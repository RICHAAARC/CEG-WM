"""Production worker for the frozen LF whitening fit and score screening."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from time import monotonic
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.protocol.lf_whitened_score_screening import (
    canonical_digest,
    load_lf_whitened_score_screening_protocol,
)
from experiments.runners.development_persistence import (
    DIAGNOSTIC_SCHEMA_VERSION,
    GPU_MIX_POLICY,
    HARD_SESSION_CAP_SECONDS,
    SOFT_STOP_SECONDS,
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
    SessionReceipt,
)
from experiments.runners.lf_whitened_score_screening import (
    LfWhitenedScoreRunnerError,
    LfWhitenedScoreScreeningRunner,
)
from main import identify_root_key, key_schedule_sha256_counter
from main.shared.key_schedule import stable_json_utf8
from runtime import Sd35PipelineBackend, create_runtime_adapter
from scripts.experiment_execution.delivery_support import (
    _base_latent,
    _canonical_bytes,
    _environment_digest,
    _session_runtime_identity,
)


PROTOCOL_PATH = Path("configs/experiments/lf_whitened_score_screening.json")
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")


class LfWhitenedScoreEntrypointError(RuntimeError):
    """The LF whitening worker could not preserve its frozen boundary."""


def _derive_registered_experiment_root(
    base_root_key: str,
    *,
    protocol_digest: str,
    screening_manifest_digest: str,
    key_family_namespace: str,
) -> str:
    root_material_envelope = {
        "base_root_key_text": base_root_key,
        "derivation_version": "ceg_wm_registered_experiment_root_v2",
        "family": "lf_whitened_score_screening",
        "key_family_namespace": key_family_namespace,
        "protocol_digest": protocol_digest,
        "screening_manifest_digest": screening_manifest_digest,
    }
    registered_root_material = stable_json_utf8(root_material_envelope).decode(
        "utf-8", errors="strict"
    )
    stream = key_schedule_sha256_counter(
        registered_root_material,
        {
            "candidate_id": "lf_low_pass",
            "operator": "carrier_template",
            "responsibility_domain": "lf_carrier",
            "tensor_role": "base_gaussian",
        },
        (8,),
    )
    return "ceg-wm-lf-whitened-screening-registered-v2:" + stream.domain_digest


def _is_retryable_resource_failure(error: BaseException) -> bool:
    resource_types = tuple(
        dict.fromkeys(
            (
                MemoryError,
                getattr(torch, "OutOfMemoryError", MemoryError),
                getattr(torch.cuda, "OutOfMemoryError", MemoryError),
            )
        )
    )
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, resource_types):
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def execute_lf_whitened_score_screening_session(
    *,
    repository_root: str | Path,
    expected_revision: str,
    persistent_root: str | Path,
    cache_root: str | Path,
    run_id: str,
    session_id: str,
    execution_package_sha256: str,
    environment: Mapping[str, str],
) -> tuple[int, dict[str, object]]:
    """Run or resume the frozen 32-clean fit plus 8-paired screening roster."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_key = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    if not root_key or not hf_token:
        raise LfWhitenedScoreEntrypointError(
            "HF_TOKEN and CEG_WM_ROOT_KEY are required"
        )
    if (
        type(execution_package_sha256) is not str
        or len(execution_package_sha256) != 64
    ):
        raise LfWhitenedScoreEntrypointError(
            "execution package digest is invalid"
        )
    protocol, fit_manifest, screening_manifest = (
        load_lf_whitened_score_screening_protocol(
            repository / PROTOCOL_PATH,
            repository_root=repository,
        )
    )
    if run_id != protocol.run_id:
        raise LfWhitenedScoreEntrypointError("run identity drifted")
    backend = Sd35PipelineBackend(
        cache_root=cache,
        persistent_root=persistent,
        hf_token=hf_token,
        prompt=protocol.operational_smoke_prompt,
    )
    runtime_adapter = create_runtime_adapter(backend, repository / RUNTIME_PATH)
    runtime_session = runtime_adapter.initialize("cuda")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(repository / COMPONENT_PATH)
    )
    protocol_digest = protocol.digest()
    registered_root_key = _derive_registered_experiment_root(
        root_key,
        protocol_digest=protocol_digest,
        screening_manifest_digest=screening_manifest.digest(),
        key_family_namespace=screening_manifest.key_family_namespace,
    )
    public_root = identify_root_key(registered_root_key).root_key_public_digest
    if public_root == identify_root_key(root_key).root_key_public_digest:
        runtime_adapter.close()
        raise LfWhitenedScoreEntrypointError(
            "screening root must differ from the base root"
        )
    candidate_config_digest = canonical_digest(
        {
            "adapter_config_digest": adapter.configuration.config_digest,
            "candidate_identity": protocol.candidate_identity,
            "null_fit_manifest_digest": fit_manifest.digest(),
            "runtime_config_digest": runtime_session.runtime_config_digest,
            "screening_manifest_digest": screening_manifest.digest(),
        }
    )
    authority_digest = canonical_digest(
        {
            "null_fit_manifest_digest": fit_manifest.digest(),
            "protocol_digest": protocol_digest,
            "root_key_public_digest": public_root,
            "run_id": run_id,
            "screening_manifest_digest": screening_manifest.digest(),
        }
    )
    runner = LfWhitenedScoreScreeningRunner(
        protocol=protocol,
        null_fit_manifest=fit_manifest,
        screening_manifest=screening_manifest,
        adapter=adapter,
        runtime_adapter=runtime_adapter,
        method_code_revision=expected_revision,
        run_id=run_id,
        registered_root_key=registered_root_key,
        root_key_public_digest=public_root,
        protocol_digest=protocol_digest,
        execution_intent_authority_digest=authority_digest,
        candidate_config_digest=candidate_config_digest,
    )
    identity = FrozenWorkerIdentity(
        revision=expected_revision,
        protocol_digest=protocol_digest,
        execution_intent_authority_digest=authority_digest,
        input_manifest_digest=canonical_digest(
            {
                "null_fit_manifest_digest": fit_manifest.digest(),
                "screening_manifest_digest": screening_manifest.digest(),
            }
        ),
        candidate_config_digest=candidate_config_digest,
        unit_roster_digest=protocol.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        persistent,
        run_id=run_id,
        worker_identity=identity,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    started_epoch = int(time.time())
    lease = store.acquire_lease(
        session_id=session_id,
        now_epoch_seconds=started_epoch,
        lease_duration_seconds=HARD_SESSION_CAP_SECONDS - 1,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=started_epoch)
    committed_before = cursor.initial_committed_count
    termination_reason = "frozen_roster_complete"
    failure: dict[str, object] | None = None
    screening_decision: dict[str, object] | None = None
    active_unit_index: int | None = None
    try:
        while cursor.next_unit_index < len(protocol.unit_roster):
            now = int(time.time())
            if now - started_epoch >= SOFT_STOP_SECONDS:
                termination_reason = "soft_stop_after_current_unit"
                break
            unit = protocol.unit_roster[cursor.next_unit_index]
            active_unit_index = unit.unit_index
            intent = store.create_session_intent(
                cursor, lease, now_epoch_seconds=now
            )
            attempted_at = monotonic()
            try:
                if unit.unit_index == 0:
                    backend.set_development_generation_prompts(
                        protocol.operational_smoke_prompt
                    )
                    record = runner.execute_operational_smoke(
                        base_latent=_base_latent(
                            protocol.operational_smoke_generation_seed,
                            height=runtime_session.image_height,
                            width=runtime_session.image_width,
                        ),
                        attempt_index=intent.attempt_index,
                        retry_parent_intent_digest=(
                            intent.parent_attempt_intent_digest
                        ),
                        maximum_duration_seconds=unit.maximum_duration_seconds,
                        started_monotonic=attempted_at,
                    )
                elif unit.unit_index <= 32:
                    entry = fit_manifest.entries[unit.source_cluster_ordinal]
                    backend.set_development_generation_prompts(entry.prompt)
                    evidence = (
                        ()
                        if unit.unit_index == 1
                        else store.verified_terminal_scientific_evidence_for_unit_indexes(
                            tuple(range(1, unit.unit_index)),
                            now_epoch_seconds=now,
                        )
                    )
                    record = runner.execute_null_fit_cluster(
                        cluster_ordinal=unit.source_cluster_ordinal,
                        base_latent=_base_latent(
                            entry.generation_seed,
                            height=runtime_session.image_height,
                            width=runtime_session.image_width,
                        ),
                        attempt_index=intent.attempt_index,
                        retry_parent_intent_digest=(
                            intent.parent_attempt_intent_digest
                        ),
                        maximum_duration_seconds=unit.maximum_duration_seconds,
                        prior_verified_fit_evidence=evidence,
                        started_monotonic=attempted_at,
                    )
                else:
                    entry = screening_manifest.entries[
                        unit.source_cluster_ordinal
                    ]
                    backend.set_development_generation_prompts(entry.prompt)
                    fit_evidence = (
                        store.verified_terminal_scientific_evidence_for_unit_indexes(
                            tuple(range(1, 33)),
                            now_epoch_seconds=now,
                        )
                    )
                    asset = runner.replay_whitening_asset(fit_evidence)
                    record = runner.execute_screening_cluster(
                        cluster_ordinal=unit.source_cluster_ordinal,
                        base_latent=_base_latent(
                            entry.generation_seed,
                            height=runtime_session.image_height,
                            width=runtime_session.image_width,
                        ),
                        whitening_asset=asset,
                        attempt_index=intent.attempt_index,
                        retry_parent_intent_digest=(
                            intent.parent_attempt_intent_digest
                        ),
                        maximum_duration_seconds=unit.maximum_duration_seconds,
                        started_monotonic=attempted_at,
                    )
            except Exception as exc:
                if unit.unit_index == 0:
                    raise
                resource_failure = _is_retryable_resource_failure(exc)
                retryable = (
                    resource_failure
                    and intent.attempt_index + 1
                    < intent.maximum_record_attempts
                )
                record = runner.create_failed_record(
                    unit_index=unit.unit_index,
                    attempt_index=intent.attempt_index,
                    retry_parent_intent_digest=intent.parent_attempt_intent_digest,
                    maximum_duration_seconds=unit.maximum_duration_seconds,
                    actual_elapsed_seconds=float(monotonic() - attempted_at),
                    failure_stage=(
                        "lf_clean_public_vae_runtime_preflight"
                        if unit.unit_index == 0
                        else (
                            "lf_clean_null_runtime"
                            if unit.unit_index <= 32
                            else "lf_whitening_asset_or_screening_runtime"
                        )
                    ),
                    resource_failure=resource_failure,
                    retryable_resource_failure=retryable,
                )
            store.commit_session_unit(
                cursor,
                lease,
                intent,
                record=record,
                raw_secret_values=(root_key, registered_root_key, hf_token),
                now_epoch_seconds=max(now, int(time.time())),
            )
            if (
                type(record) is DevelopmentScientificRecord
                and record.execution_status == "retry"
            ):
                termination_reason = "retryable_resource_failure"
                break
        if cursor.next_unit_index == len(protocol.unit_roster):
            evidence = store.verified_terminal_scientific_evidence(
                now_epoch_seconds=int(time.time())
            )
            try:
                screening_decision = asdict(
                    runner.replay_screening_decision(evidence)
                )
            except Exception as decision_error:
                screening_decision = {
                    "allow_request_for_lf_whitened_directional_validation": False,
                    "decision_available": False,
                    "reason": "fit_or_screening_terminal_failure",
                    "failure_type": (
                        f"{type(decision_error).__module__}."
                        f"{type(decision_error).__qualname__}"
                    ),
                    "failure_reason": (
                        str(decision_error)
                        if type(decision_error) is LfWhitenedScoreRunnerError
                        else "screening_decision_replay_failed"
                    ),
                }
    except Exception as exc:
        termination_reason = "worker_execution_failure"
        failure = {
            "failure_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "stage": "lf_whitened_score_unit_execution",
            "unit_index": active_unit_index,
            "scientific_claims_supported": False,
        }
    finally:
        runtime_adapter.close()
    ended_epoch = int(time.time())
    session_commits = tuple(
        item.unit_id
        for item in cursor.committed_units
        if item.session_id == session_id
    )
    receipt = SessionReceipt(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        session_id=session_id,
        run_id=run_id,
        started_at_utc=datetime.fromtimestamp(
            started_epoch, timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        ended_at_utc=datetime.fromtimestamp(
            ended_epoch, timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        gpu_model=_session_runtime_identity(
            role="gpu", display_value=torch.cuda.get_device_name(0)
        ),
        cuda_identity=_session_runtime_identity(
            role="cuda", display_value=torch.version.cuda or "unknown"
        ),
        environment_digest=_environment_digest(),
        revision=expected_revision,
        package_sha256=execution_package_sha256,
        walltime_seconds=float(ended_epoch - started_epoch),
        peak_vram_bytes=max(1, int(torch.cuda.max_memory_allocated(0))),
        termination_reason=termination_reason,
        soft_stop_seconds=SOFT_STOP_SECONDS,
        hard_session_cap_seconds=HARD_SESSION_CAP_SECONDS,
        gpu_mix_policy=GPU_MIX_POLICY,
        committed_unit_ids=session_commits,
        public_secret_identity_digests=(public_root,),
    )
    receipt_path = store.write_session_receipt(
        receipt,
        raw_secret_values=(root_key, registered_root_key, hf_token),
        session_cursor=cursor,
    )
    result_root = persistent / run_id / "session_results"
    result_root.mkdir(parents=True, exist_ok=True)
    archive = result_root / f"{session_id}.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as target:
        target.write(receipt_path, "session_receipt.json")
        target.writestr(
            "committed_unit_ids.json", _canonical_bytes(list(session_commits))
        )
        if screening_decision is not None:
            target.writestr(
                "screening_decision.json", _canonical_bytes(screening_decision)
            )
        if failure is not None:
            target.writestr("diagnostic.json", _canonical_bytes(failure))
    return (3 if failure is not None else 0), {
        "artifact_kind": (
            "lf_whitened_score_screening_failure"
            if failure is not None
            else "lf_whitened_score_screening_result"
        ),
        ("diagnostic_zip" if failure is not None else "result_zip"): str(archive),
        "protocol_digest": protocol_digest,
        "null_fit_manifest_digest": fit_manifest.digest(),
        "screening_manifest_digest": screening_manifest.digest(),
        "candidate_config_digest": candidate_config_digest,
        "unit_roster_digest": protocol.unit_roster_digest,
        "committed_unit_count": len(cursor.committed_units),
        "session_committed_unit_count": (
            len(cursor.committed_units) - committed_before
        ),
        "termination_reason": termination_reason,
        "screening_decision": screening_decision,
        "formal_tau_created": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
    }


__all__ = ["execute_lf_whitened_score_screening_session"]
