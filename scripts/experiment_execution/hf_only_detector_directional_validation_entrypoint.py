"""Production worker for frozen HF-only detector directional validation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
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
from experiments.protocol.hf_only_detector_directional_validation import (
    OPERATIONAL_UNIT_COUNT,
    canonical_digest,
    load_authority_deny_axes,
    load_hf_only_detector_directional_protocol,
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
from experiments.runners.hf_only_detector_directional_validation import (
    HfDetectorDirectionalEvidenceViolation,
    HfOnlyDetectorDirectionalRunner,
)
from main import identify_root_key, key_schedule_sha256_counter
from main.shared.key_schedule import stable_json_utf8
from runtime import Sd35PipelineBackend, create_runtime_adapter
from scripts.experiment_execution.delivery_support import (
    _base_latent,
    _build_or_verify_package,
    _canonical_bytes,
    _environment_digest,
    _session_runtime_identity,
    _sha256_file,
)


PROTOCOL_PATH = Path(
    "configs/experiments/hf_only_detector_directional_validation.json"
)
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")


class HfDetectorDirectionalEntrypointError(RuntimeError):
    """The HF detector directional worker could not preserve its boundary."""


def _derive_registered_experiment_root(
    base_root_key: str, *, protocol_digest: str, manifest_digest: str
) -> str:
    root_material_envelope = {
        "base_root_key_text": base_root_key,
        "derivation_version": "ceg_wm_registered_experiment_root_v2",
        "family": "hf_only_detector_directional_validation",
        "manifest_digest": manifest_digest,
        "protocol_digest": protocol_digest,
    }
    registered_root_material = stable_json_utf8(root_material_envelope).decode(
        "utf-8", errors="strict"
    )
    stream = key_schedule_sha256_counter(
        registered_root_material,
        {
            "candidate_id": "hf_sparse_tail",
            "operator": "carrier_template",
            "responsibility_domain": "hf_carrier",
            "tensor_role": "base_gaussian",
        },
        (8,),
    )
    return "ceg-wm-hf-directional-registered-v2:" + stream.domain_digest


def _is_resource_failure(error: BaseException) -> bool:
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


def execute_hf_only_detector_directional_validation_session(
    *,
    repository_root: str | Path,
    expected_revision: str,
    persistent_root: str | Path,
    cache_root: str | Path,
    run_id: str,
    session_id: str,
    environment: Mapping[str, str],
    authorized_scientific_unit_count: int,
) -> tuple[int, dict[str, object]]:
    """Run or resume two operational and thirty-two scientific units."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_key = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    if not root_key or not hf_token:
        raise HfDetectorDirectionalEntrypointError(
            "HF_TOKEN and CEG_WM_ROOT_KEY are required"
        )
    protocol, manifest = load_hf_only_detector_directional_protocol(
        repository / PROTOCOL_PATH,
        repository_root=repository,
    )
    if authorized_scientific_unit_count not in {
        protocol.initial_gpu_gate_scientific_unit_count,
        protocol.scientific_cluster_count,
    }:
        raise HfDetectorDirectionalEntrypointError(
            "authorized scientific unit count must be the first gate or full roster"
        )
    first_entry = manifest.entries[0]
    backend = Sd35PipelineBackend(
        cache_root=cache,
        persistent_root=persistent,
        hf_token=hf_token,
        prompt=first_entry.prompt,
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
        manifest_digest=manifest.digest(),
    )
    base_public_root = identify_root_key(root_key).root_key_public_digest
    public_root = identify_root_key(registered_root_key).root_key_public_digest
    registered_key_family_digest = canonical_digest(
        {
            "root_key_public_digest": public_root,
            "seed_namespace": manifest.seed_namespace,
            "role": "registered_hf_detector_directional_key_family",
        }
    )
    deny_axes = load_authority_deny_axes(
        protocol.prior_development_manifests, repository
    )
    if (
        public_root == base_public_root
        or public_root in set(deny_axes.key_control_identities)
        or registered_key_family_digest in set(deny_axes.key_control_identities)
    ):
        raise HfDetectorDirectionalEntrypointError(
            "experiment registered key family overlaps a prior authority"
        )
    candidate_config_digest = canonical_digest(
        {
            "adapter_config_digest": adapter.configuration.config_digest,
            "candidate_identity": protocol.candidate_identity,
            "public_callable": protocol.public_callable,
            "manifest_digest": manifest.digest(),
            "runtime_config_digest": runtime_session.runtime_config_digest,
        }
    )
    authority_digest = canonical_digest(
        {
            "protocol_digest": protocol_digest,
            "manifest_digest": manifest.digest(),
            "run_id": run_id,
            "root_key_public_digest": public_root,
        }
    )
    runner = HfOnlyDetectorDirectionalRunner(
        protocol=protocol,
        manifest=manifest,
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
    package = _build_or_verify_package(repository, persistent, expected_revision)
    package_sha256 = _sha256_file(package)
    identity = FrozenWorkerIdentity(
        revision=expected_revision,
        protocol_digest=protocol_digest,
        execution_intent_authority_digest=authority_digest,
        input_manifest_digest=manifest.digest(),
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
    first_gate_boundary = (
        OPERATIONAL_UNIT_COUNT + protocol.initial_gpu_gate_scientific_unit_count
    )
    if (
        authorized_scientific_unit_count == protocol.scientific_cluster_count
        and cursor.next_unit_index < first_gate_boundary
    ):
        runtime_adapter.close()
        raise HfDetectorDirectionalEntrypointError(
            "full directional roster requires verified first-gate COMMITTED evidence"
        )
    if (
        authorized_scientific_unit_count
        == protocol.initial_gpu_gate_scientific_unit_count
        and cursor.next_unit_index > first_gate_boundary
    ):
        runtime_adapter.close()
        raise HfDetectorDirectionalEntrypointError(
            "first directional gate cannot resume beyond its frozen boundary"
        )
    committed_before = cursor.initial_committed_count
    termination_reason = "frozen_roster_complete"
    failure: dict[str, object] | None = None
    aggregate: dict[str, object] | None = None
    active_unit_index: int | None = None
    try:
        while cursor.next_unit_index < len(protocol.unit_roster):
            if cursor.next_unit_index >= (
                OPERATIONAL_UNIT_COUNT + authorized_scientific_unit_count
            ):
                termination_reason = "authorized_directional_unit_boundary_reached"
                break
            now = int(time.time())
            if now - started_epoch >= SOFT_STOP_SECONDS:
                termination_reason = "soft_stop_after_current_unit"
                break
            unit = protocol.unit_roster[cursor.next_unit_index]
            active_unit_index = unit.unit_index
            intent = store.create_session_intent(
                cursor, lease, now_epoch_seconds=now
            )
            entry = (
                manifest.operational_entries[unit.source_cluster_ordinal]
                if unit.unit_index < OPERATIONAL_UNIT_COUNT
                else manifest.scientific_entries[unit.source_cluster_ordinal]
            )
            backend.set_development_generation_prompts(entry.prompt)
            attempted_at = monotonic()
            if unit.unit_index < OPERATIONAL_UNIT_COUNT:
                record = runner.execute_operational_smoke(
                    unit_index=unit.unit_index,
                    base_latent=_base_latent(
                        entry.generation_seed,
                        height=runtime_session.image_height,
                        width=runtime_session.image_width,
                    ),
                    attempt_index=intent.attempt_index,
                    retry_parent_intent_digest=intent.parent_attempt_intent_digest,
                    maximum_duration_seconds=unit.maximum_duration_seconds,
                    started_monotonic=attempted_at,
                )
            else:
                try:
                    record = runner.execute_scientific_cluster(
                        cluster_ordinal=unit.source_cluster_ordinal,
                        base_latent=_base_latent(
                            entry.generation_seed,
                            height=runtime_session.image_height,
                            width=runtime_session.image_width,
                        ),
                        attempt_index=intent.attempt_index,
                        retry_parent_intent_digest=intent.parent_attempt_intent_digest,
                        maximum_duration_seconds=unit.maximum_duration_seconds,
                        started_monotonic=attempted_at,
                    )
                except Exception as exc:
                    resource_failure = _is_resource_failure(exc)
                    record = runner.create_failed_scientific_record(
                        cluster_ordinal=unit.source_cluster_ordinal,
                        attempt_index=intent.attempt_index,
                        retry_parent_intent_digest=intent.parent_attempt_intent_digest,
                        maximum_duration_seconds=unit.maximum_duration_seconds,
                        actual_elapsed_seconds=float(monotonic() - attempted_at),
                        failure_type=(
                            f"{type(exc).__module__}.{type(exc).__qualname__}"
                        ),
                        resource_failure=resource_failure,
                        failure_category=(
                            "resource_failure"
                            if resource_failure
                            else (
                                exc.category
                                if isinstance(
                                    exc, HfDetectorDirectionalEvidenceViolation
                                )
                                else "nonfinite_violation"
                                if "finite" in str(exc).lower()
                                else "identity_violation"
                                if "identity" in str(exc).lower()
                                or "configuration" in str(exc).lower()
                                else "budget_violation"
                                if "budget" in str(exc).lower()
                                else "integrity_violation"
                                if "integrity" in str(exc).lower()
                                or "replay" in str(exc).lower()
                                else "implementation_failure"
                            )
                        ),
                        failure_diagnostics=(
                            exc.diagnostics
                            if isinstance(
                                exc, HfDetectorDirectionalEvidenceViolation
                            )
                            else None
                        ),
                    )
            marker = store.commit_session_unit(
                cursor,
                lease,
                intent,
                record=record,
                raw_secret_values=(root_key, registered_root_key, hf_token),
                now_epoch_seconds=max(now, int(time.time())),
            )
            if marker.attempt_disposition == "retryable_resource_failure":
                termination_reason = "retryable_resource_failure_after_committed_attempt"
                break
            if (
                type(record) is DevelopmentScientificRecord
                and record.failure_class == "resource_failure"
            ):
                termination_reason = "terminal_resource_failure_after_committed_attempt"
                break
        if cursor.next_unit_index == len(protocol.unit_roster):
            verified = store.verified_terminal_scientific_evidence(
                now_epoch_seconds=int(time.time())
            )
            aggregate = asdict(runner.replay_directional_aggregate(verified))
    except Exception as exc:
        termination_reason = "worker_execution_failure"
        failure = {
            "failure_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "stage": "hf_only_detector_directional_unit_execution",
            "unit_index": active_unit_index,
            "scientific_claims_supported": False,
        }
    finally:
        runtime_adapter.close()
    ended_epoch = int(time.time())
    session_commits = tuple(
        item.unit_id for item in cursor.committed_units if item.session_id == session_id
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
        package_sha256=package_sha256,
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
        if aggregate is not None:
            target.writestr(
                "directional_aggregate.json", _canonical_bytes(aggregate)
            )
        if failure is not None:
            target.writestr("diagnostic.json", _canonical_bytes(failure))
    return (3 if failure is not None else 0), {
        "artifact_kind": (
            "hf_only_detector_directional_validation_failure"
            if failure is not None
            else "hf_only_detector_directional_validation_result"
        ),
        ("diagnostic_zip" if failure is not None else "result_zip"): str(archive),
        "protocol_digest": protocol_digest,
        "input_manifest_digest": manifest.digest(),
        "candidate_config_digest": candidate_config_digest,
        "unit_roster_digest": protocol.unit_roster_digest,
        "source_cluster_deny_list_digest": protocol.source_cluster_deny_list_digest,
        "authorized_scientific_unit_count": authorized_scientific_unit_count,
        "package_sha256": package_sha256,
        "committed_unit_count": len(cursor.committed_units),
        "session_committed_unit_count": len(cursor.committed_units) - committed_before,
        "termination_reason": termination_reason,
        "directional_aggregate": aggregate,
        "formal_tau_created": False,
        "fpr_estimated": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
    }


__all__ = [
    "HfDetectorDirectionalEntrypointError",
    "execute_hf_only_detector_directional_validation_session",
]
