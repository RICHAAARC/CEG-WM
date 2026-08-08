"""Production worker for the frozen eight-cluster HF transport diagnostic."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    OPERATIONAL_RECORD_COLLECTION_ROLE,
    OPERATIONAL_RECORD_KIND,
    OPERATIONAL_RECORD_SCHEMA,
    DevelopmentOperationalRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.protocol.hf_transmission_diagnostic import (
    load_hf_transmission_protocol,
    canonical_digest,
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
from experiments.runners.hf_transmission_diagnostic import (
    HfTransmissionDiagnosticRunner,
)
from main import identify_root_key
from runtime import Sd35PipelineBackend, create_runtime_adapter
from scripts.experiment_execution.development_exploration_entrypoint import (
    _base_latent,
    _build_or_verify_package,
    _canonical_bytes,
    _environment_digest,
    _session_runtime_identity,
    _sha256_file,
)


PROTOCOL_PATH = Path("configs/experiments/hf_transmission_diagnostic.json")
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")


class HfTransmissionEntrypointError(RuntimeError):
    """The HF transport worker could not preserve its frozen boundary."""


def _operational_record(
    *,
    run_id: str,
    protocol_digest: str,
    revision: str,
    unit_index: int,
    candidate_config_digest: str,
    attempt_index: int,
    retry_parent_intent_digest: str | None,
    maximum_duration_seconds: int,
    runtime_config_digest: str,
    adapter_config_digest: str,
    elapsed_seconds: float,
) -> DevelopmentOperationalRecord:
    payload = {
        "schema_version": OPERATIONAL_RECORD_SCHEMA,
        "collection_role": OPERATIONAL_RECORD_COLLECTION_ROLE,
        "record_kind": OPERATIONAL_RECORD_KIND,
        "record_id": "0" * 64,
        "run_id": run_id,
        "protocol_digest": protocol_digest,
        "method_code_revision": revision,
        "unit_index": unit_index,
        "phase": "development_environment_preflight",
        "source_cluster_ordinal": unit_index,
        "candidate_config_digest": candidate_config_digest,
        "attempt_index": attempt_index,
        "retry_parent_intent_digest": retry_parent_intent_digest,
        "actual_elapsed_seconds": elapsed_seconds,
        "maximum_duration_seconds": maximum_duration_seconds,
        "operation_result_payload": {
            "operational_role": "environment_runtime_throughput_preflight",
            "source_cluster_ordinal": unit_index,
            "case_ids": ["hf_transmission_runtime_identity"],
            "responsibility_result_digests": [
                ["content_embedder", adapter_config_digest]
            ],
            "elapsed_seconds": elapsed_seconds,
            "runtime_config_digest": runtime_config_digest,
            "counts_as_scientific_coverage": False,
            "scientific_claims_supported": False,
        },
        "counts_as_scientific_coverage": False,
        "scientific_claims_supported": False,
        "scientific_claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
    }
    provisional = DevelopmentOperationalRecord(**payload)
    record = DevelopmentOperationalRecord(
        **{
            **payload,
            "record_id": canonical_development_value_digest(
                provisional.payload_without_record_id()
            ),
        }
    )
    record.validate()
    return record


def execute_hf_transmission_diagnostic_session(
    *,
    repository_root: str | Path,
    expected_revision: str,
    persistent_root: str | Path,
    cache_root: str | Path,
    run_id: str,
    session_id: str,
    environment: Mapping[str, str],
) -> tuple[int, dict[str, object]]:
    """Run or resume at most the frozen ten-unit HF diagnostic roster."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_key = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    if not root_key or not hf_token:
        raise HfTransmissionEntrypointError(
            "HF_TOKEN and CEG_WM_ROOT_KEY are required"
        )
    protocol, manifest = load_hf_transmission_protocol(
        repository / PROTOCOL_PATH,
        repository_root=repository,
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
    public_root = identify_root_key(root_key).root_key_public_digest
    protocol_digest = protocol.digest()
    candidate_config_digest = canonical_digest(
        {
            "adapter_config_digest": adapter.configuration.config_digest,
            "candidate_identity": protocol.candidate_identity,
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
    runner = HfTransmissionDiagnosticRunner(
        protocol=protocol,
        manifest=manifest,
        adapter=adapter,
        runtime_adapter=runtime_adapter,
        method_code_revision=expected_revision,
        run_id=run_id,
        registered_root_key=root_key,
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
    committed_before = cursor.initial_committed_count
    package_digest = _sha256_file(package)
    termination_reason = "frozen_roster_complete"
    failure: dict[str, object] | None = None
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
            if unit.unit_index < protocol.operational_unit_count:
                elapsed = 0.0
                record = _operational_record(
                    run_id=run_id,
                    protocol_digest=protocol_digest,
                    revision=expected_revision,
                    unit_index=unit.unit_index,
                    candidate_config_digest=candidate_config_digest,
                    attempt_index=intent.attempt_index,
                    retry_parent_intent_digest=intent.parent_attempt_intent_digest,
                    maximum_duration_seconds=unit.maximum_duration_seconds,
                    runtime_config_digest=runtime_session.runtime_config_digest,
                    adapter_config_digest=adapter.configuration.config_digest,
                    elapsed_seconds=elapsed,
                )
            else:
                entry = manifest.entries[unit.source_cluster_ordinal]
                backend.set_development_generation_prompts(entry.prompt)
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
                )
            store.commit_session_unit(
                cursor,
                lease,
                intent,
                record=record,
                raw_secret_values=(root_key, hf_token),
                now_epoch_seconds=max(now + 1, int(time.time())),
            )
    except Exception as exc:
        termination_reason = "worker_execution_failure"
        failure = {
            "failure_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "stage": "hf_transmission_unit_execution",
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
        package_sha256=package_digest,
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
        raw_secret_values=(root_key, hf_token),
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
        if failure is not None:
            target.writestr("diagnostic.json", _canonical_bytes(failure))
    return (3 if failure is not None else 0), {
        "artifact_kind": (
            "hf_transmission_diagnostic_failure"
            if failure is not None
            else "hf_transmission_diagnostic_result"
        ),
        ("diagnostic_zip" if failure is not None else "result_zip"): str(archive),
        "protocol_digest": protocol_digest,
        "input_manifest_digest": manifest.digest(),
        "candidate_config_digest": candidate_config_digest,
        "unit_roster_digest": protocol.unit_roster_digest,
        "package_sha256": package_sha256,
        "committed_unit_count": len(cursor.committed_units),
        "session_committed_unit_count": len(cursor.committed_units) - committed_before,
        "termination_reason": termination_reason,
        "formal_tau_created": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
    }


__all__ = ["execute_hf_transmission_diagnostic_session"]
