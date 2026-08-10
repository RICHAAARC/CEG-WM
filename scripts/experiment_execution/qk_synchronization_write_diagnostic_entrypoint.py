"""Production worker for the frozen Q/K synchronization-write diagnosis."""

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

from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.protocol.qk_synchronization_write_diagnostic import (
    OPERATIONAL_UNIT_COUNT,
    RATIO_PROBE_UNIT_COUNT,
    canonical_digest,
    load_authority_deny_axes,
    load_qk_synchronization_write_protocol,
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
from experiments.runners.qk_synchronization_write_diagnostic import (
    RGB8_MEMBER_PATH,
    QkSynchronizationWriteDiagnosticRunner,
)
from main import identify_root_key, key_schedule_sha256_counter
from runtime import Sd35PipelineBackend, create_runtime_adapter
from scripts.experiment_execution.development_exploration_entrypoint import (
    _base_latent,
    _canonical_bytes,
    _environment_digest,
    _session_runtime_identity,
)


PROTOCOL_PATH = Path("configs/experiments/qk_synchronization_write_diagnostic.json")
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")


class QkSynchronizationWriteEntrypointError(RuntimeError):
    """The Q/K diagnosis worker could not preserve its frozen boundary."""


def _registered_roots(base_root: str, *, protocol_digest: str, manifest_digest: str):
    content_stream = key_schedule_sha256_counter(
        base_root,
        {
            "candidate_id": "hf_sparse_tail",
            "operator": "carrier_template",
            "responsibility_domain": "hf_carrier",
            "model_revision": canonical_digest({"derivation": "qk_diagnosis_hf_content", "protocol_digest": protocol_digest, "manifest_digest": manifest_digest}),
            "tensor_role": "base_gaussian",
        },
        (8,),
    )
    public = identify_root_key(base_root).root_key_public_digest
    return (
        "ceg-wm-qk-diagnosis-content:" + content_stream.domain_digest,
        "ceg-wm-qk-diagnosis-geometry:" + canonical_digest({"base_root_public_digest": public, "protocol_digest": protocol_digest, "manifest_digest": manifest_digest}),
    )


def _is_resource_failure(error: BaseException) -> bool:
    resource_types = tuple(dict.fromkeys((MemoryError, getattr(torch, "OutOfMemoryError", MemoryError), getattr(torch.cuda, "OutOfMemoryError", MemoryError))))
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, resource_types):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _selected_rgb8(
    persistent: Path,
    *,
    run_id: str,
    cluster_ordinal: int,
    selected_ratio_identity: str,
    evidence,
) -> torch.Tensor:
    matches = tuple(
        (record, marker)
        for record, marker in evidence
        if record.unit_index <= RATIO_PROBE_UNIT_COUNT
        and record.analysis_unit_identity["case_id"] == selected_ratio_identity
        and record.source_cluster_ordinal == cluster_ordinal
        and marker.attempt_disposition == "success"
    )
    if len(matches) != 1:
        raise QkSynchronizationWriteEntrypointError("selected ratio bundle is not unique")
    record, marker = matches[0]
    metadata = record.operation_result_payload.get("accepted_rgb8_member")
    if type(metadata) is not dict or metadata.get("path") != RGB8_MEMBER_PATH:
        raise QkSynchronizationWriteEntrypointError("selected ratio RGB8 member metadata is missing")
    bundle = persistent / run_id / "bundles" / f"sha256_{marker.bundle_sha256}.zip"
    if sha256(bundle.read_bytes()).hexdigest() != marker.bundle_sha256:
        raise QkSynchronizationWriteEntrypointError("selected ratio bundle bytes drifted")
    with ZipFile(bundle, "r") as archive:
        payload = archive.read(RGB8_MEMBER_PATH)
    if len(payload) != metadata.get("size_bytes") or sha256(payload).hexdigest() != metadata.get("sha256"):
        raise QkSynchronizationWriteEntrypointError("selected ratio RGB8 member digest drifted")
    shape = tuple(metadata.get("shape", ()))
    if len(shape) != 4 or metadata.get("dtype") != "torch.uint8":
        raise QkSynchronizationWriteEntrypointError("selected ratio RGB8 member shape drifted")
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().reshape(shape)


def execute_qk_synchronization_write_diagnostic_session(
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
    """Run or resume one smoke, twelve ratio probes and conditional transforms."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_key = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    if not root_key or not hf_token:
        raise QkSynchronizationWriteEntrypointError("HF_TOKEN and CEG_WM_ROOT_KEY are required")
    if len(expected_revision) != 40 or len(execution_package_sha256) != 64:
        raise QkSynchronizationWriteEntrypointError("execution identity is invalid")
    protocol, manifest = load_qk_synchronization_write_protocol(repository / PROTOCOL_PATH, repository_root=repository)
    if run_id != protocol.run_id:
        raise QkSynchronizationWriteEntrypointError("run identity drifted")
    backend = Sd35PipelineBackend(cache_root=cache, persistent_root=persistent, hf_token=hf_token, prompt=protocol.operational_smoke_prompt)
    runtime = create_runtime_adapter(backend, repository / RUNTIME_PATH)
    runtime_session = runtime.initialize("cuda")
    adapter = CegWmExperimentAdapter(load_ceg_wm_experiment_adapter_configuration(repository / COMPONENT_PATH), runtime_adapter=runtime)
    protocol_digest = protocol.digest()
    content_root, geometry_root = _registered_roots(root_key, protocol_digest=protocol_digest, manifest_digest=manifest.digest())
    content_public = identify_root_key(content_root).root_key_public_digest
    geometry_public = identify_root_key(geometry_root).root_key_public_digest
    deny = load_authority_deny_axes(protocol.prior_development_manifests, repository)
    if {content_public, geometry_public} & set(deny.key_control_identities):
        runtime.close()
        raise QkSynchronizationWriteEntrypointError("Q/K diagnosis keys overlap prior authority")
    candidate_digest = canonical_digest({"candidate_identity": protocol.candidate_identity, "adapter_config_digest": adapter.configuration.config_digest, "runtime_config_digest": runtime_session.runtime_config_digest, "manifest_digest": manifest.digest(), "package_identity": execution_package_sha256})
    authority_digest = canonical_digest({"protocol_digest": protocol_digest, "manifest_digest": manifest.digest(), "content_root_public_digest": content_public, "geometry_root_public_digest": geometry_public, "run_id": run_id})
    runner = QkSynchronizationWriteDiagnosticRunner(protocol=protocol, manifest=manifest, adapter=adapter, runtime_adapter=runtime, method_code_revision=expected_revision, run_id=run_id, content_registered_root_key=content_root, geometry_registered_root_key=geometry_root, protocol_digest=protocol_digest, execution_intent_authority_digest=authority_digest, candidate_config_digest=candidate_digest, package_identity=execution_package_sha256)
    store = DevelopmentPersistentStore(persistent, run_id=run_id, worker_identity=FrozenWorkerIdentity(revision=expected_revision, protocol_digest=protocol_digest, execution_intent_authority_digest=authority_digest, input_manifest_digest=manifest.digest(), candidate_config_digest=candidate_digest, unit_roster_digest=protocol.unit_roster_digest), registered_unit_bindings=runner.create_persistence_unit_bindings())
    started_epoch = int(time.time())
    lease = store.acquire_lease(session_id=session_id, now_epoch_seconds=started_epoch, lease_duration_seconds=HARD_SESSION_CAP_SECONDS - 1)
    cursor = store.open_session_cursor(lease, now_epoch_seconds=started_epoch)
    committed_before = cursor.initial_committed_count
    termination_reason = "frozen_roster_complete"
    failure = None
    aggregate = None
    active_unit_index = None
    try:
        while cursor.next_unit_index < len(protocol.unit_roster):
            now = int(time.time())
            if now - started_epoch >= SOFT_STOP_SECONDS:
                termination_reason = "soft_stop_after_current_unit"
                break
            unit = protocol.unit_roster[cursor.next_unit_index]
            active_unit_index = unit.unit_index
            if unit.unit_index > RATIO_PROBE_UNIT_COUNT:
                ratio_aggregate = runner.replay_ratio_aggregate(cursor.terminal_scientific_evidence)
            else:
                ratio_aggregate = None
            intent = store.create_session_intent(cursor, lease, now_epoch_seconds=now)
            diagnostic_members = {}
            attempted_at = monotonic()
            try:
                if unit.unit_index == 0:
                    backend.set_development_generation_prompts(protocol.operational_smoke_prompt)
                    latent = _base_latent(protocol.operational_smoke_generation_seed, height=runtime_session.image_height, width=runtime_session.image_width)
                    record = runner.execute_operational_smoke(base_latent=latent, attempt_index=intent.attempt_index, retry_parent_intent_digest=intent.parent_attempt_intent_digest, maximum_duration_seconds=unit.maximum_duration_seconds)
                elif unit.unit_index <= RATIO_PROBE_UNIT_COUNT:
                    entry = manifest.entries[unit.source_cluster_ordinal]
                    backend.set_development_generation_prompts(entry.prompt)
                    latent = _base_latent(entry.generation_seed, height=runtime_session.image_height, width=runtime_session.image_width)
                    record, diagnostic_members = runner.execute_scientific_unit(unit_index=unit.unit_index, base_latent=latent, attempt_index=intent.attempt_index, retry_parent_intent_digest=intent.parent_attempt_intent_digest, maximum_duration_seconds=unit.maximum_duration_seconds)
                elif ratio_aggregate.selected_ratio_identity is None:
                    record = runner.create_dependency_blocked_record(unit_index=unit.unit_index, attempt_index=intent.attempt_index, retry_parent_intent_digest=intent.parent_attempt_intent_digest, maximum_duration_seconds=unit.maximum_duration_seconds)
                else:
                    rgb8 = _selected_rgb8(persistent, run_id=run_id, cluster_ordinal=unit.source_cluster_ordinal, selected_ratio_identity=ratio_aggregate.selected_ratio_identity, evidence=cursor.terminal_scientific_evidence)
                    record, diagnostic_members = runner.execute_scientific_unit(unit_index=unit.unit_index, base_latent=None, selected_ratio_identity=ratio_aggregate.selected_ratio_identity, source_rgb8=rgb8, attempt_index=intent.attempt_index, retry_parent_intent_digest=intent.parent_attempt_intent_digest, maximum_duration_seconds=unit.maximum_duration_seconds)
            except Exception as exc:
                if unit.unit_index == 0:
                    raise
                resource = _is_resource_failure(exc)
                record = runner.create_failed_record(unit_index=unit.unit_index, attempt_index=intent.attempt_index, retry_parent_intent_digest=intent.parent_attempt_intent_digest, maximum_duration_seconds=unit.maximum_duration_seconds, actual_elapsed_seconds=float(monotonic() - attempted_at), failure_type=f"{type(exc).__module__}.{type(exc).__qualname__}", resource_failure=resource)
                diagnostic_members = {}
            marker = store.commit_session_unit(cursor, lease, intent, record=record, diagnostic_members=diagnostic_members, raw_secret_values=(root_key, content_root, geometry_root, hf_token), now_epoch_seconds=max(now, int(time.time())))
            if marker.attempt_disposition == "retryable_resource_failure":
                termination_reason = "retryable_resource_failure_after_committed_attempt"
                break
            if type(record) is DevelopmentScientificRecord and record.failure_class == "resource_failure":
                termination_reason = "terminal_resource_failure_after_committed_attempt"
                break
        if cursor.next_unit_index == len(protocol.unit_roster):
            aggregate = asdict(runner.replay_synchronization_diagnosis_aggregate(store.verified_terminal_scientific_evidence(now_epoch_seconds=int(time.time()))))
    except Exception as exc:
        termination_reason = "worker_execution_failure"
        failure = {"failure_type": f"{type(exc).__module__}.{type(exc).__qualname__}", "stage": "qk_synchronization_write_diagnostic_execution", "unit_index": active_unit_index, "scientific_claims_supported": False}
    finally:
        runtime.close()
    ended_epoch = int(time.time())
    session_commits = tuple(item.unit_id for item in cursor.committed_units if item.session_id == session_id)
    receipt = SessionReceipt(schema_version=DIAGNOSTIC_SCHEMA_VERSION, session_id=session_id, run_id=run_id, started_at_utc=datetime.fromtimestamp(started_epoch, timezone.utc).isoformat().replace("+00:00", "Z"), ended_at_utc=datetime.fromtimestamp(ended_epoch, timezone.utc).isoformat().replace("+00:00", "Z"), gpu_model=_session_runtime_identity(role="gpu", display_value=torch.cuda.get_device_name(0)), cuda_identity=_session_runtime_identity(role="cuda", display_value=torch.version.cuda or "unknown"), environment_digest=_environment_digest(), revision=expected_revision, package_sha256=execution_package_sha256, walltime_seconds=float(ended_epoch-started_epoch), peak_vram_bytes=max(1, int(torch.cuda.max_memory_allocated(0))), termination_reason=termination_reason, soft_stop_seconds=SOFT_STOP_SECONDS, hard_session_cap_seconds=HARD_SESSION_CAP_SECONDS, gpu_mix_policy=GPU_MIX_POLICY, committed_unit_ids=session_commits, public_secret_identity_digests=(content_public, geometry_public))
    receipt_path = store.write_session_receipt(receipt, raw_secret_values=(root_key, content_root, geometry_root, hf_token), session_cursor=cursor)
    result_root = persistent / run_id / "session_results"
    result_root.mkdir(parents=True, exist_ok=True)
    archive = result_root / f"{session_id}.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as target:
        target.write(receipt_path, "session_receipt.json")
        target.writestr("committed_unit_ids.json", _canonical_bytes(list(session_commits)))
        if aggregate is not None:
            target.writestr("qk_synchronization_diagnosis_aggregate.json", _canonical_bytes(aggregate))
        if failure is not None:
            target.writestr("diagnostic.json", _canonical_bytes(failure))
    return (3 if failure is not None else 0), {"artifact_kind": "qk_synchronization_write_diagnostic_failure" if failure is not None else "qk_synchronization_write_diagnostic_result", "diagnostic_zip" if failure is not None else "result_zip": str(archive), "protocol_digest": protocol_digest, "input_manifest_digest": manifest.digest(), "candidate_config_digest": candidate_digest, "unit_roster_digest": protocol.unit_roster_digest, "source_cluster_deny_list_digest": protocol.source_cluster_deny_list_digest, "package_sha256": execution_package_sha256, "committed_unit_count": len(cursor.committed_units), "session_committed_unit_count": len(cursor.committed_units)-committed_before, "termination_reason": termination_reason, "qk_synchronization_diagnosis_aggregate": aggregate, "formal_tau_created": False, "fpr_estimated": False, "candidate_promoted": False, "scientific_claims_supported": False}


__all__ = ["QkSynchronizationWriteEntrypointError", "execute_qk_synchronization_write_diagnostic_session"]
