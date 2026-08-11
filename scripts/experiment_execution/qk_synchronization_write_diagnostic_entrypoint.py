"""Production worker for the frozen Q/K synchronization-write diagnosis."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from time import monotonic
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from torch.utils.checkpoint import CheckpointError

from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
from experiments.protocol.development_records import (
    DevelopmentRecordError,
    DevelopmentScientificRecord,
)
from experiments.protocol.qk_synchronization_write_diagnostic import (
    OPERATIONAL_UNIT_COUNT,
    RATIO_PROBE_UNIT_COUNT,
    canonical_digest,
    derive_qk_synchronization_analysis_identity,
    load_authority_deny_axes,
    load_qk_synchronization_write_protocol,
)
from experiments.runners.development_persistence import (
    DIAGNOSTIC_SCHEMA_VERSION,
    GPU_MIX_POLICY,
    HARD_SESSION_CAP_SECONDS,
    SOFT_STOP_SECONDS,
    DevelopmentPersistentStore,
    FrozenDevelopmentUnitBinding,
    FrozenWorkerIdentity,
    SessionReceipt,
    create_frozen_development_unit_binding,
)
from experiments.runners.qk_synchronization_write_diagnostic import (
    RGB8_MEMBER_PATH,
    QkSynchronizationWriteDiagnosticRunner,
)
from main import identify_root_key, key_schedule_sha256_counter
from runtime import Sd35PipelineBackend, create_runtime_adapter
from runtime.sd35_backend import (
    Sd35BackendDifferentiableImagePostprocessError,
    Sd35BackendDifferentiableVaeCheckpointExecutionError,
    Sd35BackendDifferentiableVaeCheckpointRecomputationError,
    Sd35BackendDifferentiableVaeInitialDecodeForwardError,
    Sd35BackendDifferentiableVaeInputPreparationError,
)
from scripts.experiment_execution.development_exploration_entrypoint import (
    _base_latent,
    _canonical_bytes,
    _environment_digest,
    _session_runtime_identity,
)


PROTOCOL_PATH = Path("configs/experiments/qk_synchronization_write_diagnostic.json")
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")
CUDA_LAUNCH_BLOCKING_IDENTITY = "cuda_launch_blocking_enabled"
WORKER_RESULT_PREFIX = "CEG_WM_QK_WORKER_RESULT="


class QkSynchronizationWriteEntrypointError(RuntimeError):
    """The Q/K diagnosis worker could not preserve its frozen boundary."""


def _authorized_persistence_bindings(
    runner: QkSynchronizationWriteDiagnosticRunner,
) -> tuple[FrozenDevelopmentUnitBinding, ...]:
    source = runner.create_persistence_unit_bindings()[0]
    unit = runner.protocol.authorized_unit_roster[0]
    return (
        create_frozen_development_unit_binding(
            unit,
            analysis_unit_identity=source.analysis_unit_identity,
            scientific_question_id=source.scientific_question_id,
            development_case_id=source.development_case_id,
            candidate_identity=source.candidate_identity,
            candidate_config_digest=source.candidate_config_digest,
        ),
    )


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


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_resource_failure(error: BaseException) -> bool:
    resource_types = tuple(dict.fromkeys((MemoryError, getattr(torch, "OutOfMemoryError", MemoryError), getattr(torch.cuda, "OutOfMemoryError", MemoryError))))
    return any(isinstance(current, resource_types) for current in _exception_chain(error))


def _qualified_exception_type_chain(error: BaseException) -> tuple[str, ...]:
    return tuple(
        f"{type(current).__module__}.{type(current).__qualname__}"
        for current in _exception_chain(error)
    )


_CUDA_MEMORY_FACT_KEYS = frozenset(
    {
        "before_allocated_bytes",
        "before_reserved_bytes",
        "before_max_allocated_bytes",
        "before_max_reserved_bytes",
        "after_allocated_bytes",
        "after_reserved_bytes",
        "after_max_allocated_bytes",
        "after_max_reserved_bytes",
        "total_device_bytes",
    }
)
_RUNTIME_FAILURE_OPERATION_IDENTITIES = {
    Sd35BackendDifferentiableVaeInputPreparationError: (
        "differentiable_vae_input_preparation"
    ),
    Sd35BackendDifferentiableVaeInitialDecodeForwardError: (
        "differentiable_vae_initial_decode_forward"
    ),
    Sd35BackendDifferentiableVaeCheckpointRecomputationError: (
        "differentiable_vae_checkpoint_recomputation"
    ),
    Sd35BackendDifferentiableVaeCheckpointExecutionError: (
        "differentiable_vae_checkpoint_execution"
    ),
    Sd35BackendDifferentiableImagePostprocessError: (
        "differentiable_image_postprocess"
    ),
    CheckpointError: "differentiable_vae_checkpoint_execution",
}
_VAE_DECODE_RUNTIME_FAILURE_TYPES = frozenset(
    {
        Sd35BackendDifferentiableVaeInitialDecodeForwardError,
        Sd35BackendDifferentiableVaeCheckpointRecomputationError,
        Sd35BackendDifferentiableVaeCheckpointExecutionError,
    }
)
_VAE_DECODE_RUNTIME_REASON_IDENTITIES = frozenset(
    {
        "runtime_reported_memory_allocation_failure",
        "cuda_kernel_execution_failure",
        "dtype_shape_operator_contract_failure",
        "checkpoint_recomputation_metadata_mismatch",
        "unclassified_runtime_failure",
    }
)


def _runtime_failure_safe_attribution(
    error: BaseException,
) -> tuple[str, str | None, dict[str, int] | None] | None:
    for current in _exception_chain(error):
        operation_identity = _RUNTIME_FAILURE_OPERATION_IDENTITIES.get(type(current))
        if operation_identity is None:
            continue
        if type(current) is CheckpointError:
            return (
                operation_identity,
                "checkpoint_recomputation_metadata_mismatch",
                None,
            )
        raw_facts = getattr(current, "cuda_memory_facts", None)
        if type(raw_facts) is not tuple:
            return None
        runtime_reason_identity = None
        if type(current) in _VAE_DECODE_RUNTIME_FAILURE_TYPES:
            observed_reason = getattr(current, "runtime_reason_identity", None)
            if observed_reason not in _VAE_DECODE_RUNTIME_REASON_IDENTITIES:
                return None
            runtime_reason_identity = observed_reason
        if not raw_facts:
            if runtime_reason_identity is None:
                return None
            return operation_identity, runtime_reason_identity, None
        try:
            facts = dict(raw_facts)
        except (TypeError, ValueError):
            return None
        if (
            set(facts) != _CUDA_MEMORY_FACT_KEYS
            or any(type(value) is not int or value < 0 for value in facts.values())
        ):
            return None
        return operation_identity, runtime_reason_identity, facts
    return None


def _failure_diagnostic(
    error: BaseException,
    *,
    active_binding: FrozenDevelopmentUnitBinding | None,
    cuda_launch_blocking_enabled: bool = False,
) -> dict[str, object]:
    type_chain = _qualified_exception_type_chain(error)
    diagnostic = {
        "failure_type": type_chain[0],
        "failure_type_chain": list(type_chain),
        "failure_class": (
            "resource_failure"
            if _is_resource_failure(error)
            else "implementation_failure"
        ),
        "stage": "qk_synchronization_write_diagnostic_execution",
        "unit_index": (
            None if active_binding is None else active_binding.unit_index
        ),
        "unit_id": None if active_binding is None else active_binding.unit_id,
        "operation_identity": (
            None
            if active_binding is None
            else active_binding.analysis_unit_identity.case_id
        ),
        "phase": None if active_binding is None else active_binding.phase,
        "counts_as_scientific_coverage": False,
        "scientific_claims_supported": False,
    }
    if cuda_launch_blocking_enabled:
        diagnostic["cuda_launch_blocking_identity"] = (
            CUDA_LAUNCH_BLOCKING_IDENTITY
        )
    runtime_failure = _runtime_failure_safe_attribution(error)
    if runtime_failure is not None:
        operation_identity, runtime_reason_identity, facts = runtime_failure
        diagnostic["runtime_failure_operation_identity"] = operation_identity
        if facts is not None:
            diagnostic["runtime_failure_cuda_memory_facts"] = facts
        if runtime_reason_identity is not None:
            diagnostic["runtime_failure_reason_identity"] = (
                runtime_reason_identity
            )
    return diagnostic


def _selected_rgb8(
    persistent: Path,
    *,
    run_id: str,
    cluster_ordinal: int,
    selected_ratio_identity: str,
    evidence,
    protocol,
    manifest,
) -> torch.Tensor:
    ratio_units = tuple(
        unit
        for unit in protocol.unit_roster
        if 1 <= unit.unit_index <= RATIO_PROBE_UNIT_COUNT
        and unit.source_cluster_ordinal == cluster_ordinal
        and unit.geometry_case_id == selected_ratio_identity
    )
    if len(ratio_units) != 1 or not 0 <= cluster_ordinal < len(manifest.entries):
        raise QkSynchronizationWriteEntrypointError(
            "selected ratio is outside the frozen roster"
        )
    expected_unit = ratio_units[0]
    expected_identity = derive_qk_synchronization_analysis_identity(
        manifest.entries[cluster_ordinal],
        expected_unit,
        content_key_family_digest=canonical_digest(
            {
                "role": "registered_hf_content_key_family",
                "protocol_digest": protocol.digest(),
                "manifest_digest": manifest.digest(),
            }
        ),
        geometry_key_family_digest=canonical_digest(
            {
                "role": "registered_geometry_key_family",
                "protocol_digest": protocol.digest(),
                "manifest_digest": manifest.digest(),
            }
        ),
    )
    matches = tuple(
        (record, marker)
        for record, marker in evidence
        if record.unit_index == expected_unit.unit_index
    )
    if len(matches) != 1:
        raise QkSynchronizationWriteEntrypointError("selected ratio bundle is not unique")
    record, marker = matches[0]
    try:
        record.validate()
    except DevelopmentRecordError as exc:
        raise QkSynchronizationWriteEntrypointError(
            "selected ratio record is invalid"
        ) from exc
    if (
        marker.attempt_disposition != "success"
        or marker.unit_index != expected_unit.unit_index
        or marker.record_id != record.record_id
        or marker.record_digest
        != sha256(_canonical_bytes(record.payload())).hexdigest()
        or record.execution_status != "success"
        or record.run_id != run_id
        or record.unit_index != expected_unit.unit_index
        or record.phase != expected_unit.phase
        or record.responsibility_id != expected_unit.responsibility_id
        or record.geometry_case_id != selected_ratio_identity
        or record.analysis_unit_identity != asdict(expected_identity)
    ):
        raise QkSynchronizationWriteEntrypointError(
            "selected ratio record or marker identity drifted"
        )
    metadata = record.operation_result_payload.get("accepted_rgb8_member")
    if type(metadata) is not dict or metadata.get("path") != RGB8_MEMBER_PATH:
        raise QkSynchronizationWriteEntrypointError("selected ratio RGB8 member metadata is missing")
    ratio_observation = record.operation_result_payload.get(
        "ratio_probe_observation"
    )
    if (
        type(ratio_observation) is not dict
        or ratio_observation.get("geometry_written_rgb8_digest")
        != metadata.get("sha256")
    ):
        raise QkSynchronizationWriteEntrypointError(
            "selected ratio record geometry digest drifted"
        )
    bundle = persistent / run_id / "bundles" / f"sha256_{marker.bundle_sha256}.zip"
    if sha256(bundle.read_bytes()).hexdigest() != marker.bundle_sha256:
        raise QkSynchronizationWriteEntrypointError("selected ratio bundle bytes drifted")
    with ZipFile(bundle, "r") as archive:
        payload = archive.read(RGB8_MEMBER_PATH)
        artifact_manifest = json.loads(archive.read("artifact_manifest.json"))
    artifact_members = {
        item.get("path"): item
        for item in artifact_manifest.get("members", ())
        if type(item) is dict
    }
    artifact_member = artifact_members.get(RGB8_MEMBER_PATH)
    if (
        type(artifact_member) is not dict
        or artifact_member.get("size_bytes") != metadata.get("size_bytes")
        or artifact_member.get("sha256") != metadata.get("sha256")
    ):
        raise QkSynchronizationWriteEntrypointError(
            "selected ratio artifact manifest drifted"
        )
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
    cuda_launch_blocking_enabled = (
        environment.get("CUDA_LAUNCH_BLOCKING") == "1"
        and os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"
    )
    if not root_key or not hf_token:
        raise QkSynchronizationWriteEntrypointError("HF_TOKEN and CEG_WM_ROOT_KEY are required")
    if not cuda_launch_blocking_enabled:
        raise QkSynchronizationWriteEntrypointError(
            "CUDA launch blocking identity is required before worker import"
        )
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
    registered_bindings = _authorized_persistence_bindings(runner)
    store = DevelopmentPersistentStore(persistent, run_id=run_id, worker_identity=FrozenWorkerIdentity(revision=expected_revision, protocol_digest=protocol_digest, execution_intent_authority_digest=authority_digest, input_manifest_digest=manifest.digest(), candidate_config_digest=candidate_digest, unit_roster_digest=protocol.authorized_unit_roster_digest), registered_unit_bindings=registered_bindings)
    started_epoch = int(time.time())
    lease = store.acquire_lease(session_id=session_id, now_epoch_seconds=started_epoch, lease_duration_seconds=HARD_SESSION_CAP_SECONDS - 1)
    cursor = store.open_session_cursor(lease, now_epoch_seconds=started_epoch)
    committed_before = cursor.initial_committed_count
    termination_reason = "operational_failure_localization_incomplete"
    failure = None
    aggregate = None
    active_binding = None
    try:
        while cursor.next_unit_index < len(protocol.authorized_unit_roster):
            now = int(time.time())
            if now - started_epoch >= SOFT_STOP_SECONDS:
                termination_reason = "soft_stop_after_current_unit"
                break
            unit = protocol.authorized_unit_roster[cursor.next_unit_index]
            active_binding = registered_bindings[unit.unit_index]
            intent = store.create_session_intent(cursor, lease, now_epoch_seconds=now)
            diagnostic_members = {}
            attempted_at = monotonic()
            try:
                if unit.unit_index != 0:
                    raise QkSynchronizationWriteEntrypointError(
                        "failure localization cannot execute scientific units"
                    )
                backend.set_development_generation_prompts(protocol.operational_smoke_prompt)
                latent = _base_latent(protocol.operational_smoke_generation_seed, height=runtime_session.image_height, width=runtime_session.image_width)
                record = runner.execute_operational_smoke(base_latent=latent, attempt_index=intent.attempt_index, retry_parent_intent_digest=intent.parent_attempt_intent_digest, maximum_duration_seconds=unit.maximum_duration_seconds)
            except Exception as exc:
                if unit.unit_index == 0:
                    raise
                resource = _is_resource_failure(exc)
                record = runner.create_failed_record(unit_index=unit.unit_index, attempt_index=intent.attempt_index, retry_parent_intent_digest=intent.parent_attempt_intent_digest, maximum_duration_seconds=unit.maximum_duration_seconds, actual_elapsed_seconds=float(monotonic() - attempted_at), failure_type=f"{type(exc).__module__}.{type(exc).__qualname__}", resource_failure=resource)
                diagnostic_members = {}
            marker = store.commit_session_unit(cursor, lease, intent, record=record, diagnostic_members=diagnostic_members, raw_secret_values=(root_key, content_root, geometry_root, hf_token), now_epoch_seconds=max(now, int(time.time())))
            if marker.attempt_disposition != "success":
                raise QkSynchronizationWriteEntrypointError(
                    "failure localization operational record did not terminate"
                )
            termination_reason = "operational_failure_localization_complete"
            break
    except Exception as exc:
        termination_reason = "operational_failure_localization_failed"
        failure = _failure_diagnostic(
            exc,
            active_binding=active_binding,
            cuda_launch_blocking_enabled=cuda_launch_blocking_enabled,
        )
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
    return (3 if failure is not None else 0), {"artifact_kind": "qk_synchronization_write_diagnostic_failure" if failure is not None else "qk_synchronization_write_diagnostic_result", "diagnostic_zip" if failure is not None else "result_zip": str(archive), "protocol_digest": protocol_digest, "input_manifest_digest": manifest.digest(), "candidate_config_digest": candidate_digest, "unit_roster_digest": protocol.authorized_unit_roster_digest, "source_cluster_deny_list_digest": protocol.source_cluster_deny_list_digest, "package_sha256": execution_package_sha256, "committed_unit_count": len(cursor.committed_units), "session_committed_unit_count": len(cursor.committed_units)-committed_before, "termination_reason": termination_reason, "qk_synchronization_diagnosis_aggregate": aggregate, "cuda_launch_blocking_identity": CUDA_LAUNCH_BLOCKING_IDENTITY, "formal_tau_created": False, "fpr_estimated": False, "candidate_promoted": False, "scientific_claims_supported": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--execution-package-sha256", required=True)
    arguments = parser.parse_args(argv)
    exit_code, result = execute_qk_synchronization_write_diagnostic_session(
        repository_root=arguments.repository_root,
        expected_revision=arguments.expected_revision,
        persistent_root=arguments.persistent_root,
        cache_root=arguments.cache_root,
        run_id=arguments.run_id,
        session_id=arguments.session_id,
        execution_package_sha256=arguments.execution_package_sha256,
        environment=os.environ,
    )
    print(WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "QkSynchronizationWriteEntrypointError",
    "execute_qk_synchronization_write_diagnostic_session",
]
