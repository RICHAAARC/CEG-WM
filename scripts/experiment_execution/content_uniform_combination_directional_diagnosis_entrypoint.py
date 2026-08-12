"""Production worker for the frozen uniform-combination directional diagnosis."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import os
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
from experiments.protocol.content_uniform_combination_directional_diagnosis import (
    CLAIM_BOUNDARY,
    canonical_digest,
    load_content_uniform_combination_directional_protocol,
)
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.runners.content_uniform_combination_directional_diagnosis import (
    ContentUniformCombinationDirectionalDiagnosisRunner,
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
from main import identify_root_key, key_schedule_sha256_counter
from runtime import (
    RuntimeSession,
    Sd35PipelineBackend,
    Sd35RuntimeAdapter,
    Sd35RuntimeConfiguration,
    load_runtime_configuration,
)
from scripts.experiment_execution.development_exploration_entrypoint import (
    _base_latent,
    _build_or_verify_package,
    _canonical_bytes,
    _environment_digest,
    _session_runtime_identity,
    _sha256_file,
)
from scripts.experiment_execution.lf_whitened_directional_validation_entrypoint import (
    _replay_verified_whitening_asset,
)


PROTOCOL_PATH = Path("configs/experiments/content_uniform_combination_directional_diagnosis.json")
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")


class ContentUniformCombinationDirectionalEntrypointError(RuntimeError):
    """The combination diagnosis worker could not preserve its frozen boundary."""


class ContentUniformCombinationDirectionalStartupError(RuntimeError):
    """The worker failed before its persistent store or first intent existed."""

    def __init__(self, *, failure_type: str, failure_class: str) -> None:
        super().__init__("content combination startup failed")
        self.failure_type = failure_type
        self.failure_class = failure_class


def _combination_candidate_config_digest(
    *,
    adapter_config_digest: str,
    protocol,
    probe_manifest_digest: str,
    reference_manifest_digest: str,
    runtime_config_digest: str,
) -> str:
    """Bind the frozen diagnostic combination and reference semantics."""

    return canonical_digest(
        {
            "adapter_config_digest": adapter_config_digest,
            "combination_functions": protocol.combination_functions,
            "combination_weights": protocol.combination_weights,
            "mixing_coefficients": protocol.mixing_coefficients,
            "probe_manifest_digest": probe_manifest_digest,
            "reference_manifest_digest": reference_manifest_digest,
            "whitening_asset_digest": protocol.whitening_asset_digest,
            "runtime_config_digest": runtime_config_digest,
        }
    )


def _registered_experiment_root(
    base_root_key: str,
    *,
    protocol_digest: str,
    reference_manifest_digest: str,
    probe_manifest_digest: str,
) -> str:
    stream = key_schedule_sha256_counter(
        base_root_key,
        {
            "candidate_id": "hf_sparse_tail",
            "operator": "carrier_template",
            "responsibility_domain": "hf_carrier",
            "model_revision": canonical_digest(
                {
                    "derivation_identity": "content_uniform_combination_registered_key_subdomain_derivation",
                    "probe_manifest_digest": probe_manifest_digest,
                    "protocol_digest": protocol_digest,
                    "reference_manifest_digest": reference_manifest_digest,
                }
            ),
            "tensor_role": "base_gaussian",
        },
        (8,),
    )
    return "ceg-wm-content-uniform-combination-registered:" + stream.domain_digest


def _resource_failure(error: BaseException) -> bool:
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


def _close_runtime_preserving_failure(runtime: Sd35RuntimeAdapter | None) -> None:
    if runtime is None:
        return
    try:
        runtime.close()
    except Exception:
        pass


def _initialize_combination_resources(
    *,
    cache: Path,
    persistent: Path,
    hf_token: str,
    prompt: str,
    runtime_configuration: Sd35RuntimeConfiguration,
) -> tuple[
    Sd35PipelineBackend,
    Sd35RuntimeAdapter,
    RuntimeSession,
]:
    """Initialize only the bounded GPU/model resources admitted as startup failures."""

    runtime = None
    try:
        backend = Sd35PipelineBackend(
            cache_root=cache,
            persistent_root=persistent,
            hf_token=hf_token,
            prompt=prompt,
        )
        runtime = Sd35RuntimeAdapter(backend, runtime_configuration)
        session = runtime.initialize("cuda")
    except Exception as exc:
        _close_runtime_preserving_failure(runtime)
        raise ContentUniformCombinationDirectionalStartupError(
            failure_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            failure_class=(
                "resource_failure" if _resource_failure(exc) else "implementation_failure"
            ),
        ) from exc
    return backend, runtime, session


def _terminal_scientific_records(
    store: DevelopmentPersistentStore,
) -> tuple[DevelopmentScientificRecord, ...]:
    """Read exact terminal scientific records through persistence recovery."""

    return store.verified_terminal_scientific_records(
        now_epoch_seconds=int(time.time())
    )


def _reference_dependency_failure_class(
    records: tuple[DevelopmentScientificRecord, ...],
) -> str:
    """Classify the complete terminal fit roster without hiding failed units."""

    if len(records) != 32:
        raise ContentUniformCombinationDirectionalEntrypointError(
            "combination reference terminal roster is incomplete"
        )
    implementation_failures = 0
    resource_failures = 0
    for record in records:
        checked = DevelopmentScientificRecord.from_payload(record.payload())
        if checked.execution_status == "success":
            if checked.failure_class is not None:
                raise ContentUniformCombinationDirectionalEntrypointError(
                    "successful combination reference carries a failure class"
                )
            continue
        if checked.execution_status != "failed":
            raise ContentUniformCombinationDirectionalEntrypointError(
                "combination reference is not terminal"
            )
        if checked.failure_class == "implementation_failure":
            implementation_failures += 1
        elif checked.failure_class == "resource_failure":
            resource_failures += 1
        else:
            raise ContentUniformCombinationDirectionalEntrypointError(
                "combination reference failure classification drifted"
            )
    if implementation_failures:
        return "implementation_failure"
    if resource_failures:
        return "resource_failure"
    raise ContentUniformCombinationDirectionalEntrypointError(
        "combination reference dependency is not blocked"
    )


def _commit_dependency_blocked_probe_records(
    *,
    store: DevelopmentPersistentStore,
    cursor,
    lease,
    runner: ContentUniformCombinationDirectionalDiagnosisRunner,
    failure_class: str,
    failure_reason: str = "combination_reference_dependency_blocked",
    raw_secret_values: tuple[str, ...],
) -> tuple[DevelopmentScientificRecord, ...]:
    """Commit the remaining fixed probe denominator without running observations."""

    if failure_class not in {"implementation_failure", "resource_failure"}:
        raise ContentUniformCombinationDirectionalEntrypointError(
            "combination dependency failure class is invalid"
        )
    if failure_reason not in {
        "combination_reference_dependency_blocked",
    }:
        raise ContentUniformCombinationDirectionalEntrypointError(
            "combination dependency failure reason is invalid"
        )
    while cursor.next_unit_index < len(runner.protocol.unit_roster):
        if cursor.next_unit_index < 33:
            raise ContentUniformCombinationDirectionalEntrypointError(
                "combination dependency block started before the probe roster"
            )
        now = int(time.time())
        intent = store.create_session_intent(cursor, lease, now_epoch_seconds=now)
        record = runner.create_failed_scientific_record(
            intent=intent,
            failure_class=failure_class,
            failure_reason=failure_reason,
            elapsed_seconds=0.0,
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            raw_secret_values=raw_secret_values,
            now_epoch_seconds=max(now, int(time.time())),
        )
    records = tuple(
        record for record in _terminal_scientific_records(store)
        if record.unit_index >= 33
    )
    if len(records) != 8:
        raise ContentUniformCombinationDirectionalEntrypointError(
            "combination dependency block did not preserve the fixed probe denominator"
        )
    return records


def execute_content_uniform_combination_directional_diagnosis_session(
    *,
    repository_root: str | Path,
    expected_revision: str,
    persistent_root: str | Path,
    whitening_asset_persistent_root: str | Path,
    cache_root: str | Path,
    run_id: str,
    session_id: str,
    execution_package_sha256: str,
    environment: Mapping[str, str],
) -> tuple[int, dict[str, object]]:
    """Run or resume the fixed one operational, thirty-two fit, eight probe roster."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    fit_persistent = Path(whitening_asset_persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_key = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    if not root_key or not hf_token:
        raise ContentUniformCombinationDirectionalEntrypointError(
            "HF_TOKEN and CEG_WM_ROOT_KEY are required"
        )
    protocol, reference_manifest, probe_manifest = (
        load_content_uniform_combination_directional_protocol(
            repository / PROTOCOL_PATH,
            repository_root=repository,
        )
    )
    if run_id != protocol.run_id:
        raise ContentUniformCombinationDirectionalEntrypointError("run identity drifted")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(repository / COMPONENT_PATH)
    )
    runtime_configuration = load_runtime_configuration(repository / RUNTIME_PATH)
    protocol_digest = protocol.digest()
    reference_manifest_digest = canonical_digest(asdict(reference_manifest))
    probe_manifest_digest = canonical_digest(asdict(probe_manifest))
    registered_root = _registered_experiment_root(
        root_key,
        protocol_digest=protocol_digest,
        reference_manifest_digest=reference_manifest_digest,
        probe_manifest_digest=probe_manifest_digest,
    )
    public_root = identify_root_key(registered_root).root_key_public_digest
    if public_root == identify_root_key(root_key).root_key_public_digest:
        raise ContentUniformCombinationDirectionalEntrypointError(
            "combination registered root must differ from the base root"
        )
    backend, runtime, session = _initialize_combination_resources(
        cache=cache,
        persistent=persistent,
        hf_token=hf_token,
        prompt=reference_manifest.entries[0].prompt,
        runtime_configuration=runtime_configuration,
    )
    try:
        whitening_asset = _replay_verified_whitening_asset(
            repository=repository,
            whitening_asset_persistent_root=fit_persistent,
            adapter=adapter,
            runtime_adapter=runtime,
            runtime_config_digest=session.runtime_config_digest,
            base_root_key=root_key,
            required_protocol=protocol,
        )
        candidate_digest = _combination_candidate_config_digest(
            adapter_config_digest=adapter.configuration.config_digest,
            protocol=protocol,
            probe_manifest_digest=probe_manifest_digest,
            reference_manifest_digest=reference_manifest_digest,
            runtime_config_digest=session.runtime_config_digest,
        )
        authority_digest = canonical_digest(
            {
                "probe_manifest_digest": probe_manifest_digest,
                "protocol_digest": protocol_digest,
                "reference_manifest_digest": reference_manifest_digest,
                "root_key_public_digest": public_root,
                "run_id": run_id,
            }
        )
        runner = ContentUniformCombinationDirectionalDiagnosisRunner(
            protocol=protocol,
            reference_manifest=reference_manifest,
            probe_manifest=probe_manifest,
            adapter=adapter,
            runtime_adapter=runtime,
            whitening_asset=whitening_asset,
            method_code_revision=expected_revision,
            registered_root_key=registered_root,
            root_key_public_digest=public_root,
            protocol_digest=protocol_digest,
            execution_intent_authority_digest=authority_digest,
            candidate_config_digest=candidate_digest,
        )
        package = _build_or_verify_package(repository, persistent, expected_revision)
        package_sha256 = _sha256_file(package)
        if package_sha256 != execution_package_sha256:
            raise ContentUniformCombinationDirectionalEntrypointError(
                "execution package identity drifted"
            )
        input_manifest_digest = canonical_digest(
            {
                "probe": probe_manifest_digest,
                "reference": reference_manifest_digest,
            }
        )
        store = DevelopmentPersistentStore(
            persistent,
            run_id=run_id,
            worker_identity=FrozenWorkerIdentity(
                revision=expected_revision,
                protocol_digest=protocol_digest,
                execution_intent_authority_digest=authority_digest,
                input_manifest_digest=input_manifest_digest,
                candidate_config_digest=candidate_digest,
                unit_roster_digest=protocol.unit_roster_digest,
            ),
            registered_unit_bindings=runner.create_persistence_unit_bindings(),
        )
        started_epoch = int(time.time())
        lease = store.acquire_lease(
            session_id=session_id,
            now_epoch_seconds=started_epoch,
            lease_duration_seconds=HARD_SESSION_CAP_SECONDS - 1,
        )
        cursor = store.open_session_cursor(lease, now_epoch_seconds=started_epoch)
    except Exception:
        _close_runtime_preserving_failure(runtime)
        raise
    committed_before = cursor.initial_committed_count
    termination_reason = "frozen_roster_complete"
    failure: dict[str, object] | None = None
    aggregate: dict[str, object] | None = None
    active_unit_index: int | None = None
    try:
        while cursor.next_unit_index < len(protocol.unit_roster):
            now = int(time.time())
            if now - started_epoch >= SOFT_STOP_SECONDS:
                termination_reason = "soft_stop_after_current_unit"
                break
            unit = protocol.unit_roster[cursor.next_unit_index]
            active_unit_index = unit.unit_index
            terminal_records = _terminal_scientific_records(store)
            terminal_references = tuple(
                record for record in terminal_records if 1 <= record.unit_index < 33
            )
            successful_references = tuple(
                record for record in terminal_references
                if record.execution_status == "success"
            )
            if unit.unit_index >= 33 and len(successful_references) != 32:
                dependency_failure_class = _reference_dependency_failure_class(
                    terminal_references
                )
                records = _commit_dependency_blocked_probe_records(
                    store=store,
                    cursor=cursor,
                    lease=lease,
                    runner=runner,
                    failure_class=dependency_failure_class,
                    raw_secret_values=(root_key, registered_root, hf_token),
                )
                aggregate = asdict(runner.replay_aggregate(records))
                break
            entry = (
                reference_manifest.entries[unit.source_cluster_ordinal]
                if unit.unit_index < 33
                else probe_manifest.entries[unit.source_cluster_ordinal]
            )
            backend.set_development_generation_prompts(entry.prompt)
            intent = store.create_session_intent(cursor, lease, now_epoch_seconds=now)
            started = monotonic()
            try:
                base_latent = _base_latent(
                    entry.generation_seed,
                    height=session.image_height,
                    width=session.image_width,
                )
                if unit.unit_index < 1:
                    record = runner.execute_operational_unit(
                        unit_index=unit.unit_index,
                        base_latent=base_latent,
                        intent=intent,
                    )
                elif unit.unit_index < 33:
                    record = runner.execute_reference_fit_unit(
                        unit_index=unit.unit_index,
                        base_latent=base_latent,
                        intent=intent,
                    )
                else:
                    record = runner.execute_probe_unit(
                        unit_index=unit.unit_index,
                        base_latent=base_latent,
                        intent=intent,
                        reference_records=successful_references,
                    )
            except Exception as exc:
                failure_class = (
                    "resource_failure" if _resource_failure(exc) else "implementation_failure"
                )
                failure_reason = f"{type(exc).__module__}.{type(exc).__qualname__}"
                elapsed = float(monotonic() - started)
                if unit.unit_index < 1:
                    raise
                record = runner.create_failed_scientific_record(
                    intent=intent,
                    failure_class=failure_class,
                    failure_reason=failure_reason,
                    elapsed_seconds=elapsed,
                )
            store.commit_session_unit(
                cursor,
                lease,
                intent,
                record=record,
                raw_secret_values=(root_key, registered_root, hf_token),
                now_epoch_seconds=max(now, int(time.time())),
            )
        if aggregate is None and cursor.next_unit_index == len(protocol.unit_roster):
            records = tuple(
                record for record in _terminal_scientific_records(store)
                if record.unit_index >= 33
            )
            aggregate = asdict(runner.replay_aggregate(records))
    except Exception as exc:
        termination_reason = "worker_execution_failure"
        failure = {
            "failure_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "stage": "content_uniform_combination_directional_diagnosis_execution",
            "unit_index": active_unit_index,
            "scientific_claims_supported": False,
        }
    finally:
        runtime.close()
    ended_epoch = int(time.time())
    session_commits = tuple(
        item.unit_id for item in cursor.committed_units if item.session_id == session_id
    )
    receipt = SessionReceipt(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        session_id=session_id,
        run_id=run_id,
        started_at_utc=datetime.fromtimestamp(started_epoch, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        ended_at_utc=datetime.fromtimestamp(ended_epoch, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
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
        raw_secret_values=(root_key, registered_root, hf_token),
        session_cursor=cursor,
    )
    result_root = persistent / run_id / "session_results"
    result_root.mkdir(parents=True, exist_ok=True)
    archive = result_root / f"{session_id}.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as target:
        target.write(receipt_path, "session_receipt.json")
        target.writestr("committed_unit_ids.json", _canonical_bytes(list(session_commits)))
        if aggregate is not None:
            target.writestr("directional_aggregate.json", _canonical_bytes(aggregate))
        if failure is not None:
            target.writestr("diagnostic.json", _canonical_bytes(failure))
    return (3 if failure is not None else 0), {
        "artifact_kind": (
            "content_uniform_combination_directional_diagnosis_failure"
            if failure is not None
            else "content_uniform_combination_directional_diagnosis_result"
        ),
        ("diagnostic_zip" if failure is not None else "result_zip"): str(archive),
        "protocol_digest": protocol_digest,
        "reference_manifest_digest": reference_manifest_digest,
        "probe_manifest_digest": probe_manifest_digest,
        "input_manifest_digest": input_manifest_digest,
        "candidate_config_digest": candidate_digest,
        "unit_roster_digest": protocol.unit_roster_digest,
        "package_sha256": package_sha256,
        "committed_unit_count": len(cursor.committed_units),
        "session_committed_unit_count": len(cursor.committed_units) - committed_before,
        "termination_reason": termination_reason,
        "content_uniform_combination_directional_aggregate": aggregate,
        "formal_tau_created": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = [
    "ContentUniformCombinationDirectionalStartupError",
    "execute_content_uniform_combination_directional_diagnosis_session",
]
