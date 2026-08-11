"""Production worker for the frozen content-routing directional diagnosis."""

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
from experiments.metrics.content_routing_directional_diagnosis import (
    ContentRoutingBlindScoreObservation,
    ContentRoutingDirectionalObservation,
    aggregate_content_routing_directional_diagnosis,
)
from experiments.protocol.content_routing_directional_diagnosis import (
    CLAIM_BOUNDARY,
    canonical_digest,
    load_content_routing_directional_protocol,
)
from experiments.protocol.development_records import (
    DevelopmentRoutingReferenceRecord,
    DevelopmentScientificRecord,
)
from experiments.runners.content_routing_directional_diagnosis import (
    ContentRoutingDirectionalDiagnosisRunner,
)
from experiments.runners.development_inputs import (
    DevelopmentSemanticObservationProducer,
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
from runtime import Sd35PipelineBackend, create_runtime_adapter
from scripts.experiment_execution.development_exploration_entrypoint import (
    _base_latent,
    _build_or_verify_package,
    _canonical_bytes,
    _environment_digest,
    _session_runtime_identity,
    _sha256_file,
)


PROTOCOL_PATH = Path("configs/experiments/content_routing_directional_diagnosis.json")
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")


class ContentRoutingDirectionalEntrypointError(RuntimeError):
    """The routing diagnosis worker could not preserve its frozen boundary."""


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
            "candidate_id": "routing_stqr",
            "operator": "content_routing_directional_diagnosis",
            "responsibility_domain": "content_router",
            "model_revision": canonical_digest(
                {
                    "probe_manifest_digest": probe_manifest_digest,
                    "protocol_digest": protocol_digest,
                    "reference_manifest_digest": reference_manifest_digest,
                }
            ),
            "tensor_role": "base_gaussian",
        },
        (8,),
    )
    return "ceg-wm-content-routing-registered:" + stream.domain_digest


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


def _replay_aggregate(records: tuple[DevelopmentScientificRecord, ...]):
    observations = []
    implementation_failures = 0
    resource_failures = 0
    for record in records:
        checked = DevelopmentScientificRecord.from_payload(record.payload())
        if checked.execution_status != "success":
            if checked.failure_class == "implementation_failure":
                implementation_failures += 1
            elif checked.failure_class == "resource_failure":
                resource_failures += 1
            else:
                raise ContentRoutingDirectionalEntrypointError(
                    "probe failure classification drifted"
                )
            continue
        payload = checked.operation_result_payload.get("routing_observation")
        if type(payload) is not dict:
            raise ContentRoutingDirectionalEntrypointError(
                "probe observation payload is unavailable"
            )
        converted = dict(payload)
        rows = converted.get("blind_score_observations")
        if type(rows) is not list:
            raise ContentRoutingDirectionalEntrypointError(
                "probe blind score rows are unavailable"
            )
        converted["blind_score_observations"] = tuple(
            ContentRoutingBlindScoreObservation(**row) for row in rows
        )
        observation = ContentRoutingDirectionalObservation(**converted)
        observation.validate()
        observations.append(observation)
    return aggregate_content_routing_directional_diagnosis(
        observations,
        implementation_failure_count=implementation_failures,
        resource_failure_count=resource_failures,
    )


def execute_content_routing_directional_diagnosis_session(
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
    """Run or resume the fixed 2 operational + 32 fit + 8 probe roster."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_key = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    if not root_key or not hf_token:
        raise ContentRoutingDirectionalEntrypointError(
            "HF_TOKEN and CEG_WM_ROOT_KEY are required"
        )
    protocol, reference_manifest, probe_manifest = (
        load_content_routing_directional_protocol(
            repository / PROTOCOL_PATH,
            repository_root=repository,
        )
    )
    if run_id != protocol.run_id:
        raise ContentRoutingDirectionalEntrypointError("run identity drifted")
    first_entry = reference_manifest.entries[0]
    backend = Sd35PipelineBackend(
        cache_root=cache,
        persistent_root=persistent,
        hf_token=hf_token,
        prompt=first_entry.prompt,
    )
    runtime = create_runtime_adapter(backend, repository / RUNTIME_PATH)
    session = runtime.initialize("cuda")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(repository / COMPONENT_PATH)
    )
    semantic = DevelopmentSemanticObservationProducer(
        cache_root=cache,
        hf_token=hf_token,
        device="cuda:0",
    )
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
        runtime.close()
        raise ContentRoutingDirectionalEntrypointError(
            "routing registered root must differ from the base root"
        )
    candidate_digest = canonical_digest(
        {
            "adapter_config_digest": adapter.configuration.config_digest,
            "mixing_coefficient": protocol.mixing_coefficient,
            "probe_manifest_digest": probe_manifest_digest,
            "reference_manifest_digest": reference_manifest_digest,
            "routing_candidate_identity": protocol.routing_candidate_identity,
            "runtime_config_digest": session.runtime_config_digest,
        }
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
    runner = ContentRoutingDirectionalDiagnosisRunner(
        protocol=protocol,
        reference_manifest=reference_manifest,
        probe_manifest=probe_manifest,
        adapter=adapter,
        runtime_adapter=runtime,
        semantic_producer=semantic,
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
        runtime.close()
        raise ContentRoutingDirectionalEntrypointError(
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
            entry = (
                reference_manifest.entries[unit.source_cluster_ordinal]
                if unit.unit_index < 34
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
                if unit.unit_index < 2:
                    record = runner.execute_operational_unit(
                        unit_index=unit.unit_index,
                        base_latent=base_latent,
                        intent=intent,
                    )
                elif unit.unit_index < 34:
                    record = runner.execute_reference_fit_unit(
                        unit_index=unit.unit_index,
                        base_latent=base_latent,
                        intent=intent,
                    )
                else:
                    if len(cursor.routing_reference_records) != 32:
                        record = runner.create_failed_probe_record(
                            intent=intent,
                            failure_class="implementation_failure",
                            failure_reason="routing_reference_dependency_blocked",
                            elapsed_seconds=float(monotonic() - started),
                        )
                    else:
                        record = runner.execute_probe_unit(
                            unit_index=unit.unit_index,
                            base_latent=base_latent,
                            intent=intent,
                            reference_records=cursor.routing_reference_records,
                        )
            except Exception as exc:
                failure_class = (
                    "resource_failure" if _resource_failure(exc) else "implementation_failure"
                )
                failure_reason = f"{type(exc).__module__}.{type(exc).__qualname__}"
                elapsed = float(monotonic() - started)
                if unit.unit_index < 2:
                    raise
                if unit.unit_index < 34:
                    record = runner.create_failed_reference_record(
                        intent=intent,
                        failure_class=failure_class,
                        failure_reason=failure_reason,
                        elapsed_seconds=elapsed,
                    )
                else:
                    record = runner.create_failed_probe_record(
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
        if cursor.next_unit_index == len(protocol.unit_roster):
            evidence = store.verified_terminal_scientific_evidence(
                now_epoch_seconds=int(time.time())
            )
            records = tuple(record for record, _marker in evidence)
            aggregate = asdict(_replay_aggregate(records))
    except Exception as exc:
        termination_reason = "worker_execution_failure"
        failure = {
            "failure_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "stage": "content_routing_directional_diagnosis_execution",
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
            "content_routing_directional_diagnosis_failure"
            if failure is not None
            else "content_routing_directional_diagnosis_result"
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
        "content_routing_directional_aggregate": aggregate,
        "formal_tau_created": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = ["execute_content_routing_directional_diagnosis_session"]
