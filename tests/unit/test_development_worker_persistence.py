"""CPU constraints for create-only stateless development workers."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from experiments.runners.development_support import replay_branch_null_calibration
from experiments.protocol.development_support import DevelopmentStudyUnit
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    DEVELOPMENT_RECORD_COLLECTION_ROLE,
    DEVELOPMENT_RECORD_MEMBER_PATH,
    RECORD_SCHEMA_VERSION,
    ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
    ROUTING_REFERENCE_RECORD_KIND,
    ROUTING_REFERENCE_RECORD_MEMBER_PATH,
    ROUTING_REFERENCE_RECORD_SCHEMA,
    DevelopmentRoutingReferenceRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)
from experiments.runners.development_persistence import (
    DIAGNOSTIC_SCHEMA_VERSION,
    GPU_MIX_POLICY,
    HARD_SESSION_CAP_SECONDS,
    SOFT_STOP_SECONDS,
    DevelopmentPersistenceError,
    DevelopmentPersistentStore,
    FrozenDevelopmentUnitBinding,
    FrozenWorkerIdentity,
    SessionReceipt,
    canonical_json_bytes,
    create_frozen_development_unit_binding,
)
from scripts.experiment_execution.delivery_support import (
    _session_runtime_identity,
)


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SESSION_ID_FORMAT = "colab-%Y%m%dT%H%M%S%fZ"
SESSION_ID_FORMAT = "colab_%Y%m%dt%H%M%S%fz"


def _utc(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _roster(count: int = 6) -> tuple[DevelopmentStudyUnit, ...]:
    return tuple(
        DevelopmentStudyUnit(
            unit_index=index,
            phase="scientific_breadth",
            responsibility_id="key_schedule",
            source_cluster_ordinal=index,
            content_branch_id="not_applicable_content_branch",
            geometry_case_id="not_applicable_geometry_case",
            maximum_record_attempts=3,
            maximum_duration_seconds=900,
        )
        for index in range(count)
    )


def _protocol_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _bindings(
    roster: tuple[DevelopmentStudyUnit, ...],
) -> tuple[FrozenDevelopmentUnitBinding, ...]:
    bindings = []
    for unit in roster:
        prompt_digest = f"{unit.unit_index + 20:064x}"
        image_lineage_digest = f"{unit.unit_index + 40:064x}"
        key_family_digest = f"{unit.unit_index + 60:064x}"
        generation_seed = 1000 + unit.unit_index
        analysis_identity = AnalysisUnitIdentity(
            unit_id=f"manifest_development_unit_{unit.unit_index:04d}",
            case_id="development_manifest_case",
            source_cluster_id=derive_source_cluster_id(
                prompt_digest=prompt_digest,
                generation_seed=generation_seed,
                image_lineage_digest=image_lineage_digest,
                registered_key_family_digest=key_family_digest,
            ),
            prompt_digest=prompt_digest,
            generation_seed=generation_seed,
            image_lineage_digest=image_lineage_digest,
            registered_key_family_digest=key_family_digest,
        )
        bindings.append(
            create_frozen_development_unit_binding(
                unit,
                analysis_unit_identity=analysis_identity,
                scientific_question_id="registered_identity_separation",
                development_case_id="development_key_identity_separation",
                candidate_identity="registered_key_identity_candidate",
                candidate_config_digest=f"{unit.unit_index + 80:064x}",
            )
        )
    return tuple(bindings)


def _store(
    tmp_path: Path,
    roster: tuple[DevelopmentStudyUnit, ...] | None = None,
) -> DevelopmentPersistentStore:
    roster = _roster() if roster is None else roster
    bindings = _bindings(roster)
    worker = FrozenWorkerIdentity(
        revision="1" * 40,
        protocol_digest="2" * 64,
        execution_intent_authority_digest="3" * 64,
        input_manifest_digest="4" * 64,
        candidate_config_digest="5" * 64,
        unit_roster_digest=_protocol_digest(tuple(asdict(item) for item in roster)),
    )
    return DevelopmentPersistentStore(
        (tmp_path / "persistent").resolve(),
        run_id="development_run",
        worker_identity=worker,
        registered_unit_bindings=bindings,
    )


def _lease(store: DevelopmentPersistentStore, *, session_id: str = "colab_session", start: int = 100, duration: int = 100):
    return store.acquire_lease(
        session_id=session_id,
        now_epoch_seconds=start,
        lease_duration_seconds=duration,
    )


def _intent(store: DevelopmentPersistentStore, lease, *, index: int = 0, now: int = 101, attempt: int = 0, parent: str | None = None):
    return store.create_intent(
        lease,
        unit_id=f"development_unit_{index:04d}",
        unit_index=index,
        attempt_index=attempt,
        parent_attempt_intent_digest=parent,
        now_epoch_seconds=now,
    )


def _commit(store: DevelopmentPersistentStore, lease, intent, *, now: int = 102):
    return store.commit_unit(
        lease,
        intent,
        record=_record(store, intent),
        now_epoch_seconds=now,
    )


def _record(
    store: DevelopmentPersistentStore,
    intent,
    *,
    execution_status: str = "success",
    failure_class: str | None = None,
    failure_reason: str | None = None,
    actual_elapsed_seconds: float = 1.0,
) -> DevelopmentScientificRecord:
    operation_result = {"observation": "registered"} if execution_status == "success" else {}
    success_trace = {
        "raw_detector_identity": "development_detector",
        "rectified_detector_identity": "development_detector",
        "raw_detector_config_digest": "a" * 64,
        "rectified_detector_config_digest": "a" * 64,
        "raw_preprocessing_identity": "rgb8_public_preprocessing",
        "rectified_preprocessing_identity": "rgb8_public_preprocessing",
    }
    metric_payload = {
        "schema_version": "ceg_wm_development_metric_observation_v1",
        "metric_role": "development_exploratory_cluster_level",
        "responsibility_id": intent.responsibility_id,
        "source_cluster_id": intent.analysis_unit_identity["source_cluster_id"],
        "registered_metric_ids": ("registered_identity_separation_metric",),
        "candidate_config_digest": intent.candidate_config_digest,
        "paired_ablation_identity": "registered_wrong_key_ablation",
        "content_branch_id": intent.content_branch_id,
        "geometry_case_id": intent.geometry_case_id,
        "sufficient_statistics": (
            ("registered_identity_separation_metric", 1.0),
        ),
        "result_identity_digests": ("f" * 64,),
        "threshold_role": None,
        "threshold_identity": None,
        "threshold_fit_source_cluster_digest": None,
    }
    metric_payload["observation_digest"] = canonical_development_value_digest(
        metric_payload
    )
    record = DevelopmentScientificRecord(
        schema_version=RECORD_SCHEMA_VERSION,
        collection_role=DEVELOPMENT_RECORD_COLLECTION_ROLE,
        record_id=f"{intent.unit_index + intent.attempt_index + 120:064x}",
        run_id=store.run_id,
        protocol_id="development_module_exploration",
        protocol_version="development_module_exploration_version",
        protocol_digest=store.worker_identity.protocol_digest,
        execution_intent_authority_digest=(
            store.worker_identity.execution_intent_authority_digest
        ),
        method_code_revision=store.worker_identity.revision,
        unit_index=intent.unit_index,
        phase=intent.phase,
        analysis_unit_identity=intent.analysis_unit_identity,
        responsibility_id=intent.responsibility_id,
        scientific_question_id=intent.scientific_question_id,
        development_case_id=intent.development_case_id,
        candidate_identity=intent.candidate_identity,
        candidate_config_digest=intent.candidate_config_digest,
        paired_ablation_identity="registered_wrong_key_ablation",
        negative_control_case_ids=("wrong_key_control",),
        metric_ids=("registered_identity_separation_metric",),
        content_branch_id=intent.content_branch_id,
        geometry_case_id=intent.geometry_case_id,
        attempt_index=intent.attempt_index,
        execution_status=execution_status,
        failure_class=failure_class,
        failure_reason=failure_reason,
        retry_parent_intent_digest=intent.parent_attempt_intent_digest,
        actual_elapsed_seconds=actual_elapsed_seconds,
        maximum_duration_seconds=intent.maximum_duration_seconds,
        duration_limit_exceeded=(
            actual_elapsed_seconds > intent.maximum_duration_seconds
        ),
        operation_result_payload=operation_result,
        operation_result_digest=canonical_development_value_digest(operation_result),
        metric_observation={} if execution_status != "success" else metric_payload,
        routing_trace={},
        branch_score_trace={},
        detector_trace=success_trace if execution_status == "success" else {},
        geometry_trace={},
        threshold_trace=(
            {
                "raw_threshold_identity": "development_threshold",
                "rectified_threshold_identity": "development_threshold",
            }
            if execution_status == "success"
            else {}
        ),
        key_control_trace={},
        decision_trace={"positive_source": None},
        provenance_trace={
            "protocol_digest": store.worker_identity.protocol_digest,
            "execution_intent_authority_digest": (
                store.worker_identity.execution_intent_authority_digest
            ),
            "method_code_revision": store.worker_identity.revision,
            "candidate_config_digest": intent.candidate_config_digest,
        },
        module_outcome=None,
        candidate_recommendation=None,
        scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
    )
    return replace(
        record,
        record_id=canonical_development_value_digest(
            record.payload_without_record_id()
        ),
    )


def _routing_reference_record(
    intent,
) -> DevelopmentRoutingReferenceRecord:
    values = [0.25, 0.5]
    record = DevelopmentRoutingReferenceRecord(
        schema_version=ROUTING_REFERENCE_RECORD_SCHEMA,
        collection_role=ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
        record_kind=ROUTING_REFERENCE_RECORD_KIND,
        record_id="0" * 64,
        run_id=intent.run_id,
        protocol_digest=intent.protocol_digest,
        method_code_revision=intent.revision,
        unit_index=intent.unit_index,
        phase=intent.phase,
        source_cluster_ordinal=intent.source_cluster_ordinal,
        fold_index=intent.source_cluster_ordinal % 4,
        prompt_roster_digest="9" * 64,
        candidate_config_digest=intent.candidate_config_digest,
        attempt_index=intent.attempt_index,
        retry_parent_intent_digest=intent.parent_attempt_intent_digest,
        actual_elapsed_seconds=1.0,
        maximum_duration_seconds=intent.maximum_duration_seconds,
        duration_limit_exceeded=False,
        execution_status="success",
        failure_class=None,
        failure_reason=None,
        measurement_payload={
            "candidate_id": "routing_stqr",
            "runtime_config_digest": "8" * 64,
            "model_id": "registered-development-model",
            "model_revision": "registered-development-revision",
            "callback_indices": list(range(20)),
            "public_probe_domain_digest": "7" * 64,
            "public_probe_values_digest": "6" * 64,
            "nominal_relative_probe_step": 0.001,
            "actual_probe_step": 0.001,
            "texture_gradient_values": values,
            "texture_spatial_shape": [1, 2],
            "response_ratio_values": values,
            "response_spatial_shape": [1, 2],
            "sensitivity_ratio_values": values,
            "sensitivity_spatial_shape": [1, 2],
        },
        counts_as_scientific_coverage=False,
        scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
    )
    return replace(
        record,
        record_id=canonical_development_value_digest(
            record.payload_without_record_id()
        ),
    )


def _receipt(store: DevelopmentPersistentStore, *, session_id: str, start: int, end: int, committed_unit_ids: tuple[str, ...]) -> SessionReceipt:
    return SessionReceipt(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        session_id=session_id,
        run_id=store.run_id,
        started_at_utc=_utc(start),
        ended_at_utc=_utc(end),
        gpu_model="nvidia_l4",
        cuda_identity="cuda_12_8",
        environment_digest="8" * 64,
        revision=store.worker_identity.revision,
        package_sha256="7" * 64,
        walltime_seconds=float(end - start),
        peak_vram_bytes=20_000_000_000,
        termination_reason="soft_stop_after_current_unit",
        soft_stop_seconds=SOFT_STOP_SECONDS,
        hard_session_cap_seconds=HARD_SESSION_CAP_SECONDS,
        gpu_mix_policy=GPU_MIX_POLICY,
        committed_unit_ids=committed_unit_ids,
        public_secret_identity_digests=("9" * 64,),
    )


@pytest.mark.quick
def test_session_identity_crosses_persistent_lease_boundary(
    tmp_path: Path,
) -> None:
    fixed_utc = datetime(2026, 8, 4, 8, 12, 7, 484422, tzinfo=timezone.utc)
    notebook_session_id = fixed_utc.strftime(SESSION_ID_FORMAT)
    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    accepted_store = _store(accepted_root)

    lease = accepted_store.acquire_lease(
        session_id=notebook_session_id,
        now_epoch_seconds=100,
        lease_duration_seconds=100,
    )
    assert lease.session_id == notebook_session_id

    legacy_session_id = fixed_utc.strftime(LEGACY_SESSION_ID_FORMAT)
    rejected_root = tmp_path / "rejected"
    rejected_root.mkdir()
    rejected_store = _store(rejected_root)
    with pytest.raises(
        DevelopmentPersistenceError,
        match="session_id is not a stable identity",
    ):
        rejected_store.acquire_lease(
            session_id=legacy_session_id,
            now_epoch_seconds=100,
            lease_duration_seconds=100,
        )


@pytest.mark.quick
def test_recovery_identity_excludes_transport_artifact_hashes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease)
    _commit(store, lease, intent)
    assert len(store.recover(now_epoch_seconds=201).ledger_digest) == 64
    identity_payload = asdict(store.worker_identity)
    assert "package_sha256" not in identity_payload
    assert "bootstrap_sha256" not in identity_payload


@pytest.mark.quick
def test_intent_contains_complete_registered_unit_binding_and_rejects_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=2)

    assert intent.unit_id == "development_unit_0002"
    assert intent.shard_id == intent.phase == "scientific_breadth"
    assert intent.responsibility_id == "key_schedule"
    assert intent.source_cluster_ordinal == 2
    assert intent.content_branch_id == "not_applicable_content_branch"
    assert intent.geometry_case_id == "not_applicable_geometry_case"
    assert intent.maximum_record_attempts == 3
    assert intent.maximum_duration_seconds == 900
    assert intent.unit_roster_digest == store.worker_identity.unit_roster_digest
    assert intent.analysis_unit_identity["unit_id"] == "manifest_development_unit_0002"
    assert len(intent.analysis_unit_identity_digest) == 64
    assert intent.scientific_question_id == "registered_identity_separation"
    assert intent.development_case_id == "development_key_identity_separation"
    assert intent.candidate_identity == "registered_key_identity_candidate"
    assert len(intent.candidate_config_digest) == 64
    assert len(intent.unit_descriptor_digest) == 64

    with pytest.raises(DevelopmentPersistenceError, match="outside frozen roster"):
        store.create_intent(
            lease,
            unit_id="development_unit_999999",
            unit_index=999999,
            attempt_index=0,
            parent_attempt_intent_digest=None,
            now_epoch_seconds=102,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("shard", "descriptor digest|registered binding"),
        ("analysis_identity", "analysis identity"),
        ("scientific_question", "descriptor digest|registered binding"),
        ("candidate_config", "descriptor digest|registered binding"),
        ("unit_descriptor", "descriptor digest"),
    ),
)
def test_recovery_rejects_registered_scientific_binding_tamper(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease)
    path = store.run_root / "intents" / f"{intent.unit_id}__attempt_0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "shard":
        payload["shard_id"] = "critical_pair_extension"
    elif mutation == "analysis_identity":
        payload["analysis_unit_identity"]["prompt_digest"] = "f" * 64
    elif mutation == "scientific_question":
        payload["scientific_question_id"] = "drifted_scientific_question"
    elif mutation == "candidate_config":
        payload["candidate_config_digest"] = "f" * 64
    else:
        payload["unit_descriptor_digest"] = "f" * 64
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(DevelopmentPersistenceError, match=reason):
        store.recover(now_epoch_seconds=201)


@pytest.mark.quick
def test_create_only_commit_rebuilds_ledger_from_verified_markers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease)
    marker = _commit(store, lease, intent)
    recovered = store.recover(now_epoch_seconds=201)

    assert recovered.committed_units == (marker,)
    assert recovered.interrupted_attempts == ()
    assert recovered.next_attempt_by_unit == ()
    assert len(recovered.ledger_digest) == 64
    assert not (store.run_root / "ledger.json").exists()
    with pytest.raises(DevelopmentPersistenceError, match="committed unit"):
        store.next_attempt_index(intent.unit_id)


@pytest.mark.quick
def test_commit_writes_only_exact_formal_record_at_fixed_member_path(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease)
    marker = _commit(store, lease, intent)
    bundle = store.run_root / "bundles" / f"sha256_{marker.bundle_sha256}.zip"
    with ZipFile(bundle, "r") as archive:
        assert DEVELOPMENT_RECORD_MEMBER_PATH in archive.namelist()
        payload = json.loads(archive.read(DEVELOPMENT_RECORD_MEMBER_PATH))
    assert payload["unit_index"] == intent.unit_index
    assert marker.record_digest == sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    assert marker.attempt_disposition == "success"
    verified_records = store.verified_terminal_scientific_records(
        now_epoch_seconds=201
    )
    assert len(verified_records) == 1
    assert verified_records[0].record_id == payload["record_id"]
    verified_evidence = store.verified_terminal_scientific_evidence(
        now_epoch_seconds=201
    )
    assert verified_evidence == ((verified_records[0], marker),)
    assert verified_evidence[0][1].digest() == marker.digest()
    assert store.verified_terminal_scientific_evidence_for_unit_indexes(
        (0,),
        now_epoch_seconds=201,
    ) == verified_evidence
    with pytest.raises(
        DevelopmentPersistenceError,
        match="lack terminal COMMITTED evidence",
    ):
        store.verified_terminal_scientific_evidence_for_unit_indexes(
            (0, 1),
            now_epoch_seconds=201,
        )


@pytest.mark.quick
def test_commit_rejects_pseudo_json_and_record_identity_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease)
    with pytest.raises(DevelopmentPersistenceError, match="exact type"):
        store.commit_unit(
            lease,
            intent,
            record={"execution_status": "success"},  # type: ignore[arg-type]
            now_epoch_seconds=102,
        )
    with pytest.raises(DevelopmentPersistenceError, match="identity drifted"):
        store.commit_unit(
            lease,
            intent,
            record=replace(_record(store, intent), unit_index=1),
            now_epoch_seconds=102,
        )
    empty_metric_record = replace(
        _record(store, intent),
        metric_observation={},
    )
    empty_metric_record = replace(
        empty_metric_record,
        record_id=canonical_development_value_digest(
            empty_metric_record.payload_without_record_id()
        ),
    )
    with pytest.raises(DevelopmentPersistenceError, match="metric observation"):
        store.commit_unit(
            lease,
            intent,
            record=empty_metric_record,
            now_epoch_seconds=102,
        )
    with pytest.raises(DevelopmentPersistenceError, match="reserved"):
        store.commit_unit(
            lease,
            intent,
            record=_record(store, intent),
            diagnostic_members={DEVELOPMENT_RECORD_MEMBER_PATH: b"{}\n"},
            now_epoch_seconds=102,
        )


@pytest.mark.quick
def test_retryable_failure_and_success_commit_contiguous_attempt_lineage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_lease = _lease(store, session_id="resource_session", duration=10)
    first_intent = _intent(store, first_lease, index=1)
    retry_marker = store.commit_unit(
        first_lease,
        first_intent,
        record=_record(
            store,
            first_intent,
            execution_status="retry",
            failure_class="resource_failure",
            failure_reason="gpu_memory_pressure",
        ),
        now_epoch_seconds=102,
    )
    assert retry_marker.attempt_disposition == "retryable_resource_failure"
    assert store.next_attempt_index(first_intent.unit_id) == 1
    assert store.recover(now_epoch_seconds=111).next_attempt_by_unit == (
        (first_intent.unit_id, 1),
    )

    resumed = _lease(store, session_id="recovery_session", start=111, duration=20)
    second_intent = _intent(
        store,
        resumed,
        index=1,
        now=112,
        attempt=1,
        parent=first_intent.digest(),
    )
    success_marker = _commit(store, resumed, second_intent, now=113)
    recovered = store.recover(now_epoch_seconds=132)
    assert recovered.committed_units == (retry_marker, success_marker)
    assert recovered.next_attempt_by_unit == ()
    with pytest.raises(DevelopmentPersistenceError, match="terminal"):
        store.next_attempt_index(first_intent.unit_id)


@pytest.mark.quick
def test_final_failure_is_committed_and_cannot_be_rerun(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=2)
    marker = store.commit_unit(
        lease,
        intent,
        record=_record(
            store,
            intent,
            execution_status="failed",
            failure_class="implementation_failure",
            failure_reason="registered_operation_failed",
        ),
        now_epoch_seconds=102,
    )
    assert marker.attempt_disposition == "final_failure"
    with pytest.raises(DevelopmentPersistenceError, match="terminal"):
        _intent(store, lease, index=2, now=103, attempt=1, parent=intent.digest())


@pytest.mark.quick
def test_third_attempt_cannot_remain_retryable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store, duration=100)
    parent = None
    latest_intent = None
    for attempt in (0, 1):
        latest_intent = _intent(
            store,
            lease,
            index=4,
            now=101 + attempt * 2,
            attempt=attempt,
            parent=parent,
        )
        store.commit_unit(
            lease,
            latest_intent,
            record=_record(
                store,
                latest_intent,
                execution_status="retry",
                failure_class="resource_failure",
                failure_reason="transient_resource_failure",
            ),
            now_epoch_seconds=102 + attempt * 2,
        )
        parent = latest_intent.digest()
    assert latest_intent is not None
    last = _intent(
        store,
        lease,
        index=4,
        now=105,
        attempt=2,
        parent=latest_intent.digest(),
    )
    with pytest.raises(DevelopmentPersistenceError, match="cannot remain retryable"):
        store.commit_unit(
            lease,
            last,
            record=_record(
                store,
                last,
                execution_status="retry",
                failure_class="resource_failure",
                failure_reason="resource_budget_exhausted",
            ),
            now_epoch_seconds=106,
        )


@pytest.mark.quick
def test_recovery_rejects_orphan_bundle_not_uniquely_referenced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    _commit(store, lease, _intent(store, lease))
    orphan = store.run_root / "bundles" / f"sha256_{'f' * 64}.zip"
    orphan.write_bytes(b"orphan")
    with pytest.raises(DevelopmentPersistenceError, match="orphan|unreferenced"):
        store.recover(now_epoch_seconds=201)


@pytest.mark.quick
def test_record_elapsed_time_and_frozen_duration_are_validated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=3)
    with pytest.raises(DevelopmentPersistenceError, match="duration limit status"):
        store.commit_unit(
            lease,
            intent,
            record=replace(
                _record(store, intent),
                actual_elapsed_seconds=901.0,
                duration_limit_exceeded=False,
            ),
            now_epoch_seconds=102,
        )


@pytest.mark.quick
def test_only_expired_or_closed_leaf_intent_becomes_interrupted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_lease = _lease(store, session_id="lost_session", duration=10)
    first = _intent(store, first_lease, index=1)

    active = store.recover(now_epoch_seconds=105)
    assert active.interrupted_attempts == ()
    assert active.next_attempt_by_unit == ()

    recovered = store.recover(now_epoch_seconds=111)
    interrupted = recovered.interrupted_attempts[0]
    assert interrupted.failure_class == "resource_failure"
    assert interrupted.failure_reason == "colab_session_interrupted"
    assert interrupted.retry_parent_intent_digest == first.digest()
    assert recovered.next_attempt_by_unit == ((first.unit_id, 1),)

    resumed = _lease(store, session_id="resumed_session", start=111, duration=20)
    second = _intent(
        store,
        resumed,
        index=1,
        now=112,
        attempt=1,
        parent=first.digest(),
    )
    marker = _commit(store, resumed, second, now=113)
    assert marker.parent_attempt_intent_digest == first.digest()
    assert store.recover(now_epoch_seconds=132).committed_units == (marker,)
    assert store.recover(now_epoch_seconds=132).interrupted_attempts == ()


@pytest.mark.quick
def test_routing_reference_reuses_common_interruption_commit_and_recovery(
    tmp_path: Path,
) -> None:
    roster = (
        DevelopmentStudyUnit(
            unit_index=0,
            phase=ROUTING_REFERENCE_RECORD_KIND,
            responsibility_id="content_router",
            source_cluster_ordinal=0,
            content_branch_id=ROUTING_REFERENCE_RECORD_KIND,
            geometry_case_id="geometry_case_not_applicable",
            maximum_record_attempts=3,
            maximum_duration_seconds=900,
        ),
    )
    store = _store(tmp_path, roster)
    lost_lease = _lease(
        store,
        session_id="routing_reference_lost_session",
        duration=10,
    )
    first = _intent(store, lost_lease, now=101)
    interrupted = store.recover(now_epoch_seconds=111)
    assert interrupted.next_attempt_by_unit == ((first.unit_id, 1),)
    assert interrupted.interrupted_attempts[0].failure_reason == (
        "colab_session_interrupted"
    )

    resumed_lease = _lease(
        store,
        session_id="routing_reference_resumed_session",
        start=111,
        duration=20,
    )
    second = _intent(
        store,
        resumed_lease,
        now=112,
        attempt=1,
        parent=first.digest(),
    )
    marker = store.commit_unit(
        resumed_lease,
        second,
        record=_routing_reference_record(second),
        now_epoch_seconds=113,
    )

    records = store.verified_terminal_routing_reference_records(
        now_epoch_seconds=114
    )
    assert len(records) == 1
    assert records[0].measurement_payload["candidate_id"] == "routing_stqr"
    assert marker.record_kind == ROUTING_REFERENCE_RECORD_KIND
    bundle = store.run_root / "bundles" / f"sha256_{marker.bundle_sha256}.zip"
    with ZipFile(bundle) as archive:
        assert ROUTING_REFERENCE_RECORD_MEMBER_PATH in archive.namelist()
        assert DEVELOPMENT_RECORD_MEMBER_PATH not in archive.namelist()
    with pytest.raises(DevelopmentPersistenceError, match="terminal"):
        store.next_attempt_index(second.unit_id)


@pytest.mark.quick
def test_routing_reference_resumes_across_two_sessions_without_duplicate_units(
    tmp_path: Path,
) -> None:
    roster = tuple(
        DevelopmentStudyUnit(
            unit_index=index,
            phase=ROUTING_REFERENCE_RECORD_KIND,
            responsibility_id="content_router",
            source_cluster_ordinal=index,
            content_branch_id=ROUTING_REFERENCE_RECORD_KIND,
            geometry_case_id="geometry_case_not_applicable",
            maximum_record_attempts=3,
            maximum_duration_seconds=900,
        )
        for index in range(64)
    )
    store = _store(tmp_path, roster)
    first_lease = _lease(
        store,
        session_id="routing_reference_prefix_session",
        start=100,
        duration=10,
    )
    first_cursor = store.open_session_cursor(first_lease, now_epoch_seconds=100)
    for offset in range(17):
        intent = store.create_session_intent(
            first_cursor,
            first_lease,
            now_epoch_seconds=101,
        )
        store.commit_session_unit(
            first_cursor,
            first_lease,
            intent,
            record=_routing_reference_record(intent),
            now_epoch_seconds=102,
        )
    assert first_cursor.next_unit_index == 17

    second_lease = _lease(
        store,
        session_id="routing_reference_completion_session",
        start=111,
        duration=100,
    )
    second_cursor = store.open_session_cursor(second_lease, now_epoch_seconds=111)
    assert second_cursor.next_unit_index == 17
    assert tuple(item.source_cluster_ordinal for item in second_cursor.routing_reference_records) == tuple(range(17))
    for _ in range(17, 64):
        intent = store.create_session_intent(
            second_cursor,
            second_lease,
            now_epoch_seconds=112,
        )
        store.commit_session_unit(
            second_cursor,
            second_lease,
            intent,
            record=_routing_reference_record(intent),
            now_epoch_seconds=113,
        )

    recovered = store.open_session_cursor(second_lease, now_epoch_seconds=114)
    records = recovered.routing_reference_records
    assert recovered.next_unit_index == 64
    assert len(records) == 64
    assert tuple(item.unit_index for item in records) == tuple(range(64))
    assert len({item.record_id for item in records}) == 64


@pytest.mark.quick
def test_content_null_replay_excludes_current_cross_fit_fold() -> None:
    bindings = _bindings(_roster(8))
    ordinals = {
        item.analysis_unit_identity.source_cluster_id: item.unit_index
        for item in bindings
    }
    evidence = tuple(
        (
            SimpleNamespace(
                responsibility_id="hf_detector",
                content_branch_id="clean_control",
                execution_status="success",
                operation_result_payload={
                    "hf_score": float(item.unit_index),
                    "detector_identity": "registered_hf_detector",
                },
                analysis_unit_identity=asdict(item.analysis_unit_identity),
                unit_index=item.unit_index,
            ),
            SimpleNamespace(record_id=f"{item.unit_index:064x}"),
        )
        for item in bindings
    )
    current = bindings[0].analysis_unit_identity.source_cluster_id

    calibration = replay_branch_null_calibration(
        evidence,
        branch="hf",
        current_source_cluster_id=current,
        source_cluster_ordinals=ordinals,
    )

    assert calibration.partition_identity.endswith("fold_0")
    assert {record.source_cluster_id for record in calibration.records} == {
        item.analysis_unit_identity.source_cluster_id
        for item in bindings
        if item.unit_index % 4 != 0
    }
    assert current not in {record.source_cluster_id for record in calibration.records}


@pytest.mark.quick
def test_session_cursor_recovers_once_then_commits_units_incrementally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path, _roster(2))
    lease = _lease(store, duration=100)
    recover_calls = 0
    original_recover = store.recover

    def counted_recover(*, now_epoch_seconds: int | None = None):
        nonlocal recover_calls
        recover_calls += 1
        return original_recover(now_epoch_seconds=now_epoch_seconds)

    monkeypatch.setattr(store, "recover", counted_recover)
    cursor = store.open_session_cursor(lease, now_epoch_seconds=100)

    for expected_index in range(2):
        assert cursor.next_unit_index == expected_index
        intent = store.create_session_intent(
            cursor,
            lease,
            now_epoch_seconds=101 + expected_index,
        )
        marker = store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=_record(store, intent),
            now_epoch_seconds=102 + expected_index,
        )
        assert marker.unit_index == expected_index

    assert cursor.next_unit_index == 2
    assert len(cursor.committed_units) == 2
    assert len(cursor.terminal_scientific_evidence) == 2
    assert recover_calls == 1


@pytest.mark.quick
def test_single_writer_lease_soft_stop_and_stale_fence_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _lease(store, session_id="first_session", duration=10)
    with pytest.raises(DevelopmentPersistenceError, match="active writer"):
        _lease(store, session_id="second_session", start=105, duration=10)
    second = _lease(store, session_id="second_session", start=110, duration=HARD_SESSION_CAP_SECONDS - 1)
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(DevelopmentPersistenceError, match="stale fencing"):
        _intent(store, first, index=2, now=111)
    _intent(store, second, index=2, now=110 + SOFT_STOP_SECONDS - 1)
    with pytest.raises(DevelopmentPersistenceError, match="soft stop"):
        _intent(store, second, index=3, now=110 + SOFT_STOP_SECONDS)


@pytest.mark.quick
def test_valid_session_receipt_closes_unexpired_lease_for_immediate_resume(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = _lease(
        store,
        session_id="completed_session",
        start=100,
        duration=100,
    )
    marker = _commit(store, first, _intent(store, first), now=102)
    store.write_session_receipt(
        _receipt(
            store,
            session_id=first.session_id,
            start=100,
            end=102,
            committed_unit_ids=(marker.unit_id,),
        )
    )

    with pytest.raises(DevelopmentPersistenceError, match="active writer"):
        _lease(
            store,
            session_id="premature_session",
            start=101,
            duration=100,
        )

    second = _lease(
        store,
        session_id="resumed_session",
        start=102,
        duration=100,
    )
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(DevelopmentPersistenceError, match="stale fencing"):
        _intent(store, first, index=1, now=103)


@pytest.mark.quick
def test_invalid_session_receipt_does_not_close_unexpired_lease(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lease = _lease(
        store,
        session_id="invalid_receipt_session",
        start=100,
        duration=100,
    )
    marker = _commit(store, lease, _intent(store, lease), now=102)
    invalid_receipt = replace(
        _receipt(
            store,
            session_id=lease.session_id,
            start=100,
            end=102,
            committed_unit_ids=(marker.unit_id,),
        ),
        walltime_seconds=1.0,
    )
    receipt_path = store.run_root / "receipts" / f"{lease.session_id}.json"
    receipt_path.write_bytes(canonical_json_bytes(asdict(invalid_receipt)))

    with pytest.raises(DevelopmentPersistenceError, match="active writer"):
        _lease(
            store,
            session_id="blocked_resume_session",
            start=103,
            duration=100,
        )
    assert tuple(path.name for path in (store.run_root / "leases").iterdir()) == (
        "fence_00000001.json",
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    "member_path",
    (
        "../escape.json",
        "/absolute.json",
        "nested\\windows.json",
        "nested/../escape.json",
        "payload.py",
        "payload.PYC",
        "payload.sh",
        "payload.so",
        "payload.exe",
        "payload.pkl",
    ),
)
def test_unsafe_or_executable_bundle_members_are_rejected(tmp_path: Path, member_path: str) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=3)
    with pytest.raises(DevelopmentPersistenceError, match="path|deserialization"):
        store.commit_unit(
            lease,
            intent,
            record=_record(store, intent),
            diagnostic_members={member_path: b"payload"},
            now_epoch_seconds=102,
        )


@pytest.mark.quick
def test_bundle_member_identities_are_unique_under_casefold(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=4)
    with pytest.raises(DevelopmentPersistenceError, match="casefold"):
        store.commit_unit(
            lease,
            intent,
            record=_record(store, intent),
            diagnostic_members={
                "records/Result.json": b"one",
                "records/result.JSON": b"two",
            },
            now_epoch_seconds=102,
        )


@pytest.mark.quick
def test_secret_sentinel_never_reaches_bundle_or_receipt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    intent = _intent(store, lease, index=4)
    with pytest.raises(DevelopmentPersistenceError, match="secret"):
        store.commit_unit(
            lease,
            intent,
            record=_record(store, intent),
            diagnostic_members={
                "diagnostics/secret.json": b'{"token":"raw-secret-sentinel"}\n'
            },
            raw_secret_values=("raw-secret-sentinel",),
            now_epoch_seconds=102,
        )
    assert not tuple((store.run_root / "bundles").iterdir())
    assert not tuple((store.run_root / "markers").iterdir())


@pytest.mark.quick
def test_session_receipt_is_strict_and_matches_lease_and_committed_markers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    marker = _commit(store, lease, _intent(store, lease), now=102)
    receipt = _receipt(
        store,
        session_id=lease.session_id,
        start=100,
        end=102,
        committed_unit_ids=(marker.unit_id,),
    )
    path = store.write_session_receipt(receipt)
    assert json.loads(path.read_text(encoding="utf-8"))["soft_stop_seconds"] == 75600
    assert store.recover(now_epoch_seconds=102).committed_units == (marker,)

    mutations = (
        (replace(receipt, session_id="missing_session"), "lease lineage"),
        (replace(receipt, started_at_utc=_utc(99)), "start does not match lease"),
        (replace(receipt, ended_at_utc=_utc(99), walltime_seconds=-1.0), "UTC order|hard cap"),
        (replace(receipt, walltime_seconds=1.0), "walltime and UTC interval"),
        (replace(receipt, gpu_model=""), "GPU model"),
        (replace(receipt, cuda_identity=""), "CUDA identity"),
        (replace(receipt, peak_vram_bytes=0), "peak VRAM"),
        (replace(receipt, termination_reason=""), "termination reason"),
        (replace(receipt, committed_unit_ids=()), "differ from markers"),
    )
    for mutated, reason in mutations:
        with pytest.raises(DevelopmentPersistenceError, match=reason):
            store.write_session_receipt(mutated)


@pytest.mark.quick
def test_runtime_display_metadata_is_normalized_before_session_receipt_persistence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    receipt = replace(
        _receipt(
            store,
            session_id=lease.session_id,
            start=100,
            end=101,
            committed_unit_ids=(),
        ),
        gpu_model=_session_runtime_identity(
            role="gpu",
            display_value="NVIDIA L4",
        ),
        cuda_identity=_session_runtime_identity(
            role="cuda",
            display_value="12.8",
        ),
    )

    path = store.write_session_receipt(receipt)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["gpu_model"] == "gpu_nvidia_l4"
    assert persisted["cuda_identity"] == "cuda_12_8"

    rejected_root = tmp_path / "rejected"
    rejected_root.mkdir()
    rejected_store = _store(rejected_root)
    rejected_lease = _lease(rejected_store)
    rejected_receipt = replace(
        _receipt(
            rejected_store,
            session_id=rejected_lease.session_id,
            start=100,
            end=101,
            committed_unit_ids=(),
        ),
        gpu_model="NVIDIA_L4",
        cuda_identity="12_8",
    )
    with pytest.raises(
        DevelopmentPersistenceError,
        match="session GPU model is not a stable identity",
    ):
        rejected_store.write_session_receipt(rejected_receipt)


@pytest.mark.quick
def test_recovery_rejects_deleted_lease_and_marker_lineage_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _lease(store)
    _commit(store, lease, _intent(store, lease))
    lease_path = store.run_root / "leases" / "fence_00000001.json"
    lease_path.unlink()
    with pytest.raises(DevelopmentPersistenceError, match="lease/session/fence lineage"):
        store.recover(now_epoch_seconds=201)
