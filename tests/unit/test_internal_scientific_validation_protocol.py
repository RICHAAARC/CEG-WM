"""内部科学验证协议的 CPU schema 与约束测试。"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.protocol.internal_matrix import (
    REQUIRED_METHOD_RESPONSIBILITIES,
    RESPONSIBILITY_VALIDATION_MATRIX,
    decide_split_promotion,
    validate_responsibility_matrix,
)
from experiments.protocol.internal_records import (
    BranchScoreTrace,
    DecisionTrace,
    DetectorTrace,
    GeometryTrace,
    InternalValidationRecord,
    INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION,
    INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
    KeyControlTrace,
    MAXIMUM_RECORD_ATTEMPTS,
    PromotionGateAssessment,
    ProvenanceTrace,
    RunCaseRecordCollection,
    RoutingTrace,
    ThresholdTrace,
    validate_internal_record,
    validate_run_case_record_collection,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    CURRENT_EXECUTION_ALLOWED_SPLITS,
    FrozenSplitManifest,
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
    INTERNAL_VALIDATION_SPLITS,
    SplitAccessGrant,
    SplitAssignment,
    authorize_split_access,
    derive_source_cluster_id,
)
from experiments.protocol.internal_validation import (
    load_frozen_internal_validation_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/experiments/internal_scientific_validation_protocol.json"
PROTOCOL_MODULES = (
    ROOT / "experiments/protocol/internal_splits.py",
    ROOT / "experiments/protocol/internal_matrix.py",
    ROOT / "experiments/protocol/internal_records.py",
    ROOT / "experiments/protocol/internal_validation.py",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "main",
    "runtime",
    "experiments.methods",
    "experiments.attacks",
    "experiments.metrics",
    "experiments.runners",
    "governance",
)


def _unit(index: int) -> AnalysisUnitIdentity:
    prompt_digest = f"{index + 1:064x}"
    image_lineage_digest = f"{index + 101:064x}"
    key_family_digest = f"{index + 201:064x}"
    cluster_id = derive_source_cluster_id(
        prompt_digest=prompt_digest,
        generation_seed=index,
        image_lineage_digest=image_lineage_digest,
        registered_key_family_digest=key_family_digest,
    )
    return AnalysisUnitIdentity(
        unit_id=f"unit_{index}",
        case_id=f"case_{index}",
        source_cluster_id=cluster_id,
        prompt_digest=prompt_digest,
        generation_seed=index,
        image_lineage_digest=image_lineage_digest,
        registered_key_family_digest=key_family_digest,
    )


def _manifest() -> FrozenSplitManifest:
    return FrozenSplitManifest(
        protocol_id=INTERNAL_VALIDATION_PROTOCOL_ID,
        protocol_version=INTERNAL_VALIDATION_PROTOCOL_VERSION,
        manifest_id="frozen_split_manifest_test",
        manifest_revision="manifest_revision_1",
        assignments=tuple(
            SplitAssignment(identity=_unit(index), split=split_name)
            for index, split_name in enumerate(INTERNAL_VALIDATION_SPLITS)
        ),
    )


def _record(**changes: object) -> InternalValidationRecord:
    values: dict[str, object] = {
        "record_id": "record_1",
        "run_id": "run_1",
        "protocol_id": INTERNAL_VALIDATION_PROTOCOL_ID,
        "protocol_version": INTERNAL_VALIDATION_PROTOCOL_VERSION,
        "record_schema_version": INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
        "analysis_unit_identity": _unit(0),
        "split": "end_to_end_check",
        "record_sequence_index": 0,
        "record_attempt_index": 0,
        "execution_status": "success",
        "failure_reason": None,
        "exclusion_reason": None,
        "exclusion_rule_id": None,
        "retry_of_record_id": None,
        "detector_trace": DetectorTrace(
            raw_detector_identity="content_detector_identity_1",
            rectified_detector_identity="content_detector_identity_1",
            raw_detector_config_digest="b" * 64,
            rectified_detector_config_digest="b" * 64,
            raw_preprocessing_identity="preprocessing_identity_1",
            rectified_preprocessing_identity="preprocessing_identity_1",
            raw_content_score=0.4,
            rectified_content_score=None,
        ),
        "branch_score_trace": BranchScoreTrace(
            lf_score=0.2,
            hf_score=0.4,
            combined_score=0.35,
        ),
        "routing_trace": RoutingTrace(
            routing_identity="routing_stqr",
            routing_control="routed",
            routing_observation_digest="1" * 64,
            routing_mask_digest="2" * 64,
        ),
        "geometry_trace": GeometryTrace(
            geometry_triggered=False,
            geometry_estimation_identity=None,
            geometry_reliability_identity=None,
            geometry_reliable=None,
            geometry_transform=None,
            geometry_raw_metrics=None,
            geometry_failure_reason=None,
            rectification_status="not_attempted",
        ),
        "threshold_trace": ThresholdTrace(
            raw_threshold_identity="threshold_identity_1",
            rectified_threshold_identity="threshold_identity_1",
            tau=0.8,
            tau_rescue=0.6,
        ),
        "key_control_trace": KeyControlTrace(
            registered_key_public_digest="3" * 64,
            detection_key_public_digest="3" * 64,
            key_role="registered",
            control_identity="registered_key_control",
        ),
        "decision_trace": DecisionTrace(
            watermark_decision="negative",
            positive_source=None,
            decision_reason="raw_below_rescue_threshold",
        ),
        "provenance_trace": ProvenanceTrace(
            protocol_digest="4" * 64,
            split_manifest_digest="5" * 64,
            method_code_revision="method_revision_1",
            method_config_digest="6" * 64,
            model_revision="model_revision_1",
            environment_digest="7" * 64,
            input_artifact_digest="8" * 64,
            attack_config_digest="9" * 64,
            metric_set_digest="a" * 64,
        ),
    }
    values.update(changes)
    return InternalValidationRecord(**values)


def _collection(
    *,
    records: tuple[InternalValidationRecord, ...] | None = None,
    promotion_gate_assessments: tuple[PromotionGateAssessment, ...] = (),
    promotion_stop_gate_id: str | None = None,
    **changes: object,
) -> RunCaseRecordCollection:
    values: dict[str, object] = {
        "record_collection_schema_version": (
            INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
        ),
        "run_id": "run_1",
        "case_id": "case_0",
        "protocol_id": INTERNAL_VALIDATION_PROTOCOL_ID,
        "protocol_version": INTERNAL_VALIDATION_PROTOCOL_VERSION,
        "record_schema_version": INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
        "maximum_record_attempts": MAXIMUM_RECORD_ATTEMPTS,
        "records": records if records is not None else (_record(),),
        "promotion_gate_assessments": promotion_gate_assessments,
        "promotion_stop_gate_id": promotion_stop_gate_id,
    }
    values.update(changes)
    return RunCaseRecordCollection(**values)


@pytest.mark.unit
def test_frozen_protocol_config_has_exact_splits_and_denies_held_out_access() -> None:
    protocol = load_frozen_internal_validation_protocol(CONFIG_PATH)
    assert protocol.splits == INTERNAL_VALIDATION_SPLITS
    assert protocol.validate() == ()
    assert protocol.maximum_record_attempts == MAXIMUM_RECORD_ATTEMPTS
    assert protocol.scientific_claim_boundary.endswith("no_scientific_validity_claim")
    with pytest.raises(PermissionError, match="held_out_evaluation"):
        authorize_split_access(
            _manifest(),
            ("held_out_evaluation",),
            SplitAccessGrant.current_execution(),
        )


@pytest.mark.unit
def test_forged_or_extended_split_access_grants_are_rejected() -> None:
    forged_identity = SplitAccessGrant(
        access_identity="forged_current_execution_authority",
        allowed_splits=CURRENT_EXECUTION_ALLOWED_SPLITS,
    )
    expanded = SplitAccessGrant(
        access_identity=SplitAccessGrant.current_execution().access_identity,
        allowed_splits=frozenset(INTERNAL_VALIDATION_SPLITS),
    )
    for grant in (forged_identity, expanded):
        with pytest.raises(PermissionError, match="split_access_grant_not_current_authority"):
            authorize_split_access(_manifest(), ("development",), grant)


@pytest.mark.unit
def test_split_manifest_is_stable_and_keeps_source_clusters_disjoint() -> None:
    manifest = _manifest()
    assert manifest.validate() == ()
    assert manifest.digest() == manifest.digest()
    leaked = replace(
        manifest.assignments[1],
        identity=replace(
            manifest.assignments[1].identity,
            source_cluster_id=manifest.assignments[0].identity.source_cluster_id,
        ),
    )
    bad_manifest = replace(
        manifest,
        assignments=(manifest.assignments[0], leaked, *manifest.assignments[2:]),
    )
    violations = bad_manifest.validate()
    assert "source_cluster_id_identity_mismatch" in violations
    assert "source_cluster_split_leakage" in violations


@pytest.mark.unit
def test_internal_record_contains_all_scientific_trace_groups() -> None:
    record = _record()
    assert validate_internal_record(record) == ()
    assert set(record.to_dict()) >= {
        "analysis_unit_identity",
        "detector_trace",
        "branch_score_trace",
        "routing_trace",
        "geometry_trace",
        "threshold_trace",
        "key_control_trace",
        "decision_trace",
        "provenance_trace",
    }


@pytest.mark.unit
def test_internal_record_protocol_version_is_required_and_frozen() -> None:
    missing = replace(_record(), protocol_version="")
    drifted = replace(_record(), protocol_version="1.0.1")
    assert "protocol_version_missing" in validate_internal_record(missing)
    assert "protocol_version_frozen_identity_mismatch" in validate_internal_record(missing)
    assert "protocol_version_frozen_identity_mismatch" in validate_internal_record(drifted)


@pytest.mark.unit
def test_success_failed_excluded_and_retry_semantics_are_mutually_exclusive() -> None:
    success = _record()
    failed = replace(
        success,
        execution_status="failed",
        failure_reason="runtime_failure",
        decision_trace=DecisionTrace("failed", None, "runtime_failure"),
    )
    excluded = replace(
        success,
        execution_status="excluded",
        exclusion_reason="input_corrupt_before_method_execution",
        exclusion_rule_id="predeclared_input_integrity_rule",
        decision_trace=DecisionTrace("excluded", None, "predeclared_exclusion"),
    )
    retry = replace(
        success,
        execution_status="retry",
        record_attempt_index=1,
        failure_reason="retryable_resource_failure",
        retry_of_record_id=success.record_id,
        decision_trace=DecisionTrace("retry", None, "retryable_resource_failure"),
    )
    assert validate_internal_record(failed) == ()
    assert validate_internal_record(excluded) == ()
    assert validate_internal_record(retry) == ()
    invalid_retry = replace(retry, record_attempt_index=0)
    assert "retry_record_attempt_index_must_be_positive" in validate_internal_record(
        invalid_retry
    )


@pytest.mark.unit
def test_raw_rectified_identity_and_threshold_must_be_identical() -> None:
    record = _record()
    mismatched = replace(
        record,
        threshold_trace=replace(
            record.threshold_trace,
            rectified_threshold_identity="different_threshold",
        ),
        detector_trace=replace(
            record.detector_trace,
            rectified_detector_identity="different_detector",
            rectified_detector_config_digest="c" * 64,
        ),
    )
    violations = validate_internal_record(mismatched)
    assert "raw_rectified_detector_identity_mismatch" in violations
    assert "raw_rectified_detector_config_digest_mismatch" in violations
    assert "raw_rectified_threshold_identity_mismatch" in violations


@pytest.mark.unit
def test_geometry_cannot_be_a_positive_source() -> None:
    record = _record(
        decision_trace=DecisionTrace("positive", "geometry", "geometry_confidence"),
    )
    assert "positive_content_source_missing" in validate_internal_record(record)


@pytest.mark.unit
def test_rescue_positive_requires_near_threshold_reliable_geometry_and_same_tau() -> None:
    base = _record()
    rescued = replace(
        base,
        detector_trace=replace(
            base.detector_trace,
            raw_content_score=0.7,
            rectified_content_score=0.85,
        ),
        geometry_trace=GeometryTrace(
            geometry_triggered=True,
            geometry_estimation_identity="estimation_identity_1",
            geometry_reliability_identity="reliability_identity_1",
            geometry_reliable=True,
            geometry_transform={"rotation_degrees": 3.0, "scale": 1.0},
            geometry_raw_metrics={"coverage": 0.8, "gap": 0.2},
            geometry_failure_reason=None,
            rectification_status="succeeded",
        ),
        decision_trace=DecisionTrace(
            "positive",
            "rectified_content",
            "same_detector_rectified_score_reached_tau",
        ),
    )
    assert validate_internal_record(rescued) == ()
    unreliable = replace(
        rescued,
        geometry_trace=replace(rescued.geometry_trace, geometry_reliable=False),
    )
    violations = validate_internal_record(unreliable)
    assert "unreliable_geometry_rectification_forbidden" in violations
    assert "rectified_positive_requirements_not_met" in violations


@pytest.mark.unit
def test_successful_negative_cannot_hide_raw_or_valid_rectified_threshold_crossing() -> None:
    raw_crossing = replace(
        _record(),
        detector_trace=replace(_record().detector_trace, raw_content_score=0.8),
    )
    assert "negative_raw_score_reached_tau" in validate_internal_record(raw_crossing)

    base = _record()
    rectified_crossing = replace(
        base,
        detector_trace=replace(
            base.detector_trace,
            raw_content_score=0.7,
            rectified_content_score=0.85,
        ),
        geometry_trace=GeometryTrace(
            geometry_triggered=True,
            geometry_estimation_identity="estimation_identity_1",
            geometry_reliability_identity="reliability_identity_1",
            geometry_reliable=True,
            geometry_transform={"rotation_degrees": 3.0, "scale": 1.0},
            geometry_raw_metrics={"coverage": 0.8, "gap": 0.2},
            geometry_failure_reason=None,
            rectification_status="succeeded",
        ),
    )
    assert "negative_rectified_score_reached_tau" in validate_internal_record(
        rectified_crossing
    )


def _failed_initial_record() -> InternalValidationRecord:
    return replace(
        _record(),
        execution_status="failed",
        failure_reason="retryable_resource_failure",
        decision_trace=DecisionTrace("failed", None, "retryable_resource_failure"),
    )


def _retry_record(
    parent: InternalValidationRecord,
    *,
    record_id: str,
    sequence_index: int,
    attempt_index: int,
) -> InternalValidationRecord:
    return replace(
        parent,
        record_id=record_id,
        record_sequence_index=sequence_index,
        record_attempt_index=attempt_index,
        execution_status="retry",
        retry_of_record_id=parent.record_id,
        decision_trace=DecisionTrace("retry", None, "retryable_resource_failure"),
    )


@pytest.mark.unit
def test_record_collection_rejects_orphan_cross_case_and_non_failed_retry_parent() -> None:
    failed = _failed_initial_record()
    retry = _retry_record(
        failed,
        record_id="record_retry_1",
        sequence_index=1,
        attempt_index=1,
    )
    assert validate_run_case_record_collection(
        _collection(records=(failed, retry))
    ) == ()

    orphan = replace(retry, retry_of_record_id="missing_parent")
    assert "retry_parent_record_missing" in validate_run_case_record_collection(
        _collection(records=(failed, orphan))
    )

    cross_case = replace(
        retry,
        analysis_unit_identity=replace(retry.analysis_unit_identity, case_id="different_case"),
    )
    cross_case_violations = validate_run_case_record_collection(
        _collection(records=(failed, cross_case))
    )
    assert "retry_parent_identity_mismatch" in cross_case_violations
    assert "record_case_id_collection_mismatch" in cross_case_violations

    successful_parent = _record()
    retry_after_success = _retry_record(
        successful_parent,
        record_id="record_retry_after_success",
        sequence_index=1,
        attempt_index=1,
    )
    assert "retry_parent_status_invalid" in validate_run_case_record_collection(
        _collection(records=(successful_parent, retry_after_success))
    )


@pytest.mark.unit
def test_record_collection_rejects_skipped_or_unbounded_retry_attempts() -> None:
    failed = _failed_initial_record()
    skipped = _retry_record(
        failed,
        record_id="record_retry_skipped",
        sequence_index=1,
        attempt_index=2,
    )
    skipped_violations = validate_run_case_record_collection(
        _collection(records=(failed, skipped))
    )
    assert "record_attempt_index_not_contiguous" in skipped_violations
    assert "retry_parent_attempt_not_contiguous" in skipped_violations

    retry_1 = _retry_record(
        failed,
        record_id="record_retry_1",
        sequence_index=1,
        attempt_index=1,
    )
    retry_2 = _retry_record(
        retry_1,
        record_id="record_retry_2",
        sequence_index=2,
        attempt_index=2,
    )
    retry_3 = _retry_record(
        retry_2,
        record_id="record_retry_3",
        sequence_index=3,
        attempt_index=3,
    )
    over_limit = validate_run_case_record_collection(
        _collection(records=(failed, retry_1, retry_2, retry_3))
    )
    assert "maximum_record_attempts_exceeded" in over_limit
    assert "record_attempt_index_exceeds_frozen_limit" in over_limit


@pytest.mark.unit
def test_record_collection_requires_structured_stop_and_rejects_continuation() -> None:
    failed = _failed_initial_record()
    failed_gate = PromotionGateAssessment(
        gate_id="content_branch_promotion_gate_passed",
        gate_status="failed",
        evidence_record_ids=(failed.record_id,),
        stop_outcome="content_branch_research_question_closed_negative",
    )
    stopped = _collection(
        records=(failed,),
        promotion_gate_assessments=(failed_gate,),
        promotion_stop_gate_id=failed_gate.gate_id,
    )
    assert validate_run_case_record_collection(stopped) == ()

    continuation = replace(
        _record(),
        record_id="record_after_stop",
        record_sequence_index=1,
        record_attempt_index=1,
    )
    assert "record_continues_after_promotion_stop" in validate_run_case_record_collection(
        _collection(
            records=(failed, continuation),
            promotion_gate_assessments=(failed_gate,),
            promotion_stop_gate_id=failed_gate.gate_id,
        )
    )

    unstructured = replace(
        failed_gate,
        gate_id="arbitrary_nonempty_gate",
        stop_outcome="arbitrary_nonempty_outcome",
    )
    unstructured_violations = validate_run_case_record_collection(
        _collection(
            records=(failed,),
            promotion_gate_assessments=(unstructured,),
            promotion_stop_gate_id=unstructured.gate_id,
        )
    )
    assert "promotion_gate_id_invalid" in unstructured_violations
    assert "failed_promotion_gate_stop_outcome_invalid" in unstructured_violations


@pytest.mark.unit
def test_responsibility_matrix_has_one_complete_row_per_method_responsibility() -> None:
    assert validate_responsibility_matrix() == ()
    assert tuple(
        spec.responsibility for spec in RESPONSIBILITY_VALIDATION_MATRIX
    ) == REQUIRED_METHOD_RESPONSIBILITIES
    for spec in RESPONSIBILITY_VALIDATION_MATRIX:
        assert spec.scientific_question
        assert spec.metrics
        assert spec.negative_controls
        assert spec.promotion_gates
        assert spec.record_fields


@pytest.mark.unit
def test_promotion_stops_when_prerequisite_gate_is_missing() -> None:
    stopped = decide_split_promotion("content_threshold_fit", frozenset())
    assert not stopped.approved
    assert stopped.stop_outcome == "content_branch_research_question_closed_negative"
    approved = decide_split_promotion(
        "end_to_end_check",
        frozenset(
            {
                "content_threshold_gate_passed",
                "rescue_threshold_gate_passed",
                "geometry_reliability_gate_passed",
            }
        ),
    )
    assert approved.approved


@pytest.mark.unit
def test_protocol_modules_do_not_import_method_runtime_or_experiment_execution_layers() -> None:
    for path in PROTOCOL_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        forbidden = [
            name
            for name in imported
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_IMPORT_PREFIXES
            )
        ]
        assert forbidden == []
