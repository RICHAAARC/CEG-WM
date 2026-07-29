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
    KeyControlTrace,
    ProvenanceTrace,
    RoutingTrace,
    ThresholdTrace,
    validate_internal_record,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
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
        protocol_id="ceg_wm_internal_scientific_validation_v1",
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
        "protocol_id": "ceg_wm_internal_scientific_validation_v1",
        "record_schema_version": "ceg_wm_internal_sample_record_v1",
        "analysis_unit_identity": _unit(0),
        "split": "end_to_end_check",
        "attempt_index": 0,
        "execution_status": "success",
        "failure_reason": None,
        "exclusion_reason": None,
        "exclusion_rule_id": None,
        "retry_of_record_id": None,
        "detector_trace": DetectorTrace(
            raw_detector_identity="content_detector_identity_1",
            rectified_detector_identity="content_detector_identity_1",
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


@pytest.mark.unit
def test_frozen_protocol_config_has_exact_splits_and_denies_held_out_access() -> None:
    protocol = load_frozen_internal_validation_protocol(CONFIG_PATH)
    assert protocol.splits == INTERNAL_VALIDATION_SPLITS
    assert protocol.validate() == ()
    assert protocol.scientific_claim_boundary.endswith("no_scientific_validity_claim")
    with pytest.raises(PermissionError, match="held_out_evaluation"):
        authorize_split_access(
            _manifest(),
            ("held_out_evaluation",),
            SplitAccessGrant.current_execution(),
        )


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
        attempt_index=1,
        failure_reason="retryable_resource_failure",
        retry_of_record_id=success.record_id,
        decision_trace=DecisionTrace("retry", None, "retryable_resource_failure"),
    )
    assert validate_internal_record(failed) == ()
    assert validate_internal_record(excluded) == ()
    assert validate_internal_record(retry) == ()
    invalid_retry = replace(retry, attempt_index=0)
    assert "retry_attempt_index_must_be_positive" in validate_internal_record(invalid_retry)


@pytest.mark.unit
def test_raw_rectified_identity_and_threshold_must_be_identical() -> None:
    record = _record()
    mismatched = replace(
        record,
        detector_trace=replace(
            record.detector_trace,
            rectified_detector_identity="different_detector",
        ),
        threshold_trace=replace(
            record.threshold_trace,
            rectified_threshold_identity="different_threshold",
        ),
    )
    violations = validate_internal_record(mismatched)
    assert "raw_rectified_detector_identity_mismatch" in violations
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
