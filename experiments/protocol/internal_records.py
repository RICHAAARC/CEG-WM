"""内部科学验证逐样本正式记录与 fail-closed 约束。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

from experiments.protocol.internal_matrix import (
    PROMOTION_GATE_IDENTITIES,
    PROMOTION_STOP_OUTCOMES,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
    INTERNAL_VALIDATION_SPLITS,
    SplitAssignment,
)


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXECUTION_STATUSES = frozenset({"success", "failed", "excluded", "retry"})
FAILURE_CLASSES = frozenset(
    {"execution_failure", "resource_failure", "scientific_failure"}
)
KEY_ROLES = frozenset({"registered", "wrong_key", "unwatermarked_primary_null"})
WATERMARK_DECISIONS = frozenset({"positive", "negative", "failed", "excluded", "retry"})
POSITIVE_SOURCES = frozenset({"raw_content", "rectified_content"})
INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION = "ceg_wm_internal_sample_record_v3"
INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION = (
    "ceg_wm_internal_run_case_record_collection_v1"
)
MAXIMUM_RECORD_ATTEMPTS = 3
RETRYABLE_PARENT_STATUSES = frozenset({"failed", "retry"})


def _nonempty(value: str | None) -> bool:
    return bool(value and value.strip())


def _digest_valid(value: str) -> bool:
    return bool(_DIGEST_PATTERN.fullmatch(value))


def _finite_optional(value: float | None) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class DetectorTrace:
    raw_detector_identity: str
    rectified_detector_identity: str
    raw_detector_config_digest: str
    rectified_detector_config_digest: str
    raw_preprocessing_identity: str
    rectified_preprocessing_identity: str
    raw_content_score: float | None
    rectified_content_score: float | None


@dataclass(frozen=True)
class BranchScoreTrace:
    lf_score: float | None
    hf_score: float | None
    combined_score: float | None


@dataclass(frozen=True)
class RoutingTrace:
    routing_identity: str
    routing_control: str
    routing_observation_digest: str
    routing_mask_digest: str


@dataclass(frozen=True)
class GeometryTrace:
    geometry_triggered: bool
    geometry_estimation_identity: str | None
    geometry_reliability_identity: str | None
    geometry_reliable: bool | None
    geometry_transform: dict[str, float] | None
    geometry_raw_metrics: dict[str, float] | None
    geometry_failure_reason: str | None
    rectification_status: str


@dataclass(frozen=True)
class ThresholdTrace:
    raw_threshold_identity: str
    rectified_threshold_identity: str
    tau: float
    tau_rescue: float


@dataclass(frozen=True)
class KeyControlTrace:
    registered_key_public_digest: str
    detection_key_public_digest: str
    key_role: str
    control_identity: str


@dataclass(frozen=True)
class DecisionTrace:
    watermark_decision: str
    positive_source: str | None
    decision_reason: str


@dataclass(frozen=True)
class ProvenanceTrace:
    protocol_digest: str
    split_manifest_digest: str
    input_manifest_digest: str
    method_code_revision: str
    candidate_config_digest: str
    method_config_digest: str
    execution_config_digest: str
    model_revision: str
    environment_digest: str
    resource_identity_digest: str
    input_artifact_digest: str
    attack_config_digest: str
    metric_set_digest: str


@dataclass(frozen=True)
class InternalValidationRecord:
    """一个 unit/case 的一次尝试；runner 只能序列化，不得重解释字段。"""

    record_id: str
    run_id: str
    protocol_id: str
    protocol_version: str
    record_schema_version: str
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    record_sequence_index: int
    record_attempt_index: int
    execution_status: str
    failure_class: str | None
    failure_reason: str | None
    exclusion_reason: str | None
    exclusion_rule_id: str | None
    retry_of_record_id: str | None
    detector_trace: DetectorTrace
    branch_score_trace: BranchScoreTrace
    routing_trace: RoutingTrace
    geometry_trace: GeometryTrace
    threshold_trace: ThresholdTrace
    key_control_trace: KeyControlTrace
    decision_trace: DecisionTrace
    provenance_trace: ProvenanceTrace

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionGateAssessment:
    """一个由具体 record IDs 支撑的结构化晋升门裁决。"""

    gate_id: str
    gate_status: str
    evidence_record_ids: tuple[str, ...]
    stop_outcome: str | None


@dataclass(frozen=True)
class RunCaseRecordCollection:
    """一个 run/case 的有序 records、retry 上限和 promotion stop 事实。"""

    record_collection_schema_version: str
    run_id: str
    case_id: str
    protocol_id: str
    protocol_version: str
    protocol_digest: str
    split_manifest_digest: str
    record_schema_version: str
    maximum_record_attempts: int
    records: tuple[InternalValidationRecord, ...]
    promotion_gate_assessments: tuple[PromotionGateAssessment, ...]
    promotion_stop_gate_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_internal_record(record: InternalValidationRecord) -> tuple[str, ...]:
    violations: list[str] = list(record.analysis_unit_identity.validate())
    for name in (
        "record_id",
        "run_id",
        "protocol_id",
        "protocol_version",
        "record_schema_version",
    ):
        if not getattr(record, name).strip():
            violations.append(f"{name}_missing")
    if record.protocol_id != INTERNAL_VALIDATION_PROTOCOL_ID:
        violations.append("protocol_id_frozen_identity_mismatch")
    if record.protocol_version != INTERNAL_VALIDATION_PROTOCOL_VERSION:
        violations.append("protocol_version_frozen_identity_mismatch")
    if record.record_schema_version != INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION:
        violations.append("record_schema_version_frozen_identity_mismatch")
    if record.split not in INTERNAL_VALIDATION_SPLITS:
        violations.append("split_invalid")
    for name in ("record_sequence_index", "record_attempt_index"):
        value = getattr(record, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            violations.append(f"{name}_invalid")
    if record.execution_status not in EXECUTION_STATUSES:
        violations.append("execution_status_invalid")

    _validate_status(record, violations)
    _validate_detector_and_threshold(record, violations)
    _validate_routing(record.routing_trace, violations)
    _validate_geometry(record.geometry_trace, violations)
    _validate_key_control(record.key_control_trace, violations)
    _validate_provenance(record.provenance_trace, violations)
    _validate_decision(record, violations)
    return tuple(dict.fromkeys(violations))


def _validate_status(record: InternalValidationRecord, violations: list[str]) -> None:
    status = record.execution_status
    has_failure = _nonempty(record.failure_reason)
    has_failure_class = _nonempty(record.failure_class)
    has_exclusion = _nonempty(record.exclusion_reason)
    has_exclusion_rule = _nonempty(record.exclusion_rule_id)
    has_retry_parent = _nonempty(record.retry_of_record_id)
    if record.record_attempt_index == 0 and has_retry_parent:
        violations.append("initial_attempt_retry_parent_forbidden")
    if record.record_attempt_index > 0 and not has_retry_parent:
        violations.append("subsequent_attempt_retry_parent_missing")
    if status == "success":
        if has_failure_class or has_failure or has_exclusion or has_exclusion_rule:
            violations.append("success_failure_or_exclusion_field_forbidden")
        required_scores = (
            record.detector_trace.raw_content_score,
            record.branch_score_trace.hf_score,
        )
        if any(
            value is None or not _finite_optional(value)
            for value in required_scores
        ):
            violations.append("success_required_score_missing_or_non_finite")
    elif status == "failed":
        if not has_failure or record.failure_class not in FAILURE_CLASSES:
            violations.append("failure_class_or_reason_missing")
        if has_exclusion or has_exclusion_rule:
            violations.append("failed_exclusion_field_forbidden")
    elif status == "excluded":
        if not has_exclusion or not has_exclusion_rule:
            violations.append("exclusion_reason_or_rule_missing")
        if has_failure:
            violations.append("excluded_failure_field_forbidden")
        if has_failure_class:
            violations.append("excluded_failure_class_forbidden")
    elif status == "retry":
        if (
            not has_failure
            or not has_retry_parent
            or record.failure_class != "resource_failure"
        ):
            violations.append("retry_resource_failure_reason_or_parent_missing")
        if has_exclusion or has_exclusion_rule:
            violations.append("retry_exclusion_field_forbidden")
        if record.record_attempt_index == 0:
            violations.append("retry_record_attempt_index_must_be_positive")


def _validate_detector_and_threshold(
    record: InternalValidationRecord,
    violations: list[str],
) -> None:
    detector = record.detector_trace
    threshold = record.threshold_trace
    for name, value in (
        ("raw_detector_identity", detector.raw_detector_identity),
        ("rectified_detector_identity", detector.rectified_detector_identity),
        ("raw_preprocessing_identity", detector.raw_preprocessing_identity),
        ("rectified_preprocessing_identity", detector.rectified_preprocessing_identity),
        ("raw_threshold_identity", threshold.raw_threshold_identity),
        ("rectified_threshold_identity", threshold.rectified_threshold_identity),
    ):
        if not value.strip():
            violations.append(f"{name}_missing")
    if detector.raw_detector_identity != detector.rectified_detector_identity:
        violations.append("raw_rectified_detector_identity_mismatch")
    for name, value in (
        ("raw_detector_config_digest", detector.raw_detector_config_digest),
        ("rectified_detector_config_digest", detector.rectified_detector_config_digest),
    ):
        if not _digest_valid(value):
            violations.append(f"{name}_invalid")
    if detector.raw_detector_config_digest != detector.rectified_detector_config_digest:
        violations.append("raw_rectified_detector_config_digest_mismatch")
    if detector.raw_preprocessing_identity != detector.rectified_preprocessing_identity:
        violations.append("raw_rectified_preprocessing_identity_mismatch")
    if threshold.raw_threshold_identity != threshold.rectified_threshold_identity:
        violations.append("raw_rectified_threshold_identity_mismatch")
    if not math.isfinite(threshold.tau) or not math.isfinite(threshold.tau_rescue):
        violations.append("threshold_non_finite")
    elif threshold.tau_rescue >= threshold.tau:
        violations.append("tau_rescue_must_be_lower_than_tau")
    for name, value in (
        ("raw_content_score", detector.raw_content_score),
        ("rectified_content_score", detector.rectified_content_score),
        ("lf_score", record.branch_score_trace.lf_score),
        ("hf_score", record.branch_score_trace.hf_score),
        ("combined_score", record.branch_score_trace.combined_score),
    ):
        if not _finite_optional(value):
            violations.append(f"{name}_non_finite")


def _validate_routing(trace: RoutingTrace, violations: list[str]) -> None:
    for name, value in (
        ("routing_identity", trace.routing_identity),
        ("routing_control", trace.routing_control),
    ):
        if not value.strip():
            violations.append(f"{name}_missing")
    for name, value in (
        ("routing_observation_digest", trace.routing_observation_digest),
        ("routing_mask_digest", trace.routing_mask_digest),
    ):
        if not _digest_valid(value):
            violations.append(f"{name}_invalid")


def validate_routing_trace(trace: RoutingTrace) -> tuple[str, ...]:
    """Validate one frozen routing trace before runner execution."""

    if type(trace) is not RoutingTrace:
        return ("routing_trace_exact_type_required",)
    violations: list[str] = []
    _validate_routing(trace, violations)
    return tuple(dict.fromkeys(violations))


def _validate_geometry(trace: GeometryTrace, violations: list[str]) -> None:
    if trace.rectification_status not in {"not_attempted", "succeeded", "failed"}:
        violations.append("rectification_status_invalid")
    if not trace.geometry_triggered:
        if any(
            value is not None
            for value in (
                trace.geometry_estimation_identity,
                trace.geometry_reliability_identity,
                trace.geometry_reliable,
                trace.geometry_transform,
                trace.geometry_raw_metrics,
                trace.geometry_failure_reason,
            )
        ):
            violations.append("untriggered_geometry_payload_forbidden")
        if trace.rectification_status != "not_attempted":
            violations.append("untriggered_rectification_status_invalid")
        return
    estimation_present = _nonempty(trace.geometry_estimation_identity)
    reliability_present = _nonempty(trace.geometry_reliability_identity)
    if not estimation_present:
        if (
            reliability_present
            or trace.geometry_reliable is not None
            or trace.geometry_transform is not None
            or trace.geometry_raw_metrics is not None
            or trace.rectification_status != "not_attempted"
            or not _nonempty(trace.geometry_failure_reason)
        ):
            violations.append("geometry_pre_estimation_failure_invalid")
        return
    if not reliability_present:
        if (
            trace.geometry_reliable is not None
            or trace.rectification_status != "not_attempted"
            or not _nonempty(trace.geometry_failure_reason)
        ):
            violations.append("geometry_pre_reliability_failure_invalid")
    elif trace.geometry_reliable is None:
        violations.append("geometry_reliable_missing")
    for mapping_name, mapping in (
        ("geometry_transform", trace.geometry_transform),
        ("geometry_raw_metrics", trace.geometry_raw_metrics),
    ):
        if mapping is None or not mapping:
            violations.append(f"{mapping_name}_missing")
        elif any(not _finite_optional(value) for value in mapping.values()):
            violations.append(f"{mapping_name}_non_finite")
    if trace.geometry_reliable and trace.rectification_status == "not_attempted":
        violations.append("reliable_geometry_rectification_not_attempted")
    if not trace.geometry_reliable and trace.rectification_status == "succeeded":
        violations.append("unreliable_geometry_rectification_forbidden")
    if trace.rectification_status == "failed" and not _nonempty(trace.geometry_failure_reason):
        violations.append("rectification_failure_reason_missing")


def _validate_key_control(trace: KeyControlTrace, violations: list[str]) -> None:
    for name, value in (
        ("registered_key_public_digest", trace.registered_key_public_digest),
        ("detection_key_public_digest", trace.detection_key_public_digest),
    ):
        if not _digest_valid(value):
            violations.append(f"{name}_invalid")
    if trace.key_role not in KEY_ROLES:
        violations.append("key_role_invalid")
    if not trace.control_identity.strip():
        violations.append("control_identity_missing")
    if (
        trace.key_role == "registered"
        and trace.registered_key_public_digest != trace.detection_key_public_digest
    ):
        violations.append("registered_key_identity_mismatch")
    if (
        trace.key_role == "wrong_key"
        and trace.registered_key_public_digest == trace.detection_key_public_digest
    ):
        violations.append("wrong_key_identity_not_distinct")


def validate_key_control_trace(trace: KeyControlTrace) -> tuple[str, ...]:
    """Validate one frozen public key/control trace before runner execution."""

    if type(trace) is not KeyControlTrace:
        return ("key_control_trace_exact_type_required",)
    violations: list[str] = []
    _validate_key_control(trace, violations)
    return tuple(dict.fromkeys(violations))


def _validate_provenance(trace: ProvenanceTrace, violations: list[str]) -> None:
    for name in (
        "protocol_digest",
        "split_manifest_digest",
        "input_manifest_digest",
        "candidate_config_digest",
        "method_config_digest",
        "execution_config_digest",
        "environment_digest",
        "resource_identity_digest",
        "input_artifact_digest",
        "attack_config_digest",
        "metric_set_digest",
    ):
        if not _digest_valid(getattr(trace, name)):
            violations.append(f"{name}_invalid")
    for name in ("method_code_revision", "model_revision"):
        if not getattr(trace, name).strip():
            violations.append(f"{name}_missing")


def _validate_decision(record: InternalValidationRecord, violations: list[str]) -> None:
    decision = record.decision_trace
    detector = record.detector_trace
    geometry = record.geometry_trace
    threshold = record.threshold_trace
    if decision.watermark_decision not in WATERMARK_DECISIONS:
        violations.append("watermark_decision_invalid")
    if not decision.decision_reason.strip():
        violations.append("decision_reason_missing")
    expected_non_success = {
        "failed": "failed",
        "excluded": "excluded",
        "retry": "retry",
    }.get(record.execution_status)
    if expected_non_success and decision.watermark_decision != expected_non_success:
        violations.append("execution_status_decision_mismatch")
    if record.execution_status != "success":
        if decision.positive_source is not None:
            violations.append("non_success_positive_source_forbidden")
        return

    if decision.watermark_decision not in {"positive", "negative"}:
        violations.append("success_watermark_decision_invalid")
    if decision.watermark_decision == "negative" and decision.positive_source is not None:
        violations.append("negative_positive_source_forbidden")
    rectified_is_valid_content_source = (
        detector.rectified_content_score is not None
        and geometry.geometry_triggered
        and geometry.geometry_reliable is True
        and geometry.rectification_status == "succeeded"
    )
    if detector.rectified_content_score is not None and not rectified_is_valid_content_source:
        violations.append("rectified_score_without_valid_content_source")
    if decision.watermark_decision == "negative":
        if detector.raw_content_score is not None and detector.raw_content_score >= threshold.tau:
            violations.append("negative_raw_score_reached_tau")
        if (
            rectified_is_valid_content_source
            and detector.rectified_content_score is not None
            and detector.rectified_content_score >= threshold.tau
        ):
            violations.append("negative_rectified_score_reached_tau")
    if decision.watermark_decision == "positive":
        if decision.positive_source not in POSITIVE_SOURCES:
            violations.append("positive_content_source_missing")
        elif decision.positive_source == "raw_content":
            if detector.raw_content_score is None or detector.raw_content_score < threshold.tau:
                violations.append("raw_positive_threshold_not_met")
        elif decision.positive_source == "rectified_content":
            if (
                detector.rectified_content_score is None
                or detector.rectified_content_score < threshold.tau
                or not geometry.geometry_triggered
                or not geometry.geometry_reliable
                or geometry.rectification_status != "succeeded"
            ):
                violations.append("rectified_positive_requirements_not_met")
    if detector.raw_content_score is not None:
        near_threshold = threshold.tau_rescue <= detector.raw_content_score < threshold.tau
        if geometry.geometry_triggered != near_threshold:
            violations.append("geometry_trigger_near_threshold_mismatch")


def _validate_run_case_record_collection_structure(
    collection: RunCaseRecordCollection,
    *,
    frozen_protocol_id: str,
    frozen_protocol_version: str,
    frozen_record_schema_version: str,
    frozen_record_collection_schema_version: str,
    frozen_maximum_record_attempts: int,
    actual_protocol_digest: str,
    actual_split_manifest_digest: str,
    split_manifest_protocol_id: str,
    split_manifest_protocol_version: str,
    manifest_assignments: tuple[SplitAssignment, ...],
) -> tuple[str, ...]:
    """由正式 trust-anchor 入口调用的 collection 结构校验。"""
    violations: list[str] = []
    for name in (
        "record_collection_schema_version",
        "run_id",
        "case_id",
        "protocol_id",
        "protocol_version",
        "protocol_digest",
        "split_manifest_digest",
        "record_schema_version",
    ):
        if not getattr(collection, name).strip():
            violations.append(f"{name}_missing")
    if (
        collection.record_collection_schema_version
        != INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
    ):
        violations.append("record_collection_schema_version_frozen_identity_mismatch")
    if (
        collection.record_collection_schema_version
        != frozen_record_collection_schema_version
    ):
        violations.append("collection_frozen_record_collection_schema_version_mismatch")
    if collection.protocol_id != INTERNAL_VALIDATION_PROTOCOL_ID:
        violations.append("protocol_id_frozen_identity_mismatch")
    if collection.protocol_version != INTERNAL_VALIDATION_PROTOCOL_VERSION:
        violations.append("protocol_version_frozen_identity_mismatch")
    if collection.protocol_id != frozen_protocol_id:
        violations.append("collection_frozen_protocol_id_mismatch")
    if collection.protocol_version != frozen_protocol_version:
        violations.append("collection_frozen_protocol_version_mismatch")
    if collection.protocol_digest != actual_protocol_digest:
        violations.append("collection_protocol_digest_mismatch")
    if collection.split_manifest_digest != actual_split_manifest_digest:
        violations.append("collection_split_manifest_digest_mismatch")
    if not _digest_valid(collection.protocol_digest):
        violations.append("collection_protocol_digest_invalid")
    if not _digest_valid(collection.split_manifest_digest):
        violations.append("collection_split_manifest_digest_invalid")
    if split_manifest_protocol_id != frozen_protocol_id:
        violations.append("manifest_frozen_protocol_id_mismatch")
    if split_manifest_protocol_version != frozen_protocol_version:
        violations.append("manifest_frozen_protocol_version_mismatch")
    if collection.record_schema_version != INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION:
        violations.append("record_schema_version_frozen_identity_mismatch")
    if collection.record_schema_version != frozen_record_schema_version:
        violations.append("collection_frozen_record_schema_version_mismatch")
    if collection.maximum_record_attempts != MAXIMUM_RECORD_ATTEMPTS:
        violations.append("maximum_record_attempts_frozen_value_mismatch")
    if collection.maximum_record_attempts != frozen_maximum_record_attempts:
        violations.append("collection_frozen_maximum_record_attempts_mismatch")
    if not collection.records:
        violations.append("records_missing")

    records_by_id: dict[str, InternalValidationRecord] = {}
    attempts_by_identity: dict[tuple[str, str, str], list[InternalValidationRecord]] = {}
    manifest_assignment_pairs = {
        (assignment.identity, assignment.split) for assignment in manifest_assignments
    }
    for sequence_index, record in enumerate(collection.records):
        violations.extend(validate_internal_record(record))
        if record.record_id in records_by_id:
            violations.append("record_id_duplicate")
        records_by_id[record.record_id] = record
        if record.record_sequence_index != sequence_index:
            violations.append("record_sequence_index_not_contiguous")
        if record.run_id != collection.run_id:
            violations.append("record_run_id_collection_mismatch")
        if record.analysis_unit_identity.case_id != collection.case_id:
            violations.append("record_case_id_collection_mismatch")
        if record.protocol_id != collection.protocol_id:
            violations.append("record_protocol_id_collection_mismatch")
        if record.protocol_version != collection.protocol_version:
            violations.append("record_protocol_version_collection_mismatch")
        if record.record_schema_version != collection.record_schema_version:
            violations.append("record_schema_version_collection_mismatch")
        if record.provenance_trace.protocol_digest != actual_protocol_digest:
            violations.append("record_protocol_digest_binding_mismatch")
        if record.provenance_trace.split_manifest_digest != actual_split_manifest_digest:
            violations.append("record_split_manifest_digest_binding_mismatch")
        if (record.analysis_unit_identity, record.split) not in manifest_assignment_pairs:
            violations.append("record_manifest_assignment_missing")
        identity = (
            record.analysis_unit_identity.unit_id,
            record.analysis_unit_identity.case_id,
            record.analysis_unit_identity.source_cluster_id,
        )
        attempts_by_identity.setdefault(identity, []).append(record)

    for identity_records in attempts_by_identity.values():
        ordered = sorted(identity_records, key=lambda item: item.record_attempt_index)
        attempt_indices = tuple(item.record_attempt_index for item in ordered)
        if attempt_indices != tuple(range(len(ordered))):
            violations.append("record_attempt_index_not_contiguous")
        if len(ordered) > collection.maximum_record_attempts:
            violations.append("maximum_record_attempts_exceeded")
        if any(
            item.record_attempt_index >= collection.maximum_record_attempts for item in ordered
        ):
            violations.append("record_attempt_index_exceeds_frozen_limit")

    for record in collection.records:
        if record.record_attempt_index == 0:
            continue
        parent = records_by_id.get(record.retry_of_record_id or "")
        if parent is None:
            violations.append("attempt_parent_record_missing")
            continue
        child_identity = (
            record.run_id,
            record.analysis_unit_identity.unit_id,
            record.analysis_unit_identity.case_id,
            record.analysis_unit_identity.source_cluster_id,
            record.split,
        )
        parent_identity = (
            parent.run_id,
            parent.analysis_unit_identity.unit_id,
            parent.analysis_unit_identity.case_id,
            parent.analysis_unit_identity.source_cluster_id,
            parent.split,
        )
        if child_identity != parent_identity:
            violations.append("attempt_parent_identity_mismatch")
        if parent.protocol_id != record.protocol_id or parent.protocol_version != record.protocol_version:
            violations.append("attempt_parent_protocol_identity_mismatch")
        if parent.execution_status not in RETRYABLE_PARENT_STATUSES:
            violations.append("attempt_parent_status_not_retryable")
        if record.record_attempt_index != parent.record_attempt_index + 1:
            violations.append("attempt_parent_index_not_contiguous")
        if parent.record_sequence_index >= record.record_sequence_index:
            violations.append("attempt_parent_not_earlier")

    _validate_promotion_stop(collection, records_by_id, violations)
    return tuple(dict.fromkeys(violations))


def _validate_promotion_stop(
    collection: RunCaseRecordCollection,
    records_by_id: dict[str, InternalValidationRecord],
    violations: list[str],
) -> None:
    seen_gate_ids: set[str] = set()
    failed_assessment: PromotionGateAssessment | None = None
    failed_assessment_index: int | None = None
    for assessment_index, assessment in enumerate(collection.promotion_gate_assessments):
        if assessment.gate_id in seen_gate_ids:
            violations.append("promotion_gate_id_duplicate")
        seen_gate_ids.add(assessment.gate_id)
        if assessment.gate_id not in PROMOTION_GATE_IDENTITIES:
            violations.append("promotion_gate_id_invalid")
        if assessment.gate_status not in {"passed", "failed"}:
            violations.append("promotion_gate_status_invalid")
        if not assessment.evidence_record_ids:
            violations.append("promotion_gate_evidence_missing")
        if len(set(assessment.evidence_record_ids)) != len(assessment.evidence_record_ids):
            violations.append("promotion_gate_evidence_record_duplicate")
        if any(record_id not in records_by_id for record_id in assessment.evidence_record_ids):
            violations.append("promotion_gate_evidence_record_missing")
        if assessment.gate_status == "passed" and assessment.stop_outcome is not None:
            violations.append("passed_promotion_gate_stop_outcome_forbidden")
        if assessment.gate_status == "failed":
            if assessment.stop_outcome not in PROMOTION_STOP_OUTCOMES:
                violations.append("failed_promotion_gate_stop_outcome_invalid")
            if failed_assessment is None:
                failed_assessment = assessment
                failed_assessment_index = assessment_index
    if failed_assessment is None:
        if collection.promotion_stop_gate_id is not None:
            violations.append("promotion_stop_without_failed_gate")
        return
    if collection.promotion_stop_gate_id != failed_assessment.gate_id:
        violations.append("promotion_stop_gate_id_mismatch")
    if (
        failed_assessment_index is not None
        and failed_assessment_index != len(collection.promotion_gate_assessments) - 1
    ):
        violations.append("promotion_gate_assessment_after_stop")
    evidence_records = [
        records_by_id[record_id]
        for record_id in failed_assessment.evidence_record_ids
        if record_id in records_by_id
    ]
    if not evidence_records:
        return
    stop_sequence_index = max(record.record_sequence_index for record in evidence_records)
    if any(
        record.record_sequence_index > stop_sequence_index for record in collection.records
    ):
        violations.append("record_continues_after_promotion_stop")
