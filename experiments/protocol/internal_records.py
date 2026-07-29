"""内部科学验证逐样本正式记录与 fail-closed 约束。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

from experiments.protocol.internal_splits import AnalysisUnitIdentity, INTERNAL_VALIDATION_SPLITS


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXECUTION_STATUSES = frozenset({"success", "failed", "excluded", "retry"})
KEY_ROLES = frozenset({"registered", "wrong_key", "unwatermarked_primary_null"})
WATERMARK_DECISIONS = frozenset({"positive", "negative", "failed", "excluded", "retry"})
POSITIVE_SOURCES = frozenset({"raw_content", "rectified_content"})


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
    method_code_revision: str
    method_config_digest: str
    model_revision: str
    environment_digest: str
    input_artifact_digest: str
    attack_config_digest: str
    metric_set_digest: str


@dataclass(frozen=True)
class InternalValidationRecord:
    """一个 unit/case 的一次尝试；runner 只能序列化，不得重解释字段。"""

    record_id: str
    run_id: str
    protocol_id: str
    record_schema_version: str
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    attempt_index: int
    execution_status: str
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


def validate_internal_record(record: InternalValidationRecord) -> tuple[str, ...]:
    violations: list[str] = list(record.analysis_unit_identity.validate())
    for name in ("record_id", "run_id", "protocol_id", "record_schema_version"):
        if not getattr(record, name).strip():
            violations.append(f"{name}_missing")
    if record.split not in INTERNAL_VALIDATION_SPLITS:
        violations.append("split_invalid")
    if not isinstance(record.attempt_index, int) or isinstance(record.attempt_index, bool):
        violations.append("attempt_index_invalid")
    elif record.attempt_index < 0:
        violations.append("attempt_index_invalid")
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
    has_exclusion = _nonempty(record.exclusion_reason)
    has_exclusion_rule = _nonempty(record.exclusion_rule_id)
    has_retry_parent = _nonempty(record.retry_of_record_id)
    if status == "success":
        if has_failure or has_exclusion or has_exclusion_rule or has_retry_parent:
            violations.append("success_reason_or_retry_field_forbidden")
        scores = (
            record.detector_trace.raw_content_score,
            record.branch_score_trace.lf_score,
            record.branch_score_trace.hf_score,
            record.branch_score_trace.combined_score,
        )
        if any(value is None or not _finite_optional(value) for value in scores):
            violations.append("success_required_score_missing_or_non_finite")
    elif status == "failed":
        if not has_failure:
            violations.append("failure_reason_missing")
        if has_exclusion or has_exclusion_rule or has_retry_parent:
            violations.append("failed_exclusion_or_retry_field_forbidden")
    elif status == "excluded":
        if not has_exclusion or not has_exclusion_rule:
            violations.append("exclusion_reason_or_rule_missing")
        if has_failure or has_retry_parent:
            violations.append("excluded_failure_or_retry_field_forbidden")
    elif status == "retry":
        if not has_failure or not has_retry_parent:
            violations.append("retry_reason_or_parent_missing")
        if has_exclusion or has_exclusion_rule:
            violations.append("retry_exclusion_field_forbidden")
        if record.attempt_index == 0:
            violations.append("retry_attempt_index_must_be_positive")


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
    if not _nonempty(trace.geometry_estimation_identity):
        violations.append("geometry_estimation_identity_missing")
    if not _nonempty(trace.geometry_reliability_identity):
        violations.append("geometry_reliability_identity_missing")
    if trace.geometry_reliable is None:
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


def _validate_provenance(trace: ProvenanceTrace, violations: list[str]) -> None:
    for name in (
        "protocol_digest",
        "split_manifest_digest",
        "method_config_digest",
        "environment_digest",
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
