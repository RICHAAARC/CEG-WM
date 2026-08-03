"""Exact formal records for development module exploration.

This protocol module owns the persisted scientific record schema.  It has no
filesystem or runner dependency; the governed runner is the sole writer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Mapping

from experiments.protocol.internal_splits import AnalysisUnitIdentity


DEVELOPMENT_CLAIM_BOUNDARY = (
    "preliminary_development_signal_only_no_promotion_or_scientific_claim"
)
RECORD_SCHEMA_VERSION = "ceg_wm_development_scientific_record_v1"
DEVELOPMENT_RECORD_COLLECTION_ROLE = "runner_only_formal_development_records"
DEVELOPMENT_RECORD_MEMBER_PATH = "records/development_scientific_record.json"
ATTEMPT_DISPOSITIONS = frozenset(
    {"success", "final_failure", "retryable_resource_failure"}
)
_EXECUTION_STATUSES = frozenset({"success", "failed", "excluded", "retry"})
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){1,2}$")
_METRIC_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "metric_role",
        "responsibility_id",
        "source_cluster_id",
        "registered_metric_ids",
        "candidate_config_digest",
        "paired_ablation_identity",
        "content_branch_id",
        "geometry_case_id",
        "sufficient_statistics",
        "result_identity_digests",
        "threshold_role",
        "threshold_identity",
        "threshold_fit_source_cluster_digest",
        "observation_digest",
    }
)
METRIC_SCHEMA_VERSION = "ceg_wm_development_metric_observation_v1"
_METRIC_ROLE = "development_exploratory_cluster_level"
_PROVENANCE_BINDINGS = (
    ("protocol_digest", "protocol_digest"),
    ("execution_intent_authority_digest", "execution_intent_authority_digest"),
    ("method_code_revision", "method_code_revision"),
    ("candidate_config_digest", "candidate_config_digest"),
)


class DevelopmentRecordError(ValueError):
    """Formal development record schema or semantic validation failed."""


def canonical_development_value_digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DevelopmentRecordError("development record value is not JSON data") from exc
    return sha256(payload).hexdigest()


def _identity(value: object, role: str) -> str:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        raise DevelopmentRecordError(f"{role} is not a stable identity")
    return value


def _digest(value: object, role: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise DevelopmentRecordError(f"{role} is not a SHA-256 digest")
    return value


def _mapping(value: object, role: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise DevelopmentRecordError(f"{role} must be a string-keyed mapping")
    return value


@dataclass(frozen=True, slots=True)
class DevelopmentScientificRecord:
    schema_version: str
    collection_role: str
    record_id: str
    run_id: str
    protocol_id: str
    protocol_version: str
    protocol_digest: str
    execution_intent_authority_digest: str
    method_code_revision: str
    unit_index: int
    phase: str
    analysis_unit_identity: dict[str, object]
    responsibility_id: str
    scientific_question_id: str
    development_case_id: str
    candidate_identity: str
    candidate_config_digest: str
    paired_ablation_identity: str
    negative_control_case_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    content_branch_id: str
    geometry_case_id: str
    attempt_index: int
    execution_status: str
    failure_class: str | None
    failure_reason: str | None
    retry_parent_intent_digest: str | None
    actual_elapsed_seconds: float
    maximum_duration_seconds: int
    duration_limit_exceeded: bool
    operation_result_payload: dict[str, object]
    operation_result_digest: str
    metric_observation: dict[str, object]
    routing_trace: dict[str, object]
    branch_score_trace: dict[str, object]
    detector_trace: dict[str, object]
    geometry_trace: dict[str, object]
    threshold_trace: dict[str, object]
    key_control_trace: dict[str, object]
    decision_trace: dict[str, object]
    provenance_trace: dict[str, object]
    module_outcome: str | None
    candidate_recommendation: str | None
    scientific_claim_boundary: str

    def payload(self) -> dict[str, object]:
        return asdict(self)

    def payload_without_record_id(self) -> dict[str, object]:
        payload = self.payload()
        payload.pop("record_id")
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DevelopmentScientificRecord":
        if type(payload) is not dict:
            raise DevelopmentRecordError("development scientific record must be a mapping")
        converted = dict(payload)
        for field_name in ("negative_control_case_ids", "metric_ids"):
            value = converted.get(field_name)
            if type(value) is not list:
                raise DevelopmentRecordError(
                    f"development scientific record {field_name} must be a list"
                )
            converted[field_name] = tuple(value)
        try:
            record = cls(**converted)
        except TypeError as exc:
            raise DevelopmentRecordError(
                "development scientific record exact schema mismatch"
            ) from exc
        record.validate()
        return record

    def attempt_disposition(self) -> str:
        if self.execution_status == "success":
            return "success"
        if self.execution_status == "retry":
            return "retryable_resource_failure"
        return "final_failure"

    def validate(self) -> None:
        if self.schema_version != RECORD_SCHEMA_VERSION:
            raise DevelopmentRecordError("development scientific record schema drifted")
        if self.collection_role != DEVELOPMENT_RECORD_COLLECTION_ROLE:
            raise DevelopmentRecordError("development record collection role drifted")
        for role, value in (
            ("run_id", self.run_id),
            ("protocol_id", self.protocol_id),
            ("phase", self.phase),
            ("responsibility_id", self.responsibility_id),
            ("scientific_question_id", self.scientific_question_id),
            ("development_case_id", self.development_case_id),
            ("candidate_identity", self.candidate_identity),
            ("paired_ablation_identity", self.paired_ablation_identity),
            ("content_branch_id", self.content_branch_id),
            ("geometry_case_id", self.geometry_case_id),
        ):
            _identity(value, role)
        if (
            type(self.protocol_version) is not str
            or (
                _VERSION.fullmatch(self.protocol_version) is None
                and _IDENTITY.fullmatch(self.protocol_version) is None
            )
        ):
            raise DevelopmentRecordError("protocol version is invalid")
        for role, value in (
            ("record_id", self.record_id),
            ("protocol_digest", self.protocol_digest),
            ("execution_intent_authority_digest", self.execution_intent_authority_digest),
            ("candidate_config_digest", self.candidate_config_digest),
            ("operation_result_digest", self.operation_result_digest),
        ):
            _digest(value, role)
        if type(self.method_code_revision) is not str or _REVISION.fullmatch(self.method_code_revision) is None:
            raise DevelopmentRecordError("method code revision is invalid")
        if type(self.unit_index) is not int or self.unit_index < 0:
            raise DevelopmentRecordError("development record unit index is invalid")
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise DevelopmentRecordError("development record attempt index is invalid")
        if type(self.maximum_duration_seconds) is not int or self.maximum_duration_seconds < 1:
            raise DevelopmentRecordError("development record duration limit is invalid")
        if (
            isinstance(self.actual_elapsed_seconds, bool)
            or not isinstance(self.actual_elapsed_seconds, (int, float))
            or not isfinite(float(self.actual_elapsed_seconds))
            or float(self.actual_elapsed_seconds) < 0.0
        ):
            raise DevelopmentRecordError("development record elapsed time is invalid")
        expected_exceeded = float(self.actual_elapsed_seconds) > float(
            self.maximum_duration_seconds
        )
        if type(self.duration_limit_exceeded) is not bool or self.duration_limit_exceeded is not expected_exceeded:
            raise DevelopmentRecordError("development record duration limit status drifted")
        if self.execution_status not in _EXECUTION_STATUSES:
            raise DevelopmentRecordError("development record execution status is invalid")
        if self.execution_status == "success":
            if self.failure_class is not None or self.failure_reason is not None:
                raise DevelopmentRecordError("successful development record cannot carry failure")
            if self.duration_limit_exceeded:
                raise DevelopmentRecordError("successful development record exceeded duration limit")
        else:
            _identity(self.failure_class, "failure_class")
            if type(self.failure_reason) is not str or not self.failure_reason.strip():
                raise DevelopmentRecordError("failed development record requires a reason")
        if self.execution_status == "retry" and self.failure_class != "resource_failure":
            raise DevelopmentRecordError("only resource failure is retryable")
        if self.attempt_index == 0:
            if self.retry_parent_intent_digest is not None:
                raise DevelopmentRecordError("initial record cannot have retry parent")
        else:
            _digest(self.retry_parent_intent_digest, "retry parent intent")
        if type(self.analysis_unit_identity) is not dict:
            raise DevelopmentRecordError("analysis unit identity payload is invalid")
        try:
            analysis_identity = AnalysisUnitIdentity(**self.analysis_unit_identity)
        except TypeError as exc:
            raise DevelopmentRecordError("analysis unit identity schema is invalid") from exc
        if analysis_identity.validate():
            raise DevelopmentRecordError("analysis unit identity is invalid")
        for role, values in (
            ("negative control case ids", self.negative_control_case_ids),
            ("metric ids", self.metric_ids),
        ):
            if type(values) is not tuple or not values:
                raise DevelopmentRecordError(f"{role} must be a non-empty tuple")
            for value in values:
                _identity(value, role)
            if len(values) != len(set(values)):
                raise DevelopmentRecordError(f"{role} contain duplicates")
        for role, value in (
            ("operation result payload", self.operation_result_payload),
            ("metric observation", self.metric_observation),
            ("routing trace", self.routing_trace),
            ("branch score trace", self.branch_score_trace),
            ("detector trace", self.detector_trace),
            ("geometry trace", self.geometry_trace),
            ("threshold trace", self.threshold_trace),
            ("key control trace", self.key_control_trace),
            ("decision trace", self.decision_trace),
            ("provenance trace", self.provenance_trace),
        ):
            _mapping(value, role)
        if self.operation_result_digest != canonical_development_value_digest(
            self.operation_result_payload
        ):
            raise DevelopmentRecordError("operation result digest drifted")
        for trace_name, record_name in _PROVENANCE_BINDINGS:
            if self.provenance_trace.get(trace_name) != getattr(self, record_name):
                raise DevelopmentRecordError(
                    f"development provenance {trace_name} binding drifted"
                )
        if self.execution_status == "success":
            metric = self.metric_observation
            if set(metric) != _METRIC_OBSERVATION_FIELDS:
                raise DevelopmentRecordError(
                    "development metric observation exact schema mismatch"
                )
            if (
                metric.get("schema_version") != METRIC_SCHEMA_VERSION
                or metric.get("metric_role") != _METRIC_ROLE
                or metric.get("responsibility_id") != self.responsibility_id
                or metric.get("source_cluster_id")
                != analysis_identity.source_cluster_id
                or tuple(metric.get("registered_metric_ids", ())) != self.metric_ids
                or metric.get("candidate_config_digest")
                != self.candidate_config_digest
                or metric.get("paired_ablation_identity")
                != self.paired_ablation_identity
                or metric.get("content_branch_id") != self.content_branch_id
                or metric.get("geometry_case_id") != self.geometry_case_id
            ):
                raise DevelopmentRecordError(
                    "development metric observation binding drifted"
                )
            metric_without_digest = dict(metric)
            metric_digest = metric_without_digest.pop("observation_digest")
            if metric_digest != canonical_development_value_digest(
                metric_without_digest
            ):
                raise DevelopmentRecordError(
                    "development metric observation digest drifted"
                )
        elif self.metric_observation:
            raise DevelopmentRecordError(
                "failed development record cannot carry scientific metric"
            )
        if self.record_id != canonical_development_value_digest(
            self.payload_without_record_id()
        ):
            raise DevelopmentRecordError("development record identity drifted")
        if self.module_outcome is not None or self.candidate_recommendation is not None:
            raise DevelopmentRecordError("per-unit record cannot preempt module outcome")
        if self.scientific_claim_boundary != DEVELOPMENT_CLAIM_BOUNDARY:
            raise DevelopmentRecordError("development scientific claim boundary drifted")
        positive_source = self.decision_trace.get("positive_source")
        if positive_source not in {None, "raw_content", "rectified_content"}:
            raise DevelopmentRecordError("geometry cannot be a positive source")
        if self.execution_status == "success":
            raw_threshold = self.threshold_trace.get("raw_threshold_identity")
            rectified_threshold = self.threshold_trace.get(
                "rectified_threshold_identity"
            )
            if raw_threshold != rectified_threshold:
                raise DevelopmentRecordError("raw and rectified threshold identities differ")
            for raw_name, rectified_name in (
                ("raw_detector_identity", "rectified_detector_identity"),
                ("raw_detector_config_digest", "rectified_detector_config_digest"),
                ("raw_preprocessing_identity", "rectified_preprocessing_identity"),
            ):
                if self.detector_trace.get(raw_name) != self.detector_trace.get(
                    rectified_name
                ):
                    raise DevelopmentRecordError(
                        "raw and rectified detector semantics differ"
                    )


def validate_record_against_intent(record: DevelopmentScientificRecord, intent: object) -> None:
    """Bind the formal record to an already validated persistence UnitIntent."""

    record.validate()
    expected_pairs = (
        (record.run_id, getattr(intent, "run_id", None), "run_id"),
        (record.protocol_digest, getattr(intent, "protocol_digest", None), "protocol_digest"),
        (record.method_code_revision, getattr(intent, "revision", None), "method_code_revision"),
        (record.unit_index, getattr(intent, "unit_index", None), "unit_index"),
        (record.phase, getattr(intent, "phase", None), "phase"),
        (record.analysis_unit_identity, getattr(intent, "analysis_unit_identity", None), "analysis_unit_identity"),
        (record.responsibility_id, getattr(intent, "responsibility_id", None), "responsibility_id"),
        (record.scientific_question_id, getattr(intent, "scientific_question_id", None), "scientific_question_id"),
        (record.development_case_id, getattr(intent, "development_case_id", None), "development_case_id"),
        (record.candidate_identity, getattr(intent, "candidate_identity", None), "candidate_identity"),
        (record.candidate_config_digest, getattr(intent, "candidate_config_digest", None), "candidate_config_digest"),
        (record.content_branch_id, getattr(intent, "content_branch_id", None), "content_branch_id"),
        (record.geometry_case_id, getattr(intent, "geometry_case_id", None), "geometry_case_id"),
        (record.attempt_index, getattr(intent, "attempt_index", None), "attempt_index"),
        (record.retry_parent_intent_digest, getattr(intent, "parent_attempt_intent_digest", None), "retry_parent_intent_digest"),
        (record.maximum_duration_seconds, getattr(intent, "maximum_duration_seconds", None), "maximum_duration_seconds"),
    )
    for observed, expected, role in expected_pairs:
        if observed != expected:
            raise DevelopmentRecordError(
                f"development record {role} differs from unit intent"
            )
