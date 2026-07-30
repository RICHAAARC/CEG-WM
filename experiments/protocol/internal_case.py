"""Trusted per-case execution expectations for internal governed records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import re

from experiments.protocol.internal_records import (
    KeyControlTrace,
    RoutingTrace,
    validate_key_control_trace,
    validate_routing_trace,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
)
from experiments.protocol.internal_validation import (
    FrozenInternalValidationProtocol,
)


INPUT_MANIFEST_SCHEMA_VERSION = "ceg_wm_internal_case_input_manifest_v2"
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _digest_valid(value: object) -> bool:
    return type(value) is str and bool(_DIGEST_PATTERN.fullmatch(value))


@dataclass(frozen=True, slots=True)
class FrozenCaseExecutionExpectation:
    """Record-visible declarations fixed before one case is executed."""

    content_detector_binding_digest: str
    raw_detector_identity: str
    rectified_detector_identity: str
    raw_detector_config_digest: str
    rectified_detector_config_digest: str
    raw_preprocessing_identity: str
    rectified_preprocessing_identity: str
    raw_threshold_identity: str
    rectified_threshold_identity: str
    calibration_identity: str
    tau: float
    tau_rescue: float
    geometry_operation_identity: str
    geometry_reliability_config_digest: str | None

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        for role in (
            "content_detector_binding_digest",
            "raw_detector_config_digest",
            "rectified_detector_config_digest",
        ):
            if not _digest_valid(getattr(self, role)):
                violations.append(f"{role}_invalid")
        for role in (
            "raw_detector_identity",
            "rectified_detector_identity",
            "raw_preprocessing_identity",
            "rectified_preprocessing_identity",
            "raw_threshold_identity",
            "rectified_threshold_identity",
            "calibration_identity",
            "geometry_operation_identity",
        ):
            value = getattr(self, role)
            if type(value) is not str or not value:
                violations.append(f"{role}_missing")
        if self.raw_detector_identity != self.rectified_detector_identity:
            violations.append("expected_detector_identity_mismatch")
        if (
            self.raw_detector_config_digest
            != self.rectified_detector_config_digest
        ):
            violations.append("expected_detector_config_digest_mismatch")
        if (
            self.raw_preprocessing_identity
            != self.rectified_preprocessing_identity
        ):
            violations.append("expected_preprocessing_identity_mismatch")
        if self.raw_threshold_identity != self.rectified_threshold_identity:
            violations.append("expected_threshold_identity_mismatch")
        if (
            isinstance(self.tau, bool)
            or not isinstance(self.tau, (int, float))
            or not isfinite(float(self.tau))
            or isinstance(self.tau_rescue, bool)
            or not isinstance(self.tau_rescue, (int, float))
            or not isfinite(float(self.tau_rescue))
        ):
            violations.append("expected_threshold_non_finite")
        elif float(self.tau_rescue) >= float(self.tau):
            violations.append("expected_tau_rescue_not_lower")
        if (
            self.geometry_reliability_config_digest is not None
            and not _digest_valid(
                self.geometry_reliability_config_digest
            )
        ):
            violations.append(
                "geometry_reliability_config_digest_invalid"
            )
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class InternalCaseManifestEntry:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    input_artifact_digest: str
    attack_config_digest: str
    metric_set_digest: str
    routing_trace: RoutingTrace
    key_control_trace: KeyControlTrace
    execution_expectation: FrozenCaseExecutionExpectation

    def validate(self) -> tuple[str, ...]:
        if type(self.analysis_unit_identity) is not AnalysisUnitIdentity:
            return ("analysis_unit_identity_exact_type_required",)
        violations = list(self.analysis_unit_identity.validate())
        for role in (
            "input_artifact_digest",
            "attack_config_digest",
            "metric_set_digest",
        ):
            if not _digest_valid(getattr(self, role)):
                violations.append(f"{role}_invalid")
        violations.extend(validate_routing_trace(self.routing_trace))
        violations.extend(validate_key_control_trace(self.key_control_trace))
        if type(self.execution_expectation) is not FrozenCaseExecutionExpectation:
            violations.append("execution_expectation_exact_type_required")
        else:
            violations.extend(self.execution_expectation.validate())
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class FrozenCaseInputManifest:
    manifest_schema_version: str
    manifest_id: str
    manifest_revision: str
    protocol_digest: str
    split_manifest_digest: str
    entries: tuple[InternalCaseManifestEntry, ...]

    def digest(self) -> str:
        return _canonical_digest(asdict(self))

    def validate(
        self,
        *,
        protocol: FrozenInternalValidationProtocol,
        split_manifest: FrozenSplitManifest,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if self.manifest_schema_version != INPUT_MANIFEST_SCHEMA_VERSION:
            violations.append("input_manifest_schema_version_invalid")
        for role in ("manifest_id", "manifest_revision"):
            if type(getattr(self, role)) is not str or not getattr(self, role):
                violations.append(f"{role}_missing")
        if self.protocol_digest != protocol.digest():
            violations.append("input_manifest_protocol_digest_mismatch")
        if self.split_manifest_digest != split_manifest.digest():
            violations.append("input_manifest_split_manifest_digest_mismatch")
        if not self.entries:
            violations.append("input_manifest_entries_missing")
        assignment_pairs = {
            (assignment.identity, assignment.split)
            for assignment in split_manifest.assignments
        }
        seen_units: set[str] = set()
        for entry in self.entries:
            if type(entry) is not InternalCaseManifestEntry:
                violations.append("input_manifest_entry_exact_type_required")
                continue
            violations.extend(entry.validate())
            if (entry.analysis_unit_identity, entry.split) not in assignment_pairs:
                violations.append("input_manifest_split_assignment_missing")
            unit_id = entry.analysis_unit_identity.unit_id
            if unit_id in seen_units:
                violations.append("input_manifest_unit_duplicate")
            seen_units.add(unit_id)
        return tuple(dict.fromkeys(violations))


def derive_internal_record_id(
    *,
    run_id: str,
    case_id: str,
    input_manifest_digest: str,
    analysis_unit_identity: AnalysisUnitIdentity,
    attempt_index: int,
) -> str:
    """Derive one record identity from trusted run and case inputs."""

    if type(run_id) is not str or not run_id:
        raise ValueError("run_id is required")
    if type(case_id) is not str or not case_id:
        raise ValueError("case_id is required")
    if not _digest_valid(input_manifest_digest):
        raise ValueError("input_manifest_digest is invalid")
    if type(analysis_unit_identity) is not AnalysisUnitIdentity:
        raise ValueError("analysis_unit_identity exact type is required")
    identity_violations = analysis_unit_identity.validate()
    if identity_violations:
        raise ValueError(",".join(identity_violations))
    if (
        type(attempt_index) is not int
        or isinstance(attempt_index, bool)
        or attempt_index < 0
    ):
        raise ValueError("attempt_index is invalid")
    return _canonical_digest(
        {
            "attempt_index": attempt_index,
            "case_id": case_id,
            "input_manifest_digest": input_manifest_digest,
            "run_id": run_id,
            "source_cluster_id": analysis_unit_identity.source_cluster_id,
            "unit_id": analysis_unit_identity.unit_id,
        }
    )
