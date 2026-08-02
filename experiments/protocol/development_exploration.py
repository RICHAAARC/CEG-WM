"""Fail-closed development exploration protocol for all 13 method duties.

The objects in this module freeze study identities, budgets, split isolation,
cross-fit threshold inputs, and module outcomes.  They do not execute methods,
write records, promote candidates, or create formal scientific evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from experiments.protocol.internal_matrix import REQUIRED_METHOD_RESPONSIBILITIES
from experiments.protocol.internal_records import (
    INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION,
    INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
    MAXIMUM_RECORD_ATTEMPTS,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
    INTERNAL_VALIDATION_SPLITS,
    SplitAssignment,
)


PROTOCOL_ID = "ceg_wm_development_module_exploration"
DEVELOPMENT_EXPLORATION_PROTOCOL_VERSION = "2.0.0"
SCHEMA_VERSION = "ceg_wm_development_module_exploration_protocol_schema_v2"
DEVELOPMENT_SPLIT = "development"
FORMAL_LATER_SPLIT_DENY_LIST = INTERNAL_VALIDATION_SPLITS[1:]

PREFLIGHT_SOURCE_CLUSTER_COUNT = 2
WIRING_SOURCE_CLUSTER_COUNT = 8
BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT = 16
CRITICAL_PAIR_SOURCE_CLUSTER_COUNT = 32
CHEAP_DETECTION_SOURCE_CLUSTER_COUNT = 64
SCIENTIFIC_SOURCE_CLUSTER_SCALES = (16, 32, 64)
CRITICAL_PAIR_RESPONSIBILITIES = (
    "qk_geometry_sync",
    "geometric_transform_estimator",
)
CHEAP_DETECTION_RESPONSIBILITIES = (
    "lf_detector",
    "hf_detector",
    "content_detector",
)
MAXIMUM_UNIT_DURATION_SECONDS = 900

DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT = 4
DEVELOPMENT_THRESHOLD_ROLE = "development_exploratory"
DEVELOPMENT_THRESHOLD_FIT_ROLE = "development_primary_null_and_key_control_fit"
DEVELOPMENT_THRESHOLD_SCORE_ROLE = "development_recovery_probe_score"
DEVELOPMENT_THRESHOLD_INPUT_ROLES = (
    "primary_null",
    "wrong_key_control",
)
DEVELOPMENT_THRESHOLD_INVALIDATION = (
    "invalidate_before_candidate_selection_and_all_later_splits"
)
DEVELOPMENT_CLAIM_BOUNDARY = (
    "preliminary_development_signal_only_no_promotion_or_scientific_claim"
)

CONTENT_BRANCH_IDS = (
    "clean_control",
    "hf_only",
    "lf_only",
    "lf_hf_disabled_uniform_control",
    "lf_hf_routed_combination",
)
CONTENT_MIXING_COEFFICIENTS = (0.25, 0.50, 0.75)
CONTENT_COMBINATION_FUNCTION_IDS = (
    "hf_only_standardized_score",
    "weighted_hf_lf_standardized_score",
    "maximum_hf_lf_standardized_score",
)

GEOMETRY_OPERATION_FAMILIES = (
    "identity",
    "crop",
    "scale",
    "rotation",
    "compound",
)
GEOMETRY_NEGATIVE_CONTROL_CASE_IDS = (
    "ambiguous_transform_control",
    "boundary_transform_control",
    "extreme_crop_control",
)

MODULE_OUTCOMES = (
    "mechanism_signal_observed",
    "mechanism_signal_not_observed",
    "implementation_blocked",
    "resource_blocked",
)
CANDIDATE_RECOMMENDATIONS = (
    "candidate_worth_further_selection",
    "candidate_not_recommended_for_selection",
)
DEPENDENCY_STOP_RULE = (
    "stop_when_any_prerequisite_lacks_mechanism_signal_observed"
)
MODULE_OUTCOME_RULE = (
    "classify_mechanism_signal_separately_from_candidate_recommendation"
)

ISOLATION_DIMENSIONS = (
    "prompt_digest",
    "source_cluster_id",
    "seed_namespace",
    "registered_key_family_digest",
    "image_lineage_digest",
)
REGISTERED_STUDY_ROLE_BINDINGS = (
    ("development_exploration", "development", "candidate_configured", False),
    (
        "candidate_selection_selection",
        "candidate_selection",
        "candidate_configured",
        False,
    ),
    (
        "content_candidate_confirmation",
        "untouched_confirmation",
        "combined",
        False,
    ),
    (
        "hf_only_reference_confirmation",
        "untouched_confirmation",
        "hf_only",
        True,
    ),
    (
        "content_threshold_calibration",
        "content_threshold_fit",
        "combined",
        False,
    ),
    (
        "rescue_threshold_calibration",
        "rescue_threshold_fit",
        "combined",
        False,
    ),
    (
        "geometry_reliability_calibration",
        "reliability_fit",
        "combined",
        False,
    ),
    ("end_to_end_safety_check", "end_to_end_check", "combined", False),
    (
        "formal_held_out_evaluation",
        "held_out_evaluation",
        "frozen_candidate",
        True,
    ),
)

DEVELOPMENT_UNIT_ORDER = (
    "scientific_breadth_source_cluster_ordinal",
    "module_dependency_order",
    "critical_pair_extension_source_cluster_ordinal",
    "cheap_detection_extension_source_cluster_ordinal",
)
UNIT_PHASES = (
    "scientific_breadth",
    "critical_pair_extension",
    "cheap_detection_extension",
)
PREFLIGHT_CASE_IDS = (
    "environment_identity_preflight",
    "runtime_identity_preflight",
    "throughput_preflight",
)
COMMON_MODULE_RECORD_FIELDS = frozenset(
    {
        "responsibility_id",
        "scientific_question_id",
        "development_case_id",
        "candidate_identity",
        "candidate_config_digest",
        "paired_ablation_identity",
        "module_outcome",
        "candidate_recommendation",
        "recommendation_reason",
        "evidence_record_ids",
    }
)
REGISTERED_DEVELOPMENT_RECORD_FIELDS = COMMON_MODULE_RECORD_FIELDS | {
    "branch_score_trace",
    "decision_trace",
    "detector_trace",
    "geometry_trace",
    "key_control_trace",
    "provenance_trace",
    "routing_trace",
    "threshold_trace",
}

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_EXACT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "protocol_version",
        "split_policy",
        "split_isolation",
        "preflight",
        "study_budget",
        "provisional_threshold_cross_fit",
        "content_study",
        "geometry_study",
        "module_matrix",
        "module_outcomes",
        "scientific_claim_boundary",
    }
)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _digest_sequence(values: Sequence[str]) -> str:
    return _canonical_digest(tuple(values))


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    role: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = ",".join(sorted(expected - observed))
        extra = ",".join(sorted(observed - expected))
        raise ValueError(f"{role}_keys_invalid:missing={missing}:extra={extra}")


def _require_mapping(value: object, role: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{role}_must_be_mapping")
    return value


def _require_sequence(value: object, role: str) -> Sequence[Any]:
    if type(value) is not list:
        raise ValueError(f"{role}_must_be_list")
    return value


def _require_identity(value: object, role: str) -> str:
    if not isinstance(value, str) or _IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{role}_invalid")
    return value


def _require_digest(value: object, role: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{role}_invalid")
    return value


@dataclass(frozen=True, slots=True)
class DevelopmentSplitPolicy:
    allowed_split: str
    formal_later_split_deny_list: tuple[str, ...]
    candidate_selection_mapping: str

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.allowed_split != DEVELOPMENT_SPLIT:
            violations.append("development_allowed_split_invalid")
        if self.formal_later_split_deny_list != FORMAL_LATER_SPLIT_DENY_LIST:
            violations.append("formal_later_split_deny_list_invalid")
        if self.candidate_selection_mapping != "one_unique_case_per_responsibility":
            violations.append("candidate_selection_mapping_rule_invalid")
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class RegisteredStudyRole:
    role_id: str
    registered_split: str
    detector_mode: str
    requires_frozen_hf_only_tau: bool
    execution_allowed_in_development: bool
    identity_dimension_digests: tuple[tuple[str, str], ...]
    roster_digest: str

    def payload_without_roster_digest(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("roster_digest")
        return payload

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        expected = {
            role_id: (split, detector_mode, requires_tau)
            for role_id, split, detector_mode, requires_tau in (
                REGISTERED_STUDY_ROLE_BINDINGS
            )
        }
        if self.role_id not in expected:
            violations.append("study_role_identity_invalid")
            return tuple(violations)
        if (
            self.registered_split,
            self.detector_mode,
            self.requires_frozen_hf_only_tau,
        ) != expected[self.role_id]:
            violations.append("study_role_registered_binding_invalid")
        if self.execution_allowed_in_development is not (
            self.role_id == "development_exploration"
        ):
            violations.append("study_role_development_access_invalid")
        if tuple(name for name, _ in self.identity_dimension_digests) != (
            ISOLATION_DIMENSIONS
        ):
            violations.append("study_role_isolation_dimensions_invalid")
        for dimension, digest in self.identity_dimension_digests:
            expected_digest = _canonical_digest(
                {
                    "dimension": dimension,
                    "registered_split": self.registered_split,
                    "role_id": self.role_id,
                }
            )
            if digest != expected_digest:
                violations.append("study_role_isolation_digest_invalid")
        if self.roster_digest != _canonical_digest(
            self.payload_without_roster_digest()
        ):
            violations.append("study_role_roster_digest_invalid")
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class DevelopmentSplitIsolation:
    isolation_dimensions: tuple[str, ...]
    role_bindings: tuple[RegisteredStudyRole, ...]
    formal_later_deny_roster_digest: str

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.isolation_dimensions != ISOLATION_DIMENSIONS:
            violations.append("split_isolation_dimensions_invalid")
        if tuple(item.role_id for item in self.role_bindings) != tuple(
            item[0] for item in REGISTERED_STUDY_ROLE_BINDINGS
        ):
            violations.append("study_role_order_or_identity_invalid")
        for binding in self.role_bindings:
            violations.extend(binding.validate())
        for dimension in ISOLATION_DIMENSIONS:
            digests = [dict(item.identity_dimension_digests)[dimension] for item in self.role_bindings]
            if len(digests) != len(set(digests)):
                violations.append(f"split_isolation_{dimension}_digest_reused")
        later_rosters = tuple(
            item.roster_digest
            for item in self.role_bindings
            if not item.execution_allowed_in_development
        )
        if self.formal_later_deny_roster_digest != _digest_sequence(later_rosters):
            violations.append("formal_later_deny_roster_digest_invalid")
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class DevelopmentPreflight:
    source_cluster_count: int
    case_ids: tuple[str, ...]
    purpose: str
    counts_as_scientific_coverage: bool

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.source_cluster_count != PREFLIGHT_SOURCE_CLUSTER_COUNT:
            violations.append("preflight_source_cluster_count_invalid")
        if self.case_ids != PREFLIGHT_CASE_IDS:
            violations.append("preflight_case_ids_invalid")
        if self.purpose != "environment_identity_and_throughput_only":
            violations.append("preflight_purpose_invalid")
        if self.counts_as_scientific_coverage is not False:
            violations.append("preflight_scientific_coverage_forbidden")
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class DevelopmentContentStudy:
    branch_ids: tuple[str, ...]
    mixing_coefficients: tuple[float, ...]
    combination_function_ids: tuple[str, ...]
    matched_total_budget_required: bool
    attack_condition_switching_forbidden: bool

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.branch_ids != CONTENT_BRANCH_IDS:
            violations.append("development_content_branch_ids_invalid")
        if self.mixing_coefficients != CONTENT_MIXING_COEFFICIENTS:
            violations.append("development_mixing_coefficients_invalid")
        if self.combination_function_ids != CONTENT_COMBINATION_FUNCTION_IDS:
            violations.append("development_combination_function_ids_invalid")
        if self.matched_total_budget_required is not True:
            violations.append("matched_total_budget_required")
        if self.attack_condition_switching_forbidden is not True:
            violations.append("attack_condition_switching_must_be_forbidden")
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class GeometryOperationCase:
    case_id: str
    operation_family: str
    crop_fraction: float
    scale_factor: float
    rotation_degrees: float


@dataclass(frozen=True, slots=True)
class DevelopmentGeometryStudy:
    operation_cases: tuple[GeometryOperationCase, ...]
    negative_control_case_ids: tuple[str, ...]

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if tuple(item.operation_family for item in self.operation_cases) != (
            GEOMETRY_OPERATION_FAMILIES
        ):
            violations.append("geometry_operation_families_invalid")
        if tuple(item.case_id for item in self.operation_cases) != (
            "identity_transform",
            "bounded_crop_transform",
            "bounded_scale_transform",
            "bounded_rotation_transform",
            "single_compound_transform",
        ):
            violations.append("geometry_operation_case_ids_invalid")
        expected_parameters = (
            (1.0, 1.0, 0.0),
            (0.75, 1.0, 0.0),
            (1.0, 0.7071067811865476, 0.0),
            (1.0, 1.0, 16.0),
            (0.75, 0.7071067811865476, 16.0),
        )
        observed_parameters = tuple(
            (item.crop_fraction, item.scale_factor, item.rotation_degrees)
            for item in self.operation_cases
        )
        if observed_parameters != expected_parameters:
            violations.append("geometry_operation_parameters_invalid")
        if self.negative_control_case_ids != GEOMETRY_NEGATIVE_CONTROL_CASE_IDS:
            violations.append("geometry_negative_control_case_ids_invalid")
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class DevelopmentModuleStudy:
    responsibility_id: str
    scientific_question_id: str
    development_case_id: str
    candidate_selection_case_id: str
    candidate_identity: str
    candidate_config_digest: str
    paired_ablation_identity: str
    negative_control_case_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    record_field_names: tuple[str, ...]
    prerequisite_responsibility_ids: tuple[str, ...]
    dependency_stop_rule: str
    module_outcome_rule: str
    allowed_module_outcomes: tuple[str, ...]
    scientific_source_cluster_scale: int
    content_branch_ids: tuple[str, ...]
    geometry_case_ids: tuple[str, ...]

    def candidate_config_payload(self) -> dict[str, object]:
        return {
            "candidate_identity": self.candidate_identity,
            "content_branch_ids": self.content_branch_ids,
            "geometry_case_ids": self.geometry_case_ids,
            "paired_ablation_identity": self.paired_ablation_identity,
            "responsibility_id": self.responsibility_id,
        }


def _expected_module_scale(responsibility_id: str) -> int:
    if responsibility_id in CHEAP_DETECTION_RESPONSIBILITIES:
        return CHEAP_DETECTION_SOURCE_CLUSTER_COUNT
    if responsibility_id in CRITICAL_PAIR_RESPONSIBILITIES:
        return CRITICAL_PAIR_SOURCE_CLUSTER_COUNT
    return BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT


def validate_development_module_matrix(
    matrix: Sequence[DevelopmentModuleStudy],
    geometry_case_ids: frozenset[str],
) -> tuple[str, ...]:
    violations: list[str] = []
    responsibilities = tuple(item.responsibility_id for item in matrix)
    if responsibilities != REQUIRED_METHOD_RESPONSIBILITIES:
        violations.append("development_responsibility_order_or_identity_mismatch")
    unique_fields = (
        ("scientific_question_id", tuple(item.scientific_question_id for item in matrix)),
        ("development_case_id", tuple(item.development_case_id for item in matrix)),
        (
            "candidate_selection_case_id",
            tuple(item.candidate_selection_case_id for item in matrix),
        ),
        ("candidate_identity", tuple(item.candidate_identity for item in matrix)),
        (
            "paired_ablation_identity",
            tuple(item.paired_ablation_identity for item in matrix),
        ),
    )
    for field_name, values in unique_fields:
        if any(_IDENTITY_PATTERN.fullmatch(value) is None for value in values):
            violations.append(f"{field_name}_missing_or_invalid")
        if len(values) != len(set(values)):
            violations.append(f"{field_name}_duplicate")
    seen: set[str] = set()
    geometry_responsibilities = set(REQUIRED_METHOD_RESPONSIBILITIES[8:])
    for item in matrix:
        prefix = item.responsibility_id
        if item.candidate_config_digest != _canonical_digest(
            item.candidate_config_payload()
        ):
            violations.append(f"{prefix}:candidate_config_digest_invalid")
        if not item.negative_control_case_ids:
            violations.append(f"{prefix}:negative_control_case_ids_missing")
        if not item.metric_ids:
            violations.append(f"{prefix}:metric_ids_missing")
        if not item.record_field_names:
            violations.append(f"{prefix}:record_field_names_missing")
        if any(_IDENTITY_PATTERN.fullmatch(value) is None for value in (
            *item.negative_control_case_ids,
            *item.metric_ids,
            *item.record_field_names,
        )):
            violations.append(f"{prefix}:module_identity_value_invalid")
        if len(item.record_field_names) != len(set(item.record_field_names)):
            violations.append(f"{prefix}:record_field_name_duplicate")
        if not COMMON_MODULE_RECORD_FIELDS.issubset(set(item.record_field_names)):
            violations.append(f"{prefix}:common_module_record_fields_missing")
        if set(item.record_field_names) - REGISTERED_DEVELOPMENT_RECORD_FIELDS:
            violations.append(f"{prefix}:record_field_name_unregistered")
        if any(
            dependency not in seen
            for dependency in item.prerequisite_responsibility_ids
        ):
            violations.append(f"{prefix}:dependency_order_invalid")
        if item.dependency_stop_rule != DEPENDENCY_STOP_RULE:
            violations.append(f"{prefix}:dependency_stop_rule_invalid")
        if item.module_outcome_rule != MODULE_OUTCOME_RULE:
            violations.append(f"{prefix}:module_outcome_rule_invalid")
        if item.allowed_module_outcomes != MODULE_OUTCOMES:
            violations.append(f"{prefix}:allowed_module_outcomes_invalid")
        if item.scientific_source_cluster_scale != _expected_module_scale(prefix):
            violations.append(f"{prefix}:scientific_source_cluster_scale_invalid")
        if set(item.content_branch_ids) - set(CONTENT_BRANCH_IDS):
            violations.append(f"{prefix}:content_branch_identity_invalid")
        if (
            prefix in set(REQUIRED_METHOD_RESPONSIBILITIES[1:8])
            and "clean_control" not in item.content_branch_ids
        ):
            violations.append(f"{prefix}:clean_control_missing")
        if set(item.geometry_case_ids) - geometry_case_ids:
            violations.append(f"{prefix}:geometry_case_identity_invalid")
        if (
            prefix in geometry_responsibilities
            and set(item.geometry_case_ids) != geometry_case_ids
        ):
            violations.append(f"{prefix}:geometry_case_coverage_invalid")
        if prefix not in geometry_responsibilities and item.geometry_case_ids:
            violations.append(f"{prefix}:geometry_case_ids_forbidden")
        seen.add(prefix)
    return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class DevelopmentStudyUnit:
    unit_index: int
    phase: str
    responsibility_id: str
    source_cluster_ordinal: int
    content_branch_ids: tuple[str, ...]
    geometry_case_ids: tuple[str, ...]
    maximum_record_attempts: int
    maximum_duration_seconds: int


def _build_study_unit_roster(
    matrix: Sequence[DevelopmentModuleStudy],
) -> tuple[DevelopmentStudyUnit, ...]:
    by_responsibility = {item.responsibility_id: item for item in matrix}
    ordered: list[tuple[str, str, int]] = []
    for cluster_ordinal in range(BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT):
        ordered.extend(
            ("scientific_breadth", responsibility_id, cluster_ordinal)
            for responsibility_id in REQUIRED_METHOD_RESPONSIBILITIES
        )
    for cluster_ordinal in range(
        BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT,
        CRITICAL_PAIR_SOURCE_CLUSTER_COUNT,
    ):
        ordered.extend(
            ("critical_pair_extension", responsibility_id, cluster_ordinal)
            for responsibility_id in CRITICAL_PAIR_RESPONSIBILITIES
        )
    for cluster_ordinal in range(
        BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT,
        CHEAP_DETECTION_SOURCE_CLUSTER_COUNT,
    ):
        ordered.extend(
            ("cheap_detection_extension", responsibility_id, cluster_ordinal)
            for responsibility_id in CHEAP_DETECTION_RESPONSIBILITIES
        )
    return tuple(
        DevelopmentStudyUnit(
            unit_index=index,
            phase=phase,
            responsibility_id=responsibility_id,
            source_cluster_ordinal=cluster_ordinal,
            content_branch_ids=by_responsibility[responsibility_id].content_branch_ids,
            geometry_case_ids=by_responsibility[responsibility_id].geometry_case_ids,
            maximum_record_attempts=MAXIMUM_RECORD_ATTEMPTS,
            maximum_duration_seconds=MAXIMUM_UNIT_DURATION_SECONDS,
        )
        for index, (phase, responsibility_id, cluster_ordinal) in enumerate(ordered)
    )


@dataclass(frozen=True, slots=True)
class DevelopmentStudyBudget:
    preflight_source_cluster_count: int
    wiring_source_cluster_count: int
    wiring_counts_as_scientific_coverage: bool
    scientific_source_cluster_scales: tuple[int, ...]
    maximum_scientific_units: int
    maximum_total_branch_units: int
    maximum_record_attempts_per_unit: int
    maximum_total_record_attempts: int
    maximum_duration_seconds_per_unit: int
    unit_order: tuple[str, ...]
    score_adaptive_unit_changes_forbidden: bool
    unit_roster_digest: str

    def validate(
        self,
        matrix: Sequence[DevelopmentModuleStudy],
        roster: Sequence[DevelopmentStudyUnit],
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if self.preflight_source_cluster_count != PREFLIGHT_SOURCE_CLUSTER_COUNT:
            violations.append("study_budget_preflight_count_invalid")
        if self.wiring_source_cluster_count != WIRING_SOURCE_CLUSTER_COUNT:
            violations.append("study_budget_wiring_count_invalid")
        if self.wiring_counts_as_scientific_coverage is not False:
            violations.append("wiring_scientific_coverage_forbidden")
        if self.scientific_source_cluster_scales != SCIENTIFIC_SOURCE_CLUSTER_SCALES:
            violations.append("scientific_source_cluster_scales_invalid")
        if self.maximum_record_attempts_per_unit != MAXIMUM_RECORD_ATTEMPTS:
            violations.append("maximum_record_attempts_per_unit_invalid")
        if self.maximum_duration_seconds_per_unit != MAXIMUM_UNIT_DURATION_SECONDS:
            violations.append("maximum_duration_seconds_per_unit_invalid")
        if self.unit_order != DEVELOPMENT_UNIT_ORDER:
            violations.append("development_unit_order_invalid")
        if self.score_adaptive_unit_changes_forbidden is not True:
            violations.append("score_adaptive_unit_changes_must_be_forbidden")
        if self.maximum_scientific_units != len(roster):
            violations.append("maximum_scientific_units_invalid")
        branch_units = sum(max(1, len(unit.content_branch_ids)) for unit in roster)
        if self.maximum_total_branch_units != branch_units:
            violations.append("maximum_total_branch_units_invalid")
        if self.maximum_total_record_attempts != branch_units * MAXIMUM_RECORD_ATTEMPTS:
            violations.append("maximum_total_record_attempts_invalid")
        if self.unit_roster_digest != _canonical_digest(
            tuple(asdict(unit) for unit in roster)
        ):
            violations.append("unit_roster_digest_invalid")
        if tuple(unit.unit_index for unit in roster) != tuple(range(len(roster))):
            violations.append("unit_roster_index_invalid")
        first_breadth = roster[: len(REQUIRED_METHOD_RESPONSIBILITIES)]
        if tuple(unit.responsibility_id for unit in first_breadth) != (
            REQUIRED_METHOD_RESPONSIBILITIES
        ):
            violations.append("unit_roster_breadth_first_invalid")
        observed_counts = {
            responsibility_id: sum(
                unit.responsibility_id == responsibility_id for unit in roster
            )
            for responsibility_id in REQUIRED_METHOD_RESPONSIBILITIES
        }
        expected_counts = {
            item.responsibility_id: item.scientific_source_cluster_scale
            for item in matrix
        }
        if observed_counts != expected_counts:
            violations.append("unit_roster_module_scale_mismatch")
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class DevelopmentThresholdCrossFitPolicy:
    source_split: str
    fold_count: int
    fit_role: str
    score_role: str
    threshold_role: str
    allowed_input_roles: tuple[str, ...]
    input_manifest_binding_required: bool
    detector_binding_required: bool
    threshold_rule_binding_required: bool
    recovery_probe_cluster_exclusion_required: bool
    invalidation_semantics: str
    invalid_for_splits: tuple[str, ...]

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        expected = (
            (self.source_split, DEVELOPMENT_SPLIT, "cross_fit_source_split_invalid"),
            (self.fold_count, DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT, "cross_fit_fold_count_invalid"),
            (self.fit_role, DEVELOPMENT_THRESHOLD_FIT_ROLE, "cross_fit_fit_role_invalid"),
            (self.score_role, DEVELOPMENT_THRESHOLD_SCORE_ROLE, "cross_fit_score_role_invalid"),
            (self.threshold_role, DEVELOPMENT_THRESHOLD_ROLE, "cross_fit_threshold_role_invalid"),
            (self.allowed_input_roles, DEVELOPMENT_THRESHOLD_INPUT_ROLES, "cross_fit_input_roles_invalid"),
            (self.invalidation_semantics, DEVELOPMENT_THRESHOLD_INVALIDATION, "cross_fit_invalidation_semantics_invalid"),
            (self.invalid_for_splits, FORMAL_LATER_SPLIT_DENY_LIST, "cross_fit_invalid_split_set_invalid"),
        )
        violations.extend(reason for observed, wanted, reason in expected if observed != wanted)
        for value, reason in (
            (self.input_manifest_binding_required, "threshold_manifest_binding_required"),
            (self.detector_binding_required, "threshold_detector_binding_required"),
            (self.threshold_rule_binding_required, "threshold_rule_binding_required"),
            (
                self.recovery_probe_cluster_exclusion_required,
                "threshold_recovery_probe_exclusion_required",
            ),
        ):
            if value is not True:
                violations.append(reason)
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentExplorationProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
    split_policy: DevelopmentSplitPolicy
    split_isolation: DevelopmentSplitIsolation
    preflight: DevelopmentPreflight
    study_budget: DevelopmentStudyBudget
    provisional_threshold_cross_fit: DevelopmentThresholdCrossFitPolicy
    content_study: DevelopmentContentStudy
    geometry_study: DevelopmentGeometryStudy
    module_matrix: tuple[DevelopmentModuleStudy, ...]
    unit_roster: tuple[DevelopmentStudyUnit, ...]
    module_outcomes: tuple[str, ...]
    candidate_recommendations: tuple[str, ...]
    scientific_claim_boundary: str

    def digest(self) -> str:
        return _canonical_digest(asdict(self))

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            violations.append("development_schema_version_invalid")
        if self.protocol_id != PROTOCOL_ID:
            violations.append("development_protocol_id_invalid")
        if self.protocol_version != DEVELOPMENT_EXPLORATION_PROTOCOL_VERSION:
            violations.append("development_protocol_version_invalid")
        violations.extend(self.split_policy.validate())
        violations.extend(self.split_isolation.validate())
        violations.extend(self.preflight.validate())
        violations.extend(self.provisional_threshold_cross_fit.validate())
        violations.extend(self.content_study.validate())
        violations.extend(self.geometry_study.validate())
        geometry_ids = frozenset(
            (*[item.case_id for item in self.geometry_study.operation_cases], *self.geometry_study.negative_control_case_ids)
        )
        violations.extend(validate_development_module_matrix(self.module_matrix, geometry_ids))
        violations.extend(self.study_budget.validate(self.module_matrix, self.unit_roster))
        if self.module_outcomes != MODULE_OUTCOMES:
            violations.append("development_module_outcomes_invalid")
        if self.candidate_recommendations != CANDIDATE_RECOMMENDATIONS:
            violations.append("candidate_recommendations_invalid")
        if self.scientific_claim_boundary != DEVELOPMENT_CLAIM_BOUNDARY:
            violations.append("development_scientific_claim_boundary_invalid")
        return tuple(dict.fromkeys(violations))


def _load_study_roles(raw_roles: object) -> tuple[RegisteredStudyRole, ...]:
    roles: list[RegisteredStudyRole] = []
    for index, raw_value in enumerate(_require_sequence(raw_roles, "study_role_bindings")):
        item = _require_mapping(raw_value, f"study_role_binding:{index}")
        _require_exact_keys(
            item,
            frozenset(
                {
                    "role_id",
                    "registered_split",
                    "detector_mode",
                    "requires_frozen_hf_only_tau",
                    "execution_allowed_in_development",
                }
            ),
            f"study_role_binding:{index}",
        )
        role_id = _require_identity(item["role_id"], "study_role_id")
        registered_split = _require_identity(
            item["registered_split"], "study_role_registered_split"
        )
        dimensions = tuple(
            (
                dimension,
                _canonical_digest(
                    {
                        "dimension": dimension,
                        "registered_split": registered_split,
                        "role_id": role_id,
                    }
                ),
            )
            for dimension in ISOLATION_DIMENSIONS
        )
        payload = {
            "role_id": role_id,
            "registered_split": registered_split,
            "detector_mode": _require_identity(
                item["detector_mode"], "study_role_detector_mode"
            ),
            "requires_frozen_hf_only_tau": item["requires_frozen_hf_only_tau"],
            "execution_allowed_in_development": item[
                "execution_allowed_in_development"
            ],
            "identity_dimension_digests": dimensions,
        }
        roles.append(
            RegisteredStudyRole(
                **payload,
                roster_digest=_canonical_digest(payload),
            )
        )
    return tuple(roles)


def load_frozen_development_exploration_protocol(
    path: str | Path,
) -> FrozenDevelopmentExplorationProtocol:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if type(raw) is not dict:
        raise ValueError("development_protocol_root_must_be_mapping")
    _require_exact_keys(raw, _EXACT_TOP_LEVEL_KEYS, "development_protocol")

    split_raw = _require_mapping(raw["split_policy"], "split_policy")
    _require_exact_keys(
        split_raw,
        frozenset(
            {
                "allowed_split",
                "formal_later_split_deny_list",
                "candidate_selection_mapping",
            }
        ),
        "split_policy",
    )
    split_policy = DevelopmentSplitPolicy(
        allowed_split=split_raw["allowed_split"],
        formal_later_split_deny_list=tuple(
            _require_sequence(
                split_raw["formal_later_split_deny_list"],
                "formal_later_split_deny_list",
            )
        ),
        candidate_selection_mapping=split_raw["candidate_selection_mapping"],
    )

    isolation_raw = _require_mapping(raw["split_isolation"], "split_isolation")
    _require_exact_keys(
        isolation_raw,
        frozenset({"isolation_dimensions", "role_bindings"}),
        "split_isolation",
    )
    role_bindings = _load_study_roles(isolation_raw["role_bindings"])
    split_isolation = DevelopmentSplitIsolation(
        isolation_dimensions=tuple(
            _require_sequence(
                isolation_raw["isolation_dimensions"], "isolation_dimensions"
            )
        ),
        role_bindings=role_bindings,
        formal_later_deny_roster_digest=_digest_sequence(
            tuple(
                item.roster_digest
                for item in role_bindings
                if not item.execution_allowed_in_development
            )
        ),
    )

    preflight_raw = _require_mapping(raw["preflight"], "preflight")
    _require_exact_keys(
        preflight_raw,
        frozenset(
            {
                "source_cluster_count",
                "case_ids",
                "purpose",
                "counts_as_scientific_coverage",
            }
        ),
        "preflight",
    )
    preflight = DevelopmentPreflight(
        source_cluster_count=preflight_raw["source_cluster_count"],
        case_ids=tuple(_require_sequence(preflight_raw["case_ids"], "preflight_case_ids")),
        purpose=preflight_raw["purpose"],
        counts_as_scientific_coverage=preflight_raw[
            "counts_as_scientific_coverage"
        ],
    )

    content_raw = _require_mapping(raw["content_study"], "content_study")
    _require_exact_keys(
        content_raw,
        frozenset(
            {
                "branch_ids",
                "mixing_coefficients",
                "combination_function_ids",
                "matched_total_budget_required",
                "attack_condition_switching_forbidden",
            }
        ),
        "content_study",
    )
    content_study = DevelopmentContentStudy(
        branch_ids=tuple(_require_sequence(content_raw["branch_ids"], "content_branch_ids")),
        mixing_coefficients=tuple(
            _require_sequence(content_raw["mixing_coefficients"], "mixing_coefficients")
        ),
        combination_function_ids=tuple(
            _require_sequence(
                content_raw["combination_function_ids"],
                "combination_function_ids",
            )
        ),
        matched_total_budget_required=content_raw["matched_total_budget_required"],
        attack_condition_switching_forbidden=content_raw[
            "attack_condition_switching_forbidden"
        ],
    )

    geometry_raw = _require_mapping(raw["geometry_study"], "geometry_study")
    _require_exact_keys(
        geometry_raw,
        frozenset({"operation_cases", "negative_control_case_ids"}),
        "geometry_study",
    )
    operation_cases: list[GeometryOperationCase] = []
    for index, raw_value in enumerate(
        _require_sequence(geometry_raw["operation_cases"], "geometry_operation_cases")
    ):
        item = _require_mapping(raw_value, f"geometry_operation_case:{index}")
        _require_exact_keys(
            item,
            frozenset(
                {
                    "case_id",
                    "operation_family",
                    "crop_fraction",
                    "scale_factor",
                    "rotation_degrees",
                }
            ),
            f"geometry_operation_case:{index}",
        )
        operation_cases.append(GeometryOperationCase(**item))
    geometry_study = DevelopmentGeometryStudy(
        operation_cases=tuple(operation_cases),
        negative_control_case_ids=tuple(
            _require_sequence(
                geometry_raw["negative_control_case_ids"],
                "geometry_negative_control_case_ids",
            )
        ),
    )

    matrix_raw = _require_sequence(raw["module_matrix"], "module_matrix")
    matrix: list[DevelopmentModuleStudy] = []
    module_keys = frozenset(
        {
            "responsibility_id",
            "scientific_question_id",
            "development_case_id",
            "candidate_selection_case_id",
            "candidate_identity",
            "candidate_config_digest",
            "paired_ablation_identity",
            "negative_control_case_ids",
            "metric_ids",
            "record_field_names",
            "prerequisite_responsibility_ids",
            "dependency_stop_rule",
            "module_outcome_rule",
            "allowed_module_outcomes",
            "scientific_source_cluster_scale",
            "content_branch_ids",
            "geometry_case_ids",
        }
    )
    for index, raw_value in enumerate(matrix_raw):
        item = _require_mapping(raw_value, f"module_matrix_entry:{index}")
        _require_exact_keys(item, module_keys, f"module_matrix_entry:{index}")
        matrix.append(
            DevelopmentModuleStudy(
                responsibility_id=_require_identity(
                    item["responsibility_id"], "responsibility_id"
                ),
                scientific_question_id=_require_identity(
                    item["scientific_question_id"], "scientific_question_id"
                ),
                development_case_id=_require_identity(
                    item["development_case_id"], "development_case_id"
                ),
                candidate_selection_case_id=_require_identity(
                    item["candidate_selection_case_id"],
                    "candidate_selection_case_id",
                ),
                candidate_identity=_require_identity(
                    item["candidate_identity"], "candidate_identity"
                ),
                candidate_config_digest=_require_digest(
                    item["candidate_config_digest"], "candidate_config_digest"
                ),
                paired_ablation_identity=_require_identity(
                    item["paired_ablation_identity"], "paired_ablation_identity"
                ),
                negative_control_case_ids=tuple(
                    _require_sequence(
                        item["negative_control_case_ids"],
                        "negative_control_case_ids",
                    )
                ),
                metric_ids=tuple(_require_sequence(item["metric_ids"], "metric_ids")),
                record_field_names=tuple(
                    _require_sequence(item["record_field_names"], "record_field_names")
                ),
                prerequisite_responsibility_ids=tuple(
                    _require_sequence(
                        item["prerequisite_responsibility_ids"],
                        "prerequisite_responsibility_ids",
                    )
                ),
                dependency_stop_rule=item["dependency_stop_rule"],
                module_outcome_rule=item["module_outcome_rule"],
                allowed_module_outcomes=tuple(
                    _require_sequence(
                        item["allowed_module_outcomes"], "allowed_module_outcomes"
                    )
                ),
                scientific_source_cluster_scale=item[
                    "scientific_source_cluster_scale"
                ],
                content_branch_ids=tuple(
                    _require_sequence(item["content_branch_ids"], "module_content_branch_ids")
                ),
                geometry_case_ids=tuple(
                    _require_sequence(item["geometry_case_ids"], "module_geometry_case_ids")
                ),
            )
        )
    matrix_tuple = tuple(matrix)
    unit_roster = _build_study_unit_roster(matrix_tuple)

    budget_raw = _require_mapping(raw["study_budget"], "study_budget")
    budget_keys = frozenset(
        {
            "preflight_source_cluster_count",
            "wiring_source_cluster_count",
            "wiring_counts_as_scientific_coverage",
            "scientific_source_cluster_scales",
            "maximum_scientific_units",
            "maximum_total_branch_units",
            "maximum_record_attempts_per_unit",
            "maximum_total_record_attempts",
            "maximum_duration_seconds_per_unit",
            "unit_order",
            "score_adaptive_unit_changes_forbidden",
            "unit_roster_digest",
        }
    )
    _require_exact_keys(budget_raw, budget_keys, "study_budget")
    study_budget = DevelopmentStudyBudget(
        preflight_source_cluster_count=budget_raw["preflight_source_cluster_count"],
        wiring_source_cluster_count=budget_raw["wiring_source_cluster_count"],
        wiring_counts_as_scientific_coverage=budget_raw[
            "wiring_counts_as_scientific_coverage"
        ],
        scientific_source_cluster_scales=tuple(
            _require_sequence(
                budget_raw["scientific_source_cluster_scales"],
                "scientific_source_cluster_scales",
            )
        ),
        maximum_scientific_units=budget_raw["maximum_scientific_units"],
        maximum_total_branch_units=budget_raw["maximum_total_branch_units"],
        maximum_record_attempts_per_unit=budget_raw[
            "maximum_record_attempts_per_unit"
        ],
        maximum_total_record_attempts=budget_raw["maximum_total_record_attempts"],
        maximum_duration_seconds_per_unit=budget_raw[
            "maximum_duration_seconds_per_unit"
        ],
        unit_order=tuple(_require_sequence(budget_raw["unit_order"], "unit_order")),
        score_adaptive_unit_changes_forbidden=budget_raw[
            "score_adaptive_unit_changes_forbidden"
        ],
        unit_roster_digest=_require_digest(
            budget_raw["unit_roster_digest"], "unit_roster_digest"
        ),
    )

    threshold_raw = _require_mapping(
        raw["provisional_threshold_cross_fit"],
        "provisional_threshold_cross_fit",
    )
    threshold_keys = frozenset(
        {
            "source_split",
            "fold_count",
            "fit_role",
            "score_role",
            "threshold_role",
            "allowed_input_roles",
            "input_manifest_binding_required",
            "detector_binding_required",
            "threshold_rule_binding_required",
            "recovery_probe_cluster_exclusion_required",
            "invalidation_semantics",
            "invalid_for_splits",
        }
    )
    _require_exact_keys(threshold_raw, threshold_keys, "provisional_threshold_cross_fit")
    threshold_policy = DevelopmentThresholdCrossFitPolicy(
        source_split=threshold_raw["source_split"],
        fold_count=threshold_raw["fold_count"],
        fit_role=threshold_raw["fit_role"],
        score_role=threshold_raw["score_role"],
        threshold_role=threshold_raw["threshold_role"],
        allowed_input_roles=tuple(
            _require_sequence(threshold_raw["allowed_input_roles"], "threshold_input_roles")
        ),
        input_manifest_binding_required=threshold_raw[
            "input_manifest_binding_required"
        ],
        detector_binding_required=threshold_raw["detector_binding_required"],
        threshold_rule_binding_required=threshold_raw[
            "threshold_rule_binding_required"
        ],
        recovery_probe_cluster_exclusion_required=threshold_raw[
            "recovery_probe_cluster_exclusion_required"
        ],
        invalidation_semantics=threshold_raw["invalidation_semantics"],
        invalid_for_splits=tuple(
            _require_sequence(threshold_raw["invalid_for_splits"], "threshold_invalid_for_splits")
        ),
    )

    outcome_raw = _require_mapping(raw["module_outcomes"], "module_outcomes")
    _require_exact_keys(
        outcome_raw,
        frozenset({"allowed", "candidate_recommendations"}),
        "module_outcomes",
    )
    protocol = FrozenDevelopmentExplorationProtocol(
        schema_version=raw["schema_version"],
        protocol_id=raw["protocol_id"],
        protocol_version=raw["protocol_version"],
        split_policy=split_policy,
        split_isolation=split_isolation,
        preflight=preflight,
        study_budget=study_budget,
        provisional_threshold_cross_fit=threshold_policy,
        content_study=content_study,
        geometry_study=geometry_study,
        module_matrix=matrix_tuple,
        unit_roster=unit_roster,
        module_outcomes=tuple(
            _require_sequence(outcome_raw["allowed"], "module_outcomes_allowed")
        ),
        candidate_recommendations=tuple(
            _require_sequence(
                outcome_raw["candidate_recommendations"],
                "candidate_recommendations",
            )
        ),
        scientific_claim_boundary=raw["scientific_claim_boundary"],
    )
    violations = protocol.validate()
    if violations:
        raise ValueError(",".join(violations))
    return protocol


def enumerate_development_study_units(
    protocol: FrozenDevelopmentExplorationProtocol,
) -> tuple[DevelopmentStudyUnit, ...]:
    if type(protocol) is not FrozenDevelopmentExplorationProtocol or protocol.validate():
        raise ValueError("development_protocol_invalid")
    return protocol.unit_roster


def development_assignments_only(
    manifest: FrozenSplitManifest,
) -> tuple[SplitAssignment, ...]:
    if type(manifest) is not FrozenSplitManifest:
        raise TypeError("development_split_manifest_exact_type_required")
    violations = manifest.validate(require_all_splits=False)
    if violations:
        raise ValueError(",".join(violations))
    observed = {assignment.split for assignment in manifest.assignments}
    if observed != {DEVELOPMENT_SPLIT}:
        forbidden = sorted(observed - {DEVELOPMENT_SPLIT})
        raise PermissionError(
            f"development_exploration_split_forbidden:{','.join(forbidden)}"
        )
    return manifest.assignments


@dataclass(frozen=True, slots=True)
class DevelopmentCrossFitFold:
    fold_index: int
    fit_source_cluster_ids: tuple[str, ...]
    recovery_probe_source_cluster_ids: tuple[str, ...]
    fit_source_cluster_digest: str
    recovery_probe_source_cluster_digest: str

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.fold_index not in range(DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT):
            violations.append("cross_fit_fold_index_invalid")
        if not self.fit_source_cluster_ids or not self.recovery_probe_source_cluster_ids:
            violations.append("cross_fit_fold_cluster_set_empty")
        if set(self.fit_source_cluster_ids) & set(self.recovery_probe_source_cluster_ids):
            violations.append("cross_fit_threshold_recovery_probe_leakage")
        if self.fit_source_cluster_digest != _digest_sequence(self.fit_source_cluster_ids):
            violations.append("cross_fit_fit_cluster_digest_invalid")
        if self.recovery_probe_source_cluster_digest != _digest_sequence(
            self.recovery_probe_source_cluster_ids
        ):
            violations.append("cross_fit_recovery_probe_cluster_digest_invalid")
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentCrossFitPlan:
    responsibility_id: str
    source_split: str
    source_cluster_count: int
    threshold_role: str
    invalid_for_splits: tuple[str, ...]
    folds: tuple[DevelopmentCrossFitFold, ...]
    scientific_claims_supported: bool

    def digest(self) -> str:
        return _canonical_digest(asdict(self))

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.responsibility_id not in REQUIRED_METHOD_RESPONSIBILITIES:
            violations.append("cross_fit_responsibility_invalid")
        if self.source_split != DEVELOPMENT_SPLIT:
            violations.append("cross_fit_plan_split_invalid")
        if self.source_cluster_count not in SCIENTIFIC_SOURCE_CLUSTER_SCALES:
            violations.append("cross_fit_plan_scientific_scale_invalid")
        if self.threshold_role != DEVELOPMENT_THRESHOLD_ROLE:
            violations.append("cross_fit_plan_threshold_role_invalid")
        if self.invalid_for_splits != FORMAL_LATER_SPLIT_DENY_LIST:
            violations.append("cross_fit_plan_invalidation_invalid")
        if len(self.folds) != DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT:
            violations.append("cross_fit_plan_fold_count_invalid")
        for fold in self.folds:
            violations.extend(fold.validate())
        probed = tuple(
            cluster
            for fold in self.folds
            for cluster in fold.recovery_probe_source_cluster_ids
        )
        if len(probed) != self.source_cluster_count or len(set(probed)) != len(probed):
            violations.append("cross_fit_recovery_probe_partition_invalid")
        all_clusters = set(probed)
        for fold in self.folds:
            if set(fold.fit_source_cluster_ids) | set(
                fold.recovery_probe_source_cluster_ids
            ) != all_clusters:
                violations.append("cross_fit_plan_fold_coverage_invalid")
        if self.scientific_claims_supported is not False:
            violations.append("development_cross_fit_scientific_claim_forbidden")
        return tuple(dict.fromkeys(violations))


def build_development_cross_fit_plan(
    *,
    responsibility_id: str,
    assignments: Sequence[SplitAssignment],
    expected_source_cluster_count: int,
) -> FrozenDevelopmentCrossFitPlan:
    if expected_source_cluster_count == WIRING_SOURCE_CLUSTER_COUNT:
        raise ValueError("wiring_clusters_do_not_count_as_scientific_coverage")
    if expected_source_cluster_count not in SCIENTIFIC_SOURCE_CLUSTER_SCALES:
        raise ValueError("development_scientific_source_cluster_scale_invalid")
    if not assignments:
        raise ValueError("development_cross_fit_assignments_missing")
    for assignment in assignments:
        if type(assignment) is not SplitAssignment:
            raise TypeError("development_cross_fit_assignment_exact_type_required")
        if assignment.split != DEVELOPMENT_SPLIT:
            raise PermissionError("development_cross_fit_later_split_forbidden")
        if type(assignment.identity) is not AnalysisUnitIdentity:
            raise TypeError("development_cross_fit_identity_exact_type_required")
        violations = assignment.identity.validate()
        if violations:
            raise ValueError(",".join(violations))
    cluster_ids = tuple(
        sorted({assignment.identity.source_cluster_id for assignment in assignments})
    )
    if len(cluster_ids) != expected_source_cluster_count:
        raise ValueError("development_cross_fit_source_cluster_count_mismatch")
    folds: list[DevelopmentCrossFitFold] = []
    for fold_index in range(DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT):
        probes = tuple(
            cluster
            for index, cluster in enumerate(cluster_ids)
            if index % DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT == fold_index
        )
        fit = tuple(cluster for cluster in cluster_ids if cluster not in set(probes))
        folds.append(
            DevelopmentCrossFitFold(
                fold_index=fold_index,
                fit_source_cluster_ids=fit,
                recovery_probe_source_cluster_ids=probes,
                fit_source_cluster_digest=_digest_sequence(fit),
                recovery_probe_source_cluster_digest=_digest_sequence(probes),
            )
        )
    plan = FrozenDevelopmentCrossFitPlan(
        responsibility_id=responsibility_id,
        source_split=DEVELOPMENT_SPLIT,
        source_cluster_count=len(cluster_ids),
        threshold_role=DEVELOPMENT_THRESHOLD_ROLE,
        invalid_for_splits=FORMAL_LATER_SPLIT_DENY_LIST,
        folds=tuple(folds),
        scientific_claims_supported=False,
    )
    violations = plan.validate()
    if violations:
        raise ValueError(",".join(violations))
    return plan


@dataclass(frozen=True, slots=True)
class DevelopmentThresholdFitInput:
    source_split: str
    case_role: str
    source_cluster_ids: tuple[str, ...]

    def validate(
        self,
        fold: DevelopmentCrossFitFold,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if self.source_split != DEVELOPMENT_SPLIT:
            violations.append("threshold_fit_input_split_invalid")
        if self.case_role not in DEVELOPMENT_THRESHOLD_INPUT_ROLES:
            violations.append("threshold_fit_input_role_invalid")
        if not self.source_cluster_ids:
            violations.append("threshold_fit_input_clusters_missing")
        if not set(self.source_cluster_ids).issubset(set(fold.fit_source_cluster_ids)):
            violations.append("threshold_fit_input_cluster_not_in_fit_fold")
        if set(self.source_cluster_ids) & set(fold.recovery_probe_source_cluster_ids):
            violations.append("threshold_fit_input_recovery_probe_leakage")
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class DevelopmentProvisionalThreshold:
    threshold_identity: str
    responsibility_id: str
    fold_index: int
    threshold: float
    input_manifest_digest: str
    detector_identity: str
    detector_config_digest: str
    threshold_rule_digest: str
    fit_inputs: tuple[DevelopmentThresholdFitInput, ...]
    fit_source_cluster_digest: str
    recovery_probe_source_cluster_digest: str
    source_split: str
    threshold_role: str
    invalid_for_splits: tuple[str, ...]
    scientific_claims_supported: bool

    def payload_without_identity(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("threshold_identity")
        return payload

    def validate(self, plan: FrozenDevelopmentCrossFitPlan) -> tuple[str, ...]:
        violations: list[str] = []
        if type(plan) is not FrozenDevelopmentCrossFitPlan or plan.validate():
            return ("development_cross_fit_plan_invalid",)
        if self.responsibility_id != plan.responsibility_id:
            violations.append("provisional_threshold_responsibility_mismatch")
        if self.fold_index not in range(len(plan.folds)):
            violations.append("provisional_threshold_fold_index_invalid")
            return tuple(violations)
        fold = plan.folds[self.fold_index]
        for fit_input in self.fit_inputs:
            if type(fit_input) is not DevelopmentThresholdFitInput:
                violations.append("threshold_fit_input_exact_type_required")
            else:
                violations.extend(fit_input.validate(fold))
        if {item.case_role for item in self.fit_inputs} != set(
            DEVELOPMENT_THRESHOLD_INPUT_ROLES
        ):
            violations.append("threshold_fit_input_roles_incomplete")
        covered_fit_clusters = {
            cluster for item in self.fit_inputs for cluster in item.source_cluster_ids
        }
        if covered_fit_clusters != set(fold.fit_source_cluster_ids):
            violations.append("threshold_fit_input_cluster_coverage_invalid")
        if self.fit_source_cluster_digest != fold.fit_source_cluster_digest:
            violations.append("provisional_threshold_fit_digest_mismatch")
        if self.recovery_probe_source_cluster_digest != (
            fold.recovery_probe_source_cluster_digest
        ):
            violations.append("provisional_threshold_recovery_probe_digest_mismatch")
        for value, reason in (
            (self.input_manifest_digest, "provisional_threshold_manifest_digest_invalid"),
            (self.detector_config_digest, "provisional_threshold_detector_config_digest_invalid"),
            (self.threshold_rule_digest, "provisional_threshold_rule_digest_invalid"),
        ):
            if _DIGEST_PATTERN.fullmatch(value) is None:
                violations.append(reason)
        if _IDENTITY_PATTERN.fullmatch(self.detector_identity) is None:
            violations.append("provisional_threshold_detector_identity_invalid")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            violations.append("provisional_threshold_value_invalid")
        elif not isfinite(float(self.threshold)):
            violations.append("provisional_threshold_value_non_finite")
        if self.source_split != DEVELOPMENT_SPLIT:
            violations.append("provisional_threshold_source_split_invalid")
        if self.threshold_role != DEVELOPMENT_THRESHOLD_ROLE:
            violations.append("provisional_threshold_role_invalid")
        if self.invalid_for_splits != FORMAL_LATER_SPLIT_DENY_LIST:
            violations.append("provisional_threshold_invalidation_invalid")
        if self.scientific_claims_supported is not False:
            violations.append("provisional_threshold_scientific_claim_forbidden")
        if self.threshold_identity != _canonical_digest(self.payload_without_identity()):
            violations.append("provisional_threshold_identity_invalid")
        return tuple(dict.fromkeys(violations))


def create_development_provisional_threshold(
    plan: FrozenDevelopmentCrossFitPlan,
    *,
    fold_index: int,
    threshold: float,
    input_manifest_digest: str,
    detector_identity: str,
    detector_config_digest: str,
    threshold_rule_digest: str,
    fit_inputs: Sequence[DevelopmentThresholdFitInput],
) -> DevelopmentProvisionalThreshold:
    if type(plan) is not FrozenDevelopmentCrossFitPlan or plan.validate():
        raise ValueError("development_cross_fit_plan_invalid")
    if fold_index not in range(len(plan.folds)):
        raise ValueError("provisional_threshold_fold_index_invalid")
    fold = plan.folds[fold_index]
    payload = {
        "responsibility_id": plan.responsibility_id,
        "fold_index": fold_index,
        "threshold": threshold,
        "input_manifest_digest": input_manifest_digest,
        "detector_identity": detector_identity,
        "detector_config_digest": detector_config_digest,
        "threshold_rule_digest": threshold_rule_digest,
        "fit_inputs": tuple(fit_inputs),
        "fit_source_cluster_digest": fold.fit_source_cluster_digest,
        "recovery_probe_source_cluster_digest": (
            fold.recovery_probe_source_cluster_digest
        ),
        "source_split": DEVELOPMENT_SPLIT,
        "threshold_role": DEVELOPMENT_THRESHOLD_ROLE,
        "invalid_for_splits": FORMAL_LATER_SPLIT_DENY_LIST,
        "scientific_claims_supported": False,
    }
    provisional = DevelopmentProvisionalThreshold(
        threshold_identity=_canonical_digest(
            {
                key: tuple(asdict(item) for item in value)
                if key == "fit_inputs"
                else value
                for key, value in payload.items()
            }
        ),
        **payload,
    )
    violations = provisional.validate(plan)
    if violations:
        raise ValueError(",".join(violations))
    return provisional


def authorize_development_provisional_threshold(
    threshold: DevelopmentProvisionalThreshold,
    plan: FrozenDevelopmentCrossFitPlan,
    *,
    requested_split: str,
    source_cluster_id: str,
) -> None:
    if type(threshold) is not DevelopmentProvisionalThreshold:
        raise TypeError("development_provisional_threshold_exact_type_required")
    violations = threshold.validate(plan)
    if violations:
        raise ValueError(",".join(violations))
    if requested_split != DEVELOPMENT_SPLIT:
        raise PermissionError(
            f"development_provisional_threshold_invalid_for_split:{requested_split}"
        )
    probes = plan.folds[threshold.fold_index].recovery_probe_source_cluster_ids
    if source_cluster_id not in probes:
        raise PermissionError("development_provisional_threshold_fold_leakage")


@dataclass(frozen=True, slots=True)
class DevelopmentModuleExecutionDecision:
    approved: bool
    responsibility_id: str
    missing_prerequisites: tuple[str, ...]
    blocking_responsibilities: tuple[str, ...]
    decision_reason: str


def decide_development_module_execution(
    protocol: FrozenDevelopmentExplorationProtocol,
    responsibility_id: str,
    outcomes_by_responsibility: Mapping[str, str],
) -> DevelopmentModuleExecutionDecision:
    if type(protocol) is not FrozenDevelopmentExplorationProtocol or protocol.validate():
        raise ValueError("development_protocol_invalid")
    studies = {item.responsibility_id: item for item in protocol.module_matrix}
    if responsibility_id not in studies:
        raise ValueError("development_responsibility_invalid")
    unknown = set(outcomes_by_responsibility) - set(studies)
    if unknown:
        raise ValueError("development_outcome_responsibility_unknown")
    if any(value not in MODULE_OUTCOMES for value in outcomes_by_responsibility.values()):
        raise ValueError("development_module_outcome_invalid")
    study = studies[responsibility_id]
    missing = tuple(
        dependency
        for dependency in study.prerequisite_responsibility_ids
        if dependency not in outcomes_by_responsibility
    )
    blocking = tuple(
        dependency
        for dependency in study.prerequisite_responsibility_ids
        if dependency not in missing
        and outcomes_by_responsibility[dependency] != "mechanism_signal_observed"
    )
    if missing:
        return DevelopmentModuleExecutionDecision(
            False,
            responsibility_id,
            missing,
            (),
            "prerequisite_outcome_missing",
        )
    if blocking:
        return DevelopmentModuleExecutionDecision(
            False,
            responsibility_id,
            (),
            blocking,
            DEPENDENCY_STOP_RULE,
        )
    return DevelopmentModuleExecutionDecision(
        True,
        responsibility_id,
        (),
        (),
        "development_execution_authorized",
    )


@dataclass(frozen=True, slots=True)
class DevelopmentModuleOutcomeRecord:
    outcome_record_id: str
    responsibility_id: str
    module_outcome: str
    candidate_recommendation: str
    recommendation_reason: str
    blocking_responsibilities: tuple[str, ...]
    evidence_record_ids: tuple[str, ...]
    provisional_threshold_identities: tuple[str, ...]
    source_record_schema_version: str
    source_record_collection_schema_version: str
    scientific_claims_supported: bool

    def payload_without_identity(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("outcome_record_id")
        return payload

    def validate(
        self,
        protocol: FrozenDevelopmentExplorationProtocol,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if type(protocol) is not FrozenDevelopmentExplorationProtocol or protocol.validate():
            return ("development_protocol_invalid",)
        studies = {item.responsibility_id: item for item in protocol.module_matrix}
        if self.responsibility_id not in studies:
            violations.append("module_outcome_responsibility_invalid")
        if self.module_outcome not in MODULE_OUTCOMES:
            violations.append("module_outcome_invalid")
        if self.candidate_recommendation not in CANDIDATE_RECOMMENDATIONS:
            violations.append("candidate_recommendation_invalid")
        if (
            self.candidate_recommendation == "candidate_worth_further_selection"
            and self.module_outcome != "mechanism_signal_observed"
        ):
            violations.append("candidate_recommendation_not_supported_by_outcome")
        if not self.recommendation_reason.strip():
            violations.append("recommendation_reason_missing")
        if self.module_outcome == "implementation_blocked":
            if not self.blocking_responsibilities:
                violations.append("implementation_blocking_responsibility_missing")
            elif self.responsibility_id in studies and not set(
                self.blocking_responsibilities
            ).issubset(
                set(studies[self.responsibility_id].prerequisite_responsibility_ids)
            ):
                violations.append("implementation_blocking_responsibility_invalid")
        elif self.blocking_responsibilities:
            violations.append("blocking_responsibility_forbidden_for_outcome")
        if not self.evidence_record_ids or any(not value for value in self.evidence_record_ids):
            violations.append("module_outcome_evidence_record_ids_invalid")
        if len(set(self.evidence_record_ids)) != len(self.evidence_record_ids):
            violations.append("module_outcome_evidence_record_id_duplicate")
        if any(
            _DIGEST_PATTERN.fullmatch(value) is None
            for value in self.provisional_threshold_identities
        ):
            violations.append("module_outcome_provisional_threshold_identity_invalid")
        if self.source_record_schema_version != INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION:
            violations.append("source_record_schema_version_invalid")
        if self.source_record_collection_schema_version != (
            INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
        ):
            violations.append("source_record_collection_schema_version_invalid")
        if self.scientific_claims_supported is not False:
            violations.append("development_module_outcome_scientific_claim_forbidden")
        if self.outcome_record_id != _canonical_digest(self.payload_without_identity()):
            violations.append("development_module_outcome_identity_invalid")
        return tuple(dict.fromkeys(violations))


def create_development_module_outcome_record(
    protocol: FrozenDevelopmentExplorationProtocol,
    *,
    responsibility_id: str,
    module_outcome: str,
    candidate_recommendation: str,
    recommendation_reason: str,
    evidence_record_ids: Sequence[str],
    blocking_responsibilities: Sequence[str] = (),
    provisional_threshold_identities: Sequence[str] = (),
) -> DevelopmentModuleOutcomeRecord:
    studies = {item.responsibility_id: item for item in protocol.module_matrix}
    if responsibility_id not in studies or module_outcome not in MODULE_OUTCOMES:
        raise ValueError("development_module_outcome_input_invalid")
    payload = {
        "responsibility_id": responsibility_id,
        "module_outcome": module_outcome,
        "candidate_recommendation": candidate_recommendation,
        "recommendation_reason": recommendation_reason,
        "blocking_responsibilities": tuple(blocking_responsibilities),
        "evidence_record_ids": tuple(evidence_record_ids),
        "provisional_threshold_identities": tuple(provisional_threshold_identities),
        "source_record_schema_version": INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
        "source_record_collection_schema_version": (
            INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
        ),
        "scientific_claims_supported": False,
    }
    outcome = DevelopmentModuleOutcomeRecord(
        outcome_record_id=_canonical_digest(payload),
        **payload,
    )
    violations = outcome.validate(protocol)
    if violations:
        raise ValueError(",".join(violations))
    return outcome
