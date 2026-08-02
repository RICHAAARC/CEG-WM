"""Development-only preliminary scientific exploration for all 13 duties.

This protocol freezes study identities and fail-closed boundaries.  It does not
run a method, fit a formal threshold, write records, or promote a candidate.
Per-unit evidence remains the responsibility of ``GovernedRecordWriter`` and
uses the existing internal record schemas bound below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from experiments.protocol.internal_matrix import (
    REQUIRED_METHOD_RESPONSIBILITIES,
    REQUIRED_RECORD_FIELD_GROUPS,
)
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
DEVELOPMENT_EXPLORATION_PROTOCOL_VERSION = "1.0.0"
SCHEMA_VERSION = (
    "ceg_wm_development_module_exploration_protocol_schema"
)
DEVELOPMENT_SPLIT = "development"
FORMAL_LATER_SPLIT_DENY_LIST = INTERNAL_VALIDATION_SPLITS[1:]
WIRING_SOURCE_CLUSTER_COUNT = 8
SCIENTIFIC_SOURCE_CLUSTER_SCALES = (16, 32, 64)
DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT = 4
DEVELOPMENT_THRESHOLD_ROLE = "development_provisional_cross_fit"
DEVELOPMENT_THRESHOLD_FIT_ROLE = "all_other_source_cluster_folds"
DEVELOPMENT_THRESHOLD_SCORE_ROLE = "one_held_out_source_cluster_fold"
DEVELOPMENT_THRESHOLD_INVALIDATION = (
    "invalidate_before_candidate_selection_and_all_later_splits"
)
DEVELOPMENT_CLAIM_BOUNDARY = (
    "preliminary_development_signal_only_no_promotion_or_scientific_claim"
)
DEVELOPMENT_UNIT_ORDER = (
    "module_dependency_order",
    "source_cluster_id",
    "content_branch_id",
    "geometry_grid_index",
    "key_control_identity",
    "record_attempt_index",
)
CONTENT_BRANCH_IDS = (
    "hf_only",
    "lf_only",
    "lf_hf_uniform_control",
    "lf_hf_content_adaptive",
)
CONTENT_MIXING_COEFFICIENTS = (0.25, 0.50, 0.75)
CONTENT_COMBINATION_FUNCTION_IDS = (
    "hf_only_standardized_score",
    "weighted_hf_lf_standardized_score",
    "maximum_hf_lf_standardized_score",
)
MODULE_OUTCOMES = (
    "development_signal_observed",
    "development_signal_not_observed",
    "development_execution_inconclusive",
    "development_resource_inconclusive",
    "development_dependency_blocked",
)
RECOMMENDATION_BY_MODULE_OUTCOME = {
    "development_signal_observed": "proceed_to_candidate_selection",
    "development_signal_not_observed": (
        "record_closed_negative_and_stop_dependent_modules"
    ),
    "development_execution_inconclusive": (
        "repair_execution_before_budget_bounded_repeat"
    ),
    "development_resource_inconclusive": (
        "revise_resource_plan_before_budget_bounded_repeat"
    ),
    "development_dependency_blocked": (
        "do_not_execute_until_prerequisite_is_reopened"
    ),
}

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXACT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "protocol_version",
        "split_policy",
        "study_budget",
        "provisional_threshold_cross_fit",
        "content_study",
        "geometry_grid",
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


def _digest_sequence(values: Sequence[str]) -> str:
    return _canonical_digest(tuple(values))


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
class DevelopmentStudyBudget:
    wiring_source_cluster_count: int
    scientific_source_cluster_scales: tuple[int, ...]
    maximum_module_cluster_assignments: int
    maximum_record_attempts_per_unit: int
    maximum_total_record_attempts: int
    unit_order: tuple[str, ...]

    def validate(self, matrix: Sequence[DevelopmentModuleStudy]) -> tuple[str, ...]:
        violations: list[str] = []
        if self.wiring_source_cluster_count != WIRING_SOURCE_CLUSTER_COUNT:
            violations.append("wiring_source_cluster_count_invalid")
        if self.scientific_source_cluster_scales != SCIENTIFIC_SOURCE_CLUSTER_SCALES:
            violations.append("scientific_source_cluster_scales_invalid")
        if self.maximum_record_attempts_per_unit != MAXIMUM_RECORD_ATTEMPTS:
            violations.append("maximum_record_attempts_per_unit_invalid")
        expected_assignments = sum(item.scientific_source_cluster_scale for item in matrix)
        if self.maximum_module_cluster_assignments != expected_assignments:
            violations.append("maximum_module_cluster_assignments_invalid")
        if (
            self.maximum_total_record_attempts
            != expected_assignments * self.maximum_record_attempts_per_unit
        ):
            violations.append("maximum_total_record_attempts_invalid")
        if self.unit_order != DEVELOPMENT_UNIT_ORDER:
            violations.append("development_unit_order_invalid")
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class DevelopmentThresholdCrossFitPolicy:
    source_split: str
    fold_count: int
    fit_role: str
    score_role: str
    threshold_role: str
    invalidation_semantics: str
    invalid_for_splits: tuple[str, ...]

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.source_split != DEVELOPMENT_SPLIT:
            violations.append("cross_fit_source_split_invalid")
        if self.fold_count != DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT:
            violations.append("cross_fit_fold_count_invalid")
        if self.fit_role != DEVELOPMENT_THRESHOLD_FIT_ROLE:
            violations.append("cross_fit_fit_role_invalid")
        if self.score_role != DEVELOPMENT_THRESHOLD_SCORE_ROLE:
            violations.append("cross_fit_score_role_invalid")
        if self.threshold_role != DEVELOPMENT_THRESHOLD_ROLE:
            violations.append("cross_fit_threshold_role_invalid")
        if self.invalidation_semantics != DEVELOPMENT_THRESHOLD_INVALIDATION:
            violations.append("cross_fit_invalidation_semantics_invalid")
        if self.invalid_for_splits != FORMAL_LATER_SPLIT_DENY_LIST:
            violations.append("cross_fit_invalid_split_set_invalid")
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
class GeometryGridPoint:
    geometry_grid_index: int
    crop_fraction: float
    scale_factor: float
    rotation_degrees: float
    attack_id: str


@dataclass(frozen=True, slots=True)
class DevelopmentGeometryGrid:
    crop_fractions: tuple[float, ...]
    scale_factors: tuple[float, ...]
    rotation_degrees: tuple[float, ...]
    grid_points: tuple[GeometryGridPoint, ...]

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.crop_fractions != (1.0, 0.75, 0.45):
            violations.append("geometry_crop_grid_invalid")
        if self.scale_factors != (
            1.0,
            0.7071067811865476,
            1.4142135623730951,
        ):
            violations.append("geometry_scale_grid_invalid")
        if self.rotation_degrees != (0.0, -16.0, 16.0, -32.0, 32.0):
            violations.append("geometry_rotation_grid_invalid")
        expected = _expand_geometry_grid(
            self.crop_fractions,
            self.scale_factors,
            self.rotation_degrees,
        )
        if self.grid_points != expected:
            violations.append("geometry_grid_expansion_invalid")
        if len(self.grid_points) != 45:
            violations.append("geometry_grid_size_invalid")
        return tuple(violations)


def _attack_identity(crop: float, scale: float, rotation: float) -> str:
    changed = (crop != 1.0, scale != 1.0, rotation != 0.0)
    if not any(changed):
        return "identity"
    if changed == (True, False, False):
        return "crop"
    if changed == (False, True, False):
        return "scale"
    if changed == (False, False, True):
        return "rotation"
    return "crop_scale_rotation"


def _expand_geometry_grid(
    crop_fractions: Sequence[float],
    scale_factors: Sequence[float],
    rotation_degrees: Sequence[float],
) -> tuple[GeometryGridPoint, ...]:
    return tuple(
        GeometryGridPoint(
            geometry_grid_index=index,
            crop_fraction=float(crop),
            scale_factor=float(scale),
            rotation_degrees=float(rotation),
            attack_id=_attack_identity(float(crop), float(scale), float(rotation)),
        )
        for index, (crop, scale, rotation) in enumerate(
            product(crop_fractions, scale_factors, rotation_degrees)
        )
    )


@dataclass(frozen=True, slots=True)
class DevelopmentModuleStudy:
    responsibility: str
    scientific_question: str
    development_case_id: str
    candidate_selection_case_id: str
    prerequisite_responsibilities: tuple[str, ...]
    scientific_source_cluster_scale: int
    content_branch_ids: tuple[str, ...]
    geometry_grid_required: bool
    negative_controls: tuple[str, ...]
    record_field_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentExplorationProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
    split_policy: DevelopmentSplitPolicy
    study_budget: DevelopmentStudyBudget
    provisional_threshold_cross_fit: DevelopmentThresholdCrossFitPolicy
    content_study: DevelopmentContentStudy
    geometry_grid: DevelopmentGeometryGrid
    module_matrix: tuple[DevelopmentModuleStudy, ...]
    module_outcomes: tuple[str, ...]
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
        violations.extend(self.provisional_threshold_cross_fit.validate())
        violations.extend(self.content_study.validate())
        violations.extend(self.geometry_grid.validate())
        violations.extend(validate_development_module_matrix(self.module_matrix))
        violations.extend(self.study_budget.validate(self.module_matrix))
        if self.module_outcomes != MODULE_OUTCOMES:
            violations.append("development_module_outcomes_invalid")
        if self.scientific_claim_boundary != DEVELOPMENT_CLAIM_BOUNDARY:
            violations.append("development_scientific_claim_boundary_invalid")
        return tuple(dict.fromkeys(violations))


def validate_development_module_matrix(
    matrix: Sequence[DevelopmentModuleStudy],
) -> tuple[str, ...]:
    violations: list[str] = []
    responsibilities = tuple(item.responsibility for item in matrix)
    if responsibilities != REQUIRED_METHOD_RESPONSIBILITIES:
        violations.append("development_responsibility_order_or_identity_mismatch")
    development_case_ids = tuple(item.development_case_id for item in matrix)
    candidate_case_ids = tuple(item.candidate_selection_case_id for item in matrix)
    if any(not value for value in development_case_ids):
        violations.append("development_case_id_missing")
    if len(set(development_case_ids)) != len(development_case_ids):
        violations.append("development_case_id_duplicate")
    if any(not value for value in candidate_case_ids):
        violations.append("candidate_selection_case_id_missing")
    if len(set(candidate_case_ids)) != len(candidate_case_ids):
        violations.append("candidate_selection_case_id_duplicate")
    seen: set[str] = set()
    content_responsibilities = set(REQUIRED_METHOD_RESPONSIBILITIES[1:8])
    geometry_responsibilities = set(REQUIRED_METHOD_RESPONSIBILITIES[8:12])
    for item in matrix:
        if not item.scientific_question:
            violations.append(f"{item.responsibility}:scientific_question_missing")
        if item.scientific_source_cluster_scale not in SCIENTIFIC_SOURCE_CLUSTER_SCALES:
            violations.append(f"{item.responsibility}:scientific_scale_invalid")
        if any(dependency not in seen for dependency in item.prerequisite_responsibilities):
            violations.append(f"{item.responsibility}:dependency_order_invalid")
        if set(item.content_branch_ids) - set(CONTENT_BRANCH_IDS):
            violations.append(f"{item.responsibility}:content_branch_invalid")
        if item.responsibility in content_responsibilities and not item.content_branch_ids:
            violations.append(f"{item.responsibility}:content_branch_missing")
        if item.responsibility not in content_responsibilities and item.content_branch_ids:
            if item.responsibility != "conditional_recovery_decision":
                violations.append(f"{item.responsibility}:content_branch_forbidden")
        if item.responsibility in geometry_responsibilities and not item.geometry_grid_required:
            violations.append(f"{item.responsibility}:geometry_grid_required")
        if (
            item.responsibility not in geometry_responsibilities
            and item.responsibility != "conditional_recovery_decision"
            and item.geometry_grid_required
        ):
            violations.append(f"{item.responsibility}:geometry_grid_forbidden")
        if not item.negative_controls:
            violations.append(f"{item.responsibility}:negative_controls_missing")
        if not item.record_field_groups:
            violations.append(f"{item.responsibility}:record_field_groups_missing")
        if set(item.record_field_groups) - REQUIRED_RECORD_FIELD_GROUPS:
            violations.append(f"{item.responsibility}:record_field_group_invalid")
        seen.add(item.responsibility)
    return tuple(dict.fromkeys(violations))


def load_frozen_development_exploration_protocol(
    path: str | Path,
) -> FrozenDevelopmentExplorationProtocol:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if type(raw) is not dict:
        raise ValueError("development_protocol_root_must_be_mapping")
    _require_exact_keys(raw, _EXACT_TOP_LEVEL_KEYS, "development_protocol")

    split_raw = raw["split_policy"]
    budget_raw = raw["study_budget"]
    threshold_raw = raw["provisional_threshold_cross_fit"]
    content_raw = raw["content_study"]
    geometry_raw = raw["geometry_grid"]
    outcome_raw = raw["module_outcomes"]
    for role, value in (
        ("split_policy", split_raw),
        ("study_budget", budget_raw),
        ("provisional_threshold_cross_fit", threshold_raw),
        ("content_study", content_raw),
        ("geometry_grid", geometry_raw),
        ("module_outcomes", outcome_raw),
    ):
        if type(value) is not dict:
            raise ValueError(f"{role}_must_be_mapping")

    _require_exact_keys(
        split_raw,
        frozenset(
            {"allowed_split", "formal_later_split_deny_list", "candidate_selection_mapping"}
        ),
        "split_policy",
    )
    split_policy = DevelopmentSplitPolicy(
        allowed_split=split_raw["allowed_split"],
        formal_later_split_deny_list=tuple(split_raw["formal_later_split_deny_list"]),
        candidate_selection_mapping=split_raw["candidate_selection_mapping"],
    )

    _require_exact_keys(
        budget_raw,
        frozenset(
            {
                "wiring_source_cluster_count",
                "scientific_source_cluster_scales",
                "maximum_module_cluster_assignments",
                "maximum_record_attempts_per_unit",
                "maximum_total_record_attempts",
                "unit_order",
            }
        ),
        "study_budget",
    )
    study_budget = DevelopmentStudyBudget(
        wiring_source_cluster_count=budget_raw["wiring_source_cluster_count"],
        scientific_source_cluster_scales=tuple(budget_raw["scientific_source_cluster_scales"]),
        maximum_module_cluster_assignments=budget_raw["maximum_module_cluster_assignments"],
        maximum_record_attempts_per_unit=budget_raw["maximum_record_attempts_per_unit"],
        maximum_total_record_attempts=budget_raw["maximum_total_record_attempts"],
        unit_order=tuple(budget_raw["unit_order"]),
    )

    _require_exact_keys(
        threshold_raw,
        frozenset(
            {
                "source_split",
                "fold_count",
                "fit_role",
                "score_role",
                "threshold_role",
                "invalidation_semantics",
                "invalid_for_splits",
            }
        ),
        "provisional_threshold_cross_fit",
    )
    threshold_policy = DevelopmentThresholdCrossFitPolicy(
        source_split=threshold_raw["source_split"],
        fold_count=threshold_raw["fold_count"],
        fit_role=threshold_raw["fit_role"],
        score_role=threshold_raw["score_role"],
        threshold_role=threshold_raw["threshold_role"],
        invalidation_semantics=threshold_raw["invalidation_semantics"],
        invalid_for_splits=tuple(threshold_raw["invalid_for_splits"]),
    )

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
        branch_ids=tuple(content_raw["branch_ids"]),
        mixing_coefficients=tuple(content_raw["mixing_coefficients"]),
        combination_function_ids=tuple(content_raw["combination_function_ids"]),
        matched_total_budget_required=content_raw["matched_total_budget_required"],
        attack_condition_switching_forbidden=content_raw[
            "attack_condition_switching_forbidden"
        ],
    )

    _require_exact_keys(
        geometry_raw,
        frozenset({"crop_fractions", "scale_factors", "rotation_degrees"}),
        "geometry_grid",
    )
    crop_fractions = tuple(geometry_raw["crop_fractions"])
    scale_factors = tuple(geometry_raw["scale_factors"])
    rotation_degrees = tuple(geometry_raw["rotation_degrees"])
    geometry_grid = DevelopmentGeometryGrid(
        crop_fractions=crop_fractions,
        scale_factors=scale_factors,
        rotation_degrees=rotation_degrees,
        grid_points=_expand_geometry_grid(
            crop_fractions,
            scale_factors,
            rotation_degrees,
        ),
    )

    matrix: list[DevelopmentModuleStudy] = []
    for index, item in enumerate(raw["module_matrix"]):
        if type(item) is not dict:
            raise ValueError(f"module_matrix_entry_must_be_mapping:{index}")
        _require_exact_keys(
            item,
            frozenset(
                {
                    "responsibility",
                    "scientific_question",
                    "development_case_id",
                    "candidate_selection_case_id",
                    "prerequisite_responsibilities",
                    "scientific_source_cluster_scale",
                    "content_branch_ids",
                    "geometry_grid_required",
                    "negative_controls",
                    "record_field_groups",
                }
            ),
            f"module_matrix_entry:{index}",
        )
        matrix.append(
            DevelopmentModuleStudy(
                responsibility=item["responsibility"],
                scientific_question=item["scientific_question"],
                development_case_id=item["development_case_id"],
                candidate_selection_case_id=item["candidate_selection_case_id"],
                prerequisite_responsibilities=tuple(item["prerequisite_responsibilities"]),
                scientific_source_cluster_scale=item["scientific_source_cluster_scale"],
                content_branch_ids=tuple(item["content_branch_ids"]),
                geometry_grid_required=item["geometry_grid_required"],
                negative_controls=tuple(item["negative_controls"]),
                record_field_groups=tuple(item["record_field_groups"]),
            )
        )

    _require_exact_keys(
        outcome_raw,
        frozenset({"allowed", "recommendation_by_outcome"}),
        "module_outcomes",
    )
    if outcome_raw["recommendation_by_outcome"] != RECOMMENDATION_BY_MODULE_OUTCOME:
        raise ValueError("recommendation_by_module_outcome_invalid")
    protocol = FrozenDevelopmentExplorationProtocol(
        schema_version=raw["schema_version"],
        protocol_id=raw["protocol_id"],
        protocol_version=raw["protocol_version"],
        split_policy=split_policy,
        study_budget=study_budget,
        provisional_threshold_cross_fit=threshold_policy,
        content_study=content_study,
        geometry_grid=geometry_grid,
        module_matrix=tuple(matrix),
        module_outcomes=tuple(outcome_raw["allowed"]),
        scientific_claim_boundary=raw["scientific_claim_boundary"],
    )
    violations = protocol.validate()
    if violations:
        raise ValueError(",".join(violations))
    return protocol


def development_assignments_only(
    manifest: FrozenSplitManifest,
) -> tuple[SplitAssignment, ...]:
    """Return a development-only manifest or reject any later-split presence."""

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
    score_source_cluster_ids: tuple[str, ...]
    fit_source_cluster_digest: str
    score_source_cluster_digest: str

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.fold_index not in range(DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT):
            violations.append("cross_fit_fold_index_invalid")
        if not self.fit_source_cluster_ids or not self.score_source_cluster_ids:
            violations.append("cross_fit_fold_cluster_set_empty")
        if set(self.fit_source_cluster_ids) & set(self.score_source_cluster_ids):
            violations.append("cross_fit_fold_fit_score_leakage")
        if self.fit_source_cluster_digest != _digest_sequence(
            self.fit_source_cluster_ids
        ):
            violations.append("cross_fit_fit_cluster_digest_invalid")
        if self.score_source_cluster_digest != _digest_sequence(
            self.score_source_cluster_ids
        ):
            violations.append("cross_fit_score_cluster_digest_invalid")
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentCrossFitPlan:
    responsibility: str
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
        if self.responsibility not in REQUIRED_METHOD_RESPONSIBILITIES:
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
        scored = tuple(
            cluster
            for fold in self.folds
            for cluster in fold.score_source_cluster_ids
        )
        if len(scored) != self.source_cluster_count or len(set(scored)) != len(scored):
            violations.append("cross_fit_plan_score_partition_invalid")
        all_clusters = set(scored)
        for fold in self.folds:
            if set(fold.fit_source_cluster_ids) | set(fold.score_source_cluster_ids) != all_clusters:
                violations.append("cross_fit_plan_fold_coverage_invalid")
        if self.scientific_claims_supported is not False:
            violations.append("development_cross_fit_scientific_claim_forbidden")
        return tuple(dict.fromkeys(violations))


def build_development_cross_fit_plan(
    *,
    responsibility: str,
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
        score = tuple(
            cluster
            for index, cluster in enumerate(cluster_ids)
            if index % DEVELOPMENT_THRESHOLD_CROSS_FIT_FOLD_COUNT == fold_index
        )
        fit = tuple(cluster for cluster in cluster_ids if cluster not in set(score))
        folds.append(
            DevelopmentCrossFitFold(
                fold_index=fold_index,
                fit_source_cluster_ids=fit,
                score_source_cluster_ids=score,
                fit_source_cluster_digest=_digest_sequence(fit),
                score_source_cluster_digest=_digest_sequence(score),
            )
        )
    plan = FrozenDevelopmentCrossFitPlan(
        responsibility=responsibility,
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
class DevelopmentProvisionalThreshold:
    threshold_identity: str
    responsibility: str
    fold_index: int
    threshold: float
    fit_source_cluster_digest: str
    score_source_cluster_digest: str
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
        if type(plan) is not FrozenDevelopmentCrossFitPlan:
            return ("development_cross_fit_plan_exact_type_required",)
        if plan.validate():
            violations.append("development_cross_fit_plan_invalid")
        if self.responsibility != plan.responsibility:
            violations.append("provisional_threshold_responsibility_mismatch")
        if self.fold_index not in range(len(plan.folds)):
            violations.append("provisional_threshold_fold_index_invalid")
        else:
            fold = plan.folds[self.fold_index]
            if self.fit_source_cluster_digest != fold.fit_source_cluster_digest:
                violations.append("provisional_threshold_fit_digest_mismatch")
            if self.score_source_cluster_digest != fold.score_source_cluster_digest:
                violations.append("provisional_threshold_score_digest_mismatch")
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
) -> DevelopmentProvisionalThreshold:
    if type(plan) is not FrozenDevelopmentCrossFitPlan or plan.validate():
        raise ValueError("development_cross_fit_plan_invalid")
    if fold_index not in range(len(plan.folds)):
        raise ValueError("provisional_threshold_fold_index_invalid")
    fold = plan.folds[fold_index]
    payload = {
        "responsibility": plan.responsibility,
        "fold_index": fold_index,
        "threshold": threshold,
        "fit_source_cluster_digest": fold.fit_source_cluster_digest,
        "score_source_cluster_digest": fold.score_source_cluster_digest,
        "source_split": DEVELOPMENT_SPLIT,
        "threshold_role": DEVELOPMENT_THRESHOLD_ROLE,
        "invalid_for_splits": FORMAL_LATER_SPLIT_DENY_LIST,
        "scientific_claims_supported": False,
    }
    provisional = DevelopmentProvisionalThreshold(
        threshold_identity=_canonical_digest(payload),
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
    if source_cluster_id not in plan.folds[threshold.fold_index].score_source_cluster_ids:
        raise PermissionError("development_provisional_threshold_fold_leakage")


@dataclass(frozen=True, slots=True)
class DevelopmentModuleExecutionDecision:
    approved: bool
    responsibility: str
    missing_prerequisites: tuple[str, ...]
    blocking_prerequisites: tuple[str, ...]
    decision_reason: str


def decide_development_module_execution(
    protocol: FrozenDevelopmentExplorationProtocol,
    responsibility: str,
    outcomes_by_responsibility: Mapping[str, str],
) -> DevelopmentModuleExecutionDecision:
    if type(protocol) is not FrozenDevelopmentExplorationProtocol or protocol.validate():
        raise ValueError("development_protocol_invalid")
    studies = {item.responsibility: item for item in protocol.module_matrix}
    if responsibility not in studies:
        raise ValueError("development_responsibility_invalid")
    study = studies[responsibility]
    unknown_outcomes = set(outcomes_by_responsibility) - set(studies)
    if unknown_outcomes:
        raise ValueError("development_outcome_responsibility_unknown")
    if any(value not in MODULE_OUTCOMES for value in outcomes_by_responsibility.values()):
        raise ValueError("development_module_outcome_invalid")
    missing = tuple(
        dependency
        for dependency in study.prerequisite_responsibilities
        if dependency not in outcomes_by_responsibility
    )
    blocking = tuple(
        dependency
        for dependency in study.prerequisite_responsibilities
        if outcomes_by_responsibility.get(dependency) != "development_signal_observed"
        and dependency not in missing
    )
    if missing:
        return DevelopmentModuleExecutionDecision(
            False,
            responsibility,
            missing,
            (),
            "prerequisite_outcome_missing",
        )
    if blocking:
        return DevelopmentModuleExecutionDecision(
            False,
            responsibility,
            (),
            blocking,
            "dependency_stop_rule",
        )
    return DevelopmentModuleExecutionDecision(
        True,
        responsibility,
        (),
        (),
        "development_execution_authorized",
    )


@dataclass(frozen=True, slots=True)
class DevelopmentModuleOutcomeRecord:
    outcome_record_id: str
    responsibility: str
    module_outcome: str
    recommended_next_action: str
    recommendation_reason: str
    candidate_selection_case_id: str | None
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
        if type(protocol) is not FrozenDevelopmentExplorationProtocol:
            return ("development_protocol_exact_type_required",)
        if protocol.validate():
            violations.append("development_protocol_invalid")
        studies = {item.responsibility: item for item in protocol.module_matrix}
        if self.responsibility not in studies:
            violations.append("module_outcome_responsibility_invalid")
        if self.module_outcome not in MODULE_OUTCOMES:
            violations.append("module_outcome_invalid")
        expected_recommendation = RECOMMENDATION_BY_MODULE_OUTCOME.get(
            self.module_outcome
        )
        if self.recommended_next_action != expected_recommendation:
            violations.append("module_outcome_recommendation_mismatch")
        if not self.recommendation_reason.strip():
            violations.append("recommendation_reason_missing")
        expected_case = (
            studies[self.responsibility].candidate_selection_case_id
            if self.responsibility in studies
            and self.module_outcome == "development_signal_observed"
            else None
        )
        if self.candidate_selection_case_id != expected_case:
            violations.append("candidate_selection_recommendation_mapping_invalid")
        if self.module_outcome == "development_dependency_blocked":
            if not self.blocking_responsibilities:
                violations.append("dependency_blocking_responsibility_missing")
            elif self.responsibility in studies and not set(
                self.blocking_responsibilities
            ).issubset(set(studies[self.responsibility].prerequisite_responsibilities)):
                violations.append("dependency_blocking_responsibility_invalid")
        elif self.blocking_responsibilities:
            violations.append("blocking_responsibility_forbidden_for_outcome")
        if not self.evidence_record_ids:
            violations.append("module_outcome_evidence_record_ids_missing")
        if any(not value for value in self.evidence_record_ids):
            violations.append("module_outcome_evidence_record_id_invalid")
        if len(set(self.evidence_record_ids)) != len(self.evidence_record_ids):
            violations.append("module_outcome_evidence_record_id_duplicate")
        if any(not _DIGEST_PATTERN.fullmatch(value) for value in self.provisional_threshold_identities):
            violations.append("module_outcome_provisional_threshold_identity_invalid")
        if self.source_record_schema_version != INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION:
            violations.append("source_record_schema_version_invalid")
        if (
            self.source_record_collection_schema_version
            != INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
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
    responsibility: str,
    module_outcome: str,
    recommendation_reason: str,
    evidence_record_ids: Sequence[str],
    blocking_responsibilities: Sequence[str] = (),
    provisional_threshold_identities: Sequence[str] = (),
) -> DevelopmentModuleOutcomeRecord:
    studies = {item.responsibility: item for item in protocol.module_matrix}
    if responsibility not in studies or module_outcome not in MODULE_OUTCOMES:
        raise ValueError("development_module_outcome_input_invalid")
    payload = {
        "responsibility": responsibility,
        "module_outcome": module_outcome,
        "recommended_next_action": RECOMMENDATION_BY_MODULE_OUTCOME[
            module_outcome
        ],
        "recommendation_reason": recommendation_reason,
        "candidate_selection_case_id": (
            studies[responsibility].candidate_selection_case_id
            if module_outcome == "development_signal_observed"
            else None
        ),
        "blocking_responsibilities": tuple(blocking_responsibilities),
        "evidence_record_ids": tuple(evidence_record_ids),
        "provisional_threshold_identities": tuple(
            provisional_threshold_identities
        ),
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
