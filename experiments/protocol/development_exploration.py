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

from experiments.protocol.internal_matrix import (
    REQUIRED_METHOD_RESPONSIBILITIES,
    RESPONSIBILITY_VALIDATION_MATRIX,
)
from experiments.protocol.internal_records import (
    INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION,
    INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
    MAXIMUM_RECORD_ATTEMPTS,
    InternalValidationRecord,
    validate_internal_record,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
    INTERNAL_VALIDATION_SPLITS,
    SplitAssignment,
)


PROTOCOL_ID = "ceg_wm_development_module_exploration"
DEVELOPMENT_EXPLORATION_PROTOCOL_VERSION = "4.0.0"
SCHEMA_VERSION = "ceg_wm_development_module_exploration_protocol_schema_v4"
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
DEVELOPMENT_THRESHOLD_FIT_ROLE = "development_primary_null_fit"
DEVELOPMENT_THRESHOLD_SCORE_ROLE = "development_recovery_probe_score"
DEVELOPMENT_THRESHOLD_INPUT_ROLES = (
    "primary_null",
)
DEVELOPMENT_PRIMARY_NULL_CASE_ID = "development_primary_null_threshold_fit"
DEVELOPMENT_WRONG_KEY_CONTROL_CASE_ID = (
    "development_wrong_key_threshold_control"
)
DEVELOPMENT_THRESHOLD_RULE_PAYLOAD = {
    "order_statistic": "maximum",
    "rule_id": "development_primary_null_maximum_score",
}
DEVELOPMENT_THRESHOLD_INVALIDATION = (
    "invalidate_before_candidate_selection_and_all_later_splits"
)
DEVELOPMENT_CLAIM_BOUNDARY = (
    "preliminary_development_signal_only_no_promotion_or_scientific_claim"
)
DEVELOPMENT_THRESHOLD_AUTHORITY_ID = (
    "development_high_frequency_detector_threshold_authority"
)
DEVELOPMENT_THRESHOLD_RESPONSIBILITY_ID = "hf_detector"
DEVELOPMENT_THRESHOLD_DETECTOR_IDENTITY = (
    "development_blind_high_frequency_detector"
)
DEVELOPMENT_THRESHOLD_DETECTOR_MODE = "hf_only"
DEVELOPMENT_THRESHOLD_PREPROCESSING_IDENTITY = (
    "rgb8_public_image_float32_unit_interval"
)
DEVELOPMENT_THRESHOLD_PUBLIC_KEY_RELATION = (
    "registered_detection_public_digests_distinct"
)
REGISTERED_KEY_SCHEDULE_DERIVATION_IDENTITY = (
    "main_shared_key_schedule_identify_root_key"
)
REGISTERED_KEY_SCHEDULE_CONFIG_DIGEST = (
    "8696a3fbaabb39149a3b7b30f08cdcd12b64d45bebd14658d08c09805f5b33c0"
)
DEVELOPMENT_EXECUTION_INTENT_ROLE = "create_only_before_scientific_records"
DEVELOPMENT_EXECUTION_INTENT_RAW_SECRET_POLICY = "raw_secret_prohibited"
RECORD_SCHEMA_VERSION = (
    "ceg_wm_development_scientific_record_v1"
)
RECORD_COLLECTION_SCHEMA_VERSION = (
    "ceg_wm_development_scientific_record_collection_v1"
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
NOT_APPLICABLE_CONTENT_BRANCH_ID = "content_branch_not_applicable"
NOT_APPLICABLE_GEOMETRY_CASE_ID = "geometry_case_not_applicable"

REGISTERED_CANDIDATE_IDS = frozenset(
    {
        "key_schedule_sha256_counter",
        "runtime_sd35_flowmatch",
        "hf_sparse_tail",
        "lf_low_pass",
        "routing_stqr",
        "routing_uniform_control",
        "content_combination_calibrated",
        "qk_relation_similarity",
        "rectification_similarity",
        "joint_conditional_recovery",
    }
)
CONTENT_RELATIVE_L2 = "3/250"
GEOMETRY_CONTENT_RATIO_CANDIDATES = ("1/16", "1/8", "1/4")
COMBINATION_WEIGHT_IDENTITIES = ("1/4", "1/2", "3/4")
MODULE_CANDIDATE_IDS = {
    "key_schedule": ("key_schedule_sha256_counter",),
    "content_router": (
        "key_schedule_sha256_counter",
        "routing_stqr",
        "routing_uniform_control",
    ),
    "lf_carrier": ("key_schedule_sha256_counter", "lf_low_pass"),
    "hf_carrier": (
        "key_schedule_sha256_counter",
        "runtime_sd35_flowmatch",
        "hf_sparse_tail",
    ),
    "content_embedder": (
        "runtime_sd35_flowmatch",
        "hf_sparse_tail",
        "lf_low_pass",
        "routing_stqr",
        "routing_uniform_control",
    ),
    "lf_detector": ("key_schedule_sha256_counter", "lf_low_pass"),
    "hf_detector": ("key_schedule_sha256_counter", "hf_sparse_tail"),
    "content_detector": (
        "hf_sparse_tail",
        "lf_low_pass",
        "content_combination_calibrated",
    ),
    "qk_geometry_sync": (
        "key_schedule_sha256_counter",
        "runtime_sd35_flowmatch",
        "qk_relation_similarity",
    ),
    "geometric_transform_estimator": (
        "key_schedule_sha256_counter",
        "qk_relation_similarity",
        "rectification_similarity",
    ),
    "geometry_reliability": (
        "key_schedule_sha256_counter",
        "qk_relation_similarity",
        "rectification_similarity",
    ),
    "image_rectifier": ("rectification_similarity",),
    "conditional_recovery_decision": ("joint_conditional_recovery",),
}
MODULE_CANDIDATE_PARAMETERS = {
    "key_schedule": (("key_stream_candidate", ("key_schedule_sha256_counter",)),),
    "content_router": (
        ("adaptive_router_candidate", ("routing_stqr",)),
        ("disabled_uniform_control_candidate", ("routing_uniform_control",)),
    ),
    "lf_carrier": (("carrier_candidate", ("lf_low_pass",)),),
    "hf_carrier": (("carrier_candidate", ("hf_sparse_tail",)),),
    "content_embedder": (
        ("high_frequency_candidate", ("hf_sparse_tail",)),
        ("low_frequency_candidate", ("lf_low_pass",)),
        ("adaptive_router_candidate", ("routing_stqr",)),
        ("disabled_uniform_control_candidate", ("routing_uniform_control",)),
        ("content_relative_l2", (CONTENT_RELATIVE_L2,)),
        ("mixing_coefficients", COMBINATION_WEIGHT_IDENTITIES),
    ),
    "lf_detector": (("detector_candidate", ("lf_low_pass",)),),
    "hf_detector": (("detector_candidate", ("hf_sparse_tail",)),),
    "content_detector": (
        ("combination_candidate", ("content_combination_calibrated",)),
        ("combination_functions", CONTENT_COMBINATION_FUNCTION_IDS),
        ("mixing_coefficients", COMBINATION_WEIGHT_IDENTITIES),
    ),
    "qk_geometry_sync": (
        ("relation_candidate", ("qk_relation_similarity",)),
        ("geometry_content_ratio_candidates", GEOMETRY_CONTENT_RATIO_CANDIDATES),
    ),
    "geometric_transform_estimator": (
        ("relation_candidate", ("qk_relation_similarity",)),
        ("estimator_candidate", ("rectification_similarity",)),
        ("geometry_content_ratio_candidates", GEOMETRY_CONTENT_RATIO_CANDIDATES),
    ),
    "geometry_reliability": (
        ("relation_candidate", ("qk_relation_similarity",)),
        ("reliability_candidate", ("rectification_similarity",)),
        ("geometry_content_ratio_candidates", GEOMETRY_CONTENT_RATIO_CANDIDATES),
    ),
    "image_rectifier": (
        ("rectifier_candidate", ("rectification_similarity",)),
        ("geometry_content_ratio_candidates", GEOMETRY_CONTENT_RATIO_CANDIDATES),
    ),
    "conditional_recovery_decision": (
        ("joint_candidate", ("joint_conditional_recovery",)),
        ("geometry_content_ratio_candidates", GEOMETRY_CONTENT_RATIO_CANDIDATES),
    ),
}
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
MODULE_CANDIDATE_IDENTITIES = {
    "key_schedule": "registered_key_identity_candidate",
    "content_router": "content_adaptive_router_candidate",
    "lf_carrier": "low_frequency_carrier_candidate",
    "hf_carrier": "high_frequency_carrier_candidate",
    "content_embedder": "matched_budget_content_embedder_candidate",
    "lf_detector": "blind_low_frequency_detector_candidate",
    "hf_detector": "blind_high_frequency_detector_candidate",
    "content_detector": "standardized_content_combination_candidate",
    "qk_geometry_sync": "keyed_query_key_sync_candidate",
    "geometric_transform_estimator": "blind_transform_estimator_candidate",
    "geometry_reliability": "fail_closed_geometry_reliability_candidate",
    "image_rectifier": "coordinate_preserving_rectifier_candidate",
    "conditional_recovery_decision": "same_detector_conditional_recovery_candidate",
}
MODULE_PAIRED_ABLATION_IDENTITIES = {
    "key_schedule": "wrong_and_public_key_identity_ablation",
    "content_router": "disabled_uniform_routing_ablation",
    "lf_carrier": "low_frequency_write_disabled_ablation",
    "hf_carrier": "high_frequency_write_disabled_ablation",
    "content_embedder": "single_branch_embedding_ablation",
    "lf_detector": "low_frequency_detector_disabled_ablation",
    "hf_detector": "high_frequency_detector_disabled_ablation",
    "content_detector": "high_frequency_only_detector_ablation",
    "qk_geometry_sync": "wrong_geometry_key_sync_ablation",
    "geometric_transform_estimator": "oracle_transform_input_ablation",
    "geometry_reliability": "geometry_reliability_disabled_ablation",
    "image_rectifier": "rectification_disabled_ablation",
    "conditional_recovery_decision": "raw_content_decision_ablation",
}
MODULE_NEGATIVE_CONTROL_CASE_IDS = {
    item.responsibility: item.negative_controls
    for item in RESPONSIBILITY_VALIDATION_MATRIX
}
MODULE_METRIC_IDS = {
    item.responsibility: item.metrics for item in RESPONSIBILITY_VALIDATION_MATRIX
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
        "threshold_detector_authority",
        "execution_intent_policy",
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
    manifest_reference: str
    manifest_availability: str
    frozen_manifest_digest: str | None

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
        if self.role_id == "development_exploration":
            if self.manifest_availability != "required_at_execution":
                violations.append("development_manifest_availability_invalid")
            if self.manifest_reference != "development_split_manifest_runtime_binding":
                violations.append("development_manifest_reference_invalid")
        else:
            if self.manifest_availability != "unavailable_until_role_authorized":
                violations.append("later_manifest_availability_invalid")
            if self.manifest_reference != (
                f"{self.role_id}_manifest_unavailable_until_authorized"
            ):
                violations.append("later_manifest_reference_invalid")
        if self.frozen_manifest_digest is not None:
            violations.append("unprovided_manifest_digest_must_be_null")
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class DevelopmentSplitIsolation:
    isolation_dimensions: tuple[str, ...]
    role_bindings: tuple[RegisteredStudyRole, ...]
    cross_role_identity_overlap_forbidden: bool
    formal_later_deny_policy_digest: str

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
        if self.cross_role_identity_overlap_forbidden is not True:
            violations.append("cross_role_identity_overlap_must_be_forbidden")
        later_policy = tuple(
            (
                item.role_id,
                item.registered_split,
                item.manifest_reference,
                item.manifest_availability,
            )
            for item in self.role_bindings
            if not item.execution_allowed_in_development
        )
        if self.formal_later_deny_policy_digest != _canonical_digest(later_policy):
            violations.append("formal_later_deny_policy_digest_invalid")
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class FrozenStudyRoleManifestBinding:
    role_id: str
    registered_split: str
    seed_namespace: str
    manifest: FrozenSplitManifest
    manifest_digest: str
    prompt_digests: tuple[str, ...]
    source_cluster_ids: tuple[str, ...]
    registered_key_family_digests: tuple[str, ...]
    image_lineage_digests: tuple[str, ...]

    def dimension_values(self) -> dict[str, frozenset[str]]:
        return {
            "prompt_digest": frozenset(self.prompt_digests),
            "source_cluster_id": frozenset(self.source_cluster_ids),
            "seed_namespace": frozenset((self.seed_namespace,)),
            "registered_key_family_digest": frozenset(
                self.registered_key_family_digests
            ),
            "image_lineage_digest": frozenset(self.image_lineage_digests),
        }

    def validate(
        self,
        protocol: FrozenDevelopmentExplorationProtocol,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if type(protocol) is not FrozenDevelopmentExplorationProtocol:
            return ("development_protocol_exact_type_required",)
        role_by_id = {
            item.role_id: item for item in protocol.split_isolation.role_bindings
        }
        if self.role_id not in role_by_id:
            return ("study_manifest_role_unregistered",)
        role = role_by_id[self.role_id]
        if self.registered_split != role.registered_split:
            violations.append("study_manifest_registered_split_mismatch")
        if _IDENTITY_PATTERN.fullmatch(self.seed_namespace) is None:
            violations.append("study_manifest_seed_namespace_invalid")
        if type(self.manifest) is not FrozenSplitManifest:
            return (*violations, "study_manifest_exact_type_required")
        violations.extend(self.manifest.validate(require_all_splits=False))
        if not self.manifest.assignments:
            violations.append("study_manifest_assignments_missing")
        if any(
            assignment.split != self.registered_split
            for assignment in self.manifest.assignments
        ):
            violations.append("study_manifest_contains_wrong_split")
        identities = tuple(assignment.identity for assignment in self.manifest.assignments)
        expected_values = {
            "prompt_digests": tuple(sorted({item.prompt_digest for item in identities})),
            "source_cluster_ids": tuple(
                sorted({item.source_cluster_id for item in identities})
            ),
            "registered_key_family_digests": tuple(
                sorted({item.registered_key_family_digest for item in identities})
            ),
            "image_lineage_digests": tuple(
                sorted({item.image_lineage_digest for item in identities})
            ),
        }
        for field_name, expected in expected_values.items():
            if getattr(self, field_name) != expected:
                violations.append(f"study_manifest_{field_name}_binding_invalid")
        if self.manifest_digest != self.manifest.digest():
            violations.append("study_manifest_digest_binding_invalid")
        return tuple(dict.fromkeys(violations))


def bind_study_role_manifest(
    protocol: FrozenDevelopmentExplorationProtocol,
    *,
    role_id: str,
    seed_namespace: str,
    manifest: FrozenSplitManifest,
) -> FrozenStudyRoleManifestBinding:
    if type(protocol) is not FrozenDevelopmentExplorationProtocol or protocol.validate():
        raise ValueError("development_protocol_invalid")
    role_by_id = {
        item.role_id: item for item in protocol.split_isolation.role_bindings
    }
    if role_id not in role_by_id:
        raise ValueError("study_manifest_role_unregistered")
    if type(manifest) is not FrozenSplitManifest:
        raise TypeError("study_manifest_exact_type_required")
    identities = tuple(assignment.identity for assignment in manifest.assignments)
    binding = FrozenStudyRoleManifestBinding(
        role_id=role_id,
        registered_split=role_by_id[role_id].registered_split,
        seed_namespace=seed_namespace,
        manifest=manifest,
        manifest_digest=manifest.digest(),
        prompt_digests=tuple(sorted({item.prompt_digest for item in identities})),
        source_cluster_ids=tuple(sorted({item.source_cluster_id for item in identities})),
        registered_key_family_digests=tuple(
            sorted({item.registered_key_family_digest for item in identities})
        ),
        image_lineage_digests=tuple(
            sorted({item.image_lineage_digest for item in identities})
        ),
    )
    violations = binding.validate(protocol)
    if violations:
        raise ValueError(",".join(violations))
    return binding


def assert_study_role_manifests_isolated(
    protocol: FrozenDevelopmentExplorationProtocol,
    bindings: Sequence[FrozenStudyRoleManifestBinding],
) -> None:
    if type(protocol) is not FrozenDevelopmentExplorationProtocol or protocol.validate():
        raise ValueError("development_protocol_invalid")
    if not bindings:
        raise ValueError("study_manifest_bindings_missing")
    seen_roles: set[str] = set()
    validated: list[FrozenStudyRoleManifestBinding] = []
    for binding in bindings:
        if type(binding) is not FrozenStudyRoleManifestBinding:
            raise TypeError("study_manifest_binding_exact_type_required")
        violations = binding.validate(protocol)
        if violations:
            raise ValueError(",".join(violations))
        if binding.role_id in seen_roles:
            raise ValueError("study_manifest_role_duplicate")
        seen_roles.add(binding.role_id)
        validated.append(binding)
    for left_index, left in enumerate(validated):
        for right in validated[left_index + 1 :]:
            left_values = left.dimension_values()
            right_values = right.dimension_values()
            for dimension in ISOLATION_DIMENSIONS:
                if left_values[dimension] & right_values[dimension]:
                    raise PermissionError(
                        "study_manifest_identity_overlap:"
                        f"{dimension}:{left.role_id}:{right.role_id}"
                    )


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
    candidate_ids: tuple[str, ...]
    candidate_parameter_bindings: tuple[tuple[str, tuple[str, ...]], ...]
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
            "candidate_ids": self.candidate_ids,
            "candidate_parameter_bindings": self.candidate_parameter_bindings,
            "content_branch_ids": self.content_branch_ids,
            "geometry_case_ids": self.geometry_case_ids,
            "metric_ids": self.metric_ids,
            "negative_control_case_ids": self.negative_control_case_ids,
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
        if item.candidate_identity != MODULE_CANDIDATE_IDENTITIES.get(prefix):
            violations.append(f"{prefix}:candidate_identity_unregistered")
        if item.candidate_ids != MODULE_CANDIDATE_IDS.get(prefix):
            violations.append(f"{prefix}:candidate_ids_unregistered")
        if set(item.candidate_ids) - REGISTERED_CANDIDATE_IDS:
            violations.append(f"{prefix}:candidate_id_unknown")
        if item.candidate_parameter_bindings != MODULE_CANDIDATE_PARAMETERS.get(prefix):
            violations.append(f"{prefix}:candidate_parameters_unregistered")
        if item.paired_ablation_identity != MODULE_PAIRED_ABLATION_IDENTITIES.get(prefix):
            violations.append(f"{prefix}:paired_ablation_unregistered")
        if item.negative_control_case_ids != MODULE_NEGATIVE_CONTROL_CASE_IDS.get(prefix):
            violations.append(f"{prefix}:negative_controls_unregistered")
        if item.metric_ids != MODULE_METRIC_IDS.get(prefix):
            violations.append(f"{prefix}:metric_ids_unregistered")
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
class DevelopmentThresholdDetectorAuthority:
    authority_id: str
    responsibility_id: str
    detector_identity: str
    detector_mode: str
    preprocessing_identity: str
    raw_rectified_preprocessing_same: bool
    registered_candidate_identity: str
    registered_candidate_ids: tuple[str, ...]
    registered_candidate_parameter_bindings: tuple[
        tuple[str, tuple[str, ...]], ...
    ]
    registered_candidate_config_digest: str
    registered_key_schedule_candidate_id: str
    registered_key_schedule_derivation_identity: str
    registered_key_schedule_config_digest: str
    public_key_relation: str

    def digest(self) -> str:
        return _canonical_digest(asdict(self))

    def detector_base_config_payload(self) -> dict[str, object]:
        return {
            "detector_mode": self.detector_mode,
            "registered_candidate_config_digest": (
                self.registered_candidate_config_digest
            ),
            "registered_candidate_identity": self.registered_candidate_identity,
            "registered_candidate_ids": self.registered_candidate_ids,
            "registered_candidate_parameter_bindings": (
                self.registered_candidate_parameter_bindings
            ),
            "registered_key_schedule_candidate_id": (
                self.registered_key_schedule_candidate_id
            ),
            "registered_key_schedule_config_digest": (
                self.registered_key_schedule_config_digest
            ),
            "registered_key_schedule_derivation_identity": (
                self.registered_key_schedule_derivation_identity
            ),
        }

    def validate(
        self,
        module_matrix: Sequence[DevelopmentModuleStudy],
    ) -> tuple[str, ...]:
        violations: list[str] = []
        expected_scalars = (
            (
                self.authority_id,
                DEVELOPMENT_THRESHOLD_AUTHORITY_ID,
                "threshold_authority_id_invalid",
            ),
            (
                self.responsibility_id,
                DEVELOPMENT_THRESHOLD_RESPONSIBILITY_ID,
                "threshold_authority_responsibility_invalid",
            ),
            (
                self.detector_identity,
                DEVELOPMENT_THRESHOLD_DETECTOR_IDENTITY,
                "threshold_authority_detector_identity_invalid",
            ),
            (
                self.detector_mode,
                DEVELOPMENT_THRESHOLD_DETECTOR_MODE,
                "threshold_authority_detector_mode_invalid",
            ),
            (
                self.preprocessing_identity,
                DEVELOPMENT_THRESHOLD_PREPROCESSING_IDENTITY,
                "threshold_authority_preprocessing_identity_invalid",
            ),
            (
                self.registered_key_schedule_candidate_id,
                MODULE_CANDIDATE_IDS["key_schedule"][0],
                "threshold_authority_key_schedule_candidate_invalid",
            ),
            (
                self.registered_key_schedule_derivation_identity,
                REGISTERED_KEY_SCHEDULE_DERIVATION_IDENTITY,
                "threshold_authority_key_schedule_derivation_invalid",
            ),
            (
                self.registered_key_schedule_config_digest,
                REGISTERED_KEY_SCHEDULE_CONFIG_DIGEST,
                "threshold_authority_key_schedule_config_invalid",
            ),
            (
                self.public_key_relation,
                DEVELOPMENT_THRESHOLD_PUBLIC_KEY_RELATION,
                "threshold_authority_public_key_relation_invalid",
            ),
        )
        violations.extend(
            reason for observed, expected, reason in expected_scalars
            if observed != expected
        )
        if self.raw_rectified_preprocessing_same is not True:
            violations.append("threshold_authority_same_preprocessing_required")
        if self.registered_candidate_identity != MODULE_CANDIDATE_IDENTITIES[
            DEVELOPMENT_THRESHOLD_RESPONSIBILITY_ID
        ]:
            violations.append("threshold_authority_candidate_identity_unregistered")
        if self.registered_candidate_ids != MODULE_CANDIDATE_IDS[
            DEVELOPMENT_THRESHOLD_RESPONSIBILITY_ID
        ]:
            violations.append("threshold_authority_candidate_ids_unregistered")
        if self.registered_candidate_parameter_bindings != (
            MODULE_CANDIDATE_PARAMETERS[DEVELOPMENT_THRESHOLD_RESPONSIBILITY_ID]
        ):
            violations.append("threshold_authority_candidate_parameters_unregistered")
        registered_studies = {
            item.responsibility_id: item for item in module_matrix
        }
        study = registered_studies.get(DEVELOPMENT_THRESHOLD_RESPONSIBILITY_ID)
        if study is None:
            violations.append("threshold_authority_module_study_missing")
        else:
            if self.registered_candidate_identity != study.candidate_identity:
                violations.append("threshold_authority_candidate_identity_mismatch")
            if self.registered_candidate_ids != study.candidate_ids:
                violations.append("threshold_authority_candidate_ids_mismatch")
            if (
                self.registered_candidate_parameter_bindings
                != study.candidate_parameter_bindings
            ):
                violations.append("threshold_authority_candidate_parameters_mismatch")
            if self.registered_candidate_config_digest != study.candidate_config_digest:
                violations.append("threshold_authority_candidate_config_mismatch")
        if self.registered_key_schedule_candidate_id not in (
            self.registered_candidate_ids
        ):
            violations.append("threshold_authority_key_schedule_not_in_candidate_roster")
        return tuple(dict.fromkeys(violations))


def derive_development_primary_null_key_family_digest(
    authority: DevelopmentThresholdDetectorAuthority,
    *,
    registered_key_public_digest: str,
    detection_key_public_digest: str,
) -> str:
    if type(authority) is not DevelopmentThresholdDetectorAuthority:
        raise TypeError("threshold_detector_authority_exact_type_required")
    for field_name, value in (
        ("registered_key_public_digest", registered_key_public_digest),
        ("detection_key_public_digest", detection_key_public_digest),
    ):
        if _DIGEST_PATTERN.fullmatch(value) is None:
            raise ValueError(f"primary_null_{field_name}_invalid")
    if registered_key_public_digest == detection_key_public_digest:
        raise ValueError("primary_null_public_key_relation_mismatch")
    return _canonical_digest(
        {
            "authority_digest": authority.digest(),
            "detection_key_public_digest": detection_key_public_digest,
            "public_key_relation": authority.public_key_relation,
            "registered_key_public_digest": registered_key_public_digest,
            "registered_key_schedule_candidate_id": (
                authority.registered_key_schedule_candidate_id
            ),
            "registered_key_schedule_config_digest": (
                authority.registered_key_schedule_config_digest
            ),
            "registered_key_schedule_derivation_identity": (
                authority.registered_key_schedule_derivation_identity
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class DevelopmentStudyUnit:
    unit_index: int
    phase: str
    responsibility_id: str
    source_cluster_ordinal: int
    content_branch_id: str
    geometry_case_id: str
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
    atomic_descriptors: list[tuple[str, str, int, str, str]] = []
    grouped: dict[tuple[str, int], list[str]] = {}
    phase_by_group: dict[tuple[str, int], str] = {}
    for phase, responsibility_id, cluster_ordinal in ordered:
        group = (phase, cluster_ordinal)
        grouped.setdefault(group, []).append(responsibility_id)
        phase_by_group[group] = phase
    for group, responsibility_ids in grouped.items():
        phase = phase_by_group[group]
        cluster_ordinal = group[1]
        variants_by_responsibility: dict[str, tuple[tuple[str, str], ...]] = {}
        for responsibility_id in responsibility_ids:
            study = by_responsibility[responsibility_id]
            branches = study.content_branch_ids or (NOT_APPLICABLE_CONTENT_BRANCH_ID,)
            geometries = study.geometry_case_ids or (NOT_APPLICABLE_GEOMETRY_CASE_ID,)
            variants_by_responsibility[responsibility_id] = tuple(
                (branch_id, geometry_case_id)
                for branch_id in branches
                for geometry_case_id in geometries
            )
        # Each cluster first reaches every responsibility once; remaining frozen
        # branch x geometry variants follow without score-adaptive reordering.
        for responsibility_id in responsibility_ids:
            branch_id, geometry_case_id = variants_by_responsibility[responsibility_id][0]
            atomic_descriptors.append(
                (phase, responsibility_id, cluster_ordinal, branch_id, geometry_case_id)
            )
        for responsibility_id in responsibility_ids:
            for branch_id, geometry_case_id in variants_by_responsibility[
                responsibility_id
            ][1:]:
                atomic_descriptors.append(
                    (phase, responsibility_id, cluster_ordinal, branch_id, geometry_case_id)
                )
    return tuple(
        DevelopmentStudyUnit(
            unit_index=index,
            phase=phase,
            responsibility_id=responsibility_id,
            source_cluster_ordinal=cluster_ordinal,
            content_branch_id=content_branch_id,
            geometry_case_id=geometry_case_id,
            maximum_record_attempts=MAXIMUM_RECORD_ATTEMPTS,
            maximum_duration_seconds=MAXIMUM_UNIT_DURATION_SECONDS,
        )
        for index, (
            phase,
            responsibility_id,
            cluster_ordinal,
            content_branch_id,
            geometry_case_id,
        ) in enumerate(atomic_descriptors)
    )


@dataclass(frozen=True, slots=True)
class DevelopmentStudyBudget:
    preflight_source_cluster_count: int
    wiring_source_cluster_count: int
    wiring_counts_as_scientific_coverage: bool
    scientific_source_cluster_scales: tuple[int, ...]
    maximum_scientific_units: int
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
        if self.maximum_total_record_attempts != sum(
            unit.maximum_record_attempts for unit in roster
        ):
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
        observed_clusters = {
            responsibility_id: {
                unit.source_cluster_ordinal
                for unit in roster
                if unit.responsibility_id == responsibility_id
            }
            for responsibility_id in REQUIRED_METHOD_RESPONSIBILITIES
        }
        expected_cluster_counts = {
            item.responsibility_id: item.scientific_source_cluster_scale
            for item in matrix
        }
        if {
            responsibility_id: len(cluster_ids)
            for responsibility_id, cluster_ids in observed_clusters.items()
        } != expected_cluster_counts:
            violations.append("unit_roster_module_cluster_scale_mismatch")
        observed_atomic = {
            (
                unit.responsibility_id,
                unit.source_cluster_ordinal,
                unit.content_branch_id,
                unit.geometry_case_id,
            )
            for unit in roster
        }
        if len(observed_atomic) != len(roster):
            violations.append("unit_roster_atomic_identity_duplicate")
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
class DevelopmentExecutionIntentPolicy:
    authority_role: str
    create_only_required: bool
    freeze_before_scientific_records: bool
    raw_secret_policy: str
    expected_digest_required_at_result_boundaries: bool
    later_runner_must_pin_digest: bool

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self.authority_role != DEVELOPMENT_EXECUTION_INTENT_ROLE:
            violations.append("execution_intent_authority_role_invalid")
        if self.raw_secret_policy != DEVELOPMENT_EXECUTION_INTENT_RAW_SECRET_POLICY:
            violations.append("execution_intent_raw_secret_policy_invalid")
        for value, reason in (
            (self.create_only_required, "execution_intent_create_only_required"),
            (
                self.freeze_before_scientific_records,
                "execution_intent_must_precede_scientific_records",
            ),
            (
                self.expected_digest_required_at_result_boundaries,
                "execution_intent_expected_digest_required",
            ),
            (
                self.later_runner_must_pin_digest,
                "execution_intent_runner_pin_required",
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
    threshold_detector_authority: DevelopmentThresholdDetectorAuthority
    execution_intent_policy: DevelopmentExecutionIntentPolicy
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
        violations.extend(
            self.threshold_detector_authority.validate(self.module_matrix)
        )
        violations.extend(self.execution_intent_policy.validate())
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
                    "manifest_reference",
                    "manifest_availability",
                    "frozen_manifest_digest",
                }
            ),
            f"study_role_binding:{index}",
        )
        role_id = _require_identity(item["role_id"], "study_role_id")
        registered_split = _require_identity(
            item["registered_split"], "study_role_registered_split"
        )
        roles.append(
            RegisteredStudyRole(
                role_id=role_id,
                registered_split=registered_split,
                detector_mode=_require_identity(
                    item["detector_mode"], "study_role_detector_mode"
                ),
                requires_frozen_hf_only_tau=item["requires_frozen_hf_only_tau"],
                execution_allowed_in_development=item[
                    "execution_allowed_in_development"
                ],
                manifest_reference=_require_identity(
                    item["manifest_reference"], "study_role_manifest_reference"
                ),
                manifest_availability=_require_identity(
                    item["manifest_availability"], "study_role_manifest_availability"
                ),
                frozen_manifest_digest=(
                    None
                    if item["frozen_manifest_digest"] is None
                    else _require_digest(
                        item["frozen_manifest_digest"],
                        "study_role_frozen_manifest_digest",
                    )
                ),
            )
        )
    return tuple(roles)


def _load_candidate_parameter_bindings(
    raw_bindings: object,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    bindings: list[tuple[str, tuple[str, ...]]] = []
    for index, raw_binding in enumerate(
        _require_sequence(raw_bindings, "candidate_parameter_bindings")
    ):
        item = _require_mapping(
            raw_binding,
            f"candidate_parameter_binding:{index}",
        )
        _require_exact_keys(
            item,
            frozenset({"parameter_id", "values"}),
            f"candidate_parameter_binding:{index}",
        )
        values = tuple(
            _require_sequence(item["values"], "candidate_parameter_values")
        )
        if not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("candidate_parameter_values_invalid")
        bindings.append(
            (
                _require_identity(item["parameter_id"], "candidate_parameter_id"),
                values,
            )
        )
    return tuple(bindings)


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
        frozenset(
            {
                "isolation_dimensions",
                "role_bindings",
                "cross_role_identity_overlap_forbidden",
            }
        ),
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
        cross_role_identity_overlap_forbidden=isolation_raw[
            "cross_role_identity_overlap_forbidden"
        ],
        formal_later_deny_policy_digest=_canonical_digest(
            tuple(
                (
                    item.role_id,
                    item.registered_split,
                    item.manifest_reference,
                    item.manifest_availability,
                )
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
            "candidate_ids",
            "candidate_parameter_bindings",
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
                candidate_ids=tuple(
                    _require_sequence(item["candidate_ids"], "candidate_ids")
                ),
                candidate_parameter_bindings=_load_candidate_parameter_bindings(
                    item["candidate_parameter_bindings"]
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
    authority_raw = _require_mapping(
        raw["threshold_detector_authority"],
        "threshold_detector_authority",
    )
    authority_keys = frozenset(
        {
            "authority_id",
            "responsibility_id",
            "detector_identity",
            "detector_mode",
            "preprocessing_identity",
            "raw_rectified_preprocessing_same",
            "registered_candidate_identity",
            "registered_candidate_ids",
            "registered_candidate_parameter_bindings",
            "registered_candidate_config_digest",
            "registered_key_schedule_candidate_id",
            "registered_key_schedule_derivation_identity",
            "registered_key_schedule_config_digest",
            "public_key_relation",
        }
    )
    _require_exact_keys(
        authority_raw,
        authority_keys,
        "threshold_detector_authority",
    )
    threshold_detector_authority = DevelopmentThresholdDetectorAuthority(
        authority_id=_require_identity(
            authority_raw["authority_id"], "threshold_authority_id"
        ),
        responsibility_id=_require_identity(
            authority_raw["responsibility_id"],
            "threshold_authority_responsibility_id",
        ),
        detector_identity=_require_identity(
            authority_raw["detector_identity"],
            "threshold_authority_detector_identity",
        ),
        detector_mode=_require_identity(
            authority_raw["detector_mode"], "threshold_authority_detector_mode"
        ),
        preprocessing_identity=_require_identity(
            authority_raw["preprocessing_identity"],
            "threshold_authority_preprocessing_identity",
        ),
        raw_rectified_preprocessing_same=authority_raw[
            "raw_rectified_preprocessing_same"
        ],
        registered_candidate_identity=_require_identity(
            authority_raw["registered_candidate_identity"],
            "threshold_authority_candidate_identity",
        ),
        registered_candidate_ids=tuple(
            _require_sequence(
                authority_raw["registered_candidate_ids"],
                "threshold_authority_candidate_ids",
            )
        ),
        registered_candidate_parameter_bindings=(
            _load_candidate_parameter_bindings(
                authority_raw["registered_candidate_parameter_bindings"]
            )
        ),
        registered_candidate_config_digest=_require_digest(
            authority_raw["registered_candidate_config_digest"],
            "threshold_authority_candidate_config_digest",
        ),
        registered_key_schedule_candidate_id=_require_identity(
            authority_raw["registered_key_schedule_candidate_id"],
            "threshold_authority_key_schedule_candidate_id",
        ),
        registered_key_schedule_derivation_identity=_require_identity(
            authority_raw["registered_key_schedule_derivation_identity"],
            "threshold_authority_key_schedule_derivation_identity",
        ),
        registered_key_schedule_config_digest=_require_digest(
            authority_raw["registered_key_schedule_config_digest"],
            "threshold_authority_key_schedule_config_digest",
        ),
        public_key_relation=_require_identity(
            authority_raw["public_key_relation"],
            "threshold_authority_public_key_relation",
        ),
    )
    unit_roster = _build_study_unit_roster(matrix_tuple)

    budget_raw = _require_mapping(raw["study_budget"], "study_budget")
    budget_keys = frozenset(
        {
            "preflight_source_cluster_count",
            "wiring_source_cluster_count",
            "wiring_counts_as_scientific_coverage",
            "scientific_source_cluster_scales",
            "maximum_scientific_units",
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

    intent_policy_raw = _require_mapping(
        raw["execution_intent_policy"],
        "execution_intent_policy",
    )
    _require_exact_keys(
        intent_policy_raw,
        frozenset(
            {
                "authority_role",
                "create_only_required",
                "freeze_before_scientific_records",
                "raw_secret_policy",
                "expected_digest_required_at_result_boundaries",
                "later_runner_must_pin_digest",
            }
        ),
        "execution_intent_policy",
    )
    execution_intent_policy = DevelopmentExecutionIntentPolicy(
        authority_role=_require_identity(
            intent_policy_raw["authority_role"],
            "execution_intent_authority_role",
        ),
        create_only_required=intent_policy_raw["create_only_required"],
        freeze_before_scientific_records=intent_policy_raw[
            "freeze_before_scientific_records"
        ],
        raw_secret_policy=_require_identity(
            intent_policy_raw["raw_secret_policy"],
            "execution_intent_raw_secret_policy",
        ),
        expected_digest_required_at_result_boundaries=intent_policy_raw[
            "expected_digest_required_at_result_boundaries"
        ],
        later_runner_must_pin_digest=intent_policy_raw[
            "later_runner_must_pin_digest"
        ],
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
        threshold_detector_authority=threshold_detector_authority,
        execution_intent_policy=execution_intent_policy,
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
    *,
    protocol: FrozenDevelopmentExplorationProtocol,
    seed_namespace: str,
    known_role_manifest_bindings: Sequence[
        FrozenStudyRoleManifestBinding
    ] = (),
) -> tuple[SplitAssignment, ...]:
    binding = bind_study_role_manifest(
        protocol,
        role_id="development_exploration",
        seed_namespace=seed_namespace,
        manifest=manifest,
    )
    assert_study_role_manifests_isolated(
        protocol,
        (binding, *known_role_manifest_bindings),
    )
    return manifest.assignments


@dataclass(frozen=True, slots=True)
class DevelopmentPrimaryNullKeyBinding:
    source_cluster_id: str
    registered_key_family_digest: str
    registered_key_public_digest: str
    detection_key_public_digest: str

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        for field_name in (
            "source_cluster_id",
            "registered_key_family_digest",
            "registered_key_public_digest",
            "detection_key_public_digest",
        ):
            if _DIGEST_PATTERN.fullmatch(getattr(self, field_name)) is None:
                violations.append(f"primary_null_{field_name}_invalid")
        return tuple(violations)


def _manifest_assignment_identity_digest(manifest: FrozenSplitManifest) -> str:
    return _canonical_digest(
        tuple(asdict(assignment.identity) for assignment in manifest.assignments)
    )


def _manifest_cluster_identities(
    manifest: FrozenSplitManifest,
) -> tuple[AnalysisUnitIdentity, ...]:
    by_cluster: dict[str, AnalysisUnitIdentity] = {}
    for assignment in manifest.assignments:
        identity = assignment.identity
        existing = by_cluster.get(identity.source_cluster_id)
        if existing is not None and existing != identity:
            raise ValueError("development_manifest_cluster_identity_not_unique")
        by_cluster[identity.source_cluster_id] = identity
    return tuple(by_cluster[key] for key in sorted(by_cluster))


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentExecutionIntentAuthority:
    authority_digest: str
    authority_role: str
    run_id: str
    seed_namespace: str
    protocol: FrozenDevelopmentExplorationProtocol
    protocol_digest: str
    input_manifest: FrozenSplitManifest
    input_manifest_digest: str
    assignment_identity_digest: str
    source_cluster_identity_digest: str
    public_key_roster: tuple[DevelopmentPrimaryNullKeyBinding, ...]
    public_key_roster_digest: str
    detector_authority_digest: str
    key_schedule_candidate_id: str
    key_schedule_derivation_identity: str
    key_schedule_config_digest: str
    raw_secret_policy: str

    def payload_without_authority_digest(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("authority_digest")
        return payload

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if type(self.protocol) is not FrozenDevelopmentExplorationProtocol:
            return ("execution_intent_protocol_exact_type_required",)
        if self.protocol.validate():
            violations.append("execution_intent_protocol_invalid")
        if self.protocol_digest != self.protocol.digest():
            violations.append("execution_intent_protocol_digest_invalid")
        policy = self.protocol.execution_intent_policy
        authority = self.protocol.threshold_detector_authority
        if self.authority_role != policy.authority_role:
            violations.append("execution_intent_authority_role_mismatch")
        if _IDENTITY_PATTERN.fullmatch(self.run_id) is None:
            violations.append("execution_intent_run_id_invalid")
        if _IDENTITY_PATTERN.fullmatch(self.seed_namespace) is None:
            violations.append("execution_intent_seed_namespace_invalid")
        if type(self.input_manifest) is not FrozenSplitManifest:
            return tuple((*violations, "execution_intent_manifest_exact_type_required"))
        manifest_violations = self.input_manifest.validate(require_all_splits=False)
        if manifest_violations:
            violations.append("execution_intent_manifest_invalid")
        if any(
            assignment.split != DEVELOPMENT_SPLIT
            for assignment in self.input_manifest.assignments
        ):
            violations.append("execution_intent_manifest_split_invalid")
        if self.input_manifest_digest != self.input_manifest.digest():
            violations.append("execution_intent_manifest_digest_invalid")
        if self.assignment_identity_digest != _manifest_assignment_identity_digest(
            self.input_manifest
        ):
            violations.append("execution_intent_assignment_identity_digest_invalid")
        try:
            cluster_identities = _manifest_cluster_identities(self.input_manifest)
        except ValueError:
            violations.append("execution_intent_cluster_identity_not_unique")
            cluster_identities = ()
        if self.source_cluster_identity_digest != _canonical_digest(
            tuple(asdict(identity) for identity in cluster_identities)
        ):
            violations.append("execution_intent_cluster_identity_digest_invalid")
        expected_clusters = {
            identity.source_cluster_id: identity for identity in cluster_identities
        }
        observed_roster: dict[str, DevelopmentPrimaryNullKeyBinding] = {}
        for item in self.public_key_roster:
            if type(item) is not DevelopmentPrimaryNullKeyBinding:
                violations.append("execution_intent_key_binding_exact_type_required")
                continue
            violations.extend(item.validate())
            if item.source_cluster_id in observed_roster:
                violations.append("execution_intent_key_binding_cluster_duplicate")
            observed_roster[item.source_cluster_id] = item
            identity = expected_clusters.get(item.source_cluster_id)
            if identity is None:
                violations.append("execution_intent_key_binding_cluster_unknown")
                continue
            if item.registered_key_family_digest != (
                identity.registered_key_family_digest
            ):
                violations.append("execution_intent_key_family_manifest_mismatch")
            try:
                derived_family = derive_development_primary_null_key_family_digest(
                    authority,
                    registered_key_public_digest=item.registered_key_public_digest,
                    detection_key_public_digest=item.detection_key_public_digest,
                )
            except ValueError:
                violations.append("execution_intent_public_key_relation_invalid")
            else:
                if derived_family != item.registered_key_family_digest:
                    violations.append("execution_intent_key_family_roster_mismatch")
        if set(observed_roster) != set(expected_clusters):
            violations.append("execution_intent_public_key_roster_coverage_invalid")
        if self.public_key_roster_digest != _canonical_digest(
            tuple(asdict(item) for item in self.public_key_roster)
        ):
            violations.append("execution_intent_public_key_roster_digest_invalid")
        if self.detector_authority_digest != authority.digest():
            violations.append("execution_intent_detector_authority_digest_invalid")
        if self.key_schedule_candidate_id != (
            authority.registered_key_schedule_candidate_id
        ):
            violations.append("execution_intent_key_schedule_candidate_mismatch")
        if self.key_schedule_derivation_identity != (
            authority.registered_key_schedule_derivation_identity
        ):
            violations.append("execution_intent_key_schedule_derivation_mismatch")
        if self.key_schedule_config_digest != (
            authority.registered_key_schedule_config_digest
        ):
            violations.append("execution_intent_key_schedule_config_mismatch")
        if self.raw_secret_policy != policy.raw_secret_policy:
            violations.append("execution_intent_raw_secret_policy_mismatch")
        if self.authority_digest != _canonical_digest(
            self.payload_without_authority_digest()
        ):
            violations.append("execution_intent_authority_digest_invalid")
        return tuple(dict.fromkeys(violations))


def create_frozen_development_execution_intent_authority(
    protocol: FrozenDevelopmentExplorationProtocol,
    *,
    run_id: str,
    seed_namespace: str,
    input_manifest: FrozenSplitManifest,
    public_key_roster: Sequence[DevelopmentPrimaryNullKeyBinding],
) -> FrozenDevelopmentExecutionIntentAuthority:
    if type(protocol) is not FrozenDevelopmentExplorationProtocol:
        raise TypeError("execution_intent_protocol_exact_type_required")
    protocol_violations = protocol.validate()
    if protocol_violations:
        raise ValueError(",".join(protocol_violations))
    if type(input_manifest) is not FrozenSplitManifest:
        raise TypeError("execution_intent_manifest_exact_type_required")
    cluster_identities = _manifest_cluster_identities(input_manifest)
    roster = tuple(public_key_roster)
    authority = protocol.threshold_detector_authority
    payload = {
        "authority_role": protocol.execution_intent_policy.authority_role,
        "run_id": run_id,
        "seed_namespace": seed_namespace,
        "protocol": protocol,
        "protocol_digest": protocol.digest(),
        "input_manifest": input_manifest,
        "input_manifest_digest": input_manifest.digest(),
        "assignment_identity_digest": _manifest_assignment_identity_digest(
            input_manifest
        ),
        "source_cluster_identity_digest": _canonical_digest(
            tuple(asdict(identity) for identity in cluster_identities)
        ),
        "public_key_roster": roster,
        "public_key_roster_digest": _canonical_digest(
            tuple(asdict(item) for item in roster)
        ),
        "detector_authority_digest": authority.digest(),
        "key_schedule_candidate_id": (
            authority.registered_key_schedule_candidate_id
        ),
        "key_schedule_derivation_identity": (
            authority.registered_key_schedule_derivation_identity
        ),
        "key_schedule_config_digest": authority.registered_key_schedule_config_digest,
        "raw_secret_policy": protocol.execution_intent_policy.raw_secret_policy,
    }
    intent = FrozenDevelopmentExecutionIntentAuthority(
        authority_digest=_canonical_digest(
            {
                key: (
                    asdict(value)
                    if key in {"protocol", "input_manifest"}
                    else tuple(asdict(item) for item in value)
                    if key == "public_key_roster"
                    else value
                )
                for key, value in payload.items()
            }
        ),
        **payload,
    )
    violations = intent.validate()
    if violations:
        raise ValueError(",".join(violations))
    return intent


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
    execution_intent_authority: FrozenDevelopmentExecutionIntentAuthority
    expected_execution_intent_authority_digest: str
    input_manifest: FrozenSplitManifest
    input_manifest_digest: str
    assignment_identity_digest: str
    source_cluster_ids: tuple[str, ...]
    source_cluster_identity_digest: str
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
        if type(self.execution_intent_authority) is not (
            FrozenDevelopmentExecutionIntentAuthority
        ):
            return tuple((*violations, "cross_fit_execution_intent_exact_type_required"))
        intent_violations = self.execution_intent_authority.validate()
        if intent_violations:
            violations.append("cross_fit_execution_intent_invalid")
        if self.expected_execution_intent_authority_digest != (
            self.execution_intent_authority.authority_digest
        ):
            violations.append("cross_fit_expected_execution_intent_digest_mismatch")
        if type(self.input_manifest) is not FrozenSplitManifest:
            return tuple((*violations, "cross_fit_input_manifest_exact_type_required"))
        if self.input_manifest != self.execution_intent_authority.input_manifest:
            violations.append("cross_fit_input_manifest_intent_mismatch")
        if self.input_manifest_digest != self.input_manifest.digest():
            violations.append("cross_fit_input_manifest_digest_invalid")
        if self.input_manifest_digest != (
            self.execution_intent_authority.input_manifest_digest
        ):
            violations.append("cross_fit_input_manifest_digest_intent_mismatch")
        if self.assignment_identity_digest != _manifest_assignment_identity_digest(
            self.input_manifest
        ):
            violations.append("cross_fit_assignment_identity_digest_invalid")
        if self.assignment_identity_digest != (
            self.execution_intent_authority.assignment_identity_digest
        ):
            violations.append("cross_fit_assignment_identity_digest_intent_mismatch")
        try:
            manifest_identities = _manifest_cluster_identities(self.input_manifest)
        except ValueError:
            violations.append("cross_fit_manifest_cluster_identity_not_unique")
            manifest_identities = ()
        expected_cluster_ids = tuple(
            identity.source_cluster_id for identity in manifest_identities
        )
        if self.source_cluster_ids != expected_cluster_ids:
            violations.append("cross_fit_source_cluster_roster_invalid")
        if self.source_cluster_identity_digest != _canonical_digest(
            tuple(asdict(identity) for identity in manifest_identities)
        ):
            violations.append("cross_fit_source_cluster_identity_digest_invalid")
        if self.source_cluster_identity_digest != (
            self.execution_intent_authority.source_cluster_identity_digest
        ):
            violations.append("cross_fit_source_cluster_identity_intent_mismatch")
        if self.source_cluster_count != len(expected_cluster_ids):
            violations.append("cross_fit_source_cluster_count_manifest_mismatch")
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
        all_clusters = set(self.source_cluster_ids)
        if set(probed) != all_clusters:
            violations.append("cross_fit_recovery_probe_manifest_partition_invalid")
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
    execution_intent_authority: FrozenDevelopmentExecutionIntentAuthority,
    expected_execution_intent_authority_digest: str,
    expected_source_cluster_count: int,
) -> FrozenDevelopmentCrossFitPlan:
    if expected_source_cluster_count == WIRING_SOURCE_CLUSTER_COUNT:
        raise ValueError("wiring_clusters_do_not_count_as_scientific_coverage")
    if expected_source_cluster_count not in SCIENTIFIC_SOURCE_CLUSTER_SCALES:
        raise ValueError("development_scientific_source_cluster_scale_invalid")
    if type(execution_intent_authority) is not FrozenDevelopmentExecutionIntentAuthority:
        raise TypeError("cross_fit_execution_intent_exact_type_required")
    intent_violations = execution_intent_authority.validate()
    if intent_violations:
        raise ValueError(",".join(intent_violations))
    if expected_execution_intent_authority_digest != (
        execution_intent_authority.authority_digest
    ):
        raise PermissionError("cross_fit_expected_execution_intent_digest_mismatch")
    manifest = execution_intent_authority.input_manifest
    assignments = manifest.assignments
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
    cluster_identities = _manifest_cluster_identities(manifest)
    cluster_ids = tuple(identity.source_cluster_id for identity in cluster_identities)
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
        execution_intent_authority=execution_intent_authority,
        expected_execution_intent_authority_digest=(
            expected_execution_intent_authority_digest
        ),
        input_manifest=manifest,
        input_manifest_digest=manifest.digest(),
        assignment_identity_digest=_manifest_assignment_identity_digest(manifest),
        source_cluster_ids=cluster_ids,
        source_cluster_identity_digest=_canonical_digest(
            tuple(asdict(identity) for identity in cluster_identities)
        ),
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
class FrozenDevelopmentThresholdDetectorBinding:
    binding_identity: str
    execution_intent_authority: FrozenDevelopmentExecutionIntentAuthority
    expected_execution_intent_authority_digest: str
    protocol: FrozenDevelopmentExplorationProtocol
    protocol_digest: str
    authority_digest: str
    detector_identity: str
    preprocessing_identity: str
    public_key_relation: str
    primary_null_key_bindings: tuple[DevelopmentPrimaryNullKeyBinding, ...]
    primary_null_key_roster_digest: str
    detector_base_config_payload_json: str
    detector_config_payload_json: str
    detector_config_digest: str

    def payload_without_identity(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("binding_identity")
        return payload

    def validate(
        self,
        plan: FrozenDevelopmentCrossFitPlan,
        manifest: FrozenSplitManifest,
        fold_index: int,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if type(plan) is not FrozenDevelopmentCrossFitPlan or plan.validate():
            return ("development_cross_fit_plan_invalid",)
        if type(self.execution_intent_authority) is not (
            FrozenDevelopmentExecutionIntentAuthority
        ):
            return ("threshold_execution_intent_exact_type_required",)
        if self.execution_intent_authority.validate():
            violations.append("threshold_execution_intent_invalid")
        if self.execution_intent_authority != plan.execution_intent_authority:
            violations.append("threshold_execution_intent_plan_mismatch")
        if self.expected_execution_intent_authority_digest != (
            self.execution_intent_authority.authority_digest
        ):
            violations.append("threshold_expected_execution_intent_digest_mismatch")
        if self.expected_execution_intent_authority_digest != (
            plan.expected_execution_intent_authority_digest
        ):
            violations.append("threshold_expected_execution_intent_plan_mismatch")
        if type(self.protocol) is not FrozenDevelopmentExplorationProtocol:
            return ("threshold_detector_protocol_exact_type_required",)
        protocol_violations = self.protocol.validate()
        if protocol_violations:
            violations.append("threshold_detector_protocol_invalid")
        if self.protocol_digest != self.protocol.digest():
            violations.append("threshold_detector_protocol_digest_invalid")
        if self.protocol != self.execution_intent_authority.protocol:
            violations.append("threshold_detector_protocol_intent_mismatch")
        authority = self.protocol.threshold_detector_authority
        if self.authority_digest != authority.digest():
            violations.append("threshold_detector_authority_digest_invalid")
        if plan.responsibility_id != authority.responsibility_id:
            violations.append("threshold_detector_responsibility_authority_mismatch")
        if fold_index not in range(len(plan.folds)):
            return ("threshold_detector_binding_fold_invalid",)
        if manifest != plan.input_manifest:
            violations.append("threshold_detector_manifest_plan_mismatch")
        if manifest.digest() != plan.input_manifest_digest:
            violations.append("threshold_detector_manifest_digest_plan_mismatch")
        if self.detector_identity != authority.detector_identity:
            violations.append("threshold_detector_identity_authority_mismatch")
        if self.preprocessing_identity != authority.preprocessing_identity:
            violations.append("threshold_preprocessing_identity_authority_mismatch")
        if self.public_key_relation != authority.public_key_relation:
            violations.append("primary_null_public_key_relation_authority_mismatch")
        for item in self.primary_null_key_bindings:
            if type(item) is not DevelopmentPrimaryNullKeyBinding:
                violations.append("primary_null_key_binding_exact_type_required")
            else:
                violations.extend(item.validate())
                if item.registered_key_public_digest == item.detection_key_public_digest:
                    violations.append("primary_null_public_key_relation_mismatch")
                elif item.registered_key_family_digest != (
                    derive_development_primary_null_key_family_digest(
                        authority,
                        registered_key_public_digest=(
                            item.registered_key_public_digest
                        ),
                        detection_key_public_digest=(
                            item.detection_key_public_digest
                        ),
                    )
                ):
                    violations.append("primary_null_key_family_public_roster_mismatch")
        cluster_ids = tuple(
            item.source_cluster_id for item in self.primary_null_key_bindings
        )
        if len(cluster_ids) != len(set(cluster_ids)):
            violations.append("primary_null_key_binding_cluster_duplicate")
        expected_clusters = set(plan.folds[fold_index].fit_source_cluster_ids)
        if set(cluster_ids) != expected_clusters:
            violations.append("primary_null_key_binding_fold_coverage_invalid")
        manifest_families = {
            assignment.identity.source_cluster_id: (
                assignment.identity.registered_key_family_digest
            )
            for assignment in manifest.assignments
            if assignment.identity.source_cluster_id in expected_clusters
        }
        if {
            item.source_cluster_id: item.registered_key_family_digest
            for item in self.primary_null_key_bindings
        } != manifest_families:
            violations.append("primary_null_key_family_manifest_mapping_invalid")
        full_roster = {
            item.source_cluster_id: item
            for item in self.execution_intent_authority.public_key_roster
        }
        expected_fit_roster = {
            cluster_id: full_roster[cluster_id]
            for cluster_id in expected_clusters
            if cluster_id in full_roster
        }
        if {
            item.source_cluster_id: item for item in self.primary_null_key_bindings
        } != expected_fit_roster:
            violations.append("primary_null_key_roster_execution_intent_mismatch")
        if self.primary_null_key_roster_digest != _canonical_digest(
            tuple(asdict(item) for item in self.primary_null_key_bindings)
        ):
            violations.append("primary_null_key_roster_digest_invalid")
        try:
            base_payload = json.loads(self.detector_base_config_payload_json)
            config_payload = json.loads(self.detector_config_payload_json)
        except (TypeError, json.JSONDecodeError):
            violations.append("threshold_detector_config_payload_unreadable")
        else:
            expected_base_json = json.dumps(
                base_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            if type(base_payload) is not dict or not base_payload:
                violations.append("threshold_detector_base_config_missing")
            elif self.detector_base_config_payload_json != expected_base_json:
                violations.append("threshold_detector_base_config_not_canonical")
            expected_config = {
                "detector_base_config": base_payload,
                "detector_identity": self.detector_identity,
                "preprocessing_identity": self.preprocessing_identity,
                "primary_null_key_roster_digest": self.primary_null_key_roster_digest,
                "public_key_relation": self.public_key_relation,
                "protocol_digest": self.protocol_digest,
                "threshold_detector_authority_digest": self.authority_digest,
                "execution_intent_authority_digest": (
                    self.expected_execution_intent_authority_digest
                ),
            }
            authority_base_payload = json.loads(
                json.dumps(
                    authority.detector_base_config_payload(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            if base_payload != authority_base_payload:
                violations.append("threshold_detector_base_config_authority_mismatch")
            if config_payload != expected_config:
                violations.append("threshold_detector_config_payload_binding_invalid")
            expected_config_json = json.dumps(
                expected_config,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            if self.detector_config_payload_json != expected_config_json:
                violations.append("threshold_detector_config_payload_not_canonical")
            if self.detector_config_digest != _canonical_digest(expected_config):
                violations.append("threshold_detector_config_digest_invalid")
        if self.binding_identity != _canonical_digest(self.payload_without_identity()):
            violations.append("threshold_detector_binding_identity_invalid")
        return tuple(dict.fromkeys(violations))


def create_development_threshold_detector_binding(
    plan: FrozenDevelopmentCrossFitPlan,
    *,
    expected_execution_intent_authority_digest: str,
    fold_index: int,
    input_manifest: FrozenSplitManifest,
    primary_null_key_bindings: Sequence[DevelopmentPrimaryNullKeyBinding],
) -> FrozenDevelopmentThresholdDetectorBinding:
    if type(plan) is not FrozenDevelopmentCrossFitPlan or plan.validate():
        raise ValueError("development_cross_fit_plan_invalid")
    intent = plan.execution_intent_authority
    if expected_execution_intent_authority_digest != intent.authority_digest:
        raise PermissionError("threshold_expected_execution_intent_digest_mismatch")
    protocol = intent.protocol
    authority = protocol.threshold_detector_authority
    detector_base_config_payload = authority.detector_base_config_payload()
    key_bindings = tuple(primary_null_key_bindings)
    key_roster_digest = _canonical_digest(tuple(asdict(item) for item in key_bindings))
    base_json = json.dumps(
        detector_base_config_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    config_payload = {
        "detector_base_config": detector_base_config_payload,
        "detector_identity": authority.detector_identity,
        "preprocessing_identity": authority.preprocessing_identity,
        "primary_null_key_roster_digest": key_roster_digest,
        "protocol_digest": protocol.digest(),
        "public_key_relation": authority.public_key_relation,
        "threshold_detector_authority_digest": authority.digest(),
        "execution_intent_authority_digest": (
            expected_execution_intent_authority_digest
        ),
    }
    config_json = json.dumps(
        config_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    payload = {
        "execution_intent_authority": intent,
        "expected_execution_intent_authority_digest": (
            expected_execution_intent_authority_digest
        ),
        "protocol": protocol,
        "protocol_digest": protocol.digest(),
        "authority_digest": authority.digest(),
        "detector_identity": authority.detector_identity,
        "preprocessing_identity": authority.preprocessing_identity,
        "public_key_relation": authority.public_key_relation,
        "primary_null_key_bindings": key_bindings,
        "primary_null_key_roster_digest": key_roster_digest,
        "detector_base_config_payload_json": base_json,
        "detector_config_payload_json": config_json,
        "detector_config_digest": _canonical_digest(config_payload),
    }
    binding = FrozenDevelopmentThresholdDetectorBinding(
        binding_identity=_canonical_digest(
            {
                key: tuple(asdict(item) for item in value)
                if key == "primary_null_key_bindings"
                else asdict(value)
                if key in {"protocol", "execution_intent_authority"}
                else value
                for key, value in payload.items()
            }
        ),
        **payload,
    )
    violations = binding.validate(plan, input_manifest, fold_index)
    if violations:
        raise ValueError(",".join(violations))
    return binding


@dataclass(frozen=True, slots=True)
class DevelopmentThresholdFitInput:
    source_record: InternalValidationRecord
    case_role: str
    expected_execution_intent_authority_digest: str
    source_record_digest: str

    def payload_without_digest(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("source_record_digest")
        return payload

    def validate(
        self,
        fold: DevelopmentCrossFitFold,
        manifest: FrozenSplitManifest,
        detector_binding: FrozenDevelopmentThresholdDetectorBinding,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if type(self.source_record) is not InternalValidationRecord:
            return ("threshold_fit_source_record_exact_type_required",)
        record = self.source_record
        identity = record.analysis_unit_identity
        record_violations = validate_internal_record(record)
        if record_violations:
            violations.append("threshold_fit_source_record_invalid")
        if record.split != DEVELOPMENT_SPLIT:
            violations.append("threshold_fit_input_split_invalid")
        if self.expected_execution_intent_authority_digest != (
            detector_binding.expected_execution_intent_authority_digest
        ):
            violations.append("threshold_fit_input_execution_intent_mismatch")
        if record.run_id != detector_binding.execution_intent_authority.run_id:
            violations.append("threshold_fit_input_run_identity_mismatch")
        if self.case_role != "primary_null":
            violations.append("threshold_fit_input_role_invalid")
        if identity.case_id != DEVELOPMENT_PRIMARY_NULL_CASE_ID:
            violations.append("threshold_fit_input_case_identity_invalid")
        if identity.source_cluster_id not in set(fold.fit_source_cluster_ids):
            violations.append("threshold_fit_input_cluster_not_in_fit_fold")
        if identity.source_cluster_id in set(fold.recovery_probe_source_cluster_ids):
            violations.append("threshold_fit_input_recovery_probe_leakage")
        if SplitAssignment(identity=identity, split=record.split) not in manifest.assignments:
            violations.append("threshold_fit_input_not_in_bound_manifest")
        if record.provenance_trace.split_manifest_digest != manifest.digest():
            violations.append("threshold_fit_source_record_manifest_digest_mismatch")
        detector = record.detector_trace
        if (
            detector.raw_detector_identity != detector_binding.detector_identity
            or detector.rectified_detector_identity
            != detector_binding.detector_identity
        ):
            violations.append("threshold_fit_input_detector_identity_mismatch")
        if (
            detector.raw_detector_config_digest
            != detector_binding.detector_config_digest
            or detector.rectified_detector_config_digest
            != detector_binding.detector_config_digest
        ):
            violations.append("threshold_fit_input_detector_config_mismatch")
        if (
            detector.raw_preprocessing_identity
            != detector_binding.preprocessing_identity
            or detector.rectified_preprocessing_identity
            != detector_binding.preprocessing_identity
        ):
            violations.append("threshold_fit_input_preprocessing_identity_mismatch")
        if record.key_control_trace.key_role != "unwatermarked_primary_null":
            violations.append("threshold_fit_input_key_role_invalid")
        if record.key_control_trace.control_identity != "primary_null":
            violations.append("threshold_fit_input_control_identity_invalid")
        key_by_cluster = {
            item.source_cluster_id: item
            for item in detector_binding.primary_null_key_bindings
        }
        expected_key = key_by_cluster.get(identity.source_cluster_id)
        if expected_key is None:
            violations.append("threshold_fit_input_key_binding_missing")
        else:
            if (
                identity.registered_key_family_digest
                != expected_key.registered_key_family_digest
            ):
                violations.append("threshold_fit_input_key_family_mismatch")
            if (
                record.key_control_trace.registered_key_public_digest
                != expected_key.registered_key_public_digest
                or record.key_control_trace.detection_key_public_digest
                != expected_key.detection_key_public_digest
            ):
                violations.append("threshold_fit_input_public_key_mapping_mismatch")
        if record.execution_status != "success":
            violations.append("threshold_fit_input_success_record_required")
        score = record.detector_trace.raw_content_score
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            violations.append("threshold_fit_input_score_invalid")
        elif not isfinite(float(score)):
            violations.append("threshold_fit_input_score_non_finite")
        if self.source_record_digest != _canonical_digest(
            self.payload_without_digest()
        ):
            violations.append("threshold_fit_source_record_digest_invalid")
        return tuple(dict.fromkeys(violations))


def create_development_threshold_fit_input(
    *,
    expected_execution_intent_authority_digest: str,
    source_record: InternalValidationRecord,
) -> DevelopmentThresholdFitInput:
    payload = {
        "source_record": source_record,
        "case_role": "primary_null",
        "expected_execution_intent_authority_digest": (
            expected_execution_intent_authority_digest
        ),
    }
    return DevelopmentThresholdFitInput(
        **payload,
        source_record_digest=_canonical_digest(
            {
                "source_record": asdict(source_record),
                "case_role": "primary_null",
                "expected_execution_intent_authority_digest": (
                    expected_execution_intent_authority_digest
                ),
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class DevelopmentProvisionalThreshold:
    threshold_identity: str
    responsibility_id: str
    fold_index: int
    threshold: float
    input_manifest: FrozenSplitManifest
    input_manifest_digest: str
    detector_binding: FrozenDevelopmentThresholdDetectorBinding
    expected_execution_intent_authority_digest: str
    protocol_digest: str
    threshold_detector_authority_digest: str
    detector_identity: str
    detector_config_payload_json: str
    detector_config_digest: str
    threshold_rule_payload_json: str
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
        if self.expected_execution_intent_authority_digest != (
            plan.expected_execution_intent_authority_digest
        ):
            violations.append("provisional_threshold_execution_intent_plan_mismatch")
        if self.fold_index not in range(len(plan.folds)):
            violations.append("provisional_threshold_fold_index_invalid")
            return tuple(violations)
        fold = plan.folds[self.fold_index]
        if type(self.detector_binding) is not FrozenDevelopmentThresholdDetectorBinding:
            violations.append("threshold_detector_binding_exact_type_required")
        else:
            violations.extend(
                self.detector_binding.validate(
                    plan,
                    self.input_manifest,
                    self.fold_index,
                )
            )
        for fit_input in self.fit_inputs:
            if type(fit_input) is not DevelopmentThresholdFitInput:
                violations.append("threshold_fit_input_exact_type_required")
            elif type(self.detector_binding) is FrozenDevelopmentThresholdDetectorBinding:
                violations.extend(
                    fit_input.validate(
                        fold,
                        self.input_manifest,
                        self.detector_binding,
                    )
                )
        if {item.case_role for item in self.fit_inputs} != set(
            DEVELOPMENT_THRESHOLD_INPUT_ROLES
        ):
            violations.append("threshold_fit_input_roles_incomplete")
        covered_fit_clusters = {
            item.source_record.analysis_unit_identity.source_cluster_id
            for item in self.fit_inputs
            if type(item) is DevelopmentThresholdFitInput
            and type(item.source_record) is InternalValidationRecord
        }
        observed_record_ids = tuple(
            item.source_record.record_id
            for item in self.fit_inputs
            if type(item) is DevelopmentThresholdFitInput
            and type(item.source_record) is InternalValidationRecord
        )
        if len(observed_record_ids) != len(set(observed_record_ids)):
            violations.append("threshold_fit_source_record_id_duplicate")
        if len(covered_fit_clusters) != len(self.fit_inputs):
            violations.append("threshold_fit_source_cluster_duplicate")
        if covered_fit_clusters != set(fold.fit_source_cluster_ids):
            violations.append("threshold_fit_input_cluster_coverage_invalid")
        if self.fit_source_cluster_digest != fold.fit_source_cluster_digest:
            violations.append("provisional_threshold_fit_digest_mismatch")
        if self.recovery_probe_source_cluster_digest != (
            fold.recovery_probe_source_cluster_digest
        ):
            violations.append("provisional_threshold_recovery_probe_digest_mismatch")
        if type(self.input_manifest) is not FrozenSplitManifest:
            violations.append("provisional_threshold_manifest_exact_type_required")
        else:
            manifest_violations = self.input_manifest.validate(
                require_all_splits=False
            )
            if manifest_violations:
                violations.append("provisional_threshold_manifest_invalid")
            if self.input_manifest_digest != self.input_manifest.digest():
                violations.append("provisional_threshold_manifest_digest_invalid")
            if self.input_manifest != plan.input_manifest:
                violations.append("provisional_threshold_manifest_plan_mismatch")
            if self.input_manifest_digest != plan.input_manifest_digest:
                violations.append("provisional_threshold_manifest_digest_plan_mismatch")
        for payload_json, digest, reason in (
            (
                self.detector_config_payload_json,
                self.detector_config_digest,
                "provisional_threshold_detector_config_digest_invalid",
            ),
            (
                self.threshold_rule_payload_json,
                self.threshold_rule_digest,
                "provisional_threshold_rule_digest_invalid",
            ),
        ):
            try:
                payload = json.loads(payload_json)
            except (TypeError, json.JSONDecodeError):
                violations.append(reason)
            else:
                if type(payload) is not dict or not payload:
                    violations.append(reason)
                elif payload_json != json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                ):
                    violations.append(reason)
                elif digest != _canonical_digest(payload):
                    violations.append(reason)
        if self.threshold_rule_payload_json != json.dumps(
            DEVELOPMENT_THRESHOLD_RULE_PAYLOAD,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ):
            violations.append("provisional_threshold_rule_unregistered")
        if type(self.detector_binding) is FrozenDevelopmentThresholdDetectorBinding:
            if self.expected_execution_intent_authority_digest != (
                self.detector_binding.expected_execution_intent_authority_digest
            ):
                violations.append(
                    "provisional_threshold_execution_intent_binding_mismatch"
                )
            if self.protocol_digest != self.detector_binding.protocol_digest:
                violations.append("provisional_threshold_protocol_digest_mismatch")
            if self.threshold_detector_authority_digest != (
                self.detector_binding.authority_digest
            ):
                violations.append("provisional_threshold_authority_digest_mismatch")
            if self.detector_identity != self.detector_binding.detector_identity:
                violations.append("provisional_threshold_detector_binding_mismatch")
            if (
                self.detector_config_payload_json
                != self.detector_binding.detector_config_payload_json
                or self.detector_config_digest
                != self.detector_binding.detector_config_digest
            ):
                violations.append("provisional_threshold_detector_config_binding_mismatch")
        if _IDENTITY_PATTERN.fullmatch(self.detector_identity) is None:
            violations.append("provisional_threshold_detector_identity_invalid")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            violations.append("provisional_threshold_value_invalid")
        elif not isfinite(float(self.threshold)):
            violations.append("provisional_threshold_value_non_finite")
        elif self.fit_inputs and self.threshold != max(
            float(item.source_record.detector_trace.raw_content_score)
            for item in self.fit_inputs
            if type(item) is DevelopmentThresholdFitInput
        ):
            violations.append("provisional_threshold_value_not_rule_derived")
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
    expected_execution_intent_authority_digest: str,
    fold_index: int,
    input_manifest: FrozenSplitManifest,
    detector_binding: FrozenDevelopmentThresholdDetectorBinding,
    fit_inputs: Sequence[DevelopmentThresholdFitInput],
) -> DevelopmentProvisionalThreshold:
    if type(plan) is not FrozenDevelopmentCrossFitPlan or plan.validate():
        raise ValueError("development_cross_fit_plan_invalid")
    if expected_execution_intent_authority_digest != (
        plan.expected_execution_intent_authority_digest
    ):
        raise PermissionError("provisional_threshold_execution_intent_mismatch")
    if fold_index not in range(len(plan.folds)):
        raise ValueError("provisional_threshold_fold_index_invalid")
    if type(input_manifest) is not FrozenSplitManifest:
        raise TypeError("development_threshold_manifest_exact_type_required")
    if type(detector_binding) is not FrozenDevelopmentThresholdDetectorBinding:
        raise TypeError("threshold_detector_binding_exact_type_required")
    if not fit_inputs:
        raise ValueError("development_threshold_fit_inputs_missing")
    threshold_rule_payload_json = json.dumps(
        DEVELOPMENT_THRESHOLD_RULE_PAYLOAD,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    fold = plan.folds[fold_index]
    payload = {
        "responsibility_id": plan.responsibility_id,
        "fold_index": fold_index,
        "threshold": max(
            float(item.source_record.detector_trace.raw_content_score)
            for item in fit_inputs
        ),
        "input_manifest": input_manifest,
        "input_manifest_digest": input_manifest.digest(),
        "detector_binding": detector_binding,
        "expected_execution_intent_authority_digest": (
            expected_execution_intent_authority_digest
        ),
        "protocol_digest": detector_binding.protocol_digest,
        "threshold_detector_authority_digest": detector_binding.authority_digest,
        "detector_identity": detector_binding.detector_identity,
        "detector_config_payload_json": (
            detector_binding.detector_config_payload_json
        ),
        "detector_config_digest": detector_binding.detector_config_digest,
        "threshold_rule_payload_json": threshold_rule_payload_json,
        "threshold_rule_digest": _canonical_digest(
            DEVELOPMENT_THRESHOLD_RULE_PAYLOAD
        ),
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
                key: (
                    tuple(asdict(item) for item in value)
                    if key == "fit_inputs"
                    else asdict(value)
                    if key in {"input_manifest", "detector_binding"}
                    else value
                )
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
    expected_execution_intent_authority_digest: str,
    requested_split: str,
    requested_analysis_unit_identity: AnalysisUnitIdentity,
) -> None:
    if type(threshold) is not DevelopmentProvisionalThreshold:
        raise TypeError("development_provisional_threshold_exact_type_required")
    violations = threshold.validate(plan)
    if violations:
        raise ValueError(",".join(violations))
    if expected_execution_intent_authority_digest != (
        plan.expected_execution_intent_authority_digest
    ):
        raise PermissionError("development_execution_intent_digest_mismatch")
    if type(requested_analysis_unit_identity) is not AnalysisUnitIdentity:
        raise TypeError("development_recovery_identity_exact_type_required")
    if requested_analysis_unit_identity.validate():
        raise ValueError("development_recovery_identity_invalid")
    if requested_split != DEVELOPMENT_SPLIT:
        raise PermissionError(
            f"development_provisional_threshold_invalid_for_split:{requested_split}"
        )
    probes = plan.folds[threshold.fold_index].recovery_probe_source_cluster_ids
    source_cluster_id = requested_analysis_unit_identity.source_cluster_id
    if source_cluster_id not in probes:
        raise PermissionError("development_provisional_threshold_fold_leakage")
    requested_assignment = SplitAssignment(
        identity=requested_analysis_unit_identity,
        split=DEVELOPMENT_SPLIT,
    )
    if requested_assignment not in plan.input_manifest.assignments:
        raise PermissionError("development_recovery_identity_not_in_plan_manifest")
    if requested_assignment not in threshold.input_manifest.assignments:
        raise PermissionError("development_recovery_identity_not_in_threshold_manifest")
    expected_identity = next(
        (
            identity
            for identity in _manifest_cluster_identities(plan.input_manifest)
            if identity.source_cluster_id == source_cluster_id
        ),
        None,
    )
    if expected_identity != requested_analysis_unit_identity:
        raise PermissionError("development_recovery_identity_plan_mismatch")


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
                {
                    self.responsibility_id,
                    *studies[
                        self.responsibility_id
                    ].prerequisite_responsibility_ids,
                }
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
        if self.source_record_schema_version != RECORD_SCHEMA_VERSION:
            violations.append("source_record_schema_version_invalid")
        if self.source_record_collection_schema_version != (
            RECORD_COLLECTION_SCHEMA_VERSION
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
        "source_record_schema_version": RECORD_SCHEMA_VERSION,
        "source_record_collection_schema_version": (
            RECORD_COLLECTION_SCHEMA_VERSION
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
