"""13 项方法职责到内部科学问题与证据门的一一验证矩阵。"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.protocol.internal_splits import INTERNAL_VALIDATION_SPLITS


REQUIRED_METHOD_RESPONSIBILITIES = (
    "key_schedule",
    "content_router",
    "lf_carrier",
    "hf_carrier",
    "content_embedder",
    "lf_detector",
    "hf_detector",
    "content_detector",
    "qk_geometry_sync",
    "geometric_transform_estimator",
    "geometry_reliability",
    "image_rectifier",
    "conditional_recovery_decision",
)

REQUIRED_RECORD_FIELD_GROUPS = frozenset(
    {
        "detector_trace",
        "branch_score_trace",
        "routing_trace",
        "geometry_trace",
        "threshold_trace",
        "key_control_trace",
        "decision_trace",
        "provenance_trace",
    }
)


@dataclass(frozen=True)
class ResponsibilityValidationSpec:
    responsibility: str
    scientific_question: str
    splits: tuple[str, ...]
    metrics: tuple[str, ...]
    negative_controls: tuple[str, ...]
    promotion_gates: tuple[str, ...]
    record_fields: tuple[str, ...]


RESPONSIBILITY_VALIDATION_MATRIX = (
    ResponsibilityValidationSpec(
        "key_schedule",
        "Registered, wrong-key and public-noise identities remain separated without secret persistence.",
        ("candidate_selection", "untouched_confirmation"),
        ("key_attribution_separation", "domain_collision_count"),
        ("wrong_key_roster", "public_noise_secret_independence"),
        ("key_identity_gate_passed",),
        ("key_control_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "content_router",
        "Content-adaptive routing improves a preregistered condition at matched total energy.",
        ("candidate_selection", "untouched_confirmation"),
        ("matched_budget_incremental_tpr", "routing_coverage", "quality_delta"),
        ("routing_uniform_control", "attack_label_blindness"),
        ("routing_confirmation_gate_passed",),
        ("routing_trace", "branch_score_trace", "key_control_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "lf_carrier",
        "The LF write direction provides key-specific signal rather than a generic low-frequency shift.",
        ("candidate_selection", "untouched_confirmation"),
        ("lf_attribution_tpr", "lf_primary_null_fpr", "quality_delta"),
        ("wrong_key", "unwatermarked", "hf_only"),
        ("lf_identity_gate_passed",),
        ("branch_score_trace", "key_control_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "hf_carrier",
        "The frozen HF carrier preserves the registered candidate identity and image-quality boundary.",
        ("development", "untouched_confirmation"),
        ("hf_attribution_tpr", "hf_primary_null_fpr", "quality_delta"),
        ("wrong_key", "unwatermarked", "clean_pair"),
        ("hf_candidate_identity_gate_passed",),
        ("branch_score_trace", "key_control_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "content_embedder",
        "LF-only, HF-only, routed and combined writes are compared under one realized total-content budget.",
        ("candidate_selection", "untouched_confirmation"),
        ("realized_total_relative_l2", "matched_budget_quality_delta"),
        ("hf_only", "lf_only", "routing_uniform_control"),
        ("matched_total_budget_gate_passed",),
        ("branch_score_trace", "routing_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "lf_detector",
        "Blind LF scoring retains registered-key attribution on ordinary images.",
        ("candidate_selection", "untouched_confirmation"),
        ("lf_tpr_at_selection_fpr", "lf_wrong_key_rate"),
        ("wrong_key", "unwatermarked", "content_disabled"),
        ("lf_detector_attribution_gate_passed",),
        ("detector_trace", "branch_score_trace", "key_control_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "hf_detector",
        "Blind HF direct scoring provides the frozen HF-only reference for later non-degradation checks.",
        ("content_threshold_fit", "untouched_confirmation"),
        ("hf_tpr_at_frozen_fpr", "hf_wrong_key_rate"),
        ("wrong_key", "unwatermarked"),
        ("hf_detector_reference_gate_passed",),
        ("detector_trace", "branch_score_trace", "threshold_trace", "key_control_trace"),
    ),
    ResponsibilityValidationSpec(
        "content_detector",
        "A preregistered LF/HF combination beats HF-only without masking key failures.",
        (
            "candidate_selection",
            "untouched_confirmation",
            "content_threshold_fit",
            "end_to_end_check",
        ),
        ("combined_tpr", "combined_primary_null_fpr", "hf_non_degradation", "wrong_key_rate"),
        ("hf_only_c0", "wrong_key", "unwatermarked"),
        ("content_branch_promotion_gate_passed", "content_threshold_gate_passed"),
        (
            "detector_trace",
            "branch_score_trace",
            "threshold_trace",
            "key_control_trace",
            "decision_trace",
        ),
    ),
    ResponsibilityValidationSpec(
        "qk_geometry_sync",
        "Keyed Q/K synchronization is observable for the registered key and rejects wrong-key controls.",
        ("candidate_selection", "reliability_fit", "untouched_confirmation"),
        ("relation_score_gain", "wrong_key_relation_margin", "quality_delta"),
        ("wrong_geometry_key", "geometry_disabled"),
        ("qk_sync_gate_passed",),
        ("geometry_trace", "key_control_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "geometric_transform_estimator",
        "Blind bounded estimation recovers supported crop, scale and rotation without oracle inputs.",
        ("reliability_fit", "untouched_confirmation", "end_to_end_check"),
        ("rotation_error", "scale_error", "translation_error", "coverage", "residual"),
        ("identity", "wrong_geometry_key", "ambiguous_transform", "oracle_diagnostic_only"),
        ("transform_estimation_gate_passed",),
        ("geometry_trace", "key_control_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "geometry_reliability",
        "The independent reliability conjunction rejects ambiguous, unsupported and wrong-key estimates.",
        ("reliability_fit", "end_to_end_check"),
        ("reliable_accept_rate", "unreliable_reject_rate", "false_reliable_rate"),
        ("wrong_geometry_key", "low_coverage", "boundary_solution", "non_finite_metric"),
        ("geometry_reliability_gate_passed",),
        ("geometry_trace", "key_control_trace", "decision_trace"),
    ),
    ResponsibilityValidationSpec(
        "image_rectifier",
        "Blind rectification improves the same detector without claiming deleted crop content.",
        ("untouched_confirmation", "end_to_end_check"),
        ("rectification_quality", "same_detector_score_delta", "valid_support"),
        ("identity_warp", "rectification_disabled", "oracle_diagnostic_only"),
        ("rectification_gate_passed",),
        ("detector_trace", "geometry_trace", "decision_trace", "provenance_trace"),
    ),
    ResponsibilityValidationSpec(
        "conditional_recovery_decision",
        "Near-threshold recovery improves TPR while preserving one detector, key semantics and threshold.",
        ("rescue_threshold_fit", "end_to_end_check", "held_out_evaluation"),
        ("incremental_tpr", "end_to_end_fpr", "trigger_rate", "false_rescue_rate"),
        ("raw_only", "geometry_always", "geometry_disabled", "oracle_diagnostic_only"),
        ("rescue_threshold_gate_passed", "end_to_end_gate_passed"),
        (
            "detector_trace",
            "branch_score_trace",
            "geometry_trace",
            "threshold_trace",
            "key_control_trace",
            "decision_trace",
            "provenance_trace",
        ),
    ),
)


def validate_responsibility_matrix(
    matrix: tuple[ResponsibilityValidationSpec, ...] = RESPONSIBILITY_VALIDATION_MATRIX,
) -> tuple[str, ...]:
    violations: list[str] = []
    responsibilities = tuple(spec.responsibility for spec in matrix)
    if responsibilities != REQUIRED_METHOD_RESPONSIBILITIES:
        violations.append("responsibility_order_or_identity_mismatch")
    if len(set(responsibilities)) != len(responsibilities):
        violations.append("responsibility_duplicate")
    for spec in matrix:
        for name in (
            "scientific_question",
            "splits",
            "metrics",
            "negative_controls",
            "promotion_gates",
            "record_fields",
        ):
            if not getattr(spec, name):
                violations.append(f"{spec.responsibility}:{name}_missing")
        if set(spec.splits) - set(INTERNAL_VALIDATION_SPLITS):
            violations.append(f"{spec.responsibility}:split_invalid")
        if set(spec.record_fields) - REQUIRED_RECORD_FIELD_GROUPS:
            violations.append(f"{spec.responsibility}:record_field_group_invalid")
    represented_fields = {
        field_name for spec in matrix for field_name in spec.record_fields
    }
    if represented_fields != REQUIRED_RECORD_FIELD_GROUPS:
        violations.append("record_field_groups_not_fully_represented")
    return tuple(dict.fromkeys(violations))


SPLIT_PREREQUISITE_GATES = {
    "development": (),
    "candidate_selection": (),
    "untouched_confirmation": ("candidate_selection_frozen",),
    "content_threshold_fit": ("content_branch_promotion_gate_passed",),
    "rescue_threshold_fit": ("content_threshold_gate_passed",),
    "reliability_fit": ("qk_sync_gate_passed", "transform_estimation_gate_passed"),
    "end_to_end_check": (
        "content_threshold_gate_passed",
        "rescue_threshold_gate_passed",
        "geometry_reliability_gate_passed",
    ),
    "held_out_evaluation": ("end_to_end_gate_passed",),
}


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    requested_split: str
    missing_gates: tuple[str, ...]
    stop_outcome: str | None


def decide_split_promotion(
    requested_split: str,
    passed_gates: frozenset[str],
) -> PromotionDecision:
    if requested_split not in SPLIT_PREREQUISITE_GATES:
        raise ValueError("requested_split_invalid")
    missing = tuple(
        gate for gate in SPLIT_PREREQUISITE_GATES[requested_split] if gate not in passed_gates
    )
    if missing:
        stop_outcome = (
            "content_branch_research_question_closed_negative"
            if requested_split in {"content_threshold_fit", "rescue_threshold_fit"}
            else "stop_and_return_to_prerequisite_gate"
        )
        return PromotionDecision(False, requested_split, missing, stop_outcome)
    return PromotionDecision(True, requested_split, (), None)
