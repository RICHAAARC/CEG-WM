"""Frozen loader for the clean v2 content-adaptive dual-branch protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

_UNIT_FIELDS = {"unit_id", "split", "source_id", "prompt", "seed", "height", "width"}
_ALLOWED_DETECTION_INPUTS = ("image", "detection_key", "frozen_public_assets")
_COUNTERFACTUAL_EFFECT_FIELDS = [
    "semantic_importance_counterfactual_effect",
    "texture_complexity_counterfactual_effect",
    "lf_transfer_stability_counterfactual_effect",
    "hf_transfer_stability_counterfactual_effect",
    "lf_local_perturbation_sensitivity_counterfactual_effect",
    "hf_local_perturbation_sensitivity_counterfactual_effect",
]
_CONTENT_ANALYSIS = {
    "asset_id": "facebook/dinov2-small",
    "attention_implementation": "eager",
    "attention_layer": "last",
    "attention_statistic": "mean_head_cls_to_patch",
    "tile_grid": [4, 4],
    "tile_count": 16,
    "probe_evaluations_per_branch_tile": 2,
    "probe_evaluations_per_unit": 64,
    "probe_relative_l2_scales_in_order": [0.0005, 0.001],
    "probe_measurement": "baseline_differenced_branch_tile_two_scale_v1",
    "probe_independence": "each_probe_relative_to_complete_current_callback_latent_non_cumulative_never_evolving",
    "probe_domain": "public_key_independent_branch_tile_public_shape_v2",
    "probe_domain_forbidden_inputs": [
        "detection_key", "unit", "prompt", "seed", "content", "source", "candidate_outcome", "results",
    ],
    "baseline_observations": ["I0_equals_D_of_z", "Y0_equals_E_of_I0"],
    "response_rule": "all_probe_responses_use_only_DeltaI_and_DeltaY_never_absolute_observation_magnitude",
    "stability_rule": "two_scale_alignment_times_gain_consistency_equal_mean_RGB_and_VAE",
    "sensitivity_rule": "RGB_and_VAE_each_divide_by_rho_then_unit_positive_median_kappa_then_x_over_x_plus_kappa_all_zero_modality_stays_zero_no_fallback_p_bi_equal_mean_two_scales_by_two_modalities",
    "signals": [
        "semantic_importance", "texture_complexity", "lf_transfer_stability",
        "hf_transfer_stability", "lf_local_perturbation_sensitivity",
        "hf_local_perturbation_sensitivity",
    ],
    "allocation_directions": "semantic_uses_only_magnitude_away_from_0.5_texture_monotonic_opposite_branches_stability_helps_only_own_branch_sensitivity_cannot_help",
    "flat_texture_rule": "neutral_0.5",
    "export": "irreversible_aggregate_scalars_only",
}
_DETECTION_ACCESS = {
    "allowed_inputs": list(_ALLOWED_DETECTION_INPUTS),
    "forbidden_inputs": [
        "original_image", "prompt", "embed_record", "private_latent", "embedding_latent",
        "embed_side_route", "route", "mask", "cached_qk", "qk",
        "lf_branch_share", "hf_branch_share", *_COUNTERFACTUAL_EFFECT_FIELDS,
        "minimum_counterfactual_effect",
    ],
    "threshold_status": "deferred_calibration_not_stage_a",
    "hf_detector": "frozen_hf_final_rgb_public_vae_global_normalized_correlation",
    "lf_detector": "frozen_lf_final_rgb_public_vae_block_centered_normalized_median_correlation",
    "joint_score": "min(s_LF,s_HF)",
}
_AGGREGATE_MEASUREMENT = {
    "counterfactual_effect_fields_in_order": _COUNTERFACTUAL_EFFECT_FIELDS,
    "counterfactual_effect_source": "single_ContentAllocation_counterfactual_effects_pass_through_without_recompute_copy_swap_default_alias_or_fallback",
    "counterfactual_effect_validation": "each_finite_and_nonnegative_zero_allowed",
    "minimum_counterfactual_effect": "min_of_the_six_counterfactual_effect_fields",
    "public_branch_share_fields_in_order": ["lf_branch_share", "hf_branch_share"],
    "public_branch_share_source": "same_ContentAllocation_values_pass_through",
    "public_branch_share_validation": "each_finite_strictly_between_0_and_1_and_sum_to_1_with_absolute_tolerance",
    "branch_share_sum_absolute_tolerance": 1e-12,
    "population_std_fields_in_order": [
        "lf_branch_share_population_std", "hf_branch_share_population_std",
    ],
    "population_std_formula": "sqrt(sum((x_i-mean(x))^2)/8)_ddof_0_each_field_independently",
    "population_std_fixed_roster_units": 8,
    "population_std_absolute_tolerance": 1e-12,
    "population_std_validation": "both_finite_and_theoretically_equal_with_absolute_tolerance",
    "population_std_availability": "complete_finite_identity_valid_RC0_only_else_both_null_and_no_scientific_outcome",
    "allocation_not_all_identical_support": "both_population_std_fields_strictly_positive",
    "blind_score_consumes_aggregate_measurement": False,
    "forbidden_private_exports": [
        "mask", "tile_weights", "attention_map", "latent", "delta", "probe_state",
    ],
}


@dataclass(frozen=True, slots=True)
class ContentChainUnit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ContentChainProtocol:
    protocol_id: str
    config: Mapping[str, Any]
    roster: tuple[ContentChainUnit, ...]
    protocol_digest: str


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _load_roster(path: Path) -> tuple[ContentChainUnit, ...]:
    units: list[ContentChainUnit] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} cannot be blank")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number} is invalid JSON") from error
            if not isinstance(payload, dict) or set(payload) != _UNIT_FIELDS:
                raise ValueError(f"{path.name}:{line_number} has unexpected fields")
            if any(not isinstance(payload[name], str) or not payload[name].strip() for name in ("unit_id", "split", "source_id", "prompt")):
                raise ValueError(f"{path.name}:{line_number} has empty identity text")
            if payload["split"] != "content_adaptive_dual_branch_v2_clean_v1":
                raise ValueError(f"{path.name}:{line_number} has the wrong split")
            if any(not isinstance(payload[name], int) or isinstance(payload[name], bool) for name in ("seed", "height", "width")):
                raise ValueError(f"{path.name}:{line_number} has non-integer runtime values")
            if payload["seed"] < 0 or payload["height"] < 256 or payload["width"] < 256:
                raise ValueError(f"{path.name}:{line_number} has invalid runtime values")
            units.append(ContentChainUnit(**payload))
    expected_ids = [f"content-adaptive-v2-{index:04d}" for index in range(1, 9)]
    expected_sources = [f"content-v2-prompt-{index}" for index in range(8101, 8109)]
    if len(units) != 8 or [unit.unit_id for unit in units] != expected_ids or [unit.source_id for unit in units] != expected_sources:
        raise ValueError("content-adaptive roster differs from the frozen eight units")
    if len({unit.seed for unit in units}) != 8:
        raise ValueError("content-adaptive roster seeds must be unique")
    return tuple(units)


def _validate_global_disjoint(roster_path: Path, roster: tuple[ContentChainUnit, ...]) -> None:
    """Reject unit/source collisions with every other current configuration manifest."""

    unit_ids = {unit.unit_id for unit in roster}
    source_ids = {unit.source_id for unit in roster}
    config_root = roster_path.resolve().parents[1]
    for candidate in config_root.rglob("*.jsonl"):
        if candidate.resolve() == roster_path.resolve():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and (
                    payload.get("unit_id") in unit_ids or payload.get("source_id") in source_ids
                ):
                    raise ValueError("content-adaptive roster is not globally disjoint")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != 1 or config.get("protocol_id") != "cegwm-stage-a-content-adaptive-dual-branch-v2-clean-v1":
        raise ValueError("unexpected content-adaptive protocol identity")
    if config.get("execution_scope_id") != "content_adaptive_dual_branch_v2_clean_engineering_and_stage_a_evaluation_v1":
        raise ValueError("unexpected content-adaptive execution scope")
    if config.get("scientific_status") != "not_evaluated_until_complete_real_gpu_rc0":
        raise ValueError("content-adaptive protocol cannot preclaim scientific evidence")
    runtime = _mapping(config, "generation_runtime")
    if runtime != {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "inference_steps": 20,
        "injection_step_index_zero_based": 18,
        "generation_rule": "independent_same_seed_generators_for_joint_and_primary_null",
    }:
        raise ValueError("content-adaptive generation runtime differs")
    analysis = _mapping(config, "content_analysis")
    if analysis != _CONTENT_ANALYSIS:
        raise ValueError("content analysis identity or fail-closed rules differ")
    identities = _mapping(config, "method_identities")
    if identities != {
        "hf_base_carrier_method_id": "hf_tail_rademacher_v1",
        "hf_base_evaluated_candidate_id": "hf_tail_rademacher_v1_rankgate_v2",
        "lf_base_carrier_method_id": "lf_shell_balanced_blocks_v2",
        "lf_base_evaluated_candidate_id": "lf_shell_balanced_blocks_v2_blocknorm_median_v1",
        "hf_adaptive_embedding_transform_id": "hf_content_tiles_semantic_stability_sensitivity_probe_v2",
        "lf_adaptive_embedding_transform_id": "lf_content_tiles_texture_stability_sensitivity_probe_v2",
        "combined_budget_projector_id": "dual_branch_actual_dtype_relative_l2_v1",
        "evaluated_candidate_id": "content_adaptive_dual_branch_v2_clean_v1",
        "base_prg_domain_rule": "base_carrier_ids_only_adaptive_and_joint_ids_never_enter_base_hf_or_lf_prg_domain",
    }:
        raise ValueError("content-adaptive method identities differ")
    budget = _mapping(config, "budget")
    if budget != {
        "combined_total_relative_l2": 0.012,
        "measurement": "actual_dtype_final_minus_actual_dtype_base",
        "single_shared_budget_not_per_branch": True,
        "both_effective_branches_nonzero": True,
    }:
        raise ValueError("content-adaptive combined budget differs")
    aggregate = _mapping(config, "aggregate_measurement")
    if aggregate != _AGGREGATE_MEASUREMENT:
        raise ValueError("content-adaptive aggregate measurement contract differs")
    access = _mapping(config, "detection_access")
    if access != _DETECTION_ACCESS:
        raise ValueError("blind detection access differs")
    keying = _mapping(config, "keying")
    if keying != {
        "task": "zero_bit_keyed_attribution",
        "normalization": "NFC_UTF8_for_text_exact_bytes_for_binary",
        "prg": "HMAC_SHA256_counter_v1",
        "wrong_key_count": 16,
        "wrong_key_derivation_domain": "stage-a/content-adaptive-v2-external-wrong-key/v1",
        "primary_null": True,
        "payload_bits": 0,
    }:
        raise ValueError("content-adaptive key controls differ")
    flow = _mapping(config, "execution_flow")
    if flow != {
        "roster_manifest": "content_adaptive_dual_branch_v2_clean.jsonl",
        "split": "content_adaptive_dual_branch_v2_clean_v1",
        "fixed_units": 8,
        "record_arms_in_order": ["content_adaptive_dual_branch_v2_clean_v1", "primary_null__content_adaptive_dual_branch_v2_clean_v1"],
        "unit_transaction_record_count": 2,
        "fixed_records": 16,
        "record_score_prefixes_in_order": ["lf", "hf", "joint"],
        "score_labels_per_prefix": "registered_then_wrong_00_through_wrong_15",
        "flat_score_field_rule": "prefix_double_underscore_label_within_StageARecord_v1",
        "failure_units_remain_in_denominator": True,
        "replacement_units_allowed": False,
        "outcome_requires_complete_rc0": True,
    }:
        raise ValueError("content-adaptive transaction denominator differs")
    decision = _mapping(config, "decision_rule")
    expected_gates = {
        "fixed_units": 8,
        "lf_gate_a_registered_top_rank_among_17_min_units": 7,
        "lf_gate_b_joint_registered_gt_primary_null_registered_min_units": 7,
        "hf_gate_a_registered_top_rank_among_17_min_units": 7,
        "hf_gate_b_joint_registered_gt_primary_null_registered_min_units": 7,
        "joint_gate_a_registered_top_rank_among_17_min_units": 7,
        "joint_gate_b_joint_registered_gt_primary_null_registered_min_units": 7,
        "strict_comparison_ties_fail": True,
        "combined_budget_pass_units": 8,
        "both_nonzero_branches_pass_units": 8,
        "baseline_differenced_probe_response_pass_units": 8,
        "probe_evaluation_count_64_pass_units": 8,
        "public_branch_share_valid_pass_units": 8,
        "paired_rgb_psnr_min_db": 30.0,
        "paired_rgb_psnr_pass_units": 8,
        "formal_fpr_claim": False,
    }
    if decision != expected_gates:
        raise ValueError("content-adaptive strict gate choices differ")
    if config.get("limitations") != [
        "cpu_and_fake_tests_are_engineering_only",
        "no_mechanism_or_scientific_completion_without_real_gpu_complete_rc0",
        "no_calibrated_threshold_or_fixed_fpr_claim",
        "clean_only_no_attack_or_geometry_claim",
    ]:
        raise ValueError("content-adaptive limitations differ")


def load_content_adaptive_dual_branch_v2_clean_protocol(
    config_path: str | Path,
    roster_path: str | Path,
) -> ContentChainProtocol:
    """Load and bind the exact 8-unit, 16-record clean protocol."""

    config_path = Path(config_path)
    roster_path = Path(roster_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("content-adaptive config must be an object")
    _validate_config(config)
    roster = _load_roster(roster_path)
    _validate_global_disjoint(roster_path, roster)
    canonical = json.dumps(
        {"config": config, "roster": [asdict(unit) for unit in roster]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return ContentChainProtocol(
        protocol_id=config["protocol_id"],
        config=_freeze(config),
        roster=roster,
        protocol_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
