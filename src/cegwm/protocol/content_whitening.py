"""Frozen loader for the clean content-whitening whitened-LF evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from cegwm.method.content_whitening import (
    ASSET_REPO_PATH,
    ASSET_ROLE_ID,
    ASSET_SCHEMA_ID,
    ASSET_SHA256,
    ASSET_SIDECAR_SHA256,
    CONTENT_WHITENING_EVALUATED_CANDIDATE_ID,
    CONTENT_WHITENING_LF_SCORER_ID,
    CONTENT_WHITENING_METHOD_ID,
    OBSERVATION_CONTRACT_ID,
    WHITENING_ORDER,
    WHITENING_SHAPE,
)
from cegwm.protocol.content_adaptive import (
    _CONTENT_ANALYSIS,
    _DETECTION_ACCESS as _ADAPTIVE_DETECTION_ACCESS,
    _EXPECTED_ROSTER,
    _freeze,
    _load_roster,
    ContentChainProtocol,
    ContentChainUnit,
)
from cegwm.protocol.content_unweighted import (
    _AGGREGATE_MEASUREMENT as _UNWEIGHTED_AGGREGATE_MEASUREMENT,
)

CONTENT_WHITENING_PROTOCOL_ID = "cegwm-stage-a-content-v4-whitened-lf-adaptive-hf-clean-v1"
CONTENT_WHITENING_EXECUTION_SCOPE_ID = (
    "content_v4_whitened_lf_adaptive_hf_engineering_and_stage_a_evaluation_v1"
)
CONTENT_WHITENING_RECORD_CONTRACT_ID = "content_v4_whitened_lf_adaptive_hf_record_v1"
CONTENT_WHITENING_STATE_SCHEMA_ID = "content_v4_resumable_state_v1"
CONTENT_WHITENING_RUN_PREFIX = "content-v4"
CONTENT_WHITENING_ARMS = (
    CONTENT_WHITENING_EVALUATED_CANDIDATE_ID,
    f"primary_null__{CONTENT_WHITENING_EVALUATED_CANDIDATE_ID}",
)
CONTENT_WHITENING_ROSTER_SHA256 = (
    "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
)
CONTENT_WHITENING_PROTOCOL_DIGEST = (
    "a9fdf3e5d384976c11bbd542c3248483806473ed1dca91dc5d753ab10ec5beb0"
)

_GENERATION_RUNTIME = {
    "model_id": "stabilityai/stable-diffusion-3.5-medium",
    "inference_steps": 20,
    "injection_step_index_zero_based": 18,
    "generation_rule": "independent_same_seed_generators_for_joint_and_primary_null",
}
_METHOD_IDENTITIES = {
    "content_method_id": CONTENT_WHITENING_METHOD_ID,
    "hf_base_carrier_method_id": "hf_tail_rademacher_v1",
    "hf_base_evaluated_candidate_id": "hf_tail_rademacher_v1_rankgate_v2",
    "lf_base_carrier_method_id": "lf_shell_balanced_blocks_v2",
    "hf_adaptive_embedding_transform_id": "hf_content_tiles_semantic_gate_texture_two_scale_response_consistency_sensitivity_v1",
    "lf_embedding_transform_id": "lf_unweighted_balanced_blocks_content_allocated_amplitude_v3",
    "lf_embedding_direction_rule": "delta_LF_equals_A_LF_of_content_times_normalize_reconstruct_lf_carrier_no_lf_tile_weight_spatial_transform",
    "hf_embedding_direction_rule": "delta_HF_equals_A_HF_of_content_times_normalize_w_HF_of_content_times_reconstruct_hf_carrier",
    "branch_amplitude_rule": "A_branch_equals_actual_callback_base_l2_times_0.012_times_real_content_branch_share_before_common_projection",
    "combined_budget_projector_id": "dual_branch_actual_dtype_relative_l2_v1",
    "evaluated_candidate_id": CONTENT_WHITENING_EVALUATED_CANDIDATE_ID,
    "base_prg_domain_rule": "base_carrier_ids_only_content_v4_and_joint_ids_never_enter_base_hf_or_lf_prg_domain",
}
_LF_DETECTION_OPERATOR = {
    "scorer_id": CONTENT_WHITENING_LF_SCORER_ID,
    "asset_role_id": ASSET_ROLE_ID,
    "asset_schema_id": ASSET_SCHEMA_ID,
    "asset_repo_path": ASSET_REPO_PATH,
    "asset_sha256": ASSET_SHA256,
    "asset_sidecar_sha256": ASSET_SIDECAR_SHA256,
    "observation_contract_id": OBSERVATION_CONTRACT_ID,
    "whitening_shape": list(WHITENING_SHAPE),
    "whitening_order": WHITENING_ORDER,
    "whitening_key_independent": True,
    "carrier_template_id": "lf_shell_balanced_blocks_v2",
    "statistic": "same_W_on_affine_detrended_orthonormal_DCT_II_observation_and_keyed_carrier_six_non_DC_Chebyshev_rings_matched_cosine",
}
_BUDGET = {
    "combined_total_relative_l2": 0.012,
    "measurement": "actual_dtype_final_minus_actual_dtype_base",
    "single_shared_budget_not_per_branch": True,
    "both_effective_branches_nonzero": True,
}
_DETECTION_ACCESS = {
    **_ADAPTIVE_DETECTION_ACCESS,
    "lf_detector": CONTENT_WHITENING_LF_SCORER_ID,
}
_EXECUTION_FLOW = {
    "roster_manifest": "content_adaptive_dual_branch_v2_clean.jsonl",
    "formal_roster_sha256": CONTENT_WHITENING_ROSTER_SHA256,
    "split": "content_adaptive_dual_branch_v2_clean_v1",
    "fixed_units": 8,
    "record_arms_in_order": list(CONTENT_WHITENING_ARMS),
    "unit_transaction_record_count": 2,
    "fixed_records": 16,
    "record_score_prefixes_in_order": ["lf", "hf", "joint"],
    "score_labels_per_prefix": "registered_then_wrong_00_through_wrong_15",
    "flat_score_field_rule": "prefix_double_underscore_label_within_content_v4_whitened_lf_adaptive_hf_record_v1",
    "record_contract_id": CONTENT_WHITENING_RECORD_CONTRACT_ID,
    "record_fields_in_order": [
        "run_id", "unit_id", "source_cluster_id", "arm", "condition",
        "code_revision", "config_digest", "key_public_digest", "status",
        "failure_reason", "scores", "metrics", "record_contract_id",
    ],
    "failure_units_remain_in_denominator": True,
    "replacement_units_allowed": False,
    "retry_units_allowed": False,
    "outcome_requires_complete_rc0": True,
}
_DECISION_RULE = {
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
_LIMITATIONS = [
    "cpu_and_fake_tests_are_engineering_only",
    "no_mechanism_or_scientific_completion_without_real_gpu_complete_rc0",
    "no_calibrated_threshold_or_fixed_fpr_claim",
    "clean_only_no_attack_or_geometry_claim",
]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != 1 or config.get("protocol_id") != CONTENT_WHITENING_PROTOCOL_ID:
        raise ValueError("unexpected content-whitening protocol identity")
    if config.get("execution_scope_id") != CONTENT_WHITENING_EXECUTION_SCOPE_ID:
        raise ValueError("unexpected content-whitening execution scope")
    if config.get("scientific_status") != "not_evaluated_until_complete_real_gpu_rc0":
        raise ValueError("content-whitening protocol cannot preclaim scientific evidence")
    expected_sections = {
        "generation_runtime": _GENERATION_RUNTIME,
        "content_analysis": _CONTENT_ANALYSIS,
        "method_identities": _METHOD_IDENTITIES,
        "lf_detection_operator": _LF_DETECTION_OPERATOR,
        "budget": _BUDGET,
        "aggregate_measurement": _UNWEIGHTED_AGGREGATE_MEASUREMENT,
        "detection_access": _DETECTION_ACCESS,
        "keying": {
            "task": "zero_bit_keyed_attribution",
            "normalization": "NFC_UTF8_for_text_exact_bytes_for_binary",
            "prg": "HMAC_SHA256_counter_v1",
            "wrong_key_count": 16,
            "wrong_key_derivation_domain": "stage-a/content-adaptive-v2-external-wrong-key/v1",
            "primary_null": True,
            "payload_bits": 0,
        },
        "execution_flow": _EXECUTION_FLOW,
        "decision_rule": _DECISION_RULE,
    }
    for name, expected in expected_sections.items():
        if _mapping(config, name) != expected:
            raise ValueError(f"content-whitening {name.replace('_', ' ')} differs")
    if config.get("limitations") != _LIMITATIONS:
        raise ValueError("content-whitening limitations differ")
    if set(config) != {
        "protocol_version", "protocol_id", "execution_scope_id", "scientific_status",
        *expected_sections, "limitations",
    }:
        raise ValueError("content-whitening config fields differ")


def load_content_whitening_protocol(
    config_path: str | Path,
    roster_path: str | Path,
) -> ContentChainProtocol:
    """Load content-whitening and bind it to the unchanged ordered eight-unit roster."""

    config_path = Path(config_path)
    roster_path = Path(roster_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("content-whitening config must be an object")
    _validate_config(config)
    roster_bytes = roster_path.read_bytes()
    if hashlib.sha256(roster_bytes).hexdigest() != CONTENT_WHITENING_ROSTER_SHA256:
        raise ValueError("content-whitening formal roster SHA differs")
    roster = _load_roster(roster_path)
    received = tuple(
        (unit.unit_id, unit.split, unit.source_id, unit.prompt, unit.seed, unit.height, unit.width)
        for unit in roster
    )
    if received != _EXPECTED_ROSTER:
        raise ValueError("content-whitening ordered roster differs")
    canonical = json.dumps(
        {"config": config, "roster": [asdict(unit) for unit in roster]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    protocol_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if protocol_digest != CONTENT_WHITENING_PROTOCOL_DIGEST:
        raise ValueError("content-whitening canonical protocol digest differs")
    return ContentChainProtocol(
        protocol_id=config["protocol_id"],
        config=_freeze(config),
        roster=roster,
        protocol_digest=protocol_digest,
    )


__all__ = [
    "CONTENT_WHITENING_ARMS",
    "CONTENT_WHITENING_EXECUTION_SCOPE_ID",
    "CONTENT_WHITENING_PROTOCOL_DIGEST",
    "CONTENT_WHITENING_PROTOCOL_ID",
    "CONTENT_WHITENING_RECORD_CONTRACT_ID",
    "CONTENT_WHITENING_ROSTER_SHA256",
    "CONTENT_WHITENING_RUN_PREFIX",
    "CONTENT_WHITENING_STATE_SCHEMA_ID",
    "ContentChainProtocol",
    "ContentChainUnit",
    "load_content_whitening_protocol",
]
