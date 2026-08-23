"""Frozen loader for the clean Content V3 evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from cegwm.protocol.content_chain_v2 import (
    _AGGREGATE_MEASUREMENT as _V2_AGGREGATE_MEASUREMENT,
    _CONTENT_ANALYSIS,
    _DETECTION_ACCESS,
    _EXPECTED_ROSTER,
    _freeze,
    _load_roster,
    ContentChainProtocol,
    ContentChainUnit,
)

CONTENT_V3_PROTOCOL_ID = "cegwm-stage-a-content-v3-unweighted-lf-adaptive-hf-clean-v1"
CONTENT_V3_EXECUTION_SCOPE_ID = (
    "content_v3_unweighted_lf_adaptive_hf_engineering_and_stage_a_evaluation_v1"
)
CONTENT_V3_RECORD_CONTRACT_ID = (
    "content_v3_unweighted_lf_adaptive_hf_record_v1"
)
CONTENT_V3_STATE_SCHEMA_ID = "content_v3_resumable_state_v1"
CONTENT_V3_RUN_PREFIX = "content-v3"
CONTENT_V3_EVALUATED_CANDIDATE_ID = (
    "content_v3_unweighted_lf_adaptive_hf_semantic_gate_v1"
)
CONTENT_V3_ARMS = (
    CONTENT_V3_EVALUATED_CANDIDATE_ID,
    f"primary_null__{CONTENT_V3_EVALUATED_CANDIDATE_ID}",
)
CONTENT_V3_ROSTER_SHA256 = (
    "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
)
CONTENT_V3_PROTOCOL_DIGEST = (
    "0ba7e55556892b49a873429a1d76a021a119069e03abf512b2dfd4adb50d1c56"
)
RUNTIME_ASSET_VALIDATION_CONTRACT_ID = (
    "dinov2_small_eager_bit_image_processor_public_size_semantics_v3"
)

_METHOD_IDENTITIES = {
    "content_method_id": "content_v3_unweighted_lf_adaptive_hf_v1",
    "hf_base_carrier_method_id": "hf_tail_rademacher_v1",
    "hf_base_evaluated_candidate_id": "hf_tail_rademacher_v1_rankgate_v2",
    "lf_base_carrier_method_id": "lf_shell_balanced_blocks_v2",
    "lf_base_evaluated_candidate_id": "lf_shell_balanced_blocks_v2_blocknorm_median_v1",
    "hf_adaptive_embedding_transform_id": "hf_content_tiles_semantic_gate_texture_two_scale_response_consistency_sensitivity_v1",
    "lf_embedding_transform_id": "lf_unweighted_balanced_blocks_content_allocated_amplitude_v3",
    "lf_embedding_direction_rule": "delta_LF_equals_A_LF_of_content_times_normalize_reconstruct_lf_carrier_no_lf_tile_weight_spatial_transform",
    "hf_embedding_direction_rule": "delta_HF_equals_A_HF_of_content_times_normalize_w_HF_of_content_times_reconstruct_hf_carrier",
    "branch_amplitude_rule": "A_branch_equals_actual_callback_base_l2_times_0.012_times_real_content_branch_share_before_common_projection",
    "combined_budget_projector_id": "dual_branch_actual_dtype_relative_l2_v1",
    "evaluated_candidate_id": CONTENT_V3_EVALUATED_CANDIDATE_ID,
    "base_prg_domain_rule": "base_carrier_ids_only_content_v3_and_joint_ids_never_enter_base_hf_or_lf_prg_domain",
}
_AGGREGATE_MEASUREMENT = {
    **_V2_AGGREGATE_MEASUREMENT,
    "public_branch_share_source": "same_ContentAllocation_values_pass_through_and_control_Content_V3_branch_amplitudes",
}
_EXECUTION_FLOW = {
    "roster_manifest": "content_adaptive_dual_branch_v2_clean.jsonl",
    "formal_roster_sha256": CONTENT_V3_ROSTER_SHA256,
    "split": "content_adaptive_dual_branch_v2_clean_v1",
    "fixed_units": 8,
    "record_arms_in_order": list(CONTENT_V3_ARMS),
    "unit_transaction_record_count": 2,
    "fixed_records": 16,
    "record_score_prefixes_in_order": ["lf", "hf", "joint"],
    "score_labels_per_prefix": "registered_then_wrong_00_through_wrong_15",
    "flat_score_field_rule": "prefix_double_underscore_label_within_content_v3_unweighted_lf_adaptive_hf_record_v1",
    "record_contract_id": CONTENT_V3_RECORD_CONTRACT_ID,
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
_PROVENANCE_EXCLUSIONS = [
    "Content_V2_negative_exact_run_artifacts_and_gate_counts_are_provenance_only_not_Content_V3_evidence",
    "docs/content_runtime_v3_scientific_negative_adjudication.md_is_immutable_and_not_a_Content_V3_result",
]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != 1 or config.get("protocol_id") != CONTENT_V3_PROTOCOL_ID:
        raise ValueError("unexpected Content V3 protocol identity")
    if config.get("execution_scope_id") != CONTENT_V3_EXECUTION_SCOPE_ID:
        raise ValueError("unexpected Content V3 execution scope")
    if config.get("scientific_status") != "not_evaluated_until_complete_real_gpu_rc0":
        raise ValueError("Content V3 protocol cannot preclaim scientific evidence")
    if _mapping(config, "generation_runtime") != {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "inference_steps": 20,
        "injection_step_index_zero_based": 18,
        "generation_rule": "independent_same_seed_generators_for_joint_and_primary_null",
    }:
        raise ValueError("Content V3 generation runtime differs")
    analysis = _mapping(config, "content_analysis")
    if analysis != _CONTENT_ANALYSIS:
        raise ValueError("Content V3 substantive content analysis differs")
    if analysis.get("runtime_asset_validation_contract_id") != RUNTIME_ASSET_VALIDATION_CONTRACT_ID:
        raise ValueError("Content V3 runtime asset contract differs")
    if _mapping(config, "method_identities") != _METHOD_IDENTITIES:
        raise ValueError("Content V3 method identities differ")
    if _mapping(config, "budget") != {
        "combined_total_relative_l2": 0.012,
        "measurement": "actual_dtype_final_minus_actual_dtype_base",
        "single_shared_budget_not_per_branch": True,
        "both_effective_branches_nonzero": True,
    }:
        raise ValueError("Content V3 combined budget differs")
    if _mapping(config, "aggregate_measurement") != _AGGREGATE_MEASUREMENT:
        raise ValueError("Content V3 aggregate measurement differs")
    if _mapping(config, "detection_access") != _DETECTION_ACCESS:
        raise ValueError("Content V3 blind detection access differs")
    if _mapping(config, "keying") != {
        "task": "zero_bit_keyed_attribution",
        "normalization": "NFC_UTF8_for_text_exact_bytes_for_binary",
        "prg": "HMAC_SHA256_counter_v1",
        "wrong_key_count": 16,
        "wrong_key_derivation_domain": "stage-a/content-adaptive-v2-external-wrong-key/v1",
        "primary_null": True,
        "payload_bits": 0,
    }:
        raise ValueError("Content V3 key controls differ")
    if _mapping(config, "execution_flow") != _EXECUTION_FLOW:
        raise ValueError("Content V3 transaction denominator or identity differs")
    if _mapping(config, "decision_rule") != _DECISION_RULE:
        raise ValueError("Content V3 strict science gates differ")
    if config.get("provenance_exclusions") != _PROVENANCE_EXCLUSIONS:
        raise ValueError("Content V3 provenance exclusion differs")
    if config.get("limitations") != [
        "cpu_and_fake_tests_are_engineering_only",
        "no_mechanism_or_scientific_completion_without_real_gpu_complete_rc0",
        "no_calibrated_threshold_or_fixed_fpr_claim",
        "clean_only_no_attack_or_geometry_claim",
    ]:
        raise ValueError("Content V3 limitations differ")


def load_content_v3_clean_protocol(
    config_path: str | Path,
    roster_path: str | Path,
) -> ContentChainProtocol:
    """Load and bind Content V3 to the unchanged ordered eight-unit roster."""

    config_path = Path(config_path)
    roster_path = Path(roster_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Content V3 config must be an object")
    _validate_config(config)
    roster_bytes = roster_path.read_bytes()
    if hashlib.sha256(roster_bytes).hexdigest() != CONTENT_V3_ROSTER_SHA256:
        raise ValueError("Content V3 formal roster SHA differs")
    roster = _load_roster(roster_path)
    received = tuple(
        (
            unit.unit_id, unit.split, unit.source_id, unit.prompt,
            unit.seed, unit.height, unit.width,
        )
        for unit in roster
    )
    if received != _EXPECTED_ROSTER:
        raise ValueError("Content V3 ordered roster differs")
    canonical = json.dumps(
        {"config": config, "roster": [asdict(unit) for unit in roster]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    protocol_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if protocol_digest != CONTENT_V3_PROTOCOL_DIGEST:
        raise ValueError("Content V3 canonical protocol digest differs")
    return ContentChainProtocol(
        protocol_id=config["protocol_id"],
        config=_freeze(config),
        roster=roster,
        protocol_digest=protocol_digest,
    )


__all__ = [
    "CONTENT_V3_ARMS",
    "CONTENT_V3_EVALUATED_CANDIDATE_ID",
    "CONTENT_V3_EXECUTION_SCOPE_ID",
    "CONTENT_V3_PROTOCOL_DIGEST",
    "CONTENT_V3_PROTOCOL_ID",
    "CONTENT_V3_RECORD_CONTRACT_ID",
    "CONTENT_V3_ROSTER_SHA256",
    "CONTENT_V3_RUN_PREFIX",
    "CONTENT_V3_STATE_SCHEMA_ID",
    "ContentChainProtocol",
    "ContentChainUnit",
    "load_content_v3_clean_protocol",
]
