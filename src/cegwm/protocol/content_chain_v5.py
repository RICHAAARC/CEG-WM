"""Method-only definition for Content V5 branchwise-OR decisions.

No roster, manifest digest, canonical protocol digest, or deterministic run identity is
defined here. Formal execution remains unavailable until the user freezes a new,
disjoint manifest binding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from cegwm.protocol.content_chain_v2 import _CONTENT_ANALYSIS, _freeze
from cegwm.protocol.content_chain_v3 import (
    _AGGREGATE_MEASUREMENT as _V3_AGGREGATE_MEASUREMENT,
)
from cegwm.protocol.content_chain_v4 import (
    _BUDGET as _V4_BUDGET,
    _DETECTION_ACCESS as _V4_DETECTION_ACCESS,
    _GENERATION_RUNTIME as _V4_GENERATION_RUNTIME,
    _LF_DETECTION_OPERATOR as _V4_LF_DETECTION_OPERATOR,
    _LIMITATIONS as _V4_LIMITATIONS,
    _METHOD_IDENTITIES as _V4_METHOD_IDENTITIES,
)

CONTENT_V5_METHOD_ID = (
    "content_v5_clean_null_whitened_lf_adaptive_hf_branchwise_or_v1"
)
CONTENT_V5_EVALUATED_CANDIDATE_ID = (
    "content_v5_clean_null_whitened_lf_adaptive_hf_branchwise_or_gate_v1"
)
CONTENT_V5_DECISION_RULE_ID = "content_v5_per_unit_lf_or_hf_strict_gates_v1"
CONTENT_V5_PROTOCOL_ID = (
    "cegwm-stage-a-content-v5-whitened-lf-adaptive-hf-branchwise-or-clean-v1"
)
CONTENT_V5_RECORD_CONTRACT_ID = (
    "content_v5_whitened_lf_adaptive_hf_branchwise_or_record_v1"
)
CONTENT_V5_STATE_SCHEMA_ID = "content_v5_resumable_state_v1"
CONTENT_V5_RUN_PREFIX = "content-v5"
CONTENT_V5_EXECUTION_SCOPE_ID = "content_v5_method_only_no_formal_manifest_binding_v1"
CONTENT_V5_ARMS = (
    CONTENT_V5_EVALUATED_CANDIDATE_ID,
    f"primary_null__{CONTENT_V5_EVALUATED_CANDIDATE_ID}",
)

_METHOD_IDENTITIES = {
    **_V4_METHOD_IDENTITIES,
    "content_method_id": CONTENT_V5_METHOD_ID,
    "evaluated_candidate_id": CONTENT_V5_EVALUATED_CANDIDATE_ID,
}
_EXECUTION_FLOW = {
    "formal_execution_status": "blocked_pending_user_frozen_disjoint_manifest_binding",
    "approved_disjoint_manifest_binding_required": True,
    "approved_disjoint_manifest_binding_present": False,
    "fixed_units": 8,
    "record_arms_in_order": list(CONTENT_V5_ARMS),
    "unit_transaction_record_count": 2,
    "fixed_records": 16,
    "record_score_prefixes_in_order": ["lf", "hf", "joint"],
    "joint_score_prefix_status": "diagnostic_only_never_consumed_by_v5_decision",
    "score_labels_per_prefix": "registered_then_wrong_00_through_wrong_15",
    "flat_score_field_rule": (
        "prefix_double_underscore_label_within_"
        "content_v5_whitened_lf_adaptive_hf_branchwise_or_record_v1"
    ),
    "record_contract_id": CONTENT_V5_RECORD_CONTRACT_ID,
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
    "decision_rule_id": CONTENT_V5_DECISION_RULE_ID,
    "fixed_units": 8,
    "per_unit_gate_a": (
        "lf_registered_gt_max_16_lf_wrong_OR_"
        "hf_registered_gt_max_16_hf_wrong"
    ),
    "per_unit_gate_b": (
        "lf_registered_gt_same_unit_lf_primary_null_registered_OR_"
        "hf_registered_gt_same_unit_hf_primary_null_registered"
    ),
    "branch_score_combination": "none_boolean_or_after_within_branch_comparisons",
    "joint_min_score_status": "diagnostic_only_unconsumed_by_decision",
    "individual_lf_hf_counts_status": "diagnostic_only_not_required_conjunctions",
    "branchwise_or_gate_a_min_units": 7,
    "branchwise_or_gate_b_min_units": 7,
    "strict_comparison_ties_fail_within_each_branch": True,
    "combined_budget_pass_units": 8,
    "both_nonzero_branches_pass_units": 8,
    "baseline_differenced_probe_response_pass_units": 8,
    "probe_evaluation_count_64_pass_units": 8,
    "public_branch_share_valid_pass_units": 8,
    "paired_rgb_psnr_min_db": 30.0,
    "paired_rgb_psnr_pass_units": 8,
    "formal_fpr_claim": False,
}


class ContentV5ManifestBindingRequired(RuntimeError):
    """Raised because no user-approved disjoint V5 evidence manifest is frozen."""


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != 1 or config.get("protocol_id") != CONTENT_V5_PROTOCOL_ID:
        raise ValueError("unexpected Content V5 protocol identity")
    if config.get("execution_scope_id") != CONTENT_V5_EXECUTION_SCOPE_ID:
        raise ValueError("unexpected Content V5 execution scope")
    if config.get("scientific_status") != "method_defined_formal_execution_blocked":
        raise ValueError("Content V5 cannot preclaim execution or scientific evidence")
    expected_sections = {
        "generation_runtime": _V4_GENERATION_RUNTIME,
        "content_analysis": _CONTENT_ANALYSIS,
        "method_identities": _METHOD_IDENTITIES,
        "lf_detection_operator": _V4_LF_DETECTION_OPERATOR,
        "budget": _V4_BUDGET,
        "aggregate_measurement": _V3_AGGREGATE_MEASUREMENT,
        "detection_access": _V4_DETECTION_ACCESS,
        "keying": {
            "task": "zero_bit_keyed_attribution",
            "normalization": "NFC_UTF8_for_text_exact_bytes_for_binary",
            "prg": "HMAC_SHA256_counter_v1",
            "wrong_key_count": 16,
            "wrong_key_derivation_domain": (
                "stage-a/content-adaptive-v2-external-wrong-key/v1"
            ),
            "primary_null": True,
            "payload_bits": 0,
        },
        "execution_flow": _EXECUTION_FLOW,
        "decision_rule": _DECISION_RULE,
    }
    for name, expected in expected_sections.items():
        if _mapping(config, name) != expected:
            raise ValueError(f"Content V5 {name.replace('_', ' ')} differs")
    if config.get("limitations") != _V4_LIMITATIONS:
        raise ValueError("Content V5 limitations differ")
    if set(config) != {
        "protocol_version", "protocol_id", "execution_scope_id", "scientific_status",
        *expected_sections, "limitations",
    }:
        raise ValueError("Content V5 config fields differ")


def load_content_v5_method_definition(config_path: str | Path) -> Mapping[str, Any]:
    """Load the method definition without fabricating an executable protocol."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Content V5 config must be an object")
    _validate_config(config)
    return _freeze(config)


def require_content_v5_manifest_binding() -> None:
    """Fail closed until a separate user-approved manifest binding exists."""

    raise ContentV5ManifestBindingRequired(
        "Content_V5_requires_a_user_frozen_new_disjoint_manifest_binding"
    )


def evaluate_content_v5_decision(
    records: list[dict[str, Any]],
    arms: tuple[str, str],
) -> dict[str, Any]:
    """Evaluate strict within-branch comparisons, then OR the branch booleans."""

    if arms != CONTENT_V5_ARMS:
        raise ValueError("Content V5 decision arms differ")
    by_unit: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_unit.setdefault(record["unit_id"], {})[record["arm"]] = record
    branch_counts = {
        "lf": {"gate_a_pass_units": 0, "gate_b_pass_units": 0},
        "hf": {"gate_a_pass_units": 0, "gate_b_pass_units": 0},
    }
    gate_a_or = 0
    gate_b_or = 0
    for transaction in by_unit.values():
        candidate_scores = transaction[arms[0]]["scores"]
        null_scores = transaction[arms[1]]["scores"]
        unit_results: dict[str, tuple[bool, bool]] = {}
        for branch in ("lf", "hf"):
            registered = float(candidate_scores[f"{branch}__registered"])
            wrong = [
                float(candidate_scores[f"{branch}__wrong_{index:02d}"])
                for index in range(16)
            ]
            branch_gate_a = registered > max(wrong)
            branch_gate_b = registered > float(
                null_scores[f"{branch}__registered"]
            )
            branch_counts[branch]["gate_a_pass_units"] += int(branch_gate_a)
            branch_counts[branch]["gate_b_pass_units"] += int(branch_gate_b)
            unit_results[branch] = (branch_gate_a, branch_gate_b)
        gate_a_or += int(unit_results["lf"][0] or unit_results["hf"][0])
        gate_b_or += int(unit_results["lf"][1] or unit_results["hf"][1])
    branches = {
        branch: {
            **counts,
            "diagnostic_only": True,
            "strict_ties_fail": True,
        }
        for branch, counts in branch_counts.items()
    }
    branchwise_or = {
        "gate_a_pass_units": gate_a_or,
        "gate_b_pass_units": gate_b_or,
        "gate_a_pass": gate_a_or >= 7,
        "gate_b_pass": gate_b_or >= 7,
        "strict_ties_fail_within_each_branch": True,
    }
    return {
        "branches": branches,
        "branchwise_or": branchwise_or,
        "all_decision_gates_pass": (
            branchwise_or["gate_a_pass"] and branchwise_or["gate_b_pass"]
        ),
    }


__all__ = [
    "CONTENT_V5_ARMS",
    "CONTENT_V5_DECISION_RULE_ID",
    "CONTENT_V5_EVALUATED_CANDIDATE_ID",
    "CONTENT_V5_EXECUTION_SCOPE_ID",
    "CONTENT_V5_METHOD_ID",
    "CONTENT_V5_PROTOCOL_ID",
    "CONTENT_V5_RECORD_CONTRACT_ID",
    "CONTENT_V5_RUN_PREFIX",
    "CONTENT_V5_STATE_SCHEMA_ID",
    "ContentV5ManifestBindingRequired",
    "evaluate_content_v5_decision",
    "load_content_v5_method_definition",
    "require_content_v5_manifest_binding",
]
