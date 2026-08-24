"""Frozen paired-cohort protocol for Content V5 branchwise-OR decisions."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from cegwm.protocol.content_chain_v2 import (
    _CONTENT_ANALYSIS,
    _freeze,
    ContentChainProtocol,
    ContentChainUnit,
)
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
CONTENT_V5_EXECUTION_SCOPE_ID = (
    "content_v5_paired_primary_and_reference_cohort_evaluation_v1"
)
CONTENT_V5_PRIMARY_EXECUTION_SCOPE_ID = (
    "content_v5_primary_1_whitened_lf_adaptive_hf_branchwise_or_evaluation_v1"
)
CONTENT_V5_CONTROL_EXECUTION_SCOPE_ID = (
    "content_v5_control_1_whitened_lf_adaptive_hf_branchwise_or_"
    "reference_evaluation_v1"
)
CONTENT_V5_PRIMARY_RUN_PREFIX = "content-v5-primary-1"
CONTENT_V5_CONTROL_RUN_PREFIX = "content-v5-control-1"
CONTENT_V5_PROTOCOL_DIGEST = (
    "7d8f1ebef662a45dfd760261efbc81b733eed97bfea7bfd9fde72fa15f025314"
)
CONTENT_V5_ARMS = (
    CONTENT_V5_EVALUATED_CANDIDATE_ID,
    f"primary_null__{CONTENT_V5_EVALUATED_CANDIDATE_ID}",
)

_METHOD_IDENTITIES = {
    **_V4_METHOD_IDENTITIES,
    "content_method_id": CONTENT_V5_METHOD_ID,
    "evaluated_candidate_id": CONTENT_V5_EVALUATED_CANDIDATE_ID,
}
_UNIT_FIELDS = {"unit_id", "split", "source_id", "prompt", "seed", "height", "width"}
_COHORTS_IN_ORDER = [
    {
        "cohort_id": "primary_1",
        "cohort_role": "primary_evaluation",
        "manifest_path": "configs/content_chain/content_v5_primary_evaluation_v1.jsonl",
        "manifest_sha256": (
            "5303a0284e36d2e6e159526c7ba61a7106fb3db72de35f0ada98fcfd5da2ec2c"
        ),
        "manifest_git_blob": "1b134b998820427b53be0d82ba61cab1b4a8ad79",
        "split": "content_v5_primary_evaluation_v1",
        "run_prefix": CONTENT_V5_PRIMARY_RUN_PREFIX,
        "execution_scope_id": CONTENT_V5_PRIMARY_EXECUTION_SCOPE_ID,
    },
    {
        "cohort_id": "control_1",
        "cohort_role": "reference_cohort",
        "manifest_path": (
            "configs/content_chain/content_adaptive_dual_branch_v2_clean.jsonl"
        ),
        "manifest_sha256": (
            "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
        ),
        "manifest_git_blob": "7e0415ca14a3c37475ec796d4985698afbde4f89",
        "split": "content_adaptive_dual_branch_v2_clean_v1",
        "run_prefix": CONTENT_V5_CONTROL_RUN_PREFIX,
        "execution_scope_id": CONTENT_V5_CONTROL_EXECUTION_SCOPE_ID,
    },
]
_EXECUTION_FLOW = {
    "cohorts_in_order": _COHORTS_IN_ORDER,
    "cohort_selection_required": True,
    "fixed_units_per_cohort": 8,
    "record_arms_in_order": list(CONTENT_V5_ARMS),
    "unit_transaction_record_count": 2,
    "fixed_records_per_cohort": 16,
    "cohort_denominators_independent": True,
    "pooling_to_16_units_forbidden": True,
    "pass_transfer_forbidden": True,
    "conditional_omission_forbidden": True,
    "cross_cohort_conjunction": False,
    "both_cohort_results_always_reported": True,
    "fresh_execution_required_on_final_v5_exact": True,
    "reuse_from_prior_content_versions_forbidden": [
        "images", "scores", "records", "results", "checkpoints", "artifacts",
    ],
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
    if config.get("scientific_status") != (
        "not_evaluated_until_each_cohort_complete_real_gpu_rc0"
    ):
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


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _load_bound_roster(path: Path, cohort: Mapping[str, Any]) -> tuple[ContentChainUnit, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != cohort["manifest_sha256"]:
        raise ValueError(f"Content V5 {cohort['cohort_id']} manifest SHA differs")
    if _git_blob_sha1(payload) != cohort["manifest_git_blob"]:
        raise ValueError(f"Content V5 {cohort['cohort_id']} manifest Git blob differs")
    units: list[ContentChainUnit] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"{path.name}:{line_number} cannot be blank")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{line_number} is invalid JSON") from error
        if not isinstance(item, dict) or set(item) != _UNIT_FIELDS:
            raise ValueError(f"{path.name}:{line_number} has unexpected fields")
        if any(
            not isinstance(item[name], str) or not item[name].strip()
            for name in ("unit_id", "split", "source_id", "prompt")
        ):
            raise ValueError(f"{path.name}:{line_number} has empty identity text")
        if item["split"] != cohort["split"]:
            raise ValueError(f"{path.name}:{line_number} has the wrong split")
        if any(
            not isinstance(item[name], int) or isinstance(item[name], bool)
            for name in ("seed", "height", "width")
        ):
            raise ValueError(f"{path.name}:{line_number} has non-integer runtime values")
        if item["seed"] < 0 or item["height"] < 256 or item["width"] < 256:
            raise ValueError(f"{path.name}:{line_number} has invalid runtime values")
        units.append(ContentChainUnit(**item))
    if len(units) != 8:
        raise ValueError(f"Content V5 {cohort['cohort_id']} must contain exactly 8 units")
    if len({unit.unit_id for unit in units}) != 8:
        raise ValueError(f"Content V5 {cohort['cohort_id']} unit identities collide")
    if len({unit.source_id for unit in units}) != 8:
        raise ValueError(f"Content V5 {cohort['cohort_id']} source identities collide")
    return tuple(units)


def load_content_v5_clean_protocol(
    config_path: str | Path,
    primary_roster_path: str | Path,
    control_roster_path: str | Path,
    *,
    cohort_id: str,
) -> ContentChainProtocol:
    """Bind both exact rosters and select one independent eight-unit denominator."""

    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Content V5 config must be an object")
    _validate_config(config)
    repo_root = config_path.parents[2]
    received_paths = {
        "primary_1": Path(primary_roster_path).resolve(),
        "control_1": Path(control_roster_path).resolve(),
    }
    cohorts: dict[str, tuple[ContentChainUnit, ...]] = {}
    ordered_cohorts: list[dict[str, Any]] = []
    for cohort in _COHORTS_IN_ORDER:
        selected_id = cohort["cohort_id"]
        expected_path = (repo_root / cohort["manifest_path"]).resolve()
        if received_paths[selected_id] != expected_path:
            raise ValueError(f"Content V5 {selected_id} manifest path differs")
        roster = _load_bound_roster(expected_path, cohort)
        cohorts[selected_id] = roster
        ordered_cohorts.append({
            "cohort_id": selected_id,
            "cohort_role": cohort["cohort_role"],
            "roster": [asdict(unit) for unit in roster],
        })
    primary = cohorts["primary_1"]
    control = cohorts["control_1"]
    if {unit.unit_id for unit in primary} & {unit.unit_id for unit in control}:
        raise ValueError("Content V5 cohort unit identities collide")
    if {unit.source_id for unit in primary} & {unit.source_id for unit in control}:
        raise ValueError("Content V5 cohort source identities collide")
    if cohort_id not in cohorts:
        raise ValueError("Content V5 cohort selection differs")
    canonical = json.dumps(
        {"config": config, "ordered_cohorts": ordered_cohorts},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    protocol_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if protocol_digest != CONTENT_V5_PROTOCOL_DIGEST:
        raise ValueError("Content V5 paired canonical protocol digest differs")
    return ContentChainProtocol(
        protocol_id=config["protocol_id"],
        config=_freeze(config),
        roster=cohorts[cohort_id],
        protocol_digest=protocol_digest,
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
    "CONTENT_V5_CONTROL_EXECUTION_SCOPE_ID",
    "CONTENT_V5_CONTROL_RUN_PREFIX",
    "CONTENT_V5_EXECUTION_SCOPE_ID",
    "CONTENT_V5_METHOD_ID",
    "CONTENT_V5_PRIMARY_EXECUTION_SCOPE_ID",
    "CONTENT_V5_PRIMARY_RUN_PREFIX",
    "CONTENT_V5_PROTOCOL_DIGEST",
    "CONTENT_V5_PROTOCOL_ID",
    "CONTENT_V5_RECORD_CONTRACT_ID",
    "CONTENT_V5_RUN_PREFIX",
    "CONTENT_V5_STATE_SCHEMA_ID",
    "ContentChainProtocol",
    "ContentChainUnit",
    "evaluate_content_v5_decision",
    "load_content_v5_clean_protocol",
]
