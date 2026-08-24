from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.protocol.content_chain_v5 import (
    CONTENT_V5_ARMS,
    CONTENT_V5_ARTIFACT_CONTRACT_ID,
    CONTENT_V5_DECISION_RULE_ID,
    CONTENT_V5_EVALUATED_CANDIDATE_ID,
    CONTENT_V5_METHOD_ID,
    CONTENT_V5_PROTOCOL_DIGEST,
    CONTENT_V5_PROTOCOL_ID,
    CONTENT_V5_STATE_SCHEMA_ID,
    evaluate_content_v5_decision,
    load_content_v5_clean_protocol,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _ROOT / "configs" / "content_chain"
_CONFIG = _CONFIG_ROOT / "content_v5_lf_or_hf_clean_v1.json"
_PRIMARY = _CONFIG_ROOT / "content_v5_primary_evaluation_v1.jsonl"
_CONTROL = _CONFIG_ROOT / "content_adaptive_dual_branch_v2_clean.jsonl"
_V4_CONFIG = _CONFIG_ROOT / "content_v4_clean_v1.json"


def _protocol():
    return load_content_v5_clean_protocol(_CONFIG, _PRIMARY, _CONTROL)


def _scores(registered: float, wrong: float, joint: float) -> dict[str, float]:
    return {
        "lf__registered": registered,
        **{f"lf__wrong_{index:02d}": wrong for index in range(16)},
        "hf__registered": registered,
        **{f"hf__wrong_{index:02d}": wrong for index in range(16)},
        "joint__registered": joint,
        **{f"joint__wrong_{index:02d}": -joint for index in range(16)},
    }


def _transaction(
    *,
    unit_id: str = "unit-0",
    lf: tuple[float, float, float],
    hf: tuple[float, float, float],
    joint: float = 0.0,
) -> list[dict[str, object]]:
    candidate = _scores(0.0, 0.0, joint)
    candidate["lf__registered"], lf_wrong, lf_null = lf
    candidate["hf__registered"], hf_wrong, hf_null = hf
    candidate.update({f"lf__wrong_{index:02d}": lf_wrong for index in range(16)})
    candidate.update({f"hf__wrong_{index:02d}": hf_wrong for index in range(16)})
    primary_null = _scores(0.0, 0.0, -joint)
    primary_null["lf__registered"] = lf_null
    primary_null["hf__registered"] = hf_null
    return [
        {"unit_id": unit_id, "arm": CONTENT_V5_ARMS[0], "scores": candidate},
        {"unit_id": unit_id, "arm": CONTENT_V5_ARMS[1], "scores": primary_null},
    ]


@pytest.mark.unit
def test_content_v5_protocol_freezes_paired_manifest_and_method_identities() -> None:
    paired = _protocol()
    primary = paired.cohort_protocol("primary_1")
    control = paired.cohort_protocol("control_1")
    config = paired.config
    assert config["protocol_id"] == CONTENT_V5_PROTOCOL_ID
    assert primary.protocol_digest == control.protocol_digest == CONTENT_V5_PROTOCOL_DIGEST
    assert CONTENT_V5_PROTOCOL_DIGEST == (
        "c5a0c4bf7d6d3521ae233756ea07753dd002d842662b50f82a86de6a0f96c204"
    )
    assert config["method_identities"]["content_method_id"] == CONTENT_V5_METHOD_ID
    assert (
        config["method_identities"]["evaluated_candidate_id"]
        == CONTENT_V5_EVALUATED_CANDIDATE_ID
    )
    assert config["decision_rule"]["decision_rule_id"] == CONTENT_V5_DECISION_RULE_ID
    flow = config["execution_flow"]
    assert tuple(item["cohort_id"] for item in flow["cohorts_in_order"]) == (
        "control_1", "primary_1"
    )
    assert tuple(item["cohort_role"] for item in flow["cohorts_in_order"]) == (
        "reference_cohort", "primary_evaluation"
    )
    assert flow["single_top_level_invocation"] is True
    assert flow["cohort_selection_argument_allowed"] is False
    assert flow["umbrella_state_schema_id"] == CONTENT_V5_STATE_SCHEMA_ID
    assert flow["umbrella_artifact_contract_id"] == CONTENT_V5_ARTIFACT_CONTRACT_ID
    assert flow["single_local_and_artifact_run_root"] is True
    assert flow["single_terminal_zip_and_sha_pair"] is True
    assert flow["cohort_execution_order_unconditional"] is True
    assert flow["fixed_units_per_cohort"] == 8
    assert flow["fixed_records_per_cohort"] == 16
    assert flow["cohort_denominators_independent"] is True
    assert flow["pooling_to_16_units_forbidden"] is True
    assert flow["pass_transfer_forbidden"] is True
    assert flow["conditional_omission_forbidden"] is True
    assert flow["cross_cohort_conjunction"] is False
    assert flow["both_cohort_results_always_reported"] is True
    assert flow["fresh_execution_required_on_final_v5_exact"] is True
    assert flow["unit_failures_recorded_and_execution_continues"] is True
    assert flow["fatal_interruption_checkpoint"] == "last_complete_whole_unit_only"
    assert flow["checkpoint_purpose"] == "audit_only_never_resume"
    assert flow["existing_local_or_artifact_run_root_rejected"] is True
    assert flow["automatic_resume_allowed"] is False
    assert flow["manual_resume_allowed"] is False
    assert flow["automatic_retry_allowed"] is False
    assert tuple(flow["reuse_from_prior_content_versions_forbidden"]) == (
        "images", "scores", "records", "results", "checkpoints", "artifacts"
    )
    assert config["decision_rule"]["formal_fpr_claim"] is False
    serialized = _CONFIG.read_text(encoding="utf-8")
    assert "protocol_digest" not in serialized
    assert "formal_roster_sha256" not in serialized


@pytest.mark.unit
def test_content_v5_primary_manifest_exact_bytes_sha_blob_and_order() -> None:
    payload = _PRIMARY.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "5303a0284e36d2e6e159526c7ba61a7106fb3db72de35f0ada98fcfd5da2ec2c"
    )
    header = f"blob {len(payload)}\0".encode("ascii")
    assert hashlib.sha1(header + payload).hexdigest() == (
        "1b134b998820427b53be0d82ba61cab1b4a8ad79"
    )
    assert payload.endswith(b"\n") and b"\r" not in payload
    lines = payload.decode("utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    assert len(entries) == 8
    assert [entry["unit_id"] for entry in entries] == [
        f"content-v5-primary-{index:04d}" for index in range(1, 9)
    ]
    assert [entry["source_id"] for entry in entries] == [
        f"content-v5-primary-prompt-{index}" for index in range(9201, 9209)
    ]
    assert [entry["seed"] for entry in entries] == [
        2026082401, 2026083410, 2026084419, 2026085428,
        2026086437, 2026087446, 2026088455, 2026089464,
    ]
    assert lines == [
        json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
        for entry in entries
    ]


@pytest.mark.unit
def test_content_v5_reference_manifest_exact_identity_and_cohorts_are_disjoint() -> None:
    payload = _CONTROL.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
    )
    header = f"blob {len(payload)}\0".encode("ascii")
    assert hashlib.sha1(header + payload).hexdigest() == (
        "7e0415ca14a3c37475ec796d4985698afbde4f89"
    )
    paired = _protocol()
    primary = paired.cohort_protocol("primary_1")
    control = paired.cohort_protocol("control_1")
    assert len(primary.roster) == len(control.roster) == 8
    assert {unit.unit_id for unit in primary.roster}.isdisjoint(
        unit.unit_id for unit in control.roster
    )
    assert {unit.source_id for unit in primary.roster}.isdisjoint(
        unit.source_id for unit in control.roster
    )


@pytest.mark.unit
def test_content_v5_preserves_v4_runtime_embedding_scorers_keying_and_mechanics() -> None:
    v5 = json.loads(_CONFIG.read_text(encoding="utf-8"))
    v4 = json.loads(_V4_CONFIG.read_text(encoding="utf-8"))
    for section in (
        "generation_runtime", "content_analysis", "lf_detection_operator", "budget",
        "aggregate_measurement", "detection_access", "keying", "limitations",
    ):
        assert v5[section] == v4[section]
    preserved_method_fields = set(v4["method_identities"]) - {
        "content_method_id", "evaluated_candidate_id"
    }
    for field in preserved_method_fields:
        assert v5["method_identities"][field] == v4["method_identities"][field]
    for field in (
        "combined_budget_pass_units", "both_nonzero_branches_pass_units",
        "baseline_differenced_probe_response_pass_units",
        "probe_evaluation_count_64_pass_units", "public_branch_share_valid_pass_units",
        "paired_rgb_psnr_min_db", "paired_rgb_psnr_pass_units", "formal_fpr_claim",
    ):
        assert v5["decision_rule"][field] == v4["decision_rule"][field]
    assert v5["generation_runtime"]["injection_step_index_zero_based"] == 18
    assert v5["content_analysis"]["probe_evaluations_per_unit"] == 64
    assert v5["budget"]["combined_total_relative_l2"] == 0.012


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lf_pass", "hf_pass", "expected"),
    ((False, False, 0), (True, False, 1), (False, True, 1), (True, True, 1)),
)
def test_content_v5_per_unit_or_truth_table(
    lf_pass: bool,
    hf_pass: bool,
    expected: int,
) -> None:
    passing = (0.8, 0.2, 0.2)
    failing = (0.2, 0.8, 0.8)
    evidence = evaluate_content_v5_decision(
        _transaction(lf=passing if lf_pass else failing, hf=passing if hf_pass else failing),
        CONTENT_V5_ARMS,
    )
    assert evidence["branchwise_or"]["gate_a_pass_units"] == expected
    assert evidence["branchwise_or"]["gate_b_pass_units"] == expected
    assert evidence["branches"]["lf"]["gate_a_pass_units"] == int(lf_pass)
    assert evidence["branches"]["hf"]["gate_a_pass_units"] == int(hf_pass)
    assert evidence["branches"]["lf"]["diagnostic_only"] is True
    assert evidence["branches"]["hf"]["diagnostic_only"] is True


@pytest.mark.unit
def test_content_v5_strict_ties_fail_within_both_branches() -> None:
    evidence = evaluate_content_v5_decision(
        _transaction(lf=(0.4, 0.4, 0.4), hf=(-0.2, -0.2, -0.2)),
        CONTENT_V5_ARMS,
    )
    assert evidence["branchwise_or"]["gate_a_pass_units"] == 0
    assert evidence["branchwise_or"]["gate_b_pass_units"] == 0


@pytest.mark.unit
def test_content_v5_never_substitutes_raw_max_and_never_consumes_joint_min() -> None:
    records = _transaction(
        lf=(0.9, 1.0, 1.0),
        hf=(0.2, 0.1, 0.1),
        joint=-0.95,
    )
    first = evaluate_content_v5_decision(records, CONTENT_V5_ARMS)
    records[0]["scores"].update(  # type: ignore[union-attr]
        {"joint__registered": 0.99, **{f"joint__wrong_{i:02d}": 1.0 for i in range(16)}}
    )
    records[1]["scores"]["joint__registered"] = 1.0  # type: ignore[index]
    second = evaluate_content_v5_decision(records, CONTENT_V5_ARMS)
    assert first == second
    assert first["branches"]["lf"]["gate_a_pass_units"] == 0
    assert first["branches"]["hf"]["gate_a_pass_units"] == 1
    assert first["branchwise_or"]["gate_a_pass_units"] == 1


@pytest.mark.unit
def test_content_v5_definition_rejects_identity_drift(tmp_path: Path) -> None:
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    config["decision_rule"]["branchwise_or_gate_a_min_units"] = 6
    modified = tmp_path / "modified.json"
    modified.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="decision rule"):
        load_content_v5_clean_protocol(
            modified, _PRIMARY, _CONTROL
        )
