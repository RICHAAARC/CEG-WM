from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from cegwm.protocol.stage_a import (
    load_hf_v2_confirmation_protocol,
    load_lf_a3_selection_protocol,
    load_stage_a_protocol,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "stage_a" / "stage_a_v1.json"
_SELECTION = _ROOT / "configs" / "stage_a" / "candidate_selection.jsonl"
_CONFIRMATION = _ROOT / "configs" / "stage_a" / "untouched_confirmation.jsonl"
_V2_CONFIG = _ROOT / "configs" / "stage_a" / "stage_a_hf_v2_rankgate.json"
_LF_A3_CONFIG = _ROOT / "configs" / "stage_a" / "stage_a_lf_a3_clean_selection_v1.json"


@pytest.mark.unit
def test_frozen_stage_a_protocol_is_finite_disjoint_and_digestible() -> None:
    protocol = load_stage_a_protocol(_CONFIG, _SELECTION, _CONFIRMATION)

    assert protocol.protocol_id == "cegwm-stage-a-v1"
    assert len(protocol.candidate_selection) == 8
    assert len(protocol.untouched_confirmation) == 8
    assert len(protocol.protocol_digest) == 64
    assert {unit.source_id for unit in protocol.candidate_selection}.isdisjoint(
        unit.source_id for unit in protocol.untouched_confirmation
    )
    assert len(protocol.config["lf_candidates"]) == 2
    assert isinstance(protocol.config["detection_access"], MappingProxyType)
    assert protocol.config["execution_flow"]["failure_units_remain_in_denominator"] is True
    assert protocol.config["execution_flow"]["replacement_units_allowed"] is False
    assert protocol.config["generation_runtime"]["public_asset_rule"] == (
        "protocol_model_id_default_hub_resolution_without_revision_or_weight_digest"
    )


@pytest.mark.unit
def test_detection_access_is_exact_and_private_state_is_fail_closed(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["detection_access"]["allowed_inputs"].append("prompt")
    modified = tmp_path / "invalid.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="allowed_inputs must be exactly"):
        load_stage_a_protocol(modified, _SELECTION, _CONFIRMATION)


@pytest.mark.unit
def test_protocol_stops_if_failures_are_removed_from_fixed_denominator(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["execution_flow"]["failure_units_remain_in_denominator"] = False
    modified = tmp_path / "invalid.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed denominator"):
        load_stage_a_protocol(modified, _SELECTION, _CONFIRMATION)


@pytest.mark.unit
def test_protocol_rejects_overlapping_lf_hf_bands(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["bands"]["lf_radius"] = [0.04, 0.7]
    modified = tmp_path / "invalid.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_stage_a_protocol(modified, _SELECTION, _CONFIRMATION)


@pytest.mark.unit
def test_protocol_rejects_candidate_outside_frozen_lf_band(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["lf_candidates"][0]["radial_subband"] = [0.01, 0.14]
    modified = tmp_path / "invalid.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the frozen LF band"):
        load_stage_a_protocol(modified, _SELECTION, _CONFIRMATION)


@pytest.mark.unit
def test_hf_v2_protocol_freezes_rank_gate_on_untouched_confirmation() -> None:
    protocol = load_hf_v2_confirmation_protocol(_V2_CONFIG, _CONFIRMATION)

    assert protocol.protocol_id == "cegwm-stage-a-hf-v2-rankgate"
    assert protocol.candidate_selection == ()
    assert [unit.unit_id for unit in protocol.untouched_confirmation] == [
        f"confirmation-{index:04d}" for index in range(1, 9)
    ]
    assert len(protocol.protocol_digest) == 64
    assert protocol.config["development_evidence"]["candidate_selection_role"] == (
        "protocol_v1_pilot_only_not_v2_confirmation_evidence"
    )
    candidate = protocol.config["hf_confirmation_candidate"]
    assert candidate["evaluated_candidate_id"] == "hf_tail_rademacher_v1_rankgate_v2"
    assert candidate["carrier_method_id"] == "hf_tail_rademacher_v1"
    assert candidate["injection_step_index_zero_based"] == 18
    rule = protocol.config["confirmation_rule"]
    assert rule["registered_top_rank_among_17_min_units"] == 7
    assert rule["paired_hf_registered_gt_primary_null_registered_min_units"] == 7
    assert rule["median_correct_minus_wrong_key_max_role"] == (
        "reported_effect_size_only_no_pass_threshold"
    )
    assert rule["primary_null_role"] == "reported_separately_no_pass_cutoff"
    assert protocol.config["budget"]["quality_evidence"]["lpips_alex"] == "not_evaluated"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("confirmation_rule", "registered_top_rank_among_17_min_units"), 6, "rank-gate"),
        (("confirmation_rule", "median_correct_minus_wrong_key_max_min"), 0.08, "unregistered"),
        (("confirmation_rule", "primary_null_abs_score_p95_max"), 0.2, "unregistered"),
        (("hf_confirmation_candidate", "carrier_method_id"), "changed", "decision identity"),
        (("hf_confirmation_candidate", "injection_step_index_zero_based"), 17, "decision identity"),
        (("keying", "wrong_key_derivation_domain"), "changed", "key and control"),
        (("budget", "total_relative_l2"), 0.011, "0.012 budget"),
    ],
)
def test_hf_v2_protocol_rejects_gate_or_preserved_method_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    payload = json.loads(_V2_CONFIG.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = value
    modified = tmp_path / "invalid-v2.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_hf_v2_confirmation_protocol(modified, _CONFIRMATION)


@pytest.mark.unit
def test_lf_a3_protocol_freezes_selection_roster_partition_arms_and_ranking() -> None:
    protocol = load_lf_a3_selection_protocol(_LF_A3_CONFIG, _SELECTION)

    assert protocol.protocol_id == "cegwm-stage-a-lf-a3-clean-selection-v1"
    assert [unit.unit_id for unit in protocol.candidate_selection] == [
        f"selection-{index:04d}" for index in range(1, 9)
    ]
    assert protocol.untouched_confirmation == ()
    assert len(protocol.protocol_digest) == 64
    assert protocol.config["record_arms_in_exact_unit_order"] == (
        "lf_core_rademacher_v1",
        "primary_null__lf_core_rademacher_v1",
        "lf_shell_rademacher_v1",
        "primary_null__lf_shell_rademacher_v1",
    )
    rule = protocol.config["selection_rule"]
    assert rule["registered_top_rank_among_17_min_units"] == 7
    assert rule[
        "paired_lf_registered_gt_candidate_scored_primary_null_registered_min_units"
    ] == 7
    assert rule["both_eligible_ranking"][-1] == "candidate_id_asc"
    assert rule["absolute_margin_role"] == "report_and_ranking_only_no_pass_threshold"
    assert protocol.config["execution_flow"]["fixed_records"] == 32


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("bands", "core_upper_bound"), "inclusive", "exact disjoint"),
        (("selection_rule", "registered_top_rank_among_17_min_units"), 6, "scale-free"),
        (("selection_rule", "absolute_margin_min"), 0.08, "scale-free"),
        (("execution_flow", "fixed_records"), 16, "8-unit/32-record"),
        (("lf_candidates", "replace"), [], "frozen core and shell"),
    ],
)
def test_lf_a3_protocol_rejects_band_gate_candidate_or_denominator_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    payload = json.loads(_LF_A3_CONFIG.read_text(encoding="utf-8"))
    if path[0] == "lf_candidates":
        payload["lf_candidates"] = value
    else:
        payload[path[0]][path[1]] = value
    modified = tmp_path / "invalid-lf-a3.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_lf_a3_selection_protocol(modified, _SELECTION)
