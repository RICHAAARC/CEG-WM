from __future__ import annotations

import json
from pathlib import Path

import pytest

from cegwm.protocol.content_chain import load_content_adaptive_dual_branch_clean_protocol

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_clean_v1.json"
_ROSTER = _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_clean.jsonl"


@pytest.mark.unit
def test_protocol_freezes_real_analysis_joint_identity_and_16_record_denominator() -> None:
    protocol = load_content_adaptive_dual_branch_clean_protocol(_CONFIG, _ROSTER)
    assert len(protocol.roster) == 8
    assert len(protocol.protocol_digest) == 64
    analysis = protocol.config["content_analysis"]
    assert set(analysis) == {
        "asset_id", "attention_implementation", "attention_layer", "attention_statistic",
        "tile_grid", "tile_count", "probe_evaluations_per_tile", "probe_evaluations_per_unit",
        "probe_relative_l2", "probe_measurement", "probe_independence", "probe_domain",
        "probe_domain_forbidden_inputs", "signals", "signal_requirement", "export",
    }
    assert analysis["asset_id"] == "facebook/dinov2-small"
    assert analysis["attention_implementation"] == "eager"
    assert analysis["tile_grid"] == (4, 4)
    assert analysis["probe_evaluations_per_unit"] == 32
    identities = protocol.config["method_identities"]
    assert identities["hf_base_carrier_method_id"] == "hf_tail_rademacher_v1"
    assert identities["lf_base_carrier_method_id"] == "lf_shell_balanced_blocks_v2"
    assert identities["evaluated_candidate_id"] == "content_adaptive_dual_branch_clean_v1"
    assert protocol.config["budget"]["single_shared_budget_not_per_branch"] is True
    aggregate = protocol.config["aggregate_measurement"]
    assert aggregate["counterfactual_effect_fields_in_order"] == (
        "semantic_attention_counterfactual_effect",
        "texture_energy_counterfactual_effect",
        "lf_probe_response_counterfactual_effect",
        "hf_probe_response_counterfactual_effect",
    )
    assert aggregate["minimum_counterfactual_effect"] == (
        "min_of_the_four_counterfactual_effect_fields"
    )
    assert aggregate["branch_share_sum_absolute_tolerance"] == 1e-12
    assert aggregate["population_std_formula"] == (
        "sqrt(sum((x_i-mean(x))^2)/8)_ddof_0_each_field_independently"
    )
    assert aggregate["population_std_fixed_roster_units"] == 8
    assert aggregate["population_std_availability"].startswith(
        "complete_finite_identity_valid_RC0_only"
    )
    assert aggregate["blind_score_consumes_aggregate_measurement"] is False
    assert set(aggregate["forbidden_private_exports"]) == {
        "mask", "tile_weights", "attention_map", "latent", "delta", "probe_state",
    }
    assert protocol.config["execution_flow"]["fixed_records"] == 16
    assert protocol.config["execution_flow"]["flat_score_field_rule"].endswith("StageARecord_v1")
    assert protocol.config["detection_access"]["joint_score"] == "min(s_LF,s_HF)"
    assert protocol.config["detection_access"]["hf_detector"] == (
        "frozen_hf_final_rgb_public_vae_global_normalized_correlation"
    )
    assert protocol.config["detection_access"]["lf_detector"] == (
        "frozen_lf_final_rgb_public_vae_block_centered_normalized_median_correlation"
    )
    assert "lf_branch_share" in protocol.config["detection_access"]["forbidden_inputs"]
    assert "hf_branch_share" in protocol.config["detection_access"]["forbidden_inputs"]


@pytest.mark.unit
def test_protocol_rejects_per_branch_budget_or_private_detector_input(tmp_path: Path) -> None:
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    config["budget"]["single_shared_budget_not_per_branch"] = False
    modified = tmp_path / "config.json"
    modified.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="combined budget"):
        load_content_adaptive_dual_branch_clean_protocol(modified, _ROSTER)
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    config["detection_access"]["allowed_inputs"].append("prompt")
    modified.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="detection access"):
        load_content_adaptive_dual_branch_clean_protocol(modified, _ROSTER)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("counterfactual_effect_fields_in_order", ["swapped"]),
        ("branch_share_sum_absolute_tolerance", 1e-6),
        ("population_std_formula", "copied"),
        ("population_std_fixed_roster_units", 7),
        ("population_std_availability", "all_return_codes"),
        ("blind_score_consumes_aggregate_measurement", True),
        ("forbidden_private_exports", []),
    ],
)
def test_protocol_rejects_aggregate_measurement_contract_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    config["aggregate_measurement"][field] = value
    modified = tmp_path / "config.json"
    modified.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate measurement"):
        load_content_adaptive_dual_branch_clean_protocol(modified, _ROSTER)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("content_analysis", "unexpected", "not_allowed", "content analysis"),
        ("detection_access", "hf_detector", "drifted", "detection access"),
        ("detection_access", "lf_detector", "drifted", "detection access"),
        ("detection_access", "unexpected", "not_allowed", "detection access"),
    ],
)
def test_protocol_rejects_extra_analysis_fields_and_detector_identity_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: str,
    message: str,
) -> None:
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    config[section][field] = value
    modified = tmp_path / "config.json"
    modified.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_content_adaptive_dual_branch_clean_protocol(modified, _ROSTER)
