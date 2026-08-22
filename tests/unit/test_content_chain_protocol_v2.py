from __future__ import annotations

import json
from pathlib import Path

import pytest

from cegwm.protocol.content_chain_v2 import load_content_adaptive_dual_branch_v2_clean_protocol

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_v2_clean_v1.json"
_ROSTER = _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_v2_clean.jsonl"


@pytest.mark.unit
def test_v2_protocol_freezes_64_probe_six_effect_and_fresh_fixed_denominator_identity() -> None:
    protocol = load_content_adaptive_dual_branch_v2_clean_protocol(_CONFIG, _ROSTER)
    assert protocol.protocol_id == "cegwm-stage-a-content-adaptive-dual-branch-v2-clean-v1"
    assert len(protocol.roster) == 8
    assert len({unit.unit_id for unit in protocol.roster}) == 8
    assert len({unit.source_id for unit in protocol.roster}) == 8
    assert all("v2" in unit.unit_id and "v2" in unit.source_id for unit in protocol.roster)
    analysis = protocol.config["content_analysis"]
    assert analysis["probe_evaluations_per_unit"] == 64
    assert analysis["probe_relative_l2_scales_in_order"] == (0.0005, 0.001)
    assert analysis["probe_measurement"] == "baseline_differenced_branch_tile_two_scale_v1"
    assert analysis["probe_domain"].endswith("v2")
    assert analysis["baseline_observations"] == ("I0_equals_D_of_z", "Y0_equals_E_of_I0")
    assert len(analysis["signals"]) == 6
    identities = protocol.config["method_identities"]
    assert identities["evaluated_candidate_id"] == "content_adaptive_dual_branch_v2_clean_v1"
    assert identities["hf_adaptive_embedding_transform_id"].endswith("v2")
    assert identities["lf_adaptive_embedding_transform_id"].endswith("v2")
    aggregate = protocol.config["aggregate_measurement"]
    assert len(aggregate["counterfactual_effect_fields_in_order"]) == 6
    assert aggregate["counterfactual_effect_validation"].endswith("zero_allowed")
    assert aggregate["population_std_fixed_roster_units"] == 8
    assert aggregate["population_std_absolute_tolerance"] == 1e-12
    assert aggregate["blind_score_consumes_aggregate_measurement"] is False
    assert protocol.config["execution_flow"]["fixed_records"] == 16
    assert protocol.config["keying"]["wrong_key_count"] == 16
    assert protocol.config["decision_rule"]["lf_gate_a_registered_top_rank_among_17_min_units"] == 7
    assert protocol.config["decision_rule"]["paired_rgb_psnr_pass_units"] == 8


@pytest.mark.unit
def test_v2_protocol_rejects_formula_identity_detector_or_private_export_drift(tmp_path: Path) -> None:
    mutations = (
        ("content_analysis", "probe_measurement", "candidate_observation_magnitude", "content analysis"),
        ("method_identities", "evaluated_candidate_id", "v1", "method identities"),
        ("aggregate_measurement", "counterfactual_effect_validation", "strictly_positive", "aggregate measurement"),
        ("aggregate_measurement", "blind_score_consumes_aggregate_measurement", True, "aggregate measurement"),
        ("detection_access", "allowed_inputs", ["image", "route"], "detection access"),
    )
    for section, field, value, message in mutations:
        config = json.loads(_CONFIG.read_text(encoding="utf-8"))
        config[section][field] = value
        modified = tmp_path / f"{section}-{field}.json"
        modified.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_content_adaptive_dual_branch_v2_clean_protocol(modified, _ROSTER)


@pytest.mark.unit
def test_v2_protocol_detection_is_blind_and_private_maps_are_never_exported() -> None:
    protocol = load_content_adaptive_dual_branch_v2_clean_protocol(_CONFIG, _ROSTER)
    access = protocol.config["detection_access"]
    assert access["allowed_inputs"] == ("image", "detection_key", "frozen_public_assets")
    assert access["joint_score"] == "min(s_LF,s_HF)"
    assert set(protocol.config["aggregate_measurement"]["forbidden_private_exports"]) == {
        "mask", "tile_weights", "attention_map", "latent", "delta", "probe_state",
    }
    forbidden = set(access["forbidden_inputs"])
    assert {"prompt", "embed_record", "private_latent", "lf_branch_share", "hf_branch_share"} <= forbidden
