from __future__ import annotations

import json
from pathlib import Path

import pytest

from cegwm.protocol.content_chain_v2 import load_content_adaptive_dual_branch_v2_clean_protocol

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_v2_clean_v1.json"
_ROSTER = _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_v2_clean.jsonl"
_PRE_V2_MANIFESTS = (
    _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_clean.jsonl",
    _ROOT / "configs" / "stage_a" / "candidate_selection.jsonl",
    _ROOT / "configs" / "stage_a" / "hf_lf_attack_complementarity.jsonl",
    _ROOT / "configs" / "stage_a" / "lf_balanced_blocks_selection.jsonl",
    _ROOT / "configs" / "stage_a" / "lf_balanced_blocks_untouched_confirmation.jsonl",
    _ROOT / "configs" / "stage_a" / "lf_v2_blocknorm_selection.jsonl",
    _ROOT / "configs" / "stage_a" / "lf_v2_blocknorm_untouched_confirmation.jsonl",
    _ROOT / "configs" / "stage_a" / "untouched_confirmation.jsonl",
)


@pytest.mark.unit
def test_v2_protocol_freezes_64_probe_six_effect_and_fresh_fixed_denominator_identity() -> None:
    protocol = load_content_adaptive_dual_branch_v2_clean_protocol(_CONFIG, _ROSTER)
    assert protocol.protocol_id == "cegwm-stage-a-content-adaptive-dual-branch-v2-semantic-gate-v1"
    assert protocol.protocol_digest == "bfd9b7464195107f7dc57a43ab3042501500f5e2c07a322269859bb908a3dbb8"
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
    assert analysis["texture_complexity_rule"].endswith("no_minmax_rank_kappa_or_fallback")
    assert analysis["derived_ranges"] == {
        "Q_L": (0.2, 0.85), "Q_H": (0.35, 1.0), "d": (-0.5, 0.8),
        "a_H": (0.375, 0.7), "a_L": (0.3, 0.625),
    }
    assert "gamma_0.25" in analysis["direct_per_tile_gate_rule"]
    assert "increasing_g" in analysis["allocation_directions"]
    assert analysis["counterfactual_neutral_rule"] == (
        "semantic_gate_0_texture_raw_x_0_response_consistency_and_sensitivity_0.5_unchanged"
    )
    identities = protocol.config["method_identities"]
    assert identities["evaluated_candidate_id"] == "content_adaptive_dual_branch_v2_semantic_gate_v1"
    assert "semantic_gate" in identities["hf_adaptive_embedding_transform_id"]
    assert "semantic_gate" in identities["lf_adaptive_embedding_transform_id"]
    aggregate = protocol.config["aggregate_measurement"]
    assert len(aggregate["counterfactual_effect_fields_in_order"]) == 6
    assert aggregate["counterfactual_effect_validation"].endswith("zero_allowed")
    assert aggregate["population_std_fixed_roster_units"] == 8
    assert aggregate["population_std_absolute_tolerance"] == 1e-12
    assert aggregate["blind_score_consumes_aggregate_measurement"] is False
    assert protocol.config["execution_flow"]["fixed_records"] == 16
    assert protocol.config["execution_flow"]["record_contract_id"] == (
        "content_adaptive_dual_branch_v2_semantic_gate_record_v1"
    )
    assert protocol.config["keying"]["wrong_key_count"] == 16
    assert protocol.config["decision_rule"]["lf_gate_a_registered_top_rank_among_17_min_units"] == 7
    assert protocol.config["decision_rule"]["paired_rgb_psnr_pass_units"] == 8


@pytest.mark.unit
def test_v2_protocol_rejects_formula_identity_detector_or_private_export_drift(tmp_path: Path) -> None:
    mutations = (
        ("content_analysis", "probe_measurement", "candidate_observation_magnitude", "content analysis"),
        ("content_analysis", "direct_per_tile_gate_rule", "gamma_0.20", "content analysis"),
        ("content_analysis", "derived_ranges", {"d": [-1.0, 1.0]}, "content analysis"),
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
    serialized = _CONFIG.read_text(encoding="utf-8")
    assert "transfer_stability" not in serialized
    assert "two_scale_response_consistency" in serialized


@pytest.mark.unit
def test_v2_roster_is_disjoint_from_the_explicit_tracked_pre_v2_manifests() -> None:
    protocol = load_content_adaptive_dual_branch_v2_clean_protocol(_CONFIG, _ROSTER)
    unit_ids = {unit.unit_id for unit in protocol.roster}
    source_ids = {unit.source_id for unit in protocol.roster}
    for manifest in _PRE_V2_MANIFESTS:
        assert manifest.is_file()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            assert payload["unit_id"] not in unit_ids
            assert payload["source_id"] not in source_ids


@pytest.mark.unit
def test_v2_loader_uses_only_the_explicit_config_and_ordered_roster(tmp_path: Path) -> None:
    config = tmp_path / _CONFIG.name
    roster = tmp_path / _ROSTER.name
    config.write_bytes(_CONFIG.read_bytes())
    roster.write_bytes(_ROSTER.read_bytes())
    (tmp_path / "unrelated-collision.jsonl").write_text(
        json.dumps({"unit_id": "content-adaptive-v2-0001"}) + "\n",
        encoding="utf-8",
    )
    protocol = load_content_adaptive_dual_branch_v2_clean_protocol(config, roster)
    assert len(protocol.roster) == 8


@pytest.mark.unit
def test_v2_loader_rejects_any_ordered_roster_identity_drift(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in _ROSTER.read_text(encoding="utf-8").splitlines()]
    rows[0]["prompt"] += " drift"
    drifted = tmp_path / _ROSTER.name
    drifted.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen eight units"):
        load_content_adaptive_dual_branch_v2_clean_protocol(_CONFIG, drifted)
