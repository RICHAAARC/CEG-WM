from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from cegwm.protocol.stage_a import (
    load_hf_v2_confirmation_protocol,
    load_hf_lf_attack_complementarity_protocol,
    load_lf_a3_selection_protocol,
    load_lf_balanced_blocks_confirmation_protocol,
    load_lf_balanced_blocks_selection_protocol,
    load_lf_v2_blocknorm_selection_protocol,
    load_stage_a_protocol,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "stage_a" / "stage_a_v1.json"
_SELECTION = _ROOT / "configs" / "stage_a" / "candidate_selection.jsonl"
_CONFIRMATION = _ROOT / "configs" / "stage_a" / "untouched_confirmation.jsonl"
_V2_CONFIG = _ROOT / "configs" / "stage_a" / "stage_a_hf_v2_rankgate.json"
_LF_A3_CONFIG = _ROOT / "configs" / "stage_a" / "stage_a_lf_a3_clean_selection_v1.json"
_LF_V2_CONFIG = _ROOT / "configs" / "stage_a" / "stage_a_lf_v2_blocknorm_selection_v1.json"
_LF_V2_SELECTION = _ROOT / "configs" / "stage_a" / "lf_v2_blocknorm_selection.jsonl"
_LF_V2_CONFIRMATION = (
    _ROOT / "configs" / "stage_a" / "lf_v2_blocknorm_untouched_confirmation.jsonl"
)
_LF_BALANCED_CONFIG = (
    _ROOT / "configs" / "stage_a" / "stage_a_lf_balanced_blocks_selection_v1.json"
)
_LF_BALANCED_SELECTION = (
    _ROOT / "configs" / "stage_a" / "lf_balanced_blocks_selection.jsonl"
)
_LF_BALANCED_CONFIRMATION = (
    _ROOT / "configs" / "stage_a" / "lf_balanced_blocks_untouched_confirmation.jsonl"
)
_LF_BALANCED_CONFIRMATION_CONFIG = (
    _ROOT / "configs" / "stage_a" / "stage_a_lf_balanced_blocks_confirmation_v1.json"
)
_ATTACK_CONFIG = (
    _ROOT / "configs" / "stage_a" / "stage_a_hf_lf_attack_complementarity_v1.json"
)
_ATTACK_ROSTER = (
    _ROOT / "configs" / "stage_a" / "hf_lf_attack_complementarity.jsonl"
)


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
def test_attack_complementarity_protocol_freezes_reference_attacks_and_128_records() -> None:
    protocol = load_hf_lf_attack_complementarity_protocol(
        _ATTACK_CONFIG, _ATTACK_ROSTER
    )
    assert protocol.protocol_id == "cegwm-stage-a-hf-lf-attack-complementarity-v1"
    assert protocol.untouched_confirmation == ()
    assert [unit.unit_id for unit in protocol.candidate_selection] == [
        f"attack-comp-{index:04d}" for index in range(1, 9)
    ]
    assert protocol.config["condition_order"] == (
        "identity_reference",
        "jpeg_q75",
        "gaussian_blur_sigma_1",
        "gaussian_noise_std_0_01",
    )
    assert protocol.config["attack_ids"] == protocol.config["condition_order"][1:]
    assert protocol.config["methods"]["hf"]["detector_statistic_id"] == (
        "vae_reencode_hf_masked_normalized_correlation"
    )
    assert protocol.config["execution_flow"]["unit_transaction_record_count"] == 16
    assert protocol.config["execution_flow"]["fixed_records"] == 128
    assert protocol.config["decision_rule"]["paired_clean_failure_outcome"] == (
        "SCIENTIFIC_NEGATIVE_FOR_PAIRED_CLEAN_PREREQUISITE_"
        "ATTACK_COMPLEMENTARITY_NOT_EVALUABLE_AND_STOP"
    )
    assert len(protocol.protocol_digest) == 64


@pytest.mark.unit
def test_attack_roster_is_globally_disjoint_from_prior_stage_a_manifests() -> None:
    attack = load_hf_lf_attack_complementarity_protocol(
        _ATTACK_CONFIG, _ATTACK_ROSTER
    ).candidate_selection
    prior_units: set[str] = set()
    prior_sources: set[str] = set()
    prior_prompts: set[str] = set()
    prior_seeds: set[int] = set()
    for manifest in (_ROOT / "configs" / "stage_a").glob("*.jsonl"):
        if manifest == _ATTACK_ROSTER:
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            prior_units.add(payload["unit_id"])
            prior_sources.add(payload["source_id"])
            prior_prompts.add(payload["prompt"])
            prior_seeds.add(payload["seed"])
    assert {unit.unit_id for unit in attack}.isdisjoint(prior_units)
    assert {unit.source_id for unit in attack}.isdisjoint(prior_sources)
    assert {unit.prompt for unit in attack}.isdisjoint(prior_prompts)
    assert {unit.seed for unit in attack}.isdisjoint(prior_seeds)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("condition_order",), ["identity_reference"], "condition order"),
        (("attack_ids",), ["identity_reference"], "separate"),
        (("decision_rule", "gate_a_registered_top_rank_among_17_min_units"), 6, "decision rule"),
        (("execution_flow", "fixed_records"), 96, "128-record"),
        (("budget", "total_relative_l2_per_method"), 0.011, "0.012"),
        (("methods", "hf", "carrier_method_id"), "changed", "method"),
    ],
)
def test_attack_protocol_rejects_identity_gate_denominator_or_method_drift(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = json.loads(_ATTACK_CONFIG.read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    modified = tmp_path / "invalid-attack.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_hf_lf_attack_complementarity_protocol(modified, _ATTACK_ROSTER)


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


@pytest.mark.unit
def test_lf_v2_protocol_freezes_blocknorm_identity_fresh_rosters_and_scale_free_gates() -> None:
    protocol = load_lf_v2_blocknorm_selection_protocol(_LF_V2_CONFIG, _LF_V2_SELECTION)
    assert protocol.protocol_id == "cegwm-stage-a-lf-v2-blocknorm-selection-v1"
    assert [unit.unit_id for unit in protocol.candidate_selection] == [
        f"lfv2-selection-{index:04d}" for index in range(1, 9)
    ]
    assert protocol.untouched_confirmation == ()
    candidate = protocol.config["candidate"]
    assert candidate["carrier_method_id"] == "lf_shell_rademacher_v1"
    assert candidate["evaluated_candidate_id"] == (
        "lf_shell_rademacher_v1_blocknorm_median_v2"
    )
    assert candidate["detector_statistic_id"] == (
        "lf_block_centered_normalized_median_corr_v2"
    )
    assert candidate["detector_radial_blocks"] == (
        MappingProxyType({"radius": (0.14, 0.165), "upper_bound": "exclusive"}),
        MappingProxyType({"radius": (0.165, 0.19), "upper_bound": "exclusive"}),
        MappingProxyType({"radius": (0.19, 0.215), "upper_bound": "exclusive"}),
        MappingProxyType({"radius": (0.215, 0.24), "upper_bound": "inclusive"}),
    )
    assert protocol.config["execution_flow"]["fixed_records"] == 16
    assert protocol.config["selection_rule"]["registered_top_rank_among_17_min_units"] == 7
    assert protocol.config["selection_rule"][
        "paired_lf_registered_gt_primary_null_registered_min_units"
    ] == 7
    assert protocol.config["selection_rule"]["absolute_margin_role"] == (
        "reported_effect_size_only_no_pass_threshold"
    )

    old_units = [
        json.loads(line)
        for path in (_SELECTION, _CONFIRMATION)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    new_selection = [
        json.loads(line) for line in _LF_V2_SELECTION.read_text(encoding="utf-8").splitlines()
    ]
    new_confirmation = [
        json.loads(line)
        for line in _LF_V2_CONFIRMATION.read_text(encoding="utf-8").splitlines()
    ]
    for field in ("unit_id", "source_id", "prompt", "seed"):
        old_values = {unit[field] for unit in old_units}
        selection_values = {unit[field] for unit in new_selection}
        confirmation_values = {unit[field] for unit in new_confirmation}
        assert old_values.isdisjoint(selection_values | confirmation_values)
        assert selection_values.isdisjoint(confirmation_values)
        assert len(selection_values) == len(confirmation_values) == 8
    assert [unit["split"] for unit in new_confirmation] == [
        "lf_v2_blocknorm_untouched_confirmation"
    ] * 8


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("candidate", "carrier_method_id"), "changed", "candidate or block-normalized"),
        (("candidate", "detector_statistic_id"), "changed", "candidate or block-normalized"),
        (("selection_rule", "registered_top_rank_among_17_min_units"), 6, "scale-free"),
        (("selection_rule", "absolute_margin_min"), 0.08, "scale-free"),
        (("execution_flow", "fixed_records"), 32, "8-unit/16-record"),
        (("budget", "total_relative_l2"), 0.013, "0.012 budget"),
    ],
)
def test_lf_v2_protocol_rejects_method_gate_or_denominator_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    payload = json.loads(_LF_V2_CONFIG.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = value
    modified = tmp_path / "invalid-lf-v2.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_lf_v2_blocknorm_selection_protocol(modified, _LF_V2_SELECTION)


@pytest.mark.unit
def test_lf_balanced_blocks_protocol_freezes_carrier_gate_and_globally_fresh_rosters() -> None:
    protocol = load_lf_balanced_blocks_selection_protocol(
        _LF_BALANCED_CONFIG,
        _LF_BALANCED_SELECTION,
    )
    assert protocol.protocol_id == "cegwm-stage-a-lf-balanced-blocks-selection-v1"
    assert [unit.unit_id for unit in protocol.candidate_selection] == [
        f"lfbb-selection-{index:04d}" for index in range(1, 9)
    ]
    assert protocol.untouched_confirmation == ()
    candidate = protocol.config["candidate"]
    assert candidate["carrier_method_id"] == "lf_shell_balanced_blocks_v2"
    assert candidate["evaluated_candidate_id"] == (
        "lf_shell_balanced_blocks_v2_blocknorm_median_v1"
    )
    assert candidate["detector_statistic_id"] == (
        "lf_block_centered_normalized_median_corr_v2"
    )
    assert [block["canonical_bound_token"] for block in candidate["radial_blocks"]] == [
        "0.14<=r<0.165",
        "0.165<=r<0.19",
        "0.19<=r<0.215",
        "0.215<=r<=0.24",
    ]
    assert [block["block_index"] for block in candidate["radial_blocks"]] == [0, 1, 2, 3]
    assert protocol.config["selection_rule"]["registered_top_rank_among_17_min_units"] == 7
    assert protocol.config["selection_rule"][
        "paired_lf_registered_gt_primary_null_registered_min_units"
    ] == 7
    assert protocol.config["execution_flow"]["fixed_records"] == 16

    current_paths = {_LF_BALANCED_SELECTION, _LF_BALANCED_CONFIRMATION}
    old_units = [
        json.loads(line)
        for path in (_ROOT / "configs" / "stage_a").glob("*.jsonl")
        if path not in current_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    selection = [
        json.loads(line)
        for line in _LF_BALANCED_SELECTION.read_text(encoding="utf-8").splitlines()
    ]
    confirmation = [
        json.loads(line)
        for line in _LF_BALANCED_CONFIRMATION.read_text(encoding="utf-8").splitlines()
    ]
    for field in ("unit_id", "source_id", "prompt", "seed"):
        old_values = {unit[field] for unit in old_units}
        selection_values = {unit[field] for unit in selection}
        confirmation_values = {unit[field] for unit in confirmation}
        assert old_values.isdisjoint(selection_values | confirmation_values)
        assert selection_values.isdisjoint(confirmation_values)
        assert len(selection_values) == len(confirmation_values) == 8
    assert {unit["split"] for unit in selection} == {"lf_balanced_blocks_selection"}
    assert {unit["split"] for unit in confirmation} == {
        "lf_balanced_blocks_untouched_confirmation"
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("candidate", "carrier_method_id"), "changed", "carrier or detector"),
        (("candidate", "detector_statistic_id"), "changed", "carrier or detector"),
        (("candidate", "construction_dtype"), "float32", "carrier or detector"),
        (("selection_rule", "registered_top_rank_among_17_min_units"), 6, "scale-free"),
        (("selection_rule", "absolute_margin_min"), 0.03, "scale-free"),
        (("execution_flow", "fixed_records"), 8, "8-unit/16-record"),
        (("budget", "total_relative_l2"), 0.013, "0.012 budget"),
    ],
)
def test_lf_balanced_blocks_protocol_rejects_formula_gate_or_denominator_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    payload = json.loads(_LF_BALANCED_CONFIG.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = value
    modified = tmp_path / "invalid-lf-balanced.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_lf_balanced_blocks_selection_protocol(modified, _LF_BALANCED_SELECTION)


@pytest.mark.unit
def test_lf_balanced_blocks_confirmation_loader_freezes_scope_roster_digest_and_method() -> None:
    confirmation = load_lf_balanced_blocks_confirmation_protocol(
        _LF_BALANCED_CONFIRMATION_CONFIG,
        _LF_BALANCED_CONFIRMATION,
    )
    selection = load_lf_balanced_blocks_selection_protocol(
        _LF_BALANCED_CONFIG,
        _LF_BALANCED_SELECTION,
    )
    assert confirmation.protocol_id == "cegwm-stage-a-lf-balanced-blocks-confirmation-v1"
    assert confirmation.config["execution_scope_id"] == (
        "lf_balanced_blocks_untouched_confirmation_v1"
    )
    assert confirmation.candidate_selection == ()
    assert [unit.unit_id for unit in confirmation.untouched_confirmation] == [
        f"lfbb-confirmation-{index:04d}" for index in range(1, 9)
    ]
    for field in (
        "generation_runtime",
        "keying",
        "candidate",
        "budget",
        "record_arms_in_exact_unit_order",
        "controls",
    ):
        assert confirmation.config[field] == selection.config[field]
    assert confirmation.config["selection_provenance"] == MappingProxyType({
        "selection_exact": "12833721415683bdc6028013080ec28bf8e529e3",
        "selection_run_id": "lfbbsel-45fcf9fdd8480f450aeaf9d6",
        "selection_artifact_sha256": "6be7f792695fc7ac3c194a1fddb9d81e7d04c2dc7032e17c72c61f52041ec8e2",
        "selection_protocol_digest": "f023b307a7822f8584bd641ab7b3accff762d86831d9b386e7ee40a66c01cf85",
        "selection_agent5_verdict": "FINAL_APPROVE_selection_only",
    })
    config = json.loads(_LF_BALANCED_CONFIRMATION_CONFIG.read_text(encoding="utf-8"))
    units = [
        json.loads(line)
        for line in _LF_BALANCED_CONFIRMATION.read_text(encoding="utf-8").splitlines()
    ]
    canonical = json.dumps(
        {"config": config, "untouched_confirmation": units},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert confirmation.protocol_digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert hashlib.sha256(_LF_BALANCED_CONFIRMATION.read_bytes()).hexdigest() == (
        "129f4a0633eddbc2ae16bd9a400d6bc6e0e940b42f862b7473cfe231e4d21713"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("execution_scope_id", ""), "changed", "execution scope"),
        (("scope_exclusions", ""), [], "scope exclusions"),
        (("selection_provenance", "selection_run_id"), "changed", "selection provenance"),
        (("candidate", "carrier_method_id"), "changed", "carrier or detector"),
        (("confirmation_rule", "registered_top_rank_among_17_min_units"), 6, "scale-free"),
        (("confirmation_rule", "absolute_margin_min"), 0.03, "scale-free"),
        (("execution_flow", "fixed_records"), 8, "8-unit/16-record"),
        (("budget", "total_relative_l2"), 0.013, "0.012 budget"),
    ],
)
def test_lf_balanced_blocks_confirmation_rejects_identity_gate_or_denominator_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    payload = json.loads(_LF_BALANCED_CONFIRMATION_CONFIG.read_text(encoding="utf-8"))
    if path[1]:
        payload[path[0]][path[1]] = value
    else:
        payload[path[0]] = value
    modified = tmp_path / "invalid-lf-balanced-confirmation.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_lf_balanced_blocks_confirmation_protocol(modified, _LF_BALANCED_CONFIRMATION)


@pytest.mark.unit
def test_lf_balanced_blocks_confirmation_loader_refuses_selection_roster() -> None:
    with pytest.raises(ValueError, match="wrong split"):
        load_lf_balanced_blocks_confirmation_protocol(
            _LF_BALANCED_CONFIRMATION_CONFIG,
            _LF_BALANCED_SELECTION,
        )
