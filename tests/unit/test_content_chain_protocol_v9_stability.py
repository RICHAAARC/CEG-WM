from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from cegwm.protocol import content_chain_v9_stability as stability

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _ROOT / "configs" / "content_chain"


@pytest.mark.unit
def test_v9_stability_contract_has_exact_independent_sections_and_digest() -> None:
    contract = stability.load_content_v9_stability_contract(_ROOT)

    assert contract.protocol_digest == stability.CONTENT_V9_STABILITY_PROTOCOL_DIGEST
    assert contract.protocol_digest == (
        "9bc8a94c1d022cfaaf3c36018422b245e42764571314ee048d612e58a19ca031"
    )
    assert len(contract.old_roster_reference) == 8
    assert len(contract.current_v6_roster_reference) == 8
    assert len(contract.novel_seed_01) == 32
    assert len(contract.novel_seed_02) == 32

    config = contract.config
    assert config["protocol_id"] == stability.CONTENT_V9_STABILITY_PROTOCOL_ID
    assert config["identities"]["method_id"] == stability.CONTENT_V9_STABILITY_METHOD_ID
    assert config["identities"]["evaluated_candidate_id"] == (
        stability.CONTENT_V9_STABILITY_EVALUATED_CANDIDATE_ID
    )
    assert config["identities"]["record_contract_id"] == (
        stability.CONTENT_V9_STABILITY_RECORD_CONTRACT_ID
    )
    assert [section["section_id"] for section in config["sections"]] == [
        "old_roster_reference",
        "current_v6_roster_reference",
        "novel_seed_stability",
    ]
    execution = config["execution_contract"]
    assert execution["section_order"] == [
        "old_roster_reference",
        "current_v6_roster_reference",
        "novel_seed_stability_seed_01",
        "novel_seed_stability_seed_02",
    ]
    assert execution["separate_denominators"] is True
    for forbidden in (
        "pooled_denominator",
        "cross_section_conjunction",
        "combined_result",
        "outcome_conditioned_stop",
        "replacement_units_allowed",
        "selective_omission_allowed",
    ):
        assert execution[forbidden] is False
    assert execution["total_units_metadata_only_not_a_denominator"] == 80
    assert execution["total_records_metadata_only_not_a_denominator"] == 160
    assert [
        (section["weighted_gate_a_min_units"], section["weighted_gate_b_min_units"])
        for section in config["sections"][:2]
    ] == [(7, 7), (7, 7)]
    assert config["inherited_v9"]["lf_weight"] == 0.25
    assert config["inherited_v9"]["hf_weight"] == 0.75
    assert config["identities"]["source_calibration_protocol_digest"] == (
        "68f37585eb6eab123bad7c1703767df08404718ce4771f73fbbec236491a1e01"
    )


@pytest.mark.unit
def test_v9_stability_novel_manifest_bytes_order_and_disjointness() -> None:
    contract = stability.load_content_v9_stability_contract(_ROOT)
    path = _CONFIG_ROOT / stability.CONTENT_V9_STABILITY_NOVEL_MANIFEST
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        stability.CONTENT_V9_STABILITY_NOVEL_MANIFEST_SHA256
    )
    lines = raw.splitlines(keepends=True)
    assert len(lines) == 64
    assert hashlib.sha256(b"".join(lines[:32])).hexdigest() == (
        stability.CONTENT_V9_STABILITY_SEED_01_SLICE_SHA256
    )
    assert hashlib.sha256(b"".join(lines[32:])).hexdigest() == (
        stability.CONTENT_V9_STABILITY_SEED_02_SLICE_SHA256
    )
    assert tuple(unit.seed for unit in contract.novel_seed_01) == tuple(
        range(2026101000, 2026101032)
    )
    assert tuple(unit.seed for unit in contract.novel_seed_02) == tuple(
        range(2026102000, 2026102032)
    )
    assert tuple(unit.prompt for unit in contract.novel_seed_01) == tuple(
        unit.prompt for unit in contract.novel_seed_02
    )
    prompt_bytes = b"".join(
        unit.prompt.encode("utf-8") + b"\n" for unit in contract.novel_seed_01
    )
    assert hashlib.sha256(prompt_bytes).hexdigest() == (
        stability.CONTENT_V9_STABILITY_NOVEL_PROMPT_LIST_SHA256
    )

    prior: set[tuple[str, int]] = {
        (
            "A book conservator examining an illuminated manuscript under neutral studio light",
            1415149,
        )
    }
    for name in (
        "content_adaptive_dual_branch_clean.jsonl",
        "content_adaptive_dual_branch_v2_clean.jsonl",
        "content_v6_iss_development_v1.jsonl",
        "content_v6_iss_clean.jsonl",
        "content_v9_calibration_v1.jsonl",
        "content_v9_clean_evaluation_v1.jsonl",
    ):
        for line in (_CONFIG_ROOT / name).read_bytes().splitlines():
            value = json.loads(line)
            prior.add((value["prompt"], value["seed"]))
    fit = json.loads(
        (_CONFIG_ROOT / "content_v4_clean_null_whitening_fit_v1.json").read_bytes()
    )
    prior.update((entry["prompt"], entry["generation_seed"]) for entry in fit["entries"])
    assert not prior.intersection(
        (unit.prompt, unit.seed)
        for unit in (*contract.novel_seed_01, *contract.novel_seed_02)
    )


@pytest.mark.unit
def test_v9_stability_weighted_gates_are_section_local_and_strict() -> None:
    old_a = [1.0] * 7 + [0.0]
    old_b = [1.0] * 6 + [0.0, -1.0]
    assert stability.strict_weighted_gate(old_a, required=7) == (7, True)
    assert stability.strict_weighted_gate(old_b, required=7) == (6, False)
    assert stability.strict_weighted_gate([1.0] * 28 + [0.0] * 4, required=28) == (
        28,
        True,
    )
    assert stability.strict_weighted_gate([1.0] * 27 + [0.0] * 5, required=28) == (
        27,
        False,
    )
    with pytest.raises(ValueError, match="finite"):
        stability.strict_weighted_gate([1.0] * 31 + [float("nan")], required=28)
    with pytest.raises(ValueError, match="real numbers"):
        stability.strict_weighted_gate([True] + [1.0] * 31, required=28)

    config = stability.load_content_v9_stability_contract(_ROOT).config
    novel = config["sections"][2]
    assert [stratum["weighted_gate_a_min_units"] for stratum in novel["seed_strata"]] == [
        28,
        28,
    ]
    assert novel["two_seed_prompt_agreement"] == (
        "descriptive_only_no_threshold_or_conjunction"
    )
    assert config["joint_operator"]["lf_hf_gates"] == "diagnostic_only_no_hard_veto"


@pytest.mark.unit
def test_v9_stability_run_identity_requires_a_real_calibration_asset_digest() -> None:
    digest = stability.CONTENT_V9_STABILITY_PROTOCOL_DIGEST
    run_id = stability.deterministic_stability_run_id(digest, "b" * 64, "c" * 64)
    assert run_id == f"content-v9-stability-{digest[:12]}-{'b' * 12}-{'c' * 12}"
    with pytest.raises(ValueError, match="lowercase 64-hex"):
        stability.deterministic_stability_run_id(digest, "asset-not-yet-produced", "c" * 64)
    config = stability.load_content_v9_stability_contract(_ROOT).config
    assert config["calibration_asset_input"]["required_before_execution"] is True
    assert config["calibration_asset_input"][
        "numeric_asset_sha256_bound_only_after_user_acceptance"
    ] is True
    assert config["run_identity"]["numeric_run_id_deferred_until_user_accepted_calibration_asset"] is True
    assert "asset_sha256" not in config["calibration_asset_input"]


@pytest.mark.unit
def test_v9_stability_binds_the_exact_accepted_asset_and_final_run_identity() -> None:
    contract = stability.load_content_v9_stability_contract(_ROOT)
    asset_path = _CONFIG_ROOT / stability.CONTENT_V9_STABILITY_CALIBRATION_ASSET
    sidecar_path = asset_path.with_name(f"{asset_path.name}.sha256")
    assert hashlib.sha256(asset_path.read_bytes()).hexdigest() == (
        stability.CONTENT_V9_STABILITY_CALIBRATION_ASSET_SHA256
    )
    assert hashlib.sha256(sidecar_path.read_bytes()).hexdigest() == (
        stability.CONTENT_V9_STABILITY_CALIBRATION_ASSET_SIDECAR_FILE_SHA256
    )
    assert contract.calibration_asset.payload["producer_exact"] == (
        stability.CONTENT_V9_STABILITY_CALIBRATION_PRODUCER_EXACT
    )
    assert contract.calibration_asset.payload["calibration_protocol_digest"] == (
        stability.CONTENT_V9_STABILITY_CALIBRATION_PROTOCOL_DIGEST
    )
    assert contract.calibration_asset.payload["calibration_public_key_digest"] == (
        stability.CONTENT_V9_STABILITY_CALIBRATION_PUBLIC_KEY_DIGEST
    )
    assert stability.deterministic_stability_run_id(
        contract.protocol_digest,
        stability.CONTENT_V9_STABILITY_CALIBRATION_ASSET_SHA256,
        stability.CONTENT_V9_STABILITY_PUBLIC_KEY_DIGEST,
    ) == "content-v9-stability-9bc8a94c1d02-63c17e8200a9-805bc21e173a"


@pytest.mark.unit
def test_v9_stability_preserves_unused_future_manifest_and_fails_on_drift(
    tmp_path: Path,
) -> None:
    assert hashlib.sha256(
        (_CONFIG_ROOT / stability.CONTENT_V9_UNUSED_EVALUATION_MANIFEST).read_bytes()
    ).hexdigest() == stability.CONTENT_V9_UNUSED_EVALUATION_MANIFEST_SHA256
    contract = stability.load_content_v9_stability_contract(_ROOT)
    assert contract.config["preserved_unused_future_evaluation"] == {
        "manifest": "content_v9_clean_evaluation_v1.jsonl",
        "manifest_sha256": stability.CONTENT_V9_UNUSED_EVALUATION_MANIFEST_SHA256,
        "prompt_list_sha256": (
            "ed8796c183200ec826023d44f9ebe06293968f5a387dbcf242addaf036e46617"
        ),
        "status": "immutable_unexecuted_provenance_excluded_from_all_sections",
    }

    target = tmp_path / "configs" / "content_chain"
    target.mkdir(parents=True)
    for name in (
        stability.CONTENT_V9_STABILITY_OLD_MANIFEST,
        stability.CONTENT_V9_STABILITY_CURRENT_MANIFEST,
        stability.CONTENT_V9_STABILITY_NOVEL_MANIFEST,
        stability.CONTENT_V9_UNUSED_EVALUATION_MANIFEST,
        "content_v9_multi_cohort_stability_v1.json",
    ):
        shutil.copyfile(_CONFIG_ROOT / name, target / name)
    config_path = target / "content_v9_multi_cohort_stability_v1.json"
    drifted = json.loads(config_path.read_bytes())
    drifted["execution_contract"]["pooled_denominator"] = True
    config_path.write_text(json.dumps(drifted, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical protocol digest differs"):
        stability.load_content_v9_stability_contract(tmp_path)


@pytest.mark.unit
def test_v9_stability_config_bytes_are_stable_and_have_no_result_import() -> None:
    path = _CONFIG_ROOT / "content_v9_multi_cohort_stability_v1.json"
    config = json.loads(path.read_bytes())
    assert path.read_bytes() == (json.dumps(config, indent=2) + "\n").encode("utf-8")
    assert config["later_read_only_comparison_provenance"][
        "result_bytes_consumed_by_this_protocol"
    ] is False
    serialized = json.dumps(config, sort_keys=True)
    for forbidden in (
        "all_predeclared_gates_pass",
        "combined_section_result",
        "pooled_gate",
        "calibration_asset_sha256\":",
    ):
        assert forbidden not in serialized
