from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.protocol.content_chain_v2 import (
    load_content_adaptive_dual_branch_v2_clean_protocol,
)
from cegwm.protocol.content_chain_v3 import (
    CONTENT_V3_PROTOCOL_DIGEST,
    CONTENT_V3_PROTOCOL_ID,
    CONTENT_V3_ROSTER_SHA256,
    RUNTIME_ASSET_VALIDATION_CONTRACT_ID,
    load_content_v3_clean_protocol,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "content_chain" / "content_v3_clean_v1.json"
_V2_CONFIG = (
    _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_v2_clean_v1.json"
)
_ROSTER = (
    _ROOT / "configs" / "content_chain" / "content_adaptive_dual_branch_v2_clean.jsonl"
)
_NEGATIVE = _ROOT / "docs" / "content_runtime_v3_scientific_negative_adjudication.md"


@pytest.mark.unit
def test_content_v3_protocol_has_distinct_method_protocol_digest_record_and_run_inputs() -> None:
    protocol = load_content_v3_clean_protocol(_CONFIG, _ROSTER)
    v2 = load_content_adaptive_dual_branch_v2_clean_protocol(_V2_CONFIG, _ROSTER)
    assert protocol.protocol_id == CONTENT_V3_PROTOCOL_ID
    assert protocol.protocol_digest == CONTENT_V3_PROTOCOL_DIGEST
    assert protocol.protocol_digest != v2.protocol_digest
    assert protocol.config["protocol_version"] == 2
    identities = protocol.config["method_identities"]
    assert identities["content_method_id"] == "content_v3_unweighted_lf_adaptive_hf_v1"
    assert identities["evaluated_candidate_id"] == (
        "content_v3_unweighted_lf_adaptive_hf_semantic_gate_v1"
    )
    assert identities["lf_base_carrier_method_id"] == "lf_shell_balanced_blocks_v2"
    assert identities["lf_embedding_transform_id"] == (
        "lf_unweighted_balanced_blocks_content_allocated_amplitude_v3"
    )
    assert identities["hf_adaptive_embedding_transform_id"] == (
        v2.config["method_identities"]["hf_adaptive_embedding_transform_id"]
    )
    flow = protocol.config["execution_flow"]
    assert flow["record_contract_id"] == (
        "content_v3_unweighted_lf_adaptive_hf_record_v1"
    )
    assert flow["record_arms_in_order"] == (
        "content_v3_unweighted_lf_adaptive_hf_semantic_gate_v1",
        "primary_null__content_v3_unweighted_lf_adaptive_hf_semantic_gate_v1",
    )
    assert flow["fixed_units"] == 8 and flow["fixed_records"] == 16


@pytest.mark.unit
def test_content_v3_reuses_exact_roster_science_key_and_blind_detection_contracts() -> None:
    protocol = load_content_v3_clean_protocol(_CONFIG, _ROSTER)
    v2 = load_content_adaptive_dual_branch_v2_clean_protocol(_V2_CONFIG, _ROSTER)
    assert hashlib.sha256(_ROSTER.read_bytes()).hexdigest() == CONTENT_V3_ROSTER_SHA256
    assert protocol.config["execution_flow"]["formal_roster_sha256"] == CONTENT_V3_ROSTER_SHA256
    assert protocol.roster == v2.roster
    assert tuple((unit.prompt, unit.seed) for unit in protocol.roster) == tuple(
        (unit.prompt, unit.seed) for unit in v2.roster
    )
    assert protocol.config["decision_rule"] == v2.config["decision_rule"]
    assert protocol.config["keying"] == v2.config["keying"]
    assert protocol.config["keying"]["wrong_key_count"] == 16
    assert protocol.config["detection_access"] == v2.config["detection_access"]
    assert protocol.config["detection_access"]["allowed_inputs"] == (
        "image", "detection_key", "frozen_public_assets",
    )
    assert protocol.config["detection_access"]["joint_score"] == "min(s_LF,s_HF)"
    flow = protocol.config["execution_flow"]
    assert flow["failure_units_remain_in_denominator"] is True
    assert flow["replacement_units_allowed"] is False
    assert flow["retry_units_allowed"] is False


@pytest.mark.unit
def test_content_v3_keeps_runtime_asset_contract_separate_from_method_version() -> None:
    protocol = load_content_v3_clean_protocol(_CONFIG, _ROSTER)
    analysis = protocol.config["content_analysis"]
    assert analysis["runtime_asset_validation_contract_id"] == (
        RUNTIME_ASSET_VALIDATION_CONTRACT_ID
    )
    assert protocol.config["protocol_version"] == 2
    assert analysis["probe_evaluations_per_unit"] == 64
    assert analysis["probe_measurement"] == (
        "baseline_differenced_branch_tile_two_scale_v1"
    )
    assert protocol.config["generation_runtime"]["injection_step_index_zero_based"] == 18
    assert "dependency_version" not in analysis
    assert "version_equality" not in analysis


@pytest.mark.unit
def test_content_v2_negative_is_immutable_provenance_and_excluded_from_v3() -> None:
    protocol = load_content_v3_clean_protocol(_CONFIG, _ROSTER)
    assert hashlib.sha256(_NEGATIVE.read_bytes()).hexdigest() == (
        "4afbc2b1a71a80838d4760931d108ac66e5509befcd784202922a4f688380a66"
    )
    exclusions = protocol.config["provenance_exclusions"]
    assert len(exclusions) == 2
    assert "not_Content_V3_evidence" in exclusions[0]
    assert "immutable" in exclusions[1]
    assert protocol.config["scientific_status"] == (
        "not_evaluated_until_complete_real_gpu_rc0"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("content_analysis", "runtime_asset_validation_contract_id", "v4", "content analysis"),
        ("method_identities", "lf_embedding_transform_id", "weighted", "method identities"),
        ("decision_rule", "strict_comparison_ties_fail", False, "science gates"),
        ("detection_access", "allowed_inputs", ["image", "route"], "detection access"),
        ("keying", "wrong_key_count", 15, "key controls"),
        ("execution_flow", "retry_units_allowed", True, "transaction denominator"),
    ),
)
def test_content_v3_loader_fails_closed_on_identity_or_science_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    config[section][field] = value
    modified = tmp_path / "modified.json"
    modified.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_content_v3_clean_protocol(modified, _ROSTER)
