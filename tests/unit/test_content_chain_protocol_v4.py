from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.method import content_whitening_v4 as method
from cegwm.protocol.content_chain_v3 import load_content_v3_clean_protocol
from cegwm.protocol.content_chain_v4 import (
    CONTENT_V4_PROTOCOL_DIGEST,
    CONTENT_V4_PROTOCOL_ID,
    CONTENT_V4_ROSTER_SHA256,
    load_content_v4_clean_protocol,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _ROOT / "configs" / "content_chain"
_CONFIG = _CONFIG_ROOT / "content_v4_clean_v1.json"
_V3_CONFIG = _CONFIG_ROOT / "content_v3_clean_v1.json"
_ROSTER = _CONFIG_ROOT / "content_adaptive_dual_branch_v2_clean.jsonl"


@pytest.mark.unit
def test_content_v4_protocol_binds_method_detector_asset_and_canonical_digest() -> None:
    protocol = load_content_v4_clean_protocol(_CONFIG, _ROSTER)
    assert protocol.protocol_id == CONTENT_V4_PROTOCOL_ID
    assert protocol.protocol_digest == CONTENT_V4_PROTOCOL_DIGEST
    assert protocol.config["protocol_version"] == 1
    identities = protocol.config["method_identities"]
    assert identities["content_method_id"] == method.CONTENT_V4_METHOD_ID
    assert identities["evaluated_candidate_id"] == method.CONTENT_V4_EVALUATED_CANDIDATE_ID
    detector = protocol.config["lf_detection_operator"]
    assert detector["scorer_id"] == method.CONTENT_V4_LF_SCORER_ID
    assert detector["asset_role_id"] == method.ASSET_ROLE_ID
    assert detector["asset_schema_id"] == method.ASSET_SCHEMA_ID
    assert detector["asset_sha256"] == method.ASSET_SHA256
    assert detector["asset_sidecar_sha256"] == method.ASSET_SIDECAR_SHA256
    assert detector["observation_contract_id"] == method.OBSERVATION_CONTRACT_ID
    assert detector["whitening_shape"] == (16, 6)
    assert detector["whitening_order"] == method.WHITENING_ORDER
    assert detector["whitening_key_independent"] is True


@pytest.mark.unit
def test_content_v4_reuses_exact_v3_embed_hf_budget_roster_and_gates() -> None:
    v4 = load_content_v4_clean_protocol(_CONFIG, _ROSTER)
    v3 = load_content_v3_clean_protocol(_V3_CONFIG, _ROSTER)
    assert hashlib.sha256(_ROSTER.read_bytes()).hexdigest() == CONTENT_V4_ROSTER_SHA256
    assert v4.roster == v3.roster
    assert v4.config["generation_runtime"] == v3.config["generation_runtime"]
    assert v4.config["content_analysis"] == v3.config["content_analysis"]
    assert v4.config["budget"] == v3.config["budget"]
    assert v4.config["aggregate_measurement"] == v3.config["aggregate_measurement"]
    assert v4.config["keying"] == v3.config["keying"]
    assert v4.config["decision_rule"] == v3.config["decision_rule"]
    assert v4.config["decision_rule"]["strict_comparison_ties_fail"] is True
    assert v4.config["decision_rule"]["formal_fpr_claim"] is False
    assert v4.config["execution_flow"]["fixed_units"] == 8
    assert v4.config["execution_flow"]["fixed_records"] == 16
    assert v4.config["execution_flow"]["retry_units_allowed"] is False


@pytest.mark.unit
def test_content_v4_detection_access_is_blind_and_has_no_v3_lf_fallback() -> None:
    protocol = load_content_v4_clean_protocol(_CONFIG, _ROSTER)
    access = protocol.config["detection_access"]
    assert access["allowed_inputs"] == ("image", "detection_key", "frozen_public_assets")
    assert access["lf_detector"] == method.CONTENT_V4_LF_SCORER_ID
    assert access["joint_score"] == "min(s_LF,s_HF)"
    assert set(access["forbidden_inputs"]) >= {
        "original_image", "prompt", "private_latent", "embed_side_route", "route"
    }
    serialized = json.dumps(dict(protocol.config["lf_detection_operator"]), sort_keys=True)
    assert "block_centered_normalized_median" not in serialized
    assert "fallback" not in serialized


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("lf_detection_operator", "asset_sha256", "0" * 64, "lf detection operator"),
        ("method_identities", "content_method_id", "v3", "method identities"),
        ("decision_rule", "strict_comparison_ties_fail", False, "decision rule"),
        ("keying", "wrong_key_count", 15, "keying"),
        ("execution_flow", "retry_units_allowed", True, "execution flow"),
    ),
)
def test_content_v4_loader_fails_closed_on_identity_or_science_drift(
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
        load_content_v4_clean_protocol(modified, _ROSTER)
