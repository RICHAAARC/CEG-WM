from __future__ import annotations

# Unit coverage for the content-whitening protocol.

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.method import content_whitening as method
from cegwm.protocol.content_unweighted import load_content_unweighted_clean_protocol
from cegwm.protocol.content_whitening import (
    CONTENT_WHITENING_PROTOCOL_DIGEST,
    CONTENT_WHITENING_PROTOCOL_ID,
    CONTENT_WHITENING_ROSTER_SHA256,
    load_content_whitening_protocol,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _ROOT / "configs" / "content_chain"
_CONFIG = _CONFIG_ROOT / "content_v4_clean_v1.json"
_UNWEIGHTED_CONFIG = _CONFIG_ROOT / "content_v3_clean_v1.json"
_ROSTER = _CONFIG_ROOT / "content_adaptive_dual_branch_v2_clean.jsonl"


@pytest.mark.unit
def test_content_whitening_protocol_binds_method_detector_asset_and_canonical_digest() -> None:
    protocol = load_content_whitening_protocol(_CONFIG, _ROSTER)
    assert protocol.protocol_id == CONTENT_WHITENING_PROTOCOL_ID
    assert protocol.protocol_digest == CONTENT_WHITENING_PROTOCOL_DIGEST
    assert protocol.config["protocol_version"] == 1
    identities = protocol.config["method_identities"]
    assert identities["content_method_id"] == method.CONTENT_WHITENING_METHOD_ID
    assert identities["evaluated_candidate_id"] == method.CONTENT_WHITENING_EVALUATED_CANDIDATE_ID
    detector = protocol.config["lf_detection_operator"]
    assert detector["scorer_id"] == method.CONTENT_WHITENING_LF_SCORER_ID
    assert detector["asset_role_id"] == method.ASSET_ROLE_ID
    assert detector["asset_schema_id"] == method.ASSET_SCHEMA_ID
    assert detector["asset_sha256"] == method.ASSET_SHA256
    assert detector["asset_sidecar_sha256"] == method.ASSET_SIDECAR_SHA256
    assert detector["observation_contract_id"] == method.OBSERVATION_CONTRACT_ID
    assert detector["whitening_shape"] == (16, 6)
    assert detector["whitening_order"] == method.WHITENING_ORDER
    assert detector["whitening_key_independent"] is True


@pytest.mark.unit
def test_content_whitening_reuses_unweighted_embed_hf_budget_roster_and_gates() -> None:
    whitening = load_content_whitening_protocol(_CONFIG, _ROSTER)
    unweighted = load_content_unweighted_clean_protocol(_UNWEIGHTED_CONFIG, _ROSTER)
    assert hashlib.sha256(_ROSTER.read_bytes()).hexdigest() == CONTENT_WHITENING_ROSTER_SHA256
    assert whitening.roster == unweighted.roster
    assert whitening.config["generation_runtime"] == unweighted.config["generation_runtime"]
    assert whitening.config["content_analysis"] == unweighted.config["content_analysis"]
    assert whitening.config["budget"] == unweighted.config["budget"]
    assert whitening.config["aggregate_measurement"] == unweighted.config["aggregate_measurement"]
    assert whitening.config["keying"] == unweighted.config["keying"]
    assert whitening.config["decision_rule"] == unweighted.config["decision_rule"]
    assert whitening.config["decision_rule"]["strict_comparison_ties_fail"] is True
    assert whitening.config["decision_rule"]["formal_fpr_claim"] is False
    assert whitening.config["execution_flow"]["fixed_units"] == 8
    assert whitening.config["execution_flow"]["fixed_records"] == 16
    assert whitening.config["execution_flow"]["retry_units_allowed"] is False


@pytest.mark.unit
def test_content_whitening_detection_access_is_blind_and_has_no_unweighted_lf_fallback() -> None:
    protocol = load_content_whitening_protocol(_CONFIG, _ROSTER)
    access = protocol.config["detection_access"]
    assert access["allowed_inputs"] == ("image", "detection_key", "frozen_public_assets")
    assert access["lf_detector"] == method.CONTENT_WHITENING_LF_SCORER_ID
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
def test_content_whitening_loader_fails_closed_on_identity_or_science_drift(
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
        load_content_whitening_protocol(modified, _ROSTER)
