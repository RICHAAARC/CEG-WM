from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.protocol.content_chain_v6 import load_content_v6_clean_protocol
from cegwm.protocol.content_chain_v6_reference_oldroster import (
    CONTENT_V6_REFERENCE_OLDROSTER_ARMS,
    CONTENT_V6_REFERENCE_OLDROSTER_CANDIDATE_ID,
    CONTENT_V6_REFERENCE_OLDROSTER_METHOD_ID,
    CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_DIGEST,
    CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_ID,
    CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID,
    CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256,
    _load_reference_roster,
    load_content_v6_reference_oldroster_protocol,
)

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_reference_protocol_binds_old_roster_and_fresh_identity() -> None:
    protocol = load_content_v6_reference_oldroster_protocol(_ROOT)
    assert protocol.protocol_id == CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_ID
    assert protocol.protocol_digest == CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_DIGEST
    assert len(protocol.roster) == 8
    assert tuple(unit.unit_id for unit in protocol.roster) == tuple(
        f"content-adaptive-v2-{index:04d}" for index in range(1, 9)
    )
    assert tuple(unit.seed for unit in protocol.roster) == (
        1213061, 1238321, 1263581, 1288843,
        1314103, 1339367, 1364627, 1389887,
    )
    manifest = _ROOT / "configs/content_chain/content_adaptive_dual_branch_v2_clean.jsonl"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        CONTENT_V6_REFERENCE_OLDROSTER_ROSTER_SHA256
    )
    assert protocol.config["method_identities"]["content_method_id"] == (
        CONTENT_V6_REFERENCE_OLDROSTER_METHOD_ID
    )
    assert protocol.config["method_identities"]["evaluated_candidate_id"] == (
        CONTENT_V6_REFERENCE_OLDROSTER_CANDIDATE_ID
    )
    assert tuple(protocol.config["execution_flow"]["record_arms_in_order"]) == (
        CONTENT_V6_REFERENCE_OLDROSTER_ARMS
    )
    assert protocol.config["execution_flow"]["record_contract_id"] == (
        CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID
    )


@pytest.mark.unit
def test_reference_protocol_changes_only_identity_and_cohort_from_v6() -> None:
    reference = load_content_v6_reference_oldroster_protocol(_ROOT)
    base = load_content_v6_clean_protocol(_ROOT)
    unchanged_sections = (
        "scientific_status", "generation_runtime", "content_analysis",
        "lf_detection_operator", "iss_controller", "budget",
        "aggregate_measurement", "detection_access", "keying",
        "decision_rule", "limitations",
    )
    for name in unchanged_sections:
        assert reference.config[name] == base.config[name]

    changed_method_fields = {"content_method_id", "evaluated_candidate_id"}
    assert {
        name
        for name in base.config["method_identities"]
        if reference.config["method_identities"][name]
        != base.config["method_identities"][name]
    } == changed_method_fields
    changed_flow_fields = {
        "roster_manifest", "formal_roster_sha256", "split",
        "record_arms_in_order", "flat_score_field_rule", "record_contract_id",
    }
    assert {
        name
        for name in base.config["execution_flow"]
        if reference.config["execution_flow"][name]
        != base.config["execution_flow"][name]
    } == changed_flow_fields
    assert reference.config["execution_flow"]["fixed_units"] == 8
    assert reference.config["execution_flow"]["fixed_records"] == 16

    definition = json.loads(
        (_ROOT / "configs/content_chain/content_v6_iss_reference_oldroster_v1.json")
        .read_text(encoding="utf-8")
    )
    assert definition["execution_separation"] == {
        "fresh_generation_required": True,
        "independent_artifact_required": True,
        "pooling_allowed": False,
        "current_v6_evaluation_roster_allowed": False,
        "imported_result_or_artifact_allowed": False,
    }


@pytest.mark.unit
def test_reference_roster_loader_fails_closed_on_byte_drift(tmp_path: Path) -> None:
    source = _ROOT / "configs/content_chain/content_adaptive_dual_branch_v2_clean.jsonl"
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes().replace(b"violin", b"cello", 1))
    with pytest.raises(ValueError, match="roster bytes differ"):
        _load_reference_roster(target)
