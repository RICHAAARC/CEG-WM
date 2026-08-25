from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.protocol.content_chain_v7 import (
    CONTENT_V7_PROTOCOL_ID,
    V7_DEVELOPMENT_MANIFEST_SHA256,
    V7_EVALUATION_1_MANIFEST_SHA256,
    V7_EVALUATION_2_MANIFEST_SHA256,
    load_content_v7_formal_protocol,
)

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_v7_formal_contract_binds_dev32_then_two_exact_independent_rosters() -> None:
    protocol = load_content_v7_formal_protocol(_ROOT)
    assert protocol.protocol_id == CONTENT_V7_PROTOCOL_ID
    assert len(protocol.data.development) == 32
    assert tuple(len(item.roster) for item in protocol.evaluations) == (8, 8)
    assert protocol.data.development_manifest_sha256 == V7_DEVELOPMENT_MANIFEST_SHA256
    assert protocol.data.evaluation_manifest_sha256s == (
        V7_EVALUATION_1_MANIFEST_SHA256,
        V7_EVALUATION_2_MANIFEST_SHA256,
    )
    assert tuple(unit.seed for unit in protocol.data.development) == tuple(
        range(2026082400, 2026082432)
    )
    assert protocol.data.development[0].unit_id == "content-v6-iss-dev-0001"
    assert protocol.data.development[-1].unit_id == "content-v6-iss-dev-0032"
    assert protocol.evaluations[1].roster[0].unit_id == "content-v6-iss-eval-0001"
    assert protocol.evaluations[1].roster[-1].unit_id == "content-v6-iss-eval-0008"
    assert len(protocol.protocol_digest) == 64
    assert protocol.evaluations[0].protocol_digest != protocol.evaluations[1].protocol_digest


@pytest.mark.unit
def test_v7_formal_flow_is_fit_first_nonresumable_and_never_pools() -> None:
    protocol = load_content_v7_formal_protocol(_ROOT)
    flow = protocol.config["execution_flow"]
    assert flow["phase_order"] == (
        "fit_and_publish_asset", "evaluation_01", "evaluation_02", "terminal"
    )
    assert tuple(
        item["roster_sha256"] for item in flow["evaluation_invocations"]
    ) == (V7_EVALUATION_1_MANIFEST_SHA256, V7_EVALUATION_2_MANIFEST_SHA256)
    assert flow["independent_failures_denominators_and_gates"] is True
    assert flow["pooling_allowed"] is False
    assert flow["outcome_conditioned_control_allowed"] is False
    assert flow["cross_invocation_resume_allowed"] is False
    assert flow["implicit_resume_allowed"] is False
    assert flow["terminal_result_count"] == 2
    assert flow["terminal_reporting"] == {
        "evaluation_01": {"fixed_units": 8, "fixed_records": 16},
        "evaluation_02": {"fixed_units": 8, "fixed_records": 16},
    }
    assert flow["cross_cohort_conjunction_allowed"] is False
    assert flow["combined_result_allowed"] is False
    assert "terminal_joint_rule" not in flow
    rule = protocol.config["decision_rule"]
    assert rule["fixed_units_per_invocation"] == 8
    assert rule["fixed_records_per_invocation"] == 16
    assert "scientific_outcome_requires_both_independent_rc0" not in rule
    assert "joint_result" not in rule
    assert "asset_sha256" not in protocol.config["iss_fit"]


@pytest.mark.unit
def test_v7_manifest_loader_fails_closed_on_exact_byte_drift(tmp_path: Path) -> None:
    source = _ROOT / "configs" / "content_chain"
    target = tmp_path / "configs" / "content_chain"
    target.mkdir(parents=True)
    names = (
        "content_v7_ordinary_iss_formal_initial_v1.json",
        "content_v7_ordinary_iss_development_v1.jsonl",
        "content_v6_iss_clean.jsonl",
        "content_v3_clean_v1.json",
        "content_adaptive_dual_branch_v2_clean.jsonl",
    )
    for name in names:
        (target / name).write_bytes((source / name).read_bytes())
    rows = (target / "content_v7_ordinary_iss_development_v1.jsonl").read_text().splitlines()
    changed = json.loads(rows[0])
    changed["seed"] += 1
    rows[0] = json.dumps(changed, separators=(",", ":"))
    (target / "content_v7_ordinary_iss_development_v1.jsonl").write_text(
        "\n".join(rows) + "\n"
    )
    with pytest.raises(ValueError, match="development manifest bytes differ"):
        load_content_v7_formal_protocol(tmp_path)
    assert hashlib.sha256(
        (source / "content_v7_ordinary_iss_development_v1.jsonl").read_bytes()
    ).hexdigest() == V7_DEVELOPMENT_MANIFEST_SHA256
