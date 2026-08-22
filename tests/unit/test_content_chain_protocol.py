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
    assert analysis["asset_id"] == "facebook/dinov2-small"
    assert analysis["attention_implementation"] == "eager"
    assert analysis["tile_grid"] == (4, 4)
    assert analysis["probe_evaluations_per_unit"] == 32
    identities = protocol.config["method_identities"]
    assert identities["hf_base_carrier_method_id"] == "hf_tail_rademacher_v1"
    assert identities["lf_base_carrier_method_id"] == "lf_shell_balanced_blocks_v2"
    assert identities["evaluated_candidate_id"] == "content_adaptive_dual_branch_clean_v1"
    assert protocol.config["budget"]["single_shared_budget_not_per_branch"] is True
    assert protocol.config["execution_flow"]["fixed_records"] == 16
    assert protocol.config["execution_flow"]["flat_score_field_rule"].endswith("StageARecord_v1")
    assert protocol.config["detection_access"]["joint_score"] == "min(s_LF,s_HF)"


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
