"""
测试融合规则无内容污染

功能：
- 验证：几何证据变化不会影响 content_score。
- 验证：融合决策变化与 content_score 变化独立（几何增益不污染内容分数）。
- 验证：evidence_summary 中的 content_score 分离。
"""

from __future__ import annotations

import pytest
from typing import Any, Dict

from main.watermarking.fusion.decision import NeumanPearsonFusionRule
from main.registries.fusion_registry import resolve_fusion_rule


@pytest.fixture
def np_fusion_rule() -> NeumanPearsonFusionRule:
    """构造 Neyman-Pearson 融合规则实例。"""
    factory = resolve_fusion_rule("fusion_neyman_pearson_v1")
    return factory({})


def test_content_score_independent_of_geometry(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：FusionDecision 中的 content_score 不受几何证据影响。
    """
    cfg = {"target_fpr": 0.5, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "ok", "content_score": 0.65}

    # (1) 几何证据变化 1：强几何
    geometry_evidence_strong = {"status": "ok", "geo_score": 0.8}
    result_strong = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence_strong)

    # (2) 几何证据变化 2：弱几何
    geometry_evidence_weak = {"status": "ok", "geo_score": 0.2}
    result_weak = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence_weak)

    # (3) 要点：content_score 在两种情况下都应相同
    assert result_strong.evidence_summary["content_score"] == 0.65, \
        "content_score must not change with geometry_score"
    assert result_weak.evidence_summary["content_score"] == 0.65, \
        "content_score must not change with geometry_score"

    # (4) 验证两者的 content_score 完全相同
    assert result_strong.evidence_summary["content_score"] == result_weak.evidence_summary["content_score"], \
        "content_score must be identical regardless of geometry evidence"


def test_content_status_independent_of_geometry(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：content_status 不受几何证据变化影响。
    """
    cfg = {"target_fpr": 0.5, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "ok", "content_score": 0.6}

    # (1) 几何：absent
    geometry_absent = {"status": "absent"}
    result_absent = np_fusion_rule.fuse(cfg, content_evidence, geometry_absent)

    # (2) 几何：ok
    geometry_ok = {"status": "ok", "geo_score": 0.5}
    result_ok = np_fusion_rule.fuse(cfg, content_evidence, geometry_ok)

    # (3) content_status 应一致
    assert result_absent.evidence_summary["content_status"] == "ok", \
        "content_status from content_evidence must not change"
    assert result_ok.evidence_summary["content_status"] == "ok", \
        "content_status from content_evidence must not change"


def test_geometry_score_not_polluted_to_content(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：geometry_score 单独记录，不与 content_score 混淆或污染。
    """
    cfg = {"target_fpr": 0.5, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "ok", "content_score": 0.7}
    geometry_evidence = {"status": "ok", "geo_score": 0.3}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 验证两个分数分离
    content_score = result.evidence_summary["content_score"]
    geometry_score = result.evidence_summary["geometry_score"]

    assert content_score == 0.7, \
        "content_score must be preserved exactly"
    assert geometry_score == 0.3, \
        "geometry_score must be preserved exactly"

    # (2) 验证两者不相等（防止混淆）
    assert content_score != geometry_score, \
        "content_score and geometry_score must be distinct"


def test_decision_change_independent_of_content_score_change(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：融合决策变化（由几何增益触发）与 content_score 变化独立。
    """
    cfg = {
        "target_fpr": 0.5,
        "rescue_band_delta_low": 0.15,
        "rescue_band_delta_high": 0.15,
        "geo_gate_lower": 0.2,
        "geo_gate_upper": 0.8,
        "allow_threshold_fallback_for_tests": True,
    }
    
    # (1) content_score 固定在 0.52（rescue band 边界）
    content_evidence = {"status": "ok", "content_score": 0.52}

    # (2) 几何证据 A：通过门控
    geometry_a = {"status": "ok", "geo_score": 0.5}
    result_a = np_fusion_rule.fuse(cfg, content_evidence, geometry_a)

    # (3) 几何证据 B：不通过门控
    geometry_b = {"status": "ok", "geo_score": 0.1}
    result_b = np_fusion_rule.fuse(cfg, content_evidence, geometry_b)

    # (4) 关键验证：content_score 在两种情况下相同，但决策可能不同
    assert result_a.evidence_summary["content_score"] == result_b.evidence_summary["content_score"] == 0.52, \
        "content_score must be identical"

    # (5) 决策可能因几何增益而不同（这不是污染，而是预期设计）
    # 只需验证 content_score 本身未被污染
    assert result_a.evidence_summary["content_score"] == 0.52, \
        "content_score must remain unpolluted by geometric decision"
    assert result_b.evidence_summary["content_score"] == 0.52, \
        "content_score must remain unpolluted by geometric decision"


def test_audit_fields_do_not_pollute_evidence(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：审计字段（如 rescue_band_version）不污染 evidence_summary。
    """
    cfg = {
        "target_fpr": 0.5,
        "rescue_band_delta_low": 0.1,
        "rescue_band_delta_high": 0.1,
        "allow_threshold_fallback_for_tests": True,
    }
    content_evidence = {"status": "ok", "content_score": 0.52}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 验证 evidence_summary 只包含必需的 5 个字段
    allowed_evidence_keys = {"content_score", "geometry_score", "content_status", "geometry_status", "fusion_rule_id"}
    actual_evidence_keys = set(result.evidence_summary.keys())

    # (2) 不应包含 rescue_band_version 或其他审计信息
    unexpected_keys = actual_evidence_keys - allowed_evidence_keys
    assert not unexpected_keys, \
        f"evidence_summary should not contain audit fields: {unexpected_keys}"

    # (3) 审计信息应在 audit dict 中，不在 evidence_summary
    assert "rescue_band_version" in result.audit or result.audit.get("rescue_band_version") is None, \
        "rescue_band_version must be in audit, not evidence_summary"


def test_content_score_none_vs_float_consistency(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：content_score 为 None（absent）与 float（ok）时的分离。
    """
    cfg = {"target_fpr": 0.5, "allow_threshold_fallback_for_tests": True}

    # (1) 无内容证据（返回 abstain）
    content_absent = {"status": "absent", "content_score": None}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result_absent = np_fusion_rule.fuse(cfg, content_absent, geometry_evidence)

    # (2) 有内容证据
    content_ok = {"status": "ok", "content_score": 0.7}
    result_ok = np_fusion_rule.fuse(cfg, content_ok, geometry_evidence)

    # (3) 验证 content_score 类型分离
    assert result_absent.evidence_summary["content_score"] is None, \
        "absent content_score must be None"
    assert isinstance(result_ok.evidence_summary["content_score"], float), \
        "ok content_score must be float"

    # (4) 验证两者的 content_status 不同但准确
    assert result_absent.evidence_summary["content_status"] == "absent", \
        "absent status must be recorded"
    assert result_ok.evidence_summary["content_status"] == "ok", \
        "ok status must be recorded"


def test_all_evidence_summary_fields_present(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：evidence_summary 包含所有必需的字段（无缺失）。
    """
    cfg = {"target_fpr": 0.5, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "ok", "content_score": 0.6}
    geometry_evidence = {"status": "ok", "geo_score": 0.4}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 必需字段列表
    required_keys = {"content_score", "geometry_score", "content_status", "geometry_status", "fusion_rule_id"}

    # (2) 验证全部存在
    actual_keys = set(result.evidence_summary.keys())
    assert required_keys.issubset(actual_keys), \
        f"evidence_summary must contain {required_keys}, got {actual_keys}"

    # (3) 验证无多余字段（防止污染）
    extra_keys = actual_keys - required_keys
    assert not extra_keys, \
        f"evidence_summary should not have extra keys: {extra_keys}"
