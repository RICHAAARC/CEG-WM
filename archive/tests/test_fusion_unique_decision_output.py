"""
测试融合规则唯一决策出口

功能：
- 验证 orchestrator 返回唯一 FusionDecision，禁止二次决策或 detect 侧绕过。
- 验证 decision_status ∈ {"decided", "abstain", "error"}。
- 验证 is_watermarked ∈ {bool, None}（status dependent）。
"""

from __future__ import annotations

import pytest
from typing import Any, Dict

from main.watermarking.fusion.decision import NeumanPearsonFusionRule
from main.watermarking.fusion.interfaces import FusionDecision
from main.registries.fusion_registry import resolve_fusion_rule


@pytest.fixture
def np_fusion_rule() -> NeumanPearsonFusionRule:
    """构造 Neyman-Pearson 融合规则 v2 实例。"""
    factory = resolve_fusion_rule("fusion_neyman_pearson")
    return factory({})


def test_fusion_returns_unique_decision_instance(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：fusion.fuse() 返回恰好一个 FusionDecision，无二次决策出口。
    """
    cfg = {"target_fpr": 0.1, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "ok", "content_score": 0.7}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 验证返回类型
    assert isinstance(result, FusionDecision), "fuse must return FusionDecision"

    # (2) 验证 decision_status 取值
    assert result.decision_status in {"decided", "abstain", "error"}, \
        f"decision_status must be in {{'decided', 'abstain', 'error'}}, got {result.decision_status}"

    # (3) 验证 is_watermarked 与 decision_status 一致性
    if result.decision_status == "decided":
        assert isinstance(result.is_watermarked, bool), \
            "is_watermarked must be bool when decision_status='decided'"
    else:
        assert result.is_watermarked is None, \
            f"is_watermarked must be None when decision_status='{result.decision_status}'"


def test_fusion_no_second_decision_path(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：FusionDecision 是唯一决策结果，detect 侧不应再构建第二个决策。
    """
    cfg = {"target_fpr": 0.05, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "ok", "content_score": 0.55}
    geometry_evidence = {"status": "ok", "geo_score": 0.6}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 验证关键字段唯一性
    assert hasattr(result, "is_watermarked"), "FusionDecision must have is_watermarked"
    assert hasattr(result, "decision_status"), "FusionDecision must have decision_status"

    # (2) 验证没有多个决策字段（无 "second_is_watermarked" 或类似）
    fields = vars(result)
    decision_fields = [k for k in fields.keys() if k.startswith("is_watermarked")]
    assert len(decision_fields) == 1, \
        f"Must have exactly one is_watermarked field, got {decision_fields}"

    # (3) 验证审计信息存在但不包含冗余决策
    assert isinstance(result.audit, dict), "audit must be dict"
    assert "decision_status" in result.audit, "audit must record decision_status"
    assert result.audit.get("decision_status") == result.decision_status, \
        "audit decision_status must match FusionDecision.decision_status"


def test_fusion_absent_status_abstain(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：无内容证据时返回 decision_status="abstain"，is_watermarked=None。
    """
    cfg = {"target_fpr": 0.1, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "absent", "content_score": None}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    assert result.decision_status == "abstain", \
        f"absent content evidence must yield decision_status='abstain', got {result.decision_status}"
    assert result.is_watermarked is None, \
        "abstain status must have is_watermarked=None"


def test_fusion_mismatch_status_error(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：mismatch 故障时返回 decision_status="error"，is_watermarked=None。
    """
    cfg = {"target_fpr": 0.1, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "mismatch", "content_score": None}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    assert result.decision_status == "error", \
        f"mismatch status must yield decision_status='error', got {result.decision_status}"
    assert result.is_watermarked is None, \
        "error status must have is_watermarked=None"

    # (1) 验证审计记录故障原因
    assert "failure_reason" in result.audit, \
        "audit must record failure_reason for mismatch"
    assert result.audit.get("failure_reason") == "content_mismatch", \
        "audit must specify 'content_mismatch' as failure_reason"


def test_fusion_decided_status_bool(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：可决策时返回 decision_status="decided"，is_watermarked ∈ {True, False}。
    """
    cfg = {"target_fpr": 0.5, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "ok", "content_score": 0.7}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    assert result.decision_status == "decided", \
        f"ok content evidence must yield decision_status='decided', got {result.decision_status}"
    assert isinstance(result.is_watermarked, bool), \
        f"decided status must have is_watermarked as bool, got {type(result.is_watermarked)}"


def test_fusion_decided_evidence_summary_complete(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：决策时 evidence_summary 包含完整的分数与状态字段。
    """
    cfg = {"target_fpr": 0.5, "allow_threshold_fallback_for_tests": True}
    content_evidence = {"status": "ok", "content_score": 0.65}
    geometry_evidence = {"status": "ok", "geo_score": 0.4}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 验证必需的 evidence_summary 字段
    required_keys = {"content_score", "geometry_score", "content_status", "geometry_status", "fusion_rule_id"}
    actual_keys = set(result.evidence_summary.keys())
    assert required_keys.issubset(actual_keys), \
        f"evidence_summary must contain {required_keys}, got {actual_keys}"

    # (2) 验证分数值匹配
    assert result.evidence_summary["content_score"] == 0.65, \
        "evidence_summary.content_score must match input"
    assert result.evidence_summary["geometry_score"] == 0.4, \
        "evidence_summary.geometry_score must match input"

    # (3) 验证 fusion_rule_id 正确
    assert result.evidence_summary["fusion_rule_id"] == "fusion_neyman_pearson", \
        "evidence_summary.fusion_rule_id must match implementation"
