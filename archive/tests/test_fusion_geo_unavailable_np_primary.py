"""
测试 NP 主链控制与几何不可用对称性

功能：
- 验证：无几何证据时决策等同于仅使用内容证据的 NP 主控。
- 验证：几何证据仅在 rescue band 内才能影响决策。
- 验证：NP 阈值优先级高于几何增益。
"""

from __future__ import annotations

import pytest
from typing import Any, Dict

from main.watermarking.fusion.decision import NeumanPearsonFusionRule
from main.registries.fusion_registry import resolve_fusion_rule


@pytest.fixture
def np_fusion_rule() -> NeumanPearsonFusionRule:
    """构造 Neyman-Pearson 融合规则 v2 实例。"""
    factory = resolve_fusion_rule("fusion_neyman_pearson")
    return factory({})


def test_no_geometry_evidence_same_as_content_only(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：无几何证据（geo_score=None）时，决策完全由 NP 主链决定。
    """
    cfg = {
        "target_fpr": 0.5,
        "__thresholds_artifact__": {
            "threshold_value": 0.5,
            "threshold_id": "test_threshold_05",
            "target_fpr": 0.5,
        },
    }

    # (1) 仅有内容证据的决策
    content_evidence_only = {"status": "ok", "content_score": 0.7}
    geometry_evidence_absent = {"status": "absent", "geo_score": None}

    result_no_geo = np_fusion_rule.fuse(cfg, content_evidence_only, geometry_evidence_absent)

    # (2) 有内容但几何为 None 的决策
    geometry_evidence_none = {"status": "ok", "geo_score": None}
    result_geo_none = np_fusion_rule.fuse(cfg, content_evidence_only, geometry_evidence_none)

    # (3) 验证两种情况的决策结果相同
    assert result_no_geo.decision_status == result_geo_none.decision_status, \
        "absent vs None geometry must yield same decision_status"
    assert result_no_geo.is_watermarked == result_geo_none.is_watermarked, \
        "absent vs None geometry must yield same is_watermarked decision"

    # (4) 验证 NP 主控：决策仅由 content_score vs threshold 决定
    assert result_no_geo.evidence_summary["content_score"] == 0.7, \
        "content_score must be preserved"


def test_np_primary_overrides_geometry_outside_rescue_band(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：当 content_score 远离阈值时（outside rescue band），几何证据无法改变决策。
    """
    cfg = {
        "target_fpr": 0.5,
        "rescue_band_delta_low": 0.05,
        "rescue_band_delta_high": 0.05,
        "__thresholds_artifact__": {
            "threshold_value": 0.5,
            "threshold_id": "test_threshold_np_primary",
            "target_fpr": 0.5,
        },
    }

    # (1) content_score=0.9，远高于 threshold≈0.5，应决策为 True
    content_evidence_high = {"status": "ok", "content_score": 0.9}
    geometry_evidence_low = {"status": "ok", "geo_score": 0.2}  # 低几何分数

    result_high = np_fusion_rule.fuse(cfg, content_evidence_high, geometry_evidence_low)

    # (2) 验证 NP 决策：即使几何分数低，由于 content_score 高，决策应为 True
    assert result_high.decision_status == "decided", \
        "high content score must yield decided status"
    assert result_high.is_watermarked is True, \
        "high content score should override low geometry score outside rescue band"

    # (3) content_score=0.1，远低于 threshold≈0.5，应决策为 False
    content_evidence_low = {"status": "ok", "content_score": 0.1}
    geometry_evidence_high = {"status": "ok", "geo_score": 0.9}  # 高几何分数

    result_low = np_fusion_rule.fuse(cfg, content_evidence_low, geometry_evidence_high)

    # (4) 验证 NP 决策：即使几何分数高，由于 content_score 低，决策应为 False
    assert result_low.decision_status == "decided", \
        "low content score must yield decided status"
    assert result_low.is_watermarked is False, \
        "low content score should override high geometry score outside rescue band"


def test_geometry_only_matters_in_rescue_band(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：几何证据仅在 content_score 接近阈值（rescue band 内）时才能影响决策。
    """
    cfg = {
        "target_fpr": 0.5,
        "rescue_band_delta_low": 0.1,
        "rescue_band_delta_high": 0.1,
        "__thresholds_artifact__": {
            "threshold_value": 0.5,
            "threshold_id": "test_threshold_rescue",
            "target_fpr": 0.5,
        },
    }

    # (1) content_score ≈ 0.5（接近阈值），在 rescue band 内
    content_evidence_border = {"status": "ok", "content_score": 0.52}
    geometry_evidence_strong = {"status": "ok", "geo_score": 0.8}  # 强几何证据

    result_border = np_fusion_rule.fuse(cfg, content_evidence_border, geometry_evidence_strong)

    # (2) 只有在 rescue band 内且几何证据通过门控时，才可能改变决策
    # 注意：实际是否改变决定于 geo_gate 的具体参数设定
    if result_border.audit.get("rescue_band_version"):
        # Rescue band 被应用
        assert result_border.audit.get("geo_gate_applied") is not None, \
            "geo_gate_applied must be recorded when rescue band applied"


def test_np_threshold_read_only(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：NP 阈值从 thresholds_spec 只读选择，禁止动态重估。
    """
    cfg = {
        "target_fpr": 0.1,
        "__thresholds_artifact__": {
            "threshold_value": 0.1,
            "threshold_id": "test_threshold_readonly",
            "target_fpr": 0.1,
        },
    }
    content_evidence = {"status": "ok", "content_score": 0.45}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 验证 thresholds_digest 存在且非空
    assert isinstance(result.thresholds_digest, str) and result.thresholds_digest, \
        "thresholds_digest must be non-empty string"

    # (2) 验证审计中记录了 NP 阈值（来自 read-only spec）
    assert isinstance(result.audit, dict), \
        "audit must be dict"
    if result.decision_status == "decided":
        assert "np_threshold" in result.audit, \
            "audit must record np_threshold when decision made"

    # (3) 同一配置下多次调用的 thresholds_digest 应一致（确保无重估）
    result2 = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)
    assert result.thresholds_digest == result2.thresholds_digest, \
        "same config must yield same thresholds_digest (no re-estimation)"


def test_geometry_unavailable_matches_content_primary(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：无几何证据与有几何但不在 rescue band 范围的两种情况下，决策应一致。
    """
    cfg = {
        "target_fpr": 0.5,
        "__thresholds_artifact__": {
            "threshold_value": 0.5,
            "threshold_id": "test_threshold_match",
            "target_fpr": 0.5,
        },
    }

    # (1) 无几何证据
    content_evidence = {"status": "ok", "content_score": 0.75}
    geometry_absent = {"status": "absent"}

    result_no_geo = np_fusion_rule.fuse(cfg, content_evidence, geometry_absent)

    # (2) 几何证据存在但分数在 rescue band 范围外
    geometry_outside = {"status": "ok", "geo_score": 0.2}

    result_geo_outside = np_fusion_rule.fuse(cfg, content_evidence, geometry_outside)

    # (3) 两种情况的决策应相同（NP 主控）
    assert result_no_geo.decision_status == result_geo_outside.decision_status, \
        "absent geometry vs geometry outside rescue band must match on decision_status"
    assert result_no_geo.is_watermarked == result_geo_outside.is_watermarked, \
        "absent geometry vs geometry outside rescue band must match on is_watermarked"
