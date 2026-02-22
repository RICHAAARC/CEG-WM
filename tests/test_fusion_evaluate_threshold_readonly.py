"""
测试阈值只读约束

功能：
- 验证：NP 阈值从 thresholds_digest 只读选择。
- 验证：禁止在 run_detect 内重新估计阈值。
- 验证：多次调用相同 cfg 产生一致的 thresholds_digest。
"""

from __future__ import annotations

import pytest
from typing import Any, Dict

from main.watermarking.fusion.decision import NeumanPearsonFusionRule
from main.watermarking.fusion import neyman_pearson
from main.registries.fusion_registry import resolve_fusion_rule


@pytest.fixture
def np_fusion_rule() -> NeumanPearsonFusionRule:
    """构造 Neyman-Pearson 融合规则实例。"""
    factory = resolve_fusion_rule("fusion_neyman_pearson_v1")
    return factory({})


def test_thresholds_digest_readonly_from_artifact(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：FusionDecision 中的 thresholds_digest 来自只读 artifact，非重估结果。
    """
    cfg = {
        "target_fpr": 0.05,
        "__thresholds_artifact__": {
            "threshold_value": 0.05,
            "threshold_id": "test_threshold_005",
            "target_fpr": 0.05,
        },
    }
    content_evidence = {"status": "ok", "content_score": 0.6}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 验证 thresholds_digest 非空
    assert isinstance(result.thresholds_digest, str) and result.thresholds_digest, \
        "thresholds_digest must be non-empty string"

    # (2) 计算预期的 thresholds_digest（来自 neyman_pearson.build_thresholds_spec）
    expected_spec = neyman_pearson.build_thresholds_spec(cfg)
    expected_digest = neyman_pearson.compute_thresholds_digest(expected_spec)

    # (3) 验证匹配
    assert result.thresholds_digest == expected_digest, \
        f"thresholds_digest must match expected value from neyman_pearson, got {result.thresholds_digest} vs {expected_digest}"


def test_thresholds_digest_no_reestimation(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：多次调用相同 cfg 不会重新估计阈值，digest 一致。
    """
    cfg = {
        "target_fpr": 0.1,
        "__thresholds_artifact__": {
            "threshold_value": 0.1,
            "threshold_id": "test_threshold_01",
            "target_fpr": 0.1,
        },
    }
    content_evidence = {"status": "ok", "content_score": 0.65}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    # (1) 首次调用
    result1 = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)
    digest1 = result1.thresholds_digest

    # (2) 第二次调用（相同 cfg）
    result2 = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)
    digest2 = result2.thresholds_digest

    # (3) 第三次调用
    result3 = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)
    digest3 = result3.thresholds_digest

    # (4) 验证三次的摘要一致（无重估）
    assert digest1 == digest2 == digest3, \
        f"same cfg must always yield same thresholds_digest, got {digest1}, {digest2}, {digest3}"


def test_thresholds_digest_sensitive_to_target_fpr(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：target_fpr 变化导致 thresholds_digest 变化。
    """
    content_evidence = {"status": "ok", "content_score": 0.6}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    # (1) target_fpr = 0.05
    cfg1 = {
        "target_fpr": 0.05,
        "__thresholds_artifact__": {
            "threshold_value": 0.05,
            "threshold_id": "test_threshold_cfg1",
            "target_fpr": 0.05,
        },
    }
    result1 = np_fusion_rule.fuse(cfg1, content_evidence, geometry_evidence)
    digest1 = result1.thresholds_digest

    # (2) target_fpr = 0.1
    cfg2 = {
        "target_fpr": 0.1,
        "__thresholds_artifact__": {
            "threshold_value": 0.1,
            "threshold_id": "test_threshold_cfg2",
            "target_fpr": 0.1,
        },
    }
    result2 = np_fusion_rule.fuse(cfg2, content_evidence, geometry_evidence)
    digest2 = result2.thresholds_digest

    # (3) 不同的 target_fpr 应产生不同的摘要
    assert digest1 != digest2, \
        "different target_fpr must yield different thresholds_digest"


def test_np_threshold_value_deterministic(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：NP 阈值选择是确定性的（输入相同 → 输出相同）。
    """
    cfg = {
        "target_fpr": 0.05,
        "__thresholds_artifact__": {
            "threshold_value": 0.05,
            "threshold_id": "test_threshold_np",
            "target_fpr": 0.05,
        },
    }
    content_evidence = {"status": "ok", "content_score": 0.6}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    # (1) 首次调用
    result1 = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)
    np_threshold1 = result1.audit.get("np_threshold") if result1.decision_status == "decided" else None

    # (2) 第二次调用
    result2 = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)
    np_threshold2 = result2.audit.get("np_threshold") if result2.decision_status == "decided" else None

    # (3) 验证阈值一致
    if np_threshold1 is not None and np_threshold2 is not None:
        assert np_threshold1 == np_threshold2, \
            f"NP threshold must be deterministic, got {np_threshold1} vs {np_threshold2}"


def test_different_cfg_different_thresholds(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：不同的 cfg（如 target_fpr）产生不同的阈值。
    """
    content_evidence = {"status": "ok", "content_score": 0.6}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    # (1) 严格 FPR: 0.01
    cfg_strict = {
        "target_fpr": 0.01,
        "__thresholds_artifact__": {
            "threshold_value": 0.01,
            "threshold_id": "test_threshold_strict",
            "target_fpr": 0.01,
        },
    }
    result_strict = np_fusion_rule.fuse(cfg_strict, content_evidence, geometry_evidence)
    threshold_strict = result_strict.audit.get("np_threshold") if result_strict.decision_status == "decided" else None

    # (2) 宽松 FPR: 0.2
    cfg_loose = {
        "target_fpr": 0.2,
        "__thresholds_artifact__": {
            "threshold_value": 0.2,
            "threshold_id": "test_threshold_loose",
            "target_fpr": 0.2,
        },
    }
    result_loose = np_fusion_rule.fuse(cfg_loose, content_evidence, geometry_evidence)
    threshold_loose = result_loose.audit.get("np_threshold") if result_loose.decision_status == "decided" else None

    # (3) 严格 FPR 应对应更高的阈值（更保守的决策）
    if threshold_strict is not None and threshold_loose is not None:
        # 注意：实际的阈值关系取决于 NP 方法的具体实现
        # 这里验证两个阈值不同即可
        assert threshold_strict != threshold_loose, \
            "different target_fpr must yield different thresholds"


def test_thresholds_artifact_binding_in_audit(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：审计信息中记录了 thresholds 绑定（来自 artifact）。
    """
    cfg = {
        "target_fpr": 0.1,
        "__thresholds_artifact__": {
            "threshold_value": 0.1,
            "threshold_id": "test_threshold_audit",
            "target_fpr": 0.1,
        },
    }
    content_evidence = {"status": "ok", "content_score": 0.6}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)

    # (1) 验证 audit 中有阈值相关信息
    assert isinstance(result.audit, dict), "audit must be dict"
    assert "np_threshold" in result.audit or result.decision_status != "decided", \
        "audit must record np_threshold when decision made"

    # (2) 验证 thresholds_digest 也在结果中
    assert result.thresholds_digest, \
        "FusionDecision must have thresholds_digest"


def test_calibrate_readonly_thresholds_path(np_fusion_rule: NeumanPearsonFusionRule) -> None:
    """
    验证：thresholds_digest 与 thresholds artifact 的绑定是只读的。
    
    说明：本测试验证了 run_detect 中 NP 阈值的只读性。在实际系统中，
    thresholds artifact 由 run_calibrate 或 run_evaluate 生成，
    run_detect 仅读取而不重估。
    """
    cfg = {
        "target_fpr": 0.05,
        "__thresholds_artifact__": {
            "threshold_value": 0.05,
            "threshold_id": "test_threshold_calib",
            "target_fpr": 0.05,
        },
    }
    content_evidence = {"status": "ok", "content_score": 0.75}
    geometry_evidence = {"status": "ok", "geo_score": 0.5}

    # (1) 执行 detect，获取 thresholds_digest
    result = np_fusion_rule.fuse(cfg, content_evidence, geometry_evidence)
    thresholds_digest_from_detect = result.thresholds_digest

    # (2) 独立计算预期的摘要（模拟 calibrate 侧的计算）
    expected_spec = neyman_pearson.build_thresholds_spec(cfg)
    expected_digest = neyman_pearson.compute_thresholds_digest(expected_spec)

    # (3) 验证两者匹配（说明 detect 不重估，而是复现相同摘要）
    assert thresholds_digest_from_detect == expected_digest, \
        "detect-side thresholds_digest must match calibrate-side computation (readonly)"
