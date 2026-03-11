"""
File purpose: 验证几何链失败不会改变 NP 主链判决。
Module type: General module
"""

from __future__ import annotations

from main.registries.fusion_registry import resolve_fusion_rule


def test_geo_chain_failure_does_not_change_primary_decision() -> None:
    """
    功能：在相同 content_score 下，geometry status=fail 不得改变 NP 主链判决。

    Geometry failure must not alter NP primary decision.

    Args:
        None.

    Returns:
        None.
    """
    factory = resolve_fusion_rule("fusion_neyman_pearson_v2")
    fusion_rule = factory({})

    cfg = {
        "target_fpr": 0.4,
        "rescue_band_enabled": False,
        "allow_threshold_fallback_for_tests": True,
    }
    content_evidence = {
        "status": "ok",
        "content_score": 0.7,
    }

    geometry_ok = {
        "status": "ok",
        "geo_score": 0.8,
    }
    geometry_fail = {
        "status": "absent",
        "geo_score": None,
        "geo_failure_reason": "sync_unavailable",
    }

    decision_ok = fusion_rule.fuse(cfg, content_evidence, geometry_ok)
    decision_fail = fusion_rule.fuse(cfg, content_evidence, geometry_fail)

    assert decision_ok.is_watermarked is True
    assert decision_fail.is_watermarked is True
    assert decision_ok.audit.get("np_primary_decision") == decision_fail.audit.get("np_primary_decision")
