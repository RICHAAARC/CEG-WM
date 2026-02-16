"""
File purpose: 验证几何链失败/缺失不会污染 NP 主链判决。
Module type: General module
"""

from __future__ import annotations

from main.registries.fusion_registry import resolve_fusion_rule


def test_geo_failure_must_not_flip_primary_decision() -> None:
    """
    功能：几何链 absent/fail/mismatch 不得污染 NP 主判决语义。

    Geometry chain absent/fail/mismatch must not contaminate NP primary decision.

    Args:
        None.

    Returns:
        None.
    """
    factory = resolve_fusion_rule("fusion_neyman_pearson_v1")
    fusion_rule = factory({})

    cfg = {
        "target_fpr": 0.5,
        "allow_threshold_fallback_for_tests": True,
        "rescue_band_delta_low": 0.05,
        "rescue_band_delta_high": 0.05,
        "geo_gate_lower": 0.3,
        "geo_gate_upper": 0.7,
    }
    content_evidence = {
        "status": "ok",
        "content_score": 0.48,
    }

    geometry_ok = {"status": "ok", "geo_score": 0.6}
    geometry_absent = {"status": "absent", "geo_score": None}
    geometry_fail = {"status": "fail", "geo_score": None, "geo_failure_reason": "sync_failed"}
    geometry_mismatch = {"status": "mismatch", "geo_score": None, "geo_failure_reason": "digest_mismatch"}

    decision_ok = fusion_rule.fuse(cfg, content_evidence, geometry_ok)
    decision_absent = fusion_rule.fuse(cfg, content_evidence, geometry_absent)
    decision_fail = fusion_rule.fuse(cfg, content_evidence, geometry_fail)
    decision_mismatch = fusion_rule.fuse(cfg, content_evidence, geometry_mismatch)

    assert decision_ok.audit.get("np_primary_decision") is False
    assert decision_ok.is_watermarked is True

    assert decision_absent.audit.get("np_primary_decision") is False
    assert decision_absent.is_watermarked is False
    assert decision_absent.audit.get("rescue_triggered") is False

    assert decision_fail.decision_status == "error"
    assert decision_fail.audit.get("failure_reason") == "geometry_fail"

    assert decision_mismatch.decision_status == "error"
    assert decision_mismatch.audit.get("failure_reason") == "geometry_mismatch"
