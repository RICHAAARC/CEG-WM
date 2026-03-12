"""
File purpose: rescue 可信度门控回归测试。
Module type: General module
"""

from main.registries.fusion_registry import resolve_fusion_rule


def test_proxy_geometry_blocks_rescue_when_sync_not_ok() -> None:
    fusion = resolve_fusion_rule("fusion_neyman_pearson_v1")({})
    cfg = {
        "target_fpr": 0.5,
        "rescue_band_delta_low": 0.1,
        "rescue_band_delta_high": 0.1,
        "geo_gate_lower": 0.4,
        "geo_gate_upper": 0.9,
        "allow_threshold_fallback_for_tests": True,
    }
    content_evidence = {"status": "ok", "content_score": 0.45}
    geometry_evidence = {
        "status": "ok",
        "geo_score": 0.6,
        "anchor_evidence_level": "proxy",
        "sync_status": "absent",
    }

    decision = fusion.fuse(cfg, content_evidence, geometry_evidence)

    assert decision.is_watermarked is False
    assert decision.audit.get("rescue_triggered") is False
    assert decision.audit.get("rescue_blocked_reason") == "proxy_geometry_not_trusted"
    assert decision.audit.get("rescue_sync_status_normalized") == "absent"
