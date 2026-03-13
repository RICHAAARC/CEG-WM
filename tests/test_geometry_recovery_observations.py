"""
File purpose: 几何恢复观测与闭环一致性回归测试。
Module type: General module
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

from main.watermarking.geometry_chain.align_invariance_extractor import GeometryAlignInvarianceExtractor
from main.watermarking.geometry_chain.attention_anchor_extractor import AttentionAnchorExtractor
from main.watermarking.geometry_chain.sync.latent_sync_template import LatentSyncTemplate


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def _build_geometry_cfg() -> Dict[str, Any]:
    return {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "enable_latent_sync": True,
                "enable_align_invariance": True,
                "align_model_type": "similarity",
                "align_min_inlier_ratio": 0.05,
                "align_available_max_residual_mad": 1.0,
                "align_available_max_param_variance": 1.0,
                "align_inverse_max_residual": 1.0,
                "align_template_overlap_min": 0.0,
                "align_recovered_sync_consistency_min": 0.0,
                "align_recovered_anchor_consistency_min": 0.0,
            }
        },
    }


def test_sync_extract_emits_recovery_observations() -> None:
    """
    功能：sync 主链必须输出恢复用观测。 

    Verify sync extraction emits recovery observations.

    Args:
        None.

    Returns:
        None.
    """
    sync_module = LatentSyncTemplate("geometry_latent_sync_sd3_v1", "v1", "a" * 64)
    pipeline = _Pipeline()
    latents = np.random.RandomState(11).randn(1, 4, 16, 16).astype(np.float32)

    result = sync_module.extract_sync(pipeline, latents, cfg=_build_geometry_cfg(), rng=None)

    assert result.status == "ok"
    assert isinstance(result.sync_observations, dict)
    observed_sync_peaks = result.sync_observations.get("observed_sync_peaks")
    assert isinstance(observed_sync_peaks, list)
    assert len(observed_sync_peaks) > 0
    assert isinstance(result.sync_observations.get("template_support_points"), list)
    quality_components = result.sync_quality_metrics.get("quality_components")
    assert isinstance(quality_components, dict)
    assert quality_components.get("version") == "latent_sync_quality_components"


def test_anchor_extract_emits_recovery_observations() -> None:
    """
    功能：anchor 主链必须输出候选与匹配观测。 

    Verify anchor extraction emits candidate and match observations.

    Args:
        None.

    Returns:
        None.
    """
    extractor = AttentionAnchorExtractor("geometry_attention_anchor_sd3_v1", "v1", "a" * 64)
    pipeline = _Pipeline()
    latents = np.random.RandomState(13).randn(1, 4, 16, 16).astype(np.float32)

    result = extractor.extract_anchors(pipeline, latents, cfg=_build_geometry_cfg(), rng=None)

    assert result.status == "ok"
    assert isinstance(result.anchor_observations, dict)
    observed_anchor_candidates = result.anchor_observations.get("observed_anchor_candidates")
    anchor_match_candidates = result.anchor_observations.get("anchor_match_candidates")
    assert isinstance(observed_anchor_candidates, list)
    assert len(observed_anchor_candidates) > 0
    assert isinstance(anchor_match_candidates, list)
    assert len(anchor_match_candidates) > 0
    assert "candidate_confidence_mean" in result.stability_metrics
    assert "visible_candidate_count" in result.stability_metrics


def test_align_extract_emits_observed_correspondence_and_recovery_consistency() -> None:
    """
    功能：align 主链必须写出观测对应与恢复一致性指标。 

    Verify align extraction records observed correspondences and recovery consistency metrics.

    Args:
        None.

    Returns:
        None.
    """
    extractor = GeometryAlignInvarianceExtractor("geometry_align_invariance_sd3_v1", "v2", "a" * 64)
    latents = np.random.RandomState(17).randn(1, 4, 16, 16).astype(np.float32)

    evidence = extractor.extract(_build_geometry_cfg(), inputs={"pipeline": _Pipeline(), "latents": latents})

    assert evidence.get("status") == "ok"
    observed_correspondences = evidence.get("observed_correspondences")
    assert isinstance(observed_correspondences, dict)
    items = observed_correspondences.get("items")
    assert isinstance(items, list)
    assert len(items) > 0

    summary = evidence.get("observed_correspondence_summary")
    assert isinstance(summary, dict)
    assert int(summary.get("count", 0)) >= int(summary.get("visible_count", 0)) >= 0

    align_metrics = evidence.get("align_metrics")
    assert isinstance(align_metrics, dict)
    assert "template_overlap_consistency" in align_metrics
    assert "recovered_sync_consistency" in align_metrics
    assert "recovered_anchor_consistency" in align_metrics
    assert align_metrics.get("observed_correspondence_count", 0) >= 1


def test_attestation_conditioned_recovery_changes_geometry_digests() -> None:
    """
    功能：事件级 attestation 条件化必须进入 geometry recovery 摘要。 

    Verify event-level attestation conditioning affects geometry recovery digests.

    Args:
        None.

    Returns:
        None.
    """
    extractor = GeometryAlignInvarianceExtractor("geometry_align_invariance_sd3_v1", "v2", "a" * 64)
    latents = np.random.RandomState(19).randn(1, 4, 16, 16).astype(np.float32)
    cfg_a = _build_geometry_cfg()
    cfg_b = copy.deepcopy(cfg_a)
    cfg_a["attestation_runtime"] = {"event_binding_digest": "1" * 64, "geo_anchor_seed": 7}
    cfg_b["attestation_runtime"] = {"event_binding_digest": "2" * 64, "geo_anchor_seed": 7}

    result_a = extractor.extract(cfg_a, inputs={"pipeline": _Pipeline(), "latents": latents})
    result_b = extractor.extract(cfg_b, inputs={"pipeline": _Pipeline(), "latents": latents})

    assert result_a.get("status") == "fail"
    assert result_b.get("status") == "fail"
    assert result_a.get("geo_score") is None
    assert result_b.get("geo_score") is None
    assert result_a.get("anchor_digest") != result_b.get("anchor_digest")
    assert result_a.get("sync_digest") != result_b.get("sync_digest")
    assert isinstance(result_a.get("anchor_observations"), dict)
    assert isinstance(result_b.get("anchor_observations"), dict)