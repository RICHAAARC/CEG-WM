"""
File purpose: 几何恢复观测与闭环一致性回归测试。
Module type: General module
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

from main.watermarking.geometry_chain.align_invariance_extractor import GeometryAlignInvarianceExtractor, GeometryAligner
from main.watermarking.geometry_chain.attention_anchor_extractor import (
    ATTENTION_ANCHOR_EXTRACTOR_ID,
    ATTENTION_ANCHOR_EXTRACTOR_VERSION,
    AttentionAnchorExtractor,
)
from main.watermarking.geometry_chain.sync.latent_sync_template import (
    GEOMETRY_LATENT_SYNC_SD3_ID,
    GEOMETRY_LATENT_SYNC_SD3_VERSION,
    LatentSyncTemplate,
)


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


def _apply_similarity(point: Dict[str, float], rotation_degree: float, scale_factor: float, translation_x: float, translation_y: float) -> Dict[str, float]:
    theta = np.deg2rad(rotation_degree)
    cos_value = float(np.cos(theta))
    sin_value = float(np.sin(theta))
    x_value = float(point["x"])
    y_value = float(point["y"])
    return {
        "x": scale_factor * (cos_value * x_value - sin_value * y_value) + translation_x,
        "y": scale_factor * (sin_value * x_value + cos_value * y_value) + translation_y,
    }


def _build_recovery_sync_observations(rotation_degree: float, scale_factor: float, translation_x: float, translation_y: float) -> Dict[str, Any]:
    reference_points = [
        {"x": -0.55, "y": -0.35},
        {"x": 0.45, "y": -0.20},
        {"x": 0.10, "y": 0.50},
    ]
    observed_sync_peaks = []
    template_support_points = []
    for index, reference_point in enumerate(reference_points):
        observed_point = _apply_similarity(reference_point, rotation_degree, scale_factor, translation_x, translation_y)
        observed_sync_peaks.append(
            {
                "rank": index,
                "normalized_coord": observed_point,
                "peak_strength": 4.0 - 0.5 * index,
                "local_contrast": 3.2 - 0.2 * index,
                "confidence": 0.94 - 0.04 * index,
                "visibility": True,
            }
        )
        template_support_points.append(
            {
                "rank": index,
                "reference_coord": reference_point,
                "visibility": True,
                "confidence": 1.0,
            }
        )
    return {
        "observed_sync_peaks": observed_sync_peaks,
        "template_support_points": template_support_points,
    }


def _build_anchor_observations(rotation_degree: float, scale_factor: float, translation_x: float, translation_y: float) -> Dict[str, Any]:
    reference_points = [
        {"x": -0.45, "y": 0.10},
        {"x": 0.25, "y": 0.30},
        {"x": -0.15, "y": -0.60},
    ]
    observed_anchor_candidates = []
    anchor_match_candidates = []
    for index, reference_point in enumerate(reference_points):
        observed_point = _apply_similarity(reference_point, rotation_degree, scale_factor, translation_x, translation_y)
        observed_anchor_candidates.append(
            {
                "reference_coord": reference_point,
                "observed_coord": observed_point,
                "confidence": 0.91 - 0.05 * index,
                "visibility": True,
            }
        )
        anchor_match_candidates.append(
            {
                "source_reference_coord": reference_point,
                "observed_target_coord": observed_point,
                "match_score": 0.88 - 0.04 * index,
                "visibility": True,
            }
        )
    return {
        "observed_anchor_candidates": observed_anchor_candidates,
        "anchor_match_candidates": anchor_match_candidates,
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
    sync_module = LatentSyncTemplate(GEOMETRY_LATENT_SYNC_SD3_ID, GEOMETRY_LATENT_SYNC_SD3_VERSION, "a" * 64)
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
    extractor = AttentionAnchorExtractor(ATTENTION_ANCHOR_EXTRACTOR_ID, ATTENTION_ANCHOR_EXTRACTOR_VERSION, "a" * 64)
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
    assert "coarse_registration_success" in align_metrics
    assert "coarse_registration_confidence" in align_metrics
    assert "recovered_sync_match_score" in align_metrics
    assert "sync_parameter_agreement" in align_metrics
    assert align_metrics.get("observed_correspondence_count", 0) >= 1


def test_recovered_sync_consistency_comes_from_recovered_observations() -> None:
    """
    功能：recovered_sync_consistency 必须来自恢复域真实观测，而不是参数代理。 

    Verify recovered_sync_consistency is computed from recovered-domain sync
    observations rather than from rotation or scale parameter agreement.

    Args:
        None.

    Returns:
        None.
    """
    aligner = GeometryAligner()
    cfg = _build_geometry_cfg()
    cfg["detect"]["geometry"]["align_inverse_max_residual"] = 0.4

    correspondences = [
        {"src": [-0.55, -0.35], "dst": [-0.55, -0.35], "weight": 0.95, "visibility": True, "source": "sync_peak", "source_type": "sync"},
        {"src": [0.45, -0.20], "dst": [0.45, -0.20], "weight": 0.90, "visibility": True, "source": "sync_peak", "source_type": "sync"},
        {"src": [0.10, 0.50], "dst": [0.10, 0.50], "weight": 0.88, "visibility": True, "source": "anchor_relation", "source_type": "anchor"},
    ]
    transform = {
        "model_type": "similarity",
        "transform_quantized": {
            "rotation_degree_q": 0.0,
            "scale_factor_q": 1.0,
            "translation_x_q": 0.0,
            "translation_y_q": 0.0,
        },
        "support": {
            "rotation_bins": 36,
            "scale_bins": 16,
        },
    }
    sync_data = {
        "sync_metrics": {
            "rotation_bin": 9,
            "scale_bin": 15,
            "rotation_bins": 36,
            "scale_bins": 16,
        },
        "sync_observations": _build_recovery_sync_observations(0.0, 1.0, 0.0, 0.0),
    }

    recovery_validation = aligner._validate_inverse_recovery(correspondences, transform, cfg, {}, sync_data)

    assert recovery_validation["sync_parameter_agreement"] < 0.4
    assert recovery_validation["recovered_sync_match_score"] > 0.8
    assert recovery_validation["recovered_sync_support_overlap"] > 0.8
    assert recovery_validation["recovered_sync_consistency"] > 0.8
    assert recovery_validation["recovered_sync_consistency"] > recovery_validation["sync_parameter_agreement"]


def test_coarse_registration_is_observation_driven() -> None:
    """
    功能：coarse registration 必须由观测对应驱动，而不是 anchor_concentration 代理。 

    Verify coarse registration is driven by observed correspondences and remains
    invariant to unrelated anchor concentration proxy changes.

    Args:
        None.

    Returns:
        None.
    """
    aligner = GeometryAligner()
    cfg = _build_geometry_cfg()
    rotation_degree = 18.0
    scale_factor = 1.08
    translation_x = 0.14
    translation_y = -0.09
    sync_observations = _build_recovery_sync_observations(rotation_degree, scale_factor, translation_x, translation_y)
    anchor_observations = _build_anchor_observations(rotation_degree, scale_factor, translation_x, translation_y)
    sync_data = {
        "sync_metrics": {
            "rotation_bin": 0,
            "scale_bin": 0,
        },
        "sync_observations": sync_observations,
    }
    anchor_data_a = {
        "stability_metrics": {"top1_concentration": 0.98},
        "anchor_observations": anchor_observations,
    }
    anchor_data_b = {
        "stability_metrics": {"top1_concentration": 0.02},
        "anchor_observations": anchor_observations,
    }

    correspondences_a = aligner._build_correspondences(anchor_data_a, sync_data, {}, cfg)
    coarse_a = aligner._build_initial_transform_from_observations(anchor_data_a, sync_data, correspondences_a, cfg)
    correspondences_b = aligner._build_correspondences(anchor_data_b, sync_data, {}, cfg)
    coarse_b = aligner._build_initial_transform_from_observations(anchor_data_b, sync_data, correspondences_b, cfg)

    assert coarse_a["status"] == "ok"
    assert coarse_b["status"] == "ok"
    assert coarse_a["coarse_hypothesis_summary"]["coarse_registration_success"] is True
    assert coarse_b["coarse_hypothesis_summary"]["coarse_registration_success"] is True
    assert coarse_a["transform_quantized"] == coarse_b["transform_quantized"]
    assert abs(float(coarse_a["transform_quantized"]["translation_x_q"]) - translation_x) < 0.08
    assert abs(float(coarse_a["transform_quantized"]["translation_y_q"]) - translation_y) < 0.08


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