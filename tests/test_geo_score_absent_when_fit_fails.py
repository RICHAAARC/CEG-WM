"""
File purpose: 验证拟合失败时 geo_score 必须缺失。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from main.watermarking.geometry_chain.align_invariance_extractor import GeometryAlignInvarianceExtractor


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def test_geo_score_absent_when_fit_fails() -> None:
    """
    功能：当可用性门槛无法通过时，几何链必须返回 fail 且不输出 geo_score。

    Geo score must be absent when fitting/availability gate fails.

    Args:
        None.

    Returns:
        None.
    """
    extractor = GeometryAlignInvarianceExtractor("geometry_align_invariance_sd3_v1", "v2", "a" * 64)
    cfg: Dict[str, Any] = {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "enable_latent_sync": True,
                "enable_align_invariance": True,
                "align_min_inlier_ratio": 0.9999,
            }
        },
    }
    latents = np.random.RandomState(7).randn(1, 4, 8, 8).astype(np.float32)
    evidence = extractor.extract(cfg, inputs={"pipeline": _Pipeline(), "latents": latents})

    assert evidence.get("status") == "fail"
    assert evidence.get("geo_score") is None
    assert isinstance(evidence.get("geo_failure_reason"), str)
