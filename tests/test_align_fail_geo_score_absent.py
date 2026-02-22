"""
File purpose: 验证几何对齐失败时 geo_score 不得产出。
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


def _build_cfg_for_fail() -> Dict[str, Any]:
    return {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "enable_latent_sync": True,
                "enable_align_invariance": True,
                "align_model_type": "similarity",
                "align_min_inlier_ratio": 0.9999,
            }
        },
    }


def test_align_fail_must_not_emit_geo_score() -> None:
    """
    功能：对齐失败时 geo_score 必须为空。

    Geo score must be absent when alignment fails.

    Args:
        None.

    Returns:
        None.
    """
    extractor = GeometryAlignInvarianceExtractor("geometry_align_invariance_sd3_v1", "v1", "a" * 64)
    cfg = _build_cfg_for_fail()
    latents = np.random.RandomState(7).randn(1, 4, 8, 8).astype(np.float32)
    evidence = extractor.extract(cfg, inputs={"pipeline": _Pipeline(), "latents": latents})

    assert evidence.get("status") == "fail"
    assert evidence.get("geo_score") is None
    assert evidence.get("geo_failure_reason") == "align_inlier_ratio_below_threshold"
