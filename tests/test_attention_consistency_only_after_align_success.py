"""
File purpose: 验证 attention consistency 仅在主路径成功且稳定时启用。
Module type: General module
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

from main.watermarking.geometry_chain.align_invariance_extractor import GeometryAlignInvarianceExtractor


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def _base_cfg() -> Dict[str, Any]:
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
                "align_min_inlier_ratio": 0.2,
                "align_available_max_residual_mad": 1.0,
                "align_available_max_param_variance": 1.0,
                "align_inverse_max_residual": 1.0,
                "align_template_overlap_min": 0.0,
                "align_recovered_sync_consistency_min": 0.0,
                "align_recovered_anchor_consistency_min": 0.0,
                "align_attention_consistency_min_stability": 0.2,
            }
        },
    }


def test_attention_consistency_only_after_align_success() -> None:
    """
    功能：仅当拟合成功且 fit_stability 达到阈值时，align_metrics 才允许出现 attention_consistency。

    attention_consistency must appear only after successful and stable alignment.

    Args:
        None.

    Returns:
        None.
    """
    extractor = GeometryAlignInvarianceExtractor("geometry_align_invariance_sd3_v1", "v2", "a" * 64)
    latents = np.random.RandomState(9).randn(1, 4, 8, 8).astype(np.float32)
    runtime_inputs = {"pipeline": _Pipeline(), "latents": latents}

    cfg_loose = _base_cfg()
    cfg_strict = copy.deepcopy(cfg_loose)
    cfg_strict["detect"]["geometry"]["align_attention_consistency_min_stability"] = 1.0

    result_loose = extractor.extract(cfg_loose, inputs=runtime_inputs)
    result_strict = extractor.extract(cfg_strict, inputs=runtime_inputs)

    assert result_loose.get("status") == "ok"
    assert result_strict.get("status") == "ok"

    align_metrics_loose = result_loose.get("align_metrics") or {}
    align_metrics_strict = result_strict.get("align_metrics") or {}

    assert "attention_consistency" in align_metrics_loose
    assert "attention_consistency" not in align_metrics_strict
