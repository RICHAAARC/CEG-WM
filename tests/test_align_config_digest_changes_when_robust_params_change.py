"""
验证 稳健拟合参数变化会影响 align 配置 digest
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


def _build_cfg() -> Dict[str, Any]:
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
                "align_min_inlier_ratio": 0.2,
                "align_robust_loss": "huber",
                "align_max_iterations": 8,
                "align_inlier_threshold": 0.22,
            }
        },
    }


def test_align_config_digest_changes_when_robust_params_change() -> None:
    """
    功能：稳健拟合参数变化必须导致 align_config_digest 和 align_trace_digest 变化。

    Robust fitting params must change align digests.

    Args:
        None.

    Returns:
        None.
    """
    extractor = GeometryAlignInvarianceExtractor("geometry_align_invariance_sd3_v1", "v2", "a" * 64)
    latents = np.random.RandomState(42).randn(1, 4, 8, 8).astype(np.float32)
    runtime_inputs = {"pipeline": _Pipeline(), "latents": latents}

    cfg_a = _build_cfg()
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b["detect"]["geometry"]["align_robust_loss"] = "tukey"
    cfg_b["detect"]["geometry"]["align_max_iterations"] = 12

    result_a = extractor.extract(cfg_a, inputs=runtime_inputs)
    result_b = extractor.extract(cfg_b, inputs=runtime_inputs)

    assert result_a.get("status") == "ok"
    assert result_b.get("status") == "ok"
    assert result_a.get("align_config_digest") != result_b.get("align_config_digest")
    assert result_a.get("align_trace_digest") != result_b.get("align_trace_digest")
