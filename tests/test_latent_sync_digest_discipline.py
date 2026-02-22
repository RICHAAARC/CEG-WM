"""
File purpose: 验证 latent sync 配置变化会影响 sync digest。
Module type: General module
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

from main.watermarking.geometry_chain.sync.latent_sync_template import LatentSyncTemplate


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def _build_cfg(sync_fft_bins: int) -> Dict[str, Any]:
    return {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_latent_sync": True,
                "sync_fft_bins": sync_fft_bins,
                "sync_rotation_bins": 36,
                "sync_scale_bins": 16,
            }
        },
    }


def test_sync_digest_must_change_when_sync_config_changes() -> None:
    """
    功能：sync 配置变化必须导致 sync_digest 变化。

    Sync digest must change when sync config domain changes.

    Args:
        None.

    Returns:
        None.
    """
    template = LatentSyncTemplate("geometry_latent_sync_sd3_v1", "v1", "a" * 64)
    pipeline = _Pipeline()
    latents = np.random.RandomState(21).randn(1, 4, 8, 8).astype(np.float32)

    cfg_a = _build_cfg(sync_fft_bins=16)
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b["detect"]["geometry"]["sync_fft_bins"] = 24

    result_a = template.extract_sync(pipeline, latents, cfg=cfg_a, rng=None)
    result_b = template.extract_sync(pipeline, latents, cfg=cfg_b, rng=None)

    assert result_a.status == "ok"
    assert result_b.status == "ok"
    assert result_a.sync_config_digest != result_b.sync_config_digest
    assert result_a.sync_digest != result_b.sync_digest
