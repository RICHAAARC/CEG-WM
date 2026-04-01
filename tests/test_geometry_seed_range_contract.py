"""
文件目的：验证 latent sync 对大整数 seed 的合法归一化合同。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from main.core.time_utils import stable_seed_from_parts
from main.watermarking.geometry_chain.sync.latent_sync_template import LatentSyncTemplate


def _build_sync_cfg(seed_value: int) -> Dict[str, Any]:
    """
    功能：构造 latent sync 最小测试配置。

    Build the minimal sync config used by the seed-range regression tests.

    Args:
        seed_value: Candidate deterministic seed.

    Returns:
        Minimal config mapping for latent sync tests.
    """
    return {
        "seed": seed_value,
        "embed": {
            "geometry": {
                "enable_latent_sync": True,
                "sync_strength": 0.05,
            }
        },
        "detect": {
            "geometry": {
                "enable_latent_sync": True,
                "enabled": True,
                "sync_fft_bins": 8,
            }
        },
    }


def _build_template() -> LatentSyncTemplate:
    """
    功能：构造 latent sync 模板实例。

    Build one LatentSyncTemplate instance for direct unit testing.

    Args:
        None.

    Returns:
        LatentSyncTemplate instance.
    """
    return LatentSyncTemplate(
        impl_id="geometry_latent_sync_sd3",
        impl_version="v1",
        impl_digest="a" * 64,
    )


def test_embed_inject_accepts_large_stable_seed_value() -> None:
    """
    功能：embed latent sync 必须接受 stable_seed_from_parts 生成的大整数种子。

    Verify embed-side latent sync accepts the large integer seed produced by
    stable_seed_from_parts without raising a RandomState range error.

    Args:
        None.

    Returns:
        None.
    """
    large_seed = stable_seed_from_parts({
        "key_id": "<absent>",
        "sample_idx": 0,
        "purpose": "embed",
    })
    assert large_seed > 2 ** 32 - 1

    template = _build_template()
    latents = np.zeros((1, 4, 16, 16), dtype=np.float32)
    modified_latents, inject_trace = template.embed_inject(latents, _build_sync_cfg(large_seed), large_seed)

    assert isinstance(modified_latents, np.ndarray)
    assert modified_latents.shape == latents.shape
    assert inject_trace["status"] == "ok"
    assert inject_trace["sync_inject_status"] == "ok"


def test_template_match_metrics_record_normalized_seed_for_large_input() -> None:
    """
    功能：template match 诊断必须记录归一化后的合法 seed。 

    Verify template-match diagnostics record the normalized uint32 seed when a
    larger deterministic seed is provided by the caller.

    Args:
        None.

    Returns:
        None.
    """
    large_seed = stable_seed_from_parts({
        "key_id": "<absent>",
        "sample_idx": 0,
        "purpose": "embed",
    })
    template = _build_template()
    latents = np.zeros((1, 4, 16, 16), dtype=np.float32)

    metrics = template._compute_template_match_metrics(latents, _build_sync_cfg(large_seed), large_seed)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(metrics["template_seed"], int)
    assert 0 <= metrics["template_seed"] <= 2 ** 32 - 1
    assert metrics["template_seed"] != large_seed