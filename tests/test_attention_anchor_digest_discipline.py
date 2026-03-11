"""
验证 注意力锚点摘要输入域与配置变更绑定纪律
"""

from __future__ import annotations

import copy

import numpy as np

from main.policy.runtime_whitelist import load_runtime_whitelist
from main.watermarking.geometry_chain.attention_anchor_extractor import AttentionAnchorExtractor


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def _build_cfg(anchor_top_k: int) -> dict:
    return {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "anchor_top_k": anchor_top_k,
            }
        },
    }


def test_anchor_digest_must_change_when_anchor_top_k_changes() -> None:
    """
    功能：配置变化必须影响 anchor digest。

    Anchor digest must change when anchor extraction config changes.

    Args:
        None.

    Returns:
        None.
    """
    extractor = AttentionAnchorExtractor(
        "geometry_attention_anchor_sd3_v1",
        "v1",
        "a" * 64,
    )
    pipeline = _Pipeline()
    rng = np.random.RandomState(7)
    latents = rng.randn(1, 4, 8, 8).astype(np.float32)

    cfg_a = _build_cfg(anchor_top_k=3)
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b["detect"]["geometry"]["anchor_top_k"] = 6

    result_a = extractor.extract_anchors(pipeline, latents, cfg=cfg_a, rng=None)
    result_b = extractor.extract_anchors(pipeline, latents, cfg=cfg_b, rng=None)

    assert result_a.anchor_config_digest != result_b.anchor_config_digest
    assert result_a.anchor_digest != result_b.anchor_digest


def test_geometry_attention_impl_is_in_runtime_whitelist() -> None:
    """
    功能：新几何实现必须在 runtime whitelist 中注册。

    New geometry implementation id must exist in runtime whitelist.

    Args:
        None.

    Returns:
        None.
    """
    whitelist = load_runtime_whitelist()
    impl_cfg = whitelist.data.get("impl_id", {})
    allowed_by_domain = impl_cfg.get("allowed_by_domain", {})
    geometry_impls = allowed_by_domain.get("geometry_extractor", [])
    assert "attention_anchor_map_relation_v2" in geometry_impls
