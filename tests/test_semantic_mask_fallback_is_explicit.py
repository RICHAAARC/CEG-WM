"""
文件目的：语义掩码 fallback 显式标记回归测试。
Module type: General module
"""

from main.core import digests
from main.watermarking.content_chain.semantic_mask_provider import (
    SEMANTIC_MASK_PROVIDER_ID,
    SEMANTIC_MASK_PROVIDER_VERSION,
    SemanticMaskProvider,
)


def test_semantic_mask_fallback_is_explicit(monkeypatch) -> None:
    provider = SemanticMaskProvider(
        impl_id=SEMANTIC_MASK_PROVIDER_ID,
        impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
        impl_digest=digests.canonical_sha256({
            "impl_id": SEMANTIC_MASK_PROVIDER_ID,
            "impl_version": SEMANTIC_MASK_PROVIDER_VERSION,
        }),
    )

    def _raise_saliency(*args, **kwargs):
        raise RuntimeError("forced_saliency_unavailable")

    monkeypatch.setattr(
        "main.watermarking.content_chain.semantic_mask_provider.build_semantic_saliency_mask_v1",
        _raise_saliency,
    )

    cfg = {
        "enable_mask": True,
        "mask": {
            "impl_id": "semantic_saliency_v1",
        },
    }
    inputs = {
        "image": [1, 2, 3],
        "image_shape": (64, 64, 3),
    }

    result = provider.extract(cfg, inputs=inputs, cfg_digest="cfg_digest_anchor")

    assert result.status == "ok"
    assert isinstance(result.mask_stats, dict)
    assert result.mask_stats.get("mask_impl_id") == "texture_gradient_v1"
    assert isinstance(result.mask_stats.get("mask_fallback_reason"), str)
    assert result.mask_stats.get("mask_fallback_reason") != "<absent>"
    assert result.audit.get("mask_impl_id") == "texture_gradient_v1"
