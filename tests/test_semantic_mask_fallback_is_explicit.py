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


def test_saliency_source_impl_selection_is_explicit(monkeypatch) -> None:
    """
    功能：验证 saliency 模型不可用时，formal path 不写出 mask_fallback_reason，而是通过 mask_impl_id 显式记录实现选择。

    Verify that when the semantic saliency model is unavailable, the formal path
    does not write `mask_fallback_reason` to mask_stats; instead, the implementation
    selection is recorded explicitly via `mask_impl_id` and `saliency_source_selected`.
    """
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
        "main.watermarking.content_chain.semantic_mask_provider.build_semantic_saliency_mask_proxy",
        _raise_saliency,
    )

    cfg = {
        "enable_mask": True,
        "mask": {
            "impl_id": "semantic_saliency_proxy",
        },
    }
    inputs = {
        "image": [1, 2, 3],
        "image_shape": (64, 64, 3),
    }

    result = provider.extract(cfg, inputs=inputs, cfg_digest="cfg_digest_anchor")

    assert result.status == "ok"
    assert isinstance(result.mask_stats, dict)
    assert result.mask_stats.get("mask_impl_id") == "texture_gradient"
    # v3 闭包：formal path 不再写出 mask_fallback_reason，实现选择由 mask_impl_id 显式表达
    assert "mask_fallback_reason" not in result.mask_stats
    assert result.audit.get("mask_impl_id") == "texture_gradient"


def test_saliency_source_auto_fallback_is_auditable() -> None:
    provider = SemanticMaskProvider(
        impl_id=SEMANTIC_MASK_PROVIDER_ID,
        impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
        impl_digest=digests.canonical_sha256({
            "impl_id": SEMANTIC_MASK_PROVIDER_ID,
            "impl_version": SEMANTIC_MASK_PROVIDER_VERSION,
        }),
    )

    cfg = {
        "enable_mask": True,
        "mask": {
            "saliency_source": "auto_fallback",
            "semantic_model_path": "Z:/path/not_exists/model.safetensors",
        },
    }
    inputs = {
        "image": [1, 2, 3],
        "image_shape": (64, 64, 3),
    }

    result = provider.extract(cfg, inputs=inputs, cfg_digest="cfg_digest_anchor")

    assert result.status == "ok"
    assert isinstance(result.mask_stats, dict)
    assert result.mask_stats.get("saliency_source_selected") == "proxy"
    assert "fallback_used" not in result.mask_stats
    assert "fallback_reason" not in result.mask_stats
    saliency_provenance = result.mask_stats.get("saliency_provenance")
    assert isinstance(saliency_provenance, dict)
    assert saliency_provenance.get("source_selected") == "proxy"
    assert len(saliency_provenance.get("source_attempted", [])) > 1


def test_saliency_source_model_unavailable_returns_explicit_fail() -> None:
    provider = SemanticMaskProvider(
        impl_id=SEMANTIC_MASK_PROVIDER_ID,
        impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
        impl_digest=digests.canonical_sha256({
            "impl_id": SEMANTIC_MASK_PROVIDER_ID,
            "impl_version": SEMANTIC_MASK_PROVIDER_VERSION,
        }),
    )

    cfg = {
        "enable_mask": True,
        "mask": {
            "saliency_source": "model",
            "semantic_model_path": "Z:/path/not_exists/model.safetensors",
        },
    }
    inputs = {
        "image": [1, 2, 3],
        "image_shape": (64, 64, 3),
    }

    result = provider.extract(cfg, inputs=inputs, cfg_digest="cfg_digest_anchor")

    assert result.status == "failed"
    assert result.content_failure_reason == "saliency_source_model_unavailable"
    assert isinstance(result.audit.get("probe_failure_reason"), str)


def test_saliency_source_model_v2_runtime_failure_is_fail_fast(monkeypatch) -> None:
    provider = SemanticMaskProvider(
        impl_id=SEMANTIC_MASK_PROVIDER_ID,
        impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
        impl_digest=digests.canonical_sha256({
            "impl_id": SEMANTIC_MASK_PROVIDER_ID,
            "impl_version": SEMANTIC_MASK_PROVIDER_VERSION,
        }),
    )

    def _fake_probe(_params):
        return True, None

    def _raise_v2(*args, **kwargs):
        raise RuntimeError("forced_v2_runtime_failure")

    monkeypatch.setattr(
        "main.watermarking.content_chain.semantic_mask_provider._probe_model_v2_availability",
        _fake_probe,
    )
    monkeypatch.setattr(
        "main.watermarking.content_chain.semantic_mask_provider.build_semantic_saliency_mask_model",
        _raise_v2,
    )

    cfg = {
        "enable_mask": True,
        "mask": {
            "saliency_source": "model",
            "semantic_model_path": "C:/fake/model.pt",
        },
    }
    inputs = {
        "image": [1, 2, 3],
        "image_shape": (64, 64, 3),
    }

    result = provider.extract(cfg, inputs=inputs, cfg_digest="cfg_digest_anchor")

    assert result.status == "failed"
    assert result.content_failure_reason == "saliency_source_model_v2_runtime_failed"
    assert result.mask_digest is None
    assert result.audit.get("mask_source_type") == "semantic_model"
