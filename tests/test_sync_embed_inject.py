"""
File purpose: 同步模板 embed 注入回归测试。
Module type: General module
"""

import numpy as np

from main.watermarking.geometry_chain.sync.latent_sync_template import LatentSyncTemplate


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def test_sync_embed_inject_updates_latents_and_trace() -> None:
    sync_module = LatentSyncTemplate(
        impl_id="latent_sync_template_v1",
        impl_version="v1",
        impl_digest="unit_digest",
    )
    cfg = {
        "embed": {"geometry": {"enable_latent_sync": True, "sync_strength": 0.1}},
        "detect": {"geometry": {"sync_fft_bins": 8}},
    }
    latents = np.zeros((1, 4, 16, 16), dtype=np.float32)

    modified_latents, inject_trace = sync_module.embed_inject(latents, cfg, seed=123)

    assert isinstance(modified_latents, np.ndarray)
    assert not np.array_equal(modified_latents, latents)
    assert inject_trace.get("status") == "ok"
    assert inject_trace.get("sync_inject_status") == "ok"
    assert isinstance(inject_trace.get("template_digest"), str)
    assert len(inject_trace.get("template_digest")) == 64


def test_sync_embed_detect_roundtrip_improves_template_match_score() -> None:
    sync_module = LatentSyncTemplate(
        impl_id="latent_sync_template_v1",
        impl_version="v1",
        impl_digest="unit_digest",
    )
    cfg = {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "seed": 123,
        "embed": {"geometry": {"enable_latent_sync": True, "sync_strength": 0.05}},
        "detect": {
            "geometry": {
                "enable_latent_sync": True,
                "enabled": True,
                "sync_fft_bins": 16,
                "sync_rotation_bins": 36,
                "sync_scale_bins": 16,
            }
        },
        "inference_height": 512,
        "inference_width": 512,
    }

    latents = np.zeros((1, 4, 32, 32), dtype=np.float32)
    modified_latents, _ = sync_module.embed_inject(latents, cfg, seed=123)

    pipeline = _Pipeline()
    base_result = sync_module.extract_sync(pipeline, latents, cfg=cfg, rng=None)
    injected_result = sync_module.extract_sync(pipeline, modified_latents, cfg=cfg, rng=None)

    assert base_result.status == "ok"
    assert injected_result.status == "ok"
    assert isinstance(base_result.sync_quality_metrics, dict)
    assert isinstance(injected_result.sync_quality_metrics, dict)

    base_match = float(base_result.sync_quality_metrics.get("template_match_score", 0.0))
    injected_match = float(injected_result.sync_quality_metrics.get("template_match_score", 0.0))
    assert injected_match >= base_match
    assert isinstance(injected_result.sync_quality_metrics.get("template_digest"), str)


def test_support_aware_template_match_detects_noise_diluted_aligned_signal() -> None:
    sync_module = LatentSyncTemplate(
        impl_id="latent_sync_template_v1",
        impl_version="v1",
        impl_digest="unit_digest",
    )
    cfg = {
        "embed": {"geometry": {"enable_latent_sync": True, "sync_strength": 0.05}},
        "detect": {"geometry": {"enable_latent_sync": True, "enabled": True, "sync_fft_bins": 12}},
    }

    template = sync_module._build_sync_template((1, 1, 32, 32), cfg, 123)  # pyright: ignore[reportPrivateUsage]
    rng = np.random.default_rng(2026)
    fft_noise = 6.0 * (
        rng.normal(0.0, 1.0, size=template.shape)
        + 1j * rng.normal(0.0, 1.0, size=template.shape)
    )
    aligned_latents = np.fft.ifft2(template + fft_noise).real.astype(np.float32)[None, None, :, :]
    scrambled_latents = np.fft.ifft2(np.roll(template + fft_noise, shift=5, axis=0)).real.astype(np.float32)[None, None, :, :]

    aligned_metrics = sync_module._compute_template_match_metrics(aligned_latents, cfg, 123)  # pyright: ignore[reportPrivateUsage]
    scrambled_metrics = sync_module._compute_template_match_metrics(scrambled_latents, cfg, 123)  # pyright: ignore[reportPrivateUsage]

    assert aligned_metrics.get("template_match_score", 0.0) > aligned_metrics.get("template_match_threshold", 0.0)
    assert aligned_metrics.get("template_match_score", 0.0) > scrambled_metrics.get("template_match_score", 0.0)
