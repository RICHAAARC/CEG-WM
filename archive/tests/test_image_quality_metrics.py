"""
File purpose: 图像质量指标（PSNR/SSIM）回归测试。
Module type: General module
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import main.evaluation.image_quality as image_quality_module
from main.evaluation.image_quality import compute_psnr, compute_psnr_batch, compute_quality_metrics_batch, compute_ssim, compute_ssim_batch


def _reference_psnr(img_original: np.ndarray, img_watermarked: np.ndarray, max_val: float = 255.0) -> float:
    image_original = image_quality_module._to_float_image(img_original)
    image_watermarked = image_quality_module._to_float_image(img_watermarked)
    if image_original.shape != image_watermarked.shape:
        raise ValueError("img_original and img_watermarked must have the same shape")
    mse = float(np.mean((image_original - image_watermarked) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def _reference_ssim(img_original: np.ndarray, img_watermarked: np.ndarray, win_size: int = 7, data_range: float = 255.0) -> float:
    image_original = image_quality_module._to_float_image(img_original)
    image_watermarked = image_quality_module._to_float_image(img_watermarked)
    if image_original.shape != image_watermarked.shape:
        raise ValueError("img_original and img_watermarked must have the same shape")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be odd integer >= 3")
    if image_original.ndim == 2:
        return image_quality_module._ssim_single_channel(image_original, image_watermarked, win_size, data_range)
    if image_original.ndim == 3:
        channel_scores = [
            image_quality_module._ssim_single_channel(
                image_original[:, :, channel_index],
                image_watermarked[:, :, channel_index],
                win_size,
                data_range,
            )
            for channel_index in range(image_original.shape[2])
        ]
        return float(np.mean(channel_scores)) if channel_scores else 0.0
    raise ValueError("image must be 2D or 3D array")


def test_psnr_ssim_for_identical_images() -> None:
    image = np.full((16, 16, 3), 128, dtype=np.uint8)

    psnr_value = compute_psnr(image, image)
    ssim_value = compute_ssim(image, image)

    assert psnr_value == float("inf")
    assert 0.99 <= ssim_value <= 1.0


def test_quality_metrics_batch_returns_summary() -> None:
    original = np.zeros((12, 12, 3), dtype=np.uint8)
    watermarked = np.ones((12, 12, 3), dtype=np.uint8)
    summary = compute_quality_metrics_batch([(original, watermarked), (original, original)])

    assert isinstance(summary, dict)
    assert summary.get("count") == 2
    assert isinstance(summary.get("per_image"), list)
    assert summary.get("mean_psnr") is not None
    assert summary.get("mean_ssim") is not None


def test_compute_psnr_batch_matches_legacy_reference() -> None:
    image_pairs = [
        (
            np.full((16, 16, 3), 32 + pair_index, dtype=np.uint8),
            np.full((16, 16, 3), 34 + pair_index, dtype=np.uint8),
        )
        for pair_index in range(3)
    ]

    batch_values = compute_psnr_batch(
        [pair[0] for pair in image_pairs],
        [pair[1] for pair in image_pairs],
    )
    reference_values = [_reference_psnr(pair[0], pair[1]) for pair in image_pairs]

    assert len(batch_values) == len(reference_values)
    for batch_value, reference_value in zip(batch_values, reference_values, strict=False):
        assert batch_value == pytest.approx(reference_value, abs=1e-6)


def test_compute_ssim_batch_matches_legacy_reference() -> None:
    rng = np.random.default_rng(7)
    image_pairs = []
    for _ in range(3):
        original = rng.integers(0, 256, size=(18, 18, 3), dtype=np.uint8)
        delta = rng.integers(-4, 5, size=(18, 18, 3), dtype=np.int16)
        watermarked = np.clip(original.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        image_pairs.append((original, watermarked))

    batch_values = compute_ssim_batch(
        [pair[0] for pair in image_pairs],
        [pair[1] for pair in image_pairs],
    )
    reference_values = [_reference_ssim(pair[0], pair[1]) for pair in image_pairs]

    assert len(batch_values) == len(reference_values)
    for batch_value, reference_value in zip(batch_values, reference_values, strict=False):
        assert batch_value == pytest.approx(reference_value, abs=1e-4)


def test_compute_psnr_ssim_batch_accepts_explicit_cpu_device() -> None:
    """
    Verify explicit CPU device requests preserve the existing numerical results.

    Args:
        None.

    Returns:
        None.
    """
    original = np.full((16, 16, 3), 96, dtype=np.uint8)
    watermarked = np.full((16, 16, 3), 99, dtype=np.uint8)

    default_psnr_values = compute_psnr_batch([original], [watermarked])
    default_ssim_values = compute_ssim_batch([original], [watermarked])
    psnr_values = compute_psnr_batch([original], [watermarked], device="cpu")
    ssim_values = compute_ssim_batch([original], [watermarked], device="cpu")

    assert psnr_values == default_psnr_values
    assert ssim_values == default_ssim_values


def test_resolve_torch_compute_device_cuda_request_falls_back_to_cpu_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify CUDA requests degrade to CPU when torch reports CUDA as unavailable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda value: f"device:{value}",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    resolved_device, resolved_label = image_quality_module._resolve_torch_compute_device("cuda")

    assert resolved_device == "device:cpu"
    assert resolved_label == "cpu"
