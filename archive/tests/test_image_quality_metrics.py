"""
File purpose: 图像质量指标（PSNR/SSIM）回归测试。
Module type: General module
"""

import numpy as np

from main.evaluation.image_quality import compute_psnr, compute_quality_metrics_batch, compute_ssim


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
