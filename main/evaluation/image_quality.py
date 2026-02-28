"""
File purpose: 图像质量指标计算（PSNR / SSIM）。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt


def _to_float_image(image: Any) -> npt.NDArray[np.float32]:
    """
    功能：将输入图像转换为 float32 数组。

    Convert input image to float32 array.

    Args:
        image: Input numpy image array.

    Returns:
        Float32 image array.

    Raises:
        TypeError: If image type is invalid.
        ValueError: If image is empty.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be np.ndarray")
    if image.size == 0:
        raise ValueError("image must be non-empty")
    return image.astype(np.float32)


def _uniform_filter_2d(image: npt.NDArray[np.float32], win_size: int) -> npt.NDArray[np.float32]:
    """
    功能：对二维图像执行均值窗口滤波。

    Apply 2D mean filter with same-size output.

    Args:
        image: 2D float image.
        win_size: Odd window size.

    Returns:
        Filtered image.
    """
    pad = win_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    for row_index in range(image.shape[0]):
        for col_index in range(image.shape[1]):
            window = padded[row_index:row_index + win_size, col_index:col_index + win_size]
            output[row_index, col_index] = float(np.mean(window))
    return output


def _ssim_single_channel(
    img_original: npt.NDArray[np.float32],
    img_watermarked: npt.NDArray[np.float32],
    win_size: int,
    data_range: float,
) -> float:
    """
    功能：计算单通道 SSIM。

    Compute SSIM for single-channel images.

    Args:
        img_original: Original single-channel image.
        img_watermarked: Watermarked single-channel image.
        win_size: Sliding window size.
        data_range: Pixel dynamic range.

    Returns:
        SSIM score in [0, 1].
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = _uniform_filter_2d(img_original, win_size)
    mu_y = _uniform_filter_2d(img_watermarked, win_size)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = _uniform_filter_2d(img_original * img_original, win_size) - mu_x_sq
    sigma_y_sq = _uniform_filter_2d(img_watermarked * img_watermarked, win_size) - mu_y_sq
    sigma_xy = _uniform_filter_2d(img_original * img_watermarked, win_size) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    denominator = np.maximum(denominator, 1e-12)
    ssim_map = numerator / denominator
    ssim_map = np.clip(ssim_map, 0.0, 1.0)
    return float(np.mean(ssim_map))


def compute_psnr(img_original: Any, img_watermarked: Any, max_val: float = 255.0) -> float:
    """
    功能：计算 Peak Signal-to-Noise Ratio。

    Compute PSNR between original and watermarked image.

    Args:
        img_original: Original image as numpy array.
        img_watermarked: Watermarked image with same shape.
        max_val: Maximum pixel value.

    Returns:
        PSNR value in dB, or inf for identical images.

    Raises:
        ValueError: If shapes mismatch.
    """
    image_original = _to_float_image(img_original)
    image_watermarked = _to_float_image(img_watermarked)
    if image_original.shape != image_watermarked.shape:
        raise ValueError("img_original and img_watermarked must have the same shape")

    mse = float(np.mean((image_original - image_watermarked) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def compute_ssim(
    img_original: Any,
    img_watermarked: Any,
    win_size: int = 7,
    data_range: float = 255.0,
) -> float:
    """
    功能：计算 Structural Similarity Index。

    Compute SSIM with sliding-window statistics.

    Args:
        img_original: Original image array in [H, W] or [H, W, C].
        img_watermarked: Watermarked image with same shape.
        win_size: Odd window size.
        data_range: Pixel dynamic range.

    Returns:
        Mean SSIM score in [0, 1].

    Raises:
        ValueError: If shape mismatch or win_size is invalid.
    """
    image_original = _to_float_image(img_original)
    image_watermarked = _to_float_image(img_watermarked)

    if image_original.shape != image_watermarked.shape:
        raise ValueError("img_original and img_watermarked must have the same shape")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be odd integer >= 3")

    if image_original.ndim == 2:
        return _ssim_single_channel(image_original, image_watermarked, win_size, data_range)

    if image_original.ndim == 3:
        channel_scores: List[float] = []
        for channel_index in range(image_original.shape[2]):
            channel_scores.append(
                _ssim_single_channel(
                    image_original[:, :, channel_index],
                    image_watermarked[:, :, channel_index],
                    win_size,
                    data_range,
                )
            )
        return float(np.mean(channel_scores)) if channel_scores else 0.0

    raise ValueError("image must be 2D or 3D array")


def compute_quality_metrics_batch(image_pairs: Sequence[Tuple[Any, Any]]) -> Dict[str, Any]:
    """
    功能：批量计算图像质量指标。

    Compute PSNR and SSIM for a sequence of image pairs.

    Args:
        image_pairs: Sequence of (original, watermarked) arrays.

    Returns:
        Summary mapping with mean and per-image metrics.

    Raises:
        TypeError: If input type is invalid.
    """
    if not isinstance(image_pairs, (list, tuple)):
        raise TypeError("image_pairs must be list or tuple")

    per_image: List[Dict[str, float]] = []
    psnr_values: List[float] = []
    ssim_values: List[float] = []

    for pair in image_pairs:
        if len(pair) != 2:
            continue
        original_image, watermarked_image = pair
        try:
            psnr_value = compute_psnr(original_image, watermarked_image)
            ssim_value = compute_ssim(original_image, watermarked_image)
        except Exception:
            continue

        per_image.append({"psnr": float(psnr_value), "ssim": float(ssim_value)})
        psnr_values.append(float(psnr_value))
        ssim_values.append(float(ssim_value))

    mean_psnr = float(np.mean(psnr_values)) if psnr_values else None
    mean_ssim = float(np.mean(ssim_values)) if ssim_values else None

    return {
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "per_image": per_image,
        "count": len(per_image),
    }
