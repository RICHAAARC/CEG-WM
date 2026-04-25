"""
File purpose: 图像质量指标计算（PSNR / SSIM）。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt


_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


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


def _prepare_batched_image_array(images: Any) -> npt.NDArray[np.float32]:
    """
    功能：将输入规范化为批量图像数组。 

    Normalize input images into one batched float32 array with NHWC semantics.

    Args:
        images: One image, one batched image array, or a sequence of images.

    Returns:
        Batched float32 image array with shape [N, H, W, C].

    Raises:
        TypeError: If the input type is invalid.
        ValueError: If image ranks or shapes are invalid.
    """
    if isinstance(images, np.ndarray):
        image_batch = np.ascontiguousarray(_to_float_image(images), dtype=np.float32)
        if image_batch.ndim == 2:
            return image_batch[np.newaxis, :, :, np.newaxis]
        if image_batch.ndim == 3:
            return image_batch[np.newaxis, :, :, :]
        if image_batch.ndim == 4:
            return image_batch
        raise ValueError("image batch must be 2D, 3D, or 4D array")

    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise TypeError("images must be np.ndarray or Sequence")
    if not images:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)

    normalized_images: List[npt.NDArray[np.float32]] = []
    for image in cast(Sequence[Any], images):
        normalized_image = np.ascontiguousarray(_to_float_image(image), dtype=np.float32)
        if normalized_image.ndim == 2:
            normalized_images.append(normalized_image[:, :, np.newaxis])
            continue
        if normalized_image.ndim == 3:
            normalized_images.append(normalized_image)
            continue
        raise ValueError("image must be 2D or 3D array")
    return np.stack(normalized_images, axis=0).astype(np.float32, copy=False)


def _resolve_torch_compute_device(device: Any | None = None) -> tuple[Any, str]:
    """
    功能：解析 batched 图像质量计算所使用的 torch device。 

    Resolve the effective torch device for batched image-quality computation.

    Args:
        device: Optional requested torch device token or torch.device instance.

    Returns:
        Tuple of (resolved_torch_device, resolved_device_label).

    Raises:
        RuntimeError: If torch is unavailable.
        ValueError: If the requested device token is invalid.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch unavailable for batched image quality: {type(exc).__name__}: {exc}") from exc

    if device is None:
        return torch.device("cpu"), "cpu"

    device_text = str(device).strip()
    if not device_text:
        return torch.device("cpu"), "cpu"

    if device_text.lower().startswith("cuda"):
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if not cuda_available:
            return torch.device("cpu"), "cpu"

    return torch.device(device_text), device_text


def _prepare_batched_image_tensor(images: Any, *, device: Any | None = None) -> Any:
    """
    功能：将批量图像数组转换为 NCHW float32 tensor。 

    Convert batched image data into one NCHW float32 tensor.

    Args:
        images: One image, one batched image array, or a sequence of images.
        device: Optional torch device request for tensor placement.

    Returns:
        Torch tensor with shape [N, C, H, W].

    Raises:
        RuntimeError: If torch is unavailable.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch unavailable for batched image quality: {type(exc).__name__}: {exc}") from exc

    image_batch = _prepare_batched_image_array(images)
    image_tensor = torch.as_tensor(image_batch, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    resolved_device, _ = _resolve_torch_compute_device(device)
    if hasattr(image_tensor, "to"):
        image_tensor = image_tensor.to(resolved_device)
    return image_tensor


def _compute_ssim_batch_reference(
    original_batch: npt.NDArray[np.float32],
    watermarked_batch: npt.NDArray[np.float32],
    *,
    win_size: int,
    data_range: float,
) -> List[float]:
    """
    功能：使用 legacy 单样本实现计算一批 SSIM。 

    Compute one SSIM batch using the legacy reference implementation.

    Args:
        original_batch: Batched original images in NHWC format.
        watermarked_batch: Batched compared images in NHWC format.
        win_size: Sliding window size.
        data_range: Pixel dynamic range.

    Returns:
        Per-image SSIM scores.
    """
    output_values: List[float] = []
    for original_image, watermarked_image in zip(original_batch, watermarked_batch, strict=False):
        channel_count = int(original_image.shape[2])
        if channel_count == 1:
            output_values.append(
                _ssim_single_channel(
                    original_image[:, :, 0],
                    watermarked_image[:, :, 0],
                    win_size,
                    data_range,
                )
            )
            continue
        channel_scores: List[float] = []
        for channel_index in range(channel_count):
            channel_scores.append(
                _ssim_single_channel(
                    original_image[:, :, channel_index],
                    watermarked_image[:, :, channel_index],
                    win_size,
                    data_range,
                )
            )
        output_values.append(float(np.mean(channel_scores)) if channel_scores else 0.0)
    return output_values


def compute_psnr_batch(
    img_original_batch: Any,
    img_watermarked_batch: Any,
    max_val: float = 255.0,
    device: Any | None = None,
) -> List[float]:
    """
    功能：批量计算 Peak Signal-to-Noise Ratio。 

    Compute PSNR for one aligned image batch.

    Args:
        img_original_batch: Original images with NHWC batch semantics or an image sequence.
        img_watermarked_batch: Compared images aligned with img_original_batch.
        max_val: Maximum pixel value.
        device: Optional torch device request for batched tensor execution.

    Returns:
        Per-image PSNR values in dB.

    Raises:
        ValueError: If batch shapes mismatch.
    """
    image_original_batch = _prepare_batched_image_array(img_original_batch)
    image_watermarked_batch = _prepare_batched_image_array(img_watermarked_batch)
    if image_original_batch.shape != image_watermarked_batch.shape:
        raise ValueError("img_original_batch and img_watermarked_batch must have the same shape")
    if image_original_batch.shape[0] == 0:
        return []

    try:
        import torch
    except Exception:
        mse_values = np.mean((image_original_batch - image_watermarked_batch) ** 2, axis=(1, 2, 3), dtype=np.float32)
        output_values: List[float] = []
        for mse_value in mse_values.tolist():
            if float(mse_value) <= 0.0:
                output_values.append(float("inf"))
            else:
                output_values.append(float(20.0 * np.log10(max_val) - 10.0 * np.log10(float(mse_value))))
        return output_values

    with torch.inference_mode():
        resolved_device, _ = _resolve_torch_compute_device(device)
        image_original_tensor = _prepare_batched_image_tensor(image_original_batch, device=resolved_device)
        image_watermarked_tensor = _prepare_batched_image_tensor(image_watermarked_batch, device=resolved_device)
        mse_values = torch.mean((image_original_tensor - image_watermarked_tensor) ** 2, dim=(1, 2, 3))
        max_val_tensor = torch.full_like(mse_values, float(max_val))
        psnr_values = (20.0 * torch.log10(max_val_tensor)) - (10.0 * torch.log10(torch.clamp(mse_values, min=1e-12)))
        psnr_values = torch.where(mse_values <= 0.0, torch.full_like(psnr_values, float("inf")), psnr_values)
    return [float(value.item()) for value in psnr_values.cpu()]


def compute_ssim_batch(
    img_original_batch: Any,
    img_watermarked_batch: Any,
    win_size: int = 7,
    data_range: float = 255.0,
    device: Any | None = None,
) -> List[float]:
    """
    功能：批量计算 Structural Similarity Index。 

    Compute SSIM for one aligned image batch while preserving the legacy
    sliding-window semantics.

    Args:
        img_original_batch: Original images with NHWC batch semantics or an image sequence.
        img_watermarked_batch: Compared images aligned with img_original_batch.
        win_size: Odd window size.
        data_range: Pixel dynamic range.
        device: Optional torch device request for batched tensor execution.

    Returns:
        Per-image mean SSIM scores in [0, 1].

    Raises:
        ValueError: If shapes mismatch or win_size is invalid.
    """
    image_original_batch = _prepare_batched_image_array(img_original_batch)
    image_watermarked_batch = _prepare_batched_image_array(img_watermarked_batch)

    if image_original_batch.shape != image_watermarked_batch.shape:
        raise ValueError("img_original_batch and img_watermarked_batch must have the same shape")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be odd integer >= 3")
    if image_original_batch.shape[0] == 0:
        return []

    pad = win_size // 2
    if image_original_batch.shape[1] <= pad or image_original_batch.shape[2] <= pad:
        return _compute_ssim_batch_reference(
            image_original_batch,
            image_watermarked_batch,
            win_size=win_size,
            data_range=data_range,
        )

    try:
        import torch
        import torch.nn.functional as torch_functional
    except Exception:
        return _compute_ssim_batch_reference(
            image_original_batch,
            image_watermarked_batch,
            win_size=win_size,
            data_range=data_range,
        )

    c1 = (_SSIM_K1 * data_range) ** 2
    c2 = (_SSIM_K2 * data_range) ** 2
    with torch.inference_mode():
        resolved_device, _ = _resolve_torch_compute_device(device)
        image_original_tensor = _prepare_batched_image_tensor(image_original_batch, device=resolved_device)
        image_watermarked_tensor = _prepare_batched_image_tensor(image_watermarked_batch, device=resolved_device)
        channel_count = int(image_original_tensor.shape[1])
        kernel = torch.full(
            (channel_count, 1, win_size, win_size),
            1.0 / float(win_size * win_size),
            dtype=image_original_tensor.dtype,
            device=image_original_tensor.device,
        )
        padded_original = torch_functional.pad(image_original_tensor, (pad, pad, pad, pad), mode="reflect")
        padded_watermarked = torch_functional.pad(image_watermarked_tensor, (pad, pad, pad, pad), mode="reflect")

        mu_x = torch_functional.conv2d(padded_original, kernel, groups=channel_count)
        mu_y = torch_functional.conv2d(padded_watermarked, kernel, groups=channel_count)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            torch_functional.conv2d(
                torch_functional.pad(image_original_tensor * image_original_tensor, (pad, pad, pad, pad), mode="reflect"),
                kernel,
                groups=channel_count,
            )
            - mu_x_sq
        )
        sigma_y_sq = (
            torch_functional.conv2d(
                torch_functional.pad(
                    image_watermarked_tensor * image_watermarked_tensor,
                    (pad, pad, pad, pad),
                    mode="reflect",
                ),
                kernel,
                groups=channel_count,
            )
            - mu_y_sq
        )
        sigma_xy = (
            torch_functional.conv2d(
                torch_functional.pad(
                    image_original_tensor * image_watermarked_tensor,
                    (pad, pad, pad, pad),
                    mode="reflect",
                ),
                kernel,
                groups=channel_count,
            )
            - mu_xy
        )

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / torch.clamp(denominator, min=1e-12)
        ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)
        ssim_values = torch.mean(ssim_map, dim=(1, 2, 3))
    return [float(value.item()) for value in ssim_values.cpu()]


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
    c1 = (_SSIM_K1 * data_range) ** 2
    c2 = (_SSIM_K2 * data_range) ** 2

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
    return float(compute_psnr_batch([image_original], [image_watermarked], max_val=max_val)[0])


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

    if image_original.ndim not in {2, 3}:
        raise ValueError("image must be 2D or 3D array")
    return float(
        compute_ssim_batch(
            [image_original],
            [image_watermarked],
            win_size=win_size,
            data_range=data_range,
        )[0]
    )


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
