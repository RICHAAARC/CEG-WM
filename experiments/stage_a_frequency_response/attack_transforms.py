"""Fixed ordinary-RGB transforms for the standalone frequency-response plan."""

from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Any

import numpy as np
from PIL import Image

from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.prg import prg_normal

from experiments.stage_a_frequency_response.protocol import CONDITIONS

_PUBLIC_NOISE_ROOT = hashlib.sha256(b"CEG-WM/frequency-response/public-noise/v1").digest()
_JPEG_QUALITY = {"jpeg_q90": 90, "jpeg_q75": 75, "jpeg_q50": 50}
_BLUR_SIGMA = {"gaussian_blur_sigma_0_5": 0.5, "gaussian_blur_sigma_1": 1.0, "gaussian_blur_sigma_2": 2.0}
_NOISE_STD = {"gaussian_noise_std_0_005": 0.005, "gaussian_noise_std_0_01": 0.01, "gaussian_noise_std_0_02": 0.02}


def _rgb8(image: Any) -> Image.Image:
    result = require_ordinary_rgb_image(image)
    pixels = np.asarray(result)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("attack input must be ordinary RGB8")
    return result


def _quantize(pixels: np.ndarray) -> Image.Image:
    if pixels.ndim != 3 or pixels.shape[2] != 3 or not np.isfinite(pixels).all():
        raise ValueError("attack output must be finite RGB")
    return Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def public_noise_domain(*, protocol_id: str, condition: str, unit_id: str, source_id: str, generation_seed: int, height: int, width: int) -> str:
    """Public attack identity; it deliberately excludes key, method, pixels, and outcome."""

    if condition not in _NOISE_STD:
        raise ValueError("public noise domain requires a declared noise condition")
    payload = {"protocol_id": protocol_id, "condition": condition, "unit_id": unit_id, "source_id": source_id, "generation_seed": generation_seed, "height": height, "width": width}
    if not all(isinstance(value, str) and value for value in (protocol_id, condition, unit_id, source_id)) or not isinstance(generation_seed, int) or generation_seed < 0 or not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1:
        raise ValueError("public noise identity is invalid")
    return "frequency-response/public-noise/v1/" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=quality, subsampling=2, optimize=False, progressive=False, exif=b"", icc_profile=None)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        result = decoded.convert("RGB").copy()
    if result.size != image.size:
        raise ValueError("JPEG condition changed ordinary image geometry")
    return _rgb8(result)


def _blur(image: Image.Image, sigma: float) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float64) / 255.0
    if min(pixels.shape[:2]) < 2:
        raise ValueError("Gaussian blur requires dimensions at least 2")
    radius = math.ceil(3 * sigma)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates / sigma))
    kernel /= kernel.sum()
    padded_horizontal = np.pad(pixels, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    horizontal = np.empty_like(pixels)
    for index in range(pixels.shape[1]):
        horizontal[:, index, :] = np.tensordot(padded_horizontal[:, index:index + 2 * radius + 1, :], kernel, axes=(1, 0))
    padded_vertical = np.pad(horizontal, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    vertical = np.empty_like(pixels)
    for index in range(pixels.shape[0]):
        vertical[index, :, :] = np.tensordot(padded_vertical[index:index + 2 * radius + 1, :, :], kernel, axes=(0, 0))
    return _quantize(vertical)


def apply_condition(image: Any, condition: str, *, noise_domain: str | None = None) -> Image.Image:
    """Transform a generated ordinary RGB image after all watermark callbacks complete."""

    source = _rgb8(image)
    if condition == "identity":
        if noise_domain is not None:
            raise ValueError("identity condition does not accept a noise domain")
        return source
    if condition in _JPEG_QUALITY:
        if noise_domain is not None:
            raise ValueError("JPEG condition does not accept a noise domain")
        return _jpeg(source, _JPEG_QUALITY[condition])
    if condition in _BLUR_SIGMA:
        if noise_domain is not None:
            raise ValueError("blur condition does not accept a noise domain")
        return _blur(source, _BLUR_SIGMA[condition])
    if condition in _NOISE_STD:
        if not isinstance(noise_domain, str) or not noise_domain:
            raise ValueError("noise condition requires its public domain")
        pixels = np.asarray(source, dtype=np.float64) / 255.0
        return _quantize(pixels + _NOISE_STD[condition] * prg_normal(_PUBLIC_NOISE_ROOT, noise_domain, pixels.shape, dtype=np.float64))
    if condition not in CONDITIONS:
        raise ValueError("condition is not frozen")
    raise RuntimeError("declared condition has no transform")
