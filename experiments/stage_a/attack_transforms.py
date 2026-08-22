"""Frozen ordinary-RGB transforms for the Stage-A attack comparison."""

from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Any

import numpy as np
from PIL import Image

from cegwm.shared.prg import prg_normal

IDENTITY_REFERENCE = "identity_reference"
ATTACK_IDS = (
    "jpeg_q75",
    "gaussian_blur_sigma_1",
    "gaussian_noise_std_0_01",
)
CONDITION_ORDER = (IDENTITY_REFERENCE, *ATTACK_IDS)
_PUBLIC_NOISE_ROOT = hashlib.sha256(
    b"CEG-WM/stage-a/public-attack-noise/root/v1"
).digest()


def _rgb8(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise TypeError("attack input must be an ordinary Pillow RGB image")
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("attack input must be RGB8")
    if pixels.shape[0] < 1 or pixels.shape[1] < 1:
        raise ValueError("attack input must have nonempty spatial dimensions")
    return image


def _float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(_rgb8(image), dtype=np.float64) / 255.0


def _quantize_rgb8(pixels: np.ndarray) -> Image.Image:
    if pixels.ndim != 3 or pixels.shape[2] != 3 or not np.isfinite(pixels).all():
        raise ValueError("attack output must be a finite RGB array")
    quantized = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(quantized, mode="RGB")


def _jpeg_q75(image: Image.Image) -> Image.Image:
    source = _rgb8(image)
    encoded = io.BytesIO()
    source.save(
        encoded,
        format="JPEG",
        quality=75,
        subsampling=2,
        optimize=False,
        progressive=False,
        exif=b"",
        icc_profile=None,
    )
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        result = decoded.convert("RGB").copy()
    if result.size != source.size:
        raise ValueError("JPEG attack changed the image shape")
    return _rgb8(result)


def _gaussian_kernel_sigma_one_radius_three() -> np.ndarray:
    coordinates = np.arange(-3, 4, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates))
    kernel /= kernel.sum(dtype=np.float64)
    if not np.isfinite(kernel).all() or not math.isclose(
        float(kernel.sum()), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise RuntimeError("frozen Gaussian kernel is invalid")
    return kernel


def _convolve_reflect(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    padding = [(0, 0)] * values.ndim
    padding[axis] = (3, 3)
    padded = np.pad(values, padding, mode="reflect")
    output = np.empty_like(values, dtype=np.float64)
    for index in range(values.shape[axis]):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(index, index + 7)
        window = padded[tuple(slices)]
        output_slices = [slice(None)] * values.ndim
        output_slices[axis] = index
        output[tuple(output_slices)] = np.tensordot(
            kernel, window, axes=(0, axis)
        )
    return output


def _gaussian_blur_sigma_one(image: Image.Image) -> Image.Image:
    pixels = _float_rgb(image)
    if pixels.shape[0] < 2 or pixels.shape[1] < 2:
        raise ValueError("reflect Gaussian blur requires both image dimensions >= 2")
    kernel = _gaussian_kernel_sigma_one_radius_three()
    horizontal = _convolve_reflect(pixels, kernel, axis=1)
    vertical = _convolve_reflect(horizontal, kernel, axis=0)
    return _quantize_rgb8(vertical)


def public_noise_domain(
    *,
    protocol_id: str,
    attack_id: str,
    unit_id: str,
    source_id: str,
    generation_seed: int,
    height: int,
    width: int,
) -> str:
    if attack_id != "gaussian_noise_std_0_01":
        raise ValueError("public noise domain is only defined for the frozen noise attack")
    payload = {
        "attack_id": attack_id,
        "generation_seed": generation_seed,
        "height": height,
        "protocol_id": protocol_id,
        "source_id": source_id,
        "unit_id": unit_id,
        "width": width,
    }
    if (
        not all(isinstance(payload[name], str) and payload[name] for name in (
            "attack_id", "protocol_id", "source_id", "unit_id"
        ))
        or not isinstance(generation_seed, int)
        or isinstance(generation_seed, bool)
        or generation_seed < 0
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height < 1
        or width < 1
    ):
        raise ValueError("public attack noise identity is invalid")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"stage-a/public-attack-noise/v1/identity-sha256={digest}"


def _gaussian_noise_std_001(image: Image.Image, *, domain: str) -> Image.Image:
    pixels = _float_rgb(image)
    noise = prg_normal(
        _PUBLIC_NOISE_ROOT,
        domain,
        pixels.shape,
        dtype=np.float64,
    )
    return _quantize_rgb8(pixels + 0.01 * noise)


def apply_attack(
    image: Image.Image,
    attack_id: str,
    *,
    noise_domain: str | None = None,
) -> Image.Image:
    """Apply one frozen attack after ordinary-image generation."""

    if attack_id == "jpeg_q75":
        if noise_domain is not None:
            raise ValueError("JPEG attack does not accept a noise domain")
        return _jpeg_q75(image)
    if attack_id == "gaussian_blur_sigma_1":
        if noise_domain is not None:
            raise ValueError("blur attack does not accept a noise domain")
        return _gaussian_blur_sigma_one(image)
    if attack_id == "gaussian_noise_std_0_01":
        if not isinstance(noise_domain, str) or not noise_domain:
            raise ValueError("noise attack requires its frozen public domain")
        return _gaussian_noise_std_001(image, domain=noise_domain)
    raise ValueError("attack identity is not frozen")
