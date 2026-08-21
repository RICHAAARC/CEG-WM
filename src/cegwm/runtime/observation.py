"""Final ordinary-image boundary and frozen-public-VAE observation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def require_ordinary_rgb_image(image: Any) -> Image.Image:
    """Accept only a final RGB PIL image or an RGB uint8 pixel array."""

    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            raise ValueError("final pipeline image must already be RGB")
        return image.copy()
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("array image must be uint8 HWC RGB")
        return Image.fromarray(image, mode="RGB")
    raise TypeError("blind observation accepts only an ordinary RGB image")


def _vae_device_dtype(vae: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(vae.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise TypeError("frozen VAE must expose parameters for device and dtype") from error
    if not parameter.dtype.is_floating_point:
        raise TypeError("frozen VAE parameters must use a floating dtype")
    return parameter.device, parameter.dtype


def encode_final_rgb_image(image: Any, image_processor: Any, vae: Any) -> torch.Tensor:
    """Re-encode a final RGB image without accepting any embedding-side latent."""

    rgb_image = require_ordinary_rgb_image(image)
    preprocess = getattr(image_processor, "preprocess", None)
    if not callable(preprocess):
        raise TypeError("frozen image processor must provide preprocess")
    encode = getattr(vae, "encode", None)
    if not callable(encode):
        raise TypeError("frozen VAE must provide encode")
    pixels = preprocess(rgb_image)
    if not isinstance(pixels, torch.Tensor):
        raise TypeError("frozen image processor must return a torch Tensor")
    if pixels.ndim != 4 or pixels.shape[0] != 1 or not pixels.dtype.is_floating_point:
        raise ValueError("preprocessed final image must be floating 1CHW")
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("preprocessed final image must be finite")
    device, dtype = _vae_device_dtype(vae)
    pixels = pixels.to(device=device, dtype=dtype)
    with torch.no_grad():
        encoded = encode(pixels)
        latent_distribution = getattr(encoded, "latent_dist", None)
        mode = getattr(latent_distribution, "mode", None)
        if not callable(mode):
            raise TypeError("frozen VAE encode result must expose latent_dist.mode")
        observation = mode()
    if not isinstance(observation, torch.Tensor) or observation.ndim != 4:
        raise TypeError("frozen VAE mode must return an NCHW torch Tensor")
    config = getattr(vae, "config", None)
    scaling_factor = getattr(config, "scaling_factor", None)
    if not isinstance(scaling_factor, (int, float)) or not math_is_finite_positive(scaling_factor):
        raise ValueError("frozen VAE config must provide a finite positive scaling_factor")
    observation = observation * float(scaling_factor)
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("final-image VAE observation must be finite")
    return observation.detach()


def math_is_finite_positive(value: int | float) -> bool:
    numeric = float(value)
    return bool(np.isfinite(numeric) and numeric > 0.0)
