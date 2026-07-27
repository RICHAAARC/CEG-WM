"""Shared value identity for ordinary RGB8 method inputs."""

from __future__ import annotations

from hashlib import sha256

import torch

from .key_schedule import stable_json_utf8


class Rgb8ImageError(ValueError):
    """An image is not an ordinary RGB8 [1,3,H,W] method input."""


def validate_rgb8_image(image: object) -> torch.Tensor:
    """Return a validated ordinary RGB8 tensor without changing its values."""

    if (
        not isinstance(image, torch.Tensor)
        or image.dtype is not torch.uint8
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or image.shape[2] <= 1
        or image.shape[3] <= 1
    ):
        raise Rgb8ImageError(
            "image must be RGB uint8 [1,3,H,W] with H,W > 1"
        )
    return image


def clone_rgb8_image(image: object) -> torch.Tensor:
    """Return a detached contiguous value copy of a validated RGB8 image."""

    validated = validate_rgb8_image(image)
    return validated.detach().clone(memory_format=torch.contiguous_format)


def rgb8_image_digest(image: object) -> str:
    """Bind RGB8 dtype, shape, and exact values in one stable SHA-256 digest."""

    validated = validate_rgb8_image(image)
    normalized = validated.detach().to(device="cpu").contiguous()
    values_digest = sha256(
        bytes(normalized.reshape(-1).tolist())
    ).hexdigest()
    return sha256(
        stable_json_utf8(
            {
                "dtype": "torch.uint8",
                "shape": list(normalized.shape),
                "values_sha256": values_digest,
            }
        )
    ).hexdigest()


def validate_rgb8_image_digest(value: object) -> str:
    """Return a syntactically valid lowercase RGB8 SHA-256 digest."""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Rgb8ImageError(
            "RGB8 image digest must be lowercase SHA-256 hex"
        )
    return value


__all__ = [
    "Rgb8ImageError",
    "clone_rgb8_image",
    "rgb8_image_digest",
    "validate_rgb8_image",
    "validate_rgb8_image_digest",
]
