"""Thin current-stack clean RGB observation path for the Content V4 W fit."""

from __future__ import annotations

from typing import Any

import torch

from cegwm.method.content_whitening_v4 import (
    FIT_HEIGHT,
    FIT_WIDTH,
    OBSERVATION_SHAPE,
    OBSERVATION_STRIDE,
    FitEntry,
)
from cegwm.runtime.diffusers_sd35 import run_sd35_plain
from cegwm.runtime.observation import encode_final_rgb_image


def materialize_clean_fit_observation(
    image: Any,
    image_processor: Any,
    vae: Any,
) -> torch.Tensor:
    """Encode an ordinary final RGB and materialize exact CPU C-order float32."""

    observation = encode_final_rgb_image(image, image_processor, vae)
    if tuple(observation.shape) != OBSERVATION_SHAPE:
        raise ValueError("Content V4 fit observation must have shape 1x16x64x64")
    materialized = observation.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (
        tuple(materialized.shape) != OBSERVATION_SHAPE
        or tuple(materialized.stride()) != OBSERVATION_STRIDE
        or not materialized.is_contiguous()
    ):
        raise ValueError("Content V4 fit observation must be exact C-order storage")
    if not bool(torch.isfinite(materialized).all()):
        raise ValueError("Content V4 fit observation must be finite")
    return materialized


def run_clean_fit_observation(
    pipeline: Any,
    entry: FitEntry,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate one clean image without a callback/key and immediately observe it."""

    if not isinstance(entry, FitEntry):
        raise TypeError("Content V4 fit runtime requires a validated FitEntry")
    if not isinstance(generator, torch.Generator):
        raise TypeError("Content V4 fit runtime requires an explicit torch Generator")
    image = run_sd35_plain(
        pipeline,
        entry.prompt,
        height=FIT_HEIGHT,
        width=FIT_WIDTH,
        generator=generator,
    )
    image_processor = getattr(pipeline, "image_processor", None)
    vae = getattr(pipeline, "vae", None)
    return materialize_clean_fit_observation(image, image_processor, vae)


__all__ = ["materialize_clean_fit_observation", "run_clean_fit_observation"]
