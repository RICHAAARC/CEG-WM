"""Keyed high-frequency Stage-A anchor and blind final-image score."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from cegwm.method.frequency import (
    FrequencyCarrierSpec,
    inject_frequency_carrier,
    reconstruct_frequency_carrier,
    score_frequency_image,
)
from cegwm.shared.numerics import BudgetMeasurement

HF_CANDIDATE_ID = "hf_tail_rademacher_v1"
HF_INJECTION_STEP_INDEX = 18
HF_TOTAL_RELATIVE_L2 = 0.012
HF_MIN_RADIUS = 0.58
HF_MAX_RADIUS = 1.0
_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


@dataclass(frozen=True, slots=True)
class FrozenHFPublicAssets:
    """One immutable in-memory view of the protocol-named public assets."""

    vae: Any
    image_processor: Any
    image_processor_id: str
    model_id: str = _MODEL_ID
    candidate_id: str = HF_CANDIDATE_ID
    injection_step_index: int = HF_INJECTION_STEP_INDEX
    total_relative_l2: float = HF_TOTAL_RELATIVE_L2
    hf_min_radius: float = HF_MIN_RADIUS
    hf_max_radius: float = HF_MAX_RADIUS

    def __post_init__(self) -> None:
        if self.model_id != _MODEL_ID:
            raise ValueError("HF public assets must use the frozen SD3.5 model identity")
        if self.candidate_id != HF_CANDIDATE_ID:
            raise ValueError("HF candidate identity differs from the frozen Stage-A anchor")
        if self.injection_step_index != HF_INJECTION_STEP_INDEX:
            raise ValueError("HF injection step differs from the frozen Stage-A anchor")
        if not math.isclose(self.total_relative_l2, HF_TOTAL_RELATIVE_L2, abs_tol=0.0):
            raise ValueError("HF budget differs from the frozen Stage-A anchor")
        if not math.isclose(self.hf_min_radius, HF_MIN_RADIUS, abs_tol=0.0):
            raise ValueError("HF minimum radius differs from the frozen Stage-A anchor")
        if not math.isclose(self.hf_max_radius, HF_MAX_RADIUS, abs_tol=0.0):
            raise ValueError("HF maximum radius differs from the frozen Stage-A anchor")
        if not isinstance(self.image_processor_id, str) or not self.image_processor_id.strip():
            raise ValueError("image_processor_id must be non-empty")
        if not callable(getattr(self.vae, "encode", None)):
            raise TypeError("frozen VAE must provide encode")
        if not callable(getattr(self.image_processor, "preprocess", None)):
            raise TypeError("frozen image processor must provide preprocess")


def _spec(assets: FrozenHFPublicAssets) -> FrequencyCarrierSpec:
    return FrequencyCarrierSpec(
        domain_prefix="hf",
        carrier_method_id=assets.candidate_id,
        min_radius=assets.hf_min_radius,
        max_radius=assets.hf_max_radius,
        max_inclusive=True,
        total_relative_l2=assets.total_relative_l2,
    )


def reconstruct_hf_carrier(
    detection_key: str | bytes | bytearray | memoryview,
    shape: tuple[int, ...],
    frozen_public_assets: FrozenHFPublicAssets,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Rebuild the public-domain HF carrier from key and observable shape only."""

    return reconstruct_frequency_carrier(
        detection_key,
        shape,
        _spec(frozen_public_assets),
        dtype=dtype,
        device=device,
    )


def inject_hf_carrier(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: FrozenHFPublicAssets,
) -> tuple[torch.Tensor, BudgetMeasurement]:
    """Inject the HF carrier under one budget measured on callback dtype."""

    return inject_frequency_carrier(latents, detection_key, _spec(frozen_public_assets))


def score_hf_image(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: FrozenHFPublicAssets,
) -> float:
    """Blind HF score from only an ordinary image, key, and public assets."""

    return score_frequency_image(
        image,
        detection_key,
        frozen_public_assets,
        _spec(frozen_public_assets),
    )
