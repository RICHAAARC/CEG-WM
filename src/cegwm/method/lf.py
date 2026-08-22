"""Finite Stage-A LF carriers and blind final-image scores."""

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

LF_CORE_CANDIDATE_ID = "lf_core_rademacher_v1"
LF_SHELL_CANDIDATE_ID = "lf_shell_rademacher_v1"
LF_CANDIDATE_IDS = (LF_CORE_CANDIDATE_ID, LF_SHELL_CANDIDATE_ID)
LF_INJECTION_STEP_INDEX = 18
LF_TOTAL_RELATIVE_L2 = 0.012
_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
_BANDS = {
    LF_CORE_CANDIDATE_ID: (0.04, 0.14, False),
    LF_SHELL_CANDIDATE_ID: (0.14, 0.24, True),
}


@dataclass(frozen=True, slots=True)
class FrozenLFPublicAssets:
    """Public VAE assets plus one finite LF carrier identity."""

    vae: Any
    image_processor: Any
    image_processor_id: str
    candidate_id: str
    model_id: str = _MODEL_ID
    injection_step_index: int = LF_INJECTION_STEP_INDEX
    total_relative_l2: float = LF_TOTAL_RELATIVE_L2

    def __post_init__(self) -> None:
        if self.model_id != _MODEL_ID:
            raise ValueError("LF public assets must use the frozen SD3.5 model identity")
        if self.candidate_id not in LF_CANDIDATE_IDS:
            raise ValueError("LF candidate identity is not one of the two frozen candidates")
        if self.injection_step_index != LF_INJECTION_STEP_INDEX:
            raise ValueError("LF injection step differs from the frozen Stage-A plan")
        if not math.isclose(self.total_relative_l2, LF_TOTAL_RELATIVE_L2, abs_tol=0.0):
            raise ValueError("LF budget differs from the frozen Stage-A plan")
        if not isinstance(self.image_processor_id, str) or not self.image_processor_id.strip():
            raise ValueError("image_processor_id must be non-empty")
        if not callable(getattr(self.vae, "encode", None)):
            raise TypeError("frozen VAE must provide encode")
        if not callable(getattr(self.image_processor, "preprocess", None)):
            raise TypeError("frozen image processor must provide preprocess")


def _spec(assets: FrozenLFPublicAssets) -> FrequencyCarrierSpec:
    minimum, maximum, inclusive = _BANDS[assets.candidate_id]
    return FrequencyCarrierSpec(
        domain_prefix="lf",
        carrier_method_id=assets.candidate_id,
        min_radius=minimum,
        max_radius=maximum,
        max_inclusive=inclusive,
        total_relative_l2=assets.total_relative_l2,
    )


def reconstruct_lf_carrier(
    detection_key: str | bytes | bytearray | memoryview,
    shape: tuple[int, ...],
    frozen_public_assets: FrozenLFPublicAssets,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Rebuild one LF carrier from key, candidate identity, and public shape."""

    return reconstruct_frequency_carrier(
        detection_key,
        shape,
        _spec(frozen_public_assets),
        dtype=dtype,
        device=device,
    )


def inject_lf_carrier(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: FrozenLFPublicAssets,
) -> tuple[torch.Tensor, BudgetMeasurement]:
    """Inject exactly one LF candidate under the full single-carrier budget."""

    return inject_frequency_carrier(latents, detection_key, _spec(frozen_public_assets))


def score_lf_image(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: FrozenLFPublicAssets,
) -> float:
    """Blind LF score from only ordinary image, key, and frozen public assets."""

    return score_frequency_image(
        image,
        detection_key,
        frozen_public_assets,
        _spec(frozen_public_assets),
    )
