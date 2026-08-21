"""Keyed high-frequency Stage-A anchor and blind final-image score."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cegwm.runtime.observation import encode_final_rgb_image
from cegwm.shared.bands import make_frequency_band_masks
from cegwm.shared.numerics import BudgetMeasurement
from cegwm.shared.prg import prg_rademacher

HF_CANDIDATE_ID = "hf_tail_rademacher_v1"
HF_INJECTION_STEP_INDEX = 18
HF_TOTAL_RELATIVE_L2 = 0.012
HF_MIN_RADIUS = 0.58
HF_MAX_RADIUS = 1.0
_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
_CARRIER_VERSION = "spatial-irfft2-real-rademacher-v1"


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


def _validate_latent_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) != 4 or any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("HF carrier shape must be positive NCHW")
    batch, channels, height, width = shape
    if height < 4 or width < 4:
        raise ValueError("HF carrier spatial dimensions must be at least 4")
    return batch, channels, height, width


def _carrier_domain(assets: FrozenHFPublicAssets, channels: int, height: int, width: int, channel: int) -> str:
    return (
        f"hf/{assets.candidate_id}/{_CARRIER_VERSION}/"
        f"channels={channels}/height={height}/width={width}/channel={channel}"
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

    batch, channels, height, width = _validate_latent_shape(shape)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise TypeError("HF carrier requires a floating torch dtype")
    masks = make_frequency_band_masks(
        height,
        width,
        lf_min_radius=0.04,
        lf_max_radius=0.24,
        hf_min_radius=frozen_public_assets.hf_min_radius,
        hf_max_radius=frozen_public_assets.hf_max_radius,
    )
    hf_mask = torch.from_numpy(masks.hf).to(device=device)
    channel_carriers: list[torch.Tensor] = []
    for channel in range(channels):
        domain = _carrier_domain(frozen_public_assets, channels, height, width, channel)
        signs = prg_rademacher(
            detection_key,
            domain,
            masks.hf.shape,
            dtype=np.float32,
        )
        spectrum = torch.from_numpy(signs).to(device=device)
        spectrum = torch.where(hf_mask, spectrum, torch.zeros_like(spectrum))
        spatial = torch.fft.irfft2(spectrum, s=(height, width), norm="ortho")
        channel_carriers.append(spatial)
    carrier = torch.stack(channel_carriers, dim=0)
    norm = torch.linalg.vector_norm(carrier.to(torch.float64))
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("reconstructed HF carrier has invalid norm")
    carrier = carrier / norm.to(carrier.dtype)
    carrier = carrier.unsqueeze(0).expand(batch, -1, -1, -1)
    return carrier.to(dtype=dtype)


def _relative_l2(base: torch.Tensor, candidate: torch.Tensor) -> BudgetMeasurement:
    if base.shape != candidate.shape or base.dtype != candidate.dtype or base.device != candidate.device:
        raise ValueError("actual callback base and candidate identity must match")
    base64 = base.to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    base_l2 = float(torch.linalg.vector_norm(base64).item())
    if not math.isfinite(base_l2) or base_l2 == 0.0:
        raise ValueError("HF relative L2 budget requires a finite nonzero latent")
    perturbation_l2 = float(torch.linalg.vector_norm(candidate64 - base64).item())
    relative_l2 = perturbation_l2 / base_l2
    return BudgetMeasurement(
        dtype=str(base.dtype),
        base_l2=base_l2,
        perturbation_l2=perturbation_l2,
        relative_l2=relative_l2,
    )


def inject_hf_carrier(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: FrozenHFPublicAssets,
) -> tuple[torch.Tensor, BudgetMeasurement]:
    """Inject the HF carrier under one budget measured on callback dtype."""

    if not isinstance(latents, torch.Tensor):
        raise TypeError("callback latents must be a torch Tensor")
    if latents.ndim != 4 or not latents.dtype.is_floating_point:
        raise ValueError("callback latents must be floating NCHW")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("callback latents must be finite")
    with torch.no_grad():
        carrier = reconstruct_hf_carrier(
            detection_key,
            tuple(latents.shape),
            frozen_public_assets,
            dtype=torch.float32,
            device=latents.device,
        ).to(torch.float64)
        base64 = latents.to(torch.float64)
        base_l2 = torch.linalg.vector_norm(base64)
        carrier_l2 = torch.linalg.vector_norm(carrier)
        if not bool(torch.isfinite(base_l2)) or float(base_l2.item()) == 0.0:
            raise ValueError("HF injection requires a finite nonzero latent")
        proposed_delta = carrier * (
            base_l2 * frozen_public_assets.total_relative_l2 / carrier_l2
        )

        def candidate_at(scale: float) -> torch.Tensor:
            return (base64 + scale * proposed_delta).to(dtype=latents.dtype)

        candidate = candidate_at(1.0)
        measurement = _relative_l2(latents, candidate)
        if measurement.relative_l2 > frozen_public_assets.total_relative_l2:
            low = 0.0
            high = 1.0
            best = latents.detach().clone()
            best_measurement = _relative_l2(latents, best)
            for _ in range(80):
                middle = (low + high) / 2.0
                trial = candidate_at(middle)
                trial_measurement = _relative_l2(latents, trial)
                if trial_measurement.relative_l2 <= frozen_public_assets.total_relative_l2:
                    low = middle
                    best = trial
                    best_measurement = trial_measurement
                else:
                    high = middle
            candidate = best
            measurement = best_measurement
        if measurement.perturbation_l2 == 0.0:
            raise RuntimeError("HF budget produced no change in the actual callback dtype")
        if measurement.relative_l2 > frozen_public_assets.total_relative_l2:
            raise RuntimeError("actual callback tensor exceeds the frozen HF budget")
        return candidate, measurement


def score_hf_image(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: FrozenHFPublicAssets,
) -> float:
    """Blind HF score from only an ordinary image, key, and public assets."""

    observation = encode_final_rgb_image(
        image,
        frozen_public_assets.image_processor,
        frozen_public_assets.vae,
    )
    carrier = reconstruct_hf_carrier(
        detection_key,
        tuple(observation.shape),
        frozen_public_assets,
        dtype=torch.float32,
        device=observation.device,
    )
    observation_spectrum = torch.fft.rfft2(observation.to(torch.float32), norm="ortho")
    carrier_spectrum = torch.fft.rfft2(carrier, norm="ortho")
    _, _, height, width = observation.shape
    masks = make_frequency_band_masks(
        height,
        width,
        lf_min_radius=0.04,
        lf_max_radius=0.24,
        hf_min_radius=frozen_public_assets.hf_min_radius,
        hf_max_radius=frozen_public_assets.hf_max_radius,
    )
    mask = torch.from_numpy(masks.hf).to(device=observation.device)
    observed = observation_spectrum.real[..., mask].reshape(-1).to(torch.float64)
    expected = carrier_spectrum.real[..., mask].reshape(-1).to(torch.float64)
    observed = observed - observed.mean()
    expected = expected - expected.mean()
    denominator = torch.linalg.vector_norm(observed) * torch.linalg.vector_norm(expected)
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) == 0.0:
        raise ValueError("blind HF score requires non-constant finite image evidence")
    score = float(torch.dot(observed, expected).item() / denominator.item())
    if not math.isfinite(score):
        raise RuntimeError("blind HF score is not finite")
    return score
