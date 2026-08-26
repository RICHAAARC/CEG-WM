"""Narrow keyed frequency-carrier primitives shared by content HF and LF."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cegwm.runtime.observation import encode_final_rgb_image
from cegwm.shared.numerics import BudgetMeasurement
from cegwm.shared.prg import prg_bytes, prg_rademacher

_CARRIER_VERSION = "spatial-irfft2-real-rademacher-v1"
BalancedRadialBlock = tuple[float, float, bool, str]


@dataclass(frozen=True, slots=True)
class FrequencyCarrierSpec:
    """One finite carrier identity and its exact radial interval."""

    domain_prefix: str
    carrier_method_id: str
    min_radius: float
    max_radius: float
    max_inclusive: bool
    total_relative_l2: float

    def __post_init__(self) -> None:
        if self.domain_prefix not in {"hf", "lf"}:
            raise ValueError("frequency carrier domain prefix must be hf or lf")
        if not self.carrier_method_id.strip():
            raise ValueError("carrier_method_id must be non-empty")
        if not (
            math.isfinite(self.min_radius)
            and math.isfinite(self.max_radius)
            and 0.0 <= self.min_radius < self.max_radius <= 1.0
        ):
            raise ValueError("frequency carrier radial interval is invalid")
        if not math.isclose(self.total_relative_l2, 0.012, abs_tol=0.0):
            raise ValueError("content frequency carrier budget must remain 0.012")


def _validate_latent_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) != 4 or any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("frequency carrier shape must be positive NCHW")
    batch, channels, height, width = shape
    if height < 4 or width < 4:
        raise ValueError("frequency carrier spatial dimensions must be at least 4")
    return batch, channels, height, width


def radial_frequency_mask(height: int, width: int, spec: FrequencyCarrierSpec) -> np.ndarray:
    """Return the exact rFFT mask for one finite carrier interval."""

    if not isinstance(height, int) or isinstance(height, bool) or height < 2:
        raise ValueError("height must be an integer of at least 2")
    if not isinstance(width, int) or isinstance(width, bool) or width < 2:
        raise ValueError("width must be an integer of at least 2")
    vertical = np.fft.fftfreq(height)[:, None]
    horizontal = np.fft.rfftfreq(width)[None, :]
    radius = np.hypot(vertical, horizontal) / np.hypot(0.5, 0.5)
    upper = radius <= spec.max_radius if spec.max_inclusive else radius < spec.max_radius
    mask = (radius >= spec.min_radius) & upper
    if not np.any(mask):
        raise ValueError("frequency carrier interval is empty for the requested shape")
    return mask


def _carrier_domain(spec: FrequencyCarrierSpec, channels: int, height: int, width: int, channel: int) -> str:
    return (
        f"{spec.domain_prefix}/{spec.carrier_method_id}/{_CARRIER_VERSION}/"
        f"channels={channels}/height={height}/width={width}/channel={channel}"
    )


def reconstruct_frequency_carrier(
    detection_key: str | bytes | bytearray | memoryview,
    shape: tuple[int, ...],
    spec: FrequencyCarrierSpec,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Rebuild a carrier from key, frozen method identity, and public shape only."""

    batch, channels, height, width = _validate_latent_shape(shape)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise TypeError("frequency carrier requires a floating torch dtype")
    mask = torch.from_numpy(radial_frequency_mask(height, width, spec)).to(device=device)
    channel_carriers: list[torch.Tensor] = []
    for channel in range(channels):
        signs = prg_rademacher(
            detection_key,
            _carrier_domain(spec, channels, height, width, channel),
            tuple(mask.shape),
            dtype=np.float32,
        )
        spectrum = torch.from_numpy(signs).to(device=device)
        spectrum = torch.where(mask, spectrum, torch.zeros_like(spectrum))
        channel_carriers.append(torch.fft.irfft2(spectrum, s=(height, width), norm="ortho"))
    carrier = torch.stack(channel_carriers, dim=0)
    norm = torch.linalg.vector_norm(carrier.to(torch.float64))
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("reconstructed frequency carrier has invalid norm")
    carrier = (carrier / norm.to(carrier.dtype)).unsqueeze(0).expand(batch, -1, -1, -1)
    return carrier.to(dtype=dtype)


def _balanced_pair_domain(
    spec: FrequencyCarrierSpec,
    channels: int,
    height: int,
    width: int,
    channel: int,
    block_index: int,
    canonical_bound_token: str,
    purpose: str,
) -> str:
    if purpose not in {"pair-order", "pair-sign"}:
        raise ValueError("balanced carrier purpose is not frozen")
    if not canonical_bound_token or any(
        character.isspace() for character in canonical_bound_token
    ):
        raise ValueError("balanced carrier canonical bound token is invalid")
    return (
        f"carrier_method_id={spec.carrier_method_id}/"
        f"channels={channels}/height={height}/width={width}/channel={channel}/"
        f"block={block_index}/bounds={canonical_bound_token}/purpose={purpose}"
    )


def _balanced_block_components(
    detection_key: str | bytes | bytearray | memoryview,
    shape: tuple[int, ...],
    spec: FrequencyCarrierSpec,
    radial_blocks: tuple[BalancedRadialBlock, ...],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, ...]:
    """Construct fixed float64 equal-energy channel-block components."""

    batch, channels, height, width = _validate_latent_shape(shape)
    if batch != 1:
        raise ValueError("balanced frequency carrier requires public NCHW shape with N=1")
    if len(radial_blocks) != 4:
        raise ValueError("balanced frequency carrier requires exactly four radial blocks")
    shell_mask = radial_frequency_mask(height, width, spec)
    union = np.zeros_like(shell_mask)
    boundary_columns = np.zeros_like(shell_mask)
    boundary_columns[:, 0] = True
    if width % 2 == 0:
        boundary_columns[:, width // 2] = True
    component_weight = 1.0 / math.sqrt(4.0 * channels)
    components: list[torch.Tensor] = []
    for block_index, (minimum, maximum, maximum_inclusive, bound_token) in enumerate(
        radial_blocks
    ):
        block_spec = FrequencyCarrierSpec(
            domain_prefix=spec.domain_prefix,
            carrier_method_id=spec.carrier_method_id,
            min_radius=minimum,
            max_radius=maximum,
            max_inclusive=maximum_inclusive,
            total_relative_l2=spec.total_relative_l2,
        )
        block_mask = radial_frequency_mask(height, width, block_spec)
        if np.any(union & block_mask):
            raise ValueError("balanced frequency blocks must be disjoint")
        union |= block_mask
        active_mask = block_mask & ~boundary_columns
        coordinates = [
            tuple(int(value) for value in coordinate)
            for coordinate in np.argwhere(active_mask)
        ]
        if len(coordinates) < 2:
            raise ValueError("balanced frequency block has fewer than two active coefficients")
        for channel in range(channels):
            order_domain = _balanced_pair_domain(
                spec,
                channels,
                height,
                width,
                channel,
                block_index,
                bound_token,
                "pair-order",
            )
            raw_order = prg_bytes(detection_key, order_domain, 32 * len(coordinates))
            tokens = [
                raw_order[index * 32 : (index + 1) * 32]
                for index in range(len(coordinates))
            ]
            ordered = sorted(
                zip(tokens, coordinates),
                key=lambda item: (item[0], item[1][0], item[1][1]),
            )
            pair_count = len(coordinates) // 2
            sign_domain = _balanced_pair_domain(
                spec,
                channels,
                height,
                width,
                channel,
                block_index,
                bound_token,
                "pair-sign",
            )
            signs = prg_rademacher(
                detection_key,
                sign_domain,
                (pair_count,),
                dtype=np.float32,
            )
            spectrum = torch.zeros(
                (height, width // 2 + 1),
                dtype=torch.float64,
                device=device,
            )
            for pair_index, sign in enumerate(signs):
                first = ordered[2 * pair_index][1]
                second = ordered[2 * pair_index + 1][1]
                spectrum[first] = float(sign)
                spectrum[second] = -float(sign)
            if not bool(torch.isfinite(spectrum).all()):
                raise ValueError("balanced frequency spectrum is not finite")
            if float(spectrum.sum().item()) != 0.0:
                raise RuntimeError("balanced frequency block does not have exact zero sum")
            component = torch.fft.irfft2(spectrum, s=(height, width), norm="ortho")
            roundtrip = torch.fft.rfft2(component, norm="ortho").real
            if not torch.allclose(roundtrip, spectrum, rtol=1e-10, atol=1e-10):
                raise RuntimeError("balanced frequency component failed rFFT roundtrip")
            support = torch.from_numpy(active_mask).to(device=device)
            if not torch.allclose(
                roundtrip[~support],
                torch.zeros_like(roundtrip[~support]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError("balanced frequency component escaped its public support")
            norm = torch.linalg.vector_norm(component)
            if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
                raise RuntimeError("balanced frequency component has invalid norm")
            normalized = component / norm * component_weight
            if not bool(torch.isfinite(normalized).all()):
                raise RuntimeError("balanced normalized component is not finite")
            components.append(normalized)
    if not np.array_equal(union, shell_mask):
        raise ValueError("balanced frequency blocks must exactly partition the carrier band")
    return tuple(components)


def reconstruct_balanced_frequency_carrier(
    detection_key: str | bytes | bytearray | memoryview,
    shape: tuple[int, ...],
    spec: FrequencyCarrierSpec,
    radial_blocks: tuple[BalancedRadialBlock, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Rebuild the frozen zero-sum, equal-precast-energy block carrier."""

    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise TypeError("balanced frequency carrier requires a floating torch dtype")
    _, channels, height, width = _validate_latent_shape(shape)
    components = _balanced_block_components(
        detection_key,
        shape,
        spec,
        radial_blocks,
        device=device,
    )
    carrier = torch.zeros((channels, height, width), dtype=torch.float64, device=device)
    for component_index, component in enumerate(components):
        channel = component_index % channels
        carrier[channel] += component
    norm = torch.linalg.vector_norm(carrier)
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("balanced reconstructed carrier has invalid norm")
    if not math.isclose(float(norm.item()), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("balanced precast component energies do not combine to unit L2")
    return carrier.unsqueeze(0).to(dtype=dtype)


def _relative_l2(base: torch.Tensor, candidate: torch.Tensor) -> BudgetMeasurement:
    if base.shape != candidate.shape or base.dtype != candidate.dtype or base.device != candidate.device:
        raise ValueError("actual callback base and candidate identity must match")
    base64 = base.to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    base_l2 = float(torch.linalg.vector_norm(base64).item())
    if not math.isfinite(base_l2) or base_l2 == 0.0:
        raise ValueError("relative L2 budget requires a finite nonzero latent")
    perturbation_l2 = float(torch.linalg.vector_norm(candidate64 - base64).item())
    return BudgetMeasurement(
        dtype=str(base.dtype),
        base_l2=base_l2,
        perturbation_l2=perturbation_l2,
        relative_l2=perturbation_l2 / base_l2,
    )


def _inject_frequency_carrier_with_reconstructor(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    spec: FrequencyCarrierSpec,
    reconstructor: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, BudgetMeasurement]:
    if not isinstance(latents, torch.Tensor):
        raise TypeError("callback latents must be a torch Tensor")
    if latents.ndim != 4 or not latents.dtype.is_floating_point:
        raise ValueError("callback latents must be floating NCHW")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("callback latents must be finite")
    with torch.no_grad():
        carrier = reconstructor(
            detection_key,
            tuple(latents.shape),
            spec,
            dtype=torch.float32,
            device=latents.device,
        ).to(torch.float64)
        base64 = latents.to(torch.float64)
        base_l2 = torch.linalg.vector_norm(base64)
        carrier_l2 = torch.linalg.vector_norm(carrier)
        if not bool(torch.isfinite(base_l2)) or float(base_l2.item()) == 0.0:
            raise ValueError("frequency injection requires a finite nonzero latent")
        proposed_delta = carrier * (base_l2 * spec.total_relative_l2 / carrier_l2)

        def candidate_at(scale: float) -> torch.Tensor:
            return (base64 + scale * proposed_delta).to(dtype=latents.dtype)

        candidate = candidate_at(1.0)
        measurement = _relative_l2(latents, candidate)
        if measurement.relative_l2 > spec.total_relative_l2:
            low = 0.0
            high = 1.0
            best = latents.detach().clone()
            best_measurement = _relative_l2(latents, best)
            for _ in range(80):
                middle = (low + high) / 2.0
                trial = candidate_at(middle)
                trial_measurement = _relative_l2(latents, trial)
                if trial_measurement.relative_l2 <= spec.total_relative_l2:
                    low = middle
                    best = trial
                    best_measurement = trial_measurement
                else:
                    high = middle
            candidate = best
            measurement = best_measurement
        if measurement.perturbation_l2 == 0.0:
            raise RuntimeError("frequency budget produced no change in the actual callback dtype")
        if measurement.relative_l2 > spec.total_relative_l2:
            raise RuntimeError("actual callback tensor exceeds the frozen carrier budget")
        return candidate, measurement


def inject_frequency_carrier(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    spec: FrequencyCarrierSpec,
) -> tuple[torch.Tensor, BudgetMeasurement]:
    """Inject one carrier under a budget measured on the actual callback dtype."""

    return _inject_frequency_carrier_with_reconstructor(
        latents,
        detection_key,
        spec,
        reconstruct_frequency_carrier,
    )


def inject_balanced_frequency_carrier(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    spec: FrequencyCarrierSpec,
    radial_blocks: tuple[BalancedRadialBlock, ...],
) -> tuple[torch.Tensor, BudgetMeasurement]:
    """Inject the frozen balanced-block carrier using the shared actual-dtype budget path."""

    def reconstruct(
        key: str | bytes | bytearray | memoryview,
        shape: tuple[int, ...],
        carrier_spec: FrequencyCarrierSpec,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        return reconstruct_balanced_frequency_carrier(
            key,
            shape,
            carrier_spec,
            radial_blocks,
            dtype=dtype,
            device=device,
        )

    return _inject_frequency_carrier_with_reconstructor(latents, detection_key, spec, reconstruct)


def score_frequency_image(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: Any,
    spec: FrequencyCarrierSpec,
) -> float:
    """Blind normalized correlation from final RGB, key, and public assets."""

    observation = encode_final_rgb_image(
        image,
        frozen_public_assets.image_processor,
        frozen_public_assets.vae,
    )
    carrier = reconstruct_frequency_carrier(
        detection_key,
        tuple(observation.shape),
        spec,
        dtype=torch.float32,
        device=observation.device,
    )
    observation_spectrum = torch.fft.rfft2(observation.to(torch.float32), norm="ortho")
    carrier_spectrum = torch.fft.rfft2(carrier, norm="ortho")
    _, _, height, width = observation.shape
    mask = torch.from_numpy(radial_frequency_mask(height, width, spec)).to(observation.device)
    observed = observation_spectrum.real[..., mask].reshape(-1).to(torch.float64)
    expected = carrier_spectrum.real[..., mask].reshape(-1).to(torch.float64)
    observed = observed - observed.mean()
    expected = expected - expected.mean()
    denominator = torch.linalg.vector_norm(observed) * torch.linalg.vector_norm(expected)
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) == 0.0:
        raise ValueError("blind frequency score requires non-constant finite image evidence")
    score = float(torch.dot(observed, expected).item() / denominator.item())
    if not math.isfinite(score):
        raise RuntimeError("blind frequency score is not finite")
    return score


def score_frequency_image_block_centered_normalized_median(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: Any,
    spec: FrequencyCarrierSpec,
    radial_blocks: tuple[tuple[float, float, bool], ...],
) -> float:
    """Blind equal-weight median of independently centered channel-block correlations."""

    if not radial_blocks:
        raise ValueError("block-normalized frequency score requires fixed radial blocks")
    observation = encode_final_rgb_image(
        image,
        frozen_public_assets.image_processor,
        frozen_public_assets.vae,
    )
    if observation.shape[0] != 1:
        raise ValueError("block-normalized frequency score requires one ordinary image")
    carrier = reconstruct_frequency_carrier(
        detection_key,
        tuple(observation.shape),
        spec,
        dtype=torch.float32,
        device=observation.device,
    )
    return _score_block_centered_normalized_median_observation(
        observation,
        carrier,
        spec,
        radial_blocks,
    )


def score_balanced_frequency_image_block_centered_normalized_median(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: Any,
    spec: FrequencyCarrierSpec,
    radial_blocks: tuple[BalancedRadialBlock, ...],
) -> float:
    """Blind unchanged block statistic using the balanced carrier reconstruction."""

    observation = encode_final_rgb_image(
        image,
        frozen_public_assets.image_processor,
        frozen_public_assets.vae,
    )
    if observation.shape[0] != 1:
        raise ValueError("block-normalized frequency score requires one ordinary image")
    carrier = reconstruct_balanced_frequency_carrier(
        detection_key,
        tuple(observation.shape),
        spec,
        radial_blocks,
        dtype=torch.float32,
        device=observation.device,
    )
    return _score_block_centered_normalized_median_observation(
        observation,
        carrier,
        spec,
        radial_blocks,
    )


def _score_block_centered_normalized_median_observation(
    observation: torch.Tensor,
    carrier: torch.Tensor,
    spec: FrequencyCarrierSpec,
    radial_blocks: tuple[tuple[float, float, bool] | BalancedRadialBlock, ...],
) -> float:
    observation_spectrum = torch.fft.rfft2(observation.to(torch.float32), norm="ortho").real
    carrier_spectrum = torch.fft.rfft2(carrier, norm="ortho").real
    if not bool(torch.isfinite(observation_spectrum).all()) or not bool(
        torch.isfinite(carrier_spectrum).all()
    ):
        raise ValueError("block-normalized frequency score requires finite spectra")

    _, channels, height, width = observation.shape
    block_scores: list[float] = []
    union = torch.zeros(
        (height, width // 2 + 1),
        dtype=torch.bool,
        device=observation.device,
    )
    for block in radial_blocks:
        minimum, maximum, maximum_inclusive = block[:3]
        block_spec = FrequencyCarrierSpec(
            domain_prefix=spec.domain_prefix,
            carrier_method_id=spec.carrier_method_id,
            min_radius=minimum,
            max_radius=maximum,
            max_inclusive=maximum_inclusive,
            total_relative_l2=spec.total_relative_l2,
        )
        mask = torch.from_numpy(radial_frequency_mask(height, width, block_spec)).to(
            device=observation.device
        )
        if bool(torch.any(union & mask)):
            raise ValueError("block-normalized frequency blocks must be disjoint")
        union |= mask
        for channel in range(channels):
            observed = observation_spectrum[0, channel][mask].reshape(-1).to(torch.float64)
            expected = carrier_spectrum[0, channel][mask].reshape(-1).to(torch.float64)
            if observed.numel() == 0 or expected.numel() == 0:
                raise ValueError("block-normalized frequency score contains an empty block")
            observed = observed - observed.mean()
            expected = expected - expected.mean()
            denominator = torch.linalg.vector_norm(observed) * torch.linalg.vector_norm(expected)
            if not bool(torch.isfinite(denominator)) or float(denominator.item()) == 0.0:
                raise ValueError(
                    "block-normalized frequency score requires finite nonzero block variance"
                )
            block_score = float(torch.dot(observed, expected).item() / denominator.item())
            if not math.isfinite(block_score):
                raise ValueError("block-normalized frequency correlation is not finite")
            block_scores.append(block_score)

    shell_mask = torch.from_numpy(radial_frequency_mask(height, width, spec)).to(
        device=observation.device
    )
    if not torch.equal(union, shell_mask):
        raise ValueError("block-normalized frequency blocks must exactly partition the carrier band")
    score = float(np.median(np.asarray(block_scores, dtype=np.float64)))
    if not math.isfinite(score):
        raise RuntimeError("block-normalized median frequency score is not finite")
    return score
