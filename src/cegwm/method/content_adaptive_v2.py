"""V2 content-adaptive dual-branch embedding with blind frozen detectors.

The six 4x4 signal maps and both tile-weight maps are private embed-side state.
Only irreversible scalar effects and branch shares may leave this module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from cegwm.method.hf import FrozenHFPublicAssets, reconstruct_hf_carrier, score_hf_image
from cegwm.method.lf import FrozenLFPublicAssets, reconstruct_lf_carrier, score_lf_image
from cegwm.shared.numerics import BudgetMeasurement

DINO_ASSET_ID = "facebook/dinov2-small"
DINO_ATTENTION_IMPLEMENTATION = "eager"
DINO_ATTENTION_LAYER = "last"
DINO_ATTENTION_STATISTIC = "mean_head_cls_to_patch"
TILE_GRID_SIDE = 4
TILE_COUNT = 16
PROBE_RELATIVE_L2_SCALES = (0.0005, 0.001)
PROBE_EVALUATION_COUNT = 64
PROBE_MEASUREMENT_ID = "baseline_differenced_branch_tile_two_scale_v1"
COMBINED_RELATIVE_L2 = 0.012
HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID = (
    "hf_content_tiles_semantic_stability_sensitivity_probe_v2"
)
LF_ADAPTIVE_EMBEDDING_TRANSFORM_ID = (
    "lf_content_tiles_texture_stability_sensitivity_probe_v2"
)
COMBINED_BUDGET_PROJECTOR_ID = "dual_branch_actual_dtype_relative_l2_v1"
JOINT_EVALUATED_CANDIDATE_ID = "content_adaptive_dual_branch_v2_clean_v1"
BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE = 1e-12
RGB8_TEXTURE_COMPLEXITY_MAX = 255.0 * math.sqrt(2.0)
COUNTERFACTUAL_EFFECT_FIELDS = (
    "semantic_importance_counterfactual_effect",
    "texture_complexity_counterfactual_effect",
    "lf_transfer_stability_counterfactual_effect",
    "hf_transfer_stability_counterfactual_effect",
    "lf_local_perturbation_sensitivity_counterfactual_effect",
    "hf_local_perturbation_sensitivity_counterfactual_effect",
)
_SIGNAL_FIELDS = (
    "semantic_importance",
    "texture_complexity",
    "lf_transfer_stability",
    "hf_transfer_stability",
    "lf_local_perturbation_sensitivity",
    "hf_local_perturbation_sensitivity",
)
_PUBLIC_PROBE_MASTER = b"CEG-WM/public-content-probe-master/v2"


@dataclass(frozen=True, slots=True)
class ContentSignals:
    """Six private real per-tile maps used only while embedding."""

    semantic_importance: tuple[float, ...]
    texture_complexity: tuple[float, ...]
    lf_transfer_stability: tuple[float, ...]
    hf_transfer_stability: tuple[float, ...]
    lf_local_perturbation_sensitivity: tuple[float, ...]
    hf_local_perturbation_sensitivity: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ProbeObservation:
    """Temporary public RGB/VAE observations; never runner output."""

    rgb: torch.Tensor
    vae: torch.Tensor


@dataclass(frozen=True, slots=True)
class PublicProbeMaps:
    """Four private maps derived only from baseline-differenced responses."""

    lf_transfer_stability: tuple[float, ...]
    hf_transfer_stability: tuple[float, ...]
    lf_local_perturbation_sensitivity: tuple[float, ...]
    hf_local_perturbation_sensitivity: tuple[float, ...]
    evaluation_count: int = PROBE_EVALUATION_COUNT


@dataclass(frozen=True, slots=True)
class ContentAllocation:
    """Private embed-side tile allocation; never serialize this object."""

    lf_tile_weights: tuple[float, ...]
    hf_tile_weights: tuple[float, ...]
    lf_branch_share: float
    hf_branch_share: float
    counterfactual_effects: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        _positive_vector(self.lf_tile_weights, "lf_tile_weights")
        _positive_vector(self.hf_tile_weights, "hf_tile_weights")
        _validate_public_branch_shares(self.lf_branch_share, self.hf_branch_share)
        if len(self.counterfactual_effects) != len(COUNTERFACTUAL_EFFECT_FIELDS):
            raise ValueError("content allocation must carry exactly six counterfactual effects")
        for name, value in zip(COUNTERFACTUAL_EFFECT_FIELDS, self.counterfactual_effects, strict=True):
            _nonnegative_finite_scalar(value, name)


@dataclass(frozen=True, slots=True)
class ContentAdaptiveMeasurement:
    """Irreversible aggregate scalars safe for runner export."""

    combined_budget: BudgetMeasurement
    lf_effective_relative_l2: float
    hf_effective_relative_l2: float
    lf_branch_share: float
    hf_branch_share: float
    semantic_importance_counterfactual_effect: float
    texture_complexity_counterfactual_effect: float
    lf_transfer_stability_counterfactual_effect: float
    hf_transfer_stability_counterfactual_effect: float
    lf_local_perturbation_sensitivity_counterfactual_effect: float
    hf_local_perturbation_sensitivity_counterfactual_effect: float
    probe_evaluation_count: int

    def __post_init__(self) -> None:
        _validate_public_branch_shares(self.lf_branch_share, self.hf_branch_share)
        for name in COUNTERFACTUAL_EFFECT_FIELDS:
            _nonnegative_finite_scalar(getattr(self, name), name)
        if self.probe_evaluation_count != PROBE_EVALUATION_COUNT:
            raise ValueError("v2 measurement must report exactly 64 probe evaluations")

    @property
    def minimum_counterfactual_effect(self) -> float:
        return min(getattr(self, name) for name in COUNTERFACTUAL_EFFECT_FIELDS)


@dataclass(frozen=True, slots=True)
class ContentBlindScores:
    """Blind base-detector scores from the ordinary final RGB image."""

    lf: float
    hf: float
    content: float


def _nonnegative_finite_scalar(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return scalar


def _validate_public_branch_shares(lf_share: Any, hf_share: Any) -> None:
    values: list[float] = []
    for name, value in (("lf_branch_share", lf_share), ("hf_branch_share", hf_share)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real scalar")
        scalar = float(value)
        if not math.isfinite(scalar) or not 0.0 < scalar < 1.0:
            raise ValueError(f"{name} must be finite and strictly between zero and one")
        values.append(scalar)
    if not math.isclose(
        values[0] + values[1], 1.0, rel_tol=0.0,
        abs_tol=BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("public branch shares must sum to one within the frozen tolerance")


def _finite_vector(values: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim != 1 or tensor.numel() != TILE_COUNT:
        raise ValueError(f"{name} must contain exactly 16 tile scalars")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _positive_vector(values: Any, name: str) -> torch.Tensor:
    tensor = _finite_vector(values, name)
    if not bool((tensor > 0.0).all()):
        raise ValueError(f"{name} must be strictly positive")
    return tensor


def _bounded_vector(values: Any, name: str) -> torch.Tensor:
    tensor = _finite_vector(values, name)
    if not bool(((0.0 <= tensor) & (tensor <= 1.0)).all()):
        raise ValueError(f"{name} must lie in the closed unit interval")
    return tensor


def _rgb8_texture_vector(values: Any) -> torch.Tensor:
    tensor = _finite_vector(values, "texture_complexity")
    if not bool(((0.0 <= tensor) & (tensor <= RGB8_TEXTURE_COMPLEXITY_MAX)).all()):
        raise ValueError(
            "texture_complexity must lie in the frozen RGB8 range "
            "[0, 255*sqrt(2)]"
        )
    return tensor


def _unit_interval(values: torch.Tensor) -> torch.Tensor:
    minimum = values.min()
    span = values.max() - minimum
    if float(span.item()) == 0.0:
        return torch.full_like(values, 0.5)
    return (values - minimum) / span


def dino_last_layer_cls_patch_tiles(
    image: Any,
    processor: Any,
    model: Any,
) -> tuple[float, ...]:
    """Extract only semantic importance from real final-layer DINO attention."""

    if not callable(processor) or not callable(model):
        raise TypeError("DINO processor and model must be callable")
    if getattr(getattr(model, "config", None), "_attn_implementation", None) != DINO_ATTENTION_IMPLEMENTATION:
        raise RuntimeError("DINO attention implementation must be eager")
    encoded = processor(images=image, return_tensors="pt")
    if not isinstance(encoded, Mapping) or not encoded:
        raise RuntimeError("DINO processor did not return model inputs")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("DINO model device cannot be resolved") from error
    inputs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in encoded.items()
    }
    with torch.no_grad():
        output = model(**inputs, output_attentions=True, return_dict=True)
    attentions = getattr(output, "attentions", None)
    if not isinstance(attentions, (tuple, list)) or not attentions:
        raise RuntimeError("DINO did not return real attention tensors")
    last = attentions[-1]
    if not isinstance(last, torch.Tensor) or last.ndim != 4:
        raise RuntimeError("DINO last-layer attention must be a rank-4 tensor")
    if last.shape[0] != 1 or last.shape[1] < 1 or last.shape[2] != last.shape[3]:
        raise RuntimeError("DINO last-layer attention has an invalid shape")
    if not bool(torch.isfinite(last).all()):
        raise RuntimeError("DINO last-layer attention is nonfinite")
    patch_count = last.shape[-1] - 1
    patch_side = math.isqrt(patch_count)
    if patch_count < TILE_COUNT or patch_side * patch_side != patch_count:
        raise RuntimeError("DINO CLS-to-patch attention is not a square patch grid")
    patch_grid = last[0, :, 0, 1:].mean(dim=0).reshape(1, 1, patch_side, patch_side)
    vector = functional.adaptive_avg_pool2d(
        patch_grid, (TILE_GRID_SIDE, TILE_GRID_SIDE)
    ).reshape(-1).to(torch.float64)
    if vector.numel() != TILE_COUNT or not bool(torch.isfinite(vector).all()):
        raise RuntimeError("DINO semantic importance is invalid")
    if float(torch.linalg.vector_norm(vector).item()) == 0.0:
        raise RuntimeError("DINO semantic importance is identically zero")
    return tuple(float(value) for value in vector.tolist())


def rgb_texture_tiles(image: Any) -> tuple[float, ...]:
    """Compute bounded raw RGB8 gradient complexity; flat RGB maps to raw zero."""

    pixels = np.asarray(image).copy()
    if (
        pixels.ndim != 3
        or pixels.shape[2] != 3
        or pixels.shape[0] < 4
        or pixels.shape[1] < 4
        or pixels.dtype != np.uint8
    ):
        raise ValueError("texture input must be an RGB8 image of at least 4x4")
    values = torch.as_tensor(pixels, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("texture image must be finite")
    dy = torch.zeros_like(values)
    dx = torch.zeros_like(values)
    dy[1:] = values[1:] - values[:-1]
    dx[:, 1:] = values[:, 1:] - values[:, :-1]
    magnitude = torch.sqrt(dx.square() + dy.square()).mean(dim=2)
    magnitude = magnitude.reshape(1, 1, *magnitude.shape)
    pooled = functional.adaptive_avg_pool2d(
        magnitude, (TILE_GRID_SIDE, TILE_GRID_SIDE)
    ).reshape(-1)
    if not bool(
        torch.isfinite(pooled).all()
        and ((0.0 <= pooled) & (pooled <= RGB8_TEXTURE_COMPLEXITY_MAX)).all()
    ):
        raise RuntimeError("RGB8 texture complexity escaped its frozen finite range")
    return tuple(float(value) for value in pooled.tolist())


def _relative_l2(base: torch.Tensor, candidate: torch.Tensor) -> BudgetMeasurement:
    if base.shape != candidate.shape or base.dtype != candidate.dtype or base.device != candidate.device:
        raise ValueError("actual callback base and candidate identity must match")
    base64 = base.to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    base_l2 = float(torch.linalg.vector_norm(base64).item())
    if not math.isfinite(base_l2) or base_l2 == 0.0:
        raise ValueError("relative L2 requires a finite nonzero callback latent")
    delta_l2 = float(torch.linalg.vector_norm(candidate64 - base64).item())
    return BudgetMeasurement(str(base.dtype), base_l2, delta_l2, delta_l2 / base_l2)


def _public_probe_direction(shape: tuple[int, ...], branch: str, tile_index: int) -> torch.Tensor:
    if branch not in {"lf", "hf"} or not 0 <= tile_index < TILE_COUNT:
        raise ValueError("public probe branch or tile is invalid")
    batch, channels, height, width = shape
    spectrum_shape = (batch, channels, height, width // 2 + 1)
    count = math.prod(spectrum_shape)
    domain = (
        f"branch={branch}/tile={tile_index}/shape={'x'.join(str(value) for value in shape)}/v2"
    ).encode("ascii")
    output = bytearray()
    counter = 0
    while len(output) < count:
        output.extend(hashlib.sha256(
            _PUBLIC_PROBE_MASTER + b"/" + domain + counter.to_bytes(8, "big")
        ).digest())
        counter += 1
    signs = torch.tensor(
        [1.0 if value & 1 else -1.0 for value in output[:count]], dtype=torch.float64,
    ).reshape(spectrum_shape)
    vertical = np.fft.fftfreq(height)[:, None]
    horizontal = np.fft.rfftfreq(width)[None, :]
    radius = np.hypot(vertical, horizontal) / np.hypot(0.5, 0.5)
    band = (
        (radius >= 0.14) & (radius <= 0.24)
        if branch == "lf" else (radius >= 0.58) & (radius <= 1.0)
    )
    if not np.any(band):
        raise ValueError("public probe branch band is empty for the callback shape")
    spectrum = signs * torch.from_numpy(band).reshape(1, 1, height, width // 2 + 1)
    direction = torch.fft.irfft2(spectrum, s=(height, width), norm="ortho")
    if not bool(torch.isfinite(direction).all()) or float(torch.linalg.vector_norm(direction).item()) == 0.0:
        raise RuntimeError("public branch probe direction is invalid")
    return direction


def make_public_tile_probe(
    latents: torch.Tensor,
    branch: str,
    tile_index: int,
    relative_l2: float,
) -> tuple[torch.Tensor, BudgetMeasurement]:
    """Create one non-cumulative key-independent actual-dtype v2 probe."""

    if relative_l2 not in PROBE_RELATIVE_L2_SCALES:
        raise ValueError("probe relative-L2 scale is not frozen")
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or not latents.dtype.is_floating_point:
        raise TypeError("probe base must be a floating NCHW torch Tensor")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("probe base must be finite")
    _, _, height, width = latents.shape
    row, column = divmod(tile_index, TILE_GRID_SIDE)
    row_start, row_end = height * row // 4, height * (row + 1) // 4
    col_start, col_end = width * column // 4, width * (column + 1) // 4
    if row_start == row_end or col_start == col_end:
        raise ValueError("callback latent is too small for the fixed 4x4 probe grid")
    direction = _public_probe_direction(tuple(latents.shape), branch, tile_index).to(latents.device)
    mask = torch.zeros_like(direction)
    mask[..., row_start:row_end, col_start:col_end] = 1.0
    direction *= mask
    base64 = latents.to(torch.float64)
    base_l2 = torch.linalg.vector_norm(base64)
    direction_l2 = torch.linalg.vector_norm(direction)
    if float(base_l2.item()) == 0.0 or float(direction_l2.item()) == 0.0:
        raise ValueError("probe requires nonzero base and direction")
    delta = direction * (base_l2 * relative_l2 / direction_l2)

    def candidate_at(scale: float) -> torch.Tensor:
        return (base64 + scale * delta).to(latents.dtype)

    low, high = 0.0, 2.0
    best = latents.detach().clone()
    best_measurement = _relative_l2(latents, best)
    for _ in range(96):
        middle = (low + high) / 2.0
        trial = candidate_at(middle)
        measurement = _relative_l2(latents, trial)
        if measurement.relative_l2 <= relative_l2:
            low, best, best_measurement = middle, trial, measurement
        else:
            high = middle
    if best_measurement.perturbation_l2 == 0.0:
        raise RuntimeError("actual callback dtype cannot form a nonzero probe within budget")
    return best, best_measurement


def _observation_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be nonempty and finite")
    return tensor


def _difference(candidate: torch.Tensor, baseline: torch.Tensor, name: str) -> torch.Tensor:
    value = _observation_tensor(candidate, name)
    if value.shape != baseline.shape:
        raise ValueError(f"{name} shape differs from its baseline")
    return (value - baseline).reshape(-1)


def _alignment_and_gain_consistency(
    small: torch.Tensor,
    large: torch.Tensor,
    small_scale: float,
    large_scale: float,
) -> float:
    small_norm = float(torch.linalg.vector_norm(small).item())
    large_norm = float(torch.linalg.vector_norm(large).item())
    if small_norm == 0.0 or large_norm == 0.0:
        return 0.0
    alignment = float(torch.dot(small, large).item()) / (small_norm * large_norm)
    alignment = min(1.0, max(0.0, alignment))
    small_gain = small_norm / small_scale
    large_gain = large_norm / large_scale
    consistency = min(small_gain, large_gain) / max(small_gain, large_gain)
    value = alignment * consistency
    if not math.isfinite(value):
        raise RuntimeError("probe transfer stability is nonfinite")
    return value


def _robust_sensitivity(values: list[float]) -> list[float]:
    positives = [value for value in values if value > 0.0]
    if not positives:
        return [0.0] * len(values)
    kappa = float(np.median(np.asarray(positives, dtype=np.float64)))
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise RuntimeError("positive sensitivity median is invalid")
    return [value / (value + kappa) for value in values]


def evaluate_public_probes(
    latents: torch.Tensor,
    baseline: ProbeObservation,
    evaluator: Callable[[str, torch.Tensor], ProbeObservation],
) -> PublicProbeMaps:
    """Run 64 probes and derive response-only two-scale stability/sensitivity."""

    if not callable(evaluator):
        raise TypeError("probe evaluator must be callable")
    baseline_rgb = _observation_tensor(baseline.rgb, "baseline RGB observation")
    baseline_vae = _observation_tensor(baseline.vae, "baseline VAE observation")
    differences: dict[tuple[str, int, int, str], torch.Tensor] = {}
    raw: dict[str, list[float]] = {"rgb": [], "vae": []}
    order: list[tuple[str, int, int]] = []
    for tile_index in range(TILE_COUNT):
        for branch in ("lf", "hf"):
            for scale_index, scale in enumerate(PROBE_RELATIVE_L2_SCALES):
                probe, measurement = make_public_tile_probe(latents, branch, tile_index, scale)
                if measurement.perturbation_l2 == 0.0 or measurement.relative_l2 > scale:
                    raise RuntimeError("invalid actual-dtype public probe")
                observation = evaluator(branch, probe)
                if not isinstance(observation, ProbeObservation):
                    raise TypeError("probe evaluator must return ProbeObservation")
                rgb_delta = _difference(observation.rgb, baseline_rgb, "candidate RGB observation")
                vae_delta = _difference(observation.vae, baseline_vae, "candidate VAE observation")
                differences[(branch, tile_index, scale_index, "rgb")] = rgb_delta
                differences[(branch, tile_index, scale_index, "vae")] = vae_delta
                raw["rgb"].append(float(torch.linalg.vector_norm(rgb_delta).item()) / scale)
                raw["vae"].append(float(torch.linalg.vector_norm(vae_delta).item()) / scale)
                order.append((branch, tile_index, scale_index))
    if len(order) != PROBE_EVALUATION_COUNT:
        raise RuntimeError("content analysis did not execute exactly 64 probe evaluations")
    normalized = {modality: _robust_sensitivity(values) for modality, values in raw.items()}
    normalized_by_key = {
        (branch, tile, scale_index, modality): normalized[modality][index]
        for modality in ("rgb", "vae")
        for index, (branch, tile, scale_index) in enumerate(order)
    }
    stability: dict[str, list[float]] = {"lf": [], "hf": []}
    sensitivity: dict[str, list[float]] = {"lf": [], "hf": []}
    small_scale, large_scale = PROBE_RELATIVE_L2_SCALES
    for branch in ("lf", "hf"):
        for tile_index in range(TILE_COUNT):
            modality_stability = [
                _alignment_and_gain_consistency(
                    differences[(branch, tile_index, 0, modality)],
                    differences[(branch, tile_index, 1, modality)],
                    small_scale,
                    large_scale,
                )
                for modality in ("rgb", "vae")
            ]
            stability[branch].append(sum(modality_stability) / 2.0)
            sensitivity[branch].append(sum(
                normalized_by_key[(branch, tile_index, scale_index, modality)]
                for scale_index in range(2)
                for modality in ("rgb", "vae")
            ) / 4.0)
    return PublicProbeMaps(
        tuple(stability["lf"]),
        tuple(stability["hf"]),
        tuple(sensitivity["lf"]),
        tuple(sensitivity["hf"]),
    )


def _allocation_vectors(signals: ContentSignals) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    semantic = _unit_interval(_finite_vector(signals.semantic_importance, "semantic_importance"))
    texture_raw = _rgb8_texture_vector(signals.texture_complexity)
    texture = 0.5 + 0.5 * texture_raw / RGB8_TEXTURE_COMPLEXITY_MAX
    lf_stability = _bounded_vector(signals.lf_transfer_stability, "lf_transfer_stability")
    hf_stability = _bounded_vector(signals.hf_transfer_stability, "hf_transfer_stability")
    lf_sensitivity = _bounded_vector(
        signals.lf_local_perturbation_sensitivity, "lf_local_perturbation_sensitivity"
    )
    hf_sensitivity = _bounded_vector(
        signals.hf_local_perturbation_sensitivity, "hf_local_perturbation_sensitivity"
    )
    semantic_magnitude = 2.0 * torch.abs(semantic - 0.5)
    # Frozen monotonic directions: raw RGB8 texture disfavors LF and favours HF; each
    # stability helps only its own branch; sensitivity is always a penalty.
    lf_raw = (
        0.25 + 0.20 * semantic_magnitude + 0.30 * (1.0 - texture)
        + 0.30 * lf_stability + 0.20 * (1.0 - lf_sensitivity)
    )
    hf_raw = (
        0.25 + 0.20 * semantic_magnitude + 0.30 * texture
        + 0.30 * hf_stability + 0.20 * (1.0 - hf_sensitivity)
    )
    lf_strength = float(lf_raw.mean().item())
    hf_strength = float(hf_raw.mean().item())
    total = lf_strength + hf_strength
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("content signals cannot allocate the two branches")
    lf_weights = lf_raw / lf_raw.mean()
    hf_weights = hf_raw / hf_raw.mean()
    return lf_weights, hf_weights, lf_strength / total, hf_strength / total


def allocate_content(signals: ContentSignals) -> ContentAllocation:
    """Allocate both v2 transforms and measure six neutral counterfactuals."""

    lf, hf, lf_share, hf_share = _allocation_vectors(signals)
    original = torch.cat((lf, hf, torch.tensor([lf_share, hf_share], dtype=torch.float64)))
    effects: list[float] = []
    for field in _SIGNAL_FIELDS:
        values = {name: getattr(signals, name) for name in _SIGNAL_FIELDS}
        neutral_value = 0.0 if field == "texture_complexity" else 0.5
        values[field] = (neutral_value,) * TILE_COUNT
        alternate_signals = ContentSignals(**values)
        cf_lf, cf_hf, cf_lf_share, cf_hf_share = _allocation_vectors(alternate_signals)
        alternate = torch.cat((cf_lf, cf_hf, torch.tensor([cf_lf_share, cf_hf_share])))
        effect = float(torch.linalg.vector_norm(original - alternate).item())
        if not math.isfinite(effect) or effect < 0.0:
            raise RuntimeError(f"{field} neutral-counterfactual effect is invalid")
        effects.append(effect)
    return ContentAllocation(
        tuple(float(value) for value in lf.tolist()),
        tuple(float(value) for value in hf.tolist()),
        lf_share,
        hf_share,
        tuple(effects),
    )


def _tile_weight_map(weights: tuple[float, ...], latents: torch.Tensor) -> torch.Tensor:
    vector = _positive_vector(weights, "tile_weights").reshape(1, 1, 4, 4)
    return functional.interpolate(vector, size=latents.shape[-2:], mode="nearest").to(
        device=latents.device, dtype=torch.float64
    )


def _branch_transformed_delta(
    carrier: torch.Tensor,
    weights: tuple[float, ...],
    amplitude: torch.Tensor,
) -> torch.Tensor:
    transformed = carrier.to(torch.float64) * _tile_weight_map(weights, carrier)
    norm = torch.linalg.vector_norm(transformed)
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("adaptive embedding transform produced an invalid branch")
    return transformed / norm * amplitude


def _lf_transformed_delta(
    carrier: torch.Tensor, weights: tuple[float, ...], amplitude: torch.Tensor,
) -> torch.Tensor:
    return _branch_transformed_delta(carrier, weights, amplitude)


def _hf_transformed_delta(
    carrier: torch.Tensor, weights: tuple[float, ...], amplitude: torch.Tensor,
) -> torch.Tensor:
    return _branch_transformed_delta(carrier, weights, amplitude)


def embed_content_adaptive(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets,
    allocation: ContentAllocation,
) -> tuple[torch.Tensor, ContentAdaptiveMeasurement]:
    """Simultaneously embed separate v2 LF/HF transforms under one 0.012 budget."""

    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or not latents.dtype.is_floating_point:
        raise TypeError("content embedding requires floating NCHW callback latents")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("content embedding callback latents must be finite")
    if hf_assets.injection_step_index != 18 or lf_assets.injection_step_index != 18:
        raise ValueError("both frozen base carriers must use callback step 18")
    base64 = latents.to(torch.float64)
    base_l2 = torch.linalg.vector_norm(base64)
    if not bool(torch.isfinite(base_l2)) or float(base_l2.item()) == 0.0:
        raise ValueError("content embedding requires a finite nonzero latent")
    hf_base = reconstruct_hf_carrier(
        detection_key, tuple(latents.shape), hf_assets, dtype=torch.float32,
        device=latents.device,
    )
    lf_base = reconstruct_lf_carrier(
        detection_key, tuple(latents.shape), lf_assets, dtype=torch.float32,
        device=latents.device,
    )
    # V2/adaptive/joint identities never enter either frozen base PRG domain.
    hf_delta = _hf_transformed_delta(
        hf_base, allocation.hf_tile_weights,
        base_l2 * COMBINED_RELATIVE_L2 * allocation.hf_branch_share,
    )
    lf_delta = _lf_transformed_delta(
        lf_base, allocation.lf_tile_weights,
        base_l2 * COMBINED_RELATIVE_L2 * allocation.lf_branch_share,
    )
    if float(torch.linalg.vector_norm(hf_delta).item()) == 0.0 or float(torch.linalg.vector_norm(lf_delta).item()) == 0.0:
        raise RuntimeError("both adaptive branches must be nonzero")

    def candidate_at(scale: float) -> torch.Tensor:
        return (base64 + scale * (hf_delta + lf_delta)).to(latents.dtype)

    low, high = 0.0, 2.0
    best = latents.detach().clone()
    measurement = _relative_l2(latents, best)
    for _ in range(96):
        middle = (low + high) / 2.0
        trial = candidate_at(middle)
        trial_measurement = _relative_l2(latents, trial)
        if trial_measurement.relative_l2 <= COMBINED_RELATIVE_L2:
            low, best, measurement = middle, trial, trial_measurement
        else:
            high = middle
    hf_actual = _relative_l2(latents, (base64 + low * hf_delta).to(latents.dtype))
    lf_actual = _relative_l2(latents, (base64 + low * lf_delta).to(latents.dtype))
    if measurement.perturbation_l2 == 0.0 or measurement.relative_l2 > COMBINED_RELATIVE_L2:
        raise RuntimeError("combined actual-dtype embedding is zero or over budget")
    if hf_actual.perturbation_l2 == 0.0 or lf_actual.perturbation_l2 == 0.0:
        raise RuntimeError("both actual-dtype adaptive branches must remain nonzero")
    return best, ContentAdaptiveMeasurement(
        measurement,
        lf_actual.relative_l2,
        hf_actual.relative_l2,
        allocation.lf_branch_share,
        allocation.hf_branch_share,
        *allocation.counterfactual_effects,
        PROBE_EVALUATION_COUNT,
    )


def score_content_image(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets,
) -> ContentBlindScores:
    """Blind joint score from image, key, and frozen public VAE assets only."""

    lf = float(score_lf_image(image, detection_key, lf_assets))
    hf = float(score_hf_image(image, detection_key, hf_assets))
    if not math.isfinite(lf) or not math.isfinite(hf):
        raise ValueError("blind content detector returned a nonfinite base score")
    return ContentBlindScores(lf=lf, hf=hf, content=min(lf, hf))
