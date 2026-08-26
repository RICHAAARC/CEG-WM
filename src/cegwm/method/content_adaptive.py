"""Content-adaptive dual-branch embedding with blind frozen base detectors.

All spatial state in this module is embed-side only.  The public HF/LF detector
identities remain unchanged and never receive the allocations produced here.
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
PROBE_RELATIVE_L2 = 0.001
COMBINED_RELATIVE_L2 = 0.012
HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID = "hf_content_tiles_attention_probe_v1"
LF_ADAPTIVE_EMBEDDING_TRANSFORM_ID = "lf_content_tiles_texture_probe_v1"
COMBINED_BUDGET_PROJECTOR_ID = "dual_branch_actual_dtype_relative_l2_v1"
JOINT_EVALUATED_CANDIDATE_ID = "content_adaptive_dual_branch_clean_v1"
BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE = 1e-12
COUNTERFACTUAL_EFFECT_FIELDS = (
    "semantic_attention_counterfactual_effect",
    "texture_energy_counterfactual_effect",
    "lf_probe_response_counterfactual_effect",
    "hf_probe_response_counterfactual_effect",
)
_PUBLIC_PROBE_MASTER = b"CEG-WM/public-content-probe-master/v1"


@dataclass(frozen=True, slots=True)
class ContentSignals:
    """Four real, per-tile embed-side signals."""

    semantic_attention: tuple[float, ...]
    texture_energy: tuple[float, ...]
    lf_probe_response: tuple[float, ...]
    hf_probe_response: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ContentAllocation:
    """Private embed-side tile allocation; never serialize this object."""

    lf_tile_weights: tuple[float, ...]
    hf_tile_weights: tuple[float, ...]
    lf_branch_share: float
    hf_branch_share: float
    counterfactual_effects: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _validate_public_branch_shares(self.lf_branch_share, self.hf_branch_share)
        if len(self.counterfactual_effects) != len(COUNTERFACTUAL_EFFECT_FIELDS):
            raise ValueError("content allocation must carry exactly four counterfactual effects")
        semantic, texture, lf_probe, hf_probe = self.counterfactual_effects
        _positive_finite_scalar(semantic, "semantic_attention_counterfactual_effect")
        _positive_finite_scalar(texture, "texture_energy_counterfactual_effect")
        _positive_finite_scalar(lf_probe, "lf_probe_response_counterfactual_effect")
        _positive_finite_scalar(hf_probe, "hf_probe_response_counterfactual_effect")


@dataclass(frozen=True, slots=True)
class ContentAdaptiveMeasurement:
    """Irreversible aggregate scalars safe for runner export."""

    combined_budget: BudgetMeasurement
    lf_effective_relative_l2: float
    hf_effective_relative_l2: float
    lf_branch_share: float
    hf_branch_share: float
    semantic_attention_counterfactual_effect: float
    texture_energy_counterfactual_effect: float
    lf_probe_response_counterfactual_effect: float
    hf_probe_response_counterfactual_effect: float
    probe_evaluation_count: int

    def __post_init__(self) -> None:
        _validate_public_branch_shares(self.lf_branch_share, self.hf_branch_share)
        _positive_finite_scalar(
            self.semantic_attention_counterfactual_effect,
            "semantic_attention_counterfactual_effect",
        )
        _positive_finite_scalar(
            self.texture_energy_counterfactual_effect,
            "texture_energy_counterfactual_effect",
        )
        _positive_finite_scalar(
            self.lf_probe_response_counterfactual_effect,
            "lf_probe_response_counterfactual_effect",
        )
        _positive_finite_scalar(
            self.hf_probe_response_counterfactual_effect,
            "hf_probe_response_counterfactual_effect",
        )

    @property
    def minimum_counterfactual_effect(self) -> float:
        """Minimum of the four exported effects; never independent constructor state."""

        return min(
            self.semantic_attention_counterfactual_effect,
            self.texture_energy_counterfactual_effect,
            self.lf_probe_response_counterfactual_effect,
            self.hf_probe_response_counterfactual_effect,
        )


@dataclass(frozen=True, slots=True)
class ContentBlindScores:
    """Blind base-detector scores from the ordinary final RGB image."""

    lf: float
    hf: float
    content: float


def _positive_finite_scalar(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
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
        values[0] + values[1],
        1.0,
        rel_tol=0.0,
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
    """Extract actual last-layer mean-head CLS-to-patch attention into 4x4 tiles."""

    if not callable(processor):
        raise TypeError("DINO image processor must be callable")
    if not callable(model):
        raise TypeError("DINO model must be callable")
    config = getattr(model, "config", None)
    implementation = getattr(config, "_attn_implementation", None)
    if implementation != DINO_ATTENTION_IMPLEMENTATION:
        raise RuntimeError("DINO attention implementation must be eager")
    encoded = processor(images=image, return_tensors="pt")
    if not isinstance(encoded, Mapping) or not encoded:
        raise RuntimeError("DINO processor did not return model inputs")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("DINO model device cannot be resolved") from error
    inputs: dict[str, Any] = {}
    for name, value in encoded.items():
        inputs[name] = value.to(device) if isinstance(value, torch.Tensor) else value
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
    cls_to_patch = last[0, :, 0, 1:].mean(dim=0)
    patch_grid = cls_to_patch.reshape(1, 1, patch_side, patch_side)
    tiles = functional.adaptive_avg_pool2d(patch_grid, (TILE_GRID_SIDE, TILE_GRID_SIDE))
    vector = tiles.reshape(-1).to(torch.float64)
    if vector.numel() != TILE_COUNT or not bool(torch.isfinite(vector).all()):
        raise RuntimeError("DINO tile attention is invalid")
    if float(torch.linalg.vector_norm(vector).item()) == 0.0:
        raise RuntimeError("DINO tile attention is identically zero")
    return tuple(float(value) for value in vector.tolist())


def rgb_texture_tiles(image: Any) -> tuple[float, ...]:
    """Compute public local RGB gradient energy on the fixed 4x4 grid."""

    pixels = np.asarray(image).copy()
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.shape[0] < 4 or pixels.shape[1] < 4:
        raise ValueError("texture input must be an RGB image of at least 4x4")
    values = torch.as_tensor(pixels, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("texture image must be finite")
    gray = values.mean(dim=2)
    dy = torch.zeros_like(gray)
    dx = torch.zeros_like(gray)
    dy[1:] = gray[1:] - gray[:-1]
    dx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    energy = torch.sqrt(dx.square() + dy.square()).reshape(1, 1, *gray.shape)
    pooled = functional.adaptive_avg_pool2d(energy, (TILE_GRID_SIDE, TILE_GRID_SIDE))
    result = pooled.reshape(-1)
    if float(torch.linalg.vector_norm(result).item()) == 0.0:
        raise RuntimeError("texture signal is identically zero")
    return tuple(float(value) for value in result.tolist())


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
        f"branch={branch}/tile={tile_index}/shape={'x'.join(str(value) for value in shape)}/v1"
    ).encode("ascii")
    output = bytearray()
    counter = 0
    while len(output) < count:
        output.extend(hashlib.sha256(_PUBLIC_PROBE_MASTER + b"/" + domain + counter.to_bytes(8, "big")).digest())
        counter += 1
    signs = torch.tensor(
        [1.0 if value & 1 else -1.0 for value in output[:count]],
        dtype=torch.float64,
    ).reshape(spectrum_shape)
    vertical = np.fft.fftfreq(height)[:, None]
    horizontal = np.fft.rfftfreq(width)[None, :]
    radius = np.hypot(vertical, horizontal) / np.hypot(0.5, 0.5)
    band = (
        (radius >= 0.14) & (radius <= 0.24)
        if branch == "lf"
        else (radius >= 0.58) & (radius <= 1.0)
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
) -> tuple[torch.Tensor, BudgetMeasurement]:
    """Create one non-cumulative, public, key-independent actual-dtype probe."""

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
    delta = direction * (base_l2 * PROBE_RELATIVE_L2 / direction_l2)

    def candidate_at(scale: float) -> torch.Tensor:
        return (base64 + scale * delta).to(latents.dtype)

    # Maximize the representable scale that remains within the frozen budget.
    low, high = 0.0, 2.0
    best = latents.detach().clone()
    best_measurement = _relative_l2(latents, best)
    for _ in range(96):
        middle = (low + high) / 2.0
        trial = candidate_at(middle)
        measurement = _relative_l2(latents, trial)
        if measurement.relative_l2 <= PROBE_RELATIVE_L2:
            low, best, best_measurement = middle, trial, measurement
        else:
            high = middle
    if best_measurement.perturbation_l2 == 0.0:
        raise RuntimeError("actual callback dtype cannot form a nonzero probe within budget")
    if best_measurement.relative_l2 > PROBE_RELATIVE_L2:
        raise RuntimeError("public probe exceeded its actual-dtype relative-L2 budget")
    return best, best_measurement


def evaluate_public_probes(
    latents: torch.Tensor,
    evaluator: Callable[[str, torch.Tensor], float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Execute exactly one LF and one HF temporary evaluation for each tile."""

    if not callable(evaluator):
        raise TypeError("probe evaluator must be callable")
    responses: dict[str, list[float]] = {"lf": [], "hf": []}
    for tile_index in range(TILE_COUNT):
        for branch in ("lf", "hf"):
            probe, measurement = make_public_tile_probe(latents, branch, tile_index)
            if measurement.perturbation_l2 == 0.0 or measurement.relative_l2 > PROBE_RELATIVE_L2:
                raise RuntimeError("invalid actual-dtype public probe")
            value = evaluator(branch, probe)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise RuntimeError("public probe evaluation must return one finite real scalar")
            responses[branch].append(float(value))
    if len(responses["lf"]) + len(responses["hf"]) != 32:
        raise RuntimeError("content analysis did not execute exactly 32 probe evaluations")
    return tuple(responses["lf"]), tuple(responses["hf"])


def _allocation_vectors(signals: ContentSignals) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    attention = _unit_interval(_finite_vector(signals.semantic_attention, "semantic_attention"))
    texture = _unit_interval(_finite_vector(signals.texture_energy, "texture_energy"))
    lf_probe = _unit_interval(_finite_vector(signals.lf_probe_response, "lf_probe_response"))
    hf_probe = _unit_interval(_finite_vector(signals.hf_probe_response, "hf_probe_response"))
    lf_raw = 0.30 * (1.0 - attention) + 0.30 * texture + 0.30 * lf_probe + 0.10 * (1.0 - hf_probe)
    hf_raw = 0.30 * attention + 0.20 * (1.0 - texture) + 0.30 * hf_probe + 0.20 * (1.0 - lf_probe)
    lf_weights = 0.25 + lf_raw
    hf_weights = 0.25 + hf_raw
    lf_weights /= lf_weights.mean()
    hf_weights /= hf_weights.mean()
    lf_strength = float((texture.mean() + lf_probe.mean() + (1.0 - attention).mean()).item())
    hf_strength = float((attention.mean() + hf_probe.mean() + (1.0 - texture).mean()).item())
    total = lf_strength + hf_strength
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("content signals cannot allocate the two branches")
    return lf_weights, hf_weights, lf_strength / total, hf_strength / total


def allocate_content(signals: ContentSignals) -> ContentAllocation:
    """Allocate both transformed embeddings and verify four real signal effects."""

    lf, hf, lf_share, hf_share = _allocation_vectors(signals)
    fields = (
        "semantic_attention",
        "texture_energy",
        "lf_probe_response",
        "hf_probe_response",
    )
    effects: list[float] = []
    neutral = (0.5,) * TILE_COUNT
    original = torch.cat((lf, hf, torch.tensor([lf_share, hf_share], dtype=torch.float64)))
    for field in fields:
        values = {name: getattr(signals, name) for name in fields}
        values[field] = neutral
        counterfactual = ContentSignals(**values)
        cf_lf, cf_hf, cf_lf_share, cf_hf_share = _allocation_vectors(counterfactual)
        alternate = torch.cat((cf_lf, cf_hf, torch.tensor([cf_lf_share, cf_hf_share])))
        effect = float(torch.linalg.vector_norm(original - alternate).item())
        if not math.isfinite(effect) or effect == 0.0:
            raise RuntimeError(f"{field} has no nonzero neutral-counterfactual allocation effect")
        effects.append(effect)
    return ContentAllocation(
        tuple(float(value) for value in lf.tolist()),
        tuple(float(value) for value in hf.tolist()),
        lf_share,
        hf_share,
        tuple(effects),
    )


def _tile_weight_map(weights: tuple[float, ...], latents: torch.Tensor) -> torch.Tensor:
    vector = _finite_vector(weights, "tile_weights").reshape(1, 1, 4, 4)
    return functional.interpolate(vector, size=latents.shape[-2:], mode="nearest").to(
        device=latents.device, dtype=torch.float64
    )


def _transformed_delta(carrier: torch.Tensor, weights: tuple[float, ...], amplitude: torch.Tensor) -> torch.Tensor:
    transformed = carrier.to(torch.float64) * _tile_weight_map(weights, carrier)
    norm = torch.linalg.vector_norm(transformed)
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("adaptive embedding transform produced an invalid branch")
    return transformed / norm * amplitude


def embed_content_adaptive(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets,
    allocation: ContentAllocation,
) -> tuple[torch.Tensor, ContentAdaptiveMeasurement]:
    """Simultaneously embed transformed LF/HF branches under one 0.012 budget."""

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
        detection_key, tuple(latents.shape), hf_assets, dtype=torch.float32, device=latents.device
    )
    lf_base = reconstruct_lf_carrier(
        detection_key, tuple(latents.shape), lf_assets, dtype=torch.float32, device=latents.device
    )
    # Adaptive/joint IDs intentionally never enter either frozen base PRG domain.
    hf_delta = _transformed_delta(
        hf_base, allocation.hf_tile_weights, base_l2 * COMBINED_RELATIVE_L2 * allocation.hf_branch_share
    )
    lf_delta = _transformed_delta(
        lf_base, allocation.lf_tile_weights, base_l2 * COMBINED_RELATIVE_L2 * allocation.lf_branch_share
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
        32,
    )


def score_content_image(
    image: Any,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets,
) -> ContentBlindScores:
    """Blind joint score from final RGB, key, and frozen public VAE assets only."""

    lf = float(score_lf_image(image, detection_key, lf_assets))
    hf = float(score_hf_image(image, detection_key, hf_assets))
    if not math.isfinite(lf) or not math.isfinite(hf):
        raise ValueError("blind content detector returned a nonfinite base score")
    return ContentBlindScores(lf=lf, hf=hf, content=min(lf, hf))
