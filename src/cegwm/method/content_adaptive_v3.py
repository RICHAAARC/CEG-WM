"""Content V3: content-allocated LF amplitude with adaptive spatial HF.

Content analysis and allocation are the substantive frozen V2 implementation.
Content V3 changes only the LF embedding direction: the standard balanced-block
LF carrier is normalized without a spatial tile-weight transform.  LF content
allocation remains real because its branch share controls the LF amplitude.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as functional

from cegwm.method import content_adaptive_v2 as _content_v2
from cegwm.method.content_adaptive_v2 import (
    BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE,
    COMBINED_BUDGET_PROJECTOR_ID,
    COMBINED_RELATIVE_L2,
    COUNTERFACTUAL_EFFECT_FIELDS,
    DINO_ASSET_ID,
    DINO_ATTENTION_IMPLEMENTATION,
    DINO_ATTENTION_LAYER,
    DINO_ATTENTION_STATISTIC,
    HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID,
    PROBE_EVALUATION_COUNT,
    PROBE_MEASUREMENT_ID,
    PROBE_RELATIVE_L2_SCALES,
    RGB8_TEXTURE_COMPLEXITY_MAX,
    SEMANTIC_GATE_GAMMA,
    TILE_COUNT,
    TILE_GRID_SIDE,
    ContentAdaptiveMeasurement,
    ContentAllocation,
    ContentBlindScores,
    ContentSignals,
    ProbeObservation,
    PublicProbeMaps,
    dino_last_layer_cls_patch_tiles,
    evaluate_public_probes,
    rgb_texture_tiles,
    score_content_image,
)
from cegwm.method.hf import FrozenHFPublicAssets, reconstruct_hf_carrier
from cegwm.method.lf import FrozenLFPublicAssets, reconstruct_lf_carrier
from cegwm.shared.numerics import BudgetMeasurement

LF_CONTENT_V3_EMBEDDING_TRANSFORM_ID = (
    "lf_unweighted_balanced_blocks_content_allocated_amplitude_v3"
)
CONTENT_V3_METHOD_ID = "content_v3_unweighted_lf_adaptive_hf_v1"
CONTENT_V3_EVALUATED_CANDIDATE_ID = (
    "content_v3_unweighted_lf_adaptive_hf_semantic_gate_v1"
)
_CONTENT_V3_COUNTERFACTUAL_SIGNAL_FIELDS = (
    "semantic_importance",
    "texture_complexity",
    "lf_two_scale_response_consistency",
    "hf_two_scale_response_consistency",
    "lf_local_perturbation_sensitivity",
    "hf_local_perturbation_sensitivity",
)


def _content_v3_production_control_vector(
    allocation: ContentAllocation,
) -> torch.Tensor:
    """Return only allocation controls consumed by Content V3 production."""

    return torch.tensor(
        (
            *allocation.hf_tile_weights,
            allocation.lf_branch_share,
            allocation.hf_branch_share,
        ),
        dtype=torch.float64,
    )


def allocate_content(signals: ContentSignals) -> ContentAllocation:
    """Allocate with V2 formulas but measure only controls consumed by V3."""

    allocation = _content_v2.allocate_content(signals)
    observed = _content_v3_production_control_vector(allocation)
    effects: list[float] = []
    for field in _CONTENT_V3_COUNTERFACTUAL_SIGNAL_FIELDS:
        neutral_value = 0.0 if field in {
            "semantic_importance",
            "texture_complexity",
        } else 0.5
        counterfactual = _content_v2.allocate_content(
            replace(signals, **{field: (neutral_value,) * TILE_COUNT})
        )
        effect = float(torch.linalg.vector_norm(
            observed - _content_v3_production_control_vector(counterfactual)
        ).item())
        if not math.isfinite(effect) or effect < 0.0:
            raise RuntimeError(f"{field} Content V3 counterfactual effect is invalid")
        effects.append(effect)
    return ContentAllocation(
        allocation.lf_tile_weights,
        allocation.hf_tile_weights,
        allocation.lf_branch_share,
        allocation.hf_branch_share,
        tuple(effects),
    )


def _relative_l2(base: torch.Tensor, candidate: torch.Tensor) -> BudgetMeasurement:
    if (
        base.shape != candidate.shape
        or base.dtype != candidate.dtype
        or base.device != candidate.device
    ):
        raise ValueError("actual callback base and candidate identity must match")
    base64 = base.to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    base_l2 = float(torch.linalg.vector_norm(base64).item())
    if not math.isfinite(base_l2) or base_l2 == 0.0:
        raise ValueError("relative L2 requires a finite nonzero callback latent")
    delta_l2 = float(torch.linalg.vector_norm(candidate64 - base64).item())
    return BudgetMeasurement(str(base.dtype), base_l2, delta_l2, delta_l2 / base_l2)


def _normalized_amplitude(carrier: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    direction = carrier.to(torch.float64)
    norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("Content V3 carrier direction is invalid")
    return direction / norm * amplitude


def _hf_weighted_amplitude(
    carrier: torch.Tensor,
    weights: tuple[float, ...],
    amplitude: torch.Tensor,
) -> torch.Tensor:
    vector = torch.as_tensor(weights, dtype=torch.float64)
    if (
        vector.ndim != 1
        or vector.numel() != TILE_COUNT
        or not bool(torch.isfinite(vector).all())
        or not bool((vector > 0.0).all())
    ):
        raise ValueError("Content V3 HF tile weights must be 16 finite positive scalars")
    weight_map = functional.interpolate(
        vector.reshape(1, 1, TILE_GRID_SIDE, TILE_GRID_SIDE),
        size=carrier.shape[-2:],
        mode="nearest",
    ).to(device=carrier.device, dtype=torch.float64)
    return _normalized_amplitude(carrier.to(torch.float64) * weight_map, amplitude)


def _content_v3_branch_deltas(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets,
    allocation: ContentAllocation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the exact production LF and HF deltas before joint projection."""

    base_l2 = torch.linalg.vector_norm(latents.to(torch.float64))
    if not bool(torch.isfinite(base_l2)) or float(base_l2.item()) == 0.0:
        raise ValueError("Content V3 requires a finite nonzero latent")
    hf_carrier = reconstruct_hf_carrier(
        detection_key,
        tuple(latents.shape),
        hf_assets,
        dtype=torch.float32,
        device=latents.device,
    )
    lf_carrier = reconstruct_lf_carrier(
        detection_key,
        tuple(latents.shape),
        lf_assets,
        dtype=torch.float32,
        device=latents.device,
    )
    hf_delta = _hf_weighted_amplitude(
        hf_carrier,
        allocation.hf_tile_weights,
        base_l2 * COMBINED_RELATIVE_L2 * allocation.hf_branch_share,
    )
    # Content V3 intentionally does not consume allocation.lf_tile_weights here.
    lf_delta = _normalized_amplitude(
        lf_carrier,
        base_l2 * COMBINED_RELATIVE_L2 * allocation.lf_branch_share,
    )
    return lf_delta, hf_delta


def embed_content_v3(
    latents: torch.Tensor,
    detection_key: str | bytes | bytearray | memoryview,
    hf_assets: FrozenHFPublicAssets,
    lf_assets: FrozenLFPublicAssets,
    allocation: ContentAllocation,
) -> tuple[torch.Tensor, ContentAdaptiveMeasurement]:
    """Embed Content V3 LF and HF simultaneously under one actual-dtype budget."""

    if (
        not isinstance(latents, torch.Tensor)
        or latents.ndim != 4
        or not latents.dtype.is_floating_point
    ):
        raise TypeError("Content V3 embedding requires floating NCHW callback latents")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("Content V3 callback latents must be finite")
    if not isinstance(allocation, ContentAllocation):
        raise TypeError("Content V3 embedding requires a real ContentAllocation")
    if hf_assets.injection_step_index != 18 or lf_assets.injection_step_index != 18:
        raise ValueError("both frozen base carriers must use callback step 18")
    base64 = latents.to(torch.float64)
    lf_delta, hf_delta = _content_v3_branch_deltas(
        latents, detection_key, hf_assets, lf_assets, allocation
    )
    if any(
        float(torch.linalg.vector_norm(delta).item()) == 0.0
        for delta in (lf_delta, hf_delta)
    ):
        raise RuntimeError("both Content V3 branches must be nonzero")

    def candidate_at(scale: float) -> torch.Tensor:
        return (base64 + scale * (lf_delta + hf_delta)).to(latents.dtype)

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
    lf_actual = _relative_l2(latents, (base64 + low * lf_delta).to(latents.dtype))
    hf_actual = _relative_l2(latents, (base64 + low * hf_delta).to(latents.dtype))
    if measurement.perturbation_l2 == 0.0 or measurement.relative_l2 > COMBINED_RELATIVE_L2:
        raise RuntimeError("Content V3 actual-dtype embedding is zero or over budget")
    if lf_actual.perturbation_l2 == 0.0 or hf_actual.perturbation_l2 == 0.0:
        raise RuntimeError("both actual-dtype Content V3 branches must remain nonzero")
    return best, ContentAdaptiveMeasurement(
        measurement,
        lf_actual.relative_l2,
        hf_actual.relative_l2,
        allocation.lf_branch_share,
        allocation.hf_branch_share,
        *allocation.counterfactual_effects,
        PROBE_EVALUATION_COUNT,
    )


__all__ = [
    "BRANCH_SHARE_SUM_ABSOLUTE_TOLERANCE",
    "COMBINED_BUDGET_PROJECTOR_ID",
    "COMBINED_RELATIVE_L2",
    "CONTENT_V3_EVALUATED_CANDIDATE_ID",
    "CONTENT_V3_METHOD_ID",
    "COUNTERFACTUAL_EFFECT_FIELDS",
    "DINO_ASSET_ID",
    "DINO_ATTENTION_IMPLEMENTATION",
    "DINO_ATTENTION_LAYER",
    "DINO_ATTENTION_STATISTIC",
    "HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID",
    "LF_CONTENT_V3_EMBEDDING_TRANSFORM_ID",
    "PROBE_EVALUATION_COUNT",
    "PROBE_MEASUREMENT_ID",
    "PROBE_RELATIVE_L2_SCALES",
    "RGB8_TEXTURE_COMPLEXITY_MAX",
    "SEMANTIC_GATE_GAMMA",
    "TILE_COUNT",
    "TILE_GRID_SIDE",
    "ContentAdaptiveMeasurement",
    "ContentAllocation",
    "ContentBlindScores",
    "ContentSignals",
    "ProbeObservation",
    "PublicProbeMaps",
    "allocate_content",
    "dino_last_layer_cls_patch_tiles",
    "embed_content_v3",
    "evaluate_public_probes",
    "rgb_texture_tiles",
    "score_content_image",
]
