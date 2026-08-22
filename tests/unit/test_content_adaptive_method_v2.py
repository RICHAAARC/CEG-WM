from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from cegwm.method.content_adaptive_v2 import (
    COMBINED_RELATIVE_L2,
    COUNTERFACTUAL_EFFECT_FIELDS,
    PROBE_RELATIVE_L2_SCALES,
    ContentAdaptiveMeasurement,
    ContentAllocation,
    ContentSignals,
    ProbeObservation,
    allocate_content,
    embed_content_adaptive,
    evaluate_public_probes,
    rgb_texture_tiles,
)
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
        return pixels.unsqueeze(0) / 255.0


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pixels[:, :1].repeat(1, 4, 1, 1)))


def _assets() -> tuple[FrozenHFPublicAssets, FrozenLFPublicAssets]:
    vae, processor = _VAE(), _Processor()
    hf = FrozenHFPublicAssets(vae, processor, "fixture")
    lf = FrozenLFPublicAssets(
        vae, processor, "fixture", LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    return hf, lf


def _signals(**updates: tuple[float, ...]) -> ContentSignals:
    base = {
        "semantic_importance": tuple(float(index) for index in range(16)),
        "texture_complexity": tuple(float(index) for index in range(16)),
        "lf_transfer_stability": (0.5,) * 16,
        "hf_transfer_stability": (0.5,) * 16,
        "lf_local_perturbation_sensitivity": (0.5,) * 16,
        "hf_local_perturbation_sensitivity": (0.5,) * 16,
    }
    base.update(updates)
    return ContentSignals(**base)


@pytest.mark.unit
def test_v2_allocation_directions_semantic_magnitude_and_flat_texture_neutral() -> None:
    flat = rgb_texture_tiles(Image.new("RGB", (16, 16), color=(90, 90, 90)))
    assert flat == (0.0,) * 16

    anchors = (0.0, 1.0) + (0.5,) * 14
    neutral = allocate_content(_signals(
        semantic_importance=anchors,
        texture_complexity=(0.5,) * 16,
    ))
    low = allocate_content(_signals(
        semantic_importance=(0.0, 1.0, 0.2) + (0.5,) * 13,
        texture_complexity=(0.5,) * 16,
    ))
    high = allocate_content(_signals(
        semantic_importance=(0.0, 1.0, 0.8) + (0.5,) * 13,
        texture_complexity=(0.5,) * 16,
    ))
    assert low.lf_tile_weights == pytest.approx(high.lf_tile_weights)
    assert low.hf_tile_weights == pytest.approx(high.hf_tile_weights)
    assert low.lf_tile_weights[2] > neutral.lf_tile_weights[2]
    assert low.hf_tile_weights[2] > neutral.hf_tile_weights[2]

    texture_low = allocate_content(_signals(texture_complexity=(0.2,) * 16))
    texture_high = allocate_content(_signals(texture_complexity=(0.8,) * 16))
    # A spatial directional check avoids normalization cancelling a flat map.
    texture_step = allocate_content(_signals(texture_complexity=(0.8,) + (0.2,) * 15))
    assert texture_step.lf_tile_weights[0] > texture_low.lf_tile_weights[0]
    assert texture_step.hf_tile_weights[0] < texture_high.hf_tile_weights[0]


@pytest.mark.unit
def test_v2_stability_only_helps_own_branch_and_sensitivity_cannot_help() -> None:
    baseline = allocate_content(_signals())
    lf_stable = allocate_content(_signals(lf_transfer_stability=(0.8,) + (0.5,) * 15))
    hf_stable = allocate_content(_signals(hf_transfer_stability=(0.8,) + (0.5,) * 15))
    lf_sensitive = allocate_content(_signals(
        lf_local_perturbation_sensitivity=(0.8,) + (0.5,) * 15
    ))
    hf_sensitive = allocate_content(_signals(
        hf_local_perturbation_sensitivity=(0.8,) + (0.5,) * 15
    ))
    assert lf_stable.lf_tile_weights[0] > baseline.lf_tile_weights[0]
    assert lf_stable.hf_tile_weights == pytest.approx(baseline.hf_tile_weights)
    assert hf_stable.hf_tile_weights[0] > baseline.hf_tile_weights[0]
    assert hf_stable.lf_tile_weights == pytest.approx(baseline.lf_tile_weights)
    assert lf_sensitive.lf_tile_weights[0] < baseline.lf_tile_weights[0]
    assert hf_sensitive.hf_tile_weights[0] < baseline.hf_tile_weights[0]


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_v2_64_actual_dtype_probes_are_noncumulative_baseline_differenced_and_invariant(
    dtype: torch.dtype,
) -> None:
    base = torch.linspace(-1.0, 1.0, 4 * 32 * 32, dtype=torch.float32).reshape(1, 4, 32, 32).to(dtype)
    original = base.clone()
    calls: list[tuple[str, torch.Tensor]] = []
    baseline = ProbeObservation(torch.zeros(8), torch.zeros(6))

    def evaluator(branch: str, candidate: torch.Tensor) -> ProbeObservation:
        calls.append((branch, candidate.clone()))
        delta = (candidate - base).to(torch.float64).reshape(-1)
        rgb = torch.stack((delta.sum(), delta.square().sum())).repeat(4)
        vae = torch.stack((delta.abs().sum(), delta.sum())).repeat(3)
        return ProbeObservation(rgb, vae)

    first = evaluate_public_probes(base, baseline, evaluator)
    assert torch.equal(base, original)
    assert len(calls) == 64 and first.evaluation_count == 64
    assert [branch for branch, _ in calls] == [
        branch for _ in range(16) for branch in ("lf", "hf") for _ in PROBE_RELATIVE_L2_SCALES
    ]
    for index, (_, candidate) in enumerate(calls):
        scale = PROBE_RELATIVE_L2_SCALES[index % 2]
        relative = float(
            torch.linalg.vector_norm((candidate - base).to(torch.float64))
            / torch.linalg.vector_norm(base.to(torch.float64))
        )
        assert 0.0 < relative <= scale

    offset = 37.0
    shifted_baseline = ProbeObservation(baseline.rgb + offset, baseline.vae - offset)
    calls.clear()

    def shifted_evaluator(branch: str, candidate: torch.Tensor) -> ProbeObservation:
        value = evaluator(branch, candidate)
        return ProbeObservation(value.rgb + offset, value.vae - offset)

    shifted = evaluate_public_probes(base, shifted_baseline, shifted_evaluator)
    assert shifted.lf_transfer_stability == pytest.approx(first.lf_transfer_stability, abs=1e-10)
    assert shifted.hf_transfer_stability == pytest.approx(first.hf_transfer_stability, abs=1e-10)
    assert shifted.lf_local_perturbation_sensitivity == pytest.approx(
        first.lf_local_perturbation_sensitivity, abs=1e-10
    )
    assert shifted.hf_local_perturbation_sensitivity == pytest.approx(
        first.hf_local_perturbation_sensitivity, abs=1e-10
    )


@pytest.mark.unit
def test_v2_zero_response_modality_stays_zero_without_fallback() -> None:
    base = torch.linspace(-1.0, 1.0, 4 * 16 * 16).reshape(1, 4, 16, 16)
    baseline = ProbeObservation(torch.zeros(4), torch.zeros(4))

    def evaluator(branch: str, candidate: torch.Tensor) -> ProbeObservation:
        del branch
        response = torch.full((4,), float((candidate - base).abs().sum()))
        return ProbeObservation(torch.zeros(4), response)

    maps = evaluate_public_probes(base, baseline, evaluator)
    # Equal modality mean: the zero RGB side contributes exactly zero, not a copied fallback.
    assert all(0.0 <= value <= 0.5 for value in maps.lf_local_perturbation_sensitivity)
    assert all(0.0 <= value <= 0.5 for value in maps.hf_local_perturbation_sensitivity)


@pytest.mark.unit
def test_v2_six_ordered_nonnegative_effects_one_budget_and_no_private_measurement() -> None:
    allocation = allocate_content(_signals())
    assert len(allocation.counterfactual_effects) == 6
    assert min(allocation.counterfactual_effects) >= 0.0
    zero = ContentAllocation(
        (1.0,) * 16, (1.0,) * 16, 0.5, 0.5, (0.0,) * 6,
    )
    assert zero.counterfactual_effects == (0.0,) * 6
    base = torch.linspace(-2.0, 2.0, 4 * 64 * 64).reshape(1, 4, 64, 64)
    hf, lf = _assets()
    candidate, measurement = embed_content_adaptive(base, b"registered-key-01", hf, lf, allocation)
    assert candidate.dtype == base.dtype and candidate.shape == base.shape
    assert 0.0 < measurement.combined_budget.relative_l2 <= COMBINED_RELATIVE_L2
    assert measurement.lf_effective_relative_l2 > 0.0
    assert measurement.hf_effective_relative_l2 > 0.0
    assert tuple(getattr(measurement, name) for name in COUNTERFACTUAL_EFFECT_FIELDS) == (
        allocation.counterfactual_effects
    )
    names = tuple(field.name for field in fields(ContentAdaptiveMeasurement))
    start = names.index(COUNTERFACTUAL_EFFECT_FIELDS[0])
    assert names[start:start + 6] == COUNTERFACTUAL_EFFECT_FIELDS
    assert measurement.probe_evaluation_count == 64
    assert not hasattr(measurement, "tile_weights")
    assert not hasattr(measurement, "latent")
