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
    RGB8_TEXTURE_COMPLEXITY_MAX,
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
        "lf_two_scale_response_consistency": (0.5,) * 16,
        "hf_two_scale_response_consistency": (0.5,) * 16,
        "lf_local_perturbation_sensitivity": (0.5,) * 16,
        "hf_local_perturbation_sensitivity": (0.5,) * 16,
    }
    base.update(updates)
    return ContentSignals(**base)


@pytest.mark.unit
def test_v2_direct_semantic_gate_and_rgb8_texture_coordinate() -> None:
    flat = rgb_texture_tiles(Image.new("RGB", (16, 16), color=(90, 90, 90)))
    assert flat == (0.0,) * 16
    alternating = np.zeros((16, 16, 3), dtype=np.uint8)
    alternating[::2, ::2] = (255, 0, 0)
    alternating[1::2, 1::2] = (255, 0, 0)
    alternating[::2, 1::2] = (0, 255, 0)
    alternating[1::2, ::2] = (0, 255, 0)
    raw_texture = rgb_texture_tiles(Image.fromarray(alternating, mode="RGB"))
    assert all(0.0 < value <= RGB8_TEXTURE_COMPLEXITY_MAX for value in raw_texture)

    semantic_zero = allocate_content(_signals(semantic_importance=(0.0,) * 16))
    assert semantic_zero.lf_branch_share == 0.5
    assert semantic_zero.hf_branch_share == 0.5
    assert semantic_zero.lf_tile_weights == pytest.approx((1.0,) * 16)
    assert semantic_zero.hf_tile_weights == pytest.approx((1.0,) * 16)

    # Equal LF/HF suitability remains exact balance for every frozen gate value.
    for gate in (0.0, 0.25, 0.5, 1.0):
        balanced = allocate_content(_signals(
            semantic_importance=(gate, 1.0) + (0.0,) * 14,
            texture_complexity=(0.0,) * 16,
            lf_two_scale_response_consistency=(0.5,) * 16,
            hf_two_scale_response_consistency=(0.5,) * 16,
        ))
        assert balanced.lf_branch_share == pytest.approx(0.5)
        assert balanced.hf_branch_share == pytest.approx(0.5)

    texture_min = allocate_content(_signals(
        semantic_importance=(1.0,) * 16,
        texture_complexity=(0.0,) * 16,
    ))
    texture_max = allocate_content(
        _signals(
            semantic_importance=(1.0,) * 16,
            texture_complexity=(RGB8_TEXTURE_COMPLEXITY_MAX,) * 16,
        )
    )
    texture_step = allocate_content(_signals(
        semantic_importance=(1.0,) * 16,
        texture_complexity=(RGB8_TEXTURE_COMPLEXITY_MAX,) + (0.0,) * 15,
    ))
    assert texture_step.lf_tile_weights[0] < texture_min.lf_tile_weights[0]
    assert texture_step.hf_tile_weights[0] > texture_min.hf_tile_weights[0]
    assert texture_step.lf_branch_share < texture_min.lf_branch_share
    assert texture_step.hf_branch_share > texture_min.hf_branch_share
    assert texture_max.lf_branch_share < texture_min.lf_branch_share
    assert texture_max.hf_branch_share > texture_min.hf_branch_share
    assert texture_min.lf_branch_share == pytest.approx(0.5)
    assert texture_min.hf_branch_share == pytest.approx(0.5)

    for invalid in (-1.0, RGB8_TEXTURE_COMPLEXITY_MAX + 1.0, float("nan")):
        with pytest.raises(ValueError, match="texture_complexity"):
            allocate_content(_signals(texture_complexity=(invalid,) + (0.0,) * 15))


@pytest.mark.unit
def test_v2_counterfactual_neutrals_are_semantic_zero_texture_raw_zero_and_other_half() -> None:
    neutral = ContentSignals(
        semantic_importance=(0.0,) * 16,
        texture_complexity=(0.0,) * 16,
        lf_two_scale_response_consistency=(0.5,) * 16,
        hf_two_scale_response_consistency=(0.5,) * 16,
        lf_local_perturbation_sensitivity=(0.5,) * 16,
        hf_local_perturbation_sensitivity=(0.5,) * 16,
    )
    allocation = allocate_content(neutral)
    assert allocation.lf_branch_share == pytest.approx(0.5)
    assert allocation.hf_branch_share == pytest.approx(0.5)
    assert allocation.counterfactual_effects == pytest.approx((0.0,) * 6)


@pytest.mark.unit
def test_v2_response_consistency_only_helps_own_branch_and_sensitivity_cannot_help() -> None:
    baseline = allocate_content(_signals())
    lf_stable = allocate_content(_signals(
        lf_two_scale_response_consistency=(0.5,) * 15 + (0.8,)
    ))
    hf_stable = allocate_content(_signals(
        hf_two_scale_response_consistency=(0.5,) * 15 + (0.8,)
    ))
    lf_sensitive = allocate_content(_signals(
        lf_local_perturbation_sensitivity=(0.5,) * 15 + (0.8,)
    ))
    hf_sensitive = allocate_content(_signals(
        hf_local_perturbation_sensitivity=(0.5,) * 15 + (0.8,)
    ))
    assert lf_stable.lf_tile_weights[-1] > baseline.lf_tile_weights[-1]
    assert lf_stable.lf_branch_share > baseline.lf_branch_share
    assert lf_stable.hf_branch_share < baseline.hf_branch_share
    assert hf_stable.hf_tile_weights[-1] > baseline.hf_tile_weights[-1]
    assert hf_stable.hf_branch_share > baseline.hf_branch_share
    assert hf_stable.lf_branch_share < baseline.lf_branch_share
    assert lf_sensitive.lf_tile_weights[-1] < baseline.lf_tile_weights[-1]
    assert lf_sensitive.lf_branch_share < baseline.lf_branch_share
    assert hf_sensitive.hf_tile_weights[-1] < baseline.hf_tile_weights[-1]
    assert hf_sensitive.hf_branch_share < baseline.hf_branch_share


@pytest.mark.unit
def test_v2_direct_gate_prevents_old_global_ratio_reversal() -> None:
    # In the former mean-normalized rule, a locally HF-favouring tile can reduce
    # global HF share when its Q_H/Q_L ratio is below the existing global ratio.
    old_q_h = torch.tensor([0.61] + [1.0] * 15, dtype=torch.float64)
    old_q_l = torch.tensor([0.60] + [0.20] * 15, dtype=torch.float64)
    old_g_low = torch.tensor([0.25] + [1.0] * 15, dtype=torch.float64)
    old_g_high = torch.tensor([0.75] + [1.0] * 15, dtype=torch.float64)

    def old_share(gate: torch.Tensor) -> float:
        mean_h = float((0.25 + gate * old_q_h).mean())
        mean_l = float((0.25 + gate * old_q_l).mean())
        return mean_h / (mean_h + mean_l)

    assert old_q_h[0] > old_q_l[0]
    assert old_share(old_g_high) < old_share(old_g_low)

    # The direct gate has d=0.30 at tile 0 and an unchanged max-attention anchor
    # at tile 1, so normalization leaves Delta(g_0)=0.50 exactly.
    common = {
        "texture_complexity": (RGB8_TEXTURE_COMPLEXITY_MAX,) + (0.0,) * 15,
        "lf_two_scale_response_consistency": (0.5,) * 16,
        "hf_two_scale_response_consistency": (0.5,) * 16,
        "lf_local_perturbation_sensitivity": (0.5,) * 16,
        "hf_local_perturbation_sensitivity": (0.5,) * 16,
    }
    low = allocate_content(ContentSignals(
        semantic_importance=(0.25, 1.0) + (0.0,) * 14,
        **common,
    ))
    high = allocate_content(ContentSignals(
        semantic_importance=(0.75, 1.0) + (0.0,) * 14,
        **common,
    ))
    expected = 0.25 * (0.75 - 0.25) * 0.30 / 16.0
    assert high.hf_branch_share - low.hf_branch_share == pytest.approx(expected, abs=1e-15)
    assert high.hf_branch_share > low.hf_branch_share
    assert high.lf_branch_share < low.lf_branch_share


@pytest.mark.unit
def test_v2_direct_gate_bounds_and_lf_hf_semantic_dominance() -> None:
    gate_low = (0.25, 1.0) + (0.0,) * 14
    gate_high = (0.75, 1.0) + (0.0,) * 14
    hf_low = allocate_content(_signals(
        semantic_importance=gate_low,
        texture_complexity=(RGB8_TEXTURE_COMPLEXITY_MAX,) + (0.0,) * 15,
    ))
    hf_high = allocate_content(_signals(
        semantic_importance=gate_high,
        texture_complexity=(RGB8_TEXTURE_COMPLEXITY_MAX,) + (0.0,) * 15,
    ))
    assert hf_high.hf_branch_share > hf_low.hf_branch_share

    lf_low = allocate_content(_signals(
        semantic_importance=gate_low,
        lf_two_scale_response_consistency=(1.0,) + (0.5,) * 15,
        hf_two_scale_response_consistency=(0.0,) + (0.5,) * 15,
    ))
    lf_high = allocate_content(_signals(
        semantic_importance=gate_high,
        lf_two_scale_response_consistency=(1.0,) + (0.5,) * 15,
        hf_two_scale_response_consistency=(0.0,) + (0.5,) * 15,
    ))
    assert lf_high.lf_branch_share > lf_low.lf_branch_share
    for allocation in (hf_low, hf_high, lf_low, lf_high):
        assert 0.0 < allocation.lf_branch_share < 1.0
        assert 0.0 < allocation.hf_branch_share < 1.0
        assert np.mean(allocation.lf_tile_weights) == pytest.approx(1.0, abs=1e-12)
        assert np.mean(allocation.hf_tile_weights) == pytest.approx(1.0, abs=1e-12)

    maximum_hf = allocate_content(ContentSignals(
        semantic_importance=(1.0,) * 16,
        texture_complexity=(RGB8_TEXTURE_COMPLEXITY_MAX,) * 16,
        lf_two_scale_response_consistency=(0.0,) * 16,
        hf_two_scale_response_consistency=(1.0,) * 16,
        lf_local_perturbation_sensitivity=(1.0,) * 16,
        hf_local_perturbation_sensitivity=(0.0,) * 16,
    ))
    maximum_lf = allocate_content(ContentSignals(
        semantic_importance=(1.0,) * 16,
        texture_complexity=(0.0,) * 16,
        lf_two_scale_response_consistency=(1.0,) * 16,
        hf_two_scale_response_consistency=(0.0,) * 16,
        lf_local_perturbation_sensitivity=(0.0,) * 16,
        hf_local_perturbation_sensitivity=(1.0,) * 16,
    ))
    assert maximum_hf.hf_branch_share == pytest.approx(0.70)
    assert maximum_hf.lf_branch_share == pytest.approx(0.30)
    assert maximum_lf.hf_branch_share == pytest.approx(0.375)
    assert maximum_lf.lf_branch_share == pytest.approx(0.625)
    for invalid in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="semantic_importance"):
            allocate_content(_signals(semantic_importance=(invalid,) + (1.0,) * 15))
    with pytest.raises(ValueError, match="unit mean"):
        ContentAllocation((2.0,) * 16, (1.0,) * 16, 0.5, 0.5, (0.0,) * 6)


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
    assert shifted.lf_two_scale_response_consistency == pytest.approx(
        first.lf_two_scale_response_consistency, abs=1e-10
    )
    assert shifted.hf_two_scale_response_consistency == pytest.approx(
        first.hf_two_scale_response_consistency, abs=1e-10
    )
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
