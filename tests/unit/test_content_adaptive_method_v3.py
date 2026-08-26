from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from cegwm.method import content_adaptive_v2 as v2
from cegwm.method import content_adaptive_v3 as v3
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
    reconstruct_lf_carrier,
)


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(
            2, 0, 1
        )
        return pixels.unsqueeze(0) / 255.0


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        mode = pixels[:, :1].repeat(1, 4, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mode))


def _assets() -> tuple[FrozenHFPublicAssets, FrozenLFPublicAssets]:
    vae, processor = _VAE(), _Processor()
    hf = FrozenHFPublicAssets(vae, processor, "fixture")
    lf = FrozenLFPublicAssets(
        vae,
        processor,
        "fixture",
        LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    return hf, lf


def _signals(**updates: tuple[float, ...]) -> v3.ContentSignals:
    values = {
        "semantic_importance": tuple(float(index + 1) for index in range(16)),
        "texture_complexity": tuple(float(index * 8) for index in range(16)),
        "lf_two_scale_response_consistency": (0.5,) * 16,
        "hf_two_scale_response_consistency": (0.5,) * 16,
        "lf_local_perturbation_sensitivity": (0.5,) * 16,
        "hf_local_perturbation_sensitivity": (0.5,) * 16,
    }
    values.update(updates)
    return v3.ContentSignals(**values)


def _allocation(
    *,
    lf_weights: tuple[float, ...] = (1.0,) * 16,
    hf_weights: tuple[float, ...] = (1.0,) * 16,
    lf_share: float = 0.4,
) -> v3.ContentAllocation:
    return v3.ContentAllocation(
        lf_weights,
        hf_weights,
        lf_share,
        1.0 - lf_share,
        (0.1,) * 6,
    )


@pytest.mark.unit
def test_content_v3_lf_is_collinear_unweighted_and_real_share_sets_amplitude() -> None:
    base = torch.linspace(-2.0, 2.0, 4 * 64 * 64).reshape(1, 4, 64, 64)
    hf, lf = _assets()
    nonuniform = (0.5,) * 8 + (1.5,) * 8
    low = _allocation(lf_weights=nonuniform, lf_share=0.35)
    high = _allocation(lf_weights=tuple(reversed(nonuniform)), lf_share=0.55)
    low_lf, _ = v3._content_v3_branch_deltas(
        base, b"registered-key-01", hf, lf, low
    )
    high_lf, _ = v3._content_v3_branch_deltas(
        base, b"registered-key-01", hf, lf, high
    )
    carrier = reconstruct_lf_carrier(
        b"registered-key-01",
        tuple(base.shape),
        lf,
        dtype=torch.float32,
        device=base.device,
    ).to(torch.float64)
    normalized = carrier / torch.linalg.vector_norm(carrier)
    for delta in (low_lf, high_lf):
        cosine = torch.dot(delta.reshape(-1), normalized.reshape(-1)) / torch.linalg.vector_norm(delta)
        assert float(cosine) == pytest.approx(1.0, abs=1e-12)
    base_l2 = float(torch.linalg.vector_norm(base.to(torch.float64)))
    assert float(torch.linalg.vector_norm(low_lf)) == pytest.approx(
        base_l2 * 0.012 * low.lf_branch_share, rel=1e-12
    )
    assert float(torch.linalg.vector_norm(high_lf)) == pytest.approx(
        base_l2 * 0.012 * high.lf_branch_share, rel=1e-12
    )
    assert torch.allclose(
        low_lf / torch.linalg.vector_norm(low_lf),
        high_lf / torch.linalg.vector_norm(high_lf),
        atol=1e-15,
        rtol=1e-13,
    )


@pytest.mark.unit
def test_content_v3_real_signals_change_allocation_shares_and_branch_amplitudes() -> None:
    baseline = v3.allocate_content(_signals())
    controlled = (
        v3.allocate_content(_signals(
            semantic_importance=(0.05, 1.0) + (0.0,) * 14,
        )),
        v3.allocate_content(_signals(
            texture_complexity=(v3.RGB8_TEXTURE_COMPLEXITY_MAX,) * 16,
        )),
        v3.allocate_content(_signals(
            lf_two_scale_response_consistency=(1.0,) * 16,
        )),
        v3.allocate_content(_signals(
            hf_local_perturbation_sensitivity=(1.0,) * 16,
        )),
    )
    assert all(
        (item.lf_branch_share, item.hf_branch_share)
        != (baseline.lf_branch_share, baseline.hf_branch_share)
        for item in controlled
    )

    low_hf = v3.allocate_content(_signals(
        semantic_importance=(1.0,) * 16,
        texture_complexity=(0.0,) * 16,
        lf_two_scale_response_consistency=(1.0,) * 16,
        hf_two_scale_response_consistency=(0.0,) * 16,
        lf_local_perturbation_sensitivity=(0.0,) * 16,
        hf_local_perturbation_sensitivity=(1.0,) * 16,
    ))
    high_hf = v3.allocate_content(_signals(
        semantic_importance=(1.0,) * 16,
        texture_complexity=(v3.RGB8_TEXTURE_COMPLEXITY_MAX,) * 16,
        lf_two_scale_response_consistency=(0.0,) * 16,
        hf_two_scale_response_consistency=(1.0,) * 16,
        lf_local_perturbation_sensitivity=(1.0,) * 16,
        hf_local_perturbation_sensitivity=(0.0,) * 16,
    ))
    assert high_hf.hf_branch_share > low_hf.hf_branch_share
    assert high_hf.lf_branch_share < low_hf.lf_branch_share
    assert high_hf.counterfactual_effects != low_hf.counterfactual_effects

    base = torch.linspace(-1.0, 1.0, 4 * 64 * 64).reshape(1, 4, 64, 64)
    hf, lf = _assets()
    low_lf_delta, low_hf_delta = v3._content_v3_branch_deltas(
        base, b"registered-key-01", hf, lf, low_hf
    )
    high_lf_delta, high_hf_delta = v3._content_v3_branch_deltas(
        base, b"registered-key-01", hf, lf, high_hf
    )
    assert torch.linalg.vector_norm(high_hf_delta) > torch.linalg.vector_norm(low_hf_delta)
    assert torch.linalg.vector_norm(high_lf_delta) < torch.linalg.vector_norm(low_lf_delta)


@pytest.mark.unit
def test_content_v3_six_neutral_effects_use_real_allocation_and_only_production_controls() -> None:
    signals = _signals(
        semantic_importance=tuple(float((index * 5) % 17 + 1) for index in range(16)),
        texture_complexity=tuple(float(index * 13 + 7) for index in range(16)),
        lf_two_scale_response_consistency=tuple(0.1 + index * 0.03 for index in range(16)),
        hf_two_scale_response_consistency=tuple(0.8 - index * 0.025 for index in range(16)),
        lf_local_perturbation_sensitivity=tuple(0.2 + index * 0.02 for index in range(16)),
        hf_local_perturbation_sensitivity=tuple(0.7 - index * 0.02 for index in range(16)),
    )
    allocation = v3.allocate_content(signals)
    v2_allocation = v2.allocate_content(signals)
    assert allocation.lf_tile_weights == v2_allocation.lf_tile_weights
    assert allocation.hf_tile_weights == v2_allocation.hf_tile_weights
    assert allocation.lf_branch_share == v2_allocation.lf_branch_share
    assert allocation.hf_branch_share == v2_allocation.hf_branch_share

    observed_v3 = torch.tensor(
        (*allocation.hf_tile_weights, allocation.lf_branch_share, allocation.hf_branch_share),
        dtype=torch.float64,
    )
    expected_v3: list[float] = []
    for signal_field in fields(v3.ContentSignals):
        neutral = (
            0.0
            if signal_field.name in {"semantic_importance", "texture_complexity"}
            else 0.5
        )
        counterfactual = v2.allocate_content(
            replace(signals, **{signal_field.name: (neutral,) * v3.TILE_COUNT})
        )
        counterfactual_v3 = torch.tensor(
            (
                *counterfactual.hf_tile_weights,
                counterfactual.lf_branch_share,
                counterfactual.hf_branch_share,
            ),
            dtype=torch.float64,
        )
        expected_v3.append(float(torch.linalg.vector_norm(observed_v3 - counterfactual_v3)))

    assert allocation.counterfactual_effects == pytest.approx(expected_v3, abs=1e-15)
    assert any(
        v3_effect != pytest.approx(v2_effect, abs=1e-15)
        for v3_effect, v2_effect in zip(
            allocation.counterfactual_effects,
            v2_allocation.counterfactual_effects,
            strict=True,
        )
    )
    neutral = v3.ContentSignals(
        (0.0,) * v3.TILE_COUNT,
        (0.0,) * v3.TILE_COUNT,
        (0.5,) * v3.TILE_COUNT,
        (0.5,) * v3.TILE_COUNT,
        (0.5,) * v3.TILE_COUNT,
        (0.5,) * v3.TILE_COUNT,
    )
    assert v3.allocate_content(neutral).counterfactual_effects == (0.0,) * 6


@pytest.mark.unit
def test_content_v3_hf_nonuniform_weights_change_direction_and_v2_stays_adaptive() -> None:
    base = torch.linspace(-1.0, 1.0, 4 * 64 * 64).reshape(1, 4, 64, 64)
    hf, lf = _assets()
    nonuniform = (0.5,) * 8 + (1.5,) * 8
    uniform_allocation = _allocation()
    weighted_allocation = _allocation(
        lf_weights=nonuniform,
        hf_weights=nonuniform,
    )
    _, uniform_hf = v3._content_v3_branch_deltas(
        base, b"registered-key-01", hf, lf, uniform_allocation
    )
    _, weighted_hf = v3._content_v3_branch_deltas(
        base, b"registered-key-01", hf, lf, weighted_allocation
    )
    assert not torch.allclose(
        uniform_hf / torch.linalg.vector_norm(uniform_hf),
        weighted_hf / torch.linalg.vector_norm(weighted_hf),
    )

    carrier = reconstruct_lf_carrier(
        b"registered-key-01",
        tuple(base.shape),
        lf,
        dtype=torch.float32,
        device=base.device,
    )
    amplitude = torch.tensor(1.0, dtype=torch.float64)
    v2_uniform = v2._lf_transformed_delta(carrier, (1.0,) * 16, amplitude)
    v2_weighted = v2._lf_transformed_delta(carrier, nonuniform, amplitude)
    assert not torch.allclose(v2_uniform, v2_weighted)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_content_v3_joint_actual_dtype_budget_branches_and_step18(dtype: torch.dtype) -> None:
    base = torch.linspace(-2.0, 2.0, 4 * 64 * 64).reshape(1, 4, 64, 64).to(dtype)
    hf, lf = _assets()
    allocation = v3.allocate_content(_signals())
    candidate, measurement = v3.embed_content_v3(
        base, b"registered-key-01", hf, lf, allocation
    )
    assert candidate.dtype == dtype and candidate.shape == base.shape
    actual = float(
        torch.linalg.vector_norm((candidate - base).to(torch.float64))
        / torch.linalg.vector_norm(base.to(torch.float64))
    )
    assert 0.0 < actual <= v3.COMBINED_RELATIVE_L2
    assert measurement.combined_budget.relative_l2 == pytest.approx(actual)
    assert measurement.lf_effective_relative_l2 > 0.0
    assert measurement.hf_effective_relative_l2 > 0.0
    assert measurement.probe_evaluation_count == 64
    assert hf.injection_step_index == lf.injection_step_index == 18
