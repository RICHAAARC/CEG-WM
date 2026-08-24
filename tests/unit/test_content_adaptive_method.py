from __future__ import annotations

from dataclasses import fields
import inspect
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from cegwm.method.content_adaptive import (
    COUNTERFACTUAL_EFFECT_FIELDS,
    COMBINED_RELATIVE_L2,
    ContentAdaptiveMeasurement,
    ContentAllocation,
    ContentSignals,
    allocate_content,
    dino_last_layer_cls_patch_tiles,
    embed_content_adaptive,
    evaluate_public_probes,
    make_public_tile_probe,
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


class _Dino(torch.nn.Module):
    def __init__(self, *, eager: bool = True, attentions: bool = True) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(_attn_implementation="eager" if eager else "sdpa")
        self.return_attentions = attentions

    def forward(self, **kwargs: object) -> SimpleNamespace:
        assert kwargs["output_attentions"] is True
        if not self.return_attentions:
            return SimpleNamespace(attentions=None)
        tensor = torch.zeros((1, 2, 17, 17))
        tensor[0, 0, 0, 1:] = torch.arange(1, 17)
        tensor[0, 1, 0, 1:] = torch.arange(17, 33)
        return SimpleNamespace(attentions=(tensor * 0.5, tensor))


def _dino_processor(**kwargs: object) -> dict[str, torch.Tensor]:
    assert kwargs["return_tensors"] == "pt"
    return {"pixel_values": torch.ones((1, 3, 8, 8))}


@pytest.mark.unit
def test_real_dino_last_attention_and_texture_form_exact_4x4_signals() -> None:
    pixels = np.indices((16, 16)).sum(axis=0).astype(np.uint8)
    image = Image.fromarray(np.stack((pixels, pixels * 2, pixels * 3), axis=-1), mode="RGB")
    attention = dino_last_layer_cls_patch_tiles(image, _dino_processor, _Dino())
    texture = rgb_texture_tiles(image)
    assert len(attention) == len(texture) == 16
    assert attention[0] == pytest.approx(9.0)
    assert attention[-1] == pytest.approx(24.0)
    assert len(set(texture)) > 1


@pytest.mark.unit
@pytest.mark.parametrize("model", [_Dino(eager=False), _Dino(attentions=False)])
def test_dino_missing_real_eager_attention_fails_closed(model: _Dino) -> None:
    with pytest.raises(RuntimeError, match="eager|attention"):
        dino_last_layer_cls_patch_tiles(Image.new("RGB", (8, 8)), _dino_processor, model)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_32_public_probes_are_nonzero_bounded_and_non_cumulative(dtype: torch.dtype) -> None:
    base = torch.linspace(-1.0, 1.0, 4 * 32 * 32, dtype=torch.float32).reshape(1, 4, 32, 32).to(dtype)
    original = base.clone()
    calls: list[tuple[str, torch.Tensor]] = []

    def evaluator(branch: str, candidate: torch.Tensor) -> float:
        calls.append((branch, candidate.clone()))
        return float(candidate.to(torch.float64).square().mean().item()) + len(calls) * 1e-8

    lf, hf = evaluate_public_probes(base, evaluator)
    assert torch.equal(base, original)
    assert len(calls) == 32 and len(lf) == len(hf) == 16
    assert [branch for branch, _ in calls] == [value for _ in range(16) for value in ("lf", "hf")]
    for _, candidate in calls:
        delta = torch.linalg.vector_norm((candidate - base).to(torch.float64))
        relative = float(delta / torch.linalg.vector_norm(base.to(torch.float64)))
        assert 0.0 < relative <= 0.001


@pytest.mark.unit
def test_adaptive_allocation_uses_four_nonzero_counterfactual_effects_and_one_budget() -> None:
    values = tuple(float(index + 1) for index in range(16))
    signals = ContentSignals(
        values,
        tuple(reversed(values)),
        tuple(value * value for value in values),
        tuple((index % 5) + index / 20.0 for index in range(16)),
    )
    allocation = allocate_content(signals)
    assert min(allocation.counterfactual_effects) > 0.0
    assert allocation.lf_branch_share > 0.0 and allocation.hf_branch_share > 0.0
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
    assert measurement.minimum_counterfactual_effect == min(allocation.counterfactual_effects)
    measurement_field_names = tuple(field.name for field in fields(ContentAdaptiveMeasurement))
    effect_start = measurement_field_names.index(COUNTERFACTUAL_EFFECT_FIELDS[0])
    assert measurement_field_names[effect_start:effect_start + 4] == COUNTERFACTUAL_EFFECT_FIELDS
    assert "minimum_counterfactual_effect" not in inspect.signature(ContentAdaptiveMeasurement).parameters
    assert measurement.probe_evaluation_count == 32
    assert not hasattr(measurement, "mask") and not hasattr(measurement, "latents")


@pytest.mark.unit
@pytest.mark.parametrize("bad_effect", [0.0, -0.1, float("nan"), float("inf")])
def test_allocation_rejects_each_nonpositive_or_nonfinite_counterfactual_effect(
    bad_effect: float,
) -> None:
    weights = (1.0,) * 16
    for index in range(4):
        effects = [0.1, 0.2, 0.3, 0.4]
        effects[index] = bad_effect
        with pytest.raises(ValueError, match="finite and strictly positive"):
            ContentAllocation(weights, weights, 0.4, 0.6, tuple(effects))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lf_share", "hf_share"),
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (-0.1, 1.1),
        (float("nan"), 0.5),
        (0.4, float("inf")),
        (0.4, 0.59),
    ],
)
def test_allocation_rejects_invalid_public_branch_shares(
    lf_share: float,
    hf_share: float,
) -> None:
    with pytest.raises((TypeError, ValueError), match="branch_share|branch shares"):
        ContentAllocation((1.0,) * 16, (1.0,) * 16, lf_share, hf_share, (0.1, 0.2, 0.3, 0.4))


@pytest.mark.unit
def test_probe_fails_when_actual_dtype_cannot_represent_any_change() -> None:
    base = torch.full((1, 1, 32, 32), torch.finfo(torch.float16).smallest_normal / 1024, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="cannot form a nonzero probe"):
        make_public_tile_probe(base, "lf", 0)
