from __future__ import annotations

import math

import pytest
import torch

from cegwm.geometry_v2.contracts import GeometryEstimate
from cegwm.geometry_v2.neural_sync import (
    BlindCornerExtractor,
    IMAGE_SIZE,
    KeyedResidualEmbedder,
    MAX_RESIDUAL,
    n0_joint_loss,
)


@pytest.mark.unit
def test_real_modules_take_one_optimizer_step_and_keep_budget() -> None:
    torch.manual_seed(73)
    embedder = KeyedResidualEmbedder()
    extractor = BlindCornerExtractor()
    optimizer = torch.optim.Adam((*embedder.parameters(), *extractor.parameters()), lr=1.0e-3)
    rgb = torch.rand((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    code = torch.where(torch.arange(128).reshape(2, 64) % 2 == 0, 1.0, -1.0)
    truth = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]).repeat(2, 1, 1)

    before = tuple(parameter.detach().clone() for parameter in extractor.parameters())
    embedded = embedder(rgb, code)
    prediction = extractor(embedded.image)
    loss, components = n0_joint_loss(prediction, truth, embedded, code)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    assert torch.isfinite(loss)
    assert set(components) == {"corner", "sync_reconstruction", "residual_l2"}
    assert embedded.image.shape == rgb.shape
    assert embedded.image.min() >= 0 and embedded.image.max() <= 1
    assert embedded.residual.abs().max() <= MAX_RESIDUAL + 1.0e-7
    assert any(not torch.equal(old, new) for old, new in zip(before, extractor.parameters(), strict=True))


@pytest.mark.unit
def test_blind_extractor_outputs_ordered_slot_shape_and_finite_public_measurements() -> None:
    model = BlindCornerExtractor()
    attacked_rgb = torch.rand((3, 3, IMAGE_SIZE, IMAGE_SIZE))

    prediction = model(attacked_rgb)

    assert prediction.corners.shape == (3, 4, 2)
    assert prediction.confidence.shape == prediction.support.shape == (3,)
    assert torch.isfinite(prediction.corners).all()
    assert torch.all((prediction.corners >= -0.25) & (prediction.corners <= 1.25))
    assert torch.all((prediction.confidence >= 0) & (prediction.confidence <= 1))
    assert torch.equal(prediction.support, torch.ones(3))


@pytest.mark.unit
def test_real_extractor_domain_represents_frozen_crop_truth_and_keeps_candidate_gate_reachable() -> None:
    model = BlindCornerExtractor()
    crop_truth = torch.tensor(
        [[-8 / 111, -11 / 106], [120 / 111, -11 / 106],
         [120 / 111, 117 / 106], [-8 / 111, 117 / 106]],
        dtype=torch.float32,
    )
    probabilities = (crop_truth.flatten() + 0.25) / 1.5
    logits = torch.log(probabilities / (1.0 - probabilities))
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[:8].copy_(logits)

    prediction = model(torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE)))

    assert torch.allclose(prediction.corners[0], crop_truth, atol=1.0e-6, rtol=0.0)
    assert prediction.corners.min() < 0.0 and prediction.corners.max() > 1.0
    crop_h = (
        (128 / 111, 0.0, -8 / 111),
        (0.0, 128 / 106, -11 / 106),
        (0.0, 0.0, 1.0),
    )
    estimate = GeometryEstimate(tuple(map(tuple, prediction.corners[0].tolist())), crop_h)
    assert torch.allclose(torch.tensor(estimate.corners), crop_truth, atol=1.0e-6, rtol=0.0)
    errors = [float(torch.linalg.vector_norm(prediction.corners[0] - crop_truth, dim=1).mean())] * 128
    reliable_fraction = sum(max(0.0, min(1.0, 1.0 - error / 0.25)) >= 0.5 for error in errors) / 128
    assert math.isclose(torch.tensor(errors).median().item(), 0.0, abs_tol=1.0e-7)
    assert torch.quantile(torch.tensor(errors), 0.95).item() < 0.10
    assert reliable_fraction >= 0.75


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rgb", "code"),
    [
        (torch.rand((1, 1, 128, 128)), torch.ones((1, 64))),
        (torch.rand((1, 3, 64, 64)), torch.ones((1, 64))),
        (torch.rand((1, 3, 128, 128)), torch.zeros((1, 64))),
    ],
)
def test_embedder_fails_closed_on_wrong_surface(rgb: torch.Tensor, code: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        KeyedResidualEmbedder()(rgb, code)
