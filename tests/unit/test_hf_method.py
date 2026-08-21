from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cegwm.method.hf import (
    FrozenHFPublicAssets,
    inject_hf_carrier,
    reconstruct_hf_carrier,
    score_hf_image,
)


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


class _LatentDistribution:
    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.observation = torch.ones((1, 4, 16, 16), dtype=torch.float32)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        del pixels
        return SimpleNamespace(latent_dist=_LatentDistribution(self.observation))


def _assets(vae: _VAE | None = None) -> FrozenHFPublicAssets:
    return FrozenHFPublicAssets(
        vae=vae or _VAE(),
        image_processor=_Processor(),
        model_revision="a" * 40,
        vae_weight_digest="b" * 64,
        image_processor_id="sd35-vae-image-processor-v1",
    )


@pytest.mark.unit
def test_production_carrier_is_repeatable_and_key_dependent() -> None:
    assets = _assets()
    first_key = b"0123456789abcdef0123456789abcdef"
    second_key = b"abcdef0123456789abcdef0123456789"

    first = reconstruct_hf_carrier(
        first_key,
        (1, 4, 16, 16),
        assets,
        dtype=torch.float32,
        device="cpu",
    )
    repeated = reconstruct_hf_carrier(
        first_key,
        (1, 4, 16, 16),
        assets,
        dtype=torch.float32,
        device="cpu",
    )
    wrong = reconstruct_hf_carrier(
        second_key,
        (1, 4, 16, 16),
        assets,
        dtype=torch.float32,
        device="cpu",
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, wrong)
    assert first.shape == (1, 4, 16, 16)
    assert torch.linalg.vector_norm(first.float()).item() == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_injection_modifies_actual_dtype_within_total_budget(dtype: torch.dtype) -> None:
    latents = torch.linspace(-1.0, 1.0, 4 * 16 * 16, dtype=dtype).reshape(1, 4, 16, 16)

    injected, measurement = inject_hf_carrier(
        latents,
        b"0123456789abcdef0123456789abcdef",
        _assets(),
    )

    assert injected.dtype is dtype
    assert injected.device == latents.device
    assert not torch.equal(injected, latents)
    assert 0.0 < measurement.relative_l2 <= 0.012
    actual = torch.linalg.vector_norm(injected.double() - latents.double())
    base = torch.linalg.vector_norm(latents.double())
    assert measurement.relative_l2 == pytest.approx((actual / base).item())


@pytest.mark.unit
def test_blind_scorer_signature_rejects_private_embedding_inputs() -> None:
    assert tuple(inspect.signature(score_hf_image).parameters) == (
        "image",
        "detection_key",
        "frozen_public_assets",
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        score_hf_image(
            Image.new("RGB", (16, 16)),
            b"0123456789abcdef0123456789abcdef",
            _assets(),
            embedding_latent=torch.zeros((1, 4, 16, 16)),
        )


@pytest.mark.unit
def test_blind_score_rebuilds_keyed_carrier_from_final_image_observation() -> None:
    vae = _VAE()
    assets = _assets(vae)
    correct_key = b"0123456789abcdef0123456789abcdef"
    wrong_key = b"abcdef0123456789abcdef0123456789"
    vae.observation = reconstruct_hf_carrier(
        correct_key,
        (1, 4, 16, 16),
        assets,
        dtype=torch.float32,
        device="cpu",
    )
    image = Image.new("RGB", (16, 16), color=(20, 40, 60))

    correct = score_hf_image(image, correct_key, assets)
    wrong = score_hf_image(image, wrong_key, assets)

    assert correct == pytest.approx(1.0, abs=1e-6)
    assert correct > wrong
