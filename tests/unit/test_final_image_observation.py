from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image


class _TrackingProcessor:
    def __init__(self) -> None:
        self.images: list[Image.Image] = []

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        self.images.append(image.copy())
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


class _Distribution:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class _TrackingVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=0.5)
        self.inputs: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        self.inputs.append(pixels.detach().clone())
        mean = pixels.mean(dim=1, keepdim=True)
        observation = torch.cat([pixels, mean], dim=1)
        return SimpleNamespace(latent_dist=_Distribution(observation))


@pytest.mark.unit
def test_ordinary_rgb_boundary_rejects_latents_and_non_rgb_images() -> None:
    rgb = Image.new("RGB", (8, 8), color=(1, 2, 3))
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)

    assert require_ordinary_rgb_image(rgb).mode == "RGB"
    assert require_ordinary_rgb_image(pixels).mode == "RGB"
    with pytest.raises(TypeError, match="ordinary RGB"):
        require_ordinary_rgb_image(torch.zeros((1, 4, 8, 8)))
    with pytest.raises(ValueError, match="already be RGB"):
        require_ordinary_rgb_image(Image.new("RGBA", (8, 8)))


@pytest.mark.unit
def test_final_image_is_processed_and_vae_reencoded_with_input_dependence() -> None:
    processor = _TrackingProcessor()
    vae = _TrackingVAE()
    dark = Image.new("RGB", (8, 8), color=(0, 0, 0))
    bright = Image.new("RGB", (8, 8), color=(255, 128, 64))

    dark_observation = encode_final_rgb_image(dark, processor, vae)
    bright_observation = encode_final_rgb_image(bright, processor, vae)

    assert len(processor.images) == 2
    assert len(vae.inputs) == 2
    assert dark_observation.shape == (1, 4, 8, 8)
    assert not torch.equal(dark_observation, bright_observation)
    assert torch.equal(bright_observation, torch.cat(
        [vae.inputs[1], vae.inputs[1].mean(dim=1, keepdim=True)], dim=1
    ) * 0.5)


@pytest.mark.unit
def test_incompatible_vae_result_fails_without_observation_fallback() -> None:
    class IncompatibleVAE(_TrackingVAE):
        def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
            del pixels
            return SimpleNamespace(latents=torch.ones((1, 4, 8, 8)))

    with pytest.raises(TypeError, match="latent_dist.mode"):
        encode_final_rgb_image(
            Image.new("RGB", (8, 8)),
            _TrackingProcessor(),
            IncompatibleVAE(),
        )
