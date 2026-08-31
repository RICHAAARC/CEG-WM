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
        self.config = SimpleNamespace(scaling_factor=0.5, shift_factor=0.25)
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
    assert all(item.device == vae.anchor.device for item in vae.inputs)
    assert dark_observation.shape == (1, 4, 8, 8)
    assert not torch.equal(dark_observation, bright_observation)
    mode = torch.cat([vae.inputs[1], vae.inputs[1].mean(dim=1, keepdim=True)], dim=1)
    assert torch.equal(bright_observation, (mode - 0.25) * 0.5)


@pytest.mark.unit
def test_final_image_uses_accelerate_execution_device_when_hook_differs_from_parameter_device() -> None:
    processor = _TrackingProcessor()
    vae = _TrackingVAE()
    vae.anchor = torch.nn.Parameter(torch.empty((), device="meta"), requires_grad=False)
    vae._hf_hook = SimpleNamespace(execution_device="cpu")

    observation = encode_final_rgb_image(Image.new("RGB", (8, 8)), processor, vae)

    assert vae.anchor.device.type == "meta"
    assert vae.inputs[0].device.type == "cpu"
    assert observation.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize("execution_device", ["not-a-device", object()])
def test_invalid_accelerate_execution_device_fails_closed(execution_device: object) -> None:
    vae = _TrackingVAE()
    vae._hf_hook = SimpleNamespace(execution_device=execution_device)

    with pytest.raises(TypeError, match="valid execution_device"):
        encode_final_rgb_image(Image.new("RGB", (8, 8)), _TrackingProcessor(), vae)


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


@pytest.mark.unit
@pytest.mark.parametrize("missing_name", ["scaling_factor", "shift_factor"])
def test_missing_vae_scale_or_shift_fails_closed(missing_name: str) -> None:
    vae = _TrackingVAE()
    delattr(vae.config, missing_name)

    with pytest.raises(ValueError, match=missing_name):
        encode_final_rgb_image(Image.new("RGB", (8, 8)), _TrackingProcessor(), vae)


@pytest.mark.unit
@pytest.mark.parametrize("shift_factor", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_vae_shift_fails_closed(shift_factor: float) -> None:
    vae = _TrackingVAE()
    vae.config.shift_factor = shift_factor

    with pytest.raises(ValueError, match="finite shift_factor"):
        encode_final_rgb_image(Image.new("RGB", (8, 8)), _TrackingProcessor(), vae)


@pytest.mark.unit
@pytest.mark.parametrize("scaling_factor", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_vae_scale_fails_closed(scaling_factor: float) -> None:
    vae = _TrackingVAE()
    vae.config.scaling_factor = scaling_factor

    with pytest.raises(ValueError, match="scaling_factor"):
        encode_final_rgb_image(Image.new("RGB", (8, 8)), _TrackingProcessor(), vae)
