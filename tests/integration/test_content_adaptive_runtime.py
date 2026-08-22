from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as functional

from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)
from cegwm.runtime import content_adaptive_sd35 as runtime


class _Distribution:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=0.5, shift_factor=0.1)
        self.decode_calls = 0

    def decode(self, latents: torch.Tensor, return_dict: bool) -> SimpleNamespace:
        assert return_dict is True
        self.decode_calls += 1
        sample = functional.interpolate(latents[:, :3], size=(64, 64), mode="nearest")
        return SimpleNamespace(sample=sample)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        mean = pixels.mean(dim=1, keepdim=True)
        return SimpleNamespace(latent_dist=_Distribution(torch.cat((pixels, mean), dim=1)))


class _ImageProcessor:
    def postprocess(self, sample: torch.Tensor, output_type: str) -> list[Image.Image]:
        assert output_type == "pil"
        normalized = torch.sigmoid(sample[0]).permute(1, 2, 0).detach().cpu().numpy()
        return [Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="RGB")]

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0)


class _Dino(torch.nn.Module):
    def __init__(
        self,
        *,
        attentions: bool = True,
        model_id: str = runtime.DINO_ASSET_ID,
    ) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(
            _attn_implementation="eager",
            _name_or_path=model_id,
        )
        self.attentions = attentions

    def forward(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        if not self.attentions:
            return SimpleNamespace(attentions=None)
        attention = torch.ones((1, 3, 17, 17), device=self.anchor.device)
        attention[:, :, 0, 1:] *= torch.arange(1, 17, device=self.anchor.device)
        return SimpleNamespace(attentions=(attention, attention * 1.25))


class _DinoProcessor:
    def __init__(self, model_id: str = runtime.DINO_ASSET_ID) -> None:
        self.name_or_path = model_id

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        return {"pixel_values": torch.ones((1, 3, 16, 16))}


class _Pipeline:
    def __init__(self, assets: runtime.ContentEmbedAssets) -> None:
        self.vae = assets.hf_public_assets.vae
        self.image_processor = assets.hf_public_assets.image_processor

    def __call__(
        self,
        *,
        callback_on_step_end: object,
        callback_on_step_end_tensor_inputs: list[str],
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        assert callback_on_step_end_tensor_inputs == ["latents"]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
        base = torch.stack((xx, yy, xx + yy, xx - yy)).unsqueeze(0)
        early = {"latents": base}
        assert callback_on_step_end(self, 17, None, early) is early
        updated = callback_on_step_end(self, 18, None, early)
        assert not torch.equal(updated["latents"], base)
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[..., 0] = np.arange(64, dtype=np.uint8)[None, :]
        pixels[..., 1] = np.arange(64, dtype=np.uint8)[:, None]
        pixels[..., 2] = 64
        return SimpleNamespace(images=[Image.fromarray(pixels, mode="RGB")])


def _assets(dino: _Dino | None = None) -> runtime.ContentEmbedAssets:
    vae, processor = _VAE(), _ImageProcessor()
    image_processor_id = "stabilityai/stable-diffusion-3.5-medium:image_processor"
    hf = FrozenHFPublicAssets(vae, processor, image_processor_id)
    lf = FrozenLFPublicAssets(
        vae, processor, image_processor_id, LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    return runtime.ContentEmbedAssets(dino or _Dino(), _DinoProcessor(), hf, lf)


@pytest.mark.integration
def test_real_callback_boundary_executes_32_temporary_probes_and_one_combined_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _assets()
    pipeline = _Pipeline(assets)
    calls = 0

    def band_energy(image: Image.Image, branch: str, received: runtime.ContentEmbedAssets) -> float:
        nonlocal calls
        assert image.mode == "RGB" and received is assets
        calls += 1
        return float(calls + (0.125 if branch == "hf" else 0.0))

    monkeypatch.setattr(runtime, "_public_band_energy", band_energy)
    output = runtime.run_sd35_content_adaptive(
        pipeline, "runtime fixture", b"registered-key-01", assets,
        height=512, width=512,
    )
    assert output.image.mode == "RGB"
    assert calls == 32
    assert assets.hf_public_assets.vae.decode_calls == 33
    assert output.measurement.probe_evaluation_count == 32
    assert 0.0 < output.measurement.combined_budget.relative_l2 <= 0.012
    assert output.measurement.lf_effective_relative_l2 > 0.0
    assert output.measurement.hf_effective_relative_l2 > 0.0


@pytest.mark.integration
def test_callback_fails_closed_without_real_dino_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    assets = _assets(_Dino(attentions=False))
    monkeypatch.setattr(runtime, "_public_band_energy", lambda *args: 1.0)
    with pytest.raises(RuntimeError, match="attention"):
        runtime.run_sd35_content_adaptive(
            _Pipeline(assets), "runtime fixture", b"registered-key-01", assets,
            height=512, width=512,
        )


@pytest.mark.integration
def test_content_assets_fail_closed_on_dino_or_frozen_detector_identity_drift() -> None:
    vae, processor = _VAE(), _ImageProcessor()
    image_processor_id = "stabilityai/stable-diffusion-3.5-medium:image_processor"
    hf = FrozenHFPublicAssets(vae, processor, image_processor_id)
    lf = FrozenLFPublicAssets(
        vae, processor, image_processor_id, LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    with pytest.raises(RuntimeError, match="model identity"):
        runtime.ContentEmbedAssets(_Dino(model_id="drifted"), _DinoProcessor(), hf, lf)
    with pytest.raises(RuntimeError, match="processor identity"):
        runtime.ContentEmbedAssets(_Dino(), _DinoProcessor("drifted"), hf, lf)

    drifted_hf = FrozenHFPublicAssets(vae, processor, "unfrozen:image_processor")
    with pytest.raises(ValueError, match="HF frozen carrier"):
        runtime.ContentEmbedAssets(_Dino(), _DinoProcessor(), drifted_hf, lf)
