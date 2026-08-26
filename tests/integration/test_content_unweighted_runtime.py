from __future__ import annotations

# Functional coverage for the content-unweighted runtime.

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as functional
from transformers.image_utils import SizeDict

from cegwm.method import content_unweighted as method_unweighted
from cegwm.method.content_unweighted import (
    ContentAdaptiveMeasurement,
    PublicProbeMaps,
)
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)
from cegwm.runtime import content_unweighted_sd35 as runtime
from cegwm.shared.numerics import BudgetMeasurement


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
        pixels = torch.sigmoid(sample[0]).permute(1, 2, 0).detach().cpu().numpy()
        return [Image.fromarray(np.rint(pixels * 255.0).astype(np.uint8), mode="RGB")]

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0)


class _Dino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(
            _attn_implementation="eager",
            _name_or_path=runtime.DINO_ASSET_ID,
        )

    def forward(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        attention = torch.ones((1, 2, 17, 17), device=self.anchor.device)
        attention[:, :, 0, 1:] *= torch.arange(1, 17, device=self.anchor.device)
        return SimpleNamespace(attentions=(attention, attention * 1.25))


class BitImageProcessor:
    def __init__(self, *, official_size: bool = False) -> None:
        self.do_resize = True
        self.size = SizeDict(shortest_edge=256) if official_size else {"shortest_edge": 256}
        self.resample = 3
        self.do_center_crop = True
        self.crop_size = (
            SizeDict(height=224, width=224)
            if official_size
            else {"height": 224, "width": 224}
        )
        self.do_convert_rgb = True
        self.do_rescale = True
        self.rescale_factor = 1.0 / 255.0
        self.do_normalize = True
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        return {"pixel_values": torch.ones((1, 3, 16, 16))}


class _Pipeline:
    def __init__(self, assets: runtime.ContentUnweightedEmbedAssets) -> None:
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
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij"
        )
        base = torch.stack((xx, yy, xx + yy, xx - yy)).unsqueeze(0)
        state = {"latents": base}
        assert callback_on_step_end(self, 17, None, state) is state
        updated = callback_on_step_end(self, 18, None, state)
        assert not torch.equal(updated["latents"], base)
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[..., 0] = np.arange(64, dtype=np.uint8)[None, :]
        pixels[..., 1] = np.arange(64, dtype=np.uint8)[:, None]
        pixels[..., 2] = 64
        return SimpleNamespace(images=[Image.fromarray(pixels, mode="RGB")])


def _assets(*, official_size: bool = False) -> runtime.ContentUnweightedEmbedAssets:
    vae, processor = _VAE(), _ImageProcessor()
    processor_id = "stabilityai/stable-diffusion-3.5-medium:image_processor"
    hf = FrozenHFPublicAssets(vae, processor, processor_id)
    lf = FrozenLFPublicAssets(
        vae,
        processor,
        processor_id,
        LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    return runtime.ContentUnweightedEmbedAssets(
        _Dino(), BitImageProcessor(official_size=official_size), hf, lf
    )


@pytest.mark.integration
def test_content_unweighted_runtime_executes_step18_baseline_64_probes_and_real_embedding() -> None:
    assets = _assets()
    output = runtime.run_sd35_content_unweighted(
        _Pipeline(assets),
        "content-unweighted runtime fixture",
        b"registered-key-01",
        assets,
        height=512,
        width=512,
    )
    assert output.image.mode == "RGB"
    assert assets.hf_public_assets.vae.decode_calls == 65
    assert output.measurement.probe_evaluation_count == 64
    assert 0.0 < output.measurement.combined_budget.relative_l2 <= 0.012
    assert output.measurement.lf_effective_relative_l2 > 0.0
    assert output.measurement.hf_effective_relative_l2 > 0.0
    assert assets.content_method_id == "content_v3_unweighted_lf_adaptive_hf_v1"
    assert assets.runtime_asset_validation_contract_id == (
        "dinov2_small_eager_bit_image_processor_public_size_semantics_v3"
    )


@pytest.mark.integration
def test_content_unweighted_runtime_passes_real_signal_allocation_to_unweighted_embed_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _assets()
    semantic = tuple(float((index * 3) % 11 + 1) for index in range(16))
    texture = tuple(float(index * 9 + 2) for index in range(16))
    maps = PublicProbeMaps(
        tuple(0.1 + index * 0.02 for index in range(16)),
        tuple(0.8 - index * 0.02 for index in range(16)),
        tuple(0.2 + index * 0.015 for index in range(16)),
        tuple(0.7 - index * 0.015 for index in range(16)),
    )
    expected = method_unweighted.allocate_content(method_unweighted.ContentSignals(
        semantic,
        texture,
        maps.lf_two_scale_response_consistency,
        maps.hf_two_scale_response_consistency,
        maps.lf_local_perturbation_sensitivity,
        maps.hf_local_perturbation_sensitivity,
    ))
    monkeypatch.setattr(runtime, "dino_last_layer_cls_patch_tiles", lambda *args: semantic)
    monkeypatch.setattr(runtime, "rgb_texture_tiles", lambda *args: texture)
    monkeypatch.setattr(runtime, "evaluate_public_probes", lambda *args: maps)

    def embed(latents: torch.Tensor, *args: object) -> tuple[torch.Tensor, ContentAdaptiveMeasurement]:
        allocation = args[-1]
        assert isinstance(allocation, method_unweighted.ContentAllocation)
        assert allocation == expected
        measurement = ContentAdaptiveMeasurement(
            BudgetMeasurement("torch.float32", 10.0, 0.1, 0.01),
            0.004,
            0.006,
            allocation.lf_branch_share,
            allocation.hf_branch_share,
            *allocation.counterfactual_effects,
            64,
        )
        return latents + 0.01, measurement

    monkeypatch.setattr(runtime, "embed_content_unweighted", embed)
    output = runtime.run_sd35_content_unweighted(
        _Pipeline(assets),
        "content-unweighted delegation fixture",
        b"registered-key-01",
        assets,
        height=512,
        width=512,
    )
    assert tuple(
        getattr(output.measurement, name)
        for name in method_unweighted.COUNTERFACTUAL_EFFECT_FIELDS
    ) == pytest.approx(
        expected.counterfactual_effects
    )
    assert not hasattr(output.measurement, "tile_weights")
    assert not hasattr(output.measurement, "latent")


@pytest.mark.integration
def test_content_unweighted_runtime_reuses_public_builtin_and_sizedict_contract() -> None:
    builtin = _assets(official_size=False)
    official = _assets(official_size=True)
    assert builtin.runtime_asset_validation_contract_id == (
        "dinov2_small_eager_bit_image_processor_public_size_semantics_v3"
    )
    assert official.runtime_asset_validation_contract_id == (
        builtin.runtime_asset_validation_contract_id
    )
    assert builtin.evaluated_candidate_id == official.evaluated_candidate_id
