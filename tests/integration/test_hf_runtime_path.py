from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from PIL import Image

from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_CORE_CANDIDATE_ID,
    LF_SHELL_CANDIDATE_ID,
    FrozenLFPublicAssets,
)
from cegwm.runtime.diffusers_sd35 import (
    HFInjectionCallback,
    LFInjectionCallback,
    load_sd35_pipeline,
    run_sd35_hf,
    run_sd35_lf,
)


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        class Distribution:
            def mode(self) -> torch.Tensor:
                return torch.cat([pixels, pixels.mean(dim=1, keepdim=True)], dim=1)

        return SimpleNamespace(latent_dist=Distribution())


def _assets() -> FrozenHFPublicAssets:
    return FrozenHFPublicAssets(
        vae=_VAE(),
        image_processor=_Processor(),
        image_processor_id="sd35-vae-image-processor-v1",
    )


def _lf_assets(candidate_id: str) -> FrozenLFPublicAssets:
    return FrozenLFPublicAssets(
        vae=_VAE(),
        image_processor=_Processor(),
        image_processor_id="sd35-vae-image-processor-v1",
        candidate_id=candidate_id,
    )


class _FakeSD35Pipeline:
    def __init__(self) -> None:
        self.target_before: torch.Tensor | None = None
        self.target_after: torch.Tensor | None = None

    def __call__(
        self,
        *,
        prompt: str,
        num_inference_steps: int,
        height: int,
        width: int,
        generator: torch.Generator | None,
        output_type: str,
        callback_on_step_end: HFInjectionCallback,
        callback_on_step_end_tensor_inputs: list[str],
    ) -> SimpleNamespace:
        del prompt, generator
        assert num_inference_steps == 20
        assert output_type == "pil"
        assert callback_on_step_end_tensor_inputs == ["latents"]
        latents = torch.linspace(-1.0, 1.0, 4 * 32 * 32).reshape(1, 4, 32, 32)
        for step in range(num_inference_steps):
            latents = latents + float(step + 1) / 1000.0
            state = {"latents": latents}
            if step == 18:
                self.target_before = latents.clone()
            state = callback_on_step_end(self, step, torch.tensor(step), state)
            latents = state["latents"]
            if step == 18:
                self.target_after = latents.clone()
        rgb = torch.sigmoid(latents[:, :3] * 8.0)
        rgb = functional.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
        pixels = (rgb[0].permute(1, 2, 0) * 255.0).round().byte().numpy()
        return SimpleNamespace(images=[Image.fromarray(pixels, mode="RGB")])


@pytest.mark.integration
def test_callback_modifies_and_returns_the_actual_frozen_step_state() -> None:
    callback = HFInjectionCallback(
        b"0123456789abcdef0123456789abcdef",
        _assets(),
    )
    latents = torch.linspace(-1.0, 1.0, 4 * 16 * 16).reshape(1, 4, 16, 16)
    early_state = {"latents": latents}

    assert callback(None, 17, None, early_state) is early_state
    target_state = callback(None, 18, None, early_state)

    assert target_state is not early_state
    assert not torch.equal(target_state["latents"], latents)
    assert callback.measurement is not None
    assert callback.measurement.relative_l2 <= 0.012
    with pytest.raises(RuntimeError, match="more than once"):
        callback(None, 18, None, early_state)


@pytest.mark.integration
def test_runtime_path_uses_evolving_latent_and_returns_key_dependent_rgb() -> None:
    first_pipeline = _FakeSD35Pipeline()
    second_pipeline = _FakeSD35Pipeline()
    assets = _assets()

    first = run_sd35_hf(
        first_pipeline,
        "a public test prompt",
        b"0123456789abcdef0123456789abcdef",
        assets,
        height=256,
        width=256,
    )
    second = run_sd35_hf(
        second_pipeline,
        "a public test prompt",
        b"abcdef0123456789abcdef0123456789",
        assets,
        height=256,
        width=256,
    )

    assert first.image.mode == "RGB"
    assert first.image.size == (256, 256)
    assert first_pipeline.target_before is not None
    assert first_pipeline.target_after is not None
    assert not torch.equal(first_pipeline.target_before, first_pipeline.target_after)
    assert not np.array_equal(np.asarray(first.image), np.asarray(second.image))
    assert first.injection_budget.relative_l2 <= 0.012


@pytest.mark.integration
@pytest.mark.parametrize("candidate_id", [LF_CORE_CANDIDATE_ID, LF_SHELL_CANDIDATE_ID])
def test_lf_runtime_uses_same_real_callback_and_returns_final_rgb(candidate_id: str) -> None:
    pipeline = _FakeSD35Pipeline()
    result = run_sd35_lf(
        pipeline,
        "a public test prompt",
        b"0123456789abcdef0123456789abcdef",
        _lf_assets(candidate_id),
        height=256,
        width=256,
        generator=torch.Generator().manual_seed(5),
    )

    assert result.image.mode == "RGB"
    assert pipeline.target_before is not None
    assert pipeline.target_after is not None
    assert not torch.equal(pipeline.target_before, pipeline.target_after)
    assert 0.0 < result.injection_budget.relative_l2 <= 0.012


@pytest.mark.integration
def test_lf_callback_is_single_candidate_and_fail_closed() -> None:
    callback = LFInjectionCallback(
        b"0123456789abcdef0123456789abcdef",
        _lf_assets(LF_CORE_CANDIDATE_ID),
    )
    latents = torch.linspace(-1.0, 1.0, 4 * 16 * 16).reshape(1, 4, 16, 16)
    early = {"latents": latents}

    assert callback(None, 17, None, early) is early
    updated = callback(None, 18, None, early)
    assert not torch.equal(updated["latents"], latents)
    with pytest.raises(RuntimeError, match="more than once"):
        callback(None, 18, None, early)


@pytest.mark.integration
def test_model_loader_uses_protocol_name_without_revision_or_local_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class CompatiblePipeline:
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(images=[])

    class Factory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> CompatiblePipeline:
            calls.append((model_id, kwargs))
            return CompatiblePipeline()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(StableDiffusion3Pipeline=Factory),
    )

    pipeline = load_sd35_pipeline(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.float16,
        token="hf_test_token",
    )

    assert isinstance(pipeline, CompatiblePipeline)
    assert calls == [(
        "stabilityai/stable-diffusion-3.5-medium",
        {"torch_dtype": torch.float16, "token": "hf_test_token"},
    )]


@pytest.mark.integration
def test_missing_or_incompatible_diffusers_capability_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "diffusers", None)
    with pytest.raises(RuntimeError, match="diffusers is required"):
        load_sd35_pipeline(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.float16,
            token="hf_test_token",
        )

    with pytest.raises(ValueError, match="token"):
        load_sd35_pipeline(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.float16,
            token="",
        )

    class IncompatiblePipeline:
        def __call__(self, prompt: str) -> SimpleNamespace:
            del prompt
            return SimpleNamespace(images=[])

    with pytest.raises(TypeError, match="callback-on-step-end"):
        run_sd35_hf(
            IncompatiblePipeline(),
            "a public test prompt",
            b"0123456789abcdef0123456789abcdef",
            _assets(),
            height=256,
            width=256,
        )
