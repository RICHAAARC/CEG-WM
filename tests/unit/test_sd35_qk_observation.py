from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cegwm.runtime.sd35_qk_observation import (
    SD35QKObservationSpec,
    observe_sd35_image_qk,
    observe_sd35_image_qk_sampled_all_layers,
)


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image.height, image.width, 3)
        return (pixels.permute(2, 0, 1).unsqueeze(0) / 255.0)


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        value = pixels.mean(dim=1, keepdim=True)
        latent = value.repeat(1, 4, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))


class _Scheduler:
    def __init__(self) -> None:
        self.set_device: torch.device | None = None
        self.scale_timestep: torch.Tensor | None = None

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device) -> None:
        self.set_device = device
        self.timesteps = torch.arange(num_inference_steps, 0, -1, dtype=torch.float32, device=device)

    def scale_noise(self, latent: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Match the public FlowMatch behavior that iterates a batch-shaped
        # timestep; a legacy scalar therefore raises TypeError.
        tuple(timestep)
        if timestep.shape != (latent.shape[0],) or timestep.device != latent.device:
            raise ValueError("timestep must be device-aligned batch one")
        self.scale_timestep = timestep
        return latent + noise * 0.01 + timestep.to(latent.dtype) * 0.001


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4, bias=False)
        self.to_k = torch.nn.Linear(4, 4, bias=False)
        self.heads = 2
        with torch.no_grad():
            self.to_q.weight.copy_(torch.eye(4))
            self.to_k.weight.copy_(torch.eye(4) * 2.0)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_q(tokens), self.to_k(tokens)


class _Transformer(torch.nn.Module):
    def __init__(self, *, duplicate: bool = False, reach: bool = True, nonfinite: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(patch_size=2)
        self.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.blocks[0].attn = _Attention()
        self.duplicate, self.reach, self.nonfinite = duplicate, reach, nonfinite
        self.seen_null: tuple[torch.Tensor, torch.Tensor] | None = None
        self.seen_timestep: torch.Tensor | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
    ) -> torch.Tensor:
        self.seen_timestep = timestep
        self.seen_null = (encoder_hidden_states, pooled_projections)
        if not self.reach:
            return hidden_states
        tokens = torch.nn.functional.avg_pool2d(hidden_states, 2, 2).permute(0, 2, 3, 1).reshape(1, -1, 4)
        if self.nonfinite:
            tokens = tokens * torch.tensor(float("nan"))
        self.blocks[0].attn(tokens)
        if self.duplicate:
            self.blocks[0].attn(tokens)
        return hidden_states


class _Pipeline:
    def __init__(self, transformer: _Transformer | None = None) -> None:
        self.image_processor = _Processor()
        self.vae = _VAE()
        self.scheduler = _Scheduler()
        self.transformer = transformer or _Transformer()

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("image-only Q/K observation must not call a generation pipeline")

    def encode_prompt(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("image-only Q/K observation must not encode text")


def _spec(**overrides: object) -> SD35QKObservationSpec:
    values: dict[str, object] = {
        "model_id": "public/sd35",
        "revision": "frozen-revision",
        "attention_layer_paths": ("blocks.0.attn",),
        "inference_steps": 3,
        "schedule_index": 1,
        "public_noise_seed": 17,
        "max_grid": (2, 3),
        "null_encoder_hidden_states": torch.ones((1, 2, 4)),
        "null_pooled_projections": torch.ones((1, 4)),
    }
    values.update(overrides)
    return SD35QKObservationSpec(**values)  # type: ignore[arg-type]


def _image(value: int = 64) -> Image.Image:
    return Image.new("RGB", (8, 8), (value, value // 2, 0))


def test_spec_has_no_production_defaults_and_api_has_no_private_inputs() -> None:
    assert all(field.default is dataclasses.MISSING for field in dataclasses.fields(SD35QKObservationSpec))
    names = set(inspect.signature(observe_sd35_image_qk).parameters)
    assert names == {"image", "pipeline", "spec"}
    forbidden = {"prompt", "key", "original", "embed", "latent", "attack", "transform"}
    assert not forbidden.intersection(names)


def test_revision_is_explicit_and_may_be_unknown_without_altering_observation() -> None:
    assert "revision" in inspect.signature(SD35QKObservationSpec).parameters
    unknown = _spec(revision=None)
    known = _spec(revision="public-commit")
    assert unknown.revision is None and known.revision == "public-commit"
    pipeline = _Pipeline()
    assert torch.equal(
        observe_sd35_image_qk(_image(), pipeline=pipeline, spec=unknown).layers[0].query,
        observe_sd35_image_qk(_image(), pipeline=pipeline, spec=known).layers[0].query,
    )
    with pytest.raises(ValueError, match="revision"):
        observe_sd35_image_qk(_image(), pipeline=_Pipeline(), spec=_spec(revision=""))


def test_observation_uses_rgb_numeric_values_and_direct_null_conditioning() -> None:
    pipeline = _Pipeline()
    first = observe_sd35_image_qk(_image(32), pipeline=pipeline, spec=_spec())
    second = observe_sd35_image_qk(_image(192), pipeline=pipeline, spec=_spec())
    layer = first.layers[0]
    assert layer.query.shape == layer.key.shape == (6, 4)
    assert layer.query.dtype == layer.key.dtype == torch.float32
    assert layer.query.device.type == layer.key.device.type == "cpu"
    assert layer.source_dtype == torch.float32 and layer.source_device.type == "cpu"
    assert layer.source_shape == (1, 16, 4)
    assert layer.source_grid == (4, 4)
    assert layer.sample_indices.tolist() == [0, 2, 3, 12, 14, 15]
    assert layer.heads == 2 and layer.head_dim == 2
    assert not torch.equal(first.layers[0].query, second.layers[0].query)
    assert pipeline.transformer.seen_null is not None
    assert torch.equal(pipeline.transformer.seen_null[0].cpu(), _spec().null_encoder_hidden_states)
    assert pipeline.transformer.seen_null[0].dtype == next(pipeline.transformer.parameters()).dtype
    assert pipeline.transformer.seen_null[1].dtype == next(pipeline.transformer.parameters()).dtype
    assert first.latent_shape == (1, 4, 8, 8)
    assert first.schedule_index == 1 and first.public_noise_seed == 17


def test_scheduler_and_transformer_receive_the_same_rank_one_device_timestep() -> None:
    pipeline = _Pipeline()
    observation = observe_sd35_image_qk(_image(), pipeline=pipeline, spec=_spec())
    scheduler_timestep = pipeline.scheduler.scale_timestep
    transformer_timestep = pipeline.transformer.seen_timestep
    assert pipeline.scheduler.set_device is not None
    assert scheduler_timestep is transformer_timestep
    assert scheduler_timestep is not None and scheduler_timestep.shape == (1,)
    assert scheduler_timestep.device == torch.device("cpu")
    assert observation.timestep.shape == (1,)


def test_realistic_scheduler_rejects_the_legacy_scalar_timestep() -> None:
    scheduler = _Scheduler()
    scheduler.set_timesteps(3, device=torch.device("cpu"))
    with pytest.raises(TypeError):
        scheduler.scale_noise(torch.zeros((1, 1)), scheduler.timesteps[1], torch.zeros((1, 1)))


def test_scheduler_boundary_is_tagged_without_exception_text_classification() -> None:
    class _FailingScheduler(_Scheduler):
        def scale_noise(self, latent: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            raise TypeError("public scheduler boundary")

    pipeline = _Pipeline()
    pipeline.scheduler = _FailingScheduler()
    with pytest.raises(TypeError) as captured:
        observe_sd35_image_qk(_image(), pipeline=pipeline, spec=_spec())
    assert getattr(captured.value, "geometry_failure_point") == "scheduler"


def test_public_seed_is_deterministic_and_does_not_mutate_global_rng() -> None:
    pipeline = _Pipeline()
    first = observe_sd35_image_qk(_image(), pipeline=pipeline, spec=_spec())
    second = observe_sd35_image_qk(_image(), pipeline=pipeline, spec=_spec())
    assert torch.equal(first.layers[0].query, second.layers[0].query)
    torch.manual_seed(1234)
    expected = torch.rand(3)
    torch.manual_seed(1234)
    observe_sd35_image_qk(_image(), pipeline=pipeline, spec=_spec())
    assert torch.equal(torch.rand(3), expected)


@pytest.mark.parametrize(
    ("spec_change", "transformer", "match"),
    [
        ({"attention_layer_paths": ("blocks.1.attn",)}, None, "not found"),
        ({}, _Transformer(reach=False), "exactly once"),
        ({}, _Transformer(duplicate=True), "exactly once"),
        ({}, _Transformer(nonfinite=True), "finite"),
        ({"schedule_index": 3}, None, "outside"),
        ({"max_grid": (0, 1)}, None, "max_grid"),
    ],
)
def test_invalid_specs_or_projection_paths_fail_closed(
    spec_change: dict[str, object], transformer: _Transformer | None, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        observe_sd35_image_qk(_image(), pipeline=_Pipeline(transformer), spec=_spec(**spec_change))


def test_hooks_are_removed_after_transformer_failure() -> None:
    class _FailingTransformer(_Transformer):
        def forward(self, **kwargs: object) -> torch.Tensor:
            self.blocks[0].attn.to_q(torch.ones((1, 16, 4)))
            raise RuntimeError("fake failure")

    transformer = _FailingTransformer()
    with pytest.raises(RuntimeError, match="fake failure"):
        observe_sd35_image_qk(_image(), pipeline=_Pipeline(transformer), spec=_spec())
    assert not transformer.blocks[0].attn.to_q._forward_hooks
    assert not transformer.blocks[0].attn.to_k._forward_hooks


def test_all_layer_entrypoint_samples_in_hooks_and_retains_per_layer_failure() -> None:
    transformer = _Transformer()
    observation = observe_sd35_image_qk_sampled_all_layers(_image(), pipeline=_Pipeline(transformer), spec=_spec())
    assert len(observation.layers) == 1
    layer = observation.layers[0]
    assert layer.query.shape == layer.key.shape == (6, 4)
    assert layer.query.device.type == layer.key.device.type == "cpu"
    assert not observation.layer_failures
    assert not transformer.blocks[0].attn.to_q._forward_hooks
    assert not transformer.blocks[0].attn.to_k._forward_hooks
