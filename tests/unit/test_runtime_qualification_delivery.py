from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import types
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional

from runtime import (
    RuntimeAdapterError,
    RuntimeContentExecutionError,
    RuntimeDetectionConditioning,
    RuntimeGenerationPromptIdentity,
    RuntimeQkObservationError,
    RuntimeSession,
    Sd35BackendError,
    Sd35PipelineBackend,
    create_runtime_adapter,
    load_runtime_configuration,
    observe_differentiable_detection_qk,
)
from runtime import sd35_backend as sd35_backend_module
from main import ContentEmbedderError, content_embedder, hf_carrier
from scripts.experiment_execution import runtime_qualification_runner as runner
from scripts.experiment_execution.build_runtime_qualification_package import (
    PackageBuildError,
    build_runtime_qualification_package,
)


pytestmark = pytest.mark.unit


class _DecoderLocalizationMiddleBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList((torch.nn.Identity(), torch.nn.Identity()))
        self.attentions = torch.nn.ModuleList((torch.nn.Identity(),))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.resnets[0](value)
        value = self.attentions[0](value)
        return self.resnets[1](value)


class _DecoderLocalizationPassThrough(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = torch.nn.Identity()
        self.mid_block = _DecoderLocalizationMiddleBlock()
        self.up_blocks = torch.nn.ModuleList(
            torch.nn.Identity() for _index in range(4)
        )
        self.conv_norm_out = torch.nn.Identity()
        self.conv_act = torch.nn.Identity()
        self.conv_out = torch.nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv_in(value)
        value = self.mid_block(value)
        for up_block in self.up_blocks:
            value = up_block(value)
        value = self.conv_norm_out(value)
        value = self.conv_act(value)
        return self.conv_out(value)


def _runner_storage(
    tmp_path: Path,
    label: str,
) -> tuple[Path, Path, Path]:
    ephemeral = tmp_path / f"{label}-ephemeral"
    persistent = tmp_path / f"{label}-persistent"
    return ephemeral, persistent, ephemeral / f"{label}.zip"


def test_sd35_backend_is_lazy_and_accepts_disjoint_roots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "runtime.sd35_backend.importlib.import_module",
        lambda name: calls.append(name),
    )
    Sd35PipelineBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token=None,
        prompt="probe",
    )
    assert calls == []


def test_sd35_backend_reports_specific_differentiable_stage_types(
    monkeypatch,
    tmp_path: Path,
) -> None:
    configuration = load_runtime_configuration()
    latent = torch.ones((1, 16, 2, 2), dtype=torch.float32)
    image = torch.ones((1, 3, 2, 2), dtype=torch.float32)

    class Scheduler:
        def __init__(self, *, fail_step: bool = False, fail_noise: bool = False):
            self.fail_step = fail_step
            self.fail_noise = fail_noise

        def step(self, _noise, _timestep, value, *, return_dict):
            assert return_dict is False
            if self.fail_step:
                raise RuntimeError("excluded scheduler detail")
            return (value,)

        def scale_noise(self, value, _timestep, noise):
            if self.fail_noise:
                raise RuntimeError("excluded noise detail")
            return value + noise

    class Transformer:
        def __init__(self, *, fail: bool):
            self.fail = fail

        def __call__(self, **kwargs):
            if self.fail:
                raise RuntimeError("excluded transformer detail")
            return (torch.zeros_like(kwargs["hidden_states"]),)

    class Vae:
        dtype = torch.float32
        config = types.SimpleNamespace(force_upcast=False)

        def __init__(
            self,
            *,
            fail_preparation: bool = False,
            fail_decode: bool = False,
            decode_error: BaseException | None = None,
            fail_encode: bool = False,
        ):
            self.dtype = torch.float16 if fail_preparation else torch.float32
            self.config = types.SimpleNamespace(
                force_upcast=fail_preparation,
                use_post_quant_conv=False,
            )
            self.post_quant_conv = None
            self.decoder = _DecoderLocalizationPassThrough()
            self.fail_preparation = fail_preparation
            self.fail_decode = fail_decode
            self.decode_error = decode_error
            self.fail_encode = fail_encode

        def to(self, *, dtype):
            assert dtype is torch.float32
            if self.fail_preparation:
                raise RuntimeError("excluded preparation detail")
            self.dtype = dtype
            return self

        def decode(self, value, *, return_dict):
            assert return_dict is True
            if self.decode_error is not None:
                raise self.decode_error
            if self.fail_decode:
                raise RuntimeError("excluded decode detail")
            return types.SimpleNamespace(sample=value[:, :3])

        def encode(self, value, *, return_dict):
            assert return_dict is True
            if self.fail_encode:
                raise RuntimeError("excluded encode detail")
            return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: value))

    class ImageProcessor:
        def __init__(self, *, fail_postprocess: bool = False) -> None:
            self.fail_postprocess = fail_postprocess

        def postprocess(self, value, *, output_type):
            assert output_type == "pt"
            if self.fail_postprocess:
                raise RuntimeError("excluded postprocess detail")
            return value

        def preprocess(self, value, *, height, width):
            assert (height, width) == (512, 512)
            return value

    class Pipeline:
        def __init__(
            self,
            *,
            transformer: Transformer,
            scheduler: Scheduler | None = None,
            vae: Vae | None = None,
            fail_prompt_encoding: bool = False,
            fail_postprocess: bool = False,
        ) -> None:
            self.transformer = transformer
            self.scheduler = scheduler or Scheduler()
            self.vae = vae or Vae()
            self.image_processor = ImageProcessor(
                fail_postprocess=fail_postprocess
            )
            self.fail_prompt_encoding = fail_prompt_encoding

        def encode_prompt(self, **kwargs):
            assert kwargs["do_classifier_free_guidance"] is False
            if self.fail_prompt_encoding:
                raise RuntimeError("excluded conditioning detail")
            prompt = torch.zeros((1, 2, 2), dtype=torch.float32)
            pooled = torch.zeros((1, 2), dtype=torch.float32)
            return prompt, None, pooled, None

    def backend(pipeline: Pipeline) -> Sd35PipelineBackend:
        value = Sd35PipelineBackend(
            cache_root=tmp_path / "cache",
            persistent_root=tmp_path / "persistent",
            hf_token=None,
            prompt="probe",
        )
        value._configuration = configuration
        value._device = torch.device("cpu")
        value._pipeline = pipeline
        return value

    def suffix_context(value: Sd35PipelineBackend, scheduler: Scheduler):
        return sd35_backend_module._PipelineGenerationSuffixReplayContext(
            runtime_config_digest=configuration.runtime_config_digest,
            callback_index=configuration.callback_index,
            owner_identity=id(value),
            latent_shape=tuple(latent.shape),
            latent_dtype=latent.dtype,
            selected_device="cpu",
            prompt_identity=RuntimeGenerationPromptIdentity.from_prompts("probe", ""),
            prompt_embeds=torch.zeros((2, 2, 2), dtype=torch.float32),
            pooled_prompt_embeds=torch.zeros((2, 2), dtype=torch.float32),
            suffix_timesteps=torch.tensor([1.0]),
            scheduler_snapshot=scheduler,
        )

    suffix_transformer = backend(Pipeline(transformer=Transformer(fail=True)))
    with pytest.raises(sd35_backend_module.Sd35BackendError) as transformer_error:
        suffix_transformer.replay_generation_suffix(
            latent,
            suffix_context(suffix_transformer, Scheduler()),
            differentiable=True,
        )
    assert isinstance(
        transformer_error.value.__cause__,
        sd35_backend_module.Sd35BackendGenerationSuffixTransformerForwardError,
    )

    suffix_scheduler = backend(Pipeline(transformer=Transformer(fail=False)))
    with pytest.raises(sd35_backend_module.Sd35BackendError) as scheduler_error:
        suffix_scheduler.replay_generation_suffix(
            latent,
            suffix_context(suffix_scheduler, Scheduler(fail_step=True)),
            differentiable=True,
        )
    assert isinstance(
        scheduler_error.value.__cause__,
        sd35_backend_module.Sd35BackendGenerationSuffixSchedulerStepError,
    )

    memory_snapshot = {
        "allocated_bytes": 10,
        "reserved_bytes": 20,
        "max_allocated_bytes": 30,
        "max_reserved_bytes": 40,
        "total_device_bytes": 100,
    }
    monkeypatch.setattr(
        sd35_backend_module,
        "_cuda_memory_snapshot",
        lambda _device: dict(memory_snapshot),
    )

    preparation = backend(
        Pipeline(
            transformer=Transformer(fail=False),
            vae=Vae(fail_preparation=True),
        )
    )
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeInputPreparationError
    ) as preparation_error:
        preparation.vae_decode_differentiable(latent)

    decode = backend(
        Pipeline(transformer=Transformer(fail=False), vae=Vae(fail_decode=True))
    )
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError
    ) as decode_error:
        decode.vae_decode_differentiable(latent)
    assert decode_error.value.runtime_reason_identity == (
        "unclassified_runtime_failure"
    )

    postprocess = backend(
        Pipeline(
            transformer=Transformer(fail=False),
            fail_postprocess=True,
        )
    )
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableImagePostprocessError
    ) as postprocess_error:
        postprocess.vae_decode_differentiable(latent)

    for error in (
        preparation_error.value,
        decode_error.value,
        postprocess_error.value,
    ):
        assert dict(error.cuda_memory_facts) == {
            "before_allocated_bytes": 10,
            "before_reserved_bytes": 20,
            "before_max_allocated_bytes": 30,
            "before_max_reserved_bytes": 40,
            "after_allocated_bytes": 10,
            "after_reserved_bytes": 20,
            "after_max_allocated_bytes": 30,
            "after_max_reserved_bytes": 40,
            "total_device_bytes": 100,
        }

    success_latent = latent.detach().clone().requires_grad_(True)
    success = backend(Pipeline(transformer=Transformer(fail=False)))
    success_image = success.vae_decode_differentiable(success_latent)
    assert torch.equal(success_image, success_latent[:, :3])
    assert torch.equal(
        torch.autograd.grad(success_image.sum(), success_latent)[0],
        torch.cat(
            (
                torch.ones_like(success_latent[:, :3]),
                torch.zeros_like(success_latent[:, 3:]),
            ),
            dim=1,
        ),
    )

    monkeypatch.setattr(
        sd35_backend_module,
        "_cuda_memory_snapshot",
        lambda _device: None,
    )
    no_cuda_facts = backend(
        Pipeline(transformer=Transformer(fail=False), vae=Vae(fail_decode=True))
    )
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError
    ) as no_cuda_error:
        no_cuda_facts.vae_decode_differentiable(latent)
    assert no_cuda_error.value.cuda_memory_facts == ()

    reason_cases = (
        (
            torch.OutOfMemoryError("excluded native memory detail"),
            "runtime_reported_memory_allocation_failure",
        ),
        (
            RuntimeError("CUDA out of memory: excluded allocator detail"),
            "runtime_reported_memory_allocation_failure",
        ),
        (
            RuntimeError("CUDA error: excluded kernel detail"),
            "cuda_kernel_execution_failure",
        ),
        (
            RuntimeError("expected scalar type Float but found Half"),
            "dtype_shape_operator_contract_failure",
        ),
        (
            RuntimeError("excluded unknown runtime detail"),
            "unclassified_runtime_failure",
        ),
    )
    for runtime_error, expected_reason in reason_cases:
        reason_backend = backend(
            Pipeline(
                transformer=Transformer(fail=False),
                vae=Vae(decode_error=runtime_error),
            )
        )
        with pytest.raises(
            sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError
        ) as reason_error:
            reason_backend.vae_decode_differentiable(latent)
        assert reason_error.value.runtime_reason_identity == expected_reason
        assert reason_error.value.__cause__ is runtime_error

    with pytest.raises(sd35_backend_module.Sd35BackendError):
        sd35_backend_module.Sd35BackendDifferentiableVaeDecodeForwardError(
            runtime_reason_identity="unsafe_unregistered_runtime_reason"
        )

    encode = backend(
        Pipeline(transformer=Transformer(fail=False), vae=Vae(fail_encode=True))
    )
    with pytest.raises(sd35_backend_module.Sd35BackendDifferentiableVaeEncodeError):
        encode.vae_encode_differentiable(image)

    noise = backend(Pipeline(transformer=Transformer(fail=False)))
    noise._detection_scheduler = Scheduler(fail_noise=True)
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableDetectionNoiseSchedulingError
    ):
        noise.scale_detection_noise_differentiable(
            latent,
            torch.ones_like(latent),
            torch.tensor([1.0]),
        )

    qk = backend(Pipeline(transformer=Transformer(fail=True)))
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableQkTransformerForwardError
    ):
        qk.run_qk_detection_forward_differentiable(
            latent,
            torch.tensor([1.0]),
            RuntimeDetectionConditioning(
                prompt="",
                prompt_2="",
                prompt_3="",
                do_classifier_free_guidance=False,
                detection_conditioning_protocol=(
                    "sd3_empty_text_triplet_without_cfg"
                ),
            ),
        )

    conditioning = backend(
        Pipeline(
            transformer=Transformer(fail=False),
            fail_prompt_encoding=True,
        )
    )
    with pytest.raises(sd35_backend_module.Sd35BackendError) as conditioning_error:
        conditioning.run_qk_detection_forward_differentiable(
            latent,
            torch.tensor([1.0]),
            RuntimeDetectionConditioning(
                prompt="",
                prompt_2="",
                prompt_3="",
                do_classifier_free_guidance=False,
                detection_conditioning_protocol=(
                    "sd3_empty_text_triplet_without_cfg"
                ),
            ),
        )
    assert not isinstance(
        conditioning_error.value,
        sd35_backend_module.Sd35BackendDifferentiableQkTransformerForwardError,
    )


def test_sd35_backend_checkpointed_suffix_preserves_values_gradients_and_call_boundaries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    configuration = load_runtime_configuration()
    latent_shape = (1, 16, 2, 2)

    class Scheduler:
        def __init__(self, events: list[str]) -> None:
            self.events = events

        def __deepcopy__(self, _memo):
            return self

        def step(self, noise, timestep, value, *, return_dict):
            assert return_dict is False
            assert timestep.shape == ()
            self.events.append("scheduler")
            return (value - noise * 0.125,)

    class Transformer(torch.nn.Module):
        def __init__(
            self,
            events: list[str],
            *,
            fail_on_call: int | None = None,
        ) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.events = events
            self.fail_on_call = fail_on_call
            self.call_count = 0

        def forward(self, **kwargs):
            self.call_count += 1
            self.events.append("transformer")
            if self.call_count == self.fail_on_call:
                raise torch.OutOfMemoryError("excluded transformer detail")
            hidden_states = kwargs["hidden_states"]
            assert kwargs["return_dict"] is False
            assert kwargs["joint_attention_kwargs"] is None
            assert kwargs["timestep"].shape == (2,)
            return (hidden_states * self.weight,)

    class Pipeline:
        def __init__(self, transformer: Transformer, scheduler: Scheduler) -> None:
            self.transformer = transformer
            self.scheduler = scheduler

    def backend(
        transformer: Transformer,
        scheduler: Scheduler,
    ) -> Sd35PipelineBackend:
        value = Sd35PipelineBackend(
            cache_root=tmp_path / f"cache-{id(transformer)}",
            persistent_root=tmp_path / f"persistent-{id(transformer)}",
            hf_token=None,
            prompt="probe",
        )
        value._configuration = configuration
        value._device = torch.device("cpu")
        value._pipeline = Pipeline(transformer, scheduler)
        return value

    def suffix_context(
        value: Sd35PipelineBackend,
        scheduler: Scheduler,
    ) -> sd35_backend_module._PipelineGenerationSuffixReplayContext:
        return sd35_backend_module._PipelineGenerationSuffixReplayContext(
            runtime_config_digest=configuration.runtime_config_digest,
            callback_index=configuration.callback_index,
            owner_identity=id(value),
            latent_shape=latent_shape,
            latent_dtype=torch.float32,
            selected_device="cpu",
            prompt_identity=RuntimeGenerationPromptIdentity.from_prompts("probe", ""),
            prompt_embeds=torch.zeros((2, 2, 2), dtype=torch.float32),
            pooled_prompt_embeds=torch.zeros((2, 2), dtype=torch.float32),
            suffix_timesteps=torch.tensor([1.0]),
            scheduler_snapshot=scheduler,
        )

    def direct_reference(
        callback_latent: torch.Tensor,
        transformer: Transformer,
        scheduler: Scheduler,
    ) -> torch.Tensor:
        latent_model_input = torch.cat([callback_latent, callback_latent], dim=0)
        timestep = torch.tensor(1.0).expand(latent_model_input.shape[0])
        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=torch.zeros((2, 2, 2), dtype=torch.float32),
            pooled_projections=torch.zeros((2, 2), dtype=torch.float32),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        guided_noise = noise_pred_uncond + configuration.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return scheduler.step(
            guided_noise,
            torch.tensor(1.0),
            callback_latent,
            return_dict=False,
        )[0]

    reference_events: list[str] = []
    reference_transformer = Transformer(reference_events)
    assert reference_transformer.weight.requires_grad
    reference_scheduler = Scheduler(reference_events)
    reference_latent = torch.linspace(
        -1.0,
        1.0,
        steps=64,
        dtype=torch.float32,
    ).reshape(latent_shape).requires_grad_(True)
    reference_terminal = direct_reference(
        reference_latent,
        reference_transformer,
        reference_scheduler,
    )
    reference_gradient = torch.autograd.grad(
        reference_terminal.sum(),
        reference_latent,
    )[0]

    initial_failure_events: list[str] = []
    initial_failure_transformer = Transformer(
        initial_failure_events,
        fail_on_call=1,
    )
    initial_failure_transformer.requires_grad_(False)
    initial_failure_scheduler = Scheduler(initial_failure_events)
    initial_failure_backend = backend(
        initial_failure_transformer,
        initial_failure_scheduler,
    )
    with pytest.raises(sd35_backend_module.Sd35BackendError) as initial_error:
        initial_failure_backend.replay_generation_suffix(
            reference_latent.detach().clone().requires_grad_(True),
            suffix_context(initial_failure_backend, initial_failure_scheduler),
            differentiable=True,
        )
    assert isinstance(
        initial_error.value.__cause__,
        sd35_backend_module.Sd35BackendGenerationSuffixTransformerForwardError,
    )
    assert isinstance(initial_error.value.__cause__.__cause__, torch.OutOfMemoryError)
    assert initial_failure_events == ["transformer"]

    checkpoint_events: list[str] = []
    checkpoint_transformer = Transformer(checkpoint_events)
    checkpoint_transformer.requires_grad_(False)
    checkpoint_scheduler = Scheduler(checkpoint_events)
    checkpoint_backend = backend(checkpoint_transformer, checkpoint_scheduler)
    checkpoint_latent = reference_latent.detach().clone().requires_grad_(True)
    checkpoint_terminal = checkpoint_backend.replay_generation_suffix(
        checkpoint_latent,
        suffix_context(checkpoint_backend, checkpoint_scheduler),
        differentiable=True,
    )
    checkpoint_gradient = torch.autograd.grad(
        checkpoint_terminal.sum(),
        checkpoint_latent,
    )[0]

    assert torch.equal(checkpoint_terminal, reference_terminal)
    assert torch.equal(checkpoint_gradient, reference_gradient)
    assert bool(torch.isfinite(checkpoint_gradient).all())
    assert bool(torch.count_nonzero(checkpoint_gradient))
    assert checkpoint_events == ["transformer", "scheduler", "transformer"]
    assert checkpoint_transformer.call_count == 2
    assert checkpoint_transformer.weight.grad is None

    failing_events: list[str] = []
    failing_transformer = Transformer(failing_events, fail_on_call=2)
    failing_transformer.requires_grad_(False)
    failing_scheduler = Scheduler(failing_events)
    failing_backend = backend(failing_transformer, failing_scheduler)
    failing_latent = reference_latent.detach().clone().requires_grad_(True)
    failing_terminal = failing_backend.replay_generation_suffix(
        failing_latent,
        suffix_context(failing_backend, failing_scheduler),
        differentiable=True,
    )
    with pytest.raises(
        sd35_backend_module.Sd35BackendGenerationSuffixTransformerForwardError
    ) as recompute_error:
        torch.autograd.grad(failing_terminal.sum(), failing_latent)
    assert isinstance(recompute_error.value.__cause__, torch.OutOfMemoryError)
    assert failing_events == ["transformer", "scheduler", "transformer"]
    assert failing_transformer.weight.grad is None

    def forbidden_checkpoint(*_args, **_kwargs):
        raise AssertionError("non-differentiable replay must not use checkpoint")

    monkeypatch.setattr(
        sd35_backend_module,
        "activation_checkpoint",
        forbidden_checkpoint,
    )
    inference_events: list[str] = []
    inference_transformer = Transformer(inference_events)
    inference_transformer.requires_grad_(False)
    inference_scheduler = Scheduler(inference_events)
    inference_backend = backend(inference_transformer, inference_scheduler)
    inference_terminal = inference_backend.replay_generation_suffix(
        reference_latent.detach(),
        suffix_context(inference_backend, inference_scheduler),
        differentiable=False,
    )
    assert not inference_terminal.requires_grad
    assert inference_events == ["transformer", "scheduler"]
    assert inference_transformer.call_count == 1
    assert inference_transformer.weight.grad is None


def test_sd35_backend_checkpointed_differentiable_vae_decode_preserves_values_gradients_and_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    configuration = load_runtime_configuration()

    class Vae(torch.nn.Module):
        def __init__(self, *, fail_on_call: int | None = None) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.75))
            self.config = types.SimpleNamespace(
                force_upcast=False,
                use_post_quant_conv=False,
            )
            self.post_quant_conv = None
            self.decoder = _DecoderLocalizationPassThrough()
            self.dtype = torch.float32
            self.fail_on_call = fail_on_call
            self.call_count = 0

        def decode(self, value, *, return_dict):
            assert return_dict is True
            self.call_count += 1
            if self.call_count == self.fail_on_call:
                raise torch.OutOfMemoryError("excluded VAE decoder detail")
            expanded = torch_functional.interpolate(
                value[:, :3] * self.weight + value[:, 3:6] * 0.125,
                scale_factor=8,
                mode="nearest",
            )
            decoded = expanded
            for scale in (0.875, 0.75, 0.625, 0.5):
                decoded = torch.tanh(decoded * scale)
            return types.SimpleNamespace(sample=decoded)

    class ImageProcessor:
        @staticmethod
        def postprocess(value, *, output_type):
            assert output_type == "pt"
            return value

    class Pipeline:
        def __init__(self, vae: Vae) -> None:
            self.vae = vae
            self.image_processor = ImageProcessor()

    def backend(vae: Vae) -> Sd35PipelineBackend:
        value = Sd35PipelineBackend(
            cache_root=tmp_path / f"cache-{id(vae)}",
            persistent_root=tmp_path / f"persistent-{id(vae)}",
            hf_token=None,
            prompt="probe",
        )
        value._configuration = configuration
        value._device = torch.device("cpu")
        value._pipeline = Pipeline(vae)
        return value

    reference_vae = Vae()
    assert reference_vae.weight.requires_grad
    reference_input = torch.linspace(
        -1.0,
        1.0,
        steps=16 * 4 * 4,
        dtype=torch.float32,
    ).reshape(1, 16, 4, 4).requires_grad_(True)
    reference_image = reference_vae.decode(
        reference_input,
        return_dict=True,
    ).sample
    reference_gradient = torch.autograd.grad(
        reference_image.square().sum(),
        reference_input,
    )[0]

    checkpoint_vae = Vae()
    checkpoint_vae.requires_grad_(False)
    checkpoint_backend = backend(checkpoint_vae)
    checkpoint_input = reference_input.detach().clone().requires_grad_(True)
    checkpoint_image = checkpoint_backend.vae_decode_differentiable(
        checkpoint_input
    )
    checkpoint_gradient = torch.autograd.grad(
        checkpoint_image.square().sum(),
        checkpoint_input,
    )[0]
    assert torch.equal(checkpoint_image, reference_image)
    assert torch.equal(checkpoint_gradient, reference_gradient)
    assert bool(torch.isfinite(checkpoint_gradient).all())
    assert bool(torch.count_nonzero(checkpoint_gradient))
    assert checkpoint_vae.call_count == 2
    assert checkpoint_vae.weight.grad is None

    def saved_tensor_bytes(callable_decode) -> tuple[torch.Tensor, int]:
        total = 0

        def record_saved_tensor(value: torch.Tensor) -> torch.Tensor:
            nonlocal total
            total += value.numel() * value.element_size()
            return value

        with torch.autograd.graph.saved_tensors_hooks(
            record_saved_tensor,
            lambda value: value,
        ):
            image = callable_decode()
        return image, total

    direct_memory_vae = Vae()
    direct_memory_vae.requires_grad_(False)
    direct_memory_input = reference_input.detach().clone().requires_grad_(True)
    direct_memory_image, direct_saved_bytes = saved_tensor_bytes(
        lambda: direct_memory_vae.decode(
            direct_memory_input,
            return_dict=True,
        ).sample
    )
    checkpoint_memory_vae = Vae()
    checkpoint_memory_vae.requires_grad_(False)
    checkpoint_memory_input = reference_input.detach().clone().requires_grad_(True)
    checkpoint_memory_image, checkpoint_saved_bytes = saved_tensor_bytes(
        lambda: backend(checkpoint_memory_vae).vae_decode_differentiable(
            checkpoint_memory_input
        )
    )
    assert torch.equal(checkpoint_memory_image, direct_memory_image)
    assert checkpoint_saved_bytes < direct_saved_bytes

    initial_failure_vae = Vae(fail_on_call=1)
    initial_failure_vae.requires_grad_(False)
    initial_failure_backend = backend(initial_failure_vae)
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError
    ) as initial_error:
        initial_failure_backend.vae_decode_differentiable(
            reference_input.detach().clone().requires_grad_(True)
        )
    assert isinstance(initial_error.value.__cause__, torch.OutOfMemoryError)
    assert initial_error.value.operation_identity == (
        "differentiable_vae_initial_decode_forward"
    )
    assert initial_error.value.runtime_reason_identity == (
        "runtime_reported_memory_allocation_failure"
    )
    initial_decoder_boundary = initial_error.value.decoder_operation_identity
    assert initial_decoder_boundary == "differentiable_vae_decode_entry"

    initial_failure_vae.fail_on_call = None
    recovered_input = reference_input.detach().clone().requires_grad_(True)
    recovered_image = initial_failure_backend.vae_decode_differentiable(
        recovered_input
    )
    recovered_gradient = torch.autograd.grad(
        recovered_image.square().sum(),
        recovered_input,
    )[0]
    assert torch.equal(recovered_image, reference_image)
    assert torch.equal(recovered_gradient, reference_gradient)

    recompute_failure_vae = Vae(fail_on_call=2)
    recompute_failure_vae.requires_grad_(False)
    recompute_failure_backend = backend(recompute_failure_vae)
    recompute_input = reference_input.detach().clone().requires_grad_(True)
    recompute_image = recompute_failure_backend.vae_decode_differentiable(
        recompute_input
    )
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeCheckpointRecomputationError
    ) as recompute_error:
        torch.autograd.grad(recompute_image.square().sum(), recompute_input)
    assert isinstance(recompute_error.value.__cause__, torch.OutOfMemoryError)
    assert recompute_error.value.operation_identity == (
        "differentiable_vae_checkpoint_recomputation"
    )
    assert recompute_error.value.runtime_reason_identity == (
        "runtime_reported_memory_allocation_failure"
    )
    recomputed_decoder_boundary = recompute_error.value.decoder_operation_identity
    assert recomputed_decoder_boundary == "differentiable_vae_decode_entry"

    framework_failure = MemoryError("excluded checkpoint framework detail")

    def raise_checkpoint_framework_failure(*_args, **_kwargs):
        raise framework_failure

    with monkeypatch.context() as checkpoint_patch:
        checkpoint_patch.setattr(
            sd35_backend_module,
            "activation_checkpoint",
            raise_checkpoint_framework_failure,
        )
        with pytest.raises(
            sd35_backend_module.Sd35BackendDifferentiableVaeCheckpointExecutionError
        ) as framework_error:
            backend(Vae()).vae_decode_differentiable(
                reference_input.detach().clone().requires_grad_(True)
            )
    assert framework_error.value.__cause__ is framework_failure
    assert framework_error.value.operation_identity == (
        "differentiable_vae_checkpoint_execution"
    )
    assert framework_error.value.runtime_reason_identity == (
        "runtime_reported_memory_allocation_failure"
    )

    def forbidden_checkpoint(*_args, **_kwargs):
        raise AssertionError("non-differentiable VAE decode must not use checkpoint")

    monkeypatch.setattr(
        sd35_backend_module,
        "activation_checkpoint",
        forbidden_checkpoint,
    )
    inference_vae = Vae()
    inference_vae.requires_grad_(False)
    inference_image = backend(inference_vae).vae_decode(reference_input.detach())
    assert torch.equal(inference_image, reference_image)
    assert not inference_image.requires_grad
    assert inference_vae.call_count == 1
    assert inference_vae.weight.grad is None


@pytest.mark.parametrize(
    "failing_decoder_boundary",
    tuple(
        sorted(
            sd35_backend_module.DIFFERENTIABLE_VAE_DECODER_OPERATION_IDENTITIES
            - {"differentiable_vae_post_quant_projection"}
        )
    ),
)
def test_differentiable_vae_decoder_localization_reports_bounded_operation(
    failing_decoder_boundary: str,
    tmp_path: Path,
) -> None:
    configuration = load_runtime_configuration()

    class Operation(torch.nn.Module):
        def __init__(self, operation_identity: str) -> None:
            super().__init__()
            self.operation_identity = operation_identity

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if self.operation_identity == failing_decoder_boundary:
                raise RuntimeError("excluded decoder operation detail")
            return torch.sin(value * 0.5)

    class Middle(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resnets = torch.nn.ModuleList(
                (
                    Operation(
                        "differentiable_vae_decoder_middle_input_residual"
                    ),
                    Operation(
                        "differentiable_vae_decoder_middle_output_residual"
                    ),
                )
            )
            self.attentions = torch.nn.ModuleList(
                (Operation("differentiable_vae_decoder_middle_attention"),)
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if (
                failing_decoder_boundary
                == "differentiable_vae_decoder_middle_block_dispatch"
            ):
                raise RuntimeError("excluded middle block detail")
            value = self.resnets[0](value)
            value = self.attentions[0](value)
            return self.resnets[1](value)

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_in = Operation(
                "differentiable_vae_decoder_input_convolution"
            )
            self.mid_block = Middle()
            self.up_blocks = torch.nn.ModuleList(
                Operation(identity)
                for identity in (
                    "differentiable_vae_decoder_lowest_resolution_upsampling",
                    "differentiable_vae_decoder_lower_middle_resolution_upsampling",
                    "differentiable_vae_decoder_upper_middle_resolution_upsampling",
                    "differentiable_vae_decoder_highest_resolution_upsampling",
                )
            )
            self.conv_norm_out = Operation(
                "differentiable_vae_decoder_output_normalization"
            )
            self.conv_act = Operation(
                "differentiable_vae_decoder_output_activation"
            )
            self.conv_out = Operation(
                "differentiable_vae_decoder_output_convolution"
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = self.conv_in(value)
            value = self.mid_block(value)
            for up_block in self.up_blocks:
                value = up_block(value)
            value = self.conv_norm_out(value)
            value = self.conv_act(value)
            return self.conv_out(value)

    class Vae:
        dtype = torch.float32

        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                force_upcast=False,
                use_post_quant_conv=False,
            )
            self.post_quant_conv = None
            self.decoder = Decoder()

        def decode(self, value: torch.Tensor, *, return_dict: bool):
            assert return_dict is True
            if failing_decoder_boundary == "differentiable_vae_decode_entry":
                raise RuntimeError("excluded decode entry detail")
            return types.SimpleNamespace(sample=self.decoder(value))

    class ImageProcessor:
        @staticmethod
        def postprocess(value: torch.Tensor, *, output_type: str) -> torch.Tensor:
            assert output_type == "pt"
            return value

    vae = Vae()
    backend = Sd35PipelineBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token=None,
        prompt="probe",
    )
    backend._configuration = configuration
    backend._device = torch.device("cpu")
    backend._pipeline = types.SimpleNamespace(
        vae=vae,
        image_processor=ImageProcessor(),
    )
    latent = torch.ones((1, 16, 2, 2), dtype=torch.float32, requires_grad=True)

    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError
    ) as failure:
        backend.vae_decode_differentiable(latent)

    assert failure.value.decoder_operation_identity == failing_decoder_boundary
    assert set(
        sd35_backend_module.DIFFERENTIABLE_VAE_DECODER_OPERATION_IDENTITIES
    ) == {
        "differentiable_vae_decode_entry",
        "differentiable_vae_post_quant_projection",
        "differentiable_vae_decoder_input_convolution",
        "differentiable_vae_decoder_middle_block_dispatch",
        "differentiable_vae_decoder_middle_input_residual",
        "differentiable_vae_decoder_middle_attention",
        "differentiable_vae_decoder_middle_output_residual",
        "differentiable_vae_decoder_lowest_resolution_upsampling",
        "differentiable_vae_decoder_lower_middle_resolution_upsampling",
        "differentiable_vae_decoder_upper_middle_resolution_upsampling",
        "differentiable_vae_decoder_highest_resolution_upsampling",
        "differentiable_vae_decoder_output_normalization",
        "differentiable_vae_decoder_output_activation",
        "differentiable_vae_decoder_output_convolution",
    }
    for module in vae.decoder.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


def test_differentiable_vae_decoder_localization_preserves_values_gradients_and_absent_projection(
    tmp_path: Path,
) -> None:
    configuration = load_runtime_configuration()

    class InputConvolution(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fail = False

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if self.fail:
                raise RuntimeError("excluded input convolution detail")
            return torch.tanh(value[:, :3] * 0.75)

    class Decoder(_DecoderLocalizationPassThrough):
        def __init__(self) -> None:
            super().__init__()
            self.conv_in = InputConvolution()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.conv_in(value)

    class Vae:
        dtype = torch.float32
        config = types.SimpleNamespace(
            force_upcast=False,
            use_post_quant_conv=False,
        )
        post_quant_conv = None

        def __init__(self) -> None:
            self.decoder = Decoder()

        def decode(self, value: torch.Tensor, *, return_dict: bool):
            assert return_dict is True
            return types.SimpleNamespace(sample=self.decoder(value))

    class ImageProcessor:
        @staticmethod
        def postprocess(value: torch.Tensor, *, output_type: str) -> torch.Tensor:
            assert output_type == "pt"
            return value

    vae = Vae()
    direct_input = torch.linspace(-1.0, 1.0, 64).reshape(1, 16, 2, 2)
    direct_input.requires_grad_(True)
    direct_image = vae.decode(direct_input, return_dict=True).sample
    direct_gradient = torch.autograd.grad(direct_image.square().sum(), direct_input)[0]
    backend = Sd35PipelineBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token=None,
        prompt="probe",
    )
    backend._configuration = configuration
    backend._device = torch.device("cpu")
    backend._pipeline = types.SimpleNamespace(
        vae=vae,
        image_processor=ImageProcessor(),
    )
    tracked_input = direct_input.detach().clone().requires_grad_(True)

    tracked_image = backend.vae_decode_differentiable(tracked_input)
    tracked_gradient = torch.autograd.grad(
        tracked_image.square().sum(), tracked_input
    )[0]

    assert torch.equal(tracked_image, direct_image)
    assert torch.equal(tracked_gradient, direct_gradient)
    assert not vae.decoder._forward_pre_hooks
    assert not vae.decoder._forward_hooks

    vae.decoder.conv_in.fail = True
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError
    ) as absent_projection_failure:
        backend.vae_decode_differentiable(
            direct_input.detach().clone().requires_grad_(True)
        )
    absent_projection_decoder_boundary = (
        absent_projection_failure.value.decoder_operation_identity
    )
    assert (
        absent_projection_decoder_boundary
        == "differentiable_vae_decoder_input_convolution"
    )


@pytest.mark.parametrize(
    "structure_drift",
    (
        "post_quant_configuration_enabled",
        "post_quant_module_present",
        "decoder_missing",
        "input_convolution_missing",
        "middle_block_missing",
        "middle_residual_missing",
        "middle_attention_missing",
        "middle_attention_extra",
        "upsampling_block_missing",
        "upsampling_block_extra",
        "output_normalization_missing",
        "output_activation_missing",
        "output_convolution_missing",
    ),
)
def test_differentiable_vae_decoder_localization_rejects_structure_drift(
    structure_drift: str,
    tmp_path: Path,
) -> None:
    configuration = load_runtime_configuration()

    class Vae:
        dtype = torch.float32

        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                force_upcast=False,
                use_post_quant_conv=False,
            )
            self.post_quant_conv = None
            self.decoder = _DecoderLocalizationPassThrough()

        def decode(self, value: torch.Tensor, *, return_dict: bool):
            assert return_dict is True
            return types.SimpleNamespace(sample=self.decoder(value))

    vae = Vae()
    if structure_drift == "post_quant_configuration_enabled":
        vae.config.use_post_quant_conv = True
    elif structure_drift == "post_quant_module_present":
        vae.post_quant_conv = torch.nn.Identity()
    elif structure_drift == "decoder_missing":
        vae.decoder = None
    elif structure_drift == "input_convolution_missing":
        vae.decoder.conv_in = None
    elif structure_drift == "middle_block_missing":
        vae.decoder.mid_block = None
    elif structure_drift == "middle_residual_missing":
        vae.decoder.mid_block.resnets = torch.nn.ModuleList(
            (torch.nn.Identity(),)
        )
    elif structure_drift == "middle_attention_missing":
        vae.decoder.mid_block.attentions = torch.nn.ModuleList()
    elif structure_drift == "middle_attention_extra":
        vae.decoder.mid_block.attentions = torch.nn.ModuleList(
            (torch.nn.Identity(), torch.nn.Identity())
        )
    elif structure_drift == "upsampling_block_missing":
        vae.decoder.up_blocks = torch.nn.ModuleList(
            tuple(vae.decoder.up_blocks)[:-1]
        )
    elif structure_drift == "upsampling_block_extra":
        vae.decoder.up_blocks.append(torch.nn.Identity())
    elif structure_drift == "output_normalization_missing":
        vae.decoder.conv_norm_out = None
    elif structure_drift == "output_activation_missing":
        vae.decoder.conv_act = None
    else:
        vae.decoder.conv_out = None

    backend = Sd35PipelineBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token=None,
        prompt="probe",
    )
    backend._configuration = configuration
    backend._device = torch.device("cpu")
    backend._pipeline = types.SimpleNamespace(
        vae=vae,
        image_processor=types.SimpleNamespace(postprocess=lambda value, **_kwargs: value),
    )

    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeInitialDecodeForwardError
    ) as failure:
        backend.vae_decode_differentiable(
            torch.ones(
                (1, 16, 2, 2),
                dtype=torch.float32,
                requires_grad=True,
            )
        )

    assert type(failure.value.__cause__) is sd35_backend_module.Sd35BackendError
    structure_drift_decoder_boundary = failure.value.decoder_operation_identity
    assert structure_drift_decoder_boundary == "differentiable_vae_decode_entry"


def test_differentiable_vae_decoder_localization_tracks_recomputed_operation(
    tmp_path: Path,
) -> None:
    configuration = load_runtime_configuration()

    class RecomputedAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.call_count += 1
            if self.call_count == 2:
                raise RuntimeError("excluded recomputed attention detail")
            return torch.sin(value)

    class Middle(_DecoderLocalizationMiddleBlock):
        def __init__(self) -> None:
            super().__init__()
            self.attentions = torch.nn.ModuleList((RecomputedAttention(),))

    class Decoder(_DecoderLocalizationPassThrough):
        def __init__(self) -> None:
            super().__init__()
            self.mid_block = Middle()

    class Vae:
        dtype = torch.float32
        config = types.SimpleNamespace(
            force_upcast=False,
            use_post_quant_conv=False,
        )
        post_quant_conv = None

        def __init__(self) -> None:
            self.decoder = Decoder()

        def decode(self, value: torch.Tensor, *, return_dict: bool):
            assert return_dict is True
            return types.SimpleNamespace(sample=self.decoder(value))

    class ImageProcessor:
        @staticmethod
        def postprocess(value: torch.Tensor, *, output_type: str) -> torch.Tensor:
            assert output_type == "pt"
            return value

    vae = Vae()
    backend = Sd35PipelineBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token=None,
        prompt="probe",
    )
    backend._configuration = configuration
    backend._device = torch.device("cpu")
    backend._pipeline = types.SimpleNamespace(
        vae=vae,
        image_processor=ImageProcessor(),
    )
    latent = torch.ones((1, 16, 2, 2), dtype=torch.float32, requires_grad=True)

    image = backend.vae_decode_differentiable(latent)
    with pytest.raises(
        sd35_backend_module.Sd35BackendDifferentiableVaeCheckpointRecomputationError
    ) as failure:
        torch.autograd.grad(image.square().sum(), latent)

    recomputed_decoder_boundary = failure.value.decoder_operation_identity
    assert (
        recomputed_decoder_boundary
        == "differentiable_vae_decoder_middle_attention"
    )
    assert vae.decoder.mid_block.attentions[0].call_count == 2
    for module in vae.decoder.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


@pytest.mark.parametrize(
    ("cache_relative", "persistent_relative"),
    (
        ("shared", "shared"),
        ("persistent/cache", "persistent"),
        ("cache", "cache/persistent"),
    ),
)
def test_sd35_backend_rejects_equal_or_nested_storage_roots(
    tmp_path: Path,
    cache_relative: str,
    persistent_relative: str,
) -> None:
    with pytest.raises(Sd35BackendError, match="bidirectionally disjoint"):
        Sd35PipelineBackend(
            cache_root=tmp_path / cache_relative,
            persistent_root=tmp_path / persistent_relative,
            hf_token=None,
            prompt="probe",
        )


def test_sd35_backend_preparation_binds_frozen_identity(monkeypatch, tmp_path: Path) -> None:
    operation_grad_modes: dict[str, list[bool]] = {
        "detection_noise": [],
        "image_postprocess": [],
        "image_preprocess": [],
        "qk_transformer": [],
        "vae_decode": [],
        "vae_encode": [],
    }

    class Scheduler:
        __module__ = "diffusers"

        def __init__(self):
            self.config = {"frozen": True}
            self.timesteps = torch.arange(20, dtype=torch.float32)

        @classmethod
        def from_config(cls, _config):
            return cls()

        def set_timesteps(self, count, device):
            assert str(device) == "cpu"
            self.timesteps = torch.arange(count, dtype=torch.float32)

        def scale_noise(self, sample, timestep, noise):
            assert timestep.numel() == 1
            operation_grad_modes["detection_noise"].append(
                torch.is_grad_enabled()
            )
            return sample + noise

    Scheduler.__name__ = "FlowMatchEulerDiscreteScheduler"

    class Posterior:
        def __init__(self, value):
            self.value = value

        def mode(self):
            return self.value

    class Vae(torch.nn.Module):
        config = types.SimpleNamespace(
            scaling_factor=1.5,
            shift_factor=0.25,
            force_upcast=False,
            use_post_quant_conv=False,
        )
        post_quant_conv = None

        def __init__(self) -> None:
            super().__init__()
            self.frozen_parameter = torch.nn.Parameter(torch.ones(()))
            self.decoder = _DecoderLocalizationPassThrough()

        def decode(self, latent, return_dict):
            assert return_dict is True
            operation_grad_modes["vae_decode"].append(torch.is_grad_enabled())
            decoded = self.decoder(latent)
            return types.SimpleNamespace(
                sample=torch_functional.interpolate(
                    decoded[:, :3],
                    size=(512, 512),
                    mode="nearest",
                )
            )

        def encode(self, image, return_dict):
            assert return_dict is True
            operation_grad_modes["vae_encode"].append(torch.is_grad_enabled())
            mode = torch_functional.interpolate(
                image,
                size=(64, 64),
                mode="nearest",
            )
            mode = mode.repeat(1, 6, 1, 1)[:, :16].to(torch.float16)
            return types.SimpleNamespace(latent_dist=Posterior(mode))

    class ImageProcessor:
        def postprocess(self, value, output_type):
            assert output_type == "pt"
            operation_grad_modes["image_postprocess"].append(
                torch.is_grad_enabled()
            )
            return value

        def preprocess(self, value, height, width):
            assert (height, width) == (512, 512)
            operation_grad_modes["image_preprocess"].append(
                torch.is_grad_enabled()
            )
            return value

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = 1
            self.to_q = torch.nn.Identity()
            self.to_k = torch.nn.Identity()
            self.norm_q = None
            self.norm_k = None

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_parameter = torch.nn.Parameter(torch.ones(()))
            self.transformer_blocks = torch.nn.ModuleList(
                Block() for _ in range(24)
            )

        def forward(self, **kwargs):
            assert kwargs["return_dict"] is False
            operation_grad_modes["qk_transformer"].append(
                torch.is_grad_enabled()
            )
            hidden = kwargs["hidden_states"].flatten(2).transpose(1, 2)
            for index in (0, 23):
                attention = self.transformer_blocks[index].attn
                attention.to_q(hidden)
                attention.to_k(hidden)
            return (kwargs["hidden_states"],)

    class Pipeline:
        __module__ = "diffusers"

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            value = cls()
            value.scheduler = Scheduler()
            value.vae = Vae()
            value.transformer = Transformer()
            value.image_processor = ImageProcessor()
            return value

        def to(self, _device):
            return self

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            callback = kwargs["callback_on_step_end"]
            state = {"latents": kwargs["latents"]}
            for index in range(kwargs["num_inference_steps"]):
                state = callback(self, index, torch.tensor(index), state)
            return types.SimpleNamespace(images=state["latents"])

        def encode_prompt(self, **kwargs):
            assert kwargs["do_classifier_free_guidance"] is False
            assert not torch.is_grad_enabled()
            empty = torch.zeros((1, 2, 2), dtype=torch.float16)
            pooled = torch.zeros((1, 2), dtype=torch.float16)
            return empty, None, pooled, None

    Pipeline.__name__ = "StableDiffusion3Pipeline"
    module = types.SimpleNamespace(
        StableDiffusion3Pipeline=Pipeline,
        FlowMatchEulerDiscreteScheduler=Scheduler,
    )
    actual_import_module = sd35_backend_module.importlib.import_module

    def import_runtime_dependency(name, package=None):
        if name == "diffusers":
            return module
        return actual_import_module(name, package)

    monkeypatch.setattr(
        "runtime.sd35_backend.importlib.import_module",
        import_runtime_dependency,
    )
    class CpuTestBackend(Sd35PipelineBackend):
        def prepare(self, configuration, selected_device):
            assert selected_device == "cpu"
            identity = super().prepare(configuration, "cuda:0")
            self._device = torch.device("cpu")
            return replace(identity, selected_device="cpu")

    backend = CpuTestBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token="memory-only",
        prompt="probe",
    )
    configuration = load_runtime_configuration()
    adapter = create_runtime_adapter(backend=backend)
    identity = adapter.initialize(requested_device="cpu")
    assert isinstance(identity, RuntimeSession)
    assert identity.runtime_config_digest == configuration.runtime_config_digest
    assert identity.runtime_backend_name == "diffusers_sd35_pipeline"
    assert identity.callback_index == 18
    assert all(
        not parameter.requires_grad
        for parameter in backend._pipeline.transformer.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in backend._pipeline.vae.parameters()
    )
    latent = torch.ones((1, 16, 64, 64), dtype=torch.float16)
    callback_indices: list[int] = []
    assert backend.run_generation(
        latent,
        lambda index, value: callback_indices.append(index) or value,
    ).shape == latent.shape
    assert callback_indices == list(range(20))
    assert backend.vae_factors().scaling_factor == 1.5
    image = backend.vae_decode(latent)
    assert backend.vae_encode(image).mode().shape == latent.shape
    schedule = backend.create_detection_schedule(20)
    assert schedule.detection_schedule_index == 7
    assert torch.equal(
        backend.scale_detection_noise(
            latent,
            torch.ones_like(latent),
            schedule.detection_timestep,
        ),
        latent + 1,
    )
    assert backend.attention_module("transformer_blocks.23.attn").heads == 1
    forward = backend.run_qk_detection_forward(
        latent,
        schedule.detection_timestep,
        RuntimeDetectionConditioning(
            prompt="",
            prompt_2="",
            prompt_3="",
            do_classifier_free_guidance=False,
            detection_conditioning_protocol=(
                "sd3_empty_text_triplet_without_cfg"
            ),
        ),
    )
    assert forward.qk_layer_names == configuration.qk_layer_names

    differentiable_latent = torch.linspace(
        -0.5,
        0.5,
        steps=16 * 64 * 64,
        dtype=torch.float32,
    ).reshape(1, 16, 64, 64).requires_grad_(True)
    differentiable_image = backend.vae_decode_differentiable(
        differentiable_latent
    )
    differentiable_qk = observe_differentiable_detection_qk(
        backend,
        configuration,
        identity,
        differentiable_image,
    )
    differentiable_objective = sum(
        observation.query.sum() + observation.attention_key.sum()
        for observation in differentiable_qk.qk_layer_observations
    )
    differentiable_gradient = torch.autograd.grad(
        differentiable_objective,
        differentiable_latent,
    )[0]
    assert bool(torch.isfinite(differentiable_gradient).all())
    assert bool(torch.count_nonzero(differentiable_gradient))
    assert all(
        parameter.grad is None
        for parameter in backend._pipeline.transformer.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in backend._pipeline.vae.parameters()
    )
    assert operation_grad_modes == {
        "detection_noise": [False, True],
        "image_postprocess": [False, True],
        "image_preprocess": [False, True],
        "qk_transformer": [False, True],
        "vae_decode": [False, True],
        "vae_encode": [False, True],
    }

    carrier = hf_carrier("delivery-e2e-key", tuple(latent.shape))
    paired = adapter.execute_content_write_and_vae(
        latent,
        lambda baseline: content_embedder(baseline, carrier),
    )
    observed = adapter.observe_detection_qk(paired.watermarked_image)
    assert paired.content_materialization_result.budget_status == "accepted"
    assert tuple(
        item.layer_name for item in observed.qk_layer_observations
    ) == configuration.qk_layer_names
    adapter.close()


def _package_manifest(root: Path, revision: str) -> None:
    configuration = root / "configs/runtime/runtime_sd35_flowmatch.json"
    configuration.parent.mkdir(parents=True)
    lock = [
        {"package_name": "python", "version_specifier": ">=3.12"},
        {"package_name": "diffusers", "version_specifier": "0.38.0"},
        {"package_name": "torch", "version_specifier": "2.11.0"},
        {"package_name": "transformers", "version_specifier": "5.12.1"},
        {"package_name": "accelerate", "version_specifier": "1.14.0"},
        {"package_name": "numpy", "version_specifier": "2.0.2"},
        {"package_name": "Pillow", "version_specifier": "11.3.0"},
        {"package_name": "safetensors", "version_specifier": "0.8.0"},
        {"package_name": "huggingface-hub", "version_specifier": "1.20.1"},
    ]
    configuration.write_text(
        json.dumps({"dependency_lock": lock}),
        encoding="utf-8",
    )
    package_files = (
        root / "README.md",
        root / "main/__init__.py",
        root / "runtime/__init__.py",
        root / "pyproject.toml",
        root / "requirements_runtime_qualification.txt",
        root / "scripts/experiment_execution/__init__.py",
        root / "scripts/experiment_execution/runtime_qualification_runner.py",
    )
    for path in package_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    (root / "requirements_runtime_qualification.txt").write_text(
        "\n".join(
            f"{item['package_name']}=={item['version_specifier']}"
            for item in lock
            if item["package_name"] != "python"
        )
        + "\n",
        encoding="utf-8",
    )
    copied = []
    for path in (configuration, *package_files):
        value = path.read_bytes()
        copied.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        )
    (root / "runtime_execution_manifest.json").write_text(
        json.dumps(
            {
                "package_schema_version": 1,
                "profile_name": "experiment_execution_package",
                "package_ready": True,
                "runtime_candidate_revision": revision,
                "copied_files": copied,
                "excluded_parts": sorted(runner.PACKAGE_EXCLUDED_PARTS),
            }
        ),
        encoding="utf-8",
    )


def _versions() -> dict[str, str]:
    return {
        "python": "3.12.9",
        "diffusers": "0.38.0",
        "torch": "2.11.0",
        "transformers": "5.12.1",
        "accelerate": "1.14.0",
        "numpy": "2.0.2",
        "Pillow": "11.3.0",
        "safetensors": "0.8.0",
        "huggingface-hub": "1.20.1",
    }


def _record(
    key_control: str = "registered",
    *,
    run_id: str = "run-001",
    revision: str = "1" * 40,
    seed: int = 20260728,
    prompt_identity: str = runner.PROMPT_IDENTITY,
    prompt_sha256: str = hashlib.sha256(b"probe").hexdigest(),
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "runtime_candidate_revision": revision,
        "runtime_config_digest": "0" * 64,
        "runtime_backend_name": "synthetic_backend",
        "cuda_available": True,
        "cuda_runtime": "12.8",
        "gpu_name": "Fake GPU",
        "key_control": key_control,
        "key_public_digest": (
            "1" * 64 if key_control == "registered" else "2" * 64
        ),
        "selected_device": "cuda:0",
        "model_id": "model",
        "model_revision": "3" * 40,
        "seed": seed,
        "prompt_identity": prompt_identity,
        "prompt_sha256": prompt_sha256,
        "callback_index": 18,
        "callback_status": "passed",
        "content_relative_l2_nominal": 0.012,
        "content_relative_l2_limit": 0.012,
        "realized_total_l2": 0.5,
        "realized_relative_l2": 0.011,
        "budget_utilization": 0.916,
        "materialization_scale": 1.0,
        "materialization_attempt_count": 1,
        "integrity_status": "passed",
        "budget_status": "accepted",
        "materialization_replay_identity": "4" * 64,
        "paired_base_latent_digest": "5" * 64,
        "vae_scaling_factor_actual": 1.5305,
        "vae_shift_factor_actual": 0.0609,
        "vae_status": "passed",
        "clean_image_sha256": "6" * 64,
        "watermarked_image_sha256": "7" * 64,
        "detection_latent_sha256": "8" * 64,
        "qk_actual_dtype": "float16",
        "qk_status": "passed",
        "qk_layer_names": list(runner.REGISTERED_QK_LAYERS),
        "qk_operator_identities": ["registered_query_operator", "registered_key_operator"],
        "qk_layer_value_digests": [
            {
                "layer_name": runner.REGISTERED_QK_LAYERS[0],
                "query_sha256": "9" * 64,
                "attention_key_sha256": "a" * 64,
            },
            {
                "layer_name": runner.REGISTERED_QK_LAYERS[1],
                "query_sha256": "c" * 64,
                "attention_key_sha256": "d" * 64,
            },
        ],
        "public_noise_domain_digest": "b" * 64,
        "public_noise_values_float32_be_sha256": "b" * 64,
    }


def test_runner_profiles_create_minimal_result_zip(monkeypatch, tmp_path: Path) -> None:
    revision = "1" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    def execute(**kwargs):
        assert kwargs["cache_root"] == ephemeral / "cache"
        assert kwargs["persistent_root"] == persistent
        return _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
        )

    monkeypatch.setattr(runner, "_execute_once", execute)
    ephemeral, persistent, output = _runner_storage(tmp_path, "result")
    result = runner.run_runtime_qualification(
        profile="qualification",
        run_id="run-001",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="qualification-key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "passed"
    assert result["run_id"] == "run-001"
    assert result["result_zip_filename"] == "result.zip"
    assert result["seed"] == 20260728
    assert result["prompt_sha256"] == hashlib.sha256(b"probe").hexdigest()
    assert result["key_controls"] == [
        "registered",
        "registered",
        "negative_identity",
    ]
    assert {
        result["callback_status"],
        result["actual_dtype_status"],
        result["vae_status"],
        result["qk_status"],
        result["determinism_status"],
        result["package_status"],
        result["dependency_status"],
    } == {"passed", "verified"}
    assert result["repetition_count"] == 3
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "environment_summary.json",
            "failures.jsonl",
            "run_summary.json",
            "runtime_checks.jsonl",
        }
        summary = json.loads(archive.read("run_summary.json"))
        environment = json.loads(archive.read("environment_summary.json"))
        records = [
            json.loads(line)
            for line in archive.read("runtime_checks.jsonl").decode().splitlines()
        ]
        assert summary["run_id"] == "run-001"
        assert summary["result_zip_filename"] == "result.zip"
        assert environment["result_schema_version"] == 2
        assert environment["profile"] == "qualification"
        assert environment["run_id"] == summary["run_id"]
        assert (
            environment["runtime_candidate_revision"]
            == summary["runtime_candidate_revision"]
        )
        assert environment["seed"] == summary["seed"]
        assert environment["prompt_identity"] == summary["prompt_identity"]
        assert environment["prompt_sha256"] == summary["prompt_sha256"]
        assert environment["record_digests"] == summary["record_digests"]
        assert environment["key_controls"] == summary["key_controls"]
        assert all(
            record["gpu_name"] == environment["gpu_name"]
            and record["cuda_runtime"] == environment["cuda_runtime"]
            and record["cuda_available"] == environment["cuda_available"]
            for record in records
        )
        assert archive.read("failures.jsonl") == b""


def test_runner_packages_classified_failure(monkeypatch, tmp_path: Path) -> None:
    revision = "2" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)

    def fail(**_kwargs):
        try:
            raise Sd35BackendError("CUDA out of memory")
        except Sd35BackendError as cause:
            raise RuntimeAdapterError("wrapped") from cause

    monkeypatch.setattr(runner, "_execute_once", fail)
    ephemeral, persistent, output = _runner_storage(tmp_path, "failure")
    result = runner.run_runtime_qualification(
        profile="smoke",
        run_id="run-002",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="qualification-key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "failed"
    with zipfile.ZipFile(output) as archive:
        failures = [
            json.loads(line)
            for line in archive.read("failures.jsonl").decode().splitlines()
        ]
    assert failures[0]["failure_class"] == "resource_failure"


def test_runner_packages_preflight_manifest_failure(tmp_path: Path) -> None:
    ephemeral, persistent, output = _runner_storage(
        tmp_path,
        "preflight-failure",
    )
    result = runner.run_runtime_qualification(
        profile="smoke",
        run_id="run-003",
        package_root=tmp_path / "missing-package",
        runtime_candidate_revision="3" * 40,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="qualification-key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "failed"
    assert output.is_file()


@pytest.mark.parametrize(
    ("ephemeral_relative", "persistent_relative"),
    (
        ("shared", "shared"),
        ("persistent/ephemeral", "persistent"),
        ("ephemeral", "ephemeral/persistent"),
    ),
)
def test_runner_rejects_equal_or_nested_storage_roots(
    tmp_path: Path,
    ephemeral_relative: str,
    persistent_relative: str,
) -> None:
    ephemeral = tmp_path / ephemeral_relative
    persistent = tmp_path / persistent_relative
    with pytest.raises(
        runner.QualificationRunnerError,
        match="bidirectionally disjoint",
    ):
        runner.run_runtime_qualification(
            profile="smoke",
            run_id="storage-overlap",
            package_root=tmp_path / "package",
            runtime_candidate_revision="a" * 40,
            result_zip=ephemeral / "result.zip",
            ephemeral_root=ephemeral,
            persistent_root=persistent,
            hf_token=None,
            root_key="key",
            prompt="probe",
        )


def test_runner_rejects_result_outside_ephemeral_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runner.QualificationRunnerError,
        match="strictly within ephemeral_root",
    ):
        runner.run_runtime_qualification(
            profile="smoke",
            run_id="result-outside",
            package_root=tmp_path / "package",
            runtime_candidate_revision="a" * 40,
            result_zip=tmp_path / "outside.zip",
            ephemeral_root=tmp_path / "ephemeral",
            persistent_root=tmp_path / "persistent",
            hf_token=None,
            root_key="key",
            prompt="probe",
        )


@pytest.mark.parametrize(
    ("profile", "source_location", "message"),
    (
        ("replay", None, "replay source is required"),
        ("smoke", "persistent/source.zip", "only allowed for replay"),
        ("qualification", "persistent/source.zip", "only allowed for replay"),
        ("replay", "outside/source.zip", "strictly within persistent_root"),
        ("replay", "persistent", "strictly within persistent_root"),
    ),
)
def test_runner_enforces_profile_specific_replay_source_boundary(
    tmp_path: Path,
    profile: str,
    source_location: str | None,
    message: str,
) -> None:
    ephemeral = tmp_path / "ephemeral"
    persistent = tmp_path / "persistent"
    replay_source = (
        None if source_location is None else tmp_path / source_location
    )
    with pytest.raises(runner.QualificationRunnerError, match=message):
        runner.run_runtime_qualification(
            profile=profile,
            run_id="replay-boundary",
            package_root=tmp_path / "package",
            runtime_candidate_revision="a" * 40,
            result_zip=ephemeral / "result.zip",
            ephemeral_root=ephemeral,
            persistent_root=persistent,
            hf_token=None,
            root_key="key",
            prompt="probe",
            replay_source=replay_source,
        )


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (ContentEmbedderError("hard budget failed"), "budget_failure"),
        (RuntimeContentExecutionError("bitwise replay failed"), "integrity_failure"),
        (RuntimeQkObservationError("missing hook"), "qk_failure"),
        (Sd35BackendError("CUDA out of memory"), "resource_failure"),
    ],
)
def test_failure_classification_follows_adapter_cause(
    cause: BaseException,
    expected: str,
) -> None:
    try:
        raise cause
    except BaseException as inner:
        try:
            raise RuntimeAdapterError("adapter wrapped failure") from inner
        except RuntimeAdapterError as outer:
            assert runner._classify_failure(outer) == expected


def test_failure_classification_follows_implicit_context_chain() -> None:
    try:
        raise RuntimeQkObservationError("registered hook missing")
    except RuntimeQkObservationError:
        try:
            raise RuntimeAdapterError("adapter wrapped without explicit cause")
        except RuntimeAdapterError as outer:
            assert runner._classify_failure(outer) == "qk_failure"


def test_runner_rejects_incomplete_success_record(monkeypatch, tmp_path: Path) -> None:
    revision = "4" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    monkeypatch.setattr(runner, "_execute_once", lambda **_kwargs: {"budget_status": "accepted"})
    ephemeral, persistent, output = _runner_storage(tmp_path, "schema")
    result = runner.run_runtime_qualification(
        profile="smoke",
        run_id="schema-failure",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "failed"
    assert result["failure_classes"] == ["incomplete"]


def test_qualification_classifies_independent_repetition_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    revision = "e" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    calls = 0

    def execute(**kwargs):
        nonlocal calls
        record = _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
        )
        if calls == 1:
            record["gpu_name"] = "Drift GPU"
        calls += 1
        return record

    monkeypatch.setattr(runner, "_execute_once", execute)
    ephemeral, persistent, output = _runner_storage(
        tmp_path,
        "determinism",
    )
    result = runner.run_runtime_qualification(
        profile="qualification",
        run_id="determinism-drift",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "failed"
    assert result["failure_classes"] == ["determinism_failure"]
    assert result["determinism_status"] == "failed"


def test_dependency_lock_drift_fails_closed(tmp_path: Path) -> None:
    revision = "5" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    versions = _versions()
    versions["torch"] = "0.0.0"
    with pytest.raises(runner.QualificationRunnerError, match="dependency lock drifted"):
        runner.verify_dependency_lock(
            package,
            versions,
        )


@pytest.mark.parametrize(
    "actual",
    (
        "2.11.0",
        "2.11.0+cu128",
        "2.11.0+cpu",
        "2.11.0+cu128.ubuntu20.04",
        "2.11.0+CU128_UBUNTU-20.04",
    ),
)
def test_torch_dependency_lock_accepts_only_frozen_public_version_with_valid_local_label(
    tmp_path: Path,
    actual: str,
) -> None:
    revision = "d" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    versions = _versions()
    versions["torch"] = actual

    evidence = runner.verify_dependency_lock(package, versions)

    torch_evidence = next(
        item for item in evidence if item["package_name"] == "torch"
    )
    assert torch_evidence == {
        "package_name": "torch",
        "expected_version": "2.11.0",
        "actual_version": actual,
    }


@pytest.mark.parametrize(
    "actual",
    (
        "2.11.00",
        "2.11.0rc1+cu128",
        "2.11.1+cu128",
        "2.11.0+",
        "2.11.0+cu128..ubuntu",
        "2.11.0+cu128.",
        "2.11.0+.cu128",
        "2.11.0+cu128+ubuntu",
        "2.11.0+cu 128",
        "2.11.0+cuda-β",
    ),
)
def test_torch_dependency_lock_rejects_public_or_local_version_drift(
    tmp_path: Path,
    actual: str,
) -> None:
    revision = "e" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    versions = _versions()
    versions["torch"] = actual

    with pytest.raises(
        runner.QualificationRunnerError,
        match="dependency lock drifted",
    ):
        runner.verify_dependency_lock(package, versions)


def test_non_torch_dependency_lock_rejects_local_version_label(
    tmp_path: Path,
) -> None:
    revision = "9" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    versions = _versions()
    versions["diffusers"] = "0.38.0+cu128"

    with pytest.raises(
        runner.QualificationRunnerError,
        match="dependency lock drifted",
    ):
        runner.verify_dependency_lock(package, versions)


def test_dependency_lock_uses_metadata_for_every_frozen_package(
    monkeypatch,
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    calls: list[str] = []
    versions = _versions()
    versions["torch"] = "2.11.0+cu128"

    def version(name: str) -> str:
        calls.append(name)
        return versions[name]

    monkeypatch.setattr(runner.platform, "python_version", lambda: versions["python"])
    monkeypatch.setattr(runner.importlib.metadata, "version", version)
    evidence = runner.verify_dependency_lock(package)
    assert calls == [
        "diffusers",
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "Pillow",
        "safetensors",
        "huggingface-hub",
    ]
    assert {item["package_name"] for item in evidence} == set(versions)
    assert next(
        item["actual_version"]
        for item in evidence
        if item["package_name"] == "torch"
    ) == "2.11.0+cu128"


def test_requirements_must_exactly_match_complete_dependency_lock(
    tmp_path: Path,
) -> None:
    revision = "b" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    requirements = package / "requirements_runtime_qualification.txt"
    requirements.write_text(
        requirements.read_text(encoding="utf-8")
        + "unregistered-package==1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(
        runner.QualificationRunnerError,
        match="requirements do not exactly match",
    ):
        runner.verify_dependency_lock(package, _versions())


def test_repository_requirements_match_frozen_dependency_lock() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = runner.verify_dependency_lock(root, _versions())
    assert tuple(item["package_name"] for item in evidence) == tuple(
        package_name for package_name, _version in runner.REGISTERED_DEPENDENCY_LOCK
    )


def test_requirements_reject_same_dependency_set_in_wrong_order(
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    requirements = package / "requirements_runtime_qualification.txt"
    lines = requirements.read_text(encoding="utf-8").splitlines()
    reordered = (lines[1], lines[0], *lines[2:])
    assert set(reordered) == set(lines)
    requirements.write_text("\n".join(reordered) + "\n", encoding="utf-8")
    with pytest.raises(
        runner.QualificationRunnerError,
        match="requirements do not exactly match",
    ):
        runner.verify_dependency_lock(package, _versions())


def test_dependency_lock_must_include_every_frozen_entry(
    tmp_path: Path,
) -> None:
    revision = "f" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    configuration = package / "configs/runtime/runtime_sd35_flowmatch.json"
    payload = json.loads(configuration.read_text(encoding="utf-8"))
    payload["dependency_lock"].pop()
    configuration.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        runner.QualificationRunnerError,
        match="frozen complete lock",
    ):
        runner.verify_dependency_lock(package, _versions())


def test_manifest_rejects_extra_and_tampered_files(tmp_path: Path) -> None:
    revision = "6" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    runner.verify_execution_package(package, revision)
    (package / "extra.py").write_text("extra\n")
    with pytest.raises(
        runner.QualificationRunnerError,
        match="unallowlisted|file set",
    ):
        runner.verify_execution_package(package, revision)
    (package / "extra.py").unlink()
    (package / "README.md").write_text("tampered\n")
    with pytest.raises(runner.QualificationRunnerError, match="identity drifted"):
        runner.verify_execution_package(package, revision)


@pytest.mark.parametrize(
    "manifest_mutation",
    (
        lambda manifest: manifest.update(runtime_candidate_revision="f" * 40),
        lambda manifest: manifest.update(package_ready=False),
        lambda manifest: manifest["copied_files"][0].update(path="/absolute.py"),
        lambda manifest: manifest["copied_files"][0].update(path="C:\\escape.py"),
        lambda manifest: manifest["copied_files"][0].update(path=".env"),
        lambda manifest: manifest["copied_files"][0].update(path="unallowlisted.py"),
    ),
)
def test_manifest_rejects_revision_readiness_and_unsafe_paths(
    manifest_mutation,
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    manifest_path = package / "runtime_execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.QualificationRunnerError):
        runner.verify_execution_package(package, revision)


def test_replay_validates_source_then_reruns(monkeypatch, tmp_path: Path) -> None:
    revision = "7" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    calls: list[str] = []

    def execute(**kwargs):
        calls.append(kwargs["key_control"])
        return _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
        )

    monkeypatch.setattr(runner, "_execute_once", execute)
    qualification_ephemeral, persistent, qualification_zip = _runner_storage(
        tmp_path,
        "qualification",
    )
    qualification = runner.run_runtime_qualification(
        profile="qualification",
        run_id="qualification-source",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=qualification_zip,
        ephemeral_root=qualification_ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    replay_source = persistent / "runs" / revision / qualification_zip.name
    replay_source.parent.mkdir(parents=True)
    shutil.copy2(qualification_zip, replay_source)
    calls.clear()
    replay_ephemeral = tmp_path / "replay-ephemeral"
    replay = runner.run_runtime_qualification(
        profile="replay",
        run_id="replay-run",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=replay_ephemeral / "replay.zip",
        ephemeral_root=replay_ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        replay_source=replay_source,
        supplied_dependency_versions=_versions(),
    )
    assert qualification["run_status"] == replay["run_status"] == "passed"
    assert calls == ["registered", "registered", "negative_identity"]
    assert replay["replay_source_record_digests"] == qualification["record_digests"]


def _rewrite_result_zip(
    source: Path,
    target: Path,
    mutations: dict[str, bytes],
) -> None:
    target.parent.mkdir(parents=True)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as output:
        for info in original.infolist():
            output.writestr(
                info,
                mutations.get(info.filename, original.read(info.filename)),
            )


def _qualification_source(
    monkeypatch,
    tmp_path: Path,
    revision: str,
) -> Path:
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)

    def execute(**kwargs):
        return _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
            seed=kwargs["seed"],
            prompt_sha256=hashlib.sha256(
                kwargs["prompt"].encode("utf-8")
            ).hexdigest(),
        )

    monkeypatch.setattr(runner, "_execute_once", execute)
    ephemeral, persistent, source = _runner_storage(tmp_path, "source")
    result = runner.run_runtime_qualification(
        profile="qualification",
        run_id="source-run",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=source,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "passed"
    return source


@pytest.mark.parametrize(
    ("member_name", "mutation", "message"),
    (
        (
            "environment_summary.json",
            lambda value: {**value, "gpu_name": "tampered-gpu"},
            "environment or failures drifted",
        ),
        (
            "failures.jsonl",
            lambda _value: {
                "failure_class": "runtime_failure",
                "exception_type": "Injected",
                "message": "injected",
            },
            "environment or failures drifted",
        ),
    ),
)
def test_replay_rejects_environment_tamper_and_failure_injection(
    monkeypatch,
    tmp_path: Path,
    member_name: str,
    mutation,
    message: str,
) -> None:
    revision = "8" * 40
    source = _qualification_source(monkeypatch, tmp_path, revision)
    with zipfile.ZipFile(source) as archive:
        if member_name.endswith(".json"):
            original = json.loads(archive.read(member_name))
            replacement = (
                json.dumps(mutation(original), sort_keys=True) + "\n"
            ).encode()
        else:
            replacement = (
                json.dumps(mutation(None), sort_keys=True) + "\n"
            ).encode()
    tampered = tmp_path / "tampered" / source.name
    _rewrite_result_zip(source, tampered, {member_name: replacement})
    with pytest.raises(runner.QualificationRunnerError, match=message):
        runner._load_replay_source(
            tampered,
            revision,
            hashlib.sha256(b"probe").hexdigest(),
            20260728,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", 20260729),
        ("prompt_identity", "runtime_qualification_prompt_tampered"),
        ("prompt_sha256", "f" * 64),
    ),
)
def test_replay_rejects_summary_record_request_identity_mismatch(
    monkeypatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    revision = "9" * 40
    source = _qualification_source(monkeypatch, tmp_path, revision)
    with zipfile.ZipFile(source) as archive:
        summary = json.loads(archive.read("run_summary.json"))
        environment = json.loads(archive.read("environment_summary.json"))
        records = [
            json.loads(line)
            for line in archive.read("runtime_checks.jsonl").decode().splitlines()
        ]
    records[0][field] = value
    digests = [runner._record_digest(record) for record in records]
    summary["checks"] = records
    summary["record_digests"] = digests
    environment["record_digests"] = digests
    mutations = {
        "run_summary.json": (
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "environment_summary.json": (
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "runtime_checks.jsonl": "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in records
        ).encode(),
    }
    tampered = tmp_path / field / source.name
    _rewrite_result_zip(source, tampered, mutations)
    with pytest.raises(
        runner.QualificationRunnerError,
        match="required success semantics",
    ):
        runner._load_replay_source(
            tampered,
            revision,
            hashlib.sha256(b"probe").hexdigest(),
            20260728,
        )


def test_replay_rejects_record_bytes_that_drift_from_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    revision = "d" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)

    def execute(**kwargs):
        return _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
        )

    monkeypatch.setattr(runner, "_execute_once", execute)
    source_ephemeral, persistent, source = _runner_storage(
        tmp_path,
        "record-source",
    )
    source_result = runner.run_runtime_qualification(
        profile="qualification",
        run_id="source-run",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=source,
        ephemeral_root=source_ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert source_result["run_status"] == "passed"
    tampered = persistent / "tampered" / source.name
    with zipfile.ZipFile(source) as original:
        record_payload = original.read("runtime_checks.jsonl").replace(
            b'"gpu_name": "Fake GPU"',
            b'"gpu_name": "Drift GPU"',
            1,
        )
    _rewrite_result_zip(
        source,
        tampered,
        {"runtime_checks.jsonl": record_payload},
    )
    replay_ephemeral = tmp_path / "record-replay-ephemeral"
    replay = runner.run_runtime_qualification(
        profile="replay",
        run_id="replay-run",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=replay_ephemeral / "replay.zip",
        ephemeral_root=replay_ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        replay_source=tampered,
        supplied_dependency_versions=_versions(),
    )
    assert replay["run_status"] == "failed"
    assert replay["failure_classes"] == ["incomplete"]


@pytest.mark.parametrize(
    ("result", "expected"),
    (
        ({"run_status": "passed", "failure_classes": []}, 0),
        ({"run_status": "failed", "failure_classes": ["runtime_failure"]}, 1),
        ({"run_status": "failed", "failure_classes": ["incomplete"]}, 2),
    ),
)
def test_cli_exit_code_matches_result_status(
    monkeypatch,
    tmp_path: Path,
    result: dict[str, object],
    expected: int,
) -> None:
    monkeypatch.setattr(
        runner,
        "run_runtime_qualification",
        lambda **_kwargs: result,
    )
    ephemeral = tmp_path / "cli-ephemeral"
    assert runner.main(
        [
            "--run-id",
            "cli-run",
            "--result-zip",
            str(ephemeral / "result.zip"),
            "--ephemeral-root",
            str(ephemeral),
            "--persistent-root",
            str(tmp_path / "cli-persistent"),
        ]
    ) == expected


def test_cli_requires_run_id() -> None:
    with pytest.raises(SystemExit) as exc:
        runner.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "missing_option",
    ("--result-zip", "--ephemeral-root", "--persistent-root"),
)
def test_cli_requires_explicit_storage_arguments(
    tmp_path: Path,
    missing_option: str,
) -> None:
    ephemeral = tmp_path / "cli-required-ephemeral"
    arguments = [
        "--run-id",
        "cli-required",
        "--result-zip",
        str(ephemeral / "result.zip"),
        "--ephemeral-root",
        str(ephemeral),
        "--persistent-root",
        str(tmp_path / "cli-required-persistent"),
    ]
    option_index = arguments.index(missing_option)
    del arguments[option_index : option_index + 2]
    with pytest.raises(SystemExit) as exc:
        runner.main(arguments)
    assert exc.value.code == 2


def test_execute_once_passes_persistent_root_to_backend_factory(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FactoryStop(RuntimeError):
        pass

    def factory(**kwargs):
        captured.update(kwargs)
        raise FactoryStop

    with pytest.raises(FactoryStop):
        runner._execute_once(
            backend_factory=factory,
            cache_root=tmp_path / "cache",
            persistent_root=tmp_path / "persistent",
            hf_token=None,
            root_key="key",
            prompt="probe",
            seed=1,
            key_control="registered",
            run_id="factory-boundary",
            runtime_candidate_revision="a" * 40,
        )
    assert captured["cache_root"] == tmp_path / "cache"
    assert captured["persistent_root"] == tmp_path / "persistent"


def test_delivery_python_sources_have_no_scanned_local_absolute_path() -> None:
    root = Path(__file__).resolve().parents[2]
    scanned_paths = (
        root / "runtime/sd35_backend.py",
        root / "scripts/experiment_execution/runtime_qualification_runner.py",
    )
    local_prefixes = (
        "/home/",
        "/Users/",
        "/mnt/",
        "/content/",
        "/tmp/",
        "/var/",
        "/opt/",
        "/root/",
    )
    for path in scanned_paths:
        source = path.read_text(encoding="utf-8")
        assert not any(prefix in source for prefix in local_prefixes)


def _write_package_fixture(repo: Path) -> None:
    dependency_lock = [
        {"package_name": package_name, "version_specifier": version}
        for package_name, version in runner.REGISTERED_DEPENDENCY_LOCK
    ]
    for relative in (
        "main/__init__.py",
        "runtime/__init__.py",
        "configs/runtime/runtime_sd35_flowmatch.json",
        "scripts/experiment_execution/__init__.py",
        "scripts/experiment_execution/README.md",
        "scripts/experiment_execution/runtime_qualification_runner.py",
        "pyproject.toml",
        "requirements_runtime_qualification.txt",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "configs/runtime/runtime_sd35_flowmatch.json":
            path.write_text(
                json.dumps({"dependency_lock": dependency_lock}) + "\n",
                encoding="utf-8",
            )
        elif relative == "requirements_runtime_qualification.txt":
            path.write_text(
                "\n".join(
                    f"{package_name}=={version}"
                    for package_name, version in runner.REGISTERED_DEPENDENCY_LOCK
                    if package_name != "python"
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text("# fixture\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_package_builder_requires_clean_exact_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_package_fixture(repo)
    (
        repo
        / "scripts/experiment_execution/runtime_qualification_bootstrap.py"
    ).write_text("# package-external bootstrap\n", encoding="utf-8")
    (repo / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=CEG-WM Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = _git(repo, "rev-parse", "HEAD")
    ignored = repo / "runtime/ignored.pyc"
    ignored.write_bytes(b"ignored working-tree cache")
    assert _git(repo, "status", "--porcelain") == ""
    output = tmp_path / "package.zip"
    result = build_runtime_qualification_package(
        root=repo,
        output_zip=output,
        runtime_candidate_revision=revision,
    )
    assert result["package_ready"] is True
    assert result["runtime_candidate_revision"] == revision
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("runtime_execution_manifest.json"))
        assert manifest["package_ready"] is True
        assert "README.md" in archive.namelist()
        assert "runtime/ignored.pyc" not in archive.namelist()
        assert (
            "scripts/experiment_execution/build_runtime_qualification_package.py"
            not in archive.namelist()
        )
        assert (
            "scripts/experiment_execution/runtime_qualification_bootstrap.py"
            not in archive.namelist()
        )
        assert not any(name.startswith(".codex/") for name in archive.namelist())
        for entry in manifest["copied_files"]:
            payload = archive.read(entry["path"])
            assert len(payload) == entry["size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    with pytest.raises(PackageBuildError, match="does not equal HEAD"):
        build_runtime_qualification_package(
            root=repo,
            output_zip=tmp_path / "wrong-revision.zip",
            runtime_candidate_revision="0" * 40,
        )
    (repo / "runtime/__init__.py").write_text("# drift\n")
    with pytest.raises(PackageBuildError, match="clean"):
        build_runtime_qualification_package(
            root=repo,
            output_zip=tmp_path / "dirty.zip",
            runtime_candidate_revision=revision,
        )


def test_package_builder_rejects_wrong_requirement_order_without_output(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_package_fixture(repo)
    requirements = repo / "requirements_runtime_qualification.txt"
    lines = requirements.read_text(encoding="utf-8").splitlines()
    reordered = (lines[1], lines[0], *lines[2:])
    assert set(reordered) == set(lines)
    requirements.write_text("\n".join(reordered) + "\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=CEG-WM Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = _git(repo, "rev-parse", "HEAD")
    output = tmp_path / "delivery" / "package.zip"
    with pytest.raises(
        PackageBuildError,
        match="requirements do not exactly match",
    ):
        build_runtime_qualification_package(
            root=repo,
            output_zip=output,
            runtime_candidate_revision=revision,
        )
    assert not output.exists()
    assert not output.parent.exists()


def test_package_builder_rejects_local_absolute_path_blob(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_package_fixture(repo)
    (repo / "runtime/local_path.py").write_text(
        'PATH = "/home/example/private"\n',
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=CEG-WM Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(PackageBuildError, match="local absolute path"):
        build_runtime_qualification_package(
            root=repo,
            output_zip=tmp_path / "package.zip",
            runtime_candidate_revision=revision,
        )


def test_built_package_unpacks_and_runs_independently(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_package_fixture(repo)
    (repo / "scripts/experiment_execution/runtime_qualification_runner.py").write_bytes(
        (
            source_root
            / "scripts/experiment_execution/runtime_qualification_runner.py"
        ).read_bytes()
    )
    (repo / "main/__init__.py").write_text(
        'PACKAGE_TEST_VALUE = "main"\n',
        encoding="utf-8",
    )
    (repo / "runtime/__init__.py").write_text(
        'PACKAGE_TEST_VALUE = "runtime"\n',
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=CEG-WM Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = _git(repo, "rev-parse", "HEAD")
    package_zip = tmp_path / "package.zip"
    build_runtime_qualification_package(
        root=repo,
        output_zip=package_zip,
        runtime_candidate_revision=revision,
    )
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(package_zip) as archive:
        archive.extractall(unpacked)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import main, runtime; "
                "assert main.PACKAGE_TEST_VALUE == 'main'; "
                "assert runtime.PACKAGE_TEST_VALUE == 'runtime'"
            ),
        ],
        cwd=unpacked,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    invoked = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.experiment_execution.runtime_qualification_runner",
            "--help",
        ],
        cwd=unpacked,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invoked.returncode == 0, invoked.stderr
    assert "--run-id" in invoked.stdout
    assert not list(unpacked.rglob("__pycache__"))

    (unpacked / "main/__init__.py").write_text(
        "raise AssertionError('main imported before package verification')\n",
        encoding="utf-8",
    )
    preimport_ephemeral = tmp_path / "preimport-ephemeral"
    preimport_persistent = tmp_path / "preimport-persistent"
    failure_zip = preimport_ephemeral / "preimport-failure.zip"
    preimport = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts.experiment_execution.runtime_qualification_runner "
                "import run_runtime_qualification; "
                "result = run_runtime_qualification("
                "profile='smoke', run_id='preimport-check', package_root='.', "
                f"runtime_candidate_revision='{revision}', "
                f"result_zip={str(failure_zip)!r}, "
                f"ephemeral_root={str(preimport_ephemeral)!r}, "
                f"persistent_root={str(preimport_persistent)!r}, "
                "hf_token=None, root_key='test-key', prompt='probe', "
                "supplied_dependency_versions={}); "
                "assert result['run_status'] == 'failed'; "
                "assert result['failure_classes'] == ['incomplete']"
            ),
        ],
        cwd=unpacked,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert preimport.returncode == 0, preimport.stderr
    assert failure_zip.is_file()


def test_notebook_is_unique_thin_and_output_free() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_notebook = (
        root / "notebooks/colab/runtime_qualification.ipynb"
    )
    notebooks = sorted((root / "notebooks").rglob("*.ipynb"))
    assert notebooks == sorted(
        [
            runtime_notebook,
            root / "notebooks/colab/development_exploration.ipynb",
            root / "notebooks/colab/experiment_execution.ipynb",
            root / "notebooks/colab/hf_only_detector_directional_validation.ipynb",
            root / "notebooks/colab/hf_transmission_diagnostic.ipynb",
            root / "notebooks/colab/lf_transmission_diagnostic.ipynb",
            root / "notebooks/colab/lf_whitened_directional_validation.ipynb",
            root / "notebooks/colab/lf_whitened_score_screening.ipynb",
            root / "notebooks/colab/qk_synchronization_write_diagnostic.ipynb",
            root / "notebooks/colab/salient_local_lf_mask_write_validation.ipynb",
            root / "notebooks/colab/thirteen_module_mechanism_screening.ipynb",
        ]
    )
    document = json.loads(runtime_notebook.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in document["cells"]
    )
    source_lines = sources.splitlines()
    assert all(cell.get("execution_count") is None for cell in document["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("outputs", []) == [] for cell in document["cells"])
    assert 5 <= len(document["cells"]) <= 7
    assert "runtime_qualification_bootstrap.py" in sources
    assert "HF_TOKEN" in sources
    assert "CEG_WM_ROOT_KEY" in sources
    assert "/content/drive/MyDrive/CEG-WM/runtime_qualification" in sources
    assert source_lines.count('PROFILE = "qualification"') == 1
    assert (
        'EXPECTED_RUNTIME_CANDIDATE_REVISION = '
        '"8b2344756c4c247906ff0d4eab68e46a773e13f5"'
    ) in source_lines
    assert (
        'PACKAGE_ZIP = "/content/drive/MyDrive/CEG-WM/runtime_qualification/'
        'execution_packages/current/ceg_wm_runtime_execution.zip"'
    ) in source_lines
    assert (
        'EXPECTED_PACKAGE_SHA256 = '
        '"8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38"'
    ) in source_lines
    assert "REPLAY_SOURCE = None" in source_lines
    assert "input(" not in sources
    assert "widget" not in sources.lower()
    assert "PACKAGE_DEFAULT" not in sources
    assert 'os.environ.get("PROFILE"' not in sources
    assert 'os.environ.get("PACKAGE_ZIP"' not in sources
    assert 'os.environ.get("EXPECTED_PACKAGE_SHA256"' not in sources
    assert 'os.environ.get("REPLAY_SOURCE"' not in sources
    assert "sidecar" not in sources.lower()
    assert '"--expected-package-sha256"' in sources
    assert '"--persistent-root"' in sources
    assert "completed.returncode" in sources
    assert "completed.returncode not in (0, 1, 2, 3)" in sources
    assert "EXPECTED_BOOTSTRAP_SHA256" in sources
    assert "hashlib.sha256" in sources
    assert 'pathlib.Path(BOOTSTRAP).read_bytes()' in sources
    assert 'open("xb")' in sources
    assert "str(TRUSTED_BOOTSTRAP)" in sources
    assert sources.count("pathlib.Path(BOOTSTRAP).read_bytes()") == 1
    bootstrap_path = (
        root
        / "scripts/experiment_execution/runtime_qualification_bootstrap.py"
    )
    bootstrap_digest = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
    assert f'EXPECTED_BOOTSTRAP_SHA256 = "{bootstrap_digest}"' in sources
    assert sources.index("hashlib.sha256") < sources.index(
        "subprocess.run(command"
    )
    assert sources.index("TRUSTED_BOOTSTRAP =") < sources.index(
        "subprocess.run(command"
    )
    for forbidden in (
        "runtime_qualification_runner",
        "runtime_execution_manifest",
        "package_schema_version",
        "result_schema_version",
        "zipfile",
        "extractall",
        "pip install",
        "content_embedder",
        "hf_carrier",
        "to_q",
        "to_k",
        "tau_actual_budget",
        "from_pretrained",
    ):
        assert forbidden not in sources


def test_notebook_direct_run_snapshot_requires_no_manual_parameters() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook = root / "notebooks/colab/runtime_qualification.ipynb"
    before = hashlib.sha256(notebook.read_bytes()).hexdigest()
    document = json.loads(notebook.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )
    source_lines = sources.splitlines()
    assert source_lines.count('PROFILE = "qualification"') == 1
    assert (
        'EXPECTED_RUNTIME_CANDIDATE_REVISION = '
        '"8b2344756c4c247906ff0d4eab68e46a773e13f5"'
    ) in source_lines
    assert (
        'PACKAGE_ZIP = "/content/drive/MyDrive/CEG-WM/runtime_qualification/'
        'execution_packages/current/ceg_wm_runtime_execution.zip"'
    ) in source_lines
    assert (
        'EXPECTED_PACKAGE_SHA256 = '
        '"8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38"'
    ) in source_lines
    assert "REPLAY_SOURCE = None" in source_lines
    assert "input(" not in sources
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == before


def _notebook_cell_source(marker: str) -> str:
    root = Path(__file__).resolve().parents[2]
    document = json.loads(
        (
            root / "notebooks/colab/runtime_qualification.ipynb"
        ).read_text(encoding="utf-8")
    )
    return next(
        "".join(cell.get("source", []))
        for cell in document["cells"]
        if marker in "".join(cell.get("source", []))
    )


def _install_fake_colab_userdata(monkeypatch) -> None:
    google_module = types.ModuleType("google")
    colab_module = types.ModuleType("google.colab")
    colab_module.userdata = types.SimpleNamespace(get=lambda _name: "memory-only")
    google_module.colab = colab_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)


def test_notebook_bootstrap_digest_mismatch_runs_no_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_colab_userdata(monkeypatch)
    source = tmp_path / "bootstrap.py"
    source.write_bytes(b"untrusted")
    subprocess_calls: list[object] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )
    namespace = {
        "BOOTSTRAP": str(source),
        "EXPECTED_BOOTSTRAP_SHA256": "0" * 64,
        "PROFILE": "qualification",
        "CONTENT_ROOT": str(tmp_path),
    }
    with pytest.raises(RuntimeError, match="bootstrap SHA-256 mismatch"):
        exec(_notebook_cell_source("bootstrap_payload ="), namespace)
    assert subprocess_calls == []
    assert not list(tmp_path.glob("ceg_wm_trusted_bootstrap_*"))


def test_notebook_executes_verified_local_snapshot_after_drive_source_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_colab_userdata(monkeypatch)
    drive_source = tmp_path / "drive_bootstrap.py"
    trusted_bytes = b"print('trusted bootstrap')\n"
    drive_source.write_bytes(trusted_bytes)
    digest = hashlib.sha256(trusted_bytes).hexdigest()

    def gpu_check(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, "Fake GPU, 1 MiB\n", "")

    monkeypatch.setattr(subprocess, "run", gpu_check)
    namespace = {
        "BOOTSTRAP": str(drive_source),
        "EXPECTED_BOOTSTRAP_SHA256": digest,
        "PROFILE": "qualification",
        "CONTENT_ROOT": str(tmp_path),
    }
    exec(_notebook_cell_source("bootstrap_payload ="), namespace)
    trusted_snapshot = Path(namespace["TRUSTED_BOOTSTRAP"])
    assert trusted_snapshot.is_file()
    assert trusted_snapshot.read_bytes() == trusted_bytes
    assert trusted_snapshot.parent != drive_source.parent

    drive_source.write_bytes(b"replacement after verification\n")
    observed_commands: list[tuple[str, ...]] = []

    def bootstrap_call(command, **_kwargs):
        observed_commands.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "artifact_kind": "qualification_result",
                    "profile": "qualification",
                    "run_status": "passed",
                    "runtime_candidate_revision": "8" * 40,
                    "result_zip": "/persistent/result.zip",
                }
            ),
            "",
        )

    monkeypatch.setattr(subprocess, "run", bootstrap_call)
    namespace.update(
        {
            "PACKAGE_ZIP": "/persistent/package.zip",
            "EXPECTED_PACKAGE_SHA256": "1" * 64,
            "EXPECTED_RUNTIME_CANDIDATE_REVISION": "8" * 40,
            "REPLAY_SOURCE": None,
            "DRIVE_ROOT": "/persistent",
        }
    )
    exec(_notebook_cell_source("command = [sys.executable"), namespace)
    assert observed_commands[0][1] == str(trusted_snapshot)
    assert observed_commands[0][1] != str(drive_source)
    assert trusted_snapshot.read_bytes() == trusted_bytes


def _execute_notebook_status(
    monkeypatch,
    payload: dict[str, object],
    *,
    returncode: int,
) -> dict[str, object]:
    def bootstrap_call(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            returncode,
            json.dumps(payload),
            "",
        )

    monkeypatch.setattr(subprocess, "run", bootstrap_call)
    namespace: dict[str, object] = {
        "sys": sys,
        "subprocess": subprocess,
        "json": json,
        "os": os,
        "TRUSTED_BOOTSTRAP": Path("/content/trusted_bootstrap.py"),
        "PROFILE": "qualification",
        "PACKAGE_ZIP": "/persistent/package.zip",
        "EXPECTED_PACKAGE_SHA256": "1" * 64,
        "EXPECTED_RUNTIME_CANDIDATE_REVISION": "8" * 40,
        "EPHEMERAL_ROOT": "/content/ceg_wm_runtime",
        "DRIVE_ROOT": "/persistent",
        "REPLAY_SOURCE": None,
    }
    exec(_notebook_cell_source("command = [sys.executable"), namespace)
    return namespace


def test_notebook_accepts_bound_qualification_result_status(monkeypatch) -> None:
    namespace = _execute_notebook_status(
        monkeypatch,
        {
            "artifact_kind": "qualification_result",
            "profile": "qualification",
            "run_status": "passed",
            "runtime_candidate_revision": "8" * 40,
            "result_zip": "/persistent/result.zip",
        },
        returncode=0,
    )
    assert namespace["artifact_kind"] == "qualification_result"
    assert namespace["status"]["profile"] == "qualification"
    assert namespace["status"]["run_status"] == "passed"


@pytest.mark.parametrize(
    "payload",
    (
        {
            "artifact_kind": "qualification_result",
            "profile": "qualification",
            "run_status": "passed",
            "result_zip": "/persistent/result.zip",
        },
        {
            "artifact_kind": "qualification_result",
            "profile": "qualification",
            "run_status": "passed",
            "runtime_candidate_revision": "9" * 40,
            "result_zip": "/persistent/result.zip",
        },
    ),
    ids=("missing-revision", "revision-drift"),
)
def test_notebook_rejects_unbound_qualification_result(
    monkeypatch,
    payload: dict[str, object],
) -> None:
    with pytest.raises(RuntimeError, match="runtime candidate revision drifted"):
        _execute_notebook_status(monkeypatch, payload, returncode=0)


def test_notebook_preserves_bootstrap_failure_without_revision(
    monkeypatch,
    capsys,
) -> None:
    namespace = _execute_notebook_status(
        monkeypatch,
        {
            "artifact_kind": "bootstrap_failure",
            "profile": "qualification",
            "run_status": "failed",
            "diagnostic_zip": "/persistent/bootstrap_failure.zip",
        },
        returncode=3,
    )
    with pytest.raises(
        RuntimeError,
        match="artifact preserved at /persistent/bootstrap_failure.zip",
    ):
        exec(_notebook_cell_source("artifact_path ="), namespace)
    assert namespace["artifact_path"] == "/persistent/bootstrap_failure.zip"
    assert "/persistent/bootstrap_failure.zip" in capsys.readouterr().out


def test_notebook_rejects_unknown_bootstrap_artifact_kind(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="unknown bootstrap artifact kind"):
        _execute_notebook_status(
            monkeypatch,
            {
                "artifact_kind": "unknown",
                "profile": "qualification",
                "run_status": "failed",
            },
            returncode=3,
        )
