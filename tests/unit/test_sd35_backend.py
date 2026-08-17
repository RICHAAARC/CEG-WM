"""Focused CPU coverage for the SD3.5 runtime backend boundary."""

from __future__ import annotations

import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional

from main import content_embedder, hf_carrier
from runtime import (
    RuntimeDetectionConditioning,
    RuntimeGenerationPromptIdentity,
    RuntimeSession,
    Sd35BackendError,
    Sd35PipelineBackend,
    create_runtime_adapter,
    load_runtime_configuration,
    observe_differentiable_detection_qk,
)
from runtime import sd35_backend as sd35_backend_module


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


def test_sd35_backend_sources_have_no_local_absolute_path() -> None:
    source = (Path(__file__).resolve().parents[2] / "runtime/sd35_backend.py").read_text("utf-8")
    assert "/home/" not in source
    assert "C:\\\\" not in source
