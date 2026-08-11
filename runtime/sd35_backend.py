"""Lazy real-model backend for the frozen SD3.5 runtime candidate.

Imports and model loading deliberately happen only in :meth:`prepare`, so the
default CPU test profile can validate the boundary without diffusers or model
weights.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop

from .backend import (
    GenerationCallback,
    RuntimeBackendError,
    RuntimeBackendIdentity,
    RuntimeDetectionConditioning,
    RuntimeDetectionScheduleStep,
    RuntimeDeviceCapabilities,
    RuntimeGenerationPromptIdentity,
    RuntimeGenerationSuffixContext,
    RuntimeGenerationWithSuffixContextResult,
    RuntimeQkForwardIdentity,
    RuntimeVaeFactors,
    RuntimeVaePosterior,
)
from .configuration import Sd35RuntimeConfiguration


class Sd35BackendError(RuntimeBackendError):
    """The real SD3.5 backend failed closed."""


class Sd35BackendGenerationSuffixTransformerForwardError(Sd35BackendError):
    """The generation-suffix transformer forward failed closed."""


class Sd35BackendGenerationSuffixSchedulerStepError(Sd35BackendError):
    """The generation-suffix scheduler step failed closed."""


class Sd35BackendDifferentiableVaeDecodeError(Sd35BackendError):
    """The differentiable VAE decode failed closed."""


class _Sd35BackendDifferentiableVaeOperationError(
    Sd35BackendDifferentiableVaeDecodeError
):
    """One finite differentiable VAE sub-operation failed closed."""

    operation_identity = ""

    def __init__(
        self,
        *,
        cuda_memory_facts: tuple[tuple[str, int], ...] = (),
    ) -> None:
        super().__init__()
        allowed = {
            "before_allocated_bytes",
            "before_reserved_bytes",
            "before_max_allocated_bytes",
            "before_max_reserved_bytes",
            "after_allocated_bytes",
            "after_reserved_bytes",
            "after_max_allocated_bytes",
            "after_max_reserved_bytes",
            "total_device_bytes",
        }
        facts_are_complete = (
            len(cuda_memory_facts) == len(allowed)
            and {name for name, _value in cuda_memory_facts} == allowed
        )
        if (
            not self.operation_identity
            or (cuda_memory_facts and not facts_are_complete)
            or any(
                type(name) is not str
                or type(value) is not int
                or value < 0
                for name, value in cuda_memory_facts
            )
        ):
            raise Sd35BackendError(
                "differentiable VAE failure resource facts are invalid"
            )
        self.cuda_memory_facts = tuple(sorted(cuda_memory_facts))


class Sd35BackendDifferentiableVaeInputPreparationError(
    _Sd35BackendDifferentiableVaeOperationError
):
    """Differentiable VAE input preparation failed closed."""

    operation_identity = "differentiable_vae_input_preparation"


class Sd35BackendDifferentiableVaeDecodeForwardError(
    _Sd35BackendDifferentiableVaeOperationError
):
    """A differentiable VAE checkpointed decode operation failed closed."""

    operation_identity = "differentiable_vae_decode_forward"

    def __init__(
        self,
        *,
        cuda_memory_facts: tuple[tuple[str, int], ...] = (),
        runtime_reason_identity: str = "unclassified_runtime_failure",
    ) -> None:
        super().__init__(cuda_memory_facts=cuda_memory_facts)
        if runtime_reason_identity not in _VAE_DECODE_RUNTIME_REASON_IDENTITIES:
            raise Sd35BackendError(
                "differentiable VAE runtime reason identity is invalid"
            )
        self.runtime_reason_identity = runtime_reason_identity


class Sd35BackendDifferentiableVaeInitialDecodeForwardError(
    Sd35BackendDifferentiableVaeDecodeForwardError
):
    """The initial differentiable VAE decoder forward failed closed."""

    operation_identity = "differentiable_vae_initial_decode_forward"


class Sd35BackendDifferentiableVaeCheckpointRecomputationError(
    Sd35BackendDifferentiableVaeDecodeForwardError
):
    """The differentiable VAE checkpoint recomputation failed closed."""

    operation_identity = "differentiable_vae_checkpoint_recomputation"


class Sd35BackendDifferentiableVaeCheckpointExecutionError(
    Sd35BackendDifferentiableVaeDecodeForwardError
):
    """The differentiable VAE checkpoint framework execution failed closed."""

    operation_identity = "differentiable_vae_checkpoint_execution"


class Sd35BackendDifferentiableImagePostprocessError(
    _Sd35BackendDifferentiableVaeOperationError
):
    """Differentiable decoded-image postprocessing failed closed."""

    operation_identity = "differentiable_image_postprocess"


class Sd35BackendDifferentiableVaeEncodeError(Sd35BackendError):
    """The differentiable VAE encode failed closed."""


class Sd35BackendDifferentiableDetectionNoiseSchedulingError(Sd35BackendError):
    """The differentiable detection-noise scheduling failed closed."""


class Sd35BackendDifferentiableQkTransformerForwardError(Sd35BackendError):
    """The differentiable image-only Q/K transformer forward failed closed."""


_CUDA_MEMORY_FACT_NAMES = (
    "allocated_bytes",
    "reserved_bytes",
    "max_allocated_bytes",
    "max_reserved_bytes",
)
_VAE_DECODE_RUNTIME_REASON_IDENTITIES = frozenset(
    {
        "runtime_reported_memory_allocation_failure",
        "cuda_kernel_execution_failure",
        "dtype_shape_operator_contract_failure",
        "unclassified_runtime_failure",
    }
)
_RUNTIME_REPORTED_MEMORY_PATTERNS = (
    "out of memory",
    "cannot allocate memory",
    "can't allocate memory",
    "allocation failed",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
)
_CUDA_KERNEL_EXECUTION_PATTERNS = (
    "cuda error",
    "device-side assert",
    "illegal memory access",
    "kernel launch",
    "cudnn_status_execution_failed",
    "cublas_status_execution_failed",
)
_DTYPE_SHAPE_OPERATOR_PATTERNS = (
    "expected scalar type",
    "input type",
    "dtype",
    "shape",
    "size mismatch",
    "sizes of tensors",
    "not implemented for",
    "unsupported",
    "expected all tensors to be on the same device",
    "must have the same dtype",
    "mat1 and mat2",
)


def _vae_decode_runtime_reason_identity(error: BaseException) -> str:
    """Map one runtime failure to a finite identity without retaining its text."""

    resource_types = tuple(
        dict.fromkeys(
            (
                MemoryError,
                getattr(torch, "OutOfMemoryError", MemoryError),
                getattr(torch.cuda, "OutOfMemoryError", MemoryError),
            )
        )
    )
    if isinstance(error, resource_types):
        return "runtime_reported_memory_allocation_failure"
    try:
        normalized = str(error).casefold()
    except Exception:
        return "unclassified_runtime_failure"
    if any(pattern in normalized for pattern in _RUNTIME_REPORTED_MEMORY_PATTERNS):
        return "runtime_reported_memory_allocation_failure"
    if any(pattern in normalized for pattern in _CUDA_KERNEL_EXECUTION_PATTERNS):
        return "cuda_kernel_execution_failure"
    if any(pattern in normalized for pattern in _DTYPE_SHAPE_OPERATOR_PATTERNS):
        return "dtype_shape_operator_contract_failure"
    return "unclassified_runtime_failure"


def _cuda_memory_snapshot(device: torch.device) -> dict[str, int] | None:
    """Return bounded CUDA allocator facts without exposing tensors or paths."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        values = {
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            "total_device_bytes": int(
                torch.cuda.get_device_properties(index).total_memory
            ),
        }
    except Exception:
        return None
    if any(type(value) is not int or value < 0 for value in values.values()):
        return None
    return values


def _cuda_failure_facts(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> tuple[tuple[str, int], ...]:
    if before is None or after is None:
        return ()
    return tuple(
        sorted(
            (
                *((f"before_{name}", before[name]) for name in _CUDA_MEMORY_FACT_NAMES),
                *((f"after_{name}", after[name]) for name in _CUDA_MEMORY_FACT_NAMES),
                ("total_device_bytes", after["total_device_bytes"]),
            )
        )
    )


@dataclass(frozen=True, slots=True)
class _PipelineGenerationSuffixReplayContext:
    """Backend-private tensors needed to replay one captured generation suffix."""

    runtime_config_digest: str
    callback_index: int
    owner_identity: int
    latent_shape: tuple[int, ...]
    latent_dtype: torch.dtype
    selected_device: str
    prompt_identity: RuntimeGenerationPromptIdentity
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    suffix_timesteps: torch.Tensor
    scheduler_snapshot: Any


def _explicit_absolute_root(value: str | Path, field_name: str) -> Path:
    root = Path(value)
    if not root.is_absolute():
        raise Sd35BackendError(f"{field_name} must be an explicit absolute path")
    return root.resolve()


def _roots_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


class Sd35PipelineBackend:
    """Diffusers SD3.5 backend connected to content_write_and_vae and qk_observation."""

    def __init__(
        self,
        *,
        cache_root: str | Path,
        persistent_root: str | Path,
        hf_token: str | None,
        prompt: str,
        negative_prompt: str = "",
    ) -> None:
        root = _explicit_absolute_root(cache_root, "cache_root")
        persistent = _explicit_absolute_root(
            persistent_root,
            "persistent_root",
        )
        if _roots_overlap(root, persistent):
            raise Sd35BackendError(
                "cache_root and persistent_root must be bidirectionally disjoint"
            )
        if not isinstance(hf_token, (str, type(None))):
            raise Sd35BackendError("hf_token must be text or None")
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise Sd35BackendError("generation prompts must be text")
        self._cache_root = root
        self._persistent_root = persistent
        self._hf_token = hf_token
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._configuration: Sd35RuntimeConfiguration | None = None
        self._device: torch.device | None = None
        self._pipeline: Any | None = None
        self._scheduler_type: type[Any] | None = None
        self._detection_scheduler: Any | None = None
        self._generation_running = False
        self._generation_prompt_identity: RuntimeGenerationPromptIdentity | None = None
        self._clear_prompt_after_generation = False
        self._requires_generation_prompt_selection = False

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=torch.cuda.device_count(),
        )

    def prepare(
        self,
        configuration: Sd35RuntimeConfiguration,
        selected_device: str,
    ) -> RuntimeBackendIdentity:
        if self._pipeline is not None:
            raise Sd35BackendError("SD3.5 backend may only be prepared once")
        if type(configuration) is not Sd35RuntimeConfiguration:
            raise Sd35BackendError("configuration must be Sd35RuntimeConfiguration")
        if selected_device != "cuda:0":
            raise Sd35BackendError("real SD3.5 qualification requires cuda:0")
        try:
            diffusers = importlib.import_module("diffusers")
            pipeline_type = getattr(diffusers, "StableDiffusion3Pipeline")
            scheduler_type = getattr(
                diffusers,
                "FlowMatchEulerDiscreteScheduler",
            )
        except (ImportError, AttributeError) as exc:
            raise Sd35BackendError(
                "registered diffusers SD3.5 classes are unavailable"
            ) from exc
        if f"diffusers.{pipeline_type.__name__}" != configuration.pipeline_class:
            raise Sd35BackendError("loaded pipeline class identity drifted")
        if f"diffusers.{scheduler_type.__name__}" != configuration.scheduler_class:
            raise Sd35BackendError("loaded scheduler class identity drifted")
        self._cache_root.mkdir(parents=True, exist_ok=True)
        try:
            pipeline = pipeline_type.from_pretrained(
                configuration.model_id,
                revision=configuration.model_revision,
                torch_dtype=torch.float16,
                token=self._hf_token,
                cache_dir=str(self._cache_root / "huggingface"),
            )
            pipeline = pipeline.to(selected_device)
        except Exception as exc:
            raise Sd35BackendError("SD3.5 model preparation failed") from exc
        if not isinstance(getattr(pipeline, "scheduler", None), scheduler_type):
            raise Sd35BackendError("prepared scheduler class identity drifted")
        if getattr(pipeline, "vae", None) is None or getattr(
            pipeline, "transformer", None
        ) is None:
            raise Sd35BackendError("prepared pipeline lacks VAE or transformer")
        for component_name in ("transformer", "vae"):
            component = getattr(pipeline, component_name)
            if not isinstance(component, torch.nn.Module):
                raise Sd35BackendError(
                    f"prepared {component_name} does not expose frozen parameters"
                )
            try:
                component.requires_grad_(False)
            except Exception as exc:
                raise Sd35BackendError(
                    f"prepared {component_name} parameter freezing failed"
                ) from exc
            if any(parameter.requires_grad for parameter in component.parameters()):
                raise Sd35BackendError(
                    f"prepared {component_name} parameters remain trainable"
                )
        self._configuration = configuration
        self._device = torch.device(selected_device)
        self._pipeline = pipeline
        self._scheduler_type = scheduler_type
        return RuntimeBackendIdentity(
            candidate_id=configuration.candidate_id,
            runtime_config_digest=configuration.runtime_config_digest,
            runtime_backend_name="diffusers_sd35_pipeline",
            selected_device=selected_device,
            model_id=configuration.model_id,
            model_revision=configuration.model_revision,
            pipeline_class=configuration.pipeline_class,
            scheduler_class=configuration.scheduler_class,
            inference_steps=configuration.inference_steps,
            guidance_scale=configuration.guidance_scale,
            image_height=configuration.image_height,
            image_width=configuration.image_width,
            generation_seed_device=configuration.generation_seed_device,
            latent_dtype=configuration.latent_dtype,
            template_dtype=configuration.template_dtype,
            score_dtype=configuration.score_dtype,
            callback_index=configuration.callback_index,
            callback_hold_scheduler_intervals=(
                configuration.callback_hold_scheduler_intervals
            ),
            vae_decode_protocol=configuration.vae_decode_protocol,
            vae_encode_protocol=configuration.vae_encode_protocol,
            vae_scaling_factor_source=configuration.vae_scaling_factor_source,
            vae_shift_factor_source=configuration.vae_shift_factor_source,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_conditioning_protocol=(
                configuration.detection_conditioning_protocol
            ),
            qk_layer_names=configuration.qk_layer_names,
            dependency_lock=configuration.dependency_lock,
        )

    def _prepared(self) -> tuple[Sd35RuntimeConfiguration, torch.device, Any]:
        if (
            self._configuration is None
            or self._device is None
            or self._pipeline is None
        ):
            raise Sd35BackendError("SD3.5 backend is not prepared")
        return self._configuration, self._device, self._pipeline

    def set_generation_prompts(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> RuntimeGenerationPromptIdentity:
        """Bind one hf_only_reference_validation prompt after preparation without reloading the model."""

        self._prepared()
        if self._generation_running:
            raise Sd35BackendError("generation prompts cannot change while running")
        if type(prompt) is not str or not prompt or negative_prompt != "":
            raise Sd35BackendError(
                "hf_only_reference_validation generation requires nonempty prompt and exact empty negative prompt"
            )
        try:
            identity = RuntimeGenerationPromptIdentity.from_prompts(
                prompt,
                negative_prompt,
            )
        except RuntimeBackendError as exc:
            raise Sd35BackendError("generation prompt identity is invalid") from exc
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._generation_prompt_identity = identity
        self._clear_prompt_after_generation = True
        self._requires_generation_prompt_selection = False
        return identity

    def set_development_generation_prompts(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> RuntimeGenerationPromptIdentity:
        """Bind one development cluster prompt for its repeated paired operations."""

        self._prepared()
        if self._generation_running:
            raise Sd35BackendError("generation prompts cannot change while running")
        if type(prompt) is not str or not prompt or negative_prompt != "":
            raise Sd35BackendError(
                "development generation requires nonempty prompt and exact empty negative prompt"
            )
        try:
            identity = RuntimeGenerationPromptIdentity.from_prompts(prompt, negative_prompt)
        except RuntimeBackendError as exc:
            raise Sd35BackendError("generation prompt identity is invalid") from exc
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._generation_prompt_identity = identity
        self._clear_prompt_after_generation = False
        self._requires_generation_prompt_selection = False
        return identity

    def run_generation(
        self,
        initial_latent: torch.Tensor,
        callback: GenerationCallback,
    ) -> torch.Tensor:
        """Run the established generation path without exposing suffix state."""

        terminal, _context = self._run_generation(
            initial_latent,
            callback,
            capture_suffix_context=False,
        )
        return terminal

    def run_generation_with_suffix_context(
        self,
        initial_latent: torch.Tensor,
        callback: GenerationCallback,
    ) -> RuntimeGenerationWithSuffixContextResult:
        """Run generation and retain only the execution-local registered suffix."""

        terminal, context = self._run_generation(
            initial_latent,
            callback,
            capture_suffix_context=True,
        )
        if context is None:
            raise Sd35BackendError("generation suffix context was not captured")
        return RuntimeGenerationWithSuffixContextResult(
            terminal_latent=terminal,
            suffix_context=context,
        )

    def _run_generation(
        self,
        initial_latent: torch.Tensor,
        callback: GenerationCallback,
        *,
        capture_suffix_context: bool,
    ) -> tuple[torch.Tensor, _PipelineGenerationSuffixReplayContext | None]:
        configuration, _device, pipeline = self._prepared()
        if self._generation_running:
            raise Sd35BackendError("overlapping generation is forbidden")
        if (
            self._requires_generation_prompt_selection
            and self._generation_prompt_identity is None
        ):
            raise Sd35BackendError(
                "next hf_only_reference_validation generation requires an explicit per-unit prompt"
            )
        try:
            prompt_identity = self._generation_prompt_identity or (
                RuntimeGenerationPromptIdentity.from_prompts(
                    self._prompt,
                    self._negative_prompt,
                )
            )
        except RuntimeBackendError as exc:
            raise Sd35BackendError("generation prompt snapshot is invalid") from exc
        prompt_snapshot = self._prompt
        negative_prompt_snapshot = self._negative_prompt
        if prompt_identity != RuntimeGenerationPromptIdentity.from_prompts(
            prompt_snapshot,
            negative_prompt_snapshot,
        ):
            raise Sd35BackendError("generation prompt snapshot identity drifted")
        self._generation_running = True
        suffix_context: _PipelineGenerationSuffixReplayContext | None = None

        def on_step_end(
            _pipeline: Any,
            step_index: int,
            _timestep: torch.Tensor,
            callback_kwargs: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            nonlocal suffix_context
            latent = callback_kwargs.get("latents")
            if not isinstance(latent, torch.Tensor):
                raise Sd35BackendError("generation callback did not expose latents")
            callback_kwargs["latents"] = callback(step_index, latent)
            if capture_suffix_context and step_index == configuration.callback_index:
                if suffix_context is not None:
                    raise Sd35BackendError(
                        "generation suffix context was captured more than once"
                    )
                prompt_embeds = callback_kwargs.get("prompt_embeds")
                pooled_prompt_embeds = callback_kwargs.get("pooled_prompt_embeds")
                timesteps = getattr(pipeline.scheduler, "timesteps", None)
                if (
                    not isinstance(prompt_embeds, torch.Tensor)
                    or not isinstance(pooled_prompt_embeds, torch.Tensor)
                    or not isinstance(timesteps, torch.Tensor)
                ):
                    raise Sd35BackendError(
                        "generation suffix conditioning or schedule is unavailable"
                    )
                suffix_timesteps = timesteps[step_index + 1 :].detach().clone()
                if (
                    suffix_timesteps.ndim != 1
                    or int(suffix_timesteps.numel())
                    != configuration.callback_hold_scheduler_intervals
                ):
                    raise Sd35BackendError(
                        "generation suffix interval count drifted"
                    )
                suffix_context = _PipelineGenerationSuffixReplayContext(
                    runtime_config_digest=configuration.runtime_config_digest,
                    callback_index=configuration.callback_index,
                    owner_identity=id(self),
                    latent_shape=tuple(int(size) for size in latent.shape),
                    latent_dtype=latent.dtype,
                    selected_device=str(latent.device),
                    prompt_identity=prompt_identity,
                    prompt_embeds=prompt_embeds.detach().clone(),
                    pooled_prompt_embeds=pooled_prompt_embeds.detach().clone(),
                    suffix_timesteps=suffix_timesteps,
                    scheduler_snapshot=deepcopy(pipeline.scheduler),
                )
            return callback_kwargs

        try:
            with torch.inference_mode():
                output = pipeline(
                    prompt=prompt_snapshot,
                    negative_prompt=negative_prompt_snapshot,
                    latents=initial_latent,
                    num_inference_steps=configuration.inference_steps,
                    guidance_scale=configuration.guidance_scale,
                    height=configuration.image_height,
                    width=configuration.image_width,
                    output_type="latent",
                    return_dict=True,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=(
                        ["latents", "prompt_embeds", "pooled_prompt_embeds"]
                        if capture_suffix_context
                        else ["latents"]
                    ),
                )
        except Exception as exc:
            raise Sd35BackendError("SD3.5 generation failed") from exc
        finally:
            self._generation_running = False
            if self._clear_prompt_after_generation:
                self._prompt = ""
                self._negative_prompt = ""
                self._generation_prompt_identity = None
                self._clear_prompt_after_generation = False
                self._requires_generation_prompt_selection = True
        latent = getattr(output, "images", None)
        if not isinstance(latent, torch.Tensor):
            raise Sd35BackendError("SD3.5 generation did not return a latent tensor")
        if capture_suffix_context and suffix_context is None:
            raise Sd35BackendError("registered generation suffix was not captured")
        if suffix_context is not None:
            captured = suffix_context
            suffix_context = _PipelineGenerationSuffixReplayContext(
                runtime_config_digest=captured.runtime_config_digest,
                callback_index=captured.callback_index,
                owner_identity=captured.owner_identity,
                latent_shape=captured.latent_shape,
                latent_dtype=captured.latent_dtype,
                selected_device=captured.selected_device,
                prompt_identity=captured.prompt_identity,
                prompt_embeds=captured.prompt_embeds.clone(),
                pooled_prompt_embeds=captured.pooled_prompt_embeds.clone(),
                suffix_timesteps=captured.suffix_timesteps.clone(),
                scheduler_snapshot=deepcopy(captured.scheduler_snapshot),
            )
        return latent, suffix_context

    def replay_generation_suffix(
        self,
        callback_latent: torch.Tensor,
        suffix_context: RuntimeGenerationSuffixContext,
        *,
        differentiable: bool,
    ) -> torch.Tensor:
        """Replay the captured scheduler suffix using the original conditioning."""

        configuration, device, pipeline = self._prepared()
        if type(suffix_context) is not _PipelineGenerationSuffixReplayContext:
            raise Sd35BackendError(
                "generation suffix context belongs to another backend"
            )
        context = suffix_context
        if (
            context.owner_identity != id(self)
            or context.runtime_config_digest != configuration.runtime_config_digest
            or context.callback_index != configuration.callback_index
            or context.selected_device != str(device)
            or context.latent_shape
            != tuple(int(size) for size in callback_latent.shape)
            or type(differentiable) is not bool
        ):
            raise Sd35BackendError("generation suffix context identity drifted")
        if (
            not isinstance(callback_latent, torch.Tensor)
            or callback_latent.device != device
            or not bool(torch.isfinite(callback_latent).all())
        ):
            raise Sd35BackendError("generation suffix latent is invalid")
        scheduler = deepcopy(context.scheduler_snapshot)

        def replay() -> torch.Tensor:
            latents = callback_latent.to(dtype=context.latent_dtype)
            prompt_embeds = context.prompt_embeds.to(device=device)
            pooled_prompt_embeds = context.pooled_prompt_embeds.to(device=device)

            def transformer_forward(
                hidden_states: torch.Tensor,
                timestep: torch.Tensor,
            ) -> torch.Tensor:
                nonlocal initial_transformer_failure, transformer_invocation_count
                transformer_invocation_count += 1
                try:
                    return pipeline.transformer(
                        hidden_states=hidden_states,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                except Exception as exc:
                    if differentiable and transformer_invocation_count == 1:
                        initial_transformer_failure = exc
                        return hidden_states
                    raise Sd35BackendGenerationSuffixTransformerForwardError from exc

            for timestep_value in context.suffix_timesteps:
                initial_transformer_failure: Exception | None = None
                transformer_invocation_count = 0
                latent_model_input = torch.cat([latents, latents], dim=0)
                timestep = timestep_value.to(device=device).expand(
                    latent_model_input.shape[0]
                )
                try:
                    if differentiable:
                        with set_checkpoint_early_stop(False):
                            noise_pred = activation_checkpoint(
                                transformer_forward,
                                latent_model_input,
                                timestep,
                                use_reentrant=False,
                                preserve_rng_state=True,
                            )
                        if initial_transformer_failure is not None:
                            raise Sd35BackendGenerationSuffixTransformerForwardError from (
                                initial_transformer_failure
                            )
                    else:
                        noise_pred = transformer_forward(
                            latent_model_input,
                            timestep,
                        )
                except Sd35BackendGenerationSuffixTransformerForwardError:
                    raise
                except Exception as exc:
                    raise Sd35BackendGenerationSuffixTransformerForwardError from exc
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                guided_noise = noise_pred_uncond + configuration.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                try:
                    latents = scheduler.step(
                        guided_noise,
                        timestep_value.to(device=device),
                        latents,
                        return_dict=False,
                    )[0]
                except Exception as exc:
                    raise Sd35BackendGenerationSuffixSchedulerStepError from exc
            return latents

        execution_context = nullcontext() if differentiable else torch.inference_mode()
        try:
            with execution_context:
                terminal = replay()
        except Exception as exc:
            raise Sd35BackendError("generation suffix replay failed") from exc
        if (
            not isinstance(terminal, torch.Tensor)
            or terminal.shape != callback_latent.shape
            or terminal.device != callback_latent.device
            or not bool(torch.isfinite(terminal).all())
        ):
            raise Sd35BackendError(
                "generation suffix replay returned an invalid latent"
            )
        return terminal

    def vae_factors(self) -> RuntimeVaeFactors:
        _configuration, _device, pipeline = self._prepared()
        config = getattr(pipeline.vae, "config", None)
        try:
            return RuntimeVaeFactors(
                scaling_factor=float(config.scaling_factor),
                shift_factor=float(config.shift_factor),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise Sd35BackendError("prepared VAE factors are unavailable") from exc

    @staticmethod
    def _vae_execution_dtype(pipeline: Any) -> torch.dtype | None:
        vae = pipeline.vae
        vae_dtype = getattr(vae, "dtype", None)
        force_upcast = bool(
            getattr(getattr(vae, "config", None), "force_upcast", False)
        )
        if force_upcast and vae_dtype is torch.float16:
            try:
                vae.to(dtype=torch.float32)
            except Exception as exc:
                raise Sd35BackendError("prepared VAE upcast failed") from exc
            vae_dtype = getattr(vae, "dtype", torch.float32)
        return vae_dtype if isinstance(vae_dtype, torch.dtype) else None

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        _configuration, device, pipeline = self._prepared()
        try:
            with torch.inference_mode():
                vae_dtype = self._vae_execution_dtype(pipeline)
                decode_input = latent.to(
                    device=device,
                    dtype=vae_dtype or latent.dtype,
                )
                decoded = pipeline.vae.decode(
                    decode_input,
                    return_dict=True,
                ).sample
                image = pipeline.image_processor.postprocess(
                    decoded,
                    output_type="pt",
                )
        except Exception as exc:
            raise Sd35BackendError("prepared VAE decode failed") from exc
        if not isinstance(image, torch.Tensor):
            raise Sd35BackendError("prepared VAE decode returned a non-tensor")
        return image

    def vae_decode_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode through the prepared VAE without disabling autograd."""

        _configuration, device, pipeline = self._prepared()
        before = _cuda_memory_snapshot(device)
        try:
            vae_dtype = self._vae_execution_dtype(pipeline)
            decode_input = latent.to(
                device=device,
                dtype=vae_dtype or latent.dtype,
            )
        except Exception as exc:
            raise Sd35BackendDifferentiableVaeInputPreparationError(
                cuda_memory_facts=_cuda_failure_facts(
                    before,
                    _cuda_memory_snapshot(device),
                )
            ) from exc
        before = _cuda_memory_snapshot(device)

        initial_decode_failure: Exception | None = None
        decode_invocation_count = 0

        def decode_forward(value: torch.Tensor) -> torch.Tensor:
            nonlocal initial_decode_failure, decode_invocation_count
            decode_invocation_count += 1
            try:
                return pipeline.vae.decode(
                    value,
                    return_dict=True,
                ).sample
            except Exception as exc:
                if decode_invocation_count == 1:
                    initial_decode_failure = exc
                    return value
                raise Sd35BackendDifferentiableVaeCheckpointRecomputationError(
                    cuda_memory_facts=_cuda_failure_facts(
                        before,
                        _cuda_memory_snapshot(device),
                    ),
                    runtime_reason_identity=_vae_decode_runtime_reason_identity(exc),
                ) from exc

        try:
            with set_checkpoint_early_stop(False):
                decoded = activation_checkpoint(
                    decode_forward,
                    decode_input,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            if initial_decode_failure is not None:
                raise Sd35BackendDifferentiableVaeInitialDecodeForwardError(
                    cuda_memory_facts=_cuda_failure_facts(
                        before,
                        _cuda_memory_snapshot(device),
                    ),
                    runtime_reason_identity=_vae_decode_runtime_reason_identity(
                        initial_decode_failure
                    ),
                ) from initial_decode_failure
        except (
            Sd35BackendDifferentiableVaeInitialDecodeForwardError,
            Sd35BackendDifferentiableVaeCheckpointRecomputationError,
            Sd35BackendDifferentiableVaeCheckpointExecutionError,
        ):
            raise
        except Exception as exc:
            raise Sd35BackendDifferentiableVaeCheckpointExecutionError(
                cuda_memory_facts=_cuda_failure_facts(
                    before,
                    _cuda_memory_snapshot(device),
                ),
                runtime_reason_identity=_vae_decode_runtime_reason_identity(exc),
            ) from exc
        before = _cuda_memory_snapshot(device)
        try:
            image = pipeline.image_processor.postprocess(decoded, output_type="pt")
        except Exception as exc:
            raise Sd35BackendDifferentiableImagePostprocessError(
                cuda_memory_facts=_cuda_failure_facts(
                    before,
                    _cuda_memory_snapshot(device),
                )
            ) from exc
        if not isinstance(image, torch.Tensor) or not bool(
            torch.isfinite(image).all()
        ):
            raise Sd35BackendError(
                "differentiable VAE decode returned an invalid image"
            )
        return image

    def vae_encode(self, image: torch.Tensor) -> RuntimeVaePosterior:
        configuration, device, pipeline = self._prepared()
        try:
            with torch.inference_mode():
                vae_dtype = self._vae_execution_dtype(pipeline)
                prepared_image = pipeline.image_processor.preprocess(
                    image,
                    height=configuration.image_height,
                    width=configuration.image_width,
                ).to(
                    device=device,
                    dtype=vae_dtype or image.dtype,
                )
                posterior = pipeline.vae.encode(
                    prepared_image,
                    return_dict=True,
                ).latent_dist
        except Exception as exc:
            raise Sd35BackendError("prepared VAE posterior encode failed") from exc
        if not isinstance(posterior, RuntimeVaePosterior):
            raise Sd35BackendError("prepared VAE did not expose posterior mode()")
        return posterior

    def vae_encode_differentiable(
        self,
        image: torch.Tensor,
    ) -> RuntimeVaePosterior:
        """Encode through the prepared VAE without disabling autograd."""

        configuration, device, pipeline = self._prepared()
        try:
            vae_dtype = self._vae_execution_dtype(pipeline)
            prepared_image = pipeline.image_processor.preprocess(
                image,
                height=configuration.image_height,
                width=configuration.image_width,
            ).to(device=device, dtype=vae_dtype or image.dtype)
            posterior = pipeline.vae.encode(
                prepared_image,
                return_dict=True,
            ).latent_dist
        except Exception as exc:
            raise Sd35BackendDifferentiableVaeEncodeError from exc
        if not isinstance(posterior, RuntimeVaePosterior):
            raise Sd35BackendError("differentiable VAE did not expose posterior mode()")
        return posterior

    def create_detection_schedule(
        self,
        inference_steps: int,
    ) -> RuntimeDetectionScheduleStep:
        configuration, device, pipeline = self._prepared()
        if inference_steps != configuration.inference_steps:
            raise Sd35BackendError("detection inference-step identity drifted")
        assert self._scheduler_type is not None
        try:
            scheduler = self._scheduler_type.from_config(
                pipeline.scheduler.config
            )
            scheduler.set_timesteps(inference_steps, device=device)
            timestep = scheduler.timesteps[
                configuration.detection_schedule_index
            ].reshape(1)
        except Exception as exc:
            raise Sd35BackendError("detection schedule creation failed") from exc
        self._detection_scheduler = scheduler
        return RuntimeDetectionScheduleStep(
            scheduler_class=configuration.scheduler_class,
            inference_steps=inference_steps,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_timestep=timestep,
        )

    def scale_detection_noise(
        self,
        detection_latent: torch.Tensor,
        public_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self._detection_scheduler is None:
            raise Sd35BackendError("detection schedule was not established")
        try:
            with torch.inference_mode():
                return self._detection_scheduler.scale_noise(
                    detection_latent,
                    timestep,
                    public_noise,
                )
        except Exception as exc:
            raise Sd35BackendError("detection scheduler scale_noise failed") from exc

    def scale_detection_noise_differentiable(
        self,
        detection_latent: torch.Tensor,
        public_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the established detection scheduler with autograd enabled."""

        if self._detection_scheduler is None:
            raise Sd35BackendError("detection schedule was not established")
        try:
            return self._detection_scheduler.scale_noise(
                detection_latent,
                timestep,
                public_noise,
            )
        except Exception as exc:
            raise Sd35BackendDifferentiableDetectionNoiseSchedulingError from exc

    def attention_module(self, layer_name: str) -> torch.nn.Module:
        _configuration, _device, pipeline = self._prepared()
        value: object = pipeline.transformer
        for part in layer_name.split("."):
            if part.isdecimal():
                value = value[int(part)]  # type: ignore[index]
            else:
                value = getattr(value, part)
        if not isinstance(value, torch.nn.Module):
            raise Sd35BackendError("registered attention path is not a module")
        return value

    def run_qk_detection_forward(
        self,
        noisy_detection_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: RuntimeDetectionConditioning,
    ) -> RuntimeQkForwardIdentity:
        configuration, device, pipeline = self._prepared()
        if (
            conditioning.prompt
            or conditioning.prompt_2
            or conditioning.prompt_3
            or conditioning.do_classifier_free_guidance
        ):
            raise Sd35BackendError("Q/K detection requires empty text without CFG")
        try:
            with torch.inference_mode():
                encoded = pipeline.encode_prompt(
                    prompt="",
                    prompt_2="",
                    prompt_3="",
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                prompt_embeds = encoded[0]
                pooled_prompt_embeds = encoded[2]
                pipeline.transformer(
                    hidden_states=noisy_detection_latent,
                    timestep=timestep.expand(noisy_detection_latent.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )
        except Exception as exc:
            raise Sd35BackendError("image-only Q/K transformer forward failed") from exc
        return RuntimeQkForwardIdentity(
            runtime_config_digest=configuration.runtime_config_digest,
            model_id=configuration.model_id,
            model_revision=configuration.model_revision,
            scheduler_class=configuration.scheduler_class,
            inference_steps=configuration.inference_steps,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_conditioning_protocol=(
                conditioning.detection_conditioning_protocol
            ),
            prompt=conditioning.prompt,
            prompt_2=conditioning.prompt_2,
            prompt_3=conditioning.prompt_3,
            do_classifier_free_guidance=(
                conditioning.do_classifier_free_guidance
            ),
            qk_layer_names=configuration.qk_layer_names,
        )

    def run_qk_detection_forward_differentiable(
        self,
        noisy_detection_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: RuntimeDetectionConditioning,
    ) -> RuntimeQkForwardIdentity:
        """Run the same image-only forward without the blind path's detach boundary."""

        configuration, device, pipeline = self._prepared()
        if (
            conditioning.prompt
            or conditioning.prompt_2
            or conditioning.prompt_3
            or conditioning.do_classifier_free_guidance
        ):
            raise Sd35BackendError("Q/K detection requires empty text without CFG")
        try:
            with torch.inference_mode():
                encoded = pipeline.encode_prompt(
                    prompt="",
                    prompt_2="",
                    prompt_3="",
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                prompt_embeds = encoded[0].detach().clone()
                pooled_prompt_embeds = encoded[2].detach().clone()
        except Exception as exc:
            raise Sd35BackendError(
                "differentiable Q/K conditioning failed"
            ) from exc
        try:
            pipeline.transformer(
                hidden_states=noisy_detection_latent,
                timestep=timestep.expand(noisy_detection_latent.shape[0]),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )
        except Exception as exc:
            raise Sd35BackendDifferentiableQkTransformerForwardError from exc
        return RuntimeQkForwardIdentity(
            runtime_config_digest=configuration.runtime_config_digest,
            model_id=configuration.model_id,
            model_revision=configuration.model_revision,
            scheduler_class=configuration.scheduler_class,
            inference_steps=configuration.inference_steps,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_conditioning_protocol=(
                conditioning.detection_conditioning_protocol
            ),
            prompt=conditioning.prompt,
            prompt_2=conditioning.prompt_2,
            prompt_3=conditioning.prompt_3,
            do_classifier_free_guidance=conditioning.do_classifier_free_guidance,
            qk_layer_names=configuration.qk_layer_names,
        )

    def close(self) -> None:
        if self._generation_running:
            raise Sd35BackendError("backend cannot close during generation")
        self._detection_scheduler = None
        self._pipeline = None
        self._configuration = None
        self._device = None
        self._prompt = ""
        self._negative_prompt = ""
        self._generation_prompt_identity = None
        self._clear_prompt_after_generation = False
        self._requires_generation_prompt_selection = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
