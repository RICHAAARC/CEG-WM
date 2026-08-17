"""Fail-closed image-only Q/K observation for the frozen SD3.5 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isqrt
from typing import Literal

import torch

from main import QkLayerObservation, derive_public_noise_stream

from .adapter import RuntimeSession
from .backend import (
    RuntimeDetectionConditioning,
    RuntimeDetectionScheduleStep,
    RuntimeDifferentiableQkBackend,
    RuntimeQkBackend,
    RuntimeQkForwardIdentity,
    RuntimeVaeFactors,
    RuntimeVaePosterior,
)
from .configuration import Sd35RuntimeConfiguration
from .content_write import (
    RuntimeContentExecutionError,
    _encode_detection_image,
    _tensor,
)


class RuntimeQkObservationError(RuntimeError):
    """The image-only Q/K path violated its frozen runtime boundary."""


@dataclass(frozen=True, slots=True)
class RuntimeQkObservationResult:
    """Actual projected Q/K tensors and their complete runtime identity."""

    candidate_id: str
    runtime_config_digest: str
    model_id: str
    model_revision: str
    scheduler_class: str
    detection_schedule_index: int
    detection_timestep: float
    detection_conditioning_protocol: str
    public_noise_domain_digest: str
    public_noise_values_float32_be_sha256: str
    qk_actual_dtype: str
    qk_layer_observations: tuple[QkLayerObservation, ...]


@dataclass(frozen=True, slots=True)
class _AttentionBinding:
    layer_name: str
    attention: torch.nn.Module
    to_q: torch.nn.Module
    to_k: torch.nn.Module
    norm_q: torch.nn.Module | None
    norm_k: torch.nn.Module | None
    head_count: int


def _qualified_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _validate_session(
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
) -> None:
    if type(configuration) is not Sd35RuntimeConfiguration:
        raise RuntimeQkObservationError(
            "configuration must be Sd35RuntimeConfiguration"
        )
    if type(session) is not RuntimeSession:
        raise RuntimeQkObservationError("runtime session is invalid")
    expected = {
        "candidate_id": configuration.candidate_id,
        "runtime_config_digest": configuration.runtime_config_digest,
        "scheduler_class": configuration.scheduler_class,
        "inference_steps": configuration.inference_steps,
        "latent_dtype": configuration.latent_dtype,
        "vae_encode_protocol": configuration.vae_encode_protocol,
        "detection_schedule_index": configuration.detection_schedule_index,
        "detection_conditioning_protocol": (
            configuration.detection_conditioning_protocol
        ),
        "qk_layer_names": configuration.qk_layer_names,
    }
    for field, expected_value in expected.items():
        if getattr(session, field) != expected_value:
            raise RuntimeQkObservationError(
                f"runtime session {field} drifted before Q/K observation"
            )


def _conditioning(
    configuration: Sd35RuntimeConfiguration,
) -> RuntimeDetectionConditioning:
    conditioning = RuntimeDetectionConditioning(
        prompt="",
        prompt_2="",
        prompt_3="",
        do_classifier_free_guidance=False,
        detection_conditioning_protocol=(
            configuration.detection_conditioning_protocol
        ),
    )
    if (
        conditioning.prompt
        or conditioning.prompt_2
        or conditioning.prompt_3
        or conditioning.do_classifier_free_guidance
        or conditioning.detection_conditioning_protocol
        != "sd3_empty_text_triplet_without_cfg"
    ):
        raise RuntimeQkObservationError(
            "detection conditioning does not match the frozen image-only protocol"
        )
    return conditioning


def _schedule_step(
    backend: RuntimeQkBackend,
    configuration: Sd35RuntimeConfiguration,
) -> RuntimeDetectionScheduleStep:
    try:
        step = backend.create_detection_schedule(configuration.inference_steps)
    except Exception as exc:
        raise RuntimeQkObservationError(
            "backend failed to establish the detection schedule"
        ) from exc
    if type(step) is not RuntimeDetectionScheduleStep:
        raise RuntimeQkObservationError(
            "backend returned an invalid detection schedule step"
        )
    if (
        step.scheduler_class != configuration.scheduler_class
        or step.inference_steps != configuration.inference_steps
        or step.detection_schedule_index
        != configuration.detection_schedule_index
    ):
        raise RuntimeQkObservationError("detection schedule identity drifted")
    timestep = step.detection_timestep
    if (
        not isinstance(timestep, torch.Tensor)
        or timestep.numel() != 1
        or not bool(torch.isfinite(timestep.detach().to("cpu")).all())
    ):
        raise RuntimeQkObservationError(
            "detection schedule timestep must be one finite tensor scalar"
        )
    return step


def _attention_binding(
    backend: RuntimeQkBackend,
    layer_name: str,
) -> _AttentionBinding:
    try:
        attention = backend.attention_module(layer_name)
    except Exception as exc:
        raise RuntimeQkObservationError(
            f"registered Q/K layer is unavailable: {layer_name}"
        ) from exc
    if not isinstance(attention, torch.nn.Module):
        raise RuntimeQkObservationError(
            f"registered Q/K layer is not an attention module: {layer_name}"
        )
    to_q = getattr(attention, "to_q", None)
    to_k = getattr(attention, "to_k", None)
    if not isinstance(to_q, torch.nn.Module) or not isinstance(
        to_k, torch.nn.Module
    ):
        raise RuntimeQkObservationError(
            f"registered attention layer lacks real to_q/to_k modules: {layer_name}"
        )
    if to_q is to_k:
        raise RuntimeQkObservationError(
            f"registered attention layer aliases to_q and to_k: {layer_name}"
        )
    head_count = getattr(attention, "heads", None)
    if type(head_count) is not int or head_count <= 0:
        raise RuntimeQkObservationError(
            f"registered attention layer has invalid head layout: {layer_name}"
        )
    norm_q = getattr(attention, "norm_q", None)
    norm_k = getattr(attention, "norm_k", None)
    if norm_q is not None and not isinstance(norm_q, torch.nn.Module):
        raise RuntimeQkObservationError(
            f"registered attention layer has invalid norm_q: {layer_name}"
        )
    if norm_k is not None and not isinstance(norm_k, torch.nn.Module):
        raise RuntimeQkObservationError(
            f"registered attention layer has invalid norm_k: {layer_name}"
        )
    return _AttentionBinding(
        layer_name=layer_name,
        attention=attention,
        to_q=to_q,
        to_k=to_k,
        norm_q=norm_q,
        norm_k=norm_k,
        head_count=head_count,
    )


def _reshape_projection(
    raw: object,
    binding: _AttentionBinding,
    projection_role: Literal["query", "attention_key"],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
    *,
    preserve_gradient: bool = False,
) -> torch.Tensor:
    if not isinstance(raw, torch.Tensor):
        raise RuntimeQkObservationError(
            f"{binding.layer_name} {projection_role} hook did not capture a tensor"
        )
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise RuntimeQkObservationError(
            f"{binding.layer_name} {projection_role} must have [1,tokens,width] shape"
        )
    if raw.dtype is not expected_dtype or raw.device != expected_device:
        raise RuntimeQkObservationError(
            f"{binding.layer_name} {projection_role} actual dtype or device drifted"
        )
    token_count = int(raw.shape[1])
    projected_width = int(raw.shape[2])
    if (
        token_count <= 1
        or projected_width <= 0
        or projected_width % binding.head_count
    ):
        raise RuntimeQkObservationError(
            f"{binding.layer_name} {projection_role} shape mismatches head layout"
        )
    grid_side = isqrt(token_count)
    if grid_side * grid_side != token_count or grid_side < 2:
        raise RuntimeQkObservationError(
            f"{binding.layer_name} {projection_role} image tokens are not square"
        )
    head_width = projected_width // binding.head_count
    shaped = raw.reshape(
        1,
        token_count,
        binding.head_count,
        head_width,
    ).transpose(1, 2)
    normalizer = (
        binding.norm_q if projection_role == "query" else binding.norm_k
    )
    try:
        normalized = shaped if normalizer is None else normalizer(shaped)
    except Exception as exc:
        raise RuntimeQkObservationError(
            f"{binding.layer_name} {projection_role} normalization failed"
        ) from exc
    if (
        not isinstance(normalized, torch.Tensor)
        or normalized.shape
        != (1, binding.head_count, token_count, head_width)
        or normalized.dtype is not expected_dtype
        or normalized.device != expected_device
        or not bool(torch.isfinite(normalized).all())
    ):
        raise RuntimeQkObservationError(
            f"{binding.layer_name} {projection_role} normalized output is invalid"
        )
    if preserve_gradient:
        return normalized[0].clone()
    return normalized[0].detach().clone()


def _operator_identity(
    binding: _AttentionBinding,
    head_width: int,
    actual_dtype: str,
) -> str:
    norm_q_identity = (
        "none" if binding.norm_q is None else _qualified_name(binding.norm_q)
    )
    norm_k_identity = (
        "none" if binding.norm_k is None else _qualified_name(binding.norm_k)
    )
    return "|".join(
        (
            "sd35_real_to_q_to_k",
            f"layer={binding.layer_name}",
            f"attention={_qualified_name(binding.attention)}",
            f"to_q={_qualified_name(binding.to_q)}",
            f"to_k={_qualified_name(binding.to_k)}",
            f"norm_q={norm_q_identity}",
            f"norm_k={norm_k_identity}",
            f"heads={binding.head_count}",
            f"head_width={head_width}",
            f"dtype={actual_dtype}",
            "layout=batch_heads_tokens_head_width",
            "relation_scale=inverse_sqrt_head_width",
        )
    )


def _validate_forward_identity(
    identity: object,
    configuration: Sd35RuntimeConfiguration,
    conditioning: RuntimeDetectionConditioning,
) -> RuntimeQkForwardIdentity:
    if type(identity) is not RuntimeQkForwardIdentity:
        raise RuntimeQkObservationError(
            "backend did not return a Q/K forward identity"
        )
    expected = {
        "runtime_config_digest": configuration.runtime_config_digest,
        "scheduler_class": configuration.scheduler_class,
        "inference_steps": configuration.inference_steps,
        "detection_schedule_index": configuration.detection_schedule_index,
        "detection_conditioning_protocol": (
            conditioning.detection_conditioning_protocol
        ),
        "prompt": conditioning.prompt,
        "prompt_2": conditioning.prompt_2,
        "prompt_3": conditioning.prompt_3,
        "do_classifier_free_guidance": (
            conditioning.do_classifier_free_guidance
        ),
        "qk_layer_names": configuration.qk_layer_names,
    }
    for field, expected_value in expected.items():
        if getattr(identity, field) != expected_value:
            raise RuntimeQkObservationError(
                f"Q/K forward {field} identity drifted"
            )
    return identity


def _observe_detection_qk_core(
    backend: RuntimeQkBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    detection_image: torch.Tensor,
    *,
    differentiable: bool,
) -> RuntimeQkObservationResult:
    """Capture the same image-only Q/K path with an explicit gradient boundary."""

    if not isinstance(backend, RuntimeQkBackend):
        raise RuntimeQkObservationError(
            "prepared backend lacks the qk_observation Q/K execution protocol"
        )
    _validate_session(configuration, session)
    image = _tensor(detection_image, role="detection_image")
    if (
        image.ndim != 4
        or image.shape[0] != 1
        or image.shape[1] != 3
        or tuple(image.shape[2:])
        != (configuration.image_height, configuration.image_width)
    ):
        raise RuntimeQkObservationError(
            "detection_image must be one frozen-resolution RGB image"
        )

    factors = backend.vae_factors()
    if type(factors) is not RuntimeVaeFactors:
        raise RuntimeQkObservationError(
            "backend VAE factors do not match the detection protocol"
        )
    try:
        if differentiable:
            if not isinstance(backend, RuntimeDifferentiableQkBackend):
                raise RuntimeQkObservationError(
                    "prepared backend lacks differentiable Q/K execution"
                )
            posterior = backend.vae_encode_differentiable(image)
            if not isinstance(posterior, RuntimeVaePosterior):
                raise RuntimeQkObservationError(
                    "differentiable VAE encode lacks posterior mode"
                )
            mode = _tensor(
                posterior.mode(),
                role="qk_detection_differentiable_posterior_mode",
            )
            detection_latent = (
                mode.to(dtype=torch.float32) - float(factors.shift_factor)
            ) * float(factors.scaling_factor)
            detection_latent = _tensor(
                detection_latent,
                role="qk_detection_differentiable_latent",
            )
        else:
            detection_latent = _encode_detection_image(
                backend,
                image,
                factors,
                "qk_detection",
            )
    except RuntimeContentExecutionError as exc:
        raise RuntimeQkObservationError(
            "detection image VAE posterior-mode encoding failed"
        ) from exc
    if detection_latent.ndim != 4 or detection_latent.shape[0] != 1:
        raise RuntimeQkObservationError(
            "detection VAE latent must have shape [1,C,H,W]"
        )
    if str(detection_latent.device) != session.selected_device:
        raise RuntimeQkObservationError(
            "detection VAE latent device does not match the runtime session"
        )
    if configuration.latent_dtype != "float16":
        raise RuntimeQkObservationError(
            "registered Q/K path requires the frozen float16 latent dtype"
        )
    detection_latent_actual = detection_latent.to(dtype=torch.float16)
    if not differentiable:
        detection_latent_actual = detection_latent_actual.detach()
    if not bool(torch.isfinite(detection_latent_actual).all()):
        raise RuntimeQkObservationError(
            "detection VAE latent contains non-finite values"
        )

    conditioning = _conditioning(configuration)
    noise_stream = derive_public_noise_stream(
        {
            "candidate_id": "qk_relation_similarity",
            "operator": "public_image_only_qk_detection_noise",
            "responsibility_domain": "public_noise",
            "schedule_index": configuration.detection_schedule_index,
            "conditioning_protocol": configuration.detection_conditioning_protocol,
            "tensor_role": "scheduler_noise",
        },
        tuple(int(size) for size in detection_latent_actual.shape),
    )
    public_noise = torch.tensor(
        noise_stream.values,
        dtype=torch.float32,
        device=detection_latent_actual.device,
    ).reshape(detection_latent_actual.shape).to(dtype=torch.float16)
    if not bool(torch.isfinite(public_noise).all()):
        raise RuntimeQkObservationError("public detection noise is non-finite")

    schedule_step = _schedule_step(backend, configuration)
    try:
        scale_noise = (
            backend.scale_detection_noise_differentiable
            if differentiable
            and isinstance(backend, RuntimeDifferentiableQkBackend)
            else backend.scale_detection_noise
        )
        noisy_latent = _tensor(
            scale_noise(
                detection_latent_actual.clone(),
                public_noise.detach().clone(),
                schedule_step.detection_timestep.detach().clone(),
            ),
            role="noisy_detection_latent",
            shape=detection_latent_actual.shape,
            dtype=torch.float16,
            device=detection_latent_actual.device,
        ).clone()
        if not differentiable:
            noisy_latent = noisy_latent.detach()
    except RuntimeContentExecutionError as exc:
        raise RuntimeQkObservationError(
            "scheduler scale_noise returned an invalid latent"
        ) from exc
    except Exception as exc:
        raise RuntimeQkObservationError(
            "scheduler scale_noise failed"
        ) from exc

    bindings = tuple(
        _attention_binding(backend, layer_name)
        for layer_name in configuration.qk_layer_names
    )
    attention_ids = [id(binding.attention) for binding in bindings]
    projection_ids = [
        id(module)
        for binding in bindings
        for module in (binding.to_q, binding.to_k)
    ]
    if len(set(attention_ids)) != len(attention_ids) or len(
        set(projection_ids)
    ) != len(projection_ids):
        raise RuntimeQkObservationError(
            "registered layers alias attention or projection modules"
        )

    captures: dict[tuple[str, str], torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture_hook(layer_name: str, projection_role: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            capture_key = (layer_name, projection_role)
            if capture_key in captures:
                raise RuntimeQkObservationError(
                    f"{layer_name} {projection_role} was captured more than once"
                )
            if not isinstance(output, torch.Tensor):
                raise RuntimeQkObservationError(
                    f"{layer_name} {projection_role} hook output is not a tensor"
                )
            captures[capture_key] = (
                output.clone() if differentiable else output.detach().clone()
            )

        return hook

    try:
        for binding in bindings:
            handles.append(
                binding.to_q.register_forward_hook(
                    capture_hook(binding.layer_name, "query")
                )
            )
            handles.append(
                binding.to_k.register_forward_hook(
                    capture_hook(binding.layer_name, "attention_key")
                )
            )
        forward_operation = (
            backend.run_qk_detection_forward_differentiable
            if differentiable
            and isinstance(backend, RuntimeDifferentiableQkBackend)
            else backend.run_qk_detection_forward
        )
        forward_identity = forward_operation(
            noisy_latent.clone(),
            schedule_step.detection_timestep.detach().clone(),
            conditioning,
        )
    except RuntimeQkObservationError:
        raise
    except Exception as exc:
        raise RuntimeQkObservationError(
            "image-only Q/K transformer forward failed"
        ) from exc
    finally:
        for handle in handles:
            handle.remove()

    _validate_forward_identity(
        forward_identity,
        configuration,
        conditioning,
    )
    expected_capture_count = len(bindings) * 2
    if len(captures) != expected_capture_count:
        raise RuntimeQkObservationError(
            "one or more registered Q/K projections were not captured exactly once"
        )

    observations: list[QkLayerObservation] = []
    for binding in bindings:
        query = _reshape_projection(
            captures[(binding.layer_name, "query")],
            binding,
            "query",
            torch.float16,
            noisy_latent.device,
            preserve_gradient=differentiable,
        )
        attention_key = _reshape_projection(
            captures[(binding.layer_name, "attention_key")],
            binding,
            "attention_key",
            torch.float16,
            noisy_latent.device,
            preserve_gradient=differentiable,
        )
        if query.shape != attention_key.shape:
            raise RuntimeQkObservationError(
                f"{binding.layer_name} query/key shapes do not match"
            )
        observations.append(
            QkLayerObservation(
                layer_name=binding.layer_name,
                query=query,
                attention_key=attention_key,
                operator_identity=_operator_identity(
                    binding,
                    int(query.shape[2]),
                    configuration.latent_dtype,
                ),
            )
        )

    timestep_value = float(
        schedule_step.detection_timestep.detach().to("cpu").item()
    )
    if not isfinite(timestep_value):
        raise RuntimeQkObservationError("detection timestep is non-finite")
    return RuntimeQkObservationResult(
        candidate_id=configuration.candidate_id,
        runtime_config_digest=configuration.runtime_config_digest,
        model_id=configuration.model_id,
        model_revision=configuration.model_revision,
        scheduler_class=configuration.scheduler_class,
        detection_schedule_index=configuration.detection_schedule_index,
        detection_timestep=timestep_value,
        detection_conditioning_protocol=(
            configuration.detection_conditioning_protocol
        ),
        public_noise_domain_digest=noise_stream.domain_digest,
        public_noise_values_float32_be_sha256=(
            noise_stream.values_float32_be_sha256
        ),
        qk_actual_dtype=configuration.latent_dtype,
        qk_layer_observations=tuple(observations),
    )


def observe_detection_qk(
    backend: RuntimeQkBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    detection_image: torch.Tensor,
) -> RuntimeQkObservationResult:
    """Capture detached registered-layer Q/K from one ordinary detection image."""

    return _observe_detection_qk_core(
        backend,
        configuration,
        session,
        detection_image,
        differentiable=False,
    )


def observe_differentiable_detection_qk(
    backend: RuntimeDifferentiableQkBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    detection_image: torch.Tensor,
) -> RuntimeQkObservationResult:
    """Capture the frozen image-only Q/K tensors while preserving autograd."""

    if not isinstance(backend, RuntimeDifferentiableQkBackend):
        raise RuntimeQkObservationError(
            "prepared backend lacks differentiable Q/K execution"
        )
    return _observe_detection_qk_core(
        backend,
        configuration,
        session,
        detection_image,
        differentiable=True,
    )
