"""Execution-local suffix replay for the frozen Q/K synchronization write.

This module exposes only public runtime operations.  The captured generation
conditioning remains an opaque, non-persistable backend capability and never
becomes a detector input or record field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .adapter import RuntimeSession
from .backend import (
    RuntimeGenerationSuffixContext,
    RuntimeGeometrySynchronizationBackend,
    RuntimeVaeFactors,
)
from .configuration import Sd35RuntimeConfiguration
from .content_write import (
    RuntimeContentExecutionError,
    _decode_generation_latent,
    _tensor,
)
from .qk_observation import (
    RuntimeQkObservationError,
    RuntimeQkObservationResult,
    observe_detection_qk,
    observe_differentiable_detection_qk,
)


class RuntimeGeometrySynchronizationError(RuntimeError):
    """The real suffix/Q/K replay violated the frozen runtime boundary."""


@dataclass(frozen=True, slots=True)
class RuntimeDifferentiableQkSuffixResult:
    """Autograd-bearing callback latent and final-image Q/K observations."""

    runtime_config_digest: str
    callback_index: int
    callback_latent_float32: torch.Tensor
    generation_terminal_latent: torch.Tensor
    rgb8_ste_image: torch.Tensor
    qk_observation: RuntimeQkObservationResult


@dataclass(frozen=True, slots=True)
class RuntimeActualQkSuffixResult:
    """Actual-dtype suffix, materialized RGB8 image and blind Q/K observation."""

    runtime_config_digest: str
    callback_index: int
    generation_terminal_latent: torch.Tensor
    rgb8_image: torch.Tensor
    qk_observation: RuntimeQkObservationResult


def _validate_boundary(
    backend: RuntimeGeometrySynchronizationBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    suffix_context: RuntimeGenerationSuffixContext,
) -> None:
    if not isinstance(backend, RuntimeGeometrySynchronizationBackend):
        raise RuntimeGeometrySynchronizationError(
            "prepared backend lacks geometry synchronization execution"
        )
    if type(configuration) is not Sd35RuntimeConfiguration:
        raise RuntimeGeometrySynchronizationError(
            "configuration must be Sd35RuntimeConfiguration"
        )
    if type(session) is not RuntimeSession:
        raise RuntimeGeometrySynchronizationError("runtime session is invalid")
    if not isinstance(suffix_context, RuntimeGenerationSuffixContext):
        raise RuntimeGeometrySynchronizationError(
            "generation suffix context is invalid"
        )
    if (
        session.runtime_config_digest != configuration.runtime_config_digest
        or session.callback_index != configuration.callback_index
        or session.detection_schedule_index
        != configuration.detection_schedule_index
        or session.detection_conditioning_protocol
        != configuration.detection_conditioning_protocol
        or suffix_context.runtime_config_digest
        != configuration.runtime_config_digest
        or suffix_context.callback_index != configuration.callback_index
    ):
        raise RuntimeGeometrySynchronizationError(
            "geometry synchronization runtime identity drifted"
        )


def materialize_geometry_candidate(
    candidate_latent: torch.Tensor,
    *,
    expected_shape: torch.Size,
    expected_device: torch.device,
    actual_dtype: str,
) -> torch.Tensor:
    """Perform the candidate's sole actual-dtype materialization operation."""

    if actual_dtype != "float16":
        raise RuntimeGeometrySynchronizationError(
            "geometry synchronization requires frozen float16 actual dtype"
        )
    try:
        candidate = _tensor(
            candidate_latent,
            role="geometry_candidate_latent",
            shape=expected_shape,
            device=expected_device,
        )
        materialized = candidate.detach().to(dtype=torch.float16)
        return _tensor(
            materialized,
            role="geometry_candidate_actual",
            shape=expected_shape,
            dtype=torch.float16,
            device=expected_device,
        ).detach().clone()
    except RuntimeContentExecutionError as exc:
        raise RuntimeGeometrySynchronizationError(
            "geometry candidate materialization failed"
        ) from exc


def _vae_factors(
    backend: RuntimeGeometrySynchronizationBackend,
) -> RuntimeVaeFactors:
    factors = backend.vae_factors()
    if type(factors) is not RuntimeVaeFactors:
        raise RuntimeGeometrySynchronizationError(
            "backend VAE factors do not match the frozen protocol"
        )
    return factors


def observe_differentiable_qk_from_generation_suffix(
    backend: RuntimeGeometrySynchronizationBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    suffix_context: RuntimeGenerationSuffixContext,
    content_written_latent: torch.Tensor,
) -> RuntimeDifferentiableQkSuffixResult:
    """Replay the real suffix and RGB8-STE image-only Q/K path for one gradient."""

    _validate_boundary(backend, configuration, session, suffix_context)
    if (
        not isinstance(content_written_latent, torch.Tensor)
        or content_written_latent.ndim != 4
        or content_written_latent.shape[0] != 1
        or content_written_latent.device != torch.device(session.selected_device)
        or not bool(torch.isfinite(content_written_latent).all())
    ):
        raise RuntimeGeometrySynchronizationError(
            "content-written callback latent is invalid"
        )
    callback_latent = (
        content_written_latent.detach().to(dtype=torch.float32).clone()
    ).requires_grad_(True)
    try:
        terminal = _tensor(
            backend.replay_generation_suffix(
                callback_latent,
                suffix_context,
                differentiable=True,
            ),
            role="differentiable_generation_terminal_latent",
            shape=content_written_latent.shape,
            dtype=torch.float16,
            device=content_written_latent.device,
        )
        factors = _vae_factors(backend)
        decode_input = (
            terminal.to(dtype=torch.float32) / float(factors.scaling_factor)
            + float(factors.shift_factor)
        )
        image = _tensor(
            backend.vae_decode_differentiable(decode_input),
            role="differentiable_generation_image",
        )
        if (
            image.shape
            != (
                1,
                3,
                configuration.image_height,
                configuration.image_width,
            )
            or image.device != content_written_latent.device
        ):
            raise RuntimeGeometrySynchronizationError(
                "differentiable generation image identity drifted"
            )
        quantized = torch.floor(torch.clamp(image, 0.0, 1.0) * 255.0) / 255.0
        rgb8_ste = image + (quantized - image).detach()
        rgb8_ste = _tensor(rgb8_ste, role="generation_rgb8_ste_image")
        qk_result = observe_differentiable_detection_qk(
            backend,
            configuration,
            session,
            rgb8_ste,
        )
    except RuntimeGeometrySynchronizationError:
        raise
    except (RuntimeContentExecutionError, RuntimeQkObservationError) as exc:
        raise RuntimeGeometrySynchronizationError(
            "differentiable suffix Q/K boundary failed"
        ) from exc
    except Exception as exc:
        raise RuntimeGeometrySynchronizationError(
            "differentiable suffix Q/K replay failed"
        ) from exc
    if not callback_latent.requires_grad or not any(
        observation.query.requires_grad
        or observation.attention_key.requires_grad
        for observation in qk_result.qk_layer_observations
    ):
        raise RuntimeGeometrySynchronizationError(
            "differentiable suffix Q/K replay lost its gradient path"
        )
    return RuntimeDifferentiableQkSuffixResult(
        runtime_config_digest=configuration.runtime_config_digest,
        callback_index=configuration.callback_index,
        callback_latent_float32=callback_latent,
        generation_terminal_latent=terminal,
        rgb8_ste_image=rgb8_ste,
        qk_observation=qk_result,
    )


def observe_actual_qk_from_generation_suffix(
    backend: RuntimeGeometrySynchronizationBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    suffix_context: RuntimeGenerationSuffixContext,
    candidate_latent_actual: torch.Tensor,
) -> RuntimeActualQkSuffixResult:
    """Replay one actual candidate through suffix, RGB8 and blind public Q/K."""

    _validate_boundary(backend, configuration, session, suffix_context)
    try:
        candidate = _tensor(
            candidate_latent_actual,
            role="geometry_candidate_actual",
            dtype=torch.float16,
            device=torch.device(session.selected_device),
        ).detach().clone()
        terminal = _tensor(
            backend.replay_generation_suffix(
                candidate,
                suffix_context,
                differentiable=False,
            ),
            role="actual_generation_terminal_latent",
            shape=candidate.shape,
            dtype=torch.float16,
            device=candidate.device,
        ).detach().clone()
        factors = _vae_factors(backend)
        image = _decode_generation_latent(
            backend,
            terminal,
            factors,
            "geometry_actual",
        )
        if image.shape != (
            1,
            3,
            configuration.image_height,
            configuration.image_width,
        ):
            raise RuntimeGeometrySynchronizationError(
                "actual generation image identity drifted"
            )
        rgb8 = torch.floor(torch.clamp(image, 0.0, 1.0) * 255.0).to(
            dtype=torch.uint8
        )
        reread_image = rgb8.to(dtype=torch.float32) / 255.0
        qk_result = observe_detection_qk(
            backend,
            configuration,
            session,
            reread_image,
        )
    except RuntimeGeometrySynchronizationError:
        raise
    except (RuntimeContentExecutionError, RuntimeQkObservationError) as exc:
        raise RuntimeGeometrySynchronizationError(
            "actual suffix Q/K boundary failed"
        ) from exc
    except Exception as exc:
        raise RuntimeGeometrySynchronizationError(
            "actual suffix Q/K replay failed"
        ) from exc
    if any(
        observation.query.requires_grad
        or observation.attention_key.requires_grad
        for observation in qk_result.qk_layer_observations
    ):
        raise RuntimeGeometrySynchronizationError(
            "blind actual Q/K observation retained autograd state"
        )
    return RuntimeActualQkSuffixResult(
        runtime_config_digest=configuration.runtime_config_digest,
        callback_index=configuration.callback_index,
        generation_terminal_latent=terminal,
        rgb8_image=reread_image.detach().clone(),
        qk_observation=qk_result,
    )


__all__ = [
    "RuntimeActualQkSuffixResult",
    "RuntimeDifferentiableQkSuffixResult",
    "RuntimeGeometrySynchronizationError",
    "materialize_geometry_candidate",
    "observe_actual_qk_from_generation_suffix",
    "observe_differentiable_qk_from_generation_suffix",
]
