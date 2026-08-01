"""Backend boundary for model loading and runtime identity verification."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Callable, Protocol, runtime_checkable

import torch

from .configuration import RuntimeDependencyLock, Sd35RuntimeConfiguration


class RuntimeBackendError(RuntimeError):
    """A backend could not provide the frozen runtime identity."""


GenerationCallback = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class RuntimeGenerationPromptIdentity:
    """UTF-8 identity of the exact prompts snapshotted for one generation."""

    prompt_digest: str
    negative_prompt_digest: str

    @classmethod
    def from_prompts(
        cls,
        prompt: str,
        negative_prompt: str,
    ) -> "RuntimeGenerationPromptIdentity":
        if type(prompt) is not str or type(negative_prompt) is not str:
            raise RuntimeBackendError("generation prompts must be text")
        return cls(
            prompt_digest=sha256(prompt.encode("utf-8")).hexdigest(),
            negative_prompt_digest=sha256(
                negative_prompt.encode("utf-8")
            ).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class RuntimeDeviceCapabilities:
    """Devices visible to a backend before any model is loaded."""

    cpu_available: bool
    cuda_device_count: int

    def __post_init__(self) -> None:
        if type(self.cpu_available) is not bool:
            raise RuntimeBackendError("cpu_available must be boolean")
        if type(self.cuda_device_count) is not int or self.cuda_device_count < 0:
            raise RuntimeBackendError(
                "cuda_device_count must be a non-negative integer"
            )
        if not self.cpu_available and self.cuda_device_count == 0:
            raise RuntimeBackendError("backend reported no usable devices")


@dataclass(frozen=True, slots=True)
class RuntimeBackendIdentity:
    """Actual backend identity observed after preparation."""

    candidate_id: str
    runtime_config_digest: str
    runtime_backend_name: str
    selected_device: str
    model_id: str
    model_revision: str
    pipeline_class: str
    scheduler_class: str
    inference_steps: int
    guidance_scale: float
    image_height: int
    image_width: int
    generation_seed_device: str
    latent_dtype: str
    template_dtype: str
    score_dtype: str
    callback_index: int
    callback_hold_scheduler_intervals: int
    vae_decode_protocol: str
    vae_encode_protocol: str
    vae_scaling_factor_source: str
    vae_shift_factor_source: str
    detection_schedule_index: int
    detection_conditioning_protocol: str
    qk_layer_names: tuple[str, ...]
    dependency_lock: RuntimeDependencyLock


@runtime_checkable
class RuntimeBackend(Protocol):
    """Minimal runtime_configuration_and_adapter backend protocol; model execution is added later."""

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        """Return device availability without loading a model."""

    def prepare(
        self,
        configuration: Sd35RuntimeConfiguration,
        selected_device: str,
    ) -> RuntimeBackendIdentity:
        """Prepare the backend and return the identity actually materialized."""

    def close(self) -> None:
        """Release any resources acquired by ``prepare``."""


@dataclass(frozen=True, slots=True)
class RuntimeVaeFactors:
    """Actual VAE scaling and shift factors read from the prepared backend."""

    scaling_factor: float
    shift_factor: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.scaling_factor, bool)
            or not isinstance(self.scaling_factor, (int, float))
            or not isfinite(float(self.scaling_factor))
            or float(self.scaling_factor) <= 0.0
        ):
            raise RuntimeBackendError(
                "VAE scaling_factor must be finite and positive"
            )
        if (
            isinstance(self.shift_factor, bool)
            or not isinstance(self.shift_factor, (int, float))
            or not isfinite(float(self.shift_factor))
        ):
            raise RuntimeBackendError("VAE shift_factor must be finite")


@runtime_checkable
class RuntimeVaePosterior(Protocol):
    """Narrow posterior boundary; runtime is only allowed to call ``mode``."""

    def mode(self) -> torch.Tensor:
        """Return the deterministic posterior mode."""


@dataclass(frozen=True, slots=True)
class RuntimeDetectionConditioning:
    """Frozen image-only conditioning passed to the prepared backend."""

    prompt: str
    prompt_2: str
    prompt_3: str
    do_classifier_free_guidance: bool
    detection_conditioning_protocol: str


@dataclass(frozen=True, slots=True)
class RuntimeDetectionScheduleStep:
    """One timestep selected from a newly established detection schedule."""

    scheduler_class: str
    inference_steps: int
    detection_schedule_index: int
    detection_timestep: torch.Tensor


@dataclass(frozen=True, slots=True)
class RuntimeQkForwardIdentity:
    """Backend-observed identity for one image-only Q/K forward."""

    runtime_config_digest: str
    model_id: str
    model_revision: str
    scheduler_class: str
    inference_steps: int
    detection_schedule_index: int
    detection_conditioning_protocol: str
    prompt: str
    prompt_2: str
    prompt_3: str
    do_classifier_free_guidance: bool
    qk_layer_names: tuple[str, ...]


@runtime_checkable
class RuntimeVaeBackend(Protocol):
    """Shared deterministic VAE boundary used by generation and detection."""

    def vae_factors(self) -> RuntimeVaeFactors:
        """Read actual scaling and shift factors from the prepared VAE."""

    def vae_encode(self, image: torch.Tensor) -> RuntimeVaePosterior:
        """Return the posterior for one ordinary image."""


@runtime_checkable
class RuntimeContentBackend(RuntimeVaeBackend, Protocol):
    """content_write_and_vae tensor execution boundary for a prepared model backend."""

    def run_generation(
        self,
        initial_latent: torch.Tensor,
        callback: GenerationCallback,
    ) -> torch.Tensor:
        """Run a generation path and consume callback return tensors."""

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent already transformed by the frozen protocol."""


@runtime_checkable
class RuntimeQkBackend(RuntimeVaeBackend, Protocol):
    """qk_observation image-only Q/K execution boundary for a prepared model."""

    def create_detection_schedule(
        self,
        inference_steps: int,
    ) -> RuntimeDetectionScheduleStep:
        """Rebuild the frozen scheduler and select its registered step."""

    def scale_detection_noise(
        self,
        detection_latent: torch.Tensor,
        public_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Call the prepared scheduler's ``scale_noise`` operation."""

    def attention_module(self, layer_name: str) -> torch.nn.Module:
        """Return the actual registered attention module by exact path."""

    def run_qk_detection_forward(
        self,
        noisy_detection_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: RuntimeDetectionConditioning,
    ) -> RuntimeQkForwardIdentity:
        """Run one image-only transformer forward while hooks are active."""


def validate_backend_identity(
    identity: object,
    configuration: Sd35RuntimeConfiguration,
    selected_device: str,
) -> RuntimeBackendIdentity:
    """Reject any silent backend substitution or dtype/QK drift."""

    if type(identity) is not RuntimeBackendIdentity:
        raise RuntimeBackendError(
            "backend prepare must return RuntimeBackendIdentity"
        )
    expected = {
        "candidate_id": configuration.candidate_id,
        "runtime_config_digest": configuration.runtime_config_digest,
        "selected_device": selected_device,
        "model_id": configuration.model_id,
        "model_revision": configuration.model_revision,
        "pipeline_class": configuration.pipeline_class,
        "scheduler_class": configuration.scheduler_class,
        "inference_steps": configuration.inference_steps,
        "guidance_scale": configuration.guidance_scale,
        "image_height": configuration.image_height,
        "image_width": configuration.image_width,
        "generation_seed_device": configuration.generation_seed_device,
        "latent_dtype": configuration.latent_dtype,
        "template_dtype": configuration.template_dtype,
        "score_dtype": configuration.score_dtype,
        "callback_index": configuration.callback_index,
        "callback_hold_scheduler_intervals": (
            configuration.callback_hold_scheduler_intervals
        ),
        "vae_decode_protocol": configuration.vae_decode_protocol,
        "vae_encode_protocol": configuration.vae_encode_protocol,
        "vae_scaling_factor_source": configuration.vae_scaling_factor_source,
        "vae_shift_factor_source": configuration.vae_shift_factor_source,
        "detection_schedule_index": configuration.detection_schedule_index,
        "detection_conditioning_protocol": (
            configuration.detection_conditioning_protocol
        ),
        "qk_layer_names": configuration.qk_layer_names,
        "dependency_lock": configuration.dependency_lock,
    }
    for field, expected_value in expected.items():
        if getattr(identity, field) != expected_value:
            raise RuntimeBackendError(
                f"backend {field} does not match the frozen configuration"
            )
    if (
        type(identity.runtime_backend_name) is not str
        or not identity.runtime_backend_name
    ):
        raise RuntimeBackendError(
            "runtime_backend_name must be a non-empty string"
        )
    return identity
