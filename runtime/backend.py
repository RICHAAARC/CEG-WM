"""Backend boundary for model loading and runtime identity verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .configuration import RuntimeDependencyLock, Sd35RuntimeConfiguration


class RuntimeBackendError(RuntimeError):
    """A backend could not provide the frozen runtime identity."""


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
    """Minimal Batch-1 backend protocol; model execution is added later."""

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
