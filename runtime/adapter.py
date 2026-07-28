"""Fail-closed Batch-1 control flow for the SD3.5 runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import torch

from main import ContentEmbeddingResult

from .backend import (
    RuntimeBackend,
    RuntimeBackendError,
    RuntimeBackendIdentity,
    RuntimeContentBackend,
    RuntimeDeviceCapabilities,
    validate_backend_identity,
)
from .configuration import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    RuntimeDependencyLock,
    Sd35RuntimeConfiguration,
    load_runtime_configuration,
)


DeviceRequest = Literal["auto", "cpu", "cuda"]

if TYPE_CHECKING:
    from .content_write import ContentWriteVaeResult


class RuntimeAdapterError(RuntimeError):
    """Runtime initialization or lifecycle control failed closed."""


class RuntimeAdapterState(str, Enum):
    CREATED = "created"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class RuntimeSession:
    """Prepared runtime identity; no model outputs or method decisions."""

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


def select_runtime_device(
    capabilities: RuntimeDeviceCapabilities,
    requested_device: DeviceRequest = "auto",
) -> str:
    """Resolve a device deterministically without probing or loading a model."""

    if type(capabilities) is not RuntimeDeviceCapabilities:
        raise RuntimeAdapterError(
            "device capabilities must be RuntimeDeviceCapabilities"
        )
    if requested_device not in ("auto", "cpu", "cuda"):
        raise RuntimeAdapterError(
            "requested_device must be one of auto, cpu, or cuda"
        )
    if requested_device == "cuda":
        if capabilities.cuda_device_count == 0:
            raise RuntimeAdapterError(
                "CUDA was requested but the backend reported no CUDA device"
            )
        return "cuda:0"
    if requested_device == "cpu":
        if not capabilities.cpu_available:
            raise RuntimeAdapterError(
                "CPU was requested but the backend reported no CPU device"
            )
        return "cpu"
    if capabilities.cuda_device_count > 0:
        return "cuda:0"
    if capabilities.cpu_available:
        return "cpu"
    raise RuntimeAdapterError("backend reported no selectable device")


class Sd35RuntimeAdapter:
    """Initialize exactly one frozen runtime backend and preserve its identity."""

    __slots__ = (
        "_backend",
        "_configuration",
        "_owns_backend_resources",
        "_session",
        "_state",
    )

    def __init__(
        self,
        backend: RuntimeBackend,
        configuration: Sd35RuntimeConfiguration,
    ) -> None:
        if not isinstance(backend, RuntimeBackend):
            raise RuntimeAdapterError(
                "backend does not implement the RuntimeBackend protocol"
            )
        if type(configuration) is not Sd35RuntimeConfiguration:
            raise RuntimeAdapterError(
                "configuration must be Sd35RuntimeConfiguration"
            )
        self._backend = backend
        self._configuration = configuration
        self._owns_backend_resources = False
        self._session: RuntimeSession | None = None
        self._state = RuntimeAdapterState.CREATED

    @property
    def configuration(self) -> Sd35RuntimeConfiguration:
        return self._configuration

    @property
    def state(self) -> RuntimeAdapterState:
        return self._state

    @property
    def session(self) -> RuntimeSession:
        if self._session is None:
            raise RuntimeAdapterError("runtime adapter is not ready")
        return self._session

    def initialize(
        self,
        requested_device: DeviceRequest = "auto",
    ) -> RuntimeSession:
        if self._state is not RuntimeAdapterState.CREATED:
            raise RuntimeAdapterError(
                f"runtime adapter cannot initialize from state {self._state.value}"
            )
        try:
            capabilities = self._backend.probe_devices()
            selected_device = select_runtime_device(
                capabilities,
                requested_device,
            )
            self._owns_backend_resources = True
            actual_identity = self._backend.prepare(
                self._configuration,
                selected_device,
            )
            identity = validate_backend_identity(
                actual_identity,
                self._configuration,
                selected_device,
            )
            self._session = _runtime_session(identity)
            self._state = RuntimeAdapterState.READY
            return self._session
        except (RuntimeAdapterError, RuntimeBackendError) as exc:
            self._state = RuntimeAdapterState.FAILED
            self._release_after_failure(exc)
            raise RuntimeAdapterError(
                "runtime backend initialization failed closed"
            ) from exc
        except Exception as exc:
            self._state = RuntimeAdapterState.FAILED
            self._release_after_failure(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected initialization error"
            ) from exc

    def close(self) -> None:
        if self._state is RuntimeAdapterState.CLOSED:
            return
        if self._state is RuntimeAdapterState.CREATED:
            raise RuntimeAdapterError(
                f"runtime adapter cannot close from state {self._state.value}"
            )
        prior_state = self._state
        if not self._owns_backend_resources:
            if prior_state is RuntimeAdapterState.READY:
                self._state = RuntimeAdapterState.FAILED
                raise RuntimeAdapterError(
                    "ready runtime adapter lost backend resource ownership"
                )
            return
        try:
            self._release_backend_resources()
        except Exception as exc:
            self._state = RuntimeAdapterState.FAILED
            raise RuntimeAdapterError("runtime backend close failed") from exc
        self._session = None
        if prior_state is RuntimeAdapterState.READY:
            self._state = RuntimeAdapterState.CLOSED

    def execute_content_write_and_vae(
        self,
        base_latent: torch.Tensor,
        content_embedding_operation: Callable[
            [tuple[float, ...]],
            ContentEmbeddingResult,
        ],
    ) -> ContentWriteVaeResult:
        """Run the Batch-2 path only after identity-checked preparation."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before Batch-2 execution"
            )
        if not isinstance(self._backend, RuntimeContentBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks the Batch-2 execution protocol"
            )
        from .content_write import (
            RuntimeContentExecutionError,
            execute_content_write_and_vae,
        )

        try:
            return execute_content_write_and_vae(
                self._backend,
                self._configuration,
                self.session,
                base_latent,
                content_embedding_operation,
            )
        except RuntimeContentExecutionError as exc:
            self._state = RuntimeAdapterState.FAILED
            self._release_after_failure(exc)
            raise RuntimeAdapterError(
                "runtime Batch-2 execution failed closed"
            ) from exc
        except Exception as exc:
            self._state = RuntimeAdapterState.FAILED
            self._release_after_failure(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected Batch-2 error"
            ) from exc

    def _release_backend_resources(self) -> None:
        if not self._owns_backend_resources:
            return
        self._backend.close()
        self._owns_backend_resources = False

    def _release_after_failure(self, cause: Exception) -> None:
        try:
            self._release_backend_resources()
        except Exception as close_exc:
            cause.add_note(f"backend cleanup also failed: {close_exc!r}")


def _runtime_session(
    identity: RuntimeBackendIdentity,
) -> RuntimeSession:
    return RuntimeSession(
        candidate_id=identity.candidate_id,
        runtime_config_digest=identity.runtime_config_digest,
        runtime_backend_name=identity.runtime_backend_name,
        selected_device=identity.selected_device,
        model_id=identity.model_id,
        model_revision=identity.model_revision,
        pipeline_class=identity.pipeline_class,
        scheduler_class=identity.scheduler_class,
        inference_steps=identity.inference_steps,
        guidance_scale=identity.guidance_scale,
        image_height=identity.image_height,
        image_width=identity.image_width,
        generation_seed_device=identity.generation_seed_device,
        latent_dtype=identity.latent_dtype,
        template_dtype=identity.template_dtype,
        score_dtype=identity.score_dtype,
        callback_index=identity.callback_index,
        callback_hold_scheduler_intervals=(
            identity.callback_hold_scheduler_intervals
        ),
        vae_decode_protocol=identity.vae_decode_protocol,
        vae_encode_protocol=identity.vae_encode_protocol,
        vae_scaling_factor_source=identity.vae_scaling_factor_source,
        vae_shift_factor_source=identity.vae_shift_factor_source,
        detection_schedule_index=identity.detection_schedule_index,
        detection_conditioning_protocol=(
            identity.detection_conditioning_protocol
        ),
        qk_layer_names=identity.qk_layer_names,
        dependency_lock=identity.dependency_lock,
    )


def create_runtime_adapter(
    backend: RuntimeBackend,
    config_path: str | Path = DEFAULT_RUNTIME_CONFIG_PATH,
) -> Sd35RuntimeAdapter:
    """Load the frozen configuration and construct an uninitialized adapter."""

    return Sd35RuntimeAdapter(
        backend=backend,
        configuration=load_runtime_configuration(config_path),
    )
