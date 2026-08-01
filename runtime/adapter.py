"""Fail-closed runtime_configuration_and_adapter control flow for the SD3.5 runtime adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Callable, Literal

import torch

from main import ContentEmbeddingResult

from .backend import (
    RuntimeBackend,
    RuntimeBackendError,
    RuntimeBackendIdentity,
    RuntimeContentBackend,
    RuntimeDeviceCapabilities,
    RuntimeQkBackend,
    validate_backend_identity,
)
from .configuration import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    RuntimeDependencyLock,
    Sd35RuntimeConfiguration,
    load_runtime_configuration,
    parse_runtime_configuration,
)


DeviceRequest = Literal["auto", "cpu", "cuda"]

if TYPE_CHECKING:
    from .content_write import ContentWriteVaeResult
    from .qk_observation import RuntimeQkObservationResult


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


@dataclass(frozen=True, slots=True)
class RuntimeExecutionIdentity:
    """Canonical public identity of the adapter's current execution boundary."""

    identity_schema_version: str
    backend_type_identity: str
    runtime_config_digest: str
    runtime_state: str
    backend_resources_owned: bool
    qk_observation_callable_identity: str
    runtime_backend_name: str | None
    selected_device: str | None
    runtime_session_identity_digest: str | None

    def identity_mapping(self) -> dict[str, object]:
        return {
            "backend_resources_owned": self.backend_resources_owned,
            "backend_type_identity": self.backend_type_identity,
            "identity_schema_version": self.identity_schema_version,
            "qk_observation_callable_identity": (
                self.qk_observation_callable_identity
            ),
            "runtime_backend_name": self.runtime_backend_name,
            "runtime_config_digest": self.runtime_config_digest,
            "runtime_session_identity_digest": (
                self.runtime_session_identity_digest
            ),
            "runtime_state": self.runtime_state,
            "selected_device": self.selected_device,
        }


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
        "_backend_anchor",
        "_backend_type_anchor",
        "_configuration",
        "_configuration_digest_anchor",
        "_owns_backend_resources",
        "_qk_observation_module_anchor",
        "_observe_detection_qk_anchor",
        "_observe_detection_qk_identity_anchor",
        "_session",
        "_session_anchor",
        "_session_identity_digest_anchor",
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
        qk_observation_module = import_module(
            ".qk_observation",
            __package__,
        )
        observe_detection_qk = getattr(
            qk_observation_module,
            "observe_detection_qk",
            None,
        )
        if type(observe_detection_qk) is not FunctionType:
            raise RuntimeAdapterError(
                "runtime Q/K observation callable must be an exact function"
            )
        self._backend = backend
        self._backend_anchor = backend
        self._backend_type_anchor = type(backend)
        self._configuration = configuration
        self._configuration_digest_anchor = (
            configuration.runtime_config_digest
        )
        self._owns_backend_resources = False
        self._qk_observation_module_anchor = qk_observation_module
        self._observe_detection_qk_anchor = observe_detection_qk
        self._observe_detection_qk_identity_anchor = (
            _qualified_function_identity(observe_detection_qk)
        )
        self._session: RuntimeSession | None = None
        self._session_anchor: RuntimeSession | None = None
        self._session_identity_digest_anchor: str | None = None
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

    def revalidate_execution_identity(self) -> RuntimeExecutionIdentity:
        """Fail closed on backend/config/lifecycle drift and return public identity."""

        self._validated_qk_observation_callable()
        if (
            self._backend is not self._backend_anchor
            or type(self._backend) is not self._backend_type_anchor
        ):
            raise RuntimeAdapterError(
                "runtime backend object or exact type drifted"
            )
        if type(self._configuration) is not Sd35RuntimeConfiguration:
            raise RuntimeAdapterError(
                "runtime configuration exact type drifted"
            )
        try:
            rebuilt_configuration = parse_runtime_configuration(
                self._configuration.identity_mapping()
            )
        except Exception as exc:
            raise RuntimeAdapterError(
                "runtime configuration identity drifted"
            ) from exc
        if (
            rebuilt_configuration != self._configuration
            or self._configuration.runtime_config_digest
            != self._configuration_digest_anchor
        ):
            raise RuntimeAdapterError(
                "runtime configuration digest drifted"
            )
        if type(self._state) is not RuntimeAdapterState:
            raise RuntimeAdapterError("runtime adapter state drifted")
        if type(self._owns_backend_resources) is not bool:
            raise RuntimeAdapterError(
                "runtime backend resource ownership drifted"
            )

        session_digest: str | None = None
        runtime_backend_name: str | None = None
        selected_device: str | None = None
        if self._state is RuntimeAdapterState.READY:
            if self._owns_backend_resources is not True:
                raise RuntimeAdapterError(
                    "ready runtime adapter lost backend resource ownership"
                )
            if (
                type(self._session) is not RuntimeSession
                or self._session is not self._session_anchor
                or self._session_identity_digest_anchor is None
            ):
                raise RuntimeAdapterError(
                    "ready runtime session identity drifted"
                )
            _validate_runtime_session(
                self._session,
                self._configuration,
            )
            session_digest = _runtime_session_identity_digest(
                self._session
            )
            if session_digest != self._session_identity_digest_anchor:
                raise RuntimeAdapterError(
                    "ready runtime session content drifted"
                )
            runtime_backend_name = self._session.runtime_backend_name
            selected_device = self._session.selected_device
        elif self._state is RuntimeAdapterState.CREATED:
            if (
                self._owns_backend_resources
                or self._session is not None
                or self._session_anchor is not None
                or self._session_identity_digest_anchor is not None
            ):
                raise RuntimeAdapterError(
                    "created runtime lifecycle identity drifted"
                )
        elif self._state is RuntimeAdapterState.CLOSED:
            if (
                self._owns_backend_resources
                or self._session is not None
                or self._session_anchor is not None
                or self._session_identity_digest_anchor is not None
            ):
                raise RuntimeAdapterError(
                    "closed runtime lifecycle identity drifted"
                )
        elif (
            self._owns_backend_resources
            or self._session is not None
            or self._session_anchor is not None
            or self._session_identity_digest_anchor is not None
        ):
            raise RuntimeAdapterError(
                "failed runtime lifecycle retains residual execution state"
            )

        return RuntimeExecutionIdentity(
            identity_schema_version=(
                "ceg_wm_runtime_execution_identity_v2"
            ),
            backend_type_identity=(
                f"{self._backend_type_anchor.__module__}."
                f"{self._backend_type_anchor.__qualname__}"
            ),
            runtime_config_digest=self._configuration_digest_anchor,
            runtime_state=self._state.value,
            backend_resources_owned=self._owns_backend_resources,
            qk_observation_callable_identity=(
                self._observe_detection_qk_identity_anchor
            ),
            runtime_backend_name=runtime_backend_name,
            selected_device=selected_device,
            runtime_session_identity_digest=session_digest,
        )

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
            self._session_anchor = self._session
            self._session_identity_digest_anchor = (
                _runtime_session_identity_digest(self._session)
            )
            self._state = RuntimeAdapterState.READY
            self.revalidate_execution_identity()
            return self._session
        except (RuntimeAdapterError, RuntimeBackendError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend initialization failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
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
                self._mark_failed_clean()
                raise RuntimeAdapterError(
                    "ready runtime adapter lost backend resource ownership"
                )
            if prior_state is RuntimeAdapterState.FAILED:
                self._mark_failed_clean()
            return
        try:
            self._release_backend_resources()
        except Exception as exc:
            self._mark_failed_clean()
            raise RuntimeAdapterError("runtime backend close failed") from exc
        self._clear_session_identity()
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
        """Run content_write_and_vae while leaving actual-budget semantics in ``main``."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before content_write_and_vae execution"
            )
        if not isinstance(self._backend, RuntimeContentBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks the content_write_and_vae execution protocol"
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
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime content_write_and_vae execution failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected content_write_and_vae error"
            ) from exc

    def observe_detection_qk(
        self,
        detection_image: torch.Tensor,
    ) -> RuntimeQkObservationResult:
        """Run the frozen image-only qk_observation observation path."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before qk_observation execution"
            )
        if not isinstance(self._backend, RuntimeQkBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks the qk_observation Q/K execution protocol"
            )
        from .qk_observation import RuntimeQkObservationError

        try:
            self.revalidate_execution_identity()
            result = self._observe_detection_qk_anchor(
                self._backend,
                self._configuration,
                self.session,
                detection_image,
            )
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeQkObservationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime qk_observation Q/K observation failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected qk_observation error"
            ) from exc

    def _release_backend_resources(self) -> None:
        if not self._owns_backend_resources:
            return
        self._backend.close()
        self._owns_backend_resources = False

    def _validated_qk_observation_callable(self) -> Callable[..., object]:
        current_module = import_module(
            ".qk_observation",
            __package__,
        )
        if (
            type(current_module) is not ModuleType
            or current_module is not self._qk_observation_module_anchor
        ):
            raise RuntimeAdapterError(
                "runtime Q/K observation module identity drifted"
            )
        current_callable = getattr(
            current_module,
            "observe_detection_qk",
            None,
        )
        if (
            type(current_callable) is not FunctionType
            or current_callable is not self._observe_detection_qk_anchor
            or _qualified_function_identity(current_callable)
            != self._observe_detection_qk_identity_anchor
        ):
            raise RuntimeAdapterError(
                "runtime Q/K observation callable identity drifted"
            )
        return self._observe_detection_qk_anchor

    def _clear_session_identity(self) -> None:
        self._session = None
        self._session_anchor = None
        self._session_identity_digest_anchor = None

    def _mark_failed_clean(self) -> None:
        self._state = RuntimeAdapterState.FAILED
        self._owns_backend_resources = False
        self._clear_session_identity()

    def _transition_to_failed(self, cause: Exception) -> None:
        self._state = RuntimeAdapterState.FAILED
        try:
            self._release_backend_resources()
        except Exception as close_exc:
            cause.add_note(f"backend cleanup also failed: {close_exc!r}")
        finally:
            self._mark_failed_clean()


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


def _qualified_function_identity(function: FunctionType) -> str:
    return f"{function.__module__}.{function.__qualname__}"


def _runtime_session_identity_mapping(
    session: RuntimeSession,
) -> dict[str, object]:
    identity = asdict(session)
    identity["dependency_lock"] = (
        session.dependency_lock.as_config_entries()
    )
    identity["qk_layer_names"] = list(session.qk_layer_names)
    return identity


def _runtime_session_identity_digest(session: RuntimeSession) -> str:
    return sha256(
        json.dumps(
            _runtime_session_identity_mapping(session),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_runtime_session(
    session: RuntimeSession,
    configuration: Sd35RuntimeConfiguration,
) -> None:
    identity = RuntimeBackendIdentity(
        **{
            field_name: getattr(session, field_name)
            for field_name in RuntimeBackendIdentity.__dataclass_fields__
        }
    )
    try:
        validate_backend_identity(
            identity,
            configuration,
            session.selected_device,
        )
    except RuntimeBackendError as exc:
        raise RuntimeAdapterError(
            "ready runtime session differs from frozen configuration"
        ) from exc


def create_runtime_adapter(
    backend: RuntimeBackend,
    config_path: str | Path = DEFAULT_RUNTIME_CONFIG_PATH,
) -> Sd35RuntimeAdapter:
    """Load the frozen configuration and construct an uninitialized adapter."""

    return Sd35RuntimeAdapter(
        backend=backend,
        configuration=load_runtime_configuration(config_path),
    )
