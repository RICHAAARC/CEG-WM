"""Fail-closed runtime_configuration_and_adapter control flow for the SD3.5 runtime adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import torch

from main import ContentEmbeddingResult, SemanticTextureRoutingResult

from .backend import (
    RuntimeBackend,
    RuntimeBackendError,
    RuntimeBackendIdentity,
    RuntimeContentBackend,
    RuntimeDeviceCapabilities,
    RuntimeGenerationSuffixBackend,
    RuntimeGenerationSuffixContext,
    RuntimeGeometrySynchronizationBackend,
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
    from .content_write import (
        CleanImageVaeObservationResult,
        ContentWriteGeometrySuffixResult,
        ContentWriteVaeResult,
    )
    from .geometry_synchronization import (
        RuntimeActualQkSuffixResult,
        RuntimeDifferentiableQkSuffixResult,
    )
    from .qk_observation import RuntimeQkObservationResult
    from .routing_observation import RuntimeRoutingObservationResult
    from .routing_observation import RuntimeRoutingReferenceMeasurement
    from .routing_observation import (
        InspyrenetSemanticRuntime,
        RuntimeSemanticTextureDetectionObservationResult,
        RuntimeSemanticTextureObservationResult,
    )


class RuntimeAdapterError(RuntimeError):
    """Runtime initialization or lifecycle control failed closed."""


class RuntimeAdapterState(str, Enum):
    CREATED = "created"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class RuntimeSession:
    """Prepared runtime behavior identity and observed metadata."""

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
    dependency_lock: RuntimeDependencyLock = field(compare=False)


@dataclass(frozen=True, slots=True)
class RuntimeExecutionIdentity:
    """Public runtime/config/session identity without Python-object anchors."""

    identity_schema_version: str
    runtime_config_digest: str
    runtime_state: str
    backend_resources_owned: bool
    runtime_backend_name: str | None
    selected_device: str | None
    runtime_session_identity_digest: str | None

    def identity_mapping(self) -> dict[str, object]:
        return {
            "backend_resources_owned": self.backend_resources_owned,
            "identity_schema_version": self.identity_schema_version,
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
        if capabilities.current_cuda_device_index is None:
            return "cuda"
        return f"cuda:{capabilities.current_cuda_device_index}"
    if requested_device == "cpu":
        if not capabilities.cpu_available:
            raise RuntimeAdapterError(
                "CPU was requested but the backend reported no CPU device"
            )
        return "cpu"
    if capabilities.cuda_device_count > 0:
        if capabilities.current_cuda_device_index is None:
            return "cuda"
        return f"cuda:{capabilities.current_cuda_device_index}"
    if capabilities.cpu_available:
        return "cpu"
    raise RuntimeAdapterError("backend reported no selectable device")


class Sd35RuntimeAdapter:
    """Initialize exactly one frozen runtime backend and preserve its identity."""

    __slots__ = (
        "_backend",
        "_configuration",
        "_configuration_digest_anchor",
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
        self._configuration_digest_anchor = (
            configuration.runtime_config_digest
        )
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

    def revalidate_execution_identity(self) -> RuntimeExecutionIdentity:
        """Fail closed on public config/session/lifecycle drift and return identity."""

        if type(self._configuration) is not Sd35RuntimeConfiguration:
            raise RuntimeAdapterError(
                "runtime configuration exact type drifted"
            )
        try:
            rebuilt_configuration = parse_runtime_configuration(
                self._configuration.configuration_mapping()
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
            runtime_backend_name = self._session.runtime_backend_name
            selected_device = self._session.selected_device
        elif self._state is RuntimeAdapterState.CREATED:
            if (
                self._owns_backend_resources
                or self._session is not None
            ):
                raise RuntimeAdapterError(
                    "created runtime lifecycle identity drifted"
                )
        elif self._state is RuntimeAdapterState.CLOSED:
            if (
                self._owns_backend_resources
                or self._session is not None
            ):
                raise RuntimeAdapterError(
                    "closed runtime lifecycle identity drifted"
                )
        elif (
            self._owns_backend_resources
            or self._session is not None
        ):
            raise RuntimeAdapterError(
                "failed runtime lifecycle retains residual execution state"
            )

        return RuntimeExecutionIdentity(
            identity_schema_version="ceg_wm_runtime_execution_identity",
            runtime_config_digest=self._configuration_digest_anchor,
            runtime_state=self._state.value,
            backend_resources_owned=self._owns_backend_resources,
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

    def execute_semantic_texture_content_write_and_vae(
        self,
        base_latent: torch.Tensor,
        semantic_runtime: InspyrenetSemanticRuntime,
        semantic_texture_embedding_operation: Callable[
            [tuple[float, ...], tuple[int, ...], object],
            tuple[SemanticTextureRoutingResult, ContentEmbeddingResult],
        ],
    ) -> object:
        """Run the live callback traversal for the experiment method adapter."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before semantic-texture content write"
            )
        if not isinstance(self._backend, RuntimeContentBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks semantic-texture content write execution"
            )
        from .routing_observation import InspyrenetSemanticRuntime

        if type(semantic_runtime) is not InspyrenetSemanticRuntime:
            raise RuntimeAdapterError("semantic runtime identity is invalid")
        from .content_write import (
            RuntimeContentExecutionError,
            execute_semantic_texture_content_write_and_vae,
        )

        try:
            self.revalidate_execution_identity()
            result = execute_semantic_texture_content_write_and_vae(
                self._backend,
                self._configuration,
                self.session,
                base_latent,
                semantic_runtime,
                semantic_texture_embedding_operation,
            )
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeContentExecutionError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime semantic-texture content write failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime raised an unexpected semantic-texture content write error"
            ) from exc

    def execute_clean_image_and_vae_observation(
        self,
        base_latent: torch.Tensor,
    ) -> CleanImageVaeObservationResult:
        """Run one clean generation and its public posterior-mode observation."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before clean image observation"
            )
        if not isinstance(self._backend, RuntimeContentBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks the clean image observation protocol"
            )
        from .content_write import (
            RuntimeContentExecutionError,
            execute_clean_image_and_vae_observation,
        )

        try:
            return execute_clean_image_and_vae_observation(
                self._backend,
                self._configuration,
                self.session,
                base_latent,
            )
        except RuntimeContentExecutionError as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime clean image observation failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected clean observation error"
            ) from exc

    def execute_content_write_and_capture_geometry_suffix(
        self,
        base_latent: torch.Tensor,
        content_embedding_operation: Callable[
            [tuple[float, ...]],
            ContentEmbeddingResult,
        ],
    ) -> ContentWriteGeometrySuffixResult:
        """Run paired content execution and retain one in-memory suffix capability."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before geometry suffix capture"
            )
        if not isinstance(self._backend, RuntimeGenerationSuffixBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks generation suffix execution"
            )
        from .content_write import (
            RuntimeContentExecutionError,
            execute_content_write_and_capture_geometry_suffix,
        )

        try:
            self.revalidate_execution_identity()
            result = execute_content_write_and_capture_geometry_suffix(
                self._backend,
                self._configuration,
                self.session,
                base_latent,
                content_embedding_operation,
            )
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeContentExecutionError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime geometry suffix capture failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected geometry suffix capture error"
            ) from exc

    def materialize_geometry_candidate(
        self,
        candidate_latent: torch.Tensor,
        *,
        expected_shape: torch.Size,
        expected_device: torch.device,
    ) -> torch.Tensor:
        """Materialize one line-search candidate at the frozen actual dtype."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before geometry materialization"
            )
        from .geometry_synchronization import (
            RuntimeGeometrySynchronizationError,
            materialize_geometry_candidate,
        )

        try:
            self.revalidate_execution_identity()
            return materialize_geometry_candidate(
                candidate_latent,
                expected_shape=expected_shape,
                expected_device=expected_device,
                actual_dtype=self._configuration.latent_dtype,
            )
        except (RuntimeAdapterError, RuntimeGeometrySynchronizationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime geometry candidate materialization failed closed"
            ) from exc

    def observe_differentiable_qk_from_generation_suffix(
        self,
        suffix_context: RuntimeGenerationSuffixContext,
        content_written_latent: torch.Tensor,
    ) -> RuntimeDifferentiableQkSuffixResult:
        """Run the real suffix and RGB8-STE image-only Q/K gradient path."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before differentiable suffix Q/K"
            )
        if not isinstance(self._backend, RuntimeGeometrySynchronizationBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks geometry synchronization execution"
            )
        from .geometry_synchronization import (
            RuntimeGeometrySynchronizationError,
            observe_differentiable_qk_from_generation_suffix,
        )

        try:
            self.revalidate_execution_identity()
            result = observe_differentiable_qk_from_generation_suffix(
                self._backend,
                self._configuration,
                self.session,
                suffix_context,
                content_written_latent,
            )
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeGeometrySynchronizationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime differentiable suffix Q/K replay failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime raised an unexpected differentiable suffix Q/K error"
            ) from exc

    def observe_actual_qk_from_generation_suffix(
        self,
        suffix_context: RuntimeGenerationSuffixContext,
        candidate_latent_actual: torch.Tensor,
    ) -> RuntimeActualQkSuffixResult:
        """Run one actual candidate through suffix, RGB8 and blind Q/K."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before actual suffix Q/K"
            )
        if not isinstance(self._backend, RuntimeGeometrySynchronizationBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks geometry synchronization execution"
            )
        from .geometry_synchronization import (
            RuntimeGeometrySynchronizationError,
            observe_actual_qk_from_generation_suffix,
        )

        try:
            self.revalidate_execution_identity()
            result = observe_actual_qk_from_generation_suffix(
                self._backend,
                self._configuration,
                self.session,
                suffix_context,
                candidate_latent_actual,
            )
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeGeometrySynchronizationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime actual suffix Q/K replay failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime raised an unexpected actual suffix Q/K error"
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
            from .qk_observation import observe_detection_qk

            result = observe_detection_qk(
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

    def observe_generation_routing(
        self,
        base_latent: torch.Tensor,
        *,
        sample_index: int,
        reference_gradient: float,
        reference_response: float,
        reference_sensitivity: float,
    ) -> RuntimeRoutingObservationResult:
        """Materialize the frozen generation-time T/R/Q routing observations."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before routing observation execution"
            )
        if not isinstance(self._backend, RuntimeContentBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks the routing observation execution protocol"
            )
        from .routing_observation import (
            RuntimeRoutingObservationError,
            observe_generation_routing,
        )

        try:
            self.revalidate_execution_identity()
            result = observe_generation_routing(
                self._backend,
                self._configuration,
                self.session,
                base_latent,
                sample_index=sample_index,
                reference_gradient=reference_gradient,
                reference_response=reference_response,
                reference_sensitivity=reference_sensitivity,
            )
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeRoutingObservationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime generation routing observation failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected routing observation error"
            ) from exc

    def observe_semantic_texture_rgb8(
        self,
        image_rgb8: torch.Tensor,
        semantic_runtime: InspyrenetSemanticRuntime,
    ) -> RuntimeSemanticTextureObservationResult:
        """Rebuild public M/T from one current ordinary RGB8 image."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before semantic-texture observation"
            )
        from .routing_observation import (
            InspyrenetSemanticRuntime,
            RuntimeRoutingObservationError,
        )
        if type(semantic_runtime) is not InspyrenetSemanticRuntime:
            raise RuntimeAdapterError("semantic runtime identity is invalid")
        try:
            self.revalidate_execution_identity()
            result = semantic_runtime.observe(image_rgb8)
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeRoutingObservationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime semantic-texture observation failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected semantic-texture error"
            ) from exc

    def observe_semantic_texture_detection(
        self,
        image_rgb8: torch.Tensor,
        semantic_runtime: InspyrenetSemanticRuntime,
    ) -> RuntimeSemanticTextureDetectionObservationResult:
        """Run same-image M/T reconstruction and public posterior-mode VAE."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before semantic-texture detection"
            )
        if not isinstance(self._backend, RuntimeContentBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks semantic-texture detection execution"
            )
        from .routing_observation import (
            InspyrenetSemanticRuntime,
            RuntimeRoutingObservationError,
            observe_semantic_texture_detection,
        )
        if type(semantic_runtime) is not InspyrenetSemanticRuntime:
            raise RuntimeAdapterError("semantic runtime identity is invalid")
        try:
            self.revalidate_execution_identity()
            result = observe_semantic_texture_detection(
                self._backend,
                self._configuration,
                self.session,
                image_rgb8,
                semantic_runtime,
            )
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeRoutingObservationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime semantic-texture detection failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected semantic-texture detection error"
            ) from exc

    def measure_generation_routing_reference_inputs(
        self,
        base_latent: torch.Tensor,
        *,
        sample_index: int,
    ) -> RuntimeRoutingReferenceMeasurement:
        """Measure raw T/R/Q routing reference inputs in one generation."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before routing measurement"
            )
        if not isinstance(self._backend, RuntimeContentBackend):
            raise RuntimeAdapterError(
                "runtime backend lacks the routing observation execution protocol"
            )
        from .routing_observation import (
            RuntimeRoutingObservationError,
            measure_generation_routing_reference_inputs,
        )

        try:
            self.revalidate_execution_identity()
            measurement = measure_generation_routing_reference_inputs(
                self._backend,
                self._configuration,
                self.session,
                base_latent,
                sample_index=sample_index,
            )
            self.revalidate_execution_identity()
            return measurement
        except (RuntimeAdapterError, RuntimeRoutingObservationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime generation routing measurement failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime backend raised an unexpected routing measurement error"
            ) from exc

    def normalize_generation_routing_measurement(
        self,
        measurement: RuntimeRoutingReferenceMeasurement,
        *,
        reference_gradient: float,
        reference_response: float,
        reference_sensitivity: float,
    ) -> RuntimeRoutingObservationResult:
        """Normalize a prior measurement without repeating generation or VAE."""

        if self._state is not RuntimeAdapterState.READY:
            raise RuntimeAdapterError(
                "runtime adapter must be ready before routing normalization"
            )
        from .routing_observation import (
            RuntimeRoutingObservationError,
            normalize_generation_routing_measurement,
        )

        try:
            self.revalidate_execution_identity()
            result = normalize_generation_routing_measurement(
                measurement,
                reference_gradient=reference_gradient,
                reference_response=reference_response,
                reference_sensitivity=reference_sensitivity,
            )
            if (
                result.runtime_config_digest
                != self._configuration.runtime_config_digest
                or result.callback_indices
                != tuple(range(self._configuration.inference_steps))
            ):
                raise RuntimeRoutingObservationError(
                    "routing measurement does not match the prepared runtime"
                )
            self.revalidate_execution_identity()
            return result
        except (RuntimeAdapterError, RuntimeRoutingObservationError) as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime generation routing normalization failed closed"
            ) from exc
        except Exception as exc:
            self._transition_to_failed(exc)
            raise RuntimeAdapterError(
                "runtime raised an unexpected routing normalization error"
            ) from exc

    def _release_backend_resources(self) -> None:
        if not self._owns_backend_resources:
            return
        self._backend.close()
        self._owns_backend_resources = False

    def _clear_session_identity(self) -> None:
        self._session = None

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


def _runtime_session_identity_mapping(
    session: RuntimeSession,
) -> dict[str, object]:
    identity = asdict(session)
    del identity["model_id"]
    del identity["model_revision"]
    del identity["dependency_lock"]
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
