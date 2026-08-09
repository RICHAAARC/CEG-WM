"""CEG-WM model runtime boundary.

runtime_configuration_and_adapter exposes frozen identity/lifecycle control. content_write_and_vae adds paired content
materialization measurements and deterministic VAE boundaries. qk_observation adds
image-only registered-layer Q/K capture. Method semantics remain in ``main``.
"""

from .adapter import (
    RuntimeAdapterError,
    RuntimeAdapterState,
    RuntimeExecutionIdentity,
    RuntimeSession,
    Sd35RuntimeAdapter,
    create_runtime_adapter,
    select_runtime_device,
)
from .backend import (
    GenerationCallback,
    RuntimeBackend,
    RuntimeBackendError,
    RuntimeBackendIdentity,
    RuntimeContentBackend,
    RuntimeDetectionConditioning,
    RuntimeDetectionScheduleStep,
    RuntimeDeviceCapabilities,
    RuntimeGenerationPromptIdentity,
    RuntimeQkBackend,
    RuntimeQkForwardIdentity,
    RuntimeVaeBackend,
    RuntimeVaeFactors,
    RuntimeVaePosterior,
)
from .configuration import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    RUNTIME_CANDIDATE_ID,
    RuntimeConfigurationError,
    RuntimeDependencyLock,
    Sd35RuntimeConfiguration,
    load_runtime_configuration,
    parse_runtime_configuration,
)
from .content_write import (
    CleanImageVaeObservationResult,
    ContentEmbeddingOperation,
    ContentMaterializationAttempt,
    ContentMaterializationMeasurement,
    ContentWriteVaeResult,
    RuntimeContentExecutionError,
    execute_clean_image_and_vae_observation,
    measure_content_materialization,
)
from .qk_observation import (
    RuntimeQkObservationError,
    RuntimeQkObservationResult,
    observe_detection_qk,
)
from .routing_observation import (
    ROUTING_OBSERVATION_CANDIDATE_ID,
    ROUTING_PROBE_RELATIVE_STEP,
    RuntimeRoutingObservationError,
    RuntimeRoutingObservationResult,
    RuntimeRoutingReferenceMeasurement,
    measure_generation_routing_reference_inputs,
    normalize_generation_routing_measurement,
    observe_generation_routing,
)
from .sd35_backend import (
    Sd35BackendError,
    Sd35PipelineBackend,
)

__all__ = [
    "DEFAULT_RUNTIME_CONFIG_PATH",
    "CleanImageVaeObservationResult",
    "ContentEmbeddingOperation",
    "ContentMaterializationAttempt",
    "ContentMaterializationMeasurement",
    "ContentWriteVaeResult",
    "GenerationCallback",
    "RUNTIME_CANDIDATE_ID",
    "RuntimeAdapterError",
    "RuntimeAdapterState",
    "RuntimeBackend",
    "RuntimeBackendError",
    "RuntimeBackendIdentity",
    "RuntimeContentBackend",
    "RuntimeContentExecutionError",
    "RuntimeConfigurationError",
    "RuntimeDetectionConditioning",
    "RuntimeDetectionScheduleStep",
    "RuntimeDependencyLock",
    "RuntimeDeviceCapabilities",
    "RuntimeGenerationPromptIdentity",
    "RuntimeExecutionIdentity",
    "RuntimeQkBackend",
    "RuntimeQkForwardIdentity",
    "RuntimeQkObservationError",
    "RuntimeQkObservationResult",
    "RuntimeRoutingObservationError",
    "RuntimeRoutingObservationResult",
    "RuntimeRoutingReferenceMeasurement",
    "RuntimeSession",
    "RuntimeVaeBackend",
    "RuntimeVaeFactors",
    "RuntimeVaePosterior",
    "Sd35RuntimeAdapter",
    "Sd35RuntimeConfiguration",
    "ROUTING_OBSERVATION_CANDIDATE_ID",
    "ROUTING_PROBE_RELATIVE_STEP",
    "Sd35BackendError",
    "Sd35PipelineBackend",
    "create_runtime_adapter",
    "execute_clean_image_and_vae_observation",
    "load_runtime_configuration",
    "measure_content_materialization",
    "measure_generation_routing_reference_inputs",
    "normalize_generation_routing_measurement",
    "observe_detection_qk",
    "observe_generation_routing",
    "parse_runtime_configuration",
    "select_runtime_device",
]
