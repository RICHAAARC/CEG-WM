"""CEG-WM model runtime boundary.

Batch 1 exposes frozen identity/lifecycle control. Batch 2 adds paired content
materialization measurements and deterministic VAE boundaries while delegating
all content-budget semantics to the public ``main`` method interface.
"""

from .adapter import (
    RuntimeAdapterError,
    RuntimeAdapterState,
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
    RuntimeDeviceCapabilities,
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
    ContentEmbeddingOperation,
    ContentMaterializationAttempt,
    ContentMaterializationMeasurement,
    ContentWriteVaeResult,
    RuntimeContentExecutionError,
    measure_content_materialization,
)

__all__ = [
    "DEFAULT_RUNTIME_CONFIG_PATH",
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
    "RuntimeDependencyLock",
    "RuntimeDeviceCapabilities",
    "RuntimeSession",
    "RuntimeVaeFactors",
    "RuntimeVaePosterior",
    "Sd35RuntimeAdapter",
    "Sd35RuntimeConfiguration",
    "create_runtime_adapter",
    "load_runtime_configuration",
    "measure_content_materialization",
    "parse_runtime_configuration",
    "select_runtime_device",
]
