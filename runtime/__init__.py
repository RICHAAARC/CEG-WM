"""CEG-WM model runtime boundary.

Batch 1 exposes frozen configuration parsing, device selection, backend identity
verification, and initialization lifecycle only. Model execution is deliberately
absent until the later runtime batches.
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
    RuntimeBackend,
    RuntimeBackendError,
    RuntimeBackendIdentity,
    RuntimeDeviceCapabilities,
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

__all__ = [
    "DEFAULT_RUNTIME_CONFIG_PATH",
    "RUNTIME_CANDIDATE_ID",
    "RuntimeAdapterError",
    "RuntimeAdapterState",
    "RuntimeBackend",
    "RuntimeBackendError",
    "RuntimeBackendIdentity",
    "RuntimeConfigurationError",
    "RuntimeDependencyLock",
    "RuntimeDeviceCapabilities",
    "RuntimeSession",
    "Sd35RuntimeAdapter",
    "Sd35RuntimeConfiguration",
    "create_runtime_adapter",
    "load_runtime_configuration",
    "parse_runtime_configuration",
    "select_runtime_device",
]
