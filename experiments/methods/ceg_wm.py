"""Thin experiment-facing delegation to the implemented CEG-WM method."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Concatenate, Generic, ParamSpec, Sequence, TypeVar

import torch

from main import (
    BranchNullCalibration,
    ConditionalRecoveryResult,
    ContentDetectionResult,
    ContentDetectorBinding,
    ContentEmbeddingResult,
    ContentRoutingResult,
    DerivedWrongKeyMaterial,
    GeometricTransformEstimation,
    GeometryEstimationOperation,
    GeometryReliabilityResult,
    GeometryReliabilityThresholds,
    HfCarrierResult,
    HfDetectionObservation,
    HfDetectionResult,
    ImageRectificationResult,
    JointDecisionThresholds,
    KeyScheduleConfig,
    KeyStreamResult,
    LfCarrierResult,
    LfDetectionObservation,
    LfDetectionResult,
    QkGeometrySyncResult,
    RootKeyIdentity,
    RoutingObservations,
    content_detector,
    content_embedder,
    content_router,
    conditional_recovery_decision,
    derive_public_noise_stream,
    derive_wrong_key_material,
    derive_wrong_key_stream,
    geometric_transform_estimator,
    geometry_reliability,
    hf_carrier,
    hf_detector,
    identify_root_key,
    image_rectifier,
    key_schedule_sha256_counter,
    lf_carrier,
    lf_detector,
    qk_geometry_sync,
)
from runtime import RuntimeQkObservationResult, Sd35RuntimeAdapter


DEFAULT_COMPONENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "experiments"
    / "internal_execution_components.json"
)
REQUIRED_COMPONENT_BINDINGS = (
    ("key_schedule", "main.identify_root_key", "config_digest"),
    ("content_router", "main.content_router", "route_identity"),
    ("lf_carrier", "main.lf_carrier", "carrier_config_digest"),
    ("hf_carrier", "main.hf_carrier", "carrier_config_digest"),
    (
        "content_embedder",
        "main.content_embedder",
        "embedding_result_identity",
    ),
    ("lf_detector", "main.lf_detector", "detector_identity"),
    ("hf_detector", "main.hf_detector", "detector_identity"),
    ("content_detector", "main.content_detector", "detector_identity"),
    (
        "qk_geometry_sync",
        (
            "runtime.Sd35RuntimeAdapter.observe_detection_qk"
            " -> main.qk_geometry_sync"
        ),
        "geometry_config_digest",
    ),
    (
        "geometric_transform_estimator",
        "main.geometric_transform_estimator",
        "estimation_identity_digest",
    ),
    (
        "geometry_reliability",
        "main.geometry_reliability",
        "reliability_identity_digest",
    ),
    (
        "image_rectifier",
        "main.image_rectifier",
        "rectified_image_digest",
    ),
    (
        "conditional_recovery_decision",
        "main.conditional_recovery_decision",
        "decision_identity_digest",
    ),
)
REQUIRED_RESPONSIBILITIES = tuple(
    binding[0] for binding in REQUIRED_COMPONENT_BINDINGS
)
REQUIRED_KEY_SCHEDULE_OPERATIONS = (
    (
        "identify_key",
        "main.identify_root_key",
    ),
    (
        "derive_registered_key_stream",
        "main.key_schedule_sha256_counter",
    ),
    (
        "derive_wrong_key_stream",
        (
            "main.derive_wrong_key_material"
            " -> main.derive_wrong_key_stream"
        ),
    ),
    (
        "derive_public_noise",
        "main.derive_public_noise_stream",
    ),
)
T = TypeVar("T")
P = ParamSpec("P")


class CegWmExperimentAdapterError(ValueError):
    """The experiment adapter configuration or delegation failed closed."""


def _canonical_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class MethodComponentBinding:
    responsibility: str
    public_callable: str
    result_identity_field: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (
                self.responsibility,
                self.public_callable,
                self.result_identity_field,
            )
        ):
            raise CegWmExperimentAdapterError(
                "method component binding fields must be non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class KeyScheduleOperationBinding:
    operation_id: str
    public_callable: str

    def __post_init__(self) -> None:
        if (
            type(self.operation_id) is not str
            or not self.operation_id
            or type(self.public_callable) is not str
            or not self.public_callable
        ):
            raise CegWmExperimentAdapterError(
                "key schedule operation binding fields must be non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class CegWmExperimentAdapterConfiguration:
    schema_version: str
    registry_version: str
    adapter_id: str
    adapter_version: str
    component_bindings: tuple[MethodComponentBinding, ...]
    key_schedule_operations: tuple[KeyScheduleOperationBinding, ...]
    config_digest: str = field(init=False)

    def __post_init__(self) -> None:
        payload = _validated_adapter_configuration_payload(self)
        object.__setattr__(self, "config_digest", _canonical_digest(payload))


def _validated_adapter_configuration_payload(
    configuration: object,
) -> dict[str, object]:
    if type(configuration) is not CegWmExperimentAdapterConfiguration:
        raise CegWmExperimentAdapterError(
            "configuration must be CegWmExperimentAdapterConfiguration"
        )
    for value in (
        configuration.schema_version,
        configuration.registry_version,
        configuration.adapter_id,
        configuration.adapter_version,
    ):
        if type(value) is not str or not value:
            raise CegWmExperimentAdapterError(
                "adapter configuration identities must be non-empty strings"
            )
    if (
        type(configuration.component_bindings) is not tuple
        or any(
            type(binding) is not MethodComponentBinding
            for binding in configuration.component_bindings
        )
    ):
        raise CegWmExperimentAdapterError(
            "adapter component bindings drifted from the canonical registry"
        )
    component_bindings = tuple(
        (
            binding.responsibility,
            binding.public_callable,
            binding.result_identity_field,
        )
        for binding in configuration.component_bindings
    )
    if component_bindings != REQUIRED_COMPONENT_BINDINGS:
        raise CegWmExperimentAdapterError(
            "adapter component bindings drifted from the canonical registry"
        )
    if (
        type(configuration.key_schedule_operations) is not tuple
        or any(
            type(binding) is not KeyScheduleOperationBinding
            for binding in configuration.key_schedule_operations
        )
    ):
        raise CegWmExperimentAdapterError(
            "key schedule operation bindings or order drifted"
        )
    operations = tuple(
        (binding.operation_id, binding.public_callable)
        for binding in configuration.key_schedule_operations
    )
    if operations != REQUIRED_KEY_SCHEDULE_OPERATIONS:
        raise CegWmExperimentAdapterError(
            "key schedule operation bindings or order drifted"
        )
    return {
        "adapter_id": configuration.adapter_id,
        "adapter_version": configuration.adapter_version,
        "component_bindings": [
            {
                "public_callable": binding.public_callable,
                "responsibility": binding.responsibility,
                "result_identity_field": binding.result_identity_field,
            }
            for binding in configuration.component_bindings
        ],
        "key_schedule_operations": [
            {
                "operation_id": binding.operation_id,
                "public_callable": binding.public_callable,
            }
            for binding in configuration.key_schedule_operations
        ],
        "registry_version": configuration.registry_version,
        "schema_version": configuration.schema_version,
    }


def _revalidate_adapter_configuration(
    configuration: object,
) -> CegWmExperimentAdapterConfiguration:
    payload = _validated_adapter_configuration_payload(configuration)
    if (
        type(configuration.config_digest) is not str
        or configuration.config_digest != _canonical_digest(payload)
    ):
        raise CegWmExperimentAdapterError(
            "adapter configuration digest mismatch"
        )
    return configuration


def _revalidate_configuration_before_call(
    method: Callable[
        Concatenate[CegWmExperimentAdapter, P],
        T,
    ],
) -> Callable[Concatenate[CegWmExperimentAdapter, P], T]:
    @wraps(method)
    def guarded(
        adapter: CegWmExperimentAdapter,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        _revalidate_adapter_configuration(adapter._configuration)
        return method(adapter, *args, **kwargs)

    return guarded


@dataclass(frozen=True, slots=True)
class ComponentCallObservation(Generic[T]):
    """One actual delegated result plus its experiment-observable identity."""

    responsibility: str
    public_callable: str
    adapter_config_digest: str
    result_type: str
    result_identity: str
    upstream_runtime_identity: str | None
    result: T


def load_ceg_wm_experiment_adapter_configuration(
    path: str | Path = DEFAULT_COMPONENT_CONFIG_PATH,
) -> CegWmExperimentAdapterConfiguration:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if type(document) is not dict or set(document) != {
        "attack_registry",
        "method_adapter",
        "metric_registry",
        "schema_version",
    }:
        raise CegWmExperimentAdapterError(
            "execution component configuration fields drifted"
        )
    if document.get("schema_version") != "ceg_wm_internal_execution_components_v1":
        raise CegWmExperimentAdapterError("execution component schema drifted")
    raw = document.get("method_adapter")
    if type(raw) is not dict or set(raw) != {
        "adapter_id",
        "adapter_version",
        "component_bindings",
        "key_schedule_operations",
        "registry_version",
    }:
        raise CegWmExperimentAdapterError("method_adapter configuration missing")
    bindings_raw = raw.get("component_bindings")
    if type(bindings_raw) is not list:
        raise CegWmExperimentAdapterError("component bindings must be a list")
    key_operations_raw = raw.get("key_schedule_operations")
    if type(key_operations_raw) is not list:
        raise CegWmExperimentAdapterError(
            "key schedule operations must be a list"
        )
    try:
        if any(
            type(item) is not dict
            or set(item)
            != {
                "public_callable",
                "responsibility",
                "result_identity_field",
            }
            for item in bindings_raw
        ):
            raise CegWmExperimentAdapterError(
                "method component binding fields drifted"
            )
        bindings = tuple(
            MethodComponentBinding(
                responsibility=item["responsibility"],
                public_callable=item["public_callable"],
                result_identity_field=item["result_identity_field"],
            )
            for item in bindings_raw
        )
        if any(
            type(item) is not dict
            or set(item) != {"operation_id", "public_callable"}
            for item in key_operations_raw
        ):
            raise CegWmExperimentAdapterError(
                "key schedule operation fields drifted"
            )
        key_schedule_operations = tuple(
            KeyScheduleOperationBinding(
                operation_id=item["operation_id"],
                public_callable=item["public_callable"],
            )
            for item in key_operations_raw
        )
        return CegWmExperimentAdapterConfiguration(
            schema_version=document["schema_version"],
            registry_version=raw["registry_version"],
            adapter_id=raw["adapter_id"],
            adapter_version=raw["adapter_version"],
            component_bindings=bindings,
            key_schedule_operations=key_schedule_operations,
        )
    except (KeyError, TypeError) as exc:
        raise CegWmExperimentAdapterError(
            "method adapter configuration is incomplete"
        ) from exc


class CegWmExperimentAdapter:
    """Call the production method/runtime surface without owning algorithms."""

    def __init__(
        self,
        configuration: CegWmExperimentAdapterConfiguration,
        runtime_adapter: Sd35RuntimeAdapter | None = None,
    ) -> None:
        validated_configuration = _revalidate_adapter_configuration(
            configuration
        )
        if runtime_adapter is not None and type(runtime_adapter) is not Sd35RuntimeAdapter:
            raise CegWmExperimentAdapterError(
                "runtime_adapter must be Sd35RuntimeAdapter"
            )
        self._configuration = validated_configuration
        self._runtime_adapter = runtime_adapter
        self._bindings = {
            binding.responsibility: binding
            for binding in validated_configuration.component_bindings
        }
        self._key_schedule_operations = {
            binding.operation_id: binding
            for binding in validated_configuration.key_schedule_operations
        }

    @property
    def configuration(self) -> CegWmExperimentAdapterConfiguration:
        return self._configuration

    @_revalidate_configuration_before_call
    def require_no_runtime_binding(self) -> None:
        """Fail closed unless this adapter owns no hidden runtime execution path."""

        if self._runtime_adapter is not None:
            raise CegWmExperimentAdapterError(
                "method adapter must not retain a runtime binding"
            )

    @_revalidate_configuration_before_call
    def identify_key(self, root_key_text: str) -> ComponentCallObservation[RootKeyIdentity]:
        result = identify_root_key(root_key_text)
        return self._observe_key_schedule(
            "identify_key",
            result,
            result_identity_field="root_key_public_digest",
        )

    @_revalidate_configuration_before_call
    def derive_registered_key_stream(
        self,
        root_key_text: str,
        domain_fields: dict[str, object],
        shape: Sequence[int],
        *,
        distribution: str = "gaussian",
        config: KeyScheduleConfig = KeyScheduleConfig(),
    ) -> ComponentCallObservation[KeyStreamResult]:
        result = key_schedule_sha256_counter(
            root_key_text,
            domain_fields,
            shape,
            distribution=distribution,
            config=config,
        )
        return self._observe_key_schedule(
            "derive_registered_key_stream",
            result,
            result_identity_field="domain_digest",
        )

    @_revalidate_configuration_before_call
    def derive_wrong_key_stream(
        self,
        registered_root_key_public_digest: str,
        wrong_key_index: int,
        domain_fields: dict[str, object],
        shape: Sequence[int],
        *,
        distribution: str = "gaussian",
        config: KeyScheduleConfig = KeyScheduleConfig(),
    ) -> ComponentCallObservation[KeyStreamResult]:
        wrong_material: DerivedWrongKeyMaterial = derive_wrong_key_material(
            registered_root_key_public_digest,
            wrong_key_index,
        )
        result = derive_wrong_key_stream(
            wrong_material,
            domain_fields,
            shape,
            distribution=distribution,
            config=config,
        )
        return self._observe_key_schedule(
            "derive_wrong_key_stream",
            result,
            result_identity_field="domain_digest",
        )

    @_revalidate_configuration_before_call
    def derive_public_noise(
        self,
        domain_fields: dict[str, object],
        shape: Sequence[int],
        *,
        distribution: str = "gaussian",
        config: KeyScheduleConfig = KeyScheduleConfig(),
    ) -> ComponentCallObservation[KeyStreamResult]:
        result = derive_public_noise_stream(
            domain_fields,
            shape,
            distribution=distribution,
            config=config,
        )
        return self._observe_key_schedule(
            "derive_public_noise",
            result,
            result_identity_field="domain_digest",
        )

    @_revalidate_configuration_before_call
    def route_content(
        self,
        latent_shape: Sequence[int],
        *,
        mode: str,
        observations: RoutingObservations | None = None,
    ) -> ComponentCallObservation[ContentRoutingResult]:
        result = content_router(
            latent_shape,
            mode=mode,
            observations=observations,
        )
        return self._observe("content_router", result)

    @_revalidate_configuration_before_call
    def build_lf_carrier(
        self,
        detection_key: str,
        shape: Sequence[int],
        *,
        routing_result: ContentRoutingResult | None = None,
    ) -> ComponentCallObservation[LfCarrierResult]:
        result = lf_carrier(
            detection_key,
            shape,
            routing_result=routing_result,
        )
        return self._observe("lf_carrier", result)

    @_revalidate_configuration_before_call
    def build_hf_carrier(
        self,
        detection_key: str,
        shape: Sequence[int],
        *,
        routing_result: ContentRoutingResult | None = None,
    ) -> ComponentCallObservation[HfCarrierResult]:
        result = hf_carrier(
            detection_key,
            shape,
            routing_result=routing_result,
        )
        return self._observe("hf_carrier", result)

    @_revalidate_configuration_before_call
    def embed_content(
        self,
        latent_values: Sequence[float],
        hf_carrier_result: HfCarrierResult | None = None,
        *,
        lf_carrier_result: LfCarrierResult | None = None,
        mixing_coefficient: float | None = None,
        routing_result: ContentRoutingResult | None = None,
    ) -> ComponentCallObservation[ContentEmbeddingResult]:
        result = content_embedder(
            latent_values,
            hf_carrier_result,
            lf_carrier_result=lf_carrier_result,
            mixing_coefficient=mixing_coefficient,
            routing_result=routing_result,
        )
        return self._observe("content_embedder", result)

    @_revalidate_configuration_before_call
    def detect_lf(
        self,
        observation: LfDetectionObservation,
        detection_key: str,
    ) -> ComponentCallObservation[LfDetectionResult]:
        result = lf_detector(observation, detection_key)
        return self._observe("lf_detector", result)

    @_revalidate_configuration_before_call
    def detect_hf(
        self,
        observation: HfDetectionObservation,
        detection_key: str,
    ) -> ComponentCallObservation[HfDetectionResult]:
        result = hf_detector(observation, detection_key)
        return self._observe("hf_detector", result)

    @_revalidate_configuration_before_call
    def detect_content(
        self,
        hf_result: HfDetectionResult,
        lf_result: LfDetectionResult | None = None,
        *,
        hf_null: BranchNullCalibration | None = None,
        lf_null: BranchNullCalibration | None = None,
        combination: str | None = None,
        weight: float | None = None,
    ) -> ComponentCallObservation[ContentDetectionResult]:
        result = content_detector(
            hf_result,
            lf_result,
            hf_null=hf_null,
            lf_null=lf_null,
            combination=combination,
            weight=weight,
        )
        return self._observe("content_detector", result)

    @_revalidate_configuration_before_call
    def observe_qk_geometry(
        self,
        detection_image: torch.Tensor,
        detection_key: str,
    ) -> ComponentCallObservation[QkGeometrySyncResult]:
        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "qk_geometry_sync requires a prepared runtime adapter"
            )
        runtime_result = self._runtime_adapter.observe_detection_qk(
            detection_image
        )
        return self.synchronize_qk_observation(runtime_result, detection_key)

    @_revalidate_configuration_before_call
    def synchronize_qk_observation(
        self,
        runtime_result: RuntimeQkObservationResult,
        detection_key: str,
    ) -> ComponentCallObservation[QkGeometrySyncResult]:
        """Consume only the runtime public image-side Q/K result."""

        if type(runtime_result) is not RuntimeQkObservationResult:
            raise CegWmExperimentAdapterError(
                "qk_geometry_sync requires RuntimeQkObservationResult"
            )
        result = qk_geometry_sync(
            runtime_result.qk_layer_observations,
            detection_key,
        )
        if result.model_revision != runtime_result.model_revision:
            raise CegWmExperimentAdapterError(
                "runtime and method Q/K model revisions differ"
            )
        return self._observe(
            "qk_geometry_sync",
            result,
            upstream_runtime_identity=_runtime_observation_identity(runtime_result),
        )

    @_revalidate_configuration_before_call
    def estimate_geometric_transform(
        self,
        observation: QkGeometrySyncResult,
        detection_key: str | DerivedWrongKeyMaterial,
        *,
        epsilon_inlier: float,
    ) -> ComponentCallObservation[GeometricTransformEstimation]:
        result = geometric_transform_estimator(
            observation,
            detection_key,
            epsilon_inlier=epsilon_inlier,
        )
        return self._observe("geometric_transform_estimator", result)

    @_revalidate_configuration_before_call
    def assess_geometry_reliability(
        self,
        estimation: GeometricTransformEstimation,
        thresholds: GeometryReliabilityThresholds | None = None,
    ) -> ComponentCallObservation[GeometryReliabilityResult]:
        result = geometry_reliability(estimation, thresholds)
        return self._observe("geometry_reliability", result)

    @_revalidate_configuration_before_call
    def rectify_image(
        self,
        image: torch.Tensor,
        estimation: GeometricTransformEstimation,
        reliability: GeometryReliabilityResult,
    ) -> ComponentCallObservation[ImageRectificationResult]:
        result = image_rectifier(image, estimation, reliability)
        return self._observe("image_rectifier", result)

    @_revalidate_configuration_before_call
    def decide_conditional_recovery(
        self,
        image: torch.Tensor,
        detection_key: str,
        *,
        content_detector_binding: ContentDetectorBinding,
        thresholds: JointDecisionThresholds,
        geometry_estimation_operation: GeometryEstimationOperation,
        geometry_reliability_thresholds: GeometryReliabilityThresholds | None,
    ) -> ComponentCallObservation[ConditionalRecoveryResult]:
        result = conditional_recovery_decision(
            image,
            detection_key,
            content_detector_binding=content_detector_binding,
            thresholds=thresholds,
            geometry_estimation_operation=geometry_estimation_operation,
            geometry_reliability_thresholds=geometry_reliability_thresholds,
        )
        return self._observe("conditional_recovery_decision", result)

    def _observe(
        self,
        responsibility: str,
        result: T,
        *,
        upstream_runtime_identity: str | None = None,
        result_identity_field: str | None = None,
        public_callable: str | None = None,
    ) -> ComponentCallObservation[T]:
        _revalidate_adapter_configuration(self._configuration)
        binding = self._bindings[responsibility]
        identity_field = result_identity_field or binding.result_identity_field
        result_identity = getattr(result, identity_field, None)
        if type(result_identity) is not str or not result_identity:
            raise CegWmExperimentAdapterError(
                f"{responsibility} returned no observable result identity"
            )
        result_type = f"{type(result).__module__}.{type(result).__qualname__}"
        return ComponentCallObservation(
            responsibility=responsibility,
            public_callable=public_callable or binding.public_callable,
            adapter_config_digest=self._configuration.config_digest,
            result_type=result_type,
            result_identity=result_identity,
            upstream_runtime_identity=upstream_runtime_identity,
            result=result,
        )

    def _observe_key_schedule(
        self,
        operation_id: str,
        result: T,
        *,
        result_identity_field: str,
    ) -> ComponentCallObservation[T]:
        _revalidate_adapter_configuration(self._configuration)
        try:
            operation = self._key_schedule_operations[operation_id]
        except KeyError as exc:
            raise CegWmExperimentAdapterError(
                "key schedule operation is not registered"
            ) from exc
        return self._observe(
            "key_schedule",
            result,
            result_identity_field=result_identity_field,
            public_callable=operation.public_callable,
        )


def _runtime_observation_identity(result: RuntimeQkObservationResult) -> str:
    if type(result) is not RuntimeQkObservationResult:
        raise CegWmExperimentAdapterError(
            "runtime Q/K observation has an invalid result type"
        )
    return _canonical_digest(
        {
            "candidate_id": result.candidate_id,
            "model_revision": result.model_revision,
            "public_noise_domain_digest": result.public_noise_domain_digest,
            "public_noise_values_float32_be_sha256": (
                result.public_noise_values_float32_be_sha256
            ),
            "runtime_config_digest": result.runtime_config_digest,
            "qk_layers": [
                {
                    "layer_name": observation.layer_name,
                    "operator_identity": observation.operator_identity,
                }
                for observation in result.qk_layer_observations
            ],
        }
    )
