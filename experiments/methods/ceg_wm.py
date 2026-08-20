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
    ContrastiveLfDetectionResult,
    ContrastiveLfNullAsset,
    ContrastiveLfRawObservation,
    DerivedWrongKeyMaterial,
    GeometricTransformEstimation,
    GeometryEstimationOperation,
    GeometryReliabilityResult,
    GeometryReliabilityThresholds,
    GeometrySynchronizationWriteResult,
    HfCarrierResult,
    HfDetectionObservation,
    HfDetectionResult,
    HfPopulationNullAsset,
    HfPopulationStandardizedResult,
    ImageRectificationResult,
    JointDecisionThresholds,
    KeyScheduleConfig,
    KeyStreamResult,
    LfCarrierResult,
    LfDetectionObservation,
    LfDetectionResult,
    LfNullWhitenedDetectionResult,
    LfNullWhiteningAsset,
    NullScoreRecord,
    PreparedLfWhitenedObservation,
    PreparedLfWhitenedTemplate,
    QkGeometrySyncResult,
    RootKeyIdentity,
    RoutingObservations,
    SemanticTextureBranchNullCalibration,
    SemanticTextureContentDetectionResult,
    SemanticTextureHfDetectionResult,
    SemanticTextureLfWhiteningAsset,
    SemanticTextureLfDetectionResult,
    SemanticTextureRoutingObservations,
    SemanticTextureRoutingResult,
    content_detector,
    content_embedder,
    content_router,
    contrastive_lf_carrier,
    contrastive_lf_detector,
    contrastive_lf_raw_observation,
    semantic_texture_content_detector,
    semantic_texture_content_arm_embedder,
    semantic_texture_content_embedder,
    semantic_texture_content_router,
    semantic_texture_hf_detector,
    semantic_texture_lf_detector,
    conditional_recovery_decision,
    differentiable_qk_relation_objective,
    derive_public_noise_stream,
    derive_wrong_key_material,
    derive_wrong_key_stream,
    geometric_transform_estimator,
    geometry_synchronization_write,
    geometry_reliability,
    hf_carrier,
    hf_detector,
    identify_root_key,
    image_rectifier,
    key_schedule_sha256_counter,
    lf_carrier,
    lf_detector,
    lf_null_whitened_matched_detector,
    qk_geometry_sync,
    standardize_hf_population_score,
)
from runtime import (
    ContentWriteGeometrySuffixResult,
    ContentWriteVaeResult,
    RuntimeActualQkSuffixResult,
    RuntimeDifferentiableQkSuffixResult,
    RuntimeQkObservationResult,
    InspyrenetSemanticRuntime,
    RuntimeSemanticTextureDetectionObservationResult,
    RuntimeSemanticTextureObservationResult,
    Sd35RuntimeAdapter,
    materialize_ordinary_rgb8_snapshot,
)
from experiments.protocol.semantic_texture_soft_detector_assets import (
    SemanticTextureBranchNullPayload,
    SemanticTextureBranchNullRecordPayload,
    SemanticTextureSoftDetectorAssetBundle,
    SemanticTextureSoftDetectorAssetProtocolError,
    create_asset_bundle,
)


DEFAULT_COMPONENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "experiments"
    / "internal_execution_components.json"
)
QK_OBSERVATION_PUBLIC_CALLABLE = (
    "runtime.Sd35RuntimeAdapter.observe_detection_qk"
    " -> main.qk_geometry_sync"
)
QK_SYNCHRONIZATION_WRITE_PUBLIC_CALLABLE = (
    "runtime.Sd35RuntimeAdapter."
    "execute_content_write_and_capture_geometry_suffix"
    " -> runtime.Sd35RuntimeAdapter."
    "observe_differentiable_qk_from_generation_suffix"
    " -> main.differentiable_qk_relation_objective"
    " -> main.geometry_synchronization_write"
    " -> runtime.Sd35RuntimeAdapter.materialize_geometry_candidate"
    " -> runtime.Sd35RuntimeAdapter."
    "observe_actual_qk_from_generation_suffix"
    " -> main.qk_geometry_sync"
)
SEMANTIC_TEXTURE_DETECTION_PUBLIC_CALLABLE = (
    "runtime.Sd35RuntimeAdapter.observe_semantic_texture_detection"
    " -> main.semantic_texture_content_router"
    " -> main.semantic_texture_hf_detector"
    " + main.semantic_texture_lf_detector"
    " -> main.semantic_texture_content_detector"
)
SEMANTIC_TEXTURE_EMBEDDING_PUBLIC_CALLABLE = (
    "runtime.Sd35RuntimeAdapter callback-18 transient VAE RGB8 observation"
    " -> main.semantic_texture_content_router"
    " -> main.hf_carrier + main.lf_carrier"
    " -> main.semantic_texture_content_embedder"
    " -> runtime actual-dtype reconciliation"
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
        QK_SYNCHRONIZATION_WRITE_PUBLIC_CALLABLE,
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


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismContentArmWriteResult:
    """Uniform public wrapper for a soft-route mechanism validation arm materialization."""

    arm_id: str
    content_write_result: ContentWriteVaeResult
    write_identity: str


@dataclass(frozen=True, slots=True)
class ContrastiveLfContentArmWriteResult:
    """Public Stage-A write wrapper retaining the budgeted runtime result."""

    arm_id: str
    candidate_id: str
    content_write_result: ContentWriteVaeResult
    write_identity: str


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


@dataclass(frozen=True, slots=True)
class QkSynchronizationWriteExecutionResult:
    """Ephemeral public results from one real Q/K synchronization write."""

    content_write_result: ContentWriteVaeResult
    pre_write_observation: QkGeometrySyncResult
    geometry_write_result: GeometrySynchronizationWriteResult
    accepted_post_write_observation: QkGeometrySyncResult | None
    accepted_actual_runtime_result: RuntimeActualQkSuffixResult | None
    gradient_objective: float
    gradient_l2: float
    runtime_config_digest: str
    callback_index: int

    @property
    def geometry_config_digest(self) -> str:
        return self.pre_write_observation.geometry_config_digest


@dataclass(frozen=True, slots=True)
class SemanticTextureCleanPrimaryNullObservation:
    """One clean ordinary RGB8 observation for Phase-B asset fitting only."""

    detection_image_rgb8: torch.Tensor
    runtime_detection: RuntimeSemanticTextureDetectionObservationResult
    routing_result: SemanticTextureRoutingResult
    lf_carrier_config_digest: str


@dataclass(frozen=True, slots=True)
class SemanticTexturePrimaryNullBranchObservation:
    """Paired raw public branch observations before diagnostic CDF construction."""

    hf_result: SemanticTextureHfDetectionResult
    lf_result: SemanticTextureLfDetectionResult


def serialize_semantic_texture_soft_detector_asset_bundle(
    *,
    whitening_manifest_digest: str,
    branch_null_manifest_digest: str,
    whitening_asset: SemanticTextureLfWhiteningAsset,
    hf_null: SemanticTextureBranchNullCalibration,
    lf_null: SemanticTextureBranchNullCalibration,
) -> SemanticTextureSoftDetectorAssetBundle:
    """Bind existing public detector assets to a pure Phase-B transport bundle."""

    def calibration_payload(
        calibration: SemanticTextureBranchNullCalibration,
    ) -> SemanticTextureBranchNullPayload:
        return SemanticTextureBranchNullPayload(
            branch=calibration.branch,
            detector_identity=calibration.detector_identity,
            partition_identity=calibration.partition_identity,
            records=tuple(
                SemanticTextureBranchNullRecordPayload(
                    score_float64_hex=record.score.hex(),
                    source_cluster_id=record.source_cluster_id,
                    sample_id=record.sample_id,
                )
                for record in calibration.records
            ),
        )

    try:
        whitening_asset.validate()
        hf_null_payload = calibration_payload(hf_null)
        lf_null_payload = calibration_payload(lf_null)
        return create_asset_bundle(
            whitening_manifest_digest=whitening_manifest_digest,
            branch_null_manifest_digest=branch_null_manifest_digest,
            lf_carrier_config_digest=whitening_asset.lf_carrier_config_digest,
            whitening_asset_payload=whitening_asset.canonical_payload,
            whitening_asset_digest=whitening_asset.whitening_asset_digest,
            hf_null_payload=hf_null_payload,
            lf_null_payload=lf_null_payload,
        )
    except (SemanticTextureSoftDetectorAssetProtocolError, ValueError) as exc:
        raise CegWmExperimentAdapterError(
            "semantic-texture detector asset serialization is invalid"
        ) from exc


def materialize_semantic_texture_soft_detector_asset_bundle(
    bundle: SemanticTextureSoftDetectorAssetBundle,
) -> tuple[
    SemanticTextureLfWhiteningAsset,
    SemanticTextureBranchNullCalibration,
    SemanticTextureBranchNullCalibration,
]:
    """Materialize a pure validated bundle through the existing public main API."""

    if type(bundle) is not SemanticTextureSoftDetectorAssetBundle:
        raise CegWmExperimentAdapterError("semantic-texture asset bundle type is invalid")
    try:
        bundle.validate()
        whitening = SemanticTextureLfWhiteningAsset.from_canonical_payload(
            bundle.whitening_asset_payload,
            whitening_asset_digest=bundle.whitening_asset_digest,
        )
        whitening.validate()

        def calibration(
            payload: SemanticTextureBranchNullPayload,
        ) -> SemanticTextureBranchNullCalibration:
            return SemanticTextureBranchNullCalibration(
                branch=payload.branch,
                detector_identity=payload.detector_identity,
                partition_identity=payload.partition_identity,
                records=tuple(
                    NullScoreRecord(
                        score=float.fromhex(record.score_float64_hex),
                        source_cluster_id=record.source_cluster_id,
                        sample_id=record.sample_id,
                    )
                    for record in payload.records
                ),
            )

        hf_null, lf_null = calibration(bundle.hf_null_payload), calibration(
            bundle.lf_null_payload
        )
    except (SemanticTextureSoftDetectorAssetProtocolError, TypeError, ValueError) as exc:
        raise CegWmExperimentAdapterError(
            "semantic-texture detector asset materialization is invalid"
        ) from exc
    if (
        whitening.fit_manifest_sha256 != bundle.whitening_manifest_digest
        or whitening.lf_carrier_config_digest != bundle.lf_carrier_config_digest
        or hf_null.branch != "hf"
        or lf_null.branch != "lf"
        or hf_null.partition_identity != bundle.branch_null_manifest_digest
        or lf_null.partition_identity != bundle.branch_null_manifest_digest
        or len(hf_null.records) != 32
        or len(lf_null.records) != 32
    ):
        raise CegWmExperimentAdapterError("semantic-texture detector asset binding drifted")
    return whitening, hf_null, lf_null


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
    def route_semantic_texture(
        self,
        latent_shape: Sequence[int],
        *,
        mode: str,
        observations: SemanticTextureRoutingObservations | object | None = None,
    ) -> ComponentCallObservation[SemanticTextureRoutingResult]:
        result = semantic_texture_content_router(
            latent_shape,
            mode=mode,
            observations=observations,
        )
        return self._observe(
            "content_router",
            result,
            public_callable="main.semantic_texture_content_router",
        )

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
    def execute_semantic_texture_content_write_and_vae(
        self,
        base_latent: torch.Tensor,
        detection_key: str,
        semantic_runtime: InspyrenetSemanticRuntime,
    ) -> ComponentCallObservation[object]:
        """At the registered write callback index, decode the current callback
        latent once through the transient VAE to ordinary RGB8, strictly observe
        semantic texture from that same RGB8, compose the existing main router,
        independent HF/LF carriers and embedder, and preserve the existing
        embedder-owned actual-dtype reconciliation boundary.
        """

        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "semantic-texture embedding requires a prepared runtime adapter"
            )

        def compose_from_runtime_observation(
            latent_values: tuple[float, ...],
            latent_shape: tuple[int, ...],
            runtime_observation: object,
        ) -> tuple[SemanticTextureRoutingResult, ContentEmbeddingResult]:
            if (
                type(runtime_observation)
                is not RuntimeSemanticTextureObservationResult
            ):
                raise CegWmExperimentAdapterError(
                    "semantic-texture composition requires the runtime observation"
                )
            route = semantic_texture_content_router(
                latent_shape,
                mode="routing_semantic_texture_soft",
                observations=runtime_observation.observations,
            )
            hf_carrier_result = hf_carrier(
                detection_key,
                route.latent_shape,
                routing_result=route,
            )
            lf_carrier_result = lf_carrier(
                detection_key,
                route.latent_shape,
                routing_result=route,
            )
            embedding = semantic_texture_content_embedder(
                latent_values,
                hf_carrier_result,
                lf_carrier_result=lf_carrier_result,
                routing_result=route,
            )
            return route, embedding

        result = (
            self._runtime_adapter.execute_semantic_texture_content_write_and_vae(
                base_latent,
                semantic_runtime,
                compose_from_runtime_observation,
            )
        )
        witness = getattr(result, "witness", None)
        upstream_runtime_identity = getattr(
            witness,
            "semantic_observation_identity",
            None,
        )
        return self._observe(
            "content_embedder",
            result,
            upstream_runtime_identity=upstream_runtime_identity,
            result_identity_field="witness_identity",
            public_callable=SEMANTIC_TEXTURE_EMBEDDING_PUBLIC_CALLABLE,
        )

    @_revalidate_configuration_before_call
    def execute_semantic_texture_content_arm_write_and_vae(
        self,
        base_latent: torch.Tensor,
        detection_key: str,
        semantic_runtime: InspyrenetSemanticRuntime,
        *,
        arm_id: str,
    ) -> ComponentCallObservation[object]:
        """Execute one registered soft-route mechanism validation write arm through public surfaces."""

        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "semantic-texture arm embedding requires a prepared runtime adapter"
            )
        if arm_id not in {
            "hf_only",
            "lf_only",
            "semantic_texture_soft_routed",
            "semantic_texture_route_disabled",
        }:
            raise CegWmExperimentAdapterError("semantic-texture arm is not registered")

        def compose_from_runtime_observation(
            latent_values: tuple[float, ...],
            latent_shape: tuple[int, ...],
            runtime_observation: object,
        ) -> tuple[SemanticTextureRoutingResult, ContentEmbeddingResult]:
            if type(runtime_observation) is not RuntimeSemanticTextureObservationResult:
                raise CegWmExperimentAdapterError(
                    "semantic-texture arm composition requires the runtime observation"
                )
            route_mode = (
                "routing_semantic_texture_soft"
                if arm_id != "semantic_texture_route_disabled"
                else "semantic_texture_route_disabled"
            )
            route = semantic_texture_content_router(
                latent_shape,
                mode=route_mode,
                observations=(
                    runtime_observation.observations
                    if arm_id != "semantic_texture_route_disabled"
                    else None
                ),
            )
            hf_carrier_result = hf_carrier(
                detection_key,
                route.latent_shape,
                routing_result=route,
            )
            lf_carrier_result = lf_carrier(
                detection_key,
                route.latent_shape,
                routing_result=route,
            )
            return route, semantic_texture_content_arm_embedder(
                latent_values,
                hf_carrier_result,
                lf_carrier_result=lf_carrier_result,
                routing_result=route,
                arm_id=arm_id,
            )

        if arm_id == "semantic_texture_soft_routed":
            result = self._runtime_adapter.execute_semantic_texture_content_write_and_vae(
                base_latent,
                semantic_runtime,
                compose_from_runtime_observation,
            )
            witness = getattr(result, "witness", None)
            return self._observe(
                "content_embedder",
                result,
                upstream_runtime_identity=getattr(
                    witness,
                    "semantic_observation_identity",
                    None,
                ),
                result_identity_field="witness_identity",
                public_callable=SEMANTIC_TEXTURE_EMBEDDING_PUBLIC_CALLABLE,
            )

        latent_shape = tuple(int(value) for value in base_latent.shape)

        def compose_without_semantic_observation(
            latent_values: tuple[float, ...],
        ) -> ContentEmbeddingResult:
            route = semantic_texture_content_router(
                latent_shape,
                mode="semantic_texture_route_disabled",
                observations=None,
            )
            hf_carrier_result = hf_carrier(
                detection_key,
                route.latent_shape,
                routing_result=route,
            )
            lf_carrier_result = lf_carrier(
                detection_key,
                route.latent_shape,
                routing_result=route,
            )
            return semantic_texture_content_arm_embedder(
                latent_values,
                hf_carrier_result,
                lf_carrier_result=lf_carrier_result,
                routing_result=route,
                arm_id=arm_id,
            )

        content_write = self._runtime_adapter.execute_content_write_and_vae(
            base_latent,
            compose_without_semantic_observation,
        )
        write_identity = (
            content_write.content_materialization.materialization_replay_identity
        )
        result = SoftRouteMechanismContentArmWriteResult(
            arm_id=arm_id,
            content_write_result=content_write,
            write_identity=write_identity,
        )
        return self._observe(
            "content_embedder",
            result,
            result_identity_field="write_identity",
            public_callable=(
                "runtime.Sd35RuntimeAdapter.execute_content_write_and_vae"
                " -> main.semantic_texture_content_arm_embedder"
            ),
        )

    @_revalidate_configuration_before_call
    def observe_semantic_texture_candidate_branches(
        self,
        detection_image_rgb8: torch.Tensor,
        detection_key: str | DerivedWrongKeyMaterial,
        semantic_runtime: InspyrenetSemanticRuntime,
        whitening_asset: SemanticTextureLfWhiteningAsset,
    ) -> SemanticTexturePrimaryNullBranchObservation:
        """Read raw public soft-route branch scores without CDF/decision logic."""

        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "semantic-texture branch observation requires a prepared runtime adapter"
            )
        runtime_result = self._runtime_adapter.observe_semantic_texture_detection(
            detection_image_rgb8,
            semantic_runtime,
        )
        if type(runtime_result) is not RuntimeSemanticTextureDetectionObservationResult:
            raise CegWmExperimentAdapterError(
                "runtime returned an invalid semantic-texture observation"
            )
        route = semantic_texture_content_router(
            runtime_result.hf_observation.shape,
            mode="routing_semantic_texture_soft",
            observations=runtime_result.semantic_texture.observations,
        )
        return SemanticTexturePrimaryNullBranchObservation(
            hf_result=semantic_texture_hf_detector(
                runtime_result.hf_observation,
                detection_key,
                route,
            ),
            lf_result=semantic_texture_lf_detector(
                runtime_result.lf_observation,
                detection_key,
                route,
                whitening_asset,
            ),
        )

    @_revalidate_configuration_before_call
    def combine_semantic_texture_candidate_branches(
        self,
        branches: SemanticTexturePrimaryNullBranchObservation,
        *,
        hf_null: SemanticTextureBranchNullCalibration,
        lf_null: SemanticTextureBranchNullCalibration,
    ) -> ComponentCallObservation[SemanticTextureContentDetectionResult]:
        """Apply existing public diagnostic CDFs to already observed branches."""

        if type(branches) is not SemanticTexturePrimaryNullBranchObservation:
            raise CegWmExperimentAdapterError(
                "semantic-texture branch observation identity is invalid"
            )
        result = semantic_texture_content_detector(
            branches.hf_result,
            branches.lf_result,
            hf_null=hf_null,
            lf_null=lf_null,
        )
        return self._observe(
            "content_detector",
            result,
            public_callable="main.semantic_texture_content_detector",
        )

    @_revalidate_configuration_before_call
    def build_semantic_texture_provisional_calibrations(
        self,
        observations: Sequence[
            tuple[str, str, SemanticTexturePrimaryNullBranchObservation]
        ],
        *,
        partition_identity: str,
    ) -> tuple[SemanticTextureBranchNullCalibration, SemanticTextureBranchNullCalibration]:
        """Convert declared clean branch observations into public CDF objects."""

        if len(observations) != 32 or not partition_identity:
            raise CegWmExperimentAdapterError(
                "semantic-texture provisional calibration inputs are invalid"
            )
        try:
            first = observations[0][2]
            hf_records = tuple(
                NullScoreRecord(
                    score=branches.hf_result.hf_score,
                    source_cluster_id=cluster_id,
                    sample_id=sample_id,
                )
                for cluster_id, sample_id, branches in observations
            )
            lf_records = tuple(
                NullScoreRecord(
                    score=branches.lf_result.lf_score,
                    source_cluster_id=cluster_id,
                    sample_id=sample_id,
                )
                for cluster_id, sample_id, branches in observations
            )
            return (
                SemanticTextureBranchNullCalibration(
                    branch="hf",
                    detector_identity=first.hf_result.detector_identity,
                    partition_identity=partition_identity,
                    records=hf_records,
                ),
                SemanticTextureBranchNullCalibration(
                    branch="lf",
                    detector_identity=first.lf_result.detector_identity,
                    partition_identity=partition_identity,
                    records=lf_records,
                ),
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise CegWmExperimentAdapterError(
                "semantic-texture provisional calibration is invalid"
            ) from exc

    @_revalidate_configuration_before_call
    def materialize_semantic_texture_provisional_calibrations(
        self,
        *,
        hf_detector_identity: str,
        lf_detector_identity: str,
        partition_identity: str,
        hf_records: Sequence[tuple[str, str, float]],
        lf_records: Sequence[tuple[str, str, float]],
    ) -> tuple[SemanticTextureBranchNullCalibration, SemanticTextureBranchNullCalibration]:
        """Materialize an authenticated soft-route mechanism validation provisional CDF without refitting."""

        if (
            len(hf_records) != 32
            or len(lf_records) != 32
            or not partition_identity
        ):
            raise CegWmExperimentAdapterError("provisional calibration authority is invalid")
        try:
            hf_null = SemanticTextureBranchNullCalibration(
                branch="hf",
                detector_identity=hf_detector_identity,
                partition_identity=partition_identity,
                records=tuple(NullScoreRecord(score=score, source_cluster_id=cluster, sample_id=sample) for cluster, sample, score in hf_records),
            )
            lf_null = SemanticTextureBranchNullCalibration(
                branch="lf",
                detector_identity=lf_detector_identity,
                partition_identity=partition_identity,
                records=tuple(NullScoreRecord(score=score, source_cluster_id=cluster, sample_id=sample) for cluster, sample, score in lf_records),
            )
            return hf_null, lf_null
        except (TypeError, ValueError) as exc:
            raise CegWmExperimentAdapterError("provisional calibration payload is invalid") from exc

    @_revalidate_configuration_before_call
    def materialize_semantic_texture_written_rgb8(
        self,
        write_observation: ComponentCallObservation[object],
    ) -> torch.Tensor:
        """Expose the ordinary RGB8 form of the exact public write result."""

        runtime_result = getattr(write_observation, "result", None)
        content_write = getattr(runtime_result, "content_write_result", None)
        image = getattr(content_write, "watermarked_image", None)
        if image is None:
            raise CegWmExperimentAdapterError(
                "semantic-texture write RGB8 observation is unavailable"
            )
        return materialize_ordinary_rgb8_snapshot(image)

    @_revalidate_configuration_before_call
    def execute_contrastive_lf_content_arm_write_and_vae(
        self,
        base_latent: torch.Tensor,
        detection_key: str,
        *,
        arm_id: str,
    ) -> ComponentCallObservation[ContrastiveLfContentArmWriteResult]:
        """Execute HF or either real Stage-A LF-only carrier through the public runtime."""

        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "contrastive LF embedding requires a prepared runtime adapter"
            )
        candidate_by_arm = {
            "hf_only": "hf_sparse_tail",
            "multiscale_low_frequency_only": "lf_multiscale_lowpass_contrastive",
            "single_scale_low_frequency_only": "lf_five_by_five_lowpass_contrastive",
        }
        if arm_id not in candidate_by_arm:
            raise CegWmExperimentAdapterError("contrastive LF write arm is not registered")

        def embed(latent_values: tuple[float, ...]) -> ContentEmbeddingResult:
            shape = tuple(int(size) for size in base_latent.shape)
            if arm_id == "hf_only":
                return content_embedder(
                    latent_values,
                    hf_carrier(detection_key, shape),
                )
            carrier = contrastive_lf_carrier(
                detection_key,
                shape,
                candidate_id=candidate_by_arm[arm_id],
            )
            return content_embedder(
                latent_values,
                lf_carrier_result=carrier.as_embedding_carrier(),
            )

        runtime_result = self._runtime_adapter.execute_content_write_and_vae(
            base_latent, embed
        )
        embedding = runtime_result.content_materialization_result.embedding_result
        identity = _canonical_digest(
            {
                "arm_id": arm_id,
                "candidate_id": candidate_by_arm[arm_id],
                "embedding_result_identity": embedding.embedding_result_identity,
                "materialization_replay_identity": runtime_result.content_materialization_result.observation.materialization_replay_identity,
                "runtime_config_digest": runtime_result.runtime_config_digest,
            }
        )
        wrapped = ContrastiveLfContentArmWriteResult(
            arm_id=arm_id,
            candidate_id=candidate_by_arm[arm_id],
            content_write_result=runtime_result,
            write_identity=identity,
        )
        return self._observe(
            "content_embedder",
            wrapped,
            result_identity_field="write_identity",
            public_callable=(
                "runtime.Sd35RuntimeAdapter.execute_content_write_and_vae"
                " -> main.contrastive_lf_carrier -> main.content_embedder"
            ),
        )

    @_revalidate_configuration_before_call
    def observe_contrastive_lf_candidate(
        self,
        detection_image_rgb8: torch.Tensor,
        detection_key: str | DerivedWrongKeyMaterial,
        null_asset: ContrastiveLfNullAsset,
    ) -> ComponentCallObservation[ContrastiveLfDetectionResult]:
        """Blindly rebuild one candidate from the current ordinary RGB8 image."""

        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "contrastive LF detection requires a prepared runtime adapter"
            )
        runtime_observation = self._runtime_adapter.observe_public_rgb8_vae(
            detection_image_rgb8
        )
        latent = runtime_observation.detection_latent.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        observation = LfDetectionObservation.from_public_image_encoding(
            tuple(float(value) for value in latent.reshape(-1).tolist()),
            tuple(int(size) for size in latent.shape),
        )
        raw: ContrastiveLfRawObservation = contrastive_lf_raw_observation(
            observation, detection_key, candidate_id=null_asset.candidate_id
        )
        result = contrastive_lf_detector(raw, null_asset)
        return self._observe(
            "lf_detector",
            result,
            upstream_runtime_identity=runtime_observation.observation_identity,
            public_callable=(
                "runtime.Sd35RuntimeAdapter.observe_public_rgb8_vae"
                " -> main.contrastive_lf_raw_observation"
                " -> main.contrastive_lf_detector"
            ),
        )

    @_revalidate_configuration_before_call
    def observe_contrastive_lf_raw(
        self,
        detection_image_rgb8: torch.Tensor,
        detection_key: str | DerivedWrongKeyMaterial,
        *,
        candidate_id: str,
    ) -> ContrastiveLfRawObservation:
        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError("contrastive LF raw observation requires runtime")
        runtime_observation = self._runtime_adapter.observe_public_rgb8_vae(detection_image_rgb8)
        latent = runtime_observation.detection_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
        observation = LfDetectionObservation.from_public_image_encoding(
            tuple(float(value) for value in latent.reshape(-1).tolist()),
            tuple(int(size) for size in latent.shape),
        )
        return contrastive_lf_raw_observation(
            observation, detection_key, candidate_id=candidate_id
        )

    @_revalidate_configuration_before_call
    def observe_stage_a_hf_raw(
        self,
        detection_image_rgb8: torch.Tensor,
        detection_key: str | DerivedWrongKeyMaterial,
    ) -> HfDetectionResult:
        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError("Stage-A HF raw observation requires runtime")
        runtime_observation = self._runtime_adapter.observe_public_rgb8_vae(detection_image_rgb8)
        latent = runtime_observation.detection_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
        observation = HfDetectionObservation.from_public_image_encoding(
            tuple(float(value) for value in latent.reshape(-1).tolist()),
            tuple(int(size) for size in latent.shape),
        )
        return hf_detector(observation, detection_key)

    @_revalidate_configuration_before_call
    def observe_stage_a_hf(
        self,
        detection_image_rgb8: torch.Tensor,
        detection_key: str | DerivedWrongKeyMaterial,
        null_asset: HfPopulationNullAsset,
    ) -> ComponentCallObservation[HfPopulationStandardizedResult]:
        """Apply the fresh Stage-A HF population asset on public RGB8 VAE evidence."""

        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError("Stage-A HF detection requires a prepared runtime")
        runtime_observation = self._runtime_adapter.observe_public_rgb8_vae(
            detection_image_rgb8
        )
        latent = runtime_observation.detection_latent.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        observation = HfDetectionObservation.from_public_image_encoding(
            tuple(float(value) for value in latent.reshape(-1).tolist()),
            tuple(int(size) for size in latent.shape),
        )
        raw = hf_detector(observation, detection_key)
        result = standardize_hf_population_score(raw, null_asset)
        return self._observe(
            "hf_detector",
            result,
            upstream_runtime_identity=runtime_observation.observation_identity,
            public_callable=(
                "runtime.Sd35RuntimeAdapter.observe_public_rgb8_vae"
                " -> main.hf_detector -> main.standardize_hf_population_score"
            ),
        )

    @_revalidate_configuration_before_call
    def derive_semantic_texture_wrong_key_material(
        self,
        registered_root_key_public_digest: str,
        wrong_key_index: int,
    ) -> DerivedWrongKeyMaterial:
        """Expose exactly one public wrong-key control identity for soft-route mechanism validation."""

        try:
            return derive_wrong_key_material(
                registered_root_key_public_digest,
                wrong_key_index,
            )
        except (TypeError, ValueError) as exc:
            raise CegWmExperimentAdapterError("wrong-key control identity is invalid") from exc

    @_revalidate_configuration_before_call
    def detect_lf(
        self,
        observation: LfDetectionObservation,
        detection_key: str | DerivedWrongKeyMaterial,
    ) -> ComponentCallObservation[LfDetectionResult]:
        result = lf_detector(observation, detection_key)
        return self._observe("lf_detector", result)

    @_revalidate_configuration_before_call
    def detect_lf_null_whitened(
        self,
        observation: LfDetectionObservation,
        detection_key: str | DerivedWrongKeyMaterial,
        whitening_asset: LfNullWhiteningAsset,
        *,
        prepared_observation: PreparedLfWhitenedObservation | None = None,
        prepared_template: PreparedLfWhitenedTemplate | None = None,
    ) -> ComponentCallObservation[LfNullWhitenedDetectionResult]:
        """Delegate the explicit no-fallback LF whitening candidate."""

        result = lf_null_whitened_matched_detector(
            observation,
            detection_key,
            whitening_asset,
            prepared_observation=prepared_observation,
            prepared_template=prepared_template,
        )
        return self._observe(
            "lf_detector",
            result,
            public_callable="main.lf_null_whitened_matched_detector",
        )

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
    def prepare_semantic_texture_clean_primary_null(
        self,
        base_latent: torch.Tensor,
        detection_key: str,
        semantic_runtime: InspyrenetSemanticRuntime,
    ) -> SemanticTextureCleanPrimaryNullObservation:
        """Materialize one clean RGB8 input for the declared Phase-B partitions."""

        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "semantic-texture primary-null preparation requires a prepared runtime"
            )
        clean = self._runtime_adapter.execute_clean_image_and_vae_observation(
            base_latent
        )
        image_rgb8 = materialize_ordinary_rgb8_snapshot(clean.clean_image)
        runtime_detection = self._runtime_adapter.observe_semantic_texture_detection(
            image_rgb8,
            semantic_runtime,
        )
        route = semantic_texture_content_router(
            runtime_detection.hf_observation.shape,
            mode="routing_semantic_texture_soft",
            observations=runtime_detection.semantic_texture.observations,
        )
        carrier = lf_carrier(
            detection_key,
            route.latent_shape,
        )
        return SemanticTextureCleanPrimaryNullObservation(
            detection_image_rgb8=image_rgb8,
            runtime_detection=runtime_detection,
            routing_result=route,
            lf_carrier_config_digest=carrier.carrier_config_digest,
        )

    @_revalidate_configuration_before_call
    def observe_semantic_texture_primary_null_branches(
        self,
        prepared: SemanticTextureCleanPrimaryNullObservation,
        detection_key: str,
        whitening_asset: SemanticTextureLfWhiteningAsset,
    ) -> SemanticTexturePrimaryNullBranchObservation:
        """Read paired raw soft-route scores for the declared null partition."""

        if type(prepared) is not SemanticTextureCleanPrimaryNullObservation:
            raise CegWmExperimentAdapterError(
                "semantic-texture primary-null preparation identity is invalid"
            )
        hf_result = semantic_texture_hf_detector(
            prepared.runtime_detection.hf_observation,
            detection_key,
            prepared.routing_result,
        )
        lf_result = semantic_texture_lf_detector(
            prepared.runtime_detection.lf_observation,
            detection_key,
            prepared.routing_result,
            whitening_asset,
        )
        return SemanticTexturePrimaryNullBranchObservation(
            hf_result=hf_result,
            lf_result=lf_result,
        )

    @_revalidate_configuration_before_call
    def detect_semantic_texture_candidate(
        self,
        detection_image_rgb8: torch.Tensor,
        detection_key: str | DerivedWrongKeyMaterial,
        semantic_runtime: InspyrenetSemanticRuntime,
        whitening_asset: SemanticTextureLfWhiteningAsset,
        *,
        hf_null: SemanticTextureBranchNullCalibration,
        lf_null: SemanticTextureBranchNullCalibration,
    ) -> ComponentCallObservation[SemanticTextureContentDetectionResult]:
        """Traverse the real public runtime and the fixed candidate max API."""

        if self._runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "semantic-texture detection requires a prepared runtime adapter"
            )
        runtime_result = self._runtime_adapter.observe_semantic_texture_detection(
            detection_image_rgb8,
            semantic_runtime,
        )
        if type(runtime_result) is not RuntimeSemanticTextureDetectionObservationResult:
            raise CegWmExperimentAdapterError(
                "runtime returned an invalid semantic-texture observation"
            )
        route = semantic_texture_content_router(
            runtime_result.hf_observation.shape,
            mode="routing_semantic_texture_soft",
            observations=runtime_result.semantic_texture.observations,
        )
        hf_result = semantic_texture_hf_detector(
            runtime_result.hf_observation,
            detection_key,
            route,
        )
        lf_result = semantic_texture_lf_detector(
            runtime_result.lf_observation,
            detection_key,
            route,
            whitening_asset,
        )
        result = semantic_texture_content_detector(
            hf_result,
            lf_result,
            hf_null=hf_null,
            lf_null=lf_null,
        )
        return self._observe(
            "content_detector",
            result,
            upstream_runtime_identity=runtime_result.observation_identity,
            public_callable=SEMANTIC_TEXTURE_DETECTION_PUBLIC_CALLABLE,
        )

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
        synchronized = self.synchronize_qk_observation(
            runtime_result,
            detection_key,
        )
        return ComponentCallObservation(
            responsibility=synchronized.responsibility,
            public_callable=QK_OBSERVATION_PUBLIC_CALLABLE,
            adapter_config_digest=synchronized.adapter_config_digest,
            result_type=synchronized.result_type,
            result_identity=synchronized.result_identity,
            upstream_runtime_identity=synchronized.upstream_runtime_identity,
            result=synchronized.result,
        )

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
            model_revision=runtime_result.model_revision,
        )
        return self._observe(
            "qk_geometry_sync",
            result,
            upstream_runtime_identity=_runtime_observation_identity(runtime_result),
            public_callable="main.qk_geometry_sync",
        )

    @_revalidate_configuration_before_call
    def execute_qk_synchronization_write(
        self,
        base_latent: torch.Tensor,
        content_embedding_operation: Callable[
            [tuple[float, ...]],
            ContentEmbeddingResult,
        ],
        content_directions: Sequence[torch.Tensor],
        *,
        geometry_ratio: float,
        detection_key: str | DerivedWrongKeyMaterial,
    ) -> ComponentCallObservation[QkSynchronizationWriteExecutionResult]:
        """Delegate one ratio to the frozen public Q/K write execution chain."""

        runtime_adapter = self._runtime_adapter
        if runtime_adapter is None:
            raise CegWmExperimentAdapterError(
                "Q/K synchronization write requires a prepared runtime adapter"
            )
        try:
            captured = (
                runtime_adapter.execute_content_write_and_capture_geometry_suffix(
                    base_latent,
                    content_embedding_operation,
                )
            )
            if type(captured) is not ContentWriteGeometrySuffixResult:
                raise CegWmExperimentAdapterError(
                    "runtime returned an invalid content suffix result"
                )
            content_result = captured.content_write_result
            if type(content_result) is not ContentWriteVaeResult:
                raise CegWmExperimentAdapterError(
                    "runtime returned an invalid content write result"
                )
            measurement = content_result.content_materialization
            differentiable = (
                runtime_adapter.observe_differentiable_qk_from_generation_suffix(
                    captured.suffix_context,
                    measurement.written_latent_actual,
                )
            )
            if type(differentiable) is not RuntimeDifferentiableQkSuffixResult:
                raise CegWmExperimentAdapterError(
                    "runtime returned an invalid differentiable Q/K result"
                )
            objective = differentiable_qk_relation_objective(
                differentiable.qk_observation.qk_layer_observations,
                detection_key,
            )
            gradient = torch.autograd.grad(
                objective,
                differentiable.callback_latent_float32,
                allow_unused=False,
            )[0]
            pre_write = qk_geometry_sync(
                differentiable.qk_observation.qk_layer_observations,
                detection_key,
            )
            objective_value = float(objective.detach())
            if objective_value != pre_write.relation_score:
                raise CegWmExperimentAdapterError(
                    "differentiable and public Q/K relation scores differ"
                )

            accepted_runtime: RuntimeActualQkSuffixResult | None = None
            accepted_post_write: QkGeometrySyncResult | None = None

            def materialize(candidate: torch.Tensor) -> torch.Tensor:
                return runtime_adapter.materialize_geometry_candidate(
                    candidate,
                    expected_shape=measurement.written_latent_actual.shape,
                    expected_device=measurement.written_latent_actual.device,
                )

            def replay_score(candidate_actual: torch.Tensor) -> float:
                nonlocal accepted_runtime, accepted_post_write
                actual = runtime_adapter.observe_actual_qk_from_generation_suffix(
                    captured.suffix_context,
                    candidate_actual,
                )
                if type(actual) is not RuntimeActualQkSuffixResult:
                    raise CegWmExperimentAdapterError(
                        "runtime returned an invalid actual Q/K result"
                    )
                post_write = qk_geometry_sync(
                    actual.qk_observation.qk_layer_observations,
                    detection_key,
                )
                accepted_runtime = actual
                accepted_post_write = post_write
                return post_write.relation_score

            write = geometry_synchronization_write(
                measurement.baseline_latent_actual,
                measurement.written_latent_actual,
                gradient,
                content_directions,
                geometry_ratio=geometry_ratio,
                baseline_score=pre_write.relation_score,
                materialize=materialize,
                replay_score=replay_score,
            )
            if write.accepted:
                if (
                    accepted_runtime is None
                    or accepted_post_write is None
                    or write.accepted_score
                    != accepted_post_write.relation_score
                ):
                    raise CegWmExperimentAdapterError(
                        "accepted Q/K synchronization result is incomplete"
                    )
            else:
                accepted_runtime = None
                accepted_post_write = None
            result = QkSynchronizationWriteExecutionResult(
                content_write_result=content_result,
                pre_write_observation=pre_write,
                geometry_write_result=write,
                accepted_post_write_observation=accepted_post_write,
                accepted_actual_runtime_result=accepted_runtime,
                gradient_objective=objective_value,
                gradient_l2=float(
                    torch.linalg.vector_norm(
                        gradient.detach().to(device="cpu", dtype=torch.float32)
                    )
                ),
                runtime_config_digest=differentiable.runtime_config_digest,
                callback_index=differentiable.callback_index,
            )
            return self._observe(
                "qk_geometry_sync",
                result,
                upstream_runtime_identity=_runtime_observation_identity(
                    differentiable.qk_observation
                ),
            )
        except CegWmExperimentAdapterError:
            raise
        except Exception as exc:
            raise CegWmExperimentAdapterError(
                "Q/K synchronization write execution failed closed"
            ) from exc

    @_revalidate_configuration_before_call
    def estimate_geometric_transform(
        self,
        observation: QkGeometrySyncResult,
        detection_key: str | DerivedWrongKeyMaterial,
        *,
        epsilon_inlier: float | None,
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
