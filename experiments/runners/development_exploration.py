"""Real 13-responsibility development exploration runner.

The runner owns orchestration and formal development record construction.  It
accepts raw keys, tensors, registered protocol objects, and runtime adapters;
it never accepts precomputed module results or a result-provider callback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite, isnan
from pathlib import Path
from statistics import median
from time import monotonic, time
from typing import Mapping, Sequence

import torch

from experiments.attacks import (
    AttackArtifact,
    AttackRegistry,
    GeometricAttackSpec,
    apply_geometric_attack,
)
from experiments.methods import CegWmExperimentAdapter
from experiments.metrics.development_exploration import (
    DEVELOPMENT_THRESHOLD_ROLE,
    DevelopmentClusterAggregate,
    DevelopmentMetricObservation,
    bind_development_metric_observation,
    cross_fit_development_detection_metrics,
    metric_conditional_recovery_record,
    metric_content_detector,
    metric_content_embedder,
    metric_content_router,
    metric_geometric_transform_estimator,
    metric_geometry_reliability,
    metric_hf_carrier,
    metric_hf_detector,
    metric_image_rectifier,
    metric_key_schedule,
    metric_lf_carrier,
    metric_lf_detector,
    metric_qk_geometry_sync,
)
from experiments.protocol.development_exploration import (
    CANDIDATE_RECOMMENDATIONS,
    COMBINATION_WEIGHT_IDENTITIES,
    CONTENT_COMBINATION_FUNCTION_IDS,
    DEVELOPMENT_CLAIM_BOUNDARY,
    DEVELOPMENT_DEPENDENCY_LAYERS,
    RECORD_SCHEMA_VERSION,
    DEVELOPMENT_SPLIT,
    GEOMETRY_NEGATIVE_CONTROL_CASE_IDS,
    MODULE_OUTCOMES,
    OPERATIONAL_UNIT_PHASES,
    thirteen_module_mechanism_screening_protocol_id,
    DevelopmentProvisionalThreshold,
    DevelopmentModuleOutcomeRecord,
    DevelopmentModuleStudy,
    DevelopmentModuleExecutionDecision,
    DevelopmentVerifiedModuleOutcome,
    DevelopmentVerifiedOutcomeEvidenceContext,
    DevelopmentStudyUnit,
    FrozenDevelopmentCrossFitPlan,
    FrozenDevelopmentExecutionIntentAuthority,
    FrozenDevelopmentExplorationProtocol,
    authorize_development_provisional_threshold,
    create_development_provisional_threshold,
    create_development_threshold_detector_binding,
    create_development_threshold_fit_input,
    _create_verified_development_module_outcome_record,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_RECORD_COLLECTION_ROLE,
    OPERATIONAL_RECORD_COLLECTION_ROLE,
    OPERATIONAL_RECORD_KIND,
    OPERATIONAL_RECORD_SCHEMA,
    ROUTING_REFERENCE_RECORD_KIND,
    DevelopmentRecordError,
    DevelopmentOperationalRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.protocol.internal_records import InternalValidationRecord
from experiments.protocol.internal_splits import AnalysisUnitIdentity
from experiments.runners.development_persistence import (
    CommittedUnit,
    DevelopmentPersistentStore,
    DevelopmentSessionCursor,
    FrozenDevelopmentUnitBinding,
    PersistentLease,
    UnitIntent,
    create_frozen_development_unit_binding,
)
from experiments.runners.formal_operations import (
    FormalHfContentDetectionOperation,
    FormalRuntimeGeometryEstimationOperation,
    create_formal_content_detector_binding,
)
from experiments.runners.internal import (
    InternalCaseExecutionPayload,
    InternalRunnerContext,
    execute_internal_case,
)
from main import (
    BranchNullCalibration,
    ConditionalRecoveryResult,
    ContentDetectionResult,
    ContentRoutingResult,
    GeometricTransformEstimation,
    GeometryReliabilityThresholds,
    GeometryReliabilityResult,
    HfCarrierResult,
    HfDetectionObservation,
    HfDetectionResult,
    ImageRectificationResult,
    JointDecisionThresholds,
    KeyStreamResult,
    LfCarrierResult,
    LfDetectionObservation,
    LfDetectionResult,
    QkGeometrySyncResult,
    RootKeyIdentity,
    RoutingObservations,
    derive_wrong_key_material,
)
from runtime import (
    RuntimeAdapterError,
    RuntimeAdapterState,
    RuntimeContentExecutionError,
    Sd35RuntimeAdapter,
    create_runtime_adapter,
)
from runtime.content_write import ContentWriteVaeResult


DEVELOPMENT_RECORD_SCHEMA = RECORD_SCHEMA_VERSION
THRESHOLD_BOUND_RESPONSIBILITIES = frozenset(
    {
        "conditional_recovery_decision",
    }
)
WIRING_ONLY_NON_SCIENTIFIC_TAU = 1_000_000.0
WIRING_ONLY_NON_SCIENTIFIC_TAU_RESCUE = 999_999.0
WIRING_ONLY_NON_SCIENTIFIC_CALIBRATION_IDENTITY = (
    "wiring_only_non_scientific"
)


class DevelopmentRunnerError(RuntimeError):
    """Frozen unit, direct method call, metric, or record boundary failed."""


class DevelopmentUnitExcluded(DevelopmentRunnerError):
    """A frozen scientific negative/control rule excluded this unit."""


class DevelopmentUnitDurationExceeded(DevelopmentRunnerError):
    """A completed operation exceeded its frozen per-unit walltime budget."""

    def __init__(self, elapsed_seconds: float) -> None:
        super().__init__("development unit exceeded frozen duration")
        self.elapsed_seconds = elapsed_seconds


@dataclass(frozen=True, slots=True)
class DevelopmentCleanControlResult:
    """Experiment-only no-write identity for the clean content control."""

    control_identity: str
    base_latent_digest: str
    realized_relative_l2: float = 0.0
    image_quality_delta: float = 0.0


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _intent_created_epoch(intent: UnitIntent) -> int:
    return int(datetime.fromisoformat(intent.created_at_utc.replace("Z", "+00:00")).timestamp())


def _wiring_identity_rgb8(height: int, width: int) -> torch.Tensor:
    """Encode a deterministic identity coordinate image for synthetic Q/K wiring."""

    horizontal = torch.linspace(64.0, 254.0, width, dtype=torch.float32)
    vertical = torch.linspace(64.0, 254.0, height, dtype=torch.float32)
    x_coordinates = horizontal.reshape(1, width).expand(height, width)
    y_coordinates = vertical.reshape(height, 1).expand(height, width)
    return torch.stack(
        (
            x_coordinates,
            y_coordinates,
            torch.full((height, width), 255.0, dtype=torch.float32),
        )
    ).round().to(dtype=torch.uint8).unsqueeze(0)


def _scientific_record_digest(record: DevelopmentScientificRecord) -> str:
    """Match the persistence marker digest of canonical full record bytes."""

    return _canonical_digest(record.payload())


def _public_payload(value: object) -> object:
    """Convert method results to JSON without executable or private state."""

    if value is None or type(value) in {str, int, bool}:
        return value
    if isinstance(value, float):
        if not isfinite(value):
            return {
                "nonfinite_float": (
                    "not_a_number"
                    if isnan(value)
                    else ("positive_infinity" if value > 0.0 else "negative_infinity")
                )
            }
        return value
    if isinstance(value, torch.Tensor):
        cpu = value.detach().to(device="cpu").contiguous()
        return {
            "tensor_dtype": str(cpu.dtype),
            "tensor_shape": list(cpu.shape),
            "tensor_values_sha256": sha256(cpu.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, tuple):
        return [_public_payload(item) for item in value]
    if isinstance(value, list):
        return [_public_payload(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise DevelopmentRunnerError("method result mapping keys must be strings")
        return {key: _public_payload(item) for key, item in sorted(value.items())}
    if is_dataclass(value) and not isinstance(value, type):
        payload: dict[str, object] = {}
        for item in fields(value):
            field_value = getattr(value, item.name)
            if callable(field_value):
                payload[item.name] = (
                    f"{type(field_value).__module__}.{type(field_value).__qualname__}"
                )
            else:
                payload[item.name] = _public_payload(field_value)
        return payload
    raise DevelopmentRunnerError(
        f"method result type is not persistable: {type(value).__module__}.{type(value).__qualname__}"
    )


_SAFE_RESULT_CONTRACTS: dict[str, tuple[type[object], tuple[str, ...]]] = {
    "content_router": (ContentRoutingResult, ("candidate_id", "mode", "routing_map_digest", "mask_lf_digest", "mask_hf_digest", "mean_routing_map", "mean_mask_lf", "mean_mask_hf", "route_config_digest", "route_identity")),
    "lf_carrier": (LfCarrierResult, ("candidate_id", "template_digest", "direction_digest", "mask_digest", "route_identity", "root_key_public_digest", "key_role", "wrong_key_index", "key_domain_digest", "carrier_config_digest")),
    "hf_carrier": (HfCarrierResult, ("candidate_id", "template_digest", "direction_digest", "mask_digest", "route_identity", "root_key_public_digest", "key_role", "wrong_key_index", "key_domain_digest", "carrier_config_digest")),
    "lf_detector": (LfDetectionResult, ("candidate_id", "lf_score", "detector_identity", "detector_config_digest", "observation_digest", "root_key_public_digest", "key_role", "wrong_key_index", "template_digest")),
    "hf_detector": (HfDetectionResult, ("candidate_id", "hf_score", "detector_identity", "detector_config_digest", "observation_digest", "root_key_public_digest", "key_role", "wrong_key_index", "template_digest")),
    "content_detector": (ContentDetectionResult, ("formal_mode", "content_score", "hf_score", "lf_score", "combined_score", "detector_identity", "content_config_digest", "diagnostic_identity", "content_input_image_digest")),
    "qk_geometry_sync": (QkGeometrySyncResult, ("model_revision", "relation_score", "root_key_public_digest", "key_role", "wrong_key_index", "descriptor_digest", "projection_digest", "geometry_config_digest")),
    "geometric_transform_estimator": (GeometricTransformEstimation, ("transform", "registered_objective", "second_registered_objective", "exact_identity_objective", "canonical_score", "observation_score", "coverage", "uniqueness", "gap", "identity_margin", "key_margin", "anchor_residuals", "inlier_ratio", "mean_residual", "epsilon_inlier", "registered_root_key_public_digest", "observation_descriptor_digest", "observation_projection_digest", "observation_geometry_config_digest", "search_config_digest", "estimation_identity_digest")),
    "geometry_reliability": (GeometryReliabilityResult, ("reliable", "allow_rectification", "status", "failure_reasons", "threshold_config_digest", "estimator_search_config_digest", "estimation_identity_digest", "registered_root_key_public_digest", "reliability_identity_digest")),
    "image_rectifier": (ImageRectificationResult, ("source_image_digest", "rectified_image_digest", "token_crop_support", "pixel_crop_support", "crop_support", "canonical_to_observed_matrix", "rectification_config_digest")),
}

_CONDITIONAL_RECOVERY_SAFE_RESULT_FIELDS = (
    "candidate_id",
    "formal_mode",
    "full_ceg_wm_eligible",
    "source_image_digest",
    "raw_content_score",
    "geometry_triggered",
    "trigger_reason",
    "rectified_content_score",
    "joint_content_positive",
    "positive_source",
    "positive_path",
    "status",
    "failure_reason",
    "detector_identity",
    "content_config_digest",
    "hf_detector_identity",
    "hf_detector_config_digest",
    "hf_template_digest",
    "preprocessing_identity",
    "detector_binding_digest",
    "tau",
    "tau_rescue",
    "threshold_identity",
    "calibration_identity",
    "root_key_public_digest",
    "key_role",
    "wrong_key_index",
    "decision_identity_digest",
)


def _safe_result_payload(responsibility_id: str, result: object) -> dict[str, object]:
    if responsibility_id == "key_schedule":
        if type(result) is not dict or set(result) != {
            "identity", "registered", "wrong", "public_noise"
        }:
            raise DevelopmentRunnerError("key schedule result schema drifted")
        identity = result["identity"]
        if type(identity) is not RootKeyIdentity or any(
            type(result[name]) is not KeyStreamResult
            for name in ("registered", "wrong", "public_noise")
        ):
            raise DevelopmentRunnerError("key schedule result type drifted")
        return {
            "root_key_public_digest": identity.root_key_public_digest,
            "registered_stream_digest": result["registered"].values_float32_be_sha256,
            "wrong_stream_digest": result["wrong"].values_float32_be_sha256,
            "public_noise_digest": result["public_noise"].values_float32_be_sha256,
        }
    if responsibility_id == "content_embedder":
        if type(result) is DevelopmentCleanControlResult:
            return {
                "control_identity": result.control_identity,
                "base_latent_digest": result.base_latent_digest,
                "realized_relative_l2": result.realized_relative_l2,
                "image_quality_delta": result.image_quality_delta,
                "embedding_result_identity": None,
                "materialization_replay_identity": None,
            }
        if type(result) is not ContentWriteVaeResult:
            raise DevelopmentRunnerError("content embedder result type drifted")
        materialization = result.content_materialization
        return {
            "candidate_id": result.candidate_id,
            "runtime_config_digest": result.runtime_config_digest,
            "paired_base_latent_digest": result.paired_base_latent_digest,
            "materialization_replay_identity": materialization.materialization_replay_identity,
            "realized_relative_l2": materialization.realized_relative_l2,
            "integrity_status": materialization.integrity_status,
        }
    if responsibility_id == "conditional_recovery_decision":
        if type(result) is InternalValidationRecord:
            return result.to_dict()
        if type(result) is ConditionalRecoveryResult:
            return {
                name: _public_payload(getattr(result, name))
                for name in _CONDITIONAL_RECOVERY_SAFE_RESULT_FIELDS
            }
        raise DevelopmentRunnerError(
            "conditional recovery result lacks an explicit public record schema"
        )
    contract = _SAFE_RESULT_CONTRACTS.get(responsibility_id)
    if contract is None:
        raise DevelopmentRunnerError("method result lacks an explicit public record schema")
    expected_type, fields_allowed = contract
    if type(result) is not expected_type:
        raise DevelopmentRunnerError("method result type differs from responsibility contract")
    return {name: _public_payload(getattr(result, name)) for name in fields_allowed}


def _decoded_image_to_rgb8(image: torch.Tensor) -> torch.Tensor:
    """Apply the registered public-image conversion used by formal operations."""

    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise DevelopmentRunnerError("decoded image must be a four-dimensional tensor")
    if image.shape[0] != 1 or image.shape[1] != 3 or min(image.shape[-2:]) <= 1:
        raise DevelopmentRunnerError("decoded image must have shape [1,3,H,W]")
    if image.dtype is torch.uint8:
        return image.detach().to(device="cpu").contiguous()
    if not image.dtype.is_floating_point or not bool(torch.isfinite(image).all().item()):
        raise DevelopmentRunnerError("decoded image must be finite floating point or RGB8")
    return torch.floor(image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def _tensor_relative_l2(reference: torch.Tensor, observed: torch.Tensor) -> float:
    if reference.shape != observed.shape or reference.numel() == 0:
        raise DevelopmentRunnerError("quality tensor shapes differ")
    reference_float64 = reference.detach().to(device="cpu", dtype=torch.float64)
    observed_float64 = observed.detach().to(device="cpu", dtype=torch.float64)
    denominator = float(torch.linalg.vector_norm(reference_float64).item())
    if denominator == 0.0:
        raise DevelopmentRunnerError("quality reference has zero norm")
    return float(
        torch.linalg.vector_norm(observed_float64 - reference_float64).item()
    ) / denominator


def _frozen_geometry_attack_specification(
    protocol: FrozenDevelopmentExplorationProtocol,
    geometry_case_id: str,
) -> GeometricAttackSpec:
    """Derive the attack solely from the checked-in protocol case binding."""

    try:
        case = protocol.geometry_study.case(geometry_case_id)
    except ValueError as exc:
        raise DevelopmentRunnerError("unit geometry case is not frozen") from exc
    attack_id = {
        "identity": "identity",
        "crop": "crop",
        "scale": "scale",
        "rotation": "rotation",
        "compound": "crop_scale_rotation",
    }.get(case.operation_family)
    if attack_id is None:
        raise DevelopmentRunnerError("geometry operation family is not registered")
    return GeometricAttackSpec(
        attack_id,
        crop_fraction=case.crop_fraction,
        scale_factor=case.scale_factor,
        rotation_degrees=case.rotation_degrees,
    )


@dataclass(frozen=True, slots=True)
class DevelopmentUnitInput:
    """Raw execution inputs; precomputed method results are intentionally absent."""

    registered_root_key: str
    wrong_key_index: int
    base_latent: torch.Tensor
    routing_observations: RoutingObservations | None
    mixing_coefficient: float
    combination_function_id: str
    hf_null: BranchNullCalibration | None
    lf_null: BranchNullCalibration | None
    epsilon_inlier: float | None
    geometry_reliability_thresholds: GeometryReliabilityThresholds | None
    provisional_threshold: DevelopmentProvisionalThreshold | None
    cross_fit_plan: FrozenDevelopmentCrossFitPlan | None
    development_tau_rescue: float | None
    internal_runner_context: InternalRunnerContext | None = None
    internal_case_payload: InternalCaseExecutionPayload | None = None

    def validate(self, responsibility_id: str) -> None:
        if type(self.registered_root_key) is not str or not self.registered_root_key:
            raise DevelopmentRunnerError("registered root key is missing")
        if type(self.wrong_key_index) is not int or self.wrong_key_index < 0:
            raise DevelopmentRunnerError("wrong key index is invalid")
        if not isinstance(self.base_latent, torch.Tensor) or self.base_latent.ndim != 4:
            raise DevelopmentRunnerError("base latent must be a real four-dimensional tensor")
        if not bool(torch.isfinite(self.base_latent).all().item()):
            raise DevelopmentRunnerError("base latent contains non-finite values")
        if self.routing_observations is not None and type(
            self.routing_observations
        ) is not RoutingObservations:
            raise DevelopmentRunnerError("routing observations exact type is required")
        if self.mixing_coefficient not in {0.25, 0.50, 0.75}:
            raise DevelopmentRunnerError("mixing coefficient is not registered")
        if self.combination_function_id not in CONTENT_COMBINATION_FUNCTION_IDS:
            raise DevelopmentRunnerError("combination function is not registered")
        if self.epsilon_inlier is not None and (
            isinstance(self.epsilon_inlier, bool)
            or not isinstance(self.epsilon_inlier, (int, float))
            or not isfinite(float(self.epsilon_inlier))
            or float(self.epsilon_inlier) <= 0.0
        ):
            raise DevelopmentRunnerError("epsilon_inlier is invalid")
        if responsibility_id in {
            "image_rectifier",
            "conditional_recovery_decision",
        }:
            if type(self.geometry_reliability_thresholds) is not GeometryReliabilityThresholds:
                raise DevelopmentRunnerError(
                    "development reliability thresholds exact type required"
                )
        if responsibility_id in {
            "image_rectifier",
            "conditional_recovery_decision",
        } and self.epsilon_inlier is None:
            raise DevelopmentRunnerError(
                "fitted epsilon_inlier is required for downstream geometry"
            )
        threshold_values = (
            self.provisional_threshold,
            self.cross_fit_plan,
            self.development_tau_rescue,
        )
        if responsibility_id in THRESHOLD_BOUND_RESPONSIBILITIES:
            if type(self.provisional_threshold) is not DevelopmentProvisionalThreshold:
                raise DevelopmentRunnerError("development provisional threshold exact type required")
            if type(self.cross_fit_plan) is not FrozenDevelopmentCrossFitPlan:
                raise DevelopmentRunnerError("development cross-fit plan exact type required")
            if (
                isinstance(self.development_tau_rescue, bool)
                or not isinstance(self.development_tau_rescue, (int, float))
                or not isfinite(float(self.development_tau_rescue))
                or not float(self.development_tau_rescue) < self.provisional_threshold.threshold
            ):
                raise DevelopmentRunnerError("development rescue interval is invalid")
        elif any(value is not None for value in threshold_values):
            raise DevelopmentRunnerError("threshold inputs are forbidden for this responsibility")
        if responsibility_id == "conditional_recovery_decision":
            if type(self.internal_runner_context) is not InternalRunnerContext:
                raise DevelopmentRunnerError("joint unit requires InternalRunnerContext")
            if type(self.internal_case_payload) is not InternalCaseExecutionPayload:
                raise DevelopmentRunnerError("joint unit requires InternalCaseExecutionPayload")
            assert self.provisional_threshold is not None
            assert self.development_tau_rescue is not None
            thresholds = self.internal_case_payload.thresholds
            if (
                thresholds.tau != self.provisional_threshold.threshold
                or thresholds.tau_rescue != float(self.development_tau_rescue)
            ):
                raise DevelopmentRunnerError("joint unit threshold is not the bound development threshold")
        elif self.internal_runner_context is not None or self.internal_case_payload is not None:
            raise DevelopmentRunnerError("internal joint runner inputs are forbidden for this responsibility")


@dataclass(frozen=True, slots=True)
class DevelopmentUnitRunResult:
    record: DevelopmentScientificRecord
    intent: UnitIntent | None
    committed: CommittedUnit | None


@dataclass(frozen=True, slots=True)
class DevelopmentOperationalReceipt:
    operational_role: str
    source_cluster_ordinal: int
    case_ids: tuple[str, ...]
    responsibility_result_digests: tuple[tuple[str, str], ...]
    elapsed_seconds: float
    runtime_config_digest: str
    counts_as_scientific_coverage: bool
    scientific_claims_supported: bool


class DevelopmentExplorationRunner:
    """Execute a frozen roster without result-provider or module-result proxies."""

    def __init__(
        self,
        *,
        intent_authority: FrozenDevelopmentExecutionIntentAuthority,
        adapter: CegWmExperimentAdapter,
        runtime_adapter: Sd35RuntimeAdapter,
        attack_registry: AttackRegistry,
        method_code_revision: str,
        environment_digest: str,
        resource_identity_digest: str,
        persistence_store: DevelopmentPersistentStore | None = None,
    ) -> None:
        if type(intent_authority) is not FrozenDevelopmentExecutionIntentAuthority or intent_authority.validate():
            raise DevelopmentRunnerError("execution intent authority is invalid")
        if type(adapter) is not CegWmExperimentAdapter:
            raise DevelopmentRunnerError("CEG-WM adapter exact type is required")
        if type(runtime_adapter) is not Sd35RuntimeAdapter:
            raise DevelopmentRunnerError("SD3.5 runtime adapter exact type is required")
        try:
            adapter.require_no_runtime_binding()
        except Exception as exc:
            raise DevelopmentRunnerError(
                "method adapter must not retain a hidden runtime binding"
            ) from exc
        if runtime_adapter.state is not RuntimeAdapterState.READY:
            raise DevelopmentRunnerError("runtime adapter must be ready")
        if type(attack_registry) is not AttackRegistry:
            raise DevelopmentRunnerError("attack registry exact type is required")
        if type(method_code_revision) is not str or len(method_code_revision) != 40:
            raise DevelopmentRunnerError("method code revision must be a full Git SHA")
        for role, value in (
            ("environment_digest", environment_digest),
            ("resource_identity_digest", resource_identity_digest),
        ):
            if type(value) is not str or len(value) != 64:
                raise DevelopmentRunnerError(f"{role} is invalid")
        if persistence_store is not None:
            if persistence_store.run_id != intent_authority.run_id:
                raise DevelopmentRunnerError("persistence run identity drifted")
            identity = persistence_store.worker_identity
            if (
                identity.revision != method_code_revision
                or identity.protocol_digest != intent_authority.protocol_digest
                or identity.execution_intent_authority_digest != intent_authority.authority_digest
                or identity.input_manifest_digest != intent_authority.input_manifest_digest
            ):
                raise DevelopmentRunnerError("persistence frozen identity drifted")
        self.intent_authority = intent_authority
        self.protocol = intent_authority.protocol
        self.adapter = adapter
        self.runtime_adapter = runtime_adapter
        self.attack_registry = attack_registry
        self.method_code_revision = method_code_revision
        self.environment_digest = environment_digest
        self.resource_identity_digest = resource_identity_digest
        self.persistence_store = persistence_store
        self._clusters = self._cluster_identities()
        if persistence_store is not None:
            expected_bindings = self.create_persistence_unit_bindings()
            observed_bindings = persistence_store.registered_unit_bindings
            if observed_bindings != expected_bindings:
                raise DevelopmentRunnerError(
                    "persistence unit/analysis/candidate roster binding drifted"
                )

    def create_persistence_unit_bindings(
        self,
    ) -> tuple[FrozenDevelopmentUnitBinding, ...]:
        """Materialize the exact protocol/manifest roster consumed by persistence."""

        studies = {
            item.responsibility_id: item for item in self.protocol.module_matrix
        }
        bindings: list[FrozenDevelopmentUnitBinding] = []
        for unit in self.protocol.unit_roster:
            if unit.phase == "development_environment_preflight":
                scientific_question_id = "environment_runtime_identity_and_throughput"
                development_case_id = "development_environment_preflight"
                candidate_identity = "development_execution_environment"
                candidate_config_digest = _canonical_digest(
                    {
                        "phase": unit.phase,
                        "protocol_digest": self.protocol.digest(),
                        "preflight": asdict(self.protocol.preflight),
                    }
                )
            elif unit.phase == "development_full_chain_wiring":
                scientific_question_id = "all_responsibility_public_call_wiring"
                development_case_id = "development_full_chain_wiring"
                candidate_identity = "development_method_chain"
                candidate_config_digest = _canonical_digest(
                    {
                        "phase": unit.phase,
                        "protocol_digest": self.protocol.digest(),
                        "responsibility_ids": tuple(studies),
                        "counts_as_scientific_coverage": False,
                    }
                )
            else:
                study = studies[unit.responsibility_id]
                scientific_question_id = study.scientific_question_id
                development_case_id = study.development_case_id
                candidate_identity = study.candidate_identity
                candidate_config_digest = study.candidate_config_digest
            bindings.append(
                create_frozen_development_unit_binding(
                    unit,
                    analysis_unit_identity=self._analysis_identity(unit),
                    scientific_question_id=scientific_question_id,
                    development_case_id=development_case_id,
                    candidate_identity=candidate_identity,
                    candidate_config_digest=candidate_config_digest,
                )
            )
        return tuple(bindings)

    def create_operational_intent(
        self,
        lease: PersistentLease,
        session_cursor: DevelopmentSessionCursor,
        *,
        now_epoch_seconds: int,
    ) -> UnitIntent:
        if self.persistence_store is None:
            raise DevelopmentRunnerError("persistent store is required")
        unit = self._unit(session_cursor.next_unit_index)
        if unit.phase not in {
            "development_environment_preflight",
            "development_full_chain_wiring",
        }:
            raise DevelopmentRunnerError("next frozen unit is not preflight or wiring")
        return self.persistence_store.create_session_intent(
            session_cursor,
            lease,
            now_epoch_seconds=now_epoch_seconds,
        )

    def commit_operational_receipt(
        self,
        lease: PersistentLease,
        session_cursor: DevelopmentSessionCursor,
        intent: UnitIntent,
        receipt: DevelopmentOperationalReceipt,
        *,
        now_epoch_seconds: int,
        raw_secret_values: Sequence[str],
    ) -> CommittedUnit:
        """Persist one preflight/wiring unit through the common intent/bundle path."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError("persistent store is required")
        unit = self._unit(session_cursor.next_unit_index)
        expected_role = (
            "environment_runtime_throughput_preflight"
            if unit.phase == "development_environment_preflight"
            else "full_chain_wiring_smoke"
            if unit.phase == "development_full_chain_wiring"
            else None
        )
        if (
            type(receipt) is not DevelopmentOperationalReceipt
            or expected_role is None
            or receipt.operational_role != expected_role
            or receipt.source_cluster_ordinal != unit.source_cluster_ordinal
            or intent.unit_index != unit.unit_index
            or intent.phase != unit.phase
            or receipt.counts_as_scientific_coverage is not False
            or receipt.scientific_claims_supported is not False
        ):
            raise DevelopmentRunnerError(
                "operational receipt differs from the frozen next unit"
            )
        payload = json.loads(
            json.dumps(
                asdict(receipt),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        )
        record = DevelopmentOperationalRecord(
            schema_version=OPERATIONAL_RECORD_SCHEMA,
            collection_role=OPERATIONAL_RECORD_COLLECTION_ROLE,
            record_kind=OPERATIONAL_RECORD_KIND,
            record_id="0" * 64,
            run_id=self.intent_authority.run_id,
            protocol_digest=self.protocol.digest(),
            method_code_revision=self.method_code_revision,
            unit_index=unit.unit_index,
            phase=unit.phase,
            source_cluster_ordinal=unit.source_cluster_ordinal,
            candidate_config_digest=intent.candidate_config_digest,
            attempt_index=intent.attempt_index,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            actual_elapsed_seconds=receipt.elapsed_seconds,
            maximum_duration_seconds=intent.maximum_duration_seconds,
            operation_result_payload=payload,
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = replace(
            record,
            record_id=canonical_development_value_digest(
                record.payload_without_record_id()
            ),
        )
        return self.persistence_store.commit_session_unit(
            session_cursor,
            lease,
            intent,
            record=record,
            raw_secret_values=raw_secret_values,
            now_epoch_seconds=now_epoch_seconds,
        )

    def execute_preflight_cluster(
        self,
        source_cluster_ordinal: int,
        unit_input: DevelopmentUnitInput,
    ) -> DevelopmentOperationalReceipt:
        """Measure one of the frozen one-to-two identity/throughput clusters."""

        if type(source_cluster_ordinal) is not int or not 0 <= source_cluster_ordinal < self.protocol.preflight.source_cluster_count:
            raise DevelopmentRunnerError("preflight cluster ordinal is outside frozen budget")
        unit = self._representative_unit(
            "content_embedder",
            source_cluster_ordinal,
            content_branch_id="hf_only",
        )
        unit_input.validate(unit.responsibility_id)
        identity = self._analysis_identity(unit)
        started = monotonic()
        result, _metric, _traces, _internal = self._execute_real_operation(
            unit, identity, unit_input
        )
        elapsed = monotonic() - started
        result_payload = _safe_result_payload(unit.responsibility_id, result)
        return DevelopmentOperationalReceipt(
            operational_role="environment_runtime_throughput_preflight",
            source_cluster_ordinal=source_cluster_ordinal,
            case_ids=self.protocol.preflight.case_ids,
            responsibility_result_digests=((unit.responsibility_id, _canonical_digest(result_payload)),),
            elapsed_seconds=elapsed,
            runtime_config_digest=self.runtime_adapter.session.runtime_config_digest,
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
        )

    def execute_wiring_smoke_cluster(
        self,
        source_cluster_ordinal: int,
        unit_inputs: Mapping[str, DevelopmentUnitInput],
    ) -> DevelopmentOperationalReceipt:
        """Call every real responsibility for one of eight non-scientific smoke clusters."""

        if type(source_cluster_ordinal) is not int or not 0 <= source_cluster_ordinal < self.protocol.study_budget.wiring_source_cluster_count:
            raise DevelopmentRunnerError("wiring cluster ordinal is outside frozen budget")
        expected = tuple(item.responsibility_id for item in self.protocol.module_matrix)
        if set(unit_inputs) != set(expected):
            raise DevelopmentRunnerError("wiring inputs must cover all thirteen responsibilities")
        started = monotonic()
        results = self._execute_shared_wiring_chain(
            source_cluster_ordinal,
            unit_inputs,
        )
        digests = tuple(
            (
                responsibility,
                _canonical_digest(
                    _safe_result_payload(responsibility, results[responsibility])
                ),
            )
            for responsibility in expected
        )
        return DevelopmentOperationalReceipt(
            operational_role="full_chain_wiring_smoke",
            source_cluster_ordinal=source_cluster_ordinal,
            case_ids=("all_thirteen_responsibility_wiring",),
            responsibility_result_digests=digests,
            elapsed_seconds=monotonic() - started,
            runtime_config_digest=self.runtime_adapter.session.runtime_config_digest,
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
        )

    def _execute_shared_wiring_chain(
        self,
        source_cluster_ordinal: int,
        unit_inputs: Mapping[str, DevelopmentUnitInput],
    ) -> dict[str, object]:
        """Exercise all public responsibilities while sharing one runtime artifact."""

        for responsibility, raw in unit_inputs.items():
            raw.validate(
                "content_detector"
                if responsibility == "conditional_recovery_decision"
                else responsibility
            )
        raw = unit_inputs["content_embedder"]
        key_identity = self.adapter.identify_key(raw.registered_root_key).result
        shape = tuple(int(value) for value in raw.base_latent.shape)
        registered_domain = {
            "candidate_id": "hf_sparse_tail",
            "operator": "carrier_template",
            "responsibility_domain": "hf_carrier",
            "model_revision": self.runtime_adapter.session.model_revision,
            "tensor_role": "base_gaussian",
        }
        public_domain = {
            "candidate_id": "routing_stqr",
            "operator": "local_sensitivity_public_probe",
            "responsibility_domain": "public_noise",
            "model_revision": self.runtime_adapter.session.model_revision,
            "sample_index": source_cluster_ordinal,
            "tensor_role": "latent_probe",
        }
        key_result = {
            "identity": key_identity,
            "registered": self.adapter.derive_registered_key_stream(
                raw.registered_root_key, registered_domain, shape
            ).result,
            "wrong": self.adapter.derive_wrong_key_stream(
                key_identity.root_key_public_digest,
                raw.wrong_key_index,
                registered_domain,
                shape,
            ).result,
            "public_noise": self.adapter.derive_public_noise(
                public_domain, shape
            ).result,
        }
        routing = self.adapter.route_content(
            shape,
            mode="routing_stqr",
            observations=raw.routing_observations,
        ).result
        low_frequency = self.adapter.build_lf_carrier(
            raw.registered_root_key,
            shape,
            routing_result=routing,
        ).result
        high_frequency = self.adapter.build_hf_carrier(
            raw.registered_root_key,
            shape,
            routing_result=routing,
        ).result

        def embed(latent_values: tuple[float, ...]):
            return self.adapter.embed_content(
                latent_values,
                high_frequency,
                lf_carrier_result=low_frequency,
                mixing_coefficient=raw.mixing_coefficient,
                routing_result=routing,
            ).result

        runtime_result = self.runtime_adapter.execute_content_write_and_vae(
            raw.base_latent,
            embed,
        )
        observed = runtime_result.watermarked_detection_latent
        observed_values = tuple(
            observed.detach().to(dtype=torch.float32).reshape(-1).tolist()
        )
        high_detection = self.adapter.detect_hf(
            HfDetectionObservation.from_public_image_encoding(
                observed_values, tuple(observed.shape)
            ),
            raw.registered_root_key,
        ).result
        low_detection = self.adapter.detect_lf(
            LfDetectionObservation.from_public_image_encoding(
                observed_values, tuple(observed.shape)
            ),
            raw.registered_root_key,
        ).result
        content_detection = self.adapter.detect_content(
            high_detection,
            low_detection,
            hf_null=raw.hf_null,
            lf_null=raw.lf_null,
            combination=raw.combination_function_id,
            weight=(
                raw.mixing_coefficient
                if raw.combination_function_id
                == "weighted_hf_lf_standardized_score"
                else None
            ),
        ).result
        source_image = _decoded_image_to_rgb8(runtime_result.watermarked_image)
        thresholds = unit_inputs[
            "geometry_reliability"
        ].geometry_reliability_thresholds
        if type(thresholds) is not GeometryReliabilityThresholds:
            raise DevelopmentRunnerError(
                "wiring shared chain requires wiring-only reliability constants"
            )
        runtime_observation = self.runtime_adapter.observe_detection_qk(source_image)
        synchronized = self.adapter.synchronize_qk_observation(
            runtime_observation,
            raw.registered_root_key,
        ).result
        estimation = self.adapter.estimate_geometric_transform(
            synchronized,
            raw.registered_root_key,
            epsilon_inlier=thresholds.epsilon_inlier,
        ).result
        reliability = self.adapter.assess_geometry_reliability(
            estimation,
            thresholds,
        ).result
        rectified = self._execute_wiring_image_rectifier_call(
            unit_inputs["image_rectifier"]
        )
        conditional = self._execute_wiring_conditional_call(
            self._representative_unit(
                "conditional_recovery_decision", source_cluster_ordinal
            ),
            unit_inputs["conditional_recovery_decision"],
            source_image=source_image,
        )
        return {
            "key_schedule": key_result,
            "content_router": routing,
            "lf_carrier": low_frequency,
            "hf_carrier": high_frequency,
            "content_embedder": runtime_result,
            "lf_detector": low_detection,
            "hf_detector": high_detection,
            "content_detector": content_detection,
            "qk_geometry_sync": synchronized,
            "geometric_transform_estimator": estimation,
            "geometry_reliability": reliability,
            "image_rectifier": rectified,
            "conditional_recovery_decision": conditional,
        }

    def _execute_wiring_image_rectifier_call(
        self,
        raw: DevelopmentUnitInput,
    ) -> object:
        """Call the real rectifier after a local deterministic identity Q/K chain."""

        return self._execute_wiring_geometry_chain(raw)[-1]

    def _execute_wiring_geometry_chain(
        self,
        raw: DevelopmentUnitInput,
    ) -> tuple[
        QkGeometrySyncResult,
        GeometricTransformEstimation,
        GeometryReliabilityResult,
        ImageRectificationResult,
    ]:
        """Run one complete public Q/K-to-rectification wiring chain."""

        raw.validate("image_rectifier")
        thresholds = raw.geometry_reliability_thresholds
        if type(thresholds) is not GeometryReliabilityThresholds:
            raise DevelopmentRunnerError(
                "wiring image rectifier requires wiring-only reliability constants"
            )
        from experiments.runners.synthetic_runtime import SyntheticQkBackend

        synthetic_runtime = create_runtime_adapter(
            SyntheticQkBackend(
                root_key=raw.registered_root_key,
                model_revision=self.runtime_adapter.session.model_revision,
            )
        )
        synthetic_runtime.initialize("cpu")
        try:
            source_image = _wiring_identity_rgb8(
                synthetic_runtime.session.image_height,
                synthetic_runtime.session.image_width,
            )
            runtime_observation = synthetic_runtime.observe_detection_qk(
                source_image
            )
            synchronized = self.adapter.synchronize_qk_observation(
                runtime_observation,
                raw.registered_root_key,
            ).result
            estimation = self.adapter.estimate_geometric_transform(
                synchronized,
                raw.registered_root_key,
                epsilon_inlier=thresholds.epsilon_inlier,
            ).result
            reliability = self.adapter.assess_geometry_reliability(
                estimation,
                thresholds,
            ).result
            if not reliability.reliable or not reliability.allow_rectification:
                raise DevelopmentRunnerError(
                    "wiring image rectifier identity geometry is unreliable"
                )
            rectified = self.adapter.rectify_image(
                source_image,
                estimation,
                reliability,
            ).result
            return synchronized, estimation, reliability, rectified
        finally:
            synthetic_runtime.close()

    def _execute_wiring_conditional_call(
        self,
        unit: DevelopmentStudyUnit,
        raw: DevelopmentUnitInput,
        *,
        source_image: torch.Tensor | None = None,
    ) -> object:
        """Call the real joint API with an explicitly non-scientific wiring threshold."""

        if type(raw.geometry_reliability_thresholds) is not GeometryReliabilityThresholds:
            raise DevelopmentRunnerError(
                "wiring conditional call requires wiring-only reliability constants"
            )
        if source_image is None:
            shape = tuple(int(value) for value in raw.base_latent.shape)
            routing = self.adapter.route_content(
                shape,
                mode="routing_uniform_control",
            ).result
            low_frequency = self.adapter.build_lf_carrier(
                raw.registered_root_key,
                shape,
                routing_result=routing,
            ).result
            high_frequency = self.adapter.build_hf_carrier(
                raw.registered_root_key,
                shape,
                routing_result=routing,
            ).result

            def embed(latent_values: tuple[float, ...]):
                return self.adapter.embed_content(
                    latent_values,
                    high_frequency,
                    lf_carrier_result=low_frequency,
                    mixing_coefficient=raw.mixing_coefficient,
                    routing_result=routing,
                ).result

            runtime_result = self.runtime_adapter.execute_content_write_and_vae(
                raw.base_latent,
                embed,
            )
            source_image = _decoded_image_to_rgb8(runtime_result.watermarked_image)
        content_operation = FormalHfContentDetectionOperation(self.adapter)
        content_binding, _ = create_formal_content_detector_binding(
            content_operation,
            prototype_image=source_image,
            detection_key=raw.registered_root_key,
        )
        thresholds = JointDecisionThresholds(
            tau=WIRING_ONLY_NON_SCIENTIFIC_TAU,
            tau_rescue=WIRING_ONLY_NON_SCIENTIFIC_TAU_RESCUE,
            detector_binding_digest=content_binding.detector_binding_digest,
            calibration_identity=(
                WIRING_ONLY_NON_SCIENTIFIC_CALIBRATION_IDENTITY
            ),
        )
        geometry_operation = FormalRuntimeGeometryEstimationOperation(
            runtime_adapter=self.runtime_adapter,
            adapter_configuration=self.adapter.configuration,
            epsilon_inlier=raw.geometry_reliability_thresholds.epsilon_inlier,
            execution_scope="wiring_only_non_scientific",
        )
        return self.adapter.decide_conditional_recovery(
            source_image,
            raw.registered_root_key,
            content_detector_binding=content_binding,
            thresholds=thresholds,
            geometry_estimation_operation=geometry_operation,
            geometry_reliability_thresholds=raw.geometry_reliability_thresholds,
        ).result

    def _execute_unit(
        self,
        unit_index: int,
        unit_input: DevelopmentUnitInput,
        *,
        attempt_index: int = 0,
        retry_parent_intent_digest: str | None = None,
        attempt_started_monotonic: float | None = None,
    ) -> DevelopmentUnitRunResult:
        unit = self._unit(unit_index)
        unit_input.validate(unit.responsibility_id)
        if type(attempt_index) is not int or not 0 <= attempt_index < unit.maximum_record_attempts:
            raise DevelopmentRunnerError("record attempt exceeds frozen limit")
        identity = self._analysis_identity(unit)
        if unit.responsibility_id in THRESHOLD_BOUND_RESPONSIBILITIES:
            assert unit_input.provisional_threshold is not None
            assert unit_input.cross_fit_plan is not None
            authorize_development_provisional_threshold(
                unit_input.provisional_threshold,
                unit_input.cross_fit_plan,
                expected_execution_intent_authority_digest=self.intent_authority.authority_digest,
                requested_split=DEVELOPMENT_SPLIT,
                requested_analysis_unit_identity=identity,
            )
        started = (
            monotonic()
            if attempt_started_monotonic is None
            else attempt_started_monotonic
        )
        result, metric, trace_values, internal_record = self._execute_real_operation(
            unit,
            identity,
            unit_input,
        )
        elapsed_seconds = monotonic() - started
        if elapsed_seconds > unit.maximum_duration_seconds:
            raise DevelopmentUnitDurationExceeded(elapsed_seconds)
        record = self._record(
            unit,
            identity,
            result,
            metric,
            trace_values,
            internal_record=internal_record,
            attempt_index=attempt_index,
            retry_parent_intent_digest=retry_parent_intent_digest,
            actual_elapsed_seconds=elapsed_seconds,
        )
        return DevelopmentUnitRunResult(record=record, intent=None, committed=None)

    def execute_and_commit_next_unit(
        self,
        lease: PersistentLease,
        unit_input: DevelopmentUnitInput,
        *,
        now_epoch_seconds: int,
        raw_secret_values: Sequence[str],
    ) -> DevelopmentUnitRunResult:
        """Claim exactly the next frozen breadth-first unit and persist every outcome."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError("persistent store is required")
        recovery = self.persistence_store.recover(
            now_epoch_seconds=now_epoch_seconds
        )
        latest_commit_by_unit = {}
        for committed_unit in recovery.committed_units:
            latest_commit_by_unit[committed_unit.unit_id] = committed_unit
        terminal_indices = tuple(
            sorted(
                item.unit_index
                for item in latest_commit_by_unit.values()
                if item.attempt_disposition != "retryable_resource_failure"
            )
        )
        if terminal_indices != tuple(range(len(terminal_indices))):
            raise DevelopmentRunnerError(
                "verified terminal commits are not a breadth-first frozen roster prefix"
            )
        next_attempts = dict(recovery.next_attempt_by_unit)
        if len(next_attempts) > 1:
            raise DevelopmentRunnerError("multiple frozen units are concurrently retryable")
        if next_attempts:
            unit_id, attempt_index = next(iter(next_attempts.items()))
            unit_index = next(
                (
                    item.unit_index
                    for item in self.protocol.unit_roster
                    if self._unit_id(item) == unit_id
                ),
                -1,
            )
            if unit_index < 0:
                raise DevelopmentRunnerError("retryable unit is outside frozen roster")
            interrupted_parent = next(
                (
                    item.retry_parent_intent_digest
                    for item in recovery.interrupted_attempts
                    if item.unit_id == unit_id
                    and item.attempt_index == attempt_index - 1
                ),
                None,
            )
            committed_parent = latest_commit_by_unit.get(unit_id)
            parent_digest = (
                interrupted_parent
                if interrupted_parent is not None
                else committed_parent.intent_digest
                if committed_parent is not None
                else None
            )
        else:
            unit_index = len(terminal_indices)
            attempt_index = 0
            parent_digest = None
        if unit_index >= len(self.protocol.unit_roster):
            raise DevelopmentRunnerError("all frozen development units are committed")
        unit = self._unit(unit_index)
        if unit.phase in OPERATIONAL_UNIT_PHASES:
            raise DevelopmentRunnerError(
                "operational unit requires its registered producer"
            )
        unit_id = self._unit_id(unit)
        if self.persistence_store.next_attempt_index(unit_id) != attempt_index:
            raise DevelopmentRunnerError("persistence next attempt drifted")
        intent = self.persistence_store.create_intent(
            lease,
            unit_id=unit_id,
            unit_index=unit_index,
            attempt_index=attempt_index,
            parent_attempt_intent_digest=parent_digest,
            now_epoch_seconds=now_epoch_seconds,
        )
        return self._execute_and_commit_claimed_unit(
            lease,
            intent,
            unit_input,
            now_epoch_seconds=now_epoch_seconds,
            raw_secret_values=raw_secret_values,
            session_cursor=None,
        )

    def execute_and_commit_session_unit(
        self,
        lease: PersistentLease,
        session_cursor: DevelopmentSessionCursor,
        unit_input: DevelopmentUnitInput,
        *,
        now_epoch_seconds: int,
        raw_secret_values: Sequence[str],
        attempt_started_monotonic: float | None = None,
    ) -> DevelopmentUnitRunResult:
        """Execute the session cursor's next scientific unit without full recovery."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError("persistent store is required")
        unit_index = session_cursor.next_unit_index
        if unit_index >= len(self.protocol.unit_roster):
            raise DevelopmentRunnerError("all frozen development units are committed")
        unit = self._unit(unit_index)
        if unit.phase in OPERATIONAL_UNIT_PHASES:
            raise DevelopmentRunnerError(
                "operational unit requires its registered producer"
            )
        intent = self.persistence_store.create_session_intent(
            session_cursor,
            lease,
            now_epoch_seconds=now_epoch_seconds,
        )
        return self._execute_and_commit_claimed_unit(
            lease,
            intent,
            unit_input,
            now_epoch_seconds=now_epoch_seconds,
            raw_secret_values=raw_secret_values,
            session_cursor=session_cursor,
            attempt_started_monotonic=attempt_started_monotonic,
        )

    def create_scientific_intent(
        self,
        lease: PersistentLease,
        session_cursor: DevelopmentSessionCursor,
        *,
        now_epoch_seconds: int,
    ) -> UnitIntent:
        """Create the next scientific intent before any expensive input work."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError("persistent store is required")
        unit = self._unit(session_cursor.next_unit_index)
        if unit.phase in OPERATIONAL_UNIT_PHASES:
            raise DevelopmentRunnerError("next frozen unit is not scientific")
        return self.persistence_store.create_session_intent(
            session_cursor,
            lease,
            now_epoch_seconds=now_epoch_seconds,
        )

    def execute_and_commit_claimed_session_unit(
        self,
        lease: PersistentLease,
        session_cursor: DevelopmentSessionCursor,
        intent: UnitIntent,
        unit_input: DevelopmentUnitInput,
        *,
        attempt_started_monotonic: float,
        raw_secret_values: Sequence[str],
    ) -> DevelopmentUnitRunResult:
        """Execute a previously claimed intent and include preparation walltime."""

        return self._execute_and_commit_claimed_unit(
            lease,
            intent,
            unit_input,
            now_epoch_seconds=max(_intent_created_epoch(intent) + 1, int(time())),
            raw_secret_values=raw_secret_values,
            session_cursor=session_cursor,
            attempt_started_monotonic=attempt_started_monotonic,
        )

    def commit_claimed_terminal_failure(
        self,
        lease: PersistentLease,
        session_cursor: DevelopmentSessionCursor,
        intent: UnitIntent,
        *,
        failure_class: str,
        failure_reason: str,
        attempt_started_monotonic: float,
        raw_secret_values: Sequence[str],
    ) -> DevelopmentUnitRunResult:
        """Commit a narrow terminal build/dependency failure after an intent."""

        if failure_class not in {"implementation_failure", "dependency_blocked"}:
            raise DevelopmentRunnerError("terminal failure class is not registered")
        unit = self._unit(intent.unit_index)
        record = self._failure_record(
            unit,
            self._analysis_identity(unit),
            attempt_index=intent.attempt_index,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            execution_status="failed",
            failure_class=failure_class,
            failure_reason=failure_reason,
            actual_elapsed_seconds=monotonic() - attempt_started_monotonic,
        )
        committed = self.persistence_store.commit_session_unit(
            session_cursor,
            lease,
            intent,
            record=record,
            raw_secret_values=raw_secret_values,
            now_epoch_seconds=max(_intent_created_epoch(intent) + 1, int(time())),
        )
        return DevelopmentUnitRunResult(record, intent, committed)

    def commit_claimed_resource_failure(
        self,
        lease: PersistentLease,
        session_cursor: DevelopmentSessionCursor,
        intent: UnitIntent,
        *,
        failure_reason: str,
        attempt_started_monotonic: float,
        raw_secret_values: Sequence[str],
    ) -> DevelopmentUnitRunResult:
        """Commit a retryable resource/input-budget failure and stop the session."""

        unit = self._unit(intent.unit_index)
        retryable = intent.attempt_index + 1 < unit.maximum_record_attempts
        record = self._failure_record(
            unit,
            self._analysis_identity(unit),
            attempt_index=intent.attempt_index,
            retry_parent_intent_digest=intent.parent_attempt_intent_digest,
            execution_status="retry" if retryable else "failed",
            failure_class="resource_failure",
            failure_reason=failure_reason,
            actual_elapsed_seconds=monotonic() - attempt_started_monotonic,
        )
        committed = self.persistence_store.commit_session_unit(
            session_cursor,
            lease,
            intent,
            record=record,
            raw_secret_values=raw_secret_values,
            now_epoch_seconds=max(_intent_created_epoch(intent) + 1, int(time())),
        )
        return DevelopmentUnitRunResult(record, intent, committed)

    def _execute_and_commit_claimed_unit(
        self,
        lease: PersistentLease,
        intent: UnitIntent,
        unit_input: DevelopmentUnitInput,
        *,
        now_epoch_seconds: int,
        raw_secret_values: Sequence[str],
        session_cursor: DevelopmentSessionCursor | None,
        attempt_started_monotonic: float | None = None,
    ) -> DevelopmentUnitRunResult:
        if self.persistence_store is None:
            raise DevelopmentRunnerError("persistent store is required")
        unit_index = intent.unit_index
        unit = self._unit(unit_index)
        attempt_index = intent.attempt_index
        parent_digest = intent.parent_attempt_intent_digest
        identity = self._analysis_identity(unit)
        started = monotonic() if attempt_started_monotonic is None else attempt_started_monotonic
        try:
            executed = self._execute_unit(
                unit_index,
                unit_input,
                attempt_index=attempt_index,
                retry_parent_intent_digest=parent_digest,
                attempt_started_monotonic=started,
            )
        except DevelopmentUnitExcluded as exc:
            executed = DevelopmentUnitRunResult(
                self._failure_record(
                    unit,
                    identity,
                    attempt_index=attempt_index,
                    retry_parent_intent_digest=parent_digest,
                    execution_status="excluded",
                    failure_class="scientific_exclusion",
                    failure_reason=str(exc),
                    actual_elapsed_seconds=monotonic() - started,
                ),
                None,
                None,
            )
        except DevelopmentUnitDurationExceeded as exc:
            executed = DevelopmentUnitRunResult(
                self._failure_record(
                    unit,
                    identity,
                    attempt_index=attempt_index,
                    retry_parent_intent_digest=parent_digest,
                    execution_status=(
                        "retry"
                        if attempt_index + 1 < unit.maximum_record_attempts
                        else "failed"
                    ),
                    failure_class="resource_failure",
                    failure_reason="unit_duration_exceeded",
                    actual_elapsed_seconds=exc.elapsed_seconds,
                ),
                None,
                None,
            )
        except (
            MemoryError,
            OSError,
            RuntimeAdapterError,
            RuntimeContentExecutionError,
            torch.cuda.OutOfMemoryError,
        ) as exc:
            executed = DevelopmentUnitRunResult(
                self._failure_record(
                    unit,
                    identity,
                    attempt_index=attempt_index,
                    retry_parent_intent_digest=parent_digest,
                    execution_status=(
                        "retry"
                        if attempt_index + 1 < unit.maximum_record_attempts
                        else "failed"
                    ),
                    failure_class="resource_failure",
                    failure_reason=type(exc).__name__,
                    actual_elapsed_seconds=monotonic() - started,
                ),
                None,
                None,
            )
        except Exception as exc:
            elapsed = monotonic() - started
            duration_exceeded = elapsed > unit.maximum_duration_seconds
            executed = DevelopmentUnitRunResult(
                self._failure_record(
                    unit,
                    identity,
                    attempt_index=attempt_index,
                    retry_parent_intent_digest=parent_digest,
                    execution_status=(
                        "retry"
                        if duration_exceeded
                        and attempt_index + 1 < unit.maximum_record_attempts
                        else "failed"
                    ),
                    failure_class=(
                        "resource_failure"
                        if duration_exceeded
                        else "implementation_failure"
                    ),
                    failure_reason=(
                        "unit_duration_exceeded"
                        if duration_exceeded
                        else f"{type(exc).__module__}.{type(exc).__qualname__}"
                    ),
                    actual_elapsed_seconds=elapsed,
                ),
                None,
                None,
            )
        if session_cursor is None:
            committed = self.persistence_store.commit_unit(
                lease,
                intent,
                record=executed.record,
                raw_secret_values=raw_secret_values,
                now_epoch_seconds=max(
                    now_epoch_seconds,
                    _intent_created_epoch(intent) + 1,
                    int(time()),
                ),
            )
        else:
            committed = self.persistence_store.commit_session_unit(
                session_cursor,
                lease,
                intent,
                record=executed.record,
                raw_secret_values=raw_secret_values,
                now_epoch_seconds=max(
                    now_epoch_seconds,
                    _intent_created_epoch(intent) + 1,
                    int(time()),
                ),
            )
        return DevelopmentUnitRunResult(executed.record, intent, committed)

    def build_verified_module_outcome_record(
        self,
        *,
        responsibility_id: str,
        cross_fit_plans: Mapping[str, FrozenDevelopmentCrossFitPlan] | None = None,
        now_epoch_seconds: int,
    ) -> DevelopmentVerifiedModuleOutcome:
        """Rebuild outcomes only from terminal records verified by the store."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError("verified module outcome requires persistence")
        plans = {} if cross_fit_plans is None else dict(cross_fit_plans)
        studies = tuple(self.protocol.module_matrix)
        studies_by_responsibility = {
            study.responsibility_id: study for study in studies
        }
        if responsibility_id not in studies_by_responsibility:
            raise DevelopmentRunnerError("module outcome responsibility is unknown")
        required_responsibilities: set[str] = set()

        def include_with_prerequisites(role: str) -> None:
            if role in required_responsibilities:
                return
            study = studies_by_responsibility[role]
            for prerequisite in study.prerequisite_responsibility_ids:
                include_with_prerequisites(prerequisite)
            required_responsibilities.add(role)

        include_with_prerequisites(responsibility_id)
        required_unit_indexes = tuple(
            binding.unit_index
            for binding in self.persistence_store.registered_unit_bindings
            if binding.responsibility_id in required_responsibilities
            and binding.phase not in OPERATIONAL_UNIT_PHASES
        )
        verified_evidence = (
            self.persistence_store
            .verified_terminal_scientific_evidence_for_unit_indexes(
                required_unit_indexes,
                now_epoch_seconds=now_epoch_seconds,
            )
        )
        records = tuple(record for record, _marker in verified_evidence)
        markers_by_record_id = {
            record.record_id: marker for record, marker in verified_evidence
        }
        outcomes: dict[str, DevelopmentModuleOutcomeRecord] = {}
        outcome_contexts: dict[str, DevelopmentVerifiedOutcomeEvidenceContext] = {}
        ordered_studies = tuple(
            studies_by_responsibility[role]
            for layer in DEVELOPMENT_DEPENDENCY_LAYERS
            for role in layer
            if role in required_responsibilities
        )
        for study in ordered_studies:
            evidence = tuple(
                record
                for record in records
                if record.responsibility_id == study.responsibility_id
            )
            if not evidence:
                raise DevelopmentRunnerError(
                    "verified store lacks module outcome evidence"
                )
            plan = plans.get(study.responsibility_id)
            if study.responsibility_id in {
                "lf_detector",
                "hf_detector",
                "content_detector",
            }:
                if type(plan) is not FrozenDevelopmentCrossFitPlan:
                    raise DevelopmentRunnerError(
                        "verified detector outcome lacks frozen cross-fit plan"
                    )
            elif plan is not None:
                raise DevelopmentRunnerError(
                    "cross-fit plan supplied for non-detector outcome"
                )
            thresholds: tuple[DevelopmentProvisionalThreshold, ...] = ()
            if (
                study.responsibility_id == "hf_detector"
                and all(record.execution_status == "success" for record in evidence)
            ):
                thresholds = self._replay_hf_detector_provisional_thresholds(
                    evidence,
                    plan,
                )
            outcome, outcome_context = self._build_module_outcome_record(
                evidence,
                committed_markers=tuple(
                    markers_by_record_id[record.record_id] for record in evidence
                ),
                responsibility_id=study.responsibility_id,
                now_epoch_seconds=now_epoch_seconds,
                cross_fit_plan=plan,
                provisional_threshold_identities=tuple(
                    item.threshold_identity
                    for item in sorted(
                        thresholds,
                        key=lambda item: item.fold_index,
                    )
                ),
                prerequisite_outcome_records=tuple(
                    (outcomes[role], outcome_contexts[role])
                    for role in study.prerequisite_responsibility_ids
                ),
            )
            if outcome.evidence_record_ids != tuple(
                record.record_id for record in evidence
            ) or outcome.evidence_record_digests != tuple(
                _scientific_record_digest(record) for record in evidence
            ):
                raise DevelopmentRunnerError(
                    "module outcome evidence did not replay from verified records"
                )
            outcomes[study.responsibility_id] = outcome
            outcome_contexts[study.responsibility_id] = outcome_context
        return DevelopmentVerifiedModuleOutcome(
            outcome_record=outcomes[responsibility_id],
            evidence_context=outcome_contexts[responsibility_id],
        )

    def persist_verified_module_outcome(
        self,
        outcome: DevelopmentVerifiedModuleOutcome,
    ) -> Path:
        """Persist a validated outcome as a create-only replay cache."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError("verified module outcome requires persistence")
        if (
            type(outcome) is not DevelopmentVerifiedModuleOutcome
            or outcome.validate_structure(self.protocol)
        ):
            raise DevelopmentRunnerError("verified module outcome is invalid")
        record = outcome.outcome_record
        return self.persistence_store.persist_verified_module_outcome(
            responsibility_id=record.responsibility_id,
            outcome_record_id=record.outcome_record_id,
            payload=asdict(record),
        )

    def replay_verified_hf_provisional_thresholds(
        self,
        *,
        cross_fit_plan: FrozenDevelopmentCrossFitPlan,
        now_epoch_seconds: int,
    ) -> tuple[DevelopmentProvisionalThreshold, ...]:
        """Rebuild HF thresholds only from the exact committed primary-null units."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError(
                "verified threshold replay requires persistence"
            )
        if (
            type(cross_fit_plan) is not FrozenDevelopmentCrossFitPlan
            or cross_fit_plan.responsibility_id != "hf_detector"
            or cross_fit_plan.validate()
        ):
            raise DevelopmentRunnerError("verified HF threshold plan is invalid")
        required_unit_indexes = tuple(
            binding.unit_index
            for binding in self.persistence_store.registered_unit_bindings
            if binding.responsibility_id == "hf_detector"
            and binding.content_branch_id == "hf_only"
            and binding.analysis_unit_identity.source_cluster_id
            in set(cross_fit_plan.source_cluster_ids)
        )
        if len(required_unit_indexes) != len(cross_fit_plan.source_cluster_ids):
            raise DevelopmentRunnerError(
                "verified HF threshold units differ from frozen plan"
            )
        evidence = (
            self.persistence_store
            .verified_terminal_scientific_evidence_for_unit_indexes(
                required_unit_indexes,
                now_epoch_seconds=now_epoch_seconds,
            )
        )
        return self.replay_hf_provisional_thresholds_from_evidence(
            cross_fit_plan=cross_fit_plan,
            verified_evidence=evidence,
        )

    def replay_hf_provisional_thresholds_from_evidence(
        self,
        *,
        cross_fit_plan: FrozenDevelopmentCrossFitPlan,
        verified_evidence: Sequence[
            tuple[DevelopmentScientificRecord, CommittedUnit]
        ],
    ) -> tuple[DevelopmentProvisionalThreshold, ...]:
        """Use exact session-start or incrementally committed verified evidence."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError(
                "verified threshold replay requires persistence"
            )
        if (
            type(cross_fit_plan) is not FrozenDevelopmentCrossFitPlan
            or cross_fit_plan.responsibility_id != "hf_detector"
            or cross_fit_plan.validate()
        ):
            raise DevelopmentRunnerError("verified HF threshold plan is invalid")
        expected_indexes = tuple(
            binding.unit_index
            for binding in self.persistence_store.registered_unit_bindings
            if binding.responsibility_id == "hf_detector"
            and binding.content_branch_id == "hf_only"
            and binding.analysis_unit_identity.source_cluster_id
            in set(cross_fit_plan.source_cluster_ids)
        )
        selected = tuple(
            (record, marker)
            for record, marker in verified_evidence
            if record.unit_index in set(expected_indexes)
        )
        if (
            tuple(record.unit_index for record, _marker in selected)
            != expected_indexes
            or any(
                type(record) is not DevelopmentScientificRecord
                or marker.record_id != record.record_id
                or marker.attempt_disposition == "retryable_resource_failure"
                for record, marker in selected
            )
        ):
            raise DevelopmentRunnerError(
                "verified HF threshold evidence differs from frozen plan"
            )
        return self._replay_hf_detector_provisional_thresholds(
            tuple(record for record, _marker in selected),
            cross_fit_plan,
        )

    def verify_hf_provisional_thresholds_from_store(
        self,
        provisional_thresholds: Sequence[DevelopmentProvisionalThreshold],
        *,
        cross_fit_plan: FrozenDevelopmentCrossFitPlan,
        now_epoch_seconds: int,
    ) -> tuple[DevelopmentProvisionalThreshold, ...]:
        """Reject any threshold set that was not replayed from the same store."""

        supplied = tuple(provisional_thresholds)
        replayed = self.replay_verified_hf_provisional_thresholds(
            cross_fit_plan=cross_fit_plan,
            now_epoch_seconds=now_epoch_seconds,
        )
        if supplied != replayed:
            raise DevelopmentRunnerError(
                "provisional thresholds differ from persistent-store replay"
            )
        return replayed

    def decide_verified_module_execution(
        self,
        *,
        responsibility_id: str,
        outcomes_by_responsibility: Mapping[
            str, DevelopmentVerifiedModuleOutcome
        ],
        cross_fit_plans: Mapping[str, FrozenDevelopmentCrossFitPlan] | None = None,
        now_epoch_seconds: int,
    ) -> DevelopmentModuleExecutionDecision:
        """Approve dependencies only after replaying the same persistent store."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError(
                "verified module decision requires persistence"
            )
        studies = {
            item.responsibility_id: item for item in self.protocol.module_matrix
        }
        study = studies.get(responsibility_id)
        if study is None:
            raise DevelopmentRunnerError(
                "module decision responsibility is unknown"
            )
        unknown = set(outcomes_by_responsibility) - set(studies)
        if unknown:
            raise DevelopmentRunnerError(
                "module decision contains unknown outcome responsibility"
            )
        missing = tuple(
            role
            for role in study.prerequisite_responsibility_ids
            if role not in outcomes_by_responsibility
        )
        if missing:
            return DevelopmentModuleExecutionDecision(
                False,
                responsibility_id,
                missing,
                (),
                "prerequisite_outcome_missing",
            )
        replayed: dict[str, DevelopmentVerifiedModuleOutcome] = {}
        for role in study.prerequisite_responsibility_ids:
            supplied = outcomes_by_responsibility[role]
            if (
                type(supplied) is not DevelopmentVerifiedModuleOutcome
                or supplied.validate_structure(self.protocol)
            ):
                raise DevelopmentRunnerError(
                    "module decision requires structured verified outcome bundle"
                )
            rebuilt = self.build_verified_module_outcome_record(
                responsibility_id=role,
                cross_fit_plans=cross_fit_plans,
                now_epoch_seconds=now_epoch_seconds,
            )
            if supplied != rebuilt:
                raise DevelopmentRunnerError(
                    "module decision outcome differs from persistent-store replay"
                )
            replayed[role] = rebuilt
        blocking = tuple(
            role
            for role in study.prerequisite_responsibility_ids
            if replayed[role].outcome_record.module_outcome
            != "mechanism_signal_observed"
        )
        if blocking:
            return DevelopmentModuleExecutionDecision(
                False,
                responsibility_id,
                (),
                blocking,
                "stop_when_any_prerequisite_lacks_mechanism_signal_observed",
            )
        return DevelopmentModuleExecutionDecision(
            True,
            responsibility_id,
            (),
            (),
            "development_execution_authorized",
        )

    def _verified_outcome_context(
        self,
        *,
        study: object,
        records: Sequence[DevelopmentScientificRecord],
        committed_markers: Sequence[CommittedUnit],
        aggregate_metric_means: Sequence[tuple[str, float]],
        source_cluster_count: int,
        module_outcome: str,
        candidate_recommendation: str,
        blocking_responsibilities: Sequence[str],
        cross_fit_plan: FrozenDevelopmentCrossFitPlan | None,
        provisional_threshold_identities: Sequence[str],
    ) -> DevelopmentVerifiedOutcomeEvidenceContext:
        evidence_bindings = tuple(
            (record.record_id, _scientific_record_digest(record))
            for record in records
        )
        marker_bindings = tuple(
            (record_id, record_digest, marker.digest())
            for (record_id, record_digest), marker in zip(
                evidence_bindings,
                committed_markers,
                strict=True,
            )
        )
        metric_means = tuple(
            (metric_id, float(value))
            for metric_id, value in aggregate_metric_means
        )
        aggregate_digest = _canonical_digest(
            {
                "responsibility_id": study.responsibility_id,
                "source_cluster_count": source_cluster_count,
                "aggregate_metric_means": metric_means,
                "evidence_record_bindings": evidence_bindings,
            }
        )
        return DevelopmentVerifiedOutcomeEvidenceContext(
            protocol_digest=self.protocol.digest(),
            execution_intent_authority_digest=self.intent_authority.authority_digest,
            input_manifest_digest=self.intent_authority.input_manifest_digest,
            candidate_config_digest=study.candidate_config_digest,
            signal_criteria_digest=study.signal_criteria_digest(),
            cluster_aggregate_digest=aggregate_digest,
            source_cluster_count=source_cluster_count,
            aggregate_metric_means=metric_means,
            evidence_record_bindings=evidence_bindings,
            committed_marker_bindings=marker_bindings,
            cross_fit_plan_digest=(
                _canonical_digest(asdict(cross_fit_plan))
                if cross_fit_plan is not None
                else None
            ),
            provisional_threshold_identities=tuple(
                provisional_threshold_identities
            ),
            verified_module_outcome=module_outcome,
            verified_candidate_recommendation=candidate_recommendation,
            verified_blocking_responsibilities=tuple(
                blocking_responsibilities
            ),
        )

    def _aggregate_exact_frozen_cases(
        self,
        study: DevelopmentModuleStudy,
        records: Sequence[DevelopmentScientificRecord],
        observations: Sequence[DevelopmentMetricObservation],
    ) -> DevelopmentClusterAggregate:
        """Aggregate the exact frozen case roster with source clusters equally weighted."""

        if self.persistence_store is None or len(records) != len(observations):
            raise DevelopmentRunnerError(
                "frozen case aggregation requires persisted one-to-one observations"
            )
        bindings = {
            item.unit_index: item
            for item in self.persistence_store.registered_unit_bindings
            if item.responsibility_id == study.responsibility_id
            and item.phase not in OPERATIONAL_UNIT_PHASES
        }
        if set(bindings) != {record.unit_index for record in records}:
            raise DevelopmentRunnerError(
                "module observations do not cover the exact frozen case roster"
            )
        metric_ids = tuple(study.metric_ids)
        values_by_cluster: dict[str, dict[str, list[float]]] = {}
        observed_cases: set[tuple[str, str, str]] = set()
        for record, observation in zip(records, observations, strict=True):
            observation.validate()
            binding = bindings[record.unit_index]
            source_cluster_id = binding.analysis_unit_identity.source_cluster_id
            case_identity = (
                source_cluster_id,
                binding.content_branch_id,
                binding.geometry_case_id,
            )
            if (
                case_identity in observed_cases
                or observation.responsibility_id != study.responsibility_id
                or observation.source_cluster_id != source_cluster_id
                or observation.content_branch_id != binding.content_branch_id
                or observation.geometry_case_id != binding.geometry_case_id
                or observation.registered_metric_ids != metric_ids
                or observation.candidate_config_digest
                != study.candidate_config_digest
                or observation.paired_ablation_identity
                != study.paired_ablation_identity
            ):
                raise DevelopmentRunnerError(
                    "module observation differs from its exact frozen case"
                )
            observed_cases.add(case_identity)
            statistics = dict(observation.sufficient_statistics)
            if set(statistics) != set(metric_ids):
                raise DevelopmentRunnerError(
                    "module observation metric identities drifted"
                )
            cluster_values = values_by_cluster.setdefault(
                source_cluster_id,
                {metric_id: [] for metric_id in metric_ids},
            )
            for metric_id in metric_ids:
                cluster_values[metric_id].append(float(statistics[metric_id]))
        if len(values_by_cluster) != study.scientific_source_cluster_scale:
            raise DevelopmentRunnerError(
                "module frozen case roster source-cluster scale drifted"
            )
        cluster_means = {
            metric_id: tuple(
                sum(values[metric_id]) / len(values[metric_id])
                for _, values in sorted(values_by_cluster.items())
            )
            for metric_id in metric_ids
        }
        payload = {
            "responsibility_id": study.responsibility_id,
            "source_cluster_count": len(values_by_cluster),
            "metric_medians": tuple(
                (metric_id, float(median(cluster_means[metric_id])))
                for metric_id in metric_ids
            ),
            "metric_means": tuple(
                (
                    metric_id,
                    sum(cluster_means[metric_id]) / len(cluster_means[metric_id]),
                )
                for metric_id in metric_ids
            ),
            "source_cluster_digest": _canonical_digest(
                tuple(sorted(values_by_cluster))
            ),
        }
        return DevelopmentClusterAggregate(
            **payload,
            aggregate_digest=_canonical_digest(payload),
        )

    def _screening_content_hf_non_degradation(
        self,
        observations: Sequence[DevelopmentMetricObservation],
        *,
        paired_hf_clusters: frozenset[str],
        combined_clusters: frozenset[str],
        cross_fit_plan: FrozenDevelopmentCrossFitPlan,
    ) -> float:
        """Use the same combined operation's HF-only score for screening only."""

        if (
            self.protocol.protocol_id
            != thirteen_module_mechanism_screening_protocol_id
            or paired_hf_clusters
            or cross_fit_plan.responsibility_id != "content_detector"
        ):
            raise DevelopmentRunnerError(
                "content detector same-operation comparison is not authorized"
            )
        expected_clusters = frozenset(cross_fit_plan.source_cluster_ids)
        if (
            len(observations) != 16
            or len(combined_clusters) != 16
            or combined_clusters != expected_clusters
        ):
            raise DevelopmentRunnerError(
                "screening content detector combined cluster roster is incomplete"
            )
        required_statistics = {
            "candidate_score",
            "hf_only_score",
            "wrong_key_score",
            "primary_null_score",
            "candidate_minus_hf_only",
            "lf_score",
        }
        by_cluster: dict[str, tuple[float, float, float]] = {}
        for observation in observations:
            statistics = dict(observation.sufficient_statistics)
            if (
                observation.content_branch_id != "lf_hf_routed_combination"
                or set(statistics) != required_statistics
                or observation.source_cluster_id in by_cluster
            ):
                raise DevelopmentRunnerError(
                    "screening content detector same-operation metrics are incomplete"
                )
            candidate_score = float(statistics["candidate_score"])
            hf_only_score = float(statistics["hf_only_score"])
            candidate_minus_hf_only = float(
                statistics["candidate_minus_hf_only"]
            )
            if (
                not all(
                    isfinite(value)
                    for value in (
                        candidate_score,
                        hf_only_score,
                        candidate_minus_hf_only,
                    )
                )
                or candidate_minus_hf_only != candidate_score - hf_only_score
            ):
                raise DevelopmentRunnerError(
                    "screening content detector same-operation score binding drifted"
                )
            by_cluster[observation.source_cluster_id] = (
                candidate_score,
                hf_only_score,
                candidate_minus_hf_only,
            )
        if frozenset(by_cluster) != expected_clusters:
            raise DevelopmentRunnerError(
                "screening content detector metric clusters differ from cross-fit"
            )
        return sum(
            candidate_score >= hf_only_score
            for candidate_score, hf_only_score, _delta in by_cluster.values()
        ) / len(by_cluster)

    def _build_module_outcome_record(
        self,
        records: Sequence[DevelopmentScientificRecord],
        *,
        responsibility_id: str,
        now_epoch_seconds: int,
        committed_markers: Sequence[CommittedUnit] = (),
        cross_fit_plan: FrozenDevelopmentCrossFitPlan | None = None,
        provisional_threshold_identities: Sequence[str] = (),
        prerequisite_outcome_records: Sequence[
            tuple[DevelopmentModuleOutcomeRecord, DevelopmentVerifiedOutcomeEvidenceContext]
        ] = (),
    ) -> tuple[DevelopmentModuleOutcomeRecord, DevelopmentVerifiedOutcomeEvidenceContext]:
        """Apply frozen criteria to records supplied only by the verified scheduler."""

        if self.persistence_store is None:
            raise DevelopmentRunnerError(
                "module outcome builder requires persistent-store replay"
            )
        if len(committed_markers) != len(records):
            raise DevelopmentRunnerError(
                "module outcome requires one verified marker per record"
            )
        unit_indexes = tuple(record.unit_index for record in records)
        replayed_evidence = (
            self.persistence_store
            .verified_terminal_scientific_evidence_for_unit_indexes(
                unit_indexes,
                now_epoch_seconds=now_epoch_seconds,
            )
        )
        if tuple(zip(records, committed_markers, strict=True)) != replayed_evidence:
            raise DevelopmentRunnerError(
                "module outcome evidence differs from persistent-store replay"
            )

        study = next(
            (
                item
                for item in self.protocol.module_matrix
                if item.responsibility_id == responsibility_id
            ),
            None,
        )
        if study is None or not records:
            raise DevelopmentRunnerError("module outcome records are missing")
        prerequisite_by_responsibility = {
            item.responsibility_id: (item, context)
            for item, context in prerequisite_outcome_records
        }
        if (
            len(prerequisite_by_responsibility) != len(prerequisite_outcome_records)
            or set(prerequisite_by_responsibility)
            != set(study.prerequisite_responsibility_ids)
            or any(
                item.validate(
                    self.protocol,
                    verified_evidence_context=context,
                )
                for item, context in prerequisite_outcome_records
            )
        ):
            raise DevelopmentRunnerError(
                "module prerequisites require exact verified outcome records"
            )
        if any(
            type(record) is not DevelopmentScientificRecord
            or record.responsibility_id != responsibility_id
            or record.protocol_digest != self.protocol.digest()
            or record.execution_intent_authority_digest
            != self.intent_authority.authority_digest
            for record in records
        ):
            raise DevelopmentRunnerError("module outcome evidence records drifted")
        for record in records:
            self._validate_record(record, study)
        statuses = {record.execution_status for record in records}
        if statuses - {"success", "failed", "excluded", "retry"}:
            raise DevelopmentRunnerError("module outcome execution status is invalid")
        failures = tuple(record for record in records if record.execution_status != "success")
        if failures:
            resource_only = all(record.failure_class == "resource_failure" for record in failures)
            module_outcome = "resource_blocked" if resource_only else "implementation_blocked"
            candidate_recommendation = "candidate_not_recommended_for_selection"
            recommendation_reason = (
                "verified_resource_failures_prevented_frozen_coverage"
                if resource_only
                else "verified_implementation_or_exclusion_records_prevented_frozen_coverage"
            )
            blocking = (responsibility_id,)
            verified_blocking = blocking if not resource_only else ()
            context = self._verified_outcome_context(
                study=study,
                records=records,
                committed_markers=committed_markers,
                aggregate_metric_means=(),
                source_cluster_count=len(
                    {
                        record.analysis_unit_identity["source_cluster_id"]
                        for record in records
                    }
                ),
                module_outcome=module_outcome,
                candidate_recommendation=candidate_recommendation,
                blocking_responsibilities=verified_blocking,
                cross_fit_plan=cross_fit_plan,
                provisional_threshold_identities=provisional_threshold_identities,
            )
            outcome = _create_verified_development_module_outcome_record(
                self.protocol,
                responsibility_id=responsibility_id,
                module_outcome=module_outcome,
                candidate_recommendation=candidate_recommendation,
                recommendation_reason=recommendation_reason,
                verified_evidence_context=context,
            )
            return outcome, context
        blocked_by = tuple(
            role
            for role in study.prerequisite_responsibility_ids
            if prerequisite_by_responsibility[role][0].module_outcome
            != "mechanism_signal_observed"
        )
        if blocked_by:
            context = self._verified_outcome_context(
                study=study,
                records=records,
                committed_markers=committed_markers,
                aggregate_metric_means=(),
                source_cluster_count=len(
                    {
                        record.analysis_unit_identity["source_cluster_id"]
                        for record in records
                    }
                ),
                module_outcome="implementation_blocked",
                candidate_recommendation="candidate_not_recommended_for_selection",
                blocking_responsibilities=blocked_by,
                cross_fit_plan=cross_fit_plan,
                provisional_threshold_identities=provisional_threshold_identities,
            )
            outcome = _create_verified_development_module_outcome_record(
                self.protocol,
                responsibility_id=responsibility_id,
                module_outcome="implementation_blocked",
                candidate_recommendation="candidate_not_recommended_for_selection",
                recommendation_reason="verified_prerequisite_outcome_stopped_scientific_interpretation",
                verified_evidence_context=context,
            )
            return outcome, context
        observations = tuple(
            DevelopmentMetricObservation(**record.metric_observation)
            for record in records
        )
        if responsibility_id in {"lf_detector", "hf_detector", "content_detector"}:
            if type(cross_fit_plan) is not FrozenDevelopmentCrossFitPlan:
                raise DevelopmentRunnerError("detector outcome requires the exact frozen cross-fit plan")
            cross_fit = cross_fit_development_detection_metrics(
                responsibility_id, observations, plan=cross_fit_plan
            )
            if responsibility_id == "hf_detector":
                replayed_thresholds = self._replay_hf_detector_provisional_thresholds(
                    records, cross_fit_plan
                )
                if tuple(
                    (item.fold_index, item.threshold)
                    for item in replayed_thresholds
                ) != cross_fit.fold_thresholds:
                    raise DevelopmentRunnerError(
                        "protocol threshold replay differs from metric cross-fit"
                    )
            if responsibility_id == "lf_detector":
                means = {
                    "lf_tpr_at_selection_fpr": cross_fit.registered_accept_rate,
                    "lf_wrong_key_rate": cross_fit.wrong_key_accept_rate,
                }
            elif responsibility_id == "hf_detector":
                means = {
                    "hf_tpr_at_frozen_fpr": cross_fit.registered_accept_rate,
                    "hf_wrong_key_rate": cross_fit.wrong_key_accept_rate,
                }
            else:
                hf_observations = tuple(
                    item
                    for item in observations
                    if item.content_branch_id == "hf_only"
                )
                combined_observations = tuple(
                    item
                    for item in observations
                    if item.content_branch_id == "lf_hf_routed_combination"
                )
                hf_by_cluster = {
                    item.source_cluster_id: dict(item.sufficient_statistics)[
                        "candidate_score"
                    ]
                    for item in hf_observations
                }
                combined_by_cluster = {
                    item.source_cluster_id: dict(item.sufficient_statistics)[
                        "candidate_score"
                    ]
                    for item in combined_observations
                }
                paired_hf_clusters = {
                    binding.analysis_unit_identity.source_cluster_id
                    for binding in self.persistence_store.registered_unit_bindings
                    if binding.responsibility_id == "content_detector"
                    and binding.phase == "development_paired_ablation"
                    and binding.content_branch_id == "hf_only"
                }
                combined_clusters = {
                    binding.analysis_unit_identity.source_cluster_id
                    for binding in self.persistence_store.registered_unit_bindings
                    if binding.responsibility_id == "content_detector"
                    and binding.phase
                    == "development_scientific_responsibility_case"
                    and binding.content_branch_id == "lf_hf_routed_combination"
                }
                if (
                    self.protocol.protocol_id
                    == thirteen_module_mechanism_screening_protocol_id
                    and not paired_hf_clusters
                ):
                    if hf_observations:
                        raise DevelopmentRunnerError(
                            "screening content detector has unregistered HF paired evidence"
                        )
                    hf_non_degradation = self._screening_content_hf_non_degradation(
                        combined_observations,
                        paired_hf_clusters=frozenset(paired_hf_clusters),
                        combined_clusters=frozenset(combined_clusters),
                        cross_fit_plan=cross_fit_plan,
                    )
                else:
                    if (
                        len(hf_observations) != len(hf_by_cluster)
                        or len(combined_observations) != len(combined_by_cluster)
                        or set(hf_by_cluster) != paired_hf_clusters
                        or set(combined_by_cluster) != combined_clusters
                        or not paired_hf_clusters
                        or not paired_hf_clusters.issubset(combined_clusters)
                    ):
                        raise DevelopmentRunnerError(
                            "content detector HF/combined frozen paired clusters are incomplete"
                        )
                    hf_non_degradation = sum(
                        combined_by_cluster[key] >= hf_by_cluster[key]
                        for key in hf_by_cluster
                    ) / len(hf_by_cluster)
                means = {
                    "combined_tpr": cross_fit.registered_accept_rate,
                    "combined_primary_null_fpr": cross_fit.primary_null_false_accept_rate,
                    "hf_non_degradation": hf_non_degradation,
                    "wrong_key_rate": cross_fit.wrong_key_accept_rate,
                }
            aggregate = DevelopmentClusterAggregate(
                responsibility_id=responsibility_id,
                source_cluster_count=cross_fit.source_cluster_count,
                metric_medians=tuple((key, value) for key, value in means.items()),
                metric_means=tuple((key, value) for key, value in means.items()),
                source_cluster_digest=cross_fit.fold_assignment_digest,
                aggregate_digest=cross_fit.aggregate_digest,
            )
        else:
            aggregate = self._aggregate_exact_frozen_cases(
                study,
                records,
                observations,
            )
            if responsibility_id == "content_router":
                eligible = tuple(
                    item
                    for item in observations
                    if item.content_branch_id == "lf_hf_routed_combination"
                )
                if len(eligible) != study.scientific_source_cluster_scale:
                    raise DevelopmentRunnerError(
                        "routing outcome lacks one real paired comparison per cluster"
                    )
                eligible_values = tuple(
                    dict(item.sufficient_statistics) for item in eligible
                )
                means = {
                    metric_id: sum(values[metric_id] for values in eligible_values)
                    / len(eligible_values)
                    for metric_id in study.metric_ids
                }
                aggregate = DevelopmentClusterAggregate(
                    responsibility_id=responsibility_id,
                    source_cluster_count=len(eligible),
                    metric_medians=tuple(
                        (metric_id, means[metric_id]) for metric_id in study.metric_ids
                    ),
                    metric_means=tuple(
                        (metric_id, means[metric_id]) for metric_id in study.metric_ids
                    ),
                    source_cluster_digest=aggregate.source_cluster_digest,
                    aggregate_digest=_canonical_digest(
                        {
                            "paired_validation": aggregate.aggregate_digest,
                            "eligible_observations": tuple(
                                item.observation_digest for item in eligible
                            ),
                            "means": means,
                        }
                    ),
                )
            elif responsibility_id == "geometry_reliability":
                negative_cases = set(GEOMETRY_NEGATIVE_CONTROL_CASE_IDS)
                positive = tuple(
                    item
                    for item in observations
                    if item.geometry_case_id not in negative_cases
                )
                negative = tuple(
                    item
                    for item in observations
                    if item.geometry_case_id in negative_cases
                )
                if not positive or not negative:
                    raise DevelopmentRunnerError(
                        "geometry reliability outcome lacks positive or negative controls"
                    )
                positive_values = tuple(
                    dict(item.sufficient_statistics) for item in positive
                )
                negative_values = tuple(
                    dict(item.sufficient_statistics) for item in negative
                )
                means = {
                    "reliable_accept_rate": sum(
                        item["reliable_accept_rate"] for item in positive_values
                    )
                    / len(positive_values),
                    "unreliable_reject_rate": sum(
                        item["unreliable_reject_rate"]
                        for item in (*positive_values, *negative_values)
                    )
                    / len(observations),
                    "false_reliable_rate": sum(
                        item["false_reliable_rate"]
                        for item in (*positive_values, *negative_values)
                    )
                    / len(observations),
                }
                aggregate = DevelopmentClusterAggregate(
                    responsibility_id=responsibility_id,
                    source_cluster_count=len(
                        {item.source_cluster_id for item in observations}
                    ),
                    metric_medians=tuple(
                        (metric_id, means[metric_id]) for metric_id in study.metric_ids
                    ),
                    metric_means=tuple(
                        (metric_id, means[metric_id]) for metric_id in study.metric_ids
                    ),
                    source_cluster_digest=aggregate.source_cluster_digest,
                    aggregate_digest=_canonical_digest(
                        {
                            "paired_validation": aggregate.aggregate_digest,
                            "positive_observations": tuple(
                                item.observation_digest for item in positive
                            ),
                            "negative_observations": tuple(
                                item.observation_digest for item in negative
                            ),
                            "means": means,
                        }
                    ),
                )
        means = dict(aggregate.metric_means)
        signal = self._scientific_signal_observed(study, means)
        module_outcome = "mechanism_signal_observed" if signal else "mechanism_signal_not_observed"
        candidate_recommendation = (
            "candidate_worth_further_selection"
            if signal
            else "candidate_not_recommended_for_selection"
        )
        recommendation_reason = (
            "frozen_registered_metrics_satisfied_development_signal_criteria"
            if signal
            else "frozen_registered_metrics_did_not_satisfy_development_signal_criteria"
        )
        context = self._verified_outcome_context(
            study=study,
            records=records,
            committed_markers=committed_markers,
            aggregate_metric_means=aggregate.metric_means,
            source_cluster_count=aggregate.source_cluster_count,
            module_outcome=module_outcome,
            candidate_recommendation=candidate_recommendation,
            blocking_responsibilities=(),
            cross_fit_plan=cross_fit_plan,
            provisional_threshold_identities=provisional_threshold_identities,
        )
        outcome = _create_verified_development_module_outcome_record(
            self.protocol,
            responsibility_id=responsibility_id,
            module_outcome=module_outcome,
            candidate_recommendation=candidate_recommendation,
            recommendation_reason=recommendation_reason,
            verified_evidence_context=context,
        )
        return outcome, context

    def _replay_hf_detector_provisional_thresholds(
        self,
        records: Sequence[DevelopmentScientificRecord],
        plan: FrozenDevelopmentCrossFitPlan,
    ) -> tuple[DevelopmentProvisionalThreshold, ...]:
        """Rebuild actual provisional thresholds from persisted HF records."""

        by_cluster = {
            AnalysisUnitIdentity(
                **record.analysis_unit_identity
            ).source_cluster_id: record
            for record in records
            if record.content_branch_id == "hf_only"
        }
        if len(by_cluster) != len(plan.source_cluster_ids):
            raise DevelopmentRunnerError(
                "HF threshold replay lacks the frozen cluster roster"
            )
        thresholds: list[DevelopmentProvisionalThreshold] = []
        for fold in plan.folds:
            fit_clusters = set(fold.fit_source_cluster_ids)
            key_bindings = tuple(
                item
                for item in plan.execution_intent_authority.public_key_roster
                if item.source_cluster_id in fit_clusters
            )
            detector_binding = create_development_threshold_detector_binding(
                plan,
                expected_execution_intent_authority_digest=(
                    plan.expected_execution_intent_authority_digest
                ),
                fold_index=fold.fold_index,
                input_manifest=plan.input_manifest,
                primary_null_key_bindings=key_bindings,
            )
            fit_inputs = tuple(
                create_development_threshold_fit_input(
                    expected_execution_intent_authority_digest=(
                        plan.expected_execution_intent_authority_digest
                    ),
                    source_record=by_cluster[cluster_id],
                )
                for cluster_id in fold.fit_source_cluster_ids
            )
            provisional = create_development_provisional_threshold(
                plan,
                expected_execution_intent_authority_digest=(
                    plan.expected_execution_intent_authority_digest
                ),
                fold_index=fold.fold_index,
                input_manifest=plan.input_manifest,
                detector_binding=detector_binding,
                fit_inputs=fit_inputs,
            )
            thresholds.append(provisional)
        return tuple(thresholds)

    @staticmethod
    def _scientific_signal_observed(study, values: Mapping[str, float]) -> bool:
        """Apply every checked-in development-only metric criterion."""

        if set(values) != set(study.metric_ids):
            return False
        if tuple(criterion.metric_id for criterion in study.signal_criteria) != (
            study.metric_ids
        ):
            return False
        return all(
            criterion.satisfied_by(values[criterion.metric_id])
            for criterion in study.signal_criteria
        )

    def _execute_geometry_operation(
        self,
        unit: DevelopmentStudyUnit,
        identity: AnalysisUnitIdentity,
        raw: DevelopmentUnitInput,
        *,
        runtime_result: ContentWriteVaeResult,
        wrong_material: object,
        traces: dict[str, object],
    ) -> tuple[
        object,
        DevelopmentMetricObservation,
        dict[str, object],
        InternalValidationRecord | None,
    ]:
        """Run the public geometry chain before any content-detector construction."""

        responsibility = unit.responsibility_id
        source_image = _decoded_image_to_rgb8(runtime_result.watermarked_image)
        attack_specification = _frozen_geometry_attack_specification(
            self.protocol,
            unit.geometry_case_id,
        )
        attacked = apply_geometric_attack(
            AttackArtifact(identity, source_image),
            attack_specification,
            registry=self.attack_registry,
        )
        runtime_qk = self.runtime_adapter.observe_detection_qk(
            attacked.attacked_artifact.image
        )
        qk_registered = self.adapter.synchronize_qk_observation(
            runtime_qk,
            raw.registered_root_key,
        ).result
        qk_wrong = self.adapter.synchronize_qk_observation(
            runtime_qk,
            wrong_material,
        ).result
        traces.update(
            geometry_operation_identity=attack_specification.attack_id,
            attack_config_digest=attack_specification.attack_config_digest,
            qk_registered_relation_score=qk_registered.relation_score,
            qk_wrong_relation_score=qk_wrong.relation_score,
        )
        if responsibility == "qk_geometry_sync":
            return qk_registered, metric_qk_geometry_sync(
                identity.source_cluster_id,
                registered_relation_score=qk_registered.relation_score,
                wrong_key_relation_score=qk_wrong.relation_score,
                registered_descriptor_digest=qk_registered.descriptor_digest,
                registered_projection_digest=qk_registered.projection_digest,
                wrong_projection_digest=qk_wrong.projection_digest,
                quality_delta=_tensor_relative_l2(
                    source_image.to(dtype=torch.float32),
                    attacked.attacked_artifact.image.to(dtype=torch.float32),
                ),
            ), traces, None
        estimation = self.adapter.estimate_geometric_transform(
            qk_registered,
            raw.registered_root_key,
            epsilon_inlier=raw.epsilon_inlier,
        ).result
        traces.update(
            geometry_estimation_identity=estimation.estimation_identity_digest,
            geometry_transform=_public_payload(estimation.transform),
            geometry_raw_metrics={
                "coverage": estimation.coverage,
                "uniqueness": estimation.uniqueness,
                "gap": estimation.gap,
                "key_margin": estimation.key_margin,
                "inlier_ratio": estimation.inlier_ratio,
                "mean_residual": _public_payload(estimation.mean_residual),
            },
        )
        if responsibility == "geometric_transform_estimator":
            metric = metric_geometric_transform_estimator(
                identity.source_cluster_id,
                estimated_log_scale=estimation.transform.log_scale,
                estimated_rotation_degrees=(
                    estimation.transform.residual_rotation_degrees
                ),
                estimated_coverage=estimation.coverage,
                mean_residual=estimation.mean_residual,
                key_margin=estimation.key_margin,
                estimation_identity_digest=estimation.estimation_identity_digest,
                search_config_digest=estimation.search_config_digest,
                truth_crop_fraction=attack_specification.crop_fraction,
                truth_scale=attack_specification.scale_factor,
                truth_rotation_degrees=attack_specification.rotation_degrees,
                estimated_translation_x=estimation.transform.translation_x,
                estimated_translation_y=estimation.transform.translation_y,
                truth_translation_x=attacked.output_to_input_matrix[0][2],
                truth_translation_y=attacked.output_to_input_matrix[1][2],
            )
            return estimation, metric, traces, None
        wrong_estimation = self.adapter.estimate_geometric_transform(
            qk_wrong,
            wrong_material,
            epsilon_inlier=raw.epsilon_inlier,
        ).result
        reliability = self.adapter.assess_geometry_reliability(
            estimation,
            raw.geometry_reliability_thresholds,
        ).result
        wrong_reliability = self.adapter.assess_geometry_reliability(
            wrong_estimation,
            raw.geometry_reliability_thresholds,
        ).result
        ambiguous_control_realized = (
            unit.geometry_case_id != "ambiguous_transform_control"
            or estimation.gap
            <= self.protocol.geometry_study.ambiguous_control_max_top_two_gap
        )
        traces.update(
            geometry_reliability_identity=reliability.reliability_identity_digest,
            geometry_reliable=reliability.reliable,
            geometry_reliability_config_digest=reliability.threshold_config_digest,
            geometry_reliability_status=reliability.status,
            geometry_reliability_failure_reasons=reliability.failure_reasons,
            wrong_key_geometry_estimation_identity=(
                wrong_estimation.estimation_identity_digest
            ),
            wrong_key_geometry_reliability_identity=(
                wrong_reliability.reliability_identity_digest
            ),
            wrong_key_geometry_reliable=wrong_reliability.reliable,
            wrong_key_geometry_reliability_status=wrong_reliability.status,
            wrong_key_geometry_reliability_failure_reasons=(
                wrong_reliability.failure_reasons
            ),
            ambiguous_control_realized=ambiguous_control_realized,
        )
        if responsibility == "geometry_reliability":
            if not ambiguous_control_realized:
                raise DevelopmentUnitExcluded(
                    "control_not_realized:ambiguous_transform_control"
                )
            return reliability, metric_geometry_reliability(
                identity.source_cluster_id,
                registered_reliability_accepted=reliability.reliable,
                wrong_key_reliability_accepted=wrong_reliability.reliable,
                is_unreliable_control=(
                    unit.geometry_case_id in GEOMETRY_NEGATIVE_CONTROL_CASE_IDS
                ),
                registered_reliability_identity_digest=(
                    reliability.reliability_identity_digest
                ),
                wrong_key_reliability_identity_digest=(
                    wrong_reliability.reliability_identity_digest
                ),
                registered_estimation_identity_digest=(
                    estimation.estimation_identity_digest
                ),
                wrong_key_estimation_identity_digest=(
                    wrong_estimation.estimation_identity_digest
                ),
            ), traces, None
        if not reliability.reliable:
            raise DevelopmentUnitExcluded(
                "rectification unit was blocked by fail-closed reliability"
            )
        rectified = self.adapter.rectify_image(
            attacked.attacked_artifact.image,
            estimation,
            reliability,
        ).result
        content_operation = FormalHfContentDetectionOperation(self.adapter)
        attacked_content = content_operation(
            attacked.attacked_artifact.image,
            raw.registered_root_key,
        )
        rectified_content = content_operation(
            rectified.rectified_image,
            raw.registered_root_key,
        )
        traces.update(
            raw_detector_identity=attacked_content.detector_identity,
            rectified_detector_identity=rectified_content.detector_identity,
            raw_detector_config_digest=attacked_content.content_config_digest,
            rectified_detector_config_digest=rectified_content.content_config_digest,
            raw_preprocessing_identity=content_operation.preprocessing_identity,
            rectified_preprocessing_identity=content_operation.preprocessing_identity,
            raw_content_score=attacked_content.content_score,
            rectified_content_score=rectified_content.content_score,
            rectification_status="succeeded",
        )
        if responsibility == "image_rectifier":
            return rectified, metric_image_rectifier(
                identity.source_cluster_id,
                attacked_content_score=attacked_content.content_score,
                rectified_content_score=rectified_content.content_score,
                token_crop_support=rectified.token_crop_support,
                pixel_crop_support=rectified.pixel_crop_support,
                rectified_image_digest=rectified.rectified_image_digest,
                rectification_config_digest=rectified.rectification_config_digest,
                rectification_quality=_tensor_relative_l2(
                    source_image.to(dtype=torch.float32),
                    rectified.rectified_image.to(dtype=torch.float32),
                ),
            ), traces, None
        raise DevelopmentRunnerError("geometry responsibility dispatch is incomplete")

    def _execute_real_operation(
        self,
        unit: DevelopmentStudyUnit,
        identity: AnalysisUnitIdentity,
        raw: DevelopmentUnitInput,
    ) -> tuple[object, DevelopmentMetricObservation, dict[str, object], InternalValidationRecord | None]:
        responsibility = unit.responsibility_id
        if responsibility == "conditional_recovery_decision":
            assert raw.internal_runner_context is not None
            assert raw.internal_case_payload is not None
            payload_identity = raw.internal_case_payload.source_artifact.analysis_unit_identity
            if payload_identity != identity:
                raise DevelopmentRunnerError(
                    "joint payload analysis identity differs from frozen development unit"
                )
            internal_run = execute_internal_case(
                raw.internal_runner_context,
                unit_id=payload_identity.unit_id,
                payload=raw.internal_case_payload,
            )
            joint = internal_run.record
            assert raw.provisional_threshold is not None
            metric = metric_conditional_recovery_record(
                identity.source_cluster_id,
                joint,
                threshold_fit_source_cluster_digest=(
                    raw.provisional_threshold.fit_source_cluster_digest
                ),
            )
            operation = (
                raw.internal_case_payload.content_detector_binding
                .content_detection_operation
            )
            traces: dict[str, object] = {
                **asdict(joint.routing_trace),
                **asdict(joint.branch_score_trace),
                **asdict(joint.detector_trace),
                **asdict(joint.geometry_trace),
                **asdict(joint.threshold_trace),
                **asdict(joint.key_control_trace),
                "threshold_role": DEVELOPMENT_THRESHOLD_ROLE,
                "runtime_config_digest": joint.provenance_trace.execution_config_digest,
                "input_artifact_digest": joint.provenance_trace.input_artifact_digest,
                "attack_config_digest": joint.provenance_trace.attack_config_digest,
                "internal_record_id": joint.record_id,
                "formal_content_operation": type(operation).__qualname__,
            }
            return joint, metric, traces, joint
        key_identity = self.adapter.identify_key(raw.registered_root_key).result
        planned_key_binding = next(
            (
                item
                for item in self.intent_authority.public_key_roster
                if item.source_cluster_id == identity.source_cluster_id
            ),
            None,
        )
        if (
            planned_key_binding is None
            or planned_key_binding.registered_key_public_digest
            != key_identity.root_key_public_digest
            or planned_key_binding.detection_key_public_digest
            != key_identity.root_key_public_digest
        ):
            raise DevelopmentRunnerError(
                "registered detection key differs from frozen cluster authority"
            )
        wrong_material = derive_wrong_key_material(key_identity.root_key_public_digest, raw.wrong_key_index)
        shape = tuple(int(value) for value in raw.base_latent.shape)
        registered_domain = {
            "candidate_id": "hf_sparse_tail",
            "operator": "carrier_template",
            "responsibility_domain": "hf_carrier",
            "model_revision": self.runtime_adapter.session.model_revision,
            "tensor_role": "base_gaussian",
        }
        public_domain = {
            "candidate_id": "routing_stqr",
            "operator": "local_sensitivity_public_probe",
            "responsibility_domain": "public_noise",
            "model_revision": self.runtime_adapter.session.model_revision,
            "sample_index": unit.source_cluster_ordinal,
            "tensor_role": "latent_probe",
        }
        traces: dict[str, object] = {
            "registered_key_public_digest": key_identity.root_key_public_digest,
            "detection_key_public_digest": key_identity.root_key_public_digest,
            "key_role": "registered",
            "control_identity": "registered_key_control",
            "function_id": raw.combination_function_id,
            "mixing_coefficient": raw.mixing_coefficient,
        }
        if responsibility == "key_schedule":
            replayed = self.adapter.identify_key(raw.registered_root_key).result
            registered = self.adapter.derive_registered_key_stream(raw.registered_root_key, registered_domain, shape).result
            wrong = self.adapter.derive_wrong_key_stream(key_identity.root_key_public_digest, raw.wrong_key_index, registered_domain, shape).result
            public = self.adapter.derive_public_noise(public_domain, shape).result
            result = {"identity": key_identity, "registered": registered, "wrong": wrong, "public_noise": public}
            metric = metric_key_schedule(
                identity.source_cluster_id,
                registered_identity_digest=key_identity.root_key_public_digest,
                replayed_identity_digest=replayed.root_key_public_digest,
                registered_stream_digest=registered.values_float32_be_sha256,
                wrong_stream_digest=wrong.values_float32_be_sha256,
                public_noise_digest=public.values_float32_be_sha256,
            )
            return result, metric, traces, None
        adaptive_required = (
            responsibility == "content_router"
            or unit.content_branch_id == "lf_hf_routed_combination"
        )
        if adaptive_required and type(raw.routing_observations) is not RoutingObservations:
            raise DevelopmentRunnerError(
                "adaptive routing requires real registered observations"
            )
        adaptive = (
            self.adapter.route_content(
                shape,
                mode="routing_stqr",
                observations=raw.routing_observations,
            ).result
            if adaptive_required
            else None
        )
        uniform = self.adapter.route_content(shape, mode="routing_uniform_control").result
        if responsibility == "content_router":
            assert adaptive is not None
            content_operation = FormalHfContentDetectionOperation(self.adapter)

            def execute_routing_arm(route):
                arm_lf = self.adapter.build_lf_carrier(
                    raw.registered_root_key, shape, routing_result=route
                ).result
                arm_hf = self.adapter.build_hf_carrier(
                    raw.registered_root_key, shape, routing_result=route
                ).result
                captured: list[object] = []

                def arm_embedding(latent_values: tuple[float, ...]):
                    observation = self.adapter.embed_content(
                        latent_values,
                        arm_hf,
                        lf_carrier_result=arm_lf,
                        mixing_coefficient=raw.mixing_coefficient,
                        routing_result=route,
                    ).result
                    captured.append(observation)
                    return observation

                arm_runtime = self.runtime_adapter.execute_content_write_and_vae(
                    raw.base_latent, arm_embedding
                )
                if len(captured) != 1:
                    raise DevelopmentRunnerError(
                        "routing arm did not invoke content embedder exactly once"
                    )
                public_image = _decoded_image_to_rgb8(
                    arm_runtime.watermarked_image
                )
                content_result = content_operation(
                    public_image,
                    raw.registered_root_key,
                )
                quality = _tensor_relative_l2(
                    arm_runtime.clean_image, arm_runtime.watermarked_image
                )
                runtime_digest = _canonical_digest(
                    {
                        "content_materialization_identity": (
                            arm_runtime.content_materialization.materialization_replay_identity
                        ),
                        "paired_base_latent_digest": arm_runtime.paired_base_latent_digest,
                        "runtime_config_digest": arm_runtime.runtime_config_digest,
                    }
                )
                return arm_runtime, content_result, quality, runtime_digest

            adaptive_runtime, adaptive_content, adaptive_quality, adaptive_runtime_digest = (
                execute_routing_arm(adaptive)
            )
            uniform_runtime, uniform_content, uniform_quality, uniform_runtime_digest = (
                execute_routing_arm(uniform)
            )
            if (
                adaptive_content.detector_identity
                != uniform_content.detector_identity
                or adaptive_content.content_config_digest
                != uniform_content.content_config_digest
            ):
                raise DevelopmentRunnerError(
                    "routing arms did not reuse one content detector operation"
                )
            comparison_eligible = unit.content_branch_id != "clean_control"
            traces: dict[str, object] = {
                "registered_key_public_digest": key_identity.root_key_public_digest,
                "detection_key_public_digest": key_identity.root_key_public_digest,
                "key_role": "registered",
                "control_identity": "registered_key_control",
                "routing_identity": adaptive.route_identity,
                "routing_control": uniform.route_identity,
                "routing_observation_digest": _canonical_digest(
                    adaptive.observation_digests
                ),
                "routing_mask_digest": _canonical_digest(
                    (adaptive.mask_lf_digest, adaptive.mask_hf_digest)
                ),
                "routing_comparison_eligible": comparison_eligible,
                "adaptive_registered_score": adaptive_content.content_score,
                "uniform_control_registered_score": uniform_content.content_score,
                "adaptive_quality_delta": adaptive_quality,
                "uniform_control_quality_delta": uniform_quality,
                "adaptive_detector_identity": adaptive_content.detector_identity,
                "uniform_control_detector_identity": uniform_content.detector_identity,
                "adaptive_detector_config_digest": adaptive_content.content_config_digest,
                "uniform_control_detector_config_digest": uniform_content.content_config_digest,
                "runtime_config_digest": adaptive_runtime.runtime_config_digest,
                "input_artifact_digest": adaptive_runtime.paired_base_latent_digest,
                "routing_score_role": "hf_only_public_content_operation",
                "function_id": None,
                "mixing_coefficient": None,
            }
            metric = metric_content_router(
                identity.source_cluster_id,
                matched_budget_registered_score=(
                    adaptive_content.content_score if comparison_eligible else 0.0
                ),
                uniform_control_registered_score=(
                    uniform_content.content_score if comparison_eligible else 0.0
                ),
                routing_coverage=(adaptive.mean_mask_lf if comparison_eligible else 0.0),
                matched_budget_quality_delta=(
                    max(adaptive_quality, uniform_quality)
                    if comparison_eligible
                    else 0.0
                ),
                adaptive_route_identity=adaptive.route_identity,
                uniform_route_identity=uniform.route_identity,
                adaptive_detector_config_digest=adaptive_content.content_config_digest,
                uniform_detector_config_digest=uniform_content.content_config_digest,
                adaptive_runtime_result_digest=adaptive_runtime_digest,
                uniform_runtime_result_digest=uniform_runtime_digest,
            )
            return adaptive, metric, traces, None
        routing = adaptive if unit.content_branch_id == "lf_hf_routed_combination" else uniform
        assert routing is not None
        traces.update(
            routing_identity=routing.route_identity,
            routing_control=routing.mode,
            routing_observation_digest=_canonical_digest(routing.observation_digests),
            routing_mask_digest=_canonical_digest((routing.mask_lf_digest, routing.mask_hf_digest)),
        )
        responsibility_result: object | None = None
        replayed_carrier: object | None = None
        wrong_key_carrier: object | None = None
        lf_carrier = self.adapter.build_lf_carrier(raw.registered_root_key, shape, routing_result=routing).result
        hf_carrier = self.adapter.build_hf_carrier(raw.registered_root_key, shape, routing_result=routing).result
        if responsibility == "lf_carrier":
            responsibility_result = lf_carrier
            replayed_carrier = self.adapter.build_lf_carrier(
                raw.registered_root_key,
                shape,
                routing_result=routing,
            ).result
            wrong_key_carrier = self.adapter.build_lf_carrier(
                wrong_material,
                shape,
                routing_result=routing,
            ).result
        elif responsibility == "hf_carrier":
            responsibility_result = hf_carrier
            replayed_carrier = self.adapter.build_hf_carrier(
                raw.registered_root_key,
                shape,
                routing_result=routing,
            ).result
            wrong_key_carrier = self.adapter.build_hf_carrier(
                wrong_material,
                shape,
                routing_result=routing,
            ).result
        if responsibility == "content_embedder" and unit.content_branch_id == "clean_control":
            latent_payload = _public_payload(raw.base_latent)
            assert type(latent_payload) is dict
            base_digest = str(latent_payload["tensor_values_sha256"])
            clean_result = DevelopmentCleanControlResult(
                control_identity="development_clean_no_write_control",
                base_latent_digest=base_digest,
            )
            traces.update(
                content_write_control="no_write",
                realized_relative_l2=0.0,
                embedding_result_identity=None,
                materialization_replay_identity=None,
                input_artifact_digest=base_digest,
                runtime_config_digest=self.runtime_adapter.session.runtime_config_digest,
            )
            return clean_result, metric_content_embedder(
                identity.source_cluster_id,
                nominal_relative_l2=0.0,
                realized_relative_l2=0.0,
                clean_watermarked_image_relative_l2=0.0,
                embedding_result_identity=base_digest,
                materialization_replay_identity=base_digest,
                paired_base_latent_digest=base_digest,
            ), traces, None
        captured_embedding: list[object] = []

        def embedding_operation(latent_values: tuple[float, ...]):
            branch = unit.content_branch_id
            if branch in {"clean_control", "hf_only"}:
                observation = self.adapter.embed_content(latent_values, hf_carrier).result
            elif branch == "lf_only":
                observation = self.adapter.embed_content(latent_values, None, lf_carrier_result=lf_carrier).result
            else:
                observation = self.adapter.embed_content(
                    latent_values,
                    hf_carrier,
                    lf_carrier_result=lf_carrier,
                    mixing_coefficient=raw.mixing_coefficient,
                    routing_result=routing,
                ).result
            captured_embedding.append(observation)
            return observation

        runtime_result = self.runtime_adapter.execute_content_write_and_vae(raw.base_latent, embedding_operation)
        if len(captured_embedding) != 1:
            raise DevelopmentRunnerError("runtime did not invoke content embedder exactly once")
        embedding = captured_embedding[0]
        traces.update(
            runtime_config_digest=runtime_result.runtime_config_digest,
            input_artifact_digest=runtime_result.paired_base_latent_digest,
        )
        if responsibility == "content_embedder":
            materialization = runtime_result.content_materialization
            return runtime_result, metric_content_embedder(
                identity.source_cluster_id,
                nominal_relative_l2=embedding.target_relative_l2,
                realized_relative_l2=materialization.realized_relative_l2,
                clean_watermarked_image_relative_l2=_tensor_relative_l2(
                    runtime_result.clean_image, runtime_result.watermarked_image
                ),
                embedding_result_identity=embedding.embedding_result_identity,
                materialization_replay_identity=materialization.materialization_replay_identity,
                paired_base_latent_digest=runtime_result.paired_base_latent_digest,
            ), traces, None
        if responsibility in {"lf_carrier", "hf_carrier"}:
            assert responsibility_result is not None
            assert replayed_carrier is not None
            assert wrong_key_carrier is not None
            materialization = runtime_result.content_materialization
            materialization_result = runtime_result.content_materialization_result
            if responsibility == "lf_carrier":
                assert type(responsibility_result) is LfCarrierResult
                assert type(replayed_carrier) is LfCarrierResult
                assert type(wrong_key_carrier) is LfCarrierResult
                embedded_registered_direction = (
                    embedding.active_lf_direction == responsibility_result.direction
                    and embedding.active_hf_direction is None
                    and embedding.content_direction == responsibility_result.direction
                    and embedding.lf_carrier_config_digest
                    == responsibility_result.carrier_config_digest
                    and embedding.hf_carrier_config_digest is None
                )
            else:
                assert type(responsibility_result) is HfCarrierResult
                assert type(replayed_carrier) is HfCarrierResult
                assert type(wrong_key_carrier) is HfCarrierResult
                embedded_registered_direction = (
                    embedding.active_hf_direction == responsibility_result.direction
                    and embedding.active_lf_direction is None
                    and embedding.content_direction == responsibility_result.direction
                    and embedding.hf_carrier_config_digest
                    == responsibility_result.carrier_config_digest
                    and embedding.lf_carrier_config_digest is None
                )
            replay_matches = replayed_carrier == responsibility_result
            wrong_key_separated = (
                wrong_key_carrier.key_role == "wrong"
                and wrong_key_carrier.wrong_key_index == raw.wrong_key_index
                and wrong_key_carrier.root_key_public_digest
                == responsibility_result.root_key_public_digest
                and wrong_key_carrier.carrier_config_digest
                == responsibility_result.carrier_config_digest
                and wrong_key_carrier.direction_digest
                != responsibility_result.direction_digest
                and wrong_key_carrier.direction != responsibility_result.direction
            )
            materialized_within_budget = (
                embedded_registered_direction
                and materialization_result.embedding_result == embedding
                and materialization.integrity_status == "passed"
                and materialization_result.integrity_status == "passed"
                and materialization_result.budget_status == "accepted"
                and materialization.realized_total_l2 > 0.0
                and materialization_result.realized_relative_l2 > 0.0
                and materialization_result.realized_relative_l2
                <= materialization_result.content_relative_l2_limit
                and materialization.materialization_replay_identity
                == materialization_result.observation.materialization_replay_identity
            )
            traces.update(
                content_write_control="registered_carrier_write_with_matched_clean_pair",
                embedding_result_identity=embedding.embedding_result_identity,
                materialization_replay_identity=(
                    materialization.materialization_replay_identity
                ),
                realized_relative_l2=materialization_result.realized_relative_l2,
            )
            metric_builder = (
                metric_lf_carrier
                if responsibility == "lf_carrier"
                else metric_hf_carrier
            )
            return responsibility_result, metric_builder(
                identity.source_cluster_id,
                registered_direction_replay_match=replay_matches,
                wrong_key_direction_separated=wrong_key_separated,
                materialized_nonzero_write_within_budget=(
                    materialized_within_budget
                ),
                quality_delta=_tensor_relative_l2(
                    runtime_result.clean_image,
                    runtime_result.watermarked_image,
                ),
                registered_direction_digest=responsibility_result.direction_digest,
                replayed_direction_digest=replayed_carrier.direction_digest,
                wrong_key_direction_digest=wrong_key_carrier.direction_digest,
                carrier_config_digest=responsibility_result.carrier_config_digest,
                embedding_result_identity=embedding.embedding_result_identity,
                materialization_replay_identity=(
                    materialization.materialization_replay_identity
                ),
                materialized_delta_digest=(
                    materialization.delta_content_actual_digest
                ),
            ), traces, None
        if responsibility in {
            "qk_geometry_sync",
            "geometric_transform_estimator",
            "geometry_reliability",
            "image_rectifier",
        }:
            return self._execute_geometry_operation(
                unit,
                identity,
                raw,
                runtime_result=runtime_result,
                wrong_material=wrong_material,
                traces=traces,
            )
        observed_latent = (
            runtime_result.clean_detection_latent
            if unit.content_branch_id == "clean_control"
            else runtime_result.watermarked_detection_latent
        )
        registered_hf_observation = HfDetectionObservation.from_public_image_encoding(
            tuple(observed_latent.to(dtype=torch.float32).reshape(-1).tolist()), tuple(observed_latent.shape)
        )
        registered_lf_observation = LfDetectionObservation.from_public_image_encoding(
            tuple(observed_latent.to(dtype=torch.float32).reshape(-1).tolist()), tuple(observed_latent.shape)
        )
        clean_hf_observation = HfDetectionObservation.from_public_image_encoding(
            tuple(runtime_result.clean_detection_latent.to(dtype=torch.float32).reshape(-1).tolist()),
            tuple(runtime_result.clean_detection_latent.shape),
        )
        clean_lf_observation = LfDetectionObservation.from_public_image_encoding(
            tuple(runtime_result.clean_detection_latent.to(dtype=torch.float32).reshape(-1).tolist()),
            tuple(runtime_result.clean_detection_latent.shape),
        )
        hf_registered = self.adapter.detect_hf(registered_hf_observation, raw.registered_root_key).result
        hf_wrong = self.adapter.detect_hf(registered_hf_observation, wrong_material).result
        hf_null = self.adapter.detect_hf(
            clean_hf_observation, raw.registered_root_key
        ).result
        lf_registered = self.adapter.detect_lf(registered_lf_observation, raw.registered_root_key).result
        lf_wrong = self.adapter.detect_lf(registered_lf_observation, wrong_material).result
        lf_null = self.adapter.detect_lf(
            clean_lf_observation, raw.registered_root_key
        ).result
        traces.update(
            hf_score=hf_registered.hf_score,
            lf_score=lf_registered.lf_score,
            raw_detector_identity=hf_registered.detector_identity,
            rectified_detector_identity=hf_registered.detector_identity,
            raw_detector_config_digest=hf_registered.detector_config_digest,
            rectified_detector_config_digest=hf_registered.detector_config_digest,
            raw_preprocessing_identity="final_image_vae_posterior_mode",
            rectified_preprocessing_identity="final_image_vae_posterior_mode",
            registered_detector_identity=hf_registered.detector_identity,
            wrong_key_detector_identity=hf_wrong.detector_identity,
            primary_null_detector_identity=hf_null.detector_identity,
            registered_detector_config_digest=hf_registered.detector_config_digest,
            wrong_key_detector_config_digest=hf_wrong.detector_config_digest,
            primary_null_detector_config_digest=hf_null.detector_config_digest,
            primary_null_preprocessing_identity="final_image_vae_posterior_mode",
            primary_null_detection_key_public_digest=(
                key_identity.root_key_public_digest
            ),
            primary_null_control_identity="unwatermarked_registered_key_primary_null",
        )
        if responsibility == "lf_detector":
            traces.update(
                raw_detector_identity=lf_registered.detector_identity,
                rectified_detector_identity=lf_registered.detector_identity,
                raw_detector_config_digest=lf_registered.detector_config_digest,
                rectified_detector_config_digest=lf_registered.detector_config_digest,
                registered_detector_identity=lf_registered.detector_identity,
                wrong_key_detector_identity=lf_wrong.detector_identity,
                primary_null_detector_identity=lf_null.detector_identity,
                registered_detector_config_digest=lf_registered.detector_config_digest,
                wrong_key_detector_config_digest=lf_wrong.detector_config_digest,
                primary_null_detector_config_digest=lf_null.detector_config_digest,
            )
            metric = metric_lf_detector(
                identity.source_cluster_id,
                registered_score=lf_registered.lf_score,
                wrong_score=lf_wrong.lf_score,
                primary_null_score=lf_null.lf_score,
                detector_config_digest=lf_registered.detector_config_digest,
                registered_observation_digest=lf_registered.observation_digest,
                wrong_observation_digest=lf_wrong.observation_digest,
                primary_null_observation_digest=lf_null.observation_digest,
            )
            return lf_registered, metric, traces, None
        if responsibility == "hf_detector":
            content_operation = FormalHfContentDetectionOperation(self.adapter)
            registered_image = _decoded_image_to_rgb8(
                runtime_result.clean_image
                if unit.content_branch_id == "clean_control"
                else runtime_result.watermarked_image
            )
            clean_image = _decoded_image_to_rgb8(runtime_result.clean_image)
            registered_content = content_operation(
                registered_image,
                raw.registered_root_key,
            )
            wrong_content = content_operation(registered_image, wrong_material)
            primary_null_content = content_operation(
                clean_image,
                raw.registered_root_key,
            )
            hf_registered = registered_content.hf_result
            hf_wrong = wrong_content.hf_result
            hf_null = primary_null_content.hf_result
            traces.update(
                hf_score=hf_registered.hf_score,
                raw_detector_identity=hf_registered.detector_identity,
                rectified_detector_identity=hf_registered.detector_identity,
                raw_detector_config_digest=hf_registered.detector_config_digest,
                rectified_detector_config_digest=hf_registered.detector_config_digest,
                raw_preprocessing_identity=content_operation.preprocessing_identity,
                rectified_preprocessing_identity=content_operation.preprocessing_identity,
                registered_detector_identity=hf_registered.detector_identity,
                wrong_key_detector_identity=hf_wrong.detector_identity,
                primary_null_detector_identity=hf_null.detector_identity,
                registered_detector_config_digest=hf_registered.detector_config_digest,
                wrong_key_detector_config_digest=hf_wrong.detector_config_digest,
                primary_null_detector_config_digest=hf_null.detector_config_digest,
                primary_null_preprocessing_identity=(
                    content_operation.preprocessing_identity
                ),
                primary_null_detection_key_public_digest=(
                    key_identity.root_key_public_digest
                ),
                primary_null_control_identity=(
                    "unwatermarked_registered_key_primary_null"
                ),
            )
            metric = metric_hf_detector(
                identity.source_cluster_id,
                registered_score=hf_registered.hf_score,
                wrong_score=hf_wrong.hf_score,
                primary_null_score=hf_null.hf_score,
                detector_config_digest=hf_registered.detector_config_digest,
                registered_observation_digest=hf_registered.observation_digest,
                wrong_observation_digest=hf_wrong.observation_digest,
                primary_null_observation_digest=hf_null.observation_digest,
            )
            return hf_registered, metric, traces, None
        candidate_content = self.adapter.detect_content(
            hf_registered,
            lf_registered,
            hf_null=raw.hf_null,
            lf_null=raw.lf_null,
            combination=raw.combination_function_id,
            weight=(raw.mixing_coefficient if raw.combination_function_id == "weighted_hf_lf_standardized_score" else None),
        ).result
        if responsibility == "content_detector":
            assert raw.hf_null is not None and raw.lf_null is not None
            traces.update(
                hf_null_partition_identity=raw.hf_null.partition_identity,
                lf_null_partition_identity=raw.lf_null.partition_identity,
                hf_null_fit_source_digest=_canonical_digest(
                    tuple(asdict(item) for item in raw.hf_null.records)
                ),
                lf_null_fit_source_digest=_canonical_digest(
                    tuple(asdict(item) for item in raw.lf_null.records)
                ),
            )
        hf_only_content = self.adapter.detect_content(hf_registered).result
        wrong_content = self.adapter.detect_content(
            hf_wrong,
            lf_wrong,
            hf_null=raw.hf_null,
            lf_null=raw.lf_null,
            combination=raw.combination_function_id,
            weight=(raw.mixing_coefficient if raw.combination_function_id == "weighted_hf_lf_standardized_score" else None),
        ).result
        null_content = self.adapter.detect_content(
            hf_null,
            lf_null,
            hf_null=raw.hf_null,
            lf_null=raw.lf_null,
            combination=raw.combination_function_id,
            weight=(raw.mixing_coefficient if raw.combination_function_id == "weighted_hf_lf_standardized_score" else None),
        ).result
        traces.update(
            combined_score=candidate_content.combined_score,
            raw_content_score=candidate_content.content_score,
            registered_detector_identity=candidate_content.detector_identity,
            wrong_key_detector_identity=wrong_content.detector_identity,
            primary_null_detector_identity=null_content.detector_identity,
            registered_detector_config_digest=candidate_content.content_config_digest,
            wrong_key_detector_config_digest=wrong_content.content_config_digest,
            primary_null_detector_config_digest=null_content.content_config_digest,
        )
        if responsibility == "content_detector":
            metric = metric_content_detector(
                identity.source_cluster_id,
                candidate_score=(candidate_content.combined_score if candidate_content.combined_score is not None else candidate_content.content_score),
                hf_only_score=hf_only_content.content_score,
                wrong_key_score=(wrong_content.combined_score if wrong_content.combined_score is not None else wrong_content.content_score),
                primary_null_score=(null_content.combined_score if null_content.combined_score is not None else null_content.content_score),
                low_frequency_score=candidate_content.lf_score,
                candidate_config_digest=candidate_content.content_config_digest,
                hf_only_config_digest=hf_only_content.content_config_digest,
                wrong_key_config_digest=wrong_content.content_config_digest,
                primary_null_config_digest=null_content.content_config_digest,
            )
            return candidate_content, metric, traces, None
        raise DevelopmentRunnerError("responsibility dispatch is incomplete")

    def _record(
        self,
        unit: DevelopmentStudyUnit,
        identity: AnalysisUnitIdentity,
        result: object,
        metric: DevelopmentMetricObservation,
        traces: Mapping[str, object],
        *,
        internal_record: InternalValidationRecord | None,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        actual_elapsed_seconds: float,
    ) -> DevelopmentScientificRecord:
        study = next(item for item in self.protocol.module_matrix if item.responsibility_id == unit.responsibility_id)
        metric = bind_development_metric_observation(
            metric,
            registered_metric_ids=study.metric_ids,
            candidate_config_digest=study.candidate_config_digest,
            paired_ablation_identity=study.paired_ablation_identity,
            content_branch_id=unit.content_branch_id,
            geometry_case_id=unit.geometry_case_id,
        )
        result_payload = _safe_result_payload(unit.responsibility_id, result)
        metric.validate()
        routing_trace = {
            key: traces.get(key)
            for key in (
                "routing_identity", "routing_control", "routing_observation_digest", "routing_mask_digest",
                "routing_comparison_eligible", "adaptive_registered_score",
                "uniform_control_registered_score", "adaptive_quality_delta",
                "uniform_control_quality_delta", "adaptive_detector_identity",
                "uniform_control_detector_identity", "adaptive_detector_config_digest",
                "uniform_control_detector_config_digest", "routing_score_role",
            )
        }
        branch_score_trace = {
            "lf_score": traces.get("lf_score"),
            "hf_score": traces.get("hf_score"),
            "combined_score": traces.get("combined_score"),
            "function_id": traces.get("function_id"),
            "mixing_coefficient": traces.get("mixing_coefficient"),
        }
        detector_trace = {
            key: traces.get(key)
            for key in (
                "raw_detector_identity", "rectified_detector_identity",
                "raw_detector_config_digest", "rectified_detector_config_digest",
                "raw_preprocessing_identity", "rectified_preprocessing_identity",
                "raw_content_score", "rectified_content_score",
                "registered_detector_identity", "wrong_key_detector_identity",
                "primary_null_detector_identity", "registered_detector_config_digest",
                "wrong_key_detector_config_digest", "primary_null_detector_config_digest",
                "primary_null_preprocessing_identity",
            )
        }
        geometry_trace = {
            key: traces.get(key)
            for key in (
                "geometry_operation_identity", "geometry_reliability_config_digest",
                "geometry_estimation_identity", "geometry_reliability_identity",
                "geometry_reliable", "geometry_transform", "geometry_raw_metrics",
                "rectification_status", "qk_registered_relation_score", "qk_wrong_relation_score",
                "geometry_reliability_status", "geometry_reliability_failure_reasons",
                "wrong_key_geometry_estimation_identity",
                "wrong_key_geometry_reliability_identity",
                "wrong_key_geometry_reliable",
                "wrong_key_geometry_reliability_status",
                "wrong_key_geometry_reliability_failure_reasons",
                "ambiguous_control_realized",
            )
        }
        threshold_trace = {
            key: traces.get(key)
            for key in (
                "raw_threshold_identity", "rectified_threshold_identity", "tau", "tau_rescue",
                "threshold_role",
                "hf_null_partition_identity", "lf_null_partition_identity",
                "hf_null_fit_source_digest", "lf_null_fit_source_digest",
            )
        }
        key_control_trace = {
            key: traces.get(key)
            for key in (
                "registered_key_public_digest", "detection_key_public_digest", "key_role", "control_identity"
                , "primary_null_detection_key_public_digest",
                "primary_null_control_identity"
            )
        }
        decision_trace = {
            "watermark_decision": (
                internal_record.decision_trace.watermark_decision if internal_record is not None else None
            ),
            "positive_source": (
                internal_record.decision_trace.positive_source if internal_record is not None else None
            ),
            "decision_reason": (
                internal_record.decision_trace.decision_reason if internal_record is not None else "module_observation_not_a_watermark_decision"
            ),
            "internal_validation_record_id": (
                internal_record.record_id if internal_record is not None else None
            ),
        }
        provenance = {
            "protocol_digest": self.protocol.digest(),
            "input_manifest_digest": self.intent_authority.input_manifest_digest,
            "execution_intent_authority_digest": self.intent_authority.authority_digest,
            "method_code_revision": self.method_code_revision,
            "candidate_config_digest": study.candidate_config_digest,
            "method_config_digest": self.adapter.configuration.config_digest,
            "runtime_config_digest": traces.get("runtime_config_digest"),
            "environment_digest": self.environment_digest,
            "resource_identity_digest": self.resource_identity_digest,
            "input_artifact_digest": traces.get("input_artifact_digest"),
            "attack_config_digest": traces.get("attack_config_digest"),
            "metric_observation_digest": metric.observation_digest,
        }
        record = DevelopmentScientificRecord(
            schema_version=DEVELOPMENT_RECORD_SCHEMA,
            collection_role=DEVELOPMENT_RECORD_COLLECTION_ROLE,
            record_id="0" * 64,
            run_id=self.intent_authority.run_id,
            protocol_id=self.protocol.protocol_id,
            protocol_version=self.protocol.protocol_version,
            protocol_digest=self.protocol.digest(),
            execution_intent_authority_digest=self.intent_authority.authority_digest,
            method_code_revision=self.method_code_revision,
            unit_index=unit.unit_index,
            phase=unit.phase,
            analysis_unit_identity=asdict(identity),
            responsibility_id=unit.responsibility_id,
            scientific_question_id=study.scientific_question_id,
            development_case_id=study.development_case_id,
            candidate_identity=study.candidate_identity,
            candidate_config_digest=study.candidate_config_digest,
            paired_ablation_identity=study.paired_ablation_identity,
            negative_control_case_ids=study.negative_control_case_ids,
            metric_ids=study.metric_ids,
            content_branch_id=unit.content_branch_id,
            geometry_case_id=unit.geometry_case_id,
            attempt_index=attempt_index,
            execution_status="success",
            failure_class=None,
            failure_reason=None,
            retry_parent_intent_digest=retry_parent_intent_digest,
            actual_elapsed_seconds=actual_elapsed_seconds,
            maximum_duration_seconds=unit.maximum_duration_seconds,
            duration_limit_exceeded=False,
            operation_result_payload=result_payload,
            operation_result_digest=_canonical_digest(result_payload),
            metric_observation=asdict(metric),
            routing_trace=routing_trace,
            branch_score_trace=branch_score_trace,
            detector_trace=detector_trace,
            geometry_trace=geometry_trace,
            threshold_trace=threshold_trace,
            key_control_trace=key_control_trace,
            decision_trace=decision_trace,
            provenance_trace=provenance,
            module_outcome=None,
            candidate_recommendation=None,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = replace(
            record,
            record_id=_canonical_digest(record.payload_without_record_id()),
        )
        self._validate_record(record, study)
        return record

    def _validate_record(self, record: DevelopmentScientificRecord, study) -> None:
        try:
            record.validate()
        except DevelopmentRecordError as exc:
            raise DevelopmentRunnerError(str(exc)) from exc
        if record.schema_version != DEVELOPMENT_RECORD_SCHEMA:
            raise DevelopmentRunnerError("development record schema drifted")
        if record.module_outcome is not None or record.candidate_recommendation is not None:
            raise DevelopmentRunnerError("per-unit record cannot preempt cluster outcome")
        if record.candidate_config_digest != study.candidate_config_digest:
            raise DevelopmentRunnerError("record candidate digest drifted")
        if record.metric_ids != study.metric_ids:
            raise DevelopmentRunnerError("record metric registry drifted")
        if record.execution_status == "success":
            metric = DevelopmentMetricObservation(**record.metric_observation)
            metric.validate()
            if metric.registered_metric_ids != study.metric_ids:
                raise DevelopmentRunnerError("record metric observation is not bound to registered ids")
            if (
                metric.candidate_config_digest != study.candidate_config_digest
                or metric.paired_ablation_identity != study.paired_ablation_identity
                or metric.content_branch_id != record.content_branch_id
                or metric.geometry_case_id != record.geometry_case_id
            ):
                raise DevelopmentRunnerError("record metric paired case binding drifted")
        if record.operation_result_digest != _canonical_digest(record.operation_result_payload):
            raise DevelopmentRunnerError("operation result digest drifted")
        if record.execution_status == "success":
            if record.threshold_trace["raw_threshold_identity"] != record.threshold_trace["rectified_threshold_identity"]:
                raise DevelopmentRunnerError("raw and rectified threshold identities differ")
            detector = record.detector_trace
            for left, right in (
                ("raw_detector_identity", "rectified_detector_identity"),
                ("raw_detector_config_digest", "rectified_detector_config_digest"),
                ("raw_preprocessing_identity", "rectified_preprocessing_identity"),
            ):
                if detector[left] != detector[right]:
                    raise DevelopmentRunnerError("raw and rectified detector semantics differ")
        if record.decision_trace["positive_source"] not in {None, "raw_content", "rectified_content"}:
            raise DevelopmentRunnerError("geometry cannot be a positive source")

    def _failure_record(
        self,
        unit: DevelopmentStudyUnit,
        identity: AnalysisUnitIdentity,
        *,
        attempt_index: int,
        retry_parent_intent_digest: str | None,
        execution_status: str,
        failure_class: str,
        failure_reason: str,
        actual_elapsed_seconds: float,
    ) -> DevelopmentScientificRecord:
        if execution_status not in {"failed", "excluded", "retry"}:
            raise DevelopmentRunnerError("failure record status is invalid")
        study = next(
            item for item in self.protocol.module_matrix
            if item.responsibility_id == unit.responsibility_id
        )
        empty_result: dict[str, object] = {}
        record = DevelopmentScientificRecord(
            schema_version=DEVELOPMENT_RECORD_SCHEMA,
            collection_role=DEVELOPMENT_RECORD_COLLECTION_ROLE,
            record_id="0" * 64,
            run_id=self.intent_authority.run_id,
            protocol_id=self.protocol.protocol_id,
            protocol_version=self.protocol.protocol_version,
            protocol_digest=self.protocol.digest(),
            execution_intent_authority_digest=self.intent_authority.authority_digest,
            method_code_revision=self.method_code_revision,
            unit_index=unit.unit_index,
            phase=unit.phase,
            analysis_unit_identity=asdict(identity),
            responsibility_id=unit.responsibility_id,
            scientific_question_id=study.scientific_question_id,
            development_case_id=study.development_case_id,
            candidate_identity=study.candidate_identity,
            candidate_config_digest=study.candidate_config_digest,
            paired_ablation_identity=study.paired_ablation_identity,
            negative_control_case_ids=study.negative_control_case_ids,
            metric_ids=study.metric_ids,
            content_branch_id=unit.content_branch_id,
            geometry_case_id=unit.geometry_case_id,
            attempt_index=attempt_index,
            execution_status=execution_status,
            failure_class=failure_class,
            failure_reason=failure_reason,
            retry_parent_intent_digest=retry_parent_intent_digest,
            actual_elapsed_seconds=actual_elapsed_seconds,
            maximum_duration_seconds=unit.maximum_duration_seconds,
            duration_limit_exceeded=(
                actual_elapsed_seconds > unit.maximum_duration_seconds
            ),
            operation_result_payload=empty_result,
            operation_result_digest=_canonical_digest(empty_result),
            metric_observation={},
            routing_trace={},
            branch_score_trace={},
            detector_trace={},
            geometry_trace={},
            threshold_trace={},
            key_control_trace={},
            decision_trace={
                "watermark_decision": None,
                "positive_source": None,
                "decision_reason": "unit_execution_did_not_produce_scientific_metric",
                "internal_validation_record_id": None,
            },
            provenance_trace={
                "protocol_digest": self.protocol.digest(),
                "input_manifest_digest": self.intent_authority.input_manifest_digest,
                "execution_intent_authority_digest": self.intent_authority.authority_digest,
                "method_code_revision": self.method_code_revision,
                "candidate_config_digest": study.candidate_config_digest,
                "method_config_digest": self.adapter.configuration.config_digest,
                "environment_digest": self.environment_digest,
                "resource_identity_digest": self.resource_identity_digest,
            },
            module_outcome=None,
            candidate_recommendation=None,
            scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
        )
        record = replace(
            record,
            record_id=_canonical_digest(record.payload_without_record_id()),
        )
        self._validate_record(record, study)
        return record

    def _unit(self, unit_index: int) -> DevelopmentStudyUnit:
        if type(unit_index) is not int or not 0 <= unit_index < len(self.protocol.unit_roster):
            raise DevelopmentRunnerError("unit index is outside frozen roster")
        unit = self.protocol.unit_roster[unit_index]
        if unit.unit_index != unit_index:
            raise DevelopmentRunnerError("unit roster index drifted")
        return unit

    def _cluster_identities(self) -> tuple[AnalysisUnitIdentity, ...]:
        by_cluster: dict[str, AnalysisUnitIdentity] = {}
        ordered_clusters: list[AnalysisUnitIdentity] = []
        for assignment in self.intent_authority.input_manifest.assignments:
            existing = by_cluster.get(assignment.identity.source_cluster_id)
            if existing is not None and existing != assignment.identity:
                raise DevelopmentRunnerError("manifest cluster identity is not unique")
            if existing is None:
                ordered_clusters.append(assignment.identity)
            by_cluster[assignment.identity.source_cluster_id] = assignment.identity
        clusters = tuple(ordered_clusters)
        required_cluster_count = 1 + max(
            unit.source_cluster_ordinal for unit in self.protocol.unit_roster
        )
        if len(clusters) != required_cluster_count:
            raise DevelopmentRunnerError(
                "development manifest differs from the protocol-frozen cluster roster"
            )
        return clusters

    def _analysis_identity(self, unit: DevelopmentStudyUnit) -> AnalysisUnitIdentity:
        # Cross-fit authorization is bound to the exact manifest identity.  The
        # atomic development case remains separate in the scientific record.
        return self._clusters[unit.source_cluster_ordinal]

    def _representative_unit(
        self,
        responsibility_id: str,
        source_cluster_ordinal: int,
        *,
        content_branch_id: str | None = None,
    ) -> DevelopmentStudyUnit:
        unit = next(
            (
                item
                for item in self.protocol.unit_roster
                if item.responsibility_id == responsibility_id
                and item.source_cluster_ordinal == source_cluster_ordinal
                and item.phase not in OPERATIONAL_UNIT_PHASES
                and (
                    content_branch_id is None
                    or item.content_branch_id == content_branch_id
                )
            ),
            None,
        )
        if unit is None:
            raise DevelopmentRunnerError("frozen roster lacks operational representative")
        return unit

    @staticmethod
    def _unit_id(unit: DevelopmentStudyUnit) -> str:
        return f"development_unit_{unit.unit_index:04d}"
