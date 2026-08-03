"""Real 13-responsibility development exploration runner.

The runner owns orchestration and formal development record construction.  It
accepts raw keys, tensors, registered protocol objects, and runtime adapters;
it never accepts precomputed module results or a result-provider callback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from math import isfinite, isnan
from pathlib import Path
from time import monotonic
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
    DevelopmentMetricObservation,
    aggregate_development_cluster_metrics,
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
    RECORD_SCHEMA_VERSION,
    DEVELOPMENT_SPLIT,
    MODULE_OUTCOMES,
    DevelopmentProvisionalThreshold,
    DevelopmentModuleOutcomeRecord,
    DevelopmentStudyUnit,
    FrozenDevelopmentCrossFitPlan,
    FrozenDevelopmentExecutionIntentAuthority,
    FrozenDevelopmentExplorationProtocol,
    authorize_development_provisional_threshold,
    create_development_module_outcome_record,
    decide_development_module_execution,
)
from experiments.protocol.internal_records import InternalValidationRecord
from experiments.protocol.internal_splits import AnalysisUnitIdentity
from experiments.runners.development_persistence import (
    CommittedUnit,
    DevelopmentPersistentStore,
    PersistentLease,
    UnitIntent,
    canonical_json_bytes,
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
    GeometryReliabilityThresholds,
    HfDetectionObservation,
    JointDecisionThresholds,
    LfDetectionObservation,
    RoutingObservations,
    derive_wrong_key_material,
)
from runtime import RuntimeAdapterState, Sd35RuntimeAdapter
from runtime.content_write import ContentWriteVaeResult


DEVELOPMENT_RECORD_SCHEMA = RECORD_SCHEMA_VERSION
DEVELOPMENT_RECORD_COLLECTION_ROLE = "runner_only_formal_development_records"
THRESHOLD_BOUND_RESPONSIBILITIES = frozenset(
    {
        "conditional_recovery_decision",
    }
)


class DevelopmentRunnerError(RuntimeError):
    """Frozen unit, direct method call, metric, or record boundary failed."""


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


@dataclass(frozen=True, slots=True)
class DevelopmentUnitInput:
    """Raw execution inputs; precomputed method results are intentionally absent."""

    registered_root_key: str
    wrong_key_index: int
    base_latent: torch.Tensor
    routing_observations: RoutingObservations
    mixing_coefficient: float
    combination_function_id: str
    hf_null: BranchNullCalibration | None
    lf_null: BranchNullCalibration | None
    attack_specification: GeometricAttackSpec
    epsilon_inlier: float
    geometry_reliability_thresholds: GeometryReliabilityThresholds
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
        if type(self.routing_observations) is not RoutingObservations:
            raise DevelopmentRunnerError("routing observations exact type is required")
        if self.mixing_coefficient not in {0.25, 0.50, 0.75}:
            raise DevelopmentRunnerError("mixing coefficient is not registered")
        if self.combination_function_id not in CONTENT_COMBINATION_FUNCTION_IDS:
            raise DevelopmentRunnerError("combination function is not registered")
        if type(self.attack_specification) is not GeometricAttackSpec:
            raise DevelopmentRunnerError("attack specification exact type is required")
        if (
            isinstance(self.epsilon_inlier, bool)
            or not isinstance(self.epsilon_inlier, (int, float))
            or not isfinite(float(self.epsilon_inlier))
            or float(self.epsilon_inlier) <= 0.0
        ):
            raise DevelopmentRunnerError("epsilon_inlier is invalid")
        if type(self.geometry_reliability_thresholds) is not GeometryReliabilityThresholds:
            raise DevelopmentRunnerError("development reliability thresholds exact type required")
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
class DevelopmentScientificRecord:
    schema_version: str
    collection_role: str
    record_id: str
    run_id: str
    protocol_id: str
    protocol_version: str
    protocol_digest: str
    execution_intent_authority_digest: str
    method_code_revision: str
    unit_index: int
    phase: str
    analysis_unit_identity: dict[str, object]
    responsibility_id: str
    scientific_question_id: str
    development_case_id: str
    candidate_identity: str
    candidate_config_digest: str
    paired_ablation_identity: str
    negative_control_case_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    content_branch_id: str
    geometry_case_id: str
    attempt_index: int
    execution_status: str
    failure_class: str | None
    failure_reason: str | None
    retry_parent_intent_digest: str | None
    operation_result_payload: dict[str, object]
    operation_result_digest: str
    metric_observation: dict[str, object]
    routing_trace: dict[str, object]
    branch_score_trace: dict[str, object]
    detector_trace: dict[str, object]
    geometry_trace: dict[str, object]
    threshold_trace: dict[str, object]
    key_control_trace: dict[str, object]
    decision_trace: dict[str, object]
    provenance_trace: dict[str, object]
    module_outcome: str | None
    candidate_recommendation: str | None
    scientific_claim_boundary: str

    def payload(self) -> dict[str, object]:
        return asdict(self)


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

    def execute_preflight_cluster(
        self,
        source_cluster_ordinal: int,
        unit_input: DevelopmentUnitInput,
    ) -> DevelopmentOperationalReceipt:
        """Measure one of the frozen one-to-two identity/throughput clusters."""

        if type(source_cluster_ordinal) is not int or not 0 <= source_cluster_ordinal < self.protocol.preflight.source_cluster_count:
            raise DevelopmentRunnerError("preflight cluster ordinal is outside frozen budget")
        unit = self._representative_unit("content_embedder", source_cluster_ordinal)
        unit_input.validate(unit.responsibility_id)
        identity = self._analysis_identity(unit)
        started = monotonic()
        result, _metric, _traces, _internal = self._execute_real_operation(
            unit, identity, unit_input
        )
        elapsed = monotonic() - started
        result_payload = _public_payload(result)
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
        digests: list[tuple[str, str]] = []
        started = monotonic()
        for responsibility in expected:
            unit = self._representative_unit(responsibility, source_cluster_ordinal)
            raw = unit_inputs[responsibility]
            raw.validate(responsibility)
            identity = self._analysis_identity(unit)
            if responsibility in THRESHOLD_BOUND_RESPONSIBILITIES:
                assert raw.provisional_threshold is not None
                assert raw.cross_fit_plan is not None
                authorize_development_provisional_threshold(
                    raw.provisional_threshold,
                    raw.cross_fit_plan,
                    expected_execution_intent_authority_digest=self.intent_authority.authority_digest,
                    requested_split=DEVELOPMENT_SPLIT,
                    requested_analysis_unit_identity=identity,
                )
            result, _metric, _traces, _internal = self._execute_real_operation(
                unit, identity, raw
            )
            digests.append((responsibility, _canonical_digest(_public_payload(result))))
        return DevelopmentOperationalReceipt(
            operational_role="full_chain_wiring_smoke",
            source_cluster_ordinal=source_cluster_ordinal,
            case_ids=("all_thirteen_responsibility_wiring",),
            responsibility_result_digests=tuple(digests),
            elapsed_seconds=monotonic() - started,
            runtime_config_digest=self.runtime_adapter.session.runtime_config_digest,
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
        )

    def execute_unit(
        self,
        unit_index: int,
        unit_input: DevelopmentUnitInput,
        *,
        prerequisite_outcomes: Mapping[str, str],
        attempt_index: int = 0,
        retry_parent_intent_digest: str | None = None,
    ) -> DevelopmentUnitRunResult:
        unit = self._unit(unit_index)
        unit_input.validate(unit.responsibility_id)
        if type(attempt_index) is not int or not 0 <= attempt_index < unit.maximum_record_attempts:
            raise DevelopmentRunnerError("record attempt exceeds frozen limit")
        decision = decide_development_module_execution(
            self.protocol,
            unit.responsibility_id,
            prerequisite_outcomes,
        )
        if not decision.approved:
            raise DevelopmentRunnerError(
                f"module dependency stop:{decision.decision_reason}:"
                + ",".join((*decision.missing_prerequisites, *decision.blocking_responsibilities))
            )
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
        result, metric, trace_values, internal_record = self._execute_real_operation(
            unit,
            identity,
            unit_input,
        )
        record = self._record(
            unit,
            identity,
            result,
            metric,
            trace_values,
            internal_record=internal_record,
            attempt_index=attempt_index,
            retry_parent_intent_digest=retry_parent_intent_digest,
        )
        return DevelopmentUnitRunResult(record=record, intent=None, committed=None)

    def execute_and_commit_unit(
        self,
        lease: PersistentLease,
        unit_index: int,
        unit_input: DevelopmentUnitInput,
        *,
        prerequisite_outcomes: Mapping[str, str],
        shard_id: str,
        now_epoch_seconds: int,
        raw_secret_values: Sequence[str],
    ) -> DevelopmentUnitRunResult:
        if self.persistence_store is None:
            raise DevelopmentRunnerError("persistent store is required")
        unit = self._unit(unit_index)
        unit_id = self._unit_id(unit)
        attempt_index = self.persistence_store.next_attempt_index(unit_id)
        recovery = self.persistence_store.recover()
        parent_digest = next(
            (
                item.retry_parent_intent_digest
                for item in recovery.interrupted_attempts
                if item.unit_id == unit_id and item.attempt_index == attempt_index - 1
            ),
            None,
        )
        intent = self.persistence_store.create_intent(
            lease,
            shard_id=shard_id,
            unit_id=unit_id,
            unit_index=unit_index,
            attempt_index=attempt_index,
            parent_attempt_intent_digest=parent_digest,
            now_epoch_seconds=now_epoch_seconds,
        )
        executed = self.execute_unit(
            unit_index,
            unit_input,
            prerequisite_outcomes=prerequisite_outcomes,
            attempt_index=attempt_index,
            retry_parent_intent_digest=parent_digest,
        )
        record_bytes = canonical_json_bytes(executed.record.payload())
        committed = self.persistence_store.commit_unit(
            lease,
            intent,
            members={
                "records/development_scientific_record.json": record_bytes,
            },
            raw_secret_values=raw_secret_values,
            now_epoch_seconds=now_epoch_seconds,
        )
        return DevelopmentUnitRunResult(executed.record, intent, committed)

    def build_module_outcome_record(
        self,
        records: Sequence[DevelopmentScientificRecord],
        *,
        responsibility_id: str,
        module_outcome: str,
        candidate_recommendation: str,
        recommendation_reason: str,
        blocking_responsibilities: Sequence[str] = (),
        provisional_threshold_identities: Sequence[str] = (),
    ) -> DevelopmentModuleOutcomeRecord:
        """Recompute cluster coverage before creating the protocol outcome."""

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
        if any(
            type(record) is not DevelopmentScientificRecord
            or record.responsibility_id != responsibility_id
            or record.execution_status != "success"
            or record.protocol_digest != self.protocol.digest()
            or record.execution_intent_authority_digest
            != self.intent_authority.authority_digest
            for record in records
        ):
            raise DevelopmentRunnerError("module outcome evidence records drifted")
        observations = tuple(
            DevelopmentMetricObservation(**record.metric_observation)
            for record in records
        )
        aggregate_development_cluster_metrics(
            responsibility_id,
            observations,
            minimum_source_clusters=study.scientific_source_cluster_scale,
        )
        return create_development_module_outcome_record(
            self.protocol,
            responsibility_id=responsibility_id,
            module_outcome=module_outcome,
            candidate_recommendation=candidate_recommendation,
            recommendation_reason=recommendation_reason,
            evidence_record_ids=tuple(record.record_id for record in records),
            blocking_responsibilities=blocking_responsibilities,
            provisional_threshold_identities=provisional_threshold_identities,
        )

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
        adaptive = self.adapter.route_content(shape, mode="routing_stqr", observations=raw.routing_observations).result
        uniform = self.adapter.route_content(shape, mode="routing_uniform_control").result
        routing = adaptive if unit.content_branch_id == "lf_hf_routed_combination" else uniform
        traces.update(
            routing_identity=routing.route_identity,
            routing_control=routing.mode,
            routing_observation_digest=_canonical_digest(routing.observation_digests),
            routing_mask_digest=_canonical_digest((routing.mask_lf_digest, routing.mask_hf_digest)),
        )
        if responsibility == "content_router":
            return adaptive, metric_content_router(
                identity.source_cluster_id,
                adaptive_latent_shape=adaptive.latent_shape,
                uniform_latent_shape=uniform.latent_shape,
                adaptive_routing_map=adaptive.routing_map,
                uniform_routing_map=uniform.routing_map,
                adaptive_mean_mask_lf=adaptive.mean_mask_lf,
                adaptive_mean_mask_hf=adaptive.mean_mask_hf,
                uniform_mean_mask_lf=uniform.mean_mask_lf,
                uniform_mean_mask_hf=uniform.mean_mask_hf,
                adaptive_route_identity=adaptive.route_identity,
                uniform_route_identity=uniform.route_identity,
            ), traces, None
        lf_carrier = self.adapter.build_lf_carrier(raw.registered_root_key, shape, routing_result=routing).result
        hf_carrier = self.adapter.build_hf_carrier(raw.registered_root_key, shape, routing_result=routing).result
        if responsibility == "lf_carrier":
            return lf_carrier, metric_lf_carrier(
                identity.source_cluster_id,
                direction=lf_carrier.direction,
                template=lf_carrier.template,
                active_support_count=sum(value != 0.0 for value in lf_carrier.direction),
                direction_digest=lf_carrier.direction_digest,
                template_digest=lf_carrier.template_digest,
                carrier_config_digest=lf_carrier.carrier_config_digest,
            ), traces, None
        if responsibility == "hf_carrier":
            return hf_carrier, metric_hf_carrier(
                identity.source_cluster_id,
                direction=hf_carrier.direction,
                template=hf_carrier.template,
                active_support_count=len(hf_carrier.support_indices),
                direction_digest=hf_carrier.direction_digest,
                template_digest=hf_carrier.template_digest,
                carrier_config_digest=hf_carrier.carrier_config_digest,
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
        hf_null = self.adapter.detect_hf(clean_hf_observation, raw.registered_root_key).result
        lf_registered = self.adapter.detect_lf(registered_lf_observation, raw.registered_root_key).result
        lf_wrong = self.adapter.detect_lf(registered_lf_observation, wrong_material).result
        lf_null = self.adapter.detect_lf(clean_lf_observation, raw.registered_root_key).result
        traces.update(
            hf_score=hf_registered.hf_score,
            lf_score=lf_registered.lf_score,
            raw_detector_identity=hf_registered.detector_identity,
            rectified_detector_identity=hf_registered.detector_identity,
            raw_detector_config_digest=hf_registered.detector_config_digest,
            rectified_detector_config_digest=hf_registered.detector_config_digest,
            raw_preprocessing_identity="final_image_vae_posterior_mode",
            rectified_preprocessing_identity="final_image_vae_posterior_mode",
        )
        if responsibility == "lf_detector":
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
        source_image = _decoded_image_to_rgb8(
            runtime_result.clean_image
            if unit.content_branch_id == "clean_control"
            else runtime_result.watermarked_image
        )
        attacked = apply_geometric_attack(
            AttackArtifact(identity, source_image),
            raw.attack_specification,
            registry=self.attack_registry,
        )
        runtime_qk = self.runtime_adapter.observe_detection_qk(attacked.attacked_artifact.image)
        qk_registered = self.adapter.synchronize_qk_observation(runtime_qk, raw.registered_root_key).result
        qk_wrong = self.adapter.synchronize_qk_observation(runtime_qk, wrong_material).result
        traces.update(
            geometry_operation_identity=raw.attack_specification.attack_id,
            attack_config_digest=raw.attack_specification.attack_config_digest,
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
            ), traces, None
        estimation = self.adapter.estimate_geometric_transform(
            qk_registered, raw.registered_root_key, epsilon_inlier=raw.epsilon_inlier
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
                estimated_rotation_degrees=estimation.transform.residual_rotation_degrees,
                estimated_coverage=estimation.coverage,
                mean_residual=estimation.mean_residual,
                key_margin=estimation.key_margin,
                estimation_identity_digest=estimation.estimation_identity_digest,
                search_config_digest=estimation.search_config_digest,
                truth_crop_fraction=raw.attack_specification.crop_fraction,
                truth_scale=raw.attack_specification.scale_factor,
                truth_rotation_degrees=raw.attack_specification.rotation_degrees,
            )
            return estimation, metric, traces, None
        reliability = self.adapter.assess_geometry_reliability(
            estimation, raw.geometry_reliability_thresholds
        ).result
        wrong_estimation = self.adapter.estimate_geometric_transform(
            qk_wrong, raw.registered_root_key, epsilon_inlier=raw.epsilon_inlier
        ).result
        wrong_reliability = self.adapter.assess_geometry_reliability(
            wrong_estimation, raw.geometry_reliability_thresholds
        ).result
        traces.update(
            geometry_reliability_identity=reliability.reliability_identity_digest,
            geometry_reliable=reliability.reliable,
            geometry_reliability_config_digest=reliability.threshold_config_digest,
        )
        if responsibility == "geometry_reliability":
            return reliability, metric_geometry_reliability(
                identity.source_cluster_id,
                reliable_case_accepted=reliability.reliable,
                unreliable_control_accepted=wrong_reliability.reliable,
                reliable_identity_digest=reliability.reliability_identity_digest,
                unreliable_identity_digest=wrong_reliability.reliability_identity_digest,
            ), traces, None
        if not reliability.reliable:
            raise DevelopmentRunnerError("rectification unit was blocked by fail-closed reliability")
        rectified = self.adapter.rectify_image(attacked.attacked_artifact.image, estimation, reliability).result
        content_operation = FormalHfContentDetectionOperation(self.adapter)
        attacked_content = content_operation(attacked.attacked_artifact.image, raw.registered_root_key)
        rectified_content = content_operation(rectified.rectified_image, raw.registered_root_key)
        traces.update(
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
            ), traces, None
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
    ) -> DevelopmentScientificRecord:
        study = next(item for item in self.protocol.module_matrix if item.responsibility_id == unit.responsibility_id)
        result_payload_value = _public_payload(result)
        if type(result_payload_value) is not dict:
            result_payload_value = {"result": result_payload_value}
        result_payload = result_payload_value
        metric.validate()
        record_id = _canonical_digest(
            {
                "authority": self.intent_authority.authority_digest,
                "attempt_index": attempt_index,
                "unit_index": unit.unit_index,
            }
        )
        routing_trace = {
            key: traces.get(key)
            for key in (
                "routing_identity", "routing_control", "routing_observation_digest", "routing_mask_digest"
            )
        }
        branch_score_trace = {
            "lf_score": traces.get("lf_score"),
            "hf_score": traces.get("hf_score"),
            "combined_score": traces.get("combined_score"),
        }
        detector_trace = {
            key: traces.get(key)
            for key in (
                "raw_detector_identity", "rectified_detector_identity",
                "raw_detector_config_digest", "rectified_detector_config_digest",
                "raw_preprocessing_identity", "rectified_preprocessing_identity",
                "raw_content_score", "rectified_content_score",
            )
        }
        geometry_trace = {
            key: traces.get(key)
            for key in (
                "geometry_operation_identity", "geometry_reliability_config_digest",
                "geometry_estimation_identity", "geometry_reliability_identity",
                "geometry_reliable", "geometry_transform", "geometry_raw_metrics",
                "rectification_status", "qk_registered_relation_score", "qk_wrong_relation_score",
            )
        }
        threshold_trace = {
            key: traces.get(key)
            for key in (
                "raw_threshold_identity", "rectified_threshold_identity", "tau", "tau_rescue",
                "threshold_role",
            )
        }
        key_control_trace = {
            key: traces.get(key)
            for key in (
                "registered_key_public_digest", "detection_key_public_digest", "key_role", "control_identity"
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
            record_id=record_id,
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
        self._validate_record(record, study)
        return record

    def _validate_record(self, record: DevelopmentScientificRecord, study) -> None:
        if record.schema_version != DEVELOPMENT_RECORD_SCHEMA:
            raise DevelopmentRunnerError("development record schema drifted")
        if record.module_outcome is not None or record.candidate_recommendation is not None:
            raise DevelopmentRunnerError("per-unit record cannot preempt cluster outcome")
        if record.candidate_config_digest != study.candidate_config_digest:
            raise DevelopmentRunnerError("record candidate digest drifted")
        if record.metric_ids != study.metric_ids:
            raise DevelopmentRunnerError("record metric registry drifted")
        if record.operation_result_digest != _canonical_digest(record.operation_result_payload):
            raise DevelopmentRunnerError("operation result digest drifted")
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

    def _unit(self, unit_index: int) -> DevelopmentStudyUnit:
        if type(unit_index) is not int or not 0 <= unit_index < len(self.protocol.unit_roster):
            raise DevelopmentRunnerError("unit index is outside frozen roster")
        unit = self.protocol.unit_roster[unit_index]
        if unit.unit_index != unit_index:
            raise DevelopmentRunnerError("unit roster index drifted")
        return unit

    def _cluster_identities(self) -> tuple[AnalysisUnitIdentity, ...]:
        by_cluster: dict[str, AnalysisUnitIdentity] = {}
        for assignment in self.intent_authority.input_manifest.assignments:
            existing = by_cluster.get(assignment.identity.source_cluster_id)
            if existing is not None and existing != assignment.identity:
                raise DevelopmentRunnerError("manifest cluster identity is not unique")
            by_cluster[assignment.identity.source_cluster_id] = assignment.identity
        clusters = tuple(by_cluster[key] for key in sorted(by_cluster))
        if len(clusters) < 64:
            raise DevelopmentRunnerError("development manifest lacks frozen 64-cluster roster")
        return clusters

    def _analysis_identity(self, unit: DevelopmentStudyUnit) -> AnalysisUnitIdentity:
        # Cross-fit authorization is bound to the exact manifest identity.  The
        # atomic development case remains separate in the scientific record.
        return self._clusters[unit.source_cluster_ordinal]

    def _representative_unit(
        self,
        responsibility_id: str,
        source_cluster_ordinal: int,
    ) -> DevelopmentStudyUnit:
        unit = next(
            (
                item
                for item in self.protocol.unit_roster
                if item.responsibility_id == responsibility_id
                and item.source_cluster_ordinal == source_cluster_ordinal
            ),
            None,
        )
        if unit is None:
            raise DevelopmentRunnerError("frozen roster lacks operational representative")
        return unit

    @staticmethod
    def _unit_id(unit: DevelopmentStudyUnit) -> str:
        return f"development_scientific_unit_{unit.unit_index:04d}"
