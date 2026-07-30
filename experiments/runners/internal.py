"""Governed internal runner that composes existing method, attack, and metric APIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from typing import Callable, Mapping

from experiments.attacks import (
    AttackArtifact,
    AttackRegistry,
    GeometricAttackError,
    GeometricAttackSpec,
    apply_geometric_attack,
    validate_attack_artifact,
    validate_attack_registry,
    validate_geometric_attack_spec,
)
from experiments.methods import CegWmExperimentAdapter
from experiments.metrics import (
    MetricRegistry,
    RectificationMetricCase,
    aggregate_rectification_delta,
    validate_metric_registry,
)
from experiments.protocol.internal_records import (
    BranchScoreTrace,
    DecisionTrace,
    DetectorTrace,
    GeometryTrace,
    InternalValidationRecord,
    KeyControlTrace,
    ProvenanceTrace,
    RoutingTrace,
    RunCaseRecordCollection,
    ThresholdTrace,
)
from experiments.protocol.internal_case import (
    FrozenCaseExecutionExpectation,
    FrozenCaseInputManifest,
    InternalCaseManifestEntry,
    derive_internal_record_id,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
    SplitAccessGrant,
    authorize_split_access,
)
from experiments.protocol.internal_validation import (
    FrozenInternalValidationProtocol,
)
from experiments.runners.record_writer import (
    FrozenRecordBindings,
    GovernedRecordWriter,
    GovernedRecordWriterError,
    canonical_record_digest,
)
from main import (
    ConditionalRecoveryResult,
    ContentDetectorBinding,
    GeometryEstimationOperation,
    GeometryReliabilityThresholds,
    JointDecisionThresholds,
)


SCIENTIFIC_RESULT_FAILURE_STATUSES = frozenset(
    {
        "negative_geometry_operation_failure",
        "negative_geometry_reliability_failure",
        "negative_rectification_failure",
        "negative_rectified_content_operation_failure",
        "raw_content_operation_failure",
    }
)
FORMAL_RUNNER_SEMANTIC_DECLARATION_METHOD = (
    "formal_runner_semantic_declaration"
)
FORMAL_OPERATION_ROLES = frozenset(
    {"content_detection", "geometry_estimation"}
)


class InternalRunnerError(ValueError):
    """A frozen input, composition, resume, or replay boundary failed closed."""


class ResourceExecutionError(RuntimeError):
    """Explicit transient resource failure; never interpreted as method science."""


@dataclass(frozen=True, slots=True)
class InternalCaseExecutionPayload:
    source_artifact: AttackArtifact
    attack_specification: GeometricAttackSpec
    detection_key: str
    content_detector_binding: ContentDetectorBinding
    thresholds: JointDecisionThresholds
    geometry_estimation_operation: GeometryEstimationOperation
    geometry_operation_identity: str
    geometry_reliability_thresholds: GeometryReliabilityThresholds | None
    _content_operation_anchor: object = field(
        init=False,
        repr=False,
        compare=False,
    )
    _geometry_operation_anchor: object = field(
        init=False,
        repr=False,
        compare=False,
    )
    _content_operation_type_identity: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _geometry_operation_type_identity: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _construction_anchor_digest: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        declaration = _payload_declaration_snapshot(self)
        content_operation = (
            self.content_detector_binding.content_detection_operation
        )
        object.__setattr__(
            self,
            "_content_operation_anchor",
            content_operation,
        )
        object.__setattr__(
            self,
            "_geometry_operation_anchor",
            self.geometry_estimation_operation,
        )
        object.__setattr__(
            self,
            "_content_operation_type_identity",
            _callable_type_identity(content_operation),
        )
        object.__setattr__(
            self,
            "_geometry_operation_type_identity",
            _callable_type_identity(self.geometry_estimation_operation),
        )
        object.__setattr__(
            self,
            "_construction_anchor_digest",
            _canonical_digest(declaration),
        )


@dataclass(frozen=True, slots=True)
class InternalRunnerContext:
    protocol: FrozenInternalValidationProtocol
    split_manifest: FrozenSplitManifest
    input_manifest: FrozenCaseInputManifest
    adapter: CegWmExperimentAdapter
    attack_registry: AttackRegistry
    metric_registry: MetricRegistry
    writer: GovernedRecordWriter
    bindings: FrozenRecordBindings
    _construction_anchor_digest: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_context_components(self)
        object.__setattr__(
            self,
            "_construction_anchor_digest",
            _context_anchor_digest(self),
        )


@dataclass(frozen=True, slots=True)
class InternalCaseRunResult:
    record: InternalValidationRecord
    collection: RunCaseRecordCollection
    resumed_without_execution: bool


@dataclass(frozen=True, slots=True)
class MetricCaseEvidence:
    record_id: str
    canonical_record_digest: str
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    score_delta: float


@dataclass(frozen=True, slots=True)
class MetricAggregateEvidence:
    case_count: int
    split: str
    mean_score_delta: float
    improved_fraction: float
    detector_identity: str
    threshold_identity: str


@dataclass(frozen=True, slots=True)
class RecordReplayReport:
    run_id: str
    case_id: str
    record_count: int
    success_count: int
    resource_failure_count: int
    scientific_failure_count: int
    execution_failure_count: int
    excluded_count: int
    retry_count: int
    metric_case_count: int
    metric_registry_digest: str
    metric_evaluator_identity: str | None
    metric_aggregate_identity: str | None
    metric_case_results: tuple[MetricCaseEvidence, ...]
    metric_aggregate_values: MetricAggregateEvidence | None
    metric_observation_digest: str
    replay_digest: str


def execution_config_digest(
    *,
    protocol: FrozenInternalValidationProtocol,
    adapter: CegWmExperimentAdapter,
    attack_registry: AttackRegistry,
    metric_registry: MetricRegistry,
) -> str:
    """Bind the exact existing public component registries used by the runner."""

    attack_violations = validate_attack_registry(attack_registry)
    metric_violations = validate_metric_registry(metric_registry)
    if attack_violations or metric_violations:
        raise InternalRunnerError(
            "execution registry invalid: "
            + ",".join((*attack_violations, *metric_violations))
        )
    return _canonical_digest(
        {
            "adapter_config_digest": adapter.configuration.config_digest,
            "attack_registry_digest": attack_registry.registry_digest,
            "metric_registry_digest": metric_registry.registry_digest,
            "protocol_digest": protocol.digest(),
            "runner": "experiments.runners.internal",
        }
    )


def geometry_reliability_config_digest(
    thresholds: GeometryReliabilityThresholds,
) -> str:
    """Digest an experiments-layer reliability configuration declaration.

    This identity binds only the declared dataclass fields consumed by the
    runner.  It is not the method-layer threshold_config_digest emitted by a
    geometry reliability result.
    """

    declaration = _validated_geometry_reliability_declaration(thresholds)
    return _geometry_reliability_declaration_digest(declaration)


def formal_operation_config_digest(
    operation: object,
    *,
    operation_role: str,
) -> str:
    """Digest one explicitly declared formal-runner callable configuration."""

    return _canonical_digest(
        _formal_operation_config_snapshot(
            operation,
            operation_role=operation_role,
        )
    )


def _geometry_reliability_declaration_digest(
    declaration: Mapping[str, object],
) -> str:
    return _canonical_digest(
        {
            "configuration_kind": (
                "experiments_geometry_reliability_threshold_declaration"
            ),
            "thresholds": declaration,
        }
    )


def candidate_config_digest(
    *,
    adapter: CegWmExperimentAdapter,
    input_manifest: FrozenCaseInputManifest,
    method_code_revision: str,
) -> str:
    """Bind candidate-facing method and fitted decision identities without keys."""

    if type(input_manifest) is not FrozenCaseInputManifest:
        raise InternalRunnerError("frozen input manifest exact type is required")
    if (
        type(method_code_revision) is not str
        or len(method_code_revision) != 40
    ):
        raise InternalRunnerError("method code revision must be exact")
    routing_candidate_policy = sorted(
        {
            (
                entry.routing_trace.routing_identity,
                entry.routing_trace.routing_control,
            )
            for entry in input_manifest.entries
            if type(entry) is InternalCaseManifestEntry
            and type(entry.routing_trace) is RoutingTrace
        }
    )
    if not routing_candidate_policy:
        raise InternalRunnerError("routing candidate policy is missing")
    execution_expectations = sorted(
        (
            (
                entry.analysis_unit_identity.unit_id,
                asdict(entry.execution_expectation),
            )
            for entry in input_manifest.entries
            if type(entry) is InternalCaseManifestEntry
            and type(entry.execution_expectation)
            is FrozenCaseExecutionExpectation
        ),
        key=lambda item: item[0],
    )
    if len(execution_expectations) != len(input_manifest.entries):
        raise InternalRunnerError("case execution expectation is missing")
    return _canonical_digest(
        {
            "adapter_config_digest": adapter.configuration.config_digest,
            "case_execution_expectations": [
                {
                    "expectation": expectation,
                    "unit_id": unit_id,
                }
                for unit_id, expectation in execution_expectations
            ],
            "method_code_revision": method_code_revision,
            "routing_candidate_policy": [
                {
                    "routing_control": routing_control,
                    "routing_identity": routing_identity,
                }
                for routing_identity, routing_control in routing_candidate_policy
            ],
        }
    )


def execute_internal_case(
    context: InternalRunnerContext,
    *,
    unit_id: str,
    payload: InternalCaseExecutionPayload,
) -> InternalCaseRunResult:
    """Execute or resume one manifest-bound case through real public components."""

    entry = _validated_case(context, unit_id=unit_id, payload=payload)
    existing = context.writer.load()
    prior = _unit_records(existing, entry.analysis_unit_identity)
    if prior and _record_is_complete(prior[-1]):
        replay_internal_record_collection(context, existing)
        return InternalCaseRunResult(prior[-1], existing, True)
    attempt_index = len(prior)
    if attempt_index >= context.protocol.maximum_record_attempts:
        raise InternalRunnerError("maximum record attempts already exhausted")
    sequence_index = 0 if existing is None else len(existing.records)
    retry_parent = None if not prior else prior[-1].record_id
    record_id = derive_internal_record_id(
        run_id=context.bindings.run_id,
        case_id=context.bindings.case_id,
        input_manifest_digest=context.bindings.input_manifest_digest,
        analysis_unit_identity=entry.analysis_unit_identity,
        attempt_index=attempt_index,
    )

    _validate_payload_execution_boundary(
        payload,
        entry.execution_expectation,
    )
    try:
        attacked = apply_geometric_attack(
            payload.source_artifact,
            payload.attack_specification,
            registry=context.attack_registry,
        )
        observation = context.adapter.decide_conditional_recovery(
            attacked.attacked_artifact.image,
            payload.detection_key,
            content_detector_binding=payload.content_detector_binding,
            thresholds=payload.thresholds,
            geometry_estimation_operation=payload.geometry_estimation_operation,
            geometry_reliability_thresholds=(
                payload.geometry_reliability_thresholds
            ),
        )
        joint_result = observation.result
        result_failure_class = _joint_result_failure_class(joint_result)
        if result_failure_class is not None:
            record = _record_from_joint_result(
                context,
                entry,
                joint_result,
                record_id=record_id,
                sequence_index=sequence_index,
                attempt_index=attempt_index,
                retry_parent=retry_parent,
                execution_status="failed",
                failure_class=result_failure_class,
            )
        else:
            record = _record_from_joint_result(
                context,
                entry,
                joint_result,
                record_id=record_id,
                sequence_index=sequence_index,
                attempt_index=attempt_index,
                retry_parent=retry_parent,
                execution_status="success",
                failure_class=None,
            )
    except ResourceExecutionError as exc:
        attempt_limit_reached = (
            attempt_index + 1 >= context.protocol.maximum_record_attempts
        )
        status = (
            "failed"
            if attempt_index == 0 or attempt_limit_reached
            else "retry"
        )
        record = _failure_record(
            context,
            entry,
            record_id=record_id,
            sequence_index=sequence_index,
            attempt_index=attempt_index,
            retry_parent=retry_parent,
            execution_status=status,
            failure_class="resource_failure",
            failure_reason=_exception_identity(exc),
        )
    except (GeometricAttackError, GovernedRecordWriterError):
        raise
    except Exception as exc:
        record = _failure_record(
            context,
            entry,
            record_id=record_id,
            sequence_index=sequence_index,
            attempt_index=attempt_index,
            retry_parent=retry_parent,
            execution_status="failed",
            failure_class="execution_failure",
            failure_reason=_exception_identity(exc),
        )
    _validate_payload_execution_boundary(
        payload,
        entry.execution_expectation,
    )
    collection = context.writer.append_record(record)
    replay_internal_record_collection(context, collection)
    return InternalCaseRunResult(record, collection, False)


def record_excluded_case(
    context: InternalRunnerContext,
    *,
    unit_id: str,
    payload: InternalCaseExecutionPayload,
    exclusion_rule_id: str,
    exclusion_reason: str,
) -> InternalCaseRunResult:
    """Persist one preregistered input exclusion without invoking the method."""

    entry = _validated_case(context, unit_id=unit_id, payload=payload)
    existing = context.writer.load()
    prior = _unit_records(existing, entry.analysis_unit_identity)
    if prior:
        if _record_is_complete(prior[-1]):
            replay_internal_record_collection(context, existing)
            return InternalCaseRunResult(prior[-1], existing, True)
        raise InternalRunnerError("exclusion cannot replace an execution attempt")
    if not exclusion_rule_id or not exclusion_reason:
        raise InternalRunnerError("exclusion rule and reason are required")
    record = _base_record(
        context,
        entry,
        record_id=derive_internal_record_id(
            run_id=context.bindings.run_id,
            case_id=context.bindings.case_id,
            input_manifest_digest=context.bindings.input_manifest_digest,
            analysis_unit_identity=entry.analysis_unit_identity,
            attempt_index=0,
        ),
        sequence_index=0 if existing is None else len(existing.records),
        attempt_index=0,
        retry_parent=None,
        execution_status="excluded",
        failure_class=None,
        failure_reason=None,
        exclusion_reason=exclusion_reason,
        exclusion_rule_id=exclusion_rule_id,
        detector_trace=_empty_detector_trace(entry.execution_expectation),
        branch_score_trace=BranchScoreTrace(None, None, None),
        geometry_trace=_untriggered_geometry(
            entry.execution_expectation
        ),
        decision_trace=DecisionTrace("excluded", None, exclusion_reason),
    )
    _validate_payload_execution_boundary(
        payload,
        entry.execution_expectation,
    )
    collection = context.writer.append_record(record)
    replay_internal_record_collection(context, collection)
    return InternalCaseRunResult(record, collection, False)


def replay_internal_record_collection(
    context: InternalRunnerContext,
    collection: RunCaseRecordCollection | None = None,
) -> RecordReplayReport:
    """Reload real records and recompute schema, decision, and metric consistency."""

    _validate_context(context)
    persisted = context.writer.load()
    if persisted is None:
        raise InternalRunnerError("no persisted records are available for replay")
    if collection is not None and _canonical_digest(collection.to_dict()) != _canonical_digest(
        persisted.to_dict()
    ):
        raise InternalRunnerError("replay collection differs from persisted records")
    metric_cases: list[RectificationMetricCase] = []
    metric_records: list[InternalValidationRecord] = []
    for record in persisted.records:
        if record.execution_status != "success":
            continue
        raw_score = record.detector_trace.raw_content_score
        if raw_score is None:
            raise InternalRunnerError("successful record has no raw metric score")
        rectified_score = record.detector_trace.rectified_content_score
        if rectified_score is None:
            continue
        metric_cases.append(
            RectificationMetricCase(
                analysis_unit_identity=record.analysis_unit_identity,
                split=record.split,
                raw_detector_identity=(
                    record.detector_trace.raw_detector_identity
                ),
                rectified_detector_identity=(
                    record.detector_trace.rectified_detector_identity
                ),
                raw_threshold_identity=(
                    record.threshold_trace.raw_threshold_identity
                ),
                rectified_threshold_identity=(
                    record.threshold_trace.rectified_threshold_identity
                ),
                raw_score=raw_score,
                rectified_score=rectified_score,
            )
        )
        metric_records.append(record)
    metric_evaluator_identity: str | None = None
    metric_aggregate_identity: str | None = None
    metric_case_results: tuple[MetricCaseEvidence, ...] = ()
    metric_aggregate_values: MetricAggregateEvidence | None = None
    if metric_cases:
        aggregate = aggregate_rectification_delta(
            metric_cases,
            registry=context.metric_registry,
        )
        metric_evaluator_identity = (
            "experiments.metrics.aggregate_rectification_delta"
        )
        metric_aggregate_identity = (
            "experiments.metrics.RectificationDeltaAggregate"
        )
        metric_case_results = tuple(
            MetricCaseEvidence(
                record_id=record.record_id,
                canonical_record_digest=canonical_record_digest(record),
                unit_id=result.unit_id,
                case_id=result.case_id,
                source_cluster_id=result.source_cluster_id,
                split=result.split,
                score_delta=result.score_delta,
            )
            for record, result in zip(
                metric_records,
                aggregate.cases,
                strict=True,
            )
        )
        metric_aggregate_values = MetricAggregateEvidence(
            case_count=len(aggregate.cases),
            split=aggregate.split,
            mean_score_delta=aggregate.mean_score_delta,
            improved_fraction=aggregate.improved_fraction,
            detector_identity=aggregate.detector_identity,
            threshold_identity=aggregate.threshold_identity,
        )
    counts = {
        "success": sum(record.execution_status == "success" for record in persisted.records),
        "resource": sum(
            record.failure_class == "resource_failure" for record in persisted.records
        ),
        "scientific": sum(
            record.failure_class == "scientific_failure" for record in persisted.records
        ),
        "execution": sum(
            record.failure_class == "execution_failure" for record in persisted.records
        ),
        "excluded": sum(record.execution_status == "excluded" for record in persisted.records),
        "retry": sum(record.execution_status == "retry" for record in persisted.records),
    }
    replay_digest = _canonical_digest(
        {
            "collection": persisted.to_dict(),
            "metric_registry_digest": context.metric_registry.registry_digest,
            "replay": "schema_decision_metric_consistency",
        }
    )
    metric_observation_digest = _canonical_digest(
        {
            "metric_aggregate_identity": metric_aggregate_identity,
            "metric_aggregate_values": (
                None
                if metric_aggregate_values is None
                else asdict(metric_aggregate_values)
            ),
            "metric_case_results": [
                asdict(result) for result in metric_case_results
            ],
            "metric_evaluator_identity": metric_evaluator_identity,
            "metric_registry_digest": context.metric_registry.registry_digest,
        }
    )
    return RecordReplayReport(
        run_id=persisted.run_id,
        case_id=persisted.case_id,
        record_count=len(persisted.records),
        success_count=counts["success"],
        resource_failure_count=counts["resource"],
        scientific_failure_count=counts["scientific"],
        execution_failure_count=counts["execution"],
        excluded_count=counts["excluded"],
        retry_count=counts["retry"],
        metric_case_count=len(metric_cases),
        metric_registry_digest=context.metric_registry.registry_digest,
        metric_evaluator_identity=metric_evaluator_identity,
        metric_aggregate_identity=metric_aggregate_identity,
        metric_case_results=metric_case_results,
        metric_aggregate_values=metric_aggregate_values,
        metric_observation_digest=metric_observation_digest,
        replay_digest=replay_digest,
    )


def _validated_case(
    context: InternalRunnerContext,
    *,
    unit_id: str,
    payload: InternalCaseExecutionPayload,
) -> InternalCaseManifestEntry:
    _validate_context(context)
    if type(payload) is not InternalCaseExecutionPayload:
        raise InternalRunnerError("case execution payload exact type is required")
    entries = [
        entry
        for entry in context.input_manifest.entries
        if entry.analysis_unit_identity.unit_id == unit_id
    ]
    if len(entries) != 1:
        raise InternalRunnerError("unit_id must resolve to one frozen input entry")
    entry = entries[0]
    authorize_split_access(
        context.split_manifest,
        (entry.split,),
        SplitAccessGrant.current_execution(),
    )
    _validate_payload_execution_boundary(
        payload,
        entry.execution_expectation,
    )
    if payload.source_artifact.analysis_unit_identity != entry.analysis_unit_identity:
        raise InternalRunnerError("source artifact analysis identity drifted")
    if payload.source_artifact.image_digest != entry.input_artifact_digest:
        raise InternalRunnerError("source artifact digest drifted")
    if payload.attack_specification.attack_config_digest != entry.attack_config_digest:
        raise InternalRunnerError("attack configuration digest drifted")
    if entry.metric_set_digest != context.metric_registry.registry_digest:
        raise InternalRunnerError("metric set digest drifted")
    observed_key = context.adapter.identify_key(payload.detection_key).result
    if observed_key.root_key_public_digest != (
        entry.key_control_trace.detection_key_public_digest
    ):
        raise InternalRunnerError("detection key public identity drifted")
    if candidate_config_digest(
        adapter=context.adapter,
        input_manifest=context.input_manifest,
        method_code_revision=context.bindings.method_code_revision,
    ) != (
        context.bindings.candidate_config_digest
    ):
        raise InternalRunnerError("candidate configuration digest drifted")
    if payload.content_detector_binding.content_config_digest != (
        context.bindings.method_config_digest
    ):
        raise InternalRunnerError("method configuration digest drifted")
    return entry


def _payload_declaration_snapshot(
    payload: InternalCaseExecutionPayload,
) -> dict[str, object]:
    if type(payload.source_artifact) is not AttackArtifact:
        raise InternalRunnerError("source artifact exact type is required")
    artifact_violations = validate_attack_artifact(payload.source_artifact)
    if artifact_violations:
        raise InternalRunnerError(
            f"source artifact invalid: {','.join(artifact_violations)}"
        )
    if type(payload.attack_specification) is not GeometricAttackSpec:
        raise InternalRunnerError("attack specification exact type is required")
    attack_violations = validate_geometric_attack_spec(
        payload.attack_specification
    )
    if attack_violations:
        raise InternalRunnerError(
            f"attack specification invalid: {','.join(attack_violations)}"
        )
    if type(payload.detection_key) is not str or not payload.detection_key:
        raise InternalRunnerError("detection key is required")
    binding = payload.content_detector_binding
    if type(binding) is not ContentDetectorBinding:
        raise InternalRunnerError("content detector binding exact type is required")
    try:
        rebuilt_binding = ContentDetectorBinding(
            content_detection_operation=binding.content_detection_operation,
            detector_identity=binding.detector_identity,
            content_config_digest=binding.content_config_digest,
            hf_detector_identity=binding.hf_detector_identity,
            hf_detector_config_digest=binding.hf_detector_config_digest,
            hf_template_digest=binding.hf_template_digest,
            preprocessing_identity=binding.preprocessing_identity,
            formal_mode=binding.formal_mode,
            root_key_public_digest=binding.root_key_public_digest,
            key_role=binding.key_role,
            wrong_key_index=binding.wrong_key_index,
        )
    except Exception as exc:
        raise InternalRunnerError(
            "content detector binding declaration invalid"
        ) from exc
    if binding.detector_binding_digest != (
        rebuilt_binding.detector_binding_digest
    ):
        raise InternalRunnerError("content detector binding digest drifted")
    content_operation_config = _formal_operation_config_snapshot(
        binding.content_detection_operation,
        operation_role="content_detection",
    )
    thresholds = payload.thresholds
    if type(thresholds) is not JointDecisionThresholds:
        raise InternalRunnerError("joint thresholds exact type is required")
    try:
        rebuilt_thresholds = JointDecisionThresholds(
            tau=thresholds.tau,
            tau_rescue=thresholds.tau_rescue,
            detector_binding_digest=thresholds.detector_binding_digest,
            calibration_identity=thresholds.calibration_identity,
        )
    except Exception as exc:
        raise InternalRunnerError(
            "joint threshold declaration invalid"
        ) from exc
    if thresholds.threshold_identity != rebuilt_thresholds.threshold_identity:
        raise InternalRunnerError("joint threshold identity drifted")
    if thresholds.detector_binding_digest != binding.detector_binding_digest:
        raise InternalRunnerError("threshold detector binding drifted")
    if not callable(payload.geometry_estimation_operation):
        raise InternalRunnerError("geometry estimation operation must be callable")
    if (
        type(payload.geometry_operation_identity) is not str
        or not payload.geometry_operation_identity
    ):
        raise InternalRunnerError("geometry operation identity is required")
    geometry_operation_config = _formal_operation_config_snapshot(
        payload.geometry_estimation_operation,
        operation_role="geometry_estimation",
    )
    reliability = payload.geometry_reliability_thresholds
    reliability_digest: str | None = None
    reliability_declaration: dict[str, object] | None = None
    if reliability is not None:
        reliability_declaration = (
            _validated_geometry_reliability_declaration(reliability)
        )
        reliability_digest = _geometry_reliability_declaration_digest(
            reliability_declaration
        )
    return {
        "attack_config_digest": (
            payload.attack_specification.attack_config_digest
        ),
        "content_detector_binding": {
            "content_config_digest": binding.content_config_digest,
            "detector_binding_digest": binding.detector_binding_digest,
            "detector_identity": binding.detector_identity,
            "formal_mode": binding.formal_mode,
            "hf_detector_config_digest": binding.hf_detector_config_digest,
            "hf_detector_identity": binding.hf_detector_identity,
            "hf_template_digest": binding.hf_template_digest,
            "key_role": binding.key_role,
            "preprocessing_identity": binding.preprocessing_identity,
            "root_key_public_digest": binding.root_key_public_digest,
            "wrong_key_index": binding.wrong_key_index,
        },
        "content_operation_type_identity": _callable_type_identity(
            binding.content_detection_operation
        ),
        "content_operation_config": content_operation_config,
        "content_operation_config_digest": _canonical_digest(
            content_operation_config
        ),
        "detection_key_digest": sha256(
            payload.detection_key.encode("utf-8")
        ).hexdigest(),
        "geometry_operation_identity": payload.geometry_operation_identity,
        "geometry_operation_config": geometry_operation_config,
        "geometry_operation_config_digest": _canonical_digest(
            geometry_operation_config
        ),
        "geometry_operation_type_identity": _callable_type_identity(
            payload.geometry_estimation_operation
        ),
        "geometry_reliability_config_digest": reliability_digest,
        "geometry_reliability_thresholds": reliability_declaration,
        "input_artifact_digest": payload.source_artifact.image_digest,
        "thresholds": {
            "calibration_identity": thresholds.calibration_identity,
            "detector_binding_digest": thresholds.detector_binding_digest,
            "tau": thresholds.tau,
            "tau_rescue": thresholds.tau_rescue,
            "threshold_identity": thresholds.threshold_identity,
        },
    }


def _validate_payload_construction_anchor(
    payload: InternalCaseExecutionPayload,
) -> None:
    if type(payload) is not InternalCaseExecutionPayload:
        raise InternalRunnerError("case execution payload exact type is required")
    declaration = _payload_declaration_snapshot(payload)
    content_operation = (
        payload.content_detector_binding.content_detection_operation
    )
    if content_operation is not payload._content_operation_anchor:
        raise InternalRunnerError("content operation object identity drifted")
    if (
        _callable_type_identity(content_operation)
        != payload._content_operation_type_identity
    ):
        raise InternalRunnerError("content operation type identity drifted")
    if (
        payload.geometry_estimation_operation
        is not payload._geometry_operation_anchor
    ):
        raise InternalRunnerError("geometry operation object identity drifted")
    if (
        _callable_type_identity(payload.geometry_estimation_operation)
        != payload._geometry_operation_type_identity
    ):
        raise InternalRunnerError("geometry operation type identity drifted")
    if (
        _canonical_digest(declaration)
        != payload._construction_anchor_digest
    ):
        raise InternalRunnerError("case execution payload construction anchor drifted")


def _validate_payload_execution_boundary(
    payload: InternalCaseExecutionPayload,
    expectation: FrozenCaseExecutionExpectation,
) -> None:
    _validate_payload_construction_anchor(payload)
    _validate_payload_against_expectation(payload, expectation)


def _validate_payload_against_expectation(
    payload: InternalCaseExecutionPayload,
    expectation: FrozenCaseExecutionExpectation,
) -> None:
    if type(expectation) is not FrozenCaseExecutionExpectation:
        raise InternalRunnerError("case execution expectation exact type required")
    binding = payload.content_detector_binding
    thresholds = payload.thresholds
    reliability = payload.geometry_reliability_thresholds
    reliability_digest = (
        None
        if reliability is None
        else geometry_reliability_config_digest(reliability)
    )
    observed = {
        "calibration_identity": thresholds.calibration_identity,
        "content_detector_binding_digest": binding.detector_binding_digest,
        "content_operation_config_digest": formal_operation_config_digest(
            binding.content_detection_operation,
            operation_role="content_detection",
        ),
        "geometry_operation_identity": payload.geometry_operation_identity,
        "geometry_operation_config_digest": formal_operation_config_digest(
            payload.geometry_estimation_operation,
            operation_role="geometry_estimation",
        ),
        "geometry_reliability_config_digest": reliability_digest,
        "raw_detector_config_digest": binding.content_config_digest,
        "raw_detector_identity": binding.detector_identity,
        "raw_preprocessing_identity": binding.preprocessing_identity,
        "raw_threshold_identity": thresholds.threshold_identity,
        "rectified_detector_config_digest": binding.content_config_digest,
        "rectified_detector_identity": binding.detector_identity,
        "rectified_preprocessing_identity": binding.preprocessing_identity,
        "rectified_threshold_identity": thresholds.threshold_identity,
        "tau": thresholds.tau,
        "tau_rescue": thresholds.tau_rescue,
    }
    expected = asdict(expectation)
    if observed != expected:
        raise InternalRunnerError(
            "case execution payload differs from frozen expectation"
        )


def _validated_geometry_reliability_declaration(
    reliability: GeometryReliabilityThresholds,
) -> dict[str, object]:
    if type(reliability) is not GeometryReliabilityThresholds:
        raise InternalRunnerError(
            "geometry reliability thresholds exact type required"
        )
    try:
        current_declaration = asdict(reliability)
        rebuilt_reliability = GeometryReliabilityThresholds(
            **current_declaration
        )
        rebuilt_declaration = asdict(rebuilt_reliability)
        _geometry_reliability_declaration_digest(rebuilt_declaration)
    except (TypeError, ValueError) as exc:
        raise InternalRunnerError(
            "geometry reliability configuration declaration invalid"
        ) from exc
    if rebuilt_declaration != current_declaration:
        raise InternalRunnerError(
            "geometry reliability configuration declaration drifted"
        )
    return rebuilt_declaration


def _formal_operation_config_snapshot(
    operation: object,
    *,
    operation_role: str,
) -> dict[str, object]:
    if operation_role not in FORMAL_OPERATION_ROLES:
        raise InternalRunnerError("formal operation role is invalid")
    if not callable(operation):
        raise InternalRunnerError("formal operation must be callable")
    declaration_provider = getattr(
        operation,
        FORMAL_RUNNER_SEMANTIC_DECLARATION_METHOD,
        None,
    )
    if not callable(declaration_provider):
        raise InternalRunnerError(
            "formal operation semantic declaration is required"
        )
    try:
        declarations = (
            declaration_provider(),
            declaration_provider(),
        )
    except Exception as exc:
        raise InternalRunnerError(
            "formal operation semantic declaration failed"
        ) from exc
    canonical_declarations: list[dict[str, object]] = []
    canonical_documents: list[str] = []
    for declaration in declarations:
        if (
            type(declaration) is not dict
            or not declaration
            or any(type(key) is not str or not key for key in declaration)
        ):
            raise InternalRunnerError(
                "formal operation semantic declaration must be a nonempty string-key mapping"
            )
        try:
            canonical_document = json.dumps(
                declaration,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            canonical_declaration = json.loads(canonical_document)
        except (TypeError, ValueError) as exc:
            raise InternalRunnerError(
                "formal operation semantic declaration is not canonical JSON"
            ) from exc
        if canonical_declaration != declaration:
            raise InternalRunnerError(
                "formal operation semantic declaration is not stable JSON data"
            )
        canonical_declarations.append(canonical_declaration)
        canonical_documents.append(canonical_document)
    if canonical_documents[0] != canonical_documents[1]:
        raise InternalRunnerError(
            "formal operation semantic declaration changed during validation"
        )
    return {
        "declaration_contract": (
            "ceg_wm_formal_runner_semantic_declaration_v1"
        ),
        "operation_role": operation_role,
        "operation_type_identity": _callable_type_identity(operation),
        "semantic_declaration": canonical_declarations[0],
    }


def _callable_type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _validate_context(context: InternalRunnerContext) -> None:
    if type(context) is not InternalRunnerContext:
        raise InternalRunnerError("runner context exact type is required")
    expected_types = (
        (context.protocol, FrozenInternalValidationProtocol, "protocol"),
        (context.split_manifest, FrozenSplitManifest, "split manifest"),
        (context.input_manifest, FrozenCaseInputManifest, "input manifest"),
        (context.adapter, CegWmExperimentAdapter, "method adapter"),
        (context.attack_registry, AttackRegistry, "attack registry"),
        (context.metric_registry, MetricRegistry, "metric registry"),
        (context.writer, GovernedRecordWriter, "record writer"),
        (context.bindings, FrozenRecordBindings, "record bindings"),
    )
    for value, expected_type, role in expected_types:
        if type(value) is not expected_type:
            raise InternalRunnerError(f"{role} exact type is required")
    _validate_context_components(context)
    if _context_anchor_digest(context) != context._construction_anchor_digest:
        raise InternalRunnerError("runner context construction anchor drifted")


def _validate_context_components(context: InternalRunnerContext) -> None:
    context.writer.assert_context_anchors(
        frozen_protocol=context.protocol,
        split_manifest=context.split_manifest,
        input_manifest=context.input_manifest,
        bindings=context.bindings,
    )
    attack_violations = validate_attack_registry(context.attack_registry)
    if attack_violations:
        raise InternalRunnerError(
            f"attack registry invalid: {','.join(attack_violations)}"
        )
    metric_violations = validate_metric_registry(context.metric_registry)
    if metric_violations:
        raise InternalRunnerError(
            f"metric registry invalid: {','.join(metric_violations)}"
        )
    violations = context.input_manifest.validate(
        protocol=context.protocol,
        split_manifest=context.split_manifest,
    )
    if violations:
        raise InternalRunnerError(f"input manifest invalid: {','.join(violations)}")
    if context.input_manifest.digest() != context.bindings.input_manifest_digest:
        raise InternalRunnerError("input manifest digest drifted")
    actual_execution_digest = execution_config_digest(
        protocol=context.protocol,
        adapter=context.adapter,
        attack_registry=context.attack_registry,
        metric_registry=context.metric_registry,
    )
    if actual_execution_digest != context.bindings.execution_config_digest:
        raise InternalRunnerError("execution configuration digest drifted")


def _context_anchor_digest(context: InternalRunnerContext) -> str:
    return _canonical_digest(
        {
            "adapter_config_digest": context.adapter.configuration.config_digest,
            "attack_registry": asdict(context.attack_registry),
            "bindings": asdict(context.bindings),
            "input_manifest": asdict(context.input_manifest),
            "metric_registry": asdict(context.metric_registry),
            "protocol": asdict(context.protocol),
            "split_manifest": asdict(context.split_manifest),
            "writer_path": str(context.writer.path),
        }
    )


def _record_from_joint_result(
    context: InternalRunnerContext,
    entry: InternalCaseManifestEntry,
    result: ConditionalRecoveryResult,
    *,
    record_id: str,
    sequence_index: int,
    attempt_index: int,
    retry_parent: str | None,
    execution_status: str,
    failure_class: str | None,
) -> InternalValidationRecord:
    raw = result.raw_content_result
    branch_scores = BranchScoreTrace(
        lf_score=None if raw is None else raw.lf_score,
        hf_score=None if raw is None else raw.hf_score,
        combined_score=None if raw is None else raw.combined_score,
    )
    detector = DetectorTrace(
        raw_detector_identity=result.detector_identity,
        rectified_detector_identity=result.detector_identity,
        raw_detector_config_digest=result.content_config_digest,
        rectified_detector_config_digest=result.content_config_digest,
        raw_preprocessing_identity=result.preprocessing_identity,
        rectified_preprocessing_identity=result.preprocessing_identity,
        raw_content_score=result.raw_content_score,
        rectified_content_score=result.rectified_content_score,
    )
    geometry = _geometry_trace(result, entry.execution_expectation)
    if execution_status == "success":
        decision = DecisionTrace(
            watermark_decision=(
                "positive" if result.joint_content_positive else "negative"
            ),
            positive_source=result.positive_source,
            decision_reason=result.status,
        )
        failure_reason = None
    else:
        decision = DecisionTrace("failed", None, result.status)
        failure_reason = result.failure_reason or result.status
    return _base_record(
        context,
        entry,
        record_id=record_id,
        sequence_index=sequence_index,
        attempt_index=attempt_index,
        retry_parent=retry_parent,
        execution_status=execution_status,
        failure_class=failure_class,
        failure_reason=failure_reason,
        exclusion_reason=None,
        exclusion_rule_id=None,
        detector_trace=detector,
        branch_score_trace=branch_scores,
        geometry_trace=geometry,
        decision_trace=decision,
    )


def _failure_record(
    context: InternalRunnerContext,
    entry: InternalCaseManifestEntry,
    *,
    record_id: str,
    sequence_index: int,
    attempt_index: int,
    retry_parent: str | None,
    execution_status: str,
    failure_class: str,
    failure_reason: str,
) -> InternalValidationRecord:
    return _base_record(
        context,
        entry,
        record_id=record_id,
        sequence_index=sequence_index,
        attempt_index=attempt_index,
        retry_parent=retry_parent,
        execution_status=execution_status,
        failure_class=failure_class,
        failure_reason=failure_reason,
        exclusion_reason=None,
        exclusion_rule_id=None,
        detector_trace=_empty_detector_trace(entry.execution_expectation),
        branch_score_trace=BranchScoreTrace(None, None, None),
        geometry_trace=_untriggered_geometry(
            entry.execution_expectation
        ),
        decision_trace=DecisionTrace(execution_status, None, failure_reason),
    )


def _base_record(
    context: InternalRunnerContext,
    entry: InternalCaseManifestEntry,
    *,
    record_id: str,
    sequence_index: int,
    attempt_index: int,
    retry_parent: str | None,
    execution_status: str,
    failure_class: str | None,
    failure_reason: str | None,
    exclusion_reason: str | None,
    exclusion_rule_id: str | None,
    detector_trace: DetectorTrace,
    branch_score_trace: BranchScoreTrace,
    geometry_trace: GeometryTrace,
    decision_trace: DecisionTrace,
) -> InternalValidationRecord:
    return InternalValidationRecord(
        record_id=record_id,
        run_id=context.bindings.run_id,
        protocol_id=context.protocol.protocol_id,
        protocol_version=context.protocol.protocol_version,
        record_schema_version=context.protocol.record_schema_version,
        analysis_unit_identity=entry.analysis_unit_identity,
        split=entry.split,
        record_sequence_index=sequence_index,
        record_attempt_index=attempt_index,
        execution_status=execution_status,
        failure_class=failure_class,
        failure_reason=failure_reason,
        exclusion_reason=exclusion_reason,
        exclusion_rule_id=exclusion_rule_id,
        retry_of_record_id=retry_parent,
        detector_trace=detector_trace,
        branch_score_trace=branch_score_trace,
        routing_trace=entry.routing_trace,
        geometry_trace=geometry_trace,
        threshold_trace=ThresholdTrace(
            raw_threshold_identity=(
                entry.execution_expectation.raw_threshold_identity
            ),
            rectified_threshold_identity=(
                entry.execution_expectation.rectified_threshold_identity
            ),
            tau=entry.execution_expectation.tau,
            tau_rescue=entry.execution_expectation.tau_rescue,
        ),
        key_control_trace=entry.key_control_trace,
        decision_trace=decision_trace,
        provenance_trace=ProvenanceTrace(
            protocol_digest=context.protocol.digest(),
            split_manifest_digest=context.split_manifest.digest(),
            input_manifest_digest=context.input_manifest.digest(),
            method_code_revision=context.bindings.method_code_revision,
            candidate_config_digest=context.bindings.candidate_config_digest,
            method_config_digest=context.bindings.method_config_digest,
            execution_config_digest=context.bindings.execution_config_digest,
            model_revision=context.bindings.model_revision,
            environment_digest=context.bindings.environment_digest,
            resource_identity_digest=context.bindings.resource_identity_digest,
            input_artifact_digest=entry.input_artifact_digest,
            attack_config_digest=entry.attack_config_digest,
            metric_set_digest=entry.metric_set_digest,
        ),
    )


def _empty_detector_trace(
    expectation: FrozenCaseExecutionExpectation,
) -> DetectorTrace:
    return DetectorTrace(
        raw_detector_identity=expectation.raw_detector_identity,
        rectified_detector_identity=expectation.rectified_detector_identity,
        raw_detector_config_digest=expectation.raw_detector_config_digest,
        rectified_detector_config_digest=(
            expectation.rectified_detector_config_digest
        ),
        raw_preprocessing_identity=expectation.raw_preprocessing_identity,
        rectified_preprocessing_identity=(
            expectation.rectified_preprocessing_identity
        ),
        raw_content_score=None,
        rectified_content_score=None,
    )


def _untriggered_geometry(
    expectation: FrozenCaseExecutionExpectation,
) -> GeometryTrace:
    return GeometryTrace(
        geometry_triggered=False,
        geometry_operation_identity=expectation.geometry_operation_identity,
        geometry_reliability_config_digest=(
            expectation.geometry_reliability_config_digest
        ),
        geometry_estimation_identity=None,
        geometry_reliability_identity=None,
        geometry_reliable=None,
        geometry_transform=None,
        geometry_raw_metrics=None,
        geometry_failure_reason=None,
        rectification_status="not_attempted",
    )


def _geometry_trace(
    result: ConditionalRecoveryResult,
    expectation: FrozenCaseExecutionExpectation,
) -> GeometryTrace:
    estimation = result.geometry_estimation
    reliability = result.geometry_reliability_result
    rectification = result.image_rectification_result
    transform = None
    raw_metrics = None
    if estimation is not None:
        transform = {
            "residual_rotation_degrees": float(
                estimation.transform.residual_rotation_degrees
            ),
            "log_scale": float(estimation.transform.log_scale),
            "translation_x": float(estimation.transform.translation_x),
            "translation_y": float(estimation.transform.translation_y),
        }
        raw_metrics = {
            "registered_objective": float(estimation.registered_objective),
            "second_registered_objective": float(
                estimation.second_registered_objective
            ),
            "exact_identity_objective": float(
                estimation.exact_identity_objective
            ),
            "canonical_score": float(estimation.canonical_score),
            "observation_score": float(estimation.observation_score),
            "coverage": float(estimation.coverage),
            "uniqueness": float(estimation.uniqueness),
            "gap": float(estimation.gap),
            "identity_margin": float(estimation.identity_margin),
            "key_margin": float(estimation.key_margin),
            "inlier_ratio": float(estimation.inlier_ratio),
            "mean_residual": float(estimation.mean_residual),
        }
        if any(not isfinite(value) for value in (*transform.values(), *raw_metrics.values())):
            raise InternalRunnerError("geometry trace contains non-finite values")
    rectification_status = "not_attempted"
    if rectification is not None:
        rectification_status = "succeeded"
    elif result.status == "negative_rectification_failure":
        rectification_status = "failed"
    return GeometryTrace(
        geometry_triggered=result.geometry_triggered,
        geometry_operation_identity=expectation.geometry_operation_identity,
        geometry_reliability_config_digest=(
            expectation.geometry_reliability_config_digest
        ),
        geometry_estimation_identity=(
            None if estimation is None else estimation.estimation_identity_digest
        ),
        geometry_reliability_identity=(
            None if reliability is None else reliability.reliability_identity_digest
        ),
        geometry_reliable=None if reliability is None else reliability.reliable,
        geometry_transform=transform,
        geometry_raw_metrics=raw_metrics,
        geometry_failure_reason=result.failure_reason,
        rectification_status=rectification_status,
    )


def _joint_result_failure_class(
    result: ConditionalRecoveryResult,
) -> str | None:
    if result.failure_reason is None:
        return None
    if result.status in SCIENTIFIC_RESULT_FAILURE_STATUSES:
        return "scientific_failure"
    return "execution_failure"


def _unit_records(
    collection: RunCaseRecordCollection | None,
    identity: AnalysisUnitIdentity,
) -> tuple[InternalValidationRecord, ...]:
    if collection is None:
        return ()
    return tuple(
        sorted(
            (
                record
                for record in collection.records
                if record.analysis_unit_identity == identity
            ),
            key=lambda record: record.record_attempt_index,
        )
    )


def _record_is_complete(record: InternalValidationRecord) -> bool:
    return (
        record.execution_status in {"success", "excluded"}
        or (
            record.execution_status == "failed"
            and record.failure_class
            in {"execution_failure", "scientific_failure"}
        )
    )


def _exception_identity(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
