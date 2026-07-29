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
    validate_attack_registry,
)
from experiments.methods import CegWmExperimentAdapter
from experiments.metrics import (
    DetectionMetricCase,
    MetricRegistry,
    RescueSafetyCase,
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
    validate_key_control_trace,
    validate_routing_trace,
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
)
from main import (
    ConditionalRecoveryResult,
    ContentDetectorBinding,
    GeometryEstimationOperation,
    GeometryReliabilityThresholds,
    JointDecisionThresholds,
)


INPUT_MANIFEST_SCHEMA_VERSION = "ceg_wm_internal_case_input_manifest_v1"
SCIENTIFIC_RESULT_FAILURE_STATUSES = frozenset(
    {
        "negative_geometry_operation_failure",
        "negative_geometry_reliability_failure",
        "negative_rectification_failure",
        "negative_rectified_content_operation_failure",
        "raw_content_operation_failure",
    }
)


class InternalRunnerError(ValueError):
    """A frozen input, composition, resume, or replay boundary failed closed."""


class ResourceExecutionError(RuntimeError):
    """Explicit transient resource failure; never interpreted as method science."""


@dataclass(frozen=True, slots=True)
class InternalCaseManifestEntry:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    input_artifact_digest: str
    attack_config_digest: str
    metric_set_digest: str
    routing_trace: RoutingTrace
    key_control_trace: KeyControlTrace

    def validate(self) -> tuple[str, ...]:
        if type(self.analysis_unit_identity) is not AnalysisUnitIdentity:
            return ("analysis_unit_identity_exact_type_required",)
        violations = list(self.analysis_unit_identity.validate())
        for role in (
            "input_artifact_digest",
            "attack_config_digest",
            "metric_set_digest",
        ):
            if not _digest_valid(getattr(self, role)):
                violations.append(f"{role}_invalid")
        violations.extend(validate_routing_trace(self.routing_trace))
        violations.extend(validate_key_control_trace(self.key_control_trace))
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True, slots=True)
class FrozenCaseInputManifest:
    manifest_schema_version: str
    manifest_id: str
    manifest_revision: str
    protocol_digest: str
    split_manifest_digest: str
    entries: tuple[InternalCaseManifestEntry, ...]

    def digest(self) -> str:
        return _canonical_digest(asdict(self))

    def validate(
        self,
        *,
        protocol: FrozenInternalValidationProtocol,
        split_manifest: FrozenSplitManifest,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if self.manifest_schema_version != INPUT_MANIFEST_SCHEMA_VERSION:
            violations.append("input_manifest_schema_version_invalid")
        for role in ("manifest_id", "manifest_revision"):
            if type(getattr(self, role)) is not str or not getattr(self, role):
                violations.append(f"{role}_missing")
        if self.protocol_digest != protocol.digest():
            violations.append("input_manifest_protocol_digest_mismatch")
        if self.split_manifest_digest != split_manifest.digest():
            violations.append("input_manifest_split_manifest_digest_mismatch")
        if not self.entries:
            violations.append("input_manifest_entries_missing")
        assignment_pairs = {
            (assignment.identity, assignment.split)
            for assignment in split_manifest.assignments
        }
        seen_units: set[str] = set()
        for entry in self.entries:
            if type(entry) is not InternalCaseManifestEntry:
                violations.append("input_manifest_entry_exact_type_required")
                continue
            violations.extend(entry.validate())
            if (entry.analysis_unit_identity, entry.split) not in assignment_pairs:
                violations.append("input_manifest_split_assignment_missing")
            unit_id = entry.analysis_unit_identity.unit_id
            if unit_id in seen_units:
                violations.append("input_manifest_unit_duplicate")
            seen_units.add(unit_id)
        return tuple(dict.fromkeys(violations))


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


def candidate_config_digest(
    *,
    adapter: CegWmExperimentAdapter,
    input_manifest: FrozenCaseInputManifest,
    payload: InternalCaseExecutionPayload,
) -> str:
    """Bind candidate-facing method and fitted decision identities without keys."""

    if type(payload) is not InternalCaseExecutionPayload:
        raise InternalRunnerError("case execution payload exact type is required")
    if type(input_manifest) is not FrozenCaseInputManifest:
        raise InternalRunnerError("frozen input manifest exact type is required")
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
    reliability = payload.geometry_reliability_thresholds
    return _canonical_digest(
        {
            "adapter_config_digest": adapter.configuration.config_digest,
            "content_detector_binding_digest": (
                payload.content_detector_binding.detector_binding_digest
            ),
            "geometry_operation_identity": payload.geometry_operation_identity,
            "geometry_reliability_thresholds": (
                None if reliability is None else asdict(reliability)
            ),
            "routing_candidate_policy": [
                {
                    "routing_control": routing_control,
                    "routing_identity": routing_identity,
                }
                for routing_identity, routing_control in routing_candidate_policy
            ],
            "threshold_identity": payload.thresholds.threshold_identity,
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
    record_id = _record_id(
        context.bindings,
        entry.analysis_unit_identity,
        attempt_index,
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
                payload,
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
                payload,
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
            payload,
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
            payload,
            record_id=record_id,
            sequence_index=sequence_index,
            attempt_index=attempt_index,
            retry_parent=retry_parent,
            execution_status="failed",
            failure_class="execution_failure",
            failure_reason=_exception_identity(exc),
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
        payload,
        record_id=_record_id(context.bindings, entry.analysis_unit_identity, 0),
        sequence_index=0 if existing is None else len(existing.records),
        attempt_index=0,
        retry_parent=None,
        execution_status="excluded",
        failure_class=None,
        failure_reason=None,
        exclusion_reason=exclusion_reason,
        exclusion_rule_id=exclusion_rule_id,
        detector_trace=_empty_detector_trace(payload),
        branch_score_trace=BranchScoreTrace(None, None, None),
        geometry_trace=_untriggered_geometry(),
        decision_trace=DecisionTrace("excluded", None, exclusion_reason),
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
    for record in persisted.records:
        if record.execution_status != "success":
            continue
        raw_score = record.detector_trace.raw_content_score
        if raw_score is None:
            raise InternalRunnerError("successful record has no raw metric score")
        DetectionMetricCase(
            analysis_unit_identity=record.analysis_unit_identity,
            split=record.split,
            detector_identity=record.detector_trace.raw_detector_identity,
            key_role={
                "registered": "registered_positive",
                "wrong_key": "wrong_key",
                "unwatermarked_primary_null": "unwatermarked_primary_null",
            }[record.key_control_trace.key_role],
            score=raw_score,
        )
        RescueSafetyCase(
            analysis_unit_identity=record.analysis_unit_identity,
            split=record.split,
            raw_detector_identity=record.detector_trace.raw_detector_identity,
            rectified_detector_identity=(
                record.detector_trace.rectified_detector_identity
            ),
            raw_threshold_identity=record.threshold_trace.raw_threshold_identity,
            rectified_threshold_identity=(
                record.threshold_trace.rectified_threshold_identity
            ),
            raw_positive=raw_score >= record.threshold_trace.tau,
            rescue_triggered=record.geometry_trace.geometry_triggered,
            rectified_positive=(
                record.detector_trace.rectified_content_score is not None
                and record.detector_trace.rectified_content_score
                >= record.threshold_trace.tau
            ),
            watermark_decision_positive=(
                record.decision_trace.watermark_decision == "positive"
            ),
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
    if type(payload.source_artifact) is not AttackArtifact:
        raise InternalRunnerError("source artifact exact type is required")
    if payload.source_artifact.analysis_unit_identity != entry.analysis_unit_identity:
        raise InternalRunnerError("source artifact analysis identity drifted")
    if payload.source_artifact.image_digest != entry.input_artifact_digest:
        raise InternalRunnerError("source artifact digest drifted")
    if type(payload.attack_specification) is not GeometricAttackSpec:
        raise InternalRunnerError("attack specification exact type is required")
    if payload.attack_specification.attack_config_digest != entry.attack_config_digest:
        raise InternalRunnerError("attack configuration digest drifted")
    if entry.metric_set_digest != context.metric_registry.registry_digest:
        raise InternalRunnerError("metric set digest drifted")
    if type(payload.content_detector_binding) is not ContentDetectorBinding:
        raise InternalRunnerError("content detector binding exact type is required")
    if type(payload.thresholds) is not JointDecisionThresholds:
        raise InternalRunnerError("joint thresholds exact type is required")
    if payload.thresholds.detector_binding_digest != (
        payload.content_detector_binding.detector_binding_digest
    ):
        raise InternalRunnerError("threshold detector binding drifted")
    if not callable(payload.geometry_estimation_operation):
        raise InternalRunnerError("geometry estimation operation must be callable")
    if not payload.geometry_operation_identity:
        raise InternalRunnerError("geometry operation identity is required")
    if (
        payload.geometry_reliability_thresholds is not None
        and type(payload.geometry_reliability_thresholds)
        is not GeometryReliabilityThresholds
    ):
        raise InternalRunnerError("geometry reliability thresholds exact type required")
    observed_key = context.adapter.identify_key(payload.detection_key).result
    if observed_key.root_key_public_digest != (
        entry.key_control_trace.detection_key_public_digest
    ):
        raise InternalRunnerError("detection key public identity drifted")
    if candidate_config_digest(
        adapter=context.adapter,
        input_manifest=context.input_manifest,
        payload=payload,
    ) != (
        context.bindings.candidate_config_digest
    ):
        raise InternalRunnerError("candidate configuration digest drifted")
    if payload.content_detector_binding.content_config_digest != (
        context.bindings.method_config_digest
    ):
        raise InternalRunnerError("method configuration digest drifted")
    return entry


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
    payload: InternalCaseExecutionPayload,
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
    geometry = _geometry_trace(result)
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
        payload,
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
    payload: InternalCaseExecutionPayload,
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
        payload,
        record_id=record_id,
        sequence_index=sequence_index,
        attempt_index=attempt_index,
        retry_parent=retry_parent,
        execution_status=execution_status,
        failure_class=failure_class,
        failure_reason=failure_reason,
        exclusion_reason=None,
        exclusion_rule_id=None,
        detector_trace=_empty_detector_trace(payload),
        branch_score_trace=BranchScoreTrace(None, None, None),
        geometry_trace=_untriggered_geometry(),
        decision_trace=DecisionTrace(execution_status, None, failure_reason),
    )


def _base_record(
    context: InternalRunnerContext,
    entry: InternalCaseManifestEntry,
    payload: InternalCaseExecutionPayload,
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
            raw_threshold_identity=payload.thresholds.threshold_identity,
            rectified_threshold_identity=payload.thresholds.threshold_identity,
            tau=payload.thresholds.tau,
            tau_rescue=payload.thresholds.tau_rescue,
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
    payload: InternalCaseExecutionPayload,
) -> DetectorTrace:
    binding = payload.content_detector_binding
    return DetectorTrace(
        raw_detector_identity=binding.detector_identity,
        rectified_detector_identity=binding.detector_identity,
        raw_detector_config_digest=binding.content_config_digest,
        rectified_detector_config_digest=binding.content_config_digest,
        raw_preprocessing_identity=binding.preprocessing_identity,
        rectified_preprocessing_identity=binding.preprocessing_identity,
        raw_content_score=None,
        rectified_content_score=None,
    )


def _untriggered_geometry() -> GeometryTrace:
    return GeometryTrace(
        geometry_triggered=False,
        geometry_estimation_identity=None,
        geometry_reliability_identity=None,
        geometry_reliable=None,
        geometry_transform=None,
        geometry_raw_metrics=None,
        geometry_failure_reason=None,
        rectification_status="not_attempted",
    )


def _geometry_trace(result: ConditionalRecoveryResult) -> GeometryTrace:
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


def _record_id(
    bindings: FrozenRecordBindings,
    identity: AnalysisUnitIdentity,
    attempt_index: int,
) -> str:
    return _canonical_digest(
        {
            "attempt_index": attempt_index,
            "case_id": bindings.case_id,
            "input_manifest_digest": bindings.input_manifest_digest,
            "run_id": bindings.run_id,
            "source_cluster_id": identity.source_cluster_id,
            "unit_id": identity.unit_id,
        }
    )


def _exception_identity(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def _digest_valid(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
