"""CPU/synthetic checks for governed composition, records, resume, and replay."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from experiments.attacks import (
    AttackArtifact,
    GeometricAttackError,
    GeometricAttackSpec,
    apply_geometric_attack,
    load_attack_registry,
)
from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics import load_metric_registry
from experiments.protocol.internal_records import (
    KeyControlTrace,
    RoutingTrace,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
    INTERNAL_VALIDATION_SPLITS,
    SplitAssignment,
    derive_source_cluster_id,
)
from experiments.protocol.internal_validation import (
    load_frozen_internal_validation_protocol,
)
from experiments.runners import (
    DEVELOPMENT_ONLY_RECORD_SCOPE,
    FrozenCaseExecutionExpectation,
    FrozenCaseInputManifest,
    FrozenRecordBindings,
    FormalRuntimeGeometryEstimationOperation,
    GovernedRecordWriter,
    GovernedRecordWriterError,
    InternalCaseExecutionPayload,
    InternalCaseManifestEntry,
    InternalRunnerContext,
    InternalRunnerError,
    ResourceExecutionError,
    candidate_config_digest,
    execute_internal_case,
    execution_config_digest,
    formal_operation_config_digest,
    geometry_reliability_config_digest,
    record_excluded_case,
    replay_internal_record_collection,
)
from main import (
    ContentDetectorBinding,
    GeometryReliabilityThresholds,
    HfDetectionObservation,
    JointDecisionThresholds,
    content_detector,
    hf_detector,
)
from main.content_chain import (
    ContentDetectorError,
    validate_content_detection_result,
)
from main.geometry_chain import GeometricTransformEstimatorError
from main.shared import rgb8_image_digest
from runtime import create_runtime_adapter
from tests.unit.test_runtime_qk_observation import FakeQkBackend


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT / "configs/experiments/internal_scientific_validation_protocol.json"
)
COMPONENT_PATH = ROOT / "configs/experiments/internal_execution_components.json"
SYNTHETIC_MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
# Fixture identity only; this is deliberately not a CEG-WM repository revision.
SYNTHETIC_METHOD_CODE_REVISION = "83e4d31b0fae9e91c35db600cd97b9ae1d5f3054"
ROOT_KEY = "governed-runner-cpu-key"


def _formal_ready_geometry():
    backend = FakeQkBackend()
    runtime_adapter = create_runtime_adapter(backend)
    runtime_adapter.initialize("cpu")
    operation = FormalRuntimeGeometryEstimationOperation(
        runtime_adapter=runtime_adapter,
        adapter_configuration=(
            load_ceg_wm_experiment_adapter_configuration(COMPONENT_PATH)
        ),
        epsilon_inlier=0.8,
        execution_scope="cpu_synthetic_wiring_only",
    )
    return backend, runtime_adapter, operation


class _PublicImageContentOperation:
    def __init__(self) -> None:
        self.calls = 0
        self.behavior_mode = "hf_only_public_image_detection"

    def formal_runner_semantic_declaration(self):
        return {
            "behavior_mode": self.behavior_mode,
            "image_encoding": "rgb8_public_image_float32_unit_interval",
            "semantic_version": "cpu_synthetic_content_operation_v1",
        }

    def _detect(self, image: torch.Tensor, detection_key: str):
        observation = HfDetectionObservation.from_public_image_encoding(
            tuple((image.to(dtype=torch.float32) / 255.0).reshape(-1).tolist()),
            tuple(image.shape),
        )
        return replace(
            content_detector(hf_detector(observation, detection_key)),
            content_input_image_digest=rgb8_image_digest(image),
            content_replay_operation=self,
        )

    def __call__(self, image: torch.Tensor, detection_key: str):
        self.calls += 1
        if self.behavior_mode != "hf_only_public_image_detection":
            raise ValueError("content operation behavior mode drifted")
        return self._detect(image, detection_key)

    def replay_validate_content_result(
        self,
        result,
        input_image: object,
        detection_key: str,
    ):
        if not isinstance(input_image, torch.Tensor):
            raise ContentDetectorError("replay input must be an RGB8 tensor")
        expected = self._detect(input_image, detection_key)
        validate_content_detection_result(expected)
        if result != expected:
            raise ContentDetectorError("content replay mismatch")
        return result


class _UnexpectedGeometryOperation:
    def __init__(self) -> None:
        self.calls = 0
        self.mode = "must_not_execute"

    def formal_runner_semantic_declaration(self):
        return {
            "mode": self.mode,
            "semantic_version": "cpu_synthetic_geometry_operation_v1",
        }

    def __call__(self, _image: torch.Tensor, _registered_key: str):
        self.calls += 1
        raise AssertionError("geometry must not be called")


class _UndeclaredGeometryOperation:
    def __call__(self, _image: torch.Tensor, _registered_key: str):
        raise AssertionError("undeclared geometry must not be called")


class _UndeclaredPublicImageContentOperation(
    _PublicImageContentOperation
):
    formal_runner_semantic_declaration = None


class _ResourceFailingGeometryOperation:
    def __init__(self) -> None:
        self.calls = 0
        self.mode = "resource_failure"

    def formal_runner_semantic_declaration(self):
        return {
            "mode": self.mode,
            "semantic_version": "cpu_synthetic_geometry_operation_v1",
        }

    def __call__(self, _image: torch.Tensor, _registered_key: str):
        self.calls += 1
        raise ResourceExecutionError("synthetic device unavailable")


class _UnexpectedFailingGeometryOperation:
    def __init__(self) -> None:
        self.calls = 0
        self.mode = "unexpected_failure"

    def formal_runner_semantic_declaration(self):
        return {
            "mode": self.mode,
            "semantic_version": "cpu_synthetic_geometry_operation_v1",
        }

    def __call__(self, _image: torch.Tensor, _registered_key: str):
        self.calls += 1
        raise ValueError("synthetic method operation failed")


class _ExplicitScientificFailingGeometryOperation:
    def __init__(self) -> None:
        self.calls = 0
        self.mode = "scientific_failure"

    def formal_runner_semantic_declaration(self):
        return {
            "mode": self.mode,
            "semantic_version": "cpu_synthetic_geometry_operation_v1",
        }

    def __call__(self, _image: torch.Tensor, _registered_key: str):
        self.calls += 1
        raise GeometricTransformEstimatorError(
            "synthetic registered geometry estimation failure"
        )


def _unit(index: int, *, case_id: str = "runner_case") -> AnalysisUnitIdentity:
    prompt_digest = f"{index + 1:064x}"
    lineage_digest = f"{index + 101:064x}"
    key_family_digest = f"{index + 201:064x}"
    return AnalysisUnitIdentity(
        unit_id=f"runner_unit_{index}",
        case_id=case_id,
        source_cluster_id=derive_source_cluster_id(
            prompt_digest=prompt_digest,
            generation_seed=index,
            image_lineage_digest=lineage_digest,
            registered_key_family_digest=key_family_digest,
        ),
        prompt_digest=prompt_digest,
        generation_seed=index,
        image_lineage_digest=lineage_digest,
        registered_key_family_digest=key_family_digest,
    )


def _split_manifest(
    primary: AnalysisUnitIdentity,
    *,
    primary_split: str,
) -> FrozenSplitManifest:
    assignments = [SplitAssignment(primary, primary_split)]
    for index, split in enumerate(INTERNAL_VALIDATION_SPLITS):
        if split == primary_split:
            continue
        assignments.append(SplitAssignment(_unit(index + 20), split))
    return FrozenSplitManifest(
        protocol_id=INTERNAL_VALIDATION_PROTOCOL_ID,
        protocol_version=INTERNAL_VALIDATION_PROTOCOL_VERSION,
        manifest_id="runner_split_manifest",
        manifest_revision="runner_manifest_revision",
        assignments=tuple(assignments),
    )


def _image() -> torch.Tensor:
    generator = torch.Generator().manual_seed(91)
    return torch.randint(
        0,
        256,
        (1, 3, 9, 9),
        dtype=torch.uint8,
        generator=generator,
    )


def _reliability_thresholds() -> GeometryReliabilityThresholds:
    return GeometryReliabilityThresholds(
        gamma_coverage=0.45,
        gamma_uniqueness=0.0,
        gamma_gap=0.0,
        gamma_key=0.0,
        gamma_inlier=0.0,
        gamma_residual=1.0,
        gamma_identity=0.0,
        epsilon_inlier=0.8,
        fit_identity="runner_geometry_reliability_fit",
    )


@pytest.mark.quick
def test_experiments_reliability_config_digest_binds_declared_fields() -> None:
    thresholds = _reliability_thresholds()
    baseline = geometry_reliability_config_digest(thresholds)

    assert len(baseline) == 64
    assert baseline == geometry_reliability_config_digest(thresholds)
    assert geometry_reliability_config_digest(
        replace(thresholds, gamma_gap=0.125)
    ) != baseline
    with pytest.raises(InternalRunnerError, match="declaration invalid"):
        geometry_reliability_config_digest(
            replace(thresholds, gamma_gap=float("nan"))
        )


@pytest.mark.parametrize(
    "invalid_operation_kind",
    ("content_missing_declaration", "geometry_missing_declaration"),
)
@pytest.mark.quick
def test_payload_rejects_missing_formal_callable_declarations(
    tmp_path: Path,
    invalid_operation_kind: str,
) -> None:
    arguments = {}
    if invalid_operation_kind == "content_missing_declaration":
        arguments["content_operation"] = (
            _UndeclaredPublicImageContentOperation()
        )
    else:
        arguments["geometry_operation"] = _UndeclaredGeometryOperation()

    with pytest.raises(InternalRunnerError, match="semantic declaration"):
        _context(tmp_path, **arguments)
    assert not tuple(tmp_path.rglob("*.json"))


def _rewrite_record_document(
    path: Path,
    mutation,
) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _binding(
    operation: _PublicImageContentOperation,
    image: torch.Tensor,
) -> tuple[ContentDetectorBinding, float]:
    prototype = operation(image, ROOT_KEY)
    operation.calls = 0
    return (
        ContentDetectorBinding(
            content_detection_operation=operation,
            detector_identity=prototype.detector_identity,
            content_config_digest=prototype.content_config_digest,
            hf_detector_identity=prototype.hf_result.detector_identity,
            hf_detector_config_digest=prototype.hf_result.detector_config_digest,
            hf_template_digest=prototype.hf_result.template_digest,
            preprocessing_identity="cpu_synthetic_public_image_encoding",
            formal_mode=prototype.formal_mode,
            root_key_public_digest=prototype.hf_result.root_key_public_digest,
            key_role=prototype.hf_result.key_role,
            wrong_key_index=prototype.hf_result.wrong_key_index,
        ),
        prototype.content_score,
    )


def _context(
    tmp_path: Path,
    *,
    split: str = "development",
    geometry_operation=None,
    force_rescue: bool = False,
    attack_specification: GeometricAttackSpec | None = None,
    geometry_reliability_thresholds: GeometryReliabilityThresholds | None = None,
    content_operation: _PublicImageContentOperation | None = None,
) -> tuple[InternalRunnerContext, InternalCaseExecutionPayload, _PublicImageContentOperation]:
    protocol = load_frozen_internal_validation_protocol(PROTOCOL_PATH)
    unit = _unit(0)
    split_manifest = _split_manifest(unit, primary_split=split)
    source = AttackArtifact(unit, _image())
    attack = (
        attack_specification
        if attack_specification is not None
        else GeometricAttackSpec("identity")
    )
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENT_PATH)
    )
    attack_registry = load_attack_registry(COMPONENT_PATH)
    metric_registry = load_metric_registry(COMPONENT_PATH)
    operation = (
        content_operation
        if content_operation is not None
        else _PublicImageContentOperation()
    )
    binding, raw_score = _binding(operation, source.image)
    thresholds = JointDecisionThresholds(
        tau=raw_score + (0.1 if force_rescue else -0.1),
        tau_rescue=raw_score - (0.1 if force_rescue else 0.2),
        detector_binding_digest=binding.detector_binding_digest,
        calibration_identity="cpu_synthetic_runner_thresholds",
    )
    key_digest = adapter.identify_key(ROOT_KEY).result.root_key_public_digest
    geometry = (
        geometry_operation
        if geometry_operation is not None
        else _UnexpectedGeometryOperation()
    )
    payload = InternalCaseExecutionPayload(
        source_artifact=source,
        attack_specification=attack,
        detection_key=ROOT_KEY,
        content_detector_binding=binding,
        thresholds=thresholds,
        geometry_estimation_operation=geometry,
        geometry_operation_identity="cpu_synthetic_geometry_operation",
        geometry_reliability_thresholds=geometry_reliability_thresholds,
    )
    execution_expectation = FrozenCaseExecutionExpectation(
        content_detector_binding_digest=binding.detector_binding_digest,
        content_operation_config_digest=formal_operation_config_digest(
            binding.content_detection_operation,
            operation_role="content_detection",
        ),
        raw_detector_identity=binding.detector_identity,
        rectified_detector_identity=binding.detector_identity,
        raw_detector_config_digest=binding.content_config_digest,
        rectified_detector_config_digest=binding.content_config_digest,
        raw_preprocessing_identity=binding.preprocessing_identity,
        rectified_preprocessing_identity=binding.preprocessing_identity,
        raw_threshold_identity=thresholds.threshold_identity,
        rectified_threshold_identity=thresholds.threshold_identity,
        calibration_identity=thresholds.calibration_identity,
        tau=thresholds.tau,
        tau_rescue=thresholds.tau_rescue,
        geometry_operation_identity=payload.geometry_operation_identity,
        geometry_operation_config_digest=formal_operation_config_digest(
            payload.geometry_estimation_operation,
            operation_role="geometry_estimation",
        ),
        geometry_reliability_config_digest=(
            None
            if geometry_reliability_thresholds is None
            else geometry_reliability_config_digest(
                geometry_reliability_thresholds
            )
        ),
    )
    entry = InternalCaseManifestEntry(
        analysis_unit_identity=unit,
        split=split,
        input_artifact_digest=source.image_digest,
        attack_config_digest=attack.attack_config_digest,
        metric_set_digest=metric_registry.registry_digest,
        routing_trace=RoutingTrace(
            routing_identity="routing_uniform_control",
            routing_control="uniform_disabled",
            routing_observation_digest="1" * 64,
            routing_mask_digest="2" * 64,
        ),
        key_control_trace=KeyControlTrace(
            registered_key_public_digest=key_digest,
            detection_key_public_digest=key_digest,
            key_role="registered",
            control_identity="registered_key_control",
        ),
        execution_expectation=execution_expectation,
    )
    input_manifest = FrozenCaseInputManifest(
        manifest_schema_version="ceg_wm_internal_case_input_manifest_v3",
        manifest_id="runner_input_manifest",
        manifest_revision="runner_input_revision",
        protocol_digest=protocol.digest(),
        split_manifest_digest=split_manifest.digest(),
        entries=(entry,),
    )
    bindings = FrozenRecordBindings(
        run_id="runner_run",
        case_id=unit.case_id,
        input_manifest_digest=input_manifest.digest(),
        method_code_revision=SYNTHETIC_METHOD_CODE_REVISION,
        candidate_config_digest=candidate_config_digest(
            adapter=adapter,
            input_manifest=input_manifest,
            method_code_revision=SYNTHETIC_METHOD_CODE_REVISION,
        ),
        method_config_digest=binding.content_config_digest,
        execution_config_digest=execution_config_digest(
            protocol=protocol,
            adapter=adapter,
            attack_registry=attack_registry,
            metric_registry=metric_registry,
        ),
        model_revision=SYNTHETIC_MODEL_REVISION,
        environment_digest="3" * 64,
        resource_identity_digest="4" * 64,
    )
    writer = GovernedRecordWriter(
        records_root=tmp_path.resolve(),
        frozen_protocol=protocol,
        split_manifest=split_manifest,
        input_manifest=input_manifest,
        bindings=bindings,
    )
    return (
        InternalRunnerContext(
            protocol=protocol,
            split_manifest=split_manifest,
            input_manifest=input_manifest,
            adapter=adapter,
            attack_registry=attack_registry,
            metric_registry=metric_registry,
            writer=writer,
            bindings=bindings,
        ),
        payload,
        operation,
    )


@pytest.mark.quick
def test_runner_composes_real_adapter_attack_and_metric_replay_once(tmp_path: Path) -> None:
    context, payload, operation = _context(tmp_path)
    expectation = context.input_manifest.entries[0].execution_expectation

    assert expectation.content_operation_config_digest == (
        formal_operation_config_digest(
            payload.content_detector_binding.content_detection_operation,
            operation_role="content_detection",
        )
    )
    assert expectation.geometry_operation_config_digest == (
        formal_operation_config_digest(
            payload.geometry_estimation_operation,
            operation_role="geometry_estimation",
        )
    )

    first = execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )
    resumed = execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )
    replay = replay_internal_record_collection(context)

    assert first.record.execution_status == "success"
    assert first.record.branch_score_trace.hf_score is not None
    assert first.record.branch_score_trace.lf_score is None
    assert first.record.provenance_trace.input_manifest_digest == (
        context.input_manifest.digest()
    )
    assert operation.calls == 1  # one joint call; resume/replay add no method call
    assert resumed.resumed_without_execution
    assert len(resumed.collection.records) == 1
    assert replay.success_count == 1
    assert replay.record_count == 1
    assert replay.metric_case_count == 0
    assert replay.metric_case_results == ()
    assert replay.metric_aggregate_values is None
    assert len(replay.metric_observation_digest) == 64
    assert context.writer.path.read_bytes().endswith(b"\n")


@pytest.mark.quick
def test_development_only_scope_executes_one_registered_development_assignment(
    tmp_path: Path,
) -> None:
    formal_context, payload, operation = _context(tmp_path / "formal_source")
    entry = formal_context.input_manifest.entries[0]
    assignment = next(
        item
        for item in formal_context.split_manifest.assignments
        if item.identity == entry.analysis_unit_identity
    )
    split_manifest = replace(
        formal_context.split_manifest,
        assignments=(assignment,),
    )
    input_manifest = replace(
        formal_context.input_manifest,
        split_manifest_digest=split_manifest.digest(),
    )
    bindings = replace(
        formal_context.bindings,
        input_manifest_digest=input_manifest.digest(),
        candidate_config_digest=candidate_config_digest(
            adapter=formal_context.adapter,
            input_manifest=input_manifest,
            method_code_revision=(
                formal_context.bindings.method_code_revision
            ),
        ),
    )
    with pytest.raises(
        GovernedRecordWriterError,
        match="split manifest invalid",
    ):
        GovernedRecordWriter(
            records_root=(tmp_path / "formal_partial").resolve(),
            frozen_protocol=formal_context.protocol,
            split_manifest=split_manifest,
            input_manifest=input_manifest,
            bindings=bindings,
        )
    writer = GovernedRecordWriter(
        records_root=(tmp_path / "development_only").resolve(),
        frozen_protocol=formal_context.protocol,
        split_manifest=split_manifest,
        input_manifest=input_manifest,
        bindings=bindings,
        record_scope=DEVELOPMENT_ONLY_RECORD_SCOPE,
    )
    context = InternalRunnerContext(
        protocol=formal_context.protocol,
        split_manifest=split_manifest,
        input_manifest=input_manifest,
        adapter=formal_context.adapter,
        attack_registry=formal_context.attack_registry,
        metric_registry=formal_context.metric_registry,
        writer=writer,
        bindings=bindings,
        record_scope=DEVELOPMENT_ONLY_RECORD_SCOPE,
    )

    executed = execute_internal_case(
        context,
        unit_id=entry.analysis_unit_identity.unit_id,
        payload=payload,
    )

    assert executed.record.split == "development"
    assert executed.record.execution_status == "success"
    assert context.writer.load() == executed.collection
    assert operation.calls == 1


@pytest.mark.quick
def test_development_only_scope_rejects_future_split_assignments(
    tmp_path: Path,
) -> None:
    formal_context, _payload, _operation = _context(tmp_path / "formal_source")
    entry = formal_context.input_manifest.entries[0]
    assignment = next(
        item
        for item in formal_context.split_manifest.assignments
        if item.identity == entry.analysis_unit_identity
    )
    future_assignment = replace(assignment, split="candidate_selection")
    split_manifest = replace(
        formal_context.split_manifest,
        assignments=(future_assignment,),
    )
    input_manifest = replace(
        formal_context.input_manifest,
        split_manifest_digest=split_manifest.digest(),
        entries=(replace(entry, split="candidate_selection"),),
    )
    bindings = replace(
        formal_context.bindings,
        input_manifest_digest=input_manifest.digest(),
        candidate_config_digest=candidate_config_digest(
            adapter=formal_context.adapter,
            input_manifest=input_manifest,
            method_code_revision=(
                formal_context.bindings.method_code_revision
            ),
        ),
    )

    with pytest.raises(
        GovernedRecordWriterError,
        match="requires only development assignments",
    ):
        GovernedRecordWriter(
            records_root=(tmp_path / "future_split").resolve(),
            frozen_protocol=formal_context.protocol,
            split_manifest=split_manifest,
            input_manifest=input_manifest,
            bindings=bindings,
            record_scope=DEVELOPMENT_ONLY_RECORD_SCOPE,
        )


@pytest.mark.quick
def test_identical_frozen_inputs_produce_identical_record_bytes(
    tmp_path: Path,
) -> None:
    first_context, first_payload, _first_operation = _context(tmp_path / "first")
    second_context, second_payload, _second_operation = _context(tmp_path / "second")

    first = execute_internal_case(
        first_context,
        unit_id=first_payload.source_artifact.analysis_unit_identity.unit_id,
        payload=first_payload,
    )
    second = execute_internal_case(
        second_context,
        unit_id=second_payload.source_artifact.analysis_unit_identity.unit_id,
        payload=second_payload,
    )
    first_replay = replay_internal_record_collection(first_context)
    second_replay = replay_internal_record_collection(second_context)

    assert first.record.to_dict() == second.record.to_dict()
    assert first_context.writer.path.read_bytes() == (
        second_context.writer.path.read_bytes()
    )
    assert first_replay.metric_case_count == second_replay.metric_case_count == 0
    assert first_replay.metric_observation_digest == (
        second_replay.metric_observation_digest
    )


@pytest.mark.quick
def test_preregistered_exclusion_is_persisted_without_method_execution(
    tmp_path: Path,
) -> None:
    context, payload, operation = _context(tmp_path)

    excluded = record_excluded_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
        exclusion_rule_id="predeclared_input_integrity_rule",
        exclusion_reason="input_invalid_before_method_execution",
    )

    assert excluded.record.execution_status == "excluded"
    assert excluded.record.failure_class is None
    assert operation.calls == 0
    assert replay_internal_record_collection(context).excluded_count == 1


@pytest.mark.quick
def test_resource_failures_form_bounded_retry_lineage(tmp_path: Path) -> None:
    geometry = _ResourceFailingGeometryOperation()
    context, payload, _operation = _context(
        tmp_path,
        geometry_operation=geometry,
        force_rescue=True,
    )
    outcomes = [
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
        for _ in range(3)
    ]

    assert [outcome.record.execution_status for outcome in outcomes] == [
        "failed",
        "retry",
        "failed",
    ]
    assert all(
        outcome.record.failure_class == "resource_failure"
        for outcome in outcomes
    )
    assert outcomes[1].record.retry_of_record_id == outcomes[0].record.record_id
    assert outcomes[2].record.retry_of_record_id == outcomes[1].record.record_id
    with pytest.raises(InternalRunnerError, match="attempts already exhausted"):
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
    replay = replay_internal_record_collection(context)
    assert replay.resource_failure_count == 3
    assert replay.retry_count == 1


@pytest.mark.quick
def test_scientific_failure_is_terminal_and_not_retried(tmp_path: Path) -> None:
    geometry = _ExplicitScientificFailingGeometryOperation()
    context, payload, _operation = _context(
        tmp_path,
        geometry_operation=geometry,
        force_rescue=True,
    )
    failed = execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )
    resumed = execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )

    assert failed.record.execution_status == "failed"
    assert failed.record.failure_class == "scientific_failure"
    assert resumed.resumed_without_execution
    assert geometry.calls == 1


@pytest.mark.quick
def test_unexpected_method_bug_is_terminal_execution_failure(
    tmp_path: Path,
) -> None:
    geometry = _UnexpectedFailingGeometryOperation()
    context, payload, _operation = _context(
        tmp_path,
        geometry_operation=geometry,
        force_rescue=True,
    )

    failed = execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )
    resumed = execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )

    assert failed.record.execution_status == "failed"
    assert failed.record.failure_class == "execution_failure"
    assert failed.record.failure_reason == "builtins.ValueError"
    assert resumed.resumed_without_execution
    assert replay_internal_record_collection(context).execution_failure_count == 1
    assert geometry.calls == 1


@pytest.mark.quick
def test_current_runner_rejects_held_out_evaluation_split(tmp_path: Path) -> None:
    context, payload, _operation = _context(
        tmp_path,
        split="held_out_evaluation",
    )
    with pytest.raises(PermissionError, match="split_access_forbidden"):
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
    assert not context.writer.path.exists()


@pytest.mark.quick
def test_writer_rejects_conflict_and_canonical_digest_drift(tmp_path: Path) -> None:
    context, payload, _operation = _context(tmp_path)
    result = execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )
    conflicting = replace(
        result.record,
        provenance_trace=replace(
            result.record.provenance_trace,
            input_artifact_digest="f" * 64,
        ),
    )
    with pytest.raises(GovernedRecordWriterError, match="identity conflict"):
        context.writer.append_record(conflicting)

    document = json.loads(context.writer.path.read_text(encoding="utf-8"))
    document["records"][0]["decision_trace"]["watermark_decision"] = "negative"
    document["records"][0]["decision_trace"]["positive_source"] = None
    context.writer.path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GovernedRecordWriterError, match="schema invalid"):
        context.writer.load()


@pytest.mark.parametrize(
    "mutation_kind",
    (
        "record_id",
        "per_unit_provenance",
        "routing_trace",
        "key_control_trace",
        "detector_trace",
        "threshold_trace",
        "geometry_declarations",
    ),
)
@pytest.mark.quick
def test_persisted_record_fields_are_checked_against_frozen_expectations(
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    context, payload, operation = _context(tmp_path)
    execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )
    calls_before_tamper = operation.calls

    def mutate(document) -> None:
        record = document["records"][0]
        if mutation_kind == "record_id":
            record["record_id"] = "f" * 64
        elif mutation_kind == "per_unit_provenance":
            provenance = record["provenance_trace"]
            provenance["input_artifact_digest"] = "d" * 64
            provenance["attack_config_digest"] = "e" * 64
            provenance["metric_set_digest"] = "f" * 64
        elif mutation_kind == "routing_trace":
            record["routing_trace"] = {
                "routing_identity": "tampered_routing_candidate",
                "routing_control": "tampered_uniform_control",
                "routing_observation_digest": "d" * 64,
                "routing_mask_digest": "e" * 64,
            }
        elif mutation_kind == "key_control_trace":
            record["key_control_trace"] = {
                "registered_key_public_digest": "d" * 64,
                "detection_key_public_digest": "d" * 64,
                "key_role": "registered",
                "control_identity": "tampered_registered_key_control",
            }
        elif mutation_kind == "detector_trace":
            detector = record["detector_trace"]
            detector["raw_detector_identity"] = "tampered_detector"
            detector["rectified_detector_identity"] = "tampered_detector"
            detector["raw_detector_config_digest"] = "d" * 64
            detector["rectified_detector_config_digest"] = "d" * 64
            detector["raw_preprocessing_identity"] = "tampered_preprocess"
            detector["rectified_preprocessing_identity"] = (
                "tampered_preprocess"
            )
        elif mutation_kind == "threshold_trace":
            threshold = record["threshold_trace"]
            threshold["raw_threshold_identity"] = "tampered_threshold"
            threshold["rectified_threshold_identity"] = "tampered_threshold"
            threshold["tau"] -= 0.01
            threshold["tau_rescue"] -= 0.01
        else:
            geometry = record["geometry_trace"]
            geometry["geometry_operation_identity"] = (
                "tampered_geometry_operation"
            )
            geometry["geometry_reliability_config_digest"] = "d" * 64

    _rewrite_record_document(context.writer.path, mutate)

    with pytest.raises(GovernedRecordWriterError, match="drift"):
        context.writer.load()
    with pytest.raises(GovernedRecordWriterError, match="drift"):
        replay_internal_record_collection(context)
    with pytest.raises(GovernedRecordWriterError, match="drift"):
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
    assert operation.calls == calls_before_tamper


@pytest.mark.quick
def test_synchronized_retry_lineage_rewrite_cannot_replace_deterministic_ids(
    tmp_path: Path,
) -> None:
    geometry = _ResourceFailingGeometryOperation()
    context, payload, operation = _context(
        tmp_path,
        geometry_operation=geometry,
        force_rescue=True,
    )
    for _ in range(3):
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
    content_calls_before_tamper = operation.calls
    geometry_calls_before_tamper = geometry.calls

    def mutate(document) -> None:
        forged_ids = ("d" * 64, "e" * 64, "f" * 64)
        for index, record in enumerate(document["records"]):
            record["record_id"] = forged_ids[index]
            record["retry_of_record_id"] = (
                None if index == 0 else forged_ids[index - 1]
            )

    _rewrite_record_document(context.writer.path, mutate)

    with pytest.raises(
        GovernedRecordWriterError,
        match="deterministic identity drifted",
    ):
        context.writer.load()
    with pytest.raises(GovernedRecordWriterError, match="identity drifted"):
        replay_internal_record_collection(context)
    with pytest.raises(GovernedRecordWriterError, match="identity drifted"):
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
    assert operation.calls == content_calls_before_tamper
    assert geometry.calls == geometry_calls_before_tamper


@pytest.mark.quick
def test_atomic_failure_preserves_prior_record_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _ResourceFailingGeometryOperation()
    context, payload, _operation = _context(
        tmp_path,
        geometry_operation=geometry,
        force_rescue=True,
    )
    execute_internal_case(
        context,
        unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
        payload=payload,
    )
    before = context.writer.path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace interruption")

    monkeypatch.setattr("experiments.runners.record_writer.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace interruption"):
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
    assert context.writer.path.read_bytes() == before


@pytest.mark.quick
def test_candidate_digest_binds_routing_method_revision_and_execution_declarations(
    tmp_path: Path,
) -> None:
    context, payload, _operation = _context(tmp_path)
    baseline = candidate_config_digest(
        adapter=context.adapter,
        input_manifest=context.input_manifest,
        method_code_revision=SYNTHETIC_METHOD_CODE_REVISION,
    )
    entry = context.input_manifest.entries[0]
    changed_candidate_manifest = replace(
        context.input_manifest,
        entries=(
            replace(
                entry,
                routing_trace=replace(
                    entry.routing_trace,
                    routing_identity="routing_adaptive_candidate",
                ),
            ),
        ),
    )
    changed_random_observation_manifest = replace(
        context.input_manifest,
        entries=(
            replace(
                entry,
                routing_trace=replace(
                    entry.routing_trace,
                    routing_observation_digest="f" * 64,
                    routing_mask_digest="e" * 64,
                ),
            ),
        ),
    )
    changed_geometry_declaration_manifest = replace(
        context.input_manifest,
        entries=(
            replace(
                entry,
                execution_expectation=replace(
                    entry.execution_expectation,
                    geometry_operation_identity=(
                        "changed_geometry_operation"
                    ),
                ),
            ),
        ),
    )

    assert candidate_config_digest(
        adapter=context.adapter,
        input_manifest=changed_candidate_manifest,
        method_code_revision=SYNTHETIC_METHOD_CODE_REVISION,
    ) != baseline
    assert candidate_config_digest(
        adapter=context.adapter,
        input_manifest=changed_random_observation_manifest,
        method_code_revision=SYNTHETIC_METHOD_CODE_REVISION,
    ) == baseline
    assert candidate_config_digest(
        adapter=context.adapter,
        input_manifest=changed_geometry_declaration_manifest,
        method_code_revision=SYNTHETIC_METHOD_CODE_REVISION,
    ) != baseline
    assert candidate_config_digest(
        adapter=context.adapter,
        input_manifest=context.input_manifest,
        method_code_revision="f" * 40,
    ) != baseline


@pytest.mark.parametrize(
    "mutation_kind",
    (
        "image_in_place",
        "artifact_digest",
        "spec_parameter",
        "spec_digest",
    ),
)
@pytest.mark.quick
def test_attack_input_post_init_mutation_fails_before_grid_method_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    specification = GeometricAttackSpec(
        "rotation",
        rotation_degrees=8.0,
    )
    context, payload, operation = _context(
        tmp_path,
        attack_specification=specification,
    )
    affine_calls = 0
    grid_calls = 0

    def forbidden_affine(*_args, **_kwargs):
        nonlocal affine_calls
        affine_calls += 1
        raise AssertionError("affine_grid must not execute after attack drift")

    def forbidden_grid(*_args, **_kwargs):
        nonlocal grid_calls
        grid_calls += 1
        raise AssertionError("grid_sample must not execute after attack drift")

    monkeypatch.setattr(
        "experiments.attacks.geometric.functional.affine_grid",
        forbidden_affine,
    )
    monkeypatch.setattr(
        "experiments.attacks.geometric.functional.grid_sample",
        forbidden_grid,
    )
    if mutation_kind == "image_in_place":
        payload.source_artifact.image[0, 0, 0, 0] ^= 1
    elif mutation_kind == "artifact_digest":
        object.__setattr__(
            payload.source_artifact,
            "image_digest",
            "f" * 64,
        )
    elif mutation_kind == "spec_parameter":
        object.__setattr__(
            payload.attack_specification,
            "rotation_degrees",
            12.0,
        )
    else:
        object.__setattr__(
            payload.attack_specification,
            "attack_config_digest",
            "e" * 64,
        )

    with pytest.raises(GeometricAttackError, match="drift|invalid"):
        apply_geometric_attack(
            payload.source_artifact,
            payload.attack_specification,
            registry=context.attack_registry,
        )
    with pytest.raises(
        (GeometricAttackError, InternalRunnerError),
        match="drift|invalid",
    ):
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
    assert affine_calls == 0
    assert grid_calls == 0
    assert operation.calls == 0
    assert not context.writer.path.exists()
