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
    FrozenCaseInputManifest,
    FrozenRecordBindings,
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
    record_excluded_case,
    replay_internal_record_collection,
)
from main import (
    ContentDetectorBinding,
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


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT / "configs/experiments/internal_scientific_validation_protocol.json"
)
COMPONENT_PATH = ROOT / "configs/experiments/internal_execution_components.json"
SYNTHETIC_MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
# Fixture identity only; this is deliberately not a CEG-WM repository revision.
SYNTHETIC_METHOD_CODE_REVISION = "83e4d31b0fae9e91c35db600cd97b9ae1d5f3054"
ROOT_KEY = "governed-runner-cpu-key"


class _PublicImageContentOperation:
    def __init__(self) -> None:
        self.calls = 0

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
    def __call__(self, _image: torch.Tensor, _registered_key: str):
        raise AssertionError("geometry must not be called")


class _ResourceFailingGeometryOperation:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _image: torch.Tensor, _registered_key: str):
        self.calls += 1
        raise ResourceExecutionError("synthetic device unavailable")


class _UnexpectedFailingGeometryOperation:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _image: torch.Tensor, _registered_key: str):
        self.calls += 1
        raise ValueError("synthetic method operation failed")


class _ExplicitScientificFailingGeometryOperation:
    def __init__(self) -> None:
        self.calls = 0

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
    operation = _PublicImageContentOperation()
    binding, raw_score = _binding(operation, source.image)
    thresholds = JointDecisionThresholds(
        tau=raw_score + (0.1 if force_rescue else -0.1),
        tau_rescue=raw_score - (0.1 if force_rescue else 0.2),
        detector_binding_digest=binding.detector_binding_digest,
        calibration_identity="cpu_synthetic_runner_thresholds",
    )
    key_digest = adapter.identify_key(ROOT_KEY).result.root_key_public_digest
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
    )
    input_manifest = FrozenCaseInputManifest(
        manifest_schema_version="ceg_wm_internal_case_input_manifest_v1",
        manifest_id="runner_input_manifest",
        manifest_revision="runner_input_revision",
        protocol_digest=protocol.digest(),
        split_manifest_digest=split_manifest.digest(),
        entries=(entry,),
    )
    payload = InternalCaseExecutionPayload(
        source_artifact=source,
        attack_specification=attack,
        detection_key=ROOT_KEY,
        content_detector_binding=binding,
        thresholds=thresholds,
        geometry_estimation_operation=(
            geometry_operation
            if geometry_operation is not None
            else _UnexpectedGeometryOperation()
        ),
        geometry_operation_identity="cpu_synthetic_geometry_operation",
        geometry_reliability_thresholds=None,
    )
    bindings = FrozenRecordBindings(
        run_id="runner_run",
        case_id=unit.case_id,
        input_manifest_digest=input_manifest.digest(),
        method_code_revision=SYNTHETIC_METHOD_CODE_REVISION,
        candidate_config_digest=candidate_config_digest(
            adapter=adapter,
            input_manifest=input_manifest,
            payload=payload,
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
    assert context.writer.path.read_bytes().endswith(b"\n")


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

    assert first.record.to_dict() == second.record.to_dict()
    assert first_context.writer.path.read_bytes() == (
        second_context.writer.path.read_bytes()
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
def test_candidate_digest_binds_stable_routing_policy_only(tmp_path: Path) -> None:
    context, payload, _operation = _context(tmp_path)
    baseline = candidate_config_digest(
        adapter=context.adapter,
        input_manifest=context.input_manifest,
        payload=payload,
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

    assert candidate_config_digest(
        adapter=context.adapter,
        input_manifest=changed_candidate_manifest,
        payload=payload,
    ) != baseline
    assert candidate_config_digest(
        adapter=context.adapter,
        input_manifest=changed_random_observation_manifest,
        payload=payload,
    ) == baseline


@pytest.mark.parametrize(
    ("target_name", "attribute", "mutated_value"),
    (
        ("bindings", "run_id", "mutated_runner_run"),
        ("bindings", "input_manifest_digest", "f" * 64),
        ("protocol", "protocol_kind", "mutated_protocol"),
        ("split_manifest", "manifest_revision", "mutated_split_revision"),
        ("input_manifest", "manifest_revision", "mutated_input_revision"),
        ("attack_registry", "image_padding", "border"),
        ("attack_registry", "registry_digest", "f" * 64),
        ("metric_registry", "analysis_unit", "mutated_analysis_unit"),
        ("metric_registry", "registry_digest", "e" * 64),
    ),
)
@pytest.mark.quick
def test_post_construction_anchor_drift_fails_before_execution_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    attribute: str,
    mutated_value: str,
) -> None:
    context, payload, operation = _context(tmp_path)
    target = getattr(context, target_name)
    object.__setattr__(target, attribute, mutated_value)
    attack_calls = 0

    def forbidden_attack(*_args, **_kwargs):
        nonlocal attack_calls
        attack_calls += 1
        raise AssertionError("attack must not execute after anchor drift")

    monkeypatch.setattr(
        "experiments.runners.internal.apply_geometric_attack",
        forbidden_attack,
    )
    with pytest.raises(
        (GovernedRecordWriterError, InternalRunnerError),
        match="drift|invalid",
    ):
        execute_internal_case(
            context,
            unit_id=payload.source_artifact.analysis_unit_identity.unit_id,
            payload=payload,
        )
    assert attack_calls == 0
    assert operation.calls == 0
    assert not context.writer.path.exists()


@pytest.mark.quick
def test_writer_rechecks_its_private_construction_snapshot(tmp_path: Path) -> None:
    context, _payload, _operation = _context(tmp_path)
    object.__setattr__(context.writer._bindings, "run_id", "mutated_writer_run")

    with pytest.raises(GovernedRecordWriterError, match="anchor drift"):
        context.writer.load()


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
