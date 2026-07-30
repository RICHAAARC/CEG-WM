"""Package-contained CPU/synthetic wiring entrypoint for the A3a runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import exp, log, sqrt
from pathlib import Path
import re
from typing import Sequence

import torch

from experiments.attacks import (
    AttackArtifact,
    GeometricAttackSpec,
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
    FrozenCaseExecutionExpectation,
    FrozenCaseInputManifest,
    FrozenRecordBindings,
    FormalHfContentDetectionOperation,
    FormalRuntimeGeometryEstimationOperation,
    GovernedRecordWriter,
    InternalCaseExecutionPayload,
    InternalCaseManifestEntry,
    InternalRunnerContext,
    candidate_config_digest,
    create_formal_content_detector_binding,
    execute_internal_case,
    execution_config_digest,
    formal_operation_config_digest,
    geometry_reliability_config_digest,
    replay_internal_record_collection,
)
from experiments.runners.synthetic_runtime import (
    SyntheticDelegatingGeometryOperation,
    SyntheticQkBackend,
)
from main import GeometryReliabilityThresholds, JointDecisionThresholds
from runtime import create_runtime_adapter


ENTRYPOINT_IDENTITY = (
    "scripts.experiment_execution.experiment_execution_entrypoint:main"
)
ENTRYPOINT_SCHEMA_VERSION = 2
EXECUTION_SCOPE = "cpu_synthetic_wiring_only"
EVIDENCE_SCOPE = (
    "infrastructure_synthetic_wiring_not_scientific_experiment_evidence"
)
SYNTHETIC_ROOT_KEY = "ceg-wm-package-synthetic-wiring-key"
SYNTHETIC_MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
REVISION = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class ExperimentExecutionEntrypointError(ValueError):
    """The package entrypoint inputs or A3a wiring failed closed."""


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit(index: int, *, case_id: str) -> AnalysisUnitIdentity:
    prompt_digest = f"{index + 1:064x}"
    lineage_digest = f"{index + 101:064x}"
    key_family_digest = f"{index + 201:064x}"
    return AnalysisUnitIdentity(
        unit_id=f"synthetic_wiring_unit_{index}",
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
    primary: tuple[AnalysisUnitIdentity, ...],
) -> FrozenSplitManifest:
    assignments = [
        SplitAssignment(identity, "end_to_end_check")
        for identity in primary
    ]
    for index, split in enumerate(INTERNAL_VALIDATION_SPLITS):
        if split == "end_to_end_check":
            continue
        assignments.append(
            SplitAssignment(
                _unit(index + 20, case_id=f"synthetic_{split}"),
                split,
            )
        )
    return FrozenSplitManifest(
        protocol_id=INTERNAL_VALIDATION_PROTOCOL_ID,
        protocol_version=INTERNAL_VALIDATION_PROTOCOL_VERSION,
        manifest_id="synthetic_wiring_split_manifest",
        manifest_revision="synthetic_wiring_manifest_revision",
        assignments=tuple(assignments),
    )


def _synthetic_image() -> torch.Tensor:
    axis = torch.linspace(64, 254, 512, dtype=torch.float32).floor().to(
        dtype=torch.uint8
    )
    horizontal = axis.unsqueeze(0).expand(512, 512)
    vertical = axis.unsqueeze(1).expand(512, 512)
    diagonal = (
        horizontal.to(dtype=torch.int16)
        + vertical.to(dtype=torch.int16)
    ).remainder(256).to(dtype=torch.uint8)
    return torch.stack((horizontal, vertical, diagonal), dim=0).unsqueeze(0)


@dataclass(frozen=True, slots=True)
class SyntheticWiringCase:
    role: str
    payload: InternalCaseExecutionPayload
    execution_attempts: int


@dataclass(frozen=True, slots=True)
class SyntheticWiringPreparation:
    """Prepared package-contained cases for one deterministic wiring run."""

    context: InternalRunnerContext
    payload: InternalCaseExecutionPayload
    cases: tuple[SyntheticWiringCase, ...]
    candidate_config_digest: str
    execution_config_digest: str
    input_manifest_digest: str


def prepare_synthetic_wiring(
    *,
    package_root: str | Path,
    records_root: str | Path,
    workspace_root: str | Path,
    committed_revision: str,
    run_id: str,
) -> SyntheticWiringPreparation:
    """Construct real runner paths around deterministic development fixtures."""

    package = Path(package_root).resolve()
    records = Path(records_root).resolve()
    workspace = Path(workspace_root).resolve()
    if not package.is_dir():
        raise ExperimentExecutionEntrypointError(
            "package_root must be an existing directory"
        )
    if not records.is_absolute() or not workspace.is_absolute():
        raise ExperimentExecutionEntrypointError(
            "records_root and workspace_root must be absolute"
        )
    if (
        records == workspace
        or records in workspace.parents
        or workspace in records.parents
    ):
        raise ExperimentExecutionEntrypointError(
            "records_root and workspace_root must be disjoint"
        )
    if not REVISION.fullmatch(committed_revision):
        raise ExperimentExecutionEntrypointError(
            "committed_revision must be an exact Git revision"
        )
    if not SAFE_ID.fullmatch(run_id):
        raise ExperimentExecutionEntrypointError("run_id is invalid")

    protocol_path = (
        package
        / "configs/experiments/internal_scientific_validation_protocol.json"
    )
    component_path = (
        package / "configs/experiments/internal_execution_components.json"
    )
    runtime_config_path = (
        package / "configs/runtime/runtime_sd35_flowmatch.json"
    )
    protocol = load_frozen_internal_validation_protocol(protocol_path)
    adapter_configuration = (
        load_ceg_wm_experiment_adapter_configuration(component_path)
    )
    adapter = CegWmExperimentAdapter(adapter_configuration)
    attack_registry = load_attack_registry(component_path)
    metric_registry = load_metric_registry(component_path)

    case_id = "synthetic_wiring_case"
    source_units = tuple(_unit(index, case_id=case_id) for index in range(3))
    split_manifest = _split_manifest(source_units)
    content_operation = FormalHfContentDetectionOperation(adapter)
    content_binding, _prototype_score = create_formal_content_detector_binding(
        content_operation,
        prototype_image=_synthetic_image(),
        detection_key=SYNTHETIC_ROOT_KEY,
    )
    runtime_adapter = create_runtime_adapter(
        SyntheticQkBackend(
            root_key=SYNTHETIC_ROOT_KEY,
            model_revision=SYNTHETIC_MODEL_REVISION,
        ),
        runtime_config_path,
    )
    runtime_adapter.initialize("cpu")
    geometry_operation = FormalRuntimeGeometryEstimationOperation(
        runtime_adapter=runtime_adapter,
        adapter_configuration=adapter_configuration,
        epsilon_inlier=0.8,
        execution_scope=EXECUTION_SCOPE,
    )
    reliability_thresholds = GeometryReliabilityThresholds(
        gamma_coverage=0.45,
        gamma_uniqueness=0.0,
        gamma_gap=0.0,
        gamma_key=0.0,
        gamma_inlier=0.0,
        gamma_residual=1.0,
        gamma_identity=0.0,
        epsilon_inlier=0.8,
        fit_identity="synthetic_wiring_reliability_not_scientific_evidence",
    )
    operation_cases = (
        (
            "geometry_non_identity_success",
            geometry_operation,
            GeometricAttackSpec(
                "scale",
                scale_factor=exp(-log(sqrt(2.0)) / 2.0),
            ),
            1,
        ),
        (
            "resource_retry_resume",
            SyntheticDelegatingGeometryOperation(
                geometry_operation,
                failure_mode="resource_twice_then_qk_sync_failure",
            ),
            GeometricAttackSpec("crop", crop_fraction=0.9),
            3,
        ),
        (
            "execution_failure",
            SyntheticDelegatingGeometryOperation(
                geometry_operation,
                failure_mode="execution_failure",
            ),
            GeometricAttackSpec("identity"),
            1,
        ),
    )
    key_digest = adapter.identify_key(
        SYNTHETIC_ROOT_KEY
    ).result.root_key_public_digest
    routing_observation_digest = _canonical_digest(
        {
            "execution_scope": EXECUTION_SCOPE,
            "routing_identity": "routing_uniform_control",
        }
    )
    routing_mask_digest = _canonical_digest(
        {
            "mask_hf": "all_one",
            "mask_lf": "all_one",
            "routing_control": "uniform_disabled",
        }
    )
    cases: list[SyntheticWiringCase] = []
    entries: list[InternalCaseManifestEntry] = []
    reliability_digest = geometry_reliability_config_digest(
        reliability_thresholds
    )
    content_operation_digest = formal_operation_config_digest(
        content_operation,
        operation_role="content_detection",
    )
    for source_unit, case_definition in zip(
        source_units,
        operation_cases,
        strict=True,
    ):
        role, case_geometry_operation, attack_specification, attempts = (
            case_definition
        )
        source_artifact = AttackArtifact(
            source_unit,
            _synthetic_image(),
        )
        thresholds = JointDecisionThresholds(
            tau=1.0e300,
            tau_rescue=-1.0e300,
            detector_binding_digest=(
                content_binding.detector_binding_digest
            ),
            calibration_identity=(
                f"synthetic_{role}_thresholds_not_calibration_evidence"
            ),
        )
        geometry_operation_identity = (
            f"synthetic_wiring_{role}_geometry_operation"
        )
        payload = InternalCaseExecutionPayload(
            source_artifact=source_artifact,
            attack_specification=attack_specification,
            detection_key=SYNTHETIC_ROOT_KEY,
            content_detector_binding=content_binding,
            thresholds=thresholds,
            geometry_estimation_operation=case_geometry_operation,
            geometry_operation_identity=geometry_operation_identity,
            geometry_reliability_thresholds=reliability_thresholds,
        )
        expectation = FrozenCaseExecutionExpectation(
            content_detector_binding_digest=(
                content_binding.detector_binding_digest
            ),
            content_operation_config_digest=content_operation_digest,
            raw_detector_identity=content_binding.detector_identity,
            rectified_detector_identity=content_binding.detector_identity,
            raw_detector_config_digest=(
                content_binding.content_config_digest
            ),
            rectified_detector_config_digest=(
                content_binding.content_config_digest
            ),
            raw_preprocessing_identity=(
                content_binding.preprocessing_identity
            ),
            rectified_preprocessing_identity=(
                content_binding.preprocessing_identity
            ),
            raw_threshold_identity=thresholds.threshold_identity,
            rectified_threshold_identity=thresholds.threshold_identity,
            calibration_identity=thresholds.calibration_identity,
            tau=thresholds.tau,
            tau_rescue=thresholds.tau_rescue,
            geometry_operation_identity=geometry_operation_identity,
            geometry_operation_config_digest=formal_operation_config_digest(
                case_geometry_operation,
                operation_role="geometry_estimation",
            ),
            geometry_reliability_config_digest=reliability_digest,
        )
        entries.append(
            InternalCaseManifestEntry(
                analysis_unit_identity=source_unit,
                split="end_to_end_check",
                input_artifact_digest=source_artifact.image_digest,
                attack_config_digest=(
                    attack_specification.attack_config_digest
                ),
                metric_set_digest=metric_registry.registry_digest,
                routing_trace=RoutingTrace(
                    routing_identity="routing_uniform_control",
                    routing_control="uniform_disabled",
                    routing_observation_digest=(
                        routing_observation_digest
                    ),
                    routing_mask_digest=routing_mask_digest,
                ),
                key_control_trace=KeyControlTrace(
                    registered_key_public_digest=key_digest,
                    detection_key_public_digest=key_digest,
                    key_role="registered",
                    control_identity="registered_key_control",
                ),
                execution_expectation=expectation,
            )
        )
        cases.append(
            SyntheticWiringCase(
                role=role,
                payload=payload,
                execution_attempts=attempts,
            )
        )
    input_manifest = FrozenCaseInputManifest(
        manifest_schema_version=(
            "ceg_wm_internal_case_input_manifest_v3"
        ),
        manifest_id="synthetic_wiring_input_manifest",
        manifest_revision="synthetic_wiring_input_revision",
        protocol_digest=protocol.digest(),
        split_manifest_digest=split_manifest.digest(),
        entries=tuple(entries),
    )
    input_digest = input_manifest.digest()
    execution_digest = execution_config_digest(
        protocol=protocol,
        adapter=adapter,
        attack_registry=attack_registry,
        metric_registry=metric_registry,
    )
    candidate_digest = candidate_config_digest(
        adapter=adapter,
        input_manifest=input_manifest,
        method_code_revision=committed_revision,
    )
    bindings = FrozenRecordBindings(
        run_id=run_id,
        case_id=case_id,
        input_manifest_digest=input_digest,
        method_code_revision=committed_revision,
        candidate_config_digest=candidate_digest,
        method_config_digest=content_binding.content_config_digest,
        execution_config_digest=execution_digest,
        model_revision=SYNTHETIC_MODEL_REVISION,
        environment_digest=_canonical_digest(
            {
                "device": "cpu",
                "execution_scope": EXECUTION_SCOPE,
                "torch_version": torch.__version__,
            }
        ),
        resource_identity_digest=_canonical_digest(
            {
                "geometry_runtime_state": runtime_adapter.state.value,
                "resource_mode": "deterministic_cpu_fake_runtime",
            }
        ),
    )
    writer = GovernedRecordWriter(
        records_root=records,
        frozen_protocol=protocol,
        split_manifest=split_manifest,
        input_manifest=input_manifest,
        bindings=bindings,
    )
    context = InternalRunnerContext(
        protocol=protocol,
        split_manifest=split_manifest,
        input_manifest=input_manifest,
        adapter=adapter,
        attack_registry=attack_registry,
        metric_registry=metric_registry,
        writer=writer,
        bindings=bindings,
    )
    return SyntheticWiringPreparation(
        context=context,
        payload=cases[0].payload,
        cases=tuple(cases),
        candidate_config_digest=candidate_digest,
        execution_config_digest=execution_digest,
        input_manifest_digest=input_digest,
    )


def run_synthetic_wiring(
    *,
    package_root: str | Path,
    output_root: str | Path,
    workspace_root: str | Path,
    committed_revision: str,
    expected_candidate_config_digest: str,
    expected_execution_config_digest: str,
    expected_input_manifest_digest: str,
    run_id: str,
) -> dict[str, object]:
    """Execute package-contained non-scientific runner paths and replay them."""

    for role, digest in (
        (
            "expected_candidate_config_digest",
            expected_candidate_config_digest,
        ),
        (
            "expected_execution_config_digest",
            expected_execution_config_digest,
        ),
        (
            "expected_input_manifest_digest",
            expected_input_manifest_digest,
        ),
    ):
        if not DIGEST.fullmatch(digest):
            raise ExperimentExecutionEntrypointError(
                f"{role} must be a SHA-256 digest"
            )
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = prepare_synthetic_wiring(
        package_root=package_root,
        records_root=output / "records",
        workspace_root=workspace_root,
        committed_revision=committed_revision,
        run_id=run_id,
    )
    observed = {
        "candidate_config_digest": (
            preparation.candidate_config_digest
        ),
        "execution_config_digest": (
            preparation.execution_config_digest
        ),
        "input_manifest_digest": preparation.input_manifest_digest,
    }
    expected = {
        "candidate_config_digest": expected_candidate_config_digest,
        "execution_config_digest": expected_execution_config_digest,
        "input_manifest_digest": expected_input_manifest_digest,
    }
    if observed != expected:
        raise ExperimentExecutionEntrypointError(
            "prepared A3a identities differ from package-bound digests"
        )
    terminal_record_collection = None
    resumed_count = 0
    for case in preparation.cases:
        unit_id = (
            case.payload.source_artifact.analysis_unit_identity.unit_id
        )
        for _attempt in range(case.execution_attempts):
            case_result = execute_internal_case(
                preparation.context,
                unit_id=unit_id,
                payload=case.payload,
            )
            terminal_record_collection = case_result.collection
        resumed = execute_internal_case(
            preparation.context,
            unit_id=unit_id,
            payload=case.payload,
        )
        if not resumed.resumed_without_execution:
            raise ExperimentExecutionEntrypointError(
                f"synthetic case did not resume without execution: {case.role}"
            )
        resumed_count += 1
        terminal_record_collection = resumed.collection
    if terminal_record_collection is None:
        raise ExperimentExecutionEntrypointError(
            "synthetic case plan is empty"
        )
    replay = replay_internal_record_collection(
        preparation.context,
        terminal_record_collection,
    )
    if (
        replay.metric_case_count != replay.success_count
        or replay.metric_registry_digest
        != preparation.context.metric_registry.registry_digest
        or replay.metric_evaluator_identity
        != "experiments.metrics.aggregate_rectification_delta"
        or replay.metric_aggregate_identity
        != "experiments.metrics.RectificationDeltaAggregate"
        or replay.metric_aggregate_values is None
        or replay.metric_aggregate_values.case_count
        != replay.metric_case_count
        or not DIGEST.fullmatch(replay.metric_observation_digest)
        or replay.retry_count != 1
        or resumed_count != len(preparation.cases)
    ):
        raise ExperimentExecutionEntrypointError(
            "synthetic replay, metric, retry, or resume observation drifted"
        )
    record_path = preparation.context.writer.path
    summary = {
        "entrypoint_schema_version": ENTRYPOINT_SCHEMA_VERSION,
        "entrypoint_identity": ENTRYPOINT_IDENTITY,
        "artifact_kind": "experiment_execution_result",
        "execution_scope": EXECUTION_SCOPE,
        "evidence_scope": EVIDENCE_SCOPE,
        "run_status": "completed",
        "run_id": run_id,
        "committed_revision": committed_revision,
        "candidate_config_digest": (
            preparation.candidate_config_digest
        ),
        "execution_config_digest": (
            preparation.execution_config_digest
        ),
        "input_manifest_digest": preparation.input_manifest_digest,
        "record_collection_relative_path": (
            record_path.relative_to(output).as_posix()
        ),
        "record_collection_sha256": _file_sha256(record_path),
        "record_count": replay.record_count,
        "success_count": replay.success_count,
        "resource_failure_count": replay.resource_failure_count,
        "scientific_failure_count": replay.scientific_failure_count,
        "execution_failure_count": replay.execution_failure_count,
        "excluded_count": replay.excluded_count,
        "replay_digest": replay.replay_digest,
        "metric_registry_digest": replay.metric_registry_digest,
        "metric_evaluator_identity": replay.metric_evaluator_identity,
        "metric_aggregate_identity": replay.metric_aggregate_identity,
        "metric_case_results": [
            asdict(result) for result in replay.metric_case_results
        ],
        "metric_aggregate_values": asdict(
            replay.metric_aggregate_values
        ),
        "metric_evidence_digest": replay.metric_observation_digest,
        "scientific_claims_supported": False,
        "gpu_executed": False,
        "held_out_evaluation_accessed": False,
    }
    summary_path = output / "execution_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--committed-revision", required=True)
    parser.add_argument(
        "--expected-candidate-config-digest",
        required=True,
    )
    parser.add_argument(
        "--expected-execution-config-digest",
        required=True,
    )
    parser.add_argument(
        "--expected-input-manifest-digest",
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args(argv)
    result = run_synthetic_wiring(
        package_root=arguments.package_root,
        output_root=arguments.output_root,
        workspace_root=arguments.workspace_root,
        committed_revision=arguments.committed_revision,
        expected_candidate_config_digest=(
            arguments.expected_candidate_config_digest
        ),
        expected_execution_config_digest=(
            arguments.expected_execution_config_digest
        ),
        expected_input_manifest_digest=(
            arguments.expected_input_manifest_digest
        ),
        run_id=arguments.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
