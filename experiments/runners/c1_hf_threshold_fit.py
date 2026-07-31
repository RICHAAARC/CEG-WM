"""Threshold-fit-only C1 HF runner over the frozen 4096-unit manifest."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics import (
    C1HfMetricCaseIdentity,
    C1HfMetricImplementationBinding,
    C1HfScoreCase,
    C1HfThresholdResult,
    fit_c1_hf_tau,
    load_c1_hf_metric_implementation_binding,
)
from experiments.protocol.c1_hf_reference import (
    C1_HF_SOURCE_CLUSTERS_PER_SPLIT,
    load_c1_hf_reference_bundle,
)
from experiments.protocol.internal_splits import AnalysisUnitIdentity
from experiments.protocol.c1_hf_threshold_fit_records import (
    C1HfThresholdFitAttemptRecord,
    C1HfThresholdFitFactRecord,
    C1HfThresholdFitRecordIdentity,
    C1HfThresholdFitUnitRecordCollection,
    derive_c1_hf_threshold_fit_attempt_id,
)
from runtime import load_runtime_configuration

from .formal_operations import (
    PUBLIC_IMAGE_ENCODING,
    FormalHfContentDetectionOperation,
    create_formal_content_detector_binding,
)
from .record_writer import C1HfThresholdFitRecordWriter

DEFAULT_C1_HF_THRESHOLD_FIT_EXECUTION_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/experiments/c1_hf_threshold_fit_execution.json"
)
EXECUTION_SCHEMA_VERSION = "ceg_wm_c1_hf_threshold_fit_execution_v1"
FIT_SPLIT = "content_threshold_fit"
PRIMARY_NULL_ROLE = "unwatermarked_primary_null"
PRIMARY_NULL_CONTROL = "unwatermarked_image_with_registered_detection_key"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class C1HfThresholdFitRunnerError(ValueError):
    """Frozen authority, shard, record, or fit identity failed closed."""


class C1HfThresholdFitResourceFailure(RuntimeError):
    """Retryable accelerator or memory failure, never method science."""


class C1HfThresholdFitExecutionFailure(RuntimeError):
    """Non-resource runtime or orchestration failure."""


class C1HfThresholdFitScientificFailure(RuntimeError):
    """A completed operation produced an invalid required scientific fact."""


class C1HfThresholdFitExcluded(RuntimeError):
    """A preregistered exclusion that remains in the planned denominator."""

    def __init__(self, rule_id: str) -> None:
        if type(rule_id) is not str or not rule_id:
            raise C1HfThresholdFitRunnerError("exclusion rule identity is required")
        self.rule_id = rule_id
        super().__init__(rule_id)


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(value: object, role: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise C1HfThresholdFitRunnerError(f"{role} must be SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class C1HfThresholdFitExecutionConfiguration:
    raw: Mapping[str, object]
    execution_config_digest: str

    @property
    def shard_count(self) -> int:
        return int(self.raw["shard_count"])

    @property
    def source_clusters_per_shard(self) -> int:
        return int(self.raw["source_clusters_per_shard"])

    @property
    def maximum_record_attempts_per_unit(self) -> int:
        return int(self.raw["maximum_record_attempts_per_unit"])


@dataclass(frozen=True, slots=True)
class C1HfThresholdFitAuthority:
    repository_root: Path
    configuration: C1HfThresholdFitExecutionConfiguration
    assignments: tuple[AnalysisUnitIdentity, ...]
    prompt_text_by_digest: Mapping[str, str]
    metric_binding: C1HfMetricImplementationBinding
    adapter: CegWmExperimentAdapter
    protocol_id: str
    protocol_version: str
    candidate_config_digest: str
    method_config_digest: str
    runtime_config_digest: str
    model_revision: str


@dataclass(frozen=True, slots=True)
class C1HfThresholdFitExecutionFact:
    score: float
    image_digest: str
    detector_identity: str
    detector_config_digest: str
    detection_key_public_digest: str
    runtime_config_digest: str
    model_revision: str
    selected_device: str
    preprocessing_identity: str = PUBLIC_IMAGE_ENCODING

    def __post_init__(self) -> None:
        if type(self.score) is not float or not math.isfinite(self.score):
            raise C1HfThresholdFitScientificFailure(
                "HF score must be a finite binary64 fact"
            )
        for role in (
            "image_digest",
            "detector_config_digest",
            "detection_key_public_digest",
            "runtime_config_digest",
        ):
            _require_digest(getattr(self, role), role)
        if not _REVISION.fullmatch(self.model_revision):
            raise C1HfThresholdFitScientificFailure(
                "model revision identity drifted"
            )
        if (
            type(self.detector_identity) is not str
            or not self.detector_identity
            or self.selected_device != "cuda:0"
            or self.preprocessing_identity != PUBLIC_IMAGE_ENCODING
        ):
            raise C1HfThresholdFitScientificFailure(
                "detector or selected-device identity drifted"
            )


class C1HfThresholdFitSession(Protocol):
    """One already-prepared shard session; model preparation is never per-unit."""

    def execute(
        self,
        unit: AnalysisUnitIdentity,
        prompt_text: str,
        registered_detection_key: str,
    ) -> C1HfThresholdFitExecutionFact:
        """Execute one fit unit without taking ownership of the registered secret."""


ThresholdFitSessionFactory = Callable[
    [C1HfThresholdFitAuthority],
    AbstractContextManager[C1HfThresholdFitSession],
]


def load_c1_hf_threshold_fit_execution_configuration(
    path: str | Path = DEFAULT_C1_HF_THRESHOLD_FIT_EXECUTION_PATH,
) -> C1HfThresholdFitExecutionConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "run_phase_id",
        "accessible_split",
        "forbidden_splits",
        "authorization_base_revision",
        "c1_specification_path",
        "c1_specification_digest",
        "prompt_roster_path",
        "prompt_roster_file_sha256",
        "dataset_snapshot_path",
        "dataset_snapshot_sha256",
        "fit_manifest_path",
        "fit_manifest_file_sha256",
        "fit_manifest_digest",
        "component_registry_path",
        "method_adapter_config_digest",
        "metric_registry_digest",
        "metric_implementation_path",
        "metric_implementation_binding_digest",
        "metric_implementation_source_sha256",
        "runtime_config_path",
        "runtime_config_sha256",
        "runtime_qualification_revision",
        "model_revision",
        "detector_mode",
        "source_cluster_count",
        "shard_count",
        "source_clusters_per_shard",
        "shard_assignment",
        "maximum_record_attempts_per_unit",
        "early_stopping",
        "resource_plan",
        "failure_classes",
        "invocation_policy",
        "resume_policy",
        "claim_boundary",
        "execution_config_digest",
    }
    if type(raw) is not dict or set(raw) != expected_fields:
        raise C1HfThresholdFitRunnerError("execution configuration fields drifted")
    supplied_digest = _require_digest(
        raw["execution_config_digest"], "execution_config_digest"
    )
    payload = {key: value for key, value in raw.items() if key != "execution_config_digest"}
    if supplied_digest != _canonical_digest(payload):
        raise C1HfThresholdFitRunnerError("execution configuration digest drifted")
    resource_plan = raw["resource_plan"]
    if (
        raw["schema_version"] != EXECUTION_SCHEMA_VERSION
        or raw["accessible_split"] != FIT_SPLIT
        or raw["forbidden_splits"] != ["untouched_confirmation"]
        or raw["detector_mode"] != "hf_only"
        or raw["source_cluster_count"] != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or raw["shard_count"] != 16
        or raw["source_clusters_per_shard"] != 256
        or raw["shard_assignment"]
        != "ascending_materialized_assignment_index"
        or raw["maximum_record_attempts_per_unit"] != 3
        or raw["early_stopping"] != "forbidden"
        or raw["failure_classes"]
        != ["resource_failure", "execution_failure", "scientific_failure"]
        or type(resource_plan) is not dict
        or resource_plan.get("accelerator_count") != 1
        or resource_plan.get("selected_device") != "cuda:0"
        or resource_plan.get("accelerator_model_policy")
        != "model_agnostic_no_exact_gpu_identity_gate"
        or resource_plan.get("minimum_vram_bytes") != 23622320128
        or resource_plan.get("minimum_vram_basis")
        != "22_gib_floor_below_recorded_colab_l4_23034_mib_reference"
        or resource_plan.get("per_shard_walltime_planning_cap_seconds") != 86400
        or resource_plan.get("walltime_semantics")
        != "planning_cap_only_not_observed_runtime"
        or raw["invocation_policy"]
        != {
            "mode": "explicit_user_colab_run_only",
            "required_flag": "user_colab_run",
            "local_or_task_agent_invocation": "prohibited",
        }
        or raw["resume_policy"]
        != {
            "explicit_shard_index_required": True,
            "incremental_unit_attempt_persistence_required": True,
        }
    ):
        raise C1HfThresholdFitRunnerError("execution configuration semantics drifted")
    if (
        not _REVISION.fullmatch(str(raw["authorization_base_revision"]))
        or not _REVISION.fullmatch(str(raw["runtime_qualification_revision"]))
        or not _REVISION.fullmatch(str(raw["model_revision"]))
    ):
        raise C1HfThresholdFitRunnerError("authorization base revision is invalid")
    return C1HfThresholdFitExecutionConfiguration(raw, supplied_digest)


def load_c1_hf_threshold_fit_authority(
    repository_root: str | Path,
    execution_config_path: str | Path = DEFAULT_C1_HF_THRESHOLD_FIT_EXECUTION_PATH,
) -> C1HfThresholdFitAuthority:
    root = Path(repository_root).resolve()
    config_path = Path(execution_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    configuration = load_c1_hf_threshold_fit_execution_configuration(config_path)
    raw = configuration.raw
    bound_files = (("runtime_config_path", "runtime_config_sha256"),)
    for path_field, digest_field in bound_files:
        candidate = root / str(raw[path_field])
        if not candidate.is_file() or _file_sha256(candidate) != raw[digest_field]:
            raise C1HfThresholdFitRunnerError(f"bound file drifted: {path_field}")
    reference_bundle = load_c1_hf_reference_bundle(root)
    specification = reference_bundle.specification
    dataset_binding = specification.raw["dataset"]
    split_binding = specification.raw["split_manifests"][FIT_SPLIT]
    candidate_binding = specification.raw["candidate_binding"]
    exact_authority_fields = {
        "c1_specification_path": "configs/experiments/c1_hf_reference_run.json",
        "prompt_roster_path": dataset_binding["roster_path"],
        "prompt_roster_file_sha256": dataset_binding["roster_file_sha256"],
        "dataset_snapshot_path": dataset_binding["dataset_snapshot_path"],
        "dataset_snapshot_sha256": dataset_binding["file_sha256"],
        "fit_manifest_path": split_binding["path"],
        "fit_manifest_file_sha256": split_binding["file_sha256"],
        "fit_manifest_digest": split_binding["materialized_manifest_digest"],
        "component_registry_path": candidate_binding[
            "formal_method_adapter_config_path"
        ],
        "method_adapter_config_digest": candidate_binding[
            "formal_method_adapter_config_digest"
        ],
        "runtime_config_path": candidate_binding["runtime_config_path"],
        "runtime_config_sha256": candidate_binding["runtime_config_sha256"],
        "runtime_qualification_revision": candidate_binding[
            "runtime_qualification"
        ]["candidate_revision"],
    }
    if any(raw[field] != expected for field, expected in exact_authority_fields.items()):
        raise C1HfThresholdFitRunnerError(
            "execution configuration authority path or identity drifted"
        )
    if specification.digest() != raw["c1_specification_digest"]:
        raise C1HfThresholdFitRunnerError("C1 specification digest drifted")
    phase = specification.raw["run_phases"]["threshold_fit"]
    if (
        phase["accessible_split"] != FIT_SPLIT
        or phase["forbidden_split_access"] != ["untouched_confirmation"]
        or specification.raw["execution_budget"]["threshold_fit"]
        ["source_clusters"]
        != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
    ):
        raise C1HfThresholdFitRunnerError("C1 threshold phase identity drifted")
    roster = reference_bundle.roster
    manifest = next(
        candidate
        for candidate in reference_bundle.materialized_manifests
        if candidate.assignments[0].split == FIT_SPLIT
    )
    assignments = tuple(assignment.identity for assignment in manifest.assignments)
    if (
        len(assignments) != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or manifest.digest() != raw["fit_manifest_digest"]
    ):
        raise C1HfThresholdFitRunnerError("materialized fit manifest drifted")
    component_path = root / str(raw["component_registry_path"])
    adapter_configuration = load_ceg_wm_experiment_adapter_configuration(component_path)
    if adapter_configuration.config_digest != raw["method_adapter_config_digest"]:
        raise C1HfThresholdFitRunnerError("component registry identity drifted")
    metric_binding = load_c1_hf_metric_implementation_binding(
        root / str(raw["metric_implementation_path"])
    )
    if (
        metric_binding.binding_digest != raw["metric_implementation_binding_digest"]
        or metric_binding.implementation_source_sha256
        != raw["metric_implementation_source_sha256"]
        or metric_binding.metric_registry_digest != raw["metric_registry_digest"]
        or metric_binding.fit_manifest_digest != manifest.digest()
        or metric_binding.fit_analysis_units != frozenset(assignments)
    ):
        raise C1HfThresholdFitRunnerError("C1-M threshold binding drifted")
    runtime_configuration = load_runtime_configuration(
        root / str(raw["runtime_config_path"])
    )
    if runtime_configuration.model_revision != raw["model_revision"]:
        raise C1HfThresholdFitRunnerError("runtime candidate revision drifted")
    prompt_text_by_digest = {row.prompt_digest: row.prompt_text for row in roster.rows}
    if any(identity.prompt_digest not in prompt_text_by_digest for identity in assignments):
        raise C1HfThresholdFitRunnerError("fit assignment prompt is absent from roster")
    return C1HfThresholdFitAuthority(
        repository_root=root,
        configuration=configuration,
        assignments=assignments,
        prompt_text_by_digest=prompt_text_by_digest,
        metric_binding=metric_binding,
        adapter=CegWmExperimentAdapter(adapter_configuration),
        protocol_id=str(specification.raw["protocol_id"]),
        protocol_version=str(specification.raw["protocol_version"]),
        candidate_config_digest=str(
            specification.raw["candidate_binding"]["candidate_binding_digest"]
        ),
        method_config_digest=adapter_configuration.config_digest,
        runtime_config_digest=runtime_configuration.runtime_config_digest,
        model_revision=runtime_configuration.model_revision,
    )


def c1_hf_threshold_fit_shard(
    authority: C1HfThresholdFitAuthority,
    shard_index: int,
) -> tuple[AnalysisUnitIdentity, ...]:
    if type(authority) is not C1HfThresholdFitAuthority:
        raise C1HfThresholdFitRunnerError("threshold authority exact type is required")
    if type(shard_index) is not int or not 0 <= shard_index < authority.configuration.shard_count:
        raise C1HfThresholdFitRunnerError("threshold shard index is invalid")
    size = authority.configuration.source_clusters_per_shard
    start = shard_index * size
    shard = authority.assignments[start : start + size]
    if len(shard) != size:
        raise C1HfThresholdFitRunnerError("threshold shard is incomplete")
    return shard


def run_c1_hf_threshold_fit_shard(
    *,
    authority: C1HfThresholdFitAuthority,
    shard_index: int,
    run_id: str,
    committed_revision: str,
    registered_detection_key: str,
    environment_digest: str,
    resource_identity_digest: str,
    records_root: str | Path,
    user_colab_run: bool,
    session_factory: ThresholdFitSessionFactory,
) -> Mapping[str, object]:
    """Execute or resume one explicit shard with per-attempt atomic persistence."""

    if not _REVISION.fullmatch(committed_revision):
        raise C1HfThresholdFitRunnerError("committed revision must be exact")
    if type(run_id) is not str or not run_id or not registered_detection_key:
        raise C1HfThresholdFitRunnerError("run and registered-key inputs are required")
    _require_digest(environment_digest, "environment_digest")
    _require_digest(resource_identity_digest, "resource_identity_digest")
    if user_colab_run is not True:
        raise C1HfThresholdFitRunnerError(
            "explicit user Colab run flag is required; task-agent invocation is prohibited"
        )
    root = Path(records_root)
    if not root.is_absolute():
        raise C1HfThresholdFitRunnerError("records root must be absolute")
    shard = c1_hf_threshold_fit_shard(authority, shard_index)
    try:
        import torch
    except ImportError as exc:
        raise C1HfThresholdFitExecutionFailure(
            "PyTorch is required for detector binding preflight"
        ) from exc
    operation = FormalHfContentDetectionOperation(authority.adapter)
    detector_binding, _ = create_formal_content_detector_binding(
        operation,
        prototype_image=torch.arange(12, dtype=torch.uint8).reshape(1, 3, 2, 2),
        detection_key=registered_detection_key,
    )
    identities_and_writers: list[
        tuple[C1HfThresholdFitRecordIdentity, C1HfThresholdFitRecordWriter]
    ] = []
    collections: list[C1HfThresholdFitUnitRecordCollection | None] = []
    shard_start = shard_index * authority.configuration.source_clusters_per_shard
    for local_index, unit in enumerate(shard):
        identity = C1HfThresholdFitRecordIdentity(
            run_id=run_id,
            committed_revision=committed_revision,
            c1_specification_digest=authority.metric_binding.c1_specification_digest,
            protocol_id=authority.protocol_id,
            protocol_version=authority.protocol_version,
            protocol_digest=authority.metric_binding.protocol_digest,
            shard_index=shard_index,
            unit_index=shard_start + local_index,
            execution_config_digest=authority.configuration.execution_config_digest,
            fit_manifest_digest=authority.metric_binding.fit_manifest_digest,
            metric_binding_digest=authority.metric_binding.binding_digest,
            metric_registry_digest=authority.metric_binding.metric_registry_digest,
            candidate_config_digest=authority.candidate_config_digest,
            method_config_digest=authority.method_config_digest,
            runtime_config_digest=authority.runtime_config_digest,
            model_revision=authority.model_revision,
            detector_identity=detector_binding.detector_identity,
            detector_config_digest=detector_binding.content_config_digest,
            preprocessing_identity=detector_binding.preprocessing_identity,
            registered_key_family_digest=(
                authority.metric_binding.registered_key_family_digest
            ),
            registered_key_public_digest=detector_binding.root_key_public_digest,
            environment_digest=environment_digest,
            analysis_unit_identity=unit,
        )
        writer = C1HfThresholdFitRecordWriter(records_root=root, identity=identity)
        identities_and_writers.append((identity, writer))
        collections.append(writer.load())

    def terminal(collection: C1HfThresholdFitUnitRecordCollection | None) -> bool:
        if collection is None:
            return False
        last = collection.attempts[-1]
        return (
            last.status in {"success", "excluded"}
            or last.failure_class in {"execution_failure", "scientific_failure"}
            or len(collection.attempts)
            >= authority.configuration.maximum_record_attempts_per_unit
        )

    def summarize() -> Mapping[str, object]:
        recorded = [collection.attempts[-1] for collection in collections if collection]
        return {
            "run_id": run_id,
            "run_phase_id": "c1_hf_threshold_fit_v1",
            "split": FIT_SPLIT,
            "shard_index": shard_index,
            "planned_unit_count": len(shard),
            "recorded_unit_count": len(recorded),
            "success_count": sum(item.status == "success" for item in recorded),
            "excluded_count": sum(item.status == "excluded" for item in recorded),
            "failed_count": sum(item.status == "failed" for item in recorded),
            "retry_pending_count": sum(item.status == "retry" for item in recorded),
            "walltime_value_semantics": (
                "not_collected_planning_cap_is_not_observation"
            ),
            "scientific_claims_supported": False,
        }

    if any(not terminal(collection) for collection in collections):
        with session_factory(authority) as session:
            for offset, unit in enumerate(shard):
                collection = collections[offset]
                if terminal(collection):
                    continue
                identity, writer = identities_and_writers[offset]
                prompt_text = authority.prompt_text_by_digest[unit.prompt_digest]
                while not terminal(collection):
                    attempt_index = 0 if collection is None else len(collection.attempts)
                    retry_parent = (
                        None if collection is None else collection.attempts[-1].attempt_id
                    )
                    try:
                        fact = session.execute(
                            unit,
                            prompt_text,
                            registered_detection_key,
                        )
                        if type(fact) is not C1HfThresholdFitExecutionFact:
                            raise C1HfThresholdFitScientificFailure(
                                "session returned a non-fact result"
                            )
                        if (
                            fact.runtime_config_digest != authority.runtime_config_digest
                            or fact.model_revision != authority.model_revision
                            or fact.detector_identity != identity.detector_identity
                            or fact.detector_config_digest != identity.detector_config_digest
                            or fact.preprocessing_identity != identity.preprocessing_identity
                            or fact.detection_key_public_digest
                            != identity.registered_key_public_digest
                        ):
                            raise C1HfThresholdFitScientificFailure(
                                "execution fact differs from frozen authority"
                            )
                        attempt = C1HfThresholdFitAttemptRecord(
                            attempt_id=derive_c1_hf_threshold_fit_attempt_id(
                                identity, attempt_index
                            ),
                            attempt_index=attempt_index,
                            resource_identity_digest=resource_identity_digest,
                            status="success",
                            failure_class=None,
                            failure_type=None,
                            exclusion_rule_id=None,
                            retry_of_attempt_id=retry_parent,
                            fact=C1HfThresholdFitFactRecord(
                                score_float64_hex=fact.score.hex(),
                                image_digest=fact.image_digest,
                                input_artifact_digest=fact.image_digest,
                                detector_identity=fact.detector_identity,
                                detector_config_digest=fact.detector_config_digest,
                                detection_key_public_digest=(
                                    fact.detection_key_public_digest
                                ),
                                selected_device=fact.selected_device,
                            ),
                        )
                    except C1HfThresholdFitExcluded as exc:
                        attempt = C1HfThresholdFitAttemptRecord(
                            attempt_id=derive_c1_hf_threshold_fit_attempt_id(identity, attempt_index),
                            attempt_index=attempt_index,
                            resource_identity_digest=resource_identity_digest,
                            status="excluded",
                            failure_class=None,
                            failure_type=None,
                            exclusion_rule_id=exc.rule_id,
                            retry_of_attempt_id=retry_parent,
                            fact=None,
                        )
                    except C1HfThresholdFitResourceFailure as exc:
                        attempt_budget_exhausted = (
                            attempt_index + 1
                            >= authority.configuration.maximum_record_attempts_per_unit
                        )
                        attempt = C1HfThresholdFitAttemptRecord(
                            attempt_id=derive_c1_hf_threshold_fit_attempt_id(identity, attempt_index),
                            attempt_index=attempt_index,
                            resource_identity_digest=resource_identity_digest,
                            status="failed" if attempt_budget_exhausted else "retry",
                            failure_class="resource_failure",
                            failure_type=type(exc).__name__,
                            exclusion_rule_id=None,
                            retry_of_attempt_id=retry_parent,
                            fact=None,
                        )
                    except C1HfThresholdFitScientificFailure as exc:
                        attempt = C1HfThresholdFitAttemptRecord(
                            attempt_id=derive_c1_hf_threshold_fit_attempt_id(identity, attempt_index),
                            attempt_index=attempt_index,
                            resource_identity_digest=resource_identity_digest,
                            status="failed",
                            failure_class="scientific_failure",
                            failure_type=type(exc).__name__,
                            exclusion_rule_id=None,
                            retry_of_attempt_id=retry_parent,
                            fact=None,
                        )
                    except Exception as exc:
                        attempt = C1HfThresholdFitAttemptRecord(
                            attempt_id=derive_c1_hf_threshold_fit_attempt_id(identity, attempt_index),
                            attempt_index=attempt_index,
                            resource_identity_digest=resource_identity_digest,
                            status="failed",
                            failure_class="execution_failure",
                            failure_type=type(exc).__name__,
                            exclusion_rule_id=None,
                            retry_of_attempt_id=retry_parent,
                            fact=None,
                        )
                    collection = writer.append_attempt(attempt)
                    collections[offset] = collection
                    if attempt.failure_class == "resource_failure":
                        return summarize()
                collections[offset] = collection
    return summarize()


def _case_from_record(
    collection: C1HfThresholdFitUnitRecordCollection,
    authority: C1HfThresholdFitAuthority,
) -> C1HfScoreCase:
    attempt = collection.attempts[-1]
    if attempt.status != "success" or attempt.fact is None:
        raise C1HfThresholdFitRunnerError("threshold fit has a missing required outcome")
    unit = collection.identity.analysis_unit_identity
    fact = attempt.fact
    return C1HfScoreCase(
        identity=C1HfMetricCaseIdentity(
            analysis_unit_identity=unit,
            split=FIT_SPLIT,
            detector_identity=collection.identity.detector_identity,
            detector_config_digest=collection.identity.detector_config_digest,
            protocol_id=authority.protocol_id,
            protocol_version=authority.protocol_version,
            protocol_digest=authority.metric_binding.protocol_digest,
            c1_specification_digest=authority.metric_binding.c1_specification_digest,
            manifest_digest=authority.metric_binding.fit_manifest_digest,
            metric_registry_digest=authority.metric_binding.metric_registry_digest,
            registered_key_family_digest=(
                authority.metric_binding.registered_key_family_digest
            ),
        ),
        key_role=PRIMARY_NULL_ROLE,
        score=fact.score(),
        registered_detection_key_public_digest=(
            collection.identity.registered_key_public_digest
        ),
        detection_key_public_digest=collection.identity.registered_key_public_digest,
        control_identity=PRIMARY_NULL_CONTROL,
        image_digest=fact.image_digest,
    )


def finalize_c1_hf_threshold_fit(
    *,
    authority: C1HfThresholdFitAuthority,
    run_id: str,
    committed_revision: str,
    registered_detection_key: str,
    environment_digest: str,
    records_root: str | Path,
) -> C1HfThresholdResult:
    """Replay all 4096 typed unit records and invoke the exact C1-M fit."""

    root = Path(records_root)
    if not root.is_absolute():
        raise C1HfThresholdFitRunnerError("records root must be absolute")
    if not _REVISION.fullmatch(committed_revision):
        raise C1HfThresholdFitRunnerError("committed revision must be exact")
    _require_digest(environment_digest, "environment_digest")
    try:
        import torch
    except ImportError as exc:
        raise C1HfThresholdFitExecutionFailure("PyTorch is required") from exc
    detector_binding, _ = create_formal_content_detector_binding(
        FormalHfContentDetectionOperation(authority.adapter),
        prototype_image=torch.arange(12, dtype=torch.uint8).reshape(1, 3, 2, 2),
        detection_key=registered_detection_key,
    )
    cases: list[C1HfScoreCase] = []
    for unit_index, unit in enumerate(authority.assignments):
        shard_index = unit_index // authority.configuration.source_clusters_per_shard
        identity = C1HfThresholdFitRecordIdentity(
            run_id=run_id,
            committed_revision=committed_revision,
            c1_specification_digest=authority.metric_binding.c1_specification_digest,
            protocol_id=authority.protocol_id,
            protocol_version=authority.protocol_version,
            protocol_digest=authority.metric_binding.protocol_digest,
            shard_index=shard_index,
            unit_index=unit_index,
            execution_config_digest=authority.configuration.execution_config_digest,
            fit_manifest_digest=authority.metric_binding.fit_manifest_digest,
            metric_binding_digest=authority.metric_binding.binding_digest,
            metric_registry_digest=authority.metric_binding.metric_registry_digest,
            candidate_config_digest=authority.candidate_config_digest,
            method_config_digest=authority.method_config_digest,
            runtime_config_digest=authority.runtime_config_digest,
            model_revision=authority.model_revision,
            detector_identity=detector_binding.detector_identity,
            detector_config_digest=detector_binding.content_config_digest,
            preprocessing_identity=detector_binding.preprocessing_identity,
            registered_key_family_digest=authority.metric_binding.registered_key_family_digest,
            registered_key_public_digest=detector_binding.root_key_public_digest,
            environment_digest=environment_digest,
            analysis_unit_identity=unit,
        )
        collection = C1HfThresholdFitRecordWriter(
            records_root=root,
            identity=identity,
        ).load()
        if collection is None:
            raise C1HfThresholdFitRunnerError("threshold unit record is unavailable")
        cases.append(_case_from_record(collection, authority))
    return fit_c1_hf_tau(tuple(cases), binding=authority.metric_binding)


def _is_oom(error: BaseException, torch_module: object) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
    current: BaseException | None = error
    while current is not None:
        if isinstance(oom_type, type) and isinstance(current, oom_type):
            return True
        current = current.__cause__
    return False


class _ProductionC1HfThresholdFitSession:
    def __init__(self, authority: C1HfThresholdFitAuthority) -> None:
        self._authority = authority
        self._torch: object | None = None
        self._backend: object | None = None
        self._runtime_adapter: object | None = None
        self._runtime_session: object | None = None
        self._operation = FormalHfContentDetectionOperation(authority.adapter)

    def __enter__(self) -> "_ProductionC1HfThresholdFitSession":
        runtime_adapter_for_cleanup: object | None = None
        try:
            import os
            import torch
            from runtime import Sd35PipelineBackend, create_runtime_adapter

            if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                raise C1HfThresholdFitResourceFailure("cuda:0 is unavailable")
            total_memory = int(torch.cuda.get_device_properties(0).total_memory)
            minimum = int(
                self._authority.configuration.raw["resource_plan"][
                    "minimum_vram_bytes"
                ]
            )
            if total_memory < minimum:
                raise C1HfThresholdFitResourceFailure(
                    "cuda:0 is below the frozen model-agnostic VRAM floor"
                )
            cache_root = Path(os.environ["CEG_WM_EPHEMERAL_ROOT"]).resolve()
            persistent_root = Path(os.environ["CEG_WM_PERSISTENT_ROOT"]).resolve()
            backend = Sd35PipelineBackend(
                cache_root=cache_root / "hf_cache",
                persistent_root=persistent_root,
                hf_token=os.environ.get("HF_TOKEN"),
                prompt="",
            )
            runtime_adapter = create_runtime_adapter(
                backend,
                self._authority.repository_root
                / str(self._authority.configuration.raw["runtime_config_path"]),
            )
            runtime_adapter_for_cleanup = runtime_adapter
            runtime_session = runtime_adapter.initialize("cuda")
            self._torch = torch
            self._backend = backend
            self._runtime_adapter = runtime_adapter
            self._runtime_session = runtime_session
            return self
        except C1HfThresholdFitResourceFailure:
            self._close_partially_initialized_runtime(runtime_adapter_for_cleanup)
            raise
        except (KeyError, OSError) as exc:
            self._close_partially_initialized_runtime(runtime_adapter_for_cleanup)
            raise C1HfThresholdFitExecutionFailure(
                "threshold execution environment is incomplete"
            ) from exc
        except Exception as exc:
            self._close_partially_initialized_runtime(runtime_adapter_for_cleanup)
            torch_module = locals().get("torch")
            if torch_module is not None and _is_oom(exc, torch_module):
                raise C1HfThresholdFitResourceFailure(
                    "threshold session preparation exhausted CUDA memory"
                ) from exc
            raise C1HfThresholdFitExecutionFailure(
                "threshold session preparation failed"
            ) from exc

    @staticmethod
    def _close_partially_initialized_runtime(runtime_adapter: object | None) -> None:
        if runtime_adapter is None:
            return
        try:
            runtime_adapter.close()
        except Exception:
            pass

    def execute(
        self,
        unit: AnalysisUnitIdentity,
        prompt_text: str,
        registered_detection_key: str,
    ) -> C1HfThresholdFitExecutionFact:
        if any(
            value is None
            for value in (
                self._torch,
                self._backend,
                self._runtime_adapter,
                self._runtime_session,
            )
        ):
            raise C1HfThresholdFitExecutionFailure("threshold session is not prepared")
        torch = self._torch
        backend = self._backend
        runtime_adapter = self._runtime_adapter
        runtime_session = self._runtime_session
        try:
            prompt_identity = backend.set_generation_prompts(prompt_text, "")
            if prompt_identity.prompt_digest != unit.prompt_digest:
                raise C1HfThresholdFitScientificFailure(
                    "runtime prompt digest differs from frozen manifest unit"
                )
            configuration = runtime_adapter.configuration
            generator = torch.Generator(device="cpu").manual_seed(unit.generation_seed)
            initial = torch.randn(
                (
                    1,
                    16,
                    configuration.image_height // 8,
                    configuration.image_width // 8,
                ),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=runtime_session.selected_device, dtype=torch.float16)
            generated_latent = backend.run_generation(
                initial,
                lambda _index, latent: latent,
            )
            factors = backend.vae_factors()
            decoded = backend.vae_decode(
                generated_latent.to(dtype=torch.float32) / factors.scaling_factor
                + factors.shift_factor
            )
            rgb8 = torch.floor(decoded.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            content_result = self._operation(rgb8, registered_detection_key)
            if content_result.content_input_image_digest is None:
                raise C1HfThresholdFitScientificFailure(
                    "formal detector omitted public image digest"
                )
            return C1HfThresholdFitExecutionFact(
                score=float(content_result.content_score),
                image_digest=content_result.content_input_image_digest,
                detector_identity=content_result.detector_identity,
                detector_config_digest=content_result.content_config_digest,
                detection_key_public_digest=(
                    content_result.hf_result.root_key_public_digest
                ),
                runtime_config_digest=runtime_session.runtime_config_digest,
                model_revision=runtime_session.model_revision,
                selected_device=runtime_session.selected_device,
                preprocessing_identity=self._operation.preprocessing_identity,
            )
        except C1HfThresholdFitScientificFailure:
            raise
        except Exception as exc:
            if _is_oom(exc, torch):
                raise C1HfThresholdFitResourceFailure(
                    "threshold unit exhausted CUDA memory"
                ) from exc
            raise C1HfThresholdFitExecutionFailure(
                "threshold unit execution failed"
            ) from exc

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._runtime_adapter is not None:
            try:
                self._runtime_adapter.close()
            except Exception as close_error:
                if exc is None:
                    raise C1HfThresholdFitExecutionFailure(
                        "threshold session close failed"
                    ) from close_error
        self._runtime_adapter = None
        self._runtime_session = None
        self._backend = None
        self._torch = None
        return False


def production_c1_hf_threshold_fit_session(
    authority: C1HfThresholdFitAuthority,
) -> AbstractContextManager[C1HfThresholdFitSession]:
    """Create one lazy real-model session; CPU tests never call this factory."""

    return _ProductionC1HfThresholdFitSession(authority)


__all__ = [
    "C1HfThresholdFitAuthority",
    "C1HfThresholdFitExecutionConfiguration",
    "C1HfThresholdFitExecutionFact",
    "C1HfThresholdFitExecutionFailure",
    "C1HfThresholdFitExcluded",
    "C1HfThresholdFitResourceFailure",
    "C1HfThresholdFitRunnerError",
    "C1HfThresholdFitScientificFailure",
    "c1_hf_threshold_fit_shard",
    "finalize_c1_hf_threshold_fit",
    "load_c1_hf_threshold_fit_authority",
    "load_c1_hf_threshold_fit_execution_configuration",
    "production_c1_hf_threshold_fit_session",
    "run_c1_hf_threshold_fit_shard",
]
