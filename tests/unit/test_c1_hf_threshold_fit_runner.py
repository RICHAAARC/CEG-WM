"""CPU-only tests for typed C1 threshold-fit records and shard execution."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics import C1HfMetricImplementationBinding
from experiments.protocol.c1_hf_reference import (
    load_c1_hf_reference_specification,
    load_compact_c1_split_manifest,
    load_frozen_prompt_roster,
    materialize_c1_split_manifest,
)
from experiments.protocol.c1_hf_threshold_fit_records import (
    C1_HF_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE,
    C1_HF_THRESHOLD_FIT_RECORD_SCHEMA_VERSION,
    C1_HF_THRESHOLD_FIT_SPLIT,
    C1_HF_THRESHOLD_FIT_SYNTHETIC_EXECUTION_EVIDENCE,
    C1HfThresholdFitAttemptRecord,
    C1HfThresholdFitFactRecord,
    C1HfThresholdFitRecordError,
    C1HfThresholdFitRecordIdentity,
    C1HfThresholdFitUnitRecordCollection,
    derive_c1_hf_threshold_fit_attempt_id,
    load_c1_hf_threshold_fit_record_collection,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)
from experiments.runners.c1_hf_threshold_fit import (
    C1HfThresholdFitAuthority,
    C1HfThresholdFitExecutionFact,
    C1HfThresholdFitExecutionFailure,
    C1HfThresholdFitResourceFailure,
    C1HfThresholdFitRunnerError,
    c1_hf_threshold_fit_shard,
    finalize_c1_hf_threshold_fit,
    finalize_c1_hf_threshold_fit_synthetic_cpu_fixture,
    load_c1_hf_threshold_fit_authority,
    load_c1_hf_threshold_fit_execution_configuration,
    production_c1_hf_threshold_fit_session,
    run_c1_hf_threshold_fit_shard,
    run_c1_hf_threshold_fit_synthetic_cpu_fixture_shard,
)
import experiments.runners.c1_hf_threshold_fit as threshold_fit_runner
from experiments.runners.formal_operations import (
    PUBLIC_IMAGE_ENCODING,
    FormalHfContentDetectionOperation,
)
from experiments.runners.record_writer import C1HfThresholdFitRecordWriter
from runtime import Sd35BackendError, Sd35PipelineBackend, load_runtime_configuration


ROOT = Path(__file__).resolve().parents[2]
REVISION = "3" * 40
ENVIRONMENT_DIGEST = "e" * 64
RESOURCE_A_DIGEST = "f" * 64
RESOURCE_B_DIGEST = "0" * 64
REGISTERED_KEY = "unit-test-registered-key"


def _analysis_unit() -> AnalysisUnitIdentity:
    prompt_digest = sha256(b"prompt").hexdigest()
    image_lineage_digest = sha256(b"lineage").hexdigest()
    key_family_digest = sha256(b"family").hexdigest()
    seed = 7
    return AnalysisUnitIdentity(
        unit_id="unit-0",
        case_id="C1-HF",
        source_cluster_id=derive_source_cluster_id(
            prompt_digest=prompt_digest,
            generation_seed=seed,
            image_lineage_digest=image_lineage_digest,
            registered_key_family_digest=key_family_digest,
        ),
        prompt_digest=prompt_digest,
        generation_seed=seed,
        image_lineage_digest=image_lineage_digest,
        registered_key_family_digest=key_family_digest,
    )


def _record_identity(
    execution_evidence_kind: str = C1_HF_THRESHOLD_FIT_SYNTHETIC_EXECUTION_EVIDENCE,
) -> C1HfThresholdFitRecordIdentity:
    unit = _analysis_unit()
    return C1HfThresholdFitRecordIdentity(
        run_id="run-1",
        committed_revision=REVISION,
        execution_evidence_kind=execution_evidence_kind,
        c1_specification_digest="a" * 64,
        protocol_id="ceg_wm_internal_scientific_validation_v2",
        protocol_version="2.0.0",
        protocol_digest="b" * 64,
        shard_index=0,
        unit_index=0,
        execution_config_digest="c" * 64,
        fit_manifest_digest="d" * 64,
        metric_binding_digest="1" * 64,
        metric_registry_digest="2" * 64,
        candidate_config_digest="3" * 64,
        method_config_digest="4" * 64,
        runtime_config_digest="5" * 64,
        model_revision="6" * 40,
        detector_identity="ceg_wm_hf_only_detector",
        detector_config_digest="7" * 64,
        preprocessing_identity=PUBLIC_IMAGE_ENCODING,
        registered_key_family_digest=unit.registered_key_family_digest,
        registered_key_public_digest="8" * 64,
        environment_digest=ENVIRONMENT_DIGEST,
        analysis_unit_identity=unit,
    )


def _attempt(
    identity: C1HfThresholdFitRecordIdentity,
    index: int,
    *,
    status: str,
    failure_class: str | None = None,
    parent: str | None = None,
    resource_identity_digest: str = RESOURCE_A_DIGEST,
) -> C1HfThresholdFitAttemptRecord:
    success = status == "success"
    return C1HfThresholdFitAttemptRecord(
        attempt_id=derive_c1_hf_threshold_fit_attempt_id(identity, index),
        attempt_index=index,
        resource_identity_digest=resource_identity_digest,
        status=status,
        failure_class=failure_class,
        failure_type=None if success else "SyntheticFailure",
        exclusion_rule_id=None,
        retry_of_attempt_id=parent,
        fact=(
            C1HfThresholdFitFactRecord(
                score_float64_hex=(0.25).hex(),
                image_digest="9" * 64,
                input_artifact_digest="9" * 64,
                detector_identity=identity.detector_identity,
                detector_config_digest=identity.detector_config_digest,
                detection_key_public_digest=identity.registered_key_public_digest,
                selected_device="cuda:0",
            )
            if success
            else None
        ),
    )


@pytest.mark.quick
def test_typed_record_writer_replays_retry_lineage_and_rejects_drift(
    tmp_path: Path,
) -> None:
    identity = _record_identity()
    writer = C1HfThresholdFitRecordWriter(records_root=tmp_path, identity=identity)
    retry = _attempt(identity, 0, status="retry", failure_class="resource_failure")
    first = writer.append_attempt(retry)
    assert first.attempts == (retry,)
    success = _attempt(
        identity,
        1,
        status="success",
        parent=retry.attempt_id,
    )
    complete = writer.append_attempt(success)
    assert complete.attempts == (retry, success)
    assert writer.append_attempt(success) == complete
    assert load_c1_hf_threshold_fit_record_collection(
        writer.path,
        expected_identity=identity,
    ) == complete

    raw = json.loads(writer.path.read_text(encoding="utf-8"))
    raw["extra"] = True
    writer.path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(C1HfThresholdFitRecordError, match="fields drifted"):
        load_c1_hf_threshold_fit_record_collection(writer.path)

    tampered = asdict(complete)
    tampered["attempts"][1]["fact"]["detector_config_digest"] = "0" * 64
    writer.path.write_text(
        json.dumps(
            tampered,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(C1HfThresholdFitRecordError, match="binding identity"):
        load_c1_hf_threshold_fit_record_collection(writer.path)


@pytest.mark.quick
def test_typed_record_parser_rejects_nonfinite_json_as_custom_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"generation_seed":NaN}', encoding="utf-8")
    with pytest.raises(C1HfThresholdFitRecordError):
        load_c1_hf_threshold_fit_record_collection(path)


@pytest.mark.quick
def test_production_authority_loader_replays_full_c1_p_and_c1_m() -> None:
    authority = load_c1_hf_threshold_fit_authority(ROOT)
    assert len(authority.assignments) == 4096
    assert authority.metric_binding.fit_analysis_units == frozenset(
        authority.assignments
    )
    assert len(authority.metric_binding.confirmation_analysis_units) == 4096
    assert authority.configuration.raw["accessible_split"] == (
        "content_threshold_fit"
    )
    assert authority.configuration.raw["forbidden_splits"] == [
        "untouched_confirmation"
    ]


@pytest.mark.quick
@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("prompt_roster_path", "configs/experiments/c1_hf_metric_implementation.json"),
        ("runtime_qualification_revision", "0" * 40),
        ("run_phase_id", "rehash_accepted_phase"),
        ("authorization_base_revision", "9" * 40),
        ("claim_boundary", "rehash_accepted_claim"),
        ("shard_count", 8),
    ),
)
def test_production_authority_loader_rejects_rehashed_decorative_field_tamper(
    tmp_path: Path,
    field: str,
    tampered_value: str,
) -> None:
    raw = json.loads(
        (ROOT / "configs/experiments/c1_hf_threshold_fit_execution.json").read_text(
            encoding="utf-8"
        )
    )
    raw[field] = tampered_value
    payload = {key: value for key, value in raw.items() if key != "execution_config_digest"}
    raw["execution_config_digest"] = sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(
        C1HfThresholdFitRunnerError,
        match="frozen authority",
    ):
        load_c1_hf_threshold_fit_authority(ROOT, path)


@pytest.mark.quick
def test_formal_public_api_rejects_injected_factory_and_revision_keywords(
    tmp_path: Path,
) -> None:
    run_parameters = inspect.signature(run_c1_hf_threshold_fit_shard).parameters
    finalize_parameters = inspect.signature(finalize_c1_hf_threshold_fit).parameters
    assert "session_factory" not in run_parameters
    assert "committed_revision" not in run_parameters
    assert "committed_revision" not in finalize_parameters
    authority = _authority()
    with pytest.raises(TypeError, match="committed_revision"):
        run_c1_hf_threshold_fit_shard(
            authority=authority,
            shard_index=0,
            run_id="formal-run",
            committed_revision=REVISION,  # type: ignore[call-arg]
            registered_detection_key=REGISTERED_KEY,
            environment_digest=ENVIRONMENT_DIGEST,
            resource_identity_digest=RESOURCE_A_DIGEST,
            records_root=tmp_path,
            user_colab_run=True,
        )
    with pytest.raises(TypeError, match="committed_revision"):
        finalize_c1_hf_threshold_fit(
            authority=authority,
            run_id="formal-run",
            committed_revision=REVISION,  # type: ignore[call-arg]
            registered_detection_key=REGISTERED_KEY,
            environment_digest=ENVIRONMENT_DIGEST,
            records_root=tmp_path,
        )


@pytest.mark.quick
def test_formal_runner_rejects_shadowed_production_session_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    fake_factory = _FakeFactory(authority)
    monkeypatch.setattr(
        threshold_fit_runner,
        "production_c1_hf_threshold_fit_session",
        fake_factory,
    )
    monkeypatch.setattr(
        threshold_fit_runner,
        "_resolve_clean_repository_revision",
        lambda _root: REVISION,
    )
    with pytest.raises(C1HfThresholdFitRunnerError, match="factory identity drifted"):
        run_c1_hf_threshold_fit_shard(
            authority=authority,
            shard_index=0,
            run_id="formal-run",
            registered_detection_key=REGISTERED_KEY,
            environment_digest=ENVIRONMENT_DIGEST,
            resource_identity_digest=RESOURCE_A_DIGEST,
            records_root=tmp_path,
            user_colab_run=True,
        )
    assert fake_factory.enter_count == fake_factory.execute_count == 0


@pytest.mark.quick
def test_private_core_rejects_real_evidence_with_fake_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    fake_factory = _FakeFactory(authority)
    monkeypatch.setattr(
        threshold_fit_runner,
        "_resolve_clean_repository_revision",
        lambda _root: REVISION,
    )
    with pytest.raises(
        C1HfThresholdFitRunnerError,
        match="pinned production session factory",
    ):
        threshold_fit_runner._run_c1_hf_threshold_fit_shard_core(
            authority=authority,
            shard_index=0,
            run_id="formal-run",
            committed_revision=REVISION,
            execution_evidence_kind=C1_HF_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE,
            registered_detection_key=REGISTERED_KEY,
            environment_digest=ENVIRONMENT_DIGEST,
            resource_identity_digest=RESOURCE_A_DIGEST,
            records_root=tmp_path,
            session_factory=fake_factory,
        )
    assert fake_factory.enter_count == fake_factory.execute_count == 0


@pytest.mark.quick
def test_private_cores_reject_arbitrary_real_evidence_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    clean_head = "a" * 40
    monkeypatch.setattr(
        threshold_fit_runner,
        "_resolve_clean_repository_revision",
        lambda _root: clean_head,
    )
    with pytest.raises(
        C1HfThresholdFitRunnerError,
        match="differs from clean repository HEAD",
    ):
        threshold_fit_runner._run_c1_hf_threshold_fit_shard_core(
            authority=authority,
            shard_index=0,
            run_id="formal-run",
            committed_revision=REVISION,
            execution_evidence_kind=C1_HF_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE,
            registered_detection_key=REGISTERED_KEY,
            environment_digest=ENVIRONMENT_DIGEST,
            resource_identity_digest=RESOURCE_A_DIGEST,
            records_root=tmp_path,
            session_factory=_FakeFactory(authority),
        )
    with pytest.raises(
        C1HfThresholdFitRunnerError,
        match="differs from clean repository HEAD",
    ):
        threshold_fit_runner._finalize_c1_hf_threshold_fit_core(
            authority=authority,
            run_id="formal-run",
            committed_revision=REVISION,
            expected_execution_evidence_kind=(
                C1_HF_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE
            ),
            registered_detection_key=REGISTERED_KEY,
            environment_digest=ENVIRONMENT_DIGEST,
            records_root=tmp_path,
        )


@pytest.mark.quick
def test_formal_revision_is_derived_from_clean_git_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = iter(
        (
            SimpleNamespace(stdout=f"{REVISION}\n"),
            SimpleNamespace(stdout=""),
        )
    )
    monkeypatch.setattr(
        threshold_fit_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: next(completed),
    )
    assert threshold_fit_runner._resolve_clean_repository_revision(ROOT) == REVISION

    dirty = iter(
        (
            SimpleNamespace(stdout=f"{REVISION}\n"),
            SimpleNamespace(stdout=" M experiments/runners/c1_hf_threshold_fit.py\n"),
        )
    )
    monkeypatch.setattr(
        threshold_fit_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: next(dirty),
    )
    with pytest.raises(C1HfThresholdFitRunnerError, match="source drift"):
        threshold_fit_runner._resolve_clean_repository_revision(ROOT)


def _authority() -> C1HfThresholdFitAuthority:
    configuration = load_c1_hf_threshold_fit_execution_configuration()
    specification = load_c1_hf_reference_specification(
        ROOT / "configs/experiments/c1_hf_reference_run.json"
    )
    roster = load_frozen_prompt_roster(
        ROOT / "configs/experiments/c1_hf_prompt_roster.json"
    )
    compact = load_compact_c1_split_manifest(
        ROOT / "configs/experiments/c1_hf_content_threshold_fit_manifest.json"
    )
    manifest = materialize_c1_split_manifest(compact, roster)
    assignments = tuple(item.identity for item in manifest.assignments)
    metric_raw = json.loads(
        (ROOT / "configs/experiments/c1_hf_metric_implementation.json").read_text(
            encoding="utf-8"
        )
    )
    adapter_configuration = load_ceg_wm_experiment_adapter_configuration(
        ROOT / "configs/experiments/internal_execution_components.json"
    )
    runtime_configuration = load_runtime_configuration(
        ROOT / "configs/runtime/runtime_sd35_flowmatch.json"
    )
    metric_binding = C1HfMetricImplementationBinding(
        c1_specification_digest=metric_raw["c1_specification_digest"],
        protocol_digest=metric_raw["protocol_digest"],
        fit_manifest_digest=manifest.digest(),
        confirmation_manifest_digest=(
            metric_raw["split_manifest_digests"]["untouched_confirmation"]
        ),
        registered_key_family_digest=metric_raw["registered_key_family_digest"],
        metric_registry_digest=metric_raw["metric_registry_digest"],
        formula_identity_digest=metric_raw["formula_identity_digest"],
        implementation_source_sha256=metric_raw["implementation_source_sha256"],
        binding_digest=metric_raw["binding_digest"],
        fit_analysis_units=frozenset(assignments),
        confirmation_analysis_units=frozenset(),
    )
    return C1HfThresholdFitAuthority(
        repository_root=ROOT,
        configuration=configuration,
        assignments=assignments,
        prompt_text_by_digest={row.prompt_digest: row.prompt_text for row in roster.rows},
        metric_binding=metric_binding,
        adapter=CegWmExperimentAdapter(adapter_configuration),
        protocol_id=specification.raw["protocol_id"],
        protocol_version=specification.raw["protocol_version"],
        candidate_config_digest=(
            specification.raw["candidate_binding"]["candidate_binding_digest"]
        ),
        method_config_digest=adapter_configuration.config_digest,
        runtime_config_digest=runtime_configuration.runtime_config_digest,
        model_revision=runtime_configuration.model_revision,
    )


class _FakeSession:
    def __init__(self, authority: C1HfThresholdFitAuthority, owner: "_FakeFactory") -> None:
        self.authority = authority
        self.owner = owner
        self.operation = FormalHfContentDetectionOperation(authority.adapter)
        self.prototype = None

    def __enter__(self) -> "_FakeSession":
        self.owner.enter_count += 1
        return self

    def __exit__(self, *_args: object) -> bool:
        self.owner.exit_count += 1
        return False

    def execute(
        self,
        unit: AnalysisUnitIdentity,
        prompt_text: str,
        registered_detection_key: str,
    ) -> C1HfThresholdFitExecutionFact:
        self.owner.execute_count += 1
        if self.owner.resource_retry_unit == unit.unit_id and unit.unit_id not in self.owner.retried:
            self.owner.retried.add(unit.unit_id)
            raise C1HfThresholdFitResourceFailure("synthetic OOM")
        if self.prototype is None:
            self.prototype = self.operation(
                torch.arange(12, dtype=torch.uint8).reshape(1, 3, 2, 2),
                registered_detection_key,
            )
        index = self.owner.index_by_unit[unit.unit_id]
        return C1HfThresholdFitExecutionFact(
            score=float(index) / 8192.0,
            image_digest=sha256(unit.unit_id.encode("utf-8")).hexdigest(),
            detector_identity=self.prototype.detector_identity,
            detector_config_digest=self.prototype.content_config_digest,
            detection_key_public_digest=(
                self.prototype.hf_result.root_key_public_digest
            ),
            runtime_config_digest=self.authority.runtime_config_digest,
            model_revision=self.authority.model_revision,
            selected_device="cuda:0",
            preprocessing_identity=self.operation.preprocessing_identity,
        )


class _FakeFactory:
    def __init__(
        self,
        authority: C1HfThresholdFitAuthority,
        *,
        resource_retry_unit: str | None = None,
    ) -> None:
        self.index_by_unit = {
            unit.unit_id: index for index, unit in enumerate(authority.assignments)
        }
        self.resource_retry_unit = resource_retry_unit
        self.retried: set[str] = set()
        self.enter_count = 0
        self.exit_count = 0
        self.execute_count = 0

    def __call__(self, authority: C1HfThresholdFitAuthority) -> _FakeSession:
        return _FakeSession(authority, self)


@pytest.mark.quick
def test_threshold_fit_config_shards_and_resumable_single_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    config = authority.configuration.raw
    assert config["resource_plan"]["minimum_vram_bytes"] == 22 * 1024**3
    assert config["resource_plan"]["accelerator_model_policy"].startswith(
        "model_agnostic"
    )
    assert config["invocation_policy"]["mode"] == "explicit_user_colab_run_only"
    first_shard = c1_hf_threshold_fit_shard(authority, 0)
    last_shard = c1_hf_threshold_fit_shard(authority, 15)
    assert len(first_shard) == len(last_shard) == 256
    assert first_shard + tuple(
        unit
        for shard_index in range(1, 15)
        for unit in c1_hf_threshold_fit_shard(authority, shard_index)
    ) + last_shard == authority.assignments
    with pytest.raises(C1HfThresholdFitRunnerError):
        c1_hf_threshold_fit_shard(authority, 16)

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA touched by CPU test")),
    )
    factory = _FakeFactory(authority, resource_retry_unit=first_shard[0].unit_id)
    summary = run_c1_hf_threshold_fit_synthetic_cpu_fixture_shard(
        authority=authority,
        shard_index=0,
        run_id="fit-run",
        fixture_revision=REVISION,
        registered_detection_key=REGISTERED_KEY,
        environment_digest=ENVIRONMENT_DIGEST,
        resource_identity_digest=RESOURCE_A_DIGEST,
        records_root=tmp_path,
        session_factory=factory,
    )
    assert summary["success_count"] == 0
    assert summary["retry_pending_count"] == 1
    assert summary["recorded_unit_count"] == 1
    assert factory.enter_count == factory.exit_count == 1
    assert factory.execute_count == 1
    first_record = load_c1_hf_threshold_fit_record_collection(
        tmp_path / "fit-run/threshold_fit/shard_00/unit_0000.json"
    )
    assert first_record.identity.execution_evidence_kind == (
        C1_HF_THRESHOLD_FIT_SYNTHETIC_EXECUTION_EVIDENCE
    )
    assert tuple(item.status for item in first_record.attempts) == ("retry",)
    assert first_record.attempts[0].resource_identity_digest == RESOURCE_A_DIGEST

    resume_factory = _FakeFactory(authority)
    resumed = run_c1_hf_threshold_fit_synthetic_cpu_fixture_shard(
        authority=authority,
        shard_index=0,
        run_id="fit-run",
        fixture_revision=REVISION,
        registered_detection_key=REGISTERED_KEY,
        environment_digest=ENVIRONMENT_DIGEST,
        resource_identity_digest=RESOURCE_B_DIGEST,
        records_root=tmp_path,
        session_factory=resume_factory,
    )
    assert resumed["success_count"] == 256
    assert resume_factory.enter_count == resume_factory.exit_count == 1
    assert resume_factory.execute_count == 256
    resumed_first = load_c1_hf_threshold_fit_record_collection(
        tmp_path / "fit-run/threshold_fit/shard_00/unit_0000.json"
    )
    assert tuple(item.status for item in resumed_first.attempts) == (
        "retry",
        "success",
    )
    assert tuple(
        item.resource_identity_digest for item in resumed_first.attempts
    ) == (RESOURCE_A_DIGEST, RESOURCE_B_DIGEST)
    assert (
        resumed_first.attempts[1].retry_of_attempt_id
        == resumed_first.attempts[0].attempt_id
    )
    completed_factory = _FakeFactory(authority)
    completed = run_c1_hf_threshold_fit_synthetic_cpu_fixture_shard(
        authority=authority,
        shard_index=0,
        run_id="fit-run",
        fixture_revision=REVISION,
        registered_detection_key=REGISTERED_KEY,
        environment_digest=ENVIRONMENT_DIGEST,
        resource_identity_digest=RESOURCE_B_DIGEST,
        records_root=tmp_path,
        session_factory=completed_factory,
    )
    assert completed["success_count"] == 256
    assert completed_factory.enter_count == completed_factory.execute_count == 0
    with pytest.raises(C1HfThresholdFitRunnerError, match="explicit user Colab"):
        run_c1_hf_threshold_fit_shard(
            authority=authority,
            shard_index=1,
            run_id="blocked-run",
            registered_detection_key=REGISTERED_KEY,
            environment_digest=ENVIRONMENT_DIGEST,
            resource_identity_digest=RESOURCE_A_DIGEST,
            records_root=tmp_path,
            user_colab_run=False,
        )


@pytest.mark.quick
def test_synthetic_finalize_is_non_scientific_and_formal_finalize_rejects_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    replayed_resources: set[str] = set()
    synthetic_collections: dict[int, C1HfThresholdFitUnitRecordCollection] = {}

    def load_typed(
        writer: C1HfThresholdFitRecordWriter,
    ) -> C1HfThresholdFitUnitRecordCollection:
        identity = writer._identity  # type: ignore[attr-defined]
        index = identity.unit_index
        attempt = C1HfThresholdFitAttemptRecord(
            attempt_id=derive_c1_hf_threshold_fit_attempt_id(identity, 0),
            attempt_index=0,
            resource_identity_digest=(
                RESOURCE_A_DIGEST
                if identity.shard_index == 0
                else RESOURCE_B_DIGEST
            ),
            status="success",
            failure_class=None,
            failure_type=None,
            exclusion_rule_id=None,
            retry_of_attempt_id=None,
            fact=C1HfThresholdFitFactRecord(
                score_float64_hex=(float(index) / 8192.0).hex(),
                image_digest=sha256(
                    identity.analysis_unit_identity.unit_id.encode("utf-8")
                ).hexdigest(),
                input_artifact_digest=sha256(
                    identity.analysis_unit_identity.unit_id.encode("utf-8")
                ).hexdigest(),
                detector_identity=identity.detector_identity,
                detector_config_digest=identity.detector_config_digest,
                detection_key_public_digest=identity.registered_key_public_digest,
                selected_device="cuda:0",
            ),
        )
        replayed_resources.add(attempt.resource_identity_digest)
        collection = C1HfThresholdFitUnitRecordCollection(
            schema_version=C1_HF_THRESHOLD_FIT_RECORD_SCHEMA_VERSION,
            split=C1_HF_THRESHOLD_FIT_SPLIT,
            identity=identity,
            attempts=(attempt,),
        )
        synthetic_collections[index] = collection
        return collection

    monkeypatch.setattr(C1HfThresholdFitRecordWriter, "load", load_typed)
    summary = finalize_c1_hf_threshold_fit_synthetic_cpu_fixture(
        authority=authority,
        run_id="full-fit-run",
        fixture_revision=REVISION,
        registered_detection_key=REGISTERED_KEY,
        environment_digest=ENVIRONMENT_DIGEST,
        records_root=tmp_path,
    )
    assert summary == {
        "run_id": "full-fit-run",
        "execution_evidence_kind": (
            C1_HF_THRESHOLD_FIT_SYNTHETIC_EXECUTION_EVIDENCE
        ),
        "tau_float64_hex": math.nextafter(4095.0 / 8192.0, math.inf).hex(),
        "scientific_claims_supported": False,
    }
    assert replayed_resources == {RESOURCE_A_DIGEST, RESOURCE_B_DIGEST}

    def load_synthetic_collection(
        writer: C1HfThresholdFitRecordWriter,
    ) -> C1HfThresholdFitUnitRecordCollection:
        identity = writer._identity  # type: ignore[attr-defined]
        return synthetic_collections[identity.unit_index]

    monkeypatch.setattr(
        C1HfThresholdFitRecordWriter,
        "load",
        load_synthetic_collection,
    )
    monkeypatch.setattr(
        threshold_fit_runner,
        "_resolve_clean_repository_revision",
        lambda _root: REVISION,
    )
    with pytest.raises(C1HfThresholdFitRunnerError, match="evidence kind"):
        finalize_c1_hf_threshold_fit(
            authority=authority,
            run_id="full-fit-run",
            registered_detection_key=REGISTERED_KEY,
            environment_digest=ENVIRONMENT_DIGEST,
            records_root=tmp_path,
        )


class _PromptPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append((str(kwargs["prompt"]), str(kwargs["negative_prompt"])))
        return SimpleNamespace(images=kwargs["latents"])


@pytest.mark.quick
def test_sd35_prompt_selection_is_one_generation_and_preserves_legacy_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    persistent = tmp_path / "persistent"
    backend = Sd35PipelineBackend(
        cache_root=cache,
        persistent_root=persistent,
        hf_token=None,
        prompt="legacy prompt",
        negative_prompt="legacy negative",
    )
    pipeline = _PromptPipeline()
    backend._configuration = load_runtime_configuration()  # type: ignore[attr-defined]
    backend._device = torch.device("cpu")  # type: ignore[attr-defined]
    backend._pipeline = pipeline  # type: ignore[attr-defined]
    latent = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    backend.run_generation(latent, lambda _index, value: value)
    backend.run_generation(latent, lambda _index, value: value)
    assert pipeline.calls == [
        ("legacy prompt", "legacy negative"),
        ("legacy prompt", "legacy negative"),
    ]

    prompt = "每个 C1 unit 的提示词"
    identity = backend.set_generation_prompts(prompt, "")
    assert identity.prompt_digest == sha256(prompt.encode("utf-8")).hexdigest()
    backend.run_generation(latent, lambda _index, value: value)
    with pytest.raises(Sd35BackendError, match="explicit per-unit prompt"):
        backend.run_generation(latent, lambda _index, value: value)
    with pytest.raises(Sd35BackendError, match="exact empty negative"):
        backend.set_generation_prompts("next", "not empty")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend.close()
    with pytest.raises(Sd35BackendError, match="not prepared"):
        backend.set_generation_prompts("after close", "")


@pytest.mark.quick
def test_production_session_closes_partial_runtime_after_initialize_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runtime

    authority = _authority()
    close_count = 0

    class FakeBackend:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class FakeRuntimeAdapter:
        def initialize(self, _device: str) -> object:
            raise RuntimeError("synthetic prepare failure")

        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    monkeypatch.setenv("CEG_WM_EPHEMERAL_ROOT", str(tmp_path / "ephemeral"))
    monkeypatch.setenv("CEG_WM_PERSISTENT_ROOT", str(tmp_path / "persistent"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=24 * 1024**3),
    )
    monkeypatch.setattr(runtime, "Sd35PipelineBackend", FakeBackend)
    monkeypatch.setattr(
        runtime,
        "create_runtime_adapter",
        lambda _backend, _path: FakeRuntimeAdapter(),
    )
    with pytest.raises(C1HfThresholdFitExecutionFailure):
        with production_c1_hf_threshold_fit_session(authority):
            raise AssertionError("unreachable")
    assert close_count == 1
