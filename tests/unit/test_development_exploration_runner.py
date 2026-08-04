"""CPU wiring checks for the real development exploration dispatch."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest
import torch

from experiments.attacks import load_attack_registry
from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.runners.development_exploration import (
    DevelopmentExplorationRunner,
    DevelopmentOperationalReceipt,
    DevelopmentRunnerError,
    DevelopmentUnitExcluded,
    DevelopmentUnitInput,
    _safe_result_payload,
)
from experiments.runners.development_persistence import (
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
    development_unit_roster_digest,
)
from experiments.runners.synthetic_runtime import SyntheticQkBackend
import experiments.runners.record_writer as record_writer_module
import experiments.runners.development_exploration as development_runner_module
from experiments.protocol.development_exploration import (
    DevelopmentVerifiedModuleOutcome,
    build_development_cross_fit_plan,
    create_development_provisional_threshold,
    create_development_threshold_fit_input,
)
from experiments.protocol.internal_matrix import REQUIRED_METHOD_RESPONSIBILITIES
from experiments.protocol.development_records import canonical_development_value_digest
from scripts.experiment_execution.development_exploration_worker_inputs import (
    DevelopmentProductionInputBuilder,
)
from main import (
    BranchNullCalibration,
    ConditionalRecoveryResult,
    GeometryReliabilityThresholds,
    HfDetectionObservation,
    LfDetectionObservation,
    RoutingObservations,
    hf_detector,
    lf_detector,
)
from main.content_chain.routing import SpatialRoutingObservation
from main.content_chain.detector import NullScoreRecord
from runtime import (
    RuntimeRoutingReferenceMeasurement,
    Sd35RuntimeAdapter,
    create_runtime_adapter,
)
import scripts.experiment_execution.development_exploration_worker_inputs as worker_inputs_module
from scripts.experiment_execution import (
    development_exploration_entrypoint as development_entrypoint,
)
from tests.unit.test_development_module_exploration import (
    _development_manifest,
    _execution_intent,
    _primary_null_record,
    _redigest_scientific_record,
    _threshold_material,
)
from tests.unit.test_internal_governed_runner import _context as _internal_context
from experiments.runners.internal import execute_internal_case
import tests.unit.test_internal_governed_runner as internal_runner_test_module
from tests.unit.test_runtime_content_write_and_vae import FakeContentBackend
from tests.unit.test_runtime_qk_observation import FakeQkBackend
from tests.unit.test_development_worker_persistence import (
    _intent as _persistence_intent,
    _lease as _persistence_lease,
    _record as _persistence_record,
    _routing_reference_record,
    _store as _persistence_store,
)


ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"


def _source_digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _observations() -> RoutingObservations:
    def spatial(values: tuple[float, ...], shape: tuple[int, int], label: str):
        return SpatialRoutingObservation(
            values=values,
            spatial_shape=shape,
            source_identity_digest=_source_digest(label),
        )

    return RoutingObservations(
        semantic=spatial((0.0, 0.8, 0.3, 0.6), (2, 2), "semantic"),
        texture=spatial((0.0, 0.25, 0.75, 1.0), (2, 2), "texture"),
        response=spatial((0.0, 0.25, 0.5, 0.75), (2, 2), "response"),
        sensitivity=spatial((0.1, 0.4, 0.7, 1.0), (2, 2), "sensitivity"),
    )


def _reliability_thresholds() -> GeometryReliabilityThresholds:
    return GeometryReliabilityThresholds(
        gamma_coverage=0.45,
        gamma_uniqueness=0.0,
        gamma_gap=-1.0,
        gamma_key=-1.0,
        gamma_inlier=0.0,
        gamma_residual=100.0,
        gamma_identity=-1.0,
        epsilon_inlier=0.8,
        fit_identity="cpu_wiring_development_only",
    )


class _CombinedCpuWiringBackend:
    def __init__(self) -> None:
        callback_sequence = tuple(range(20))
        self.content = FakeContentBackend(
            callback_sequences=tuple(callback_sequence for _ in range(128))
        )
        self.geometry = FakeQkBackend()

    @property
    def run_calls(self):
        return self.content.run_calls

    def probe_devices(self):
        return self.geometry.probe_devices()

    def prepare(self, configuration, selected_device):
        return self.geometry.prepare(configuration, selected_device)

    def close(self):
        self.content.close()
        self.geometry.close()

    def run_generation(self, initial_latent, callback):
        return self.content.run_generation(initial_latent, callback)

    def vae_factors(self):
        return self.content.vae_factors()

    def vae_decode(self, latent):
        decoded = self.content.vae_decode(latent)
        size = (self.geometry.configuration.image_height, self.geometry.configuration.image_width)
        return torch.nn.functional.interpolate(
            decoded.to(torch.float32), size=size, mode="bilinear", align_corners=False
        )

    def vae_encode(self, image):
        return self.geometry.vae_encode(image)

    def create_detection_schedule(self, inference_steps):
        return self.geometry.create_detection_schedule(inference_steps)

    def scale_detection_noise(self, detection_latent, public_noise, timestep):
        return self.geometry.scale_detection_noise(
            detection_latent, public_noise, timestep
        )

    def attention_module(self, layer_name):
        return self.geometry.attention_module(layer_name)

    def run_qk_detection_forward(self, noisy_detection_latent, timestep, conditioning):
        return self.geometry.run_qk_detection_forward(
            noisy_detection_latent, timestep, conditioning
        )


def _runner(
    intent_authority=None,
    persistence_store: DevelopmentPersistentStore | None = None,
) -> DevelopmentExplorationRunner:
    runtime_adapter = create_runtime_adapter(_CombinedCpuWiringBackend())
    runtime_adapter.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS),
    )
    return DevelopmentExplorationRunner(
        intent_authority=(
            _execution_intent(
                _development_manifest(64), run_id="development_runner_wiring"
            )
            if intent_authority is None
            else intent_authority
        ),
        adapter=adapter,
        runtime_adapter=runtime_adapter,
        attack_registry=load_attack_registry(COMPONENTS),
        method_code_revision="a" * 40,
        environment_digest="b" * 64,
        resource_identity_digest="c" * 64,
        persistence_store=persistence_store,
    )


def _persistent_runner(tmp_path: Path) -> tuple[DevelopmentExplorationRunner, DevelopmentPersistentStore]:
    initial = _runner()
    package = tmp_path / "development_package.zip"
    bootstrap = tmp_path / "development_bootstrap.py"
    package.write_bytes(b"frozen package")
    bootstrap.write_bytes(b"frozen bootstrap")
    worker = FrozenWorkerIdentity(
        revision="a" * 40,
        protocol_digest=initial.intent_authority.protocol_digest,
        execution_intent_authority_digest=initial.intent_authority.authority_digest,
        input_manifest_digest=initial.intent_authority.input_manifest_digest,
        candidate_config_digest="d" * 64,
        unit_roster_digest=development_unit_roster_digest(initial.protocol.unit_roster),
        package_sha256=sha256(package.read_bytes()).hexdigest(),
        bootstrap_sha256=sha256(bootstrap.read_bytes()).hexdigest(),
    )
    store = DevelopmentPersistentStore(
        tmp_path / "persistent",
        run_id=initial.intent_authority.run_id,
        worker_identity=worker,
        package_path=package,
        bootstrap_path=bootstrap,
        registered_unit_bindings=initial.create_persistence_unit_bindings(),
    )
    return _runner(initial.intent_authority, store), store


class _ReferenceMeasurementRuntime:
    def __init__(
        self,
        *,
        fail_once_at: int | None = None,
        implementation_failure_at: int | None = None,
    ) -> None:
        self.fail_once_at = fail_once_at
        self.implementation_failure_at = implementation_failure_at
        self.failed = False
        self.sample_indexes: list[int] = []

    def measure_generation_routing_reference_inputs(
        self,
        _base_latent: torch.Tensor,
        *,
        sample_index: int,
    ) -> RuntimeRoutingReferenceMeasurement:
        self.sample_indexes.append(sample_index)
        if self.fail_once_at == sample_index and not self.failed:
            self.failed = True
            raise OSError("synthetic transient routing measurement failure")
        if self.implementation_failure_at == sample_index:
            raise ValueError("synthetic routing implementation failure")
        latent = torch.zeros((1, 1, 2, 2), dtype=torch.float16)
        values = (0.25, 0.5, 0.75, 1.0)
        return RuntimeRoutingReferenceMeasurement(
            candidate_id="routing_stqr",
            runtime_config_digest="8" * 64,
            model_id="registered-development-model",
            model_revision="registered-development-revision",
            sample_index=sample_index,
            callback_indices=tuple(range(20)),
            previous_write_latent=latent,
            routing_write_latent=latent.clone(),
            semantic_rgb=torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            texture_gradient_values=values,
            texture_spatial_shape=(2, 2),
            response_ratio_values=values,
            response_spatial_shape=(2, 2),
            sensitivity_ratio_values=values,
            sensitivity_spatial_shape=(2, 2),
            public_probe_domain_digest="7" * 64,
            public_probe_values_float32_be_sha256="6" * 64,
            nominal_relative_probe_step=0.001,
            actual_probe_step=0.001,
        )


class _ReferencePromptBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def set_development_generation_prompts(self, prompt: str) -> None:
        self.prompts.append(prompt)


def _reference_builder(
    runner: DevelopmentExplorationRunner,
    store: DevelopmentPersistentStore,
    cursor,
    runtime: _ReferenceMeasurementRuntime,
) -> DevelopmentProductionInputBuilder:
    builder = object.__new__(DevelopmentProductionInputBuilder)
    builder.prompts = SimpleNamespace(
        entries=tuple(
            SimpleNamespace(
                cluster_ordinal=index,
                prompt=f"development reference prompt {index}",
                generation_seed=index,
            )
            for index in range(64)
        ),
        digest="9" * 64,
    )
    builder.protocol = runner.protocol
    builder.authority = runner.intent_authority
    builder.root_key = "development-runner-cpu-wiring-key"
    builder.hf_token = "development-test-token"
    builder.runtime = runtime
    builder.store = store
    builder.session_cursor = cursor
    builder.runner = runner
    return builder


def _advance_session_cursor_to_routing_reference(
    runner: DevelopmentExplorationRunner,
    store: DevelopmentPersistentStore,
    lease,
    cursor,
    *,
    now_epoch_seconds: int,
) -> None:
    for _ in range(10):
        unit = runner.protocol.unit_roster[cursor.next_unit_index]
        intent = runner.create_operational_intent(
            lease,
            cursor,
            now_epoch_seconds=now_epoch_seconds,
        )
        preflight = unit.phase == "development_environment_preflight"
        roles = (
            ("content_embedder",)
            if preflight
            else REQUIRED_METHOD_RESPONSIBILITIES
        )
        receipt = DevelopmentOperationalReceipt(
            operational_role=(
                "environment_runtime_throughput_preflight"
                if preflight
                else "full_chain_wiring_smoke"
            ),
            source_cluster_ordinal=unit.source_cluster_ordinal,
            case_ids=(
                runner.protocol.preflight.case_ids
                if preflight
                else ("all_thirteen_responsibility_wiring",)
            ),
            responsibility_result_digests=tuple(
                (role, sha256(role.encode()).hexdigest()) for role in roles
            ),
            elapsed_seconds=0.25,
            runtime_config_digest=(
                runner.runtime_adapter.session.runtime_config_digest
            ),
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
        )
        runner.commit_operational_receipt(
            lease,
            cursor,
            intent,
            receipt,
            now_epoch_seconds=now_epoch_seconds + 1,
            raw_secret_values=(),
        )


def _commit_frozen_unit_indexes(
    runner: DevelopmentExplorationRunner,
    store: DevelopmentPersistentStore,
    unit_indexes: tuple[int, ...],
    *,
    session_id: str,
) -> None:
    lease = store.acquire_lease(
        session_id=session_id,
        now_epoch_seconds=100,
        lease_duration_seconds=80_000,
    )
    unit_input = _input()
    for unit_index in unit_indexes:
        intent = store.create_intent(
            lease,
            unit_id=f"development_unit_{unit_index:04d}",
            unit_index=unit_index,
            attempt_index=0,
            parent_attempt_intent_digest=None,
            now_epoch_seconds=101,
        )
        record = runner._execute_unit(unit_index, unit_input).record
        store.commit_unit(
            lease,
            intent,
            record=record,
            now_epoch_seconds=102,
        )


def _commit_hf_primary_null_fixture_records(
    store: DevelopmentPersistentStore,
    plan,
) -> None:
    lease = store.acquire_lease(
        session_id="hf_threshold_replay_session",
        now_epoch_seconds=100,
        lease_duration_seconds=80_000,
    )
    detector_bindings = {
        fold.fold_index: _threshold_material(plan, fold.fold_index)[1]
        for fold in plan.folds
    }
    assignments = {
        item.identity.source_cluster_id: item for item in plan.input_manifest.assignments
    }
    for binding in store.registered_unit_bindings:
        if (
            binding.responsibility_id != "hf_detector"
            or binding.content_branch_id != "hf_only"
        ):
            continue
        source_cluster_id = binding.analysis_unit_identity.source_cluster_id
        fold = next(
            item
            for item in plan.folds
            if source_cluster_id in item.fit_source_cluster_ids
        )
        intent = store.create_intent(
            lease,
            unit_id=binding.unit_id,
            unit_index=binding.unit_index,
            attempt_index=0,
            parent_attempt_intent_digest=None,
            now_epoch_seconds=101,
        )
        record = _primary_null_record(
            assignments[source_cluster_id],
            index=binding.unit_index,
            score=float(binding.source_cluster_ordinal) / 100.0,
            split_manifest_digest=plan.input_manifest.digest(),
            detector_binding=detector_bindings[fold.fold_index],
        )
        metric_observation = dict(record.metric_observation)
        metric_observation["geometry_case_id"] = binding.geometry_case_id
        metric_without_digest = dict(metric_observation)
        metric_without_digest.pop("observation_digest")
        metric_observation["observation_digest"] = (
            canonical_development_value_digest(metric_without_digest)
        )
        record = _redigest_scientific_record(
            record,
            phase=binding.phase,
            analysis_unit_identity=intent.analysis_unit_identity,
            geometry_case_id=binding.geometry_case_id,
            maximum_duration_seconds=binding.maximum_duration_seconds,
            metric_observation=metric_observation,
        )
        store.commit_unit(
            lease,
            intent,
            record=record,
            now_epoch_seconds=102,
        )


def _input() -> DevelopmentUnitInput:
    values = tuple((index - 16) / 16.0 for index in range(32))
    shape = (1, 2, 4, 4)
    hf_result = hf_detector(
        HfDetectionObservation.from_public_image_encoding(values, shape),
        "development-runner-cpu-wiring-key",
    )
    lf_result = lf_detector(
        LfDetectionObservation.from_public_image_encoding(values, shape),
        "development-runner-cpu-wiring-key",
    )
    return DevelopmentUnitInput(
        registered_root_key="development-runner-cpu-wiring-key",
        wrong_key_index=0,
        base_latent=torch.linspace(-1.0, 1.0, steps=48).reshape(1, 3, 4, 4).to(torch.float16),
        routing_observations=_observations(),
        mixing_coefficient=0.5,
        combination_function_id="weighted_hf_lf_standardized_score",
        hf_null=BranchNullCalibration(
            branch="hf",
            detector_identity=hf_result.detector_identity,
            partition_identity="development_content_null_cpu_wiring",
            records=(
                NullScoreRecord(hf_result.hf_score - 0.2, "null_cluster_a", "null_sample_a"),
                NullScoreRecord(hf_result.hf_score + 0.2, "null_cluster_b", "null_sample_b"),
            ),
        ),
        lf_null=BranchNullCalibration(
            branch="lf",
            detector_identity=lf_result.detector_identity,
            partition_identity="development_content_null_cpu_wiring",
            records=(
                NullScoreRecord(lf_result.lf_score - 0.2, "null_cluster_a", "null_sample_a"),
                NullScoreRecord(lf_result.lf_score + 0.2, "null_cluster_b", "null_sample_b"),
            ),
        ),
        epsilon_inlier=0.8,
        geometry_reliability_thresholds=_reliability_thresholds(),
        provisional_threshold=None,
        cross_fit_plan=None,
        development_tau_rescue=None,
    )


def _first_scientific_unit_index(
    runner: DevelopmentExplorationRunner,
    responsibility_id: str,
) -> int:
    return next(
        unit.unit_index
        for unit in runner.protocol.unit_roster
        if unit.phase == "scientific_breadth"
        and unit.responsibility_id == responsibility_id
    )


@pytest.mark.quick
def test_unfitted_geometry_inputs_are_limited_to_exploratory_responsibilities() -> None:
    unfitted = replace(
        _input(),
        epsilon_inlier=None,
        geometry_reliability_thresholds=None,
    )

    unfitted.validate("geometric_transform_estimator")
    unfitted.validate("geometry_reliability")
    with pytest.raises(
        DevelopmentRunnerError,
        match="development reliability thresholds exact type required",
    ):
        unfitted.validate("image_rectifier")


@pytest.mark.quick
def test_first_breadth_units_call_real_key_router_and_carrier_methods() -> None:
    runner = _runner()
    key = runner._execute_unit(
        _first_scientific_unit_index(runner, "key_schedule"), _input()
    )
    router = runner._execute_unit(
        _first_scientific_unit_index(runner, "content_router"), _input()
    )
    low_frequency_runner = _runner()
    low_frequency = low_frequency_runner._execute_unit(
        _first_scientific_unit_index(low_frequency_runner, "lf_carrier"), _input()
    )
    high_frequency_runner = _runner()
    high_frequency = high_frequency_runner._execute_unit(
        _first_scientific_unit_index(high_frequency_runner, "hf_carrier"), _input()
    )

    assert key.record.responsibility_id == "key_schedule"
    assert dict(key.record.metric_observation["sufficient_statistics"])[
        "key_attribution_separation"
    ] == 1.0
    assert router.record.routing_trace["routing_identity"]
    assert router.record.routing_trace["routing_comparison_eligible"] is False
    assert low_frequency.record.operation_result_payload["candidate_id"] == "lf_low_pass"
    assert high_frequency.record.operation_result_payload["candidate_id"] == "hf_sparse_tail"
    assert all(result.record.module_outcome is None for result in (key, router, low_frequency, high_frequency))


@pytest.mark.quick
def test_public_runner_does_not_accept_caller_prerequisite_outcomes() -> None:
    assert not hasattr(DevelopmentExplorationRunner, "execute_unit")


@pytest.mark.quick
def test_result_serialization_is_bound_to_exact_responsibility_type() -> None:
    runner = _runner()
    low_frequency = runner._execute_unit(
        _first_scientific_unit_index(runner, "lf_detector"), _input()
    )
    result = runner.adapter.detect_lf(
        LfDetectionObservation.from_public_image_encoding(
            tuple(_input().base_latent.reshape(-1).tolist()),
            tuple(_input().base_latent.shape),
        ),
        _input().registered_root_key,
    ).result
    assert _safe_result_payload("lf_detector", result)["lf_score"] == (
        result.lf_score
    )
    with pytest.raises(
        DevelopmentRunnerError,
        match="differs from responsibility contract",
    ):
        _safe_result_payload("hf_detector", result)
    assert low_frequency.record.operation_result_payload["candidate_id"] == "lf_low_pass"


@pytest.mark.quick
def test_content_embedding_unit_uses_actual_runtime_write_and_vae() -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == "content_embedder"
        and item.content_branch_id == "hf_only"
    )
    result = runner._execute_unit(
        unit.unit_index, _input()
    )

    assert result.record.responsibility_id == "content_embedder"
    assert result.record.provenance_trace["runtime_config_digest"] == (
        runner.runtime_adapter.session.runtime_config_digest
    )
    values = dict(result.record.metric_observation["sufficient_statistics"])
    assert values["realized_total_relative_l2"] >= 0.0
    assert runner.runtime_adapter._backend.run_calls == 2


@pytest.mark.quick
def test_clean_content_embedding_control_performs_no_hidden_write() -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == "content_embedder"
        and item.content_branch_id == "clean_control"
    )

    result = runner._execute_unit(unit.unit_index, _input())

    assert runner.runtime_adapter._backend.run_calls == 0
    assert result.record.operation_result_payload["control_identity"] == (
        "development_clean_no_write_control"
    )
    assert result.record.operation_result_payload["realized_relative_l2"] == 0.0
    assert result.record.operation_result_payload["embedding_result_identity"] is None


@pytest.mark.quick
def test_router_comparison_executes_both_frozen_runtime_arms() -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == "content_router"
        and item.source_cluster_ordinal == 0
        and item.content_branch_id == "lf_hf_routed_combination"
    )

    result = runner._execute_unit(unit.unit_index, _input())

    assert runner.runtime_adapter._backend.run_calls == 4
    assert result.record.routing_trace["routing_comparison_eligible"] is True
    assert result.record.routing_trace["adaptive_detector_identity"] == (
        result.record.routing_trace["uniform_control_detector_identity"]
    )
    assert result.record.routing_trace["adaptive_detector_config_digest"] == (
        result.record.routing_trace["uniform_control_detector_config_digest"]
    )
    assert result.record.routing_trace["routing_score_role"] == (
        "hf_only_public_content_operation"
    )
    assert result.record.branch_score_trace["function_id"] is None


@pytest.mark.quick
def test_preflight_calls_real_runtime_without_scientific_coverage() -> None:
    runner = _runner()

    receipt = runner.execute_preflight_cluster(0, _input())

    assert receipt.operational_role == "environment_runtime_throughput_preflight"
    assert receipt.case_ids == runner.protocol.preflight.case_ids
    assert receipt.counts_as_scientific_coverage is False
    assert receipt.scientific_claims_supported is False
    assert receipt.elapsed_seconds >= 0.0
    assert runner.runtime_adapter._backend.run_calls == 2


@pytest.mark.quick
def test_wiring_smoke_dispatches_all_responsibilities_without_threshold_fit(
) -> None:
    runner = _runner()
    receipt = runner.execute_wiring_smoke_cluster(
        0,
        {role: _input() for role in REQUIRED_METHOD_RESPONSIBILITIES},
    )

    assert receipt.operational_role == "full_chain_wiring_smoke"
    assert tuple(role for role, _ in receipt.responsibility_result_digests) == (
        REQUIRED_METHOD_RESPONSIBILITIES
    )
    assert receipt.counts_as_scientific_coverage is False
    assert receipt.scientific_claims_supported is False
    assert runner.runtime_adapter._backend.run_calls == 2


@pytest.mark.quick
def test_first_wiring_receipt_uses_real_runner_record_and_committed_bundle(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    now = int(time.time())
    lease = store.acquire_lease(
        session_id="real_wiring_receipt_session",
        now_epoch_seconds=now,
        lease_duration_seconds=600,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=now)
    for _ in range(2):
        unit = runner.protocol.unit_roster[cursor.next_unit_index]
        intent = runner.create_operational_intent(
            lease, cursor, now_epoch_seconds=now + 1
        )
        receipt = DevelopmentOperationalReceipt(
            operational_role="environment_runtime_throughput_preflight",
            source_cluster_ordinal=unit.source_cluster_ordinal,
            case_ids=runner.protocol.preflight.case_ids,
            responsibility_result_digests=(("content_embedder", "d" * 64),),
            elapsed_seconds=0.25,
            runtime_config_digest=runner.runtime_adapter.session.runtime_config_digest,
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
        )
        runner.commit_operational_receipt(
            lease,
            cursor,
            intent,
            receipt,
            now_epoch_seconds=now + 2,
            raw_secret_values=(),
        )
    unit = runner.protocol.unit_roster[cursor.next_unit_index]
    assert unit.unit_index == 2
    intent = runner.create_operational_intent(
        lease, cursor, now_epoch_seconds=now + 3
    )
    receipt = runner.execute_wiring_smoke_cluster(
        unit.source_cluster_ordinal,
        {role: _input() for role in REQUIRED_METHOD_RESPONSIBILITIES},
    )
    committed = runner.commit_operational_receipt(
        lease,
        cursor,
        intent,
        receipt,
        now_epoch_seconds=max(now + 4, int(time.time())),
        raw_secret_values=("development-runner-cpu-wiring-key",),
    )
    recovered = store.open_session_cursor(
        lease, now_epoch_seconds=max(now + 5, int(time.time()))
    )

    assert committed.unit_index == 2
    assert committed.record_kind == "development_operational_check"
    assert recovered.operational_records[-1].record_id == committed.record_id
    assert recovered.next_unit_index == 3


@pytest.mark.quick
def test_wiring_conditional_call_records_real_public_result_without_science_alias() -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == "conditional_recovery_decision"
    )

    result = runner._execute_wiring_conditional_call(unit, _input())
    payload = _safe_result_payload("conditional_recovery_decision", result)

    assert type(result) is ConditionalRecoveryResult
    assert payload["decision_identity_digest"] == result.decision_identity_digest
    assert payload["detector_binding_digest"] == result.detector_binding_digest
    assert payload["threshold_identity"] == result.threshold_identity
    assert payload["calibration_identity"] == "wiring_only_non_scientific"
    assert payload["joint_content_positive"] is False
    assert "source_image" not in payload
    assert "raw_content_result" not in payload
    assert "content_detector_binding" not in payload
    with pytest.raises(
        DevelopmentRunnerError,
        match="conditional recovery result lacks an explicit public record schema",
    ):
        _safe_result_payload("conditional_recovery_decision", object())


@pytest.mark.quick
def test_wiring_rectifier_uses_real_synthetic_identity_public_call_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    raw = _input()
    observed: list[str] = []
    closed: list[bool] = []

    runtime_observe = Sd35RuntimeAdapter.observe_detection_qk
    synchronize = CegWmExperimentAdapter.synchronize_qk_observation
    estimate = CegWmExperimentAdapter.estimate_geometric_transform
    assess = CegWmExperimentAdapter.assess_geometry_reliability
    rectify = CegWmExperimentAdapter.rectify_image
    backend_close = SyntheticQkBackend.close

    def observe_call(self, image):
        observed.append("runtime_qk_observation")
        return runtime_observe(self, image)

    def synchronize_call(self, observation, root_key):
        observed.append("qk_geometry_sync")
        return synchronize(self, observation, root_key)

    def estimate_call(self, observation, root_key, *, epsilon_inlier):
        observed.append("geometric_transform_estimator")
        return estimate(
            self,
            observation,
            root_key,
            epsilon_inlier=epsilon_inlier,
        )

    def assess_call(self, estimation, thresholds):
        observed.append("geometry_reliability")
        result = assess(self, estimation, thresholds)
        assert result.result.reliable is True
        assert result.result.allow_rectification is True
        return result

    def rectify_call(self, image, estimation, reliability):
        observed.append("image_rectifier")
        return rectify(self, image, estimation, reliability)

    def close_call(self):
        closed.append(True)
        return backend_close(self)

    monkeypatch.setattr(Sd35RuntimeAdapter, "observe_detection_qk", observe_call)
    monkeypatch.setattr(
        CegWmExperimentAdapter,
        "synchronize_qk_observation",
        synchronize_call,
    )
    monkeypatch.setattr(
        CegWmExperimentAdapter,
        "estimate_geometric_transform",
        estimate_call,
    )
    monkeypatch.setattr(
        CegWmExperimentAdapter,
        "assess_geometry_reliability",
        assess_call,
    )
    monkeypatch.setattr(CegWmExperimentAdapter, "rectify_image", rectify_call)
    monkeypatch.setattr(SyntheticQkBackend, "close", close_call)

    result = runner._execute_wiring_image_rectifier_call(raw)

    assert result.rectification_config_digest
    assert observed == [
        "runtime_qk_observation",
        "qk_geometry_sync",
        "geometric_transform_estimator",
        "geometry_reliability",
        "image_rectifier",
    ]
    assert closed == [True]


@pytest.mark.quick
def test_wiring_rectifier_rejects_unreliable_identity_before_real_rectification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    raw = replace(
        _input(),
        geometry_reliability_thresholds=replace(
            _reliability_thresholds(),
            gamma_gap=1_000_000.0,
            gamma_key=1_000_000.0,
        ),
    )
    rectifier_calls: list[bool] = []
    closed: list[bool] = []
    rectify = CegWmExperimentAdapter.rectify_image
    backend_close = SyntheticQkBackend.close

    def unexpected_rectify(self, image, estimation, reliability):
        rectifier_calls.append(True)
        return rectify(self, image, estimation, reliability)

    def close_call(self):
        closed.append(True)
        return backend_close(self)

    monkeypatch.setattr(
        CegWmExperimentAdapter,
        "rectify_image",
        unexpected_rectify,
    )
    monkeypatch.setattr(SyntheticQkBackend, "close", close_call)

    with pytest.raises(
        DevelopmentRunnerError,
        match="identity geometry is unreliable",
    ):
        runner._execute_wiring_image_rectifier_call(raw)

    assert rectifier_calls == []
    assert closed == [True]


@pytest.mark.quick
@pytest.mark.parametrize(
    "responsibility_id",
    (
        "lf_detector",
        "hf_detector",
        "content_detector",
        "qk_geometry_sync",
        "geometric_transform_estimator",
        "geometry_reliability",
    ),
)
def test_remaining_non_joint_responsibilities_use_real_public_calls(
    responsibility_id: str,
) -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == responsibility_id
    )
    result = runner._execute_unit(unit.unit_index, _input())

    assert result.record.responsibility_id == responsibility_id
    assert result.record.metric_observation["responsibility_id"] == responsibility_id
    assert result.record.operation_result_digest
    if responsibility_id == "content_detector":
        assert result.record.branch_score_trace["function_id"] == (
            "weighted_hf_lf_standardized_score"
        )
        assert result.record.branch_score_trace["mixing_coefficient"] == 0.5


@pytest.mark.quick
def test_real_high_frequency_unit_bridges_into_frozen_cluster_threshold_fit() -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == "hf_detector"
        and item.content_branch_id == "hf_only"
    )
    result = runner._execute_unit(unit.unit_index, _input())
    record = result.record
    authority = runner.protocol.threshold_detector_authority
    identity = record.analysis_unit_identity
    assert record.responsibility_id == "hf_detector"
    assert record.content_branch_id == "hf_only"
    assert record.detector_trace["primary_null_detector_identity"] == (
        authority.method_detector_identity
    )
    assert record.detector_trace["primary_null_detector_config_digest"] == (
        authority.method_detector_config_digest
    )
    assert record.detector_trace["primary_null_preprocessing_identity"] == (
        authority.preprocessing_identity
    )
    assert record.key_control_trace["primary_null_control_identity"] == (
        "unwatermarked_registered_key_primary_null"
    )
    assert record.key_control_trace["registered_key_public_digest"] == (
        record.key_control_trace["primary_null_detection_key_public_digest"]
    )

    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=runner.intent_authority,
        expected_execution_intent_authority_digest=(
            runner.intent_authority.authority_digest
        ),
        expected_source_cluster_count=64,
    )
    fold_index = next(
        fold.fold_index
        for fold in plan.folds
        if identity["source_cluster_id"] in fold.fit_source_cluster_ids
    )
    manifest, detector_binding, fit_inputs = _threshold_material(
        plan,
        fold_index,
    )
    real_input = create_development_threshold_fit_input(
        expected_execution_intent_authority_digest=(
            runner.intent_authority.authority_digest
        ),
        source_record=record,
    )
    rebound_inputs = tuple(
        real_input
        if item.source_record.analysis_unit_identity["source_cluster_id"]
        == identity["source_cluster_id"]
        else item
        for item in fit_inputs
    )
    threshold = create_development_provisional_threshold(
        plan,
        expected_execution_intent_authority_digest=(
            runner.intent_authority.authority_digest
        ),
        fold_index=fold_index,
        input_manifest=manifest,
        detector_binding=detector_binding,
        fit_inputs=rebound_inputs,
    )
    assert threshold.validate(plan) == ()
    assert real_input.source_record.record_id in {
        item.source_record.record_id for item in threshold.fit_inputs
    }


@pytest.mark.quick
def test_rectifier_fail_closed_is_classified_as_scientific_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == "image_rectifier"
    )
    monkeypatch.setattr(
        runner,
        "_execute_wiring_image_rectifier_call",
        lambda _raw: pytest.fail("scientific path used wiring rectifier helper"),
    )
    with pytest.raises(DevelopmentUnitExcluded, match="fail-closed reliability"):
        runner._execute_unit(unit.unit_index, _input())


@pytest.mark.quick
def test_geometry_control_uses_frozen_attack_and_official_reliability() -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == "geometry_reliability"
        and item.source_cluster_ordinal == 0
        and item.geometry_case_id == "extreme_crop_control"
    )

    result = runner._execute_unit(unit.unit_index, _input())
    statistics = dict(result.record.metric_observation["sufficient_statistics"])

    assert result.record.geometry_case_id == "extreme_crop_control"
    assert result.record.geometry_trace["geometry_operation_identity"] == "crop"
    assert result.record.geometry_trace["geometry_reliability_status"] in {
        "reliable",
        "unreliable",
    }
    assert result.record.geometry_trace["wrong_key_geometry_estimation_identity"]
    assert result.record.geometry_trace["wrong_key_geometry_reliability_identity"]
    assert type(result.record.geometry_trace["wrong_key_geometry_reliable"]) is bool
    assert statistics["reliable_accept_rate"] == 0.0
    assert (
        statistics["unreliable_reject_rate"]
        + statistics["false_reliable_rate"]
    ) == 1.0


@pytest.mark.quick
def test_conditional_recovery_unit_delegates_once_to_governed_internal_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _development_manifest(64)
    intent = _execution_intent(manifest, run_id="development_joint_wiring")
    joint_runner = _runner(intent)
    joint_unit = next(
        item
        for item in joint_runner.protocol.unit_roster
        if item.responsibility_id == "conditional_recovery_decision"
    )
    target = intent.input_manifest.assignments[
        joint_unit.source_cluster_ordinal
    ].identity
    original_unit = internal_runner_test_module._unit
    monkeypatch.setattr(
        internal_runner_test_module,
        "_unit",
        lambda index, **kwargs: target if index == 0 else original_unit(index, **kwargs),
    )
    context, payload, operation = _internal_context(tmp_path / "internal")
    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=intent,
        expected_execution_intent_authority_digest=intent.authority_digest,
        expected_source_cluster_count=64,
    )
    fold_index = next(
        fold.fold_index
        for fold in plan.folds
        if target.source_cluster_id in fold.recovery_probe_source_cluster_ids
    )
    threshold_manifest, detector_binding, fit_inputs = _threshold_material(
        plan, fold_index
    )
    desired_tau = payload.thresholds.tau
    adjusted_inputs = []
    for index, item in enumerate(fit_inputs):
        score = desired_tau if index == 0 else desired_tau - 1.0
        operation_payload = {"primary_null_score": score}
        metric_observation = dict(item.source_record.metric_observation)
        statistics = dict(metric_observation["sufficient_statistics"])
        statistics.update(
            primary_null_score=score,
            registered_score=score + 1.0,
            wrong_key_score=score - 1.0,
        )
        metric_observation["sufficient_statistics"] = tuple(
            statistics.items()
        )
        metric_without_digest = dict(metric_observation)
        metric_without_digest.pop("observation_digest")
        metric_observation["observation_digest"] = (
            canonical_development_value_digest(metric_without_digest)
        )
        branch_score_trace = dict(item.source_record.branch_score_trace)
        branch_score_trace["hf_score"] = score
        source_record = _redigest_scientific_record(
            item.source_record,
            operation_result_payload=operation_payload,
            operation_result_digest=canonical_development_value_digest(
                operation_payload
            ),
            metric_observation=metric_observation,
            branch_score_trace=branch_score_trace,
        )
        adjusted_inputs.append(
            create_development_threshold_fit_input(
                expected_execution_intent_authority_digest=intent.authority_digest,
                source_record=source_record,
            )
        )
    threshold = create_development_provisional_threshold(
        plan,
        expected_execution_intent_authority_digest=intent.authority_digest,
        fold_index=fold_index,
        input_manifest=threshold_manifest,
        detector_binding=detector_binding,
        fit_inputs=tuple(adjusted_inputs),
    )
    joint_input = replace(
        _input(),
        provisional_threshold=threshold,
        cross_fit_plan=plan,
        development_tau_rescue=payload.thresholds.tau_rescue,
        internal_runner_context=context,
        internal_case_payload=payload,
    )

    result = joint_runner._execute_unit(joint_unit.unit_index, joint_input)

    assert result.record.responsibility_id == "conditional_recovery_decision"
    assert result.record.decision_trace["internal_validation_record_id"]
    assert result.record.threshold_trace["threshold_role"] == "development_exploratory"
    assert operation.calls == 1


@pytest.mark.quick
def test_conditional_scratch_survives_drive_limited_record_writer_and_outer_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drive_root = (tmp_path / "drive_like_persistent_root").resolve()
    cache_root = (tmp_path / "local_worker_cache").resolve()
    drive_root.mkdir(parents=True)
    cache_root.mkdir(parents=True)
    original_replace = record_writer_module.os.replace
    original_fsync = record_writer_module.os.fsync
    original_flock = record_writer_module.fcntl.flock

    def _fd_path(descriptor: int) -> Path | None:
        try:
            return Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        except OSError:
            return None

    def _under_drive(path: Path | None) -> bool:
        return path is not None and (path == drive_root or drive_root in path.parents)

    def _guarded_replace(source, destination) -> None:
        if _under_drive(Path(destination).resolve()):
            raise OSError("drive atomic replace unavailable")
        original_replace(source, destination)

    def _guarded_fsync(descriptor: int) -> None:
        path = _fd_path(descriptor)
        if _under_drive(path) and any(
            Path(frame.filename).name == "record_writer.py"
            for frame in inspect.stack()[1:5]
        ):
            raise OSError("drive fsync unavailable to internal record writer")
        original_fsync(descriptor)

    def _guarded_flock(descriptor: int, operation: int) -> None:
        if _under_drive(_fd_path(descriptor)):
            raise OSError("drive flock unavailable")
        original_flock(descriptor, operation)

    monkeypatch.setattr(record_writer_module.os, "replace", _guarded_replace)
    monkeypatch.setattr(record_writer_module.os, "fsync", _guarded_fsync)
    monkeypatch.setattr(record_writer_module.fcntl, "flock", _guarded_flock)

    blocked_context, blocked_payload, _ = _internal_context(
        drive_root / "legacy_internal_records"
    )
    with pytest.raises(OSError, match="drive flock unavailable"):
        execute_internal_case(
            blocked_context,
            unit_id=blocked_payload.source_artifact.analysis_unit_identity.unit_id,
            payload=blocked_payload,
        )

    outer_store_fixture_root = drive_root / "outer_store"
    outer_store_fixture_root.mkdir()
    outer_store = _persistence_store(outer_store_fixture_root)
    outer_lease = _persistence_lease(outer_store, session_id="drive_limited_session")
    outer_intent = _persistence_intent(outer_store, outer_lease)
    builder = object.__new__(DevelopmentProductionInputBuilder)
    builder.cache_root = cache_root
    scratch_root = builder._internal_record_scratch_root(
        unit_descriptor_digest=outer_intent.unit_descriptor_digest,
        intent=outer_intent,
    )
    assert cache_root in scratch_root.parents
    assert drive_root not in scratch_root.parents

    local_context, local_payload, _ = _internal_context(scratch_root)
    internal = execute_internal_case(
        local_context,
        unit_id=local_payload.source_artifact.analysis_unit_identity.unit_id,
        payload=local_payload,
    ).record
    operation_payload = internal.to_dict()
    outer_record = _persistence_record(outer_store, outer_intent)
    outer_record = replace(
        outer_record,
        operation_result_payload=operation_payload,
        operation_result_digest=canonical_development_value_digest(
            operation_payload
        ),
    )
    outer_record = replace(
        outer_record,
        record_id=canonical_development_value_digest(
            outer_record.payload_without_record_id()
        ),
    )
    marker = outer_store.commit_unit(
        outer_lease,
        outer_intent,
        record=outer_record,
        now_epoch_seconds=102,
    )
    shutil.rmtree(cache_root)

    recovered = outer_store.verified_terminal_scientific_evidence(
        now_epoch_seconds=103
    )
    assert marker.record_id == outer_record.record_id
    assert recovered[0][0].operation_result_payload == operation_payload
    assert recovered[0][1] == marker


@pytest.mark.quick
def test_production_runner_has_no_result_provider_or_module_result_surface() -> None:
    source = inspect.getsource(DevelopmentExplorationRunner)
    signature = inspect.signature(DevelopmentExplorationRunner._execute_unit)

    assert "result_provider" not in source
    assert "module_results" not in source
    assert "result_provider" not in signature.parameters
    assert "module_results" not in signature.parameters
    assert "execute_content_write_and_vae" in source
    assert "observe_detection_qk" in source
    assert "apply_geometric_attack" in source
    assert "execute_internal_case" in source


@pytest.mark.quick
def test_persistent_runner_requires_registered_producer_for_operational_prefix(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    lease = store.acquire_lease(
        session_id="runner_session",
        now_epoch_seconds=100,
        lease_duration_seconds=100,
    )

    with pytest.raises(
        DevelopmentRunnerError,
        match="operational unit requires its registered producer",
    ):
        runner.execute_and_commit_next_unit(
            lease,
            _input(),
            now_epoch_seconds=101,
            raw_secret_values=("development-runner-cpu-wiring-key",),
        )
    assert store.recover(now_epoch_seconds=102).committed_units == ()


@pytest.mark.quick
def test_preflight_and_wiring_share_commit_recovery_before_routing_reference(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    lease = store.acquire_lease(
        session_id="operational_prefix_session",
        now_epoch_seconds=100,
        lease_duration_seconds=100,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=101)
    for _ in range(10):
        unit = runner.protocol.unit_roster[cursor.next_unit_index]
        intent = runner.create_operational_intent(
            lease,
            cursor,
            now_epoch_seconds=102,
        )
        is_preflight = unit.phase == "development_environment_preflight"
        roles = (
            ("content_embedder",)
            if is_preflight
            else REQUIRED_METHOD_RESPONSIBILITIES
        )
        receipt = DevelopmentOperationalReceipt(
            operational_role=(
                "environment_runtime_throughput_preflight"
                if is_preflight
                else "full_chain_wiring_smoke"
            ),
            source_cluster_ordinal=unit.source_cluster_ordinal,
            case_ids=(
                runner.protocol.preflight.case_ids
                if is_preflight
                else ("all_thirteen_responsibility_wiring",)
            ),
            responsibility_result_digests=tuple(
                (role, sha256(role.encode("utf-8")).hexdigest())
                for role in roles
            ),
            elapsed_seconds=0.25,
            runtime_config_digest=(
                runner.runtime_adapter.session.runtime_config_digest
            ),
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
        )
        runner.commit_operational_receipt(
            lease,
            cursor,
            intent,
            receipt,
            now_epoch_seconds=103,
            raw_secret_values=(),
        )
    assert cursor.next_unit_index == 10
    assert len(cursor.operational_records) == 10
    recovered = store.open_session_cursor(lease, now_epoch_seconds=104)
    assert recovered.next_unit_index == 10
    assert len(recovered.operational_records) == 10
    assert runner.protocol.unit_roster[recovered.next_unit_index].phase == (
        "development_routing_reference_fit"
    )
    with pytest.raises(
        DevelopmentRunnerError,
        match="operational unit requires its registered producer",
    ):
        runner.execute_and_commit_session_unit(
            lease,
            recovered,
            _input(),
            now_epoch_seconds=105,
            raw_secret_values=(),
        )


@pytest.mark.quick
def test_production_routing_reference_recovers_measurement_retry_across_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    initial_epoch = int(time.time())
    first_lease = store.acquire_lease(
        session_id="routing_reference_measurement_failure_session",
        now_epoch_seconds=initial_epoch,
        lease_duration_seconds=10,
    )
    first_cursor = store.open_session_cursor(
        first_lease,
        now_epoch_seconds=initial_epoch,
    )
    _advance_session_cursor_to_routing_reference(
        runner,
        store,
        first_lease,
        first_cursor,
        now_epoch_seconds=initial_epoch,
    )
    runtime = _ReferenceMeasurementRuntime(fail_once_at=17)
    backend = _ReferencePromptBackend()
    first_builder = _reference_builder(
        runner,
        store,
        first_cursor,
        runtime,
    )
    monkeypatch.setattr(
        worker_inputs_module.time,
        "time",
        lambda: float(initial_epoch),
    )

    assert first_builder.prepare_routing_reference_fit(
        backend,
        lambda seed: torch.tensor([float(seed)]),
        lease=first_lease,
        soft_stop_epoch_seconds=initial_epoch + 100,
    ) == "retryable_stop"
    first_recovery = store.recover(now_epoch_seconds=initial_epoch + 1)
    retry_marker = first_recovery.committed_units[-1]
    assert retry_marker.unit_index == 27
    assert retry_marker.attempt_index == 0
    assert retry_marker.attempt_disposition == "retryable_resource_failure"
    assert first_cursor.next_unit_index == 27
    assert store.next_attempt_index(retry_marker.unit_id) == 1

    resumed_epoch = initial_epoch + 11
    second_lease = store.acquire_lease(
        session_id="routing_reference_measurement_resume_session",
        now_epoch_seconds=resumed_epoch,
        lease_duration_seconds=100,
    )
    second_cursor = store.open_session_cursor(
        second_lease,
        now_epoch_seconds=resumed_epoch,
    )
    second_builder = _reference_builder(
        runner,
        store,
        second_cursor,
        runtime,
    )
    monkeypatch.setattr(
        worker_inputs_module.time,
        "time",
        lambda: float(resumed_epoch),
    )

    assert second_builder.prepare_routing_reference_fit(
        backend,
        lambda seed: torch.tensor([float(seed)]),
        lease=second_lease,
        soft_stop_epoch_seconds=resumed_epoch + 100,
    ) == "complete_success"
    records = store.verified_terminal_routing_reference_records(
        now_epoch_seconds=resumed_epoch + 1
    )
    resumed_record = next(item for item in records if item.unit_index == 27)
    assert len(records) == 64
    assert len({item.record_id for item in records}) == 64
    assert resumed_record.attempt_index == 1
    assert resumed_record.retry_parent_intent_digest == retry_marker.intent_digest
    assert runtime.sample_indexes.count(17) == 2


@pytest.mark.quick
def test_production_routing_reference_timeout_exhaustion_commits_terminal_and_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_epoch_time = time.time
    entrypoint_clock = {"now": int(real_epoch_time())}
    entrypoint_root = tmp_path / "production_entrypoint"
    entrypoint_root.mkdir()
    package = entrypoint_root / "development_execution_package.zip"
    package.write_bytes(b"production entrypoint package fixture")
    production_base_latent = torch.linspace(
        -1.0,
        1.0,
        steps=32,
        dtype=torch.float32,
    ).reshape(1, 2, 4, 4).to(torch.float16)
    measurement_state = {"calls": 0, "reference_failures": 0}

    class FixedSemanticObservationProducer:
        def __init__(self, **_keywords: object) -> None:
            pass

        def observe(
            self,
            _routing_rgb: torch.Tensor,
            _prompt: str,
        ) -> SpatialRoutingObservation:
            return _observations().semantic

    class ProductionCpuBackend(_CombinedCpuWiringBackend):
        def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
            decoded = self.content.vae_decode(latent)
            return torch.cat((decoded, decoded[:, :1]), dim=1)

    original_runtime_initialize = Sd35RuntimeAdapter.initialize
    qk_fixture_runtime = create_runtime_adapter(_CombinedCpuWiringBackend())
    qk_fixture_session = qk_fixture_runtime.initialize("cpu")
    qk_observation = qk_fixture_runtime.observe_detection_qk(
        torch.zeros(
            (
                1,
                3,
                qk_fixture_session.image_height,
                qk_fixture_session.image_width,
            ),
            dtype=torch.uint8,
        )
    )
    qk_fixture_runtime.close()
    original_runtime_qk_observation = Sd35RuntimeAdapter.observe_detection_qk

    def initialize_cpu(self, _requested_device):
        return original_runtime_initialize(self, "cpu")

    def controlled_measurement(
        self,
        base_latent: torch.Tensor,
        *,
        sample_index: int,
    ) -> RuntimeRoutingReferenceMeasurement:
        is_reference_attempt = (
            measurement_state["calls"] >= 10
            and sample_index == 0
            and measurement_state["reference_failures"] < 3
        )
        measurement_state["calls"] += 1
        if is_reference_attempt:
            measurement_state["reference_failures"] += 1
            raise OSError("controlled production reference resource failure")
        measurement = _ReferenceMeasurementRuntime().measure_generation_routing_reference_inputs(
            base_latent,
            sample_index=sample_index,
        )
        return replace(
            measurement,
            runtime_config_digest=self._configuration.runtime_config_digest,
            model_id=self._configuration.model_id,
            model_revision=self._configuration.model_revision,
            callback_indices=tuple(range(self._configuration.inference_steps)),
        )

    def cpu_runtime_factory(_backend, _configuration_path):
        return create_runtime_adapter(ProductionCpuBackend())

    def controlled_qk_observation(self, image: torch.Tensor):
        if isinstance(self._backend, ProductionCpuBackend):
            return qk_observation
        return original_runtime_qk_observation(self, image)

    def ticking_entrypoint_time() -> float:
        entrypoint_clock["now"] += 1
        return float(entrypoint_clock["now"])

    original_execute_claimed = (
        DevelopmentExplorationRunner.execute_and_commit_claimed_session_unit
    )
    original_terminal_commit = (
        DevelopmentExplorationRunner.commit_claimed_terminal_failure
    )
    routing_dependency_commit = {}

    def execute_one_scientific_then_soft_stop(self, *arguments, **keywords):
        result = original_execute_claimed(self, *arguments, **keywords)
        entrypoint_clock["now"] += (
            development_entrypoint.SOFT_STOP_SECONDS + 1
        )
        return result

    def commit_routing_dependency_then_soft_stop(
        self,
        *arguments,
        **keywords,
    ):
        result = original_terminal_commit(self, *arguments, **keywords)
        if (
            result.record.unit_index == 140
            and keywords["failure_class"] == "dependency_blocked"
            and keywords["failure_reason"]
            == "verified_dependency_input_incomplete"
        ):
            assert result.committed is not None
            routing_dependency_commit["result"] = result
            routing_dependency_commit["next_unit_index"] = (
                arguments[1].next_unit_index
            )
            routing_dependency_commit["successful_reference_count"] = len(
                arguments[1].routing_reference_records
            )
            routing_dependency_commit["terminal_reference_count"] = len(
                arguments[1].terminal_routing_reference_records
            )
            entrypoint_clock["now"] += (
                development_entrypoint.SOFT_STOP_SECONDS + 1
            )
        return result

    monkeypatch.setattr(
        worker_inputs_module,
        "DevelopmentSemanticObservationProducer",
        FixedSemanticObservationProducer,
    )
    monkeypatch.setattr(
        development_entrypoint,
        "Sd35PipelineBackend",
        lambda **_keywords: _ReferencePromptBackend(),
    )
    monkeypatch.setattr(
        development_entrypoint,
        "create_runtime_adapter",
        cpu_runtime_factory,
    )
    monkeypatch.setattr(Sd35RuntimeAdapter, "initialize", initialize_cpu)
    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "measure_generation_routing_reference_inputs",
        controlled_measurement,
    )
    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "observe_detection_qk",
        controlled_qk_observation,
    )
    monkeypatch.setattr(
        development_entrypoint,
        "_base_latent",
        lambda _seed, **_keywords: production_base_latent,
    )
    monkeypatch.setattr(
        development_entrypoint,
        "_build_or_verify_package",
        lambda *_arguments: package,
    )
    monkeypatch.setattr(
        development_entrypoint,
        "_environment_digest",
        lambda: "b" * 64,
    )
    monkeypatch.setattr(
        development_entrypoint.time,
        "time",
        ticking_entrypoint_time,
    )
    monkeypatch.setattr(
        development_runner_module,
        "time",
        lambda: float(entrypoint_clock["now"]),
    )
    monkeypatch.setattr(
        development_entrypoint.torch.cuda,
        "get_device_name",
        lambda _index: "cpu-production-fixture",
    )
    monkeypatch.setattr(
        development_entrypoint.torch.cuda,
        "max_memory_allocated",
        lambda _index: 1,
    )
    monkeypatch.setattr(
        DevelopmentExplorationRunner,
        "execute_and_commit_claimed_session_unit",
        execute_one_scientific_then_soft_stop,
    )

    production_results = []
    for session_ordinal in range(3):
        exit_code, result = (
            development_entrypoint.execute_development_exploration_session(
                repository_root=ROOT,
                expected_revision="a" * 40,
                persistent_root=entrypoint_root / "persistent",
                cache_root=entrypoint_root / "cache",
                run_id="production_routing_terminal_recovery",
                session_id=f"production_session_{session_ordinal}",
                environment={
                    "CEG_WM_ROOT_KEY": "development-runner-cpu-wiring-key",
                    "HF_TOKEN": "development-test-token",
                },
            )
        )
        assert exit_code == 0, (result, measurement_state)
        production_results.append(result)
        if session_ordinal < 2:
            assert result["termination_reason"] == (
                "resource_retry_after_committed_reference"
            )
        entrypoint_clock["now"] += (
            development_entrypoint.HARD_SESSION_CAP_SECONDS
        )

    marker_root = (
        entrypoint_root
        / "persistent"
        / "production_routing_terminal_recovery"
        / "markers"
    )
    marker_payloads = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(marker_root.glob("*.COMMITTED.json"))
    )
    reference_attempts = tuple(
        item
        for item in marker_payloads
        if item["unit_index"] == 10
    )
    assert tuple(item["attempt_disposition"] for item in reference_attempts) == (
        "retryable_resource_failure",
        "retryable_resource_failure",
        "final_failure",
    )
    assert reference_attempts[1]["parent_attempt_intent_digest"] == (
        reference_attempts[0]["intent_digest"]
    )
    assert reference_attempts[2]["parent_attempt_intent_digest"] == (
        reference_attempts[1]["intent_digest"]
    )
    assert sum(
        item["record_kind"] == "development_routing_reference_fit"
        and item["attempt_disposition"] == "success"
        for item in marker_payloads
    ) == 63
    first_scientific_marker = next(
        item for item in marker_payloads if item["unit_index"] == 74
    )
    assert first_scientific_marker["attempt_disposition"] == "success"
    assert production_results[-1]["termination_reason"] == (
        "soft_stop_after_current_unit"
    )

    measurement_calls_before_routing_dependency = measurement_state["calls"]
    monkeypatch.setattr(
        DevelopmentExplorationRunner,
        "execute_and_commit_claimed_session_unit",
        original_execute_claimed,
    )
    monkeypatch.setattr(
        DevelopmentExplorationRunner,
        "commit_claimed_terminal_failure",
        commit_routing_dependency_then_soft_stop,
    )
    exit_code, routing_dependency_result = (
        development_entrypoint.execute_development_exploration_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=entrypoint_root / "persistent",
            cache_root=entrypoint_root / "cache",
            run_id="production_routing_terminal_recovery",
            session_id="production_session_routing_dependency",
            environment={
                "CEG_WM_ROOT_KEY": "development-runner-cpu-wiring-key",
                "HF_TOKEN": "development-test-token",
            },
        )
    )
    assert exit_code == 0
    assert routing_dependency_result["termination_reason"] == (
        "soft_stop_after_current_unit"
    )
    assert measurement_state["calls"] == (
        measurement_calls_before_routing_dependency
    )

    blocked_result = routing_dependency_commit["result"]
    blocked_record = blocked_result.record
    blocked_record.validate()
    assert blocked_record.unit_index == 140
    assert blocked_record.execution_status == "failed"
    assert blocked_record.failure_class == "dependency_blocked"
    assert blocked_record.failure_reason == (
        "verified_dependency_input_incomplete"
    )
    assert routing_dependency_commit["next_unit_index"] == 141
    assert routing_dependency_commit["successful_reference_count"] == 63
    assert routing_dependency_commit["terminal_reference_count"] == 64
    blocked_marker = blocked_result.committed
    assert blocked_marker is not None
    assert blocked_marker.unit_index == 140
    assert blocked_marker.attempt_disposition == "final_failure"

    final_marker_payloads = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(marker_root.glob("*.COMMITTED.json"))
    )
    assert sum(
        item["record_kind"] == "development_routing_reference_fit"
        and item["attempt_disposition"] == "success"
        for item in final_marker_payloads
    ) == 63
    blocked_marker_payload = next(
        item for item in final_marker_payloads if item["unit_index"] == 140
    )
    assert blocked_marker_payload["record_id"] == blocked_record.record_id
    bundle_path = (
        marker_root.parent
        / "bundles"
        / f"sha256_{blocked_marker_payload['bundle_sha256']}.zip"
    )
    assert bundle_path.is_file()
    assert bundle_path.stat().st_size == blocked_marker_payload["bundle_bytes"]
    assert sha256(bundle_path.read_bytes()).hexdigest() == (
        blocked_marker_payload["bundle_sha256"]
    )


@pytest.mark.quick
def test_production_routing_reference_commits_terminal_implementation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    session_epoch = int(time.time())
    lease = store.acquire_lease(
        session_id="routing_reference_implementation_failure_session",
        now_epoch_seconds=session_epoch,
        lease_duration_seconds=100,
    )
    cursor = store.open_session_cursor(
        lease,
        now_epoch_seconds=session_epoch,
    )
    _advance_session_cursor_to_routing_reference(
        runner,
        store,
        lease,
        cursor,
        now_epoch_seconds=session_epoch,
    )
    builder = _reference_builder(
        runner,
        store,
        cursor,
        _ReferenceMeasurementRuntime(implementation_failure_at=0),
    )
    monkeypatch.setattr(
        worker_inputs_module.time,
        "time",
        lambda: float(session_epoch),
    )

    assert builder.prepare_routing_reference_fit(
        _ReferencePromptBackend(),
        lambda seed: torch.tensor([float(seed)]),
        lease=lease,
        soft_stop_epoch_seconds=session_epoch + 100,
    ) == "terminal_blocked"
    marker = next(
        item
        for item in store.recover(
            now_epoch_seconds=session_epoch + 1
        ).committed_units
        if item.unit_index == 10
    )
    record = store._verify_committed(marker)

    assert record.unit_index == 10
    assert record.execution_status == "failed"
    assert record.failure_class == "implementation_failure"
    assert record.failure_reason == "builtins.ValueError"
    assert marker.attempt_disposition == "final_failure"
    assert cursor.next_unit_index == 74
    assert len(cursor.terminal_routing_reference_records) == 64


@pytest.mark.quick
def test_intent_precedes_production_build_and_first_scientific_commit(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    now = int(time.time())
    lease = store.acquire_lease(
        session_id="intent_first_scientific_session",
        now_epoch_seconds=now,
        lease_duration_seconds=600,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=now)
    for _ in range(10):
        unit = runner.protocol.unit_roster[cursor.next_unit_index]
        intent = runner.create_operational_intent(
            lease, cursor, now_epoch_seconds=now + 1
        )
        preflight = unit.phase == "development_environment_preflight"
        roles = ("content_embedder",) if preflight else REQUIRED_METHOD_RESPONSIBILITIES
        receipt = DevelopmentOperationalReceipt(
            operational_role=(
                "environment_runtime_throughput_preflight"
                if preflight
                else "full_chain_wiring_smoke"
            ),
            source_cluster_ordinal=unit.source_cluster_ordinal,
            case_ids=(
                runner.protocol.preflight.case_ids
                if preflight
                else ("all_thirteen_responsibility_wiring",)
            ),
            responsibility_result_digests=tuple(
                (role, sha256(role.encode()).hexdigest()) for role in roles
            ),
            elapsed_seconds=0.25,
            runtime_config_digest=runner.runtime_adapter.session.runtime_config_digest,
            counts_as_scientific_coverage=False,
            scientific_claims_supported=False,
        )
        runner.commit_operational_receipt(
            lease,
            cursor,
            intent,
            receipt,
            now_epoch_seconds=now + 2,
            raw_secret_values=(),
        )
    for _ in range(64):
        intent = store.create_session_intent(
            cursor, lease, now_epoch_seconds=now + 3
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=_routing_reference_record(intent),
            now_epoch_seconds=now + 4,
        )
    unit = runner.protocol.unit_roster[cursor.next_unit_index]
    assert unit.responsibility_id == "key_schedule"
    attempt_started = time.monotonic()
    intent = runner.create_scientific_intent(
        lease, cursor, now_epoch_seconds=now + 5
    )
    builder = object.__new__(DevelopmentProductionInputBuilder)
    builder.protocol = runner.protocol
    builder.authority = runner.intent_authority
    builder.root_key = "development-runner-cpu-wiring-key"
    builder.runner = runner
    builder.store = store
    builder.session_cursor = cursor
    builder.cache_root = tmp_path / "cache"
    builder._routing_observations_by_cluster = {}
    unit_input = builder.build(
        unit,
        _input().base_latent,
        intent=intent,
        now_epoch_seconds=now + 5,
    )
    result = runner.execute_and_commit_claimed_session_unit(
        lease,
        cursor,
        intent,
        unit_input,
        attempt_started_monotonic=attempt_started,
        raw_secret_values=("development-runner-cpu-wiring-key",),
    )

    assert result.record.execution_status == "success"
    assert result.record.unit_index == 74
    assert result.committed is not None
    assert result.committed.committed_at_utc > intent.created_at_utc
    assert store.recover(now_epoch_seconds=now + 6).committed_units[-1] == (
        result.committed
    )
    terminal_started = time.monotonic()
    terminal_intent = runner.create_scientific_intent(
        lease, cursor, now_epoch_seconds=now + 7
    )
    terminal = runner.commit_claimed_terminal_failure(
        lease,
        cursor,
        terminal_intent,
        failure_class="dependency_blocked",
        failure_reason="prerequisite_outcome_missing",
        attempt_started_monotonic=terminal_started,
        raw_secret_values=(),
    )
    assert terminal.record.execution_status == "failed"
    assert cursor.next_unit_index == 76
    resource_started = time.monotonic()
    resource_intent = runner.create_scientific_intent(
        lease, cursor, now_epoch_seconds=now + 8
    )
    resource = runner.commit_claimed_resource_failure(
        lease,
        cursor,
        resource_intent,
        failure_reason="input_preparation_resource_exhausted",
        attempt_started_monotonic=resource_started,
        raw_secret_values=(),
    )
    assert resource.record.execution_status == "retry"
    assert cursor.next_unit_index == 76
    assert store.next_attempt_index(resource_intent.unit_id) == 1


@pytest.mark.quick
def test_scientific_exception_is_committed_as_formal_failure_not_interruption(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    now = int(time.time())
    lease = store.acquire_lease(
        session_id="failure_session",
        now_epoch_seconds=now,
        lease_duration_seconds=100,
    )
    unit_index = next(
        binding.unit_index
        for binding in store.registered_unit_bindings
        if binding.phase == "scientific_breadth"
    )
    intent = store.create_intent(
        lease,
        unit_id=f"development_unit_{unit_index:04d}",
        unit_index=unit_index,
        attempt_index=0,
        parent_attempt_intent_digest=None,
        now_epoch_seconds=now + 1,
    )
    runner.adapter.identify_key = lambda *_args, **_kwargs: None

    result = runner._execute_and_commit_claimed_unit(
        lease,
        intent,
        _input(),
        now_epoch_seconds=now + 1,
        raw_secret_values=("development-runner-cpu-wiring-key",),
        session_cursor=None,
    )
    recovery = store.recover(now_epoch_seconds=now + 2)

    assert result.record.execution_status == "failed"
    assert result.record.failure_class == "implementation_failure"
    assert result.record.failure_reason
    assert recovery.interrupted_attempts == ()
    assert recovery.committed_units == (result.committed,)


@pytest.mark.quick
def test_module_outcome_public_surface_accepts_no_caller_records_or_outcomes() -> None:
    runner = _runner()
    assert not hasattr(runner, "build_module_outcome_record")
    signature = inspect.signature(runner.build_verified_module_outcome_record)
    assert "records" not in signature.parameters
    assert "prerequisite_outcome_records" not in signature.parameters
    with pytest.raises(
        DevelopmentRunnerError,
        match="persistent-store replay",
    ):
        runner._build_module_outcome_record(
            (
                runner._execute_unit(
                    _first_scientific_unit_index(runner, "key_schedule"),
                    _input(),
                ).record,
            ),
            responsibility_id="key_schedule",
            now_epoch_seconds=1,
        )


@pytest.mark.quick
def test_committed_key_records_replay_into_outcome_and_dependency_decision(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    key_unit_indexes = tuple(
        binding.unit_index
        for binding in store.registered_unit_bindings
        if binding.responsibility_id == "key_schedule"
    )
    _commit_frozen_unit_indexes(
        runner,
        store,
        key_unit_indexes,
        session_id="key_outcome_replay_session",
    )

    verified = runner.build_verified_module_outcome_record(
        responsibility_id="key_schedule",
        now_epoch_seconds=103,
    )
    outcome_path = runner.persist_verified_module_outcome(verified)
    assert runner.persist_verified_module_outcome(verified) == outcome_path
    assert outcome_path.is_file()
    decision = runner.decide_verified_module_execution(
        responsibility_id="lf_carrier",
        outcomes_by_responsibility={"key_schedule": verified},
        now_epoch_seconds=103,
    )

    assert verified.outcome_record.module_outcome == "mechanism_signal_observed"
    assert len(verified.outcome_record.evidence_record_ids) == 16
    assert len(verified.evidence_context.committed_marker_bindings) == 16
    assert decision.approved is True
    assert decision.decision_reason == "development_execution_authorized"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("hf_failure_class", "expected_hf_outcome"),
    (
        ("implementation_failure", "implementation_blocked"),
        ("resource_failure", "resource_blocked"),
    ),
)
def test_committed_hf_detector_failures_persist_blocked_outcome_without_threshold_replay(
    tmp_path: Path,
    hf_failure_class: str,
    expected_hf_outcome: str,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    lease = store.acquire_lease(
        session_id=f"hf_detector_{expected_hf_outcome}_session",
        now_epoch_seconds=100,
        lease_duration_seconds=80_000,
    )
    success_roles = {"key_schedule"}
    success_indexes = tuple(
        binding.unit_index
        for binding in store.registered_unit_bindings
        if binding.responsibility_id in success_roles
    )
    for unit_index in success_indexes:
        intent = store.create_intent(
            lease,
            unit_id=f"development_unit_{unit_index:04d}",
            unit_index=unit_index,
            attempt_index=0,
            parent_attempt_intent_digest=None,
            now_epoch_seconds=101,
        )
        record = runner._execute_unit(unit_index, _input()).record
        store.commit_unit(
            lease,
            intent,
            record=record,
            now_epoch_seconds=102,
        )

    terminal_roles = {"hf_carrier", "hf_detector"}
    if expected_hf_outcome == "implementation_blocked":
        terminal_roles.update({"lf_carrier", "lf_detector"})
    for binding in store.registered_unit_bindings:
        if binding.responsibility_id not in terminal_roles:
            continue
        intent = store.create_intent(
            lease,
            unit_id=binding.unit_id,
            unit_index=binding.unit_index,
            attempt_index=0,
            parent_attempt_intent_digest=None,
            now_epoch_seconds=101,
        )
        unit = runner.protocol.unit_roster[binding.unit_index]
        if binding.responsibility_id == "lf_detector":
            execution_status = (
                "excluded"
                if binding.source_cluster_ordinal % 3 == 0
                else "failed"
            )
            failure_class = (
                "scientific_failure"
                if execution_status == "excluded"
                else "dependency_blocked"
                if binding.source_cluster_ordinal % 3 == 1
                else "implementation_failure"
            )
        elif binding.responsibility_id == "hf_detector":
            execution_status = "failed"
            failure_class = hf_failure_class
        else:
            execution_status = "failed"
            failure_class = "implementation_failure"
        record = runner._failure_record(
            unit,
            runner._analysis_identity(unit),
            attempt_index=0,
            retry_parent_intent_digest=None,
            execution_status=execution_status,
            failure_class=failure_class,
            failure_reason="registered_terminal_detector_failure",
            actual_elapsed_seconds=(
                901.0 if failure_class == "resource_failure" else 0.01
            ),
        )
        store.commit_unit(
            lease,
            intent,
            record=record,
            now_epoch_seconds=102,
        )

    plans = {
        responsibility_id: build_development_cross_fit_plan(
            responsibility_id=responsibility_id,
            execution_intent_authority=runner.intent_authority,
            expected_execution_intent_authority_digest=(
                runner.intent_authority.authority_digest
            ),
            expected_source_cluster_count=64,
        )
        for responsibility_id in ("lf_detector", "hf_detector", "content_detector")
    }
    key_outcome = runner.build_verified_module_outcome_record(
        responsibility_id="key_schedule",
        now_epoch_seconds=103,
    )
    hf_outcome = runner.build_verified_module_outcome_record(
        responsibility_id="hf_detector",
        cross_fit_plans=plans,
        now_epoch_seconds=103,
    )
    hf_path = runner.persist_verified_module_outcome(hf_outcome)

    assert hf_outcome.outcome_record.module_outcome == expected_hf_outcome
    assert hf_outcome.outcome_record.provisional_threshold_identities == ()
    assert hf_outcome.evidence_context.provisional_threshold_identities == ()
    assert runner.persist_verified_module_outcome(hf_outcome) == hf_path
    assert hf_path.is_file()
    if expected_hf_outcome == "implementation_blocked":
        lf_outcome = runner.build_verified_module_outcome_record(
            responsibility_id="lf_detector",
            cross_fit_plans=plans,
            now_epoch_seconds=103,
        )
        decision = runner.decide_verified_module_execution(
            responsibility_id="content_router",
            outcomes_by_responsibility={
                "key_schedule": key_outcome,
                "lf_detector": lf_outcome,
                "hf_detector": hf_outcome,
            },
            cross_fit_plans=plans,
            now_epoch_seconds=103,
        )
        assert decision.approved is False
        assert decision.missing_prerequisites == ()
        assert decision.blocking_responsibilities == (
            "lf_detector",
            "hf_detector",
        )
        assert decision.decision_reason == (
            "stop_when_any_prerequisite_lacks_mechanism_signal_observed"
        )


@pytest.mark.quick
def test_dependency_decision_rejects_ghost_or_foreign_verified_context(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    key_unit_indexes = tuple(
        binding.unit_index
        for binding in store.registered_unit_bindings
        if binding.responsibility_id == "key_schedule"
    )
    _commit_frozen_unit_indexes(
        runner,
        store,
        key_unit_indexes,
        session_id="ghost_rejection_session",
    )
    verified = runner.build_verified_module_outcome_record(
        responsibility_id="key_schedule",
        now_epoch_seconds=103,
    )
    ghost_context = replace(
        verified.evidence_context,
        committed_marker_bindings=tuple(
            (record_id, record_digest, "f" * 64)
            for record_id, record_digest, _marker_digest in (
                verified.evidence_context.committed_marker_bindings
            )
        ),
    )
    ghost_record = replace(
        verified.outcome_record,
        committed_marker_bindings=ghost_context.committed_marker_bindings,
    )
    ghost_record = replace(
        ghost_record,
        outcome_record_id=canonical_development_value_digest(
            ghost_record.payload_without_identity()
        ),
    )
    ghost = DevelopmentVerifiedModuleOutcome(
        outcome_record=ghost_record,
        evidence_context=ghost_context,
    )
    assert ghost.validate_structure(runner.protocol) == ()

    with pytest.raises(
        DevelopmentRunnerError,
        match="differs from persistent-store replay",
    ):
        runner.decide_verified_module_execution(
            responsibility_id="lf_carrier",
            outcomes_by_responsibility={"key_schedule": ghost},
            now_epoch_seconds=103,
        )


@pytest.mark.quick
def test_hf_threshold_replay_rejects_plan_valid_uncommitted_alternate(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=runner.intent_authority,
        expected_execution_intent_authority_digest=(
            runner.intent_authority.authority_digest
        ),
        expected_source_cluster_count=64,
    )
    _commit_hf_primary_null_fixture_records(store, plan)

    replayed = runner.replay_verified_hf_provisional_thresholds(
        cross_fit_plan=plan,
        now_epoch_seconds=103,
    )
    alternate = tuple(
        create_development_provisional_threshold(
            plan,
            expected_execution_intent_authority_digest=(
                runner.intent_authority.authority_digest
            ),
            fold_index=fold.fold_index,
            input_manifest=material[0],
            detector_binding=material[1],
            fit_inputs=material[2],
        )
        for fold in plan.folds
        for material in (_threshold_material(plan, fold.fold_index),)
    )

    assert all(item.validate(plan) == () for item in alternate)
    assert alternate != replayed
    with pytest.raises(
        DevelopmentRunnerError,
        match="differ from persistent-store replay",
    ):
        runner.verify_hf_provisional_thresholds_from_store(
            alternate,
            cross_fit_plan=plan,
            now_epoch_seconds=103,
        )
    assert runner.verify_hf_provisional_thresholds_from_store(
        replayed,
        cross_fit_plan=plan,
        now_epoch_seconds=103,
    ) == replayed


@pytest.mark.quick
@pytest.mark.parametrize(
    ("responsibility_id", "bad_values"),
    (
        (
            "qk_geometry_sync",
            {
                "relation_score_gain": 1.0,
                "wrong_key_relation_margin": 1.0,
                "quality_delta": 999.0,
            },
        ),
        (
            "image_rectifier",
            {
                "rectification_quality": 999.0,
                "same_detector_score_delta": 1.0,
                "valid_support": 1.0,
            },
        ),
        (
            "conditional_recovery_decision",
            {
                "incremental_tpr": 1.0,
                "end_to_end_fpr": 1.0,
                "trigger_rate": 1.0,
                "false_rescue_rate": 0.0,
            },
        ),
    ),
)
def test_frozen_module_signal_criteria_reject_bad_scientific_values(
    responsibility_id: str,
    bad_values: dict[str, float],
) -> None:
    runner = _runner()
    study = next(
        item
        for item in runner.protocol.module_matrix
        if item.responsibility_id == responsibility_id
    )

    assert runner._scientific_signal_observed(study, bad_values) is False
