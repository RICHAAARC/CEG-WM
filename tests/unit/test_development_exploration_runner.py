"""CPU wiring checks for the real development exploration dispatch."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
from pathlib import Path

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
)
from experiments.runners.development_persistence import (
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
    development_unit_roster_digest,
)
from experiments.protocol.development_exploration import (
    DevelopmentVerifiedModuleOutcome,
    build_development_cross_fit_plan,
    create_development_provisional_threshold,
    create_development_threshold_fit_input,
)
from experiments.protocol.internal_matrix import REQUIRED_METHOD_RESPONSIBILITIES
from experiments.protocol.development_records import canonical_development_value_digest
from main import (
    BranchNullCalibration,
    GeometryReliabilityThresholds,
    HfDetectionObservation,
    LfDetectionObservation,
    RoutingObservations,
    hf_detector,
    lf_detector,
)
from main.content_chain.routing import SpatialRoutingObservation
from main.content_chain.detector import NullScoreRecord
from runtime import create_runtime_adapter
from tests.unit.test_development_module_exploration import (
    _development_manifest,
    _execution_intent,
    _primary_null_record,
    _redigest_scientific_record,
    _threshold_material,
)
from tests.unit.test_internal_governed_runner import _context as _internal_context
import tests.unit.test_internal_governed_runner as internal_runner_test_module
from tests.unit.test_runtime_content_write_and_vae import FakeContentBackend
from tests.unit.test_runtime_qk_observation import FakeQkBackend


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
def test_content_embedding_unit_uses_actual_runtime_write_and_vae() -> None:
    runner = _runner()
    result = runner._execute_unit(
        _first_scientific_unit_index(runner, "content_embedder"), _input()
    )

    assert result.record.responsibility_id == "content_embedder"
    assert result.record.provenance_trace["runtime_config_digest"] == (
        runner.runtime_adapter.session.runtime_config_digest
    )
    values = dict(result.record.metric_observation["sufficient_statistics"])
    assert values["realized_total_relative_l2"] >= 0.0
    assert runner.runtime_adapter._backend.run_calls == 2


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    observed: list[str] = []

    def execute_operation(unit, _identity, _raw):
        observed.append(unit.responsibility_id)
        return {"responsibility_id": unit.responsibility_id}, None, {}, None

    def execute_conditional(unit, _raw):
        observed.append(unit.responsibility_id)
        return {"responsibility_id": unit.responsibility_id}

    monkeypatch.setattr(runner, "_execute_real_operation", execute_operation)
    monkeypatch.setattr(
        runner,
        "_execute_wiring_conditional_call",
        execute_conditional,
    )
    monkeypatch.setattr(
        "experiments.runners.development_exploration._safe_result_payload",
        lambda _responsibility, result: result,
    )
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
    assert tuple(observed) == REQUIRED_METHOD_RESPONSIBILITIES


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
def test_rectifier_fail_closed_is_classified_as_scientific_exclusion() -> None:
    runner = _runner()
    unit = next(
        item
        for item in runner.protocol.unit_roster
        if item.responsibility_id == "image_rectifier"
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
def test_scientific_exception_is_committed_as_formal_failure_not_interruption(
    tmp_path: Path,
) -> None:
    runner, store = _persistent_runner(tmp_path)
    lease = store.acquire_lease(
        session_id="failure_session",
        now_epoch_seconds=100,
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
        now_epoch_seconds=101,
    )
    runner.adapter.identify_key = lambda *_args, **_kwargs: None

    result = runner._execute_and_commit_claimed_unit(
        lease,
        intent,
        _input(),
        now_epoch_seconds=101,
        raw_secret_values=("development-runner-cpu-wiring-key",),
        session_cursor=None,
    )
    recovery = store.recover(now_epoch_seconds=102)

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
    decision = runner.decide_verified_module_execution(
        responsibility_id="content_router",
        outcomes_by_responsibility={"key_schedule": verified},
        now_epoch_seconds=103,
    )

    assert verified.outcome_record.module_outcome == "mechanism_signal_observed"
    assert len(verified.outcome_record.evidence_record_ids) == 16
    assert len(verified.evidence_context.committed_marker_bindings) == 16
    assert decision.approved is True
    assert decision.decision_reason == "development_execution_authorized"


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
            responsibility_id="content_router",
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
