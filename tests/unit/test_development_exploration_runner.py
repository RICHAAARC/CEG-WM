"""CPU wiring checks for the real development exploration dispatch."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
from pathlib import Path

import pytest
import torch

from experiments.attacks import GeometricAttackSpec, load_attack_registry
from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.runners.development_exploration import (
    DevelopmentExplorationRunner,
    DevelopmentRunnerError,
    DevelopmentUnitInput,
)
from experiments.protocol.development_exploration import (
    build_development_cross_fit_plan,
    create_development_provisional_threshold,
    create_development_threshold_fit_input,
)
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
        self.content = FakeContentBackend()
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


def _runner(intent_authority=None) -> DevelopmentExplorationRunner:
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
        attack_specification=GeometricAttackSpec("identity"),
        epsilon_inlier=0.8,
        geometry_reliability_thresholds=_reliability_thresholds(),
        provisional_threshold=None,
        cross_fit_plan=None,
        development_tau_rescue=None,
    )


@pytest.mark.quick
def test_first_breadth_units_call_real_key_router_and_carrier_methods() -> None:
    runner = _runner()
    key = runner.execute_unit(0, _input(), prerequisite_outcomes={})
    router = runner.execute_unit(
        1, _input(), prerequisite_outcomes={"key_schedule": "mechanism_signal_observed"}
    )
    low_frequency = runner.execute_unit(
        2, _input(), prerequisite_outcomes={"key_schedule": "mechanism_signal_observed"}
    )
    high_frequency = runner.execute_unit(
        3, _input(), prerequisite_outcomes={"key_schedule": "mechanism_signal_observed"}
    )

    assert key.record.responsibility_id == "key_schedule"
    assert dict(key.record.metric_observation["metric_values"])[
        "registered_wrong_domain_separated"
    ] == 1.0
    assert router.record.routing_trace["routing_identity"]
    assert low_frequency.record.operation_result_payload["candidate_id"] == "lf_low_pass"
    assert high_frequency.record.operation_result_payload["candidate_id"] == "hf_sparse_tail"
    assert all(result.record.module_outcome is None for result in (key, router, low_frequency, high_frequency))


@pytest.mark.quick
def test_dependency_stop_is_separate_from_resource_or_mechanism_outcome() -> None:
    runner = _runner()
    with pytest.raises(DevelopmentRunnerError, match="module dependency stop"):
        runner.execute_unit(
            1,
            _input(),
            prerequisite_outcomes={"key_schedule": "mechanism_signal_not_observed"},
        )


@pytest.mark.quick
def test_content_embedding_unit_uses_actual_runtime_write_and_vae() -> None:
    runner = _runner()
    result = runner.execute_unit(
        4,
        _input(),
        prerequisite_outcomes={
            "content_router": "mechanism_signal_observed",
            "lf_carrier": "mechanism_signal_observed",
            "hf_carrier": "mechanism_signal_observed",
        },
    )

    assert result.record.responsibility_id == "content_embedder"
    assert result.record.provenance_trace["runtime_config_digest"] == (
        runner.runtime_adapter.session.runtime_config_digest
    )
    values = dict(result.record.metric_observation["metric_values"])
    assert values["realized_relative_l2"] >= 0.0
    assert runner.runtime_adapter._backend.run_calls == 2


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
@pytest.mark.parametrize(
    ("unit_index", "prerequisite_outcomes", "responsibility_id"),
    (
        (5, {"key_schedule": "mechanism_signal_observed", "lf_carrier": "mechanism_signal_observed"}, "lf_detector"),
        (6, {"key_schedule": "mechanism_signal_observed", "hf_carrier": "mechanism_signal_observed"}, "hf_detector"),
        (7, {"content_embedder": "mechanism_signal_observed", "lf_detector": "mechanism_signal_observed", "hf_detector": "mechanism_signal_observed"}, "content_detector"),
        (8, {"key_schedule": "mechanism_signal_observed"}, "qk_geometry_sync"),
        (9, {"qk_geometry_sync": "mechanism_signal_observed"}, "geometric_transform_estimator"),
        (10, {"geometric_transform_estimator": "mechanism_signal_observed"}, "geometry_reliability"),
        (11, {"geometric_transform_estimator": "mechanism_signal_observed", "geometry_reliability": "mechanism_signal_observed"}, "image_rectifier"),
    ),
)
def test_remaining_non_joint_responsibilities_use_real_public_calls(
    unit_index: int,
    prerequisite_outcomes: dict[str, str],
    responsibility_id: str,
) -> None:
    result = _runner().execute_unit(
        unit_index,
        _input(),
        prerequisite_outcomes=prerequisite_outcomes,
    )

    assert result.record.responsibility_id == responsibility_id
    assert result.record.metric_observation["responsibility_id"] == responsibility_id
    assert result.record.operation_result_digest


@pytest.mark.quick
def test_conditional_recovery_unit_delegates_once_to_governed_internal_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _development_manifest(64)
    target = min(
        (assignment.identity for assignment in manifest.assignments),
        key=lambda identity: identity.source_cluster_id,
    )
    original_unit = internal_runner_test_module._unit
    monkeypatch.setattr(
        internal_runner_test_module,
        "_unit",
        lambda index, **kwargs: target if index == 0 else original_unit(index, **kwargs),
    )
    context, payload, operation = _internal_context(tmp_path / "internal")
    intent = _execution_intent(manifest, run_id="development_joint_wiring")
    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=intent,
        expected_execution_intent_authority_digest=intent.authority_digest,
        expected_source_cluster_count=64,
    )
    threshold_manifest, detector_binding, fit_inputs = _threshold_material(plan)
    desired_tau = payload.thresholds.tau
    adjusted_inputs = tuple(
        create_development_threshold_fit_input(
            expected_execution_intent_authority_digest=intent.authority_digest,
            source_record=replace(
                item.source_record,
                detector_trace=replace(
                    item.source_record.detector_trace,
                    raw_content_score=(
                        desired_tau if index == 0 else desired_tau - 1.0
                    ),
                ),
                branch_score_trace=replace(
                    item.source_record.branch_score_trace,
                    hf_score=(desired_tau if index == 0 else desired_tau - 1.0),
                ),
            )
        )
        for index, item in enumerate(fit_inputs)
    )
    threshold = create_development_provisional_threshold(
        plan,
        expected_execution_intent_authority_digest=intent.authority_digest,
        fold_index=0,
        input_manifest=threshold_manifest,
        detector_binding=detector_binding,
        fit_inputs=adjusted_inputs,
    )
    joint_input = replace(
        _input(),
        provisional_threshold=threshold,
        cross_fit_plan=plan,
        development_tau_rescue=payload.thresholds.tau_rescue,
        internal_runner_context=context,
        internal_case_payload=payload,
    )

    result = _runner(intent).execute_unit(
        12,
        joint_input,
        prerequisite_outcomes={
            "content_detector": "mechanism_signal_observed",
            "qk_geometry_sync": "mechanism_signal_observed",
            "geometric_transform_estimator": "mechanism_signal_observed",
            "geometry_reliability": "mechanism_signal_observed",
            "image_rectifier": "mechanism_signal_observed",
        },
    )

    assert result.record.responsibility_id == "conditional_recovery_decision"
    assert result.record.decision_trace["internal_validation_record_id"]
    assert result.record.threshold_trace["threshold_role"] == "development_exploratory"
    assert operation.calls == 1


@pytest.mark.quick
def test_production_runner_has_no_result_provider_or_module_result_surface() -> None:
    source = inspect.getsource(DevelopmentExplorationRunner)
    signature = inspect.signature(DevelopmentExplorationRunner.execute_unit)

    assert "result_provider" not in source
    assert "module_results" not in source
    assert "result_provider" not in signature.parameters
    assert "module_results" not in signature.parameters
    assert "execute_content_write_and_vae" in source
    assert "observe_detection_qk" in source
    assert "apply_geometric_attack" in source
    assert "execute_internal_case" in source
