from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from math import inf, nextafter
from pathlib import Path
import inspect
import json
from struct import pack, unpack
import time
from types import MethodType, SimpleNamespace

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)

from experiments.metrics.content_uniform_combination_directional_diagnosis import (
    ContentCombinationArmRgbQualityBudgetExceededError,
    ContentCombinationArmImageDigestInvalidError,
    ContentCombinationArmMaterializationRejectedError,
    ContentCombinationArmMeasurementNonfiniteError,
    ContentCombinationArmObservationIdentityDriftError,
    ContentCombinationArmRealizedContentBudgetExceededError,
    ContentCombinationArmRoleInvalidError,
    ContentUniformCombinationDirectionalMetricError,
    aggregate_content_uniform_combination_directional_diagnosis,
    create_content_combination_arm_observation,
    create_content_combination_reference_measurement,
    create_content_combination_score_row,
    create_content_uniform_combination_directional_observation,
    fit_content_combination_fold_reference,
)
from experiments.protocol.content_uniform_combination_directional_diagnosis import (
    ATTRIBUTION_MARGIN_FLOOR,
    COMBINATION_FUNCTIONS,
    COMBINATION_WEIGHTS,
    ContentUniformCombinationDirectionalProtocolError,
    MIXING_COEFFICIENTS,
    canonical_digest,
    reference_entries_for_probe,
    load_content_uniform_combination_directional_protocol,
)
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.runners.content_uniform_combination_directional_diagnosis import (
    ContentCombinationArmRgbQualityBudgetExceededRunnerError,
    ContentCombinationArmImageDigestInvalidRunnerError,
    ContentCombinationArmMaterializationRejectedRunnerError,
    ContentCombinationArmMeasurementNonfiniteRunnerError,
    ContentCombinationArmObservationIdentityDriftRunnerError,
    ContentCombinationArmRealizedContentBudgetExceededRunnerError,
    ContentCombinationArmRoleInvalidRunnerError,
    ContentCombinationArmObservationConstructionError,
    ContentCombinationProbeObservationConstructionError,
    ContentCombinationScoreRowConstructionError,
    ContentUniformCombinationDirectionalDiagnosisRunner,
)
from experiments.runners.development_persistence import (
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
)
from main import LfNullWhiteningAsset, identify_root_key
from main.shared.key_schedule import stable_json_utf8
from runtime import RuntimeVaeFactors, create_runtime_adapter
from scripts.experiment_execution.content_uniform_combination_directional_diagnosis_entrypoint import (
    _content_combination_observation_failure_reason,
    _resource_failure,
)
from tests.unit.test_runtime_routing_observation import _Posterior, _RoutingBackend


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/content_uniform_combination_directional_diagnosis.json"
DIGEST = "a" * 64
ARMS = (
    ("hf_only", None),
    ("lf_only", None),
    ("uniform_combined_quarter", 0.25),
    ("uniform_combined_half", 0.50),
    ("uniform_combined_three_quarters", 0.75),
)


def _load():
    return load_content_uniform_combination_directional_protocol(CONFIG, repository_root=ROOT)


def _reference_measurements():
    return tuple(
        create_content_combination_reference_measurement(
            cluster_ordinal=index,
            fold_index=index % 4,
            hf_score=-0.25 + index / 1000.0,
            lf_score=-0.15 + index / 1000.0,
            hf_detector_identity="b" * 64,
            lf_detector_identity="c" * 64,
            whitening_asset_digest="d" * 64,
            observation_digest=f"{index + 1:064x}",
        )
        for index in range(32)
    )


def _functions():
    for function in COMBINATION_FUNCTIONS:
        weights = COMBINATION_WEIGHTS if function == "weighted_hf_lf_standardized_score" else (None,)
        for weight in weights:
            yield function, weight


def _observation(index: int, *, passing: bool = True):
    rows = []
    registered_template = "1" * 64
    wrong_templates = tuple(f"{value + 2:064x}" for value in range(4))
    clean_image = "2" * 64
    for arm_number, (arm_id, coefficient) in enumerate(ARMS):
        candidate_image = f"{100 + arm_number:064x}"
        for role, wrong_index in (
            ("registered", None),
            ("paired_clean_primary_null", None),
            *(("wrong_key_control", value) for value in range(4)),
        ):
            image = clean_image if role == "paired_clean_primary_null" else candidate_image
            template = registered_template if wrong_index is None else wrong_templates[wrong_index]
            observation_digest = f"{300 + arm_number * 8 + (wrong_index or 0):064x}"
            for function, weight in _functions():
                lf = function != "hf_only_standardized_score"
                if role == "registered":
                    score = 0.020 if lf and passing else 0.002
                elif role == "paired_clean_primary_null":
                    score = 0.0
                else:
                    score = 0.0
                rows.append(create_content_combination_score_row(
                    arm_id=arm_id,
                    embedding_coefficient=coefficient,
                    control_role=role,
                    wrong_key_index=wrong_index,
                    key_role="wrong" if wrong_index is not None else "registered",
                    combination_function=function,
                    detector_weight=weight,
                    hf_raw_score=score,
                    lf_raw_score=score if lf else None,
                    hf_standardized_score=score,
                    lf_standardized_score=score if lf else None,
                    content_score=score,
                    content_detector_identity="3" * 64,
                    content_config_digest="4" * 64,
                    hf_detector_identity="5" * 64,
                    lf_detector_identity="6" * 64 if lf else None,
                    whitening_asset_digest="7" * 64 if lf else None,
                    input_image_digest=image,
                    hf_observation_digest=observation_digest,
                    lf_observation_digest=observation_digest if lf else None,
                    hf_template_digest=template,
                    lf_template_digest=template if lf else None,
                    root_key_public_digest="8" * 64,
                ))
    arms = tuple(create_content_combination_arm_observation(
        arm_id=arm_id,
        embedding_coefficient=coefficient,
        clean_to_watermarked_rgb_relative_l2=0.01,
        realized_relative_l2=0.01,
        materialization_integrity_status="passed",
        materialization_budget_status="accepted",
        image_digest=f"{500 + arm_number:064x}",
    ) for arm_number, (arm_id, coefficient) in enumerate(ARMS))
    return create_content_uniform_combination_directional_observation(
        cluster_ordinal=index,
        fold_index=index % 4,
        fold_reference_identity=f"{600 + index:064x}",
        whitening_asset_digest="7" * 64,
        score_rows=tuple(rows),
        arm_observations=arms,
        failure_class=None,
    )


def test_protocol_freezes_forty_one_attempt_zero_units_and_disjoint_manifests() -> None:
    protocol, reference, probes = _load()
    assert protocol.run_id == (
        "ceg_wm_content_uniform_combination_arm_budget_field_localization"
    )
    assert (protocol.operational_unit_count, protocol.reference_fit_cluster_count, protocol.directional_probe_cluster_count, protocol.maximum_total_units) == (1, 32, 8, 41)
    assert protocol.maximum_attempts_per_unit == 1
    assert protocol.mixing_coefficients == MIXING_COEFFICIENTS == (0.25, 0.50, 0.75)
    assert protocol.combination_weights == COMBINATION_WEIGHTS == (0.25, 0.50, 0.75)
    assert set(entry.cluster_identity for entry in reference.entries).isdisjoint(entry.cluster_identity for entry in probes.entries)
    assert len(protocol.unit_roster) == 41
    assert all(unit.maximum_record_attempts == 1 for unit in protocol.unit_roster)
    with pytest.raises(ContentUniformCombinationDirectionalProtocolError):
        replace(
            protocol,
            run_id="ceg_wm_content_uniform_combination_arm_observation_leaf_localization",
        ).validate()


def test_reference_cross_fit_excludes_probe_fold_and_uses_twenty_four() -> None:
    values = _reference_measurements()
    reference = fit_content_combination_fold_reference(values, probe_fold_index=2)
    assert len(reference.source_cluster_ordinals) == 24
    assert all(index % 4 != 2 for index in reference.source_cluster_ordinals)
    protocol, manifest, _ = _load()
    _protocol, _reference, probes = _load()
    selected = reference_entries_for_probe(probes.entries[6], manifest)
    assert len(selected) == 24
    assert all(index % 4 != 2 for index in range(32) if manifest.entries[index] in selected)


def test_reference_roster_duplicate_and_detector_drift_fail_closed() -> None:
    values = _reference_measurements()
    with pytest.raises(ContentUniformCombinationDirectionalMetricError):
        fit_content_combination_fold_reference(values[:-1], probe_fold_index=0)
    drifted = replace(values[-1], cluster_ordinal=0)
    with pytest.raises(ContentUniformCombinationDirectionalMetricError):
        fit_content_combination_fold_reference((*values[:-1], drifted), probe_fold_index=0)


def test_complete_controls_bind_same_image_key_templates_and_whitening_asset() -> None:
    observation = _observation(0)
    observation.validate()
    first = observation.score_rows[0]
    drifted = replace(first, input_image_digest="f" * 64)
    rows = (drifted, *observation.score_rows[1:])
    payload = {
        "cluster_ordinal": observation.cluster_ordinal,
        "fold_index": observation.fold_index,
        "fold_reference_identity": observation.fold_reference_identity,
        "whitening_asset_digest": observation.whitening_asset_digest,
        "score_rows": observation.score_rows,
        "arm_observations": observation.arm_observations,
        "failure_class": observation.failure_class,
    }
    payload["score_rows"] = rows
    with pytest.raises(ContentUniformCombinationDirectionalMetricError):
        create_content_uniform_combination_directional_observation(**payload)


def test_directional_gate_preserves_each_a_c_w_without_selecting_one() -> None:
    aggregate = aggregate_content_uniform_combination_directional_diagnosis(
        tuple(_observation(index) for index in range(8))
    )
    assert aggregate.outcome == "mechanism_signal_observed"
    assert aggregate.candidate_recommendation == "candidate_worth_further_selection"
    assert aggregate.allow_request_for_content_combination_candidate_selection is True
    assert aggregate.qualifying_candidate_count == 12
    assert len(aggregate.candidate_statistics) == 12
    assert {item["embedding_coefficient"] for item in aggregate.candidate_statistics} == {0.25, 0.50, 0.75}
    assert {item["detector_weight"] for item in aggregate.candidate_statistics} == {None, 0.25, 0.50, 0.75}


def test_directional_gate_is_fixed_denominator_and_failure_priority() -> None:
    negative = aggregate_content_uniform_combination_directional_diagnosis(
        tuple(_observation(index, passing=False) for index in range(8))
    )
    assert negative.outcome == "mechanism_signal_not_observed"
    assert negative.allow_request_for_content_combination_candidate_selection is False
    failed = create_content_uniform_combination_directional_observation(
        cluster_ordinal=0,
        fold_index=0,
        fold_reference_identity=DIGEST,
        whitening_asset_digest=DIGEST,
        score_rows=(),
        arm_observations=(),
        failure_class="implementation_failure",
    )
    blocked = aggregate_content_uniform_combination_directional_diagnosis(
        (failed, *tuple(_observation(index) for index in range(1, 8)))
    )
    assert blocked.scientific_cluster_count == 8
    assert blocked.failed_cluster_count == 1
    assert blocked.outcome == "implementation_blocked"


def test_budget_and_identity_violations_cannot_pass() -> None:
    observations = tuple(_observation(index) for index in range(8))
    budget = aggregate_content_uniform_combination_directional_diagnosis(
        observations, budget_violation_count=1
    )
    assert budget.allow_request_for_content_combination_candidate_selection is False
    assert budget.outcome == "mechanism_signal_not_observed"
    identity = aggregate_content_uniform_combination_directional_diagnosis(
        observations, identity_violation_count=1
    )
    assert identity.outcome == "implementation_blocked"


def test_hf_only_rows_do_not_consume_lf_and_margin_is_strict() -> None:
    observation = _observation(0)
    c0 = [row for row in observation.score_rows if row.combination_function == "hf_only_standardized_score"]
    assert c0 and all(row.lf_raw_score is None and row.whitening_asset_digest is None for row in c0)
    assert ATTRIBUTION_MARGIN_FLOOR == 2 ** -10


def test_runner_source_uses_public_hf_whitened_lf_and_combination_surfaces() -> None:
    source = inspect.getsource(ContentUniformCombinationDirectionalDiagnosisRunner)
    for required_call in (
        "self.adapter.route_content",
        "self.adapter.build_lf_carrier",
        "self.adapter.build_hf_carrier",
        "self.adapter.embed_content",
        "self.runtime.execute_content_write_and_vae",
        "self.runtime.execute_clean_image_and_vae_observation",
        "self.adapter.detect_hf",
        "self.adapter.detect_lf_null_whitened",
        "self.adapter.detect_content",
    ):
        assert required_call in source
    assert "detect_lf(" not in source
    assert "combination=function" in source
    assert "weight=weight" in source


def _construction_boundary_runner(*, realized_relative_l2: float):
    runner = object.__new__(ContentUniformCombinationDirectionalDiagnosisRunner)
    runner.registered_root_key = "content-combination-construction-boundary-root"
    runner.root_key_public_digest = "8" * 64
    runner.whitening_asset = SimpleNamespace(whitening_asset_digest="7" * 64)

    route = SimpleNamespace(result=SimpleNamespace(route_identity="routing_uniform_control"))
    carrier = SimpleNamespace(result=SimpleNamespace())
    embedded = SimpleNamespace(result=SimpleNamespace())

    class BoundaryAdapter:
        @staticmethod
        def route_content(*_args, **_kwargs):
            return route

        @staticmethod
        def build_lf_carrier(*_args, **_kwargs):
            return carrier

        @staticmethod
        def build_hf_carrier(*_args, **_kwargs):
            return carrier

        @staticmethod
        def embed_content(*_args, **_kwargs):
            return embedded

    clean_image = torch.ones((1, 3, 512, 512), dtype=torch.uint8)
    latent = torch.ones((1, 16, 64, 64), dtype=torch.float32)

    class BoundaryRuntime:
        @staticmethod
        def execute_content_write_and_vae(_base_latent, embed):
            embed(_base_latent)
            return SimpleNamespace(
                clean_image=clean_image,
                watermarked_image=clean_image.clone(),
                clean_detection_latent=latent,
                watermarked_detection_latent=latent,
                content_materialization=SimpleNamespace(
                    realized_relative_l2=realized_relative_l2,
                    integrity_status="passed",
                ),
                content_materialization_result=SimpleNamespace(
                    budget_status="accepted"
                ),
            )

    runner.adapter = BoundaryAdapter()
    runner.runtime = BoundaryRuntime()
    references = tuple(
        fit_content_combination_fold_reference(
            _reference_measurements(), probe_fold_index=index
        )
        for index in range(4)
    )
    runner.fit_fold_references = MethodType(
        lambda _self, _records: references,
        runner,
    )
    return runner


def _score_row_boundary_runner():
    runner = object.__new__(ContentUniformCombinationDirectionalDiagnosisRunner)
    runner.registered_root_key = "content-combination-score-row-boundary-root"
    runner.root_key_public_digest = "8" * 64
    runner.whitening_asset = SimpleNamespace(whitening_asset_digest="7" * 64)
    hf = SimpleNamespace(
        hf_score=0.1,
        detector_identity="5" * 64,
        observation_digest="6" * 64,
        template_digest="9" * 64,
        root_key_public_digest="8" * 64,
    )
    lf = SimpleNamespace(
        lf_score=0.2,
        detector_identity="6" * 64,
        observation_digest="6" * 64,
        template_digest="9" * 64,
        whitening_asset_digest="7" * 64,
    )
    runner._detect_branches = MethodType(lambda _self, _latent, _key: (hf, lf), runner)

    class BoundaryAdapter:
        @staticmethod
        def detect_content(_hf, _lf=None, *, combination, weight=None, **_kwargs):
            diagnostic = SimpleNamespace(
                function_id=combination,
                weight=weight,
                diagnostic_only=True,
                promoted=False,
                hf_standardization=SimpleNamespace(z_score=0.1),
                lf_standardization=(
                    None
                    if combination == "hf_only_standardized_score"
                    else SimpleNamespace(z_score=0.2)
                ),
                combined_score=0.1,
            )
            return SimpleNamespace(
                result=SimpleNamespace(
                    diagnostic_combination=diagnostic,
                    detector_identity="3" * 64,
                    content_config_digest="4" * 64,
                )
            )

    runner.adapter = BoundaryAdapter()
    return runner


def test_runner_preserves_exact_metric_causes_at_three_construction_boundaries() -> None:
    reference = fit_content_combination_fold_reference(
        _reference_measurements(), probe_fold_index=0
    )
    image = torch.ones((1, 3, 512, 512), dtype=torch.uint8)
    latent = torch.ones((1, 16, 64, 64), dtype=torch.float32)
    score_runner = _score_row_boundary_runner()
    with pytest.raises(ContentCombinationScoreRowConstructionError) as score_error:
        score_runner._score_rows(
            arm_id="unsupported_arm",
            coefficient=None,
            image=image,
            latent=latent,
            clean_image=image,
            clean_latent=latent,
            reference=reference,
        )
    assert type(score_error.value.__cause__) is ContentUniformCombinationDirectionalMetricError

    arm_runner = _construction_boundary_runner(realized_relative_l2=0.013)
    arm_runner._score_rows = MethodType(lambda _self, **_kwargs: (), arm_runner)
    with pytest.raises(ContentCombinationArmRealizedContentBudgetExceededRunnerError) as arm_error:
        arm_runner.execute_probe_unit(
            unit_index=33,
            base_latent=latent,
            intent=SimpleNamespace(unit_index=33),
            reference_records=(),
        )
    assert type(arm_error.value.__cause__) is ContentCombinationArmRealizedContentBudgetExceededError
    assert arm_error.value.arm_id == "hf_only"

    probe_runner = _construction_boundary_runner(realized_relative_l2=0.01)
    probe_runner._score_rows = MethodType(lambda _self, **_kwargs: (), probe_runner)
    with pytest.raises(ContentCombinationProbeObservationConstructionError) as probe_error:
        probe_runner.execute_probe_unit(
            unit_index=33,
            base_latent=latent,
            intent=SimpleNamespace(unit_index=33),
            reference_records=(),
        )
    assert type(probe_error.value.__cause__) is ContentUniformCombinationDirectionalMetricError


@pytest.mark.parametrize(
    ("overrides", "runner_error_type", "metric_error_type", "safe_reason"),
    (
        (
            {"arm_id": "unsupported_arm"},
            ContentCombinationArmRoleInvalidRunnerError,
            ContentCombinationArmRoleInvalidError,
            "content_combination_arm_role_invalid",
        ),
        (
            {"embedding_coefficient": 0.50},
            ContentCombinationArmRoleInvalidRunnerError,
            ContentCombinationArmRoleInvalidError,
            "content_combination_arm_role_invalid",
        ),
        (
            {"clean_to_watermarked_rgb_relative_l2": float("nan")},
            ContentCombinationArmMeasurementNonfiniteRunnerError,
            ContentCombinationArmMeasurementNonfiniteError,
            "content_combination_arm_measurement_nonfinite",
        ),
        (
            {"realized_relative_l2": float("inf")},
            ContentCombinationArmMeasurementNonfiniteRunnerError,
            ContentCombinationArmMeasurementNonfiniteError,
            "content_combination_arm_measurement_nonfinite",
        ),
        (
            {"materialization_integrity_status": "rejected"},
            ContentCombinationArmMaterializationRejectedRunnerError,
            ContentCombinationArmMaterializationRejectedError,
            "content_combination_arm_materialization_rejected",
        ),
        (
            {"materialization_budget_status": "rejected"},
            ContentCombinationArmMaterializationRejectedRunnerError,
            ContentCombinationArmMaterializationRejectedError,
            "content_combination_arm_materialization_rejected",
        ),
        (
            {"image_digest": "invalid"},
            ContentCombinationArmImageDigestInvalidRunnerError,
            ContentCombinationArmImageDigestInvalidError,
            "content_combination_arm_image_digest_invalid",
        ),
    ),
)
def test_runner_maps_real_arm_constructor_inputs_to_exact_safe_leaf_reason(
    overrides: dict[str, object],
    runner_error_type: type[RuntimeError],
    metric_error_type: type[ValueError],
    safe_reason: str,
) -> None:
    values = {
        "arm_id": "hf_only",
        "embedding_coefficient": None,
        "clean_to_watermarked_rgb_relative_l2": 0.01,
        "realized_relative_l2": 0.01,
        "materialization_integrity_status": "passed",
        "materialization_budget_status": "accepted",
        "image_digest": "a" * 64,
    }
    values.update(overrides)
    with pytest.raises(runner_error_type) as caught:
        ContentUniformCombinationDirectionalDiagnosisRunner._create_arm_observation(
            **values
        )
    assert type(caught.value) is runner_error_type
    assert type(caught.value.__cause__) is metric_error_type
    assert _content_combination_observation_failure_reason(caught.value) == safe_reason


@pytest.mark.parametrize(("arm_id", "embedding_coefficient"), ARMS)
@pytest.mark.parametrize(
    ("measurement_field", "runner_error_type", "metric_error_type"),
    (
        (
            "clean_to_watermarked_rgb_relative_l2",
            ContentCombinationArmRgbQualityBudgetExceededRunnerError,
            ContentCombinationArmRgbQualityBudgetExceededError,
        ),
        (
            "realized_relative_l2",
            ContentCombinationArmRealizedContentBudgetExceededRunnerError,
            ContentCombinationArmRealizedContentBudgetExceededError,
        ),
    ),
)
def test_runner_binds_each_arm_and_budget_measurement_to_an_exact_safe_reason(
    arm_id: str,
    embedding_coefficient: float | None,
    measurement_field: str,
    runner_error_type: type[RuntimeError],
    metric_error_type: type[ValueError],
) -> None:
    canonical_limit = unpack(">f", pack(">f", 3.0 / 250.0))[0]
    values = {
        "arm_id": arm_id,
        "embedding_coefficient": embedding_coefficient,
        "clean_to_watermarked_rgb_relative_l2": canonical_limit,
        "realized_relative_l2": canonical_limit,
        "materialization_integrity_status": "passed",
        "materialization_budget_status": "accepted",
        "image_digest": "a" * 64,
    }
    boundary = ContentUniformCombinationDirectionalDiagnosisRunner._create_arm_observation(
        **values
    )
    assert boundary.clean_to_watermarked_rgb_relative_l2 == canonical_limit
    assert boundary.realized_relative_l2 == canonical_limit

    values[measurement_field] = nextafter(canonical_limit, inf)
    with pytest.raises(runner_error_type) as caught:
        ContentUniformCombinationDirectionalDiagnosisRunner._create_arm_observation(
            **values
        )
    assert type(caught.value) is runner_error_type
    assert caught.value.arm_id == arm_id
    assert type(caught.value.__cause__) is metric_error_type
    assert caught.value.__cause__.arm_id == arm_id
    assert _content_combination_observation_failure_reason(caught.value) == (
        f"content_combination_{arm_id}_{measurement_field}_canonical_budget_exceeded"
    )


def test_runner_maps_real_probe_validation_identity_drift_to_exact_safe_leaf_reason() -> None:
    observation = _observation(0)
    arms = (
        replace(observation.arm_observations[0], arm_identity="f" * 64),
        *observation.arm_observations[1:],
    )
    with pytest.raises(ContentCombinationArmObservationIdentityDriftRunnerError) as caught:
        ContentUniformCombinationDirectionalDiagnosisRunner._create_probe_observation(
            cluster_ordinal=observation.cluster_ordinal,
            fold_index=observation.fold_index,
            fold_reference_identity=observation.fold_reference_identity,
            whitening_asset_digest=observation.whitening_asset_digest,
            score_rows=observation.score_rows,
            arm_observations=arms,
            failure_class=None,
        )
    assert type(caught.value) is ContentCombinationArmObservationIdentityDriftRunnerError
    assert type(caught.value.__cause__) is ContentCombinationArmObservationIdentityDriftError
    assert _content_combination_observation_failure_reason(caught.value) == (
        "content_combination_arm_observation_identity_drift"
    )


def _record_boundary_runner(
    *, authority_run_id: str | None = None
) -> ContentUniformCombinationDirectionalDiagnosisRunner:
    protocol, reference_manifest, probe_manifest = _load()
    if authority_run_id is not None:
        protocol = replace(
            protocol,
            run_id=authority_run_id,
        )
    runner = object.__new__(ContentUniformCombinationDirectionalDiagnosisRunner)
    runner.protocol = protocol
    runner.reference_manifest = reference_manifest
    runner.probe_manifest = probe_manifest
    runner.method_code_revision = "a" * 40
    runner.root_key_public_digest = "8" * 64
    runner.protocol_digest = (
        canonical_digest(asdict(protocol))
        if authority_run_id is not None
        else protocol.digest()
    )
    runner.execution_intent_authority_digest = "b" * 64
    runner.candidate_config_digest = "c" * 64
    return runner


def _probe_intent(runner, unit_index: int):
    return SimpleNamespace(
        unit_index=unit_index,
        phase="development_content_uniform_combination_directional_probe",
        development_case_id="six_image_uniform_combination_probe",
        content_branch_id="six_image_uniform_combination_probe",
        attempt_index=0,
        parent_attempt_intent_digest=None,
        maximum_duration_seconds=1800,
    )


def test_entrypoint_persists_only_bounded_safe_reasons_with_fixed_failure_denominator() -> None:
    failures = (
        ContentCombinationScoreRowConstructionError("sensitive score message"),
        ContentCombinationArmObservationConstructionError("sensitive arm message"),
        ContentCombinationProbeObservationConstructionError("sensitive probe message"),
    )
    reasons = tuple(
        _content_combination_observation_failure_reason(error) for error in failures
    )
    assert reasons == (
        "content_combination_score_row_construction_failed",
        "content_combination_arm_observation_construction_failed",
        "content_combination_probe_observation_construction_failed",
    )

    class DerivedScoreRowFailure(ContentCombinationScoreRowConstructionError):
        pass

    assert _content_combination_observation_failure_reason(DerivedScoreRowFailure()) is None
    assert _content_combination_observation_failure_reason(RuntimeError()) is None
    assert _resource_failure(MemoryError()) is True

    budget_leaf_failures = tuple(
        error_type(arm_id)
        for error_type in (
            ContentCombinationArmRgbQualityBudgetExceededRunnerError,
            ContentCombinationArmRealizedContentBudgetExceededRunnerError,
        )
        for arm_id, _ in ARMS
    )
    budget_leaf_reasons = tuple(
        f"content_combination_{arm_id}_{measurement_field}_canonical_budget_exceeded"
        for measurement_field in (
            "clean_to_watermarked_rgb_relative_l2",
            "realized_relative_l2",
        )
        for arm_id, _ in ARMS
    )
    leaf_failures = (
        ContentCombinationArmRoleInvalidRunnerError("sensitive role"),
        ContentCombinationArmMeasurementNonfiniteRunnerError("sensitive measurement"),
        *budget_leaf_failures,
        ContentCombinationArmMaterializationRejectedRunnerError("sensitive status"),
        ContentCombinationArmImageDigestInvalidRunnerError("sensitive digest"),
        ContentCombinationArmObservationIdentityDriftRunnerError("sensitive identity"),
    )
    leaf_reasons = tuple(
        _content_combination_observation_failure_reason(error)
        for error in leaf_failures
    )
    assert leaf_reasons == (
        "content_combination_arm_role_invalid",
        "content_combination_arm_measurement_nonfinite",
        *budget_leaf_reasons,
        "content_combination_arm_materialization_rejected",
        "content_combination_arm_image_digest_invalid",
        "content_combination_arm_observation_identity_drift",
    )

    class DerivedArmBudgetFailure(
        ContentCombinationArmRealizedContentBudgetExceededRunnerError
    ):
        pass

    assert _content_combination_observation_failure_reason(
        DerivedArmBudgetFailure("hf_only")
    ) is None

    runner = _record_boundary_runner()
    bounded_reasons = leaf_reasons
    records = tuple(
        runner.create_failed_scientific_record(
            intent=_probe_intent(runner, 33 + index),
            failure_class="implementation_failure",
            failure_reason=bounded_reasons[index % len(bounded_reasons)],
            elapsed_seconds=0.5,
        )
        for index in range(8)
    )
    assert all(
        record.operation_result_payload == {}
        and record.metric_observation == {}
        and "sensitive" not in json.dumps(record.payload(), sort_keys=True)
        for record in records
    )
    aggregate = runner.replay_aggregate(records)
    assert aggregate.scientific_cluster_count == 8
    assert aggregate.failed_cluster_count == 8
    assert aggregate.implementation_failure_count == 8
    assert aggregate.outcome == "implementation_blocked"
    assert aggregate.allow_request_for_content_combination_candidate_selection is False


def test_success_record_and_aggregate_bytes_remain_deterministic() -> None:
    runner = _record_boundary_runner(
        authority_run_id=(
            "ceg_wm_content_uniform_combination_whitening_asset_replay_"
            "correction_diagnosis"
        )
    )
    observation = _observation(0)
    identity = runner._analysis_identity(33)
    payload = {
        "combination_observation": asdict(observation),
        "clean_image_digest": "d" * 64,
    }
    metric = runner._metric_observation(
        identity=identity,
        metric_ids=("content_combination_branch_scores",),
        paired="same_generation_uniform_route_six_image_control",
        branch="six_image_uniform_combination_probe",
        statistics=(("score_row_count", float(len(observation.score_rows))),),
        result_digests=(observation.observation_identity,),
    )
    record = runner._scientific_record(
        intent=_probe_intent(runner, 33),
        identity=identity,
        phase="development_content_uniform_combination_directional_probe",
        case="six_image_uniform_combination_probe",
        branch="six_image_uniform_combination_probe",
        paired="same_generation_uniform_route_six_image_control",
        status="success",
        failure_class=None,
        failure_reason=None,
        elapsed=0.5,
        operation_payload=payload,
        metric=metric,
    )
    record_bytes = json.dumps(
        record.payload(), separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    aggregate = aggregate_content_uniform_combination_directional_diagnosis(
        tuple(_observation(index) for index in range(8))
    )
    aggregate_bytes = json.dumps(
        asdict(aggregate), separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    assert sha256(record_bytes).hexdigest() == (
        "98752f655fc225f1d24c982f227372282ec207a4e7aeb0b812a336662f6f889e"
    )
    assert sha256(aggregate_bytes).hexdigest() == (
        "4d92e9b5230627f45aae7cf5f99258d533e3c73f553a9ab3058a9658277a6586"
    )

    parent_runner = _record_boundary_runner(
        authority_run_id=(
            "ceg_wm_content_uniform_combination_observation_construction_"
            "localization"
        )
    )
    parent_identity = parent_runner._analysis_identity(33)
    parent_metric = parent_runner._metric_observation(
        identity=parent_identity,
        metric_ids=("content_combination_branch_scores",),
        paired="same_generation_uniform_route_six_image_control",
        branch="six_image_uniform_combination_probe",
        statistics=(("score_row_count", float(len(observation.score_rows))),),
        result_digests=(observation.observation_identity,),
    )
    parent_record = parent_runner._scientific_record(
        intent=_probe_intent(parent_runner, 33),
        identity=parent_identity,
        phase="development_content_uniform_combination_directional_probe",
        case="six_image_uniform_combination_probe",
        branch="six_image_uniform_combination_probe",
        paired="same_generation_uniform_route_six_image_control",
        status="success",
        failure_class=None,
        failure_reason=None,
        elapsed=0.5,
        operation_payload=payload,
        metric=parent_metric,
    )
    parent_record_bytes = json.dumps(
        parent_record.payload(),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    assert sha256(parent_record_bytes).hexdigest() == (
        "9ca68d0fb5a72f98808dad19670dfb7036c2ce647ba2df05169ed93ce8c241a4"
    )

    current_runner = _record_boundary_runner()
    current_identity = current_runner._analysis_identity(33)
    current_metric = current_runner._metric_observation(
        identity=current_identity,
        metric_ids=("content_combination_branch_scores",),
        paired="same_generation_uniform_route_six_image_control",
        branch="six_image_uniform_combination_probe",
        statistics=(("score_row_count", float(len(observation.score_rows))),),
        result_digests=(observation.observation_identity,),
    )
    current_record = current_runner._scientific_record(
        intent=_probe_intent(current_runner, 33),
        identity=current_identity,
        phase="development_content_uniform_combination_directional_probe",
        case="six_image_uniform_combination_probe",
        branch="six_image_uniform_combination_probe",
        paired="same_generation_uniform_route_six_image_control",
        status="success",
        failure_class=None,
        failure_reason=None,
        elapsed=0.5,
        operation_payload=payload,
        metric=current_metric,
    )
    assert current_record.run_id != record.run_id
    assert current_record.protocol_digest != record.protocol_digest
    assert current_record.record_id != record.record_id
    assert current_record.operation_result_payload == record.operation_result_payload
    assert current_record.operation_result_digest == record.operation_result_digest
    assert current_record.metric_observation == record.metric_observation
    current_record_bytes = json.dumps(
        current_record.payload(),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    assert sha256(current_record_bytes).hexdigest() == (
        "e09383d3b4bc134414038d682089c0273bac6ecbd73620ec6c6f6462e97256f9"
    )
    assert current_record.operation_result_payload == parent_record.operation_result_payload
    assert current_record.operation_result_digest == parent_record.operation_result_digest
    assert current_record.metric_observation == parent_record.metric_observation


class _CombinationBackend(_RoutingBackend):
    def vae_factors(self) -> RuntimeVaeFactors:
        return RuntimeVaeFactors(scaling_factor=1.0, shift_factor=0.0)

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        image = latent.detach().float().mean(dim=1, keepdim=True)
        image = torch.nn.functional.interpolate(
            image, size=(512, 512), mode="bilinear", align_corners=False
        )
        return torch.cat((image, image * 0.9, image * 0.8), dim=1).clamp(-0.9, 0.9)

    def vae_encode(self, image: torch.Tensor) -> _Posterior:
        latent = torch.nn.functional.interpolate(
            image.detach().float().mean(dim=1, keepdim=True),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        ).repeat(1, 16, 1, 1)
        return _Posterior(latent.to(torch.float16))


def _whitening_asset() -> LfNullWhiteningAsset:
    payload = {
        "artifact_role": "lf_clean_null_whitening_operator",
        "band_identity": "six_dyadic_chebyshev_frequency_rings_without_dc",
        "candidate_id": "lf_null_whitened_matched_score",
        "detrend_identity": "per_channel_affine_plane_normalized_coordinates",
        "fit_manifest_sha256": "9" * 64,
        "fit_source_cluster_count": 32,
        "latent_shape": [1, 16, 64, 64],
        "observation_protocol": "final_image_vae_posterior_mode",
        "regularization_ratio": "0x1.0000000000000p-10",
        "transform_identity": "orthonormal_dct_ii",
        "weights_binary32_be_hex": ["3f800000"] * 96,
    }
    return LfNullWhiteningAsset.from_canonical_payload(
        payload,
        whitening_asset_digest=sha256(stable_json_utf8(payload)).hexdigest(),
    )


@pytest.mark.integration
@pytest.mark.slow
def test_real_runner_store_commits_recovers_and_replays_all_forty_one_units(
    tmp_path: Path,
) -> None:
    protocol, reference_manifest, probe_manifest = _load()
    asset = _whitening_asset()
    protocol = replace(protocol, whitening_asset_digest=asset.whitening_asset_digest)
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(
            ROOT / "configs/experiments/internal_execution_components.json"
        )
    )
    backend = _CombinationBackend()
    runtime = create_runtime_adapter(backend)
    runtime.initialize("cpu")
    registered_root = "content-uniform-combination-real-public-root"
    runner = ContentUniformCombinationDirectionalDiagnosisRunner(
        protocol=protocol,
        reference_manifest=reference_manifest,
        probe_manifest=probe_manifest,
        adapter=adapter,
        runtime_adapter=runtime,
        whitening_asset=asset,
        method_code_revision="a" * 40,
        registered_root_key=registered_root,
        root_key_public_digest=identify_root_key(registered_root).root_key_public_digest,
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest="b" * 64,
        candidate_config_digest="c" * 64,
    )
    worker = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest="d" * 64,
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )

    def new_store() -> DevelopmentPersistentStore:
        return DevelopmentPersistentStore(
            tmp_path,
            run_id=runner.protocol.run_id,
            worker_identity=worker,
            registered_unit_bindings=runner.create_persistence_unit_bindings(),
        )

    store = new_store()
    epoch = int(time.time())
    lease = store.acquire_lease(
        session_id="uniform_combination_real_public_session",
        now_epoch_seconds=epoch,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch)
    base = torch.linspace(
        0.05, 0.45, steps=16 * 64 * 64, dtype=torch.float16
    ).reshape(1, 16, 64, 64)

    intent = store.create_session_intent(cursor, lease, now_epoch_seconds=epoch + 1)
    operational = runner.execute_operational_unit(
        unit_index=0, base_latent=base, intent=intent
    )
    store.commit_session_unit(
        cursor, lease, intent, record=operational, now_epoch_seconds=epoch + 2
    )
    for ordinal in range(32):
        intent = store.create_session_intent(
            cursor, lease, now_epoch_seconds=epoch + 3 + ordinal * 2
        )
        if ordinal == 0:
            failed_reference = runner.create_failed_scientific_record(
                intent=intent,
                failure_class="implementation_failure",
                failure_reason="public_reference_detection_failure",
                elapsed_seconds=0.5,
            )
            checked_failed_reference = DevelopmentScientificRecord.from_payload(
                json.loads(
                    json.dumps(
                        failed_reference.payload(),
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
            )
            assert checked_failed_reference.paired_ablation_identity == (
                "clean_primary_null_cross_fit_reference"
            )
            assert checked_failed_reference.content_branch_id == (
                "paired_clean_branch_null_reference"
            )
        record = runner.execute_reference_fit_unit(
            unit_index=ordinal + 1, base_latent=base, intent=intent
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + 4 + ordinal * 2,
        )
    store = new_store()
    cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch + 100)
    assert cursor.next_unit_index == 33
    references = tuple(
        record for record in store.verified_terminal_scientific_records(
            now_epoch_seconds=epoch + 101
        ) if record.unit_index < 33
    )
    assert len(references) == 32
    assert all(
        record.paired_ablation_identity == "clean_primary_null_cross_fit_reference"
        and record.metric_observation["paired_ablation_identity"]
        == record.paired_ablation_identity
        and record.content_branch_id == "paired_clean_branch_null_reference"
        and record.metric_observation["content_branch_id"] == record.content_branch_id
        for record in references
    )
    for ordinal in range(8):
        intent = store.create_session_intent(
            cursor, lease, now_epoch_seconds=epoch + 110 + ordinal * 2
        )
        record = runner.execute_probe_unit(
            unit_index=ordinal + 33,
            base_latent=base,
            intent=intent,
            reference_records=references,
        )
        checked = DevelopmentScientificRecord.from_payload(
            json.loads(
                json.dumps(
                    record.payload(),
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        )
        observation_payload = checked.operation_result_payload[
            "combination_observation"
        ]
        assert checked.paired_ablation_identity == (
            "same_generation_uniform_route_six_image_control"
        )
        assert checked.metric_observation["paired_ablation_identity"] == (
            checked.paired_ablation_identity
        )
        assert checked.metric_observation["content_branch_id"] == (
            checked.content_branch_id
        )
        assert len(observation_payload["arm_observations"]) == 5
        assert len({checked.operation_result_payload["clean_image_digest"]}) == 1
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + 111 + ordinal * 2,
        )
    final_store = new_store()
    final_cursor = final_store.open_session_cursor(lease, now_epoch_seconds=epoch + 200)
    records = tuple(
        record for record in final_store.verified_terminal_scientific_records(
            now_epoch_seconds=epoch + 201
        ) if record.unit_index >= 33
    )
    aggregate = runner.replay_aggregate(records)
    class DerivedScientificRecord(DevelopmentScientificRecord):
        pass

    derived = DerivedScientificRecord(**records[0].payload())
    with pytest.raises(RuntimeError, match="exact persistent scientific record type"):
        runner.observation_from_record(derived)
    for drifted in (
        replace(records[0], phase="development_content_combination_reference_fit"),
        replace(records[0], paired_ablation_identity="clean_primary_null_cross_fit_reference"),
        replace(records[0], execution_status="failed"),
        replace(records[0], operation_result_payload={}),
    ):
        with pytest.raises((RuntimeError, ValueError)):
            runner.replay_aggregate((drifted, *records[1:]))
    assert final_cursor.next_unit_index == 41
    assert len(final_cursor.committed_units) == 41
    assert len(records) == 8
    assert aggregate.scientific_cluster_count == 8
    assert aggregate.failed_cluster_count == 0
    assert backend.generation_calls == 114

    failure_root = tmp_path / "failure_recovery"
    failure_store = DevelopmentPersistentStore(
        failure_root,
        run_id=runner.protocol.run_id,
        worker_identity=worker,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    failure_lease = failure_store.acquire_lease(
        session_id="uniform_combination_failure_recovery_session",
        now_epoch_seconds=epoch,
        lease_duration_seconds=10000,
    )
    failure_cursor = failure_store.open_session_cursor(
        failure_lease, now_epoch_seconds=epoch
    )
    intent = failure_store.create_session_intent(
        failure_cursor, failure_lease, now_epoch_seconds=epoch + 1
    )
    operational = runner.execute_operational_unit(
        unit_index=0, base_latent=base, intent=intent
    )
    failure_store.commit_session_unit(
        failure_cursor,
        failure_lease,
        intent,
        record=operational,
        now_epoch_seconds=epoch + 2,
    )
    for ordinal in range(32):
        intent = failure_store.create_session_intent(
            failure_cursor,
            failure_lease,
            now_epoch_seconds=epoch + 3 + ordinal * 2,
        )
        record = runner.execute_reference_fit_unit(
            unit_index=ordinal + 1, base_latent=base, intent=intent
        )
        failure_store.commit_session_unit(
            failure_cursor,
            failure_lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + 4 + ordinal * 2,
        )
    for ordinal in range(8):
        intent = failure_store.create_session_intent(
            failure_cursor,
            failure_lease,
            now_epoch_seconds=epoch + 110 + ordinal * 2,
        )
        record = runner.create_failed_scientific_record(
            intent=intent,
            failure_class=(
                "implementation_failure" if ordinal < 4 else "resource_failure"
            ),
            failure_reason=(
                "public_combination_implementation_failure"
                if ordinal < 4
                else "public_runtime_resource_failure"
            ),
            elapsed_seconds=0.5,
        )
        if ordinal == 0:
            with pytest.raises(
                RuntimeError,
                match="scientific intent responsibility identity drifted",
            ):
                runner.create_failed_scientific_record(
                    intent=replace(
                        intent,
                        phase="development_content_combination_reference_fit",
                    ),
                    failure_class="implementation_failure",
                    failure_reason="public_combination_implementation_failure",
                    elapsed_seconds=0.5,
                )
        checked_failure = DevelopmentScientificRecord.from_payload(
            json.loads(
                json.dumps(
                    record.payload(),
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        )
        assert checked_failure.paired_ablation_identity == (
            "same_generation_uniform_route_six_image_control"
        )
        assert checked_failure.content_branch_id == (
            "six_image_uniform_combination_probe"
        )
        failure_store.commit_session_unit(
            failure_cursor,
            failure_lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + 111 + ordinal * 2,
        )
    recovered_failure_store = DevelopmentPersistentStore(
        failure_root,
        run_id=runner.protocol.run_id,
        worker_identity=worker,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    failure_records = tuple(
        record
        for record in recovered_failure_store.verified_terminal_scientific_records(
            now_epoch_seconds=epoch + 201
        )
        if record.unit_index >= 33
    )
    blocked = runner.replay_aggregate(failure_records)
    assert len(failure_records) == 8
    assert blocked.failed_cluster_count == 8
    assert blocked.implementation_failure_count == 4
    assert blocked.resource_failure_count == 4
    assert blocked.outcome == "implementation_blocked"
    assert blocked.allow_request_for_content_combination_candidate_selection is False
    runtime.close()
