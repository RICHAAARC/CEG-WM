from __future__ import annotations

from collections import Counter
from dataclasses import asdict, fields, replace
import importlib
import inspect
import json
from math import ceil
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)

from experiments.metrics.content_routing_directional_diagnosis import (
    ContentRoutingBlindScoreObservation,
    ContentRoutingDirectionalMetricError,
    ContentRoutingReferencePositiveSupportError,
    aggregate_content_routing_directional_diagnosis,
    create_content_routing_blind_score_observation,
    create_content_routing_directional_observation,
    create_content_routing_reference_measurement,
    exact_nearest_rank_positive_p95,
    fit_content_routing_fold_reference,
)
from experiments.protocol.content_routing_directional_diagnosis import (
    CLAIM_BOUNDARY,
    ContentRoutingDirectionalProtocolError,
    canonical_digest,
    load_content_routing_directional_protocol,
    reference_entries_for_probe,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
    ROUTING_REFERENCE_RECORD_KIND,
    ROUTING_REFERENCE_RECORD_SCHEMA,
    DevelopmentRecordError,
    DevelopmentRoutingReferenceRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)
from experiments.runners.content_routing_directional_diagnosis import (
    ContentRoutingDirectionalDiagnosisRunner,
    ContentRoutingReferencePositiveSupportAbsentError,
)
from experiments.runners.development_inputs import (
    DevelopmentSemanticObservationProducer,
)
from experiments.runners.development_persistence import (
    DevelopmentPersistenceError,
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
)
from experiments.runners.formal_operations import (
    FormalHfContentDetectionOperation,
)
from main import RoutingObservations, SpatialRoutingObservation, identify_root_key
from runtime import Sd35RuntimeAdapter, create_runtime_adapter
from scripts.experiment_execution import (
    content_routing_directional_diagnosis_entrypoint as entrypoint_module,
)
from scripts.experiment_execution.content_routing_directional_diagnosis_entrypoint import (
    ContentRoutingDirectionalEntrypointError,
    ContentRoutingDirectionalStartupError,
    _commit_dependency_blocked_probe_records,
    _initialize_routing_resources,
    _registered_experiment_root,
    _reference_dependency_failure_class,
    _replay_aggregate,
)
from tests.unit.test_runtime_content_write_and_vae import FakeContentBackend
from tests.unit.test_runtime_routing_observation import (
    _Posterior,
    _RoutingBackend,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/content_routing_directional_diagnosis.json"
pytestmark = pytest.mark.unit


def _protocol_bundle():
    return load_content_routing_directional_protocol(
        CONFIG,
        repository_root=ROOT,
    )


def _reference_measurements():
    return tuple(
        create_content_routing_reference_measurement(
            cluster_ordinal=ordinal,
            fold_index=ordinal % 4,
            texture_gradient_values=(float(ordinal + 1), float(ordinal + 1.25)),
            texture_spatial_shape=(1, 2),
            response_ratio_values=(
                float(ordinal + 101),
                float(ordinal + 101.25),
            ),
            response_spatial_shape=(1, 2),
            sensitivity_ratio_values=(
                float(ordinal + 201),
                float(ordinal + 201.25),
            ),
            sensitivity_spatial_shape=(1, 2),
        )
        for ordinal in range(32)
    )


def _persistent_reference_record(ordinal: int) -> DevelopmentRoutingReferenceRecord:
    value = float(ordinal + 1)
    if ordinal == 31:
        texture_values = [100.0] * 100
        response_values = [200.0] * 100
        sensitivity_values = [300.0] * 100
        spatial_shape = [10, 10]
    else:
        texture_values = [value, value + 0.25]
        response_values = [value + 100.0, value + 100.25]
        sensitivity_values = [value + 200.0, value + 200.25]
        spatial_shape = [1, 2]
    payload = {
        "schema_version": ROUTING_REFERENCE_RECORD_SCHEMA,
        "collection_role": ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
        "record_kind": ROUTING_REFERENCE_RECORD_KIND,
        "record_id": "0" * 64,
        "run_id": "ceg_wm_content_routing_backend_binding_correction_diagnosis",
        "protocol_digest": "1" * 64,
        "method_code_revision": "2" * 40,
        "unit_index": ordinal + 2,
        "phase": ROUTING_REFERENCE_RECORD_KIND,
        "source_cluster_ordinal": ordinal,
        "fold_index": ordinal % 4,
        "prompt_roster_digest": "3" * 64,
        "candidate_config_digest": "4" * 64,
        "attempt_index": 0,
        "retry_parent_intent_digest": None,
        "actual_elapsed_seconds": 1.0,
        "maximum_duration_seconds": 2700,
        "duration_limit_exceeded": False,
        "execution_status": "success",
        "failure_class": None,
        "failure_reason": None,
        "measurement_payload": {
            "candidate_id": "routing_stqr",
            "runtime_config_digest": "5" * 64,
            "model_id": "registered-routing-model",
            "model_revision": "registered-routing-model-revision",
            "callback_indices": list(range(20)),
            "public_probe_domain_digest": f"{ordinal + 6:064x}",
            "public_probe_values_digest": f"{ordinal + 38:064x}",
            "nominal_relative_probe_step": 0.001,
            "actual_probe_step": 0.001,
            "texture_gradient_values": texture_values,
            "texture_spatial_shape": spatial_shape,
            "response_ratio_values": response_values,
            "response_spatial_shape": spatial_shape,
            "sensitivity_ratio_values": sensitivity_values,
            "sensitivity_spatial_shape": spatial_shape,
        },
        "counts_as_scientific_coverage": False,
        "scientific_claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
    }
    draft = DevelopmentRoutingReferenceRecord(**payload)
    payload["record_id"] = canonical_development_value_digest(
        draft.payload_without_record_id()
    )
    return DevelopmentRoutingReferenceRecord.from_payload(payload)


def _failed_reference_record(
    ordinal: int,
    failure_class: str | None,
    *,
    execution_status: str = "failed",
) -> DevelopmentRoutingReferenceRecord:
    payload = _persistent_reference_record(ordinal).payload()
    payload.update(
        {
            "record_id": "0" * 64,
            "execution_status": execution_status,
            "failure_class": failure_class,
            "failure_reason": "registered_reference_failure",
            "measurement_payload": {},
        }
    )
    draft = DevelopmentRoutingReferenceRecord(**payload)
    payload["record_id"] = canonical_development_value_digest(
        draft.payload_without_record_id()
    )
    return DevelopmentRoutingReferenceRecord.from_payload(payload)


def _zero_reference_record(
    ordinal: int,
    *,
    axis: str | None = None,
) -> DevelopmentRoutingReferenceRecord:
    payload = _persistent_reference_record(ordinal).payload()
    measurement = dict(payload["measurement_payload"])
    axis_names = (
        "texture_gradient_values",
        "response_ratio_values",
        "sensitivity_ratio_values",
    )
    for field_name in axis_names:
        if axis is None or field_name == axis:
            measurement[field_name] = [
                0.0 for _value in measurement[field_name]
            ]
    payload.update(record_id="0" * 64, measurement_payload=measurement)
    draft = DevelopmentRoutingReferenceRecord(**payload)
    payload["record_id"] = canonical_development_value_digest(
        draft.payload_without_record_id()
    )
    return DevelopmentRoutingReferenceRecord.from_payload(payload)


def _persistence_runner() -> ContentRoutingDirectionalDiagnosisRunner:
    protocol, reference_manifest, probe_manifest = _protocol_bundle()
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(
            ROOT / "configs/experiments/internal_execution_components.json"
        )
    )
    callback_sequence = tuple(range(20))
    runtime = create_runtime_adapter(
        FakeContentBackend(
            callback_sequences=(
                callback_sequence,
                callback_sequence,
                callback_sequence,
                callback_sequence,
            )
        )
    )
    runtime.initialize("cpu")
    semantic = object.__new__(DevelopmentSemanticObservationProducer)
    registered_root = "routing-persistence-test-key"
    return ContentRoutingDirectionalDiagnosisRunner(
        protocol=protocol,
        reference_manifest=reference_manifest,
        probe_manifest=probe_manifest,
        adapter=adapter,
        runtime_adapter=runtime,
        semantic_producer=semantic,
        method_code_revision="a" * 40,
        registered_root_key=registered_root,
        root_key_public_digest=(
            identify_root_key(registered_root).root_key_public_digest
        ),
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest="b" * 64,
        candidate_config_digest="c" * 64,
    )


def _bound_reference_record(
    runner: ContentRoutingDirectionalDiagnosisRunner,
    intent,
    *,
    failure_class: str | None = None,
) -> DevelopmentRoutingReferenceRecord:
    ordinal = intent.unit_index - 2
    source = (
        _persistent_reference_record(ordinal)
        if failure_class is None
        else _failed_reference_record(ordinal, failure_class)
    )
    record = replace(
        source,
        record_id="0" * 64,
        run_id=runner.protocol.run_id,
        protocol_digest=runner.protocol_digest,
        method_code_revision=runner.method_code_revision,
        unit_index=intent.unit_index,
        source_cluster_ordinal=ordinal,
        candidate_config_digest=runner.candidate_config_digest,
        attempt_index=intent.attempt_index,
        retry_parent_intent_digest=intent.parent_attempt_intent_digest,
        maximum_duration_seconds=intent.maximum_duration_seconds,
    )
    return replace(
        record,
        record_id=canonical_development_value_digest(
            record.payload_without_record_id()
        ),
    )


def _blind_score_rows(
    ordinal: int,
    *,
    routed_wins: bool,
) -> tuple[ContentRoutingBlindScoreObservation, ...]:
    uniform_score = 0.20 + ordinal * 0.001
    routed_score = uniform_score + (0.01 if routed_wins else -0.01)
    rows: list[ContentRoutingBlindScoreObservation] = []
    for arm_id, registered_score, candidate_digest, observation_digest in (
        ("routed", routed_score, f"{ordinal + 101:064x}", f"{ordinal + 401:064x}"),
        ("uniform", uniform_score, f"{ordinal + 301:064x}", f"{ordinal + 501:064x}"),
    ):
        common = {
            "arm_id": arm_id,
            "formal_mode": "hf_only",
            "content_detector_identity": "a" * 64,
            "content_config_digest": "1" * 64,
            "hf_detector_identity": "b" * 64,
            "hf_detector_config_digest": "2" * 64,
            "root_key_public_digest": "3" * 64,
        }
        rows.append(
            create_content_routing_blind_score_observation(
                **common,
                control_role="registered",
                wrong_key_index=None,
                content_score=registered_score,
                hf_score=registered_score,
                content_input_image_digest=candidate_digest,
                hf_observation_digest=observation_digest,
                hf_template_digest="4" * 64,
                key_role="registered",
            )
        )
        rows.append(
            create_content_routing_blind_score_observation(
                **common,
                control_role="paired_clean_primary_null",
                wrong_key_index=None,
                content_score=-0.08,
                hf_score=-0.08,
                content_input_image_digest="5" * 64,
                hf_observation_digest=f"{ordinal + 601:064x}",
                hf_template_digest="4" * 64,
                key_role="registered",
            )
        )
        for wrong_key_index in range(4):
            wrong_score = -0.03 + wrong_key_index * 0.01
            rows.append(
                create_content_routing_blind_score_observation(
                    **common,
                    control_role="wrong_key_control",
                    wrong_key_index=wrong_key_index,
                    content_score=wrong_score,
                    hf_score=wrong_score,
                    content_input_image_digest=candidate_digest,
                    hf_observation_digest=observation_digest,
                    hf_template_digest=f"{wrong_key_index + 7:064x}",
                    key_role="wrong",
                )
            )
    return tuple(rows)


def _observation(ordinal: int, *, routed_wins: bool = True, l2: float = 0.01):
    return create_content_routing_directional_observation(
        cluster_ordinal=ordinal,
        fold_index=ordinal % 4,
        blind_score_observations=_blind_score_rows(
            ordinal,
            routed_wins=routed_wins,
        ),
        routed_mean_mask_lf=0.4,
        routed_mean_mask_hf=0.3,
        uniform_mean_mask_lf=1.0,
        uniform_mean_mask_hf=1.0,
        routed_clean_to_watermarked_rgb_relative_l2=l2,
        uniform_clean_to_watermarked_rgb_relative_l2=l2,
        routed_realized_relative_l2=l2,
        uniform_realized_relative_l2=l2,
        routed_materialization_integrity_status="passed",
        uniform_materialization_integrity_status="passed",
        routed_materialization_budget_status="accepted",
        uniform_materialization_budget_status="accepted",
        public_content_operation="FormalHfContentDetectionOperation",
        preprocessing_identity="rgb8_public_image_float32_unit_interval",
        routed_route_digest=f"{ordinal + 101:064x}",
        uniform_route_digest=f"{ordinal + 201:064x}",
        cross_fit_reference_digest=f"{ordinal + 301:064x}",
        routed_candidate_observation_digest=f"{ordinal + 401:064x}",
        uniform_candidate_observation_digest=f"{ordinal + 501:064x}",
        paired_clean_observation_digest=f"{ordinal + 601:064x}",
        failure_class=None,
    )


def test_content_routing_protocol_freezes_forty_two_attempt_zero_units() -> None:
    protocol, reference, probes = _protocol_bundle()
    assert (
        protocol.run_id
        == "ceg_wm_content_routing_backend_binding_correction_diagnosis"
    )
    assert protocol.mixing_coefficient == 0.50
    assert (
        protocol.passing_request
        == "allow_request_for_fixed_half_routing_directional_validation"
    )
    assert "fixed_half_mixing_only" in protocol.claim_boundary
    assert "no_alpha_generalization" in protocol.claim_boundary
    raw = json.loads(CONFIG.read_text("utf-8"))
    assert raw["mixing_coefficient"] == 0.50
    assert {
        key for key in raw if "alpha" in key or "mixing" in key
    } == {"mixing_coefficient"}
    assert len(protocol.unit_roster) == 42
    assert len(reference.entries) == 32
    assert len(probes.entries) == 8
    assert all(unit.maximum_record_attempts == 1 for unit in protocol.unit_roster)
    assert tuple(unit.unit_index for unit in protocol.unit_roster) == tuple(range(42))
    operational = protocol.unit_roster[:2]
    assert tuple(unit.source_cluster_ordinal for unit in operational) == (0, 1)
    assert all(unit.phase == "development_environment_preflight" for unit in operational)
    assert all(
        unit.responsibility_id == "development_environment_preflight"
        and unit.content_branch_id == "development_environment_preflight"
        and unit.geometry_case_id == "geometry_case_not_applicable"
        for unit in operational
    )
    assert len(protocol.operational_units) == 2
    for ordinal, authority in enumerate(protocol.operational_units):
        assert authority.unit_index == ordinal
        assert authority.source_cluster_ordinal == ordinal
        assert authority.operational_role == "environment_runtime_throughput_preflight"
        assert authority.case_ids == (
            "environment_identity_preflight",
            "runtime_identity_preflight",
            "throughput_preflight",
        )
        assert authority.responsibility_result_digest_keys == (
            "content_embedder",
        )
        assert authority.counts_as_scientific_coverage is False
        assert authority.scientific_claims_supported is False


def test_content_routing_cross_fit_excludes_probe_fold_and_uses_twenty_four() -> None:
    _, reference, probes = _protocol_bundle()
    for probe in probes.entries:
        selected = reference_entries_for_probe(probe, reference)
        assert len(selected) == 24
        assert all(item.fold_index != probe.fold_index for item in selected)
        assert {item.fold_index for item in selected} == ({0, 1, 2, 3} - {probe.fold_index})


def test_content_routing_manifests_are_disjoint_on_all_frozen_axes() -> None:
    protocol, reference, probes = _protocol_bundle()
    assert protocol.future_split_exclusion_roles == (
        "routing_directional_validation",
        "candidate_selection",
        "content_threshold_fit",
        "rescue_window_fit",
        "geometry_reliability_fit",
        "end_to_end_calibration_check",
        "evaluation",
    )
    assert {item.cluster_identity for item in reference.entries}.isdisjoint(
        {item.cluster_identity for item in probes.entries}
    )
    assert {item.prompt_digest for item in reference.entries}.isdisjoint(
        {item.prompt_digest for item in probes.entries}
    )
    assert {item.generation_seed for item in reference.entries}.isdisjoint(
        {item.generation_seed for item in probes.entries}
    )
    assert {
        item.image_lineage_digest(role_id=reference.role_id)
        for item in reference.entries
    }.isdisjoint(
        {
            item.image_lineage_digest(role_id=probes.role_id)
            for item in probes.entries
        }
    )
    assert reference.seed_namespace != probes.seed_namespace
    assert reference.key_family_namespace != probes.key_family_namespace


def test_content_routing_manifest_digest_drift_fails_closed(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["reference_fit_manifest_file_sha256"] = "0" * 64
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(
        ContentRoutingDirectionalProtocolError,
        match="manifest file digest drifted",
    ):
        load_content_routing_directional_protocol(path, repository_root=ROOT)


def test_content_routing_operational_authority_drift_fails_closed(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["operational_units"][1]["scientific_claims_supported"] = True
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(
        ContentRoutingDirectionalProtocolError,
        match="operational unit identity drifted",
    ):
        load_content_routing_directional_protocol(path, repository_root=ROOT)


def test_content_routing_optional_mixing_surface_fails_closed(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["mixing_coefficient_candidates"] = [0.25, 0.50, 0.75]
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(
        ContentRoutingDirectionalProtocolError,
        match="routing protocol schema drifted",
    ):
        load_content_routing_directional_protocol(path, repository_root=ROOT)


def test_routing_reference_fits_only_texture_response_and_sensitivity() -> None:
    reference = fit_content_routing_fold_reference(
        _reference_measurements(),
        probe_fold_index=0,
    )
    assert len(reference.fit_cluster_ordinals) == 24
    assert all(ordinal % 4 != 0 for ordinal in reference.fit_cluster_ordinals)
    assert not any("semantic" in field.name for field in fields(reference))
    assert reference.texture_gradient_reference == 31.25
    assert reference.latent_response_reference == 131.25
    assert reference.local_sensitivity_reference == 231.25


def test_routing_protocol_freezes_distinct_raw_and_fit_support_rules() -> None:
    protocol, _reference, _probes = _protocol_bundle()
    assert protocol.raw_reference_support_rule == "finite_nonnegative"
    assert protocol.fit_support_rule == "strictly_positive_values_only"
    assert protocol.reference_quantile_rule == "exact_nearest_rank_positive_p95"

    digest_arguments = {
        "adapter_config_digest": "1" * 64,
        "probe_manifest_digest": "2" * 64,
        "reference_manifest_digest": "3" * 64,
        "runtime_config_digest": "4" * 64,
    }
    identity = entrypoint_module._routing_candidate_config_digest(
        protocol=protocol,
        **digest_arguments,
    )
    assert identity != entrypoint_module._routing_candidate_config_digest(
        protocol=replace(protocol, fit_support_rule="drifted_fit_support"),
        **digest_arguments,
    )
    assert identity != entrypoint_module._routing_candidate_config_digest(
        protocol=replace(protocol, raw_reference_support_rule="drifted_raw_support"),
        **digest_arguments,
    )
    assert identity != entrypoint_module._routing_candidate_config_digest(
        protocol=replace(protocol, reference_quantile_rule="drifted_quantile"),
        **digest_arguments,
    )


@pytest.mark.parametrize(
    "changed_field",
    (
        "texture_gradient_values",
        "texture_spatial_shape",
        "response_ratio_values",
        "response_spatial_shape",
        "sensitivity_ratio_values",
        "sensitivity_spatial_shape",
    ),
)
def test_routing_reference_identity_changes_with_each_spatial_statistic(
    changed_field: str,
) -> None:
    original_values = {
        "cluster_ordinal": 0,
        "fold_index": 0,
        "texture_gradient_values": (1.0, 1.5),
        "texture_spatial_shape": (1, 2),
        "response_ratio_values": (2.0, 2.5),
        "response_spatial_shape": (1, 2),
        "sensitivity_ratio_values": (3.0, 3.5),
        "sensitivity_spatial_shape": (1, 2),
    }
    original = create_content_routing_reference_measurement(**original_values)
    changed_value = (
        (2, 1)
        if changed_field.endswith("_shape")
        else (
            original_values[changed_field][0] + 0.25,
            original_values[changed_field][1],
        )
    )
    changed_values = {**original_values, changed_field: changed_value}
    changed = create_content_routing_reference_measurement(**changed_values)
    assert changed.observation_identity != original.observation_identity


def test_routing_reference_fit_replays_thirty_two_persistent_records() -> None:
    records = tuple(_persistent_reference_record(ordinal) for ordinal in range(32))
    measurements = tuple(
        ContentRoutingDirectionalDiagnosisRunner.reference_measurement_from_committed_record(
            record
        )
        for record in records
    )
    assert len(measurements) == 32
    reference = fit_content_routing_fold_reference(
        measurements,
        probe_fold_index=0,
    )
    assert len(reference.fit_cluster_ordinals) == 24
    assert all(ordinal % 4 != 0 for ordinal in reference.fit_cluster_ordinals)
    assert reference.texture_gradient_reference == 100.0
    assert reference.latent_response_reference == 200.0
    assert reference.local_sensitivity_reference == 300.0
    hierarchical_texture_reference = exact_nearest_rank_positive_p95(
        tuple(
            exact_nearest_rank_positive_p95(measurement.texture_gradient_values)
            for measurement in measurements
            if measurement.fold_index != 0
        )
    )
    assert hierarchical_texture_reference == 31.25
    assert reference.texture_gradient_reference != hierarchical_texture_reference
    assert not any("semantic" in field.name for field in fields(measurements[0]))


@pytest.mark.parametrize(
    ("failure_classes", "expected"),
    (
        (("resource_failure",), "resource_failure"),
        (("implementation_failure",), "implementation_failure"),
        (
            (("resource_failure", "implementation_failure")),
            "implementation_failure",
        ),
    ),
)
def test_routing_reference_dependency_classification_preserves_terminal_failures(
    failure_classes: tuple[str, ...],
    expected: str,
) -> None:
    records = list(_persistent_reference_record(index) for index in range(32))
    for offset, failure_class in enumerate(failure_classes):
        records[31 - offset] = _failed_reference_record(
            31 - offset,
            failure_class,
        )
    assert _reference_dependency_failure_class(tuple(records)) == expected


@pytest.mark.parametrize(
    ("records", "expected_error"),
    (
        (
            tuple(_persistent_reference_record(index) for index in range(31)),
            ContentRoutingDirectionalEntrypointError,
        ),
        (
            (
                *tuple(_persistent_reference_record(index) for index in range(31)),
                replace(
                    _failed_reference_record(31, "resource_failure"),
                    failure_class=None,
                ),
            ),
            DevelopmentRecordError,
        ),
        (
            (
                *tuple(_persistent_reference_record(index) for index in range(31)),
                replace(
                    _failed_reference_record(31, "resource_failure"),
                    execution_status="retry",
                ),
            ),
            DevelopmentRecordError,
        ),
    ),
)
def test_routing_reference_dependency_unknown_missing_or_nonterminal_fails_closed(
    records: tuple[DevelopmentRoutingReferenceRecord, ...],
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        _reference_dependency_failure_class(records)


def test_routing_reference_positive_support_budget_validates_all_four_folds() -> None:
    records = tuple(_persistent_reference_record(index) for index in range(32))
    references = (
        ContentRoutingDirectionalDiagnosisRunner.validate_reference_positive_support(
            records
        )
    )
    assert tuple(reference.probe_fold_index for reference in references) == (
        0,
        1,
        2,
        3,
    )
    assert all(len(reference.fit_cluster_ordinals) == 24 for reference in references)


def test_routing_reference_positive_support_budget_fails_before_probe_calls() -> None:
    records = tuple(
        _zero_reference_record(index, axis="texture_gradient_values")
        for index in range(32)
    )
    with pytest.raises(
        ContentRoutingReferencePositiveSupportAbsentError,
        match="routing_reference_positive_support_absent",
    ):
        ContentRoutingDirectionalDiagnosisRunner.validate_reference_positive_support(
            records
        )


@pytest.mark.parametrize(
    ("reference_failures", "expected_failure_class", "expected_outcome"),
    (
        ({31: "resource_failure"}, "resource_failure", "resource_blocked"),
        (
            {31: "implementation_failure"},
            "implementation_failure",
            "implementation_blocked",
        ),
        (
            {30: "resource_failure", 31: "implementation_failure"},
            "implementation_failure",
            "implementation_blocked",
        ),
    ),
)
def test_routing_dependency_block_commits_fixed_probe_denominator_and_recovers(
    tmp_path: Path,
    reference_failures: dict[int, str],
    expected_failure_class: str,
    expected_outcome: str,
) -> None:
    runner = _persistence_runner()
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.protocol.run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=runner.method_code_revision,
            protocol_digest=runner.protocol_digest,
            execution_intent_authority_digest=(
                runner.execution_intent_authority_digest
            ),
            input_manifest_digest="d" * 64,
            candidate_config_digest=runner.candidate_config_digest,
            unit_roster_digest=runner.protocol.unit_roster_digest,
        ),
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    epoch = int(time.time())
    lease = store.acquire_lease(
        session_id="routing_dependency_block_session",
        now_epoch_seconds=epoch,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch)
    base = torch.ones((1, 16, 2, 2), dtype=torch.float16)
    for unit_index in range(2):
        intent = store.create_session_intent(
            cursor,
            lease,
            now_epoch_seconds=epoch + 1 + unit_index * 2,
        )
        record = runner.execute_operational_unit(
            unit_index=unit_index,
            base_latent=base,
            intent=intent,
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + 2 + unit_index * 2,
        )
    for ordinal in range(32):
        intent = store.create_session_intent(
            cursor,
            lease,
            now_epoch_seconds=epoch + 10 + ordinal * 2,
        )
        record = _bound_reference_record(
            runner,
            intent,
            failure_class=reference_failures.get(ordinal),
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + 11 + ordinal * 2,
        )
    assert len(cursor.routing_reference_records) == 32 - len(reference_failures)
    assert len(cursor.terminal_routing_reference_records) == 32
    failure_class = _reference_dependency_failure_class(
        cursor.terminal_routing_reference_records
    )
    records = _commit_dependency_blocked_probe_records(
        store=store,
        cursor=cursor,
        lease=lease,
        runner=runner,
        failure_class=failure_class,
        raw_secret_values=(),
    )
    aggregate = _replay_aggregate(records)
    recovered = store.open_session_cursor(lease, now_epoch_seconds=epoch + 200)

    assert cursor.next_unit_index == 42
    assert recovered.next_unit_index == 42
    assert len(cursor.committed_units) == 42
    assert len(records) == 8
    assert all(type(record) is DevelopmentScientificRecord for record in records)
    assert all(record.execution_status == "failed" for record in records)
    assert all(record.failure_class == expected_failure_class for record in records)
    assert all(record.operation_result_payload == {} for record in records)
    for record in records:
        replayed = DevelopmentScientificRecord.from_payload(
            json.loads(json.dumps(record.payload()))
        )
        assert replayed.record_id == record.record_id
    assert aggregate.expected_probe_count == 8
    assert aggregate.successful_probe_count == 0
    assert aggregate.failed_probe_count == 8
    assert aggregate.outcome == expected_outcome
    assert aggregate.outcome != "routing_directional_signal_not_observed"
    assert "execute_probe_unit" not in inspect.getsource(
        _commit_dependency_blocked_probe_records
    )
    assert len(tuple((store.run_root / "intents").glob("*.json"))) == 42
    assert len(tuple((store.run_root / "bundles").glob("*.zip"))) == 42
    assert len(
        tuple((store.run_root / "markers").glob("*.COMMITTED.json"))
    ) == 42
    with pytest.raises(DevelopmentPersistenceError):
        store.create_session_intent(
            recovered,
            lease,
            now_epoch_seconds=epoch + 201,
        )


def test_routing_positive_support_block_uses_fixed_denominator_and_stable_reason(
    tmp_path: Path,
) -> None:
    runner = _persistence_runner()
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.protocol.run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=runner.method_code_revision,
            protocol_digest=runner.protocol_digest,
            execution_intent_authority_digest=(
                runner.execution_intent_authority_digest
            ),
            input_manifest_digest="d" * 64,
            candidate_config_digest=runner.candidate_config_digest,
            unit_roster_digest=runner.protocol.unit_roster_digest,
        ),
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    epoch = int(time.time())
    lease = store.acquire_lease(
        session_id="routing_positive_support_block_session",
        now_epoch_seconds=epoch,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch)
    base = torch.ones((1, 16, 2, 2), dtype=torch.float16)
    for unit_index in range(2):
        intent = store.create_session_intent(
            cursor, lease, now_epoch_seconds=epoch + 1 + unit_index * 2
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=runner.execute_operational_unit(
                unit_index=unit_index,
                base_latent=base,
                intent=intent,
            ),
            now_epoch_seconds=epoch + 2 + unit_index * 2,
        )
    for ordinal in range(32):
        intent = store.create_session_intent(
            cursor, lease, now_epoch_seconds=epoch + 10 + ordinal * 2
        )
        record = _bound_reference_record(runner, intent)
        zero_payload = _zero_reference_record(ordinal).payload()
        record_payload = record.payload()
        record_payload.update(
            record_id="0" * 64,
            measurement_payload=zero_payload["measurement_payload"],
        )
        draft = DevelopmentRoutingReferenceRecord(**record_payload)
        record_payload["record_id"] = canonical_development_value_digest(
            draft.payload_without_record_id()
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=DevelopmentRoutingReferenceRecord.from_payload(record_payload),
            now_epoch_seconds=epoch + 11 + ordinal * 2,
        )
    assert cursor.next_unit_index == 34
    with pytest.raises(ContentRoutingReferencePositiveSupportAbsentError):
        runner.validate_reference_positive_support(cursor.routing_reference_records)
    records = _commit_dependency_blocked_probe_records(
        store=store,
        cursor=cursor,
        lease=lease,
        runner=runner,
        failure_class="implementation_failure",
        failure_reason="routing_reference_positive_support_absent",
        raw_secret_values=(),
    )
    aggregate = _replay_aggregate(records)
    recovered = store.open_session_cursor(lease, now_epoch_seconds=epoch + 200)
    replayed_records = _commit_dependency_blocked_probe_records(
        store=store,
        cursor=recovered,
        lease=lease,
        runner=runner,
        failure_class="implementation_failure",
        failure_reason="routing_reference_positive_support_absent",
        raw_secret_values=(),
    )
    assert len(records) == 8
    assert all(
        record.failure_reason == "routing_reference_positive_support_absent"
        and record.execution_status == "failed"
        and record.operation_result_payload == {}
        for record in records
    )
    assert aggregate.outcome == "implementation_blocked"
    assert aggregate.successful_probe_count == 0
    assert recovered.next_unit_index == 42
    assert len(recovered.committed_units) == 42
    assert tuple(record.record_id for record in replayed_records) == tuple(
        record.record_id for record in records
    )


def test_routing_positive_support_budget_precedes_probe_intent_and_execution() -> None:
    source = inspect.getsource(
        entrypoint_module.execute_content_routing_directional_diagnosis_session
    )
    budget_call = source.index("runner.validate_reference_positive_support(")
    next_intent = source.index(
        "intent = store.create_session_intent",
        source.index("if unit.unit_index == 34:"),
    )
    probe_call = source.index("record = runner.execute_probe_unit(")
    assert budget_call < next_intent < probe_call
    assert source.count("runner.validate_reference_positive_support(") == 1


def test_routing_dependency_block_rejects_unknown_failure_reason(
    tmp_path: Path,
) -> None:
    runner = _persistence_runner()
    store = object.__new__(DevelopmentPersistentStore)
    with pytest.raises(
        ContentRoutingDirectionalEntrypointError,
        match="failure reason is invalid",
    ):
        _commit_dependency_blocked_probe_records(
            store=store,
            cursor=object(),
            lease=object(),
            runner=runner,
            failure_class="implementation_failure",
            failure_reason="unregistered_reference_reason",
            raw_secret_values=(),
        )


@pytest.mark.parametrize(
    "tamper_role",
    ("duplicate_marker", "foreign_marker", "mismatched_marker"),
)
def test_routing_persistence_recovery_rejects_invalid_committed_markers(
    tmp_path: Path,
    tamper_role: str,
) -> None:
    runner = _persistence_runner()
    identity = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest="d" * 64,
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )
    bindings = runner.create_persistence_unit_bindings()
    source_root = tmp_path / "source"
    store = DevelopmentPersistentStore(
        source_root,
        run_id=runner.protocol.run_id,
        worker_identity=identity,
        registered_unit_bindings=bindings,
    )
    epoch = int(time.time())
    lease = store.acquire_lease(
        session_id="routing_marker_integrity_session",
        now_epoch_seconds=epoch,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch)
    intent = store.create_session_intent(
        cursor,
        lease,
        now_epoch_seconds=epoch + 1,
    )
    record = runner.execute_operational_unit(
        unit_index=0,
        base_latent=torch.ones((1, 16, 2, 2), dtype=torch.float16),
        intent=intent,
    )
    store.commit_session_unit(
        cursor,
        lease,
        intent,
        record=record,
        now_epoch_seconds=epoch + 2,
    )
    tampered_root = tmp_path / tamper_role
    shutil.copytree(source_root, tampered_root)
    tampered_store = DevelopmentPersistentStore(
        tampered_root,
        run_id=runner.protocol.run_id,
        worker_identity=identity,
        registered_unit_bindings=bindings,
    )
    marker = next((tampered_store.run_root / "markers").glob("*.COMMITTED.json"))
    payload = json.loads(marker.read_text("utf-8"))
    if tamper_role == "duplicate_marker":
        duplicate = marker.with_name(
            "development_unit_0001__attempt_0.COMMITTED.json"
        )
        duplicate.write_bytes(marker.read_bytes())
    elif tamper_role == "foreign_marker":
        payload["run_id"] = "ceg_wm_foreign_routing_run"
        marker.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    else:
        payload["record_digest"] = "f" * 64
        marker.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    with pytest.raises(DevelopmentPersistenceError):
        tampered_store.recover(now_epoch_seconds=epoch + 3)


def test_routing_persistence_dangling_attempt_fails_closed_without_rerun(
    tmp_path: Path,
) -> None:
    runner = _persistence_runner()
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.protocol.run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=runner.method_code_revision,
            protocol_digest=runner.protocol_digest,
            execution_intent_authority_digest=(
                runner.execution_intent_authority_digest
            ),
            input_manifest_digest="d" * 64,
            candidate_config_digest=runner.candidate_config_digest,
            unit_roster_digest=runner.protocol.unit_roster_digest,
        ),
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    epoch = int(time.time())
    lease = store.acquire_lease(
        session_id="routing_dangling_attempt_session",
        now_epoch_seconds=epoch,
        lease_duration_seconds=10,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch)
    intent = store.create_session_intent(
        cursor,
        lease,
        now_epoch_seconds=epoch + 1,
    )
    assert intent.attempt_index == 0
    assert intent.maximum_record_attempts == 1
    assert cursor.committed_units == ()
    with pytest.raises(
        DevelopmentPersistenceError,
        match="interrupted unit exhausted frozen attempts",
    ):
        store.recover(now_epoch_seconds=epoch + 11)


def test_routing_real_public_success_chain_commits_recovers_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, reference_manifest, probe_manifest = _protocol_bundle()
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(
            ROOT / "configs/experiments/internal_execution_components.json"
        )
    )
    class RoutingExecutionBackend(_RoutingBackend):
        def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
            self.decode_inputs.append(latent.detach().clone())
            values = latent.detach().to(torch.float32)
            channel_count = int(values.shape[1])
            weights = torch.linspace(
                0.5,
                1.5,
                steps=channel_count,
                dtype=torch.float32,
                device=values.device,
            ).reshape(1, channel_count, 1, 1)
            normalized = (values * weights).sum(dim=1, keepdim=True) / weights.sum()
            height, width = (int(value) for value in values.shape[-2:])
            vertical = torch.linspace(
                0.0,
                0.12,
                steps=height,
                device=values.device,
            ).reshape(1, 1, height, 1)
            horizontal = torch.linspace(
                0.0,
                0.08,
                steps=width,
                device=values.device,
            ).reshape(1, 1, 1, width)
            global_weights = torch.linspace(
                0.2,
                1.8,
                steps=values.numel(),
                device=values.device,
            ).reshape_as(values)
            globally_coupled = (values * global_weights).sum() / global_weights.sum()
            normalized = normalized + vertical + horizontal + globally_coupled * 0.05
            return torch.cat(
                (normalized, normalized * 0.9, normalized * 0.8),
                dim=1,
            ).clamp(-0.9, 0.9)

        def vae_encode(self, image: torch.Tensor) -> _Posterior:
            mean = image.detach().to(torch.float32).mean(dim=1, keepdim=True)
            return _Posterior(mean.repeat(1, 16, 1, 1).to(torch.float16))

    backend = RoutingExecutionBackend()
    runtime = create_runtime_adapter(backend)
    runtime.initialize("cpu")
    semantic = object.__new__(DevelopmentSemanticObservationProducer)
    calls: Counter[str] = Counter()

    def observe_semantic(
        routing_rgb: torch.Tensor,
        prompt: str,
    ) -> SpatialRoutingObservation:
        calls["semantic_observation"] += 1
        assert routing_rgb.ndim == 4
        assert tuple(routing_rgb.shape[:2]) == (1, 3)
        assert prompt
        height, width = (int(value) for value in routing_rgb.shape[-2:])
        return SpatialRoutingObservation(
            values=tuple(0.5 for _ in range(height * width)),
            spatial_shape=(height, width),
            source_identity_digest=f"{calls['semantic_observation']:064x}",
        )

    semantic.observe = observe_semantic
    for method_name in (
        "route_content",
        "build_lf_carrier",
        "build_hf_carrier",
        "embed_content",
        "detect_hf",
        "detect_content",
    ):
        original = getattr(CegWmExperimentAdapter, method_name)

        def counted_adapter_call(
            self,
            *args,
            _method_name=method_name,
            _original=original,
            **kwargs,
        ):
            calls[_method_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(
            CegWmExperimentAdapter,
            method_name,
            counted_adapter_call,
        )
    original_write = Sd35RuntimeAdapter.execute_content_write_and_vae

    def counted_public_write(self, *args, **kwargs):
        calls["execute_content_write_and_vae"] += 1
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "execute_content_write_and_vae",
        counted_public_write,
    )
    original_measure = Sd35RuntimeAdapter.measure_generation_routing_reference_inputs

    def counted_reference_measurement(self, *args, **kwargs):
        calls["measure_generation_routing_reference_inputs"] += 1
        return original_measure(self, *args, **kwargs)

    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "measure_generation_routing_reference_inputs",
        counted_reference_measurement,
    )
    carrier_key_domains: list[tuple[str, str, str]] = []
    for module_name in (
        "main.content_chain.hf_carrier",
        "main.content_chain.lf_carrier",
    ):
        carrier_module = importlib.import_module(module_name)
        original_schedule = carrier_module.key_schedule_sha256_counter

        def traced_schedule(
            *args,
            _original=original_schedule,
            **kwargs,
        ):
            domain_fields = args[1]
            carrier_key_domains.append(
                (
                    domain_fields["candidate_id"],
                    domain_fields["operator"],
                    domain_fields["responsibility_domain"],
                )
            )
            return _original(*args, **kwargs)

        monkeypatch.setattr(
            carrier_module,
            "key_schedule_sha256_counter",
            traced_schedule,
        )
    registered_root = _registered_experiment_root(
        "routing-real-public-success-key",
        protocol_digest=protocol.digest(),
        reference_manifest_digest=canonical_digest(asdict(reference_manifest)),
        probe_manifest_digest=canonical_digest(asdict(probe_manifest)),
    )

    def create_runner() -> ContentRoutingDirectionalDiagnosisRunner:
        return ContentRoutingDirectionalDiagnosisRunner(
            protocol=protocol,
            reference_manifest=reference_manifest,
            probe_manifest=probe_manifest,
            adapter=adapter,
            runtime_adapter=runtime,
            semantic_producer=semantic,
            method_code_revision="a" * 40,
            registered_root_key=registered_root,
            root_key_public_digest=(
                identify_root_key(registered_root).root_key_public_digest
            ),
            protocol_digest=protocol.digest(),
            execution_intent_authority_digest="b" * 64,
            candidate_config_digest="c" * 64,
        )

    runner = create_runner()
    worker_identity = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest="d" * 64,
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )

    def create_store(current_runner):
        return DevelopmentPersistentStore(
            tmp_path,
            run_id=current_runner.protocol.run_id,
            worker_identity=worker_identity,
            registered_unit_bindings=(
                current_runner.create_persistence_unit_bindings()
            ),
        )

    store = create_store(runner)
    epoch = int(time.time())
    lease = store.acquire_lease(
        session_id="routing_real_public_success_session",
        now_epoch_seconds=epoch,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch)
    base = torch.linspace(
        0.05,
        0.45,
        steps=16 * 4 * 4,
        dtype=torch.float16,
    ).reshape(1, 16, 4, 4)
    for unit_index in range(2):
        intent = store.create_session_intent(
            cursor,
            lease,
            now_epoch_seconds=epoch + unit_index * 2 + 1,
        )
        record = runner.execute_operational_unit(
            unit_index=unit_index,
            base_latent=base,
            intent=intent,
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + unit_index * 2 + 2,
        )
    for ordinal in range(32):
        intent = store.create_session_intent(
            cursor,
            lease,
            now_epoch_seconds=epoch + ordinal * 2 + 10,
        )
        record = runner.execute_reference_fit_unit(
            unit_index=ordinal + 2,
            base_latent=base,
            intent=intent,
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + ordinal * 2 + 11,
        )
        if ordinal == 15:
            generation_calls = backend.generation_calls
            runner = create_runner()
            store = create_store(runner)
            cursor = store.open_session_cursor(
                lease,
                now_epoch_seconds=epoch + 100,
            )
            assert cursor.next_unit_index == 18
            assert backend.generation_calls == generation_calls
    assert len(cursor.routing_reference_records) == 32
    for reference_record in cursor.routing_reference_records:
        payload = reference_record.measurement_payload
        for field_name in (
            "texture_gradient_values",
            "response_ratio_values",
            "sensitivity_ratio_values",
        ):
            assert min(payload[field_name]) > 0.0, (
                reference_record.source_cluster_ordinal,
                field_name,
                payload[field_name],
            )
    for ordinal in range(8):
        intent = store.create_session_intent(
            cursor,
            lease,
            now_epoch_seconds=epoch + ordinal * 2 + 200,
        )
        record = runner.execute_probe_unit(
            unit_index=ordinal + 34,
            base_latent=base,
            intent=intent,
            reference_records=cursor.routing_reference_records,
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            now_epoch_seconds=epoch + ordinal * 2 + 201,
        )
        checked = DevelopmentScientificRecord.from_payload(
            json.loads(json.dumps(record.payload()))
        )
        observation = checked.operation_result_payload["routing_observation"]
        rows = observation["blind_score_observations"]
        assert len(rows) == 12
        assert observation["routed_realized_relative_l2"] <= 0.012
        assert observation["uniform_realized_relative_l2"] <= 0.012
        assert observation["routed_materialization_budget_status"] == "accepted"
        assert observation["uniform_materialization_budget_status"] == "accepted"
        assert rows[1]["content_input_image_digest"] == rows[7][
            "content_input_image_digest"
        ]
        assert rows[0]["hf_template_digest"] == rows[1]["hf_template_digest"]
        assert rows[6]["hf_template_digest"] == rows[7]["hf_template_digest"]
        for wrong_index in range(4):
            routed_wrong = rows[wrong_index + 2]
            uniform_wrong = rows[wrong_index + 8]
            assert routed_wrong["wrong_key_index"] == wrong_index
            assert uniform_wrong["wrong_key_index"] == wrong_index
            assert routed_wrong["hf_template_digest"] == uniform_wrong[
                "hf_template_digest"
            ]
    evidence = store.verified_terminal_scientific_evidence(
        now_epoch_seconds=epoch + 300
    )
    records = tuple(record for record, _marker in evidence)
    aggregate = _replay_aggregate(records)
    final_cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch + 301)

    assert protocol.mixing_coefficient == 0.50
    assert len(final_cursor.committed_units) == 42
    assert final_cursor.next_unit_index == 42
    assert tuple(marker.unit_index for marker in final_cursor.committed_units) == tuple(
        range(42)
    )
    assert len(records) == 8
    assert all(record.execution_status == "success" for record in records)
    assert aggregate.expected_probe_count == 8
    assert aggregate.successful_probe_count == 8
    assert aggregate.failed_probe_count == 0
    assert calls["semantic_observation"] == 8
    assert calls["measure_generation_routing_reference_inputs"] == 40
    assert calls["route_content"] == 18
    assert calls["build_lf_carrier"] == 18
    assert calls["build_hf_carrier"] == 18
    assert calls["embed_content"] == 18
    assert calls["execute_content_write_and_vae"] == 18
    assert calls["detect_hf"] == 96
    assert calls["detect_content"] == 96
    assert ("hf_sparse_tail", "carrier_template", "hf_carrier") in carrier_key_domains
    assert ("lf_low_pass", "carrier_template", "lf_carrier") in carrier_key_domains


def test_routing_registered_root_uses_frozen_hf_secret_domain() -> None:
    protocol, reference_manifest, probe_manifest = _protocol_bundle()
    digests = {
        "protocol_digest": protocol.digest(),
        "reference_manifest_digest": canonical_digest(asdict(reference_manifest)),
        "probe_manifest_digest": canonical_digest(asdict(probe_manifest)),
    }
    root = _registered_experiment_root(
        "routing-registered-root-test-key",
        **digests,
    )
    assert root.startswith("ceg-wm-content-routing-registered:")
    assert root != "routing-registered-root-test-key"
    assert identify_root_key(root) != identify_root_key(
        "routing-registered-root-test-key"
    )
    for digest_role in digests:
        changed = dict(digests)
        changed[digest_role] = "0" * 64
        assert (
            _registered_experiment_root(
                "routing-registered-root-test-key",
                **changed,
            )
            != root
        )


@pytest.mark.parametrize(
    ("failure_stage", "failure_type", "expected_class", "expected_close_count"),
    (
        ("backend", MemoryError, "resource_failure", 0),
        ("runtime", RuntimeError, "implementation_failure", 1),
        ("semantic", RuntimeError, "implementation_failure", 1),
    ),
)
def test_routing_startup_diagnostic_only_wraps_resource_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
    failure_type: type[Exception],
    expected_class: str,
    expected_close_count: int,
) -> None:
    failure = failure_type("must-not-persist")
    close_calls: list[str] = []

    class FakeRuntime:
        def initialize(self, requested_device: str):
            assert requested_device == "cuda"
            if failure_stage == "runtime":
                raise failure
            return SimpleNamespace(runtime_config_digest="1" * 64)

        def close(self) -> None:
            close_calls.append("close")

    def backend_factory(**_kwargs):
        if failure_stage == "backend":
            raise failure
        return object()

    monkeypatch.setattr(entrypoint_module, "Sd35PipelineBackend", backend_factory)
    monkeypatch.setattr(
        entrypoint_module,
        "Sd35RuntimeAdapter",
        lambda _backend, _configuration: FakeRuntime(),
    )

    def semantic_factory(**_kwargs):
        if failure_stage == "semantic":
            raise failure
        return object()

    monkeypatch.setattr(
        entrypoint_module,
        "DevelopmentSemanticObservationProducer",
        semantic_factory,
    )
    with pytest.raises(ContentRoutingDirectionalStartupError) as caught:
        _initialize_routing_resources(
            cache=tmp_path / "cache",
            persistent=tmp_path / "persistent",
            hf_token="startup-test-token",
            prompt="startup test prompt",
            runtime_configuration=object(),
        )
    assert caught.value.__cause__ is failure
    assert caught.value.failure_type == (
        f"{type(failure).__module__}.{type(failure).__qualname__}"
    )
    assert caught.value.failure_class == expected_class
    assert len(close_calls) == expected_close_count


def test_routing_session_uses_initialized_backend_before_first_intent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, int | None]] = []
    created: dict[str, object] = {}

    class FakeBackend:
        def set_development_generation_prompts(self, prompt: str) -> None:
            assert prompt
            events.append(("set_prompt", id(self)))

    def backend_factory(**_kwargs):
        backend = FakeBackend()
        created["backend"] = backend
        return backend

    def runtime_init(self, backend, configuration) -> None:
        created["runtime_backend"] = backend
        self._configuration = configuration

    def runtime_initialize(self, requested_device: str):
        assert requested_device == "cuda"
        return SimpleNamespace(
            runtime_config_digest=self._configuration.runtime_config_digest,
            image_height=8,
            image_width=8,
        )

    def runtime_close(self) -> None:
        events.append(("close", None))

    monkeypatch.setattr(entrypoint_module, "Sd35PipelineBackend", backend_factory)
    monkeypatch.setattr(Sd35RuntimeAdapter, "__init__", runtime_init)
    monkeypatch.setattr(Sd35RuntimeAdapter, "initialize", runtime_initialize)
    monkeypatch.setattr(Sd35RuntimeAdapter, "close", runtime_close)
    monkeypatch.setattr(
        DevelopmentSemanticObservationProducer,
        "__init__",
        lambda self, **_kwargs: None,
    )
    monkeypatch.setattr(
        entrypoint_module,
        "_base_latent",
        lambda *_args, **_kwargs: torch.zeros((1, 16, 1, 1), dtype=torch.float16),
    )
    package = tmp_path / "execution-package.zip"
    package.write_bytes(b"routing-package")
    monkeypatch.setattr(
        entrypoint_module,
        "_build_or_verify_package",
        lambda *_args, **_kwargs: package,
    )
    monkeypatch.setattr(entrypoint_module, "_sha256_file", lambda *_args: "b" * 64)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_args: "cpu-fixture")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args: 1)

    original_create_intent = DevelopmentPersistentStore.create_session_intent

    def traced_create_intent(self, cursor, lease, *, now_epoch_seconds: int):
        events.append(("intent", None))
        return original_create_intent(
            self,
            cursor,
            lease,
            now_epoch_seconds=now_epoch_seconds,
        )

    monkeypatch.setattr(
        DevelopmentPersistentStore,
        "create_session_intent",
        traced_create_intent,
    )

    def fail_operational(self, *, unit_index, base_latent, intent):
        assert unit_index == 0
        assert intent.unit_index == 0
        events.append(("unit0", None))
        raise RuntimeError("fixture operational stop")

    monkeypatch.setattr(
        ContentRoutingDirectionalDiagnosisRunner,
        "execute_operational_unit",
        fail_operational,
    )

    return_code, summary = (
        entrypoint_module.execute_content_routing_directional_diagnosis_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="ceg_wm_content_routing_backend_binding_correction_diagnosis",
            session_id="routing_backend_binding_session",
            execution_package_sha256="b" * 64,
            environment={"HF_TOKEN": "token", "CEG_WM_ROOT_KEY": "root"},
        )
    )

    assert return_code == 3
    assert summary["committed_unit_count"] == 0
    assert created["runtime_backend"] is created["backend"]
    assert events[:3] == [
        ("set_prompt", id(created["backend"])),
        ("intent", None),
        ("unit0", None),
    ]
    assert events.count(("set_prompt", id(created["backend"]))) == 1
    assert events.count(("close", None)) == 1
    intent_paths = tuple((tmp_path / "persistent").rglob("intents/*.json"))
    assert len(intent_paths) == 1
    assert intent_paths[0].name == "development_unit_0000__attempt_0.json"
    diagnostic_zip = Path(summary["diagnostic_zip"])
    with ZipFile(diagnostic_zip) as archive:
        diagnostic = json.loads(archive.read("diagnostic.json"))
    assert diagnostic["failure_type"] == "builtins.RuntimeError"
    assert "NameError" not in json.dumps(diagnostic, sort_keys=True)


@pytest.mark.parametrize(
    "failure_stage",
    (
        "protocol_loader",
        "component_loader",
        "runtime_configuration",
        "registered_root",
    ),
)
def test_routing_authority_failures_preserve_original_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    failure = RuntimeError(f"{failure_stage} failure")
    if failure_stage == "protocol_loader":
        monkeypatch.setattr(
            entrypoint_module,
            "load_content_routing_directional_protocol",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    elif failure_stage == "component_loader":
        monkeypatch.setattr(
            entrypoint_module,
            "load_ceg_wm_experiment_adapter_configuration",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    elif failure_stage == "runtime_configuration":
        monkeypatch.setattr(
            entrypoint_module,
            "load_runtime_configuration",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            entrypoint_module,
            "_registered_experiment_root",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    monkeypatch.setattr(
        entrypoint_module,
        "_initialize_routing_resources",
        lambda **_kwargs: pytest.fail("resource initialization must be unreachable"),
    )
    with pytest.raises(RuntimeError) as caught:
        entrypoint_module.execute_content_routing_directional_diagnosis_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="ceg_wm_content_routing_backend_binding_correction_diagnosis",
            session_id="routing_authority_failure_session",
            execution_package_sha256="b" * 64,
            environment={"HF_TOKEN": "token", "CEG_WM_ROOT_KEY": "root"},
        )
    assert caught.value is failure


def test_routing_run_and_root_invariant_failures_are_not_startup_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ContentRoutingDirectionalEntrypointError, match="run identity"):
        entrypoint_module.execute_content_routing_directional_diagnosis_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="ceg_wm_content_routing_registered_key_correction_diagnosis",
            session_id="routing_run_failure_session",
            execution_package_sha256="b" * 64,
            environment={"HF_TOKEN": "token", "CEG_WM_ROOT_KEY": "root"},
        )
    monkeypatch.setattr(
        entrypoint_module,
        "_registered_experiment_root",
        lambda root_key, **_kwargs: root_key,
    )
    with pytest.raises(ContentRoutingDirectionalEntrypointError, match="must differ"):
        entrypoint_module.execute_content_routing_directional_diagnosis_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="ceg_wm_content_routing_backend_binding_correction_diagnosis",
            session_id="routing_root_failure_session",
            execution_package_sha256="b" * 64,
            environment={"HF_TOKEN": "token", "CEG_WM_ROOT_KEY": "root"},
        )


@pytest.mark.parametrize(
    "failure_stage",
    ("candidate", "authority", "runner", "package", "sha", "store"),
)
def test_routing_post_initialization_failures_close_once_and_preserve_original(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    failure = RuntimeError(f"{failure_stage} failure")
    close_calls: list[str] = []

    class FakeRuntime:
        def close(self) -> None:
            close_calls.append("close")

    runtime = FakeRuntime()
    backend = object()
    session = SimpleNamespace(runtime_config_digest="1" * 64)
    semantic = object()
    monkeypatch.setattr(
        entrypoint_module,
        "_initialize_routing_resources",
        lambda **_kwargs: (backend, runtime, session, semantic),
    )
    original_digest = entrypoint_module.canonical_digest

    def controlled_digest(value):
        if failure_stage == "candidate" and type(value) is dict and (
            "routing_candidate_identity" in value
        ):
            raise failure
        if failure_stage == "authority" and type(value) is dict and (
            "root_key_public_digest" in value and "run_id" in value
        ):
            raise failure
        return original_digest(value)

    monkeypatch.setattr(entrypoint_module, "canonical_digest", controlled_digest)

    class FakeRunner:
        def create_persistence_unit_bindings(self):
            return ()

    if failure_stage == "runner":
        monkeypatch.setattr(
            entrypoint_module,
            "ContentRoutingDirectionalDiagnosisRunner",
            lambda **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            entrypoint_module,
            "ContentRoutingDirectionalDiagnosisRunner",
            lambda **_kwargs: FakeRunner(),
        )
    package = tmp_path / "package.zip"
    package.write_bytes(b"package")
    if failure_stage == "package":
        monkeypatch.setattr(
            entrypoint_module,
            "_build_or_verify_package",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            entrypoint_module,
            "_build_or_verify_package",
            lambda *_args, **_kwargs: package,
        )
    if failure_stage == "sha":
        monkeypatch.setattr(
            entrypoint_module,
            "_sha256_file",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(entrypoint_module, "_sha256_file", lambda *_args: "b" * 64)
    if failure_stage == "store":
        monkeypatch.setattr(
            entrypoint_module,
            "DevelopmentPersistentStore",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    with pytest.raises(RuntimeError) as caught:
        entrypoint_module.execute_content_routing_directional_diagnosis_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="ceg_wm_content_routing_backend_binding_correction_diagnosis",
            session_id="routing_post_initialization_failure_session",
            execution_package_sha256="b" * 64,
            environment={"HF_TOKEN": "token", "CEG_WM_ROOT_KEY": "root"},
        )
    assert caught.value is failure
    assert close_calls == ["close"]


def test_routing_runner_calls_real_public_method_surfaces() -> None:
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(
            ROOT / "configs/experiments/internal_execution_components.json"
        )
    )
    observation = SpatialRoutingObservation(
        values=(0.2, 0.4, 0.6, 0.8),
        spatial_shape=(2, 2),
        source_identity_digest="7" * 64,
    )
    routing_observations = RoutingObservations(
        semantic=observation,
        texture=observation,
        response=observation,
        sensitivity=observation,
    )
    shape = (1, 16, 2, 2)
    routed = adapter.route_content(
        shape,
        mode="routing_stqr",
        observations=routing_observations,
    ).result
    uniform = adapter.route_content(shape, mode="routing_uniform_control").result
    key = "routing-public-method-test-key"
    for route in (routed, uniform):
        low = adapter.build_lf_carrier(key, shape, routing_result=route).result
        high = adapter.build_hf_carrier(key, shape, routing_result=route).result
        embedding = adapter.embed_content(
            (1.0,) * 64,
            high,
            lf_carrier_result=low,
            mixing_coefficient=0.50,
            routing_result=route,
        ).result
        assert embedding.active_lf_direction is not None
        assert embedding.active_hf_direction is not None
    public_image = torch.arange(48, dtype=torch.uint8).reshape(1, 3, 4, 4)
    detected = FormalHfContentDetectionOperation(adapter)(public_image, key)
    assert detected.formal_mode == "hf_only"
    assert detected.content_score == detected.hf_score
    assert detected.lf_score is None
    assert detected.lf_result is None
    assert detected.combined_score is None
    assert detected.diagnostic_combination is None
    assert detected.diagnostic_identity is None


def test_routing_runner_source_uses_runtime_semantic_and_public_detector_calls() -> None:
    source = inspect.getsource(ContentRoutingDirectionalDiagnosisRunner)
    for required_call in (
        "measure_generation_routing_reference_inputs",
        "normalize_generation_routing_measurement",
        "self.semantic.observe",
        "self.adapter.route_content",
        "self.adapter.build_lf_carrier",
        "self.adapter.build_hf_carrier",
        "self.adapter.embed_content",
        "self.runtime.execute_content_write_and_vae",
        "FormalHfContentDetectionOperation",
    ):
        assert required_call in source
    for forbidden_call in (
        "detect_lf(",
        "private_state",
        "precomputed_score",
        "content_" + "combination_calibrated",
        "select_mixing",
    ):
        assert forbidden_call not in source
    assert source.count(
        "mixing_coefficient=self.protocol.mixing_coefficient"
    ) == 2
    assert "alpha" not in source


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("texture_gradient_values", (1.0, -0.25)),
        ("response_ratio_values", (2.0, float("inf"))),
        ("sensitivity_ratio_values", (3.0, float("nan"))),
        ("texture_spatial_shape", (1, 3)),
        ("response_ratio_values", (2, 2.5)),
    ),
)
def test_routing_reference_spatial_values_fail_closed(
    changed_field: str,
    changed_value: object,
) -> None:
    values = {
        "cluster_ordinal": 0,
        "fold_index": 0,
        "texture_gradient_values": (1.0, 1.5),
        "texture_spatial_shape": (1, 2),
        "response_ratio_values": (2.0, 2.5),
        "response_spatial_shape": (1, 2),
        "sensitivity_ratio_values": (3.0, 3.5),
        "sensitivity_spatial_shape": (1, 2),
    }
    with pytest.raises(ContentRoutingDirectionalMetricError, match="measurement drifted"):
        create_content_routing_reference_measurement(
            **{**values, changed_field: changed_value}
        )


def test_routing_reference_roster_incomplete_or_duplicated_fails_closed() -> None:
    measurements = _reference_measurements()
    with pytest.raises(
        ContentRoutingDirectionalMetricError,
        match="roster is incomplete",
    ):
        fit_content_routing_fold_reference(measurements[:-1], probe_fold_index=0)
    with pytest.raises(
        ContentRoutingDirectionalMetricError,
        match="roster is duplicated",
    ):
        fit_content_routing_fold_reference(
            (*measurements[:-1], measurements[0]),
            probe_fold_index=0,
        )


def test_exact_nearest_rank_p95_excludes_zero_without_interpolation() -> None:
    assert exact_nearest_rank_positive_p95((0.0, 1.0, 2.0, 0.0, 3.0)) == 3.0
    with pytest.raises(
        ContentRoutingReferencePositiveSupportError,
        match="positive support is absent",
    ):
        exact_nearest_rank_positive_p95((0.0, 0.0))


@pytest.mark.parametrize(
    "invalid",
    (-0.25, float("nan"), float("inf"), 1),
)
def test_exact_nearest_rank_p95_rejects_non_raw_support_values(
    invalid: object,
) -> None:
    with pytest.raises(
        ContentRoutingDirectionalMetricError,
        match="exact finite nonnegative floats",
    ):
        exact_nearest_rank_positive_p95((0.0, invalid, 1.0))


def test_routing_reference_measurement_accepts_zero_and_binds_identity() -> None:
    values = {
        "cluster_ordinal": 0,
        "fold_index": 0,
        "texture_gradient_values": (0.0, 1.5),
        "texture_spatial_shape": (1, 2),
        "response_ratio_values": (0.0, 2.5),
        "response_spatial_shape": (1, 2),
        "sensitivity_ratio_values": (0.0, 3.5),
        "sensitivity_spatial_shape": (1, 2),
    }
    accepted = create_content_routing_reference_measurement(**values)
    changed = create_content_routing_reference_measurement(
        **{**values, "texture_gradient_values": (0.0, 1.75)}
    )
    assert accepted.texture_gradient_values[0] == 0.0
    assert accepted.observation_identity != changed.observation_identity


def test_routing_reference_raw_support_is_validated_before_success_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _persistence_runner()
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.protocol.run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=runner.method_code_revision,
            protocol_digest=runner.protocol_digest,
            execution_intent_authority_digest=(
                runner.execution_intent_authority_digest
            ),
            input_manifest_digest="d" * 64,
            candidate_config_digest=runner.candidate_config_digest,
            unit_roster_digest=runner.protocol.unit_roster_digest,
        ),
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    epoch = int(time.time())
    lease = store.acquire_lease(
        session_id="routing_reference_precommit_validation_session",
        now_epoch_seconds=epoch,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=epoch)
    base = torch.ones((1, 16, 2, 2), dtype=torch.float16)
    for unit_index in range(2):
        intent = store.create_session_intent(
            cursor, lease, now_epoch_seconds=epoch + unit_index * 2 + 1
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=runner.execute_operational_unit(
                unit_index=unit_index,
                base_latent=base,
                intent=intent,
            ),
            now_epoch_seconds=epoch + unit_index * 2 + 2,
        )
    intent = store.create_session_intent(
        cursor, lease, now_epoch_seconds=epoch + 10
    )
    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "measure_generation_routing_reference_inputs",
        lambda _self, _latent, *, sample_index: SimpleNamespace(
            candidate_id="routing_stqr",
            runtime_config_digest="1" * 64,
            model_id="registered-routing-model",
            model_revision="registered-routing-model-revision",
            callback_indices=tuple(range(20)),
            public_probe_domain_digest="2" * 64,
            public_probe_values_float32_be_sha256="3" * 64,
            nominal_relative_probe_step=0.001,
            actual_probe_step=0.001,
            texture_gradient_values=(-0.25, 1.0),
            texture_spatial_shape=(1, 2),
            response_ratio_values=(0.0, 1.0),
            response_spatial_shape=(1, 2),
            sensitivity_ratio_values=(0.0, 1.0),
            sensitivity_spatial_shape=(1, 2),
        ),
    )
    with pytest.raises(
        ContentRoutingDirectionalMetricError,
        match="measurement drifted",
    ):
        runner.execute_reference_fit_unit(
            unit_index=2,
            base_latent=base,
            intent=intent,
        )
    assert len(cursor.committed_units) == 2
    assert not tuple((store.run_root / "markers").glob("*0002*"))
    assert not tuple((store.run_root / "bundles").glob("*0002*"))


def test_routing_reference_fold_pools_only_positive_spatial_support() -> None:
    measurements = tuple(
        create_content_routing_reference_measurement(
            cluster_ordinal=ordinal,
            fold_index=ordinal % 4,
            texture_gradient_values=(0.0, float(ordinal + 1)),
            texture_spatial_shape=(1, 2),
            response_ratio_values=(0.0, float(ordinal + 101)),
            response_spatial_shape=(1, 2),
            sensitivity_ratio_values=(0.0, float(ordinal + 201)),
            sensitivity_spatial_shape=(1, 2),
        )
        for ordinal in range(32)
    )
    reference = fit_content_routing_fold_reference(
        measurements,
        probe_fold_index=0,
    )
    selected = tuple(
        item for item in measurements if item.fold_index != 0
    )
    positive_texture = tuple(
        value
        for item in selected
        for value in item.texture_gradient_values
        if value > 0.0
    )
    assert reference.texture_gradient_reference == exact_nearest_rank_positive_p95(
        positive_texture
    )
    assert reference.texture_gradient_reference == 31.0


def test_routing_reference_real_shape_fourfold_oracle_drops_zero_support() -> None:
    shape = (64, 64)
    count = shape[0] * shape[1]
    measurements = tuple(
        create_content_routing_reference_measurement(
            cluster_ordinal=ordinal,
            fold_index=ordinal % 4,
            texture_gradient_values=(0.0,) * (count - 1) + (float(ordinal + 1),),
            texture_spatial_shape=shape,
            response_ratio_values=(0.0,) * (count - 1) + (float(ordinal + 101),),
            response_spatial_shape=shape,
            sensitivity_ratio_values=(0.0,) * (count - 1)
            + (float(ordinal + 201),),
            sensitivity_spatial_shape=shape,
        )
        for ordinal in range(32)
    )
    for fold_index in range(4):
        selected = tuple(
            ordinal + 1
            for ordinal in range(32)
            if ordinal % 4 != fold_index
        )
        oracle = float(sorted(selected)[ceil(0.95 * len(selected)) - 1])
        reference = fit_content_routing_fold_reference(
            measurements,
            probe_fold_index=fold_index,
        )
        zero_inclusive = sorted(
            value
            for item in measurements
            if item.fold_index != fold_index
            for value in item.texture_gradient_values
        )
        zero_inclusive_p95 = zero_inclusive[
            ceil(0.95 * len(zero_inclusive)) - 1
        ]
        assert zero_inclusive_p95 == 0.0
        assert reference.texture_gradient_reference == oracle
        assert reference.texture_gradient_reference != zero_inclusive_p95
        assert reference.latent_response_reference == oracle + 100.0
        assert reference.local_sensitivity_reference == oracle + 200.0


def test_routing_observation_preserves_paired_scores_controls_and_identities() -> None:
    observation = _observation(0)
    observation.validate()
    assert observation.incremental_indicator == 1.0
    assert observation.routing_coverage == observation.routed_mean_mask_lf
    assert observation.quality_relative_l2 == max(
        observation.routed_clean_to_watermarked_rgb_relative_l2,
        observation.uniform_clean_to_watermarked_rgb_relative_l2,
    )
    assert len(observation.blind_score_observations) == 12
    for arm_id in ("routed", "uniform"):
        rows = tuple(
            row for row in observation.blind_score_observations if row.arm_id == arm_id
        )
        assert len(rows) == 6
        assert sum(row.control_role == "registered" for row in rows) == 1
        assert sum(
            row.control_role == "paired_clean_primary_null" for row in rows
        ) == 1
        assert tuple(
            sorted(
                row.wrong_key_index
                for row in rows
                if row.control_role == "wrong_key_control"
            )
        ) == (0, 1, 2, 3)
        assert all(row.formal_mode == "hf_only" for row in rows)
        assert all(row.content_score == row.hf_score for row in rows)
    assert observation.failure_class is None


def test_routing_blind_score_row_fields_are_the_frozen_hf_only_surface() -> None:
    assert tuple(field.name for field in fields(ContentRoutingBlindScoreObservation)) == (
        "arm_id",
        "control_role",
        "wrong_key_index",
        "content_score",
        "hf_score",
        "formal_mode",
        "content_detector_identity",
        "content_config_digest",
        "hf_detector_identity",
        "hf_detector_config_digest",
        "content_input_image_digest",
        "hf_observation_digest",
        "hf_template_digest",
        "root_key_public_digest",
        "key_role",
    )


@pytest.mark.parametrize(
    ("replacement", "match"),
    (
        ({"formal_mode": "combined"}, "blind score semantics"),
        ({"content_score": 0.7}, "blind score semantics"),
        ({"wrong_key_index": 0}, "blind score semantics"),
    ),
)
def test_routing_blind_registered_score_rejects_non_hf_or_wrong_key_semantics(
    replacement: dict[str, object],
    match: str,
) -> None:
    values = asdict(_blind_score_rows(0, routed_wins=True)[0])
    values.update(replacement)
    with pytest.raises(ContentRoutingDirectionalMetricError, match=match):
        create_content_routing_blind_score_observation(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("content_detector_identity", "main.content_detector"),
        ("hf_detector_identity", ""),
        ("content_config_digest", "A" * 64),
        ("hf_observation_digest", "g" * 64),
    ),
)
def test_routing_blind_score_rejects_non_sha256_identities_and_digests(
    field_name: str,
    invalid_value: str,
) -> None:
    values = asdict(_blind_score_rows(0, routed_wins=True)[0])
    values[field_name] = invalid_value
    with pytest.raises(ContentRoutingDirectionalMetricError, match="blind score semantics"):
        create_content_routing_blind_score_observation(**values)


def _directional_values_with_mutable_blind_rows() -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    values = asdict(_observation(0))
    for key in (
        "observation_identity",
        "incremental_indicator",
        "routing_coverage",
        "quality_relative_l2",
    ):
        values.pop(key)
    rows = list(values["blind_score_observations"])
    values["blind_score_observations"] = rows
    return values, rows


def test_routing_registered_template_must_match_across_arms() -> None:
    values, rows = _directional_values_with_mutable_blind_rows()
    rows[6]["hf_template_digest"] = "c" * 64
    rows[7]["hf_template_digest"] = "c" * 64
    with pytest.raises(
        ContentRoutingDirectionalMetricError,
        match="cross-arm detector template",
    ):
        create_content_routing_directional_observation(**values)


def test_routing_paired_clean_template_must_match_registered_template() -> None:
    values, rows = _directional_values_with_mutable_blind_rows()
    rows[1]["hf_template_digest"] = "c" * 64
    with pytest.raises(
        ContentRoutingDirectionalMetricError,
        match="control pairing",
    ):
        create_content_routing_directional_observation(**values)


def test_routing_each_wrong_key_template_must_match_across_arms() -> None:
    values, rows = _directional_values_with_mutable_blind_rows()
    rows[10]["hf_template_digest"] = "f" * 64
    with pytest.raises(
        ContentRoutingDirectionalMetricError,
        match="cross-arm detector template",
    ):
        create_content_routing_directional_observation(**values)


def test_routing_blind_controls_require_same_candidate_and_registered_clean_key() -> None:
    values, rows = _directional_values_with_mutable_blind_rows()
    rows[2]["content_input_image_digest"] = "f" * 64
    with pytest.raises(ContentRoutingDirectionalMetricError, match="control pairing"):
        create_content_routing_directional_observation(**values)


def test_routing_protocol_freezes_hf_only_score_and_embedder_responsibility() -> None:
    protocol, _, _ = _protocol_bundle()
    assert protocol.content_embedding_responsibility_id == "content_embedder"
    assert protocol.content_embedding_branch_identity == "lf_hf_routed_combination"
    assert protocol.public_content_operation == "FormalHfContentDetectionOperation"
    assert protocol.public_score_identity == "hf_only_public_content_operation"
    assert protocol.public_score_semantics == "content_score_equals_hf_result_hf_score"
    assert protocol.public_score_required_null_result_fields == (
        "lf_score",
        "lf_result",
        "combined_score",
        "diagnostic_combination",
        "diagnostic_identity",
    )
    assert protocol.lf_branch_responsibility_ids == (
        "lf_carrier",
        "content_embedder",
    )
    assert protocol.lf_detector_usage == "prohibited"
    for relative in (
        "configs/experiments/content_routing_directional_diagnosis.json",
        "experiments/protocol/content_routing_directional_diagnosis.py",
        "experiments/metrics/content_routing_directional_diagnosis.py",
        "tests/unit/test_content_routing_directional_diagnosis.py",
    ):
        forbidden_identity = "content_" + "combination_calibrated"
        assert forbidden_identity not in (ROOT / relative).read_text(
            encoding="utf-8"
        )


def test_routing_observation_rejects_nonuniform_disabled_control() -> None:
    values = asdict(_observation(0))
    values.pop("observation_identity")
    values.pop("incremental_indicator")
    values.pop("routing_coverage")
    values.pop("quality_relative_l2")
    values["uniform_mean_mask_lf"] = 0.9
    with pytest.raises(ContentRoutingDirectionalMetricError, match="paired metric"):
        create_content_routing_directional_observation(**values)


def test_routing_directional_passing_gate_only_allows_later_validation_request() -> None:
    observations = tuple(_observation(i, routed_wins=i < 5) for i in range(8))
    aggregate = aggregate_content_routing_directional_diagnosis(observations)
    assert aggregate.incremental_success_count == 5
    assert aggregate.incremental_indicator_mean == 0.625
    assert aggregate.routing_coverage_mean == pytest.approx(0.4)
    assert aggregate.outcome == "routing_directional_signal_observed"
    assert aggregate.allow_request_for_routing_directional_validation is True
    assert aggregate.formal_scientific_claims_supported is False
    assert "mechanism_signal_observed" not in json.dumps(asdict(aggregate))
    assert "candidate_worth_further_selection" not in json.dumps(asdict(aggregate))


def test_routing_directional_interpretable_negative_requires_all_eight_successes() -> None:
    observations = tuple(_observation(i, routed_wins=i < 4) for i in range(8))
    aggregate = aggregate_content_routing_directional_diagnosis(observations)
    assert aggregate.outcome == "routing_directional_signal_not_observed"
    assert aggregate.allow_request_for_routing_directional_validation is False


def test_routing_directional_failure_precedence_and_fixed_denominator() -> None:
    successes = tuple(_observation(i) for i in range(6))
    mixed = aggregate_content_routing_directional_diagnosis(
        successes,
        implementation_failure_count=1,
        resource_failure_count=1,
    )
    assert mixed.expected_probe_count == 8
    assert mixed.successful_probe_count == 6
    assert mixed.failed_probe_count == 2
    assert mixed.outcome == "implementation_blocked"
    resource = aggregate_content_routing_directional_diagnosis(
        tuple(_observation(i) for i in range(7)),
        resource_failure_count=1,
    )
    assert resource.outcome == "resource_blocked"
    assert resource.incremental_indicator_mean == 7 / 8


def test_routing_directional_budget_violation_cannot_pass() -> None:
    observations = tuple(
        _observation(i, l2=0.013 if i == 0 else 0.01)
        for i in range(8)
    )
    aggregate = aggregate_content_routing_directional_diagnosis(observations)
    assert aggregate.budget_violation_count == 1
    assert aggregate.routing_directional_diagnosis_passed is False
    assert aggregate.outcome == "routing_directional_signal_not_observed"


def test_routing_protocol_and_metric_remain_independent_of_execution_layers() -> None:
    for relative in (
        "experiments/protocol/content_routing_directional_diagnosis.py",
        "experiments/metrics/content_routing_directional_diagnosis.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "experiments.runners" not in source
        assert "experiments.methods" not in source
        assert "from runtime" not in source
        assert "import runtime" not in source
        assert "governance" not in source
    assert "no_threshold_no_fpr_no_combination_conclusion" in CLAIM_BOUNDARY
