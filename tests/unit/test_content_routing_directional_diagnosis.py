from __future__ import annotations

from dataclasses import asdict, fields
import inspect
import json
from pathlib import Path

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)

from experiments.metrics.content_routing_directional_diagnosis import (
    ContentRoutingBlindScoreObservation,
    ContentRoutingDirectionalMetricError,
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
    load_content_routing_directional_protocol,
    reference_entries_for_probe,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    ROUTING_REFERENCE_RECORD_COLLECTION_ROLE,
    ROUTING_REFERENCE_RECORD_KIND,
    ROUTING_REFERENCE_RECORD_SCHEMA,
    DevelopmentRoutingReferenceRecord,
    canonical_development_value_digest,
)
from experiments.runners.content_routing_directional_diagnosis import (
    ContentRoutingDirectionalDiagnosisRunner,
)
from experiments.runners.formal_operations import (
    FormalHfContentDetectionOperation,
)
from main import RoutingObservations, SpatialRoutingObservation


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
        "run_id": "ceg_wm_content_routing_directional_diagnosis",
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
    assert protocol.run_id == "ceg_wm_content_routing_directional_diagnosis"
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
    ):
        assert forbidden_call not in source


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("texture_gradient_values", (1.0, 0.0)),
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


def test_exact_nearest_rank_p95_rejects_nonpositive_reference_values() -> None:
    assert exact_nearest_rank_positive_p95((1.0, 2.0, 3.0, 4.0)) == 4.0
    with pytest.raises(ContentRoutingDirectionalMetricError, match="strictly positive"):
        exact_nearest_rank_positive_p95((1.0, 0.0, 2.0))


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
