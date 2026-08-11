from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path

import pytest

from experiments.metrics.content_routing_directional_diagnosis import (
    ContentRoutingDirectionalMetricError,
    aggregate_content_routing_directional_diagnosis,
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


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/content_routing_directional_diagnosis.json"


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
            texture_gradient_value=float(ordinal + 1),
            latent_response_value=float(ordinal + 101),
            local_sensitivity_value=float(ordinal + 201),
            semantic_observation_digest=f"{ordinal + 1:064x}",
        )
        for ordinal in range(32)
    )


def _observation(ordinal: int, *, routed_wins: bool = True, l2: float = 0.01):
    uniform_score = 0.20 + ordinal * 0.001
    routed_score = uniform_score + (0.01 if routed_wins else -0.01)
    return create_content_routing_directional_observation(
        cluster_ordinal=ordinal,
        fold_index=ordinal % 4,
        routed_registered_content_score=routed_score,
        uniform_registered_content_score=uniform_score,
        routed_registered_hf_score=routed_score,
        uniform_registered_hf_score=uniform_score,
        routed_registered_lf_score=0.03 + ordinal * 0.001,
        uniform_registered_lf_score=0.02 + ordinal * 0.001,
        routed_primary_null_content_score=-0.08,
        uniform_primary_null_content_score=-0.08,
        routed_wrong_key_content_scores=(-0.03, -0.02, -0.01, 0.0),
        uniform_wrong_key_content_scores=(-0.03, -0.02, -0.01, 0.0),
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
        detector_identity="main.hf_detector",
        detector_config_digest="1" * 64,
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
    assert protocol.operational_role == "environment_runtime_throughput_preflight"
    assert protocol.operational_case_ids == (
        "environment_identity_preflight",
        "runtime_identity_preflight",
        "throughput_preflight",
    )
    assert protocol.operational_result_responsibility_id == "content_embedder"


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


def test_routing_reference_fits_only_texture_response_and_sensitivity() -> None:
    reference = fit_content_routing_fold_reference(
        _reference_measurements(),
        probe_fold_index=0,
    )
    assert len(reference.fit_cluster_ordinals) == 24
    assert all(ordinal % 4 != 0 for ordinal in reference.fit_cluster_ordinals)
    assert reference.semantic_observation_is_not_fitted is True
    assert not any(
        "semantic" in field.name and field.name.endswith("_reference")
        for field in fields(reference)
    )
    assert reference.texture_gradient_reference == 31.0
    assert reference.latent_response_reference == 131.0
    assert reference.local_sensitivity_reference == 231.0


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
    assert len(observation.routed_wrong_key_content_scores) == 4
    assert len(observation.uniform_wrong_key_content_scores) == 4
    assert observation.routed_registered_hf_score != observation.routed_registered_lf_score
    assert observation.failure_class is None


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
