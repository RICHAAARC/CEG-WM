"""CPU calculations for development-only cluster metrics."""

from __future__ import annotations

import pytest

from experiments.metrics.development_exploration import (
    DEVELOPMENT_THRESHOLD_ROLE,
    DevelopmentMetricError,
    aggregate_development_cluster_metrics,
    bind_development_metric_observation,
    cross_fit_development_detection_metrics,
    metric_hf_detector,
)
from main import HfDetectionObservation, derive_wrong_key_material, hf_carrier, hf_detector, identify_root_key
from experiments.protocol.development_exploration import build_development_cross_fit_plan
from tests.unit.test_development_module_exploration import (
    _development_manifest,
    _execution_intent,
)


def _hf_observations(cluster_ids: tuple[str, ...]):
    root_key = "development-metric-real-hf-key"
    shape = (1, 1, 4, 4)
    carrier = hf_carrier(root_key, shape)
    registered_observation = HfDetectionObservation.from_public_image_encoding(
        carrier.direction, shape
    )
    null_observation = HfDetectionObservation.from_public_image_encoding(
        tuple(((index % 5) - 2) / 4.0 for index in range(16)), shape
    )
    wrong_material = derive_wrong_key_material(
        identify_root_key(root_key).root_key_public_digest, 0
    )
    registered = hf_detector(registered_observation, root_key)
    wrong = hf_detector(registered_observation, wrong_material)
    primary_null = hf_detector(null_observation, root_key)
    return tuple(
        bind_development_metric_observation(
            metric_hf_detector(
            cluster_id,
            registered_score=registered.hf_score,
            wrong_score=wrong.hf_score,
            primary_null_score=primary_null.hf_score,
            detector_config_digest=registered.detector_config_digest,
            registered_observation_digest=registered.observation_digest,
            wrong_observation_digest=wrong.observation_digest,
            primary_null_observation_digest=primary_null.observation_digest,
            ),
            registered_metric_ids=("hf_tpr_at_frozen_fpr", "hf_wrong_key_rate"),
            candidate_config_digest="8" * 64,
            paired_ablation_identity="high_frequency_detector_disabled_ablation",
            content_branch_id="hf_only",
            geometry_case_id="not_applicable",
        )
        for cluster_id in cluster_ids
    )


def _folds(cluster_ids: tuple[str, ...]):
    return tuple(
        (
            tuple(cluster for index, cluster in enumerate(cluster_ids) if index % 4 != fold),
            tuple(cluster for index, cluster in enumerate(cluster_ids) if index % 4 == fold),
        )
        for fold in range(4)
    )


@pytest.mark.quick
def test_detection_cross_fit_uses_disjoint_primary_null_clusters() -> None:
    intent = _execution_intent(_development_manifest(64), run_id="metric_cross_fit")
    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=intent,
        expected_execution_intent_authority_digest=intent.authority_digest,
        expected_source_cluster_count=64,
    )
    observations = _hf_observations(plan.source_cluster_ids)
    result = cross_fit_development_detection_metrics(
        "hf_detector", observations, plan=plan
    )

    assert result.threshold_role == DEVELOPMENT_THRESHOLD_ROLE
    assert result.fold_count == 4
    assert result.source_cluster_count == 64
    assert 0.0 <= result.primary_null_false_accept_rate <= 1.0
    assert 0.0 <= result.registered_accept_rate <= 1.0
    assert 0.0 <= result.wrong_key_accept_rate <= 1.0
    assert len(result.fold_assignment_digest) == 64
    assert result.candidate_config_digest == "8" * 64
    assert result.registered_metric_ids == (
        "hf_tpr_at_frozen_fpr",
        "hf_wrong_key_rate",
    )
    assert len(result.evidence_observation_digest) == 64


@pytest.mark.quick
def test_detection_cross_fit_rejects_fit_probe_cluster_leakage() -> None:
    intent = _execution_intent(_development_manifest(64), run_id="metric_cross_fit_bad")
    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=intent,
        expected_execution_intent_authority_digest=intent.authority_digest,
        expected_source_cluster_count=64,
    )
    observations = _hf_observations(plan.source_cluster_ids)
    wrong_plan = build_development_cross_fit_plan(
        responsibility_id="lf_detector",
        execution_intent_authority=intent,
        expected_execution_intent_authority_digest=intent.authority_digest,
        expected_source_cluster_count=64,
    )

    with pytest.raises(DevelopmentMetricError, match="responsibility"):
        cross_fit_development_detection_metrics(
            "hf_detector", observations, plan=wrong_plan
        )


@pytest.mark.quick
def test_detection_cross_fit_rejects_candidate_binding_drift() -> None:
    intent = _execution_intent(_development_manifest(64), run_id="metric_candidate_drift")
    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=intent,
        expected_execution_intent_authority_digest=intent.authority_digest,
        expected_source_cluster_count=64,
    )
    observations = _hf_observations(plan.source_cluster_ids)
    forged = bind_development_metric_observation(
        observations[-1],
        registered_metric_ids=observations[-1].registered_metric_ids,
        candidate_config_digest="9" * 64,
        paired_ablation_identity=observations[-1].paired_ablation_identity,
        content_branch_id=observations[-1].content_branch_id,
        geometry_case_id=observations[-1].geometry_case_id,
    )

    with pytest.raises(DevelopmentMetricError, match="candidate"):
        cross_fit_development_detection_metrics(
            "hf_detector", (*observations[:-1], forged), plan=plan
        )


@pytest.mark.quick
def test_cluster_aggregate_rejects_duplicate_source_clusters() -> None:
    observations = _hf_observations(tuple(f"{index + 1:064x}" for index in range(16)))
    with pytest.raises(DevelopmentMetricError, match="duplicated"):
        aggregate_development_cluster_metrics(
            "hf_detector",
            (*observations[:-1], observations[0]),
            minimum_source_clusters=16,
        )
