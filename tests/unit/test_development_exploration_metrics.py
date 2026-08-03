"""CPU calculations for development-only cluster metrics."""

from __future__ import annotations

import pytest

from experiments.metrics.development_exploration import (
    DEVELOPMENT_THRESHOLD_ROLE,
    DevelopmentMetricError,
    aggregate_development_cluster_metrics,
    cross_fit_development_detection_metrics,
    metric_hf_detector,
)
from main import HfDetectionObservation, derive_wrong_key_material, hf_carrier, hf_detector, identify_root_key


def _hf_observations(count: int = 16):
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
        metric_hf_detector(
            f"{index + 1:064x}",
            registered_score=registered.hf_score,
            wrong_score=wrong.hf_score,
            primary_null_score=primary_null.hf_score,
            detector_config_digest=registered.detector_config_digest,
            registered_observation_digest=registered.observation_digest,
            wrong_observation_digest=wrong.observation_digest,
            primary_null_observation_digest=primary_null.observation_digest,
        )
        for index in range(count)
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
    observations = _hf_observations()
    clusters = tuple(item.source_cluster_id for item in observations)
    result = cross_fit_development_detection_metrics(
        "hf_detector", observations, folds=_folds(clusters)
    )

    assert result.threshold_role == DEVELOPMENT_THRESHOLD_ROLE
    assert result.fold_count == 4
    assert result.source_cluster_count == 16
    assert 0.0 <= result.primary_null_false_accept_rate <= 1.0
    assert 0.0 <= result.registered_accept_rate <= 1.0
    assert 0.0 <= result.wrong_key_accept_rate <= 1.0
    assert len(result.fold_assignment_digest) == 64


@pytest.mark.quick
def test_detection_cross_fit_rejects_fit_probe_cluster_leakage() -> None:
    observations = _hf_observations()
    clusters = tuple(item.source_cluster_id for item in observations)
    bad_folds = list(_folds(clusters))
    fit, probe = bad_folds[0]
    bad_folds[0] = (fit, (*probe, fit[0]))

    with pytest.raises(DevelopmentMetricError, match="overlap"):
        cross_fit_development_detection_metrics(
            "hf_detector", observations, folds=bad_folds
        )


@pytest.mark.quick
def test_cluster_aggregate_rejects_duplicate_source_clusters() -> None:
    observations = _hf_observations()
    with pytest.raises(DevelopmentMetricError, match="duplicated"):
        aggregate_development_cluster_metrics(
            "hf_detector",
            (*observations[:-1], observations[0]),
            minimum_source_clusters=16,
        )
