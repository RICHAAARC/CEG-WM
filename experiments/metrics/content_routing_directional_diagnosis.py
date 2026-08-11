"""Threshold-free metrics for the content-routing directional diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import ceil, isfinite
from statistics import mean
from typing import Sequence

from experiments.protocol.content_routing_directional_diagnosis import (
    CONTENT_RELATIVE_L2_DENOMINATOR,
    CONTENT_RELATIVE_L2_NUMERATOR,
    CROSS_FIT_FOLD_COUNT,
    DIRECTIONAL_PROBE_CLUSTER_COUNT,
    INCREMENTAL_INDICATOR_MEAN_REQUIREMENT,
    NEGATIVE_OUTCOME,
    PASSING_OUTCOME,
    PUBLIC_CONTENT_OPERATION,
    REFERENCE_FIT_CLUSTER_COUNT,
    REFERENCE_FIT_COUNT_PER_PROBE,
    REFERENCE_QUANTILE_RULE,
    ROUTING_COVERAGE_REQUIREMENT,
    WRONG_KEY_ROSTER_SIZE,
)


class ContentRoutingDirectionalMetricError(ValueError):
    """A routing directional metric input is invalid."""


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def exact_nearest_rank_positive_p95(values: Sequence[float]) -> float:
    items = tuple(float(value) for value in values)
    positive = tuple(
        sorted(
            value
            for value in items
            if isfinite(value) and value > 0.0
        )
    )
    if len(positive) != len(items) or not positive:
        raise ContentRoutingDirectionalMetricError(
            "routing reference values must all be finite and strictly positive"
        )
    return positive[ceil(0.95 * len(positive)) - 1]


@dataclass(frozen=True, slots=True)
class ContentRoutingReferenceMeasurement:
    cluster_ordinal: int
    fold_index: int
    texture_gradient_value: float
    latent_response_value: float
    local_sensitivity_value: float
    semantic_observation_digest: str
    observation_identity: str

    def validate(self) -> None:
        numeric = (
            self.texture_gradient_value,
            self.latent_response_value,
            self.local_sensitivity_value,
        )
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < REFERENCE_FIT_CLUSTER_COUNT
            or type(self.fold_index) is not int
            or self.fold_index != self.cluster_ordinal % CROSS_FIT_FOLD_COUNT
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) <= 0.0
                for value in numeric
            )
            or type(self.semantic_observation_digest) is not str
            or len(self.semantic_observation_digest) != 64
        ):
            raise ContentRoutingDirectionalMetricError(
                "routing reference measurement drifted"
            )
        payload = asdict(self)
        identity = payload.pop("observation_identity")
        if identity != _digest(payload):
            raise ContentRoutingDirectionalMetricError(
                "routing reference measurement identity drifted"
            )


def create_content_routing_reference_measurement(
    **values: object,
) -> ContentRoutingReferenceMeasurement:
    payload = dict(values)
    measurement = ContentRoutingReferenceMeasurement(
        **payload,
        observation_identity=_digest(payload),
    )
    measurement.validate()
    return measurement


@dataclass(frozen=True, slots=True)
class ContentRoutingFoldReference:
    probe_fold_index: int
    fit_cluster_ordinals: tuple[int, ...]
    texture_gradient_reference: float
    latent_response_reference: float
    local_sensitivity_reference: float
    quantile_rule: str
    semantic_observation_is_not_fitted: bool
    reference_identity: str

    def validate(self) -> None:
        expected_ordinals = tuple(
            ordinal
            for ordinal in range(REFERENCE_FIT_CLUSTER_COUNT)
            if ordinal % CROSS_FIT_FOLD_COUNT != self.probe_fold_index
        )
        numeric = (
            self.texture_gradient_reference,
            self.latent_response_reference,
            self.local_sensitivity_reference,
        )
        if (
            type(self.probe_fold_index) is not int
            or not 0 <= self.probe_fold_index < CROSS_FIT_FOLD_COUNT
            or self.fit_cluster_ordinals != expected_ordinals
            or len(self.fit_cluster_ordinals) != REFERENCE_FIT_COUNT_PER_PROBE
            or any(not isfinite(value) or value <= 0.0 for value in numeric)
            or self.quantile_rule != REFERENCE_QUANTILE_RULE
            or self.semantic_observation_is_not_fitted is not True
        ):
            raise ContentRoutingDirectionalMetricError(
                "routing fold reference drifted"
            )
        payload = asdict(self)
        identity = payload.pop("reference_identity")
        if identity != _digest(payload):
            raise ContentRoutingDirectionalMetricError(
                "routing fold reference identity drifted"
            )


def fit_content_routing_fold_reference(
    measurements: Sequence[ContentRoutingReferenceMeasurement],
    *,
    probe_fold_index: int,
) -> ContentRoutingFoldReference:
    items = tuple(measurements)
    if (
        len(items) != REFERENCE_FIT_CLUSTER_COUNT
        or any(type(item) is not ContentRoutingReferenceMeasurement for item in items)
    ):
        raise ContentRoutingDirectionalMetricError(
            "routing reference roster is incomplete"
        )
    for item in items:
        item.validate()
    if (
        tuple(sorted(item.cluster_ordinal for item in items))
        != tuple(range(REFERENCE_FIT_CLUSTER_COUNT))
        or len({item.observation_identity for item in items}) != len(items)
    ):
        raise ContentRoutingDirectionalMetricError(
            "routing reference roster is duplicated"
        )
    selected = tuple(
        sorted(
            (item for item in items if item.fold_index != probe_fold_index),
            key=lambda item: item.cluster_ordinal,
        )
    )
    if len(selected) != REFERENCE_FIT_COUNT_PER_PROBE:
        raise ContentRoutingDirectionalMetricError(
            "routing reference fold exclusion drifted"
        )
    payload = {
        "probe_fold_index": probe_fold_index,
        "fit_cluster_ordinals": tuple(item.cluster_ordinal for item in selected),
        "texture_gradient_reference": exact_nearest_rank_positive_p95(
            tuple(item.texture_gradient_value for item in selected)
        ),
        "latent_response_reference": exact_nearest_rank_positive_p95(
            tuple(item.latent_response_value for item in selected)
        ),
        "local_sensitivity_reference": exact_nearest_rank_positive_p95(
            tuple(item.local_sensitivity_value for item in selected)
        ),
        "quantile_rule": REFERENCE_QUANTILE_RULE,
        "semantic_observation_is_not_fitted": True,
    }
    reference = ContentRoutingFoldReference(
        **payload,
        reference_identity=_digest(payload),
    )
    reference.validate()
    return reference


@dataclass(frozen=True, slots=True)
class ContentRoutingDirectionalObservation:
    cluster_ordinal: int
    fold_index: int
    routed_registered_content_score: float
    uniform_registered_content_score: float
    routed_registered_hf_score: float
    uniform_registered_hf_score: float
    routed_registered_lf_score: float
    uniform_registered_lf_score: float
    routed_primary_null_content_score: float
    uniform_primary_null_content_score: float
    routed_wrong_key_content_scores: tuple[float, ...]
    uniform_wrong_key_content_scores: tuple[float, ...]
    routed_mean_mask_lf: float
    routed_mean_mask_hf: float
    uniform_mean_mask_lf: float
    uniform_mean_mask_hf: float
    routed_clean_to_watermarked_rgb_relative_l2: float
    uniform_clean_to_watermarked_rgb_relative_l2: float
    routed_realized_relative_l2: float
    uniform_realized_relative_l2: float
    routed_materialization_integrity_status: str
    uniform_materialization_integrity_status: str
    routed_materialization_budget_status: str
    uniform_materialization_budget_status: str
    public_content_operation: str
    detector_identity: str
    detector_config_digest: str
    preprocessing_identity: str
    routed_route_digest: str
    uniform_route_digest: str
    cross_fit_reference_digest: str
    routed_candidate_observation_digest: str
    uniform_candidate_observation_digest: str
    paired_clean_observation_digest: str
    failure_class: None
    incremental_indicator: float
    routing_coverage: float
    quality_relative_l2: float
    observation_identity: str

    def validate(self) -> None:
        scores = (
            self.routed_registered_content_score,
            self.uniform_registered_content_score,
            self.routed_registered_hf_score,
            self.uniform_registered_hf_score,
            self.routed_registered_lf_score,
            self.uniform_registered_lf_score,
            self.routed_primary_null_content_score,
            self.uniform_primary_null_content_score,
            *self.routed_wrong_key_content_scores,
            *self.uniform_wrong_key_content_scores,
            self.routed_mean_mask_lf,
            self.routed_mean_mask_hf,
            self.uniform_mean_mask_lf,
            self.uniform_mean_mask_hf,
            self.routed_clean_to_watermarked_rgb_relative_l2,
            self.uniform_clean_to_watermarked_rgb_relative_l2,
            self.routed_realized_relative_l2,
            self.uniform_realized_relative_l2,
            self.incremental_indicator,
            self.routing_coverage,
            self.quality_relative_l2,
        )
        digest_values = (
            self.detector_config_digest,
            self.routed_route_digest,
            self.uniform_route_digest,
            self.cross_fit_reference_digest,
            self.routed_candidate_observation_digest,
            self.uniform_candidate_observation_digest,
            self.paired_clean_observation_digest,
        )
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < DIRECTIONAL_PROBE_CLUSTER_COUNT
            or self.fold_index != self.cluster_ordinal % CROSS_FIT_FOLD_COUNT
            or len(self.routed_wrong_key_content_scores) != WRONG_KEY_ROSTER_SIZE
            or len(self.uniform_wrong_key_content_scores) != WRONG_KEY_ROSTER_SIZE
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                for value in scores
            )
            or any(type(value) is not str or len(value) != 64 for value in digest_values)
            or self.public_content_operation != PUBLIC_CONTENT_OPERATION
            or type(self.detector_identity) is not str
            or not self.detector_identity
            or type(self.preprocessing_identity) is not str
            or not self.preprocessing_identity
            or self.routed_route_digest == self.uniform_route_digest
            or self.routed_candidate_observation_digest
            == self.uniform_candidate_observation_digest
            or self.paired_clean_observation_digest
            in {
                self.routed_candidate_observation_digest,
                self.uniform_candidate_observation_digest,
            }
            or self.failure_class is not None
            or self.routed_materialization_integrity_status != "passed"
            or self.uniform_materialization_integrity_status != "passed"
            or self.routed_materialization_budget_status != "accepted"
            or self.uniform_materialization_budget_status != "accepted"
        ):
            raise ContentRoutingDirectionalMetricError(
                "routing directional observation identity drifted"
            )
        if (
            not all(
                0.0 <= value <= 1.0
                for value in (
                    self.routed_mean_mask_lf,
                    self.routed_mean_mask_hf,
                    self.uniform_mean_mask_lf,
                    self.uniform_mean_mask_hf,
                )
            )
            or self.uniform_mean_mask_lf != 1.0
            or self.uniform_mean_mask_hf != 1.0
            or min(
                self.routed_clean_to_watermarked_rgb_relative_l2,
                self.uniform_clean_to_watermarked_rgb_relative_l2,
                self.routed_realized_relative_l2,
                self.uniform_realized_relative_l2,
            )
            < 0.0
            or self.incremental_indicator
            != float(
                self.routed_registered_content_score
                > self.uniform_registered_content_score
            )
            or self.routing_coverage != self.routed_mean_mask_lf
            or self.quality_relative_l2
            != max(
                self.routed_clean_to_watermarked_rgb_relative_l2,
                self.uniform_clean_to_watermarked_rgb_relative_l2,
            )
        ):
            raise ContentRoutingDirectionalMetricError(
                "routing paired metric semantics drifted"
            )
        payload = asdict(self)
        identity = payload.pop("observation_identity")
        if identity != _digest(payload):
            raise ContentRoutingDirectionalMetricError(
                "routing directional observation digest drifted"
            )


def create_content_routing_directional_observation(
    **values: object,
) -> ContentRoutingDirectionalObservation:
    payload = dict(values)
    for key in (
        "routed_wrong_key_content_scores",
        "uniform_wrong_key_content_scores",
    ):
        payload[key] = tuple(float(value) for value in payload[key])
    payload.update(
        incremental_indicator=float(
            float(payload["routed_registered_content_score"])
            > float(payload["uniform_registered_content_score"])
        ),
        routing_coverage=float(payload["routed_mean_mask_lf"]),
        quality_relative_l2=max(
            float(payload["routed_clean_to_watermarked_rgb_relative_l2"]),
            float(payload["uniform_clean_to_watermarked_rgb_relative_l2"]),
        ),
    )
    observation = ContentRoutingDirectionalObservation(
        **payload,
        observation_identity=_digest(payload),
    )
    observation.validate()
    return observation


@dataclass(frozen=True, slots=True)
class ContentRoutingDirectionalAggregate:
    expected_probe_count: int
    successful_probe_count: int
    failed_probe_count: int
    implementation_failure_count: int
    resource_failure_count: int
    incremental_success_count: int
    incremental_indicator_mean: float
    routing_coverage_mean: float
    maximum_quality_relative_l2: float | None
    maximum_realized_relative_l2: float | None
    identity_violation_count: int
    integrity_violation_count: int
    nonfinite_violation_count: int
    budget_violation_count: int
    routing_directional_diagnosis_passed: bool
    outcome: str
    allow_request_for_routing_directional_validation: bool
    formal_scientific_claims_supported: bool
    aggregate_identity: str

    def validate(self) -> None:
        expected_outcome = (
            PASSING_OUTCOME
            if self.routing_directional_diagnosis_passed
            else (
                "implementation_blocked"
                if self.implementation_failure_count
                else (
                    "resource_blocked"
                    if self.resource_failure_count
                    else NEGATIVE_OUTCOME
                )
            )
        )
        if (
            self.expected_probe_count != DIRECTIONAL_PROBE_CLUSTER_COUNT
            or self.failed_probe_count
            != self.implementation_failure_count + self.resource_failure_count
            or self.successful_probe_count + self.failed_probe_count
            != DIRECTIONAL_PROBE_CLUSTER_COUNT
            or self.outcome != expected_outcome
            or self.allow_request_for_routing_directional_validation
            is not self.routing_directional_diagnosis_passed
            or self.formal_scientific_claims_supported is not False
        ):
            raise ContentRoutingDirectionalMetricError(
                "routing aggregate boundary drifted"
            )
        payload = asdict(self)
        identity = payload.pop("aggregate_identity")
        if identity != _digest(payload):
            raise ContentRoutingDirectionalMetricError(
                "routing aggregate identity drifted"
            )


def aggregate_content_routing_directional_diagnosis(
    observations: Sequence[ContentRoutingDirectionalObservation],
    *,
    implementation_failure_count: int = 0,
    resource_failure_count: int = 0,
    identity_violation_count: int = 0,
    integrity_violation_count: int = 0,
    nonfinite_violation_count: int = 0,
    budget_violation_count: int = 0,
) -> ContentRoutingDirectionalAggregate:
    items = tuple(observations)
    counts = (
        implementation_failure_count,
        resource_failure_count,
        identity_violation_count,
        integrity_violation_count,
        nonfinite_violation_count,
        budget_violation_count,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ContentRoutingDirectionalMetricError(
            "routing aggregate count is invalid"
        )
    failed = implementation_failure_count + resource_failure_count
    if (
        len(items) + failed != DIRECTIONAL_PROBE_CLUSTER_COUNT
        or any(type(item) is not ContentRoutingDirectionalObservation for item in items)
    ):
        raise ContentRoutingDirectionalMetricError(
            "routing aggregate fixed denominator is incomplete"
        )
    for item in items:
        item.validate()
    if (
        len({item.cluster_ordinal for item in items}) != len(items)
        or len({item.observation_identity for item in items}) != len(items)
    ):
        raise ContentRoutingDirectionalMetricError(
            "routing aggregate observations are duplicated"
        )
    limit = CONTENT_RELATIVE_L2_NUMERATOR / CONTENT_RELATIVE_L2_DENOMINATOR
    budget_excess = sum(
        any(
            value > limit
            for value in (
                item.routed_clean_to_watermarked_rgb_relative_l2,
                item.uniform_clean_to_watermarked_rgb_relative_l2,
                item.routed_realized_relative_l2,
                item.uniform_realized_relative_l2,
            )
        )
        for item in items
    )
    total_budget_violations = budget_violation_count + budget_excess
    incremental_success_count = sum(
        int(item.incremental_indicator) for item in items
    )
    incremental_mean = incremental_success_count / DIRECTIONAL_PROBE_CLUSTER_COUNT
    coverage_mean = (
        sum(item.routing_coverage for item in items)
        / DIRECTIONAL_PROBE_CLUSTER_COUNT
    )
    quality = tuple(item.quality_relative_l2 for item in items)
    realized = tuple(
        max(item.routed_realized_relative_l2, item.uniform_realized_relative_l2)
        for item in items
    )
    violations = (
        identity_violation_count,
        integrity_violation_count,
        nonfinite_violation_count,
        total_budget_violations,
    )
    passed = (
        len(items) == DIRECTIONAL_PROBE_CLUSTER_COUNT
        and failed == 0
        and incremental_mean > INCREMENTAL_INDICATOR_MEAN_REQUIREMENT
        and coverage_mean > ROUTING_COVERAGE_REQUIREMENT
        and all(value <= limit for value in quality)
        and all(value <= limit for value in realized)
        and sum(violations) == 0
    )
    outcome = (
        PASSING_OUTCOME
        if passed
        else (
            "implementation_blocked"
            if implementation_failure_count
            else ("resource_blocked" if resource_failure_count else NEGATIVE_OUTCOME)
        )
    )
    payload = {
        "expected_probe_count": DIRECTIONAL_PROBE_CLUSTER_COUNT,
        "successful_probe_count": len(items),
        "failed_probe_count": failed,
        "implementation_failure_count": implementation_failure_count,
        "resource_failure_count": resource_failure_count,
        "incremental_success_count": incremental_success_count,
        "incremental_indicator_mean": incremental_mean,
        "routing_coverage_mean": coverage_mean,
        "maximum_quality_relative_l2": max(quality) if quality else None,
        "maximum_realized_relative_l2": max(realized) if realized else None,
        "identity_violation_count": identity_violation_count,
        "integrity_violation_count": integrity_violation_count,
        "nonfinite_violation_count": nonfinite_violation_count,
        "budget_violation_count": total_budget_violations,
        "routing_directional_diagnosis_passed": passed,
        "outcome": outcome,
        "allow_request_for_routing_directional_validation": passed,
        "formal_scientific_claims_supported": False,
    }
    aggregate = ContentRoutingDirectionalAggregate(
        **payload,
        aggregate_identity=_digest(payload),
    )
    aggregate.validate()
    return aggregate


__all__ = [
    "ContentRoutingDirectionalAggregate",
    "ContentRoutingDirectionalMetricError",
    "ContentRoutingDirectionalObservation",
    "ContentRoutingFoldReference",
    "ContentRoutingReferenceMeasurement",
    "aggregate_content_routing_directional_diagnosis",
    "create_content_routing_directional_observation",
    "create_content_routing_reference_measurement",
    "exact_nearest_rank_positive_p95",
    "fit_content_routing_fold_reference",
]
