"""Threshold-free paired summaries for LF whitened directional validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from statistics import mean, median
from struct import pack
from typing import Sequence

from experiments.metrics.binomial import clopper_pearson_lower
from experiments.protocol.lf_whitened_directional_validation import (
    CONFIDENCE_LEVEL,
    CONTENT_RELATIVE_L2_DENOMINATOR,
    CONTENT_RELATIVE_L2_NUMERATOR,
    MINIMUM_DIRECTIONAL_SUCCESS_COUNT,
    PASSING_CANDIDATE_RECOMMENDATION,
    PASSING_MODULE_OUTCOME,
    PRACTICAL_MARGIN_FLOOR,
    SCIENTIFIC_CLUSTER_COUNT,
    WRONG_KEY_ROSTER_SIZE,
)


class LfWhitenedDirectionalMetricError(ValueError):
    """An LF whitened directional statistic is invalid."""


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


@dataclass(frozen=True, slots=True)
class LfWhitenedDirectionalObservation:
    cluster_ordinal: int
    registered_score: float
    primary_null_score: float
    wrong_key_scores: tuple[float, ...]
    registered_minus_primary_null: float
    registered_minus_max_wrong: float
    candidate_observation_digest: str
    clean_observation_digest: str
    registered_detector_identity: str
    primary_null_detector_identity: str
    wrong_key_detector_identities: tuple[str, ...]
    detector_config_digest: str
    observation_protocol: str
    whitening_asset_digest: str
    registered_template_digest: str
    primary_null_template_digest: str
    wrong_key_template_digests: tuple[str, ...]
    registered_root_key_public_digest: str
    wrong_key_indexes: tuple[int, ...]
    materialization_integrity_status: str
    materialization_budget_status: str
    realized_relative_l2: float
    content_relative_l2_limit: float
    actual_runtime_dtype: str
    registered_minus_primary_null_strict_floor_passed: bool
    registered_minus_max_wrong_strict_floor_passed: bool
    registered_minus_primary_null_exact_tie: bool
    registered_minus_max_wrong_exact_tie: bool
    observation_identity: str

    def validate(self) -> None:
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < SCIENTIFIC_CLUSTER_COUNT
            or self.wrong_key_indexes != tuple(range(WRONG_KEY_ROSTER_SIZE))
            or len(self.wrong_key_scores) != WRONG_KEY_ROSTER_SIZE
            or len(self.wrong_key_detector_identities)
            != WRONG_KEY_ROSTER_SIZE
            or len(self.wrong_key_template_digests) != WRONG_KEY_ROSTER_SIZE
        ):
            raise LfWhitenedDirectionalMetricError(
                "directional cluster or wrong-key roster drifted"
            )
        numeric = (
            self.registered_score,
            self.primary_null_score,
            *self.wrong_key_scores,
            self.registered_minus_primary_null,
            self.registered_minus_max_wrong,
            self.realized_relative_l2,
            self.content_relative_l2_limit,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in numeric
        ):
            raise LfWhitenedDirectionalMetricError(
                "directional observation contains non-finite values"
            )
        if (
            self.registered_minus_primary_null
            != self.registered_score - self.primary_null_score
            or self.registered_minus_max_wrong
            != self.registered_score - max(self.wrong_key_scores)
        ):
            raise LfWhitenedDirectionalMetricError(
                "directional paired margin drifted"
            )
        identities = (
            self.candidate_observation_digest,
            self.clean_observation_digest,
            self.registered_detector_identity,
            self.primary_null_detector_identity,
            *self.wrong_key_detector_identities,
            self.detector_config_digest,
            self.observation_protocol,
            self.whitening_asset_digest,
            self.registered_template_digest,
            self.primary_null_template_digest,
            *self.wrong_key_template_digests,
            self.registered_root_key_public_digest,
            self.actual_runtime_dtype,
        )
        if any(type(value) is not str or not value for value in identities):
            raise LfWhitenedDirectionalMetricError(
                "directional detector identity is missing"
            )
        if (
            self.candidate_observation_digest == self.clean_observation_digest
            or len(
                {
                    self.registered_detector_identity,
                    self.primary_null_detector_identity,
                    *self.wrong_key_detector_identities,
                }
            )
            != 1
            or self.registered_template_digest
            != self.primary_null_template_digest
            or self.registered_template_digest
            in set(self.wrong_key_template_digests)
            or len(set(self.wrong_key_template_digests))
            != WRONG_KEY_ROSTER_SIZE
            or self.materialization_integrity_status != "passed"
            or self.materialization_budget_status != "accepted"
            or self.realized_relative_l2 < 0.0
            or pack(">f", self.content_relative_l2_limit)
            != pack(
                ">f",
                CONTENT_RELATIVE_L2_NUMERATOR
                / CONTENT_RELATIVE_L2_DENOMINATOR,
            )
            or self.realized_relative_l2 > self.content_relative_l2_limit
            or self.registered_minus_primary_null_strict_floor_passed
            is not (
                self.registered_minus_primary_null > PRACTICAL_MARGIN_FLOOR
            )
            or self.registered_minus_max_wrong_strict_floor_passed
            is not (self.registered_minus_max_wrong > PRACTICAL_MARGIN_FLOOR)
            or self.registered_minus_primary_null_exact_tie
            is not (self.registered_minus_primary_null == 0.0)
            or self.registered_minus_max_wrong_exact_tie
            is not (self.registered_minus_max_wrong == 0.0)
        ):
            raise LfWhitenedDirectionalMetricError(
                "directional detector/control binding drifted"
            )
        payload = asdict(self)
        identity = payload.pop("observation_identity")
        if identity != _digest(payload):
            raise LfWhitenedDirectionalMetricError(
                "directional observation identity drifted"
            )


def create_lf_whitened_directional_observation(
    **values: object,
) -> LfWhitenedDirectionalObservation:
    payload = dict(values)
    registered = float(payload["registered_score"])
    primary_null = float(payload["primary_null_score"])
    wrong = tuple(float(value) for value in payload["wrong_key_scores"])
    payload.update(
        registered_score=registered,
        primary_null_score=primary_null,
        wrong_key_scores=wrong,
        wrong_key_detector_identities=tuple(
            payload["wrong_key_detector_identities"]
        ),
        wrong_key_template_digests=tuple(
            payload["wrong_key_template_digests"]
        ),
        wrong_key_indexes=tuple(payload["wrong_key_indexes"]),
        registered_minus_primary_null=registered - primary_null,
        registered_minus_max_wrong=registered - max(wrong),
        registered_minus_primary_null_strict_floor_passed=(
            registered - primary_null > PRACTICAL_MARGIN_FLOOR
        ),
        registered_minus_max_wrong_strict_floor_passed=(
            registered - max(wrong) > PRACTICAL_MARGIN_FLOOR
        ),
        registered_minus_primary_null_exact_tie=(
            registered - primary_null == 0.0
        ),
        registered_minus_max_wrong_exact_tie=(
            registered - max(wrong) == 0.0
        ),
    )
    observation = LfWhitenedDirectionalObservation(
        **payload,
        observation_identity=_digest(payload),
    )
    observation.validate()
    return observation


@dataclass(frozen=True, slots=True)
class LfDirectionalMarginSummary:
    observation_count: int
    practical_success_count: int
    exact_tie_count: int
    mean_margin: float | None
    median_margin: float | None
    minimum_margin: float | None
    threshold_free_paired_ranking_auc: float
    exact_one_sided_confidence_lower_bound: float


@dataclass(frozen=True, slots=True)
class LfWhitenedDirectionalAggregate:
    expected_cluster_count: int
    successful_cluster_count: int
    failed_cluster_count: int
    registered_minus_primary_null: LfDirectionalMarginSummary
    registered_minus_max_wrong: LfDirectionalMarginSummary
    identity_violation_count: int
    budget_violation_count: int
    integrity_violation_count: int
    nonfinite_violation_count: int
    mean_realized_relative_l2: float | None
    maximum_realized_relative_l2: float | None
    directional_validation_passed: bool
    module_outcome: str
    candidate_recommendation: str
    aggregate_identity: str

    def validate(self) -> None:
        payload = asdict(self)
        identity = payload.pop("aggregate_identity")
        if identity != _digest(payload):
            raise LfWhitenedDirectionalMetricError(
                "directional aggregate identity drifted"
            )
        if (
            self.expected_cluster_count != SCIENTIFIC_CLUSTER_COUNT
            or self.successful_cluster_count + self.failed_cluster_count
            != SCIENTIFIC_CLUSTER_COUNT
            or self.directional_validation_passed
            is not (self.module_outcome == PASSING_MODULE_OUTCOME)
            or self.directional_validation_passed
            is not (
                self.candidate_recommendation
                == PASSING_CANDIDATE_RECOMMENDATION
            )
        ):
            raise LfWhitenedDirectionalMetricError(
                "directional aggregate boundary drifted"
            )


def _margin_summary(
    values: Sequence[float],
    *,
    failed_cluster_count: int,
) -> LfDirectionalMarginSummary:
    margins = tuple(float(value) for value in values)
    if (
        any(not isfinite(value) for value in margins)
        or len(margins) + failed_cluster_count != SCIENTIFIC_CLUSTER_COUNT
    ):
        raise LfWhitenedDirectionalMetricError(
            "directional margins are invalid"
        )
    successes = sum(value > PRACTICAL_MARGIN_FLOOR for value in margins)
    ties = sum(value == 0.0 for value in margins)
    return LfDirectionalMarginSummary(
        observation_count=SCIENTIFIC_CLUSTER_COUNT,
        practical_success_count=successes,
        exact_tie_count=ties,
        mean_margin=float(mean(margins)) if margins else None,
        median_margin=float(median(margins)) if margins else None,
        minimum_margin=float(min(margins)) if margins else None,
        threshold_free_paired_ranking_auc=float(
            (
                sum(value > 0.0 for value in margins)
                + 0.5 * ties
            )
            / SCIENTIFIC_CLUSTER_COUNT
        ),
        exact_one_sided_confidence_lower_bound=float(
            clopper_pearson_lower(
                successes,
                SCIENTIFIC_CLUSTER_COUNT,
                confidence_level=CONFIDENCE_LEVEL,
            )
        ),
    )


def aggregate_lf_whitened_direction(
    observations: Sequence[LfWhitenedDirectionalObservation],
    *,
    failed_cluster_count: int,
    identity_violation_count: int = 0,
    budget_violation_count: int = 0,
    integrity_violation_count: int = 0,
    nonfinite_violation_count: int = 0,
) -> LfWhitenedDirectionalAggregate:
    items = tuple(observations)
    if (
        type(failed_cluster_count) is not int
        or failed_cluster_count < 0
        or len(items) + failed_cluster_count != SCIENTIFIC_CLUSTER_COUNT
        or any(type(item) is not LfWhitenedDirectionalObservation for item in items)
    ):
        raise LfWhitenedDirectionalMetricError(
            "directional aggregate coverage is incomplete"
        )
    for item in items:
        item.validate()
    if (
        len({item.cluster_ordinal for item in items}) != len(items)
        or len({item.observation_identity for item in items}) != len(items)
    ):
        raise LfWhitenedDirectionalMetricError(
            "directional aggregate observations are duplicated"
        )
    counts = (
        identity_violation_count,
        budget_violation_count,
        integrity_violation_count,
        nonfinite_violation_count,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise LfWhitenedDirectionalMetricError(
            "directional violation count is invalid"
        )
    null_summary = _margin_summary(
        tuple(item.registered_minus_primary_null for item in items),
        failed_cluster_count=failed_cluster_count,
    )
    wrong_summary = _margin_summary(
        tuple(item.registered_minus_max_wrong for item in items),
        failed_cluster_count=failed_cluster_count,
    )
    passed = (
        len(items) == SCIENTIFIC_CLUSTER_COUNT
        and failed_cluster_count == 0
        and null_summary.practical_success_count
        >= MINIMUM_DIRECTIONAL_SUCCESS_COUNT
        and wrong_summary.practical_success_count
        >= MINIMUM_DIRECTIONAL_SUCCESS_COUNT
        and null_summary.exact_one_sided_confidence_lower_bound > 0.5
        and wrong_summary.exact_one_sided_confidence_lower_bound > 0.5
        and sum(counts) == 0
    )
    realized = tuple(item.realized_relative_l2 for item in items)
    payload = {
        "expected_cluster_count": SCIENTIFIC_CLUSTER_COUNT,
        "successful_cluster_count": len(items),
        "failed_cluster_count": failed_cluster_count,
        "registered_minus_primary_null": asdict(null_summary),
        "registered_minus_max_wrong": asdict(wrong_summary),
        "identity_violation_count": identity_violation_count,
        "budget_violation_count": budget_violation_count,
        "integrity_violation_count": integrity_violation_count,
        "nonfinite_violation_count": nonfinite_violation_count,
        "mean_realized_relative_l2": (
            float(mean(realized)) if realized else None
        ),
        "maximum_realized_relative_l2": (
            float(max(realized)) if realized else None
        ),
        "directional_validation_passed": passed,
        "module_outcome": (
            PASSING_MODULE_OUTCOME
            if passed
            else "mechanism_signal_not_observed"
        ),
        "candidate_recommendation": (
            PASSING_CANDIDATE_RECOMMENDATION
            if passed
            else "candidate_not_recommended_for_selection"
        ),
    }
    aggregate = LfWhitenedDirectionalAggregate(
        expected_cluster_count=SCIENTIFIC_CLUSTER_COUNT,
        successful_cluster_count=len(items),
        failed_cluster_count=failed_cluster_count,
        registered_minus_primary_null=null_summary,
        registered_minus_max_wrong=wrong_summary,
        identity_violation_count=identity_violation_count,
        budget_violation_count=budget_violation_count,
        integrity_violation_count=integrity_violation_count,
        nonfinite_violation_count=nonfinite_violation_count,
        mean_realized_relative_l2=payload["mean_realized_relative_l2"],
        maximum_realized_relative_l2=payload["maximum_realized_relative_l2"],
        directional_validation_passed=passed,
        module_outcome=payload["module_outcome"],
        candidate_recommendation=payload["candidate_recommendation"],
        aggregate_identity=_digest(payload),
    )
    aggregate.validate()
    return aggregate


__all__ = [
    "LfDirectionalMarginSummary",
    "LfWhitenedDirectionalAggregate",
    "LfWhitenedDirectionalMetricError",
    "LfWhitenedDirectionalObservation",
    "aggregate_lf_whitened_direction",
    "create_lf_whitened_directional_observation",
]
