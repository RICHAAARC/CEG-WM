"""Threshold-free paired summaries for HF-only detector direction validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import ceil, isfinite
from statistics import mean, median
from struct import pack
from typing import Sequence

from experiments.metrics.binomial import clopper_pearson_lower
from experiments.protocol.hf_only_detector_directional_validation import (
    CONFIDENCE_LEVEL,
    MARGIN_QUANTILE_PROBABILITY,
    MINIMUM_DIRECTIONAL_SUCCESS_COUNT,
    PRACTICAL_MARGIN_FLOOR,
    SCIENTIFIC_CLUSTER_COUNT,
    WRONG_KEY_ROSTER_SIZE,
)


class HfDetectorDirectionalMetricError(ValueError):
    """A paired HF detector directional statistic is invalid."""


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


def paired_rgb8_quality(candidate_values: Sequence[int], clean_values: Sequence[int]) -> tuple[float, float]:
    candidate = tuple(candidate_values)
    clean = tuple(clean_values)
    if (
        not candidate
        or len(candidate) != len(clean)
        or any(type(item) is not int or not 0 <= item <= 255 for item in (*candidate, *clean))
    ):
        raise HfDetectorDirectionalMetricError("paired RGB values are invalid")
    clean_norm = sum(item * item for item in clean) ** 0.5
    if clean_norm <= 0.0:
        raise HfDetectorDirectionalMetricError("paired clean RGB norm is zero")
    relative = sum(
        (candidate_item - clean_item) ** 2
        for candidate_item, clean_item in zip(candidate, clean, strict=True)
    ) ** 0.5 / clean_norm
    if not isfinite(relative):
        raise HfDetectorDirectionalMetricError("paired RGB quality is non-finite")
    mse = sum(
        (candidate_item - clean_item) ** 2
        for candidate_item, clean_item in zip(candidate, clean, strict=True)
    ) / len(candidate)
    return float(relative), float(mse)


@dataclass(frozen=True, slots=True)
class HfDetectorDirectionalObservation:
    cluster_ordinal: int
    wrong_key_index: int
    registered_score: float
    wrong_key_score: float
    primary_null_score: float
    registered_minus_wrong_key: float
    registered_minus_primary_null: float
    candidate_observation_digest: str
    clean_observation_digest: str
    registered_detector_identity: str
    wrong_key_detector_identity: str
    primary_null_detector_identity: str
    detector_config_digest: str
    observation_protocol: str
    detector_statistic_identity: str
    registered_template_digest: str
    wrong_key_template_digest: str
    primary_null_template_digest: str
    registered_root_key_public_digest: str
    wrong_key_root_key_public_digest: str
    primary_null_root_key_public_digest: str
    materialization_integrity_status: str
    realized_relative_l2: float
    content_relative_l2_limit: float
    rgb_paired_relative_l2: float
    rgb_paired_mse: float
    rgb_quality_dtype: str
    actual_runtime_dtype: str
    registered_minus_wrong_key_strict_floor_passed: bool
    registered_minus_wrong_key_exact_tie: bool
    registered_minus_primary_null_strict_floor_passed: bool
    registered_minus_primary_null_exact_tie: bool
    observation_identity: str

    def validate(self) -> None:
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < SCIENTIFIC_CLUSTER_COUNT
            or type(self.wrong_key_index) is not int
            or self.wrong_key_index != self.cluster_ordinal % WRONG_KEY_ROSTER_SIZE
        ):
            raise HfDetectorDirectionalMetricError(
                "directional cluster or wrong-key identity drifted"
            )
        numeric = (
            self.registered_score,
            self.wrong_key_score,
            self.primary_null_score,
            self.registered_minus_wrong_key,
            self.registered_minus_primary_null,
            self.realized_relative_l2,
            self.content_relative_l2_limit,
            self.rgb_paired_relative_l2,
            self.rgb_paired_mse,
        )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not isfinite(float(item))
            for item in numeric
        ):
            raise HfDetectorDirectionalMetricError(
                "directional observation contains non-finite values"
            )
        if self.registered_minus_wrong_key != self.registered_score - self.wrong_key_score or self.registered_minus_primary_null != self.registered_score - self.primary_null_score:
            raise HfDetectorDirectionalMetricError(
                "directional paired margin drifted"
            )
        identities = (
            self.candidate_observation_digest,
            self.clean_observation_digest,
            self.registered_detector_identity,
            self.wrong_key_detector_identity,
            self.primary_null_detector_identity,
            self.detector_config_digest,
            self.observation_protocol,
            self.detector_statistic_identity,
            self.registered_template_digest,
            self.wrong_key_template_digest,
            self.primary_null_template_digest,
            self.registered_root_key_public_digest,
            self.wrong_key_root_key_public_digest,
            self.primary_null_root_key_public_digest,
        )
        if any(type(item) is not str or not item for item in identities):
            raise HfDetectorDirectionalMetricError(
                "directional detector identity is missing"
            )
        if (
            self.candidate_observation_digest == self.clean_observation_digest
            or len({self.registered_detector_identity, self.wrong_key_detector_identity, self.primary_null_detector_identity}) != 1
            or self.registered_root_key_public_digest
            != self.wrong_key_root_key_public_digest
            or self.registered_root_key_public_digest
            != self.primary_null_root_key_public_digest
            or self.registered_template_digest == self.wrong_key_template_digest
            or self.registered_template_digest != self.primary_null_template_digest
            or self.materialization_integrity_status != "passed"
            or self.realized_relative_l2 < 0.0
            or pack(">f", self.content_relative_l2_limit)
            != pack(">f", 3 / 250)
            or self.realized_relative_l2 > self.content_relative_l2_limit
            or self.rgb_paired_relative_l2 < 0.0
            or self.rgb_paired_mse < 0.0
            or self.rgb_quality_dtype != "torch.uint8"
            or type(self.actual_runtime_dtype) is not str
            or not self.actual_runtime_dtype
            or self.registered_minus_wrong_key_strict_floor_passed
            is not (self.registered_minus_wrong_key > PRACTICAL_MARGIN_FLOOR)
            or self.registered_minus_primary_null_strict_floor_passed
            is not (self.registered_minus_primary_null > PRACTICAL_MARGIN_FLOOR)
            or self.registered_minus_wrong_key_exact_tie
            is not (self.registered_minus_wrong_key == 0.0)
            or self.registered_minus_primary_null_exact_tie
            is not (self.registered_minus_primary_null == 0.0)
        ):
            raise HfDetectorDirectionalMetricError(
                "directional detector/control binding drifted"
            )
        payload = asdict(self)
        identity = payload.pop("observation_identity")
        if identity != _digest(payload):
            raise HfDetectorDirectionalMetricError(
                "directional observation identity drifted"
            )


def create_hf_detector_directional_observation(**values: object) -> HfDetectorDirectionalObservation:
    payload = dict(values)
    registered_score = float(payload["registered_score"])
    wrong_key_score = float(payload["wrong_key_score"])
    primary_null_score = float(payload["primary_null_score"])
    payload.update(
        registered_score=registered_score,
        wrong_key_score=wrong_key_score,
        primary_null_score=primary_null_score,
        registered_minus_wrong_key=registered_score - wrong_key_score,
        registered_minus_primary_null=registered_score - primary_null_score,
        registered_minus_wrong_key_strict_floor_passed=(registered_score - wrong_key_score) > PRACTICAL_MARGIN_FLOOR,
        registered_minus_wrong_key_exact_tie=(registered_score - wrong_key_score) == 0.0,
        registered_minus_primary_null_strict_floor_passed=(registered_score - primary_null_score) > PRACTICAL_MARGIN_FLOOR,
        registered_minus_primary_null_exact_tie=(registered_score - primary_null_score) == 0.0,
    )
    result = HfDetectorDirectionalObservation(
        **payload,
        observation_identity=_digest(payload),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class DirectionalMarginSummary:
    observation_count: int
    practical_success_count: int
    exact_tie_count: int
    mean_margin: float | None
    median_margin: float | None
    minimum_margin: float | None
    lower_quartile_nearest_rank_margin: float | None
    threshold_free_paired_ranking_auc: float
    exact_one_sided_confidence_lower_bound: float


@dataclass(frozen=True, slots=True)
class HfDetectorDirectionalAggregate:
    expected_cluster_count: int
    successful_cluster_count: int
    failed_cluster_count: int
    registered_minus_primary_null: DirectionalMarginSummary
    registered_minus_wrong_key: DirectionalMarginSummary
    identity_violation_count: int
    budget_violation_count: int
    integrity_violation_count: int
    nonfinite_violation_count: int
    mean_realized_relative_l2: float | None
    maximum_realized_relative_l2: float | None
    mean_rgb_paired_relative_l2: float | None
    maximum_rgb_paired_relative_l2: float | None
    mean_rgb_paired_mse: float | None
    maximum_rgb_paired_mse: float | None
    allow_request_for_next_scientific_gate: bool
    aggregate_identity: str

    def validate(self) -> None:
        payload = asdict(self)
        identity = payload.pop("aggregate_identity")
        if identity != _digest(payload):
            raise HfDetectorDirectionalMetricError(
                "directional aggregate identity drifted"
            )
        if self.expected_cluster_count != SCIENTIFIC_CLUSTER_COUNT or self.successful_cluster_count + self.failed_cluster_count != SCIENTIFIC_CLUSTER_COUNT:
            raise HfDetectorDirectionalMetricError(
                "directional aggregate denominator drifted"
            )


def _margin_summary(values: Sequence[float], *, failed_cluster_count: int) -> DirectionalMarginSummary:
    margins = tuple(float(item) for item in values)
    if any(not isfinite(item) for item in margins) or len(margins) + failed_cluster_count != SCIENTIFIC_CLUSTER_COUNT:
        raise HfDetectorDirectionalMetricError("directional margins are invalid")
    ordered = tuple(sorted(margins))
    successes = sum(item > PRACTICAL_MARGIN_FLOOR for item in margins)
    ties = sum(item == 0.0 for item in margins)
    quantile_index = max(0, ceil(MARGIN_QUANTILE_PROBABILITY * len(ordered)) - 1) if ordered else 0
    return DirectionalMarginSummary(
        observation_count=SCIENTIFIC_CLUSTER_COUNT,
        practical_success_count=successes,
        exact_tie_count=ties,
        mean_margin=float(mean(margins)) if margins else None,
        median_margin=float(median(margins)) if margins else None,
        minimum_margin=float(ordered[0]) if ordered else None,
        lower_quartile_nearest_rank_margin=float(ordered[quantile_index]) if ordered else None,
        threshold_free_paired_ranking_auc=float(
            (sum(item > 0.0 for item in margins) + 0.5 * ties) / SCIENTIFIC_CLUSTER_COUNT
        ),
        exact_one_sided_confidence_lower_bound=float(
            clopper_pearson_lower(
                successes,
                SCIENTIFIC_CLUSTER_COUNT,
                confidence_level=CONFIDENCE_LEVEL,
            )
        ),
    )


def aggregate_hf_detector_direction(
    observations: Sequence[HfDetectorDirectionalObservation],
    *,
    failed_cluster_count: int,
    identity_violation_count: int = 0,
    budget_violation_count: int = 0,
    integrity_violation_count: int = 0,
    nonfinite_violation_count: int = 0,
) -> HfDetectorDirectionalAggregate:
    items = tuple(observations)
    if (
        type(failed_cluster_count) is not int
        or failed_cluster_count < 0
        or len(items) + failed_cluster_count != SCIENTIFIC_CLUSTER_COUNT
        or any(type(item) is not HfDetectorDirectionalObservation for item in items)
    ):
        raise HfDetectorDirectionalMetricError(
            "directional aggregate coverage is incomplete"
        )
    for item in items:
        item.validate()
    if len({item.cluster_ordinal for item in items}) != len(items) or len({item.observation_identity for item in items}) != len(items):
        raise HfDetectorDirectionalMetricError(
            "directional aggregate observations are duplicated"
        )
    counts = (
        identity_violation_count,
        budget_violation_count,
        integrity_violation_count,
        nonfinite_violation_count,
    )
    if any(type(item) is not int or item < 0 for item in counts):
        raise HfDetectorDirectionalMetricError(
            "directional violation count is invalid"
        )
    null_summary = _margin_summary(
        tuple(item.registered_minus_primary_null for item in items), failed_cluster_count=failed_cluster_count
    )
    wrong_summary = _margin_summary(
        tuple(item.registered_minus_wrong_key for item in items), failed_cluster_count=failed_cluster_count
    )
    realized = tuple(item.realized_relative_l2 for item in items)
    rgb_quality = tuple(item.rgb_paired_relative_l2 for item in items)
    rgb_mse = tuple(item.rgb_paired_mse for item in items)
    gate = (
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
    payload = {
        "expected_cluster_count": SCIENTIFIC_CLUSTER_COUNT,
        "successful_cluster_count": len(items),
        "failed_cluster_count": failed_cluster_count,
        "registered_minus_primary_null": asdict(null_summary),
        "registered_minus_wrong_key": asdict(wrong_summary),
        "identity_violation_count": identity_violation_count,
        "budget_violation_count": budget_violation_count,
        "integrity_violation_count": integrity_violation_count,
        "nonfinite_violation_count": nonfinite_violation_count,
        "mean_realized_relative_l2": float(mean(realized)) if realized else None,
        "maximum_realized_relative_l2": float(max(realized)) if realized else None,
        "mean_rgb_paired_relative_l2": float(mean(rgb_quality)) if rgb_quality else None,
        "maximum_rgb_paired_relative_l2": float(max(rgb_quality)) if rgb_quality else None,
        "mean_rgb_paired_mse": float(mean(rgb_mse)) if rgb_mse else None,
        "maximum_rgb_paired_mse": float(max(rgb_mse)) if rgb_mse else None,
        "allow_request_for_next_scientific_gate": gate,
    }
    aggregate = HfDetectorDirectionalAggregate(
        expected_cluster_count=payload["expected_cluster_count"],
        successful_cluster_count=payload["successful_cluster_count"],
        failed_cluster_count=payload["failed_cluster_count"],
        registered_minus_primary_null=null_summary,
        registered_minus_wrong_key=wrong_summary,
        identity_violation_count=identity_violation_count,
        budget_violation_count=budget_violation_count,
        integrity_violation_count=integrity_violation_count,
        nonfinite_violation_count=nonfinite_violation_count,
        mean_realized_relative_l2=payload["mean_realized_relative_l2"],
        maximum_realized_relative_l2=payload["maximum_realized_relative_l2"],
        mean_rgb_paired_relative_l2=payload["mean_rgb_paired_relative_l2"],
        maximum_rgb_paired_relative_l2=payload["maximum_rgb_paired_relative_l2"],
        mean_rgb_paired_mse=payload["mean_rgb_paired_mse"],
        maximum_rgb_paired_mse=payload["maximum_rgb_paired_mse"],
        allow_request_for_next_scientific_gate=gate,
        aggregate_identity=_digest(payload),
    )
    aggregate.validate()
    return aggregate


__all__ = [
    "DirectionalMarginSummary",
    "HfDetectorDirectionalAggregate",
    "HfDetectorDirectionalMetricError",
    "HfDetectorDirectionalObservation",
    "aggregate_hf_detector_direction",
    "create_hf_detector_directional_observation",
    "paired_rgb8_quality",
]
