"""Frozen development summaries for real Q/K synchronization writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
from statistics import mean, median
from typing import Sequence

from experiments.protocol.qk_synchronization_write_diagnostic import (
    GEOMETRY_RATIO_ROSTER,
    CONTENT_PROJECTION_RELATIVE_LIMIT,
    CONTENT_RELATIVE_L2_DENOMINATOR,
    CONTENT_RELATIVE_L2_NUMERATOR,
    LINE_SEARCH_FACTORS,
    PASSING_CANDIDATE_RECOMMENDATION,
    PASSING_MODULE_OUTCOME,
    RATIO_PROBE_CLUSTER_COUNT,
    RATIO_PROBE_UNIT_COUNT,
    TRANSFORM_PROBE_ROSTER,
    TRANSFORM_PROBE_UNIT_COUNT,
    WRONG_KEY_INDEXES,
)


class QkSynchronizationWriteMetricError(ValueError):
    """A Q/K synchronization-write diagnostic statistic is invalid."""


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
            default=lambda item: asdict(item),
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
    )


@dataclass(frozen=True, slots=True)
class QkRgb8QualityDelta:
    relative_l2: float
    mean_squared_error: float
    content_only_rgb8_digest: str
    geometry_written_rgb8_digest: str
    quality_identity: str

    def validate(self) -> None:
        payload = asdict(self)
        identity = payload.pop("quality_identity")
        if (
            not _finite(self.relative_l2)
            or not _finite(self.mean_squared_error)
            or self.relative_l2 < 0.0
            or self.mean_squared_error < 0.0
            or not self.content_only_rgb8_digest
            or not self.geometry_written_rgb8_digest
            or self.content_only_rgb8_digest == self.geometry_written_rgb8_digest
            or identity != _digest(payload)
        ):
            raise QkSynchronizationWriteMetricError(
                "Q/K RGB8 quality delta is invalid"
            )


def create_qk_rgb8_quality_delta(
    *,
    relative_l2: float,
    mean_squared_error: float,
    content_only_rgb8_digest: str,
    geometry_written_rgb8_digest: str,
) -> QkRgb8QualityDelta:
    payload = {
        "relative_l2": float(relative_l2),
        "mean_squared_error": float(mean_squared_error),
        "content_only_rgb8_digest": content_only_rgb8_digest,
        "geometry_written_rgb8_digest": geometry_written_rgb8_digest,
    }
    result = QkRgb8QualityDelta(**payload, quality_identity=_digest(payload))
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class QkRatioProbeObservation:
    cluster_ordinal: int
    ratio_identity: str
    geometry_ratio: float
    write_accepted: bool
    line_search_factor: float | None
    ste_acceptance_baseline_score: float
    ste_acceptance_score: float | None
    public_pre_registered_score: float
    public_pre_wrong_key_scores: tuple[float, ...]
    public_post_registered_score: float | None
    public_post_wrong_key_scores: tuple[float, ...]
    registered_gain: float | None
    wrong_key_gains: tuple[float, ...]
    maximum_wrong_gain: float | None
    keyed_gain_margin: float | None
    actual_geometry_relative_l2: float | None
    actual_total_relative_l2: float | None
    content_span_projection_relative: float | None
    rgb8_quality_delta: QkRgb8QualityDelta | None
    public_observation_identity: str
    geometry_key_family_digest: str
    registered_template_digest: str
    wrong_key_template_digests: tuple[str, ...]
    wrong_key_indexes: tuple[int, ...]
    method_identity: str
    runtime_identity: str
    runtime_config_digest: str
    model_revision: str
    package_identity: str
    identity_violation_count: int
    budget_violation_count: int
    integrity_violation_count: int
    nonfinite_violation_count: int
    ratio_eligible: bool
    observation_identity: str

    def validate(self) -> None:
        expected_ratio = dict(GEOMETRY_RATIO_ROSTER).get(self.ratio_identity)
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < RATIO_PROBE_CLUSTER_COUNT
            or expected_ratio is None
            or self.geometry_ratio != expected_ratio
            or self.wrong_key_indexes != WRONG_KEY_INDEXES
            or len(self.public_pre_wrong_key_scores) != len(WRONG_KEY_INDEXES)
            or len(self.wrong_key_template_digests) != len(WRONG_KEY_INDEXES)
            or len(set(self.wrong_key_template_digests)) != len(WRONG_KEY_INDEXES)
            or self.registered_template_digest in self.wrong_key_template_digests
        ):
            raise QkSynchronizationWriteMetricError(
                "Q/K ratio probe identity or control roster drifted"
            )
        if any(
            type(value) is not str or not value
            for value in (
                self.public_observation_identity,
                self.geometry_key_family_digest,
                self.registered_template_digest,
                *self.wrong_key_template_digests,
                self.method_identity,
                self.runtime_identity,
                self.runtime_config_digest,
                self.model_revision,
                self.package_identity,
            )
        ):
            raise QkSynchronizationWriteMetricError(
                "Q/K ratio probe authority is missing"
            )
        required_numeric = (
            self.ste_acceptance_baseline_score,
            self.public_pre_registered_score,
            *self.public_pre_wrong_key_scores,
        )
        if any(not _finite(value) for value in required_numeric):
            raise QkSynchronizationWriteMetricError(
                "Q/K ratio probe baseline is non-finite"
            )
        violations = (
            self.identity_violation_count,
            self.budget_violation_count,
            self.integrity_violation_count,
            self.nonfinite_violation_count,
        )
        if any(type(value) is not int or value < 0 for value in violations):
            raise QkSynchronizationWriteMetricError(
                "Q/K ratio probe violation count is invalid"
            )
        accepted_values = (
            self.line_search_factor,
            self.ste_acceptance_score,
            self.public_post_registered_score,
            *self.public_post_wrong_key_scores,
            self.registered_gain,
            *self.wrong_key_gains,
            self.maximum_wrong_gain,
            self.keyed_gain_margin,
            self.actual_geometry_relative_l2,
            self.actual_total_relative_l2,
            self.content_span_projection_relative,
        )
        if self.write_accepted:
            if (
                len(self.public_post_wrong_key_scores) != len(WRONG_KEY_INDEXES)
                or len(self.wrong_key_gains) != len(WRONG_KEY_INDEXES)
                or self.rgb8_quality_delta is None
                or any(not _finite(value) for value in accepted_values)
                or self.line_search_factor not in LINE_SEARCH_FACTORS
                or self.ste_acceptance_score
                <= self.ste_acceptance_baseline_score
                or self.actual_geometry_relative_l2
                > (
                    CONTENT_RELATIVE_L2_NUMERATOR
                    / CONTENT_RELATIVE_L2_DENOMINATOR
                    * self.geometry_ratio
                )
                or self.actual_total_relative_l2
                > (
                    CONTENT_RELATIVE_L2_NUMERATOR
                    / CONTENT_RELATIVE_L2_DENOMINATOR
                    * sqrt(1.0 + self.geometry_ratio * self.geometry_ratio)
                )
                or self.content_span_projection_relative
                > CONTENT_PROJECTION_RELATIVE_LIMIT
            ):
                raise QkSynchronizationWriteMetricError(
                    "accepted Q/K ratio probe is incomplete"
                )
            self.rgb8_quality_delta.validate()
            wrong_gains = tuple(
                post - pre
                for post, pre in zip(
                    self.public_post_wrong_key_scores,
                    self.public_pre_wrong_key_scores,
                    strict=True,
                )
            )
            registered_gain = (
                self.public_post_registered_score
                - self.public_pre_registered_score
            )
            if (
                self.registered_gain != registered_gain
                or self.wrong_key_gains != wrong_gains
                or self.maximum_wrong_gain != max(wrong_gains)
                or self.keyed_gain_margin
                != registered_gain - max(wrong_gains)
            ):
                raise QkSynchronizationWriteMetricError(
                    "actual public Q/K gain calculation drifted"
                )
        elif (
            self.public_post_wrong_key_scores
            or self.wrong_key_gains
            or self.rgb8_quality_delta is not None
            or any(value is not None for value in accepted_values)
        ):
            raise QkSynchronizationWriteMetricError(
                "rejected Q/K ratio probe contains post-write evidence"
            )
        eligible = (
            self.write_accepted
            and self.registered_gain is not None
            and self.registered_gain > 0.0
            and self.keyed_gain_margin is not None
            and self.keyed_gain_margin > 0.0
            and sum(violations) == 0
        )
        payload = asdict(self)
        identity = payload.pop("observation_identity")
        if self.ratio_eligible is not eligible or identity != _digest(payload):
            raise QkSynchronizationWriteMetricError(
                "Q/K ratio probe decision or identity drifted"
            )


def create_qk_ratio_probe_observation(**values: object) -> QkRatioProbeObservation:
    payload = dict(values)
    payload["public_pre_wrong_key_scores"] = tuple(
        float(value) for value in payload["public_pre_wrong_key_scores"]
    )
    payload["public_post_wrong_key_scores"] = tuple(
        float(value) for value in payload.get("public_post_wrong_key_scores", ())
    )
    payload["wrong_key_template_digests"] = tuple(
        payload["wrong_key_template_digests"]
    )
    payload["wrong_key_indexes"] = tuple(payload["wrong_key_indexes"])
    if payload["write_accepted"]:
        registered_gain = float(payload["public_post_registered_score"]) - float(
            payload["public_pre_registered_score"]
        )
        wrong_gains = tuple(
            post - pre
            for post, pre in zip(
                payload["public_post_wrong_key_scores"],
                payload["public_pre_wrong_key_scores"],
                strict=True,
            )
        )
        payload.update(
            registered_gain=registered_gain,
            wrong_key_gains=wrong_gains,
            maximum_wrong_gain=max(wrong_gains),
            keyed_gain_margin=registered_gain - max(wrong_gains),
        )
    else:
        payload.update(
            line_search_factor=None,
            ste_acceptance_score=None,
            public_post_registered_score=None,
            public_post_wrong_key_scores=(),
            registered_gain=None,
            wrong_key_gains=(),
            maximum_wrong_gain=None,
            keyed_gain_margin=None,
            actual_geometry_relative_l2=None,
            actual_total_relative_l2=None,
            content_span_projection_relative=None,
            rgb8_quality_delta=None,
        )
    violations = tuple(
        int(payload[name])
        for name in (
            "identity_violation_count",
            "budget_violation_count",
            "integrity_violation_count",
            "nonfinite_violation_count",
        )
    )
    payload["ratio_eligible"] = (
        bool(payload["write_accepted"])
        and payload["registered_gain"] is not None
        and payload["registered_gain"] > 0.0
        and payload["keyed_gain_margin"] is not None
        and payload["keyed_gain_margin"] > 0.0
        and sum(violations) == 0
    )
    observation = QkRatioProbeObservation(
        **payload, observation_identity=_digest(payload)
    )
    observation.validate()
    return observation


@dataclass(frozen=True, slots=True)
class QkTerminalFailure:
    cluster_ordinal: int
    case_identity: str
    failure_class: str

    def validate(self) -> None:
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < RATIO_PROBE_CLUSTER_COUNT
            or not self.case_identity
            or self.failure_class
            not in {"implementation_failure", "resource_failure"}
        ):
            raise QkSynchronizationWriteMetricError(
                "Q/K terminal failure classification is invalid"
            )


@dataclass(frozen=True, slots=True)
class QkRatioProbeAggregate:
    expected_unit_count: int
    successful_unit_count: int
    implementation_failure_count: int
    resource_failure_count: int
    eligible_counts_by_ratio: tuple[tuple[str, int], ...]
    selected_ratio_identity: str | None
    selected_geometry_ratio: float | None
    ratio_probe_outcome: str
    aggregate_identity: str

    def validate(self) -> None:
        expected_counts = tuple(name for name, _ in GEOMETRY_RATIO_ROSTER)
        payload = asdict(self)
        identity = payload.pop("aggregate_identity")
        if (
            self.expected_unit_count != RATIO_PROBE_UNIT_COUNT
            or self.successful_unit_count
            + self.implementation_failure_count
            + self.resource_failure_count
            != RATIO_PROBE_UNIT_COUNT
            or tuple(name for name, _ in self.eligible_counts_by_ratio)
            != expected_counts
            or any(not 0 <= count <= RATIO_PROBE_CLUSTER_COUNT for _, count in self.eligible_counts_by_ratio)
            or identity != _digest(payload)
        ):
            raise QkSynchronizationWriteMetricError(
                "Q/K ratio aggregate coverage or identity drifted"
            )
        expected_outcome = (
            "implementation_blocked"
            if self.implementation_failure_count
            else (
                "resource_blocked"
                if self.resource_failure_count
                else (
                    PASSING_MODULE_OUTCOME
                    if self.selected_ratio_identity is not None
                    else "mechanism_signal_not_observed"
                )
            )
        )
        selected = dict(GEOMETRY_RATIO_ROSTER).get(self.selected_ratio_identity)
        if (
            self.ratio_probe_outcome != expected_outcome
            or (self.selected_ratio_identity is None)
            is not (self.selected_geometry_ratio is None)
            or (selected is not None and selected != self.selected_geometry_ratio)
            or (self.selected_ratio_identity is not None and expected_outcome != PASSING_MODULE_OUTCOME)
        ):
            raise QkSynchronizationWriteMetricError(
                "Q/K ratio eligibility outcome drifted"
            )


def aggregate_qk_ratio_probes(
    observations: Sequence[QkRatioProbeObservation],
    failures: Sequence[QkTerminalFailure] = (),
) -> QkRatioProbeAggregate:
    items = tuple(observations)
    failed = tuple(failures)
    for item in items:
        item.validate()
    for item in failed:
        item.validate()
    expected = {
        (cluster, ratio_identity)
        for ratio_identity, _ in GEOMETRY_RATIO_ROSTER
        for cluster in range(RATIO_PROBE_CLUSTER_COUNT)
    }
    observed = {(item.cluster_ordinal, item.ratio_identity) for item in items}
    failed_keys = {(item.cluster_ordinal, item.case_identity) for item in failed}
    if (
        observed & failed_keys
        or observed | failed_keys != expected
        or len(observed) != len(items)
        or len(failed_keys) != len(failed)
    ):
        raise QkSynchronizationWriteMetricError(
            "Q/K ratio aggregate requires all twelve unique terminal units"
        )
    implementation_failures = sum(
        item.failure_class == "implementation_failure" for item in failed
    )
    resource_failures = sum(
        item.failure_class == "resource_failure" for item in failed
    )
    counts = tuple(
        (
            ratio_identity,
            sum(
                item.ratio_identity == ratio_identity and item.ratio_eligible
                for item in items
            ),
        )
        for ratio_identity, _ in GEOMETRY_RATIO_ROSTER
    )
    selected_identity = None
    selected_ratio = None
    if not failed:
        for ratio_identity, ratio in GEOMETRY_RATIO_ROSTER:
            if dict(counts)[ratio_identity] == RATIO_PROBE_CLUSTER_COUNT:
                selected_identity, selected_ratio = ratio_identity, ratio
                break
    outcome = (
        "implementation_blocked"
        if implementation_failures
        else (
            "resource_blocked"
            if resource_failures
            else (
                PASSING_MODULE_OUTCOME
                if selected_identity is not None
                else "mechanism_signal_not_observed"
            )
        )
    )
    payload = {
        "expected_unit_count": RATIO_PROBE_UNIT_COUNT,
        "successful_unit_count": len(items),
        "implementation_failure_count": implementation_failures,
        "resource_failure_count": resource_failures,
        "eligible_counts_by_ratio": counts,
        "selected_ratio_identity": selected_identity,
        "selected_geometry_ratio": selected_ratio,
        "ratio_probe_outcome": outcome,
    }
    aggregate = QkRatioProbeAggregate(**payload, aggregate_identity=_digest(payload))
    aggregate.validate()
    return aggregate


@dataclass(frozen=True, slots=True)
class QkTransformedRelationObservation:
    cluster_ordinal: int
    transform_identity: str
    selected_ratio_identity: str
    source_geometry_written_rgb8_digest: str
    transformed_rgb8_digest: str
    registered_score: float
    wrong_key_scores: tuple[float, ...]
    registered_minus_max_wrong: float
    public_observation_identity: str
    method_identity: str
    runtime_identity: str
    identity_violation_count: int
    integrity_violation_count: int
    nonfinite_violation_count: int
    observation_identity: str

    def validate(self) -> None:
        transform_names = tuple(item[0] for item in TRANSFORM_PROBE_ROSTER)
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < RATIO_PROBE_CLUSTER_COUNT
            or self.transform_identity not in transform_names
            or self.selected_ratio_identity not in dict(GEOMETRY_RATIO_ROSTER)
            or len(self.wrong_key_scores) != len(WRONG_KEY_INDEXES)
            or any(not _finite(value) for value in (self.registered_score, *self.wrong_key_scores))
            or self.registered_minus_max_wrong
            != self.registered_score - max(self.wrong_key_scores)
            or any(
                type(value) is not str or not value
                for value in (
                    self.source_geometry_written_rgb8_digest,
                    self.transformed_rgb8_digest,
                    self.public_observation_identity,
                    self.method_identity,
                    self.runtime_identity,
                )
            )
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.identity_violation_count,
                    self.integrity_violation_count,
                    self.nonfinite_violation_count,
                )
            )
        ):
            raise QkSynchronizationWriteMetricError(
                "transformed Q/K relation probe is invalid"
            )
        payload = asdict(self)
        identity = payload.pop("observation_identity")
        if identity != _digest(payload):
            raise QkSynchronizationWriteMetricError(
                "transformed Q/K relation identity drifted"
            )


def create_qk_transformed_relation_observation(
    **values: object,
) -> QkTransformedRelationObservation:
    payload = dict(values)
    payload["wrong_key_scores"] = tuple(
        float(value) for value in payload["wrong_key_scores"]
    )
    payload["registered_minus_max_wrong"] = float(payload["registered_score"]) - max(
        payload["wrong_key_scores"]
    )
    result = QkTransformedRelationObservation(
        **payload, observation_identity=_digest(payload)
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class QkSynchronizationDiagnosisAggregate:
    ratio_probe: QkRatioProbeAggregate
    transform_success_count: int
    transform_excluded_count: int
    transform_implementation_failure_count: int
    transform_resource_failure_count: int
    transform_margin_mean: float | None
    transform_margin_median: float | None
    transform_margin_minimum: float | None
    module_outcome: str
    candidate_recommendation: str
    aggregate_identity: str

    def validate(self) -> None:
        self.ratio_probe.validate()
        if (
            self.transform_success_count
            + self.transform_excluded_count
            + self.transform_implementation_failure_count
            + self.transform_resource_failure_count
            != TRANSFORM_PROBE_UNIT_COUNT
        ):
            raise QkSynchronizationWriteMetricError(
                "Q/K diagnosis transform denominator drifted"
            )
        expected_outcome = (
            "implementation_blocked"
            if (
                self.ratio_probe.ratio_probe_outcome == "implementation_blocked"
                or self.transform_implementation_failure_count
            )
            else (
                "resource_blocked"
                if (
                    self.ratio_probe.ratio_probe_outcome == "resource_blocked"
                    or self.transform_resource_failure_count
                )
                else (
                    PASSING_MODULE_OUTCOME
                    if self.transform_success_count == TRANSFORM_PROBE_UNIT_COUNT
                    else "mechanism_signal_not_observed"
                )
            )
        )
        payload = asdict(self)
        identity = payload.pop("aggregate_identity")
        if (
            self.module_outcome != expected_outcome
            or self.candidate_recommendation
            != (
                PASSING_CANDIDATE_RECOMMENDATION
                if expected_outcome == PASSING_MODULE_OUTCOME
                else "candidate_not_recommended_for_selection"
            )
            or identity != _digest(payload)
        ):
            raise QkSynchronizationWriteMetricError(
                "Q/K diagnosis outcome or identity drifted"
            )


def aggregate_qk_synchronization_diagnosis(
    ratio_probe: QkRatioProbeAggregate,
    transformed_observations: Sequence[QkTransformedRelationObservation] = (),
    transform_failures: Sequence[QkTerminalFailure] = (),
    *,
    dependency_blocked_excluded_count: int = 0,
) -> QkSynchronizationDiagnosisAggregate:
    ratio_probe.validate()
    transformed = tuple(transformed_observations)
    failed = tuple(transform_failures)
    for item in transformed:
        item.validate()
    for item in failed:
        item.validate()
    expected = {
        (cluster, transform_identity)
        for transform_identity, *_ in TRANSFORM_PROBE_ROSTER
        for cluster in range(RATIO_PROBE_CLUSTER_COUNT)
    }
    observed = {(item.cluster_ordinal, item.transform_identity) for item in transformed}
    failed_keys = {(item.cluster_ordinal, item.case_identity) for item in failed}
    if ratio_probe.selected_ratio_identity is None:
        if transformed or failed or dependency_blocked_excluded_count != TRANSFORM_PROBE_UNIT_COUNT:
            raise QkSynchronizationWriteMetricError(
                "ineligible ratio requires sixteen excluded transform terminals"
            )
    elif (
        dependency_blocked_excluded_count != 0
        or observed & failed_keys
        or observed | failed_keys != expected
        or len(observed) != len(transformed)
        or len(failed_keys) != len(failed)
        or any(
            item.selected_ratio_identity != ratio_probe.selected_ratio_identity
            for item in transformed
        )
    ):
        raise QkSynchronizationWriteMetricError(
            "eligible ratio requires sixteen unique transform terminals"
        )
    implementation_failures = sum(
        item.failure_class == "implementation_failure" for item in failed
    )
    resource_failures = sum(
        item.failure_class == "resource_failure" for item in failed
    )
    margins = tuple(item.registered_minus_max_wrong for item in transformed)
    module_outcome = (
        "implementation_blocked"
        if (
            ratio_probe.ratio_probe_outcome == "implementation_blocked"
            or implementation_failures
        )
        else (
            "resource_blocked"
            if (
                ratio_probe.ratio_probe_outcome == "resource_blocked"
                or resource_failures
            )
            else (
                PASSING_MODULE_OUTCOME
                if len(transformed) == TRANSFORM_PROBE_UNIT_COUNT
                else "mechanism_signal_not_observed"
            )
        )
    )
    payload = {
        "ratio_probe": asdict(ratio_probe),
        "transform_success_count": len(transformed),
        "transform_excluded_count": dependency_blocked_excluded_count,
        "transform_implementation_failure_count": implementation_failures,
        "transform_resource_failure_count": resource_failures,
        "transform_margin_mean": float(mean(margins)) if margins else None,
        "transform_margin_median": float(median(margins)) if margins else None,
        "transform_margin_minimum": float(min(margins)) if margins else None,
        "module_outcome": module_outcome,
        "candidate_recommendation": (
            PASSING_CANDIDATE_RECOMMENDATION
            if module_outcome == PASSING_MODULE_OUTCOME
            else "candidate_not_recommended_for_selection"
        ),
    }
    aggregate = QkSynchronizationDiagnosisAggregate(
        ratio_probe=ratio_probe,
        transform_success_count=len(transformed),
        transform_excluded_count=dependency_blocked_excluded_count,
        transform_implementation_failure_count=implementation_failures,
        transform_resource_failure_count=resource_failures,
        transform_margin_mean=payload["transform_margin_mean"],
        transform_margin_median=payload["transform_margin_median"],
        transform_margin_minimum=payload["transform_margin_minimum"],
        module_outcome=module_outcome,
        candidate_recommendation=payload["candidate_recommendation"],
        aggregate_identity=_digest(payload),
    )
    aggregate.validate()
    return aggregate


__all__ = [
    "QkRatioProbeAggregate",
    "QkRatioProbeObservation",
    "QkRgb8QualityDelta",
    "QkSynchronizationDiagnosisAggregate",
    "QkSynchronizationWriteMetricError",
    "QkTerminalFailure",
    "QkTransformedRelationObservation",
    "aggregate_qk_ratio_probes",
    "aggregate_qk_synchronization_diagnosis",
    "create_qk_ratio_probe_observation",
    "create_qk_rgb8_quality_delta",
    "create_qk_transformed_relation_observation",
]
