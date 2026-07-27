"""Independent fail-closed conjunction for geometry reliability."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from main.shared.key_schedule import stable_json_utf8

from .transform_estimator import (
    GeometricTransformEstimation,
    GeometricTransformEstimatorError,
    validate_geometric_transform_estimation,
)

QK_CANDIDATE_ID = "qk_relation_similarity"
RECTIFICATION_CANDIDATE_ID = "rectification_similarity"
MINIMUM_COVERAGE = 0.45


class GeometryReliabilityError(ValueError):
    """Reliability thresholds or estimator input violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class GeometryReliabilityThresholds:
    """Thresholds fitted only by the independent geometry-reliability role."""

    gamma_coverage: float
    gamma_uniqueness: float
    gamma_gap: float
    gamma_key: float
    gamma_inlier: float
    gamma_residual: float
    gamma_identity: float
    epsilon_inlier: float
    fit_identity: str


@dataclass(frozen=True, slots=True)
class GeometryReliabilityResult:
    """Rectification permission or an explicit fail-closed status."""

    reliable: bool
    allow_rectification: bool
    status: str
    failure_reasons: tuple[str, ...]
    fitted_reliability_thresholds: GeometryReliabilityThresholds | None
    threshold_config_digest: str | None
    estimator_search_config_digest: str
    estimation_identity_digest: str
    registered_root_key_public_digest: str
    reliability_identity_digest: str


@dataclass(frozen=True, slots=True)
class _ReliabilityDecision:
    reliable: bool
    allow_rectification: bool
    status: str
    failure_reasons: tuple[str, ...]


def _validated_thresholds(
    thresholds: object,
) -> GeometryReliabilityThresholds:
    if type(thresholds) is not GeometryReliabilityThresholds:
        raise GeometryReliabilityError(
            "thresholds must be GeometryReliabilityThresholds"
        )
    values = (
        thresholds.gamma_coverage,
        thresholds.gamma_uniqueness,
        thresholds.gamma_gap,
        thresholds.gamma_key,
        thresholds.gamma_inlier,
        thresholds.gamma_residual,
        thresholds.gamma_identity,
        thresholds.epsilon_inlier,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        for value in values
    ):
        raise GeometryReliabilityError("all fitted thresholds must be finite numbers")
    if not 0.0 <= float(thresholds.gamma_coverage) <= 1.0:
        raise GeometryReliabilityError("gamma_coverage must be in [0,1]")
    if not 0.0 <= float(thresholds.gamma_uniqueness) <= 1.0:
        raise GeometryReliabilityError("gamma_uniqueness must be in [0,1]")
    if not 0.0 <= float(thresholds.gamma_inlier) <= 1.0:
        raise GeometryReliabilityError("gamma_inlier must be in [0,1]")
    if float(thresholds.gamma_residual) < 0.0:
        raise GeometryReliabilityError("gamma_residual must be non-negative")
    if float(thresholds.epsilon_inlier) <= 0.0:
        raise GeometryReliabilityError("epsilon_inlier must be positive")
    if type(thresholds.fit_identity) is not str or not thresholds.fit_identity:
        raise GeometryReliabilityError("fit_identity must be a non-empty string")
    return thresholds


def _threshold_identity(
    thresholds: GeometryReliabilityThresholds,
) -> dict[str, str]:
    return {
        "epsilon_inlier_decimal": format(thresholds.epsilon_inlier, ".17g"),
        "fit_identity": thresholds.fit_identity,
        "gamma_coverage_decimal": format(thresholds.gamma_coverage, ".17g"),
        "gamma_gap_decimal": format(thresholds.gamma_gap, ".17g"),
        "gamma_identity_decimal": format(thresholds.gamma_identity, ".17g"),
        "gamma_inlier_decimal": format(thresholds.gamma_inlier, ".17g"),
        "gamma_key_decimal": format(thresholds.gamma_key, ".17g"),
        "gamma_residual_decimal": format(thresholds.gamma_residual, ".17g"),
        "gamma_uniqueness_decimal": format(thresholds.gamma_uniqueness, ".17g"),
    }


def _threshold_digest(thresholds: GeometryReliabilityThresholds) -> str:
    identity = {
        "candidate_ids": [
            "key_schedule_sha256_counter",
            QK_CANDIDATE_ID,
            RECTIFICATION_CANDIDATE_ID,
        ],
        "rule": "fitted_geometry_reliability_threshold_conjunction",
        "thresholds": _threshold_identity(thresholds),
    }
    return sha256(stable_json_utf8(identity)).hexdigest()


def _all_estimator_metrics_finite(
    estimation: GeometricTransformEstimation,
) -> bool:
    scalar_metrics = (
        estimation.registered_objective,
        estimation.second_registered_objective,
        estimation.exact_identity_objective,
        estimation.canonical_score,
        estimation.observation_score,
        estimation.coverage_forward,
        estimation.coverage_backward,
        estimation.uniqueness_forward,
        estimation.uniqueness_backward,
        estimation.coverage,
        estimation.uniqueness,
        estimation.gap,
        estimation.identity_margin,
        estimation.key_margin,
        estimation.inlier_ratio,
        estimation.mean_residual,
        estimation.epsilon_inlier,
        *estimation.wrong_key_objectives,
        *estimation.anchor_residuals,
        *(
            value
            for row in estimation.transform.matrix
            for value in row
        ),
    )
    return all(isfinite(float(value)) for value in scalar_metrics)


def _estimator_metrics_in_domain(
    estimation: GeometricTransformEstimation,
) -> bool:
    unit_interval_metrics = (
        estimation.coverage_forward,
        estimation.coverage_backward,
        estimation.uniqueness_forward,
        estimation.uniqueness_backward,
        estimation.coverage,
        estimation.uniqueness,
        estimation.inlier_ratio,
    )
    nonnegative_metrics = (
        estimation.gap,
        estimation.identity_margin,
        estimation.mean_residual,
        *estimation.anchor_residuals,
    )
    return (
        all(0.0 <= float(value) <= 1.0 for value in unit_interval_metrics)
        and all(float(value) >= 0.0 for value in nonnegative_metrics)
        and float(estimation.epsilon_inlier) > 0.0
    )


def _validated_estimation(
    estimation: object,
) -> GeometricTransformEstimation:
    if type(estimation) is not GeometricTransformEstimation:
        raise GeometryReliabilityError(
            "estimation must be GeometricTransformEstimation"
        )
    try:
        validate_geometric_transform_estimation(estimation)
    except GeometricTransformEstimatorError as exc:
        raise GeometryReliabilityError(
            "transform estimation identity validation failed"
        ) from exc
    return estimation


def _replay_reliability_decision(
    estimation: GeometricTransformEstimation,
    thresholds: GeometryReliabilityThresholds | None,
) -> _ReliabilityDecision:
    if thresholds is None:
        return _ReliabilityDecision(
            reliable=False,
            allow_rectification=False,
            status="reliability_not_fitted",
            failure_reasons=("reliability_not_fitted",),
        )
    reasons: list[str] = []
    if thresholds.epsilon_inlier != estimation.epsilon_inlier:
        reasons.append("epsilon_inlier_identity_mismatch")
    metrics_finite = _all_estimator_metrics_finite(estimation)
    if not metrics_finite:
        reasons.append("nonfinite_geometry_metric")
    elif not _estimator_metrics_in_domain(estimation):
        reasons.append("geometry_metric_outside_domain")
    if estimation.coverage < max(MINIMUM_COVERAGE, thresholds.gamma_coverage):
        reasons.append("coverage_below_threshold")
    if estimation.uniqueness < thresholds.gamma_uniqueness:
        reasons.append("uniqueness_below_threshold")
    if estimation.gap < thresholds.gamma_gap:
        reasons.append("registered_candidate_gap_below_threshold")
    if estimation.key_margin < thresholds.gamma_key:
        reasons.append("wrong_key_margin_below_threshold")
    if estimation.inlier_ratio < thresholds.gamma_inlier:
        reasons.append("inlier_ratio_below_threshold")
    if estimation.mean_residual > thresholds.gamma_residual:
        reasons.append("residual_above_threshold")
    if estimation.transform.continuous_parameter_on_search_boundary:
        reasons.append("continuous_parameter_on_search_boundary")
    if (
        not estimation.transform.is_exact_identity
        and estimation.identity_margin < thresholds.gamma_identity
    ):
        reasons.append("identity_margin_below_threshold")

    reliable = not reasons
    return _ReliabilityDecision(
        reliable=reliable,
        allow_rectification=reliable,
        status="reliable" if reliable else "unreliable",
        failure_reasons=tuple(reasons),
    )


def _reliability_identity_digest(
    *,
    decision: _ReliabilityDecision,
    fitted_thresholds: GeometryReliabilityThresholds | None,
    threshold_config_digest: str | None,
    estimator_search_config_digest: str,
    estimation_identity_digest: str,
    registered_root_key_public_digest: str,
) -> str:
    identity = {
        "allow_rectification": decision.allow_rectification,
        "estimation_identity_digest": estimation_identity_digest,
        "estimator_search_config_digest": estimator_search_config_digest,
        "failure_reasons": list(decision.failure_reasons),
        "fitted_reliability_thresholds": (
            None
            if fitted_thresholds is None
            else _threshold_identity(fitted_thresholds)
        ),
        "registered_root_key_public_digest": (
            registered_root_key_public_digest
        ),
        "reliable": decision.reliable,
        "status": decision.status,
        "threshold_config_digest": threshold_config_digest,
    }
    return sha256(stable_json_utf8(identity)).hexdigest()


def _build_reliability_result(
    estimation: GeometricTransformEstimation,
    fitted_thresholds: GeometryReliabilityThresholds | None,
    decision: _ReliabilityDecision,
) -> GeometryReliabilityResult:
    threshold_config_digest = (
        None
        if fitted_thresholds is None
        else _threshold_digest(fitted_thresholds)
    )
    reliability_identity_digest = _reliability_identity_digest(
        decision=decision,
        fitted_thresholds=fitted_thresholds,
        threshold_config_digest=threshold_config_digest,
        estimator_search_config_digest=estimation.search_config_digest,
        estimation_identity_digest=estimation.estimation_identity_digest,
        registered_root_key_public_digest=(
            estimation.registered_root_key_public_digest
        ),
    )
    return GeometryReliabilityResult(
        reliable=decision.reliable,
        allow_rectification=decision.allow_rectification,
        status=decision.status,
        failure_reasons=decision.failure_reasons,
        fitted_reliability_thresholds=fitted_thresholds,
        threshold_config_digest=threshold_config_digest,
        estimator_search_config_digest=estimation.search_config_digest,
        estimation_identity_digest=estimation.estimation_identity_digest,
        registered_root_key_public_digest=(
            estimation.registered_root_key_public_digest
        ),
        reliability_identity_digest=reliability_identity_digest,
    )


def geometry_reliability(
    estimation: GeometricTransformEstimation,
    thresholds: GeometryReliabilityThresholds | None = None,
) -> GeometryReliabilityResult:
    """Apply the frozen conjunction without reading any content statistic."""

    validated_estimation = _validated_estimation(estimation)
    fitted_thresholds = (
        None if thresholds is None else _validated_thresholds(thresholds)
    )
    decision = _replay_reliability_decision(
        validated_estimation,
        fitted_thresholds,
    )
    return _build_reliability_result(
        validated_estimation,
        fitted_thresholds,
        decision,
    )


def validate_geometry_reliability_result(
    result: GeometryReliabilityResult,
    estimation: GeometricTransformEstimation,
) -> bool:
    """Replay and validate a reliability result against one estimation."""

    validated_estimation = _validated_estimation(estimation)
    if type(result) is not GeometryReliabilityResult:
        raise GeometryReliabilityError(
            "result must be GeometryReliabilityResult"
        )
    if (
        type(result.reliable) is not bool
        or type(result.allow_rectification) is not bool
        or type(result.status) is not str
        or not result.status
        or type(result.failure_reasons) is not tuple
        or any(
            type(reason) is not str or not reason
            for reason in result.failure_reasons
        )
    ):
        raise GeometryReliabilityError(
            "geometry reliability result decision fields are invalid"
        )
    fitted_thresholds = result.fitted_reliability_thresholds
    if fitted_thresholds is None:
        if result.threshold_config_digest is not None:
            raise GeometryReliabilityError(
                "unfitted reliability result must not bind a threshold digest"
            )
    else:
        fitted_thresholds = _validated_thresholds(fitted_thresholds)
        if result.threshold_config_digest != _threshold_digest(
            fitted_thresholds
        ):
            raise GeometryReliabilityError(
                "geometry reliability threshold digest mismatch"
            )
    if (
        result.estimator_search_config_digest
        != validated_estimation.search_config_digest
        or result.estimation_identity_digest
        != validated_estimation.estimation_identity_digest
        or result.registered_root_key_public_digest
        != validated_estimation.registered_root_key_public_digest
    ):
        raise GeometryReliabilityError(
            "geometry reliability result estimation binding mismatch"
        )
    replayed_decision = _replay_reliability_decision(
        validated_estimation,
        fitted_thresholds,
    )
    supplied_decision = _ReliabilityDecision(
        reliable=result.reliable,
        allow_rectification=result.allow_rectification,
        status=result.status,
        failure_reasons=result.failure_reasons,
    )
    if supplied_decision != replayed_decision:
        raise GeometryReliabilityError(
            "geometry reliability decision replay mismatch"
        )
    expected_identity_digest = _reliability_identity_digest(
        decision=replayed_decision,
        fitted_thresholds=fitted_thresholds,
        threshold_config_digest=result.threshold_config_digest,
        estimator_search_config_digest=(
            result.estimator_search_config_digest
        ),
        estimation_identity_digest=result.estimation_identity_digest,
        registered_root_key_public_digest=(
            result.registered_root_key_public_digest
        ),
    )
    if result.reliability_identity_digest != expected_identity_digest:
        raise GeometryReliabilityError(
            "geometry reliability identity digest mismatch"
        )
    return replayed_decision.reliable


__all__ = [
    "GeometryReliabilityError",
    "GeometryReliabilityResult",
    "GeometryReliabilityThresholds",
    "geometry_reliability",
    "validate_geometry_reliability_result",
]
