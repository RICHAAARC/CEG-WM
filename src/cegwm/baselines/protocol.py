"""Frozen Baseline-V1 formal protocol and exact clean-confirmation gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from cegwm.baselines.records import BaselineObservation, validate_observation


ONE_SIDED_CONFIDENCE = 0.95
TARGET_FPR_UPPER_BOUND = 0.001
THRESHOLD_FREEZE_NEGATIVES = 2_000
CLEAN_CONFIRMATION_NEGATIVES = 3_000
EVALUATION_PHYSICAL_UNITS = 1_000
OBSERVATION_TIMEOUT_SECONDS = 20 * 60


@dataclass(frozen=True)
class AttackCondition:
    family: str
    condition: str
    parameters: tuple[tuple[str, str], ...]


FORMAL_ATTACK_CONDITIONS = (
    AttackCondition("clean", "clean_no_attack", ()),
    AttackCondition("compression", "jpeg_q50", (("quality", "50"),)),
    AttackCondition("geometric", "resize_50_bicubic_restore", (("scale", "0.50"), ("restore", "bicubic"))),
    AttackCondition("geometric", "center_crop_80_restore", (("retained_area", "0.80"), ("restore", "resize"))),
    AttackCondition("photometric", "gaussian_blur_sigma_1px", (("sigma_px", "1.0"),)),
    AttackCondition("geometric", "rotation_10deg", (("degrees", "10"), ("fill_crop_policy", "pending_user_freeze"))),
)


@dataclass(frozen=True)
class PerMethodScale:
    threshold_freeze_detections: int
    clean_confirmation_detections: int
    evaluation_detections: int
    source_generation_images: int
    attack_derivative_images: int
    quality_pair_comparisons: int


def per_method_scale() -> PerMethodScale:
    """Return fixed planned counts; no run is implied."""

    condition_count = len(FORMAL_ATTACK_CONDITIONS)
    return PerMethodScale(
        THRESHOLD_FREEZE_NEGATIVES,
        CLEAN_CONFIRMATION_NEGATIVES,
        condition_count * EVALUATION_PHYSICAL_UNITS * 2,
        THRESHOLD_FREEZE_NEGATIVES + CLEAN_CONFIRMATION_NEGATIVES + 2 * EVALUATION_PHYSICAL_UNITS,
        (condition_count - 1) * EVALUATION_PHYSICAL_UNITS * 2,
        condition_count * EVALUATION_PHYSICAL_UNITS,
    )


def rotation_execution_blocker() -> str | None:
    """Expose the one unresolved attack semantic before execution."""

    if dict(FORMAL_ATTACK_CONDITIONS[-1].parameters)["fill_crop_policy"] == "pending_user_freeze":
        return "rotation fill/crop policy requires user freeze before attack execution"
    return None


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / max(d, tiny)
    h = d
    for step in range(1, 201):
        m, m2 = float(step), 2.0 * step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / max(1.0 + aa * d, tiny)
        c = max(1.0 + aa / c, tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / max(1.0 + aa * d, tiny)
        c = max(1.0 + aa / c, tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise RuntimeError("incomplete-beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _inverse_regularized_beta(probability: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(120):
        midpoint = (low + high) / 2.0
        if _regularized_beta(midpoint, a, b) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def one_sided_clopper_pearson_upper(false_positives: int, negatives: int, *, confidence: float = ONE_SIDED_CONFIDENCE) -> float:
    """Exact upper limit: BetaInv(confidence; false_positives+1, negatives-false_positives)."""

    if negatives <= 0 or false_positives < 0 or false_positives > negatives:
        raise ValueError("false-positive count must be within a positive negative denominator")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if false_positives == negatives:
        return 1.0
    if false_positives == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / negatives)
    return _inverse_regularized_beta(confidence, false_positives + 1.0, negatives - false_positives)


@dataclass(frozen=True)
class ConfirmationGateResult:
    planned_negatives: int
    observed_negatives: int
    failure_count: int
    false_positives: int
    observed_fpr: float | None
    upper_bound: float | None
    operating_point_violation: bool | None


def operating_point_violation(false_positives: int, negatives: int) -> bool:
    """Predeclared observed-rate flag; it does not authorize threshold retuning."""

    if negatives <= 0 or false_positives < 0 or false_positives > negatives:
        raise ValueError("false-positive count must be within a positive negative denominator")
    return false_positives / negatives > TARGET_FPR_UPPER_BOUND


def evaluate_clean_confirmation(records: Iterable[BaselineObservation]) -> ConfirmationGateResult:
    """Apply the independent clean-confirmation gate with no threshold feedback."""

    items = tuple(validate_observation(record) for record in records)
    if len(items) != CLEAN_CONFIRMATION_NEGATIVES:
        raise ValueError("clean confirmation requires exactly the frozen planned denominator")
    if any(item.protocol_partition != "clean_confirmation" or item.sample_role != "confirmation_unwatermarked_negative"
           for item in items):
        raise ValueError("clean confirmation records must be confirmation unwatermarked negatives")
    if any(item.attack_family != "clean" or item.attack_condition != "clean_no_attack" for item in items):
        raise ValueError("clean confirmation cannot include attacks")
    failures = sum(item.status == "failed" for item in items)
    observed = tuple(item for item in items if item.status == "confirmation_observed")
    if failures or len(observed) != CLEAN_CONFIRMATION_NEGATIVES:
        return ConfirmationGateResult(CLEAN_CONFIRMATION_NEGATIVES, len(observed), failures, 0, None, None, None)
    false_positives = sum(item.decision for item in observed)
    upper_bound = one_sided_clopper_pearson_upper(false_positives, CLEAN_CONFIRMATION_NEGATIVES)
    observed_fpr = false_positives / CLEAN_CONFIRMATION_NEGATIVES
    return ConfirmationGateResult(CLEAN_CONFIRMATION_NEGATIVES, len(observed), 0, false_positives, observed_fpr,
                                  upper_bound, operating_point_violation(false_positives, CLEAN_CONFIRMATION_NEGATIVES))
