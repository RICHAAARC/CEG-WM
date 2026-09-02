"""Frozen Geometry-V7 R1B truth-utility and under-correction contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any, Mapping, Sequence

from PIL import Image

from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    Matrix3x3,
    homography_observed_to_canonical,
)
from cegwm.geometry_v7.r0 import ContentScore, PairedContentDecision, R0Arm
from cegwm.geometry_v7.r1a import (
    R1A_CORE_CONDITIONS,
    R1AConditionSpec,
    apply_homography,
    corner_rmse,
    truth_correspondences,
)
from cegwm.runtime.observation import require_ordinary_rgb_image


R1B_FIXED_UNIT_COUNT = 8
R1B_BOUNDARY_RATIO = 0.25
R1B_LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
R1B_MIN_GAIN_FRACTION = 0.75
R1B_MIN_NEGATIVE_RECOVERY_FRACTION = 0.5
R1B_TRUTH_UTILITY_AND_NONZERO_EPSILON_PASSED = (
    "R1B_TRUTH_UTILITY_AND_NONZERO_EPSILON_PASSED"
)
R1B_TRUTH_UTILITY_FAILED = "R1B_TRUTH_UTILITY_FAILED"
R1B_ZERO_ONLY_TOLERANCE_FAILED = "R1B_ZERO_ONLY_TOLERANCE_FAILED"
R1B_INSUFFICIENT_GEOMETRY_NECESSITY = "R1B_INSUFFICIENT_GEOMETRY_NECESSITY"
R1B_OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"
R1B_CLAIM_CEILING = (
    "small_sample_truth_utility_and_under_correction_tolerance_canary_only"
)
R1B_REPAIR_PIXEL_GRID = (0, 1, 2, 4, 6, 8)
R1B_REPAIR_CLAIM_CEILING = (
    "small_sample_real_syncseal_recovery_and_directional_pixel_tolerance_canary_only"
)
R1B_REPAIR_REAL_ALL_CORE_PASSED = "ALL_CORE_PASSED"
R1B_REPAIR_REAL_PARTIAL_CORE_PASSED = "PARTIAL_CORE_PASSED"
R1B_REPAIR_REAL_NO_CORE_PASSED = "NO_CORE_PASSED"
R1B_REPAIR_FINE_ALL_CORE_NONZERO_PREFIX = "ALL_CORE_NONZERO_PREFIX"
R1B_REPAIR_FINE_PARTIAL_CORE_NONZERO_PREFIX = "PARTIAL_CORE_NONZERO_PREFIX"
R1B_REPAIR_FINE_ZERO_ONLY_ALL_CORE = "ZERO_ONLY_ALL_CORE"
R1B_REPAIR_METHOD_PASSED = "R1B_REPAIR_REAL_H_END_TO_END_PASSED"
R1B_REPAIR_METHOD_NOT_READY = "R1B_REPAIR_REAL_H_NOT_END_TO_END_READY"


class R1BMembership(str, Enum):
    RECOVERY_NEGATIVE = "N_recovery_negative"
    BOUNDARY = "B_boundary"
    DAMAGE_ONLY = "D_damage_only"


@dataclass(frozen=True, slots=True)
class R1BScoredTriplet:
    """Raw U/G/CG blind scores and their two frozen paired decisions."""

    u: ContentScore
    g: ContentScore
    cg: ContentScore
    positive_cg_vs_g: PairedContentDecision
    negative_g_vs_u: PairedContentDecision


@dataclass(frozen=True, slots=True)
class R1BPreUnitRecord:
    unit_id: str
    condition_id: str
    clean_score: float
    scores: R1BScoredTriplet | None
    membership: R1BMembership | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R1BLambdaUnitRecord:
    unit_id: str
    condition_id: str
    lambda_value: float
    epsilon_normalized: float
    epsilon_pixels: float
    scores: R1BScoredTriplet | None
    positive_gate_a_delta: float | None
    positive_gate_b_delta: float | None
    positive_score_delta: float | None
    gain: float | None
    improved: bool | None
    recovered_negative: bool | None
    decision_harm: bool | None
    observed_negative_false_positive: bool | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R1BLambdaAggregate:
    condition_id: str
    lambda_value: float
    epsilon_normalized: float
    epsilon_pixels: float
    roster: tuple[str, ...]
    eligible_roster: tuple[str, ...]
    recovery_negative_roster: tuple[str, ...]
    damage_only_roster: tuple[str, ...]
    eligible_denominator: int
    valid_gain_count: int
    improved_count: int
    required_improved_count: int
    full_eligible_gain_median: float | None
    missing_gain_sentinel_count: int
    recovery_negative_count: int | None
    required_recovery_negative_count: int | None
    damage_harm_count: int
    observed_negative_false_positive_count: int
    observed_negative_denominator: int
    passed: bool


@dataclass(frozen=True, slots=True)
class R1BConditionEvaluation:
    condition_id: str
    roster: tuple[str, ...]
    eligible_roster: tuple[str, ...]
    damage_only_roster: tuple[str, ...]
    applicable: bool
    eligibility_status: str
    lambda_aggregates: tuple[R1BLambdaAggregate, ...]
    accepted_lambda: float | None
    accepted_epsilon_normalized: float | None
    accepted_epsilon_pixels: float | None
    truth_utility_passed: bool | None
    nonzero_epsilon_passed: bool | None


@dataclass(frozen=True, slots=True)
class R1BEvaluation:
    status: str
    conditions: tuple[R1BConditionEvaluation, ...]
    applicable_condition_count: int
    blocking_method_canary_passed: bool | None


@dataclass(frozen=True, slots=True)
class R1BStoredPrediction:
    unit_id: str
    condition_id: str
    truth_correspondences: tuple[tuple[float, float], ...]
    predicted_correspondences: tuple[tuple[float, float], ...] | None
    predicted_h_observed_to_canonical: Matrix3x3 | None
    prediction_rmse_normalized: float | None
    prediction_rmse_pixels: float | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R1BRepairPointRecord:
    unit_id: str
    condition_id: str
    point_kind: str
    radius_pixels: int | None
    scores: R1BScoredTriplet | None
    positive_gate_a_delta: float | None
    positive_gate_b_delta: float | None
    positive_score_delta: float | None
    improved: bool | None
    recovered_negative: bool | None
    decision_harm: bool | None
    observed_negative_false_positive: bool | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R1BRepairPointAggregate:
    condition_id: str
    point_kind: str
    radius_pixels: int | None
    roster: tuple[str, ...]
    eligible_roster: tuple[str, ...]
    recovery_negative_roster: tuple[str, ...]
    damage_only_roster: tuple[str, ...]
    eligible_denominator: int
    valid_gain_count: int
    improved_count: int
    required_improved_count: int
    full_eligible_gain_median: float | None
    missing_gain_sentinel_count: int
    recovery_negative_count: int | None
    required_recovery_negative_count: int | None
    damage_harm_count: int
    observed_negative_false_positive_count: int
    observed_negative_denominator: int
    passed: bool


@dataclass(frozen=True, slots=True)
class R1BRepairConditionEvaluation:
    condition_id: str
    roster: tuple[str, ...]
    eligible_roster: tuple[str, ...]
    damage_only_roster: tuple[str, ...]
    eligibility_status: str
    real_h_aggregate: R1BRepairPointAggregate
    fine_grid_aggregates: tuple[R1BRepairPointAggregate, ...]
    accepted_max_pixels: int | None
    real_h_passed: bool
    fine_nonzero_prefix_passed: bool


@dataclass(frozen=True, slots=True)
class R1BRepairEvaluation:
    status: str
    real_h_status: str
    fine_grid_status: str
    conditions: tuple[R1BRepairConditionEvaluation, ...]
    real_h_passed_condition_count: int
    fine_nonzero_prefix_condition_count: int
    r2_candidate: bool


def paired_content_decision(
    candidate: ContentScore,
    paired_null: ContentScore,
    paired_null_arm: R0Arm,
) -> PairedContentDecision:
    """Apply the unchanged strict Gate A/Gate B paired decision."""

    if not isinstance(candidate, ContentScore) or not isinstance(
        paired_null, ContentScore
    ):
        raise TypeError("R1B paired scores require frozen ContentScore values")
    gate_a = candidate.gate_a_margin
    gate_b = candidate.weighted_joint - paired_null.weighted_joint
    return PairedContentDecision(
        paired_null_arm,
        gate_a,
        gate_b,
        min(gate_a, gate_b),
        gate_a > 0.0 and gate_b > 0.0,
    )


def scored_triplet(
    *, u: ContentScore, g: ContentScore, cg: ContentScore
) -> R1BScoredTriplet:
    """Bind positive CG-vs-G and negative G-vs-U without a C arm."""

    return R1BScoredTriplet(
        u,
        g,
        cg,
        paired_content_decision(cg, g, R0Arm.G),
        paired_content_decision(g, u, R0Arm.U),
    )


def freeze_pre_recovery_record(
    *,
    unit_id: str,
    spec: R1AConditionSpec,
    clean_score: float,
    scores: R1BScoredTriplet | None,
    errors: Sequence[str] = (),
) -> R1BPreUnitRecord:
    if not isinstance(unit_id, str) or not unit_id:
        raise ValueError("R1B unit id must be nonempty")
    if spec not in R1A_CORE_CONDITIONS:
        raise ValueError("R1B requires one exact frozen R1A core condition")
    clean = float(clean_score)
    if not math.isfinite(clean) or clean <= 0.0:
        raise ValueError("accepted R0 clean CG-vs-G score must be finite and positive")
    recorded_errors = tuple(str(error) for error in errors)
    if scores is None:
        if not recorded_errors:
            recorded_errors = ("pre_content_score:missing",)
        membership = None
    elif recorded_errors:
        membership = None
    else:
        score = scores.positive_cg_vs_g.margin
        if score <= 0.0:
            membership = R1BMembership.RECOVERY_NEGATIVE
        elif score <= R1B_BOUNDARY_RATIO * clean:
            membership = R1BMembership.BOUNDARY
        else:
            membership = R1BMembership.DAMAGE_ONLY
    return R1BPreUnitRecord(
        unit_id,
        spec.condition_id,
        clean,
        scores,
        membership,
        recorded_errors,
    )


def _matrix_multiply(left: Matrix3x3, right: Matrix3x3) -> Matrix3x3:
    return tuple(
        tuple(
            math.fsum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _matrix_inverse(matrix: Matrix3x3) -> Matrix3x3:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    cofactors = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    determinant = a * cofactors[0][0] + b * cofactors[1][0] + c * cofactors[2][0]
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("R1B homography must be finite and invertible")
    result = tuple(tuple(value / determinant for value in row) for row in cofactors)
    if any(not math.isfinite(value) for row in result for value in row):
        raise ValueError("R1B inverse homography must be finite")
    return result


def controlled_correspondences(
    spec: R1AConditionSpec, lambda_value: float
) -> tuple[tuple[float, float], ...]:
    """Interpolate only along the frozen truth-to-identity point path."""

    if spec not in R1A_CORE_CONDITIONS:
        raise ValueError("R1B controlled path requires one exact core condition")
    value = float(lambda_value)
    if value not in R1B_LAMBDA_GRID:
        raise ValueError("R1B lambda must lie on the exact frozen grid")
    truth = truth_correspondences(spec)
    return tuple(
        tuple((1.0 - value) * truth_axis + value * identity_axis for truth_axis, identity_axis in zip(truth_point, identity_point, strict=True))
        for truth_point, identity_point in zip(
            truth, CANONICAL_CORNERS_NORMALIZED, strict=True
        )
    )


def controlled_homography(
    spec: R1AConditionSpec, lambda_value: float
) -> Matrix3x3:
    return homography_observed_to_canonical(
        controlled_correspondences(spec, lambda_value)
    )


def epsilon_for_lambda(
    spec: R1AConditionSpec, lambda_value: float
) -> tuple[float, float]:
    value = float(lambda_value)
    if value not in R1B_LAMBDA_GRID:
        raise ValueError("R1B lambda must lie on the exact frozen grid")
    identity_error = corner_rmse(
        CANONICAL_CORNERS_NORMALIZED, truth_correspondences(spec)
    )
    normalized = value * identity_error
    return normalized, normalized * 511.0 / 2.0


def _pixel_output_to_source(
    observed_to_canonical: Matrix3x3,
) -> tuple[float, ...]:
    pixel_to_normalized: Matrix3x3 = (
        (2.0 / 511.0, 0.0, -1.0),
        (0.0, 2.0 / 511.0, -1.0),
        (0.0, 0.0, 1.0),
    )
    normalized_to_pixel: Matrix3x3 = (
        (511.0 / 2.0, 0.0, 511.0 / 2.0),
        (0.0, 511.0 / 2.0, 511.0 / 2.0),
        (0.0, 0.0, 1.0),
    )
    canonical_to_observed = _matrix_inverse(observed_to_canonical)
    mapping = _matrix_multiply(
        normalized_to_pixel,
        _matrix_multiply(canonical_to_observed, pixel_to_normalized),
    )
    scale = mapping[2][2]
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("R1B pixel sampler must be finite")
    normalized = tuple(tuple(value / scale for value in row) for row in mapping)
    return (
        normalized[0][0],
        normalized[0][1],
        normalized[0][2],
        normalized[1][0],
        normalized[1][1],
        normalized[1][2],
        normalized[2][0],
        normalized[2][1],
    )


def rectify_attacked_rgb(
    attacked_rgb: Any, observed_to_canonical: Matrix3x3
) -> Image.Image:
    """Sample attacked observed RGB once into the canonical 512 canvas."""

    source = require_ordinary_rgb_image(attacked_rgb)
    if source.size != (512, 512):
        raise ValueError("R1B rectification requires ordinary RGB 512x512")
    return source.transform(
        (512, 512),
        Image.Transform.PERSPECTIVE,
        _pixel_output_to_source(observed_to_canonical),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def evaluate_lambda_unit(
    *,
    pre_record: R1BPreUnitRecord,
    spec: R1AConditionSpec,
    lambda_value: float,
    scores: R1BScoredTriplet | None,
    errors: Sequence[str] = (),
) -> R1BLambdaUnitRecord:
    if pre_record.condition_id != spec.condition_id:
        raise ValueError("R1B pre/lambda condition identity differs")
    value = float(lambda_value)
    epsilon_normalized, epsilon_pixels = epsilon_for_lambda(spec, value)
    recorded_errors = tuple(str(error) for error in errors)
    gate_a_delta = gate_b_delta = score_delta = None
    gain = improved = recovered = harm = false_positive = None
    if scores is None:
        if not recorded_errors:
            recorded_errors = ("rectified_content_score:missing",)
    elif not recorded_errors and pre_record.scores is not None:
        gate_a_delta = (
            scores.positive_cg_vs_g.gate_a_margin
            - pre_record.scores.positive_cg_vs_g.gate_a_margin
        )
        gate_b_delta = (
            scores.positive_cg_vs_g.gate_b_margin
            - pre_record.scores.positive_cg_vs_g.gate_b_margin
        )
        score_delta = (
            scores.positive_cg_vs_g.margin
            - pre_record.scores.positive_cg_vs_g.margin
        )
        if any(
            not math.isfinite(value)
            for value in (gate_a_delta, gate_b_delta, score_delta)
        ):
            gate_a_delta = gate_b_delta = score_delta = gain = None
            recorded_errors = ("rectified_content_score:nonfinite_gain",)
        else:
            gain = score_delta
            improved = gain > 0.0
            if pre_record.membership is R1BMembership.RECOVERY_NEGATIVE:
                recovered = scores.positive_cg_vs_g.margin > 0.0
            if pre_record.membership is R1BMembership.DAMAGE_ONLY:
                harm = not scores.positive_cg_vs_g.positive
            false_positive = scores.negative_g_vs_u.positive
    return R1BLambdaUnitRecord(
        pre_record.unit_id,
        spec.condition_id,
        value,
        epsilon_normalized,
        epsilon_pixels,
        scores,
        gate_a_delta,
        gate_b_delta,
        score_delta,
        gain,
        improved,
        recovered,
        harm,
        false_positive,
        recorded_errors,
    )


def _validate_roster(
    pre_records: Sequence[R1BPreUnitRecord], ordered_roster: Sequence[str]
) -> tuple[tuple[R1BPreUnitRecord, ...], tuple[str, ...]]:
    records = tuple(pre_records)
    roster = tuple(ordered_roster)
    if (
        len(roster) != R1B_FIXED_UNIT_COUNT
        or len(set(roster)) != R1B_FIXED_UNIT_COUNT
        or len(records) != R1B_FIXED_UNIT_COUNT
        or tuple(record.unit_id for record in records) != roster
        or len({record.condition_id for record in records}) != 1
    ):
        raise ValueError("R1B requires one fixed ordered eight-unit condition roster")
    return records, roster


def aggregate_lambda(
    *,
    spec: R1AConditionSpec,
    pre_records: Sequence[R1BPreUnitRecord],
    lambda_records: Sequence[R1BLambdaUnitRecord],
    ordered_roster: Sequence[str],
) -> R1BLambdaAggregate:
    pre, roster = _validate_roster(pre_records, ordered_roster)
    records = tuple(lambda_records)
    if (
        spec not in R1A_CORE_CONDITIONS
        or any(record.condition_id != spec.condition_id for record in pre)
        or len(records) != R1B_FIXED_UNIT_COUNT
        or tuple(record.unit_id for record in records) != roster
        or len({record.lambda_value for record in records}) != 1
        or any(record.condition_id != spec.condition_id for record in records)
    ):
        raise ValueError("R1B lambda records differ from the frozen condition/roster")
    value = records[0].lambda_value
    if value not in R1B_LAMBDA_GRID:
        raise ValueError("R1B lambda records differ from the frozen grid")
    eligible = tuple(
        record.unit_id
        for record in pre
        if record.membership in (
            R1BMembership.RECOVERY_NEGATIVE,
            R1BMembership.BOUNDARY,
        )
    )
    recovery_negative = tuple(
        record.unit_id
        for record in pre
        if record.membership is R1BMembership.RECOVERY_NEGATIVE
    )
    damage = tuple(
        record.unit_id
        for record in pre
        if record.membership is R1BMembership.DAMAGE_ONLY
    )
    by_unit = {record.unit_id: record for record in records}
    gain_values = tuple(
        by_unit[unit_id].gain
        if not by_unit[unit_id].errors
        and isinstance(by_unit[unit_id].gain, float)
        and math.isfinite(by_unit[unit_id].gain)
        else -math.inf
        for unit_id in eligible
    )
    missing_gain_count = sum(not math.isfinite(value) for value in gain_values)
    median_with_sentinel = float(median(gain_values)) if gain_values else None
    public_median = (
        median_with_sentinel
        if median_with_sentinel is not None and math.isfinite(median_with_sentinel)
        else None
    )
    valid_gain_count = len(gain_values) - missing_gain_count
    improved_count = sum(value > 0.0 for value in gain_values)
    required_improved = math.ceil(R1B_MIN_GAIN_FRACTION * len(eligible))
    gain_gate = bool(
        eligible
        and improved_count >= required_improved
        and median_with_sentinel is not None
        and median_with_sentinel > 0.0
    )
    if recovery_negative:
        recovery_count = sum(
            by_unit[unit_id].recovered_negative is True
            and not by_unit[unit_id].errors
            for unit_id in recovery_negative
        )
        required_recovery = math.ceil(
            R1B_MIN_NEGATIVE_RECOVERY_FRACTION * len(recovery_negative)
        )
        recovery_gate = recovery_count >= required_recovery
    else:
        recovery_count = required_recovery = None
        recovery_gate = True
    harm_count = sum(
        by_unit[unit_id].decision_harm is True for unit_id in damage
    )
    damage_gate = all(
        not by_unit[unit_id].errors
        and by_unit[unit_id].decision_harm is False
        for unit_id in damage
    )
    false_positive_count = sum(
        record.observed_negative_false_positive is True for record in records
    )
    negative_gate = all(
        not record.errors
        and record.observed_negative_false_positive is False
        for record in records
    )
    epsilon_normalized, epsilon_pixels = epsilon_for_lambda(spec, value)
    return R1BLambdaAggregate(
        spec.condition_id,
        value,
        epsilon_normalized,
        epsilon_pixels,
        roster,
        eligible,
        recovery_negative,
        damage,
        len(eligible),
        valid_gain_count,
        improved_count,
        required_improved,
        public_median,
        missing_gain_count,
        recovery_count,
        required_recovery,
        harm_count,
        false_positive_count,
        R1B_FIXED_UNIT_COUNT,
        gain_gate and recovery_gate and damage_gate and negative_gate,
    )


def evaluate_condition(
    *,
    spec: R1AConditionSpec,
    pre_records: Sequence[R1BPreUnitRecord],
    lambda_records: Mapping[float, Sequence[R1BLambdaUnitRecord]],
    ordered_roster: Sequence[str],
) -> R1BConditionEvaluation:
    pre, roster = _validate_roster(pre_records, ordered_roster)
    if spec not in R1A_CORE_CONDITIONS or any(
        record.condition_id != spec.condition_id for record in pre
    ):
        raise ValueError("R1B condition identity differs")
    if any(record.errors or record.membership is None for record in pre):
        raise ValueError("R1B membership must be frozen before method evaluation")
    eligible = tuple(
        record.unit_id
        for record in pre
        if record.membership in (
            R1BMembership.RECOVERY_NEGATIVE,
            R1BMembership.BOUNDARY,
        )
    )
    damage = tuple(
        record.unit_id
        for record in pre
        if record.membership is R1BMembership.DAMAGE_ONLY
    )
    if not eligible:
        if lambda_records:
            raise ValueError("non-applicable R1B condition must not be rectified")
        return R1BConditionEvaluation(
            spec.condition_id,
            roster,
            eligible,
            damage,
            False,
            "NOT_APPLICABLE/INSUFFICIENT_ELIGIBLE",
            (),
            None,
            None,
            None,
            None,
            None,
        )
    if tuple(lambda_records) != R1B_LAMBDA_GRID:
        raise ValueError("applicable R1B condition requires the full ordered lambda grid")
    aggregates = tuple(
        aggregate_lambda(
            spec=spec,
            pre_records=pre,
            lambda_records=lambda_records[value],
            ordered_roster=roster,
        )
        for value in R1B_LAMBDA_GRID
    )
    accepted = None
    for aggregate in aggregates:
        if not aggregate.passed:
            break
        accepted = aggregate
    return R1BConditionEvaluation(
        spec.condition_id,
        roster,
        eligible,
        damage,
        True,
        "APPLICABLE",
        aggregates,
        None if accepted is None else accepted.lambda_value,
        None if accepted is None else accepted.epsilon_normalized,
        None if accepted is None else accepted.epsilon_pixels,
        aggregates[0].passed,
        accepted is not None and accepted.lambda_value >= R1B_LAMBDA_GRID[1],
    )


def evaluate_r1b(
    *,
    pre_records_by_condition: Mapping[str, Sequence[R1BPreUnitRecord]],
    lambda_records_by_condition: Mapping[
        str, Mapping[float, Sequence[R1BLambdaUnitRecord]]
    ],
    ordered_roster: Sequence[str],
) -> R1BEvaluation:
    expected = tuple(spec.condition_id for spec in R1A_CORE_CONDITIONS)
    if tuple(pre_records_by_condition) != expected:
        raise ValueError("R1B requires all ten core pre-score conditions in order")
    conditions = tuple(
        evaluate_condition(
            spec=spec,
            pre_records=pre_records_by_condition[spec.condition_id],
            lambda_records=lambda_records_by_condition.get(spec.condition_id, {}),
            ordered_roster=ordered_roster,
        )
        for spec in R1A_CORE_CONDITIONS
    )
    applicable = tuple(condition for condition in conditions if condition.applicable)
    if not applicable:
        return R1BEvaluation(
            R1B_INSUFFICIENT_GEOMETRY_NECESSITY,
            conditions,
            0,
            None,
        )
    if any(condition.truth_utility_passed is not True for condition in applicable):
        return R1BEvaluation(
            R1B_TRUTH_UTILITY_FAILED,
            conditions,
            len(applicable),
            False,
        )
    if any(condition.nonzero_epsilon_passed is not True for condition in applicable):
        return R1BEvaluation(
            R1B_ZERO_ONLY_TOLERANCE_FAILED,
            conditions,
            len(applicable),
            False,
        )
    return R1BEvaluation(
        R1B_TRUTH_UTILITY_AND_NONZERO_EPSILON_PASSED,
        conditions,
        len(applicable),
        True,
    )


def directional_correspondences_for_pixels(
    prediction: R1BStoredPrediction,
    radius_pixels: int,
) -> tuple[tuple[float, float], ...]:
    """Scale or extrapolate along the stored R1A predicted-error direction."""

    if not isinstance(prediction, R1BStoredPrediction):
        raise TypeError("R1B repair direction requires a stored R1A prediction")
    if radius_pixels not in R1B_REPAIR_PIXEL_GRID:
        raise ValueError("R1B repair radius must lie on the exact pixel grid")
    truth = prediction.truth_correspondences
    if radius_pixels == 0:
        return truth
    if (
        prediction.errors
        or prediction.predicted_correspondences is None
        or prediction.prediction_rmse_normalized is None
        or prediction.prediction_rmse_pixels is None
        or not math.isfinite(prediction.prediction_rmse_normalized)
        or not math.isfinite(prediction.prediction_rmse_pixels)
        or prediction.prediction_rmse_normalized <= 0.0
        or prediction.prediction_rmse_pixels <= 0.0
    ):
        raise ValueError("stored predicted-error direction is unavailable")
    expected_pixels = prediction.prediction_rmse_normalized * 511.0 / 2.0
    if not math.isclose(
        prediction.prediction_rmse_pixels,
        expected_pixels,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("stored predicted-error pixel conversion differs")
    scale = radius_pixels / expected_pixels
    return tuple(
        tuple(
            truth_axis + scale * (predicted_axis - truth_axis)
            for truth_axis, predicted_axis in zip(
                truth_point, predicted_point, strict=True
            )
        )
        for truth_point, predicted_point in zip(
            truth, prediction.predicted_correspondences, strict=True
        )
    )


def directional_homography_for_pixels(
    prediction: R1BStoredPrediction,
    radius_pixels: int,
) -> Matrix3x3:
    return homography_observed_to_canonical(
        directional_correspondences_for_pixels(prediction, radius_pixels)
    )


def evaluate_repair_point_unit(
    *,
    pre_record: R1BPreUnitRecord,
    point_kind: str,
    radius_pixels: int | None,
    scores: R1BScoredTriplet | None,
    errors: Sequence[str] = (),
) -> R1BRepairPointRecord:
    if point_kind not in {"real_h", "directional_pixel"}:
        raise ValueError("R1B repair point kind differs")
    if point_kind == "real_h":
        if radius_pixels is not None:
            raise ValueError("real-H point has no directional radius")
    elif radius_pixels not in R1B_REPAIR_PIXEL_GRID:
        raise ValueError("directional repair point radius differs")
    recorded_errors = tuple(str(error) for error in errors)
    gate_a_delta = gate_b_delta = score_delta = None
    improved = recovered = harm = false_positive = None
    if scores is None:
        if not recorded_errors:
            recorded_errors = ("repair_score:missing",)
    elif not recorded_errors and pre_record.scores is not None:
        gate_a_delta = (
            scores.positive_cg_vs_g.gate_a_margin
            - pre_record.scores.positive_cg_vs_g.gate_a_margin
        )
        gate_b_delta = (
            scores.positive_cg_vs_g.gate_b_margin
            - pre_record.scores.positive_cg_vs_g.gate_b_margin
        )
        score_delta = (
            scores.positive_cg_vs_g.margin
            - pre_record.scores.positive_cg_vs_g.margin
        )
        if any(
            not math.isfinite(value)
            for value in (gate_a_delta, gate_b_delta, score_delta)
        ):
            gate_a_delta = gate_b_delta = score_delta = None
            recorded_errors = ("repair_score:nonfinite_delta",)
        else:
            improved = score_delta > 0.0
            if pre_record.membership is R1BMembership.RECOVERY_NEGATIVE:
                recovered = scores.positive_cg_vs_g.margin > 0.0
            if pre_record.membership is R1BMembership.DAMAGE_ONLY:
                harm = not scores.positive_cg_vs_g.positive
            false_positive = scores.negative_g_vs_u.positive
    return R1BRepairPointRecord(
        pre_record.unit_id,
        pre_record.condition_id,
        point_kind,
        radius_pixels,
        scores,
        gate_a_delta,
        gate_b_delta,
        score_delta,
        improved,
        recovered,
        harm,
        false_positive,
        recorded_errors,
    )


def aggregate_repair_point(
    *,
    pre_records: Sequence[R1BPreUnitRecord],
    point_records: Sequence[R1BRepairPointRecord],
    ordered_roster: Sequence[str],
) -> R1BRepairPointAggregate:
    pre, roster = _validate_roster(pre_records, ordered_roster)
    records = tuple(point_records)
    if (
        len(records) != R1B_FIXED_UNIT_COUNT
        or tuple(record.unit_id for record in records) != roster
        or len({record.condition_id for record in records}) != 1
        or records[0].condition_id != pre[0].condition_id
        or len({record.point_kind for record in records}) != 1
        or len({record.radius_pixels for record in records}) != 1
    ):
        raise ValueError("R1B repair point records differ from the fixed roster")
    point_kind = records[0].point_kind
    radius_pixels = records[0].radius_pixels
    eligible = tuple(
        record.unit_id
        for record in pre
        if record.membership in (
            R1BMembership.RECOVERY_NEGATIVE,
            R1BMembership.BOUNDARY,
        )
    )
    recovery_negative = tuple(
        record.unit_id
        for record in pre
        if record.membership is R1BMembership.RECOVERY_NEGATIVE
    )
    damage = tuple(
        record.unit_id
        for record in pre
        if record.membership is R1BMembership.DAMAGE_ONLY
    )
    by_unit = {record.unit_id: record for record in records}
    gains = tuple(
        by_unit[unit_id].positive_score_delta
        if not by_unit[unit_id].errors
        and isinstance(by_unit[unit_id].positive_score_delta, float)
        and math.isfinite(by_unit[unit_id].positive_score_delta)
        else -math.inf
        for unit_id in eligible
    )
    missing = sum(not math.isfinite(value) for value in gains)
    median_with_sentinel = float(median(gains)) if gains else None
    public_median = (
        median_with_sentinel
        if median_with_sentinel is not None and math.isfinite(median_with_sentinel)
        else None
    )
    improved_count = sum(value > 0.0 for value in gains)
    required_improved = math.ceil(R1B_MIN_GAIN_FRACTION * len(eligible))
    gain_gate = bool(
        eligible
        and improved_count >= required_improved
        and median_with_sentinel is not None
        and median_with_sentinel > 0.0
    )
    if recovery_negative:
        recovered_count = sum(
            not by_unit[unit_id].errors
            and by_unit[unit_id].recovered_negative is True
            for unit_id in recovery_negative
        )
        required_recovered = math.ceil(
            R1B_MIN_NEGATIVE_RECOVERY_FRACTION * len(recovery_negative)
        )
        recovery_gate = recovered_count >= required_recovered
    else:
        recovered_count = required_recovered = None
        recovery_gate = True
    harm_count = sum(
        by_unit[unit_id].decision_harm is True for unit_id in damage
    )
    damage_gate = all(
        not by_unit[unit_id].errors
        and by_unit[unit_id].decision_harm is False
        for unit_id in damage
    )
    false_positive_count = sum(
        record.observed_negative_false_positive is True for record in records
    )
    negative_gate = all(
        not record.errors
        and record.observed_negative_false_positive is False
        for record in records
    )
    return R1BRepairPointAggregate(
        pre[0].condition_id,
        point_kind,
        radius_pixels,
        roster,
        eligible,
        recovery_negative,
        damage,
        len(eligible),
        len(gains) - missing,
        improved_count,
        required_improved,
        public_median,
        missing,
        recovered_count,
        required_recovered,
        harm_count,
        false_positive_count,
        R1B_FIXED_UNIT_COUNT,
        gain_gate and recovery_gate and damage_gate and negative_gate,
    )


def evaluate_r1b_repair(
    *,
    pre_records_by_condition: Mapping[str, Sequence[R1BPreUnitRecord]],
    real_h_records_by_condition: Mapping[
        str, Sequence[R1BRepairPointRecord]
    ],
    fine_grid_records_by_condition: Mapping[
        str, Mapping[int, Sequence[R1BRepairPointRecord]]
    ],
    ordered_roster: Sequence[str],
) -> R1BRepairEvaluation:
    expected = tuple(spec.condition_id for spec in R1A_CORE_CONDITIONS)
    if (
        tuple(pre_records_by_condition) != expected
        or tuple(real_h_records_by_condition) != expected
    ):
        raise ValueError("R1B repair requires all ten core conditions in order")
    conditions: list[R1BRepairConditionEvaluation] = []
    for spec in R1A_CORE_CONDITIONS:
        pre, roster = _validate_roster(
            pre_records_by_condition[spec.condition_id], ordered_roster
        )
        if any(record.errors or record.membership is None for record in pre):
            raise ValueError("R1B repair requires complete frozen old membership")
        eligible = tuple(
            record.unit_id
            for record in pre
            if record.membership in (
                R1BMembership.RECOVERY_NEGATIVE,
                R1BMembership.BOUNDARY,
            )
        )
        damage = tuple(
            record.unit_id
            for record in pre
            if record.membership is R1BMembership.DAMAGE_ONLY
        )
        real = aggregate_repair_point(
            pre_records=pre,
            point_records=real_h_records_by_condition[spec.condition_id],
            ordered_roster=roster,
        )
        if eligible:
            grid = fine_grid_records_by_condition.get(spec.condition_id)
            if grid is None or tuple(grid) != R1B_REPAIR_PIXEL_GRID:
                raise ValueError(
                    "applicable R1B repair condition requires the full pixel grid"
                )
            fine = tuple(
                aggregate_repair_point(
                    pre_records=pre,
                    point_records=grid[radius],
                    ordered_roster=roster,
                )
                for radius in R1B_REPAIR_PIXEL_GRID
            )
            accepted = None
            for aggregate in fine:
                if not aggregate.passed:
                    break
                accepted = aggregate.radius_pixels
            eligibility_status = "APPLICABLE"
        else:
            if spec.condition_id in fine_grid_records_by_condition:
                raise ValueError(
                    "empty-E R1B repair condition must not create fine-grid records"
                )
            fine = ()
            accepted = None
            eligibility_status = "NOT_APPLICABLE/INSUFFICIENT_ELIGIBLE"
        conditions.append(
            R1BRepairConditionEvaluation(
                spec.condition_id,
                roster,
                eligible,
                damage,
                eligibility_status,
                real,
                fine,
                accepted,
                bool(eligible and real.passed),
                bool(eligible and accepted is not None and accepted >= 1),
            )
        )
    real_count = sum(condition.real_h_passed for condition in conditions)
    fine_count = sum(condition.fine_nonzero_prefix_passed for condition in conditions)
    if real_count == len(R1A_CORE_CONDITIONS):
        real_status = R1B_REPAIR_REAL_ALL_CORE_PASSED
    elif real_count:
        real_status = R1B_REPAIR_REAL_PARTIAL_CORE_PASSED
    else:
        real_status = R1B_REPAIR_REAL_NO_CORE_PASSED
    if fine_count == len(R1A_CORE_CONDITIONS):
        fine_status = R1B_REPAIR_FINE_ALL_CORE_NONZERO_PREFIX
    elif fine_count:
        fine_status = R1B_REPAIR_FINE_PARTIAL_CORE_NONZERO_PREFIX
    else:
        fine_status = R1B_REPAIR_FINE_ZERO_ONLY_ALL_CORE
    r2_candidate = real_status == R1B_REPAIR_REAL_ALL_CORE_PASSED
    return R1BRepairEvaluation(
        R1B_REPAIR_METHOD_PASSED if r2_candidate else R1B_REPAIR_METHOD_NOT_READY,
        real_status,
        fine_status,
        tuple(conditions),
        real_count,
        fine_count,
        r2_candidate,
    )


__all__ = [
    "R1B_BOUNDARY_RATIO",
    "R1B_CLAIM_CEILING",
    "R1B_FIXED_UNIT_COUNT",
    "R1B_INSUFFICIENT_GEOMETRY_NECESSITY",
    "R1B_LAMBDA_GRID",
    "R1B_OPERATIONAL_FAILURE",
    "R1B_TRUTH_UTILITY_AND_NONZERO_EPSILON_PASSED",
    "R1B_TRUTH_UTILITY_FAILED",
    "R1B_ZERO_ONLY_TOLERANCE_FAILED",
    "R1B_REPAIR_CLAIM_CEILING",
    "R1B_REPAIR_FINE_ALL_CORE_NONZERO_PREFIX",
    "R1B_REPAIR_FINE_PARTIAL_CORE_NONZERO_PREFIX",
    "R1B_REPAIR_FINE_ZERO_ONLY_ALL_CORE",
    "R1B_REPAIR_METHOD_NOT_READY",
    "R1B_REPAIR_METHOD_PASSED",
    "R1B_REPAIR_PIXEL_GRID",
    "R1B_REPAIR_REAL_ALL_CORE_PASSED",
    "R1B_REPAIR_REAL_NO_CORE_PASSED",
    "R1B_REPAIR_REAL_PARTIAL_CORE_PASSED",
    "R1BConditionEvaluation",
    "R1BEvaluation",
    "R1BLambdaAggregate",
    "R1BLambdaUnitRecord",
    "R1BMembership",
    "R1BPreUnitRecord",
    "R1BScoredTriplet",
    "R1BRepairConditionEvaluation",
    "R1BRepairEvaluation",
    "R1BRepairPointAggregate",
    "R1BRepairPointRecord",
    "R1BStoredPrediction",
    "aggregate_lambda",
    "controlled_correspondences",
    "controlled_homography",
    "directional_correspondences_for_pixels",
    "directional_homography_for_pixels",
    "epsilon_for_lambda",
    "evaluate_condition",
    "evaluate_lambda_unit",
    "evaluate_repair_point_unit",
    "evaluate_r1b",
    "evaluate_r1b_repair",
    "freeze_pre_recovery_record",
    "paired_content_decision",
    "aggregate_repair_point",
    "rectify_attacked_rgb",
    "scored_triplet",
]
