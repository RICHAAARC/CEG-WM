"""Frozen Geometry-V7 R3 advanced predicted-H runtime gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r3 import R3Unit


OLD_CYCLE_THRESHOLD_PX = 8.0
PURE_ROTATION_MIN_ABS_ANGLE_DEG = 10.0
PURE_ROTATION_MAX_ABS_ANGLE_DEG = 20.0
PURE_ROTATION_MIN_SCALE = 0.95
PURE_ROTATION_MAX_SCALE = 1.05
PURE_ROTATION_MAX_TRANSLATION = 0.02
PURE_ROTATION_MAX_PERSPECTIVE = 0.01
R3_ADVANCED_CLAIM_CEILING = "engineering_development_on_existing_observed_data_only"
R3_ADVANCED_R4_CANDIDATE = "R3_ADVANCED_EXISTING_TEST40_ACCEPTED_FOR_R4"
R3_ADVANCED_TRANSLATION_PARTIAL = "R3_ADVANCED_FULL_FAMILY_TRANSLATION_PARTIAL_SCOPE"
R3_ADVANCED_TEST40_RECORDED = "R3_ADVANCED_ENGINEERING_TEST40_RECORDED"
R3_ADVANCED_OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"


@dataclass(frozen=True, slots=True)
class PredictedHRegime:
    valid: bool
    angle_degrees: float | None
    scale: float | None
    translation: float | None
    perspective: float | None
    pure_rotation_gate: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AdvancedDecision:
    accepted: bool
    old_cycle_gate: bool
    pure_rotation_gate: bool
    regime: PredictedHRegime


@dataclass(frozen=True, slots=True)
class AdvancedRow:
    split: str
    condition_id: str
    unit_id: str
    route: str
    r2_selector_accepted: bool
    old_cycle_score_px: float | None
    homography_observed_to_canonical: object | None
    geometry_legal: bool
    geometry_error: str | None


@dataclass(frozen=True, slots=True)
class AdvancedAttackMetrics:
    condition_id: str
    baseline_accepted_count: int
    accepted_count: int
    safe_rescue_count: int
    unsafe_accept_count: int
    selected_negative_control_fp_count: int


@dataclass(frozen=True, slots=True)
class AdvancedMetrics:
    split: str
    fixed_denominator: int
    eligible_count: int
    baseline_accepted_count: int
    baseline_safe_rescue_count: int
    baseline_unsafe_accept_count: int
    baseline_negative_control_fp_count: int
    baseline_covered_attack_count: int
    accepted_count: int
    safe_rescue_count: int
    unsafe_accept_count: int
    selected_negative_control_fp_count: int
    covered_attack_count: int
    per_attack: tuple[AdvancedAttackMetrics, ...]
    translation_summary: tuple[AdvancedAttackMetrics, ...]


@dataclass(frozen=True, slots=True)
class OrientationDiagnostic:
    valid: bool
    best_transform: str | None
    best_residual_px: float | None
    second_residual_px: float | None
    second_minus_best_margin_px: float | None
    identity_residual_px: float | None
    identity_minus_best_residual_px: float | None
    identity_over_best_residual: float | None
    best_forward_cycle_px: float | None
    best_backward_cycle_px: float | None
    errors: tuple[str, ...] = ()


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("finite scalar required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("finite scalar required")
    return value


def _matrix3(value: object) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("homography must be 3x3")
    matrix = tuple(tuple(_finite(axis) for axis in row) for row in value)
    if any(len(row) != 3 for row in matrix):
        raise ValueError("homography must be 3x3")
    return matrix


def _det3(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _inverse3(m: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    determinant = _det3(m)
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("homography must be invertible")
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    adjugate = (
        (e*i-f*h, c*h-b*i, b*f-c*e),
        (f*g-d*i, a*i-c*g, c*d-a*f),
        (d*h-e*g, b*g-a*h, a*e-b*d),
    )
    return tuple(tuple(value / determinant for value in row) for row in adjugate)


def _product(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]):
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _corner_identity_rmse_px(matrix: Sequence[Sequence[float]]) -> float:
    points = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    squared = 0.0
    for x, y in points:
        denominator = matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]
        if not math.isfinite(denominator) or denominator == 0.0:
            raise ValueError("cycle projection is undefined")
        mapped_x = (matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]) / denominator
        mapped_y = (matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]) / denominator
        squared += (mapped_x-x)**2 + (mapped_y-y)**2
    return math.sqrt(squared / 8.0) * 511.0 / 2.0


def predicted_h_regime(
    homography_observed_to_canonical: object,
    *,
    geometry_legal: bool,
    geometry_error: str | None,
) -> PredictedHRegime:
    """Extract only inference-time H features; invalid geometry rejects closed."""

    try:
        if geometry_legal is not True or geometry_error is not None:
            raise ValueError("stored geometry is not legal")
        matrix = _matrix3(homography_observed_to_canonical)
        h22 = matrix[2][2]
        if h22 == 0.0:
            raise ValueError("homography h22 is zero")
        matrix = tuple(tuple(axis / h22 for axis in row) for row in matrix)
        _inverse3(matrix)
        angle = math.degrees(math.atan2(
            matrix[1][0] - matrix[0][1], matrix[0][0] + matrix[1][1]
        ))
        det_affine = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
        scale = math.sqrt(abs(det_affine))
        translation = math.hypot(matrix[0][2], matrix[1][2])
        perspective = math.hypot(matrix[2][0], matrix[2][1])
        values = (angle, scale, translation, perspective)
        if any(not math.isfinite(item) for item in values):
            raise ValueError("predicted-H regime is nonfinite")
        pure = (
            PURE_ROTATION_MIN_ABS_ANGLE_DEG <= abs(angle) <= PURE_ROTATION_MAX_ABS_ANGLE_DEG
            and PURE_ROTATION_MIN_SCALE <= scale <= PURE_ROTATION_MAX_SCALE
            and translation <= PURE_ROTATION_MAX_TRANSLATION
            and perspective <= PURE_ROTATION_MAX_PERSPECTIVE
        )
        return PredictedHRegime(True, angle, scale, translation, perspective, pure)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        return PredictedHRegime(False, None, None, None, None, False, (str(error),))


def advanced_runtime_decision(
    *,
    boundary: bool,
    r2_selector_accepted: bool,
    old_cycle_score_px: object,
    homography_observed_to_canonical: object,
    geometry_legal: bool,
    geometry_error: str | None,
) -> AdvancedDecision:
    """Frozen runtime predicate. It deliberately has no condition or outcome input."""

    regime = predicted_h_regime(
        homography_observed_to_canonical,
        geometry_legal=geometry_legal,
        geometry_error=geometry_error,
    )
    try:
        cycle = _finite(old_cycle_score_px)
        old_gate = cycle <= OLD_CYCLE_THRESHOLD_PX
    except ValueError:
        old_gate = False
    accepted = bool(
        boundary and r2_selector_accepted and regime.valid
        and (old_gate or regime.pure_rotation_gate)
    )
    return AdvancedDecision(accepted, old_gate, regime.pure_rotation_gate, regime)


def _validate(rows: Sequence[AdvancedRow], units: Sequence[R3Unit], split: str):
    rows, units = tuple(rows), tuple(units)
    roster = R2_DEV_UNIT_IDS if split == "dev" else R2_TEST_UNIT_IDS
    expected = tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in roster)
    if (
        len(rows) != 40 or len(units) != 40
        or tuple((row.condition_id, row.unit_id) for row in rows) != expected
        or tuple((unit.condition_id, unit.unit_id) for unit in units) != expected
        or any(row.split != split for row in rows)
        or any(unit.split != split for unit in units)
    ):
        raise ValueError(f"R3 advanced {split} requires exact ordered fixed 40")
    return rows, units


def evaluate_advanced(
    rows: Sequence[AdvancedRow], units: Sequence[R3Unit], *, split: str,
) -> AdvancedMetrics:
    rows, units = _validate(rows, units, split)
    baseline: list[tuple[AdvancedRow, R3Unit]] = []
    accepted: list[tuple[AdvancedRow, R3Unit]] = []
    for row, unit in zip(rows, units, strict=True):
        decision = advanced_runtime_decision(
            boundary=row.route == "BOUNDARY",
            r2_selector_accepted=row.r2_selector_accepted,
            old_cycle_score_px=row.old_cycle_score_px,
            homography_observed_to_canonical=row.homography_observed_to_canonical,
            geometry_legal=row.geometry_legal,
            geometry_error=row.geometry_error,
        )
        if row.route == "BOUNDARY" and row.r2_selector_accepted and decision.old_cycle_gate and decision.regime.valid:
            baseline.append((row, unit))
        if decision.accepted:
            accepted.append((row, unit))
    per_attack = []
    for condition in R2_CONDITION_IDS:
        old = tuple(item for item in baseline if item[0].condition_id == condition)
        new = tuple(item for item in accepted if item[0].condition_id == condition)
        per_attack.append(AdvancedAttackMetrics(
            condition, len(old), len(new), sum(unit.safe_rescue for _, unit in new),
            sum(not unit.safe for _, unit in new),
            sum(unit.observed_negative_false_positive is True for _, unit in new),
        ))
    return AdvancedMetrics(
        split, 40,
        sum(row.route == "BOUNDARY" and row.r2_selector_accepted for row in rows),
        len(baseline), sum(unit.safe_rescue for _, unit in baseline),
        sum(not unit.safe for _, unit in baseline),
        sum(unit.observed_negative_false_positive is True for _, unit in baseline),
        sum(item.baseline_accepted_count > 0 for item in per_attack),
        len(accepted), sum(unit.safe_rescue for _, unit in accepted),
        sum(not unit.safe for _, unit in accepted),
        sum(unit.observed_negative_false_positive is True for _, unit in accepted),
        sum(item.accepted_count > 0 for item in per_attack), tuple(per_attack),
        tuple(item for item in per_attack if "translation" in item.condition_id),
    )


def orientation_diagnostic(branches: Sequence[Mapping[str, object]]) -> OrientationDiagnostic:
    """Record-only orientation diagnostic; never consumed by the runtime gate."""

    try:
        branches = tuple(branches)
        if len(branches) != 8 or branches[0].get("transform") != "identity":
            raise ValueError("exact ordered D4 branches required")
        parsed = []
        for index, branch in enumerate(branches):
            geometry = branch.get("geometry")
            if not isinstance(geometry, Mapping) or geometry.get("legal") is not True or geometry.get("error") is not None:
                raise ValueError("D4 branch geometry invalid")
            h2 = _matrix3(geometry.get("homography_observed_to_canonical"))
            d_matrix = _matrix3(branch.get("d_matrix"))
            forward = _finite(branch.get("cycle_pixels"))
            backward = _corner_identity_rmse_px(_inverse3(_product(h2, d_matrix)))
            parsed.append((forward, index, str(branch.get("transform")), backward))
        ordered = sorted(parsed)
        best, second = ordered[0], ordered[1]
        identity = parsed[0][0]
        identity_over_best = None if best[0] == 0.0 else identity / best[0]
        return OrientationDiagnostic(
            True, best[2], best[0], second[0], second[0]-best[0], identity,
            identity-best[0], identity_over_best, best[0], best[3],
        )
    except (TypeError, ValueError, ZeroDivisionError) as error:
        return OrientationDiagnostic(
            False, None, None, None, None, None, None, None, None, None, (str(error),)
        )


__all__ = [name for name in globals() if name.startswith("R3_ADVANCED_") or name in {
    "OLD_CYCLE_THRESHOLD_PX", "PURE_ROTATION_MIN_ABS_ANGLE_DEG",
    "PURE_ROTATION_MAX_ABS_ANGLE_DEG", "PURE_ROTATION_MIN_SCALE",
    "PURE_ROTATION_MAX_SCALE", "PURE_ROTATION_MAX_TRANSLATION",
    "PURE_ROTATION_MAX_PERSPECTIVE", "PredictedHRegime", "AdvancedDecision",
    "AdvancedRow", "AdvancedAttackMetrics", "AdvancedMetrics", "OrientationDiagnostic",
    "predicted_h_regime", "advanced_runtime_decision", "evaluate_advanced",
    "orientation_diagnostic",
}]
