"""Minimal Geometry-V7 R3 boundary and D4 cycle method iteration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Sequence

from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS


R3_TAU = 0.0
R3_B_LOW = -3.09856202452356
R3_B_LOW_HEX = "-0x1.8c9dae2ffa661p+1"
R3_THRESHOLD_GRID_PX = (1, 2, 4, 6, 8)
R3_NOT_PROBED_INELIGIBLE = "R3_NOT_PROBED_INELIGIBLE"
R3_METHOD_IMPROVED = "R3_METHOD_IMPROVED"
R3_METHOD_NOT_IMPROVED = "R3_METHOD_NOT_IMPROVED"
R3_OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE"
R3_CLAIM_CEILING = "engineering_method_iteration_on_existing_observed_data"


@dataclass(frozen=True, slots=True)
class R3Unit:
    split: str
    condition_id: str
    unit_id: str
    s0: float | None
    r2_selector_accepted: bool
    outcome_complete: bool
    safe: bool
    safe_rescue: bool
    baseline_positive: bool
    post_positive: bool | None
    observed_negative_false_positive: bool | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class D4BranchRecord:
    transform: str
    probed: bool
    geometry: object | None
    cycle_normalized: float | None
    cycle_pixels: float | None
    errors: tuple[str, ...] = ()
    d_matrix: object | None = None
    h_probe: object | None = None
    expected_inverse_d: object | None = None


@dataclass(frozen=True, slots=True)
class CycleFeatureRow:
    split: str
    condition_id: str
    unit_id: str
    route: str
    r2_selector_accepted: bool
    feature_valid: bool
    identity_cycle_px: float | None
    d4_median_cycle_px: float | None
    cycle_score_px: float | None
    valid_d4_count: int
    branches: tuple[D4BranchRecord, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ThresholdAttackMetrics:
    condition_id: str
    baseline_accepted_count: int
    baseline_unsafe_count: int
    baseline_safe_rescue_count: int
    baseline_negative_control_fp_count: int
    accepted_count: int
    unsafe_accept_count: int
    safe_rescue_count: int
    selected_negative_control_fp_count: int


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    split: str
    threshold_px: float
    fixed_denominator: int
    boundary_count: int
    baseline_accepted_count: int
    baseline_coverage: float
    baseline_unsafe_count: int
    baseline_selective_risk: float | None
    baseline_negative_control_fp_count: int
    baseline_safe_rescue_count: int
    baseline_net_rescue_change: int
    accepted_count: int
    coverage: float
    unsafe_accept_count: int
    selective_risk: float | None
    selected_negative_control_fp_count: int
    safe_rescue_count: int
    net_rescue_change: int
    correct_reject_unsafe_count: int
    wrong_reject_safe_count: int
    covered_attack_count: int
    per_attack: tuple[ThresholdAttackMetrics, ...]
    translation_summary: tuple[ThresholdAttackMetrics, ...]
    usable: bool


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    status: str
    selected_threshold_px: float | None
    selected_metrics: ThresholdMetrics | None
    grid_metrics: tuple[ThresholdMetrics, ...]


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def route_from_s0(s0: object) -> str:
    value = _finite(s0)
    if value is None:
        return "INVALID_S0"
    if value > R3_TAU:
        return "DIRECT_POSITIVE"
    if value <= R3_B_LOW:
        return "DIRECT_NEGATIVE"
    return "BOUNDARY"


def cycle_feature_row(
    *, unit: R3Unit, branches: Sequence[D4BranchRecord], d4_order: Sequence[str],
) -> CycleFeatureRow:
    records = tuple(branches)
    if len(records) != 8 or tuple(item.transform for item in records) != tuple(d4_order):
        raise ValueError("R3 D4 branches must have exact frozen order")
    route = route_from_s0(unit.s0)
    eligible = route == "BOUNDARY" and unit.r2_selector_accepted
    if not eligible:
        if any(item.probed for item in records):
            raise ValueError("ineligible R3 unit must not invoke D4 detector")
        reason = "BOUNDARY_R2_REJECTED" if route == "BOUNDARY" else route
        return CycleFeatureRow(
            unit.split, unit.condition_id, unit.unit_id, route,
            unit.r2_selector_accepted, False, None, None, None, 0, records,
            (f"{R3_NOT_PROBED_INELIGIBLE}:{reason}",),
        )
    valid = tuple(
        item for item in records
        if item.probed and not item.errors and _finite(item.cycle_pixels) is not None
    )
    identity = records[0]
    feature_valid = (
        identity.probed and not identity.errors
        and _finite(identity.cycle_pixels) is not None and len(valid) >= 6
    )
    values = tuple(float(item.cycle_pixels) for item in valid if item.cycle_pixels is not None)
    identity_px = float(identity.cycle_pixels) if feature_valid and identity.cycle_pixels is not None else None
    median_px = float(median(values)) if feature_valid else None
    score = max(identity_px, median_px) if identity_px is not None and median_px is not None else None
    return CycleFeatureRow(
        unit.split, unit.condition_id, unit.unit_id, route,
        unit.r2_selector_accepted, feature_valid, identity_px, median_px, score,
        len(valid), records, () if feature_valid else ("cycle_feature_invalid",),
    )


def _validate(rows: Sequence[CycleFeatureRow], units: Sequence[R3Unit], split: str):
    rows, units = tuple(rows), tuple(units)
    roster = R2_DEV_UNIT_IDS if split == "dev" else R2_TEST_UNIT_IDS
    expected = tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in roster)
    if (
        len(rows) != 40 or len(units) != 40
        or tuple((item.condition_id, item.unit_id) for item in rows) != expected
        or tuple((item.condition_id, item.unit_id) for item in units) != expected
        or any(item.split != split for item in (*rows, *units))
    ):
        raise ValueError(f"R3 {split} requires exact ordered fixed 40")
    return rows, units


def evaluate_threshold(
    threshold_px: float, rows: Sequence[CycleFeatureRow], units: Sequence[R3Unit], *, split: str,
) -> ThresholdMetrics:
    if threshold_px not in R3_THRESHOLD_GRID_PX:
        raise ValueError("R3 threshold must come from fixed pixel grid")
    rows, units = _validate(rows, units, split)
    baseline = tuple(
        (row, unit) for row, unit in zip(rows, units, strict=True)
        if row.route == "BOUNDARY" and row.r2_selector_accepted
    )
    accepted = tuple(
        (row, unit) for row, unit in baseline
        if row.feature_valid and row.cycle_score_px is not None
        and row.cycle_score_px <= threshold_px
    )
    baseline_unsafe = sum(not unit.safe for _, unit in baseline)
    baseline_rescue = sum(unit.safe_rescue for _, unit in baseline)
    unsafe = sum(not unit.safe for _, unit in accepted)
    rescue = sum(unit.safe_rescue for _, unit in accepted)
    fp = sum(unit.observed_negative_false_positive is True for _, unit in accepted)
    per_attack = []
    for condition in R2_CONDITION_IDS:
        base = tuple((row, unit) for row, unit in baseline if row.condition_id == condition)
        chosen = tuple((row, unit) for row, unit in accepted if row.condition_id == condition)
        per_attack.append(ThresholdAttackMetrics(
            condition, len(base), sum(not unit.safe for _, unit in base),
            sum(unit.safe_rescue for _, unit in base),
            sum(unit.observed_negative_false_positive is True for _, unit in base),
            len(chosen), sum(not unit.safe for _, unit in chosen),
            sum(unit.safe_rescue for _, unit in chosen),
            sum(unit.observed_negative_false_positive is True for _, unit in chosen),
        ))
    accepted_count = len(accepted)
    net_rescue = sum(unit.post_positive is True for _, unit in accepted) - sum(
        unit.baseline_positive for _, unit in accepted
    )
    baseline_net_rescue = sum(unit.post_positive is True for _, unit in baseline) - sum(
        unit.baseline_positive for _, unit in baseline
    )
    usable = (
        accepted_count > 0 and fp == 0 and unsafe < baseline_unsafe
        and 4 * rescue >= 3 * baseline_rescue
    )
    translations = tuple(item for item in per_attack if "translation" in item.condition_id)
    return ThresholdMetrics(
        split, threshold_px, 40,
        sum(row.route == "BOUNDARY" for row in rows), len(baseline), len(baseline) / 40.0,
        baseline_unsafe, None if not baseline else baseline_unsafe / len(baseline),
        sum(unit.observed_negative_false_positive is True for _, unit in baseline),
        baseline_rescue, baseline_net_rescue, accepted_count,
        accepted_count / 40.0, unsafe,
        None if accepted_count == 0 else unsafe / accepted_count,
        fp, rescue, net_rescue, baseline_unsafe - unsafe,
        sum(unit.safe for _, unit in baseline) - sum(unit.safe for _, unit in accepted),
        sum(item.accepted_count > 0 for item in per_attack), tuple(per_attack),
        translations, usable,
    )


def select_threshold(
    dev_rows: Sequence[CycleFeatureRow], dev_units: Sequence[R3Unit],
) -> ThresholdSelection:
    table = tuple(
        evaluate_threshold(threshold, dev_rows, dev_units, split="dev")
        for threshold in R3_THRESHOLD_GRID_PX
    )
    usable = tuple(item for item in table if item.usable)
    if not usable:
        return ThresholdSelection(R3_METHOD_NOT_IMPROVED, None, None, table)
    selected = min(usable, key=lambda item: (
        -item.safe_rescue_count, -item.accepted_count,
        item.unsafe_accept_count, item.threshold_px,
    ))
    return ThresholdSelection(R3_METHOD_IMPROVED, selected.threshold_px, selected, table)


def evaluate_selected_threshold(
    threshold_px: float, test_rows: Sequence[CycleFeatureRow], test_units: Sequence[R3Unit],
) -> ThresholdMetrics:
    """One engineering diagnostic on the existing observed R2 test40."""

    return evaluate_threshold(threshold_px, test_rows, test_units, split="test")


__all__ = [name for name in globals() if name.startswith("R3_") or name in {
    "R3Unit", "D4BranchRecord", "CycleFeatureRow", "ThresholdAttackMetrics",
    "ThresholdMetrics", "ThresholdSelection", "route_from_s0", "cycle_feature_row",
    "evaluate_threshold", "select_threshold", "evaluate_selected_threshold",
}]
