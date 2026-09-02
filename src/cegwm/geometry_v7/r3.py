"""Frozen development selector and D4 cycle diagnostic for Geometry-V7 R3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Sequence

from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS


R3_TAU = 0.0
R3_B_LOW_QUANTILES = (0.20, 0.40, 0.60, 0.80)
R3_DEV_NO_FEASIBLE_B_LOW = "R3_DEV_NO_FEASIBLE_B_LOW"
R3_DEV_B_LOW_FROZEN = "R3_DEV_B_LOW_FROZEN"
R3_OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"
R3_B_LOW_CLAIM_CEILING = "dev_only_engineering_boundary_only"
R3_CYCLE_FEATURE_ORDER = (
    "identity_cycle_px", "d4_median_cycle_px", "d4_max_cycle_px", "invalid_d4_count"
)
R3_CYCLE_QUANTILES = (0.20, 0.40, 0.60, 0.80)
R3_NOT_PROBED_INELIGIBLE = "R3_NOT_PROBED_INELIGIBLE"
R3_DEV_NO_FEASIBLE_CYCLE = "R3_DEV_NO_FEASIBLE_CYCLE_GATE"
R3_DEV_CYCLE_GATE_FROZEN = "R3_DEV_CYCLE_GATE_FROZEN"
R3_DEV_NOT_APPLICABLE = "NOT_APPLICABLE_NO_BASELINE_FALSE_RELIABLE"
R3_PASSED = "R3_EXPLORATORY_DIAGNOSTIC_PASSED"
R3_FAILED = "R3_EXPLORATORY_DIAGNOSTIC_FAILED"
R3_NOT_APPLICABLE = "R3_EXPLORATORY_DIAGNOSTIC_NOT_APPLICABLE"
R3_DIAGNOSTIC_OPERATIONAL = "R3_EXPLORATORY_DIAGNOSTIC_OPERATIONAL"
R3_CLAIM_CEILING = (
    "small_sample_r3_exploratory_cycle_diagnostic_on_previously_seen_r2_test_only"
)


@dataclass(frozen=True, slots=True)
class R3DevUnit:
    condition_id: str
    unit_id: str
    s0: float | None
    r2_selector_accepted: bool
    outcome_complete: bool
    safe: bool
    safe_rescue: bool
    observed_negative_false_positive: bool | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BLowCandidate:
    b_low: float
    source_quantiles: tuple[float, ...]

    @property
    def candidate_id(self) -> str:
        return f"BLOW|{self.b_low.hex()}"


@dataclass(frozen=True, slots=True)
class BLowAttackMetrics:
    condition_id: str
    boundary_count: int
    accepted_count: int
    unsafe_accept_count: int
    safe_rescue_count: int
    selected_negative_control_fp_count: int


@dataclass(frozen=True, slots=True)
class BLowMetrics:
    candidate_id: str
    b_low: float
    direct_positive_count: int
    direct_negative_count: int
    boundary_count: int
    invalid_no_recovery_count: int
    accepted_count: int
    unsafe_accept_count: int
    safe_rescue_count: int
    selected_negative_control_fp_count: int
    selected_negative_control_known_denominator: int
    covered_attack_count: int
    accepted_over_fixed_40: float
    accepted_over_boundary: float | None
    selective_risk: float | None
    per_attack: tuple[BLowAttackMetrics, ...]
    feasible: bool


@dataclass(frozen=True, slots=True)
class BLowSelection:
    status: str
    selected: BLowCandidate | None
    selected_metrics: BLowMetrics | None
    candidates: tuple[BLowCandidate, ...]
    candidate_table: tuple[BLowMetrics, ...]


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
    d4_max_cycle_px: float | None
    invalid_d4_count: int
    branches: tuple[D4BranchRecord, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CycleStump:
    feature: str
    threshold: float
    source_quantiles: tuple[float, ...]

    @property
    def candidate_id(self) -> str:
        return f"B|{self.feature}|le|{self.threshold.hex()}"


@dataclass(frozen=True, slots=True)
class CycleCandidate:
    candidate_id: str
    components: tuple[CycleStump, ...]

    @property
    def complexity(self) -> int:
        return len(self.components)


@dataclass(frozen=True, slots=True)
class CycleAttackMetrics:
    condition_id: str
    baseline_count: int
    feature_valid_count: int
    accepted_count: int
    unsafe_accept_count: int
    safe_rescue_count: int
    selected_negative_control_fp_count: int


@dataclass(frozen=True, slots=True)
class CycleMetrics:
    candidate_id: str
    split: str
    baseline_count: int
    boundary_eligible_count: int
    feature_valid_baseline_count: int
    accepted_count: int
    unsafe_accept_count: int
    safe_rescue_count: int
    selected_negative_control_fp_count: int
    covered_attack_count: int
    correct_reject_unsafe_count: int
    wrong_reject_safe_count: int
    accepted_over_fixed_40: float
    accepted_over_boundary_eligible: float | None
    accepted_over_baseline_accepted: float | None
    accepted_over_feature_valid_baseline: float | None
    selective_risk: float | None
    baseline_unsafe_count: int
    false_reliable_reduction_count: int
    per_attack: tuple[CycleAttackMetrics, ...]
    gates_passed: bool


@dataclass(frozen=True, slots=True)
class CycleSelection:
    status: str
    selected: CycleCandidate | None
    selected_metrics: CycleMetrics | None
    candidates: tuple[CycleCandidate, ...]
    candidate_table: tuple[CycleMetrics, ...]


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _validate_dev_units(units: Sequence[R3DevUnit]) -> tuple[R3DevUnit, ...]:
    received = tuple(units)
    expected = tuple(
        (condition_id, unit_id)
        for condition_id in R2_CONDITION_IDS
        for unit_id in R2_DEV_UNIT_IDS
    )
    if (
        len(received) != 40
        or tuple((unit.condition_id, unit.unit_id) for unit in received) != expected
        or len({(unit.condition_id, unit.unit_id) for unit in received}) != 40
    ):
        raise ValueError("R3 b_low selector requires exact ordered R2 development 40")
    if any(
        not isinstance(unit.r2_selector_accepted, bool)
        or not isinstance(unit.outcome_complete, bool)
        or not isinstance(unit.safe, bool)
        or not isinstance(unit.safe_rescue, bool)
        or unit.observed_negative_false_positive not in (True, False, None)
        or not isinstance(unit.errors, tuple)
        for unit in received
    ):
        raise ValueError("R3 development unit fields differ")
    return received


def _type7(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * quantile
    lower = math.floor(h)
    upper = math.ceil(h)
    return ordered[lower] + (h - lower) * (ordered[upper] - ordered[lower])


def generate_b_low_candidates(dev_units: Sequence[R3DevUnit]) -> tuple[BLowCandidate, ...]:
    units = _validate_dev_units(dev_units)
    negative = tuple(
        value for unit in units
        if (value := _finite(unit.s0)) is not None and value < R3_TAU
    )
    grouped: dict[str, tuple[float, list[float]]] = {}
    for quantile in R3_B_LOW_QUANTILES:
        if not negative:
            continue
        threshold = _type7(negative, quantile)
        if threshold >= R3_TAU:
            continue
        grouped.setdefault(threshold.hex(), (threshold, []))[1].append(quantile)
    return tuple(
        BLowCandidate(threshold, tuple(quantiles))
        for threshold, quantiles in sorted(
            grouped.values(), key=lambda item: (item[0], item[0].hex())
        )
    )


def route_for_b_low(unit: R3DevUnit, b_low: float) -> str:
    threshold = _finite(b_low)
    if threshold is None or threshold >= R3_TAU:
        raise ValueError("b_low must be finite and strictly below tau=0")
    s0 = _finite(unit.s0)
    if s0 is None:
        return "INVALID_S0"
    if s0 > R3_TAU:
        return "DIRECT_POSITIVE"
    if s0 <= threshold:
        return "DIRECT_NEGATIVE"
    return "BOUNDARY"


def evaluate_b_low_candidate(
    candidate: BLowCandidate,
    dev_units: Sequence[R3DevUnit],
) -> BLowMetrics:
    units = _validate_dev_units(dev_units)
    routes = tuple(route_for_b_low(unit, candidate.b_low) for unit in units)
    accepted_pairs = tuple(
        (unit, route)
        for unit, route in zip(units, routes, strict=True)
        if route == "BOUNDARY" and unit.r2_selector_accepted
    )
    accepted = len(accepted_pairs)
    unsafe = sum(not unit.safe for unit, _ in accepted_pairs)
    rescue = sum(unit.safe_rescue for unit, _ in accepted_pairs)
    fp = sum(unit.observed_negative_false_positive is True for unit, _ in accepted_pairs)
    fp_known = sum(
        isinstance(unit.observed_negative_false_positive, bool)
        for unit, _ in accepted_pairs
    )
    per_attack = []
    for condition_id in R2_CONDITION_IDS:
        condition_pairs = tuple(
            (unit, route)
            for unit, route in zip(units, routes, strict=True)
            if unit.condition_id == condition_id
        )
        selected = tuple(
            unit for unit, route in condition_pairs
            if route == "BOUNDARY" and unit.r2_selector_accepted
        )
        per_attack.append(
            BLowAttackMetrics(
                condition_id,
                sum(route == "BOUNDARY" for _, route in condition_pairs),
                len(selected),
                sum(not unit.safe for unit in selected),
                sum(unit.safe_rescue for unit in selected),
                sum(unit.observed_negative_false_positive is True for unit in selected),
            )
        )
    boundary = sum(route == "BOUNDARY" for route in routes)
    covered = sum(item.accepted_count >= 1 for item in per_attack)
    feasible = (
        boundary >= 10
        and accepted >= 10
        and unsafe * 10 <= accepted
        and fp == 0
        and rescue >= 8
        and covered >= 5
    )
    return BLowMetrics(
        candidate.candidate_id,
        candidate.b_low,
        sum(route == "DIRECT_POSITIVE" for route in routes),
        sum(route == "DIRECT_NEGATIVE" for route in routes),
        boundary,
        sum(route == "INVALID_S0" for route in routes),
        accepted,
        unsafe,
        rescue,
        fp,
        fp_known,
        covered,
        accepted / 40.0,
        None if boundary == 0 else accepted / boundary,
        None if accepted == 0 else unsafe / accepted,
        tuple(per_attack),
        feasible,
    )


def select_b_low(dev_units: Sequence[R3DevUnit]) -> BLowSelection:
    candidates = generate_b_low_candidates(dev_units)
    table = tuple(evaluate_b_low_candidate(candidate, dev_units) for candidate in candidates)
    feasible = tuple(item for item in table if item.feasible)
    if not feasible:
        return BLowSelection(R3_DEV_NO_FEASIBLE_B_LOW, None, None, candidates, table)
    selected_metrics = min(
        feasible,
        key=lambda item: (
            -item.safe_rescue_count,
            -item.accepted_count,
            item.unsafe_accept_count,
            -item.covered_attack_count,
            -item.b_low,
            item.candidate_id,
        ),
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return BLowSelection(
        R3_DEV_B_LOW_FROZEN,
        by_id[selected_metrics.candidate_id],
        selected_metrics,
        candidates,
        table,
    )


def cycle_feature_row(
    *, split: str, unit: R3DevUnit, route: str,
    branches: Sequence[D4BranchRecord], d4_order: Sequence[str],
) -> CycleFeatureRow:
    records = tuple(branches)
    order = tuple(d4_order)
    if len(order) != 8 or tuple(item.transform for item in records) != order:
        raise ValueError("R3 D4 branches must have exact frozen order")
    eligible = route == "BOUNDARY" and unit.r2_selector_accepted
    if not eligible:
        if any(item.probed for item in records):
            raise ValueError("ineligible R3 unit must not invoke D4 detector")
        if route == "BOUNDARY":
            reason = "BOUNDARY_R2_REJECTED"
        else:
            reason = route
        return CycleFeatureRow(
            split, unit.condition_id, unit.unit_id, route,
            unit.r2_selector_accepted, False, None, None, None, 8, records,
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
    return CycleFeatureRow(
        split, unit.condition_id, unit.unit_id, route, unit.r2_selector_accepted,
        feature_valid,
        float(identity.cycle_pixels) if feature_valid and identity.cycle_pixels is not None else None,
        float(median(values)) if feature_valid else None,
        float(max(values)) if feature_valid else None,
        8 - len(valid), records,
        () if feature_valid else ("cycle_feature_invalid",),
    )


def _validate_cycle_rows(
    rows: Sequence[CycleFeatureRow], split: str,
) -> tuple[CycleFeatureRow, ...]:
    received = tuple(rows)
    units = R2_DEV_UNIT_IDS if split == "dev" else tuple(
        f"content-v6-iss-eval-{index:04d}" for index in range(5, 9)
    )
    expected = tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in units)
    if (
        len(received) != 40
        or any(item.split != split for item in received)
        or tuple((item.condition_id, item.unit_id) for item in received) != expected
    ):
        raise ValueError(f"R3 cycle {split} rows require exact fixed ordered 40")
    return received


def _feature_valid(row: CycleFeatureRow) -> bool:
    return row.feature_valid and all(
        _finite(getattr(row, feature)) is not None for feature in R3_CYCLE_FEATURE_ORDER
    )


def generate_cycle_candidates(dev_rows: Sequence[CycleFeatureRow]) -> tuple[CycleCandidate, ...]:
    rows = _validate_cycle_rows(dev_rows, "dev")
    stumps: list[CycleStump] = []
    for feature in R3_CYCLE_FEATURE_ORDER:
        values = tuple(
            float(getattr(row, feature)) for row in rows
            if row.route == "BOUNDARY" and row.r2_selector_accepted and _feature_valid(row)
        )
        grouped: dict[str, tuple[float, list[float]]] = {}
        for quantile in R3_CYCLE_QUANTILES:
            if not values:
                continue
            threshold = _type7(values, quantile)
            grouped.setdefault(threshold.hex(), (threshold, []))[1].append(quantile)
        for threshold, quantiles in sorted(grouped.values(), key=lambda item: (item[0], item[0].hex())):
            stumps.append(CycleStump(feature, threshold, tuple(quantiles)))
    stumps.sort(key=lambda item: (
        R3_CYCLE_FEATURE_ORDER.index(item.feature), item.threshold, item.threshold.hex()
    ))
    candidates = [CycleCandidate("A|CYCLE_VALID_ONLY", ())]
    candidates.extend(CycleCandidate(stump.candidate_id, (stump,)) for stump in stumps)
    for index, first in enumerate(stumps):
        for second in stumps[index + 1:]:
            if first.feature == second.feature:
                continue
            candidates.append(CycleCandidate(
                f"C|{first.candidate_id}&{second.candidate_id}", (first, second)
            ))
    return tuple(candidates)


def _cycle_accepts(candidate: CycleCandidate, row: CycleFeatureRow) -> bool:
    if not (row.route == "BOUNDARY" and row.r2_selector_accepted and _feature_valid(row)):
        return False
    return all(float(getattr(row, item.feature)) <= item.threshold for item in candidate.components)


def evaluate_cycle_candidate(
    candidate: CycleCandidate, rows: Sequence[CycleFeatureRow],
    units: Sequence[R3DevUnit], *, split: str,
) -> CycleMetrics:
    records = _validate_cycle_rows(rows, split)
    expected_units = R2_DEV_UNIT_IDS if split == "dev" else tuple(
        f"content-v6-iss-eval-{index:04d}" for index in range(5, 9)
    )
    expected = tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in expected_units)
    outcomes = tuple(units)
    if tuple((item.condition_id, item.unit_id) for item in outcomes) != expected:
        raise ValueError(f"R3 cycle {split} outcomes require exact fixed ordered 40")
    boundary_eligible = tuple(
        (row, unit) for row, unit in zip(records, outcomes, strict=True)
        if row.route == "BOUNDARY"
    )
    baseline = tuple(
        (row, unit) for row, unit in zip(records, outcomes, strict=True)
        if row.route == "BOUNDARY" and row.r2_selector_accepted
    )
    accepted = tuple(
        (row, unit) for row, unit in zip(records, outcomes, strict=True)
        if _cycle_accepts(candidate, row)
    )
    baseline_unsafe = sum(not unit.safe for _, unit in baseline)
    unsafe = sum(not unit.safe for _, unit in accepted)
    rescue = sum(unit.safe_rescue for _, unit in accepted)
    fp = sum(unit.observed_negative_false_positive is True for _, unit in accepted)
    feature_valid_count = sum(_feature_valid(row) for row, _ in baseline)
    per_attack = []
    for condition in R2_CONDITION_IDS:
        base = tuple((row, unit) for row, unit in baseline if row.condition_id == condition)
        chosen = tuple((row, unit) for row, unit in accepted if row.condition_id == condition)
        per_attack.append(CycleAttackMetrics(
            condition, len(base), sum(_feature_valid(row) for row, _ in base), len(chosen),
            sum(not unit.safe for _, unit in chosen), sum(unit.safe_rescue for _, unit in chosen),
            sum(unit.observed_negative_false_positive is True for _, unit in chosen),
        ))
    covered = sum(item.accepted_count >= 1 for item in per_attack)
    accepted_count = len(accepted)
    correct_reject = baseline_unsafe - unsafe
    wrong_reject = sum(unit.safe for _, unit in baseline) - sum(unit.safe for _, unit in accepted)
    gates = (
        accepted_count >= 8 and rescue >= 8 and covered >= 5
        and unsafe * 10 <= accepted_count and fp == 0 and accepted_count > 0
        and 4 * rescue >= 3 * sum(unit.safe_rescue for _, unit in baseline)
        and baseline_unsafe > 0 and unsafe <= baseline_unsafe - 1
    )
    return CycleMetrics(
        candidate.candidate_id, split, len(baseline), len(boundary_eligible), feature_valid_count,
        accepted_count, unsafe, rescue, fp, covered, correct_reject, wrong_reject,
        accepted_count / 40.0,
        None if not boundary_eligible else accepted_count / len(boundary_eligible),
        None if not baseline else accepted_count / len(baseline),
        None if feature_valid_count == 0 else accepted_count / feature_valid_count,
        None if accepted_count == 0 else unsafe / accepted_count,
        baseline_unsafe, correct_reject, tuple(per_attack), gates,
    )


def select_cycle_candidate(
    dev_rows: Sequence[CycleFeatureRow], dev_units: Sequence[R3DevUnit],
) -> CycleSelection:
    candidates = generate_cycle_candidates(dev_rows)
    table = tuple(
        evaluate_cycle_candidate(candidate, dev_rows, dev_units, split="dev")
        for candidate in candidates
    )
    baseline_unsafe = table[0].baseline_unsafe_count if table else 0
    if baseline_unsafe == 0:
        return CycleSelection(R3_DEV_NOT_APPLICABLE, None, None, candidates, table)
    feasible = tuple(item for item in table if item.gates_passed)
    if not feasible:
        return CycleSelection(R3_DEV_NO_FEASIBLE_CYCLE, None, None, candidates, table)
    selected_metrics = min(feasible, key=lambda item: (
        -item.correct_reject_unsafe_count, -item.safe_rescue_count, -item.accepted_count,
        item.unsafe_accept_count, -item.covered_attack_count,
        next(candidate.complexity for candidate in candidates if candidate.candidate_id == item.candidate_id),
        item.candidate_id,
    ))
    selected = next(item for item in candidates if item.candidate_id == selected_metrics.candidate_id)
    return CycleSelection(R3_DEV_CYCLE_GATE_FROZEN, selected, selected_metrics, candidates, table)


def evaluate_frozen_cycle_candidate(
    candidate: CycleCandidate, test_rows: Sequence[CycleFeatureRow],
    test_units: Sequence[R3DevUnit],
) -> tuple[str, CycleMetrics]:
    metrics = evaluate_cycle_candidate(candidate, test_rows, test_units, split="test")
    if metrics.baseline_unsafe_count == 0:
        return R3_NOT_APPLICABLE, metrics
    return (R3_PASSED if metrics.gates_passed else R3_FAILED), metrics


__all__ = [name for name in globals() if name.startswith("R3_") or name in {
    "R3DevUnit", "BLowCandidate", "BLowAttackMetrics", "BLowMetrics",
    "BLowSelection", "generate_b_low_candidates", "route_for_b_low",
    "evaluate_b_low_candidate", "select_b_low", "D4BranchRecord", "CycleFeatureRow",
    "CycleStump", "CycleCandidate", "CycleAttackMetrics", "CycleMetrics",
    "CycleSelection", "cycle_feature_row", "generate_cycle_candidates",
    "evaluate_cycle_candidate", "select_cycle_candidate", "evaluate_frozen_cycle_candidate",
}]
