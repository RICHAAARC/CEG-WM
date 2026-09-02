from __future__ import annotations

from dataclasses import replace
import math

import pytest

from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r3 import (
    R3_DEV_B_LOW_FROZEN,
    R3_DEV_CYCLE_GATE_FROZEN,
    R3_DEV_NOT_APPLICABLE,
    R3_DEV_NO_FEASIBLE_B_LOW,
    R3_NOT_APPLICABLE,
    R3_NOT_PROBED_INELIGIBLE,
    CycleFeatureRow,
    D4BranchRecord,
    R3DevUnit,
    cycle_feature_row,
    evaluate_frozen_cycle_candidate,
    generate_cycle_candidates,
    route_for_b_low,
    select_b_low,
    select_cycle_candidate,
)


D4_ORDER = (
    "identity", "rotate_90_ccw", "rotate_180", "rotate_270_ccw",
    "mirror_left_right", "mirror_left_right_then_rotate_90_ccw",
    "mirror_left_right_then_rotate_180", "mirror_left_right_then_rotate_270_ccw",
)


def _units(split: str = "dev") -> tuple[R3DevUnit, ...]:
    roster = R2_DEV_UNIT_IDS if split == "dev" else R2_TEST_UNIT_IDS
    result = []
    for condition_index, condition in enumerate(R2_CONDITION_IDS):
        for unit_index, unit in enumerate(roster):
            index = condition_index * 4 + unit_index
            s0 = -float(index + 1) / 100.0
            result.append(R3DevUnit(
                condition, unit, s0, True, True,
                index < 30, index < 30, False,
            ))
    return tuple(result)


def _branches(value: float = 1.0, *, invalid: tuple[int, ...] = ()):
    return tuple(D4BranchRecord(
        transform, True, {"legal": index not in invalid},
        None if index in invalid else value / 255.5,
        None if index in invalid else value,
        ("geometry_invalid",) if index in invalid else (),
    ) for index, transform in enumerate(D4_ORDER))


def _rows(units: tuple[R3DevUnit, ...], split: str, b_low: float):
    result = []
    for index, unit in enumerate(units):
        route = route_for_b_low(unit, b_low)
        branches = _branches(10.0 if index < 10 else 0.0)
        result.append(cycle_feature_row(
            split=split, unit=unit, route=route, branches=branches, d4_order=D4_ORDER,
        ))
    return tuple(result)


def test_b_low_type7_selection_is_dev_only_and_exact_boundary_semantics():
    units = _units()
    selection = select_b_low(units)
    assert selection.status == R3_DEV_B_LOW_FROZEN
    assert selection.selected is not None
    assert selection.selected.b_low < 0.0
    assert selection.selected_metrics is not None
    assert selection.selected_metrics.accepted_count >= 10
    assert selection.selected_metrics.accepted_over_fixed_40 == pytest.approx(
        selection.selected_metrics.accepted_count / 40
    )
    boundary = replace(units[0], s0=selection.selected.b_low)
    assert route_for_b_low(boundary, selection.selected.b_low) == "DIRECT_NEGATIVE"
    assert route_for_b_low(replace(boundary, s0=0.0), selection.selected.b_low) == "BOUNDARY"
    assert route_for_b_low(replace(boundary, s0=math.nan), selection.selected.b_low) == "INVALID_S0"
    with pytest.raises(ValueError, match="exact ordered"):
        select_b_low((*units[:-1], replace(units[-1], unit_id="test-leak")))


def test_b_low_outcome_errors_do_not_change_s0_route_and_no_feasible_is_bounded():
    unit = replace(_units()[0], errors=("outcome_incomplete",))
    assert route_for_b_low(unit, -0.2) == "BOUNDARY"
    rejected = tuple(replace(item, r2_selector_accepted=False) for item in _units())
    assert select_b_low(rejected).status == R3_DEV_NO_FEASIBLE_B_LOW


def test_cycle_features_require_identity_and_six_legal_branches():
    unit = replace(_units()[0], s0=-0.1)
    row = cycle_feature_row(
        split="dev", unit=unit, route="BOUNDARY", branches=_branches(2.0, invalid=(7,)),
        d4_order=D4_ORDER,
    )
    assert row.feature_valid is True
    assert row.identity_cycle_px == 2.0
    assert row.d4_median_cycle_px == 2.0
    assert row.d4_max_cycle_px == 2.0
    assert row.invalid_d4_count == 1
    assert cycle_feature_row(
        split="dev", unit=unit, route="BOUNDARY", branches=_branches(invalid=(0,)),
        d4_order=D4_ORDER,
    ).feature_valid is False
    assert cycle_feature_row(
        split="dev", unit=unit, route="BOUNDARY", branches=_branches(invalid=(1, 2, 3)),
        d4_order=D4_ORDER,
    ).feature_valid is False


def test_ineligible_units_are_not_probed_and_have_explicit_reason_categories():
    unit = _units()[0]
    for route, accepted, reason in (
        ("DIRECT_POSITIVE", True, "DIRECT_POSITIVE"),
        ("DIRECT_NEGATIVE", True, "DIRECT_NEGATIVE"),
        ("INVALID_S0", True, "INVALID_S0"),
        ("BOUNDARY", False, "BOUNDARY_R2_REJECTED"),
    ):
        candidate = replace(unit, r2_selector_accepted=accepted)
        branches = tuple(D4BranchRecord(item, False, None, None, None, ()) for item in D4_ORDER)
        row = cycle_feature_row(
            split="dev", unit=candidate, route=route, branches=branches, d4_order=D4_ORDER,
        )
        assert row.feature_valid is False
        assert row.errors == (f"{R3_NOT_PROBED_INELIGIBLE}:{reason}",)


def test_cycle_candidate_type7_order_gates_and_three_denominators():
    units = tuple(
        replace(item, s0=-0.01, safe=index >= 10, safe_rescue=index >= 10)
        for index, item in enumerate(_units())
    )
    rows = _rows(units, "dev", -0.02)
    candidates = generate_cycle_candidates(rows)
    assert candidates[0].candidate_id == "A|CYCLE_VALID_ONLY"
    assert all("|le|" in item.candidate_id for item in candidates[1:])
    assert any(item.candidate_id.startswith("C|") for item in candidates)
    selection = select_cycle_candidate(rows, units)
    assert selection.status == R3_DEV_CYCLE_GATE_FROZEN
    assert selection.selected is not None
    metrics = selection.selected_metrics
    assert metrics is not None and metrics.gates_passed
    assert metrics.accepted_over_fixed_40 == metrics.accepted_count / 40
    assert metrics.accepted_over_boundary_eligible == metrics.accepted_count / 40
    assert metrics.accepted_over_baseline_accepted == metrics.accepted_count / 40
    assert metrics.correct_reject_unsafe_count >= 1


def test_no_baseline_false_reliable_is_not_applicable_and_test_cannot_select():
    dev = tuple(replace(item, s0=-0.01, safe=True, safe_rescue=True) for item in _units())
    rows = _rows(dev, "dev", -0.02)
    selection = select_cycle_candidate(rows, dev)
    assert selection.status == R3_DEV_NOT_APPLICABLE
    test = tuple(replace(item, s0=-0.01) for item in _units("test"))
    test_rows = _rows(test, "test", -0.02)
    candidate = generate_cycle_candidates(rows)[0]
    status, metrics = evaluate_frozen_cycle_candidate(candidate, test_rows, test)
    assert status != R3_DEV_CYCLE_GATE_FROZEN
    assert metrics.split == "test"
    with pytest.raises(ValueError, match="dev rows"):
        generate_cycle_candidates(test_rows)
