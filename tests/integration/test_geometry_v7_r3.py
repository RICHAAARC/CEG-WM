from __future__ import annotations

from dataclasses import replace
import math

import pytest

from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r3 import (
    R3_B_LOW,
    R3_B_LOW_HEX,
    R3_METHOD_IMPROVED,
    R3_METHOD_NOT_IMPROVED,
    R3_NOT_PROBED_INELIGIBLE,
    R3_THRESHOLD_GRID_PX,
    CycleFeatureRow,
    D4BranchRecord,
    R3Unit,
    cycle_feature_row,
    evaluate_selected_threshold,
    route_from_s0,
    select_threshold,
)


D4_ORDER = (
    "identity", "rotate_90_ccw", "rotate_180", "rotate_270_ccw",
    "mirror_left_right", "mirror_left_right_then_rotate_90_ccw",
    "mirror_left_right_then_rotate_180", "mirror_left_right_then_rotate_270_ccw",
)


def _units(split: str = "dev") -> tuple[R3Unit, ...]:
    roster = R2_DEV_UNIT_IDS if split == "dev" else R2_TEST_UNIT_IDS
    return tuple(
        R3Unit(
            split, condition, unit, -1.0, True, True,
            index >= 10, index >= 10, False, True, False,
        )
        for index, (condition, unit) in enumerate(
            (pair for condition in R2_CONDITION_IDS for pair in ((condition, unit) for unit in roster))
        )
    )


def _branches(score: float, invalid: tuple[int, ...] = ()) -> tuple[D4BranchRecord, ...]:
    return tuple(D4BranchRecord(
        name, True, {"legal": index not in invalid},
        None if index in invalid else score / 255.5,
        None if index in invalid else score,
        ("geometry_invalid",) if index in invalid else (),
    ) for index, name in enumerate(D4_ORDER))


def _rows(units: tuple[R3Unit, ...], *, unsafe_score: float = 10.0):
    return tuple(cycle_feature_row(
        unit=unit,
        branches=_branches(unsafe_score if index < 10 else 1.0),
        d4_order=D4_ORDER,
    ) for index, unit in enumerate(units))


def test_boundary_partition_is_frozen_directly_without_candidate_calibration():
    assert R3_B_LOW.hex() == R3_B_LOW_HEX
    assert route_from_s0(0.01) == "DIRECT_POSITIVE"
    assert route_from_s0(0.0) == "BOUNDARY"
    assert route_from_s0(R3_B_LOW + 1e-12) == "BOUNDARY"
    assert route_from_s0(R3_B_LOW) == "DIRECT_NEGATIVE"
    assert route_from_s0(R3_B_LOW - 1e-12) == "DIRECT_NEGATIVE"
    assert route_from_s0(math.nan) == "INVALID_S0"


def test_d4_participation_is_boundary_and_frozen_r2_acceptance_only():
    unit = _units()[0]
    not_probed = tuple(D4BranchRecord(name, False, None, None, None) for name in D4_ORDER)
    for changed, reason in (
        (replace(unit, s0=0.1), "DIRECT_POSITIVE"),
        (replace(unit, s0=R3_B_LOW), "DIRECT_NEGATIVE"),
        (replace(unit, s0=None), "INVALID_S0"),
        (replace(unit, r2_selector_accepted=False), "BOUNDARY_R2_REJECTED"),
    ):
        row = cycle_feature_row(unit=changed, branches=not_probed, d4_order=D4_ORDER)
        assert row.errors == (f"{R3_NOT_PROBED_INELIGIBLE}:{reason}",)
        assert row.feature_valid is False
    with pytest.raises(ValueError, match="must not invoke"):
        cycle_feature_row(unit=replace(unit, s0=0.1), branches=_branches(1.0), d4_order=D4_ORDER)


def test_cycle_score_is_max_identity_and_median_with_six_of_eight_required():
    unit = _units()[0]
    branches = list(_branches(2.0))
    branches[0] = replace(branches[0], cycle_pixels=4.0)
    branches[7] = replace(branches[7], cycle_pixels=None, errors=("invalid",))
    row = cycle_feature_row(unit=unit, branches=branches, d4_order=D4_ORDER)
    assert row.feature_valid and row.valid_d4_count == 7
    assert row.identity_cycle_px == 4.0
    assert row.d4_median_cycle_px == 2.0
    assert row.cycle_score_px == 4.0
    assert not cycle_feature_row(
        unit=unit, branches=_branches(1.0, invalid=(1, 2, 3)), d4_order=D4_ORDER,
    ).feature_valid


def test_single_grid_selects_by_rescue_accept_unsafe_then_smaller_threshold():
    units = _units()
    selection = select_threshold(_rows(units), units)
    assert selection.status == R3_METHOD_IMPROVED
    assert tuple(item.threshold_px for item in selection.grid_metrics) == R3_THRESHOLD_GRID_PX
    assert selection.selected_threshold_px == 1.0
    metrics = selection.selected_metrics
    assert metrics is not None and metrics.accepted_count == 30
    assert metrics.unsafe_accept_count == 0 and metrics.safe_rescue_count == 30
    assert metrics.correct_reject_unsafe_count == 10
    assert len(metrics.translation_summary) == 4
    assert metrics.coverage == 30 / 40


def test_no_strict_unsafe_reduction_and_all_reject_are_not_improved():
    units = _units()
    assert select_threshold(_rows(units, unsafe_score=1.0), units).status == R3_METHOD_NOT_IMPROVED
    invalid_rows = tuple(replace(row, feature_valid=False, cycle_score_px=None) for row in _rows(units))
    result = select_threshold(invalid_rows, units)
    assert result.status == R3_METHOD_NOT_IMPROVED
    assert all(item.accepted_count == 0 for item in result.grid_metrics)


def test_existing_test40_uses_only_the_selected_scalar_threshold():
    units = _units("test")
    metrics = evaluate_selected_threshold(1.0, _rows(units), units)
    assert metrics.split == "test" and metrics.fixed_denominator == 40
    with pytest.raises(ValueError, match="fixed pixel grid"):
        evaluate_selected_threshold(3.0, _rows(units), units)
