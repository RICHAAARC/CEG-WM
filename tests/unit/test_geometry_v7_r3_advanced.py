from __future__ import annotations

import inspect
import math

from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS
from cegwm.geometry_v7.r3 import R3Unit
from cegwm.geometry_v7.r3_advanced import (
    AdvancedRow,
    advanced_runtime_decision,
    evaluate_advanced,
    orientation_diagnostic,
    predicted_h_regime,
)


def _rotation(angle: float, scale: float = 1.0, tx: float = 0.0):
    radians = math.radians(angle)
    return (
        (scale * math.cos(radians), -scale * math.sin(radians), tx),
        (scale * math.sin(radians), scale * math.cos(radians), 0.0),
        (0.0, 0.0, 1.0),
    )


def test_predicted_h_regime_uses_exact_inclusive_frozen_bounds_and_h22_normalization():
    for angle in (-20.0, -10.0, 10.0, 20.0):
        matrix = tuple(tuple(2.0 * value for value in row) for row in _rotation(angle))
        regime = predicted_h_regime(matrix, geometry_legal=True, geometry_error=None)
        assert regime.valid and regime.pure_rotation_gate
        assert abs(abs(regime.angle_degrees) - abs(angle)) < 1e-9
    assert not predicted_h_regime(_rotation(9.999), geometry_legal=True, geometry_error=None).pure_rotation_gate
    assert not predicted_h_regime(_rotation(15, scale=1.051), geometry_legal=True, geometry_error=None).pure_rotation_gate
    assert not predicted_h_regime(_rotation(15, tx=0.021), geometry_legal=True, geometry_error=None).pure_rotation_gate


def test_invalid_or_illegal_h_fails_closed_even_when_old_cycle_passes():
    for matrix, legal, error in (
        (((1, 0, 0), (0, 1, 0), (0, 0, 0)), True, None),
        (((1, 0, 0), (0, math.nan, 0), (0, 0, 1)), True, None),
        (((1, 0, 0), (0, 0, 0), (0, 0, 1)), True, None),
        (_rotation(15), False, None),
        (_rotation(15), True, "stored error"),
    ):
        decision = advanced_runtime_decision(
            boundary=True, r2_selector_accepted=True, old_cycle_score_px=1.0,
            homography_observed_to_canonical=matrix,
            geometry_legal=legal, geometry_error=error,
        )
        assert not decision.accepted and not decision.regime.valid


def test_runtime_predicate_has_no_condition_truth_attack_or_outcome_parameter():
    assert tuple(inspect.signature(advanced_runtime_decision).parameters) == (
        "boundary", "r2_selector_accepted", "old_cycle_score_px",
        "homography_observed_to_canonical", "geometry_legal", "geometry_error",
    )
    source = inspect.getsource(advanced_runtime_decision).lower()
    for forbidden in ("condition_id", "truth", "attack_label", "post_outcome"):
        assert forbidden not in source


def test_frozen_a_reproduces_dev_16_safe_zero_unsafe_zero_fp_covered7():
    baseline = {
        "core_fixed_canvas_zoom_0_8": {0, 2},
        "core_translation_pos32_x": {1},
        "core_translation_neg32_x": {0, 1},
        "core_translation_pos32_y": {0, 1},
        "core_translation_neg32_y": {1},
    }
    rows, units = [], []
    for condition in R2_CONDITION_IDS:
        for index, unit_id in enumerate(R2_DEV_UNIT_IDS):
            rotation = condition in ("core_rotation_neg15", "core_rotation_pos15")
            old_pass = index in baseline.get(condition, set())
            accepted = rotation or old_pass
            h = _rotation(-15 if condition == "core_rotation_neg15" else 15) if rotation else _rotation(0)
            rows.append(AdvancedRow(
                "dev", condition, unit_id, "BOUNDARY", True,
                1.0 if old_pass else 60.0, h, True, None,
            ))
            units.append(R3Unit(
                "dev", condition, unit_id, -1.0, True, True,
                accepted, accepted, False, accepted, False,
            ))
    metrics = evaluate_advanced(rows, units, split="dev")
    assert (metrics.fixed_denominator, metrics.baseline_accepted_count,
            metrics.baseline_safe_rescue_count, metrics.baseline_unsafe_accept_count,
            metrics.baseline_negative_control_fp_count,
            metrics.baseline_covered_attack_count) == (40, 8, 8, 0, 0, 5)
    assert (metrics.accepted_count, metrics.safe_rescue_count,
            metrics.unsafe_accept_count, metrics.selected_negative_control_fp_count,
            metrics.covered_attack_count) == (16, 16, 0, 0, 7)
    directions = {item.condition_id: item.accepted_count for item in metrics.per_attack}
    assert directions["core_rotation_neg15"] == directions["core_rotation_pos15"] == 4


def test_orientation_diagnostic_ranks_residual_and_records_margin_identity_and_cycles():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    branches = []
    for index, name in enumerate((
        "identity", "rotate_90_ccw", "rotate_180", "rotate_270_ccw",
        "mirror_left_right", "mirror_left_right_then_rotate_90_ccw",
        "mirror_left_right_then_rotate_180", "mirror_left_right_then_rotate_270_ccw",
    )):
        branches.append({
            "transform": name, "cycle_pixels": float(8 - index), "d_matrix": identity,
            "geometry": {"legal": True, "error": None,
                         "uncalibrated_sync_logit": float(index),
                         "homography_observed_to_canonical": identity},
        })
    diagnostic = orientation_diagnostic(branches)
    assert diagnostic.valid and diagnostic.best_transform.endswith("rotate_270_ccw")
    assert diagnostic.best_residual_px == 1.0
    assert diagnostic.second_residual_px == 2.0
    assert diagnostic.second_minus_best_margin_px == 1.0
    assert diagnostic.identity_residual_px == 8.0
    assert diagnostic.identity_minus_best_residual_px == 7.0
    assert diagnostic.identity_over_best_residual == 8.0
    assert diagnostic.best_forward_cycle_px == 1.0
    assert diagnostic.best_backward_cycle_px == 0.0
