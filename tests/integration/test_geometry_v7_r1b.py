from __future__ import annotations

import math

from PIL import Image, ImageDraw
import pytest

from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED
from cegwm.geometry_v7.r0 import ContentScore
from cegwm.geometry_v7.r1a import (
    R1A_CORE_CONDITIONS,
    apply_homography,
    corner_rmse,
    render_r1a_attack,
    truth_correspondences,
)
from cegwm.geometry_v7.r1b import (
    R1B_INSUFFICIENT_GEOMETRY_NECESSITY,
    R1B_LAMBDA_GRID,
    R1B_TRUTH_UTILITY_AND_NONZERO_EPSILON_PASSED,
    R1B_TRUTH_UTILITY_FAILED,
    R1B_ZERO_ONLY_TOLERANCE_FAILED,
    R1B_REPAIR_FINE_ALL_CORE_NONZERO_PREFIX,
    R1B_REPAIR_FINE_PARTIAL_CORE_NONZERO_PREFIX,
    R1B_REPAIR_METHOD_NOT_READY,
    R1B_REPAIR_METHOD_PASSED,
    R1B_REPAIR_PIXEL_GRID,
    R1B_REPAIR_REAL_ALL_CORE_PASSED,
    R1B_REPAIR_REAL_PARTIAL_CORE_PASSED,
    R1BMembership,
    R1BStoredPrediction,
    aggregate_lambda,
    controlled_correspondences,
    controlled_homography,
    directional_correspondences_for_pixels,
    directional_homography_for_pixels,
    epsilon_for_lambda,
    evaluate_condition,
    evaluate_lambda_unit,
    evaluate_repair_point_unit,
    evaluate_r1b,
    evaluate_r1b_repair,
    freeze_pre_recovery_record,
    rectify_attacked_rgb,
    scored_triplet,
)


_ROSTER = tuple(f"evaluation-{index:02d}" for index in range(8))


def _score(weighted_joint: float, *, wrong: float = -2.0) -> ContentScore:
    return ContentScore(
        0.0,
        0.0,
        weighted_joint,
        (0.0,) * 16,
        (0.0,) * 16,
        (wrong,) * 16,
    )


def _triplet(positive_score: float, *, negative_fp: bool = False):
    u = _score(-0.1 if negative_fp else 0.1)
    g = _score(0.0)
    cg = _score(positive_score)
    return scored_triplet(u=u, g=g, cg=cg)


def _pre_records(spec, values):
    return tuple(
        freeze_pre_recovery_record(
            unit_id=unit_id,
            spec=spec,
            clean_score=1.0,
            scores=_triplet(value),
        )
        for unit_id, value in zip(_ROSTER, values, strict=True)
    )


def _lambda_records(spec, pre, value: float, gains, *, failure_index=None):
    records = []
    for index, (pre_record, gain) in enumerate(zip(pre, gains, strict=True)):
        if index == failure_index:
            records.append(
                evaluate_lambda_unit(
                    pre_record=pre_record,
                    spec=spec,
                    lambda_value=value,
                    scores=None,
                    errors=("content_score:RuntimeError",),
                )
            )
        else:
            records.append(
                evaluate_lambda_unit(
                    pre_record=pre_record,
                    spec=spec,
                    lambda_value=value,
                    scores=_triplet(
                        pre_record.scores.positive_cg_vs_g.margin + gain
                    ),
                )
            )
    return tuple(records)


@pytest.mark.integration
def test_paired_score_algebra_tau_and_boundary_partition_are_exact() -> None:
    equality = _triplet(0.0)
    assert equality.positive_cg_vs_g.gate_a_margin == pytest.approx(2.0)
    assert equality.positive_cg_vs_g.gate_b_margin == 0.0
    assert equality.positive_cg_vs_g.margin == 0.0
    assert not equality.positive_cg_vs_g.positive
    assert not equality.negative_g_vs_u.positive

    spec = R1A_CORE_CONDITIONS[0]
    negative, boundary, damage = tuple(
        freeze_pre_recovery_record(
            unit_id=f"partition-{index}",
            spec=spec,
            clean_score=1.0,
            scores=_triplet(value),
        )
        for index, value in enumerate((-0.1, 0.25, 0.250001))
    )
    assert negative.membership is R1BMembership.RECOVERY_NEGATIVE
    assert boundary.membership is R1BMembership.BOUNDARY
    assert damage.membership is R1BMembership.DAMAGE_ONLY


@pytest.mark.integration
def test_controlled_truth_identity_path_and_epsilon_are_exact() -> None:
    spec = R1A_CORE_CONDITIONS[0]
    assert controlled_correspondences(spec, 0.0) == truth_correspondences(spec)
    assert controlled_correspondences(spec, 1.0) == CANONICAL_CORNERS_NORMALIZED
    halfway = controlled_correspondences(spec, 0.5)
    truth = truth_correspondences(spec)
    expected_halfway = tuple(
        ((tx + qx) / 2.0, (ty + qy) / 2.0)
        for (tx, ty), (qx, qy) in zip(
            truth, CANONICAL_CORNERS_NORMALIZED, strict=True
        )
    )
    for actual, expected in zip(halfway, expected_halfway, strict=True):
        assert actual == pytest.approx(expected)
    for value in R1B_LAMBDA_GRID:
        solved = controlled_homography(spec, value)
        actual = apply_homography(solved, CANONICAL_CORNERS_NORMALIZED)
        expected = controlled_correspondences(spec, value)
        for actual_point, expected_point in zip(actual, expected, strict=True):
            assert actual_point == pytest.approx(expected_point)
    epsilon_one, pixels_one = epsilon_for_lambda(spec, 1.0)
    epsilon_quarter, pixels_quarter = epsilon_for_lambda(spec, 0.25)
    assert epsilon_quarter == pytest.approx(0.25 * epsilon_one)
    assert pixels_one == pytest.approx(epsilon_one * 511.0 / 2.0)
    assert pixels_quarter == pytest.approx(epsilon_quarter * 511.0 / 2.0)


@pytest.mark.integration
def test_truth_rectification_uses_inverse_sampler_once() -> None:
    source = Image.new("RGB", (512, 512), "black")
    draw = ImageDraw.Draw(source)
    draw.rectangle((250, 250, 262, 262), fill="white")
    spec = next(
        item
        for item in R1A_CORE_CONDITIONS
        if item.condition_id == "core_translation_pos32_x"
    )
    attacked = render_r1a_attack(source, spec)
    rectified = rectify_attacked_rgb(attacked, controlled_homography(spec, 0.0))
    assert attacked.getbbox() is not None and rectified.getbbox() is not None
    attacked_center = (attacked.getbbox()[0] + attacked.getbbox()[2]) / 2.0
    rectified_center = (rectified.getbbox()[0] + rectified.getbbox()[2]) / 2.0
    assert attacked_center > 280.0
    assert rectified_center == pytest.approx(256.0, abs=2.0)


@pytest.mark.integration
def test_lambda_gate_uses_full_eligible_sentinel_and_all_four_gates() -> None:
    spec = R1A_CORE_CONDITIONS[0]
    pre = _pre_records(spec, (0.1,) * 6 + (0.8,) * 2)
    records = _lambda_records(spec, pre, 0.0, (0.2,) * 8)
    aggregate = aggregate_lambda(
        spec=spec,
        pre_records=pre,
        lambda_records=records,
        ordered_roster=_ROSTER,
    )
    assert aggregate.eligible_denominator == 6
    assert aggregate.required_improved_count == math.ceil(0.75 * 6) == 5
    assert aggregate.improved_count == 6
    assert aggregate.full_eligible_gain_median == pytest.approx(0.2)
    assert aggregate.recovery_negative_count is None
    assert aggregate.damage_harm_count == 0
    assert aggregate.observed_negative_false_positive_count == 0
    assert aggregate.observed_negative_denominator == 8
    assert aggregate.passed
    assert records[0].positive_gate_a_delta == pytest.approx(0.2)
    assert records[0].positive_gate_b_delta == pytest.approx(0.2)
    assert records[0].positive_score_delta == pytest.approx(0.2)
    assert records[0].gain == records[0].positive_score_delta

    failed_records = _lambda_records(
        spec, pre, 0.0, (0.2,) * 8, failure_index=0
    )
    failed = aggregate_lambda(
        spec=spec,
        pre_records=pre,
        lambda_records=failed_records,
        ordered_roster=_ROSTER,
    )
    assert failed.eligible_denominator == 6
    assert failed.valid_gain_count == 5
    assert failed.missing_gain_sentinel_count == 1
    assert failed_records[0].positive_gate_a_delta is None
    assert failed_records[0].positive_gate_b_delta is None
    assert failed_records[0].positive_score_delta is None
    assert failed.full_eligible_gain_median == pytest.approx(0.2)
    assert not failed.passed


@pytest.mark.integration
def test_recovery_damage_and_negative_gates_are_fixed_denominator() -> None:
    spec = R1A_CORE_CONDITIONS[0]
    pre = _pre_records(spec, (-0.1, -0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8))
    passing = _lambda_records(spec, pre, 0.0, (0.3,) * 8)
    aggregate = aggregate_lambda(
        spec=spec,
        pre_records=pre,
        lambda_records=passing,
        ordered_roster=_ROSTER,
    )
    assert aggregate.recovery_negative_count == 2
    assert aggregate.required_recovery_negative_count == 1
    assert aggregate.passed

    harmed = list(passing)
    harmed[4] = evaluate_lambda_unit(
        pre_record=pre[4],
        spec=spec,
        lambda_value=0.0,
        scores=_triplet(0.0),
    )
    harm_aggregate = aggregate_lambda(
        spec=spec,
        pre_records=pre,
        lambda_records=tuple(harmed),
        ordered_roster=_ROSTER,
    )
    assert harm_aggregate.damage_harm_count == 1
    assert not harm_aggregate.passed

    false_positive = list(passing)
    false_positive[7] = evaluate_lambda_unit(
        pre_record=pre[7],
        spec=spec,
        lambda_value=0.0,
        scores=_triplet(1.1, negative_fp=True),
    )
    fp_aggregate = aggregate_lambda(
        spec=spec,
        pre_records=pre,
        lambda_records=tuple(false_positive),
        ordered_roster=_ROSTER,
    )
    assert fp_aggregate.observed_negative_false_positive_count == 1
    assert fp_aggregate.observed_negative_denominator == 8
    assert not fp_aggregate.passed


def _condition_inputs(first_fail: float | None = None):
    pre_by_condition = {}
    lambda_by_condition = {}
    for index, spec in enumerate(R1A_CORE_CONDITIONS):
        values = (0.1,) * 8 if index == 0 else (0.8,) * 8
        pre = _pre_records(spec, values)
        pre_by_condition[spec.condition_id] = pre
        if index == 0:
            mapping = {}
            for value in R1B_LAMBDA_GRID:
                gain = 0.2
                if first_fail is not None and value == first_fail:
                    gain = -0.2
                mapping[value] = _lambda_records(spec, pre, value, (gain,) * 8)
            lambda_by_condition[spec.condition_id] = mapping
    return pre_by_condition, lambda_by_condition


@pytest.mark.integration
def test_contiguous_prefix_and_stage_statuses_are_exact() -> None:
    pre, lambdas = _condition_inputs(first_fail=0.5)
    evaluation = evaluate_r1b(
        pre_records_by_condition=pre,
        lambda_records_by_condition=lambdas,
        ordered_roster=_ROSTER,
    )
    first = evaluation.conditions[0]
    assert first.accepted_lambda == 0.25
    assert evaluation.status == R1B_TRUTH_UTILITY_AND_NONZERO_EPSILON_PASSED
    assert evaluation.blocking_method_canary_passed is True

    pre, lambdas = _condition_inputs(first_fail=0.0)
    truth_failed = evaluate_r1b(
        pre_records_by_condition=pre,
        lambda_records_by_condition=lambdas,
        ordered_roster=_ROSTER,
    )
    assert truth_failed.status == R1B_TRUTH_UTILITY_FAILED
    assert truth_failed.blocking_method_canary_passed is False

    pre, lambdas = _condition_inputs(first_fail=0.25)
    zero_only = evaluate_r1b(
        pre_records_by_condition=pre,
        lambda_records_by_condition=lambdas,
        ordered_roster=_ROSTER,
    )
    assert zero_only.conditions[0].accepted_lambda == 0.0
    assert zero_only.status == R1B_ZERO_ONLY_TOLERANCE_FAILED
    assert zero_only.blocking_method_canary_passed is False


@pytest.mark.integration
def test_all_empty_eligibility_stops_before_any_lambda() -> None:
    pre = {
        spec.condition_id: _pre_records(spec, (0.8,) * 8)
        for spec in R1A_CORE_CONDITIONS
    }
    evaluation = evaluate_r1b(
        pre_records_by_condition=pre,
        lambda_records_by_condition={},
        ordered_roster=_ROSTER,
    )
    assert evaluation.status == R1B_INSUFFICIENT_GEOMETRY_NECESSITY
    assert evaluation.applicable_condition_count == 0
    assert evaluation.blocking_method_canary_passed is None
    assert all(
        not condition.applicable
        and condition.eligibility_status
        == "NOT_APPLICABLE/INSUFFICIENT_ELIGIBLE"
        and not condition.lambda_aggregates
        for condition in evaluation.conditions
    )


@pytest.mark.integration
def test_membership_failure_cannot_enter_method_evaluation() -> None:
    spec = R1A_CORE_CONDITIONS[0]
    pre = list(_pre_records(spec, (0.1,) * 8))
    pre[2] = freeze_pre_recovery_record(
        unit_id=_ROSTER[2],
        spec=spec,
        clean_score=1.0,
        scores=None,
        errors=("content_score:RuntimeError",),
    )
    with pytest.raises(ValueError, match="membership must be frozen"):
        evaluate_condition(
            spec=spec,
            pre_records=tuple(pre),
            lambda_records={},
            ordered_roster=_ROSTER,
        )


def _stored_prediction(spec, unit_id: str, *, predicted_rmse=None):
    truth = truth_correspondences(spec)
    predicted = tuple((x + 0.02, y) for x, y in truth)
    rmse = corner_rmse(predicted, truth) if predicted_rmse is None else predicted_rmse
    return R1BStoredPrediction(
        unit_id,
        spec.condition_id,
        truth,
        predicted,
        controlled_homography(spec, 0.0),
        rmse,
        rmse * 511.0 / 2.0,
        (),
    )


@pytest.mark.integration
def test_directional_pixel_grid_scales_and_extrapolates_stored_error() -> None:
    spec = R1A_CORE_CONDITIONS[0]
    prediction = _stored_prediction(spec, _ROSTER[0])
    assert R1B_REPAIR_PIXEL_GRID == (0, 1, 2, 4, 6, 8)
    assert directional_correspondences_for_pixels(prediction, 0) == (
        prediction.truth_correspondences
    )
    for radius in R1B_REPAIR_PIXEL_GRID:
        points = directional_correspondences_for_pixels(prediction, radius)
        error_pixels = (
            corner_rmse(points, prediction.truth_correspondences) * 511.0 / 2.0
        )
        assert error_pixels == pytest.approx(radius)
        solved = directional_homography_for_pixels(prediction, radius)
        mapped = apply_homography(solved, CANONICAL_CORNERS_NORMALIZED)
        for actual, expected in zip(mapped, points, strict=True):
            assert actual == pytest.approx(expected)
    assert prediction.prediction_rmse_pixels < 8.0


@pytest.mark.integration
def test_zero_radius_survives_missing_direction_nonzero_points_fail() -> None:
    spec = R1A_CORE_CONDITIONS[0]
    truth = truth_correspondences(spec)
    prediction = R1BStoredPrediction(
        _ROSTER[0], spec.condition_id, truth, truth, None, 0.0, 0.0,
        ("stored_prediction:invalid_homography",),
    )
    assert directional_correspondences_for_pixels(prediction, 0) == truth
    with pytest.raises(ValueError, match="direction is unavailable"):
        directional_correspondences_for_pixels(prediction, 1)

    inconsistent = R1BStoredPrediction(
        _ROSTER[0], spec.condition_id, truth, truth, None, 0.1, 1.0, ()
    )
    with pytest.raises(ValueError, match="pixel conversion differs"):
        directional_correspondences_for_pixels(inconsistent, 1)


def _repair_inputs(*, real_fail_condition=None, fine_fail_condition=None):
    pre_by_condition = {}
    real_by_condition = {}
    fine_by_condition = {}
    for spec in R1A_CORE_CONDITIONS:
        pre = _pre_records(spec, (0.1,) * 8)
        pre_by_condition[spec.condition_id] = pre
        real_gain = -0.2 if spec.condition_id == real_fail_condition else 0.2
        real_by_condition[spec.condition_id] = tuple(
            evaluate_repair_point_unit(
                pre_record=record,
                point_kind="real_h",
                radius_pixels=None,
                scores=_triplet(record.scores.positive_cg_vs_g.margin + real_gain),
            )
            for record in pre
        )
        grid = {}
        for radius in R1B_REPAIR_PIXEL_GRID:
            gain = (
                -0.2
                if spec.condition_id == fine_fail_condition and radius == 1
                else 0.2
            )
            grid[radius] = tuple(
                evaluate_repair_point_unit(
                    pre_record=record,
                    point_kind="directional_pixel",
                    radius_pixels=radius,
                    scores=_triplet(record.scores.positive_cg_vs_g.margin + gain),
                )
                for record in pre
            )
        fine_by_condition[spec.condition_id] = grid
    return pre_by_condition, real_by_condition, fine_by_condition


@pytest.mark.integration
def test_repair_real_and_fine_statuses_are_orthogonal() -> None:
    pre, real, fine = _repair_inputs()
    passed = evaluate_r1b_repair(
        pre_records_by_condition=pre,
        real_h_records_by_condition=real,
        fine_grid_records_by_condition=fine,
        ordered_roster=_ROSTER,
    )
    assert passed.real_h_status == R1B_REPAIR_REAL_ALL_CORE_PASSED
    assert passed.fine_grid_status == R1B_REPAIR_FINE_ALL_CORE_NONZERO_PREFIX
    assert passed.status == R1B_REPAIR_METHOD_PASSED
    assert passed.r2_candidate

    pre, real, fine = _repair_inputs(
        real_fail_condition=R1A_CORE_CONDITIONS[0].condition_id
    )
    real_partial = evaluate_r1b_repair(
        pre_records_by_condition=pre,
        real_h_records_by_condition=real,
        fine_grid_records_by_condition=fine,
        ordered_roster=_ROSTER,
    )
    assert real_partial.real_h_status == R1B_REPAIR_REAL_PARTIAL_CORE_PASSED
    assert real_partial.fine_grid_status == R1B_REPAIR_FINE_ALL_CORE_NONZERO_PREFIX
    assert real_partial.status == R1B_REPAIR_METHOD_NOT_READY
    assert not real_partial.r2_candidate

    pre, real, fine = _repair_inputs(
        fine_fail_condition=R1A_CORE_CONDITIONS[0].condition_id
    )
    fine_partial = evaluate_r1b_repair(
        pre_records_by_condition=pre,
        real_h_records_by_condition=real,
        fine_grid_records_by_condition=fine,
        ordered_roster=_ROSTER,
    )
    assert fine_partial.real_h_status == R1B_REPAIR_REAL_ALL_CORE_PASSED
    assert fine_partial.fine_grid_status == R1B_REPAIR_FINE_PARTIAL_CORE_NONZERO_PREFIX
    assert fine_partial.status == R1B_REPAIR_METHOD_PASSED
    assert fine_partial.r2_candidate
    assert fine_partial.conditions[0].accepted_max_pixels == 0
    assert len(fine_partial.conditions[0].fine_grid_aggregates) == 6


@pytest.mark.integration
def test_repair_empty_e_is_not_applicable_and_cannot_pass_core() -> None:
    pre, real, fine = _repair_inputs()
    spec = R1A_CORE_CONDITIONS[0]
    damage_pre = _pre_records(spec, (0.8,) * 8)
    pre[spec.condition_id] = damage_pre
    real[spec.condition_id] = tuple(
        evaluate_repair_point_unit(
            pre_record=record,
            point_kind="real_h",
            radius_pixels=None,
            scores=_triplet(1.0),
        )
        for record in damage_pre
    )
    del fine[spec.condition_id]
    evaluation = evaluate_r1b_repair(
        pre_records_by_condition=pre,
        real_h_records_by_condition=real,
        fine_grid_records_by_condition=fine,
        ordered_roster=_ROSTER,
    )
    first = evaluation.conditions[0]
    assert first.eligibility_status == "NOT_APPLICABLE/INSUFFICIENT_ELIGIBLE"
    assert not first.real_h_passed
    assert not first.fine_nonzero_prefix_passed
    assert evaluation.real_h_status == R1B_REPAIR_REAL_PARTIAL_CORE_PASSED
    assert evaluation.status == R1B_REPAIR_METHOD_NOT_READY


@pytest.mark.integration
def test_repair_failure_stays_in_full_eligible_denominator() -> None:
    pre, real, fine = _repair_inputs()
    spec = R1A_CORE_CONDITIONS[0]
    records = list(real[spec.condition_id])
    records[0] = evaluate_repair_point_unit(
        pre_record=pre[spec.condition_id][0],
        point_kind="real_h",
        radius_pixels=None,
        scores=None,
        errors=("real_h_recovery:RuntimeError",),
    )
    real[spec.condition_id] = tuple(records)
    evaluation = evaluate_r1b_repair(
        pre_records_by_condition=pre,
        real_h_records_by_condition=real,
        fine_grid_records_by_condition=fine,
        ordered_roster=_ROSTER,
    )
    aggregate = evaluation.conditions[0].real_h_aggregate
    assert aggregate.eligible_denominator == 8
    assert aggregate.valid_gain_count == 7
    assert aggregate.missing_gain_sentinel_count == 1
    assert not aggregate.passed
    assert evaluation.real_h_status == R1B_REPAIR_REAL_PARTIAL_CORE_PASSED
