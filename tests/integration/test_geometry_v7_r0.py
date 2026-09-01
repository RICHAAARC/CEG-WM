from __future__ import annotations

import json
from dataclasses import replace

from PIL import Image
import pytest

from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED, estimate_geometry
from cegwm.geometry_v7.r0 import (
    ContentScore,
    ImageQuality,
    R0Arm,
    R0ArmRecord,
    R0MultiplierRecords,
    R0NumericGates,
    R0Stage,
    R0UnitRecord,
    evaluate_r0_test,
    r0_record_payload,
    run_r0_four_arm_unit,
    select_r0_development_multiplier,
)


@pytest.mark.integration
def test_four_arm_routing_records_raw_deltas_false_positive_quality_and_denominators() -> None:
    unwatermarked = Image.new("RGB", (512, 512), (10, 10, 10))
    content = Image.new("RGB", (512, 512), (20, 20, 20))
    sync_calls: list[tuple[int, float]] = []
    score_calls: list[int] = []
    geometry_calls: list[int] = []
    quality_calls: list[tuple[int, int]] = []

    def sync_embedder(image: Image.Image, strength: float) -> Image.Image:
        value = image.getpixel((0, 0))[0]
        sync_calls.append((value, strength))
        return Image.new("RGB", (512, 512), (value + 1,) * 3)

    def content_scorer(image: Image.Image) -> ContentScore:
        value = image.getpixel((0, 0))[0]
        score_calls.append(value)
        weighted = value / 50.0
        return ContentScore(value / 100.0, value / 200.0, weighted, weighted - 0.4, value >= 20)

    def geometry_detector(image: Image.Image):
        value = image.getpixel((0, 0))[0]
        geometry_calls.append(value)
        return estimate_geometry(value / 10.0, CANONICAL_CORNERS_NORMALIZED)

    def quality_scorer(reference: Image.Image, candidate: Image.Image) -> ImageQuality:
        pair = (reference.getpixel((0, 0))[0], candidate.getpixel((0, 0))[0])
        quality_calls.append(pair)
        return ImageQuality(40.0, 0.99, 0.01)

    record = run_r0_four_arm_unit(
        unit_id="r0-0001",
        unwatermarked_final_rgb=unwatermarked,
        content_watermarked_final_rgb=content,
        residual_strength_multiplier=0.5,
        sync_embedder=sync_embedder,
        content_scorer=content_scorer,
        geometry_detector=geometry_detector,
        quality_scorer=quality_scorer,
    )
    assert sync_calls == [(10, 0.5), (20, 0.5)]
    assert score_calls == [10, 11, 20, 21]
    assert geometry_calls == [10, 11, 20, 21]
    assert quality_calls == [(10, 11), (20, 21)]
    assert [arm.arm for arm in record.arms] == [R0Arm.U, R0Arm.G, R0Arm.C, R0Arm.CG]
    assert record.base_syncseal_alpha == 0.20
    assert record.residual_strength_multiplier == 0.50
    assert dict(record.cg_minus_c_raw or ()) == pytest.approx(
        {"lf": 0.01, "hf": 0.005, "weighted_joint": 0.02, "margin": 0.02}
    )
    assert record.cg_c_content_flip is False
    assert record.g_content_false_positive is False
    assert (record.negative_arm_denominator, record.positive_arm_denominator) == (2, 2)
    assert record.failure_arm_denominator == 4 and record.failed_arm_count == 0
    payload = r0_record_payload(record)
    assert json.loads(json.dumps(payload, allow_nan=False))["arms"][3]["geometry"]["legal"] is True


@pytest.mark.integration
def test_failed_cg_stays_in_fixed_denominator_without_retry_or_fallback() -> None:
    calls: list[int] = []

    def failing_sync(image: Image.Image, strength: float) -> Image.Image:
        value = image.getpixel((0, 0))[0]
        calls.append(value)
        if value == 20:
            raise RuntimeError("fixed failure")
        return image.copy()

    record = run_r0_four_arm_unit(
        unit_id="r0-0002",
        unwatermarked_final_rgb=Image.new("RGB", (512, 512), (10, 10, 10)),
        content_watermarked_final_rgb=Image.new("RGB", (512, 512), (20, 20, 20)),
        residual_strength_multiplier=1.0,
        sync_embedder=failing_sync,
        content_scorer=lambda image: ContentScore(0.0, 0.0, 0.0, 0.0, False),
        geometry_detector=lambda image: estimate_geometry(0.0, CANONICAL_CORNERS_NORMALIZED),
        quality_scorer=lambda left, right: ImageQuality(40.0, 0.99, 0.01),
    )
    assert calls == [10, 20]
    cg = record.arms[3]
    assert cg.arm is R0Arm.CG and cg.image is None
    assert cg.errors == ("sync_embed:RuntimeError:fixed failure",)
    assert record.failure_arm_denominator == 4 and record.failed_arm_count == 1
    assert record.cg_minus_c_raw is None and record.cg_c_content_flip is None


@pytest.mark.integration
def test_numeric_gates_are_exactly_frozen_and_reject_drift() -> None:
    gates = R0NumericGates()
    assert gates.base_syncseal_alpha == 0.20
    assert gates.residual_strength_multipliers == (0.25, 0.50, 0.75, 1.00)
    assert (gates.min_mean_psnr, gates.min_mean_ssim, gates.max_mean_lpips) == (
        40.0, 0.98, 0.05
    )
    assert gates.identity_homography_max_error_normalized == 2.0 / 255.0
    with pytest.raises(ValueError, match="exact user-frozen"):
        R0NumericGates(min_mean_psnr=39.0)


def _aggregate_record(
    unit_id: str,
    multiplier: float,
    *,
    g_u_quality: ImageQuality = ImageQuality(40.0, 0.98, 0.05),
    cg_c_quality: ImageQuality = ImageQuality(40.0, 0.98, 0.05),
    g_false_positive: bool = False,
    cg_c_flip: bool = False,
    valid_identity: bool = True,
) -> R0UnitRecord:
    image = Image.new("RGB", (512, 512), "gray")
    geometry = (
        estimate_geometry(0.0, CANONICAL_CORNERS_NORMALIZED)
        if valid_identity
        else estimate_geometry(0.0, ((-1.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)))
    )
    false_score = ContentScore(0.0, 0.0, 0.0, -1.0, False)
    g_score = replace(false_score, positive=g_false_positive)
    c_score = ContentScore(0.2, 0.3, 1.0, 0.2, True)
    cg_score = replace(c_score, positive=not cg_c_flip)
    arms = (
        R0ArmRecord(R0Arm.U, image, false_score, None, None, ()),
        R0ArmRecord(R0Arm.G, image, g_score, geometry, g_u_quality, ()),
        R0ArmRecord(R0Arm.C, image, c_score, None, None, ()),
        R0ArmRecord(R0Arm.CG, image, cg_score, geometry, cg_c_quality, ()),
    )
    return R0UnitRecord(
        unit_id,
        0.20,
        multiplier,
        arms,
        (("lf", 0.0), ("hf", 0.0), ("weighted_joint", 0.0), ("margin", 0.0)),
        cg_c_flip,
        g_false_positive,
        2,
        2,
        4,
        0,
    )


@pytest.mark.integration
def test_development_uses_separate_complete_family_means_and_selects_first_pass() -> None:
    roster = tuple(f"reference-{index}" for index in range(4))
    qualities = (
        ImageQuality(39.0, 0.97, 0.06),
        ImageQuality(41.0, 0.99, 0.04),
        ImageQuality(40.0, 0.98, 0.05),
        ImageQuality(40.0, 0.98, 0.05),
    )
    records = tuple(
        _aggregate_record(unit_id, 0.25, g_u_quality=quality, cg_c_quality=quality)
        for unit_id, quality in zip(roster, qualities, strict=True)
    )
    selection = select_r0_development_multiplier(
        attempts=(R0MultiplierRecords(0.25, records),),
        ordered_reference_roster_first_4=roster,
    )
    assert selection.complete is True
    assert selection.selected_residual_strength_multiplier == 0.25
    aggregate = selection.attempts[0]
    assert aggregate.g_u_quality.passed and aggregate.cg_c_quality.passed
    assert aggregate.g_u_quality.mean_psnr == 40.0
    assert aggregate.g_u_quality.min_psnr == 39.0
    assert aggregate.identity_coordinate_valid_rate == 1.0


@pytest.mark.integration
def test_missing_pair_and_invalid_identity_remain_in_denominators_and_fail_closed() -> None:
    roster = tuple(f"reference-{index}" for index in range(4))
    records = list(_aggregate_record(unit_id, 0.25) for unit_id in roster)
    bad_arms = list(records[0].arms)
    near_but_outside_tolerance = (
        (1.0, 0.0, 0.01), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    )
    bad_arms[1] = replace(
        bad_arms[1], quality_to_unsynchronized_pair=None, geometry=replace(
            bad_arms[1].geometry,
            homography_current_to_canonical=near_but_outside_tolerance,
        )
    )
    records[0] = replace(records[0], arms=tuple(bad_arms), failed_arm_count=1)
    aggregate = select_r0_development_multiplier(
        attempts=(R0MultiplierRecords(0.25, tuple(records)),),
        ordered_reference_roster_first_4=roster,
    )
    assert aggregate.complete is False
    aggregate = aggregate.attempts[0]
    assert aggregate.g_u_quality.denominator == 4
    assert aggregate.g_u_quality.valid_count == 3
    assert aggregate.g_u_quality.mean_psnr is None
    assert aggregate.cg_c_quality.passed is True
    assert aggregate.identity_coordinate_valid_denominator == 8
    assert aggregate.identity_coordinate_valid_count == 7
    assert aggregate.carrier_compatibility_passed is False


@pytest.mark.integration
def test_all_grid_failures_stop_boundedly_and_test_runs_selected_multiplier_once() -> None:
    reference = tuple(f"reference-{index}" for index in range(4))
    attempts = tuple(
        R0MultiplierRecords(
            multiplier,
            tuple(
                _aggregate_record(unit_id, multiplier, g_false_positive=True)
                for unit_id in reference
            ),
        )
        for multiplier in R0NumericGates().residual_strength_multipliers
    )
    selection = select_r0_development_multiplier(
        attempts=attempts, ordered_reference_roster_first_4=reference
    )
    assert selection.complete is True
    assert selection.selected_residual_strength_multiplier is None
    assert selection.stop_reason is not None and "preregistered strength grid" in selection.stop_reason
    evaluation = tuple(f"evaluation-{index}" for index in range(8))
    with pytest.raises(ValueError, match="passing development selection"):
        evaluate_r0_test(
            records=tuple(_aggregate_record(unit_id, 0.50) for unit_id in evaluation),
            ordered_evaluation_roster_8=evaluation,
            development_selection=selection,
        )

    passing_selection = select_r0_development_multiplier(
        attempts=(
            attempts[0],
            R0MultiplierRecords(
                0.50,
                tuple(_aggregate_record(unit_id, 0.50) for unit_id in reference),
            ),
        ),
        ordered_reference_roster_first_4=reference,
    )
    assert passing_selection.selected_residual_strength_multiplier == 0.50
    result = evaluate_r0_test(
        records=tuple(_aggregate_record(unit_id, 0.50) for unit_id in evaluation),
        ordered_evaluation_roster_8=evaluation,
        development_selection=passing_selection,
    )
    assert result.stage is R0Stage.EVALUATION
    assert result.carrier_compatibility_passed is True
    assert result.cg_c_decision_flip_denominator == 8
    assert result.g_content_false_positive_denominator == 8
