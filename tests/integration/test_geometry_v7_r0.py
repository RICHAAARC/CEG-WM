from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PIL import Image
import pytest

from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED, estimate_geometry
from cegwm.geometry_v7.r0 import (
    ContentScore,
    ImageQuality,
    PairedContentDecision,
    R0Arm,
    R0ArmRecord,
    R0MultiplierRecords,
    R0NumericGates,
    R0Stage,
    R0UnitRecord,
    _identity_coordinate_valid,
    evaluate_r0_test,
    r0_record_payload,
    r0_producer_failure_record,
    run_r0_four_arm_unit,
    select_r0_development_multiplier,
)
from cegwm.protocol.content_chain import load_content_chain_contract


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixed_rosters() -> tuple[tuple[str, ...], tuple[str, ...]]:
    contract = load_content_chain_contract(_REPO_ROOT)
    return (
        tuple(unit.unit_id for unit in contract.reference_roster[:4]),
        tuple(unit.unit_id for unit in contract.evaluation_roster),
    )


def _raw_content(lf: float, hf: float, weighted: float) -> ContentScore:
    return ContentScore(
        lf,
        hf,
        weighted,
        (lf - 0.1,) * 16,
        (hf - 0.1,) * 16,
        (weighted - 0.1,) * 16,
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
        wrong = weighted + 0.1 if value < 20 else weighted - 0.1
        return ContentScore(
            value / 100.0,
            value / 200.0,
            weighted,
            (0.0,) * 16,
            (0.0,) * 16,
            (wrong,) * 16,
        )

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
        {
            "lf": 0.01,
            "hf": 0.005,
            "weighted_joint": 0.02,
            "gate_a_margin": 0.0,
            "gate_b_margin": 0.0,
            "margin": 0.0,
        }
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
        content_scorer=lambda image: _raw_content(0.0, 0.0, 0.0),
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


@pytest.mark.integration
def test_identity_coordinate_valid_uses_direct_ordered_strict_convex_corners() -> None:
    tolerance = R0NumericGates().identity_homography_max_error_normalized
    identity = estimate_geometry(0.0, CANONICAL_CORNERS_NORMALIZED)
    one_official_grid_step = replace(
        identity,
        corners_current_normalized=(
            (-1.0, -1.0),
            (257.0 / 255.0, -1.0),
            (257.0 / 255.0, 257.0 / 255.0),
            (-1.0, 257.0 / 255.0),
        ),
    )
    assert _identity_coordinate_valid(one_official_grid_step, tolerance)
    assert not _identity_coordinate_valid(
        replace(
            one_official_grid_step,
            corners_current_normalized=(
                (-1.0, -1.0),
                (257.0 / 255.0 + 1e-12, -1.0),
                (257.0 / 255.0, 257.0 / 255.0),
                (-1.0, 257.0 / 255.0),
            ),
        ),
        tolerance,
    )
    assert _identity_coordinate_valid(
        replace(
            identity,
            homography_current_to_canonical=(
                (1.0, 0.0, 0.5),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        ),
        tolerance,
    )

    invalid_estimates = (
        replace(
            identity,
            corners_current_normalized=(
                (-1.0, -1.0),
                (float("nan"), -1.0),
                (1.0, 1.0),
                (-1.0, 1.0),
            ),
        ),
        replace(
            identity,
            corners_current_normalized=(
                (-1.0, -1.0),
                (-1.0, 1.0),
                (1.0, 1.0),
                (1.0, -1.0),
            ),
        ),
        replace(
            identity,
            corners_current_normalized=(
                (-1.0, -1.0),
                (1.0, -1.0),
                (-0.5, 0.0),
                (-1.0, 1.0),
            ),
        ),
        replace(
            identity,
            corners_current_normalized=(
                (-1.0, -1.0),
                (1.0, -1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
            ),
        ),
        replace(identity, legal=False),
        replace(identity, error="reported geometry error"),
    )
    assert all(
        not _identity_coordinate_valid(estimate, tolerance)
        for estimate in invalid_estimates
    )


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
    false_score = _raw_content(0.0, 0.0, 0.0)
    g_score = _raw_content(0.0, 0.0, 0.5 if g_false_positive else -1.0)
    c_score = _raw_content(0.2, 0.3, 1.0)
    cg_score = _raw_content(0.2, 0.3, -1.0 if cg_c_flip else 1.0)
    g_decision = PairedContentDecision(
        R0Arm.U,
        g_score.gate_a_margin,
        g_score.weighted_joint - false_score.weighted_joint,
        min(g_score.gate_a_margin, g_score.weighted_joint - false_score.weighted_joint),
        g_false_positive,
    )
    c_decision = PairedContentDecision(R0Arm.U, 0.1, 1.0, 0.1, True)
    cg_positive = not cg_c_flip
    cg_gate_b = cg_score.weighted_joint - g_score.weighted_joint
    cg_decision = PairedContentDecision(
        R0Arm.G,
        cg_score.gate_a_margin,
        cg_gate_b,
        min(cg_score.gate_a_margin, cg_gate_b),
        cg_positive,
    )
    arms = (
        R0ArmRecord(R0Arm.U, image, false_score, None, None, None, ()),
        R0ArmRecord(R0Arm.G, image, g_score, g_decision, geometry, g_u_quality, ()),
        R0ArmRecord(R0Arm.C, image, c_score, c_decision, None, None, ()),
        R0ArmRecord(R0Arm.CG, image, cg_score, cg_decision, geometry, cg_c_quality, ()),
    )
    return R0UnitRecord(
        unit_id,
        0.20,
        multiplier,
        arms,
        (
            ("lf", 0.0),
            ("hf", 0.0),
            ("weighted_joint", 0.0),
            ("gate_a_margin", 0.0),
            ("gate_b_margin", 0.0),
            ("margin", 0.0),
        ),
        cg_c_flip,
        g_false_positive,
        2,
        2,
        4,
        0,
    )


@pytest.mark.integration
def test_atomic_content_pair_producer_failure_remains_in_all_fixed_denominators() -> None:
    record = r0_producer_failure_record(
        unit_id="content-adaptive-v2-0001",
        residual_strength_multiplier=0.25,
        error=RuntimeError("real pair stopped"),
    )
    assert record.failed_arm_count == record.failure_arm_denominator == 4
    assert record.g_content_false_positive is None and record.cg_c_content_flip is None
    assert all(arm.image is None for arm in record.arms)
    assert all(
        arm.errors == ("content_pair_producer:RuntimeError:real pair stopped",)
        for arm in record.arms
    )


@pytest.mark.integration
def test_development_uses_separate_complete_family_means_and_selects_first_pass() -> None:
    roster, _ = _fixed_rosters()
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
        repo_root=_REPO_ROOT,
        attempts=(R0MultiplierRecords(0.25, records),),
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
    roster, _ = _fixed_rosters()
    records = list(_aggregate_record(unit_id, 0.25) for unit_id in roster)
    bad_arms = list(records[0].arms)
    corners_just_outside_tolerance = (
        (-1.0, -1.0),
        (1.0 + 2.0 / 255.0 + 1e-12, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    )
    bad_arms[1] = replace(
        bad_arms[1], quality_to_unsynchronized_pair=None, geometry=replace(
            bad_arms[1].geometry,
            corners_current_normalized=corners_just_outside_tolerance,
        )
    )
    records[0] = replace(records[0], arms=tuple(bad_arms), failed_arm_count=1)
    aggregate = select_r0_development_multiplier(
        repo_root=_REPO_ROOT,
        attempts=(R0MultiplierRecords(0.25, tuple(records)),),
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
    reference, evaluation = _fixed_rosters()
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
        repo_root=_REPO_ROOT,
        attempts=attempts,
    )
    assert selection.complete is True
    assert selection.selected_residual_strength_multiplier is None
    assert selection.stop_reason is not None and "preregistered strength grid" in selection.stop_reason
    with pytest.raises(ValueError, match="passing development selection"):
        evaluate_r0_test(
            repo_root=_REPO_ROOT,
            records=tuple(_aggregate_record(unit_id, 0.50) for unit_id in evaluation),
            development_selection=selection,
        )

    passing_selection = select_r0_development_multiplier(
        repo_root=_REPO_ROOT,
        attempts=(
            attempts[0],
            R0MultiplierRecords(
                0.50,
                tuple(_aggregate_record(unit_id, 0.50) for unit_id in reference),
            ),
        ),
    )
    assert passing_selection.selected_residual_strength_multiplier == 0.50
    result = evaluate_r0_test(
        repo_root=_REPO_ROOT,
        records=tuple(_aggregate_record(unit_id, 0.50) for unit_id in evaluation),
        development_selection=passing_selection,
    )
    assert result.stage is R0Stage.EVALUATION
    assert result.carrier_compatibility_passed is True
    assert result.cg_c_decision_flip_denominator == 8
    assert result.g_content_false_positive_denominator == 8


@pytest.mark.integration
def test_contract_rosters_reject_order_or_identity_drift() -> None:
    reference, evaluation = _fixed_rosters()
    development_records = tuple(_aggregate_record(unit_id, 0.25) for unit_id in reference)
    with pytest.raises(ValueError, match="complete fixed roster in exact order"):
        select_r0_development_multiplier(
            repo_root=_REPO_ROOT,
            attempts=(R0MultiplierRecords(0.25, development_records[::-1]),),
        )
    with pytest.raises(ValueError, match="complete fixed roster in exact order"):
        select_r0_development_multiplier(
            repo_root=_REPO_ROOT,
            attempts=(
                R0MultiplierRecords(
                    0.25,
                    tuple(_aggregate_record(f"arbitrary-{index}", 0.25) for index in range(4)),
                ),
            ),
        )

    selection = select_r0_development_multiplier(
        repo_root=_REPO_ROOT,
        attempts=(R0MultiplierRecords(0.25, development_records),),
    )
    evaluation_records = tuple(_aggregate_record(unit_id, 0.25) for unit_id in evaluation)
    with pytest.raises(ValueError, match="complete fixed roster in exact order"):
        evaluate_r0_test(
            repo_root=_REPO_ROOT,
            records=(evaluation_records[1], evaluation_records[0], *evaluation_records[2:]),
            development_selection=selection,
        )


@pytest.mark.integration
def test_test_boundary_revalidates_selection_history_fail_closed() -> None:
    reference, evaluation = _fixed_rosters()
    first_fails = R0MultiplierRecords(
        0.25,
        tuple(
            _aggregate_record(unit_id, 0.25, g_false_positive=True)
            for unit_id in reference
        ),
    )
    second_passes = R0MultiplierRecords(
        0.50,
        tuple(_aggregate_record(unit_id, 0.50) for unit_id in reference),
    )
    selection = select_r0_development_multiplier(
        repo_root=_REPO_ROOT,
        attempts=(first_fails, second_passes),
    )
    evaluation_records = tuple(_aggregate_record(unit_id, 0.50) for unit_id in evaluation)

    forged_selections = (
        replace(selection, attempts=()),
        replace(selection, selected_residual_strength_multiplier=0.75),
        replace(selection, attempts=(selection.attempts[1],)),
        replace(
            selection,
            attempts=(
                selection.attempts[0],
                replace(selection.attempts[1], roster=reference[::-1]),
            ),
        ),
    )
    for forged in forged_selections:
        with pytest.raises(ValueError):
            evaluate_r0_test(
                repo_root=_REPO_ROOT,
                records=evaluation_records,
                development_selection=forged,
            )

    incomplete = select_r0_development_multiplier(
        repo_root=_REPO_ROOT,
        attempts=(first_fails,),
    )
    assert incomplete.complete is False
    with pytest.raises(ValueError, match="completed passing development selection"):
        evaluate_r0_test(
            repo_root=_REPO_ROOT,
            records=evaluation_records,
            development_selection=incomplete,
        )
