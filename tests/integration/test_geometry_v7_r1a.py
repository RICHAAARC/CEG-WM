from __future__ import annotations

import inspect
import math

from PIL import Image, ImageDraw
import pytest

from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    estimate_geometry,
)
from cegwm.geometry_v7.r1a import (
    R1A_ALL_CONDITIONS,
    R1A_BLOCKING_METHOD_CANARY_FAILED,
    R1A_BLOCKING_METHOD_CANARY_PASSED,
    R1A_CORE_CONDITIONS,
    R1A_DELTA_NONTRIVIAL_NORMALIZED,
    R1A_FIXED_UNIT_COUNT,
    R1A_MIN_IMPROVED_COUNT,
    R1A_SANITY_CONDITIONS,
    R1AConditionKind,
    R1AConditionRecords,
    aggregate_r1a_condition,
    apply_homography,
    condition_by_id,
    corner_rmse,
    detect_attacked_rgb,
    evaluate_r1a,
    evaluate_r1a_observation,
    r1a_truth_preflight,
    render_r1a_attack,
    truth_correspondences,
)


_ROSTER = tuple(f"evaluation-{index:02d}" for index in range(8))


def _record(spec, unit_id: str, correspondences, *, errors=()):
    geometry = None if correspondences is None else estimate_geometry(
        0.0, correspondences
    )
    return evaluate_r1a_observation(
        unit_id=unit_id,
        spec=spec,
        attacked_image=Image.new("RGB", (512, 512)),
        geometry=geometry,
        errors=errors,
    )


@pytest.mark.integration
def test_frozen_condition_identities_truth_and_direction_are_exact() -> None:
    assert len(R1A_SANITY_CONDITIONS) == 3
    assert len(R1A_CORE_CONDITIONS) == 10
    assert len(R1A_ALL_CONDITIONS) == 13
    assert tuple(spec.condition_id for spec in R1A_CORE_CONDITIONS) == (
        "core_rotation_neg15",
        "core_rotation_pos15",
        "core_fixed_canvas_zoom_0_8",
        "core_fixed_canvas_zoom_1_2",
        "core_translation_pos32_x",
        "core_translation_neg32_x",
        "core_translation_pos32_y",
        "core_translation_neg32_y",
        "core_offset_crop_rescale",
        "core_composite_c0_85_t16_neg16_r10",
    )
    rotation = condition_by_id("core_rotation_pos15")
    rotated_right = apply_homography(
        rotation.forward_canonical_to_observed, ((1.0, 0.0),)
    )[0]
    assert rotated_right[1] < 0.0

    translation = condition_by_id("core_translation_pos32_x")
    assert apply_homography(
        translation.forward_canonical_to_observed, ((0.0, 0.0),)
    )[0] == pytest.approx((64.0 / 511.0, 0.0))
    assert truth_correspondences(translation)[0] == pytest.approx(
        (-1.0 - 64.0 / 511.0, -1.0)
    )

    crop = condition_by_id("core_offset_crop_rescale")
    assert truth_correspondences(crop) == (
        (-0.875, -0.625),
        (0.625, -0.625),
        (0.625, 0.875),
        (-0.875, 0.875),
    )

    composite = condition_by_id("core_composite_c0_85_t16_neg16_r10")
    cosine = math.cos(math.radians(10.0))
    sine = math.sin(math.radians(10.0))
    expected = (
        (cosine + 32.0 / 511.0) / 0.85,
        (-sine - 32.0 / 511.0) / 0.85,
    )
    observed = apply_homography(
        composite.forward_canonical_to_observed, ((1.0, 0.0),)
    )[0]
    assert observed == pytest.approx(expected)
    recovered = apply_homography(
        composite.truth_observed_to_canonical, (observed,)
    )[0]
    assert recovered == pytest.approx((1.0, 0.0))


@pytest.mark.integration
def test_cpu_truth_preflight_and_corner_rmse_are_exact() -> None:
    assert corner_rmse(
        CANONICAL_CORNERS_NORMALIZED,
        CANONICAL_CORNERS_NORMALIZED,
    ) == 0.0
    shifted = tuple((x + 0.25, y) for x, y in CANONICAL_CORNERS_NORMALIZED)
    assert corner_rmse(CANONICAL_CORNERS_NORMALIZED, shifted) == pytest.approx(
        math.sqrt(4.0 * 0.25**2 / 8.0)
    )
    preflight = r1a_truth_preflight()
    assert preflight.passed
    assert len(preflight.entries) == 10
    assert all(
        entry.eligible
        and entry.identity_baseline_rmse > R1A_DELTA_NONTRIVIAL_NORMALIZED
        for entry in preflight.entries
    )


@pytest.mark.integration
def test_renderer_landmarks_sign_fill_and_single_core_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Image.new("RGB", (512, 512), "black")
    draw = ImageDraw.Draw(source)
    draw.rectangle((415, 251, 423, 259), fill="white")
    rotation = render_r1a_attack(
        source, condition_by_id("core_rotation_pos15")
    )
    weighted = [
        (x, y, rotation.getpixel((x, y))[0])
        for y in range(512)
        for x in range(512)
        if rotation.getpixel((x, y))[0]
    ]
    assert weighted
    total = sum(value for _, _, value in weighted)
    centroid_y = sum(y * value for _, y, value in weighted) / total
    assert centroid_y < 255.5

    translated = render_r1a_attack(
        source, condition_by_id("core_translation_pos32_x")
    )
    assert translated.getbbox() is not None
    assert translated.getbbox()[0] > source.getbbox()[0]
    assert translated.getpixel((0, 0)) == (0, 0, 0)

    calls: list[str] = []
    original_transform = Image.Image.transform

    def tracked_transform(self, *args, **kwargs):
        calls.append("transform")
        return original_transform(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "transform", tracked_transform)
    for spec in R1A_CORE_CONDITIONS:
        render_r1a_attack(source, spec)
    assert calls == ["transform"] * len(R1A_CORE_CONDITIONS)


@pytest.mark.integration
def test_detector_callable_receives_only_attacked_rgb() -> None:
    source = inspect.getsource(detect_attacked_rgb)
    assert tuple(inspect.signature(detect_attacked_rgb).parameters) == (
        "detector",
        "attacked_rgb",
    )
    assert all(
        forbidden not in source
        for forbidden in (
            "truth",
            "matrix",
            "attack_param",
            "prompt",
            "key",
            "latent",
            "original",
            "content",
        )
    )
    attacked = Image.new("RGB", (512, 512), "gray")
    seen: list[Image.Image] = []

    def detector(image):
        seen.append(image)
        return estimate_geometry(0.0, CANONICAL_CORNERS_NORMALIZED)

    assert detect_attacked_rgb(detector, attacked).legal
    assert seen == [attacked]


@pytest.mark.integration
def test_condition_gate_uses_fixed_eight_paired_median_and_six_improvements() -> None:
    spec = R1A_CORE_CONDITIONS[0]
    truth = truth_correspondences(spec)
    six_improve = tuple(
        _record(
            spec,
            unit_id,
            truth if index < R1A_MIN_IMPROVED_COUNT else CANONICAL_CORNERS_NORMALIZED,
        )
        for index, unit_id in enumerate(_ROSTER)
    )
    aggregate = aggregate_r1a_condition(
        condition_records=R1AConditionRecords(spec, six_improve),
        ordered_roster=_ROSTER,
    )
    assert aggregate.denominator == R1A_FIXED_UNIT_COUNT == 8
    assert aggregate.truth_eligible_count == aggregate.valid_prediction_count == 8
    assert aggregate.improved_count == R1A_MIN_IMPROVED_COUNT == 6
    assert aggregate.paired_delta_median is not None
    assert aggregate.paired_delta_median < 0.0
    assert aggregate.passed

    five_improve = tuple(
        _record(
            spec,
            unit_id,
            truth if index < 5 else CANONICAL_CORNERS_NORMALIZED,
        )
        for index, unit_id in enumerate(_ROSTER)
    )
    failed = aggregate_r1a_condition(
        condition_records=R1AConditionRecords(spec, five_improve),
        ordered_roster=_ROSTER,
    )
    assert failed.improved_count == 5 and not failed.passed


@pytest.mark.integration
def test_failure_stays_in_denominator_and_prevents_subset_median() -> None:
    spec = R1A_CORE_CONDITIONS[0]
    truth = truth_correspondences(spec)
    records = [
        _record(spec, unit_id, truth)
        for unit_id in _ROSTER
    ]
    records[3] = _record(
        spec,
        _ROSTER[3],
        None,
        errors=("geometry_detect:RuntimeError",),
    )
    aggregate = aggregate_r1a_condition(
        condition_records=R1AConditionRecords(spec, tuple(records)),
        ordered_roster=_ROSTER,
    )
    assert aggregate.denominator == aggregate.truth_eligible_count == 8
    assert aggregate.valid_prediction_count == 7
    assert aggregate.paired_delta_median is None
    assert not aggregate.passed


@pytest.mark.integration
def test_all_sanity_and_each_core_condition_block_independently() -> None:
    complete = tuple(
        R1AConditionRecords(
            spec,
            tuple(
                _record(spec, unit_id, truth_correspondences(spec))
                for unit_id in _ROSTER
            ),
        )
        for spec in R1A_ALL_CONDITIONS
    )
    passed = evaluate_r1a(condition_records=complete, ordered_roster=_ROSTER)
    assert passed.status == R1A_BLOCKING_METHOD_CANARY_PASSED
    assert passed.all_sanity_passed and passed.all_core_passed

    first_core = len(R1A_SANITY_CONDITIONS)
    failing_records = list(complete)
    spec = failing_records[first_core].spec
    failing_records[first_core] = R1AConditionRecords(
        spec,
        tuple(
            _record(spec, unit_id, CANONICAL_CORNERS_NORMALIZED)
            for unit_id in _ROSTER
        ),
    )
    failed = evaluate_r1a(
        condition_records=tuple(failing_records),
        ordered_roster=_ROSTER,
    )
    assert failed.status == R1A_BLOCKING_METHOD_CANARY_FAILED
    assert failed.all_sanity_passed
    assert not failed.all_core_passed
    assert not failed.blocking_method_canary_passed
