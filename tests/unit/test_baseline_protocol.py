from __future__ import annotations

import pytest

from cegwm.baselines.protocol import (
    CLEAN_CONFIRMATION_NEGATIVES,
    EVALUATION_PHYSICAL_UNITS,
    FORMAL_ATTACK_CONDITIONS,
    TARGET_FPR_UPPER_BOUND,
    THRESHOLD_FREEZE_NEGATIVES,
    one_sided_clopper_pearson_upper,
    operating_point_violation,
    per_method_scale,
    rotation_execution_blocker,
)


@pytest.mark.unit
def test_exact_clean_confirmation_ucb_is_report_only_diagnostic() -> None:
    upper_zero = one_sided_clopper_pearson_upper(0, CLEAN_CONFIRMATION_NEGATIVES)
    upper_one = one_sided_clopper_pearson_upper(1, CLEAN_CONFIRMATION_NEGATIVES)
    upper_four = one_sided_clopper_pearson_upper(4, CLEAN_CONFIRMATION_NEGATIVES)

    assert upper_zero == pytest.approx(1.0 - 0.05 ** (1.0 / 3000.0), rel=0.0, abs=1e-15)
    assert upper_zero <= TARGET_FPR_UPPER_BOUND
    assert upper_one > TARGET_FPR_UPPER_BOUND
    assert upper_one > upper_zero
    assert upper_four > TARGET_FPR_UPPER_BOUND
    assert not operating_point_violation(0, CLEAN_CONFIRMATION_NEGATIVES)
    assert operating_point_violation(1, CLEAN_CONFIRMATION_NEGATIVES)
    assert operating_point_violation(4, CLEAN_CONFIRMATION_NEGATIVES)


@pytest.mark.unit
def test_frozen_scale_and_rotation_blocker() -> None:
    scale = per_method_scale()

    assert (THRESHOLD_FREEZE_NEGATIVES, CLEAN_CONFIRMATION_NEGATIVES, EVALUATION_PHYSICAL_UNITS) == (2000, 3000, 1000)
    assert scale.evaluation_detections == 12_000
    assert scale.threshold_freeze_detections + scale.clean_confirmation_detections + scale.evaluation_detections == 17_000
    assert scale.source_generation_images == 7_000
    assert scale.attack_derivative_images == 10_000
    assert scale.quality_pair_comparisons == 6_000
    assert rotation_execution_blocker() is None


@pytest.mark.unit
def test_frozen_common_attack_parameters_and_order() -> None:
    assert [(condition.family, condition.condition) for condition in FORMAL_ATTACK_CONDITIONS] == [
        ("clean", "clean_no_attack"),
        ("compression", "jpeg_q50"),
        ("geometric", "resize_50_bicubic_restore"),
        ("geometric", "center_crop_80_restore"),
        ("photometric", "gaussian_blur_sigma_1px"),
        ("geometric", "rotation_10_bicubic_reflect_center_crop_v1"),
    ]
    assert FORMAL_ATTACK_CONDITIONS[1].parameters == (
        ("format", "JPEG"), ("quality", "50"), ("subsampling", "2 (4:2:0)"),
        ("optimize", "false"), ("progressive", "false"),
    )
    assert FORMAL_ATTACK_CONDITIONS[2].parameters == (
        ("scale", "0.50"), ("rounding", "python_round_ties_to_even"),
        ("downsample_interpolation", "PIL.Image.Resampling.BICUBIC"),
        ("restore_interpolation", "PIL.Image.Resampling.BICUBIC"),
    )
    assert FORMAL_ATTACK_CONDITIONS[3].parameters == (
        ("retained_area_target", "0.80"), ("linear_scale", "sqrt(0.80)"),
        ("rounding", "python_round_ties_to_even"),
        ("restore_interpolation", "PIL.Image.Resampling.BICUBIC"),
    )
    assert FORMAL_ATTACK_CONDITIONS[4].parameters == (("sigma_px", "1.0"), ("pillow_radius", "1.0"))
