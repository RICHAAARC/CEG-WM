from __future__ import annotations

import pytest

from cegwm.baselines.protocol import (
    CLEAN_CONFIRMATION_NEGATIVES,
    EVALUATION_PHYSICAL_UNITS,
    TARGET_FPR_UPPER_BOUND,
    THRESHOLD_FREEZE_NEGATIVES,
    one_sided_clopper_pearson_upper,
    operating_point_violation,
    per_method_scale,
    rotation_execution_blocker,
)


@pytest.mark.unit
def test_exact_clean_confirmation_ucb_is_the_formal_admission_gate() -> None:
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
