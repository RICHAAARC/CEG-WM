from __future__ import annotations

import inspect
import math
from typing import Callable

import numpy as np
import PIL
import pytest

import cegwm.baselines.attacks as attacks
from cegwm.baselines.attacks import (
    CENTER_CROP_80_RESTORE_ATTACK_ID,
    GAUSSIAN_BLUR_SIGMA_1PX_ATTACK_ID,
    JPEG_Q50_ATTACK_ID,
    RESIZE_50_BICUBIC_RESTORE_ATTACK_ID,
    FrozenAttackResult,
    center_crop_80_restore,
    gaussian_blur_sigma_1px,
    jpeg_q50,
    resize_50_bicubic_restore,
)


Attack = Callable[[np.ndarray], FrozenAttackResult]
ATTACKS: tuple[tuple[str, Attack], ...] = (
    (JPEG_Q50_ATTACK_ID, jpeg_q50),
    (RESIZE_50_BICUBIC_RESTORE_ATTACK_ID, resize_50_bicubic_restore),
    (CENTER_CROP_80_RESTORE_ATTACK_ID, center_crop_80_restore),
    (GAUSSIAN_BLUR_SIGMA_1PX_ATTACK_ID, gaussian_blur_sigma_1px),
)


def _structured_rgb(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(((37 * x + 19 * y) % 256, (91 * x + 53 * y) % 256, (x * y + 71 * y) % 256), axis=-1).astype(np.uint8)


@pytest.mark.unit
@pytest.mark.parametrize("height,width", [(7, 11), (8, 12)])
@pytest.mark.parametrize("attack_id,attack", ATTACKS)
def test_common_attacks_preserve_rgb_parameters_and_are_deterministic(
    height: int, width: int, attack_id: str, attack: Attack
) -> None:
    rgb = _structured_rgb(height, width)
    first, second = attack(rgb), attack(rgb)

    assert first.rgb.shape == rgb.shape and first.rgb.dtype == np.uint8
    assert np.array_equal(first.rgb, second.rgb)
    assert not np.array_equal(first.rgb, rgb)
    provenance = first.provenance
    assert provenance["attack_id"] == attack_id
    assert provenance["positive_negative_pipeline_identical"] is True
    assert (provenance["input_width"], provenance["input_height"]) == (width, height)
    assert (provenance["output_width"], provenance["output_height"]) == (width, height)
    assert provenance["output_crop_box"] == (0, 0, width, height)
    assert provenance["numpy_version"] == np.__version__ and provenance["pillow_version"] == PIL.__version__
    assert "sample_role" not in inspect.signature(attack).parameters


@pytest.mark.unit
def test_common_attacks_run_without_git_helper_api() -> None:
    assert not hasattr(attacks, "_run_git")
    assert not hasattr(attacks, "_verify_head_blob")
    assert not hasattr(attacks, "_verified_implementation_identity")
    rgb = _structured_rgb(9, 13)
    assert all(attack(rgb).rgb.shape == rgb.shape for _, attack in ATTACKS)


@pytest.mark.unit
def test_resize_and_crop_record_frozen_actual_geometry() -> None:
    rgb = _structured_rgb(7, 5)
    resized = resize_50_bicubic_restore(rgb).provenance
    assert (resized["small_width"], resized["small_height"]) == (2, 4)
    assert resized["rounding"] == "python_round_ties_to_even"
    assert resized["downsample_interpolation"] == "PIL.Image.Resampling.BICUBIC"
    assert resized["restore_interpolation"] == "PIL.Image.Resampling.BICUBIC"

    cropped = center_crop_80_restore(rgb).provenance
    linear_scale = math.sqrt(0.80)
    crop_width, crop_height = max(1, round(5 * linear_scale)), max(1, round(7 * linear_scale))
    assert (cropped["crop_width"], cropped["crop_height"]) == (crop_width, crop_height)
    assert cropped["crop_box"] == ((5 - crop_width) // 2, (7 - crop_height) // 2,
                                   (5 - crop_width) // 2 + crop_width, (7 - crop_height) // 2 + crop_height)
    assert cropped["retained_area_target"] == 0.80
    assert cropped["actual_retained_area"] == pytest.approx(crop_width * crop_height / (5 * 7))
    assert cropped["linear_scale"] == pytest.approx(linear_scale)


@pytest.mark.unit
def test_jpeg_and_blur_record_frozen_parameters() -> None:
    rgb = _structured_rgb(9, 13)
    jpeg = jpeg_q50(rgb).provenance
    assert (jpeg["jpeg_format"], jpeg["jpeg_quality"], jpeg["jpeg_subsampling"]) == ("JPEG", 50, 2)
    assert jpeg["jpeg_subsampling_description"] == "4:2:0"
    assert jpeg["jpeg_optimize"] is False and jpeg["jpeg_progressive"] is False
    assert jpeg["jpeg_codec_library"] == "Pillow" and jpeg["jpeg_codec_version"] == PIL.__version__
    blur = gaussian_blur_sigma_1px(rgb).provenance
    assert blur["sigma_px"] == 1.0 and blur["pillow_gaussian_blur_radius"] == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("_attack_id,attack", ATTACKS)
def test_common_attacks_fail_closed_on_invalid_input(_attack_id: str, attack: Attack) -> None:
    with pytest.raises(TypeError):
        attack(np.zeros((7, 11), dtype=np.uint8))
    with pytest.raises(TypeError):
        attack(np.zeros((7, 11, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        attack(np.zeros((2, 11, 3), dtype=np.uint8))
