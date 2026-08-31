from __future__ import annotations

import inspect

import numpy as np
import pytest

from cegwm.baselines.attacks import _verify_head_blob, rotation_10_bicubic_reflect_center_crop


@pytest.mark.unit
@pytest.mark.parametrize("height,width", [(41, 73), (73, 41), (42, 74), (74, 42)])
def test_rotation_shape_dtype_mask_and_determinism(height: int, width: int) -> None:
    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    first = rotation_10_bicubic_reflect_center_crop(rgb)
    second = rotation_10_bicubic_reflect_center_crop(rgb)

    assert first.rgb.shape == rgb.shape and first.rgb.dtype == np.uint8
    assert first.valid_mask.shape == rgb.shape[:2]
    assert set(np.unique(first.valid_mask)) <= {0, 1}
    assert np.array_equal(first.rgb, second.rgb)
    assert first.provenance["output_rgb_digest"] == second.provenance["output_rgb_digest"]
    assert first.provenance["positive_negative_pipeline_identical"]
    assert np.all(first.rgb == 255)  # black Pillow fill cannot leak into the center crop
    theta = np.radians(10.0)
    a, b = (width - 1) / 2.0, (height - 1) / 2.0
    expected_px = max(0, int(np.ceil(abs(np.cos(theta)) * a + abs(np.sin(theta)) * b + 2 - a)))
    expected_py = max(0, int(np.ceil(abs(np.sin(theta)) * a + abs(np.cos(theta)) * b + 2 - b)))
    assert (first.provenance["padding_x"], first.provenance["padding_y"]) == (expected_px, expected_py)
    assert first.provenance["implementation_exact"] != "0" * 40
    assert first.provenance["implementation_digest"].startswith("sha256:")
    assert first.provenance["input_rgb_digest"].startswith("sha256:")
    assert first.provenance["output_mask_digest"].startswith("sha256:")
    assert first.provenance["numpy_version"] and first.provenance["pillow_version"]


def test_positive_angle_is_visual_counter_clockwise_not_inverted() -> None:
    rgb = np.zeros((61, 81, 3), dtype=np.uint8)
    center_y, center_x = 30, 40
    rgb[center_y, center_x + 20, 0] = 255
    result = rotation_10_bicubic_reflect_center_crop(rgb)
    marker_y, marker_x = np.unravel_index(np.argmax(result.rgb[:, :, 0]), result.rgb.shape[:2])

    assert marker_x > center_x
    assert marker_y < center_y


def test_reflected_rgb_and_zero_mask_are_separate() -> None:
    rgb = np.full((41, 73, 3), (17, 34, 51), dtype=np.uint8)
    result = rotation_10_bicubic_reflect_center_crop(rgb)

    assert np.any(result.valid_mask == 0)
    assert np.any(result.rgb[result.valid_mask == 0] != 0)
    assert result.provenance["padding_mode_rgb"] != result.provenance["padding_mode_mask"]
    assert "sample_role" not in inspect.signature(rotation_10_bicubic_reflect_center_crop).parameters


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(2, 10, 3), (10, 2, 3), (3, 100, 3)])
def test_invalid_small_or_reflect_padding_inputs_fail_closed(shape: tuple[int, int, int]) -> None:
    with pytest.raises((TypeError, ValueError)):
        rotation_10_bicubic_reflect_center_crop(np.zeros(shape, dtype=np.uint8))


@pytest.mark.unit
def test_non_rgb_or_non_uint8_input_fails_closed() -> None:
    with pytest.raises(TypeError):
        rotation_10_bicubic_reflect_center_crop(np.zeros((41, 73), dtype=np.uint8))
    with pytest.raises(TypeError):
        rotation_10_bicubic_reflect_center_crop(np.zeros((41, 73, 3), dtype=np.float32))


@pytest.mark.unit
def test_staged_implementation_divergence_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="recorded HEAD blob"):
        _verify_head_blob(head_blob="a" * 40, working_blob="b" * 40, path_clean_against_head=True)
    with pytest.raises(RuntimeError, match="clean relative to HEAD"):
        _verify_head_blob(head_blob="a" * 40, working_blob="a" * 40, path_clean_against_head=False)
