"""Deterministic, role-independent image attacks for Baseline-V1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import PIL
from PIL import Image, ImageFilter


ROTATION_ATTACK_ID = "rotation_10_bicubic_reflect_center_crop_v1"
CENTER_FORMULA_ID = "pixel_center_w_minus_1_over_2_v1"
ROTATION_ANGLE_DEGREES = 10.0
BICUBIC_MARGIN_PIXELS = 2
JPEG_Q50_ATTACK_ID = "jpeg_q50"
RESIZE_50_BICUBIC_RESTORE_ATTACK_ID = "resize_50_bicubic_restore"
CENTER_CROP_80_RESTORE_ATTACK_ID = "center_crop_80_restore"
GAUSSIAN_BLUR_SIGMA_1PX_ATTACK_ID = "gaussian_blur_sigma_1px"


@dataclass(frozen=True)
class RotationAttackResult:
    rgb: np.ndarray
    valid_mask: np.ndarray
    provenance: dict[str, Any]


@dataclass(frozen=True)
class FrozenAttackResult:
    """Result and reproducibility binding for a non-rotation frozen attack."""

    rgb: np.ndarray
    provenance: dict[str, Any]


def _require_rgb_uint8(rgb: Any) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError("rotation attack requires HxWx3 uint8 ordinary RGB")
    height, width = rgb.shape[:2]
    if height < 3 or width < 3:
        raise ValueError("rotation attack requires H and W at least 3")
    return rgb


def _common_provenance(
    attack_id: str,
    source: np.ndarray,
    output: np.ndarray,
) -> dict[str, Any]:
    height, width = source.shape[:2]
    output_height, output_width = output.shape[:2]
    return {
        "attack_id": attack_id,
        "input_width": width,
        "input_height": height,
        "output_width": output_width,
        "output_height": output_height,
        "output_crop_box": (0, 0, output_width, output_height),
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "positive_negative_pipeline_identical": True,
    }


def _frozen_result(attack_id: str, source: np.ndarray, output: np.ndarray, **parameters: Any) -> FrozenAttackResult:
    if output.shape != source.shape or output.dtype != np.uint8:
        raise RuntimeError("frozen image attack must preserve HxWx3 uint8")
    provenance = _common_provenance(attack_id, source, output)
    provenance.update(parameters)
    return FrozenAttackResult(output, provenance)


def jpeg_q50(rgb: Any) -> FrozenAttackResult:
    """Apply the frozen Pillow RGB JPEG Q50 4:2:0 round trip."""

    source = _require_rgb_uint8(rgb)
    with BytesIO() as encoded:
        Image.fromarray(source, mode="RGB").save(
            encoded,
            format="JPEG",
            quality=50,
            subsampling=2,
            optimize=False,
            progressive=False,
        )
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            output = np.asarray(decoded.convert("RGB").copy(), dtype=np.uint8)
    return _frozen_result(
        JPEG_Q50_ATTACK_ID,
        source,
        output,
        jpeg_format="JPEG",
        jpeg_quality=50,
        jpeg_subsampling=2,
        jpeg_subsampling_description="4:2:0",
        jpeg_optimize=False,
        jpeg_progressive=False,
        jpeg_codec_library="Pillow",
        jpeg_codec_version=PIL.__version__,
    )


def resize_50_bicubic_restore(rgb: Any) -> FrozenAttackResult:
    """Apply the frozen Python-round 50% bicubic downsample and restore."""

    source = _require_rgb_uint8(rgb)
    height, width = source.shape[:2]
    small_width, small_height = max(1, round(width * 0.50)), max(1, round(height * 0.50))
    image = Image.fromarray(source, mode="RGB")
    small = image.resize((small_width, small_height), Image.Resampling.BICUBIC)
    output = np.asarray(small.resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8)
    return _frozen_result(
        RESIZE_50_BICUBIC_RESTORE_ATTACK_ID,
        source,
        output,
        scale=0.50,
        rounding="python_round_ties_to_even",
        small_width=small_width,
        small_height=small_height,
        downsample_interpolation="PIL.Image.Resampling.BICUBIC",
        restore_interpolation="PIL.Image.Resampling.BICUBIC",
    )


def center_crop_80_restore(rgb: Any) -> FrozenAttackResult:
    """Apply the frozen 80%-area center crop and bicubic restore."""

    source = _require_rgb_uint8(rgb)
    height, width = source.shape[:2]
    target_area, linear_scale = 0.80, math.sqrt(0.80)
    crop_width, crop_height = max(1, round(width * linear_scale)), max(1, round(height * linear_scale))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    crop_box = (left, top, left + crop_width, top + crop_height)
    cropped = Image.fromarray(source, mode="RGB").crop(crop_box)
    output = np.asarray(cropped.resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8)
    return _frozen_result(
        CENTER_CROP_80_RESTORE_ATTACK_ID,
        source,
        output,
        retained_area_target=target_area,
        actual_retained_area=(crop_width * crop_height) / (width * height),
        linear_scale=linear_scale,
        rounding="python_round_ties_to_even",
        crop_width=crop_width,
        crop_height=crop_height,
        crop_box=crop_box,
        restore_interpolation="PIL.Image.Resampling.BICUBIC",
    )


def gaussian_blur_sigma_1px(rgb: Any) -> FrozenAttackResult:
    """Apply Pillow GaussianBlur radius 1.0, frozen here as sigma_px=1.0."""

    source = _require_rgb_uint8(rgb)
    output = np.asarray(
        Image.fromarray(source, mode="RGB").filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.uint8
    )
    return _frozen_result(
        GAUSSIAN_BLUR_SIGMA_1PX_ATTACK_ID,
        source,
        output,
        sigma_px=1.0,
        pillow_gaussian_blur_radius=1.0,
    )


def rotation_10_bicubic_reflect_center_crop(
    rgb: Any,
) -> RotationAttackResult:
    """Apply the frozen +10 degree visual counter-clockwise rotation.

    There is intentionally no sample-role argument: positive and unwatermarked
    negative inputs use this exact same function and implementation identity.
    """

    source = _require_rgb_uint8(rgb)
    height, width = source.shape[:2]
    theta = math.radians(ROTATION_ANGLE_DEGREES)
    a, b = (width - 1) / 2.0, (height - 1) / 2.0
    e_x = abs(math.cos(theta)) * a + abs(math.sin(theta)) * b
    e_y = abs(math.sin(theta)) * a + abs(math.cos(theta)) * b
    p_x = max(0, math.ceil(e_x + BICUBIC_MARGIN_PIXELS - a))
    p_y = max(0, math.ceil(e_y + BICUBIC_MARGIN_PIXELS - b))
    if p_x >= width or p_y >= height:
        raise ValueError("reflection padding exceeds NumPy reflect semantics")

    padded = np.pad(source, ((p_y, p_y), (p_x, p_x), (0, 0)), mode="reflect")
    center = (p_x + (width - 1) / 2.0, p_y + (height - 1) / 2.0)
    crop_box = (p_x, p_y, p_x + width, p_y + height)
    rgb_rotated = Image.fromarray(padded, mode="RGB").rotate(
        angle=ROTATION_ANGLE_DEGREES,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        center=center,
        fillcolor=(0, 0, 0),
    )
    output = np.asarray(rgb_rotated.crop(crop_box), dtype=np.uint8)

    original_mask = np.ones((height, width), dtype=np.uint8)
    padded_mask = np.pad(original_mask, ((p_y, p_y), (p_x, p_x)), mode="constant", constant_values=0)
    mask_rotated = Image.fromarray(padded_mask, mode="L").rotate(
        angle=ROTATION_ANGLE_DEGREES,
        resample=Image.Resampling.NEAREST,
        expand=False,
        center=center,
        fillcolor=0,
    )
    valid_mask = np.asarray(mask_rotated.crop(crop_box), dtype=np.uint8)
    if valid_mask.shape != (height, width) or not bool(np.isin(valid_mask, (0, 1)).all()):
        raise RuntimeError("rotated valid mask must be canonical HxW {0,1}")
    if output.shape != source.shape or output.dtype != np.uint8:
        raise RuntimeError("rotation output must preserve HxWx3 uint8")

    provenance = {
        "attack_id": ROTATION_ATTACK_ID,
        "angle_degrees": ROTATION_ANGLE_DEGREES,
        "angle_convention": "Pillow visual counter-clockwise positive angle",
        "center_formula_id": CENTER_FORMULA_ID,
        "center_padded": center,
        "padding_x": p_x,
        "padding_y": p_y,
        "bicubic_margin_pixels": BICUBIC_MARGIN_PIXELS,
        "padding_mode_rgb": "numpy.reflect_edge_not_repeated",
        "padding_mode_mask": "numpy.constant_zero",
        "rgb_interpolation": "PIL.Image.Resampling.BICUBIC",
        "mask_interpolation": "PIL.Image.Resampling.NEAREST",
        "crop_box": crop_box,
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "positive_negative_pipeline_identical": True,
    }
    return RotationAttackResult(output, valid_mask, provenance)
