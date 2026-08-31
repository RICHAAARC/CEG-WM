"""Deterministic, role-independent image attacks for Baseline-V1."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image


ROTATION_ATTACK_ID = "rotation_10_bicubic_reflect_center_crop_v1"
CENTER_FORMULA_ID = "pixel_center_w_minus_1_over_2_v1"
ROTATION_ANGLE_DEGREES = 10.0
BICUBIC_MARGIN_PIXELS = 2


@dataclass(frozen=True)
class RotationAttackResult:
    rgb: np.ndarray
    valid_mask: np.ndarray
    provenance: dict[str, Any]


def _sha256(array: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _require_rgb_uint8(rgb: Any) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError("rotation attack requires HxWx3 uint8 ordinary RGB")
    height, width = rgb.shape[:2]
    if height < 3 or width < 3:
        raise ValueError("rotation attack requires H and W at least 3")
    return rgb


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(("git", "-C", str(root), *arguments), check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _verified_implementation_identity() -> tuple[str, str]:
    """Bind executed module bytes to a clean checked-out Git exact."""

    module = Path(__file__).resolve()
    root = Path(_run_git(module.parent, "rev-parse", "--show-toplevel"))
    relative = module.relative_to(root).as_posix()
    if subprocess.run(("git", "-C", str(root), "diff", "--quiet", "--", relative), check=False).returncode != 0:
        raise RuntimeError("rotation implementation file must be clean before execution")
    index_blob = _run_git(root, "ls-files", "-s", "--", relative).split()[1]
    working_blob = _run_git(root, "hash-object", relative)
    if index_blob != working_blob:
        raise RuntimeError("rotation implementation bytes do not match the checked-out index blob")
    exact = _run_git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", exact):
        raise RuntimeError("rotation implementation checkout exact is invalid")
    return exact, "sha256:" + hashlib.sha256(module.read_bytes()).hexdigest()


def rotation_10_bicubic_reflect_center_crop(
    rgb: Any,
) -> RotationAttackResult:
    """Apply the frozen +10 degree visual counter-clockwise rotation.

    There is intentionally no sample-role argument: positive and unwatermarked
    negative inputs use this exact same function and implementation identity.
    """

    source = _require_rgb_uint8(rgb)
    implementation, module_digest = _verified_implementation_identity()
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
        "input_rgb_digest": _sha256(source),
        "output_rgb_digest": _sha256(output),
        "input_mask_digest": _sha256(original_mask),
        "output_mask_digest": _sha256(valid_mask),
        "implementation_exact": implementation,
        "implementation_digest": module_digest,
        "positive_negative_pipeline_identical": True,
    }
    return RotationAttackResult(output, valid_mask, provenance)
