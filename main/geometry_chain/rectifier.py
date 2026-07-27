"""Frozen PyTorch image-coordinate inverse warp for reliable geometry."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isclose, isfinite

import torch
import torch.nn.functional as functional

from main.shared.key_schedule import stable_json_utf8

from .reliability import (
    GeometryReliabilityError,
    GeometryReliabilityResult,
    validate_geometry_reliability_result,
)
from .transform_estimator import (
    GeometricTransformEstimation,
)

RECTIFICATION_CANDIDATE_ID = "rectification_similarity"


class ImageRectifierError(ValueError):
    """Image or transform cannot be rectified under the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ImageRectificationResult:
    """Rectified RGB8 image plus independent token and pixel support."""

    rectified_image: torch.Tensor
    valid_support_mask: torch.Tensor
    token_crop_support: float
    pixel_crop_support: float
    crop_support: tuple[float, float]
    canonical_to_observed_matrix: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    rectification_config_digest: str


def _rectification_config_digest(height: int, width: int) -> str:
    identity = {
        "align_corners": True,
        "candidate_id": RECTIFICATION_CANDIDATE_ID,
        "image_interpolation": "bilinear",
        "image_padding": "border",
        "image_shape": [1, 3, height, width],
        "output_quantization": "clamp_floor_uint8",
        "support_interpolation": "nearest",
        "support_padding": "zeros",
    }
    return sha256(stable_json_utf8(identity)).hexdigest()


def image_rectifier(
    image: torch.Tensor,
    estimation: GeometricTransformEstimation,
    reliability: GeometryReliabilityResult,
) -> ImageRectificationResult:
    """Rectify only a reliability-approved estimator result."""

    if not isinstance(image, torch.Tensor):
        raise ImageRectifierError("image must be a torch.Tensor")
    if image.dtype is not torch.uint8 or image.ndim != 4:
        raise ImageRectifierError("image must be RGB uint8 [1,3,H,W]")
    if (
        image.shape[0] != 1
        or image.shape[1] != 3
        or image.shape[2] <= 1
        or image.shape[3] <= 1
    ):
        raise ImageRectifierError("image must be RGB uint8 [1,3,H,W] with H,W > 1")
    try:
        replayed_reliable = validate_geometry_reliability_result(
            reliability,
            estimation,
        )
    except GeometryReliabilityError as exc:
        raise ImageRectifierError(
            "geometry reliability result validation failed"
        ) from exc
    if not replayed_reliable:
        raise ImageRectifierError("geometry reliability does not allow rectification")
    matrix = estimation.transform.tensor().to(
        device=image.device, dtype=torch.float32
    )
    if matrix.shape != (2, 3) or not bool(torch.isfinite(matrix).all()):
        raise ImageRectifierError("transform matrix must be finite with [2,3] shape")
    token_crop_support = estimation.coverage
    if not isfinite(float(token_crop_support)) or not 0.0 <= float(
        token_crop_support
    ) <= 1.0:
        raise ImageRectifierError("estimator token coverage must be finite in [0,1]")

    height, width = int(image.shape[2]), int(image.shape[3])
    try:
        grid = functional.affine_grid(
            matrix.unsqueeze(0),
            size=image.shape,
            align_corners=True,
        )
        image_float = image.to(dtype=torch.float32) / 255.0
        warped_float = functional.grid_sample(
            image_float,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        rectified = torch.floor(
            torch.clamp(warped_float, 0.0, 1.0) * 255.0
        ).to(dtype=torch.uint8)
        support_input = torch.ones(
            (1, 1, height, width),
            dtype=torch.float32,
            device=image.device,
        )
        support = functional.grid_sample(
            support_input,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        valid_support = support > 0.5
    except RuntimeError as exc:
        raise ImageRectifierError("PyTorch affine rectification failed") from exc
    if rectified.shape != image.shape or valid_support.shape != (
        1,
        1,
        height,
        width,
    ):
        raise ImageRectifierError("rectification output shape mismatch")
    if not bool(valid_support.any()):
        raise ImageRectifierError("rectification has no valid pixel support")
    pixel_support = float(valid_support.to(dtype=torch.float32).mean())
    matrix_value = tuple(
        tuple(float(value) for value in row)
        for row in matrix.to(device="cpu")
    )
    result = ImageRectificationResult(
        rectified_image=rectified,
        valid_support_mask=valid_support,
        token_crop_support=float(token_crop_support),
        pixel_crop_support=pixel_support,
        crop_support=(float(token_crop_support), pixel_support),
        canonical_to_observed_matrix=matrix_value,
        rectification_config_digest=_rectification_config_digest(height, width),
    )
    return validate_image_rectification_result(
        result,
        estimation,
        reliability,
    )


def validate_image_rectification_result(
    result: ImageRectificationResult,
    estimation: GeometricTransformEstimation,
    reliability: GeometryReliabilityResult,
) -> ImageRectificationResult:
    """Validate rectifier-owned image, support, transform, and config relations."""

    if type(result) is not ImageRectificationResult:
        raise ImageRectifierError(
            "result must be ImageRectificationResult"
        )
    try:
        replayed_reliable = validate_geometry_reliability_result(
            reliability,
            estimation,
        )
    except GeometryReliabilityError as exc:
        raise ImageRectifierError(
            "rectification reliability validation failed"
        ) from exc
    if not replayed_reliable:
        raise ImageRectifierError(
            "rectification result requires reliable geometry"
        )
    image = result.rectified_image
    support = result.valid_support_mask
    if (
        not isinstance(image, torch.Tensor)
        or image.dtype is not torch.uint8
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or image.shape[2] <= 1
        or image.shape[3] <= 1
    ):
        raise ImageRectifierError(
            "rectified image must be RGB uint8 [1,3,H,W]"
        )
    if (
        not isinstance(support, torch.Tensor)
        or support.dtype is not torch.bool
        or support.shape != (1, 1, image.shape[2], image.shape[3])
        or not bool(support.any())
    ):
        raise ImageRectifierError(
            "valid support mask must be non-empty bool [1,1,H,W]"
        )
    expected_matrix = tuple(
        tuple(float(value) for value in row)
        for row in estimation.transform.tensor()
    )
    if result.canonical_to_observed_matrix != expected_matrix:
        raise ImageRectifierError(
            "rectification matrix does not match estimation"
        )
    token_support = result.token_crop_support
    pixel_support = result.pixel_crop_support
    if (
        isinstance(token_support, bool)
        or not isinstance(token_support, (int, float))
        or not isfinite(float(token_support))
        or isinstance(pixel_support, bool)
        or not isinstance(pixel_support, (int, float))
        or not isfinite(float(pixel_support))
        or not 0.0 <= float(token_support) <= 1.0
        or not 0.0 <= float(pixel_support) <= 1.0
    ):
        raise ImageRectifierError(
            "rectification support values must be finite in [0,1]"
        )
    expected_pixel_support = float(
        support.to(dtype=torch.float32).mean()
    )
    if (
        not isclose(
            float(token_support),
            float(estimation.coverage),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isclose(
            float(pixel_support),
            expected_pixel_support,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or result.crop_support
        != (float(token_support), float(pixel_support))
    ):
        raise ImageRectifierError(
            "rectification support values are inconsistent"
        )
    expected_config_digest = _rectification_config_digest(
        int(image.shape[2]),
        int(image.shape[3]),
    )
    if result.rectification_config_digest != expected_config_digest:
        raise ImageRectifierError(
            "rectification config digest mismatch"
        )
    return result


__all__ = [
    "ImageRectificationResult",
    "ImageRectifierError",
    "image_rectifier",
    "validate_image_rectification_result",
]
