"""Frozen PyTorch image-coordinate inverse warp for reliable geometry."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isclose, isfinite

import torch
import torch.nn.functional as functional

from main.shared.key_schedule import stable_json_utf8
from main.shared.rgb8 import (
    Rgb8ImageError,
    rgb8_image_digest,
    validate_rgb8_image,
    validate_rgb8_image_digest,
)

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
    source_image_digest: str
    rectified_image_digest: str
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


def _validated_rectification_inputs(
    image: torch.Tensor,
    estimation: GeometricTransformEstimation,
    reliability: GeometryReliabilityResult,
) -> tuple[torch.Tensor, torch.Tensor, float, int, int]:
    try:
        validated_image = validate_rgb8_image(image)
    except Rgb8ImageError as exc:
        raise ImageRectifierError(
            "image must be RGB uint8 [1,3,H,W] with H,W > 1"
        ) from exc
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
        device=validated_image.device, dtype=torch.float32
    )
    if matrix.shape != (2, 3) or not bool(torch.isfinite(matrix).all()):
        raise ImageRectifierError("transform matrix must be finite with [2,3] shape")
    token_crop_support = estimation.coverage
    if not isfinite(float(token_crop_support)) or not 0.0 <= float(
        token_crop_support
    ) <= 1.0:
        raise ImageRectifierError("estimator token coverage must be finite in [0,1]")
    return (
        validated_image,
        matrix,
        float(token_crop_support),
        int(validated_image.shape[2]),
        int(validated_image.shape[3]),
    )


def _replay_rectification(
    image: torch.Tensor,
    matrix: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return rectified, valid_support


def _matrix_value(
    matrix: torch.Tensor,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return tuple(
        tuple(float(value) for value in row)
        for row in matrix.detach().to(device="cpu")
    )


def image_rectifier(
    image: torch.Tensor,
    estimation: GeometricTransformEstimation,
    reliability: GeometryReliabilityResult,
) -> ImageRectificationResult:
    """Rectify only a reliability-approved estimator result."""

    (
        validated_image,
        matrix,
        token_crop_support,
        height,
        width,
    ) = _validated_rectification_inputs(image, estimation, reliability)
    rectified, valid_support = _replay_rectification(
        validated_image,
        matrix,
        height,
        width,
    )
    pixel_support = float(valid_support.to(dtype=torch.float32).mean())
    result = ImageRectificationResult(
        rectified_image=rectified,
        valid_support_mask=valid_support,
        source_image_digest=rgb8_image_digest(validated_image),
        rectified_image_digest=rgb8_image_digest(rectified),
        token_crop_support=token_crop_support,
        pixel_crop_support=pixel_support,
        crop_support=(token_crop_support, pixel_support),
        canonical_to_observed_matrix=_matrix_value(matrix),
        rectification_config_digest=_rectification_config_digest(height, width),
    )
    return result


def validate_image_rectification_result(
    result: ImageRectificationResult,
    source_image: torch.Tensor,
    estimation: GeometricTransformEstimation,
    reliability: GeometryReliabilityResult,
) -> ImageRectificationResult:
    """Replay and validate the frozen rectifier against its actual source."""

    if type(result) is not ImageRectificationResult:
        raise ImageRectifierError(
            "result must be ImageRectificationResult"
        )
    (
        validated_source,
        matrix,
        expected_token_support,
        height,
        width,
    ) = _validated_rectification_inputs(
        source_image,
        estimation,
        reliability,
    )
    expected_image, expected_support = _replay_rectification(
        validated_source,
        matrix,
        height,
        width,
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
    try:
        validate_rgb8_image_digest(result.source_image_digest)
        validate_rgb8_image_digest(result.rectified_image_digest)
    except Rgb8ImageError as exc:
        raise ImageRectifierError(
            "rectification image digest is invalid"
        ) from exc
    expected_source_digest = rgb8_image_digest(validated_source)
    expected_rectified_digest = rgb8_image_digest(expected_image)
    if (
        result.source_image_digest != expected_source_digest
        or result.rectified_image_digest != expected_rectified_digest
        or rgb8_image_digest(image) != result.rectified_image_digest
    ):
        raise ImageRectifierError(
            "rectification source or output image digest mismatch"
        )
    if not torch.equal(image, expected_image) or not torch.equal(
        support,
        expected_support,
    ):
        raise ImageRectifierError(
            "rectification image or support replay mismatch"
        )
    expected_matrix = _matrix_value(matrix)
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
        expected_support.to(dtype=torch.float32).mean()
    )
    if (
        not isclose(
            float(token_support),
            expected_token_support,
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
        height,
        width,
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
