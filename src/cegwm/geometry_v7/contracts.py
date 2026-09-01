"""Geometry-V7 coordinate-only contracts.

The fixed observed/output-canvas corners are ``q`` in TL/TR/BR/BL order.
SyncSeal predicts their correspondences ``p_hat`` in original/canonical image
coordinates, and ``p_hat ~ H_observed_to_canonical @ q``.  Nothing in this
module can create or alter a content-watermark decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Sequence

import torch


PUBLIC_IMAGE_HEIGHT = 512
PUBLIC_IMAGE_WIDTH = 512
SYNCSEAL_MODEL_SIZE = 256
CORNER_ORDER = ("TL", "TR", "BR", "BL")
CANONICAL_CORNERS_NORMALIZED = (
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
)


class GeometryStatus(str, Enum):
    RELIABLE = "RELIABLE"
    UNRELIABLE = "UNRELIABLE"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class D4Transform(str, Enum):
    IDENTITY = "identity"
    ROTATE_90_CCW = "rotate_90_ccw"
    ROTATE_180 = "rotate_180"
    ROTATE_270_CCW = "rotate_270_ccw"
    MIRROR_LEFT_RIGHT = "mirror_left_right"
    MIRROR_LEFT_RIGHT_THEN_ROTATE_90_CCW = "mirror_left_right_then_rotate_90_ccw"
    MIRROR_LEFT_RIGHT_THEN_ROTATE_180 = "mirror_left_right_then_rotate_180"
    MIRROR_LEFT_RIGHT_THEN_ROTATE_270_CCW = "mirror_left_right_then_rotate_270_ccw"


Matrix3x3 = tuple[tuple[float, float, float], ...]
Corners4 = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class GeometryEstimate:
    status: GeometryStatus
    uncalibrated_sync_logit: float | None
    raw_syncseal_corners: Corners4 | None
    observed_corners_in_canonical_normalized: Corners4 | None
    homography_observed_to_canonical: Matrix3x3 | None
    legal: bool
    basic_observable: bool
    error: str | None = None

    @property
    def corners_current_normalized(self) -> Corners4 | None:
        """Deprecated alias; values are canonical correspondences, not locations."""

        return self.observed_corners_in_canonical_normalized

    @property
    def homography_current_to_canonical(self) -> Matrix3x3 | None:
        """Deprecated alias for :attr:`homography_observed_to_canonical`."""

        return self.homography_observed_to_canonical

    @classmethod
    def error_record(cls, error: BaseException | str) -> "GeometryEstimate":
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        return cls(GeometryStatus.ERROR, None, None, None, None, False, False, message)


def pixel_center_to_normalized(index: Real, size: int) -> float:
    """Map an integer-centered pixel lattice ``[0, size-1]`` to ``[-1, 1]``."""

    if isinstance(index, bool) or not isinstance(index, Real):
        raise TypeError("pixel-center index must be real")
    if not isinstance(size, int) or isinstance(size, bool) or size < 2:
        raise ValueError("pixel-center normalization requires integer size >= 2")
    value = float(index)
    if not math.isfinite(value):
        raise ValueError("pixel-center index must be finite")
    return 2.0 * value / float(size - 1) - 1.0


def normalized_to_pixel_center(value: Real, size: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("normalized coordinate must be real")
    if not isinstance(size, int) or isinstance(size, bool) or size < 2:
        raise ValueError("pixel-center denormalization requires integer size >= 2")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("normalized coordinate must be finite")
    return (numeric + 1.0) * float(size - 1) / 2.0


def syncseal_raw_to_public_normalized(
    raw_corners: Sequence[Real] | Sequence[Sequence[Real]],
) -> Corners4:
    """Apply the official 256-grid unwarp conversion before public normalization."""

    raw = _corners_tensor(raw_corners)
    model_centers = torch.round(raw * (SYNCSEAL_MODEL_SIZE / 2.0) + (SYNCSEAL_MODEL_SIZE / 2.0))
    public_normalized = 2.0 * model_centers / float(SYNCSEAL_MODEL_SIZE - 1) - 1.0
    return tuple(tuple(float(value) for value in row) for row in public_normalized.tolist())


def _corners_tensor(corners: Sequence[Real] | Sequence[Sequence[Real]]) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(corners, dtype=torch.float64).reshape(4, 2)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("corners must contain exactly 8 real values") from error
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("corners must be finite")
    return tensor


def _strict_convex_in_declared_order(corners: torch.Tensor) -> bool:
    edges = torch.roll(corners, shifts=-1, dims=0) - corners
    following = torch.roll(edges, shifts=-1, dims=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    return bool(torch.all(crosses > 0.0) or torch.all(crosses < 0.0))


def homography_observed_to_canonical(
    observed_corners_in_canonical_normalized: Sequence[Real]
    | Sequence[Sequence[Real]],
) -> Matrix3x3:
    """Solve ``q -> p_hat`` for the fixed observed/output-canvas square."""

    target = _corners_tensor(observed_corners_in_canonical_normalized)
    if not _strict_convex_in_declared_order(target):
        raise ValueError(
            "predicted canonical TL/TR/BR/BL correspondences must form a "
            "strict convex quadrilateral"
        )
    source = torch.tensor(CANONICAL_CORNERS_NORMALIZED, dtype=torch.float64)
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(source.tolist(), target.tolist(), strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)
    try:
        solution = torch.linalg.solve(
            torch.tensor(rows, dtype=torch.float64), torch.tensor(rhs, dtype=torch.float64)
        )
    except RuntimeError as error:
        raise ValueError("corners do not define a unique finite homography") from error
    matrix = torch.cat((solution, torch.ones(1, dtype=torch.float64))).reshape(3, 3)
    if not bool(torch.isfinite(matrix).all()) or float(torch.linalg.det(matrix)) == 0.0:
        raise ValueError("corners do not define a finite invertible homography")
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def homography_current_to_canonical(
    observed_corners_in_canonical_normalized: Sequence[Real]
    | Sequence[Sequence[Real]],
) -> Matrix3x3:
    """Deprecated name for :func:`homography_observed_to_canonical`.

    The argument uses the corrected ``p_hat`` canonical-correspondence
    semantics; this compatibility alias does not restore the former reversed
    solve direction.
    """

    return homography_observed_to_canonical(
        observed_corners_in_canonical_normalized
    )


def estimate_geometry(
    uncalibrated_sync_logit: Real,
    observed_corners_in_canonical_normalized: Sequence[Real]
    | Sequence[Sequence[Real]],
    *,
    raw_syncseal_corners: Sequence[Real] | Sequence[Sequence[Real]] | None = None,
) -> GeometryEstimate:
    """Validate raw SyncSeal output without calibrating it into reliability."""

    if isinstance(uncalibrated_sync_logit, bool) or not isinstance(uncalibrated_sync_logit, Real):
        return GeometryEstimate.error_record("uncalibrated SyncSeal logit must be real")
    logit = float(uncalibrated_sync_logit)
    if not math.isfinite(logit):
        return GeometryEstimate.error_record("uncalibrated SyncSeal logit must be finite")
    try:
        corners_tensor = _corners_tensor(
            observed_corners_in_canonical_normalized
        )
        corners = tuple(tuple(float(value) for value in row) for row in corners_tensor.tolist())
        raw = None
        if raw_syncseal_corners is not None:
            raw_tensor = _corners_tensor(raw_syncseal_corners)
            raw = tuple(tuple(float(value) for value in row) for row in raw_tensor.tolist())
        homography = homography_observed_to_canonical(corners)
    except ValueError as error:
        return GeometryEstimate(
            GeometryStatus.UNSUPPORTED, logit, None, None, None, False, False, str(error)
        )
    # P0/R0 has no approved reliability calibration.  A legal raw estimate is
    # therefore observable but deliberately cannot be labelled RELIABLE.
    return GeometryEstimate(
        GeometryStatus.UNRELIABLE, logit, raw, corners, homography, True, True, None
    )


_D4_MATRICES: dict[D4Transform, Matrix3x3] = {
    D4Transform.IDENTITY: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    D4Transform.ROTATE_90_CCW: ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    D4Transform.ROTATE_180: ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    D4Transform.ROTATE_270_CCW: ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    D4Transform.MIRROR_LEFT_RIGHT: ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    D4Transform.MIRROR_LEFT_RIGHT_THEN_ROTATE_90_CCW: (
        (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    ),
    D4Transform.MIRROR_LEFT_RIGHT_THEN_ROTATE_180: (
        (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)
    ),
    D4Transform.MIRROR_LEFT_RIGHT_THEN_ROTATE_270_CCW: (
        (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    ),
}


def d4_homography(transform: D4Transform) -> Matrix3x3:
    if not isinstance(transform, D4Transform):
        raise TypeError("D4 transform must be a D4Transform")
    return _D4_MATRICES[transform]


def compose_d4_observed_to_canonical(
    raw_observed_to_canonical: Sequence[Sequence[Real]], transform: D4Transform
) -> Matrix3x3:
    """Freeze D4 composition as ``H_candidate = D_canonical @ H_raw``."""

    try:
        raw = torch.as_tensor(
            raw_observed_to_canonical, dtype=torch.float64
        ).reshape(3, 3)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("raw homography must contain exactly 9 real values") from error
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("raw homography must be finite")
    composed = torch.tensor(d4_homography(transform), dtype=torch.float64) @ raw
    if not bool(torch.isfinite(composed).all()):
        raise ValueError("composed homography must be finite")
    return tuple(tuple(float(value) for value in row) for row in composed.tolist())


def compose_d4_current_to_canonical(
    raw_observed_to_canonical: Sequence[Sequence[Real]], transform: D4Transform
) -> Matrix3x3:
    """Deprecated alias for :func:`compose_d4_observed_to_canonical`."""

    return compose_d4_observed_to_canonical(raw_observed_to_canonical, transform)


__all__ = [
    "CANONICAL_CORNERS_NORMALIZED",
    "CORNER_ORDER",
    "D4Transform",
    "GeometryEstimate",
    "GeometryStatus",
    "PUBLIC_IMAGE_HEIGHT",
    "PUBLIC_IMAGE_WIDTH",
    "SYNCSEAL_MODEL_SIZE",
    "compose_d4_current_to_canonical",
    "compose_d4_observed_to_canonical",
    "d4_homography",
    "estimate_geometry",
    "homography_current_to_canonical",
    "homography_observed_to_canonical",
    "normalized_to_pixel_center",
    "pixel_center_to_normalized",
    "syncseal_raw_to_public_normalized",
]
