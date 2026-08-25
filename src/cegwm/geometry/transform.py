"""Bounded D4-plus-similarity fitting with a fixed transform convention."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cegwm.geometry.types import SimilarityEstimate

_D4: tuple[NDArray[np.float64], ...] = (
    np.array(((1.0, 0.0), (0.0, 1.0))), np.array(((0.0, -1.0), (1.0, 0.0))),
    np.array(((-1.0, 0.0), (0.0, -1.0))), np.array(((0.0, 1.0), (-1.0, 0.0))),
    np.array(((-1.0, 0.0), (0.0, 1.0))), np.array(((1.0, 0.0), (0.0, -1.0))),
    np.array(((0.0, 1.0), (1.0, 0.0))), np.array(((0.0, -1.0), (-1.0, 0.0))),
)


def apply_h(points: ArrayLike, h_canonical_to_observed: ArrayLike) -> NDArray[np.float64]:
    """Map canonical xy points into observed xy coordinates."""

    pts = np.asarray(points, dtype=np.float64)
    h = np.asarray(h_canonical_to_observed, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or h.shape != (3, 3):
        raise ValueError("points must be [N,2] and H must be [3,3]")
    homogeneous = np.c_[pts, np.ones(len(pts))] @ h.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def transform_corners(h_canonical_to_observed: ArrayLike, frame_shape: tuple[int, int]) -> NDArray[np.float64]:
    """Return canonical frame corners in observed coordinates (H direction fixed)."""

    height, width = frame_shape
    if height <= 0 or width <= 0:
        raise ValueError("frame_shape dimensions must be positive")
    return apply_h(np.array(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1))), h_canonical_to_observed)


def _fit_similarity(source: NDArray[np.float64], target: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float, NDArray[np.float64]]:
    src_center, dst_center = source.mean(0), target.mean(0)
    centered_source, centered_target = source - src_center, target - dst_center
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    # D4 supplies every discrete reflection.  The remaining similarity must be
    # a proper rotation so a reflection cannot silently bypass D4 selection.
    orientation = np.eye(2, dtype=np.float64)
    orientation[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ orientation @ vt
    scale = float(np.sum((centered_source @ rotation) * centered_target) / np.sum(centered_source**2))
    translation = dst_center - scale * (src_center @ rotation)
    linear = scale * rotation.T
    h = np.eye(3, dtype=np.float64)
    h[:2, :2], h[:2, 2] = linear, translation
    angle = math.atan2(rotation[0, 1], rotation[0, 0])
    return h, scale, angle, translation


def estimate_bounded_similarity(
    canonical_points: ArrayLike,
    observed_points: ArrayLike,
    frame_shape: tuple[int, int],
    *,
    scale_bounds: tuple[float, float] = (0.25, 4.0),
    max_residual_rotation_radians: float = math.pi / 4,
    max_translation: float = 1_000_000.0,
    ambiguity_tolerance: float = 1e-10,
    total_reference_points: int | None = None,
) -> SimilarityEstimate:
    """Enumerate D4 orientation then fit residual rotation, scale and translation.

    Crop and crop-rescale are represented by reduced coverage and the same bounded
    isotropic canonical-to-observed similarity; no perspective model is fitted.
    """

    source, target = np.asarray(canonical_points, dtype=np.float64), np.asarray(observed_points, dtype=np.float64)
    if source.ndim != 2 or source.shape != target.shape or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("canonical and observed points must be matching [N,2] arrays with N >= 3")
    if not np.isfinite(source).all() or not np.isfinite(target).all() or np.linalg.matrix_rank(source - source.mean(0)) < 2:
        raise ValueError("points must be finite and canonical points non-collinear")
    if not (0 < scale_bounds[0] <= scale_bounds[1] and max_residual_rotation_radians >= 0 and max_translation >= 0):
        raise ValueError("invalid bounds")
    if total_reference_points is not None and total_reference_points < len(source):
        raise ValueError("total_reference_points cannot be less than observed correspondences")
    candidates: list[tuple[float, int, NDArray[np.float64], float, float, NDArray[np.float64]]] = []
    center = source.mean(0)
    for index, d4 in enumerate(_D4):
        oriented = (source - center) @ d4.T + center
        residual_h, scale, angle, translation = _fit_similarity(oriented, target)
        d4_h = np.eye(3, dtype=np.float64)
        d4_h[:2, :2] = d4
        d4_h[:2, 2] = center - d4 @ center
        h = residual_h @ d4_h
        predicted = apply_h(source, h)
        residual = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
        if scale_bounds[0] <= abs(scale) <= scale_bounds[1] and abs(angle) <= max_residual_rotation_radians and np.linalg.norm(translation) <= max_translation:
            candidates.append((residual, index, h, abs(scale), angle, translation))
    if not candidates:
        raise ValueError("no D4/similarity candidate satisfies bounds")
    candidates.sort(key=lambda item: (item[0], item[1]))
    residual, index, h, scale, angle, translation = candidates[0]
    gap = float(candidates[1][0] - residual) if len(candidates) > 1 else float("inf")
    corners = transform_corners(h, frame_shape)
    height, width = frame_shape
    valid = bool(np.isfinite(corners).all() and np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] < width) and np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] < height))
    boundary = float(np.mean((corners[:, 0] <= 0) | (corners[:, 0] >= width - 1) | (corners[:, 1] <= 0) | (corners[:, 1] >= height - 1)))
    coverage = 1.0 if total_reference_points is None else len(source) / total_reference_points
    return SimilarityEstimate(h, index, angle, scale, translation, len(source), residual, coverage, boundary, gap, corners, valid and gap > ambiguity_tolerance)
