"""Typed, image-only data exchanged by Geometry-V1 primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class QKRelation:
    """A numeric Q/K relation and its independently keyed projection."""

    relation: NDArray[np.float64]
    projection: float
    coverage: float
    gap: float
    wrong_key_margin: float


@dataclass(frozen=True, slots=True)
class SimilarityEstimate:
    """Canonical-to-observed affine map; H always has this direction."""

    h_canonical_to_observed: NDArray[np.float64]
    d4_index: int
    residual_rotation_radians: float
    scale: float
    translation: NDArray[np.float64]
    inlier_count: int
    residual: float
    coverage: float
    boundary_fraction: float
    uniqueness_gap: float
    corners: NDArray[np.float64]
    valid_corners: bool


@dataclass(frozen=True, slots=True)
class ReliabilityAssessment:
    """Metrics-only result: V1 has no fitted acceptance thresholds."""

    reliable: bool
    reason: str
    cycle_error: float
    valid_corners: bool


@dataclass(frozen=True, slots=True)
class RectifiedImage:
    """Inverse-warp result; false support denotes unavailable crop content."""

    image: NDArray[np.generic]
    valid_support: NDArray[np.bool_]
