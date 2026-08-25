"""Fail-closed reliability reporting for unfitted Geometry-V1."""

from __future__ import annotations

import numpy as np

from cegwm.geometry.transform import apply_h
from cegwm.geometry.types import ReliabilityAssessment, SimilarityEstimate


def assess_reliability(estimate: SimilarityEstimate) -> ReliabilityAssessment:
    """Report raw cycle/corner evidence but never accept before calibration."""

    inverse = np.linalg.inv(estimate.h_canonical_to_observed)
    cycle = apply_h(estimate.corners, inverse)
    expected = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    # Scale-free corner cycle: map reconstructed unit square back through H.
    forward = apply_h(expected, estimate.h_canonical_to_observed)
    returned = apply_h(forward, inverse)
    cycle_error = float(np.max(np.abs(returned - expected)))
    if not np.isfinite(cycle).all():
        cycle_error = float("inf")
    return ReliabilityAssessment(False, "unfitted_geometry_v1_fail_closed", cycle_error, estimate.valid_corners)
