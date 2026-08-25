import numpy as np

from cegwm.geometry.reliability import assess_reliability
from cegwm.geometry.transform import estimate_bounded_similarity


def test_reliability_is_unfitted_and_fail_closed_with_cycle_metric() -> None:
    points = np.array(((1, 1), (8, 1), (8, 8), (1, 8)), dtype=float)
    estimate = estimate_bounded_similarity(points, points + np.array((2, 3)), (16, 16))
    result = assess_reliability(estimate)
    assert not result.reliable
    assert result.reason == "unfitted_geometry_v1_fail_closed"
    assert result.cycle_error < 1e-10
