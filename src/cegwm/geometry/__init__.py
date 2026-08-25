"""CPU-only Geometry-V1 synchronization primitives.

Geometry is deliberately limited to coordinate recovery.  It never produces a
watermark decision; callers must keep any content detector and threshold intact.
"""

from cegwm.geometry.qk_relation import keyed_qk_relation
from cegwm.geometry.rectifier import inverse_rectify
from cegwm.geometry.reliability import assess_reliability
from cegwm.geometry.transform import estimate_bounded_similarity, transform_corners

__all__ = [
    "assess_reliability",
    "estimate_bounded_similarity",
    "inverse_rectify",
    "keyed_qk_relation",
    "transform_corners",
]
