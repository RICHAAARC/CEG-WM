"""Explicit, non-overlapping radial frequency masks for LF and HF carriers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FrequencyBandMasks:
    """Masks aligned with the last two dimensions of ``numpy.fft.rfft2``."""

    lf: NDArray[np.bool_]
    hf: NDArray[np.bool_]
    normalized_radius: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.lf.shape != self.hf.shape or self.lf.shape != self.normalized_radius.shape:
            raise ValueError("frequency masks and radius must have identical shapes")
        if np.any(self.lf & self.hf):
            raise ValueError("LF and HF frequency masks must be mutually exclusive")


def make_frequency_band_masks(
    height: int,
    width: int,
    *,
    lf_min_radius: float,
    lf_max_radius: float,
    hf_min_radius: float,
    hf_max_radius: float = 1.0,
) -> FrequencyBandMasks:
    """Build radial masks using radius normalized by the Nyquist corner.

    Radius 0 is DC and radius 1 is the two-dimensional Nyquist corner. A gap
    between the LF and HF intervals is allowed; overlap is rejected.
    """

    if not isinstance(height, int) or isinstance(height, bool) or height < 2:
        raise ValueError("height must be an integer of at least 2")
    if not isinstance(width, int) or isinstance(width, bool) or width < 2:
        raise ValueError("width must be an integer of at least 2")
    bounds = (lf_min_radius, lf_max_radius, hf_min_radius, hf_max_radius)
    if not all(np.isfinite(value) for value in bounds):
        raise ValueError("frequency radius bounds must be finite")
    if not (0.0 <= lf_min_radius < lf_max_radius < hf_min_radius < hf_max_radius <= 1.0):
        raise ValueError("frequency bands must satisfy 0 <= lf_min < lf_max < hf_min < hf_max <= 1")

    vertical = np.fft.fftfreq(height)[:, None]
    horizontal = np.fft.rfftfreq(width)[None, :]
    corner = np.hypot(0.5, 0.5)
    radius = np.hypot(vertical, horizontal) / corner
    lf = (radius >= lf_min_radius) & (radius <= lf_max_radius)
    hf = (radius >= hf_min_radius) & (radius <= hf_max_radius)
    if not np.any(lf) or not np.any(hf):
        raise ValueError("requested shape leaves an LF or HF band empty")
    return FrequencyBandMasks(lf=lf, hf=hf, normalized_radius=radius)
