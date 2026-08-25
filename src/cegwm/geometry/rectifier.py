"""Nearest-neighbour inverse rectification with explicit valid support."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cegwm.geometry.transform import apply_h
from cegwm.geometry.types import RectifiedImage


def inverse_rectify(
    observed_image: ArrayLike,
    h_canonical_to_observed: ArrayLike,
    output_shape: tuple[int, int],
) -> RectifiedImage:
    """Sample observed content into canonical coordinates without inpainting."""

    image = np.asarray(observed_image)
    height, width = output_shape
    if image.ndim not in (2, 3) or height <= 0 or width <= 0:
        raise ValueError("image must be 2D/3D and output_shape positive")
    yy, xx = np.indices((height, width), dtype=np.float64)
    observed = apply_h(np.c_[xx.ravel(), yy.ravel()], h_canonical_to_observed)
    ox, oy = np.rint(observed[:, 0]).astype(int), np.rint(observed[:, 1]).astype(int)
    support = (ox >= 0) & (ox < image.shape[1]) & (oy >= 0) & (oy < image.shape[0])
    result = np.zeros((height, width) + image.shape[2:], dtype=image.dtype)
    result.reshape((-1,) + image.shape[2:])[support] = image[oy[support], ox[support]]
    return RectifiedImage(result, support.reshape(height, width))
