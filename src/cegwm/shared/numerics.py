"""Actual-dtype budget accounting and blind content-score primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class BudgetMeasurement:
    dtype: str
    base_l2: float
    perturbation_l2: float
    relative_l2: float


def _floating_array(value: ArrayLike, *, name: str) -> NDArray[np.floating]:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must use a floating dtype")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def relative_l2_measurement(base: ArrayLike, candidate: ArrayLike) -> BudgetMeasurement:
    """Measure relative L2 after requiring the same actual dtype and shape."""

    base_array = _floating_array(base, name="base")
    candidate_array = _floating_array(candidate, name="candidate")
    if base_array.shape != candidate_array.shape:
        raise ValueError("base and candidate must have identical shapes")
    if base_array.dtype != candidate_array.dtype:
        raise ValueError("base and candidate must have the same actual dtype")
    base64 = base_array.astype(np.float64)
    candidate64 = candidate_array.astype(np.float64)
    base_l2 = float(np.linalg.norm(base64.ravel()))
    if base_l2 == 0.0:
        raise ValueError("relative L2 budget is undefined for a zero-norm base")
    perturbation_l2 = float(np.linalg.norm((candidate64 - base64).ravel()))
    return BudgetMeasurement(
        dtype=base_array.dtype.str,
        base_l2=base_l2,
        perturbation_l2=perturbation_l2,
        relative_l2=perturbation_l2 / base_l2,
    )


def assert_relative_l2_budget(
    base: ArrayLike,
    candidate: ArrayLike,
    max_relative_l2: float,
    *,
    atol: float = 1e-12,
) -> BudgetMeasurement:
    """Fail if the actual-dtype candidate exceeds the declared total budget."""

    if not np.isfinite(max_relative_l2) or max_relative_l2 < 0.0:
        raise ValueError("max_relative_l2 must be finite and non-negative")
    measurement = relative_l2_measurement(base, candidate)
    if measurement.relative_l2 > max_relative_l2 + atol:
        raise ValueError(
            f"actual-dtype relative L2 {measurement.relative_l2:.12g} exceeds "
            f"budget {max_relative_l2:.12g}"
        )
    return measurement


def project_sum_to_relative_l2_budget(
    base: ArrayLike,
    deltas: Sequence[ArrayLike],
    max_relative_l2: float,
) -> tuple[NDArray[np.floating], BudgetMeasurement]:
    """Apply all carrier deltas under one budget measured after dtype casting.

    The returned candidate has ``base.dtype``. A monotone bisection over the
    common delta scale handles float16/float32 rounding without silently
    measuring a higher-precision proposal instead of the actual tensor.
    """

    base_array = _floating_array(base, name="base")
    if not deltas:
        raise ValueError("at least one carrier delta is required")
    if not np.isfinite(max_relative_l2) or max_relative_l2 <= 0.0:
        raise ValueError("max_relative_l2 must be finite and positive")
    total = np.zeros(base_array.shape, dtype=np.float64)
    for index, delta in enumerate(deltas):
        delta_array = _floating_array(delta, name=f"delta[{index}]")
        if delta_array.shape != base_array.shape:
            raise ValueError("every carrier delta must match the base shape")
        total += delta_array.astype(np.float64)
    if not np.any(total):
        candidate = base_array.copy()
        return candidate, relative_l2_measurement(base_array, candidate)

    base64 = base_array.astype(np.float64)

    def cast_candidate(scale: float) -> NDArray[np.floating]:
        return (base64 + scale * total).astype(base_array.dtype)

    full = cast_candidate(1.0)
    full_measurement = relative_l2_measurement(base_array, full)
    if full_measurement.relative_l2 <= max_relative_l2:
        return full, full_measurement

    low = 0.0
    high = 1.0
    best = base_array.copy()
    best_measurement = relative_l2_measurement(base_array, best)
    for _ in range(80):
        middle = (low + high) / 2.0
        candidate = cast_candidate(middle)
        measurement = relative_l2_measurement(base_array, candidate)
        if measurement.relative_l2 <= max_relative_l2:
            low = middle
            best = candidate
            best_measurement = measurement
        else:
            high = middle
    return best, assert_relative_l2_budget(base_array, best, max_relative_l2)


def masked_normalized_correlation(
    observation: ArrayLike,
    carrier: ArrayLike,
    mask: ArrayLike,
) -> float:
    """Score image-derived coefficients against a keyed carrier on one band."""

    observed = np.asarray(observation, dtype=np.float64)
    expected = np.asarray(carrier, dtype=np.float64)
    selected = np.asarray(mask, dtype=np.bool_)
    if observed.shape != expected.shape or observed.shape != selected.shape:
        raise ValueError("observation, carrier, and mask must have identical shapes")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(expected)):
        raise ValueError("score inputs must contain only finite values")
    if np.count_nonzero(selected) < 2:
        raise ValueError("score mask must select at least two coefficients")
    observed_values = observed[selected]
    expected_values = expected[selected]
    observed_values = observed_values - observed_values.mean()
    expected_values = expected_values - expected_values.mean()
    denominator = float(np.linalg.norm(observed_values) * np.linalg.norm(expected_values))
    if denominator == 0.0:
        raise ValueError("normalized correlation requires non-constant selected values")
    return float(np.dot(observed_values, expected_values) / denominator)
