"""CPU measurements for a single-direction reconstruction-channel diagnostic.

Features must already have been extracted from ordinary RGB by the runtime.
These measurements do not fit a reader, encode a payload, or adjudicate a
scientific result.  Finite differences are descriptive even when an actual
dtype perturbation has a component outside the requested probe direction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

ArrayValue = np.ndarray | torch.Tensor
ARMS = ("zero", "plus", "minus")


def _array(value: ArrayValue, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a real floating dtype")
        array = value.detach().to(dtype=torch.float64).numpy().copy()
    elif isinstance(value, np.ndarray):
        if value.dtype.kind != "f":
            raise TypeError(f"{name} must have a real floating dtype")
        array = np.array(value, dtype=np.float64, copy=True)
    else:
        raise TypeError(f"{name} must be a numpy array or CPU torch tensor")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _norm(value: np.ndarray) -> float:
    flat = value.reshape(-1)
    scale = float(np.max(np.abs(flat)))
    result = 0.0 if scale == 0 else float(scale * np.linalg.norm(flat / scale))
    if not np.isfinite(result):
        raise ValueError("L2 measurement overflowed")
    return result


def _vector(value: np.ndarray) -> list[float]:
    return value.reshape(-1).tolist()


def measure_probe_geometry(
    base: ArrayValue,
    plus: ArrayValue,
    minus: ArrayValue,
    probe: ArrayValue,
    *,
    nominal_amplitude: float,
) -> dict[str, Any]:
    """Measure actual post-cast perturbations against a unit probe.

    ``nominal_amplitude`` is the requested positive L2 displacement along the
    unit probe, not a relative budget.  Collapsed or reversed amplitudes remain
    in the returned record; finite-difference callers then report them invalid.
    """
    if not np.isfinite(nominal_amplitude) or nominal_amplitude <= 0:
        raise ValueError("nominal_amplitude must be finite and positive")
    if not all(isinstance(value, (np.ndarray, torch.Tensor)) for value in (base, plus, minus)):
        raise TypeError("base, plus and minus must be numpy arrays or CPU torch tensors")
    if not (str(base.dtype) == str(plus.dtype) == str(minus.dtype)):
        raise ValueError("base, plus and minus must retain the same actual dtype")
    arrays = [_array(value, name) for value, name in zip((base, plus, minus, probe), ("base", "plus", "minus", "probe"))]
    base64, plus64, minus64, probe64 = arrays
    if any(value.shape != base64.shape for value in arrays[1:]):
        raise ValueError("base, perturbed states and probe must have the same shape")
    probe_l2 = _norm(probe64)
    if probe_l2 == 0:
        raise ValueError("probe must be nonzero")
    unit = probe64 / probe_l2
    delta_plus, delta_minus = plus64 - base64, minus64 - base64
    a_plus = float(np.vdot(unit, delta_plus))
    a_minus = float(np.vdot(unit, delta_minus))
    return {
        "actual_dtype": str(base.dtype),
        "shape": list(base64.shape),
        "probe_l2_before_normalization": probe_l2,
        "nominal_amplitude": float(nominal_amplitude),
        "plus_signed_amplitude": a_plus,
        "minus_signed_amplitude": a_minus,
        "actual_signed_step": a_plus - a_minus,
        "signed_midpoint": (a_plus + a_minus) / 2.0,
        "plus_nominal_residual": a_plus - float(nominal_amplitude),
        "minus_nominal_residual": a_minus + float(nominal_amplitude),
        "plus_delta_l2": _norm(delta_plus),
        "minus_delta_l2": _norm(delta_minus),
        "plus_off_axis_l2": _norm(delta_plus - a_plus * unit),
        "minus_off_axis_l2": _norm(delta_minus - a_minus * unit),
        "antisymmetry_residual_l2": _norm(delta_plus + delta_minus),
        "amplitudes_bracket_zero": bool(a_minus < 0 < a_plus),
    }


def measure_feature_triple(
    zero: ArrayValue,
    plus: ArrayValue,
    minus: ArrayValue,
    *,
    probe_geometry: Mapping[str, Any],
    include_vectors: bool = True,
) -> dict[str, Any]:
    """Return actual-step secant and asymmetric three-point diagnostics.

    The secant need not be the derivative at zero.  The centered three-point
    estimate is exact for a quadratic only when the actual perturbations lie
    on the measured one-dimensional axis, not evidence of channel linearity.
    The weighted even residual compares zero with the endpoint chord evaluated
    at zero. Unequal amplitudes alone do not create an affine-map residual for
    collinear perturbations. Off-axis perturbations can create a nonzero even
    residual even when the full feature map is linear.
    """
    f_zero, f_plus, f_minus = [_array(value, name) for value, name in zip((zero, plus, minus), ARMS)]
    if f_zero.shape != f_plus.shape or f_zero.shape != f_minus.shape:
        raise ValueError("feature triples must have matching shapes")
    a_plus = float(probe_geometry["plus_signed_amplitude"])
    a_minus = float(probe_geometry["minus_signed_amplitude"])
    if not np.isfinite([a_plus, a_minus]).all() or not a_minus < 0 < a_plus:
        raise ValueError("actual signed amplitudes must strictly bracket zero")
    step = a_plus - a_minus
    secant = (f_plus - f_minus) / step
    slope_plus = (f_plus - f_zero) / a_plus
    slope_minus = (f_minus - f_zero) / a_minus
    centered = (-a_minus * slope_plus + a_plus * slope_minus) / step
    even = (a_plus * (f_minus - f_zero) - a_minus * (f_plus - f_zero)) / step
    midpoint_even = (f_plus - f_zero + f_minus - f_zero) / 2.0
    diagnostics = (secant, centered, even, midpoint_even)
    if not all(np.isfinite(value).all() for value in diagnostics):
        raise ValueError("finite-difference measurement overflowed")
    result = {
        "feature_shape": list(f_zero.shape),
        "actual_signed_step": step,
        "actual_signed_midpoint": (a_plus + a_minus) / 2.0,
        "response_secant_l2": _norm(secant),
        "response_centered_three_point_l2": _norm(centered),
        "even_residual_at_zero_l2": _norm(even),
        "midpoint_even_residual_raw_l2": _norm(midpoint_even),
        "interpretation": (
            "Descriptive finite differences on actual perturbations; off-axis "
            "quantization can contribute to the response and even residual. "
            "The even residual is not a nonlinearity verdict."
        ),
    }
    if include_vectors:
        result.update({
            "response_secant": _vector(secant),
            "response_centered_three_point": _vector(centered),
            "even_residual_at_zero": _vector(even),
            "midpoint_even_residual_raw": _vector(midpoint_even),
        })
    return result


def measure_source(
    source_id: str,
    *,
    clean_features: Mapping[str, ArrayValue | None],
    reconstruction_features_by_seed: Mapping[str | int, Mapping[str, ArrayValue | None]],
    expected_reconstruction_seeds: Sequence[str | int],
    probe_geometry: Mapping[str, Any] | None,
    failures: Mapping[str, str] | None = None,
    include_vectors: bool = True,
) -> dict[str, Any]:
    """Keep a source's expected arms, including missing and failed observations.

    Failure keys are ``clean/zero`` or ``reconstruction/<seed>/plus`` etc.
    A recorded failure wins over any stale supplied feature for that arm.
    Expected seeds come from the roster, never from successful observations.
    """
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a nonempty string")
    seeds = [str(seed) for seed in expected_reconstruction_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected reconstruction seeds must be nonempty and unique")
    recon = {str(seed): features for seed, features in reconstruction_features_by_seed.items()}
    if len(recon) != len(reconstruction_features_by_seed) or set(recon) - set(seeds):
        raise ValueError("reconstruction observations contain duplicate or unplanned seeds")
    failures = dict(failures or {})
    expected_paths = {f"clean/{arm}" for arm in ARMS} | {f"reconstruction/{seed}/{arm}" for seed in seeds for arm in ARMS}
    if set(failures) - expected_paths:
        raise ValueError("failure records contain unplanned arm paths")
    arm_states: dict[str, Any] = {}

    def collect(features: Mapping[str, ArrayValue | None], prefix: str) -> dict[str, np.ndarray]:
        if set(features) - set(ARMS):
            raise ValueError(f"{prefix} contains unplanned arms")
        available = {}
        for arm in ARMS:
            path = f"{prefix}/{arm}"
            if path in failures:
                arm_states[path] = {"status": "FAILED", "error": failures[path]}
            elif features.get(arm) is None:
                arm_states[path] = {"status": "MISSING"}
            else:
                try:
                    available[arm] = _array(features[arm], path)
                    arm_states[path] = {"status": "OBSERVED"}
                except (TypeError, ValueError) as error:
                    arm_states[path] = {"status": "INVALID", "error": str(error)}
        return available

    def triple(available: Mapping[str, np.ndarray]) -> dict[str, Any]:
        if len(available) != len(ARMS) or probe_geometry is None:
            return {"status": "INCOMPLETE", "missing_probe_geometry": probe_geometry is None}
        try:
            return {"status": "MEASURED", **measure_feature_triple(**available, probe_geometry=probe_geometry, include_vectors=include_vectors)}
        except (TypeError, ValueError, KeyError) as error:
            return {"status": "INVALID", "error": str(error)}

    clean = collect(clean_features, "clean")
    clean_measurement = triple(clean)
    reconstructed: dict[str, Any] = {}
    for seed in seeds:
        available = collect(recon.get(seed, {}), f"reconstruction/{seed}")
        measurement = triple(available)
        drift: dict[str, Any] = {"status": "INCOMPLETE"}
        if "zero" in clean and "zero" in available:
            if clean["zero"].shape != available["zero"].shape:
                drift = {"status": "INVALID", "error": "clean and reconstructed zero features must have matching shapes"}
            else:
                difference = available["zero"] - clean["zero"]
                if np.isfinite(difference).all():
                    try:
                        drift = {"status": "MEASURED", "l2": _norm(difference)}
                        if include_vectors:
                            drift["vector"] = _vector(difference)
                    except ValueError as error:
                        drift = {"status": "INVALID", "error": str(error)}
                else:
                    drift = {"status": "INVALID", "error": "host drift overflowed"}
        reconstructed[seed] = {**measurement, "host_drift": drift}
    complete = clean_measurement["status"] == "MEASURED" and all(row["status"] == "MEASURED" and row["host_drift"]["status"] == "MEASURED" for row in reconstructed.values())
    return {
        "source_id": source_id,
        "status": "MEASURED" if complete else "INCOMPLETE",
        "scope": "single_direction_channel_measurement",
        "expected_reconstruction_seeds": seeds,
        "expected_arm_count": len(expected_paths),
        "observed_arm_count": sum(row["status"] == "OBSERVED" for row in arm_states.values()),
        "arm_states": arm_states,
        "probe_geometry": dict(probe_geometry) if probe_geometry is not None else None,
        "clean": clean_measurement,
        "reconstruction": reconstructed,
    }


def planned_call_budget(
    source_count: int,
    reconstruction_seed_count: int,
    direction_count: int = 1,
) -> dict[str, int | str]:
    """Count three full generations per source; there is no extra capture run."""
    for name, value in (("source_count", source_count), ("reconstruction_seed_count", reconstruction_seed_count), ("direction_count", direction_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if direction_count != 1:
        raise ValueError("the first diagnostic supports exactly one frozen input direction")
    generations = source_count * (1 + 2 * direction_count)
    return {
        "source_count": source_count,
        "direction_count": direction_count,
        "reconstruction_seed_count": reconstruction_seed_count,
        "generation_formula": "n * (1 + 2 * d)",
        "generation_calls": generations,
        "reconstruction_calls": generations * reconstruction_seed_count,
        "public_feature_calls": generations * (1 + reconstruction_seed_count),
    }
