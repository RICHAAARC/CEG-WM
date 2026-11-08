"""CPU geometry for one-step, two-channel scalar precompensation.

The gradients are supplied surrogate sensitivities. Solver feasibility and
predicted QIM margins do not assert that a real RGB channel transmits a bit.
No model, attack, fitting, learning-rate search, or detector calibration occurs
in this module. A failed write is returned without scaling or changing cells.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cegwm.shared.prg import prg_bytes, prg_rademacher

_EPS = torch.finfo(torch.float64).eps
_DOMAIN = "channel-precompensation-v1"


@dataclass(frozen=True)
class ReaderSpec:
    key_seed: int
    delta: float = 1.0
    inner_fraction: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.key_seed, bool) or not isinstance(self.key_seed, int) or self.key_seed < 0:
            raise ValueError("key_seed must be a nonnegative integer")
        if not math.isfinite(self.delta) or self.delta <= 0 or not math.isfinite(2 * self.delta):
            raise ValueError("delta must be finite and positive with a finite two-cell period")
        if not math.isfinite(self.inner_fraction) or not 0 < self.inner_fraction < 0.5:
            raise ValueError("inner_fraction must lie strictly between zero and one half")


def _bit(bit: int) -> int:
    if isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1):
        raise ValueError("bit must be the integer zero or one")
    return bit


def _scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if value.is_complex():
            raise ValueError(f"{name} must be real")
        result = value.detach().to(dtype=torch.float64).clone()
    else:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        result = torch.as_tensor(value, dtype=torch.float64, device="cpu").clone()
    if result.numel() == 0 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def _norm(vector: torch.Tensor) -> float:
    scale = float(vector.abs().max())
    value = 0.0 if scale == 0 else float(torch.linalg.vector_norm(vector / scale)) * scale
    if not math.isfinite(value):
        raise ArithmeticError("finite inputs produced an unrepresentable L2 norm")
    return value


def _tolerance(*vectors: torch.Tensor) -> float:
    return 64 * _EPS * max(1.0, *(float(vector.abs().max()) for vector in vectors))


def make_reader(feature_dim: int, spec: ReaderSpec) -> tuple[torch.Tensor, float]:
    """Return a fixed unit Rademacher vector and independently keyed phase.

    The reader and phase use different PRG domains and no process-global RNG.
    Phase is in [0, 2*delta); its random draw does not depend on feature shape.
    """
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("feature_dim must be a positive integer")
    key = f"{_DOMAIN}/key/{spec.key_seed}"
    signs = prg_rademacher(key, f"{_DOMAIN}/reader", (feature_dim,), dtype=np.float64)
    reader = torch.from_numpy(signs) / math.sqrt(feature_dim)
    bits = int.from_bytes(prg_bytes(key, f"{_DOMAIN}/phase", 8), "big") >> 11
    phase = (bits / float(2**53)) * (2 * spec.delta)
    phase = min(phase, math.nextafter(2 * spec.delta, 0.0))
    return reader, phase


def nearest_samebit_centers(scores: Any, spec: ReaderSpec, phase: float, bit: int) -> torch.Tensor:
    """Choose one nearest center per score; exact ties choose the larger index.

    The solver does not search other centers when these fixed cells fail.
    """
    bit, phase = _bit(bit), _scalar(phase, "phase")
    values = _tensor(scores, "scores")
    centers = []
    for value in values.reshape(-1):
        position = (float(value) - phase - bit * spec.delta) / (2 * spec.delta)
        if not math.isfinite(position):
            raise ValueError("score cannot be represented on this QIM lattice")
        index = 2 * math.floor(position + 0.5) + bit
        centers.append(phase + index * spec.delta)
    result = torch.tensor(centers, dtype=torch.float64).reshape(values.shape)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nearest QIM center overflowed")
    return result


def decode_score(
    score: float, spec: ReaderSpec, phase: float, target_bit: int | None = None,
) -> dict[str, Any]:
    """Blind scalar readout; target labels affect reporting, never decoding.

    Signed target margin is distance to the decision boundary in score units:
    delta/2 minus distance to the nearest target-bit center. A boundary tie has
    zero margin regardless of the deterministic decoded parity.
    """
    score, phase = _scalar(score, "score"), _scalar(phase, "phase")
    position = (score - phase) / spec.delta
    if not math.isfinite(position):
        raise ValueError("score cannot be represented on this QIM lattice")
    index = math.floor(position + 0.5)
    center = phase + index * spec.delta
    result = {
        "score": score, "nearest_index": index, "nearest_center": center,
        "decoded_bit": index % 2,
        "decision_boundary_distance": spec.delta / 2 - abs(score - center),
    }
    if target_bit is not None:
        target_bit = _bit(target_bit)
        target_center = float(nearest_samebit_centers([score], spec, phase, target_bit)[0])
        distance = abs(score - target_center)
        numerical_tolerance = 64 * _EPS * max(1.0, abs(score), abs(target_center), spec.delta)
        result.update(target_bit=target_bit, target_center=target_center,
                      signed_margin=spec.delta / 2 - distance,
                      inner_cell_violation=max(0.0, distance - spec.delta * spec.inner_fraction),
                      inside_target_inner_cell=distance <= spec.delta * spec.inner_fraction + numerical_tolerance,
                      numerical_tolerance=numerical_tolerance)
    return result


def _prepare(gradients, baseline_scores, bit, spec, phase, l2_budget):
    gradients = _tensor(gradients, "gradients")
    if gradients.ndim < 2 or gradients.shape[0] != 2:
        raise ValueError("gradients must have shape [2, *latent_shape]")
    baseline = _tensor(baseline_scores, "baseline_scores").reshape(-1)
    if baseline.numel() != 2:
        raise ValueError("exactly two baseline scores are required")
    budget = _scalar(l2_budget, "l2_budget")
    if budget < 0:
        raise ValueError("l2_budget must be nonnegative")
    centers = nearest_samebit_centers(baseline, spec, phase, bit)
    half_width = spec.delta * spec.inner_fraction
    lower, upper = centers - half_width, centers + half_width
    return gradients.reshape(2, -1), gradients.shape[1:], baseline, centers, lower, upper, budget


def _violation(predicted, lower, upper) -> float:
    return float(torch.clamp(torch.maximum(lower - predicted, predicted - upper), min=0).max())


def evaluate_delta(
    delta, gradients, baseline_scores, *, bit: int, spec: ReaderSpec,
    phase: float, l2_budget: float,
) -> dict[str, Any]:
    """Evaluate a caller-supplied delta, including an actual post-cast delta.

    Caller owns the actual-dtype subtraction. These remain linear predictions,
    not observations of the written RGB image or a bit-transmission verdict.
    """
    matrix, _, baseline, centers, lower, upper, budget = _prepare(
        gradients, baseline_scores, bit, spec, phase, l2_budget,
    )
    delta = _tensor(delta, "delta").reshape(-1)
    if delta.numel() != matrix.shape[1]:
        raise ValueError("delta must match the latent element count")
    predicted = baseline + matrix @ delta
    if not bool(torch.isfinite(predicted).all()):
        raise ArithmeticError("predicted scores overflowed")
    norm = _norm(delta)
    violation = _violation(predicted, lower, upper)
    numerical_tolerance = _tolerance(predicted, lower, upper)
    return {
        "centers": centers.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
        "predicted_scores": predicted.tolist(), "l2_norm": norm, "l2_budget": budget,
        "within_budget": norm <= budget,
        "max_interval_violation": violation,
        "within_inner_cells": violation <= numerical_tolerance,
        "interval_numerical_tolerance": numerical_tolerance,
        "predicted_reads": [decode_score(float(value), spec, phase, bit) for value in predicted],
        "interpretation": "Surrogate linear prediction only; actual RGB reads are required.",
    }


def _equality_minimum_norm(matrix: torch.Tensor, right: torch.Tensor) -> torch.Tensor | None:
    """Solve at most two active equalities without squaring their condition number."""
    norms = torch.tensor([_norm(row) for row in matrix], dtype=torch.float64)
    nonzero = norms > 0
    if bool((right[~nonzero] != 0).any()):
        return None
    if not bool(nonzero.any()):
        return torch.zeros(matrix.shape[1], dtype=torch.float64)
    normalized = matrix[nonzero] / norms[nonzero, None]
    scaled_right = right[nonzero] / norms[nonzero]
    if not bool(torch.isfinite(scaled_right).all()):
        raise ArithmeticError("active constraint scaling overflowed")
    left, singular, right_vectors = torch.linalg.svd(normalized, full_matrices=False)
    keep = singular > (max(2, normalized.shape[0]) * _EPS * singular[0])
    coordinates = (left[:, keep].T @ scaled_right) / singular[keep]
    candidate = right_vectors[keep].T @ coordinates
    equality_prediction = matrix @ candidate
    if not bool(torch.isfinite(candidate).all()) or not bool(torch.isfinite(equality_prediction).all()):
        raise ArithmeticError("active-set solve overflowed")
    if float((equality_prediction - right).abs().max()) > _tolerance(equality_prediction, right):
        return None
    return candidate


def solve_write(
    gradients, baseline_scores, *, bit: int, spec: ReaderSpec, phase: float,
    l2_budget: float, method: str = "joint",
) -> dict[str, Any]:
    """Solve fixed inner-cell constraints or retain an explicit failed write.

    Joint: enumerate active lower/upper faces and choose the feasible minimum
    L2 perturbation. Gradient: use G.T@(centers-baseline), the negative gradient
    of the same two center-squared losses, then take the shortest feasible step
    on that positive ray. Both methods use identical cells and budget.
    """
    if method not in {"joint", "gradient"}:
        raise ValueError("method must be 'joint' or 'gradient'")
    matrix, latent_shape, baseline, centers, lower, upper, budget = _prepare(
        gradients, baseline_scores, bit, spec, phase, l2_budget,
    )
    diagnostics: dict[str, Any] = {
        "method": method, "baseline_scores": baseline.tolist(), "centers": centers.tolist(),
        "lower": lower.tolist(), "upper": upper.tolist(), "l2_budget": budget,
        "cell_selection": "nearest_samebit_center_ties_toward_larger_index",
    }

    def result(status, reason, candidate=None):
        if candidate is not None:
            diagnostics.update(evaluate_delta(candidate, matrix, baseline, bit=bit, spec=spec,
                                               phase=phase, l2_budget=budget))
        return {"status": status, "reason": reason,
                "delta": candidate.reshape(latent_shape) if status == "SOLVED" else None,
                "diagnostics": diagnostics}

    if bool(((baseline >= lower) & (baseline <= upper)).all()):
        return result("SOLVED", "baseline_already_in_both_inner_cells", torch.zeros(matrix.shape[1], dtype=torch.float64))

    try:
        if method == "joint":
            feasible = []
            numerical_issue = False
            faces = [((row,), (bound,)) for row in (0, 1) for bound in ("lower", "upper")]
            faces += [((0, 1), bounds) for bounds in itertools.product(("lower", "upper"), repeat=2)]
            for rows, bounds in faces:
                active = matrix[list(rows)]
                right = torch.tensor([
                    float((lower if bound == "lower" else upper)[row] - baseline[row])
                    for row, bound in zip(rows, bounds)
                ], dtype=torch.float64)
                try:
                    candidate = _equality_minimum_norm(active, right)
                    if candidate is None:
                        continue
                    predicted = baseline + matrix @ candidate
                    if not bool(torch.isfinite(predicted).all()):
                        raise ArithmeticError("candidate prediction overflowed")
                    if _violation(predicted, lower, upper) <= _tolerance(predicted, lower, upper):
                        feasible.append((_norm(candidate), candidate, list(zip(rows, bounds))))
                except (ArithmeticError, torch.linalg.LinAlgError):
                    numerical_issue = True
            if not feasible:
                return result("NUMERICALLY_UNRESOLVED" if numerical_issue else "INFEASIBLE",
                              "active_set_numerical_failure" if numerical_issue else "fixed_cells_have_no_linear_preimage")
            norm, candidate, active_faces = min(feasible, key=lambda entry: entry[0])
            diagnostics["active_faces"] = active_faces
        else:
            composite = matrix.T @ (centers - baseline)
            norm = _norm(composite)
            if norm == 0:
                return result("INFEASIBLE", "zero_composite_gradient_outside_target_cells")
            direction = composite / norm
            gains = matrix @ direction
            if not bool(torch.isfinite(gains).all()):
                raise ArithmeticError("ray gains overflowed")
            lower_step, upper_step = 0.0, math.inf
            for gain, lo, hi in zip(gains.tolist(), (lower - baseline).tolist(), (upper - baseline).tolist()):
                if gain == 0:
                    if not lo <= 0 <= hi:
                        return result("INFEASIBLE", "ray_is_parallel_to_a_violated_constraint")
                    continue
                first, second = lo / gain, hi / gain
                lower_step = max(lower_step, min(first, second))
                upper_step = min(upper_step, max(first, second))
            diagnostics.update(ray_gains=gains.tolist(),
                               ray_step_lower=lower_step if math.isfinite(lower_step) else None,
                               ray_step_upper=upper_step if math.isfinite(upper_step) else None)
            if lower_step > upper_step:
                return result("INFEASIBLE", "fixed_cells_do_not_intersect_the_positive_gradient_ray")
            if not math.isfinite(lower_step):
                return result("NUMERICALLY_UNRESOLVED", "required_ray_step_is_not_finite")
            candidate = lower_step * direction
            norm = _norm(candidate)
            diagnostics["ray_step"] = lower_step
        if norm > budget:
            return result("OVER_BUDGET", "shortest_feasible_write_exceeds_l2_budget", candidate)
        predicted = baseline + matrix @ candidate
        if _violation(predicted, lower, upper) > _tolerance(predicted, lower, upper):
            return result("NUMERICALLY_UNRESOLVED", "computed_write_does_not_satisfy_fixed_cells", candidate)
        return result("SOLVED", "linear_inner_cells_reached", candidate)
    except (ArithmeticError, torch.linalg.LinAlgError):
        return result("NUMERICALLY_UNRESOLVED", "finite_input_numerical_failure")
