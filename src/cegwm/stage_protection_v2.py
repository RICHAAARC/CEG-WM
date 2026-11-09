"""Frozen V2 canary geometry and reporting; no models or detector changes.

Cells are supplied by the unmarked baseline, never reselected at E. Solver
tolerances are floating-point arithmetic checks, not observed acceptance gates.
"""
from __future__ import annotations

import itertools
import math

import torch

from cegwm.channel_precompensation import (
    ReaderSpec, _equality_minimum_norm, _norm, _tensor, _tolerance,
    decode_score, make_reader, nearest_samebit_centers,
)

LIMITS = dict(final_candidates=16, sd3_forward=40, sd3_denoising_steps=258,
              sdxl_ordinary=44, sdxl_proxy=12, sdxl_forward=56,
              sdxl_denoising_steps=336, rgb_vjp=16, latent_vjp=16,
              reader_result_ordinary=70, reader_aux_ordinary=16,
              reader_proxy=16, reader_encode=102)


def fixed_cells(scores, bit, spec):
    _, phase = make_reader(1, spec)
    return nearest_samebit_centers(scores, spec, phase, bit).tolist()


def read_score(score, center, bit, spec):
    _, phase = make_reader(1, spec)
    blind = decode_score(score, spec, phase, bit)
    margin = 0.25 - abs(score - center)
    return {**blind, "fixed_center": center, "fixed_inner_margin": margin,
            "fixed_inner": margin >= 0, "bit_correct": blind["decoded_bit"] == bit}


def protection_radius(score, center):
    margin = 0.25 - abs(score - center)
    if not 0 <= margin <= 0.25:
        raise ValueError("E must be inside the frozen design inner cell")
    return 0.25 - margin / 2


def cast_action(base, requested, reference, rho=0.01):
    """Exactly the runtime FP32-add/latent-cast, with a fixed unmarked norm."""
    requested = _tensor(requested, "requested")
    if requested.shape != base.shape:
        raise ValueError("action shape differs from latent")
    after = (base.float() + requested.float()).to(base.dtype)
    if not bool(torch.isfinite(after).all()):
        raise ValueError("nonfinite cast action")
    actual = after.double() - base.double()
    denominator = _norm(_tensor(reference, "unmarked reference"))
    if denominator == 0:
        raise ValueError("zero unmarked reference norm")
    relative = _norm(actual) / denominator
    return {"after": after, "actual": actual, "relative": relative,
            "within_budget": relative <= rho / math.sqrt(2),
            "nonzero": bool(torch.count_nonzero(actual)),
            "requested_norm": _norm(requested), "actual_norm": _norm(actual),
            "reference_norm": denominator}


def early_update(current_delta, gradient, score, center, radius):
    d, g = _tensor(current_delta, "early cumulative delta"), _tensor(gradient, "gradient")
    if d.shape != g.shape or not math.isfinite(score) or not math.isfinite(radius) or radius < 0:
        raise ValueError("invalid early update")
    norm = _norm(g)
    if norm == 0:
        raise ValueError("zero early proxy gradient; not a proof of scientific infeasibility")
    candidate = d + ((center - score) / norm) * (g / norm)
    size = _norm(candidate)
    if size > radius:
        candidate = candidate * (radius / size)
    return candidate


def solve_intervals(gradients, scores, centers, widths, radius):
    """Minimum norm for one/two fixed slabs, via at most two SVD equalities."""
    g, s, c, w = (_tensor(v, n) for v, n in
                  ((gradients, "gradients"), (scores, "scores"),
                   (centers, "centers"), (widths, "widths")))
    shape = g.shape[1:]
    if g.shape[0] not in (1, 2) or s.shape != c.shape or s.shape != w.shape or s.numel() != len(g):
        raise ValueError("one or two matching constraints required")
    if bool((w < 0).any()) or not math.isfinite(radius) or radius < 0:
        raise ValueError("invalid width or budget")
    g = g.reshape(len(g), -1)
    lo, hi = c - w, c + w
    candidates = []
    if bool(((s >= lo) & (s <= hi)).all()):
        candidates.append((0.0, torch.zeros(g.shape[1], dtype=torch.float64), []))
    unresolved = False
    for count in range(1, len(g) + 1):
        for rows in itertools.combinations(range(len(g)), count):
            for sides in itertools.product((0, 1), repeat=count):
                rhs = torch.tensor([(lo if side == 0 else hi)[row] - s[row]
                                    for row, side in zip(rows, sides)])
                try:
                    delta = _equality_minimum_norm(g[list(rows)], rhs)
                    if delta is None:
                        continue
                    pred = s + g @ delta
                    if not bool(torch.isfinite(pred).all()):
                        raise ArithmeticError("nonfinite prediction")
                    tolerance = _tolerance(pred, lo, hi)
                    if bool(((pred >= lo - tolerance) & (pred <= hi + tolerance)).all()):
                        candidates.append((_norm(delta), delta, list(zip(rows, sides))))
                except (ArithmeticError, torch.linalg.LinAlgError):
                    unresolved = True
    common = dict(scores=s.tolist(), centers=c.tolist(), widths=w.tolist(), radius=radius)
    if not candidates:
        return dict(common, status="NUMERICALLY_UNRESOLVED" if unresolved else "INFEASIBLE",
                    delta=None, requested=None)
    norm, delta, active = min(candidates, key=lambda x: x[0])
    return dict(common, status="SOLVED" if norm <= radius else "INFEASIBLE_BUDGET",
                delta=delta.reshape(shape) if norm <= radius else None,
                requested=delta.reshape(shape), norm=norm, active=active,
                predicted=(s + g @ delta).tolist())


def common_accept(reads, early_relative, late_relative, rho=0.01):
    budget = (early_relative <= rho / math.sqrt(2) and late_relative <= rho / math.sqrt(2)
              and early_relative ** 2 + late_relative ** 2 <= rho ** 2)
    return bool(budget and reads["clean"]["fixed_inner"] and reads["design"]["fixed_inner"])


def grouped_verdict(trials, source_ids):
    """No significance claim. Unknown/incomplete units never become negatives."""
    groups = []
    for source_id in source_ids:
        rows = [r for r in trials if r["source_id"] == source_id]
        complete = len(rows) == 2 and all(r.get("evaluation_complete", False) for r in rows)
        strict = [r["bit"] for r in rows if r.get("mechanism_case", False)
                  and r.get("C_success") is True and r.get("B_success") is False]
        regression = any(r.get("B_success") is True and r.get("C_success") is False for r in rows)
        groups.append(dict(source_id=source_id, strict_improvement_bits=strict,
                           any_B_success_C_failure=regression, complete=complete,
                           operational_signal=complete and bool(strict) and not regression))
    signal = len(trials) == 4 and all(g["operational_signal"] for g in groups)
    complete = len(trials) == 4 and all(r.get("evaluation_complete", False) for r in trials)
    if not complete:
        outcome = "ENGINEERING_INCOMPLETE"
    elif signal:
        outcome = "OPERATIONAL_SIGNAL_PENDING_QUALITY_REVIEW"
    elif all(r.get("late_status") == "early_design_gate_failed" for r in trials):
        outcome = "EARLY_WRITER_NOT_SUPPORTED_THIS_ROUND"
    elif not any(r.get("active_early_flip") and r.get("late_status") == "repair_attempted" for r in trials):
        outcome = "MECHANISM_NOT_TRIGGERED_THIS_ROUND"
    elif not any(r.get("C_B_actual_different") for r in trials):
        outcome = "NO_DISTINGUISHABLE_C_B_ACTION_THIS_ROUND"
    else:
        outcome = "NO_TWO_CONTENT_INCREMENT_THIS_ROUND"
    return dict(content_denominator=2, bit_trial_denominator=4, groups=groups,
                operational_P2_signal=signal, outcome=outcome,
                active_early_flip_trials=sum(bool(r.get("active_early_flip")) for r in trials),
                repair_trials=sum(r.get("late_status") == "repair_attempted" for r in trials),
                quality_tradeoff_review="REQUIRED" if signal else "not_triggered",
                P3_authorized=False, scientific_pass=None, innovation_pass=None,
                interpretation="Two contents only; no significance, attribution or FPR claim. Quality tradeoff remains a separate review.")
