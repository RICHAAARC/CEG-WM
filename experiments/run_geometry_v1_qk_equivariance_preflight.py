"""CPU/fake Geometry-V1 Q/K equivariance experiment harness.

This module is deliberately an experiment-only calculation.  It does not make
route, reliability, detector, or scientific decisions.  ``H_reference_to_attacked``
is used after descriptor matching solely to construct experiment truth metrics.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

from cegwm.geometry.transform import apply_h, estimate_bounded_similarity


_PUBLIC_FIELDS = frozenset({
    "pair_id", "transform_label", "control_label", "descriptor_kind", "layer_path",
    "reference_grid", "attacked_grid", "input_identity", "h_identity", "status",
    "failure_reason", "candidate_correspondences", "true_match_ranks", "coverage",
    "ambiguity_gaps", "fit_residual", "recovery_error",
})


def pixel_center_grid(grid: tuple[int, int]) -> np.ndarray:
    """Return row-major pixel-center xy coordinates for a finite token grid."""

    rows, columns = grid
    if isinstance(rows, bool) or isinstance(columns, bool) or rows < 1 or columns < 1:
        raise ValueError("grid must contain positive rows and columns")
    yy, xx = np.indices((rows, columns), dtype=np.float64)
    return np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))


def nearest_grid_index(point: np.ndarray, grid: tuple[int, int]) -> int:
    """Quantize to the nearest pixel-center, breaking exact ties by low index."""

    points = pixel_center_grid(grid)
    distance = np.sum((points - np.asarray(point, dtype=np.float64)) ** 2, axis=1)
    if not np.isfinite(distance).all():
        raise ValueError("point must be finite")
    return int(np.argmin(distance))


def _identity(value: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(value)
    return {"shape": [int(x) for x in array.shape], "sha256": sha256(array.tobytes()).hexdigest()}


def _base(unit: Mapping[str, Any]) -> dict[str, Any]:
    kind = unit.get("descriptor_kind")
    return {
        "pair_id": unit.get("pair_id"), "transform_label": unit.get("transform_label"),
        "control_label": unit.get("control_label"), "descriptor_kind": kind,
        "layer_path": unit.get("layer_path"), "reference_grid": None,
        "attacked_grid": None, "input_identity": None, "h_identity": None,
        "status": "failed", "failure_reason": None, "candidate_correspondences": [],
        "true_match_ranks": [], "coverage": None, "ambiguity_gaps": [],
        "fit_residual": None, "recovery_error": None,
    }


def _failure(record: dict[str, Any], reason: str) -> dict[str, Any]:
    record["failure_reason"] = reason
    return record


def _descriptors(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1 or not np.isfinite(result).all():
        raise ValueError(f"invalid_{name}_descriptors")
    return result


def _grid(value: Any, token_count: int, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"invalid_{name}_grid")
    result = (int(value[0]), int(value[1]))
    if result[0] < 1 or result[1] < 1 or result[0] * result[1] != token_count:
        raise ValueError(f"invalid_{name}_grid")
    return result


def _mutual_nearest(reference: np.ndarray, attacked: np.ndarray) -> list[tuple[int, int]]:
    """Score-threshold-free, deterministic mutual-nearest descriptor matching."""

    squared = np.sum((reference[:, None, :] - attacked[None, :, :]) ** 2, axis=2)
    reference_best = np.argmin(squared, axis=1)  # np.argmin is stable at equal minima.
    attacked_best = np.argmin(squared, axis=0)
    return [(int(i), int(j)) for i, j in enumerate(reference_best) if int(attacked_best[j]) == i]


def _rank_and_gap(reference_row: np.ndarray, attacked: np.ndarray, truth_index: int | None) -> tuple[int | None, float | None]:
    distances = np.sum((attacked - reference_row) ** 2, axis=1)
    order = np.argsort(distances, kind="stable")
    rank = None if truth_index is None else int(np.flatnonzero(order == truth_index)[0]) + 1
    gap = float(distances[order[1]] - distances[order[0]]) if len(order) > 1 else None
    return rank, gap


def evaluate_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate exactly one predeclared fake observation unit and retain failures."""

    record = _base(unit)
    if record["descriptor_kind"] not in {"q", "k"}:
        return _failure(record, "invalid_descriptor_kind")
    if not isinstance(record["layer_path"], str) or not record["layer_path"]:
        return _failure(record, "invalid_layer_path")
    try:
        reference = _descriptors(unit["reference_descriptors"], "reference")
        attacked = _descriptors(unit["attacked_descriptors"], "attacked")
        if reference.shape[1] != attacked.shape[1]:
            return _failure(record, "descriptor_dimension_mismatch")
        reference_grid = _grid(unit["reference_grid"], len(reference), "reference")
        attacked_grid = _grid(unit["attacked_grid"], len(attacked), "attacked")
        h = np.asarray(unit["H_reference_to_attacked"], dtype=np.float64)
        if h.shape != (3, 3) or not np.isfinite(h).all():
            return _failure(record, "invalid_h_reference_to_attacked")
    except (KeyError, TypeError, ValueError) as error:
        return _failure(record, str(error))

    record.update({
        "reference_grid": list(reference_grid), "attacked_grid": list(attacked_grid),
        "input_identity": {"reference": _identity(reference), "attacked": _identity(attacked)},
        "h_identity": _identity(h),
    })
    reference_xy, attacked_xy = pixel_center_grid(reference_grid), pixel_center_grid(attacked_grid)
    pairs = _mutual_nearest(reference, attacked)
    record["candidate_correspondences"] = [
        {"reference_index": i, "attacked_index": j,
         "reference_xy": reference_xy[i].tolist(), "attacked_xy": attacked_xy[j].tolist()}
        for i, j in pairs
    ]

    truth_xy = apply_h(reference_xy, h)
    attacked_rows, attacked_columns = attacked_grid
    in_view = ((truth_xy[:, 0] >= 0.5) & (truth_xy[:, 0] <= attacked_columns - 0.5)
               & (truth_xy[:, 1] >= 0.5) & (truth_xy[:, 1] <= attacked_rows - 0.5))
    ranks: list[int | None] = []
    gaps: list[float | None] = []
    for index, descriptor in enumerate(reference):
        truth_index = nearest_grid_index(truth_xy[index], attacked_grid) if bool(in_view[index]) else None
        rank, gap = _rank_and_gap(descriptor, attacked, truth_index)
        ranks.append(rank)
        gaps.append(gap)
    record["true_match_ranks"], record["ambiguity_gaps"] = ranks, gaps
    record["coverage"] = float(np.mean(in_view))

    if len(pairs) < 3:
        return _failure(record, "fewer_than_three_mutual_candidates")
    source = reference_xy[[i for i, _ in pairs]]
    target = attacked_xy[[j for _, j in pairs]]
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        return _failure(record, "collinear_candidate_coordinates")
    try:
        fitted = estimate_bounded_similarity(source, target, (attacked_rows, attacked_columns), total_reference_points=len(reference_xy))
    except ValueError:
        return _failure(record, "bounded_similarity_unavailable")
    record["fit_residual"] = float(fitted.residual)
    valid_truth = truth_xy[in_view]
    predicted_truth = apply_h(reference_xy[in_view], fitted.h_canonical_to_observed)
    record["recovery_error"] = (None if not len(valid_truth) else float(np.sqrt(np.mean(np.sum((predicted_truth - valid_truth) ** 2, axis=1)))))
    record["status"] = "calculated"
    return record


def run_predeclared_units(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one bounded public record per predeclared unit, including failures."""

    return [evaluate_unit(unit) for unit in units]


def public_record_fields() -> frozenset[str]:
    """Expose the fixed experiment record whitelist for audit and tests."""

    return _PUBLIC_FIELDS
