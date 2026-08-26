"""CPU/fake image-side Q/K coordinate-recovery experiment harness.

This is deliberately calculation-only. Descriptor matching never consumes the
known homography; that value is experiment truth used after matching only.
"""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any, Mapping, Sequence

import numpy as np

from cegwm.geometry.transform import apply_h, estimate_bounded_similarity

MAX_SAMPLED_TOKENS = 64
MAX_PREDECLARED_UNITS = 64
_TRANSFORM_LABELS = frozenset({"identity", "d4", "similarity", "crop_rescale"})
_CONTROL_LABELS = frozenset({"matched_h", "shuffled_h"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$", re.ASCII)
_PUBLIC_FIELDS = frozenset({
    "pair_id", "transform_label", "control_label", "descriptor_kind", "layer_path",
    "reference_grid", "attacked_grid", "input_identity", "h_identity", "status",
    "failure_reason", "candidate_correspondences", "true_match_ranks", "coverage",
    "ambiguity_gaps", "fit_residual", "recovery_error",
})


def pixel_center_grid(grid: tuple[int, int]) -> np.ndarray:
    """Return row-major full-source-grid pixel centers."""
    rows, columns = grid
    yy, xx = np.indices((rows, columns), dtype=np.float64)
    return np.column_stack((xx.ravel() + .5, yy.ravel() + .5))


def sampled_pixel_centers(source_grid: tuple[int, int], sample_indices: np.ndarray) -> np.ndarray:
    """Recover observer token centers from flattened full-grid indices."""
    _, columns = source_grid
    indices = np.asarray(sample_indices, dtype=np.int64)
    return np.column_stack(((indices % columns).astype(np.float64) + .5,
                            (indices // columns).astype(np.float64) + .5))


def nearest_sampled_index(point: np.ndarray, sampled_xy: np.ndarray) -> int:
    """Stable sampled-position nearest: ties select lower sample order."""
    distances = np.sum((sampled_xy - np.asarray(point, dtype=np.float64)) ** 2, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("point must be finite")
    return int(np.argmin(distances))


def _identity(value: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(value)
    return {"shape": [int(x) for x in array.shape], "sha256": sha256(array.tobytes()).hexdigest()}


def _base(unit: Mapping[str, Any]) -> dict[str, Any]:
    return {"pair_id": unit.get("pair_id"), "transform_label": unit.get("transform_label"),
            "control_label": unit.get("control_label"), "descriptor_kind": unit.get("descriptor_kind"),
            "layer_path": unit.get("layer_path"), "reference_grid": None, "attacked_grid": None,
            "input_identity": None, "h_identity": None, "status": "failed", "failure_reason": None,
            "candidate_correspondences": [], "true_match_ranks": [], "coverage": None,
            "ambiguity_gaps": [], "fit_residual": None, "recovery_error": None}


def _failure(record: dict[str, Any], reason: str) -> dict[str, Any]:
    record["failure_reason"] = reason
    return record


def _identifier(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid_{name}")
    return value


def _descriptors(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1] or not np.isfinite(result).all():
        raise ValueError(f"invalid_{name}_descriptors")
    return result


def _source_grid(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"invalid_{name}_source_grid")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) < 1 for item in value):
        raise ValueError(f"invalid_{name}_source_grid")
    return int(value[0]), int(value[1])


def _sample_indices(value: Any, grid: tuple[int, int], count: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"invalid_{name}_sample_indices")
    result = raw.astype(np.int64, copy=False)
    if len(result) != count:
        raise ValueError(f"{name}_sample_count_mismatch")
    if len(result) > MAX_SAMPLED_TOKENS:
        raise ValueError("input_token_bound_exceeded")
    if np.any(result < 0) or np.any(result >= grid[0] * grid[1]) or np.any(np.diff(result) <= 0):
        raise ValueError(f"invalid_{name}_sample_indices")
    return result


def _mutual_nearest(reference: np.ndarray, attacked: np.ndarray) -> list[tuple[int, int]]:
    """Score-threshold-free one-to-one matching, bounded by input counts."""
    squared = np.sum((reference[:, None, :] - attacked[None, :, :]) ** 2, axis=2)
    reference_best = np.argmin(squared, axis=1)
    attacked_best = np.argmin(squared, axis=0)
    return [(int(i), int(j)) for i, j in enumerate(reference_best) if int(attacked_best[j]) == i]


def _rank_and_gap(reference_row: np.ndarray, attacked: np.ndarray, truth_position: int | None) -> tuple[int | None, float | None]:
    distances = np.sum((attacked - reference_row) ** 2, axis=1)
    order = np.argsort(distances, kind="stable")
    rank = None if truth_position is None else int(np.flatnonzero(order == truth_position)[0]) + 1
    gap = float(distances[order[1]] - distances[order[0]]) if len(order) > 1 else None
    return rank, gap


def evaluate_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one predeclared fake unit and retain every bounded failure."""
    record = _base(unit)
    if record["descriptor_kind"] not in {"q", "k"}:
        return _failure(record, "invalid_descriptor_kind")
    if record["transform_label"] not in _TRANSFORM_LABELS:
        return _failure(record, "invalid_transform_label")
    if record["control_label"] not in _CONTROL_LABELS:
        return _failure(record, "invalid_control_label")
    try:
        _identifier(record["pair_id"], "pair_id", 128)
        _identifier(record["layer_path"], "layer_path", 256)
        reference = _descriptors(unit["reference_descriptors"], "reference")
        attacked = _descriptors(unit["attacked_descriptors"], "attacked")
        if reference.shape[1] != attacked.shape[1]:
            return _failure(record, "descriptor_dimension_mismatch")
        if len(reference) > MAX_SAMPLED_TOKENS or len(attacked) > MAX_SAMPLED_TOKENS:
            return _failure(record, "input_token_bound_exceeded")
        reference_grid = _source_grid(unit["reference_source_grid"], "reference")
        attacked_grid = _source_grid(unit["attacked_source_grid"], "attacked")
        reference_indices = _sample_indices(unit["reference_sample_indices"], reference_grid, len(reference), "reference")
        attacked_indices = _sample_indices(unit["attacked_sample_indices"], attacked_grid, len(attacked), "attacked")
        h = np.asarray(unit["H_reference_to_attacked"], dtype=np.float64)
        if h.shape != (3, 3) or not np.isfinite(h).all():
            return _failure(record, "invalid_h_reference_to_attacked")
    except (KeyError, TypeError, ValueError) as error:
        return _failure(record, str(error))

    record.update({"reference_grid": list(reference_grid), "attacked_grid": list(attacked_grid),
                   "input_identity": {"reference": _identity(reference), "attacked": _identity(attacked),
                                      "reference_sample_indices": _identity(reference_indices),
                                      "attacked_sample_indices": _identity(attacked_indices)}, "h_identity": _identity(h)})
    reference_xy = sampled_pixel_centers(reference_grid, reference_indices)
    attacked_xy = sampled_pixel_centers(attacked_grid, attacked_indices)
    pairs = _mutual_nearest(reference, attacked)
    record["candidate_correspondences"] = [{"reference_index": i, "attacked_index": j,
                                           "reference_xy": reference_xy[i].tolist(), "attacked_xy": attacked_xy[j].tolist()}
                                          for i, j in pairs]
    truth_xy = apply_h(reference_xy, h)
    rows, columns = attacked_grid
    in_view = ((truth_xy[:, 0] >= .5) & (truth_xy[:, 0] <= columns - .5)
               & (truth_xy[:, 1] >= .5) & (truth_xy[:, 1] <= rows - .5))
    ranks, gaps = [], []
    for index, descriptor in enumerate(reference):
        truth_position = nearest_sampled_index(truth_xy[index], attacked_xy) if bool(in_view[index]) else None
        rank, gap = _rank_and_gap(descriptor, attacked, truth_position)
        ranks.append(rank); gaps.append(gap)
    record["true_match_ranks"], record["ambiguity_gaps"], record["coverage"] = ranks, gaps, float(np.mean(in_view))
    if len(pairs) < 3:
        return _failure(record, "fewer_than_three_mutual_candidates")
    source, target = reference_xy[[i for i, _ in pairs]], attacked_xy[[j for _, j in pairs]]
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        return _failure(record, "collinear_candidate_coordinates")
    try:
        fitted = estimate_bounded_similarity(source, target, attacked_grid, total_reference_points=len(reference_xy))
    except ValueError:
        return _failure(record, "bounded_similarity_unavailable")
    record["fit_residual"] = float(fitted.residual)
    valid_truth = truth_xy[in_view]
    predicted_truth = apply_h(reference_xy[in_view], fitted.h_canonical_to_observed)
    record["recovery_error"] = None if not len(valid_truth) else float(np.sqrt(np.mean(np.sum((predicted_truth - valid_truth) ** 2, axis=1))))
    record["status"] = "calculated"
    return record


def run_predeclared_units(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate a bounded all-or-nothing declared plan in its original order."""
    if len(units) > MAX_PREDECLARED_UNITS:
        raise ValueError("predeclared_unit_bound_exceeded")
    return [evaluate_unit(unit) for unit in units]


def public_record_fields() -> frozenset[str]:
    return _PUBLIC_FIELDS
