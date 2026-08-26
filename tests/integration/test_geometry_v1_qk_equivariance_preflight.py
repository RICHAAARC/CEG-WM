"""Dynamic CPU/fake tests for the experiment-only Q/K equivariance harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[2] / "experiments" / "run_geometry_v1_qk_equivariance_preflight.py"
SPEC = importlib.util.spec_from_file_location("qk_equivariance", MODULE)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _unit(kind="q", h=None, reference=None, attacked=None, reference_grid=(2, 2), attacked_grid=(2, 2)):
    reference = np.eye(4) if reference is None else reference
    attacked = np.eye(4) if attacked is None else attacked
    return {"pair_id": "pair", "transform_label": "known", "control_label": "control",
            "descriptor_kind": kind, "layer_path": "blocks.0.attn", "reference_descriptors": reference,
            "attacked_descriptors": attacked, "reference_grid": reference_grid, "attacked_grid": attacked_grid,
            "H_reference_to_attacked": np.eye(3) if h is None else h}


def test_pixel_centers_h_direction_and_nearest_quantization_are_frozen() -> None:
    h = np.array(((1., 0., .75), (0., 1., 0.), (0., 0., 1.)))
    assert np.allclose(HARNESS.pixel_center_grid((2, 2))[0], (0.5, 0.5))
    assert HARNESS.nearest_grid_index(np.array((1., .5)), (2, 2)) == 0  # equal-distance -> low index
    record = HARNESS.evaluate_unit(_unit(h=h))
    assert record["coverage"] == .5 and record["true_match_ranks"][:2] == [2, None]


def test_out_of_view_truth_is_retained_and_h_never_changes_candidates() -> None:
    shifted = np.array(((1., 0., 10.), (0., 1., 0.), (0., 0., 1.)))
    a, b = HARNESS.evaluate_unit(_unit()), HARNESS.evaluate_unit(_unit(h=shifted))
    assert a["candidate_correspondences"] == b["candidate_correspondences"]
    assert b["coverage"] == 0.0 and b["true_match_ranks"] == [None] * 4
    assert a["h_identity"] != b["h_identity"]


def test_q_and_k_are_separate_and_not_substituted_or_fused() -> None:
    q = HARNESS.evaluate_unit(_unit("q"))
    k = HARNESS.evaluate_unit(_unit("k", attacked=np.roll(np.eye(4), 1, axis=0)))
    assert q["descriptor_kind"] == "q" and k["descriptor_kind"] == "k"
    assert q["candidate_correspondences"] != k["candidate_correspondences"]


def test_mutual_nearest_is_deterministic_and_has_no_score_cutoff() -> None:
    reference = np.array(((0.,), (100.,), (200.,), (300.,)))
    attacked = np.array(((.1,), (100.1,), (200.1,), (300.1,)))
    first = HARNESS.evaluate_unit(_unit(reference=reference, attacked=attacked))
    second = HARNESS.evaluate_unit(_unit(reference=reference, attacked=attacked))
    assert first["candidate_correspondences"] == second["candidate_correspondences"]
    assert len(first["candidate_correspondences"]) == 4
    tied = HARNESS.evaluate_unit(_unit(reference=np.ones((4, 1)), attacked=np.ones((4, 1))))
    assert [(item["reference_index"], item["attacked_index"]) for item in tied["candidate_correspondences"]] == [(0, 0)]


def test_underconstrained_and_invalid_units_are_retained_as_failures() -> None:
    fewer = HARNESS.evaluate_unit(_unit(reference=np.eye(2), attacked=np.eye(2), reference_grid=(1, 2), attacked_grid=(1, 2)))
    collinear = HARNESS.evaluate_unit(_unit(reference_grid=(1, 4), attacked_grid=(1, 4)))
    bad = HARNESS.evaluate_unit(_unit(h=np.full((3, 3), np.nan)))
    mismatch = HARNESS.evaluate_unit(_unit(reference=np.ones((4, 2)), attacked=np.ones((4, 3))))
    grid_bad = HARNESS.evaluate_unit(_unit(reference_grid=(3, 3)))
    layer_bad = HARNESS.evaluate_unit({**_unit(), "layer_path": ""})
    records = HARNESS.run_predeclared_units([_unit(), {"pair_id": "bad", "descriptor_kind": "wrong"}])
    assert fewer["failure_reason"] == "fewer_than_three_mutual_candidates"
    assert collinear["failure_reason"] == "collinear_candidate_coordinates"
    assert bad["failure_reason"] == "invalid_h_reference_to_attacked"
    assert mismatch["failure_reason"] == "descriptor_dimension_mismatch"
    assert grid_bad["failure_reason"] == "invalid_reference_grid"
    assert layer_bad["failure_reason"] == "invalid_layer_path"
    assert len(records) == 2 and records[1]["status"] == "failed"


def test_public_record_whitelist_metrics_and_no_promotion_fields() -> None:
    record = HARNESS.evaluate_unit(_unit())
    assert set(record) == HARNESS.public_record_fields()
    assert record["status"] == "calculated"
    assert record["fit_residual"] is not None and record["recovery_error"] is not None
    forbidden = ("reliab", "accept", "detector", "watermark", "threshold", "tau", "route", "scientific")
    assert not any(any(fragment in str(key).lower() for fragment in forbidden) for key in record)
    assert all("score" not in correspondence for correspondence in record["candidate_correspondences"])


def test_descriptor_rank_coverage_gap_and_fit_direction_are_independently_exercised() -> None:
    h = np.eye(3)
    q = HARNESS.evaluate_unit(_unit("q", h=h))
    k = HARNESS.evaluate_unit(_unit("k", h=h, attacked=np.eye(4)[[1, 0, 3, 2]]))
    assert len(q["true_match_ranks"]) == len(k["true_match_ranks"]) == 4
    assert len(q["ambiguity_gaps"]) == len(k["ambiguity_gaps"]) == 4
    assert q["coverage"] == k["coverage"] == 1.0
    assert q["recovery_error"] is not None
