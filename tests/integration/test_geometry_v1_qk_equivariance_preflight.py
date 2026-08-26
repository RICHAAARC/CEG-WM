"""Dynamic CPU/fake coverage for sampled-token Q/K coordinate experiments."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).parents[2] / "experiments" / "run_geometry_v1_qk_equivariance_preflight.py"
SPEC = importlib.util.spec_from_file_location("qk_equivariance", MODULE)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _unit(*, kind="q", h=None, reference=None, attacked=None, reference_grid=(2, 2), attacked_grid=(2, 2), reference_indices=None, attacked_indices=None, **labels):
    reference = np.eye(4) if reference is None else reference
    attacked = np.eye(4) if attacked is None else attacked
    result = {"pair_id": labels.get("pair_id", "pair-1"), "transform_label": labels.get("transform_label", "identity"), "control_label": labels.get("control_label", "matched_h"), "descriptor_kind": kind, "layer_path": labels.get("layer_path", "blocks.0.attn"), "reference_descriptors": reference, "attacked_descriptors": attacked, "reference_source_grid": reference_grid, "attacked_source_grid": attacked_grid, "reference_sample_indices": np.arange(len(reference), dtype=np.int64) if reference_indices is None else reference_indices, "attacked_sample_indices": np.arange(len(attacked), dtype=np.int64) if attacked_indices is None else attacked_indices, "H_reference_to_attacked": np.eye(3) if h is None else h}
    result.update({key: value for key, value in labels.items() if key not in {"pair_id", "transform_label", "control_label", "layer_path"}})
    return result


def test_real_observer_coordinates_from_full_grids_not_fictional_dense_sample_grid() -> None:
    indices = np.array([0, 63, 64, 127, 511, 1023, 2048, 4095], dtype=np.int64)
    assert HARNESS.sampled_pixel_centers((64, 64), indices).tolist() == [[.5, .5], [63.5, .5], [.5, 1.5], [63.5, 1.5], [63.5, 7.5], [63.5, 15.5], [.5, 32.5], [63.5, 63.5]]
    record = HARNESS.evaluate_unit(_unit(reference=np.eye(8), attacked=np.eye(8), reference_grid=(64, 64), attacked_grid=(64, 64), reference_indices=indices, attacked_indices=indices))
    assert record["candidate_correspondences"][4]["reference_xy"] == [63.5, 7.5]


def test_h_is_truth_only_full_fov_and_quantization_is_sampled_stable() -> None:
    ids = np.array([0, 2, 8, 10], dtype=np.int64)
    shifted = np.array(((1., 0., 100.), (0., 1., 0.), (0., 0., 1.)))
    a = HARNESS.evaluate_unit(_unit(reference_grid=(4, 4), attacked_grid=(4, 4), reference_indices=ids, attacked_indices=ids))
    b = HARNESS.evaluate_unit(_unit(reference_grid=(4, 4), attacked_grid=(4, 4), reference_indices=ids, attacked_indices=ids, h=shifted, control_label="shuffled_h"))
    assert a["candidate_correspondences"] == b["candidate_correspondences"]
    assert b["coverage"] == 0.0 and b["true_match_ranks"] == [None] * 4
    assert HARNESS.nearest_sampled_index(np.array((1.5, .5)), HARNESS.sampled_pixel_centers((4, 4), np.array([0, 2]))) == 0


def test_q_k_are_separate_and_score_free_mutual_nearest_is_deterministic() -> None:
    q = HARNESS.evaluate_unit(_unit())
    k = HARNESS.evaluate_unit(_unit(kind="k", attacked=np.eye(4)[[1, 0, 3, 2]]))
    assert q["descriptor_kind"] == "q" and k["descriptor_kind"] == "k" and q["candidate_correspondences"] != k["candidate_correspondences"]
    tied = HARNESS.evaluate_unit(_unit(reference=np.ones((4, 1)), attacked=np.ones((4, 1))))
    assert [(x["reference_index"], x["attacked_index"]) for x in tied["candidate_correspondences"]] == [(0, 0)]


def test_token_bound_precedes_matching_and_64_candidate_limit_is_structural() -> None:
    over = _unit(reference=np.eye(65), attacked=np.eye(65), reference_grid=(9, 9), attacked_grid=(9, 9), reference_indices=np.arange(65), attacked_indices=np.arange(65))
    original = HARNESS._mutual_nearest
    HARNESS._mutual_nearest = lambda *_: (_ for _ in ()).throw(AssertionError("no allocation"))
    try:
        assert HARNESS.evaluate_unit(over)["failure_reason"] == "input_token_bound_exceeded"
    finally:
        HARNESS._mutual_nearest = original
    exact = HARNESS.evaluate_unit(_unit(reference=np.eye(64), attacked=np.eye(64), reference_grid=(8, 8), attacked_grid=(8, 8), reference_indices=np.arange(64), attacked_indices=np.arange(64)))
    assert len(exact["candidate_correspondences"]) == 64 <= HARNESS.MAX_SAMPLED_TOKENS


def test_unit_plan_bound_is_atomic_and_preserves_64_ordered_records() -> None:
    units = [_unit(pair_id=f"p{i}") for i in range(64)]
    assert [x["pair_id"] for x in HARNESS.run_predeclared_units(units)] == [f"p{i}" for i in range(64)]
    original = HARNESS.evaluate_unit
    HARNESS.evaluate_unit = lambda _: (_ for _ in ()).throw(AssertionError("no prefix"))
    try:
        with pytest.raises(ValueError, match="predeclared_unit_bound_exceeded"):
            HARNESS.run_predeclared_units(units + [_unit(pair_id="p64")])
    finally:
        HARNESS.evaluate_unit = original


@pytest.mark.parametrize(("key", "value", "reason"), [("transform_label", "bad", "invalid_transform_label"), ("control_label", "bad", "invalid_control_label"), ("pair_id", "-bad", "invalid_pair_id"), ("pair_id", "x" * 129, "invalid_pair_id"), ("layer_path", "", "invalid_layer_path"), ("layer_path", "x" * 257, "invalid_layer_path")])
def test_frozen_protocol_and_bounded_structural_identifiers_are_retained(key, value, reason) -> None:
    record = HARNESS.evaluate_unit(_unit(**{key: value}))
    assert record["status"] == "failed" and record["failure_reason"] == reason


def test_identifier_has_no_semantic_blacklist_or_governance_behavior() -> None:
    record = HARNESS.evaluate_unit(_unit(pair_id="ordinary.scientific-words:are-data"))
    assert record["failure_reason"] not in {"invalid_pair_id", "invalid_layer_path"}


@pytest.mark.parametrize(("key", "value", "reason"), [("reference_source_grid", (True, 2), "invalid_reference_source_grid"), ("reference_sample_indices", np.array([[0, 1]]), "invalid_reference_sample_indices"), ("reference_sample_indices", np.array([0., 1., 2., 3.]), "invalid_reference_sample_indices"), ("reference_sample_indices", np.array([0, 0, 2, 3]), "invalid_reference_sample_indices"), ("reference_sample_indices", np.array([0, 2, 1, 3]), "invalid_reference_sample_indices"), ("reference_sample_indices", np.array([0, 1, 2, 4]), "invalid_reference_sample_indices"), ("reference_sample_indices", np.array([0, 1]), "reference_sample_count_mismatch")])
def test_invalid_observer_shapes_and_indices_are_retained(key, value, reason) -> None:
    record = HARNESS.evaluate_unit(_unit(**{key: value}))
    assert record["status"] == "failed" and record["failure_reason"] == reason and record["candidate_correspondences"] == []


def test_underconstrained_failures_whitelist_and_no_promotion() -> None:
    fewer = HARNESS.evaluate_unit(_unit(reference=np.eye(2), attacked=np.eye(2), reference_grid=(1, 2), attacked_grid=(1, 2), reference_indices=np.arange(2), attacked_indices=np.arange(2)))
    collinear = HARNESS.evaluate_unit(_unit(reference_grid=(1, 4), attacked_grid=(1, 4)))
    record = HARNESS.evaluate_unit(_unit())
    assert fewer["failure_reason"] == "fewer_than_three_mutual_candidates" and collinear["failure_reason"] == "collinear_candidate_coordinates"
    assert set(record) == HARNESS.public_record_fields() and record["status"] == "calculated"
    assert not {"reliability", "accepted", "detector_score", "watermark_present", "threshold", "tau", "tau_rescue"} & set(record)


def test_separate_source_grids_keep_truth_metrics_and_identity() -> None:
    h = np.array(((1., 0., 1.), (0., 1., 1.), (0., 0., 1.)))
    record = HARNESS.evaluate_unit(_unit(reference_grid=(4, 4), attacked_grid=(6, 6), reference_indices=np.array([0, 3, 12, 15]), attacked_indices=np.array([5, 8, 25, 28]), h=h))
    assert record["reference_grid"] == [4, 4] and record["attacked_grid"] == [6, 6]
    assert len(record["true_match_ranks"]) == len(record["ambiguity_gaps"]) == 4 and record["h_identity"]["shape"] == [3, 3]
