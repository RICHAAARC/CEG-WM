from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from cegwm.channel_precompensation import (
    ReaderSpec,
    decode_score,
    evaluate_delta,
    make_reader,
    nearest_samebit_centers,
    solve_write,
)

pytestmark = pytest.mark.unit
SPEC = ReaderSpec(key_seed=7301)


def _solve(matrix, baseline=(0.0, 0.0), *, bit=1, budget=10.0, method="joint", spec=SPEC, phase=0.0):
    return solve_write(matrix, baseline, bit=bit, spec=spec, phase=phase, l2_budget=budget, method=method)


def test_reader_is_unit_keyed_and_does_not_consume_global_random_state():
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    first, phase = make_reader(65536, SPEC)
    repeated, repeated_phase = make_reader(65536, SPEC)
    other, other_phase = make_reader(65536, ReaderSpec(key_seed=7302))
    assert first.dtype == torch.float64 and first.device.type == "cpu"
    assert float(first.norm()) == pytest.approx(1)
    assert set(first.unique().tolist()) == {-1 / 256, 1 / 256}
    assert torch.equal(first, repeated) and phase == repeated_phase
    assert not torch.equal(first, other) and phase != other_phase
    assert 0 <= phase < 2 * SPEC.delta
    assert torch.equal(torch_state, torch.random.get_rng_state())
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]


def test_phase_draw_is_independent_of_dimension_and_scales_with_lattice_spacing():
    reader, phase = make_reader(32, SPEC)
    _, other_shape_phase = make_reader(128, SPEC)
    scaled_reader, scaled_phase = make_reader(32, ReaderSpec(SPEC.key_seed, delta=3))
    assert phase == other_shape_phase
    assert torch.equal(reader, scaled_reader)
    assert scaled_phase == pytest.approx(3 * phase)


@pytest.mark.parametrize("score,index,bit", [(-1.5, -1, 1), (-0.5, 0, 0), (0.5, 1, 1), (1.5, 2, 0)])
def test_blind_parity_tie_rule_is_explicit_and_does_not_use_target_label(score, index, bit):
    first = decode_score(score, SPEC, phase=0, target_bit=0)
    second = decode_score(score, SPEC, phase=0, target_bit=1)
    assert first["nearest_index"] == second["nearest_index"] == index
    assert first["decoded_bit"] == second["decoded_bit"] == bit
    assert first["signed_margin"] == second["signed_margin"] == 0


def test_target_margin_is_signed_distance_to_the_decision_boundary():
    assert decode_score(1, SPEC, 0, target_bit=1)["signed_margin"] == 0.5
    assert decode_score(1.25, SPEC, 0, target_bit=1)["signed_margin"] == 0.25
    assert decode_score(1.75, SPEC, 0, target_bit=1)["signed_margin"] == -0.25
    assert decode_score(2, SPEC, 0, target_bit=1)["signed_margin"] == -0.5
    centers = nearest_samebit_centers([0, -2, 2], SPEC, 0, bit=1)
    torch.testing.assert_close(centers, torch.tensor([1.0, -1.0, 3.0], dtype=torch.float64))


def test_inner_cell_comparison_preserves_raw_margin_and_only_tolerates_roundoff():
    adjacent = decode_score(math.nextafter(0.75, 0.0), SPEC, 0, target_bit=1)
    assert adjacent["signed_margin"] < 0.25
    assert adjacent["inside_target_inner_cell"] is True
    assert adjacent["inner_cell_violation"] > 0
    assert decode_score(0.75 - 1e-8, SPEC, 0, target_bit=1)["inside_target_inner_cell"] is False
    narrow = ReaderSpec(SPEC.key_seed, inner_fraction=0.2)
    assert decode_score(0.8, narrow, 0, target_bit=1)["inside_target_inner_cell"] is True
    assert decode_score(0.75, narrow, 0, target_bit=1)["inside_target_inner_cell"] is False


@pytest.mark.parametrize("method", ["joint", "gradient"])
def test_orthogonal_channels_reach_inner_faces_at_the_minimum_norm(method):
    result = _solve(torch.eye(2), method=method)
    assert result["status"] == "SOLVED"
    torch.testing.assert_close(result["delta"], torch.tensor([0.75, 0.75], dtype=torch.float64))
    assert result["diagnostics"]["l2_norm"] == pytest.approx(math.sqrt(2) * 0.75)
    assert result["diagnostics"]["predicted_scores"] == pytest.approx([0.75, 0.75])
    assert all(row["decoded_bit"] == 1 for row in result["diagnostics"]["predicted_reads"])
    json.dumps({key: value for key, value in result.items() if key != "delta"}, allow_nan=False)


def test_joint_solution_uses_a_correction_that_the_same_gradient_ray_cannot_supply():
    matrix = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    joint = _solve(matrix, budget=1)
    gradient = _solve(matrix, budget=1, method="gradient")
    assert joint["status"] == "SOLVED"
    # x >= .75 and 2*x+y <= 1.25 force the closest corner (.75,-.25).
    torch.testing.assert_close(joint["delta"], torch.tensor([0.75, -0.25], dtype=torch.float64))
    assert joint["diagnostics"]["l2_norm"] == pytest.approx(math.sqrt(0.75**2 + 0.25**2))
    assert gradient["status"] == "INFEASIBLE" and gradient["delta"] is None
    assert "positive_gradient_ray" in gradient["reason"]
    assert joint["diagnostics"]["centers"] == gradient["diagnostics"]["centers"]
    assert joint["diagnostics"]["lower"] == gradient["diagnostics"]["lower"]
    assert joint["diagnostics"]["upper"] == gradient["diagnostics"]["upper"]


def test_gradient_takes_first_feasible_step_not_an_arbitrary_step_to_centers():
    matrix = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    joint, gradient = _solve(matrix), _solve(matrix, method="gradient")
    torch.testing.assert_close(joint["delta"], torch.tensor([0.75, 0.0], dtype=torch.float64))
    torch.testing.assert_close(gradient["delta"], torch.tensor([0.75, 0.375], dtype=torch.float64))
    assert gradient["diagnostics"]["predicted_scores"] == pytest.approx([0.75, 1.125])
    assert joint["diagnostics"]["l2_norm"] < gradient["diagnostics"]["l2_norm"]


@pytest.mark.parametrize("method", ["joint", "gradient"])
def test_budget_failure_does_not_scale_or_return_an_unusable_write(method):
    result = _solve(torch.eye(2), budget=1.0, method=method)
    assert result["status"] == "OVER_BUDGET" and result["delta"] is None
    assert result["diagnostics"]["l2_norm"] > result["diagnostics"]["l2_budget"]
    assert result["diagnostics"]["within_budget"] is False
    assert result["diagnostics"]["predicted_scores"] == pytest.approx([0.75, 0.75])


@pytest.mark.parametrize("method", ["joint", "gradient"])
def test_zero_budget_no_op_is_valid_for_only_the_already_encoded_bit(method):
    matrix = torch.eye(2)
    no_op = _solve(matrix, bit=0, budget=0, method=method)
    opposite = _solve(matrix, bit=1, budget=0, method=method)
    assert no_op["status"] == "SOLVED"
    assert torch.count_nonzero(no_op["delta"]) == 0
    assert no_op["diagnostics"]["l2_norm"] == 0
    assert opposite["status"] == "OVER_BUDGET" and opposite["delta"] is None


def test_zero_sensitivity_is_not_a_failure_when_that_channel_already_satisfies_its_cell():
    result = _solve(torch.tensor([[0.0, 0.0], [0.0, 2.0]]), baseline=[1, 0])
    assert result["status"] == "SOLVED"
    torch.testing.assert_close(result["delta"], torch.tensor([0.0, 0.375], dtype=torch.float64))
    for method in ("joint", "gradient"):
        impossible = _solve(torch.zeros(2, 2), method=method)
        assert impossible["status"] == "INFEASIBLE" and impossible["delta"] is None


@pytest.mark.parametrize("matrix", [
    [[1.0, 0.0], [2.0, 0.0]],
    [[1.0, 0.0], [-1.0, 0.0]],
])
def test_incompatible_parallel_constraints_remain_failure_in_the_fixed_cells(matrix):
    for method in ("joint", "gradient"):
        result = _solve(matrix, method=method)
        assert result["status"] == "INFEASIBLE" and result["delta"] is None
        assert result["diagnostics"]["centers"] == [1.0, 1.0]


@pytest.mark.parametrize("method", ["joint", "gradient"])
def test_compatible_rank_deficient_constraints_are_not_discarded(method):
    result = _solve([[1.0, 0.0], [1.5, 0.0]], method=method)
    assert result["status"] == "SOLVED"
    torch.testing.assert_close(result["delta"], torch.tensor([0.75, 0.0], dtype=torch.float64))


def test_underdetermined_solution_has_no_unnecessary_nullspace_component_and_keeps_shape():
    matrix = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    result = _solve(matrix)
    assert result["delta"].shape == (1, 3)
    torch.testing.assert_close(result["delta"], torch.tensor([[0.75, 0.75, 0.0]], dtype=torch.float64))


@pytest.mark.parametrize("method", ["joint", "gradient"])
def test_changing_score_units_with_the_lattice_leaves_the_write_unchanged(method):
    matrix = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    original = _solve(matrix, method=method)
    scaled = _solve(matrix * 7, spec=ReaderSpec(SPEC.key_seed, delta=7), method=method)
    assert original["status"] == scaled["status"] == "SOLVED"
    torch.testing.assert_close(original["delta"], scaled["delta"])
    assert scaled["diagnostics"]["predicted_scores"] == pytest.approx(np.array(original["diagnostics"]["predicted_scores"]) * 7)


def test_postcast_zero_write_has_failed_predicted_margin_despite_feasible_float64_solution():
    matrix = torch.tensor([[1000.0], [1000.0]])
    solution = _solve(matrix, budget=0.001)
    assert solution["status"] == "SOLVED"
    base = torch.tensor([1000.0], dtype=torch.float16)
    written = (base.double() + solution["delta"]).to(base.dtype)
    actual = written.double() - base.double()
    assert torch.count_nonzero(actual) == 0
    observed_geometry = evaluate_delta(actual, matrix, [0, 0], bit=1, spec=SPEC, phase=0, l2_budget=0.001)
    assert observed_geometry["within_budget"] is True
    assert observed_geometry["max_interval_violation"] == 0.75
    assert observed_geometry["within_inner_cells"] is False
    assert [row["signed_margin"] for row in observed_geometry["predicted_reads"]] == [-0.5, -0.5]


@pytest.mark.parametrize("bad", [
    {"key_seed": -1}, {"key_seed": True}, {"key_seed": 1, "delta": 0},
    {"key_seed": 1, "delta": float("nan")}, {"key_seed": 1, "inner_fraction": 0.5},
])
def test_invalid_reader_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        ReaderSpec(**bad)


def test_malformed_or_nonfinite_inputs_are_not_reported_as_channel_failure():
    with pytest.raises(ValueError, match="shape"):
        _solve(torch.ones(3, 2))
    with pytest.raises(ValueError, match="finite"):
        _solve(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="nonnegative"):
        _solve(torch.eye(2), budget=-1)
    with pytest.raises(ValueError, match="bit"):
        _solve(torch.eye(2), bit=2)
