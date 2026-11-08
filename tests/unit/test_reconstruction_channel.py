from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cegwm.reconstruction_channel import (
    measure_feature_triple,
    measure_probe_geometry,
    measure_source,
    planned_call_budget,
)

pytestmark = pytest.mark.unit


def _geometry(a_plus: float = 0.3, a_minus: float = -0.2) -> dict:
    return measure_probe_geometry(
        np.zeros(1), np.array([a_plus]), np.array([a_minus]),
        np.ones(1), nominal_amplitude=0.25,
    )


def _affine_features(geometry: dict, offset: np.ndarray, slope: np.ndarray) -> dict:
    return {
        "zero": offset.copy(),
        "plus": offset + geometry["plus_signed_amplitude"] * slope,
        "minus": offset + geometry["minus_signed_amplitude"] * slope,
    }


def test_actual_float16_amplitudes_remove_nominal_and_asymmetry_bias() -> None:
    epsilon = 0.0006
    base = np.array([1.0], dtype=np.float16)
    plus = np.array([1.0 + epsilon], dtype=np.float16)
    minus = np.array([1.0 - epsilon], dtype=np.float16)
    geometry = measure_probe_geometry(base, plus, minus, np.ones(1), nominal_amplitude=epsilon)
    assert geometry["plus_signed_amplitude"] == 2 ** -10
    assert geometry["minus_signed_amplitude"] == -(2 ** -11)
    assert geometry["signed_midpoint"] != 0
    features = _affine_features(geometry, np.array([2.0, -3.0]), np.array([7.0, -4.0]))
    result = measure_feature_triple(**features, probe_geometry=geometry)
    np.testing.assert_allclose(result["response_secant"], [7.0, -4.0])
    np.testing.assert_allclose(result["response_centered_three_point"], [7.0, -4.0])
    np.testing.assert_allclose(result["even_residual_at_zero"], 0, atol=1e-15)
    assert result["midpoint_even_residual_raw_l2"] > 0
    assert not np.allclose((features["plus"] - features["minus"]) / (2 * epsilon), [7.0, -4.0])


def test_quadratic_distinguishes_secant_centered_response_and_even_residual() -> None:
    geometry = _geometry()
    offset, slope, curvature = np.array([4.0, -3.0]), np.array([2.0, 5.0]), np.array([7.0, -2.0])
    def feature(a: float) -> np.ndarray:
        return offset + a * slope + a * a * curvature
    result = measure_feature_triple(feature(0), feature(0.3), feature(-0.2), probe_geometry=geometry)
    np.testing.assert_allclose(result["response_secant"], slope + 0.1 * curvature)
    np.testing.assert_allclose(result["response_centered_three_point"], slope)
    np.testing.assert_allclose(result["even_residual_at_zero"], 0.06 * curvature)
    np.testing.assert_allclose(result["midpoint_even_residual_raw"], 0.05 * slope + 0.065 * curvature)


def test_off_axis_actual_dtype_error_is_reported_not_projected_away() -> None:
    base = np.array([1.0, 2.0], dtype=np.float16)
    plus = np.array([1.001, 2.0], dtype=np.float16)
    minus = np.array([0.9995, 1.999], dtype=np.float16)
    geometry = measure_probe_geometry(base, plus, minus, np.ones(2), nominal_amplitude=0.001)
    assert geometry["plus_off_axis_l2"] > 0
    assert geometry["minus_off_axis_l2"] > 0
    assert geometry["antisymmetry_residual_l2"] > 0
    assert geometry["plus_delta_l2"] > geometry["plus_signed_amplitude"]


def test_linear_full_feature_map_can_have_even_residual_from_off_axis_motion() -> None:
    base = np.zeros(2)
    plus, minus = np.array([0.3, 0.1]), np.array([-0.2, 0.1])
    geometry = measure_probe_geometry(base, plus, minus, np.array([1.0, 0.0]), nominal_amplitude=0.25)
    # The feature map x -> x[1] is exactly linear, while both actual
    # perturbations move away from the requested x[0] axis in the same sense.
    result = measure_feature_triple(base[1:], plus[1:], minus[1:], probe_geometry=geometry)
    np.testing.assert_allclose(result["even_residual_at_zero"], [0.1])
    assert geometry["plus_off_axis_l2"] == pytest.approx(0.1)
    assert geometry["minus_off_axis_l2"] == pytest.approx(0.1)
    assert "not a nonlinearity verdict" in result["interpretation"]


def test_cpu_bfloat16_geometry_uses_actual_values_without_numpy_dtype_loss() -> None:
    base = torch.ones(2, dtype=torch.bfloat16)
    plus, minus = base + 0.015625, base - 0.015625
    geometry = measure_probe_geometry(base, plus, minus, torch.ones(2), nominal_amplitude=0.02)
    assert geometry["actual_dtype"] == "torch.bfloat16"
    assert geometry["plus_signed_amplitude"] == pytest.approx(0.015625 * np.sqrt(2))
    features = {"zero": torch.zeros(2), "plus": torch.ones(2), "minus": -torch.ones(2)}
    result = measure_feature_triple(**features, probe_geometry=geometry)
    assert result["response_secant_l2"] > 0


def test_host_drift_and_gain_signs_are_preserved_per_seed() -> None:
    geometry = _geometry()
    clean_zero = np.array([2.0, -1.0])
    clean = _affine_features(geometry, clean_zero, np.array([3.0, 4.0]))
    reconstructed = {
        11: _affine_features(geometry, clean_zero + [-7, 0.5], np.array([-5.0, 2.0])),
        22: _affine_features(geometry, clean_zero + [1, -0.5], np.array([5.0, -2.0])),
    }
    result = measure_source("source-a", clean_features=clean, reconstruction_features_by_seed=reconstructed,
                            expected_reconstruction_seeds=[11, 22], probe_geometry=geometry)
    assert result["status"] == "MEASURED"
    assert result["expected_arm_count"] == result["observed_arm_count"] == 9
    np.testing.assert_allclose(result["reconstruction"]["11"]["response_secant"], [-5, 2])
    np.testing.assert_allclose(result["reconstruction"]["22"]["response_secant"], [5, -2])
    np.testing.assert_allclose(result["reconstruction"]["11"]["host_drift"]["vector"], [-7, 0.5])
    np.testing.assert_allclose(result["reconstruction"]["22"]["host_drift"]["vector"], [1, -0.5])
    assert "PASS" not in json.dumps(result, allow_nan=False)


def test_missing_seed_and_failed_arm_retain_the_fixed_source_denominator() -> None:
    geometry = _geometry()
    features = _affine_features(geometry, np.ones(2), np.ones(2))
    result = measure_source("source-b", clean_features=features,
                            reconstruction_features_by_seed={11: features},
                            expected_reconstruction_seeds=[11, 22], probe_geometry=geometry,
                            failures={"reconstruction/11/plus": "recorded generation failure"})
    assert result["source_id"] == "source-b"
    assert result["status"] == "INCOMPLETE"
    assert result["expected_arm_count"] == 9
    assert result["observed_arm_count"] == 5
    assert result["arm_states"]["reconstruction/11/plus"]["status"] == "FAILED"
    assert result["arm_states"]["reconstruction/22/zero"]["status"] == "MISSING"
    assert result["reconstruction"]["11"]["host_drift"]["status"] == "MEASURED"
    assert result["reconstruction"]["22"]["status"] == "INCOMPLETE"


def test_empty_source_is_retained_without_fabricated_measurements() -> None:
    result = measure_source("source-missing", clean_features={}, reconstruction_features_by_seed={},
                            expected_reconstruction_seeds=[4], probe_geometry=None)
    assert result["status"] == "INCOMPLETE"
    assert result["expected_arm_count"] == 6
    assert result["observed_arm_count"] == 0
    assert all(row["status"] == "MISSING" for row in result["arm_states"].values())


def test_nan_observation_is_recorded_invalid_without_dropping_other_arms() -> None:
    geometry = _geometry()
    features = _affine_features(geometry, np.zeros(1), np.ones(1))
    clean = {**features, "plus": np.array([np.nan])}
    result = measure_source("source-invalid", clean_features=clean,
                            reconstruction_features_by_seed={1: features},
                            expected_reconstruction_seeds=[1], probe_geometry=geometry)
    assert result["arm_states"]["clean/plus"]["status"] == "INVALID"
    assert result["observed_arm_count"] == 5
    assert result["reconstruction"]["1"]["host_drift"]["status"] == "MEASURED"
    json.dumps(result, allow_nan=False)


def test_collapsed_actual_amplitude_remains_visible_but_cannot_be_a_secant() -> None:
    geometry = measure_probe_geometry(np.ones(1, dtype=np.float16), np.ones(1, dtype=np.float16),
                                      np.ones(1, dtype=np.float16), np.ones(1), nominal_amplitude=1e-8)
    assert geometry["amplitudes_bracket_zero"] is False
    features = {arm: np.ones(1) for arm in ("zero", "plus", "minus")}
    result = measure_source("source-collapsed", clean_features=features,
                            reconstruction_features_by_seed={1: features},
                            expected_reconstruction_seeds=[1], probe_geometry=geometry)
    assert result["clean"]["status"] == "INVALID"
    assert result["status"] == "INCOMPLETE"
    assert result["observed_arm_count"] == 6


def test_compact_summary_retains_scalar_diagnostics_without_feature_vectors() -> None:
    geometry = _geometry()
    features = _affine_features(geometry, np.zeros(4), np.ones(4))
    result = measure_source("source-compact", clean_features=features,
                            reconstruction_features_by_seed={1: features},
                            expected_reconstruction_seeds=[1], probe_geometry=geometry, include_vectors=False)
    assert "response_secant" not in result["clean"]
    assert result["clean"]["response_secant_l2"] == pytest.approx(2)
    assert "vector" not in result["reconstruction"]["1"]["host_drift"]
    json.dumps(result, allow_nan=False)


def test_actual_dtype_and_feature_shape_cannot_silently_change() -> None:
    with pytest.raises(ValueError, match="same actual dtype"):
        measure_probe_geometry(np.zeros(1, dtype=np.float16), np.ones(1), -np.ones(1),
                               np.ones(1), nominal_amplitude=1)
    with pytest.raises(ValueError, match="matching shapes"):
        measure_feature_triple(np.zeros(2), np.ones(1), -np.ones(2), probe_geometry=_geometry())


@pytest.mark.parametrize("kwargs", [
    {"expected_reconstruction_seeds": [1, "1"]},
    {"reconstruction_features_by_seed": {2: {}}},
    {"failures": {"reconstruction/99/zero": "unplanned"}},
])
def test_unplanned_or_duplicate_units_are_rejected(kwargs: dict) -> None:
    arguments = dict(clean_features={}, reconstruction_features_by_seed={},
                     expected_reconstruction_seeds=[1], probe_geometry=None)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        measure_source("source", **arguments)


def test_single_direction_budget_counts_full_generation_and_reconstruction_calls() -> None:
    budget = planned_call_budget(4, 2)
    assert budget["generation_calls"] == 12
    assert budget["reconstruction_calls"] == 24
    assert budget["public_feature_calls"] == 36
    assert budget["generation_formula"] == "n * (1 + 2 * d)"
    assert "checkpoint_capture_runs" not in budget
    with pytest.raises(ValueError, match="exactly one"):
        planned_call_budget(4, 2, direction_count=2)
    with pytest.raises(ValueError, match="positive integer"):
        planned_call_budget(0, 2)
