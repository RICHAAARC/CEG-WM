from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cegwm.method import geometry_v4_proxy as proxy
from cegwm.protocol.geometry_v4 import derive_geometry_v4_key
from cegwm.protocol.geometry_v4_proxy import (
    P1_ATTACKS,
    P1_DEVELOPMENT_CANARY_ATTACKS,
    P1_DEVELOPMENT_CANARY_SEEDS,
    P1_DIGEST,
    P1_H_DIRECTION,
    P1_SCALE_BOUNDS,
    P1_SPLITS,
    load_p1_proxy,
)
from cegwm.shared.keys import normalize_detection_key

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "geometry_v4" / "geometry_v4_p1_proxy_v1.json"
KEY = "0123456789abcdef"
WRONG_KEY = "fedcba9876543210"


def _record_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | set().union(*(_record_keys(item) for item in value.values()), set())
    if isinstance(value, (tuple, list)):
        return set().union(*(_record_keys(item) for item in value), set())
    return set()


@pytest.mark.unit
def test_proxy_config_digest_split_roster_and_canonical_bytes() -> None:
    contract = load_p1_proxy(ROOT)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == P1_DIGEST
    assert CONFIG.read_bytes() == (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode()
    assert tuple(contract["attacks"]) == P1_ATTACKS
    assert tuple(contract["splits"]["P1D"]["seeds"]) == P1_SPLITS["P1D"]
    assert tuple(contract["splits"]["P1C"]["seeds"]) == P1_SPLITS["P1C"]
    assert tuple(contract["development_canary"]["seeds"]) == P1_DEVELOPMENT_CANARY_SEEDS
    assert tuple(contract["development_canary"]["attacks"]) == P1_DEVELOPMENT_CANARY_ATTACKS
    assert set(P1_SPLITS["P1D"]).isdisjoint(P1_SPLITS["P1C"])
    assert contract["attack_operator"]["public_h_direction"] == P1_H_DIRECTION == "attacked_to_canonical"
    assert tuple(contract["detector"]["coarse_scale_bounds"]) == P1_SCALE_BOUNDS
    assert contract["detector"]["cross_scale_estimation"] == "keyed_sparse_constellation_glrt_primary_v1"
    assert contract["detector"]["cross_scale_group_score"] == "four_component_joint_glrt_with_geometric_mean_completeness"
    assert contract["detector"]["whole_log_polar_role"] == "diagnostic_only"
    assert P1_SCALE_BOUNDS[1] >= 1 / 0.7
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    _, attacked_to_canonical = proxy.apply_proxy_attack(rgb, "crop_rescale_0.7")
    constructed_scale = math.sqrt(
        np.linalg.det(np.linalg.inv(np.asarray(attacked_to_canonical).reshape(3, 3))[:2, :2])
    )
    assert P1_SCALE_BOUNDS[0] <= constructed_scale <= P1_SCALE_BOUNDS[1]


@pytest.mark.unit
def test_writer_has_twelve_independent_global_components_fixed_tiles_and_final_luma_budget() -> None:
    geometry_key = derive_geometry_v4_key(normalize_detection_key(KEY))
    phase_signs = {
        proxy._phase_sign(geometry_key, f"global/{cycles}/{direction}")
        for cycles in (8, 16, 24)
        for direction in (0, 45, 90, 135)
    }
    assert len(phase_signs) == 12
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    marked, budget = proxy.write_proxy(rgb, KEY)
    energy = budget["anchor_energy"]
    assert energy["direction_count"] == 4
    assert energy["scale_count"] == 3
    assert energy["global_component_count"] == 12
    assert energy["tile_count"] == 16
    assert energy["global_energy_fraction"] == pytest.approx(0.40, abs=1e-12)
    assert energy["local_energy_fraction"] == pytest.approx(0.60, abs=1e-12)
    assert abs(energy["global_local_cross"]) < 1e-12
    assert budget["luma_rms"] > 0.0
    assert budget["luma_rms"] <= 2 / 255
    assert budget["luma_peak"] <= 8 / 255
    assert np.max(np.abs(marked - rgb)) <= 8 / 255 + 1e-12
    assert not {"key_digest", "derived_key", "root_key", "pattern"} & _record_keys(budget)


@pytest.mark.unit
def test_balanced_magnitude_codebook_and_weighted_components_are_publicly_safe() -> None:
    assert len(proxy._MAGNITUDE_CODEBOOK) == 60
    assert sum(level * level for level in proxy._MAGNITUDE_LEVELS) / 4.0 == pytest.approx(1.0)
    assert min(right - left for left, right in zip(proxy._MAGNITUDE_LEVELS, proxy._MAGNITUDE_LEVELS[1:])) >= 0.19
    assert all(sorted(label for row in code for label in row) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] for code in proxy._MAGNITUDE_CODEBOOK)
    for code in proxy._MAGNITUDE_CODEBOOK:
        assert proxy._magnitude_worst_radial_ratio(code) == pytest.approx(proxy._MAGNITUDE_OPTIMAL_RADIAL_RATIO)
    assert len(proxy._MAGNITUDE_CODEBOOK) == 60
    key = derive_geometry_v4_key(normalize_detection_key(KEY))
    assert proxy._magnitude_code(key) == proxy._magnitude_code(key)
    _, global_field, _, _, components, diagnostics = proxy._anchor_fields((64, 64), key)
    ratios = sorted(round(float(np.sqrt(np.mean(np.square(value)))), 9) for value in components.values())
    assert ratios == sorted(round(value, 9) for value in proxy._MAGNITUDE_LEVELS for _ in range(3))
    assert np.sqrt(np.mean(np.square(global_field))) == pytest.approx(1.0)
    assert not {"assignment", "weight", "label", "code", "permutation", "index", "digest"} & _record_keys(diagnostics)


@pytest.mark.unit
def test_fixed_multiradius_constellation_groups_and_orbit_order() -> None:
    key = derive_geometry_v4_key(normalize_detection_key(KEY))
    _, _, _, _, components, _ = proxy._anchor_fields((64, 64), key)
    groups = proxy._constellation_groups(components)
    identities = [item for _, group in groups for item in group]
    assert len(groups) == 3 and len(identities) == len(set(identities)) == 12
    assert all(len(group) == 4 and len({item[1] for item in group}) == 4 and len({item[0] for item in group}) >= 3 for _, group in groups)
    assert proxy._angle_orbits_45(0.0) == (0.0, 45.0, -90.0, -45.0)
    assert len(proxy._orbit_assignments()) == 64 and proxy._orbit_assignments()[0] == (0, 0, 0) and proxy._orbit_assignments()[-1] == (3, 3, 3)


@pytest.mark.unit
def test_mixed_group_consensus_and_raw_group_spreads() -> None:
    result = proxy._mixed_group_consensus([(89.0, -0.1, 1.0), (-89.0, 0.0, 1.0), (89.0, 0.1, 1.0)])
    assert result["valid"] is True
    assert result["raw_consensus"] == pytest.approx((89.0, 0.0))
    rotation, scale = proxy._raw_group_spreads(((89.0, -10.0), (-89.0, 10.0), (89.0, 0.0)), (89.0, 0.0))
    assert rotation == pytest.approx(2.0) and scale == pytest.approx(10.0)


@pytest.mark.unit
def test_quadratic_radial_peak_is_independent_and_fail_closed() -> None:
    assert proxy._quadratic_peak_delta(-1.5625, -0.0625, -0.5625) == pytest.approx(0.25)
    assert proxy._quadratic_peak_delta(-0.5625, -0.0625, -1.5625) == pytest.approx(-0.25)
    assert proxy._quadratic_peak_delta(0.0, 0.0, -1.0) is None
    assert proxy._quadratic_peak_delta(0.0, 1.0, float("nan")) is None
    assert proxy._raw_group_spreads(((0.0, -0.1), (0.0, 0.0), (0.0, 0.1)), (0.0, 0.0))[1] > 0.03


@pytest.mark.unit
def test_sparse_glrt_pure_helpers_and_identity_recovery() -> None:
    plane = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    assert np.array_equal(proxy._sparse_constant_border_reflection(plane), plane)
    for axis, reverse in ((0, False), (0, True), (1, False), (1, True)):
        padded = plane.copy()
        selector = slice(None, 7) if not reverse else slice(-7, None)
        if axis == 0:
            padded[selector, :] = 0.5
        else:
            padded[:, selector] = 0.5
        reflected = proxy._sparse_constant_border_reflection(padded)
        assert reflected is not None and not np.all(reflected == 0.5)
    assert proxy._sparse_constant_border_reflection(np.full((64, 64), 0.5)) is None
    grid = proxy._zero_anchored_log_grid()
    assert 0.0 in grid and math.log(0.65) in grid and math.log(1.55) in grid
    assert all(abs(value / 0.01 - round(value / 0.01)) < 1e-12 for value in grid[1:-1])
    image = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    assert proxy._toroidal_bilinear_patch(image, -0.5, -0.5).shape == (11, 11)
    lobe = proxy._hann_dtft_lobe(np.arange(-5, 6, dtype=np.float64))
    assert int(np.argmax(lobe)) == 5
    key = derive_geometry_v4_key(normalize_detection_key(KEY))
    global_field, _, components = proxy._global_fields((64, 64), key)
    records = proxy._sparse_constellation_diagnostic(global_field, components)
    assert len(records) == 3
    assert all(record["valid"] and abs(record["raw_rotation_deg"]) <= 0.5 and abs(record["raw_log_scale"]) <= 0.011 for record in records)


@pytest.mark.unit
def test_sparse_endpoint_raw_is_invalid_and_cannot_seed_rectification(monkeypatch: pytest.MonkeyPatch) -> None:
    key = derive_geometry_v4_key(normalize_detection_key(KEY))
    _, global_reference, _, by_scale, components, _ = proxy._anchor_fields((64, 64), key)
    identities = tuple(components)
    endpoint_records = tuple(
        {
            "identities": identities[index * 4 : (index + 1) * 4],
            "raw_rotation_deg": 16.0,
            "raw_log_scale": 0.0,
                "score": 1.0,
                "margin": 1.0,
                "boundary": True,
                "valid": False,
        }
        for index in range(3)
    )
    monkeypatch.setattr(proxy, "_sparse_constellation_diagnostic", lambda *_: endpoint_records)
    result = proxy._refine_rotation_scale(np.zeros((64, 64, 3), dtype=np.float64), global_reference, by_scale, components)
    assert result["raw_valid"] is False
    assert all(math.isnan(value) for value in result["rectification_seed"])


@pytest.mark.unit
def test_normalized_cross_power_and_blind_identity_observation_use_measured_matches() -> None:
    left = np.arange(25, dtype=np.float64).reshape(5, 5)
    mask = np.ones((5, 5), dtype=bool)
    assert proxy._masked_normalized_patch_score(left, left, mask) == pytest.approx(proxy._normalized_patch_score(left, left))
    perturbed = left.copy()
    mask[0, 0] = False
    perturbed[0, 0] = 1e9
    assert proxy._masked_normalized_patch_score(left, perturbed, mask) == pytest.approx(1.0)
    assert proxy._masked_normalized_patch_score(left, left, np.eye(5, dtype=bool)) is None
    reference = np.zeros((32, 32), dtype=np.float64)
    reference[5, 7] = 1.0
    moving = np.roll(np.roll(reference, 4, axis=0), -3, axis=1)
    correlation = proxy.normalized_phase_correlation(moving, reference)
    assert (correlation["shift_y"], correlation["shift_x"]) == (4, -3)
    assert correlation["PSR"] > 8.0
    translation = proxy._translation_phase_correlation(reference, reference, np.ones_like(reference, dtype=bool))
    assert (translation["shift_y"], translation["shift_x"]) == (0, 0)
    assert translation["PSR"] > 8.0

    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    marked, _ = proxy.write_proxy(rgb, KEY)
    correct = proxy.detect_proxy(marked, KEY)
    negative = proxy.detect_proxy(rgb, KEY)
    wrong = proxy.detect_proxy(marked, WRONG_KEY)
    assert correct["status"] == "RELIABLE"
    assert correct["support"] == len(correct["diagnostics"]["matches"])
    assert correct["support"] >= 6
    assert negative["status"] == "UNRELIABLE"
    assert wrong["status"] == "UNRELIABLE"
    assert not {"key_digest", "derived_key", "root_key", "pattern"} & _record_keys(correct)


@pytest.mark.unit
def test_mixed_log_polar_diagnostic_keeps_angular_wrap_and_radial_axis_linear() -> None:
    generator = np.random.default_rng(17)
    reference = generator.normal(size=(360, 256))

    def linear_radial_shift(values: np.ndarray, shift: int) -> np.ndarray:
        result = np.zeros_like(values)
        if shift >= 0:
            result[:, shift:] = values[:, : values.shape[1] - shift]
        else:
            result[:, :shift] = values[:, -shift:]
        return result

    moved = linear_radial_shift(np.roll(reference, 7, axis=0), 5)
    diagnostic = proxy._mixed_logpolar_correlation(moved, reference, 0.01)
    assert diagnostic["surface"].shape == (360, 511)
    assert diagnostic["global"]["shift_y"] == diagnostic["measured"]["shift_y"] == 7
    assert diagnostic["global"]["shift_x"] == diagnostic["measured"]["shift_x"] == 5
    assert diagnostic["measured"]["rotation_deg"] == pytest.approx(3.5)
    assert diagnostic["measured"]["log_scale"] == pytest.approx(-0.05, abs=1e-4)
    assert diagnostic["measured"]["valid"] is True

    moved_negative = linear_radial_shift(np.roll(reference, -9, axis=0), -6)
    negative = proxy._mixed_logpolar_correlation(moved_negative, reference, 0.01)
    assert negative["global"]["shift_y"] == -9
    assert negative["global"]["shift_x"] == -6
    assert negative["global"]["log_scale"] == pytest.approx(0.06)
    assert negative["measured"]["score_identity"] == "support_weighted_primary"
    assert negative["global"]["score_identity"] == "unweighted_coherence_diagnostic_only"
    assert negative["measured"]["score"] == pytest.approx(
        negative["measured"]["unweighted_coherence"] * math.sqrt(negative["measured"]["overlap_fraction"])
    )
    assert negative["measured"]["score"] < negative["measured"]["unweighted_coherence"]


@pytest.mark.unit
def test_mixed_log_polar_diagnostic_separates_global_and_domain_measurement() -> None:
    generator = np.random.default_rng(23)
    reference = generator.normal(size=(360, 256))
    moved = np.zeros_like(reference)
    moved[:, 70:] = reference[:, :-70]
    diagnostic = proxy._mixed_logpolar_correlation(moved, reference, 0.01)
    assert diagnostic["global"]["shift_x"] == 70
    assert diagnostic["global"]["log_scale"] == pytest.approx(-0.70)
    assert diagnostic["global"]["log_scale"] < math.log(0.65)
    assert diagnostic["measured"]["log_scale"] >= math.log(0.65)
    assert diagnostic["measured"]["index"] != diagnostic["global"]["index"]

    angular_boundary = np.roll(reference, 32, axis=0)
    boundary = proxy._mixed_logpolar_correlation(angular_boundary, reference, 0.01)
    assert boundary["measured"]["rotation_deg"] == pytest.approx(16.0)
    assert boundary["measured"]["boundary"] is True
    assert boundary["measured"]["valid"] is False


@pytest.mark.unit
def test_public_h_is_attacked_to_canonical_and_rectifies_nonidentity() -> None:
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    marked, _ = proxy.write_proxy(rgb, KEY)
    attacked, truth = proxy.apply_proxy_attack(marked, "rotation_+5")
    detection = proxy.detect_proxy(attacked, KEY)
    assert detection["diagnostics"]["public_h_direction"] == "attacked_to_canonical"
    assert detection["H_hat"] is not None
    estimate = np.asarray(detection["H_hat"]).reshape(3, 3)
    assert np.max(np.abs(estimate - np.asarray(truth).reshape(3, 3))) < 0.02
    rectified = proxy.rectify_proxy(attacked, detection["H_hat"])
    wrong_direction = proxy._sample_h(attacked, estimate, float(np.median(attacked)))
    assert np.mean(np.square(rectified - marked)) < np.mean(np.square(wrong_direction - marked)) / 2
    assert np.asarray(detection["corners_hat"]) == pytest.approx(np.asarray(proxy._corners(estimate)))


@pytest.mark.unit
def test_independent_cross_scale_disagreement_fails_closed_instead_of_being_windowed_pass() -> None:
    raw_groups = ((0.0, -0.1), (0.0, 0.0), (0.0, 0.1))
    rotation, scale = proxy._raw_group_spreads(raw_groups, (0.0, 0.0))
    assert rotation == 0.0 and scale > 0.03


@pytest.mark.unit
def test_multiscale_raw_rotation_scale_evidence_is_periodic_and_not_clamped() -> None:
    angle, scale = proxy._quality_weighted_consensus(((89.0, 0.9, 2.0), (-89.0, 1.1, 2.0)))
    assert abs(abs(angle) - 90.0) < 1e-9
    assert scale == pytest.approx(math.sqrt(0.99))

    raw_rotation_spread, raw_log_scale_spread = proxy._raw_cross_scale_spreads(
        ((89.0, math.log(0.1)), (-89.0, math.log(10.0)), (0.0, 0.0)), 0.0, 1.0
    )
    bounded_log_scale_spread = math.log(1.55) - math.log(0.65)
    assert raw_rotation_spread == pytest.approx(89.0)
    assert raw_log_scale_spread > 2.0 > bounded_log_scale_spread
    assert raw_log_scale_spread > 0.03  # frozen gate must see disagreement before search clipping

    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    marked, _ = proxy.write_proxy(rgb, KEY)
    detection = proxy.detect_proxy(marked, KEY)
    estimates = detection["diagnostics"]["cross_scale_estimates"]
    assert len(estimates) == len(detection["diagnostics"]["cross_scale_quality"]) == 3
    assert detection["diagnostics"]["cross_scale_estimates_role"] == "diagnostic_only"
    raw_groups = detection["diagnostics"]["joint_group_raw_observations"]
    resolved_groups = detection["diagnostics"]["resolved_group_raw_estimates"]
    assert len(raw_groups) == 3
    assert all({"identities", "raw_rotation_deg", "raw_log_scale", "sparse_glrt_score", "sparse_glrt_margin", "measurement_id", "valid", "boundary"} == set(item) for item in raw_groups)
    assert all(item["measurement_id"] == "sparse_keyed_spectrum_glrt_v1" for item in raw_groups)
    if detection["diagnostics"]["sparse_group_raw_valid"]:
        assert len(resolved_groups) == 3
        raw_consensus = detection["diagnostics"]["raw_group_consensus"]
        expected_spreads = proxy._raw_group_spreads(
            tuple((item["rotation_deg"], item["log_scale"]) for item in resolved_groups),
            (raw_consensus["rotation_deg"], raw_consensus["log_scale"]),
        )
        assert detection["diagnostics"]["cross_scale_rotation_spread_deg"] == pytest.approx(expected_spreads[0])
        assert detection["diagnostics"]["cross_scale_log_scale_spread"] == pytest.approx(expected_spreads[1])
    else:
        assert not resolved_groups and detection["status"] == "UNRELIABLE"
        assert math.isinf(detection["diagnostics"]["cross_scale_rotation_spread_deg"])
        assert math.isinf(detection["diagnostics"]["cross_scale_log_scale_spread"])
    if detection["diagnostics"]["sparse_group_raw_valid"]:
        assert all(0.65 <= estimate[1] <= 1.55 for estimate in estimates)
        assert detection["diagnostics"]["valid_overlap_fraction"] == pytest.approx(1.0)
        assert detection["diagnostics"]["matches"]
        assert all(math.isfinite(match["PSR"]) for match in detection["diagnostics"]["matches"])


@pytest.mark.unit
def test_robust_similarity_excludes_outliers_and_coverage_uses_inlier_geometry() -> None:
    truth = proxy._similarity_h(4.0, 1.05, 0.03, -0.02)
    canonical = (
        (0.125, 0.125),
        (0.375, 0.125),
        (0.875, 0.125),
        (0.125, 0.625),
        (0.875, 0.625),
        (0.125, 0.875),
        (0.625, 0.875),
        (0.875, 0.875),
    )
    matches = []
    for index, (x, y) in enumerate(canonical):
        point = truth @ np.asarray((x, y, 1.0))
        matches.append(
            {
                "tile": (index // 4, index % 4),
                "canonical": (x, y),
                "attacked": (float(point[0]), float(point[1])),
                "correlation": 0.9,
            }
        )
    matches.extend(
        (
            {"tile": (2, 2), "canonical": (0.375, 0.375), "attacked": (0.9, 0.1), "correlation": 0.95},
            {"tile": (2, 3), "canonical": (0.625, 0.375), "attacked": (0.1, 0.9), "correlation": 0.95},
        )
    )
    estimate, inliers, residuals, condition = proxy._robust_similarity_fit(matches)
    assert estimate is not None
    assert len(inliers) == len(canonical)
    assert all(match["tile"] not in {(2, 2), (2, 3)} for match in inliers)
    assert np.max(np.abs(estimate - truth)) < 1e-12
    assert np.max(residuals) < 1e-12 and condition < 1e4
    assert proxy._spatial_coverage(inliers) >= 0.75
    assert proxy._spatial_coverage(inliers[:3]) < 0.75
