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
        for cycles in (8, 16, 32)
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
def test_normalized_cross_power_and_blind_identity_observation_use_measured_matches() -> None:
    reference = np.zeros((32, 32), dtype=np.float64)
    reference[5, 7] = 1.0
    moving = np.roll(np.roll(reference, 4, axis=0), -3, axis=1)
    correlation = proxy.normalized_phase_correlation(moving, reference)
    assert (correlation["shift_y"], correlation["shift_x"]) == (4, -3)
    assert correlation["PSR"] > 8.0

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
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    marked, _ = proxy.write_proxy(rgb, KEY)
    attacked, _ = proxy.apply_proxy_attack(marked, "rotation_+5")
    detection = proxy.detect_proxy(attacked, KEY)
    estimates = detection["diagnostics"]["cross_scale_estimates"]
    assert len(estimates) == 3
    assert detection["diagnostics"]["cross_scale_log_scale_spread"] > 0.03
    assert detection["status"] == "UNRELIABLE"


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
    raw_estimates = detection["diagnostics"]["cross_scale_raw_estimates"]
    assert len(raw_estimates) == 3
    assert all({"rotation_deg", "log_scale", "scale"} == set(item) for item in raw_estimates)
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
