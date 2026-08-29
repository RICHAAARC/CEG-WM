from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cegwm.method import geometry_v4_proxy as proxy
from cegwm.protocol.geometry_v4 import derive_geometry_v4_key
from cegwm.protocol.geometry_v4_proxy import P1_ATTACKS, P1_DIGEST, P1_SPLITS, load_p1_proxy
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
    assert set(P1_SPLITS["P1D"]).isdisjoint(P1_SPLITS["P1C"])


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
