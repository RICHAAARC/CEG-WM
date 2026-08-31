import math
from pathlib import Path

import numpy as np
from PIL import Image

from experiments import geometry_v6_r01_engine as engine


def _record(values):
    return {"status": "success", "public_pilot_observation": values}


def test_r01_roster_is_frozen_same_model_and_excludes_r0_identity():
    roster = engine._load_roster(Path("."))
    assert len(roster) == 4
    assert tuple((item["unit_id"], item["seed"]) for item in roster) == tuple((item[0], item[2]) for item in engine._EXPECTED_ROSTER)
    assert all(item["prompt"] != "A watchmaker sorting steel springs beneath a magnifying lamp" and item["seed"] != 2026082400 for item in roster)


def test_r01_carrier_and_quality_are_strict_and_fail_closed():
    absent = _record({"search_score": 1.0, "fit_score": 2.0, "validate_score": 3.0, "aggregate_score": 4.0})
    present = _record({"search_score": 1.1, "fit_score": 2.1, "validate_score": 3.1, "aggregate_score": 4.1})
    assert engine._matched_carrier(present, absent)["status"] == "PASS"
    tied = _record({"search_score": 1.0, "fit_score": 2.1, "validate_score": 3.1, "aggregate_score": 4.1})
    assert engine._matched_carrier(tied, absent)["status"] == "FAIL_CLOSED_NONPOSITIVE_OR_NONFINITE_DELTA"
    clean = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), "RGB")
    marked = Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8), "RGB")
    metrics = engine._v4_rgb_quality(clean, marked)
    assert math.isfinite(metrics["psnr_db"]) and math.isfinite(metrics["ssim"])
    assert engine._matched_quality(None, clean)["status"] == "FAIL_CLOSED_QUALITY_UNAVAILABLE"
