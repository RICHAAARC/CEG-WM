import json
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


def _content_record(gates, positive=True):
    return {"content_evidence": {"per_unit_frozen_content_positive": positive, "per_unit_frozen_content_evidence": gates}}


def test_r01_notebook_has_fixed_execution_handoff_and_create_only_drive_sink():
    notebook = json.loads(Path("notebooks/geometry_v6_r01_colab.ipynb").read_text(encoding="utf-8"))
    text = json.dumps(notebook)
    assert notebook["cells"][0]["source"] == ["from google.colab import drive\n", "drive.mount('/content/drive')\n"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in notebook["cells"])
    assert "force_remount" not in text
    assert "APPROVED_EXECUTION_EXACT='bcecbf63a2218eabd7dd878f19dd379feedf2b26'" in text
    assert "['git','checkout','--detach',APPROVED_EXECUTION_EXACT]" in text
    assert "git('rev-parse','HEAD')==APPROVED_EXECUTION_EXACT" in text
    assert "git('branch','--show-current')==''" in text and "git('status','--porcelain')==''" in text
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V6/R01" in text
    assert "RUN_ROOT.mkdir(parents=True,exist_ok=False)" in text
    assert "[sys.executable,'-m','experiments.geometry_v6_r01_engine'" in text
    assert "--expected-exact" in text and "--output-json" in text


def test_r01_one_complete_amplitude_is_a_candidate_and_all_incomplete_fail_closed():
    summaries = [
        {"amplitude": 0.0025, "passed_units": 4, "status": "PASS"},
        {"amplitude": 0.005, "passed_units": 3, "status": "FAIL_CLOSED"},
        {"amplitude": 0.01, "passed_units": 0, "status": "FAIL_CLOSED"},
    ]
    assert engine._carrier_window(summaries) == (True, [0.0025])
    assert engine._carrier_window([{**item, "passed_units": 3, "status": "FAIL_CLOSED"} for item in summaries]) == (False, [])


def test_r01_content_compatibility_requires_matching_complete_six_gate_vector():
    gates = {
        "lf_gate_a_diagnostic": False,
        "lf_gate_b_diagnostic": True,
        "hf_gate_a_diagnostic": False,
        "hf_gate_b_diagnostic": True,
        "weighted_gate_a": True,
        "weighted_gate_b": False,
    }
    assert engine._content_compatibility(_content_record(gates), _content_record(dict(gates))) is True
    changed = dict(gates); changed["hf_gate_b_diagnostic"] = False
    assert engine._content_compatibility(_content_record(gates), _content_record(changed)) is False
    missing = dict(gates); missing.pop("weighted_gate_b")
    assert engine._content_compatibility(_content_record(missing), _content_record(missing)) is False
    non_bool = dict(gates); non_bool["weighted_gate_a"] = 1
    assert engine._content_compatibility(_content_record(non_bool), _content_record(non_bool)) is False
    assert engine._content_compatibility(_content_record(gates, positive=False), _content_record(gates)) is False
