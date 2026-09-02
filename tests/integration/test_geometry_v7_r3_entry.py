from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    D4Transform,
    GeometryEstimate,
    GeometryStatus,
    d4_homography,
)
from cegwm.geometry_v7.r1a import apply_homography
from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS
from cegwm.geometry_v7.r3 import R3DevUnit
from experiments import run_geometry_v7_r3 as entry


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "geometry_v7_r3.ipynb"


def _transpose(matrix):
    return tuple(tuple(float(matrix[j][i]) for j in range(3)) for i in range(3))


def test_d4_probe_composition_is_identity_and_detector_receives_only_rgb():
    calls = []

    def detector(image):
        assert isinstance(image, Image.Image) and image.mode == "RGB" and image.size == (512, 512)
        transform = tuple(D4Transform)[len(calls)]
        calls.append(image)
        h2 = _transpose(d4_homography(transform))
        points = apply_homography(h2, CANONICAL_CORNERS_NORMALIZED)
        return GeometryEstimate(
            GeometryStatus.UNRELIABLE, 0.0, points, points, h2, True, True, None
        )

    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    branches = entry._probe_unit(Image.new("RGB", (512, 512)), identity, detector)
    assert len(calls) == 8
    assert tuple(item.transform for item in branches) == entry.D4_ORDER
    assert all(item.errors == () and item.cycle_pixels == 0.0 for item in branches)
    assert all(item.h_probe is not None and item.expected_inverse_d is not None for item in branches)
    assert branches[0].h_probe == identity


def test_illegal_or_error_geometry_is_a_branch_failure_not_false_valid_cycle():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    illegal = GeometryEstimate(
        GeometryStatus.UNSUPPORTED, 0.0, None, None, None, False, False, "nonconvex"
    )
    branches = entry._probe_unit(Image.new("RGB", (512, 512)), identity, lambda image: illegal)
    assert len(branches) == 8
    assert all(item.geometry["legal"] is False for item in branches)
    assert all(item.cycle_pixels is None and item.errors for item in branches)


def test_ineligible_fixed_rows_never_call_detector():
    units = tuple(
        R3DevUnit(condition, unit, 0.1, True, True, True, False, False)
        for condition in R2_CONDITION_IDS for unit in R2_DEV_UNIT_IDS
    )

    def forbidden(_image):
        raise AssertionError("ineligible detector call")

    rows = entry._rows_for_split(
        split="dev", units=units, b_low=-0.1, r1a={}, r1a_root=ROOT,
        detector=forbidden, allow_probe=True,
    )
    assert len(rows) == 40
    assert all(not branch.probed for row in rows for branch in row.branches)
    assert all(row.errors == ("R3_NOT_PROBED_INELIGIBLE:DIRECT_POSITIVE",) for row in rows)


def test_runner_source_has_no_content_route_and_cli_help_is_safe():
    source = (ROOT / "experiments" / "run_geometry_v7_r3.py").read_text(encoding="utf-8")
    assert "blind_weighted_scores" not in source
    assert "root key" not in source.lower()
    assert "HF_TOKEN" not in source
    assert "detector(probe)" in source
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.run_geometry_v7_r3", "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0
    assert "--r2-artifact-root" in completed.stdout


def test_real_shape_failed_r2_artifact_is_the_only_accepted_old_verdict(tmp_path):
    roster = tuple(f"content-v6-iss-eval-{index:04d}" for index in range(1, 9))
    identities = tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in roster)
    features = [
        {"condition_id": condition, "unit_id": unit, "mandatory_valid": True,
         "area_ratio": entry.R2_SELECTED_THRESHOLD}
        for condition, unit in identities
    ]
    outcomes = [
        {"condition_id": condition, "unit_id": unit, "membership": "N_recovery_negative",
         "complete": True, "safe": True, "safe_rescue": True,
         "observed_negative_false_positive": False, "errors": []}
        for condition, unit in identities
    ]
    payload = {
        "schema": entry.R2_SCHEMA, "exact": entry.R2_PRODUCER_EXACT,
        "status": "R2_SELECTIVE_RISK_FAILED", "ordered_roster": list(roster),
        "R1B_FULL_PASS": False, "R1B_SELECTIVE_CANDIDATE": True,
        "R2_SELECTIVE_RELIABILITY_AUTHORIZED": True,
        "prior_aggregate_visibility": True,
        "feature_rows": features, "outcome_rows": outcomes,
        "selection": {"selected": {
            "candidate_id": entry.R2_SELECTED_ID,
            "components": [{"feature": "area_ratio", "direction": "ge",
                            "threshold": entry.R2_SELECTED_THRESHOLD}],
        }},
        "formal_test": {"metrics": {
            "accepted_count": 24, "unsafe_accept_count": 6,
            "selected_negative_control_fp_count": 1, "covered_attack_count": 7,
        }},
    }
    root = tmp_path / "r2"
    root.mkdir()
    (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    actual_roster, actual_features, actual_outcomes = entry._validate_r2(root)
    assert actual_roster == roster and len(actual_features) == len(actual_outcomes) == 80
    payload["status"] = "R2_SELECTIVE_RISK_PASSED_PARTIAL_FAMILY_COVERAGE"
    (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        entry._validate_r2(root)
    except ValueError as error:
        assert "identity differs" in str(error)
    else:
        raise AssertionError("old R2 verdict drift was accepted")


def test_r3_notebook_is_unexecuted_exact_bound_phase_a_and_create_only():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code[0]["source"] == [
        "from google.colab import drive\n", "drive.mount('/content/drive')"
    ]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    for cell in code:
        ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell["source"]) for cell in code)
    assert "force_remount" not in source
    assert "PENDING_AFTER_GEOMETRY_V7_R3_PUSH" in source
    assert "checkout', '--detach', APPROVED_EXACT" in source
    assert "ac590330e91aacf4b3283df1e94572a0e4f983a0/r1a-f2" in source
    assert "3b9819d80b07704a4caab8b7aaa581cf9eb8a3c5/r1b-repair" in source
    assert "ffac9d4c1e575c27240d9423bbd30e0713aa2dcd/r2-selective" in source
    assert "'r3-exploratory'" in source
    assert source.count("experiments.run_geometry_v7_r3") == 1
    assert "copytree" in source and "if DRIVE_RESULT_DIR.exists()" in source
    assert "userdata" not in source and "TOKEN" not in source and "KEY" not in source
