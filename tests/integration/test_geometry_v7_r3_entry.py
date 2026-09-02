from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED, D4Transform, GeometryEstimate, GeometryStatus,
    d4_homography,
)
from cegwm.geometry_v7.r1a import apply_homography
from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS
from cegwm.geometry_v7.r3 import R3Unit
from experiments import run_geometry_v7_r3 as entry


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "geometry_v7_r3.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _state_source():
    code = [cell for cell in _notebook()["cells"] if cell["cell_type"] == "code"]
    return "".join(code[1]["source"])


def _helpers():
    tree = ast.parse(_state_source())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.If):
            break
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef)):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def _write_complete(root, exact, status="R3_METHOD_NOT_IMPROVED"):
    root.mkdir(parents=True)
    payload = {"schema": "geometry_v7_r3_exploratory_result_v1", "exact": exact, "status": status}
    (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _transpose(matrix):
    return tuple(tuple(float(matrix[j][i]) for j in range(3)) for i in range(3))


def test_probe_records_frozen_d_hprobe_expected_inverse_h2_and_residual():
    calls = []

    def detector(image):
        assert isinstance(image, Image.Image) and image.mode == "RGB"
        transform = tuple(D4Transform)[len(calls)]
        calls.append(image)
        h2 = _transpose(d4_homography(transform))
        points = apply_homography(h2, CANONICAL_CORNERS_NORMALIZED)
        return GeometryEstimate(
            GeometryStatus.UNRELIABLE, 0.0, points, points, h2, True, True, None
        )

    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    branches = entry._probe_unit(Image.new("RGB", (512, 512)), identity, detector)
    assert len(calls) == len(branches) == 8
    assert all(item.errors == () and item.cycle_pixels == 0.0 for item in branches)
    assert branches[0].h_probe == identity
    assert all(item.d_matrix is not None and item.expected_inverse_d is not None for item in branches)
    assert all(item.geometry["homography_observed_to_canonical"] is not None for item in branches)


def test_illegal_branch_and_eligible_input_failure_remain_recorded():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    illegal = GeometryEstimate(
        GeometryStatus.UNSUPPORTED, 0.0, None, None, None, False, False, "nonconvex"
    )
    branches = entry._probe_unit(Image.new("RGB", (512, 512)), identity, lambda image: illegal)
    assert len(branches) == 8
    assert all(item.geometry["legal"] is False and item.errors for item in branches)

    units = tuple(
        R3Unit("dev", condition, unit, -1.0, True, True, True, True, False, True, False)
        for condition in R2_CONDITION_IDS for unit in R2_DEV_UNIT_IDS
    )
    rows = entry._rows_for_split(
        split="dev", units=units, r1a={}, r1a_root=ROOT,
        detector=lambda image: illegal, allow_probe=True,
    )
    assert len(rows) == 40
    assert all(len(row.branches) == 8 and row.errors == ("cycle_feature_invalid",) for row in rows)
    assert all("eligible_probe_input" in branch.errors[0] for row in rows for branch in row.branches)


def test_runtime_predicate_has_no_truth_content_result_or_attack_label_and_h0_is_unchanged():
    source = (ROOT / "experiments" / "run_geometry_v7_r3.py").read_text(encoding="utf-8")
    module = (ROOT / "src" / "cegwm" / "geometry_v7" / "r3.py").read_text(encoding="utf-8")
    assert "detector(probe)" in source
    assert "blind_weighted_scores" not in source
    assert "truth" not in module.lower()
    assert "attack_label" not in module
    assert '"final_recovery_uses_raw_h0_once": True' in source
    assert '"h0_updated_replaced_or_averaged": False' in source
    assert "generate_cycle_candidates" not in module and "CycleStump" not in module


def test_runner_accepts_and_preserves_the_observed_failed_r2_identity(tmp_path):
    roster = tuple(f"content-v6-iss-eval-{index:04d}" for index in range(1, 9))
    identities = tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in roster)
    payload = {
        "schema": entry.R2_SCHEMA, "exact": entry.R2_PRODUCER_EXACT,
        "status": "R2_SELECTIVE_RISK_FAILED", "ordered_roster": list(roster),
        "R1B_FULL_PASS": False, "R1B_SELECTIVE_CANDIDATE": True,
        "R2_SELECTIVE_RELIABILITY_AUTHORIZED": True, "prior_aggregate_visibility": True,
        "feature_rows": [{"condition_id": c, "unit_id": u, "mandatory_valid": True,
                          "area_ratio": entry.R2_SELECTED_THRESHOLD} for c, u in identities],
        "outcome_rows": [{"condition_id": c, "unit_id": u} for c, u in identities],
        "selection": {"selected": {"candidate_id": entry.R2_SELECTED_ID,
                      "components": [{"feature": "area_ratio", "direction": "ge",
                                      "threshold": entry.R2_SELECTED_THRESHOLD}]}},
        "formal_test": {"metrics": {"accepted_count": 24, "unsafe_accept_count": 6,
                        "selected_negative_control_fp_count": 1, "covered_attack_count": 7}},
    }
    root = tmp_path / "r2"
    root.mkdir()
    (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    assert len(entry._validate_r2(root)[1]) == 80


def test_top_status_requires_both_development_and_existing_test40_usable():
    root = Path("/record-only")
    selection = SimpleNamespace(
        status="R3_METHOD_IMPROVED", selected_threshold_px=1,
        selected_metrics=SimpleNamespace(usable=True),
    )
    unusable_test = SimpleNamespace(usable=False)
    payload = entry._payload(
        exact="0" * 40, r1a_root=root, repair_root=root, r2_root=root,
        selection=selection, test_metrics=unusable_test,
    )
    assert payload["status"] == "R3_METHOD_NOT_IMPROVED"
    assert payload["development_threshold_selection"] is selection
    assert payload["existing_test40_engineering_diagnostic"] is unusable_test


def test_top_status_is_improved_only_when_both_observed_splits_are_usable():
    root = Path("/record-only")
    selection = SimpleNamespace(
        status="R3_METHOD_IMPROVED", selected_threshold_px=1,
        selected_metrics=SimpleNamespace(usable=True),
    )
    usable_test = SimpleNamespace(usable=True)
    payload = entry._payload(
        exact="0" * 40, r1a_root=root, repair_root=root, r2_root=root,
        selection=selection, test_metrics=usable_test,
    )
    assert payload["status"] == "R3_METHOD_IMPROVED"


def test_drive_complete_skips_and_partial_conflict_is_precise(tmp_path):
    helpers = _helpers()
    exact = "a" * 40
    drive = tmp_path / "drive-result"
    local = tmp_path / "local-result"
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _write_complete(drive, exact, "R3_METHOD_IMPROVED")
    plan = helpers["plan_result_state"](drive, local, checkpoint, exact)
    assert plan["DRIVE_STATE"]["payload"] == payload
    assert plan["RUN_REQUIRED"] is False and plan["PUBLISH_REQUIRED"] is False

    conflict = tmp_path / "drive-partial"
    conflict.mkdir()
    with pytest.raises(RuntimeError, match=f"Drive result conflict at {conflict}: result_json_absent"):
        helpers["plan_result_state"](conflict, local, checkpoint, exact)


def test_local_complete_is_publish_only_and_residual_paths_are_preserved(tmp_path):
    helpers = _helpers()
    exact = "b" * 40
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    checkpoint = tmp_path / "checkpoint.pt"
    _write_complete(local, exact, "OPERATIONAL_FAILURE")
    plan = helpers["plan_result_state"](drive, local, checkpoint, exact)
    assert plan["LOCAL_RESULT_DIR"] == local
    assert plan["RUN_REQUIRED"] is False and plan["PUBLISH_REQUIRED"] is True

    residual = tmp_path / "residual"
    residual.mkdir()
    checkpoint.write_text("preserve", encoding="utf-8")
    plan = helpers["plan_result_state"](drive, residual, checkpoint, exact)
    assert residual.exists() and checkpoint.read_text(encoding="utf-8") == "preserve"
    assert plan["LOCAL_RESULT_DIR"] != residual and not plan["LOCAL_RESULT_DIR"].exists()
    assert plan["SYNCSEAL_CHECKPOINT"] != checkpoint and not plan["SYNCSEAL_CHECKPOINT"].exists()
    assert plan["RUN_REQUIRED"] is True and plan["PUBLISH_REQUIRED"] is True


def test_checkout_reuses_only_exact_clean_and_isolates_invalid_without_mutation(tmp_path):
    helpers = _helpers()
    exact = "c" * 40
    checkout = tmp_path / "CEG-WM"
    checkout.mkdir()

    def clean_run(args, **kwargs):
        return SimpleNamespace(stdout=exact + "\n" if "rev-parse" in args else "")

    reusable = helpers["plan_checkout"](checkout, exact, run=clean_run)
    assert reusable["reuse"] is True and reusable["path"] == checkout

    def wrong_run(args, **kwargs):
        return SimpleNamespace(stdout="d" * 40 + "\n" if "rev-parse" in args else "")

    isolated = helpers["plan_checkout"](checkout, exact, run=wrong_run)
    assert isolated["reuse"] is False and isolated["path"] != checkout
    assert checkout.exists() and not isolated["path"].exists()


def test_state_cell_defines_all_cross_cell_variables_on_sequential_reexecution(tmp_path):
    exact = "e" * 40
    source = _state_source().replace(
        "PENDING_AFTER_GEOMETRY_V7_R3_RESUME_PUSH", exact
    )
    source = source.replace("/content/drive/MyDrive", str(tmp_path / "drive"))
    source = source.replace("/content", str(tmp_path / "content"))
    namespace = {}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    for name in (
        "DRIVE_STATE", "LOCAL_STATE", "LOCAL_RESULT_DIR", "SYNCSEAL_CHECKPOINT",
        "RUN_REQUIRED", "PUBLISH_REQUIRED", "checkout", "CHECKOUT_REUSED",
    ):
        assert name in namespace


def test_runner_return_codes_require_a_new_complete_result(tmp_path):
    helpers = _helpers()
    exact = "f" * 40
    result = tmp_path / "result"
    _write_complete(result, exact)
    assert helpers["validate_runner_completion"](0, result, exact)["state"] == "COMPLETE"
    assert helpers["validate_runner_completion"](2, result, exact)["state"] == "COMPLETE"
    with pytest.raises(RuntimeError, match="unexpected code 1"):
        helpers["validate_runner_completion"](1, result, exact)
    with pytest.raises(RuntimeError, match="result incomplete"):
        helpers["validate_runner_completion"](0, tmp_path / "absent", exact)


def test_notebook_and_cli_are_thin_create_only_phase_a_guards():
    notebook = _notebook()
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code[0]["source"] == [
        "from google.colab import drive\n", "drive.mount('/content/drive')"
    ]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    for cell in code:
        ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell["source"]) for cell in code)
    assert "PENDING_AFTER_GEOMETRY_V7_R3_RESUME_PUSH" in source and "--detach" in source
    assert source.count("experiments.run_geometry_v7_r3") == 1
    assert "force_remount" not in source and "userdata" not in source
    assert all(name in source for name in ("r1a-f2", "r1b-repair", "r2-selective", "r3-exploratory"))
    assert "copytree" in source and "Drive result conflict" in source
    assert "existing_drive_artifacts_ready" in source
    assert "inspect_result(DRIVE_RESULT_DIR, APPROVED_EXACT)" in source
    assert "if not RUN_REQUIRED" in source and "if not CHECKOUT_REUSED" in source
    assert "sys.path" not in source and "exec(open" not in source
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.run_geometry_v7_r3", "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0
