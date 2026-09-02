from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from experiments import run_geometry_v7_r2 as runner
from cegwm.geometry_v7.r1a import R1A_ALL_CONDITIONS, R1A_CORE_CONDITIONS


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_runner_is_pure_cpu_and_exact_bound() -> None:
    source = (ROOT / "experiments/run_geometry_v7_r2.py").read_text(encoding="utf-8")
    assert runner.R1A_PRODUCER_EXACT == "ac590330e91aacf4b3283df1e94572a0e4f983a0"
    assert runner.REPAIR_PRODUCER_EXACT == "3b9819d80b07704a4caab8b7aaa581cf9eb8a3c5"
    for forbidden in ("torch", "PIL", "blind_weighted_scores", "HF_TOKEN", "CEG_WM_ROOT_KEY", "cuda"):
        assert forbidden not in source
    assert "real_h_passed_condition_count\") != 7" in source
    assert "fine_nonzero_prefix_condition_count\") != 9" in source


@pytest.mark.integration
def test_notebook_static_guards() -> None:
    path = ROOT / "notebooks/geometry_v7_r2.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code[0]["source"] == ["from google.colab import drive\n", "drive.mount('/content/drive')"]
    assert all(cell.get("execution_count") is None and cell.get("outputs") == [] for cell in code)
    for cell in code:
        ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell["source"]) for cell in code)
    assert "APPROVED_EXACT = 'PENDING_AFTER_GEOMETRY_V7_R2_PUSH'" in source
    assert "'checkout', '--detach', APPROVED_EXACT" in source
    assert "ac590330e91aacf4b3283df1e94572a0e4f983a0/r1a-f2" in source
    assert "3b9819d80b07704a4caab8b7aaa581cf9eb8a3c5/r1b-repair" in source
    assert " / 'r2-selective'" in source
    assert source.count("experiments.run_geometry_v7_r2") == 1
    assert "userdata" not in source and "cuda" not in source.lower()
    assert "copytree" in source and "exists()" in source


@pytest.mark.integration
def test_operational_payload_keeps_method_axes_null() -> None:
    payload = runner._payload(
        exact="1" * 40, r1a_root=Path("/r1a"), repair_root=Path("/repair"),
        input_error=ValueError("bad input"),
    )
    assert payload["status"] == "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"
    assert payload["selection"] is None and payload["formal_test"] is None
    assert payload["method_verdict"] is None
    assert payload["R1B_FULL_PASS"] is False
    assert payload["R1B_SELECTIVE_CANDIDATE"] is True
    assert payload["R2_SELECTIVE_RELIABILITY_AUTHORIZED"] is True


@pytest.mark.integration
def test_upstream_loaders_bind_counts_order_and_actual_condition_facts(tmp_path: Path) -> None:
    roster = tuple(f"content-v6-iss-eval-{index:04d}" for index in range(1, 9))
    r1a_root = tmp_path / "r1a"
    repair_root = tmp_path / "repair"
    r1a_root.mkdir()
    repair_root.mkdir()
    geometry = {
        "status": "UNRELIABLE", "uncalibrated_sync_logit": 0.0,
        "observed_corners_in_canonical_normalized": [[-1,-1],[1,-1],[1,1],[-1,1]],
        "homography_observed_to_canonical": [[1,0,0],[0,1,0],[0,0,1]],
        "legal": True, "error": None,
    }
    r1a_raw = [
        {"condition_id": spec.condition_id, "condition_kind": spec.kind.value,
         "unit_id": unit, "geometry": geometry}
        for spec in R1A_ALL_CONDITIONS for unit in roster
    ]
    (r1a_root / "result.json").write_text(json.dumps({
        "schema": runner.R1A_SCHEMA, "exact": runner.R1A_PRODUCER_EXACT,
        "status": runner.R1A_REQUIRED_STATUS,
        "r0_input": {"ordered_evaluation_cg_inputs": [{"unit_id": unit} for unit in roster]},
        "raw_records": r1a_raw,
    }), encoding="utf-8")
    memberships = []
    real = []
    evaluations = []
    scores = {"u": {}, "g": {}, "cg": {},
              "positive_cg_vs_g": {"positive": True}, "negative_g_vs_u": {}}
    for index, spec in enumerate(R1A_CORE_CONDITIONS):
        evaluations.append({
            "condition_id": spec.condition_id, "roster": list(roster),
            "real_h_passed": index < 7, "accepted_max_pixels": 1 if index < 9 else 0,
        })
        for unit in roster:
            memberships.append({"condition_id": spec.condition_id, "unit_id": unit,
                                "membership_from_old_r1b": "N_recovery_negative"})
            real.append({
                "condition_id": spec.condition_id, "unit_id": unit,
                "point_kind": "real_h", "radius_pixels": None, "errors": [],
                "scores": scores, "positive_gate_a_delta": 0.1,
                "positive_gate_b_delta": 0.1, "positive_score_delta": 0.1,
                "improved": True, "recovered_negative": True,
                "decision_harm": False, "observed_negative_false_positive": False,
            })
    (repair_root / "result.json").write_text(json.dumps({
        "schema": runner.REPAIR_SCHEMA, "exact": runner.REPAIR_PRODUCER_EXACT,
        "status": runner.REPAIR_REQUIRED_STATUS, "real_h_status": runner.REPAIR_REAL_STATUS,
        "fine_grid_status": runner.REPAIR_FINE_STATUS, "r2_candidate": False,
        "real_h_passed_condition_count": 7, "fine_nonzero_prefix_condition_count": 9,
        "inputs": {"ordered_roster": list(roster)},
        "frozen_old_membership_records": memberships, "real_h_records": real,
        "condition_evaluations": evaluations,
    }, sort_keys=True), encoding="utf-8")

    loaded_roster, features = runner._validate_r1a(r1a_root)
    outcomes, prior = runner._validate_repair(repair_root, loaded_roster)
    assert loaded_roster == roster and len(features) == 80 and len(outcomes) == 80
    assert prior["real_h_passed_condition_count"] == 7
    assert prior["fine_nonzero_prefix_condition_count"] == 9
    assert [item["condition_id"] for item in prior["per_condition_actual"]] == [
        spec.condition_id for spec in R1A_CORE_CONDITIONS
    ]

    evaluations[0]["real_h_passed"] = False
    (repair_root / "result.json").write_text(json.dumps({
        **json.loads((repair_root / "result.json").read_text()),
        "condition_evaluations": evaluations,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="per-condition facts"):
        runner._validate_repair(repair_root, roster)
