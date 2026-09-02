from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import run_geometry_v7_r4 as entry
from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r3 import R3_B_LOW


def _paired(margin: float):
    return {
        "gate_a_margin": margin,
        "gate_b_margin": margin,
        "margin": margin,
        "positive": margin > 0.0,
    }


def _identity_rows(roster):
    return [
        {"condition_id": condition, "unit_id": unit}
        for condition in R2_CONDITION_IDS for unit in roster
    ]


def _is_direct_negative(split: str, condition: str, unit: str) -> bool:
    if condition != "core_fixed_canvas_zoom_0_8":
        return condition not in ("core_rotation_neg15", "core_rotation_pos15")
    return unit in (
        {R2_DEV_UNIT_IDS[1], R2_DEV_UNIT_IDS[3]}
        if split == "dev" else {R2_TEST_UNIT_IDS[2]}
    )


def _advanced_row(split: str, condition: str, unit: str):
    direct_negative = _is_direct_negative(split, condition, unit)
    rotation = condition in ("core_rotation_neg15", "core_rotation_pos15")
    zoom = condition == "core_fixed_canvas_zoom_0_8"
    angle = -15.0 if condition == "core_rotation_neg15" else (15.0 if rotation else 0.0)
    return {
        "split": split,
        "condition_id": condition,
        "unit_id": unit,
        "route": "DIRECT_NEGATIVE" if direct_negative else "BOUNDARY",
        "r2_selector_accepted": not direct_negative,
        "old_cycle_score_px": 4.0 if zoom and not direct_negative else None,
        "decision": {
            "pure_rotation_gate": rotation,
            "regime": {
                "valid": True,
                "angle_degrees": angle,
                "scale": 1.25 if zoom else 1.0,
                "translation": 0.0,
                "perspective": 0.0,
            },
        },
    }


def _fixtures():
    roster = R2_DEV_UNIT_IDS + R2_TEST_UNIT_IDS
    pre, real = [], []
    for condition in R2_CONDITION_IDS:
        for unit in roster:
            split = "dev" if unit in R2_DEV_UNIT_IDS else "test"
            direct_negative = _is_direct_negative(split, condition, unit)
            s0 = R3_B_LOW - 0.5 if direct_negative else -1.0
            identity = {"condition_id": condition, "unit_id": unit}
            pre.append({
                **identity,
                "membership": "N_recovery_negative",
                "pre_scores_from_old_r1b": {
                    "positive_cg_vs_g": _paired(s0),
                    "negative_g_vs_u": _paired(-1.0),
                },
            })
            real.append({
                **identity,
                "errors": [],
                "positive_score_delta": 2.0,
                "recovered_negative": True,
                "scores": {
                    "positive_cg_vs_g": _paired(1.0),
                    "negative_g_vs_u": _paired(-1.0),
                },
            })
    repair = {
        "schema": entry.REPAIR_SCHEMA,
        "exact": entry.REPAIR_EXACT,
        "status": "R1B_REPAIR_REAL_H_NOT_END_TO_END_READY",
        "frozen_old_membership_records": pre,
        "real_h_records": real,
    }
    fixed = _identity_rows(roster)
    r2 = {
        "schema": entry.R2_SCHEMA,
        "exact": entry.R2_EXACT,
        "status": "R2_SELECTIVE_RISK_FAILED",
        "feature_rows": fixed,
        "outcome_rows": fixed,
    }
    dev = [
        _advanced_row("dev", condition, unit)
        for condition in R2_CONDITION_IDS for unit in R2_DEV_UNIT_IDS
    ]
    test = [
        _advanced_row("test", condition, unit)
        for condition in R2_CONDITION_IDS for unit in R2_TEST_UNIT_IDS
    ]
    advanced = {
        "schema": entry.ADVANCED_SCHEMA,
        "exact": entry.ADVANCED_EXACT,
        "status": "R3_ADVANCED_ENGINEERING_TEST40_RECORDED",
        "inputs": {
            "r1a": {"producer_exact": entry.R1A_EXACT},
            "r1b_repair": {"producer_exact": entry.REPAIR_EXACT},
            "r2": {"producer_exact": entry.R2_EXACT},
            "old_r3": {"producer_exact": entry.OLD_R3_EXACT},
        },
        "development_decisions": dev,
        "existing_test40_decisions": test,
    }
    return repair, r2, advanced


def _write_result(root: Path, payload):
    root.mkdir()
    (root / "result.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_real_shape_artifacts_replay_frozen_metrics_and_retain_all_rows():
    repair, _r2, advanced = _fixtures()
    rows, refined, scoped = entry._assemble(repair, advanced)
    assert len(rows) == 80
    assert refined[0]["accepted_count"] == refined[0]["safe_rescue_count"] == 10
    assert refined[1]["accepted_count"] == refined[1]["safe_rescue_count"] == 11
    assert all(metric["unsafe_accept_count"] == 0 for metric in refined)
    assert all(metric["selected_negative_control_fp_count"] == 0 for metric in refined)
    assert [metric["covered_attack_count"] for metric in refined] == [3, 3]
    assert [metric["fixed_denominator"] for metric in scoped] == [12, 12]
    assert [metric["final_positive_count"] for metric in scoped] == [10, 11]
    assert [metric["net_rescue_change"] for metric in scoped] == [10, 11]
    assert all(metric["baseline_positive_count"] == 0 for metric in scoped)
    assert sum(
        row["in_scope"] and row["route"] == "DIRECT_NEGATIVE" for row in rows
    ) == 3
    assert sum(not row["in_scope"] for row in rows) == 56


def test_loader_validates_exact_bindings_and_condition_major_orders(tmp_path):
    repair, r2, advanced = _fixtures()
    roots = tuple(tmp_path / name for name in ("repair", "r2", "advanced"))
    for root, payload in zip(roots, (repair, r2, advanced), strict=True):
        _write_result(root, payload)
    loaded = entry._validate_inputs(*roots)
    assert loaded[0]["exact"] == entry.REPAIR_EXACT
    assert loaded[1]["status"] == "R2_SELECTIVE_RISK_FAILED"
    assert loaded[2]["inputs"]["old_r3"]["producer_exact"] == entry.OLD_R3_EXACT

    advanced["development_decisions"][0], advanced["development_decisions"][1] = (
        advanced["development_decisions"][1], advanced["development_decisions"][0]
    )
    (roots[2] / "result.json").write_text(
        json.dumps(advanced, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="split identity/order"):
        entry._validate_inputs(*roots)


def test_loader_rejects_upstream_binding_drift(tmp_path):
    repair, r2, advanced = _fixtures()
    advanced["inputs"]["r1a"]["producer_exact"] = "0" * 40
    roots = tuple(tmp_path / name for name in ("repair", "r2", "advanced"))
    for root, payload in zip(roots, (repair, r2, advanced), strict=True):
        _write_result(root, payload)
    with pytest.raises(ValueError, match="input binding"):
        entry._validate_inputs(*roots)


def test_result_is_create_only_and_has_bounded_engineering_claim(tmp_path):
    repair, r2, advanced = _fixtures()
    rows, refined, scoped = entry._assemble(repair, advanced)
    result = tmp_path / "result"
    payload = entry._payload(
        exact="1" * 40,
        repair_root=tmp_path / "repair",
        r2_root=tmp_path / "r2",
        advanced_root=tmp_path / "advanced",
        rows=rows,
        refined=refined,
        scoped=scoped,
    )
    entry._write(result, payload)
    stored = json.loads((result / "result.json").read_text(encoding="utf-8"))
    assert stored["status"] == entry.R4_ENGINEERING_REPLAY_RECORDED
    assert stored["scientific_status"] == "not_adjudicated"
    assert stored["route"]["r4_promotion_claim"] is False
    assert stored["route"]["actual_callback_not_executed"] is True
    with pytest.raises(FileExistsError):
        entry._write(result, payload)


def test_runner_source_is_cpu_artifact_only():
    source = Path(entry.__file__).read_text(encoding="utf-8")
    for forbidden in ("cuda", "SyncSeal", "score_rgb(", "generate(", "google.colab"):
        assert forbidden not in source
