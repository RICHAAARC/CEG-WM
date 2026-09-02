from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r3 import R3Unit
from cegwm.geometry_v7.r3_advanced import (
    R3_ADVANCED_OPERATIONAL_FAILURE,
    R3_ADVANCED_TEST40_RECORDED,
)
from experiments import run_geometry_v7_r3_advanced as entry


def _old_result():
    rows = [
        {"condition_id": condition, "unit_id": unit, "split": split}
        for split, roster in (("dev", R2_DEV_UNIT_IDS), ("test", R2_TEST_UNIT_IDS))
        for condition in R2_CONDITION_IDS for unit in roster
    ]
    return {
        "schema": entry.OLD_R3_SCHEMA, "exact": entry.OLD_R3_PRODUCER_EXACT,
        "status": entry.OLD_R3_STATUS, "feature_rows": rows,
        "development_threshold_selection": {
            "selected_threshold_px": None, "selected_metrics": None,
            "grid_metrics": [{
                "threshold_px": 8.0,
                "fixed_denominator": 40, "accepted_count": 8,
                "safe_rescue_count": 8, "unsafe_accept_count": 0,
                "selected_negative_control_fp_count": 0, "covered_attack_count": 5,
            }],
        },
    }


def test_old_r3_loader_binds_exact_immutable_baseline_and_order(tmp_path):
    root = tmp_path / "old-r3"
    root.mkdir()
    (root / "result.json").write_text(json.dumps(_old_result()), encoding="utf-8")
    rows = entry._validate_old_r3(root)
    assert len(rows) == 80
    assert tuple((row["condition_id"], row["unit_id"], row["split"]) for row in rows[:40]) == tuple(
        (condition, unit, "dev") for condition in R2_CONDITION_IDS for unit in R2_DEV_UNIT_IDS
    )
    assert tuple((row["condition_id"], row["unit_id"], row["split"]) for row in rows[40:]) == tuple(
        (condition, unit, "test") for condition in R2_CONDITION_IDS for unit in R2_TEST_UNIT_IDS
    )
    changed = _old_result()
    changed["development_threshold_selection"]["grid_metrics"][0]["safe_rescue_count"] = 7
    (root / "result.json").write_text(json.dumps(changed), encoding="utf-8")
    try:
        entry._validate_old_r3(root)
    except ValueError as error:
        assert "baseline differs" in str(error)
    else:
        raise AssertionError("drifted old R3 baseline was accepted")


def test_r4_readiness_requires_complete_probes_safe_rescue_translations_and_rotations():
    conditions = []
    for condition in R2_CONDITION_IDS:
        conditions.append(SimpleNamespace(
            condition_id=condition, accepted_count=1, safe_rescue_count=1,
        ))
    metrics = SimpleNamespace(
        accepted_count=10, safe_rescue_count=10, unsafe_accept_count=0,
        selected_negative_control_fp_count=0, per_attack=tuple(conditions),
    )
    rows = []
    for condition in R2_CONDITION_IDS:
        for unit in R2_TEST_UNIT_IDS:
            branches = tuple(SimpleNamespace(probed=True) for _ in range(8))
            rows.append(SimpleNamespace(
                condition_id=condition, route="BOUNDARY", r2_selector_accepted=True,
                branches=branches,
            ))
    assert entry._r4_readiness(metrics, rows)["candidate"] is True
    conditions[4] = SimpleNamespace(
        condition_id=R2_CONDITION_IDS[4], accepted_count=0, safe_rescue_count=0,
    )
    assert entry._r4_readiness(
        SimpleNamespace(**{**metrics.__dict__, "per_attack": tuple(conditions)}), rows,
    )["candidate"] is False


def test_operational_payload_keeps_method_axes_unpromoted_and_fixed_test_denominator():
    root = Path("/record-only")
    error = RuntimeError("setup interrupted")
    payload = entry._payload(
        exact="0" * 40, r1a_root=root, repair_root=root, r2_root=root,
        old_r3_root=root, operational_error=error,
    )
    assert payload["status"] == R3_ADVANCED_OPERATIONAL_FAILURE
    assert payload["r4_candidate"] is False and payload["fallback_scope"] is None
    assert payload["test_probe_accounting"]["fixed_denominator"] == 40
    assert "setup interrupted" in payload["operational_error"]


def test_complete_test40_payload_records_engineering_status_not_self_adjudication():
    root = Path("/record-only")
    payload = entry._payload(
        exact="0" * 40, r1a_root=root, repair_root=root, r2_root=root,
        old_r3_root=root,
    )
    assert payload["status"] == R3_ADVANCED_TEST40_RECORDED
    assert payload["r4_readiness"]["requires_agent5_adjudication"] is True


def test_setup_failure_rows_retain_all_40_and_all_eligible_failures():
    units = tuple(
        R3Unit("test", condition, unit, -1.0, True, True, True, True, False, True, False)
        for condition in R2_CONDITION_IDS for unit in R2_TEST_UNIT_IDS
    )
    rows = entry.old_r3._setup_failure_rows(
        split="test", units=units, error=RuntimeError("setup"),
    )
    payload = entry._payload(
        exact="0" * 40, r1a_root=Path("/r1a"), repair_root=Path("/repair"),
        r2_root=Path("/r2"), old_r3_root=Path("/old"),
        test_cycle_rows=rows, operational_error=RuntimeError("setup"),
    )
    accounting = payload["test_probe_accounting"]
    assert len(rows) == accounting["fixed_denominator"] == 40
    assert accounting["eligible_units"] == 40
    assert accounting["attempted_branches"] == 0
    assert accounting["eligible_units_without_exact_8_attempts"] == 40


def test_runner_source_keeps_runtime_predicate_blind_and_result_create_only():
    source = Path(entry.__file__).read_text(encoding="utf-8")
    assert 'open("x", encoding="utf-8")' in source
    assert "allow_probe=True" in source
    assert "blind_weighted_scores" not in source
    assert "final_recovery_remains_raw_h0_once" in source
    assert "orientation_diagnostic_affects_acceptance\": False" in source
    assert "--old-r3-artifact-root" in source
