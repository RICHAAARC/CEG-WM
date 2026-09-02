"""Run the frozen Geometry-V7 R3 advanced predicted-H gate and test40 probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import run_geometry_v7_r3 as old_r3
from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r3 import CycleFeatureRow, R3Unit, route_from_s0
from cegwm.geometry_v7.r3_advanced import (
    R3_ADVANCED_CLAIM_CEILING,
    R3_ADVANCED_OPERATIONAL_FAILURE,
    R3_ADVANCED_R4_CANDIDATE,
    R3_ADVANCED_TEST40_RECORDED,
    R3_ADVANCED_TRANSLATION_PARTIAL,
    AdvancedRow,
    advanced_runtime_decision,
    evaluate_advanced,
    orientation_diagnostic,
)
from cegwm.geometry_v7.syncseal import SyncSealTorchScript, download_official_syncseal_torchscript


OLD_R3_PRODUCER_EXACT = "896571fd17fbc161bbb617f74677328a012ce43a"
OLD_R3_SCHEMA = "geometry_v7_r3_exploratory_result_v1"
OLD_R3_STATUS = "R3_METHOD_NOT_IMPROVED"
RESULT_SCHEMA = "geometry_v7_r3_advanced_result_v1"
STAGE_LABEL = "R3 advanced predicted-H regime and orientation diagnostic"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _expected(roster: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in roster)


def _validate_old_r3(root: Path) -> tuple[Mapping[str, Any], ...]:
    result = old_r3._read_json(root, "old R3")
    if (
        result.get("schema") != OLD_R3_SCHEMA
        or result.get("exact") != OLD_R3_PRODUCER_EXACT
        or result.get("status") != OLD_R3_STATUS
    ):
        raise ValueError("old R3 artifact identity/status differs")
    rows = result.get("feature_rows")
    if not isinstance(rows, list) or len(rows) != 80:
        raise ValueError("old R3 fixed 80 feature rows differ")
    expected = _expected(R2_DEV_UNIT_IDS) + _expected(R2_TEST_UNIT_IDS)
    if tuple((row.get("condition_id"), row.get("unit_id")) for row in rows if isinstance(row, Mapping)) != expected:
        raise ValueError("old R3 feature identity/order differs")
    selection = result.get("development_threshold_selection")
    grid = selection.get("grid_metrics") if isinstance(selection, Mapping) else None
    eight = tuple(
        item for item in grid
        if isinstance(item, Mapping) and item.get("threshold_px") == 8.0
    ) if isinstance(grid, list) else ()
    if (
        not isinstance(selection, Mapping)
        or selection.get("selected_threshold_px") is not None
        or selection.get("selected_metrics") is not None
        or len(eight) != 1
        or not isinstance(eight[0], Mapping)
    ):
        raise ValueError("old R3 frozen 8px development baseline differs")
    metrics = eight[0]
    if (
        metrics.get("threshold_px") != 8.0
        or metrics.get("fixed_denominator") != 40
        or metrics.get("accepted_count") != 8
        or metrics.get("safe_rescue_count") != 8
        or metrics.get("unsafe_accept_count") != 0
        or metrics.get("selected_negative_control_fp_count") != 0
        or metrics.get("covered_attack_count") != 5
    ):
        raise ValueError("old R3 frozen 8px development baseline differs")
    return tuple(rows)


def _ordered_split(units: Sequence[R3Unit], split: str) -> tuple[R3Unit, ...]:
    roster = R2_DEV_UNIT_IDS if split == "dev" else R2_TEST_UNIT_IDS
    return tuple(unit for unit in units if unit.unit_id in roster)


def _geometry_h0(record: Mapping[str, Any]) -> tuple[object, bool, str | None]:
    geometry = record.get("geometry")
    if not isinstance(geometry, Mapping):
        return None, False, "stored geometry missing"
    error = geometry.get("error")
    return (
        geometry.get("homography_observed_to_canonical"),
        geometry.get("legal") is True,
        None if error is None else str(error),
    )


def _advanced_rows(
    *, split: str, cycle_rows: Sequence[Mapping[str, Any] | CycleFeatureRow],
    units: Sequence[R3Unit], r1a: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[AdvancedRow, ...]:
    result = []
    for cycle, unit in zip(cycle_rows, units, strict=True):
        if isinstance(cycle, Mapping):
            cycle_score = cycle.get("cycle_score_px")
        else:
            cycle_score = cycle.cycle_score_px
        h0, legal, error = _geometry_h0(r1a[(unit.condition_id, unit.unit_id)])
        result.append(AdvancedRow(
            split, unit.condition_id, unit.unit_id, route_from_s0(unit.s0),
            unit.r2_selector_accepted,
            float(cycle_score) if isinstance(cycle_score, (int, float)) and not isinstance(cycle_score, bool)
            and math.isfinite(float(cycle_score)) else None,
            h0, legal, error,
        ))
    return tuple(result)


def _decision_rows(rows: Sequence[AdvancedRow], units: Sequence[R3Unit]):
    output = []
    for row, unit in zip(rows, units, strict=True):
        decision = advanced_runtime_decision(
            boundary=row.route == "BOUNDARY",
            r2_selector_accepted=row.r2_selector_accepted,
            old_cycle_score_px=row.old_cycle_score_px,
            homography_observed_to_canonical=row.homography_observed_to_canonical,
            geometry_legal=row.geometry_legal,
            geometry_error=row.geometry_error,
        )
        output.append({
            "split": row.split, "condition_id": row.condition_id, "unit_id": row.unit_id,
            "route": row.route, "r2_selector_accepted": row.r2_selector_accepted,
            "old_cycle_score_px": row.old_cycle_score_px, "decision": _jsonable(decision),
            "evaluation_only": {
                "safe": unit.safe, "safe_rescue": unit.safe_rescue,
                "observed_negative_false_positive": unit.observed_negative_false_positive,
            },
        })
    return output


def _negative_orientation_diagnostic(
    dev_cycle_rows: Sequence[Mapping[str, Any]], dev_units: Sequence[R3Unit],
) -> Mapping[str, Any]:
    rows = []
    for cycle, unit in zip(dev_cycle_rows, dev_units, strict=True):
        eligible = route_from_s0(unit.s0) == "BOUNDARY" and unit.r2_selector_accepted
        branches = cycle.get("branches")
        diagnostic = orientation_diagnostic(branches) if eligible and isinstance(branches, list) else None
        rows.append({
            "condition_id": unit.condition_id, "unit_id": unit.unit_id,
            "eligible": eligible, "diagnostic": _jsonable(diagnostic),
        })
    return {
        "status": "RECORD_ONLY_NOT_SELECTED",
        "reason": "no frozen orientation acceptance predicate; cannot affect runtime decision",
        "predicate_input": False,
        "rows": rows,
    }


def _assert_frozen_dev(metrics) -> None:
    expected = (
        metrics.fixed_denominator, metrics.baseline_accepted_count,
        metrics.baseline_safe_rescue_count, metrics.baseline_unsafe_accept_count,
        metrics.baseline_negative_control_fp_count, metrics.baseline_covered_attack_count,
        metrics.accepted_count, metrics.safe_rescue_count, metrics.unsafe_accept_count,
        metrics.selected_negative_control_fp_count, metrics.covered_attack_count,
    )
    if expected != (40, 8, 8, 0, 0, 5, 16, 16, 0, 0, 7):
        raise ValueError("frozen R3 advanced development result differs")


def _r4_readiness(metrics, test_rows: Sequence[CycleFeatureRow]) -> Mapping[str, Any]:
    eligible = sum(row.route == "BOUNDARY" and row.r2_selector_accepted for row in test_rows)
    attempts = sum(branch.probed for row in test_rows for branch in row.branches)
    complete = eligible > 0 and attempts == eligible * 8 and all(
        not (row.route == "BOUNDARY" and row.r2_selector_accepted)
        or sum(branch.probed for branch in row.branches) == 8
        for row in test_rows
    )
    per_attack = {
        item.condition_id: item for item in (() if metrics is None else metrics.per_attack)
    }
    translation_ids = tuple(item for item in R2_CONDITION_IDS if "translation" in item)
    rotation_ids = ("core_rotation_neg15", "core_rotation_pos15")
    translation_complete = all(
        condition in per_attack
        and per_attack[condition].accepted_count >= 1
        and per_attack[condition].safe_rescue_count >= 1
        for condition in translation_ids
    )
    rotation_complete = all(
        condition in per_attack
        and per_attack[condition].accepted_count >= 1
        and per_attack[condition].safe_rescue_count >= 1
        for condition in rotation_ids
    )
    zero_harm = bool(
        metrics is not None and metrics.accepted_count > 0
        and metrics.safe_rescue_count > 0
        and metrics.unsafe_accept_count == 0
        and metrics.selected_negative_control_fp_count == 0
    )
    candidate = complete and zero_harm and translation_complete and rotation_complete
    return {
        "candidate": candidate,
        "eligible_probe_attempts_complete": complete,
        "nonempty_safe_rescue_zero_unsafe_zero_fp": zero_harm,
        "all_four_translation_directions_have_safe_rescue": translation_complete,
        "both_rotation_directions_have_safe_rescue": rotation_complete,
        "requires_agent5_adjudication": True,
    }


def _payload(
    *, exact: str, r1a_root: Path, repair_root: Path, r2_root: Path, old_r3_root: Path,
    dev_metrics=None, test_metrics=None, dev_decisions=(), test_decisions=(),
    orientation=None, test_cycle_rows=(), operational_error: BaseException | None = None,
) -> Mapping[str, Any]:
    test_rows = tuple(test_cycle_rows)
    eligible = sum(row.route == "BOUNDARY" and row.r2_selector_accepted for row in test_rows)
    attempts = sum(branch.probed for row in test_rows for branch in row.branches)
    failures = sum(
        row.route == "BOUNDARY" and row.r2_selector_accepted
        and sum(branch.probed for branch in row.branches) != 8
        for row in test_rows
    )
    readiness = _r4_readiness(test_metrics, test_rows)
    status = R3_ADVANCED_OPERATIONAL_FAILURE if operational_error is not None else R3_ADVANCED_TEST40_RECORDED
    return {
        "schema": RESULT_SCHEMA, "stage": STAGE_LABEL, "exact": exact,
        "status": status, "scientific_status": "not_adjudicated",
        "claim_ceiling": R3_ADVANCED_CLAIM_CEILING,
        "data_used_for_development": True,
        "inputs": {
            "r1a": {"producer_exact": old_r3.R1A_PRODUCER_EXACT, "artifact_root": str(r1a_root)},
            "r1b_repair": {"producer_exact": old_r3.REPAIR_PRODUCER_EXACT, "artifact_root": str(repair_root)},
            "r2": {"producer_exact": old_r3.R2_PRODUCER_EXACT, "artifact_root": str(r2_root),
                   "recorded_status_unchanged": "R2_SELECTIVE_RISK_FAILED"},
            "old_r3": {"producer_exact": OLD_R3_PRODUCER_EXACT, "artifact_root": str(old_r3_root),
                       "recorded_status_unchanged": OLD_R3_STATUS},
        },
        "frozen_runtime_predicate": {
            "eligible": "BOUNDARY and frozen_R2_accepted",
            "accept": "old_cycle_score_px <= 8 OR pure_rotation_gate",
            "pure_rotation_gate": {
                "angle_degrees": "10 <= abs(atan2(h10-h01,h00+h11)*180/pi) <= 20",
                "scale": "0.95 <= sqrt(abs(det(H[:2,:2]))) <= 1.05",
                "translation": "hypot(h02,h12) <= 0.02",
                "perspective": "hypot(h20,h21) <= 0.01",
                "normalization": "H / h22",
            },
            "forbidden_inputs": ["condition_id", "truth", "attack_label", "post_outcome"],
        },
        "development_metrics": _jsonable(dev_metrics),
        "development_decisions": list(dev_decisions),
        "orientation_d4_negative_diagnostic": _jsonable(orientation),
        "existing_test40_metrics": _jsonable(test_metrics),
        "existing_test40_decisions": list(test_decisions),
        "existing_test40_cycle_rows": [_jsonable(row) for row in test_rows],
        "test_probe_accounting": {
            "fixed_denominator": 40, "eligible_units": eligible,
            "attempted_branches": attempts, "expected_attempted_branches": eligible * 8,
            "eligible_units_without_exact_8_attempts": failures,
        },
        "r4_readiness": readiness,
        "r4_candidate": readiness["candidate"] if operational_error is None else False,
        "fallback_scope": None if operational_error is not None or readiness["candidate"] else
        "full_family_translation_partial_scope",
        "operational_error": None if operational_error is None else (
            f"{type(operational_error).__name__}: {operational_error}"
        ),
        "route": {
            "geometry_positive_vote": False, "condition_or_truth_in_runtime_predicate": False,
            "orientation_diagnostic_affects_acceptance": False,
            "old_h0_updated_replaced_or_averaged": False,
            "final_recovery_remains_raw_h0_once": True,
            "content_detector_key_preprocess_tau_unchanged": True,
            "no_retry_fallback_or_success_subset": True,
        },
    }


def _write_result(root: Path, payload: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    with (root / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    result_root = Path(args.result_dir).resolve()
    checkpoint = Path(args.syncseal_checkpoint).resolve()
    if result_root.exists():
        raise FileExistsError("R3 advanced result directory must be create-only")
    if checkpoint.exists():
        raise FileExistsError("R3 advanced checkpoint must be create-only")
    exact = old_r3._git_exact(repo_root, args.expected_exact)
    roots = tuple(Path(value).resolve() for value in (
        args.r1a_artifact_root, args.r1b_repair_artifact_root,
        args.r2_artifact_root, args.old_r3_artifact_root,
    ))
    r1a_root, repair_root, r2_root, old_r3_root = roots
    try:
        roster, r2_features, r2_outcomes = old_r3._validate_r2(r2_root)
        units = old_r3._bind_r2_acceptance(
            old_r3._validate_repair(repair_root, roster, r2_outcomes), r2_features,
        )
        r1a = old_r3._validate_r1a(r1a_root, roster)
        old_rows = _validate_old_r3(old_r3_root)
        dev_units = _ordered_split(units, "dev")
        test_units = _ordered_split(units, "test")
        dev_cycle = old_rows[:40]
        dev_rows = _advanced_rows(split="dev", cycle_rows=dev_cycle, units=dev_units, r1a=r1a)
        dev_metrics = evaluate_advanced(dev_rows, dev_units, split="dev")
        _assert_frozen_dev(dev_metrics)
        dev_decisions = _decision_rows(dev_rows, dev_units)
        orientation = _negative_orientation_diagnostic(dev_cycle, dev_units)
    except Exception as error:
        payload = _payload(
            exact=exact, r1a_root=r1a_root, repair_root=repair_root,
            r2_root=r2_root, old_r3_root=old_r3_root, operational_error=error,
        )
        _write_result(result_root, payload)
        return payload
    if not old_r3.torch.cuda.is_available():
        error = RuntimeError("cuda_required_for_existing_test40_d4_probe")
        test_cycle = old_r3._setup_failure_rows(split="test", units=test_units, error=error)
        payload = _payload(
            exact=exact, r1a_root=r1a_root, repair_root=repair_root, r2_root=r2_root,
            old_r3_root=old_r3_root, dev_metrics=dev_metrics,
            dev_decisions=dev_decisions, orientation=orientation,
            test_cycle_rows=test_cycle, operational_error=error,
        )
        _write_result(result_root, payload)
        return payload
    try:
        loaded = download_official_syncseal_torchscript(checkpoint)
        detector = SyncSealTorchScript.from_file(loaded, device="cuda").detect_geometry
    except Exception as error:
        test_cycle = old_r3._setup_failure_rows(split="test", units=test_units, error=error)
        payload = _payload(
            exact=exact, r1a_root=r1a_root, repair_root=repair_root, r2_root=r2_root,
            old_r3_root=old_r3_root, dev_metrics=dev_metrics,
            dev_decisions=dev_decisions, orientation=orientation,
            test_cycle_rows=test_cycle, operational_error=error,
        )
        _write_result(result_root, payload)
        return payload
    test_cycle = old_r3._rows_for_split(
        split="test", units=test_units, r1a=r1a, r1a_root=r1a_root,
        detector=detector, allow_probe=True,
    )
    test_rows = _advanced_rows(split="test", cycle_rows=test_cycle, units=test_units, r1a=r1a)
    test_metrics = evaluate_advanced(test_rows, test_units, split="test")
    payload = _payload(
        exact=exact, r1a_root=r1a_root, repair_root=repair_root, r2_root=r2_root,
        old_r3_root=old_r3_root, dev_metrics=dev_metrics, test_metrics=test_metrics,
        dev_decisions=dev_decisions, test_decisions=_decision_rows(test_rows, test_units),
        orientation=orientation, test_cycle_rows=test_cycle,
    )
    _write_result(result_root, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--r1a-artifact-root", required=True)
    parser.add_argument("--r1b-repair-artifact-root", required=True)
    parser.add_argument("--r2-artifact-root", required=True)
    parser.add_argument("--old-r3-artifact-root", required=True)
    parser.add_argument("--syncseal-checkpoint", required=True)
    parser.add_argument("--result-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = _run(args)
    except Exception as error:
        print(json.dumps({"status": "RUNNER_STOPPED_BEFORE_PACKAGE",
                          "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
        return 2
    print(json.dumps({"status": payload["status"], "result_dir": args.result_dir}, sort_keys=True))
    return 0 if payload["status"] == R3_ADVANCED_TEST40_RECORDED else 2


if __name__ == "__main__":
    raise SystemExit(main())
