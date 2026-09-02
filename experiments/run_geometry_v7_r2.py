"""Run Geometry-V7 R2 as a pure CPU postprocess over accepted JSON artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from cegwm.geometry_v7.r1a import R1A_CORE_CONDITIONS
from cegwm.geometry_v7.r2 import (
    R2_CLAIM_CEILING,
    R2_DEV_NO_FEASIBLE,
    R2_OPERATIONAL_FAILURE,
    R2_PASSED_ALL,
    R2_PASSED_PARTIAL,
    FeatureRow,
    OutcomeRow,
    evaluate_frozen_candidate,
    feature_row_from_geometry,
    outcome_row_from_repair,
    select_candidate,
)


R1A_PRODUCER_EXACT = "ac590330e91aacf4b3283df1e94572a0e4f983a0"
R1A_SCHEMA = "geometry_v7_r1a_result_v1"
R1A_REQUIRED_STATUS = "R1A_BLOCKING_METHOD_CANARY_PASSED"
REPAIR_PRODUCER_EXACT = "3b9819d80b07704a4caab8b7aaa581cf9eb8a3c5"
REPAIR_SCHEMA = "geometry_v7_r1b_repair_result_v1"
REPAIR_REQUIRED_STATUS = "R1B_REPAIR_REAL_H_NOT_END_TO_END_READY"
REPAIR_REAL_STATUS = "PARTIAL_CORE_PASSED"
REPAIR_FINE_STATUS = "PARTIAL_CORE_NONZERO_PREFIX"
RESULT_SCHEMA = "geometry_v7_r2_selective_result_v1"
STAGE_LABEL = "R2 selective reliability"


def _read_json(root: Path, label: str) -> Mapping[str, Any]:
    path = root.resolve() / "result.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} result.json must be readable JSON") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} result.json must contain an object")
    return result


def _git_exact(repo_root: Path, expected: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("expected exact must be lowercase 40-hex")
    exact = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                           text=True, capture_output=True).stdout.strip()
    if exact != expected:
        raise RuntimeError("checkout differs from approved exact")
    if subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True,
                      text=True, capture_output=True).stdout:
        raise RuntimeError("execution checkout must be clean")
    return exact


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _split(unit_id: str) -> str:
    if any(unit_id.endswith(f"eval-{index:04d}") for index in range(1, 5)):
        return "dev"
    if any(unit_id.endswith(f"eval-{index:04d}") for index in range(5, 9)):
        return "test"
    raise ValueError("unit is outside frozen R2 split")


def _validate_r1a(root: Path) -> tuple[tuple[str, ...], dict[tuple[str, str], FeatureRow]]:
    result = _read_json(root, "R1A")
    if (result.get("schema") != R1A_SCHEMA or result.get("exact") != R1A_PRODUCER_EXACT
            or result.get("status") != R1A_REQUIRED_STATUS):
        raise ValueError("R1A identity/status differs")
    r0_input = result.get("r0_input")
    ordered = r0_input.get("ordered_evaluation_cg_inputs") if isinstance(r0_input, Mapping) else None
    if not isinstance(ordered, list) or len(ordered) != 8:
        raise ValueError("R1A fixed evaluation roster differs")
    roster = tuple(item.get("unit_id") for item in ordered if isinstance(item, Mapping))
    if len(roster) != 8 or len(set(roster)) != 8:
        raise ValueError("R1A fixed evaluation roster differs")
    if tuple(_split(unit) for unit in roster) != ("dev",)*4 + ("test",)*4:
        raise ValueError("R1A frozen dev/test roster order differs")
    raw = result.get("raw_records")
    if not isinstance(raw, list) or len(raw) != 104:
        raise ValueError("R1A fixed record count differs")
    condition_ids = tuple(spec.condition_id for spec in R1A_CORE_CONDITIONS)
    expected = tuple((condition, unit) for condition in condition_ids for unit in roster)
    core_items = tuple(item for item in raw if isinstance(item, Mapping)
                       and item.get("condition_id") in condition_ids)
    if tuple((item.get("condition_id"), item.get("unit_id")) for item in core_items) != expected:
        raise ValueError("R1A core identity/order differs")
    indexed = {(item.get("condition_id"), item.get("unit_id")): item
               for item in raw if isinstance(item, Mapping)}
    features = {}
    for spec in R1A_CORE_CONDITIONS:
        for unit in roster:
            item = indexed.get((spec.condition_id, unit))
            if not isinstance(item, Mapping):
                raise ValueError("R1A core identity is missing")
            geometry = item.get("geometry")
            if not isinstance(geometry, Mapping):
                geometry = {}
            features[(spec.condition_id, unit)] = feature_row_from_geometry(
                split=_split(unit), condition_id=spec.condition_id, unit_id=unit,
                geometry=geometry,
            )
    return roster, features


def _validate_repair(root: Path, roster: Sequence[str]) -> tuple[
    dict[tuple[str, str], OutcomeRow], Mapping[str, Any]
]:
    result = _read_json(root, "R1B repair")
    if (result.get("schema") != REPAIR_SCHEMA or result.get("exact") != REPAIR_PRODUCER_EXACT
            or result.get("status") != REPAIR_REQUIRED_STATUS
            or result.get("real_h_status") != REPAIR_REAL_STATUS
            or result.get("fine_grid_status") != REPAIR_FINE_STATUS
            or result.get("r2_candidate") is not False
            or result.get("real_h_passed_condition_count") != 7
            or result.get("fine_nonzero_prefix_condition_count") != 9):
        raise ValueError("R1B repair accepted top facts differ")
    inputs = result.get("inputs")
    if not isinstance(inputs, Mapping) or tuple(inputs.get("ordered_roster", ())) != tuple(roster):
        raise ValueError("R1B repair ordered roster differs")
    memberships = result.get("frozen_old_membership_records")
    real = result.get("real_h_records")
    evaluations = result.get("condition_evaluations")
    if not isinstance(memberships, list) or len(memberships) != 80:
        raise ValueError("R1B repair membership count differs")
    if not isinstance(real, list) or len(real) != 80:
        raise ValueError("R1B repair real-H count differs")
    if not isinstance(evaluations, list) or len(evaluations) != 10:
        raise ValueError("R1B repair condition count differs")
    condition_ids = tuple(spec.condition_id for spec in R1A_CORE_CONDITIONS)
    if tuple(item.get("condition_id") for item in evaluations if isinstance(item, Mapping)) != condition_ids:
        raise ValueError("R1B repair condition order differs")
    if any(tuple(item.get("roster", ())) != tuple(roster) for item in evaluations):
        raise ValueError("R1B repair condition roster differs")
    actual_real_passed = sum(item.get("real_h_passed") is True for item in evaluations)
    actual_fine_passed = sum(
        isinstance(item.get("accepted_max_pixels"), (int, float))
        and not isinstance(item.get("accepted_max_pixels"), bool)
        and item.get("accepted_max_pixels") >= 1
        for item in evaluations
    )
    if actual_real_passed != 7 or actual_fine_passed != 9:
        raise ValueError("R1B repair per-condition facts differ from accepted top counts")
    expected = tuple((condition, unit) for condition in condition_ids for unit in roster)
    membership_map = {}
    for item in memberships:
        if not isinstance(item, Mapping):
            raise ValueError("R1B repair membership malformed")
        identity = (item.get("condition_id"), item.get("unit_id"))
        membership_map[identity] = item.get("membership_from_old_r1b")
    real_map = {}
    for item in real:
        if not isinstance(item, Mapping) or item.get("point_kind") != "real_h" or item.get("radius_pixels") is not None:
            raise ValueError("R1B repair real-H record malformed")
        real_map[(item.get("condition_id"), item.get("unit_id"))] = item
    if tuple(membership_map) != expected or tuple(real_map) != expected:
        raise ValueError("R1B repair fixed record identity/order differs")
    outcomes = {}
    for condition, unit in expected:
        membership = membership_map[(condition, unit)]
        if membership not in ("N_recovery_negative", "B_boundary", "D_damage_only"):
            raise ValueError("R1B repair frozen membership differs")
        outcomes[(condition, unit)] = outcome_row_from_repair(
            split=_split(unit), condition_id=condition, unit_id=unit,
            membership=membership, record=real_map[(condition, unit)],
        )
    facts = {
        "status": result["status"], "real_h_status": result["real_h_status"],
        "fine_grid_status": result["fine_grid_status"], "r2_candidate": result["r2_candidate"],
        "real_h_passed_condition_count": result["real_h_passed_condition_count"],
        "fine_nonzero_prefix_condition_count": result["fine_nonzero_prefix_condition_count"],
        "per_condition_actual": evaluations,
    }
    return outcomes, facts


def _ordered_split(rows: Mapping[tuple[str, str], Any], roster: Sequence[str], split: str):
    return tuple(rows[(spec.condition_id, unit)] for spec in R1A_CORE_CONDITIONS
                 for unit in roster if _split(unit) == split)


def _payload(*, exact: str, r1a_root: Path, repair_root: Path,
             roster: Sequence[str] = (), features: Mapping = {}, outcomes: Mapping = {},
             prior: Mapping[str, Any] | None = None, input_error: BaseException | None = None):
    if input_error is not None:
        return {
            "schema": RESULT_SCHEMA, "stage": STAGE_LABEL, "status": R2_OPERATIONAL_FAILURE,
            "method_verdict": None, "selection": None, "formal_test": None,
            "input_error": f"{type(input_error).__name__}: {input_error}",
            "exact": exact, "claim_ceiling": R2_CLAIM_CEILING,
            "scientific_status": "not_adjudicated",
            "R1B_FULL_PASS": False, "R1B_SELECTIVE_CANDIDATE": True,
            "R2_SELECTIVE_RELIABILITY_AUTHORIZED": True, "prior_aggregate_visibility": True,
        }
    dev_features = _ordered_split(features, roster, "dev")
    test_features = _ordered_split(features, roster, "test")
    dev_outcomes = _ordered_split(outcomes, roster, "dev")
    test_outcomes = _ordered_split(outcomes, roster, "test")
    selection = select_candidate(dev_features, dev_outcomes)
    formal = None
    status = selection.status
    if selection.selected is not None:
        status, metrics = evaluate_frozen_candidate(selection.selected, test_features, test_outcomes)
        formal = {"candidate_id": selection.selected.candidate_id, "metrics": _jsonable(metrics)}
    return {
        "schema": RESULT_SCHEMA, "stage": STAGE_LABEL, "status": status,
        "method_verdict": status, "exact": exact,
        "R1B_FULL_PASS": False, "R1B_SELECTIVE_CANDIDATE": True,
        "R2_SELECTIVE_RELIABILITY_AUTHORIZED": True, "prior_aggregate_visibility": True,
        "upstream_r2_candidate": False,
        "upstream_r2_candidate_recorded_unchanged": True,
        "inputs": {"r1a": {"producer_exact": R1A_PRODUCER_EXACT, "artifact_root": str(r1a_root)},
                   "r1b_repair": {"producer_exact": REPAIR_PRODUCER_EXACT, "artifact_root": str(repair_root)}},
        "fixed": {"conditions": 10, "units": 8, "total": 80, "dev": 40, "test": 40,
                  "features": ["raw_logit", "kappa_f", "coverage", "area_ratio"],
                  "quantiles_type7": [0.20, 0.40, 0.60, 0.80]},
        "ordered_roster": list(roster), "prior_aggregate": prior,
        "feature_rows": [_jsonable(features[(spec.condition_id, unit)]) for spec in R1A_CORE_CONDITIONS for unit in roster],
        "outcome_rows": [_jsonable(outcomes[(spec.condition_id, unit)]) for spec in R1A_CORE_CONDITIONS for unit in roster],
        "selection": {"status": selection.status,
                      "selected": _jsonable(selection.selected),
                      "selected_metrics": _jsonable(selection.selected_metrics),
                      "complete_dev_candidate_table": [
                          {"candidate": _jsonable(candidate), "metrics": _jsonable(metrics)}
                          for candidate, metrics in zip(
                              selection.candidates, selection.candidate_table, strict=True
                          )
                      ]},
        "formal_test": formal,
        "claim_ceiling": R2_CLAIM_CEILING, "scientific_status": "not_adjudicated",
        "raw_logit_semantics": "uncalibrated raw SyncSeal logit; not calibrated confidence",
        "route": {"pure_cpu_json_postprocess": True, "model_invoked": False, "content_positive_vote_from_geometry": False,
                  "test_used_for_selection": False, "no_retry_fallback_or_subset": True},
    }


def _write_result(root: Path, payload: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    with (root / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    result_root = Path(args.result_dir).resolve()
    if result_root.exists():
        raise FileExistsError("create-only R2 result directory already exists")
    exact = _git_exact(repo_root, args.expected_exact)
    r1a_root = Path(args.r1a_artifact_root).resolve()
    repair_root = Path(args.r1b_repair_artifact_root).resolve()
    try:
        roster, features = _validate_r1a(r1a_root)
        outcomes, prior = _validate_repair(repair_root, roster)
        payload = _payload(exact=exact, r1a_root=r1a_root, repair_root=repair_root,
                           roster=roster, features=features, outcomes=outcomes, prior=prior)
    except Exception as error:
        payload = _payload(exact=exact, r1a_root=r1a_root, repair_root=repair_root, input_error=error)
    _write_result(result_root, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--r1a-artifact-root", required=True)
    parser.add_argument("--r1b-repair-artifact-root", required=True)
    parser.add_argument("--result-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = _run(args)
    except Exception as error:
        print(json.dumps({"status": "RUNNER_STOPPED_BEFORE_PACKAGE", "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
        return 2
    print(json.dumps({"status": payload["status"], "result_dir": args.result_dir}, sort_keys=True))
    return 0 if payload["status"] in (R2_PASSED_ALL, R2_PASSED_PARTIAL) else 2


if __name__ == "__main__":
    raise SystemExit(main())
