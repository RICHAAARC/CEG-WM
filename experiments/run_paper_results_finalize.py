"""Read-only Drive aggregation for the five-method formal result package."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from cegwm.formal_experiment import (
    CLEAN_TEST_NEGATIVES,
    EVALUATION_PAIRS,
    FORMAL_CONDITIONS,
    empty_binary_summary,
    load_formal_config,
)
from experiments.run_paper_main_worker import CONFIG_PATH, METHOD_ID, REPO_ROOT


BASELINE_JOBS = {
    "t2smark": "paper-baseline-t2smark-v1",
    "tree_ring": "paper-baseline-treering-v1",
    "gaussian_shading": "paper-baseline-gaussian-shading-v1",
    "shallow_diffuse": "paper-baseline-shallow-diffuse-v1",
}
MAIN_JOB_ID = "paper-main-v1"
RECONSTRUCTION_JOB_ID = "paper-main-reconstruction-v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"result must be an object: {path}")
    return value


def _verify_checkout(expected_exact: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != expected_exact or dirty:
        raise RuntimeError("finalizer requires the clean frozen producer exact")


def _validate_summary(summary: Mapping[str, Any], *, planned: int, role: str) -> None:
    if role not in {"negative", "positive"}:
        raise ValueError("summary role differs")
    scored = summary.get("n_scored")
    failed = summary.get("n_failed")
    missing = summary.get("n_missing")
    if (
        summary.get("n_planned") != planned
        or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (scored, failed, missing))
        or scored + failed + missing != planned
        or summary.get("status") != ("COMPLETE" if failed + missing == 0 else "INCOMPLETE_OPERATIONAL")
    ):
        raise ValueError("summary denominator accounting differs")
    rate_key = "scored_only_tpr" if role == "positive" else "scored_only_fpr"
    ci_key = "tpr_ci95" if role == "positive" else "fpr_ci95"
    lower_key = "planned_tpr_lower" if role == "positive" else "planned_fpr_lower"
    upper_key = "planned_tpr_upper" if role == "positive" else "planned_fpr_upper"
    if any(key not in summary for key in (rate_key, ci_key, lower_key, upper_key)):
        raise ValueError("summary statistical fields differ")


def _validate_method_result(result: Mapping[str, Any], *, method_id: str, exact: str) -> None:
    if result.get("schema_version") != "cegwm_formal_method_result_v1":
        raise ValueError(f"{method_id} result schema differs")
    if result.get("method_id") != method_id or result.get("producer_exact") != exact:
        raise ValueError(f"{method_id} result identity differs")
    if result.get("status") not in {"COMPLETE", "INCOMPLETE_OPERATIONAL"}:
        raise ValueError(f"{method_id} status differs")
    threshold = result.get("threshold")
    if threshold is None:
        if result.get("status") != "INCOMPLETE_OPERATIONAL" or result.get("threshold_status") != "INCOMPLETE_THRESHOLD":
            raise ValueError(f"{method_id} threshold is missing without terminal incomplete status")
    elif not isinstance(threshold, dict) or any((
        threshold.get("alpha") != 0.001,
        threshold.get("calibration_denominator") != 2000,
        threshold.get("estimator") != "nearest_rank_empirical_quantile",
        threshold.get("rank_one_based") != 1998,
        threshold.get("decision_rule") != "positive_iff_normalized_score_strictly_greater_than_tau",
        threshold.get("equality_decision") != "negative",
    )):
        raise ValueError(f"{method_id} paper threshold contract differs")
    clean = result.get("clean_negative_test")
    if not isinstance(clean, dict) or clean.get("n_planned") != 3000:
        raise ValueError(f"{method_id} clean-negative denominator differs")
    _validate_summary(clean, planned=CLEAN_TEST_NEGATIVES, role="negative")
    evaluation = result.get("evaluation")
    expected = {f"{condition}:{role}" for condition in FORMAL_CONDITIONS for role in ("negative", "positive")}
    if not isinstance(evaluation, dict) or set(evaluation) != expected:
        raise ValueError(f"{method_id} evaluation matrix differs")
    for key, value in evaluation.items():
        if not isinstance(value, dict):
            raise ValueError(f"{method_id} evaluation summary differs")
        _validate_summary(value, planned=EVALUATION_PAIRS, role=key.rsplit(":", 1)[1])
    if result.get("status") == "COMPLETE" and (
        threshold is None
        or clean.get("status") != "COMPLETE"
        or any(value.get("status") != "COMPLETE" for value in evaluation.values())
    ):
        raise ValueError(f"{method_id} complete status conflicts with incomplete evidence")


def _missing_method_result(method_id: str, exact: str) -> dict[str, Any]:
    return {
        "schema_version": "cegwm_formal_method_result_v1",
        "method_id": method_id,
        "producer_exact": exact,
        "threshold": None,
        "threshold_status": "INCOMPLETE_THRESHOLD",
        "clean_negative_test": empty_binary_summary(
            truth_positive=False, planned=CLEAN_TEST_NEGATIVES
        ),
        "evaluation": {
            f"{condition}:{role}": empty_binary_summary(
                truth_positive=role == "positive", planned=EVALUATION_PAIRS
            )
            for condition in FORMAL_CONDITIONS
            for role in ("negative", "positive")
        },
        "status": "INCOMPLETE_OPERATIONAL",
        "reason": "method result file is missing at finalization",
        "source_result_missing": True,
        "result_package_produced": True,
    }


def _read_method_or_missing(path: Path, *, method_id: str, exact: str) -> dict[str, Any]:
    return _read(path) if path.exists() else _missing_method_result(method_id, exact)


def _missing_reconstruction(expected_exact: str) -> dict[str, Any]:
    return {
        "schema_version": "cegwm_reconstruction_supplement_v1",
        "method_id": METHOD_ID,
        "producer_exact": expected_exact,
        "threshold": None,
        "summaries": {
            role: empty_binary_summary(truth_positive=role == "positive", planned=100)
            for role in ("negative", "positive")
        },
        "status": "INCOMPLETE_OPERATIONAL",
        "reason": "reconstruction supplement file is missing at finalization",
        "source_result_missing": True,
        "fpr_resolution": 0.01,
        "claim_ceiling": "supplementary_reconstruction_only_not_0.1_percent_attacked_fpr_validation",
        "result_package_produced": True,
    }


def _validate_reconstruction(result: Mapping[str, Any], expected_exact: str) -> None:
    summaries = result.get("summaries")
    if (
        result.get("schema_version") != "cegwm_reconstruction_supplement_v1"
        or result.get("method_id") != METHOD_ID
        or result.get("producer_exact") != expected_exact
        or result.get("fpr_resolution") != 0.01
        or result.get("status") not in {"COMPLETE", "INCOMPLETE_OPERATIONAL"}
        or not isinstance(summaries, dict)
        or set(summaries) != {"negative", "positive"}
        or any(not isinstance(value, dict) or value.get("n_planned") != 100 for value in summaries.values())
    ):
        raise ValueError("reconstruction supplement identity, status, denominator, or resolution differs")
    for role, summary in summaries.items():
        _validate_summary(summary, planned=100, role=role)
    if result.get("status") == "COMPLETE" and (
        not isinstance(result.get("threshold"), dict)
        or any(summary.get("status") != "COMPLETE" for summary in summaries.values())
    ):
        raise ValueError("reconstruction complete status conflicts with incomplete evidence")


def _write_csv_create_only(path: Path, methods: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "method_id", "condition", "role", "n_planned", "n_scored",
            "n_failed", "n_missing", "coverage", "conditional_rate",
            "ci95_lower", "ci95_upper", "planned_lower", "planned_upper", "status",
        ))
        writer.writeheader()
        for method_id, result in methods.items():
            for key, summary in result["evaluation"].items():
                condition, role = key.rsplit(":", 1)
                rate_key = "scored_only_tpr" if role == "positive" else "scored_only_fpr"
                ci_key = "tpr_ci95" if role == "positive" else "fpr_ci95"
                lower_key = "planned_tpr_lower" if role == "positive" else "planned_fpr_lower"
                upper_key = "planned_tpr_upper" if role == "positive" else "planned_fpr_upper"
                ci = summary[ci_key]
                writer.writerow({
                    "method_id": method_id, "condition": condition, "role": role,
                    "n_planned": summary["n_planned"], "n_scored": summary["n_scored"],
                    "n_failed": summary["n_failed"], "n_missing": summary["n_missing"],
                    "coverage": summary["coverage"], "conditional_rate": summary[rate_key],
                    "ci95_lower": ci[0], "ci95_upper": ci[1],
                    "planned_lower": summary[lower_key], "planned_upper": summary[upper_key],
                    "status": summary["status"],
                })
        stream.flush()
        os.fsync(stream.fileno())


def run_finalize(*, drive_root: Path, expected_exact: str, baseline_exact: str) -> int:
    _verify_checkout(expected_exact)
    load_formal_config(CONFIG_PATH)
    output_root = drive_root / "finalized" / "paper-formal-v1"
    final_path = output_root / "unified_result_package.json"
    if final_path.exists():
        final = _read(final_path)
        print(json.dumps({"status": final["status"], "terminal": True}, sort_keys=True))
        return 0

    methods: dict[str, dict[str, Any]] = {}
    main = _read_method_or_missing(
        drive_root / "main" / MAIN_JOB_ID / "method_final.json",
        method_id=METHOD_ID,
        exact=expected_exact,
    )
    _validate_method_result(main, method_id=METHOD_ID, exact=expected_exact)
    methods[METHOD_ID] = main
    for method_id, job_id in BASELINE_JOBS.items():
        result = _read_method_or_missing(
            drive_root / "baselines" / job_id / "method_final.json",
            method_id=method_id,
            exact=baseline_exact,
        )
        _validate_method_result(result, method_id=method_id, exact=baseline_exact)
        if method_id in methods:
            raise ValueError("duplicate method identity")
        methods[method_id] = result
    reconstruction_path = (
        drive_root / "reconstruction" / RECONSTRUCTION_JOB_ID / "reconstruction_final.json"
    )
    reconstruction = (
        _read(reconstruction_path)
        if reconstruction_path.exists()
        else _missing_reconstruction(expected_exact)
    )
    _validate_reconstruction(reconstruction, expected_exact)

    _write_csv_create_only(output_root / "unified_main_table_long.csv", methods)
    statuses = [result["status"] for result in methods.values()] + [reconstruction["status"]]
    payload = {
        "schema_version": "cegwm_unified_paper_result_package_v1",
        "paper_producer_exact": expected_exact,
        "baseline_producer_exact": baseline_exact,
        "method_order": [METHOD_ID, *BASELINE_JOBS],
        "methods": methods,
        "reconstruction_supplement": reconstruction,
        "status": "COMPLETE" if all(status == "COMPLETE" for status in statuses) else "INCOMPLETE_OPERATIONAL",
        "statistical_policy": "report_only_nonblocking",
        "result_package_produced": True,
        "final_published_last": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with final_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": payload["status"], "terminal": True}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--baseline-exact", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_formal_config(CONFIG_PATH)
    if args.validate_only:
        print(json.dumps({
            "status": "VALID", "model_execution": False,
            "method_order": [METHOD_ID, *BASELINE_JOBS],
            "main_job_id": MAIN_JOB_ID,
            "reconstruction_job_id": RECONSTRUCTION_JOB_ID,
        }, sort_keys=True))
        return 0
    return run_finalize(
        drive_root=Path(args.drive_root), expected_exact=args.expected_exact,
        baseline_exact=args.baseline_exact,
    )


if __name__ == "__main__":
    raise SystemExit(main())
