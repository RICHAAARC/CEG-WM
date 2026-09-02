"""Run Geometry-V7 R3 b_low freeze and exploratory D4 cycle diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from PIL import Image
import torch

from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    D4Transform,
    GeometryEstimate,
    GeometryStatus,
    compose_d4_observed_to_canonical,
    d4_homography,
)
from cegwm.geometry_v7.r1a import R1A_CORE_CONDITIONS, apply_homography, corner_rmse
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.geometry_v7.r2 import (
    R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS, outcome_row_from_repair,
)
from cegwm.geometry_v7.r3 import (
    R3_B_LOW,
    R3_B_LOW_HEX,
    R3_CLAIM_CEILING,
    R3_METHOD_IMPROVED,
    R3_METHOD_NOT_IMPROVED,
    R3_NOT_PROBED_INELIGIBLE,
    R3_OPERATIONAL_FAILURE,
    R3_THRESHOLD_GRID_PX,
    CycleFeatureRow,
    D4BranchRecord,
    R3Unit,
    cycle_feature_row,
    evaluate_selected_threshold,
    route_from_s0,
    select_threshold,
)
from cegwm.geometry_v7.syncseal import (
    SyncSealTorchScript,
    download_official_syncseal_torchscript,
)


R1A_PRODUCER_EXACT = "ac590330e91aacf4b3283df1e94572a0e4f983a0"
R1A_SCHEMA = "geometry_v7_r1a_result_v1"
R1A_REQUIRED_STATUS = "R1A_BLOCKING_METHOD_CANARY_PASSED"
REPAIR_PRODUCER_EXACT = "3b9819d80b07704a4caab8b7aaa581cf9eb8a3c5"
REPAIR_SCHEMA = "geometry_v7_r1b_repair_result_v1"
R2_PRODUCER_EXACT = "ffac9d4c1e575c27240d9423bbd30e0713aa2dcd"
R2_SCHEMA = "geometry_v7_r2_selective_result_v1"
R2_SELECTED_ID = "B|area_ratio|ge|0x1.f7fb98cfa00a1p-1"
R2_SELECTED_THRESHOLD = 0.9843414071510957
RESULT_SCHEMA = "geometry_v7_r3_exploratory_result_v1"
STAGE_LABEL = "R3 exploratory cycle diagnostic"
D4_ORDER = tuple(item.value for item in D4Transform)


def _read_json(root: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads((root.resolve() / "result.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} result.json must be readable JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} result.json must be an object")
    return payload


def _git_exact(repo_root: Path, expected: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("expected exact must be lowercase 40-hex")
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    if exact != expected:
        raise RuntimeError("checkout differs from approved exact")
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True,
        text=True, capture_output=True,
    ).stdout:
        raise RuntimeError("execution checkout must be clean")
    return exact


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


def _split(unit_id: str) -> str:
    if unit_id in R2_DEV_UNIT_IDS:
        return "dev"
    if unit_id in R2_TEST_UNIT_IDS:
        return "test"
    raise ValueError("unit outside frozen R3 split")


def _expected(roster: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in roster)


def _validate_r2(root: Path) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    result = _read_json(root, "R2")
    if (
        result.get("schema") != R2_SCHEMA or result.get("exact") != R2_PRODUCER_EXACT
        or result.get("status") != "R2_SELECTIVE_RISK_FAILED"
    ):
        raise ValueError("R2 artifact identity differs")
    selection = result.get("selection")
    selected = selection.get("selected") if isinstance(selection, Mapping) else None
    if (
        not isinstance(selected, Mapping)
        or selected.get("candidate_id") != R2_SELECTED_ID
        or len(tuple(selected.get("components", ()))) != 1
    ):
        raise ValueError("R2 frozen selector identity differs")
    component = selected["components"][0]
    if (
        not isinstance(component, Mapping)
        or component.get("feature") != "area_ratio"
        or component.get("direction") != "ge"
        or component.get("threshold") != R2_SELECTED_THRESHOLD
    ):
        raise ValueError("R2 frozen selector threshold differs")
    roster = tuple(result.get("ordered_roster", ()))
    if roster != R2_DEV_UNIT_IDS + R2_TEST_UNIT_IDS:
        raise ValueError("R2 frozen roster differs")
    features = result.get("feature_rows")
    outcomes = result.get("outcome_rows")
    if not isinstance(features, list) or not isinstance(outcomes, list) or len(features) != 80 or len(outcomes) != 80:
        raise ValueError("R2 fixed 80 rows differ")
    expected = _expected(roster)
    for rows, label in ((features, "feature"), (outcomes, "outcome")):
        if tuple((item.get("condition_id"), item.get("unit_id")) for item in rows if isinstance(item, Mapping)) != expected:
            raise ValueError(f"R2 {label} identity/order differs")
    return roster, tuple(features), tuple(outcomes)


def _validate_repair(
    root: Path, roster: Sequence[str], r2_outcomes: Sequence[Mapping[str, Any]],
) -> tuple[R3Unit, ...]:
    result = _read_json(root, "R1B repair")
    if (
        result.get("schema") != REPAIR_SCHEMA or result.get("exact") != REPAIR_PRODUCER_EXACT
        or result.get("status") != "R1B_REPAIR_REAL_H_NOT_END_TO_END_READY"
        or result.get("real_h_status") != "PARTIAL_CORE_PASSED"
        or result.get("fine_grid_status") != "PARTIAL_CORE_NONZERO_PREFIX"
        or result.get("r2_candidate") is not False
        or result.get("real_h_passed_condition_count") != 7
        or result.get("fine_nonzero_prefix_condition_count") != 9
    ):
        raise ValueError("R1B repair artifact identity differs")
    memberships = result.get("frozen_old_membership_records")
    real = result.get("real_h_records")
    if not isinstance(memberships, list) or not isinstance(real, list) or len(memberships) != 80 or len(real) != 80:
        raise ValueError("R1B repair fixed 80 records differ")
    expected = _expected(roster)
    if tuple((item.get("condition_id"), item.get("unit_id")) for item in memberships if isinstance(item, Mapping)) != expected:
        raise ValueError("R1B repair membership identity/order differs")
    if tuple((item.get("condition_id"), item.get("unit_id")) for item in real if isinstance(item, Mapping)) != expected:
        raise ValueError("R1B repair real-H identity/order differs")
    units = []
    for identity, membership, real_record, outcome in zip(expected, memberships, real, r2_outcomes, strict=True):
        scores = membership.get("pre_scores_from_old_r1b")
        decision = scores.get("positive_cg_vs_g") if isinstance(scores, Mapping) else None
        s0 = decision.get("margin") if isinstance(decision, Mapping) else None
        if isinstance(s0, bool) or not isinstance(s0, (int, float)) or not math.isfinite(float(s0)):
            s0 = None
        membership_name = membership.get("membership_from_old_r1b")
        reconstructed = outcome_row_from_repair(
            split=_split(identity[1]), condition_id=identity[0], unit_id=identity[1],
            membership=membership_name, record=real_record,
        )
        expected_outcome = _jsonable(reconstructed)
        if any(
            outcome.get(name) != expected_outcome[name]
            for name in (
                "split", "condition_id", "unit_id", "membership", "complete", "safe",
                "safe_rescue", "baseline_positive", "post_positive",
                "observed_negative_false_positive", "errors",
            )
        ):
            raise ValueError("R2 outcome differs from frozen repair membership")
        fp = outcome.get("observed_negative_false_positive")
        if fp not in (True, False, None):
            raise ValueError("R2 outcome false-positive field differs")
        units.append(R3Unit(
            _split(identity[1]), identity[0], identity[1],
            None if s0 is None else float(s0), False,
            bool(outcome["complete"]), bool(outcome["safe"]),
            bool(outcome["safe_rescue"]), bool(outcome["baseline_positive"]),
            outcome["post_positive"], fp,
            tuple(str(item) for item in outcome.get("errors", ()) if isinstance(item, str)),
        ))
    return tuple(units)


def _bind_r2_acceptance(
    units: Sequence[R3Unit], features: Sequence[Mapping[str, Any]],
) -> tuple[R3Unit, ...]:
    result = []
    for unit, row in zip(units, features, strict=True):
        area = row.get("area_ratio")
        accepted = (
            row.get("mandatory_valid") is True
            and isinstance(area, (int, float)) and not isinstance(area, bool)
            and math.isfinite(float(area)) and float(area) >= R2_SELECTED_THRESHOLD
        )
        result.append(R3Unit(
            unit.split, unit.condition_id, unit.unit_id, unit.s0, accepted,
            unit.outcome_complete, unit.safe, unit.safe_rescue,
            unit.baseline_positive, unit.post_positive,
            unit.observed_negative_false_positive, unit.errors,
        ))
    return tuple(result)


def _validate_r1a(root: Path, roster: Sequence[str]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = _read_json(root, "R1A")
    if (
        result.get("schema") != R1A_SCHEMA or result.get("exact") != R1A_PRODUCER_EXACT
        or result.get("status") != R1A_REQUIRED_STATUS
    ):
        raise ValueError("R1A accepted artifact identity/status differs")
    raw = result.get("raw_records")
    if not isinstance(raw, list) or len(raw) != 104:
        raise ValueError("R1A fixed raw record count differs")
    core = tuple(
        item for item in raw
        if isinstance(item, Mapping) and item.get("condition_id") in R2_CONDITION_IDS
    )
    if tuple((item.get("condition_id"), item.get("unit_id")) for item in core) != _expected(roster):
        raise ValueError("R1A core identity/order differs")
    return {(item["condition_id"], item["unit_id"]): item for item in core}


def _matrix_product(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]):
    result = tuple(tuple(sum(float(left[i][k]) * float(right[k][j]) for k in range(3))
                         for j in range(3)) for i in range(3))
    if any(not math.isfinite(value) for row in result for value in row):
        raise ValueError("cycle composition is nonfinite")
    return result


def _transpose_3x3(matrix: Sequence[Sequence[float]]):
    return tuple(tuple(float(matrix[j][i]) for j in range(3)) for i in range(3))


def _geometry_payload(geometry: GeometryEstimate) -> Mapping[str, Any]:
    return {
        "status": geometry.status.value,
        "uncalibrated_sync_logit": geometry.uncalibrated_sync_logit,
        "raw_syncseal_corners": geometry.raw_syncseal_corners,
        "observed_corners_in_canonical_normalized": geometry.observed_corners_in_canonical_normalized,
        "homography_observed_to_canonical": geometry.homography_observed_to_canonical,
        "legal": geometry.legal,
        "error": geometry.error,
    }


def _probe_unit(image: Image.Image, h0: Sequence[Sequence[float]], detector) -> tuple[D4BranchRecord, ...]:
    branches = []
    for transform in D4Transform:
        geometry = None
        d_matrix = d4_homography(transform)
        expected_inverse = _transpose_3x3(d_matrix)
        h_probe = None
        cycle_norm = cycle_px = None
        errors = []
        try:
            h_probe = compose_d4_observed_to_canonical(h0, transform)
            probe = rectify_attacked_rgb(image, h_probe)
            geometry = detector(probe)
            if not isinstance(geometry, GeometryEstimate):
                raise ValueError("geometry_type")
            if (
                geometry.status is GeometryStatus.ERROR or geometry.legal is not True
                or geometry.error is not None or geometry.homography_observed_to_canonical is None
            ):
                raise ValueError("geometry_invalid")
            composed = _matrix_product(
                geometry.homography_observed_to_canonical, d4_homography(transform)
            )
            cycle_norm = corner_rmse(
                apply_homography(composed, CANONICAL_CORNERS_NORMALIZED),
                CANONICAL_CORNERS_NORMALIZED,
            )
            cycle_px = cycle_norm * 511.0 / 2.0
        except Exception as error:
            errors.append(f"d4_probe:{type(error).__name__}:{error}")
        branches.append(D4BranchRecord(
            transform.value, True,
            None if geometry is None else _geometry_payload(geometry),
            cycle_norm, cycle_px, tuple(errors), d_matrix, h_probe, expected_inverse,
        ))
    return tuple(branches)


def _not_probed(reason: str) -> tuple[D4BranchRecord, ...]:
    return tuple(D4BranchRecord(
        item.value, False, None, None, None, (reason,), d4_homography(item), None,
        _transpose_3x3(d4_homography(item)),
    ) for item in D4Transform)


def _rows_for_split(
    *, split: str, units: Sequence[R3Unit],
    r1a: Mapping[tuple[str, str], Mapping[str, Any]], r1a_root: Path,
    detector, allow_probe: bool,
) -> tuple[CycleFeatureRow, ...]:
    result = []
    for unit in units:
        route = route_from_s0(unit.s0)
        eligible = route == "BOUNDARY" and unit.r2_selector_accepted
        branches = _not_probed(f"{R3_NOT_PROBED_INELIGIBLE}:{route}")
        if eligible and allow_probe:
            try:
                raw = r1a[(unit.condition_id, unit.unit_id)]
                geometry = raw.get("geometry")
                h0 = geometry.get("homography_observed_to_canonical") if isinstance(geometry, Mapping) else None
                relative = raw.get("attacked_image_file")
                if not isinstance(relative, str) or not isinstance(h0, (list, tuple)):
                    raise ValueError("stored H0/image is missing")
                with Image.open(r1a_root / relative) as source:
                    image = source.convert("RGB")
                branches = _probe_unit(image, h0, detector)
            except Exception as error:
                branches = _not_probed(f"eligible_probe_input:{type(error).__name__}:{error}")
        elif eligible:
            branches = _not_probed("R3_NOT_PROBED_NO_FROZEN_DEV_CANDIDATE")
        result.append(cycle_feature_row(
            unit=unit, branches=branches, d4_order=D4_ORDER,
        ))
    return tuple(result)


def _setup_failure_rows(
    *, split: str, units: Sequence[R3Unit], error: BaseException,
) -> tuple[CycleFeatureRow, ...]:
    rows = []
    for unit in units:
        route = route_from_s0(unit.s0)
        eligible = route == "BOUNDARY" and unit.r2_selector_accepted
        reason = f"syncseal_runtime_setup:{type(error).__name__}:{error}"
        branches = _not_probed(reason)
        rows.append(cycle_feature_row(
            unit=unit, branches=branches, d4_order=D4_ORDER,
        ))
    return tuple(rows)


def _ordered_split(units: Sequence[R3Unit], split: str) -> tuple[R3Unit, ...]:
    wanted = R2_DEV_UNIT_IDS if split == "dev" else R2_TEST_UNIT_IDS
    return tuple(item for item in units if item.unit_id in wanted)


def _payload(
    *, exact: str, r1a_root: Path, repair_root: Path, r2_root: Path,
    selection=None, dev_rows=(), test_rows=(), test_metrics=None,
    input_error: BaseException | None = None,
    setup_error: BaseException | None = None,
) -> Mapping[str, Any]:
    operational = input_error is not None or setup_error is not None
    status = R3_OPERATIONAL_FAILURE if operational else (
        R3_METHOD_NOT_IMPROVED if selection is None else selection.status
    )
    return {
        "schema": RESULT_SCHEMA, "stage": STAGE_LABEL, "exact": exact,
        "status": status, "scientific_status": "not_adjudicated",
        "claim_ceiling": R3_CLAIM_CEILING,
        "data_used_for_development": True,
        "inputs": {
            "r1a": {"producer_exact": R1A_PRODUCER_EXACT, "artifact_root": str(r1a_root)},
            "r1b_repair": {"producer_exact": REPAIR_PRODUCER_EXACT, "artifact_root": str(repair_root)},
            "r2": {"producer_exact": R2_PRODUCER_EXACT, "artifact_root": str(r2_root),
                   "frozen_candidate_id": R2_SELECTED_ID,
                   "recorded_status_unchanged": "R2_SELECTIVE_RISK_FAILED"},
        },
        "fixed": {
            "tau": 0.0, "b_low": R3_B_LOW, "b_low_hex": R3_B_LOW_HEX,
            "b_low_source_note": "frozen prior development q20; not a calibration claim",
            "d4_order": list(D4_ORDER), "cycle_threshold_grid_px": list(R3_THRESHOLD_GRID_PX),
            "conditions": 10, "units": 8, "dev": 40, "test": 40,
            "probe_scope": "BOUNDARY_AND_FROZEN_R2_ACCEPTED_ONLY",
        },
        "development_threshold_selection": _jsonable(selection),
        "existing_test40_engineering_diagnostic": _jsonable(test_metrics),
        "feature_rows": [_jsonable(item) for item in (*dev_rows, *test_rows)],
        "probe_accounting": {
            "eligible_units": sum(item.route == "BOUNDARY" and item.r2_selector_accepted for item in (*dev_rows, *test_rows)),
            "probed_units": sum(any(branch.probed for branch in item.branches) for item in (*dev_rows, *test_rows)),
            "eligible_units_with_exact_8_attempts": sum(
                item.route == "BOUNDARY" and item.r2_selector_accepted
                and sum(branch.probed for branch in item.branches) == 8
                for item in (*dev_rows, *test_rows)
            ),
            "eligible_units_with_probe_input_or_setup_failure": sum(
                item.route == "BOUNDARY" and item.r2_selector_accepted
                and sum(branch.probed for branch in item.branches) == 0
                for item in (*dev_rows, *test_rows)
            ),
            "executed_probe_count": sum(branch.probed for item in (*dev_rows, *test_rows) for branch in item.branches),
            "fixed_unit_table_denominator": 80,
        },
        "operational_error": None if not operational else f"{type(input_error or setup_error).__name__}: {input_error or setup_error}",
        "route": {
            "unseen_or_formal_test_claim": False, "eligible_only_d4_probes": True,
            "h0_updated_replaced_or_averaged": False, "content_scorer_invoked": False,
            "final_recovery_uses_raw_h0_once": True,
            "same_detector_key_preprocess_tau_for_recovery": True,
            "content_positive_vote_from_geometry": False, "no_retry_fallback_or_subset": True,
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
        raise FileExistsError("R3 result directory must be create-only")
    if checkpoint.exists():
        raise FileExistsError("R3 checkpoint must be create-only")
    exact = _git_exact(repo_root, args.expected_exact)
    r1a_root = Path(args.r1a_artifact_root).resolve()
    repair_root = Path(args.r1b_repair_artifact_root).resolve()
    r2_root = Path(args.r2_artifact_root).resolve()
    try:
        roster, r2_features, r2_outcomes = _validate_r2(r2_root)
        units = _bind_r2_acceptance(
            _validate_repair(repair_root, roster, r2_outcomes), r2_features
        )
        r1a = _validate_r1a(r1a_root, roster)
        dev_units = _ordered_split(units, "dev")
        test_units = _ordered_split(units, "test")
    except Exception as error:
        payload = _payload(
            exact=exact, r1a_root=r1a_root, repair_root=repair_root,
            r2_root=r2_root, input_error=error,
        )
        _write_result(result_root, payload)
        return payload
    if not torch.cuda.is_available():
        setup_error = RuntimeError("cuda_required_for_real_geometry_v7_r3")
        dev_rows = _setup_failure_rows(
            split="dev", units=dev_units, error=setup_error,
        )
        test_rows = _setup_failure_rows(
            split="test", units=test_units, error=setup_error,
        )
        payload = _payload(
            exact=exact, r1a_root=r1a_root, repair_root=repair_root,
            r2_root=r2_root, dev_rows=dev_rows,
            test_rows=test_rows, setup_error=setup_error,
        )
        _write_result(result_root, payload)
        return payload
    try:
        loaded = download_official_syncseal_torchscript(checkpoint)
        syncseal = SyncSealTorchScript.from_file(loaded, device="cuda")
    except Exception as error:
        dev_rows = _setup_failure_rows(
            split="dev", units=dev_units, error=error,
        )
        test_rows = _setup_failure_rows(
            split="test", units=test_units, error=error,
        )
        payload = _payload(
            exact=exact, r1a_root=r1a_root, repair_root=repair_root,
            r2_root=r2_root, dev_rows=dev_rows,
            test_rows=test_rows, setup_error=error,
        )
        _write_result(result_root, payload)
        return payload
    dev_rows = _rows_for_split(
        split="dev", units=dev_units, r1a=r1a,
        r1a_root=r1a_root, detector=syncseal.detect_geometry, allow_probe=True,
    )
    selection = select_threshold(dev_rows, dev_units)
    test_rows = _rows_for_split(
        split="test", units=test_units, r1a=r1a,
        r1a_root=r1a_root, detector=syncseal.detect_geometry,
        allow_probe=selection.selected_threshold_px is not None,
    )
    test_metrics = None
    if selection.selected_threshold_px is not None:
        test_metrics = evaluate_selected_threshold(
            selection.selected_threshold_px, test_rows, test_units
        )
    payload = _payload(
        exact=exact, r1a_root=r1a_root, repair_root=repair_root, r2_root=r2_root,
        selection=selection, dev_rows=dev_rows,
        test_rows=test_rows, test_metrics=test_metrics,
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
    parser.add_argument("--syncseal-checkpoint", required=True)
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
    return 0 if payload["status"] == R3_METHOD_IMPROVED else 2


if __name__ == "__main__":
    raise SystemExit(main())
