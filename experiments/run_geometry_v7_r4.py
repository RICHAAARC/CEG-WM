"""Assemble the Geometry-V7 refined-R3/R4 engineering replay on CPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r3 import R3_B_LOW
from cegwm.geometry_v7.r4 import (
    R4_CLAIM_CEILING, R4_ENGINEERING_REPLAY_RECORDED,
    R4_OPERATIONAL_FAILURE, refined_gate_from_features, score_route,
)


REPAIR_EXACT = "3b9819d80b07704a4caab8b7aaa581cf9eb8a3c5"
REPAIR_SCHEMA = "geometry_v7_r1b_repair_result_v1"
R2_EXACT = "ffac9d4c1e575c27240d9423bbd30e0713aa2dcd"
R2_SCHEMA = "geometry_v7_r2_selective_result_v1"
ADVANCED_EXACT = "580ae951740419d1ccbc0e53fb81e1df7de6c469"
ADVANCED_SCHEMA = "geometry_v7_r3_advanced_result_v1"
R1A_EXACT = "ac590330e91aacf4b3283df1e94572a0e4f983a0"
OLD_R3_EXACT = "896571fd17fbc161bbb617f74677328a012ce43a"
RESULT_SCHEMA = "geometry_v7_r4_engineering_replay_v1"
SCOPE_CONDITIONS = (
    "core_rotation_neg15", "core_rotation_pos15", "core_fixed_canvas_zoom_0_8",
)


def _read(root: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads((root.resolve() / "result.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} result.json must be readable JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} result.json must be an object")
    return payload


def _git_exact(root: Path, expected: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("expected exact must be lowercase 40-hex")
    exact = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                           text=True, capture_output=True).stdout.strip()
    if exact != expected:
        raise RuntimeError("checkout differs from expected exact")
    if subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                      text=True, capture_output=True).stdout:
        raise RuntimeError("execution checkout must be clean")
    return exact


def _expected(roster: Sequence[str]):
    return tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in roster)


def _identities(rows: Sequence[Mapping[str, Any]]):
    return tuple((row.get("condition_id"), row.get("unit_id")) for row in rows)


def _validate_inputs(repair_root: Path, r2_root: Path, advanced_root: Path):
    repair, r2, advanced = (_read(repair_root, "R1B repair"), _read(r2_root, "R2"),
                            _read(advanced_root, "advanced R3"))
    expected80 = _expected(R2_DEV_UNIT_IDS + R2_TEST_UNIT_IDS)
    if (
        repair.get("schema") != REPAIR_SCHEMA or repair.get("exact") != REPAIR_EXACT
        or repair.get("status") != "R1B_REPAIR_REAL_H_NOT_END_TO_END_READY"
    ):
        raise ValueError("R1B repair identity/status differs")
    pre, real = repair.get("frozen_old_membership_records"), repair.get("real_h_records")
    if not isinstance(pre, list) or not isinstance(real, list) or len(pre) != 80 or len(real) != 80:
        raise ValueError("R1B repair fixed records differ")
    if _identities(pre) != expected80 or _identities(real) != expected80:
        raise ValueError("R1B repair identity/order differs")
    if r2.get("schema") != R2_SCHEMA or r2.get("exact") != R2_EXACT or r2.get("status") != "R2_SELECTIVE_RISK_FAILED":
        raise ValueError("R2 identity/status differs")
    features, outcomes = r2.get("feature_rows"), r2.get("outcome_rows")
    if not isinstance(features, list) or not isinstance(outcomes, list) or len(features) != 80 or len(outcomes) != 80:
        raise ValueError("R2 fixed rows differ")
    if _identities(features) != expected80 or _identities(outcomes) != expected80:
        raise ValueError("R2 identity/order differs")
    if (
        advanced.get("schema") != ADVANCED_SCHEMA or advanced.get("exact") != ADVANCED_EXACT
        or advanced.get("status") != "R3_ADVANCED_ENGINEERING_TEST40_RECORDED"
    ):
        raise ValueError("advanced R3 identity/status differs")
    inputs = advanced.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("r1a", {}).get("producer_exact") != R1A_EXACT
        or inputs.get("r1b_repair", {}).get("producer_exact") != REPAIR_EXACT
        or inputs.get("r2", {}).get("producer_exact") != R2_EXACT
        or inputs.get("old_r3", {}).get("producer_exact") != OLD_R3_EXACT
    ):
        raise ValueError("advanced R3 input binding differs")
    dev, test = advanced.get("development_decisions"), advanced.get("existing_test40_decisions")
    if not isinstance(dev, list) or not isinstance(test, list) or len(dev) != 40 or len(test) != 40:
        raise ValueError("advanced R3 fixed decisions differ")
    if _identities(dev) != _expected(R2_DEV_UNIT_IDS) or _identities(test) != _expected(R2_TEST_UNIT_IDS):
        raise ValueError("advanced R3 split identity/order differs")
    return repair, r2, advanced


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("finite numeric field required")
    return float(value)


def _decision(payload: object) -> tuple[float, bool]:
    if not isinstance(payload, Mapping):
        raise ValueError("paired decision missing")
    gate_a, gate_b, margin = (_finite(payload.get(name)) for name in
                              ("gate_a_margin", "gate_b_margin", "margin"))
    positive = payload.get("positive")
    if not isinstance(positive, bool) or margin != min(gate_a, gate_b) or positive != (gate_a > 0.0 and gate_b > 0.0):
        raise ValueError("paired decision differs from strict frozen gates")
    return margin, positive


def _refined(decision: Mapping[str, Any]):
    regime = decision.get("decision", {}).get("regime")
    if not isinstance(regime, Mapping):
        return refined_gate_from_features(
            boundary=False, r2_selector_accepted=False, regime_valid=False,
            angle_degrees=None, scale=None, translation=None, perspective=None,
            pure_rotation=False, cycle_score_px=None,
        )
    return refined_gate_from_features(
        boundary=decision.get("route") == "BOUNDARY",
        r2_selector_accepted=decision.get("r2_selector_accepted") is True,
        regime_valid=regime.get("valid") is True,
        angle_degrees=regime.get("angle_degrees"), scale=regime.get("scale"),
        translation=regime.get("translation"), perspective=regime.get("perspective"),
        pure_rotation=decision.get("decision", {}).get("pure_rotation_gate") is True,
        cycle_score_px=decision.get("old_cycle_score_px"),
    )


def _metrics(rows: Sequence[Mapping[str, Any]], split: str):
    selected = tuple(row for row in rows if row["refined_reliable"])
    safe = sum(row["safe"] for row in selected)
    return {
        "split": split, "fixed_denominator": 40, "accepted_count": len(selected),
        "safe_rescue_count": safe, "unsafe_accept_count": len(selected) - safe,
        "selected_negative_control_fp_count": sum(row["final_negative_fp"] for row in selected),
        "covered_attack_count": len({row["condition_id"] for row in selected}),
    }


def _scope_metrics(rows: Sequence[Mapping[str, Any]], split: str):
    scoped = tuple(row for row in rows if row["in_scope"])
    return {
        "split": split, "fixed_denominator": 12,
        "baseline_positive_count": sum(row["baseline_positive"] for row in scoped),
        "reliable_count": sum(row["refined_reliable"] for row in scoped),
        "recovered_positive_count": sum(row["refined_reliable"] and row["final_positive"] for row in scoped),
        "final_positive_count": sum(row["final_positive"] for row in scoped),
        "net_rescue_change": sum(row["final_positive"] for row in scoped) - sum(row["baseline_positive"] for row in scoped),
        "negative_control_fp_count": sum(row["final_negative_fp"] for row in scoped),
        "decision_harm_count": sum(row["decision_harm"] for row in scoped),
        "failure_count": sum(bool(row["errors"]) for row in scoped),
        "known_failed_or_out_of_scope_count": sum(not row["in_scope"] for row in rows),
    }


def _assemble(repair: Mapping[str, Any], advanced: Mapping[str, Any]):
    decisions = tuple(advanced["development_decisions"] + advanced["existing_test40_decisions"])
    pre_by = {f"{row['condition_id']}|{row['unit_id']}": row for row in repair["frozen_old_membership_records"]}
    real_by = {f"{row['condition_id']}|{row['unit_id']}": row for row in repair["real_h_records"]}
    rows = []
    for decision in decisions:
        identity = f"{decision['condition_id']}|{decision['unit_id']}"
        pre, real = pre_by[identity], real_by[identity]
        errors = [str(item) for item in real.get("errors", ())]
        try:
            pre_scores = pre.get("pre_scores_from_old_r1b")
            s0, baseline_positive = _decision(pre_scores.get("positive_cg_vs_g"))
            _, pre_fp = _decision(pre_scores.get("negative_g_vs_u"))
            route = score_route(s0)
            gate = _refined(decision)
            post_positive, post_fp = False, False
            if gate.reliable:
                scores = real.get("scores")
                if not isinstance(scores, Mapping):
                    raise ValueError("reliable unit lacks real-H scores")
                _, post_positive = _decision(scores.get("positive_cg_vs_g"))
                _, post_fp = _decision(scores.get("negative_g_vs_u"))
            final_positive = baseline_positive if route == "DIRECT_POSITIVE" else (
                post_positive if gate.reliable else False
            )
            final_fp = post_fp if gate.reliable else pre_fp
            safe = bool(
                gate.reliable and real.get("positive_score_delta", 0) > 0
                and real.get("recovered_negative") is True and final_positive and not final_fp
            )
        except Exception as error:
            errors.append(f"artifact_replay:{type(error).__name__}:{error}")
            s0, route, gate = None, "INVALID_SCORE", _refined({})
            baseline_positive = final_positive = final_fp = safe = False
        split = decision["split"]
        in_scope = decision["condition_id"] in SCOPE_CONDITIONS
        rows.append({
            "split": split, "condition_id": decision["condition_id"],
            "unit_id": decision["unit_id"], "in_scope": in_scope,
            "scope_status": "R4_REFINED_SCOPE" if in_scope else "KNOWN_FAILED_OUT_OF_SCOPE",
            "s0": s0, "route": route, "refined_gate": gate.__dict__ if hasattr(gate, "__dict__") else {
                name: getattr(gate, name) for name in gate.__slots__
            },
            "refined_reliable": gate.reliable, "baseline_positive": baseline_positive,
            "final_positive": final_positive, "final_negative_fp": final_fp,
            "safe": safe, "decision_harm": baseline_positive and not final_positive,
            "errors": errors,
        })
    dev, test = tuple(row for row in rows if row["split"] == "dev"), tuple(row for row in rows if row["split"] == "test")
    refined = (_metrics(dev, "dev"), _metrics(test, "test"))
    scoped = (_scope_metrics(dev, "dev"), _scope_metrics(test, "test"))
    zoom_direct_negative = tuple(
        row for row in rows
        if row["condition_id"] == "core_fixed_canvas_zoom_0_8"
        and row["route"] == "DIRECT_NEGATIVE"
    )
    if (
        len(zoom_direct_negative) != 3
        or sum(row["split"] == "dev" for row in zoom_direct_negative) != 2
        or sum(row["split"] == "test" for row in zoom_direct_negative) != 1
    ):
        raise ValueError("R4 fixed-scope direct-negative zoom units differ")
    if refined != (
        {"split":"dev","fixed_denominator":40,"accepted_count":10,"safe_rescue_count":10,"unsafe_accept_count":0,"selected_negative_control_fp_count":0,"covered_attack_count":3},
        {"split":"test","fixed_denominator":40,"accepted_count":11,"safe_rescue_count":11,"unsafe_accept_count":0,"selected_negative_control_fp_count":0,"covered_attack_count":3},
    ):
        raise ValueError("refined R3 frozen replay metrics differ")
    expected_scope = (("dev",10),("test",11))
    for metric, (split, count) in zip(scoped, expected_scope, strict=True):
        expected = {"split":split,"fixed_denominator":12,"baseline_positive_count":0,
                    "reliable_count":count,"recovered_positive_count":count,
                    "final_positive_count":count,"net_rescue_change":count,
                    "negative_control_fp_count":0,"decision_harm_count":0,
                    "failure_count":0,"known_failed_or_out_of_scope_count":28}
        if metric != expected:
            raise ValueError("R4 fixed-scope replay metrics differ")
    return tuple(rows), refined, scoped


def _payload(*, exact: str, repair_root: Path, r2_root: Path, advanced_root: Path,
             rows=(), refined=(), scoped=(), error: BaseException | None = None):
    return {
        "schema": RESULT_SCHEMA, "exact": exact,
        "status": R4_OPERATIONAL_FAILURE if error else R4_ENGINEERING_REPLAY_RECORDED,
        "scientific_status": "not_adjudicated", "claim_ceiling": R4_CLAIM_CEILING,
        "data_used_for_development": True,
        "inputs": {"r1b_repair":{"exact":REPAIR_EXACT,"root":str(repair_root)},
                   "r2":{"exact":R2_EXACT,"root":str(r2_root)},
                   "advanced_r3":{"exact":ADVANCED_EXACT,"root":str(advanced_root)}},
        "fixed": {"total_rows":80,"dev":40,"test":40,"scope_per_split":12,
                  "scope_conditions":list(SCOPE_CONDITIONS),"tau":0.0,"b_low":R3_B_LOW,
                  "s0_zero_route":"BOUNDARY","other_per_split":28},
        "refined_r3_metrics": list(refined), "r4_scope_metrics": list(scoped),
        "rows": list(rows),
        "operational_error": None if error is None else f"{type(error).__name__}: {error}",
        "route": {"artifact_replay_only":True,"actual_callback_not_executed":True,
                  "geometry_positive_vote":False,"raw_h0_once":True,
                  "same_bound_score_callback_required":True,"no_retry_or_fallback":True,
                  "r4_promotion_claim":False},
    }


def _write(root: Path, payload: Mapping[str, Any]):
    root.mkdir(parents=True, exist_ok=False)
    with (root / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def _run(args):
    result = Path(args.result_dir).resolve()
    if result.exists():
        raise FileExistsError("R4 result directory must be create-only")
    exact = _git_exact(Path(args.repo_root).resolve(), args.expected_exact)
    roots = tuple(Path(item).resolve() for item in (args.r1b_repair_root, args.r2_root, args.advanced_r3_root))
    try:
        repair, _, advanced = _validate_inputs(*roots)
        rows, refined, scoped = _assemble(repair, advanced)
        payload = _payload(exact=exact, repair_root=roots[0], r2_root=roots[1],
                           advanced_root=roots[2], rows=rows, refined=refined, scoped=scoped)
    except Exception as error:
        payload = _payload(exact=exact, repair_root=roots[0], r2_root=roots[1],
                           advanced_root=roots[2], error=error)
    _write(result, payload)
    return payload


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True); parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--r1b-repair-root", required=True); parser.add_argument("--r2-root", required=True)
    parser.add_argument("--advanced-r3-root", required=True); parser.add_argument("--result-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try: payload = _run(args)
    except Exception as error:
        print(json.dumps({"status":"RUNNER_STOPPED_BEFORE_PACKAGE","error":f"{type(error).__name__}: {error}"}, sort_keys=True)); return 2
    print(json.dumps({"status":payload["status"],"result_dir":args.result_dir}, sort_keys=True))
    return 0 if payload["status"] == R4_ENGINEERING_REPLAY_RECORDED else 2


if __name__ == "__main__":
    raise SystemExit(main())
