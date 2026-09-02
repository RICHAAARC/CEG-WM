"""Run Geometry-V7-Direct replay or the fixed-seven real callback route."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from experiments import run_geometry_v7_r4 as reliable_runner
from cegwm.geometry_v7.contracts import GeometryEstimate
from cegwm.geometry_v7.r0 import ContentScore
from cegwm.geometry_v7.r1a import R1A_CORE_CONDITIONS, render_r1a_attack
from cegwm.geometry_v7.r1b import rectify_attacked_rgb, scored_triplet
from cegwm.geometry_v7.r2 import R2_CONDITION_IDS, R2_DEV_UNIT_IDS, R2_TEST_UNIT_IDS
from cegwm.geometry_v7.r4 import detect_direct_and_recover, score_route


REPLAY_SCHEMA = "geometry_v7_r4_direct_replay_v1"
CALLBACK_SCHEMA = "geometry_v7_r4_direct_callback_v1"
REPLAY_STATUS = "R4_DIRECT_ENGINEERING_REPLAY_RECORDED"
CALLBACK_STATUS = "R4_DIRECT_CALLBACK_RECORDED"
OPERATIONAL_STATUS = "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"
CLAIM_CEILING = "existing_observed_data_engineering_only_no_science_or_r4_promotion"
R0_EXACT = "4f0bf1560805672f786dc86dd50d793aec18aae7"
R1A_EXACT = "ac590330e91aacf4b3283df1e94572a0e4f983a0"

MAIN_CONDITIONS = (
    "core_rotation_neg15",
    "core_rotation_pos15",
    "core_fixed_canvas_zoom_0_8",
    "core_translation_neg32_x",
    "core_translation_pos32_y",
    "core_translation_neg32_y",
    "core_offset_crop_rescale",
)
CALLBACK_ROSTER = (
    ("core_rotation_neg15", R2_DEV_UNIT_IDS[0]),
    ("core_rotation_pos15", R2_DEV_UNIT_IDS[0]),
    ("core_fixed_canvas_zoom_0_8", R2_DEV_UNIT_IDS[0]),
    ("core_translation_neg32_x", R2_DEV_UNIT_IDS[0]),
    ("core_translation_pos32_y", R2_DEV_UNIT_IDS[0]),
    ("core_translation_neg32_y", R2_DEV_UNIT_IDS[0]),
    ("core_offset_crop_rescale", R2_DEV_UNIT_IDS[1]),
)
EXPECTED_ATTACK_SAFE_FP = {
    "core_rotation_neg15": (8, 0),
    "core_rotation_pos15": (8, 0),
    "core_fixed_canvas_zoom_0_8": (5, 0),
    "core_translation_neg32_x": (4, 0),
    "core_translation_pos32_y": (5, 0),
    "core_translation_neg32_y": (3, 0),
    "core_offset_crop_rescale": (3, 0),
    "core_fixed_canvas_zoom_1_2": (1, 2),
    "core_translation_pos32_x": (2, 1),
    "core_composite_c0_85_t16_neg16_r10": (1, 0),
}


@dataclass(frozen=True, slots=True)
class CallbackInput:
    condition_id: str
    unit_id: str
    u: Image.Image | None
    g: Image.Image | None
    cg: Image.Image | None
    errors: tuple[str, ...] = ()


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
    if unit_id in R2_DEV_UNIT_IDS:
        return "dev"
    if unit_id in R2_TEST_UNIT_IDS:
        return "test"
    raise ValueError("unit is outside the frozen evaluation roster")


def _expected(roster: Sequence[str]):
    return tuple((condition, unit) for condition in R2_CONDITION_IDS for unit in roster)


def _identities(rows: Sequence[Mapping[str, Any]]):
    return tuple((row.get("condition_id"), row.get("unit_id")) for row in rows)


def _paired_decision(payload: object) -> tuple[float, bool]:
    return reliable_runner._decision(payload)


def _geometry_valid(decision: Mapping[str, Any]) -> bool:
    details = decision.get("decision")
    regime = details.get("regime") if isinstance(details, Mapping) else None
    return bool(isinstance(regime, Mapping) and regime.get("valid") is True)


def _replay_rows(
    repair: Mapping[str, Any], r2: Mapping[str, Any], advanced: Mapping[str, Any]
):
    decisions = tuple(
        advanced["development_decisions"] + advanced["existing_test40_decisions"]
    )
    pre_by = {
        (row["condition_id"], row["unit_id"]): row
        for row in repair["frozen_old_membership_records"]
    }
    real_by = {
        (row["condition_id"], row["unit_id"]): row
        for row in repair["real_h_records"]
    }
    feature_by = {
        (row["condition_id"], row["unit_id"]): row for row in r2["feature_rows"]
    }
    outcome_by = {
        (row["condition_id"], row["unit_id"]): row for row in r2["outcome_rows"]
    }
    rows = []
    for decision in decisions:
        identity = (decision["condition_id"], decision["unit_id"])
        pre, real = pre_by[identity], real_by[identity]
        errors = tuple(str(item) for item in real.get("errors", ()))
        derived_errors: list[str] = []
        s0 = None
        pre_positive = pre_fp = False
        post_positive = post_fp = False
        route = "INVALID_SCORE"
        recovered = False
        try:
            pre_scores = pre.get("pre_scores_from_old_r1b")
            if not isinstance(pre_scores, Mapping):
                raise ValueError("pre scores missing")
            s0, pre_positive = _paired_decision(pre_scores.get("positive_cg_vs_g"))
            _, pre_fp = _paired_decision(pre_scores.get("negative_g_vs_u"))
            route = score_route(s0)
            if route == "BOUNDARY":
                if not _geometry_valid(decision):
                    derived_errors.append("direct_replay:invalid_stored_geometry")
                elif errors:
                    derived_errors.extend(errors)
                else:
                    scores = real.get("scores")
                    if not isinstance(scores, Mapping):
                        raise ValueError("real-H scores missing")
                    _, post_positive = _paired_decision(scores.get("positive_cg_vs_g"))
                    _, post_fp = _paired_decision(scores.get("negative_g_vs_u"))
                    recovered = True
        except Exception as error:
            derived_errors.append(f"direct_replay:{type(error).__name__}:{error}")
        direct_final = bool(
            pre_positive if route == "DIRECT_POSITIVE"
            else post_positive if route == "BOUNDARY" and recovered
            else False
        )
        direct_fp = bool(post_fp if recovered else pre_fp)
        direct_safe = bool(recovered and direct_final and not direct_fp)
        reliable_gate = reliable_runner._refined(decision)
        reliable_recovered = bool(
            reliable_gate.reliable and not errors and isinstance(real.get("scores"), Mapping)
        )
        reliable_positive = reliable_fp = False
        if reliable_recovered:
            _, reliable_positive = _paired_decision(real["scores"].get("positive_cg_vs_g"))
            _, reliable_fp = _paired_decision(real["scores"].get("negative_g_vs_u"))
        rows.append({
            "split": decision["split"],
            "condition_id": identity[0],
            "unit_id": identity[1],
            "s0": s0,
            "route": route,
            "content_only": {
                "final_positive": bool(pre_positive),
                "paired_negative_false_positive": bool(pre_fp),
            },
            "direct": {
                "geometry_valid": _geometry_valid(decision),
                "recovered": recovered,
                "final_positive": direct_final,
                "paired_negative_false_positive": direct_fp,
                "safe_recovered": direct_safe,
                "recovery_failure": route == "BOUNDARY" and not recovered,
            },
            "reliable_ablation": {
                "accepted": reliable_gate.reliable,
                "recovered": reliable_recovered,
                "final_positive": reliable_positive,
                "paired_negative_false_positive": reliable_fp,
                "safe_recovered": bool(
                    reliable_recovered and reliable_positive and not reliable_fp
                ),
            },
            "raw": {
                "pre_scores": pre.get("pre_scores_from_old_r1b"),
                "real_h_scores": real.get("scores"),
                "stored_real_h_errors": list(errors),
                "r2_feature": feature_by[identity],
                "r2_outcome": outcome_by[identity],
                "advanced_decision": decision,
            },
            "errors": derived_errors,
        })
    return tuple(rows)


def _row_metrics(rows: Sequence[Mapping[str, Any]]):
    return {
        "fixed_denominator": len(rows),
        "direct_positive_count": sum(row["route"] == "DIRECT_POSITIVE" for row in rows),
        "direct_negative_count": sum(row["route"] == "DIRECT_NEGATIVE" for row in rows),
        "boundary_count": sum(row["route"] == "BOUNDARY" for row in rows),
        "recovered_count": sum(row["direct"]["recovered"] for row in rows),
        "final_positive_count": sum(row["direct"]["final_positive"] for row in rows),
        "paired_negative_false_positive_count": sum(
            row["direct"]["paired_negative_false_positive"] for row in rows
        ),
        "safe_recovered_count": sum(row["direct"]["safe_recovered"] for row in rows),
        "recovery_failure_count": sum(row["direct"]["recovery_failure"] for row in rows),
        "content_only_final_positive_count": sum(
            row["content_only"]["final_positive"] for row in rows
        ),
    }


def _aggregate_replay(rows: Sequence[Mapping[str, Any]]):
    global_metrics = _row_metrics(rows)
    split_metrics = {
        split: _row_metrics(tuple(row for row in rows if row["split"] == split))
        for split in ("dev", "test")
    }
    per_attack = {}
    for condition in R2_CONDITION_IDS:
        selected = tuple(row for row in rows if row["condition_id"] == condition)
        metrics = _row_metrics(selected)
        metrics["condition_id"] = condition
        per_attack[condition] = metrics
    main_rows = tuple(row for row in rows if row["condition_id"] in MAIN_CONDITIONS)
    main = {
        "fixed_denominator": len(main_rows),
        "safe_recovered_count": sum(row["direct"]["safe_recovered"] for row in main_rows),
        "paired_negative_false_positive_count": sum(
            row["direct"]["paired_negative_false_positive"] for row in main_rows
        ),
        "dev": {
            "fixed_denominator": 28,
            "safe_recovered_count": sum(
                row["direct"]["safe_recovered"] for row in main_rows if row["split"] == "dev"
            ),
        },
        "test": {
            "fixed_denominator": 28,
            "safe_recovered_count": sum(
                row["direct"]["safe_recovered"] for row in main_rows if row["split"] == "test"
            ),
        },
    }
    reliable = {
        "accepted_count": sum(row["reliable_ablation"]["accepted"] for row in rows),
        "safe_recovered_count": sum(
            row["reliable_ablation"]["safe_recovered"] for row in rows
        ),
        "paired_negative_false_positive_count": sum(
            row["reliable_ablation"]["paired_negative_false_positive"] for row in rows
            if row["reliable_ablation"]["accepted"]
        ),
        "dev_accepted_count": sum(
            row["reliable_ablation"]["accepted"] for row in rows if row["split"] == "dev"
        ),
        "test_accepted_count": sum(
            row["reliable_ablation"]["accepted"] for row in rows if row["split"] == "test"
        ),
    }
    expected_global = {
        "fixed_denominator": 80,
        "direct_positive_count": 0,
        "direct_negative_count": 14,
        "boundary_count": 66,
        "recovered_count": 66,
        "final_positive_count": 43,
        "paired_negative_false_positive_count": 3,
        "safe_recovered_count": 40,
        "recovery_failure_count": 0,
        "content_only_final_positive_count": 0,
    }
    if global_metrics != expected_global:
        raise ValueError("Geometry-V7-Direct frozen global replay metrics differ")
    if main != {
        "fixed_denominator": 56,
        "safe_recovered_count": 36,
        "paired_negative_false_positive_count": 0,
        "dev": {"fixed_denominator": 28, "safe_recovered_count": 18},
        "test": {"fixed_denominator": 28, "safe_recovered_count": 18},
    }:
        raise ValueError("Geometry-V7-Direct seven-family replay metrics differ")
    for condition, expected in EXPECTED_ATTACK_SAFE_FP.items():
        actual = (
            per_attack[condition]["safe_recovered_count"],
            per_attack[condition]["paired_negative_false_positive_count"],
        )
        if actual != expected:
            raise ValueError("Geometry-V7-Direct per-attack replay metrics differ")
    if reliable != {
        "accepted_count": 21,
        "safe_recovered_count": 21,
        "paired_negative_false_positive_count": 0,
        "dev_accepted_count": 10,
        "test_accepted_count": 11,
    }:
        raise ValueError("Geometry-V7 Reliable ablation metrics differ")
    return {
        "global": global_metrics,
        "by_split": split_metrics,
        "per_attack": per_attack,
        "seven_main_conditions": main,
        "reliable_ablation": reliable,
    }


def _validate_r1a_identity(
    repo_root: Path, artifact_root: Path, ordered_roster: Sequence[str]
) -> None:
    from experiments import run_geometry_v7_r1b as repair_runner

    result = repair_runner._read_json(artifact_root.resolve() / "result.json", "R1A")
    roster = tuple(ordered_roster)
    r0_input = result.get("r0_input")
    fixed = result.get("fixed_counts")
    if (
        result.get("exact") != R1A_EXACT
        or result.get("status") != "R1A_BLOCKING_METHOD_CANARY_PASSED"
        or result.get("blocking_method_canary_passed") is not True
        or not isinstance(r0_input, Mapping)
        or r0_input.get("producer_exact") != R0_EXACT
        or r0_input.get("selected_residual_strength_multiplier") != 0.75
        or tuple(
            item.get("unit_id") for item in r0_input.get("ordered_evaluation_cg_inputs", ())
            if isinstance(item, Mapping)
        ) != roster
        or not isinstance(fixed, Mapping)
        or fixed.get("core_conditions") != 10
        or fixed.get("units_per_condition") != 8
        or fixed.get("records") != 104
    ):
        raise ValueError("R1A identity or fixed input roster differs")
    raw = result.get("raw_records")
    if not isinstance(raw, list) or len(raw) != 104:
        raise ValueError("R1A fixed raw-record identities are absent")
    for spec in R1A_CORE_CONDITIONS:
        matches = tuple(
            row for row in raw
            if isinstance(row, Mapping) and row.get("condition_id") == spec.condition_id
        )
        if _identities(matches) != tuple((spec.condition_id, unit) for unit in roster):
            raise ValueError("R1A core record identity/order differs")


def _validate_callback_repair(artifact_root: Path) -> Mapping[str, Any]:
    result = reliable_runner._read(artifact_root.resolve(), "R1B repair")
    expected = _expected(R2_DEV_UNIT_IDS + R2_TEST_UNIT_IDS)
    memberships = result.get("frozen_old_membership_records")
    real_h = result.get("real_h_records")
    if (
        result.get("schema") != reliable_runner.REPAIR_SCHEMA
        or result.get("exact") != reliable_runner.REPAIR_EXACT
        or result.get("status") != "R1B_REPAIR_REAL_H_NOT_END_TO_END_READY"
        or not isinstance(memberships, list)
        or not isinstance(real_h, list)
        or len(memberships) != 80
        or len(real_h) != 80
        or _identities(memberships) != expected
        or _identities(real_h) != expected
    ):
        raise ValueError("R1B repair callback identity/status/order differs")
    return result


def _render_callback_inputs(r0_inputs: Sequence[Any]) -> tuple[CallbackInput, ...]:
    by_unit = {item.unit_id: item for item in r0_inputs}
    by_condition = {spec.condition_id: spec for spec in R1A_CORE_CONDITIONS}
    output = []
    for condition_id, unit_id in CALLBACK_ROSTER:
        try:
            item = by_unit[unit_id]
            spec = by_condition[condition_id]
            arms = {}
            for name, path in (("u", item.u_path), ("g", item.g_path), ("cg", item.cg_path)):
                with Image.open(path) as source:
                    source.load()
                    arms[name] = render_r1a_attack(source, spec)
            output.append(CallbackInput(condition_id, unit_id, arms["u"], arms["g"], arms["cg"]))
        except Exception as error:
            output.append(CallbackInput(
                condition_id, unit_id, None, None, None,
                (f"attack_render:{type(error).__name__}",),
            ))
    return tuple(output)


def _callback_record(
    item: CallbackInput,
    *,
    scorer: Callable[[Image.Image], ContentScore],
    detector: Callable[[Image.Image], GeometryEstimate],
):
    if item.errors or item.u is None or item.g is None or item.cg is None:
        return {
            "condition_id": item.condition_id, "unit_id": item.unit_id,
            "attempted": False, "route": "ERROR", "runtime": None,
            "score_snapshots": [],
            "call_counts": {"score_rgb": 0, "content_scorer": 0,
                            "detect_geometry": 0, "paired_null_rectifications": 0},
            "final_negative_false_positive": None,
            "errors": list(item.errors or ("callback_input:missing_rgb",)),
            "operational_interruption": True,
        }
    state: dict[str, Any] = {
        "raw_h0": None, "snapshots": [], "operational": None,
        "score_rgb": 0, "content_scorer": 0, "detect_geometry": 0,
        "paired_null_rectifications": 0,
    }

    def score_rgb(candidate_rgb: Image.Image) -> float:
        phase = state["score_rgb"]
        state["score_rgb"] += 1
        try:
            if phase == 0:
                u_rgb, g_rgb = item.u, item.g
            elif phase == 1 and state["raw_h0"] is not None:
                u_rgb = rectify_attacked_rgb(item.u, state["raw_h0"])
                state["paired_null_rectifications"] += 1
                g_rgb = rectify_attacked_rgb(item.g, state["raw_h0"])
                state["paired_null_rectifications"] += 1
            else:
                raise RuntimeError("bound score callback phase differs")

            def content_score(rgb: Image.Image) -> ContentScore:
                state["content_scorer"] += 1
                return scorer(rgb)

            scores = scored_triplet(
                u=content_score(u_rgb),
                g=content_score(g_rgb),
                cg=content_score(candidate_rgb),
            )
            state["snapshots"].append(scores)
            return scores.positive_cg_vs_g.margin
        except Exception as error:
            state["operational"] = f"content_score:{type(error).__name__}"
            raise

    def detect_geometry(candidate_rgb: Image.Image) -> GeometryEstimate:
        state["detect_geometry"] += 1
        try:
            geometry = detector(candidate_rgb)
        except Exception as error:
            state["operational"] = f"geometry_detect:{type(error).__name__}"
            raise
        if isinstance(geometry, GeometryEstimate):
            state["raw_h0"] = geometry.homography_observed_to_canonical
        return geometry

    runtime = detect_direct_and_recover(
        item.cg, score_rgb=score_rgb, detect_geometry=detect_geometry
    )
    snapshots = tuple(state["snapshots"])
    final_snapshot = None
    if runtime.route in ("DIRECT_POSITIVE", "DIRECT_NEGATIVE") and snapshots:
        final_snapshot = snapshots[0]
    elif runtime.route == "BOUNDARY" and runtime.post_score is not None and len(snapshots) >= 2:
        final_snapshot = snapshots[1]
    final_fp = (
        final_snapshot.negative_g_vs_u.positive if final_snapshot is not None else None
    )
    errors = tuple(item.errors) + ((runtime.error,) if runtime.error else ())
    return {
        "condition_id": item.condition_id,
        "unit_id": item.unit_id,
        "attempted": True,
        "route": runtime.route,
        "runtime": _jsonable(runtime),
        "score_snapshots": _jsonable(snapshots),
        "call_counts": {
            "score_rgb": state["score_rgb"],
            "content_scorer": state["content_scorer"],
            "detect_geometry": state["detect_geometry"],
            "paired_null_rectifications": state["paired_null_rectifications"],
            "candidate_rectifications": 1 if runtime.recovered else 0,
        },
        "same_bound_score_callback_pre_post": True,
        "raw_h0_unmodified": True,
        "final_negative_false_positive": final_fp,
        "errors": list(errors),
        "operational_interruption": state["operational"] is not None,
    }


def _callback_records(
    inputs: Sequence[CallbackInput],
    *, scorer: Callable[[Image.Image], ContentScore],
    detector: Callable[[Image.Image], GeometryEstimate],
):
    if tuple((item.condition_id, item.unit_id) for item in inputs) != CALLBACK_ROSTER:
        raise ValueError("callback fixed-seven roster differs")
    return tuple(_callback_record(item, scorer=scorer, detector=detector) for item in inputs)


def _operational_callback_rows(
    error: BaseException, rendered: Sequence[CallbackInput] | None = None,
):
    category = f"setup:{type(error).__name__}"
    rendered_by = {
        (item.condition_id, item.unit_id): item for item in (rendered or ())
    }
    rows = []
    for condition_id, unit_id in CALLBACK_ROSTER:
        item = rendered_by.get((condition_id, unit_id))
        evidence = list(item.errors) if item is not None and item.errors else [category]
        rows.append({
            "condition_id": condition_id, "unit_id": unit_id,
            "attempted": False, "route": "ERROR", "runtime": None,
            "score_snapshots": [],
            "call_counts": {"score_rgb": 0, "content_scorer": 0,
                            "detect_geometry": 0, "paired_null_rectifications": 0},
            "final_negative_false_positive": None,
            "errors": evidence, "operational_interruption": True,
        })
    return tuple(rows)


def _setup_real_callbacks(repo_root: Path, checkpoint: Path):
    import torch
    from experiments import content_adaptive_engine as engine
    from experiments import run_content_chain as content_chain_runner
    from experiments import run_geometry_v7_r0 as r0_runner
    from cegwm.geometry_v7.syncseal import (
        SyncSealTorchScript, download_official_syncseal_torchscript,
    )
    from cegwm.protocol.content_chain import (
        CONTENT_CHAIN_PUBLIC_KEY_DIGEST, load_content_chain_contract,
    )
    from cegwm.runtime.content_weighted_joint_sd35 import (
        ContentCalibrationAssets, derive_stability_wrong_keys,
    )
    from cegwm.shared.keys import normalize_detection_key, public_key_digest

    key_text = os.environ.pop(engine.KEY_ENV, "")
    token = os.environ.pop(engine.TOKEN_ENV, "")
    try:
        if not key_text.strip():
            raise RuntimeError("CEG_WM_ROOT_KEY_is_required")
        if not token.strip():
            raise RuntimeError("HF_TOKEN_is_required")
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_required_for_geometry_v7_direct_callback")
        detection_key = normalize_detection_key(key_text)
        if public_key_digest(detection_key) != CONTENT_CHAIN_PUBLIC_KEY_DIGEST:
            raise RuntimeError("content chain public key identity differs")
        contract = load_content_chain_contract(repo_root)
        _pipeline, assets = content_chain_runner._load_pipeline_and_assets(
            contract.runtime_protocol.config["generation_runtime"]["model_id"], token
        )
        if not isinstance(assets, ContentCalibrationAssets):
            raise TypeError("real frozen content calibration assets are required")
        scorer = r0_runner._content_scorer(
            detection_key=detection_key,
            wrong_keys=derive_stability_wrong_keys(detection_key),
            assets=assets,
            contract=contract,
        )
        loaded = download_official_syncseal_torchscript(checkpoint)
        syncseal = SyncSealTorchScript.from_file(loaded, device="cuda")
        return scorer, syncseal.detect_geometry
    finally:
        key_text = ""
        token = ""
        if "detection_key" in locals():
            detection_key = b""


def _replay_payload(
    *, exact: str, roots: Sequence[Path], rows=(), aggregates=None,
    error: BaseException | None = None,
):
    return {
        "schema": REPLAY_SCHEMA, "stage": "Geometry-V7-Direct replay",
        "exact": exact, "status": OPERATIONAL_STATUS if error else REPLAY_STATUS,
        "scientific_status": "not_adjudicated", "claim_ceiling": CLAIM_CEILING,
        "data_used_for_development": True,
        "inputs": {
            "r1b_repair": {"exact": reliable_runner.REPAIR_EXACT, "root": str(roots[0])},
            "r2": {"exact": reliable_runner.R2_EXACT, "root": str(roots[1])},
            "advanced_r3": {"exact": reliable_runner.ADVANCED_EXACT, "root": str(roots[2])},
        },
        "fixed": {"conditions": list(R2_CONDITION_IDS), "units": 8, "rows": 80,
                  "main_conditions": list(MAIN_CONDITIONS)},
        "rows": list(rows), "aggregates": aggregates,
        "operational_error": None if error is None else type(error).__name__,
        "claims": {"science": False, "promotion": False, "method_complete": False},
    }


def _callback_payload(
    *, exact: str, r0_root: Path, r1a_root: Path, repair_root: Path, checkpoint: Path,
    rows: Sequence[Mapping[str, Any]], error: BaseException | None = None,
):
    operational = error is not None or any(row["operational_interruption"] for row in rows)
    return {
        "schema": CALLBACK_SCHEMA, "stage": "Geometry-V7-Direct fixed-seven callback",
        "exact": exact, "status": OPERATIONAL_STATUS if operational else CALLBACK_STATUS,
        "scientific_status": "not_adjudicated", "claim_ceiling": CLAIM_CEILING,
        "data_used_for_development": True,
        "inputs": {"r0": {"exact": R0_EXACT, "root": str(r0_root)},
                   "r1a": {"exact": R1A_EXACT, "root": str(r1a_root)},
                   "r1b_repair": {
                       "exact": reliable_runner.REPAIR_EXACT,
                       "root": str(repair_root),
                   }},
        "checkpoint": {"path": str(checkpoint), "official_url_only": True,
                       "sha_gate": False},
        "fixed_roster": [
            {"condition_id": condition, "unit_id": unit}
            for condition, unit in CALLBACK_ROSTER
        ],
        "rows": list(rows),
        "operational_error": None if error is None else type(error).__name__,
        "route": {"single_ordinary_rgb_runtime": True, "truth_runtime_input": False,
                  "stored_h_runtime_input": False, "geometry_positive_vote": False,
                  "retry_or_fallback": False, "method_complete": False},
    }


def _write_result(root: Path, payload: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    with (root / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def _run_replay(args, exact: str):
    roots = tuple(Path(value).resolve() for value in (
        args.r1b_repair_root, args.r2_root, args.advanced_r3_root
    ))
    try:
        repair, r2, advanced = reliable_runner._validate_inputs(*roots)
        rows = _replay_rows(repair, r2, advanced)
        aggregates = _aggregate_replay(rows)
        return _replay_payload(exact=exact, roots=roots, rows=rows, aggregates=aggregates)
    except Exception as error:
        return _replay_payload(exact=exact, roots=roots, error=error)


def _run_callback(args, exact: str):
    from experiments import run_geometry_v7_r1b as repair_runner

    r0_root = Path(args.r0_artifact_root).resolve()
    r1a_root = Path(args.r1a_artifact_root).resolve()
    repair_root = Path(args.r1b_repair_root).resolve()
    checkpoint = Path(args.syncseal_checkpoint).resolve()
    rendered = None
    try:
        _validate_callback_repair(repair_root)
        inputs = repair_runner._load_r0_inputs(Path(args.repo_root).resolve(), r0_root)
        roster = tuple(item.unit_id for item in inputs)
        _validate_r1a_identity(Path(args.repo_root).resolve(), r1a_root, roster)
        rendered = _render_callback_inputs(inputs)
        scorer, detector = _setup_real_callbacks(Path(args.repo_root).resolve(), checkpoint)
        rows = _callback_records(rendered, scorer=scorer, detector=detector)
        return _callback_payload(
            exact=exact, r0_root=r0_root, r1a_root=r1a_root, repair_root=repair_root,
            checkpoint=checkpoint, rows=rows,
        )
    except Exception as error:
        rows = _operational_callback_rows(error, rendered)
        return _callback_payload(
            exact=exact, r0_root=r0_root, r1a_root=r1a_root, repair_root=repair_root,
            checkpoint=checkpoint, rows=rows, error=error,
        )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "callback"), required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--r1b-repair-root")
    parser.add_argument("--r2-root")
    parser.add_argument("--advanced-r3-root")
    parser.add_argument("--r0-artifact-root")
    parser.add_argument("--r1a-artifact-root")
    parser.add_argument("--syncseal-checkpoint")
    return parser


def _required(args, names: Sequence[str]) -> None:
    if any(not getattr(args, name, None) for name in names):
        raise ValueError(f"{args.mode} mode lacks required arguments")


def execute(args) -> Mapping[str, Any]:
    result = Path(args.result_dir).resolve()
    if result.exists():
        raise FileExistsError("Geometry-V7-Direct result directory must be create-only")
    exact = reliable_runner._git_exact(Path(args.repo_root).resolve(), args.expected_exact)
    if args.mode == "replay":
        _required(args, ("r1b_repair_root", "r2_root", "advanced_r3_root"))
        payload = _run_replay(args, exact)
    else:
        _required(args, (
            "r0_artifact_root", "r1a_artifact_root", "r1b_repair_root",
            "syncseal_checkpoint",
        ))
        payload = _run_callback(args, exact)
    _write_result(result, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = execute(args)
    except Exception as error:
        print(json.dumps({"status": "RUNNER_STOPPED_BEFORE_PACKAGE",
                          "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
        return 2
    print(json.dumps({"status": payload["status"], "result_dir": args.result_dir}, sort_keys=True))
    return 0 if payload["status"] in (REPLAY_STATUS, CALLBACK_STATUS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
