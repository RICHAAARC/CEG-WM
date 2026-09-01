"""Run Geometry-V7 R1B on fixed accepted R0/R1A artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import math
import os
from importlib import metadata
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

from PIL import Image
import torch

from experiments import content_adaptive_engine as engine
from experiments import run_content_chain as content_chain_runner
from experiments import run_geometry_v7_r0 as r0_runner
from cegwm.geometry_v7.r0 import ContentScore, R0Arm
from cegwm.geometry_v7.r1a import R1A_CORE_CONDITIONS, render_r1a_attack
from cegwm.geometry_v7.r1b import (
    R1B_CLAIM_CEILING,
    R1B_FIXED_UNIT_COUNT,
    R1B_INSUFFICIENT_GEOMETRY_NECESSITY,
    R1B_LAMBDA_GRID,
    R1B_OPERATIONAL_FAILURE,
    R1B_TRUTH_UTILITY_AND_NONZERO_EPSILON_PASSED,
    R1BEvaluation,
    R1BLambdaUnitRecord,
    R1BMembership,
    R1BPreUnitRecord,
    R1BScoredTriplet,
    controlled_homography,
    evaluate_lambda_unit,
    evaluate_r1b,
    freeze_pre_recovery_record,
    rectify_attacked_rgb,
    scored_triplet,
)
from cegwm.protocol.content_chain import (
    CONTENT_CHAIN_PUBLIC_KEY_DIGEST,
    load_content_chain_contract,
)
from cegwm.runtime.content_weighted_joint_sd35 import (
    ContentCalibrationAssets,
    derive_stability_wrong_keys,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest


R0_PRODUCER_EXACT = "4f0bf1560805672f786dc86dd50d793aec18aae7"
R0_REQUIRED_STATUS = "PAIRED_COMPATIBILITY_CANARY_PASSED"
R0_SELECTED_MULTIPLIER = 0.75
R1A_PRODUCER_EXACT = "ac590330e91aacf4b3283df1e94572a0e4f983a0"
R1A_REQUIRED_STATUS = "R1A_BLOCKING_METHOD_CANARY_PASSED"
RESULT_SCHEMA = "geometry_v7_r1b_result_v1"


@dataclass(frozen=True, slots=True)
class R0R1BInput:
    unit_id: str
    u_path: Path
    g_path: Path
    cg_path: Path
    u_relative_path: str
    g_relative_path: str
    cg_relative_path: str
    clean_scores: R1BScoredTriplet
    clean_score: float


@dataclass(frozen=True, slots=True)
class AttackedTriplet:
    unit_id: str
    condition_id: str
    u: Image.Image
    g: Image.Image
    cg: Image.Image


ContentScorer = Callable[[Image.Image], ContentScore]


def _dependency_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not_installed"


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


def _git_exact(repo_root: Path, expected_exact: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected exact must be a lowercase 40-character revision")
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if exact != expected_exact:
        raise RuntimeError("resolved revision differs from approved execution exact")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("execution checkout must be clean")
    return exact


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} result.json must be readable UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} result.json must contain an object")
    return result


def _validated_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
                raise ValueError("R1B input must be an RGB 512x512 PNG")
            image.verify()
    except (OSError, ValueError) as error:
        raise ValueError("R1B input must be an RGB 512x512 PNG") from error


def _relative_png(root: Path, value: object) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ValueError("R1B image member must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("R1B image member must stay inside the artifact root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("R1B image member escaped the artifact root") from error
    if not path.is_file():
        raise ValueError("R1B image member is absent")
    _validated_png(path)
    return path, relative.as_posix()


def _content_score(payload: object) -> ContentScore:
    if not isinstance(payload, Mapping):
        raise ValueError("R0 clean content raw score is absent")
    required = (
        "lf",
        "hf",
        "weighted_joint",
        "wrong_key_lf",
        "wrong_key_hf",
        "wrong_key_weighted_joint",
        "gate_a_margin",
    )
    if len(payload) != len(required) or set(payload) != set(required):
        raise ValueError("R0 clean content raw score identity/order differs")
    try:
        score = ContentScore(
            float(payload["lf"]),
            float(payload["hf"]),
            float(payload["weighted_joint"]),
            tuple(float(value) for value in payload["wrong_key_lf"]),
            tuple(float(value) for value in payload["wrong_key_hf"]),
            tuple(float(value) for value in payload["wrong_key_weighted_joint"]),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("R0 clean content raw score is malformed") from error
    if not math.isclose(
        float(payload["gate_a_margin"]),
        score.gate_a_margin,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("R0 clean Gate A margin differs from raw scores")
    return score


def _decision_matches(payload: object, expected: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return (
        len(payload) == 5
        and set(payload)
        == {
            "paired_null_arm",
            "gate_a_margin",
            "gate_b_margin",
            "margin",
            "positive",
        }
        and payload["paired_null_arm"] == expected.paired_null_arm.value
        and payload["gate_a_margin"] == expected.gate_a_margin
        and payload["gate_b_margin"] == expected.gate_b_margin
        and payload["margin"] == expected.margin
        and payload["positive"] is expected.positive
    )


def _load_r0_inputs(
    repo_root: Path, artifact_root: Path
) -> tuple[R0R1BInput, ...]:
    root = artifact_root.resolve()
    if not root.is_dir():
        raise ValueError("R0 artifact root must be an existing directory")
    result = _read_json(root / "result.json", "R0")
    contract = load_content_chain_contract(repo_root)
    roster = tuple(unit.unit_id for unit in contract.evaluation_roster)
    selection = result.get("selection")
    rosters = result.get("rosters")
    aggregate = result.get("evaluation_aggregate")
    if (
        result.get("exact") != R0_PRODUCER_EXACT
        or result.get("status") != R0_REQUIRED_STATUS
        or not isinstance(selection, Mapping)
        or selection.get("selected_residual_strength_multiplier")
        != R0_SELECTED_MULTIPLIER
        or not isinstance(rosters, Mapping)
        or tuple(rosters.get("evaluation", ())) != roster
        or not isinstance(aggregate, Mapping)
        or aggregate.get("stage") != "evaluation_fixed_8"
        or tuple(aggregate.get("roster", ())) != roster
        or aggregate.get("residual_strength_multiplier")
        != R0_SELECTED_MULTIPLIER
        or aggregate.get("carrier_compatibility_passed") is not True
    ):
        raise ValueError("R0 artifact identity, status, multiplier, or roster differs")
    raw_records = result.get("raw_unit_records")
    if not isinstance(raw_records, list):
        raise ValueError("R0 raw unit records are absent")
    evaluation = tuple(
        record
        for record in raw_records
        if isinstance(record, Mapping) and record.get("stage") == "evaluation"
    )
    if (
        len(evaluation) != R1B_FIXED_UNIT_COUNT
        or tuple(record.get("unit_id") for record in evaluation) != roster
        or any(
            record.get("residual_strength_multiplier") != R0_SELECTED_MULTIPLIER
            for record in evaluation
        )
    ):
        raise ValueError("R0 evaluation records differ from the fixed ordered roster")
    inputs: list[R0R1BInput] = []
    allowed = {
        R0Arm.U.value: "u",
        R0Arm.G.value: "g",
        R0Arm.CG.value: "cg",
    }
    for unit_id, record in zip(roster, evaluation, strict=True):
        arms = record.get("arms")
        if not isinstance(arms, list):
            raise ValueError("R0 evaluation arm records are absent")
        selected: dict[str, Mapping[str, Any]] = {}
        for arm_value in allowed:
            matches = tuple(
                arm
                for arm in arms
                if isinstance(arm, Mapping) and arm.get("arm") == arm_value
            )
            if len(matches) != 1 or matches[0].get("errors") not in ([], ()):
                raise ValueError("R0 U/G/CG arm is missing, duplicated, or failed")
            selected[arm_value] = matches[0]
        parsed_paths: dict[str, tuple[Path, str]] = {
            name: _relative_png(root, selected[arm_value].get("image_file"))
            for arm_value, name in allowed.items()
        }
        u_score = _content_score(selected[R0Arm.U.value].get("content"))
        g_score = _content_score(selected[R0Arm.G.value].get("content"))
        cg_score = _content_score(selected[R0Arm.CG.value].get("content"))
        clean = scored_triplet(u=u_score, g=g_score, cg=cg_score)
        if (
            not _decision_matches(
                selected[R0Arm.CG.value].get("paired_content_decision"),
                clean.positive_cg_vs_g,
            )
            or not _decision_matches(
                selected[R0Arm.G.value].get("paired_content_decision"),
                clean.negative_g_vs_u,
            )
            or not clean.positive_cg_vs_g.positive
            or clean.positive_cg_vs_g.margin <= 0.0
        ):
            raise ValueError("R0 clean CG-vs-G decision or G-vs-U control differs")
        inputs.append(
            R0R1BInput(
                unit_id,
                parsed_paths["u"][0],
                parsed_paths["g"][0],
                parsed_paths["cg"][0],
                parsed_paths["u"][1],
                parsed_paths["g"][1],
                parsed_paths["cg"][1],
                clean,
                clean.positive_cg_vs_g.margin,
            )
        )
    return tuple(inputs)


def _validate_r1a_artifact(
    repo_root: Path, artifact_root: Path, ordered_roster: Sequence[str]
) -> None:
    root = artifact_root.resolve()
    if not root.is_dir():
        raise ValueError("R1A artifact root must be an existing directory")
    result = _read_json(root / "result.json", "R1A")
    roster = tuple(ordered_roster)
    r0_input = result.get("r0_input")
    fixed = result.get("fixed_counts")
    if (
        result.get("exact") != R1A_PRODUCER_EXACT
        or result.get("status") != R1A_REQUIRED_STATUS
        or result.get("blocking_method_canary_passed") is not True
        or not isinstance(r0_input, Mapping)
        or r0_input.get("producer_exact") != R0_PRODUCER_EXACT
        or r0_input.get("selected_residual_strength_multiplier")
        != R0_SELECTED_MULTIPLIER
        or tuple(
            item.get("unit_id")
            for item in r0_input.get("ordered_evaluation_cg_inputs", ())
            if isinstance(item, Mapping)
        )
        != roster
        or not isinstance(fixed, Mapping)
        or fixed.get("core_conditions") != 10
        or fixed.get("units_per_condition") != R1B_FIXED_UNIT_COUNT
        or fixed.get("records") != 104
    ):
        raise ValueError("R1A artifact identity, pass status, or fixed roster differs")
    expected_ids = tuple(spec.condition_id for spec in R1A_CORE_CONDITIONS)
    specs = result.get("condition_specs")
    if not isinstance(specs, list):
        raise ValueError("R1A condition specs are absent")
    core_specs = tuple(
        item
        for item in specs
        if isinstance(item, Mapping) and item.get("kind") == "core_nonidentity"
    )
    if tuple(item.get("condition_id") for item in core_specs) != expected_ids:
        raise ValueError("R1A ten-core condition identity/order differs")
    aggregates = result.get("condition_aggregates")
    if not isinstance(aggregates, list):
        raise ValueError("R1A condition aggregates are absent")
    core_aggregates = tuple(
        item
        for item in aggregates
        if isinstance(item, Mapping)
        and item.get("condition_kind") == "core_nonidentity"
    )
    if (
        tuple(item.get("condition_id") for item in core_aggregates) != expected_ids
        or any(
            tuple(item.get("roster", ())) != roster
            or item.get("denominator") != R1B_FIXED_UNIT_COUNT
            or item.get("passed") is not True
            for item in core_aggregates
        )
    ):
        raise ValueError("R1A core aggregates or fixed-eight identity differs")
    records = result.get("raw_records")
    if not isinstance(records, list) or len(records) != 104:
        raise ValueError("R1A raw records are absent")
    for condition_id in expected_ids:
        matches = tuple(
            item
            for item in records
            if isinstance(item, Mapping) and item.get("condition_id") == condition_id
        )
        if (
            len(matches) != R1B_FIXED_UNIT_COUNT
            or tuple(item.get("unit_id") for item in matches) != roster
        ):
            raise ValueError("R1A core raw records differ from fixed-eight identity")


def _score_triplet(scorer: ContentScorer, images: AttackedTriplet) -> R1BScoredTriplet:
    return scored_triplet(
        u=scorer(images.u),
        g=scorer(images.g),
        cg=scorer(images.cg),
    )


def _render_pre_attacks(
    inputs: Sequence[R0R1BInput],
) -> tuple[dict[tuple[str, str], AttackedTriplet], dict[tuple[str, str], str]]:
    rendered: dict[tuple[str, str], AttackedTriplet] = {}
    failures: dict[tuple[str, str], str] = {}
    for spec in R1A_CORE_CONDITIONS:
        for item in inputs:
            identity = (spec.condition_id, item.unit_id)
            try:
                arms: dict[str, Image.Image] = {}
                for name, path in (
                    ("u", item.u_path),
                    ("g", item.g_path),
                    ("cg", item.cg_path),
                ):
                    with Image.open(path) as source:
                        source.load()
                        arms[name] = render_r1a_attack(source, spec)
                rendered[identity] = AttackedTriplet(
                    item.unit_id,
                    spec.condition_id,
                    arms["u"],
                    arms["g"],
                    arms["cg"],
                )
            except Exception as error:
                failures[identity] = f"attack_render:{type(error).__name__}"
    return rendered, failures


def _pre_score_all(
    *,
    inputs: Sequence[R0R1BInput],
    rendered: Mapping[tuple[str, str], AttackedTriplet],
    render_failures: Mapping[tuple[str, str], str],
    scorer: ContentScorer,
) -> dict[str, tuple[R1BPreUnitRecord, ...]]:
    collected: dict[tuple[str, str], tuple[R1BScoredTriplet | None, str | None]] = {}
    by_unit = {item.unit_id: item for item in inputs}
    for spec in R1A_CORE_CONDITIONS:
        for unit_id in by_unit:
            identity = (spec.condition_id, unit_id)
            error = render_failures.get(identity)
            scores = None
            if error is None:
                try:
                    scores = _score_triplet(scorer, rendered[identity])
                except Exception as caught:
                    error = f"pre_content_score:{type(caught).__name__}"
            collected[identity] = (scores, error)

    # Freeze the complete 10x8 membership partition only after every pre score
    # attempt has finished.  No rectification is reachable before this pass.
    output: dict[str, tuple[R1BPreUnitRecord, ...]] = {}
    for spec in R1A_CORE_CONDITIONS:
        records: list[R1BPreUnitRecord] = []
        for unit_id in by_unit:
            scores, error = collected[(spec.condition_id, unit_id)]
            records.append(
                freeze_pre_recovery_record(
                    unit_id=unit_id,
                    spec=spec,
                    clean_score=by_unit[unit_id].clean_score,
                    scores=scores,
                    errors=() if error is None else (error,),
                )
            )
        output[spec.condition_id] = tuple(records)
    return output


def _setup_failure_pre_records(
    inputs: Sequence[R0R1BInput], error: BaseException
) -> dict[str, tuple[R1BPreUnitRecord, ...]]:
    failure = f"content_runtime_setup:{type(error).__name__}"
    return {
        spec.condition_id: tuple(
            freeze_pre_recovery_record(
                unit_id=item.unit_id,
                spec=spec,
                clean_score=item.clean_score,
                scores=None,
                errors=(failure,),
            )
            for item in inputs
        )
        for spec in R1A_CORE_CONDITIONS
    }


def _lambda_score_all(
    *,
    rendered: Mapping[tuple[str, str], AttackedTriplet],
    pre_by_condition: Mapping[str, Sequence[R1BPreUnitRecord]],
    scorer: ContentScorer,
) -> dict[str, dict[float, tuple[R1BLambdaUnitRecord, ...]]]:
    output: dict[str, dict[float, tuple[R1BLambdaUnitRecord, ...]]] = {}
    for spec in R1A_CORE_CONDITIONS:
        pre = tuple(pre_by_condition[spec.condition_id])
        if not any(
            record.membership in (
                R1BMembership.RECOVERY_NEGATIVE,
                R1BMembership.BOUNDARY,
            )
            for record in pre
        ):
            continue
        grid: dict[float, tuple[R1BLambdaUnitRecord, ...]] = {}
        for lambda_value in R1B_LAMBDA_GRID:
            records: list[R1BLambdaUnitRecord] = []
            for pre_record in pre:
                scores = None
                error = None
                if lambda_value == 1.0:
                    scores = pre_record.scores
                else:
                    try:
                        attacked = rendered[(spec.condition_id, pre_record.unit_id)]
                        homography = controlled_homography(spec, lambda_value)
                        rectified = AttackedTriplet(
                            pre_record.unit_id,
                            spec.condition_id,
                            rectify_attacked_rgb(attacked.u, homography),
                            rectify_attacked_rgb(attacked.g, homography),
                            rectify_attacked_rgb(attacked.cg, homography),
                        )
                        scores = _score_triplet(scorer, rectified)
                    except Exception as caught:
                        error = f"rectified_content_score:{type(caught).__name__}"
                records.append(
                    evaluate_lambda_unit(
                        pre_record=pre_record,
                        spec=spec,
                        lambda_value=lambda_value,
                        scores=scores,
                        errors=() if error is None else (error,),
                    )
                )
            grid[lambda_value] = tuple(records)
        output[spec.condition_id] = grid
    return output


def _has_operational_failure(
    pre: Mapping[str, Sequence[R1BPreUnitRecord]],
    lambdas: Mapping[str, Mapping[float, Sequence[R1BLambdaUnitRecord]]],
) -> bool:
    return any(record.errors for records in pre.values() for record in records) or any(
        record.errors
        for grid in lambdas.values()
        for records in grid.values()
        for record in records
    )


def _triplet_payload(scores: R1BScoredTriplet | None) -> object:
    return None if scores is None else _jsonable(scores)


def _result_payload(
    *,
    exact: str,
    r0_root: Path,
    r1a_root: Path,
    inputs: Sequence[R0R1BInput],
    pre: Mapping[str, Sequence[R1BPreUnitRecord]],
    lambdas: Mapping[str, Mapping[float, Sequence[R1BLambdaUnitRecord]]],
    evaluation: R1BEvaluation | None,
    setup_error: BaseException | None,
) -> dict[str, Any]:
    pre_payload = []
    for condition_id, records in pre.items():
        for record in records:
            pre_payload.append(
                {
                    "unit_id": record.unit_id,
                    "condition_id": condition_id,
                    "clean_score_from_accepted_r0": record.clean_score,
                    "pre_scores": _triplet_payload(record.scores),
                    "membership": None
                    if record.membership is None
                    else record.membership.value,
                    "errors": list(record.errors),
                }
            )
    lambda_payload = []
    for condition_id, grid in lambdas.items():
        for lambda_value, records in grid.items():
            for record in records:
                lambda_payload.append(
                    {
                        "unit_id": record.unit_id,
                        "condition_id": condition_id,
                        "lambda": lambda_value,
                        "epsilon_normalized": record.epsilon_normalized,
                        "epsilon_pixels": record.epsilon_pixels,
                        "scores": _triplet_payload(record.scores),
                        "positive_gate_a_delta": record.positive_gate_a_delta,
                        "positive_gate_b_delta": record.positive_gate_b_delta,
                        "positive_score_delta": record.positive_score_delta,
                        "gain": record.gain,
                        "improved": record.improved,
                        "recovered_negative": record.recovered_negative,
                        "decision_harm": record.decision_harm,
                        "observed_negative_false_positive": (
                            record.observed_negative_false_positive
                        ),
                        "errors": list(record.errors),
                    }
                )
    failures = [
        {
            "phase": "pre",
            "condition_id": item["condition_id"],
            "unit_id": item["unit_id"],
            "errors": item["errors"],
        }
        for item in pre_payload
        if item["errors"]
    ] + [
        {
            "phase": "lambda",
            "condition_id": item["condition_id"],
            "unit_id": item["unit_id"],
            "lambda": item["lambda"],
            "errors": item["errors"],
        }
        for item in lambda_payload
        if item["errors"]
    ]
    operational = setup_error is not None or _has_operational_failure(pre, lambdas)
    status = (
        R1B_OPERATIONAL_FAILURE
        if operational
        else evaluation.status
        if evaluation is not None
        else R1B_OPERATIONAL_FAILURE
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "claim_ceiling": R1B_CLAIM_CEILING,
        "scientific_status": "not_adjudicated",
        "exact": exact,
        "inputs": {
            "r0": {
                "producer_exact": R0_PRODUCER_EXACT,
                "artifact_root": str(r0_root),
                "status": R0_REQUIRED_STATUS,
                "selected_residual_strength_multiplier": R0_SELECTED_MULTIPLIER,
                "arms": ["U", "G", "CG"],
            },
            "r1a": {
                "producer_exact": R1A_PRODUCER_EXACT,
                "artifact_root": str(r1a_root),
                "status": R1A_REQUIRED_STATUS,
                "predicted_h_used_as_truth": False,
            },
            "ordered_roster": [item.unit_id for item in inputs],
            "ordered_images": [
                {
                    "unit_id": item.unit_id,
                    "U": item.u_relative_path,
                    "G": item.g_relative_path,
                    "CG": item.cg_relative_path,
                }
                for item in inputs
            ],
        },
        "fixed_counts": {
            "core_conditions": 10,
            "units_per_condition": 8,
            "pre_records": 80,
            "lambda_grid": list(R1B_LAMBDA_GRID),
            "lambda_records": len(lambda_payload),
        },
        "pre_recovery_partition_frozen_before_rectification": pre_payload,
        "lambda_records": lambda_payload,
        "condition_evaluations": []
        if evaluation is None
        else [_jsonable(item) for item in evaluation.conditions],
        "fixed_denominator_evaluation_status": None
        if evaluation is None
        else evaluation.status,
        "applicable_condition_count": 0
        if evaluation is None
        else evaluation.applicable_condition_count,
        "blocking_method_canary_passed": None
        if operational or evaluation is None
        else evaluation.blocking_method_canary_passed,
        "failures": failures,
        "failure_policy": (
            "all fixed predeclared units and scoring failures remain in their "
            "denominators; missing eligible gain is internal -inf but JSON null plus "
            "error; no retry, fallback, replacement, subset, threshold tuning, or "
            "lambda reselection"
        ),
        "route": {
            "content_scorer": (
                "unchanged blind_weighted_scores bound through the existing R0 "
                "content scorer with identical key, preprocessing, calibration asset, "
                "weighted-joint statistic, and strict tau=0 paired decision"
            ),
            "generation_invoked": False,
            "arms": "U/G/CG only; C is neither loaded nor scored",
            "attacks": "exact R1A ten core transforms on each U/G/CG",
            "rectification": (
                "truth-to-identity controlled H, Pillow bilinear black 512, one "
                "rectification resample; lambda=1 reuses cached pre scores"
            ),
            "negative_claim": (
                "observed paired G-vs-U false positives only on the fixed eight-unit "
                "roster; never a generalized or fixed-FPR claim"
            ),
        },
        "provenance": {
            "python": platform.python_version(),
            "dependencies_record_only": {
                name: _dependency_version(name)
                for name in ("torch", "Pillow", "diffusers", "transformers")
            },
            "asset_pipeline_loaded_for_scoring_assets_only": setup_error is None,
            "operational_setup_error_class": None
            if setup_error is None
            else type(setup_error).__name__,
        },
    }


def _write_result(result_root: Path, result: Mapping[str, Any]) -> None:
    payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with (result_root / "result.json").open("xb") as output:
        output.write(payload)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    r0_root = Path(args.r0_artifact_root).resolve()
    r1a_root = Path(args.r1a_artifact_root).resolve()
    result_root = Path(args.result_dir).resolve()
    if result_root.exists():
        raise FileExistsError("Geometry-V7 R1B result directory must be create-only")
    exact = _git_exact(repo_root, args.expected_exact)
    inputs = _load_r0_inputs(repo_root, r0_root)
    roster = tuple(item.unit_id for item in inputs)
    _validate_r1a_artifact(repo_root, r1a_root, roster)
    result_root.mkdir(parents=True, exist_ok=False)

    key_text = os.environ.pop(engine.KEY_ENV, "")
    token = os.environ.pop(engine.TOKEN_ENV, "")
    setup_error = None
    scorer = None
    try:
        if not key_text.strip():
            raise RuntimeError("CEG_WM_ROOT_KEY_is_required")
        if not token.strip():
            raise RuntimeError("HF_TOKEN_is_required")
        detection_key = normalize_detection_key(key_text)
        if public_key_digest(detection_key) != CONTENT_CHAIN_PUBLIC_KEY_DIGEST:
            raise RuntimeError("content chain public key identity differs")
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_required_for_real_geometry_v7_r1b")
        contract = load_content_chain_contract(repo_root)
        _pipeline, assets = content_chain_runner._load_pipeline_and_assets(
            contract.runtime_protocol.config["generation_runtime"]["model_id"],
            token,
        )
        if not isinstance(assets, ContentCalibrationAssets):
            raise TypeError("R1B requires real frozen content calibration assets")
        wrong_keys = derive_stability_wrong_keys(detection_key)
        scorer = r0_runner._content_scorer(
            detection_key=detection_key,
            wrong_keys=wrong_keys,
            assets=assets,
            contract=contract,
        )
    except Exception as error:
        setup_error = error
    finally:
        key_text = ""
        token = ""
        if "detection_key" in locals():
            detection_key = b""

    if setup_error is not None or scorer is None:
        pre = _setup_failure_pre_records(
            inputs,
            setup_error or RuntimeError("content scorer setup unavailable"),
        )
        return _result_payload(
            exact=exact,
            r0_root=r0_root,
            r1a_root=r1a_root,
            inputs=inputs,
            pre=pre,
            lambdas={},
            evaluation=None,
            setup_error=setup_error,
        )

    rendered, render_failures = _render_pre_attacks(inputs)
    pre = _pre_score_all(
        inputs=inputs,
        rendered=rendered,
        render_failures=render_failures,
        scorer=scorer,
    )
    if _has_operational_failure(pre, {}):
        return _result_payload(
            exact=exact,
            r0_root=r0_root,
            r1a_root=r1a_root,
            inputs=inputs,
            pre=pre,
            lambdas={},
            evaluation=None,
            setup_error=None,
        )
    all_empty = all(
        record.membership is R1BMembership.DAMAGE_ONLY
        for records in pre.values()
        for record in records
    )
    if all_empty:
        lambdas: dict[str, dict[float, tuple[R1BLambdaUnitRecord, ...]]] = {}
    else:
        lambdas = _lambda_score_all(
            rendered=rendered,
            pre_by_condition=pre,
            scorer=scorer,
        )
    evaluation = evaluate_r1b(
        pre_records_by_condition=pre,
        lambda_records_by_condition=lambdas,
        ordered_roster=roster,
    )
    return _result_payload(
        exact=exact,
        r0_root=r0_root,
        r1a_root=r1a_root,
        inputs=inputs,
        pre=pre,
        lambdas=lambdas,
        evaluation=evaluation,
        setup_error=None,
    )


def execute(args: argparse.Namespace) -> int:
    result_root = Path(args.result_dir).resolve()
    preexisting = result_root.exists()
    try:
        result = _run(args)
        _write_result(result_root, result)
    except BaseException:
        if not preexisting and result_root.is_dir():
            shutil.rmtree(result_root)
        raise
    print(
        "CEGWM_GEOMETRY_V7_R1B "
        + json.dumps(
            {
                "status": result["status"],
                "claim_ceiling": result["claim_ceiling"],
                "exact": result["exact"],
                "result_dir": str(result_root),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return (
        0
        if result["status"] == R1B_TRUTH_UTILITY_AND_NONZERO_EPSILON_PASSED
        else 2
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--r0-artifact-root", required=True)
    parser.add_argument("--r1a-artifact-root", required=True)
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
