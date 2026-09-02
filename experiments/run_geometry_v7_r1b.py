"""Run the Geometry-V7 R1B-R1 real-H and directional-pixel method repair."""

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
from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED, Matrix3x3
from cegwm.geometry_v7.r0 import ContentScore, R0Arm
from cegwm.geometry_v7.r1a import (
    R1A_CORE_CONDITIONS,
    apply_homography,
    corner_rmse,
    render_r1a_attack,
    truth_correspondences,
)
from cegwm.geometry_v7.r1b import (
    R1B_OPERATIONAL_FAILURE,
    R1B_REPAIR_CLAIM_CEILING,
    R1B_REPAIR_METHOD_PASSED,
    R1B_REPAIR_PIXEL_GRID,
    R1BEvaluation,
    R1BLambdaUnitRecord,
    R1BMembership,
    R1BPreUnitRecord,
    R1BRepairEvaluation,
    R1BRepairPointRecord,
    R1BScoredTriplet,
    R1BStoredPrediction,
    directional_homography_for_pixels,
    evaluate_lambda_unit,
    evaluate_repair_point_unit,
    evaluate_r1b_repair,
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
OLD_R1B_PRODUCER_EXACT = "79631ce179b335d4c445a80b07d5bde254756260"
OLD_R1B_SCHEMA = "geometry_v7_r1b_result_v1"
RESULT_SCHEMA = "geometry_v7_r1b_repair_result_v1"
STAGE_LABEL = "R1B-R1 method repair"


@dataclass(frozen=True, slots=True)
class R0R1BInput:
    unit_id: str
    u_path: Path
    g_path: Path
    cg_path: Path
    u_relative_path: str
    g_relative_path: str
    cg_relative_path: str


@dataclass(frozen=True, slots=True)
class AttackedTriplet:
    unit_id: str
    condition_id: str
    u: Image.Image
    g: Image.Image
    cg: Image.Image


@dataclass(frozen=True, slots=True)
class OldR1BArtifact:
    recorded_status: str
    pre_by_condition: Mapping[str, tuple[R1BPreUnitRecord, ...]]
    zero_by_condition: Mapping[str, tuple[R1BLambdaUnitRecord, ...]]


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
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if exact != expected_exact:
        raise RuntimeError("resolved revision differs from approved execution exact")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True,
        capture_output=True, text=True,
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
                raise ValueError("R1B repair input must be RGB 512x512 PNG")
            image.verify()
    except (OSError, ValueError) as error:
        raise ValueError("R1B repair input must be RGB 512x512 PNG") from error


def _relative_png(root: Path, value: object) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ValueError("R1B repair image member must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("R1B repair image member must stay inside the artifact root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("R1B repair image member escaped the artifact root") from error
    if not path.is_file():
        raise ValueError("R1B repair image member is absent")
    _validated_png(path)
    return path, relative.as_posix()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be a finite real scalar")
    return scalar


def _corners(value: object, label: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must be finite 4x2 points")
    try:
        points = tuple(tuple(_finite(axis, label) for axis in point) for point in value)
    except TypeError as error:
        raise ValueError(f"{label} must be finite 4x2 points") from error
    if any(len(point) != 2 for point in points):
        raise ValueError(f"{label} must be finite 4x2 points")
    return points


def _matrix(value: object, label: str) -> Matrix3x3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a finite invertible 3x3 matrix")
    try:
        matrix = tuple(tuple(_finite(axis, label) for axis in row) for row in value)
    except TypeError as error:
        raise ValueError(f"{label} must be a finite invertible 3x3 matrix") from error
    if any(len(row) != 3 for row in matrix):
        raise ValueError(f"{label} must be a finite invertible 3x3 matrix")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError(f"{label} must be a finite invertible 3x3 matrix")
    return matrix


def _content_score(payload: object, *, has_gate_a: bool) -> ContentScore:
    if not isinstance(payload, Mapping):
        raise ValueError("content raw score is absent")
    fields = {
        "lf", "hf", "weighted_joint", "wrong_key_lf", "wrong_key_hf",
        "wrong_key_weighted_joint",
    }
    expected = fields | ({"gate_a_margin"} if has_gate_a else set())
    if len(payload) != len(expected) or set(payload) != expected:
        raise ValueError("content raw score fields differ")
    try:
        score = ContentScore(
            _finite(payload["lf"], "LF score"),
            _finite(payload["hf"], "HF score"),
            _finite(payload["weighted_joint"], "weighted-joint score"),
            tuple(_finite(value, "wrong-key LF") for value in payload["wrong_key_lf"]),
            tuple(_finite(value, "wrong-key HF") for value in payload["wrong_key_hf"]),
            tuple(
                _finite(value, "wrong-key weighted-joint")
                for value in payload["wrong_key_weighted_joint"]
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("content raw score is malformed") from error
    if has_gate_a and _finite(payload["gate_a_margin"], "Gate A") != score.gate_a_margin:
        raise ValueError("stored Gate A differs from raw scores")
    return score


def _decision_matches(payload: object, expected: Any) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and len(payload) == 5
        and set(payload) == {
            "paired_null_arm", "gate_a_margin", "gate_b_margin", "margin", "positive"
        }
        and payload["paired_null_arm"] == expected.paired_null_arm.value
        and payload["gate_a_margin"] == expected.gate_a_margin
        and payload["gate_b_margin"] == expected.gate_b_margin
        and payload["margin"] == expected.margin
        and payload["positive"] is expected.positive
    )


def _triplet(payload: object) -> R1BScoredTriplet:
    if not isinstance(payload, Mapping) or set(payload) != {
        "u", "g", "cg", "positive_cg_vs_g", "negative_g_vs_u"
    }:
        raise ValueError("old R1B scored triplet fields differ")
    u = _content_score(payload["u"], has_gate_a=False)
    g = _content_score(payload["g"], has_gate_a=False)
    cg = _content_score(payload["cg"], has_gate_a=False)
    parsed = scored_triplet(u=u, g=g, cg=cg)
    if (
        not _decision_matches(payload["positive_cg_vs_g"], parsed.positive_cg_vs_g)
        or not _decision_matches(payload["negative_g_vs_u"], parsed.negative_g_vs_u)
    ):
        raise ValueError("old R1B paired decision differs from raw scores")
    return parsed


def _load_r0_inputs(repo_root: Path, artifact_root: Path) -> tuple[R0R1BInput, ...]:
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
        or selection.get("selected_residual_strength_multiplier") != R0_SELECTED_MULTIPLIER
        or not isinstance(rosters, Mapping)
        or tuple(rosters.get("evaluation", ())) != roster
        or not isinstance(aggregate, Mapping)
        or aggregate.get("stage") != "evaluation_fixed_8"
        or tuple(aggregate.get("roster", ())) != roster
        or aggregate.get("residual_strength_multiplier") != R0_SELECTED_MULTIPLIER
        or aggregate.get("carrier_compatibility_passed") is not True
    ):
        raise ValueError("R0 artifact identity, status, multiplier, or roster differs")
    raw = result.get("raw_unit_records")
    if not isinstance(raw, list):
        raise ValueError("R0 raw records are absent")
    evaluation = tuple(
        item for item in raw
        if isinstance(item, Mapping) and item.get("stage") == "evaluation"
    )
    if (
        len(evaluation) != 8
        or tuple(item.get("unit_id") for item in evaluation) != roster
        or any(item.get("residual_strength_multiplier") != 0.75 for item in evaluation)
    ):
        raise ValueError("R0 evaluation records differ from fixed ordered roster")
    inputs: list[R0R1BInput] = []
    allowed = (R0Arm.U.value, R0Arm.G.value, R0Arm.CG.value)
    for unit_id, record in zip(roster, evaluation, strict=True):
        arms = record.get("arms")
        if not isinstance(arms, list):
            raise ValueError("R0 arm records are absent")
        selected = {}
        for arm_value in allowed:
            matches = tuple(
                arm for arm in arms
                if isinstance(arm, Mapping) and arm.get("arm") == arm_value
            )
            if len(matches) != 1 or matches[0].get("errors") not in ([], ()):
                raise ValueError("R0 U/G/CG arm is missing, duplicated, or failed")
            selected[arm_value] = matches[0]
        paths = {
            arm: _relative_png(root, selected[arm].get("image_file"))
            for arm in allowed
        }
        u_score = _content_score(
            selected[R0Arm.U.value].get("content"), has_gate_a=True
        )
        g_score = _content_score(
            selected[R0Arm.G.value].get("content"), has_gate_a=True
        )
        cg_score = _content_score(
            selected[R0Arm.CG.value].get("content"), has_gate_a=True
        )
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
        inputs.append(R0R1BInput(
            unit_id,
            paths[R0Arm.U.value][0], paths[R0Arm.G.value][0], paths[R0Arm.CG.value][0],
            paths[R0Arm.U.value][1], paths[R0Arm.G.value][1], paths[R0Arm.CG.value][1],
        ))
    return tuple(inputs)


def _validate_r1a_artifact(
    repo_root: Path, artifact_root: Path, ordered_roster: Sequence[str]
) -> Mapping[tuple[str, str], R1BStoredPrediction]:
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
        or r0_input.get("selected_residual_strength_multiplier") != 0.75
        or tuple(
            item.get("unit_id")
            for item in r0_input.get("ordered_evaluation_cg_inputs", ())
            if isinstance(item, Mapping)
        ) != roster
        or not isinstance(fixed, Mapping)
        or fixed.get("core_conditions") != 10
        or fixed.get("units_per_condition") != 8
        or fixed.get("records") != 104
    ):
        raise ValueError("R1A artifact identity, status, or fixed roster differs")
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
            or item.get("denominator") != 8
            or item.get("passed") is not True
            for item in core_aggregates
        )
    ):
        raise ValueError("R1A core aggregates or fixed-eight identity differs")
    raw = result.get("raw_records")
    if not isinstance(raw, list) or len(raw) != 104:
        raise ValueError("R1A raw record schema/count differs")
    predictions: dict[tuple[str, str], R1BStoredPrediction] = {}
    for spec in R1A_CORE_CONDITIONS:
        matches = tuple(
            item for item in raw
            if isinstance(item, Mapping) and item.get("condition_id") == spec.condition_id
        )
        if len(matches) != 8 or tuple(item.get("unit_id") for item in matches) != roster:
            raise ValueError("R1A core raw record order differs")
        for unit_id, item in zip(roster, matches, strict=True):
            if item.get("condition_kind") != "core_nonidentity":
                raise ValueError("R1A core raw record kind differs")
            truth = _corners(
                item.get("truth_observed_corners_in_canonical_normalized"),
                "R1A truth correspondences",
            )
            frozen_truth = truth_correspondences(spec)
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual_point, expected_point in zip(
                    truth, frozen_truth, strict=True
                )
                for actual, expected in zip(
                    actual_point, expected_point, strict=True
                )
            ):
                raise ValueError("R1A frozen truth correspondences differ")
            errors: list[str] = []
            geometry = item.get("geometry")
            predicted = None
            matrix = None
            prediction_rmse = None
            if not isinstance(geometry, Mapping):
                errors.append("stored_prediction:missing_geometry")
            else:
                try:
                    predicted = _corners(
                        geometry.get("observed_corners_in_canonical_normalized"),
                        "R1A predicted correspondences",
                    )
                except ValueError:
                    errors.append("stored_prediction:invalid_correspondences")
                try:
                    matrix = _matrix(
                        geometry.get("homography_observed_to_canonical"),
                        "R1A predicted homography",
                    )
                except ValueError:
                    errors.append("stored_prediction:invalid_homography")
                if geometry.get("legal") is not True or geometry.get("error") is not None:
                    errors.append("stored_prediction:geometry_not_legal")
            try:
                prediction_rmse = _finite(item.get("prediction_rmse"), "R1A e_pred")
                if prediction_rmse < 0.0:
                    raise ValueError("negative")
            except ValueError:
                prediction_rmse = None
                errors.append("stored_prediction:invalid_e_pred")
            if predicted is not None and matrix is not None:
                try:
                    mapped = apply_homography(matrix, CANONICAL_CORNERS_NORMALIZED)
                    if any(
                        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
                        for actual_point, expected_point in zip(mapped, predicted, strict=True)
                        for actual, expected in zip(actual_point, expected_point, strict=True)
                    ):
                        errors.append("stored_prediction:h_correspondence_mismatch")
                except ValueError:
                    errors.append("stored_prediction:invalid_homography")
            if predicted is not None and prediction_rmse is not None:
                expected_rmse = corner_rmse(predicted, truth)
                if not math.isclose(
                    prediction_rmse, expected_rmse, rel_tol=0.0, abs_tol=1e-12
                ):
                    errors.append("stored_prediction:e_pred_mismatch")
            predictions[(spec.condition_id, unit_id)] = R1BStoredPrediction(
                unit_id,
                spec.condition_id,
                truth,
                predicted,
                matrix,
                prediction_rmse,
                None if prediction_rmse is None else prediction_rmse * 511.0 / 2.0,
                tuple(dict.fromkeys(errors)),
            )
    return predictions


def _load_old_r1b_artifact(
    artifact_root: Path,
    ordered_roster: Sequence[str],
) -> OldR1BArtifact:
    root = artifact_root.resolve()
    if not root.is_dir():
        raise ValueError("old R1B artifact root must be an existing directory")
    result = _read_json(root / "result.json", "old R1B")
    status = result.get("status")
    if (
        result.get("schema") != OLD_R1B_SCHEMA
        or result.get("exact") != OLD_R1B_PRODUCER_EXACT
        or not isinstance(status, str)
        or status == R1B_OPERATIONAL_FAILURE
    ):
        raise ValueError("old R1B artifact schema/exact/completeness differs")
    roster = tuple(ordered_roster)
    raw_pre = result.get("pre_recovery_partition_frozen_before_rectification")
    if not isinstance(raw_pre, list) or len(raw_pre) != 80:
        raise ValueError("old R1B pre-record count differs")
    pre_by_condition: dict[str, tuple[R1BPreUnitRecord, ...]] = {}
    offset = 0
    for spec in R1A_CORE_CONDITIONS:
        records: list[R1BPreUnitRecord] = []
        for unit_id in roster:
            item = raw_pre[offset]
            offset += 1
            if (
                not isinstance(item, Mapping)
                or item.get("condition_id") != spec.condition_id
                or item.get("unit_id") != unit_id
                or item.get("errors") not in ([], ())
            ):
                raise ValueError("old R1B pre-record identity/order differs")
            clean = _finite(item.get("clean_score_from_accepted_r0"), "old s_clean")
            scores = _triplet(item.get("pre_scores"))
            expected = freeze_pre_recovery_record(
                unit_id=unit_id, spec=spec, clean_score=clean, scores=scores
            )
            if item.get("membership") != expected.membership.value:
                raise ValueError("old R1B frozen membership differs from its raw scores")
            records.append(expected)
        pre_by_condition[spec.condition_id] = tuple(records)
    evaluations = result.get("condition_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 10:
        raise ValueError("old R1B condition evaluation count differs")
    for spec, item in zip(R1A_CORE_CONDITIONS, evaluations, strict=True):
        pre = pre_by_condition[spec.condition_id]
        eligible = tuple(
            record.unit_id for record in pre
            if record.membership in (R1BMembership.RECOVERY_NEGATIVE, R1BMembership.BOUNDARY)
        )
        damage = tuple(
            record.unit_id for record in pre
            if record.membership is R1BMembership.DAMAGE_ONLY
        )
        if (
            not isinstance(item, Mapping)
            or item.get("condition_id") != spec.condition_id
            or tuple(item.get("roster", ())) != roster
            or tuple(item.get("eligible_roster", ())) != eligible
            or tuple(item.get("damage_only_roster", ())) != damage
        ):
            raise ValueError("old R1B condition membership/roster differs")
    raw_lambda = result.get("lambda_records")
    if not isinstance(raw_lambda, list):
        raise ValueError("old R1B lambda records are absent")
    zero_by_condition: dict[str, tuple[R1BLambdaUnitRecord, ...]] = {}
    for spec in R1A_CORE_CONDITIONS:
        pre = pre_by_condition[spec.condition_id]
        eligible = any(
            record.membership in (R1BMembership.RECOVERY_NEGATIVE, R1BMembership.BOUNDARY)
            for record in pre
        )
        matches = tuple(
            item for item in raw_lambda
            if isinstance(item, Mapping)
            and item.get("condition_id") == spec.condition_id
            and item.get("lambda") == 0.0
        )
        if not eligible:
            if matches:
                raise ValueError("old empty-E condition unexpectedly has lambda-zero records")
            continue
        if len(matches) != 8 or tuple(item.get("unit_id") for item in matches) != roster:
            raise ValueError("old applicable condition lacks fixed lambda-zero records")
        parsed: list[R1BLambdaUnitRecord] = []
        for pre_record, item in zip(pre, matches, strict=True):
            if item.get("errors") not in ([], ()):
                raise ValueError("old lambda-zero record is incomplete")
            scores = _triplet(item.get("scores"))
            expected = evaluate_lambda_unit(
                pre_record=pre_record,
                spec=spec,
                lambda_value=0.0,
                scores=scores,
            )
            numeric_fields = {
                "epsilon_normalized": expected.epsilon_normalized,
                "epsilon_pixels": expected.epsilon_pixels,
                "positive_gate_a_delta": expected.positive_gate_a_delta,
                "positive_gate_b_delta": expected.positive_gate_b_delta,
                "positive_score_delta": expected.positive_score_delta,
                "gain": expected.gain,
                "improved": expected.improved,
                "recovered_negative": expected.recovered_negative,
                "decision_harm": expected.decision_harm,
                "observed_negative_false_positive": expected.observed_negative_false_positive,
            }
            if any(item.get(name) != value for name, value in numeric_fields.items()):
                raise ValueError("old lambda-zero numeric record differs from raw scores")
            parsed.append(expected)
        zero_by_condition[spec.condition_id] = tuple(parsed)
    return OldR1BArtifact(status, pre_by_condition, zero_by_condition)


def _render_attacks(inputs: Sequence[R0R1BInput]) -> tuple[
    Mapping[tuple[str, str], AttackedTriplet], Mapping[tuple[str, str], str]
]:
    rendered = {}
    failures = {}
    for spec in R1A_CORE_CONDITIONS:
        for item in inputs:
            identity = (spec.condition_id, item.unit_id)
            try:
                arms = {}
                for name, path in (("u", item.u_path), ("g", item.g_path), ("cg", item.cg_path)):
                    with Image.open(path) as source:
                        source.load()
                        arms[name] = render_r1a_attack(source, spec)
                rendered[identity] = AttackedTriplet(
                    item.unit_id, spec.condition_id, arms["u"], arms["g"], arms["cg"]
                )
            except Exception as error:
                failures[identity] = f"attack_render:{type(error).__name__}"
    return rendered, failures


def _score_shared_h(
    attacked: AttackedTriplet,
    observed_to_canonical: Matrix3x3,
    scorer: ContentScorer,
) -> R1BScoredTriplet:
    """Use one CG-derived H for the paired U/G/CG rectifications."""

    u = rectify_attacked_rgb(attacked.u, observed_to_canonical)
    g = rectify_attacked_rgb(attacked.g, observed_to_canonical)
    cg = rectify_attacked_rgb(attacked.cg, observed_to_canonical)
    return scored_triplet(u=scorer(u), g=scorer(g), cg=scorer(cg))


def _real_h_records(
    *,
    pre: Mapping[str, Sequence[R1BPreUnitRecord]],
    predictions: Mapping[tuple[str, str], R1BStoredPrediction],
    rendered: Mapping[tuple[str, str], AttackedTriplet],
    render_failures: Mapping[tuple[str, str], str],
    scorer: ContentScorer,
) -> Mapping[str, tuple[R1BRepairPointRecord, ...]]:
    output = {}
    for spec in R1A_CORE_CONDITIONS:
        records = []
        for pre_record in pre[spec.condition_id]:
            identity = (spec.condition_id, pre_record.unit_id)
            prediction = predictions[identity]
            errors = list(prediction.errors)
            if identity in render_failures:
                errors.append(render_failures[identity])
            scores = None
            if not errors and prediction.predicted_h_observed_to_canonical is not None:
                try:
                    scores = _score_shared_h(
                        rendered[identity], prediction.predicted_h_observed_to_canonical, scorer
                    )
                except Exception as error:
                    errors.append(f"real_h_recovery:{type(error).__name__}")
            records.append(evaluate_repair_point_unit(
                pre_record=pre_record,
                point_kind="real_h",
                radius_pixels=None,
                scores=scores,
                errors=errors,
            ))
        output[spec.condition_id] = tuple(records)
    return output


def _fine_grid_records(
    *,
    pre: Mapping[str, Sequence[R1BPreUnitRecord]],
    old_zero: Mapping[str, Sequence[R1BLambdaUnitRecord]],
    predictions: Mapping[tuple[str, str], R1BStoredPrediction],
    rendered: Mapping[tuple[str, str], AttackedTriplet],
    render_failures: Mapping[tuple[str, str], str],
    scorer: ContentScorer,
) -> Mapping[str, Mapping[int, tuple[R1BRepairPointRecord, ...]]]:
    output = {}
    for spec in R1A_CORE_CONDITIONS:
        pre_records = tuple(pre[spec.condition_id])
        if not any(
            record.membership in (R1BMembership.RECOVERY_NEGATIVE, R1BMembership.BOUNDARY)
            for record in pre_records
        ):
            continue
        zero_records = tuple(old_zero[spec.condition_id])
        grid = {}
        for radius in R1B_REPAIR_PIXEL_GRID:
            records = []
            for index, pre_record in enumerate(pre_records):
                identity = (spec.condition_id, pre_record.unit_id)
                if radius == 0:
                    scores = zero_records[index].scores
                    errors = ()
                else:
                    scores = None
                    errors_list = []
                    prediction = predictions[identity]
                    if identity in render_failures:
                        errors_list.append(render_failures[identity])
                    try:
                        homography = directional_homography_for_pixels(prediction, radius)
                    except Exception as error:
                        errors_list.append(f"directional_h:{type(error).__name__}")
                    if not errors_list:
                        try:
                            scores = _score_shared_h(rendered[identity], homography, scorer)
                        except Exception as error:
                            errors_list.append(f"directional_recovery:{type(error).__name__}")
                    errors = tuple(errors_list)
                records.append(evaluate_repair_point_unit(
                    pre_record=pre_record,
                    point_kind="directional_pixel",
                    radius_pixels=radius,
                    scores=scores,
                    errors=errors,
                ))
            grid[radius] = tuple(records)
        output[spec.condition_id] = grid
    return output


def _operational_records(
    *,
    pre: Mapping[str, Sequence[R1BPreUnitRecord]],
    old_zero: Mapping[str, Sequence[R1BLambdaUnitRecord]],
    error: BaseException,
) -> tuple[
    Mapping[str, tuple[R1BRepairPointRecord, ...]],
    Mapping[str, Mapping[int, tuple[R1BRepairPointRecord, ...]]],
]:
    failure = f"content_runtime_setup:{type(error).__name__}"
    real = {}
    fine = {}
    for spec in R1A_CORE_CONDITIONS:
        pre_records = tuple(pre[spec.condition_id])
        real[spec.condition_id] = tuple(
            evaluate_repair_point_unit(
                pre_record=record, point_kind="real_h", radius_pixels=None,
                scores=None, errors=(failure,),
            )
            for record in pre_records
        )
        if spec.condition_id in old_zero:
            grid = {}
            for radius in R1B_REPAIR_PIXEL_GRID:
                grid[radius] = tuple(
                    evaluate_repair_point_unit(
                        pre_record=record,
                        point_kind="directional_pixel",
                        radius_pixels=radius,
                        scores=old_zero[spec.condition_id][index].scores if radius == 0 else None,
                        errors=() if radius == 0 else (failure,),
                    )
                    for index, record in enumerate(pre_records)
                )
            fine[spec.condition_id] = grid
    return real, fine


def _record_payload(record: R1BRepairPointRecord) -> dict[str, Any]:
    return {
        "unit_id": record.unit_id,
        "condition_id": record.condition_id,
        "point_kind": record.point_kind,
        "radius_pixels": record.radius_pixels,
        "scores": None if record.scores is None else _jsonable(record.scores),
        "positive_gate_a_delta": record.positive_gate_a_delta,
        "positive_gate_b_delta": record.positive_gate_b_delta,
        "positive_score_delta": record.positive_score_delta,
        "improved": record.improved,
        "recovered_negative": record.recovered_negative,
        "decision_harm": record.decision_harm,
        "observed_negative_false_positive": record.observed_negative_false_positive,
        "errors": list(record.errors),
    }


def _result_payload(
    *,
    exact: str,
    r0_root: Path,
    r1a_root: Path,
    old_r1b_root: Path,
    inputs: Sequence[R0R1BInput],
    old: OldR1BArtifact,
    predictions: Mapping[tuple[str, str], R1BStoredPrediction],
    real: Mapping[str, Sequence[R1BRepairPointRecord]],
    fine: Mapping[str, Mapping[int, Sequence[R1BRepairPointRecord]]],
    evaluation: R1BRepairEvaluation | None,
    setup_error: BaseException | None,
) -> dict[str, Any]:
    real_payload = [_record_payload(record) for records in real.values() for record in records]
    fine_payload = [
        _record_payload(record)
        for grid in fine.values() for records in grid.values() for record in records
    ]
    failures = [
        {
            "point_kind": item["point_kind"],
            "radius_pixels": item["radius_pixels"],
            "condition_id": item["condition_id"],
            "unit_id": item["unit_id"],
            "errors": item["errors"],
        }
        for item in (*real_payload, *fine_payload)
        if item["errors"]
    ]
    operational = setup_error is not None
    status = R1B_OPERATIONAL_FAILURE if operational else evaluation.status
    pre_payload = [
        {
            "unit_id": record.unit_id,
            "condition_id": condition_id,
            "clean_score_from_old_r1b": record.clean_score,
            "pre_scores_from_old_r1b": _jsonable(record.scores),
            "membership_from_old_r1b": record.membership.value,
        }
        for condition_id, records in old.pre_by_condition.items()
        for record in records
    ]
    return {
        "schema": RESULT_SCHEMA,
        "stage": STAGE_LABEL,
        "status": status,
        "real_h_status": None if operational else evaluation.real_h_status,
        "fine_grid_status": None if operational else evaluation.fine_grid_status,
        "r2_candidate": None if operational else evaluation.r2_candidate,
        "claim_ceiling": R1B_REPAIR_CLAIM_CEILING,
        "scientific_status": "not_adjudicated",
        "exact": exact,
        "inputs": {
            "r0": {"producer_exact": R0_PRODUCER_EXACT, "artifact_root": str(r0_root)},
            "r1a": {"producer_exact": R1A_PRODUCER_EXACT, "artifact_root": str(r1a_root)},
            "old_r1b": {
                "producer_exact": OLD_R1B_PRODUCER_EXACT,
                "artifact_root": str(old_r1b_root),
                "recorded_status_no_new_gate": old.recorded_status,
                "lambda_zero_raw_records_reused_without_rescore": True,
            },
            "ordered_roster": [item.unit_id for item in inputs],
            "arms": ["U", "G", "CG"],
        },
        "fixed": {
            "core_conditions": 10,
            "units_per_condition": 8,
            "pixel_grid": list(R1B_REPAIR_PIXEL_GRID),
            "real_h_records": len(real_payload),
            "fine_grid_records": len(fine_payload),
        },
        "frozen_old_membership_records": pre_payload,
        "stored_r1a_predictions": [_jsonable(item) for item in predictions.values()],
        "real_h_records": real_payload,
        "fine_grid_records": fine_payload,
        "condition_evaluations": [] if evaluation is None else [
            _jsonable(item) for item in evaluation.conditions
        ],
        "real_h_passed_condition_count": None if evaluation is None else (
            evaluation.real_h_passed_condition_count
        ),
        "fine_nonzero_prefix_condition_count": None if evaluation is None else (
            evaluation.fine_nonzero_prefix_condition_count
        ),
        "failures": failures,
        "failure_policy": (
            "old membership and denominators are immutable; one CG-derived predicted H "
            "is shared by paired U/G/CG; all pixel-grid points are retained; invalid H, "
            "direction, recovery, or scoring is a fixed-denominator method failure; no "
            "retry, fallback, replacement, subset, reclassification, or grid reselection"
        ),
        "route": {
            "generation_invoked": False,
            "geometry_positive_vote": False,
            "real_h": "stored R1A CG-derived H shared across U/G/CG",
            "directional_formula": (
                "e_pred_px=prediction_rmse*511/2; "
                "p_r=p_truth+(r/e_pred_px)*(p_hat-p_truth)"
            ),
            "lambda_zero": "old R1B truth-H raw scores reused without rescoring",
            "content_scorer": (
                "unchanged blind_weighted_scores through existing R0 binding; identical "
                "key, preprocessing, calibration, weighted-joint, exact16 wrong keys, tau=0"
            ),
        },
        "provenance": {
            "python": platform.python_version(),
            "dependencies_record_only": {
                name: _dependency_version(name)
                for name in ("torch", "Pillow", "diffusers", "transformers")
            },
            "operational_setup_error_class": None if setup_error is None else type(setup_error).__name__,
        },
    }


def _write_result(result_root: Path, result: Mapping[str, Any]) -> None:
    payload = json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    with (result_root / "result.json").open("xb") as output:
        output.write(payload)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    r0_root = Path(args.r0_artifact_root).resolve()
    r1a_root = Path(args.r1a_artifact_root).resolve()
    old_r1b_root = Path(args.old_r1b_artifact_root).resolve()
    result_root = Path(args.result_dir).resolve()
    if result_root.exists():
        raise FileExistsError("Geometry-V7 R1B repair result directory must be create-only")
    exact = _git_exact(repo_root, args.expected_exact)
    inputs = _load_r0_inputs(repo_root, r0_root)
    roster = tuple(item.unit_id for item in inputs)
    predictions = _validate_r1a_artifact(repo_root, r1a_root, roster)
    old = _load_old_r1b_artifact(old_r1b_root, roster)
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
            raise RuntimeError("cuda_required_for_real_geometry_v7_r1b_repair")
        contract = load_content_chain_contract(repo_root)
        _pipeline, assets = content_chain_runner._load_pipeline_and_assets(
            contract.runtime_protocol.config["generation_runtime"]["model_id"], token
        )
        if not isinstance(assets, ContentCalibrationAssets):
            raise TypeError("R1B repair requires real frozen content calibration assets")
        scorer = r0_runner._content_scorer(
            detection_key=detection_key,
            wrong_keys=derive_stability_wrong_keys(detection_key),
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
        real, fine = _operational_records(
            pre=old.pre_by_condition, old_zero=old.zero_by_condition,
            error=setup_error or RuntimeError("content scorer unavailable"),
        )
        return _result_payload(
            exact=exact, r0_root=r0_root, r1a_root=r1a_root,
            old_r1b_root=old_r1b_root, inputs=inputs, old=old,
            predictions=predictions, real=real, fine=fine,
            evaluation=None, setup_error=setup_error,
        )

    rendered, render_failures = _render_attacks(inputs)
    real = _real_h_records(
        pre=old.pre_by_condition, predictions=predictions, rendered=rendered,
        render_failures=render_failures, scorer=scorer,
    )
    fine = _fine_grid_records(
        pre=old.pre_by_condition, old_zero=old.zero_by_condition,
        predictions=predictions, rendered=rendered,
        render_failures=render_failures, scorer=scorer,
    )
    evaluation = evaluate_r1b_repair(
        pre_records_by_condition=old.pre_by_condition,
        real_h_records_by_condition=real,
        fine_grid_records_by_condition=fine,
        ordered_roster=roster,
    )
    return _result_payload(
        exact=exact, r0_root=r0_root, r1a_root=r1a_root,
        old_r1b_root=old_r1b_root, inputs=inputs, old=old,
        predictions=predictions, real=real, fine=fine,
        evaluation=evaluation, setup_error=None,
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
    print("CEGWM_GEOMETRY_V7_R1B_REPAIR " + json.dumps(
        {
            "status": result["status"], "real_h_status": result["real_h_status"],
            "fine_grid_status": result["fine_grid_status"],
            "claim_ceiling": result["claim_ceiling"], "exact": result["exact"],
            "result_dir": str(result_root),
        }, sort_keys=True, separators=(",", ":"),
    ), flush=True)
    return 0 if result["status"] == R1B_REPAIR_METHOD_PASSED else 2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--r0-artifact-root", required=True)
    parser.add_argument("--r1a-artifact-root", required=True)
    parser.add_argument("--old-r1b-artifact-root", required=True)
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
