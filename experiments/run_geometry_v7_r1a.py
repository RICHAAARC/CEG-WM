"""Run Geometry-V7 R1A on the fixed eight R0 evaluation CG images."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from PIL import Image
import torch

from cegwm.geometry_v7.contracts import GeometryStatus
from cegwm.geometry_v7.r1a import (
    R1A_ALL_CONDITIONS,
    R1A_ATTACK_SPEC_REQUEST_CHANGES,
    R1A_BLOCKING_METHOD_CANARY_PASSED,
    R1AConditionRecords,
    R1AEvaluation,
    R1ATruthPreflight,
    R1AUnitRecord,
    detect_attacked_rgb,
    evaluate_r1a,
    evaluate_r1a_observation,
    r1a_detection_setup_failure_record,
    r1a_truth_preflight,
    render_r1a_attack,
)
from cegwm.geometry_v7.syncseal import (
    SYNCSEAL_TORCHSCRIPT_URL,
    SyncSealTorchScript,
    download_official_syncseal_torchscript,
)
from cegwm.protocol.content_chain import load_content_chain_contract


R0_PRODUCER_EXACT = "4f0bf1560805672f786dc86dd50d793aec18aae7"
R0_REQUIRED_STATUS = "PAIRED_COMPATIBILITY_CANARY_PASSED"
R0_SELECTED_MULTIPLIER = 0.75
RESULT_SCHEMA = "geometry_v7_r1a_result_v1"
CLAIM_CEILING = "small_sample_blocking_geometry_method_canary_only"
OPERATIONAL_FAILURE_STATUS = "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"


@dataclass(frozen=True, slots=True)
class R0CGInput:
    unit_id: str
    path: Path
    relative_path: str


@dataclass(frozen=True, slots=True)
class RenderedAttack:
    unit_id: str
    condition_id: str
    image: Image.Image
    relative_path: str
    sha256: str


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


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("R0 result.json must be readable UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("R0 result.json must contain an object")
    return value


def _validated_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
                raise ValueError("R0 CG input must be an RGB 512x512 PNG")
            image.verify()
    except (OSError, ValueError) as error:
        raise ValueError("R0 CG input must be an RGB 512x512 PNG") from error


def _relative_member(root: Path, value: object) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ValueError("R0 CG image member must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("R0 CG image member must stay inside the artifact root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("R0 CG image member escaped the artifact root") from error
    if not path.is_file():
        raise ValueError("R0 CG image member is absent")
    _validated_png(path)
    return path, relative.as_posix()


def _load_r0_cg_inputs(repo_root: Path, artifact_root: Path) -> tuple[R0CGInput, ...]:
    """Select only fixed evaluation CG PNGs; sidecars and hashes are not gates."""

    root = artifact_root.resolve()
    if not root.is_dir():
        raise ValueError("R0 artifact root must be an existing directory")
    result = _read_json(root / "result.json")
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
        raise ValueError("R0 artifact identity, status, selection, or roster differs")
    raw_records = result.get("raw_unit_records")
    if not isinstance(raw_records, list):
        raise ValueError("R0 raw unit records are absent")
    evaluation = tuple(
        record
        for record in raw_records
        if isinstance(record, Mapping) and record.get("stage") == "evaluation"
    )
    if (
        len(evaluation) != 8
        or tuple(record.get("unit_id") for record in evaluation) != roster
        or any(
            record.get("residual_strength_multiplier") != R0_SELECTED_MULTIPLIER
            for record in evaluation
        )
    ):
        raise ValueError("R0 evaluation records differ from the fixed ordered roster")
    inputs: list[R0CGInput] = []
    for unit_id, record in zip(roster, evaluation, strict=True):
        arms = record.get("arms")
        if not isinstance(arms, list):
            raise ValueError("R0 evaluation arm records are absent")
        matches = tuple(
            arm
            for arm in arms
            if isinstance(arm, Mapping) and arm.get("arm") == "CG_with_content_with_sync"
        )
        if len(matches) != 1 or matches[0].get("errors") not in ([], ()):
            raise ValueError("R0 evaluation CG arm is missing or failed")
        path, relative = _relative_member(root, matches[0].get("image_file"))
        inputs.append(R0CGInput(unit_id, path, relative))
    return tuple(inputs)


def _write_png(path: Path, image: Image.Image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        image.save(output, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_all(
    inputs: Sequence[R0CGInput],
    result_root: Path,
) -> tuple[RenderedAttack, ...]:
    rendered: list[RenderedAttack] = []
    for spec in R1A_ALL_CONDITIONS:
        for item in inputs:
            with Image.open(item.path) as source:
                source.load()
                attacked = render_r1a_attack(source, spec)
            relative = (
                Path("attacked") / spec.condition_id / f"{item.unit_id}.png"
            )
            digest = _write_png(result_root / relative, attacked)
            rendered.append(
                RenderedAttack(
                    item.unit_id,
                    spec.condition_id,
                    attacked,
                    relative.as_posix(),
                    digest,
                )
            )
    if len(rendered) != 13 * 8:
        raise RuntimeError("R1A renderer did not produce the fixed 104 images")
    return tuple(rendered)


def _records_after_detection(
    rendered: Sequence[RenderedAttack],
    detector: Any,
) -> tuple[R1AConditionRecords, ...]:
    condition_records: list[R1AConditionRecords] = []
    offset = 0
    for spec in R1A_ALL_CONDITIONS:
        records: list[R1AUnitRecord] = []
        for item in rendered[offset : offset + 8]:
            try:
                geometry = detect_attacked_rgb(detector, item.image)
                errors = (
                    ("geometry_detect:reported_error",)
                    if geometry.status is GeometryStatus.ERROR
                    else ()
                )
                record = evaluate_r1a_observation(
                    unit_id=item.unit_id,
                    spec=spec,
                    attacked_image=item.image,
                    geometry=geometry,
                    errors=errors,
                )
            except Exception as error:
                record = evaluate_r1a_observation(
                    unit_id=item.unit_id,
                    spec=spec,
                    attacked_image=item.image,
                    geometry=None,
                    errors=(f"geometry_detect:{type(error).__name__}",),
                )
            records.append(record)
        condition_records.append(R1AConditionRecords(spec, tuple(records)))
        offset += 8
    return tuple(condition_records)


def _records_after_setup_failure(
    rendered: Sequence[RenderedAttack], error: BaseException
) -> tuple[R1AConditionRecords, ...]:
    condition_records: list[R1AConditionRecords] = []
    offset = 0
    sanitized = RuntimeError(type(error).__name__)
    for spec in R1A_ALL_CONDITIONS:
        records = tuple(
            r1a_detection_setup_failure_record(
                unit_id=item.unit_id,
                spec=spec,
                attacked_image=item.image,
                error=sanitized,
            )
            for item in rendered[offset : offset + 8]
        )
        condition_records.append(R1AConditionRecords(spec, records))
        offset += 8
    return tuple(condition_records)


def _geometry_payload(record: R1AUnitRecord) -> Mapping[str, Any] | None:
    geometry = record.geometry
    if geometry is None:
        return None
    return {
        "status": geometry.status.value,
        "uncalibrated_sync_logit": geometry.uncalibrated_sync_logit,
        "raw_syncseal_corners": geometry.raw_syncseal_corners,
        "observed_corners_in_canonical_normalized": (
            geometry.observed_corners_in_canonical_normalized
        ),
        "homography_observed_to_canonical": (
            geometry.homography_observed_to_canonical
        ),
        "legal": geometry.legal,
        "error": geometry.error,
    }


def _record_payload(
    record: R1AUnitRecord, rendered: RenderedAttack
) -> dict[str, Any]:
    if (
        record.unit_id != rendered.unit_id
        or record.condition_id != rendered.condition_id
    ):
        raise ValueError("R1A rendered/record identity differs")
    return {
        "unit_id": record.unit_id,
        "condition_id": record.condition_id,
        "condition_kind": record.condition_kind.value,
        "attacked_image_file": rendered.relative_path,
        "attacked_image_sha256_record_only": rendered.sha256,
        "truth_observed_corners_in_canonical_normalized": (
            record.truth_observed_corners_in_canonical_normalized
        ),
        "identity_baseline_rmse": record.identity_baseline_rmse,
        "truth_eligible": record.truth_eligible,
        "geometry": _geometry_payload(record),
        "prediction_rmse": record.prediction_rmse,
        "paired_delta": record.paired_delta,
        "improved": record.improved,
        "errors": record.errors,
    }


def _result_payload(
    *,
    exact: str,
    artifact_root: Path,
    inputs: Sequence[R0CGInput],
    rendered: Sequence[RenderedAttack],
    records: Sequence[R1AConditionRecords],
    evaluation: R1AEvaluation,
    setup_error: BaseException | None,
    checkpoint: Path | None,
) -> dict[str, Any]:
    flattened = tuple(
        record for item in records for record in item.records
    )
    if len(flattened) != 104 or len(rendered) != 104:
        raise RuntimeError("R1A fixed record count differs")
    raw_records = [
        _record_payload(record, image)
        for record, image in zip(flattened, rendered, strict=True)
    ]
    failures = [
        {
            "condition_id": record["condition_id"],
            "unit_id": record["unit_id"],
            "errors": record["errors"],
        }
        for record in raw_records
        if record["errors"]
    ]
    operational_failure = setup_error is not None or any(
        any(
            str(error).startswith(
                ("syncseal_runtime_setup:", "geometry_detect:")
            )
            for error in record.errors
        )
        for record in flattened
    )
    if operational_failure:
        status = OPERATIONAL_FAILURE_STATUS
    else:
        status = evaluation.status
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "claim_ceiling": CLAIM_CEILING,
        "scientific_status": "not_adjudicated",
        "exact": exact,
        "r0_input": {
            "producer_exact": R0_PRODUCER_EXACT,
            "artifact_root": str(artifact_root),
            "status": R0_REQUIRED_STATUS,
            "selected_residual_strength_multiplier": R0_SELECTED_MULTIPLIER,
            "ordered_evaluation_cg_inputs": [
                {"unit_id": item.unit_id, "path": item.relative_path}
                for item in inputs
            ],
            "sha_or_sidecar_used_as_gate": False,
        },
        "fixed_counts": {
            "conditions": 13,
            "sanity_conditions": 3,
            "core_conditions": 10,
            "units_per_condition": 8,
            "attacked_images": 104,
            "records": 104,
        },
        "truth_preflight": _jsonable(evaluation.truth_preflight),
        "condition_specs": [_jsonable(spec) for spec in R1A_ALL_CONDITIONS],
        "condition_aggregates": [
            _jsonable(aggregate) for aggregate in evaluation.aggregates
        ],
        "fixed_denominator_evaluation_status": evaluation.status,
        "all_sanity_passed": evaluation.all_sanity_passed,
        "all_core_passed": evaluation.all_core_passed,
        "blocking_method_canary_passed": (
            evaluation.blocking_method_canary_passed
            if not operational_failure
            else None
        ),
        "raw_records": raw_records,
        "failures": failures,
        "failure_policy": (
            "all fixed condition-unit failures remain in the denominator; no retry, "
            "fallback, replacement, condition pooling, or successful subset median"
        ),
        "route": {
            "input": "fixed R0 evaluation CG final RGB only",
            "detector_input": "attacked RGB only",
            "truth_role": "CPU renderer/evaluator only, never detector input",
            "resampling": "Pillow bilinear, black fill, one core resample",
            "syncseal_url": SYNCSEAL_TORCHSCRIPT_URL,
        },
        "provenance": {
            "python": platform.python_version(),
            "dependencies_record_only": {
                name: _dependency_version(name)
                for name in ("torch", "Pillow", "numpy")
            },
            "syncseal_checkpoint_sha256_record_only": (
                None
                if checkpoint is None
                else hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ),
            "operational_setup_error_class": (
                None if setup_error is None else type(setup_error).__name__
            ),
        },
    }


def _attack_spec_result(
    *, exact: str, artifact_root: Path, preflight: R1ATruthPreflight
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "status": R1A_ATTACK_SPEC_REQUEST_CHANGES,
        "claim_ceiling": CLAIM_CEILING,
        "scientific_status": "not_adjudicated",
        "exact": exact,
        "r0_input": {"artifact_root": str(artifact_root)},
        "truth_preflight": _jsonable(preflight),
        "condition_specs": [_jsonable(spec) for spec in R1A_ALL_CONDITIONS],
        "condition_aggregates": [],
        "raw_records": [],
        "failures": [],
        "blocking_method_canary_passed": None,
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
    digest = hashlib.sha256(payload).hexdigest()
    with (result_root / "result.json.sha256").open("x", encoding="ascii") as sidecar:
        sidecar.write(f"{digest}  result.json\n")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    artifact_root = Path(args.r0_artifact_root).resolve()
    result_root = Path(args.result_dir).resolve()
    checkpoint = Path(args.syncseal_checkpoint).resolve()
    if result_root.exists():
        raise FileExistsError("Geometry-V7 R1A result directory must be create-only")
    if checkpoint.exists():
        raise FileExistsError("Geometry-V7 R1A checkpoint must be create-only")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_geometry_v7_r1a")
    exact = _git_exact(repo_root, args.expected_exact)
    inputs = _load_r0_cg_inputs(repo_root, artifact_root)
    preflight = r1a_truth_preflight()
    result_root.mkdir(parents=True, exist_ok=False)
    if not preflight.passed:
        return _attack_spec_result(
            exact=exact,
            artifact_root=artifact_root,
            preflight=preflight,
        )
    rendered = _render_all(inputs, result_root)
    setup_error = None
    loaded_checkpoint = None
    try:
        loaded_checkpoint = download_official_syncseal_torchscript(checkpoint)
        syncseal = SyncSealTorchScript.from_file(loaded_checkpoint, device="cuda")
    except Exception as error:
        setup_error = error
        records = _records_after_setup_failure(rendered, error)
    else:
        records = _records_after_detection(rendered, syncseal.detect_geometry)
    roster = tuple(item.unit_id for item in inputs)
    evaluation = evaluate_r1a(
        condition_records=records,
        ordered_roster=roster,
    )
    return _result_payload(
        exact=exact,
        artifact_root=artifact_root,
        inputs=inputs,
        rendered=rendered,
        records=records,
        evaluation=evaluation,
        setup_error=setup_error,
        checkpoint=loaded_checkpoint,
    )


def execute(args: argparse.Namespace) -> int:
    result_root = Path(args.result_dir).resolve()
    checkpoint = Path(args.syncseal_checkpoint).resolve()
    preexisting = result_root.exists()
    checkpoint_preexisting = checkpoint.exists()
    try:
        result = _run(args)
        _write_result(result_root, result)
    except BaseException:
        if not preexisting and result_root.is_dir():
            shutil.rmtree(result_root)
        if not checkpoint_preexisting and checkpoint.is_file():
            checkpoint.unlink()
        raise
    print(
        "CEGWM_GEOMETRY_V7_R1A "
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
    return 0 if result["status"] == R1A_BLOCKING_METHOD_CANARY_PASSED else 2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--r0-artifact-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--syncseal-checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
