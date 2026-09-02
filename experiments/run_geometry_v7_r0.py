"""Run the fixed Geometry-V7 R0 paired compatibility canary on real final RGB."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from experiments import content_adaptive_engine as engine
from experiments import run_content_chain as content_chain_runner
from cegwm.geometry_v7.r0 import (
    ContentScore,
    ImageQuality,
    R0AggregateEvaluation,
    R0Arm,
    R0DevelopmentSelection,
    R0MultiplierRecords,
    R0NumericGates,
    R0UnitRecord,
    evaluate_r0_test,
    r0_pre_arm_failure_record,
    r0_producer_failure_record,
    r0_record_payload,
    run_r0_four_arm_unit,
    select_r0_development_multiplier,
)
from cegwm.geometry_v7.syncseal import (
    SYNCSEAL_TORCHSCRIPT_URL,
    SyncSealTorchScript,
    download_official_syncseal_torchscript,
)
from cegwm.protocol.content_chain import (
    CONTENT_CHAIN_PUBLIC_KEY_DIGEST,
    ContentChainContract,
    ContentChainUnit,
    load_content_chain_contract,
)
from cegwm.runtime.content_iss_sd35 import (
    ContentISSRunOutput,
    run_content_iss_evaluation_pair,
)
from cegwm.runtime.content_weighted_joint_sd35 import (
    ContentCalibrationAssets,
    blind_weighted_scores,
    derive_stability_wrong_keys,
)
from cegwm.shared.keys import normalize_detection_key, public_key_digest


RESULT_SCHEMA = "geometry_v7_r0_run_all_result_v1"
CLAIM_CEILING = "small_sample_paired_compatibility_canary_only"
PAIRED_COMPATIBILITY_CONCLUSION = (
    "small-sample paired compatibility canary only; not fixed-FPR, not "
    "TPR-at-fixed-FPR, and not single-image blind FPR"
)
_SCORE_BRANCHES = ("lf", "hf", "weighted_joint")
_SCORE_LABELS = ("registered", *(f"wrong_{index:02d}" for index in range(16)))


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


def _sanitized_operational_error(error: Exception) -> RuntimeError:
    """Retain only the content runner's bounded public failure category."""

    return RuntimeError(engine._public_operational_error_class(error))


def _sanitized_record_error(value: object) -> str:
    """Remove exception messages from the published fixed-denominator record."""

    text = str(value)
    pieces = text.split(":", 2)
    if len(pieces) >= 2:
        return f"{pieces[0]}:{pieces[1]}"
    return f"{pieces[0]}:reported_error"


def _content_scorer(
    *,
    detection_key: bytes,
    wrong_keys: tuple[bytes, ...],
    assets: ContentCalibrationAssets,
    contract: ContentChainContract,
) -> Callable[[Image.Image], ContentScore]:
    """Bind only final RGB, key, and frozen public scoring assets."""

    def score(final_rgb: Image.Image) -> ContentScore:
        values = blind_weighted_scores(
            final_rgb,
            detection_key,
            wrong_keys,
            assets,
            contract.calibration_asset,
        )
        if tuple(values) != _SCORE_BRANCHES or any(
            tuple(values[branch]) != _SCORE_LABELS for branch in _SCORE_BRANCHES
        ):
            raise ValueError("content blind raw score identity or order differs")
        return ContentScore(
            float(values["lf"]["registered"]),
            float(values["hf"]["registered"]),
            float(values["weighted_joint"]["registered"]),
            tuple(float(values["lf"][label]) for label in _SCORE_LABELS[1:]),
            tuple(float(values["hf"][label]) for label in _SCORE_LABELS[1:]),
            tuple(
                float(values["weighted_joint"][label]) for label in _SCORE_LABELS[1:]
            ),
        )

    return score


def _rgb_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    pixels = np.asarray(image, dtype=np.uint8).copy()
    if pixels.shape != (512, 512, 3):
        raise ValueError("eval_sync quality input must be RGB8 512x512")
    return (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .clamp(0.0, 1.0)
    )


def _quality_scorer(device: torch.device) -> Callable[[Image.Image, Image.Image], ImageQuality]:
    """Load and bind the exact authorized official eval_sync implementations."""

    import lpips  # type: ignore[import-not-found]
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    perceptual = lpips.LPIPS(net="alex").to(device).eval()

    def score(base: Image.Image, watermarked: Image.Image) -> ImageQuality:
        base_tensor = _rgb_tensor(base, device)
        watermarked_tensor = _rgb_tensor(watermarked, device)
        with torch.no_grad():
            psnr = peak_signal_noise_ratio(
                watermarked_tensor, base_tensor, data_range=1.0
            )
            ssim = structural_similarity_index_measure(
                watermarked_tensor, base_tensor, data_range=1.0
            )
            # Authorized eval_sync passes direct clamped [0,1] RGB, with no
            # input-range transformation or convenience normalization.
            perceptual_distance = perceptual(watermarked_tensor, base_tensor)
        values = tuple(
            float(item.detach().to("cpu").reshape(-1).mean().item())
            for item in (psnr, ssim, perceptual_distance)
        )
        return ImageQuality(*values)

    return score


def _produce_pairs(
    units: Sequence[ContentChainUnit],
    *,
    pipeline: Any,
    detection_key: bytes,
    assets: ContentCalibrationAssets,
) -> tuple[tuple[ContentChainUnit, ContentISSRunOutput | BaseException], ...]:
    outcomes: list[tuple[ContentChainUnit, ContentISSRunOutput | BaseException]] = []
    for unit in units:
        try:
            output = run_content_iss_evaluation_pair(
                pipeline,
                unit.prompt,
                detection_key,
                assets.iss_assets,
                height=unit.height,
                width=unit.width,
                seed=unit.seed,
            )
            if not isinstance(output, ContentISSRunOutput):
                raise TypeError("content pair producer must return ContentISSRunOutput")
            outcomes.append((unit, output))
        except Exception as error:  # fixed unit denominator retains the real failure
            outcomes.append((unit, _sanitized_operational_error(error)))
    return tuple(outcomes)


def _attempt_records(
    outcomes: Sequence[tuple[ContentChainUnit, ContentISSRunOutput | BaseException]],
    *,
    multiplier: float,
    syncseal: SyncSealTorchScript,
    content_scorer: Callable[[Image.Image], ContentScore],
    quality_scorer: Callable[[Image.Image, Image.Image], ImageQuality],
) -> tuple[R0UnitRecord, ...]:
    records: list[R0UnitRecord] = []
    for unit, outcome in outcomes:
        if isinstance(outcome, BaseException):
            records.append(
                r0_producer_failure_record(
                    unit_id=unit.unit_id,
                    residual_strength_multiplier=multiplier,
                    error=outcome,
                )
            )
            continue
        records.append(
            run_r0_four_arm_unit(
                unit_id=unit.unit_id,
                unwatermarked_final_rgb=outcome.primary_null,
                content_watermarked_final_rgb=outcome.image,
                residual_strength_multiplier=multiplier,
                sync_embedder=syncseal.embed_final_rgb,
                content_scorer=content_scorer,
                geometry_detector=syncseal.detect_geometry,
                quality_scorer=quality_scorer,
            )
        )
    return tuple(records)


def _write_png(path: Path, image: Image.Image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        image.save(output, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_payload(
    record: R0UnitRecord,
    *,
    stage: str,
    result_root: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    payload = r0_record_payload(record)
    files: list[dict[str, str]] = []
    arms = payload["arms"]
    if not isinstance(arms, list):
        raise TypeError("R0 JSON arm projection differs")
    for arm_record, arm_payload in zip(record.arms, arms, strict=True):
        relative = None
        if arm_record.image is not None:
            relative_path = (
                Path("images")
                / stage
                / f"multiplier-{record.residual_strength_multiplier:.2f}"
                / record.unit_id
                / f"{arm_record.arm.name}.png"
            )
            digest = _write_png(result_root / relative_path, arm_record.image)
            relative = relative_path.as_posix()
            files.append({"path": relative, "sha256": digest})
        if not isinstance(arm_payload, dict):
            raise TypeError("R0 JSON arm item differs")
        errors = arm_payload.get("errors")
        if not isinstance(errors, tuple):
            raise TypeError("R0 JSON arm errors differ")
        arm_payload["errors"] = tuple(
            _sanitized_record_error(value) for value in errors
        )
        arm_payload["image_file"] = relative
    payload["stage"] = stage
    return payload, files


def _aggregate_payload(aggregate: R0AggregateEvaluation) -> dict[str, Any]:
    payload = _jsonable(aggregate)
    if not isinstance(payload, dict):
        raise TypeError("R0 aggregate JSON projection differs")
    payload["observed_paired_G_false_positive_rate"] = (
        aggregate.observed_paired_G_false_positive_rate
    )
    payload["paired_compatibility_claim"] = PAIRED_COMPATIBILITY_CONCLUSION
    return payload


def _failure_index(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for record in records:
        arms = record.get("arms")
        if not isinstance(arms, list):
            raise TypeError("R0 failure index arm projection differs")
        for arm in arms:
            if not isinstance(arm, Mapping):
                raise TypeError("R0 failure index arm item differs")
            errors = arm.get("errors")
            if errors:
                failures.append(
                    {
                        "stage": record.get("stage"),
                        "unit_id": record.get("unit_id"),
                        "residual_strength_multiplier": record.get(
                            "residual_strength_multiplier"
                        ),
                        "arm": arm.get("arm"),
                        "errors": errors,
                    }
                )
    return failures


def _setup_failure_result(
    *,
    repo_root: Path,
    exact: str,
    key_digest: str,
    contract: ContentChainContract,
    result_root: Path,
    failure_stage: str,
    error: BaseException,
) -> dict[str, Any]:
    """Project one global real-runtime failure across the complete dev grid."""

    gates = R0NumericGates()
    sanitized_error = _sanitized_operational_error(error)
    attempts: list[R0MultiplierRecords] = []
    raw_records: list[dict[str, object]] = []
    for multiplier in gates.residual_strength_multipliers:
        records = tuple(
            r0_pre_arm_failure_record(
                unit_id=unit.unit_id,
                residual_strength_multiplier=multiplier,
                failure_stage=failure_stage,
                error=sanitized_error,
            )
            for unit in contract.reference_roster[:4]
        )
        attempts.append(R0MultiplierRecords(multiplier, records))
        for record in records:
            payload, files = _record_payload(
                record, stage="development", result_root=result_root
            )
            if files:
                raise RuntimeError("setup failure cannot publish image files")
            raw_records.append(payload)
    selection = select_r0_development_multiplier(
        repo_root=repo_root,
        attempts=tuple(attempts),
        gates=gates,
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR",
        "claim_ceiling": CLAIM_CEILING,
        "conclusion": PAIRED_COMPATIBILITY_CONCLUSION,
        "exact": exact,
        "protocol_digest": contract.protocol_digest,
        "public_key_digest": key_digest,
        "rosters": {
            "development": [unit.unit_id for unit in contract.reference_roster[:4]],
            "evaluation": [unit.unit_id for unit in contract.evaluation_roster],
        },
        "selection": _jsonable(selection),
        "development_aggregates": [
            _aggregate_payload(item) for item in selection.attempts
        ],
        "evaluation_aggregate": None,
        "raw_unit_records": raw_records,
        "image_files": [],
        "failures": _failure_index(raw_records),
        "failure_policy": (
            "global real-runtime failure projected across fixed development 4 x "
            "frozen 4 multipliers; no retry, fallback, replacement, or successful subset"
        ),
        "route": {
            "content_pair_producer": (
                "cegwm.runtime.content_iss_sd35.run_content_iss_evaluation_pair"
            ),
            "content_blind_scorer": (
                "cegwm.runtime.content_weighted_joint_sd35.blind_weighted_scores"
            ),
            "geometry_role": "coordinates_only_never_content_vote",
            "setup_failure_stage": failure_stage,
            "setup_failure_public_class": str(sanitized_error),
        },
        "provenance": {
            "model_id": contract.runtime_protocol.config["generation_runtime"]["model_id"],
            "python": platform.python_version(),
            "dependencies_record_only": {
                name: _dependency_version(name)
                for name in (
                    "torch",
                    "torchmetrics",
                    "lpips",
                    "diffusers",
                    "transformers",
                )
            },
            "syncseal_checkpoint_sha256_record_only": None,
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
    result_path = result_root / "result.json"
    with result_path.open("xb") as output:
        output.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with (result_root / "result.json.sha256").open("x", encoding="ascii") as sidecar:
        sidecar.write(f"{digest}  result.json\n")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    result_root = Path(args.result_dir).resolve()
    checkpoint = Path(args.syncseal_checkpoint).resolve()
    if result_root.exists():
        raise FileExistsError("Geometry-V7 R0 result directory must be create-only")

    key_text = os.environ.pop(engine.KEY_ENV, "")
    token = os.environ.pop(engine.TOKEN_ENV, "")
    if not key_text.strip():
        token = ""
        raise RuntimeError("CEG_WM_ROOT_KEY_is_required")
    if not token.strip():
        key_text = ""
        raise RuntimeError("HF_TOKEN_is_required")
    detection_key = normalize_detection_key(key_text)
    key_text = ""
    key_digest = public_key_digest(detection_key)
    if key_digest != CONTENT_CHAIN_PUBLIC_KEY_DIGEST:
        detection_key = b""
        token = ""
        raise RuntimeError("content chain public key identity differs")
    if not torch.cuda.is_available():
        detection_key = b""
        token = ""
        raise RuntimeError("cuda_required_for_real_geometry_v7_r0")

    exact = engine._git_exact(repo_root, args.expected_exact)
    contract = load_content_chain_contract(repo_root)
    result_root.mkdir(parents=True, exist_ok=False)
    try:
        try:
            try:
                pipeline, assets = content_chain_runner._load_pipeline_and_assets(
                    contract.runtime_protocol.config["generation_runtime"]["model_id"],
                    token,
                )
                if not isinstance(assets, ContentCalibrationAssets):
                    raise TypeError(
                        "Geometry-V7 R0 requires real content calibration assets"
                    )
                wrong_keys = derive_stability_wrong_keys(detection_key)
                scorer = _content_scorer(
                    detection_key=detection_key,
                    wrong_keys=wrong_keys,
                    assets=assets,
                    contract=contract,
                )
            except Exception as error:
                return _setup_failure_result(
                    repo_root=repo_root,
                    exact=exact,
                    key_digest=key_digest,
                    contract=contract,
                    result_root=result_root,
                    failure_stage="content_runtime_setup",
                    error=error,
                )
        finally:
            token = ""
        try:
            syncseal_path = download_official_syncseal_torchscript(checkpoint)
            syncseal = SyncSealTorchScript.from_file(syncseal_path, device="cuda")
        except Exception as error:
            return _setup_failure_result(
                repo_root=repo_root,
                exact=exact,
                key_digest=key_digest,
                contract=contract,
                result_root=result_root,
                failure_stage="syncseal_runtime_setup",
                error=error,
            )
        try:
            quality = _quality_scorer(torch.device("cuda"))
        except Exception as error:
            return _setup_failure_result(
                repo_root=repo_root,
                exact=exact,
                key_digest=key_digest,
                contract=contract,
                result_root=result_root,
                failure_stage="quality_runtime_setup",
                error=error,
            )
        gates = R0NumericGates()

        development_outcomes = _produce_pairs(
            contract.reference_roster[:4],
            pipeline=pipeline,
            detection_key=detection_key,
            assets=assets,
        )
        attempts: list[R0MultiplierRecords] = []
        raw_records: list[dict[str, object]] = []
        image_files: list[dict[str, str]] = []
        selection: R0DevelopmentSelection | None = None
        for multiplier in gates.residual_strength_multipliers:
            records = _attempt_records(
                development_outcomes,
                multiplier=multiplier,
                syncseal=syncseal,
                content_scorer=scorer,
                quality_scorer=quality,
            )
            attempts.append(R0MultiplierRecords(multiplier, records))
            for record in records:
                payload, files = _record_payload(
                    record, stage="development", result_root=result_root
                )
                raw_records.append(payload)
                image_files.extend(files)
            selection = select_r0_development_multiplier(
                repo_root=repo_root,
                attempts=tuple(attempts),
                gates=gates,
            )
            if selection.complete:
                break
        if selection is None:
            raise RuntimeError("R0 development produced no frozen-grid attempt")

        evaluation = None
        if selection.selected_residual_strength_multiplier is not None:
            evaluation_outcomes = _produce_pairs(
                contract.evaluation_roster,
                pipeline=pipeline,
                detection_key=detection_key,
                assets=assets,
            )
            evaluation_records = _attempt_records(
                evaluation_outcomes,
                multiplier=selection.selected_residual_strength_multiplier,
                syncseal=syncseal,
                content_scorer=scorer,
                quality_scorer=quality,
            )
            for record in evaluation_records:
                payload, files = _record_payload(
                    record, stage="evaluation", result_root=result_root
                )
                raw_records.append(payload)
                image_files.extend(files)
            evaluation = evaluate_r0_test(
                repo_root=repo_root,
                records=evaluation_records,
                development_selection=selection,
                gates=gates,
            )

        if selection.selected_residual_strength_multiplier is None:
            status = "STOPPED_NO_PAIRED_COMPATIBILITY_WINDOW"
        elif evaluation is not None and evaluation.carrier_compatibility_passed:
            status = "PAIRED_COMPATIBILITY_CANARY_PASSED"
        else:
            status = "PAIRED_COMPATIBILITY_CANARY_FAILED"
        return {
            "schema": RESULT_SCHEMA,
            "status": status,
            "claim_ceiling": CLAIM_CEILING,
            "conclusion": PAIRED_COMPATIBILITY_CONCLUSION,
            "exact": exact,
            "protocol_digest": contract.protocol_digest,
            "public_key_digest": key_digest,
            "rosters": {
                "development": [unit.unit_id for unit in contract.reference_roster[:4]],
                "evaluation": [unit.unit_id for unit in contract.evaluation_roster],
            },
            "selection": _jsonable(selection),
            "development_aggregates": [
                _aggregate_payload(item) for item in selection.attempts
            ],
            "evaluation_aggregate": None
            if evaluation is None
            else _aggregate_payload(evaluation),
            "raw_unit_records": raw_records,
            "image_files": image_files,
            "failures": _failure_index(raw_records),
            "failure_policy": (
                "all fixed units and illegal/nonfinite/failed observations remain in the "
                "denominator; no retry, fallback, replacement, or successful subset"
            ),
            "route": {
                "content_pair_producer": (
                    "cegwm.runtime.content_iss_sd35.run_content_iss_evaluation_pair"
                ),
                "content_blind_scorer": (
                    "cegwm.runtime.content_weighted_joint_sd35.blind_weighted_scores"
                ),
                "content_decision": (
                    "paired Gate A registered-minus-max-16-wrong and Gate B "
                    "candidate-minus-paired-null; strict conjunction; margin=min(A,B)"
                ),
                "paired_nulls": {"G": "U", "C": "U", "CG": "G"},
                "syncseal_url": SYNCSEAL_TORCHSCRIPT_URL,
                "geometry_role": "coordinates_only_never_content_vote",
                "quality": {
                    "psnr": (
                        "torchmetrics.functional.image.peak_signal_noise_ratio"
                        "(watermarked,base,data_range=1.0)"
                    ),
                    "ssim": (
                        "torchmetrics.functional.image.structural_similarity_index_measure"
                        "(watermarked,base,data_range=1.0,defaults)"
                    ),
                    "lpips": (
                        "lpips.LPIPS(net='alex')(watermarked,base), direct clamped "
                        "[0,1] RGB, no [-1,1] transform"
                    ),
                },
            },
            "provenance": {
                "model_id": contract.runtime_protocol.config["generation_runtime"]["model_id"],
                "python": platform.python_version(),
                "dependencies_record_only": {
                    name: _dependency_version(name)
                    for name in (
                        "torch",
                        "torchmetrics",
                        "lpips",
                        "diffusers",
                        "transformers",
                    )
                },
                "syncseal_checkpoint_sha256_record_only": hashlib.sha256(
                    syncseal_path.read_bytes()
                ).hexdigest(),
            },
        }
    finally:
        detection_key = b""
        token = ""


def execute(args: argparse.Namespace) -> int:
    result_root = Path(args.result_dir).resolve()
    preexisting = result_root.exists()
    try:
        result = _run(args)
    except BaseException:
        if not preexisting and result_root.is_dir() and not (result_root / "result.json").exists():
            shutil.rmtree(result_root)
        raise
    try:
        _write_result(result_root, result)
    except BaseException:
        if not preexisting and result_root.is_dir():
            shutil.rmtree(result_root)
        raise
    print(
        "CEGWM_GEOMETRY_V7_R0 "
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
    return 0 if result["status"] == "PAIRED_COMPATIBILITY_CANARY_PASSED" else 2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--syncseal-checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
