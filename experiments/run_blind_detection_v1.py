#!/usr/bin/env python3
"""Prepare/freeze BlindDetection-V1 assets without silently running models."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cegwm.method.blind_detection import (  # noqa: E402
    BLIND_DEV_DENOMINATOR,
    BLIND_DEV_DISJOINT_FROM,
    BLIND_PRODUCTION_RUNTIME_ID,
    BLIND_STATISTIC_ID,
    BlindCalibrationRoster,
    BlindCalibrationUnit,
    build_threshold_asset,
    candidate_tau_blind,
    encode_binary64,
    load_threshold_asset,
    stable_json_bytes,
)
from cegwm.geometry_v7.syncseal import SyncSealTorchScript  # noqa: E402
from cegwm.method.content_weighted_joint import load_calibration_asset  # noqa: E402
from cegwm.runtime.blind_detection import (  # noqa: E402
    BLIND_PREPROCESS_ID,
    BLIND_SCORER_ID,
    BlindProductionAssets,
    detect_watermark,
    run_development_calibration,
    run_development_full_system_replay,
)
from cegwm.runtime.content_weighted_joint_sd35 import ContentCalibrationAssets  # noqa: E402
from cegwm.runtime.observation import require_ordinary_rgb_image  # noqa: E402
from cegwm.shared.keys import normalize_detection_key, public_key_digest  # noqa: E402


CALIBRATION_RESULT_SCHEMA = "cegwm_blind_detection_v1_calibration_result_v1"
CALIBRATION_CLAIM_CEILING = (
    "engineering_N_dev_256_threshold_calibration_only; science_denominator=0; "
    "not_fixed_FPR_production_reliability_or_paper_evidence"
)
CONTENT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
WEIGHTED_ASSET_REPO_PATH = Path(
    "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json"
)
WEIGHTED_ASSET_SIDECAR_REPO_PATH = Path(
    "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json.sha256"
)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_roster_inputs(
    path: str | Path,
) -> tuple[BlindCalibrationRoster, dict[str, str], str]:
    """Validate and summarize the frozen roster before any scoring occurs."""

    payload = _read_json(path)
    required = {"disjoint_evidence", "disjoint_from", "units"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("blind development roster fields differ")
    evidence = payload["disjoint_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(BLIND_DEV_DISJOINT_FROM):
        raise ValueError("blind development disjointness evidence fields differ")
    if any(not isinstance(evidence[name], str) or not evidence[name] for name in evidence):
        raise ValueError("blind development disjointness evidence must be nonempty")
    units = tuple(BlindCalibrationUnit(**unit) for unit in payload["units"])
    roster = BlindCalibrationRoster(units, tuple(payload["disjoint_from"]))
    if len({unit.image_ref for unit in roster.units}) != BLIND_DEV_DENOMINATOR:
        raise ValueError("blind development image references must be unique")
    return roster, dict(evidence), _sha256_file(path)


def load_roster(path: str | Path) -> BlindCalibrationRoster:
    return load_roster_inputs(path)[0]


def _current_rgb_digest(image: Image.Image) -> str:
    current = require_ordinary_rgb_image(image)
    framed = (
        b"CEG-WM/blind-current-rgb/v1\0"
        + current.width.to_bytes(8, "big")
        + current.height.to_bytes(8, "big")
        + current.tobytes()
    )
    return hashlib.sha256(framed).hexdigest()


def validate_unique_current_images(
    roster: BlindCalibrationRoster, image_loader
) -> dict[str, Image.Image]:
    """Load and validate all 256 physical RGBs before the first score."""

    if not callable(image_loader):
        raise TypeError("roster image loader must be callable")
    cached: dict[str, Image.Image] = {}
    digest_owner: dict[str, str] = {}
    for index, unit in enumerate(roster.units):
        try:
            current = require_ordinary_rgb_image(image_loader(unit.image_ref))
        except Exception as error:
            raise RuntimeError(
                f"roster_image_validation[{index}]:{type(error).__name__}: {error}"
            ) from error
        digest = _current_rgb_digest(current)
        if digest in digest_owner:
            raise ValueError(
                "blind development physical RGB is duplicated: "
                f"unit {unit.unit_id} matches {digest_owner[digest]}"
            )
        digest_owner[digest] = unit.unit_id
        cached[unit.image_ref] = current.copy()
    if len(cached) != BLIND_DEV_DENOMINATOR:
        raise RuntimeError("blind development physical image cache denominator differs")
    return cached


def load_runtime_config(path: str | Path) -> tuple[dict[str, str], str]:
    payload = _read_json(path)
    required = {
        "content_model_id", "device", "syncseal_checkpoint",
        "syncseal_checkpoint_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("blind calibration runtime config fields differ")
    if payload["content_model_id"] != CONTENT_MODEL_ID or payload["device"] != "cuda":
        raise ValueError("blind calibration frozen model or device identity differs")
    checkpoint = payload["syncseal_checkpoint"]
    checkpoint_sha256 = payload["syncseal_checkpoint_sha256"]
    if not isinstance(checkpoint, str) or not checkpoint or not Path(checkpoint).is_absolute():
        raise ValueError("SyncSeal checkpoint path must be absolute and nonempty")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise ValueError("SyncSeal checkpoint SHA-256 must be lowercase 64-hex")
    return dict(payload), _sha256_file(path)


def build_production_runtime(
    repo_root: Path, config: dict[str, str], *, hf_token: str
) -> BlindProductionAssets:
    """Construct the real typed detector through existing repository loaders."""

    if not isinstance(hf_token, str) or not hf_token.strip():
        raise RuntimeError("HF_TOKEN is required to load frozen public content assets")
    from experiments import content_iss_engine

    _, content_runner_assets = content_iss_engine._load_pipeline_and_assets(
        config["content_model_id"], hf_token
    )
    content_assets = ContentCalibrationAssets(content_runner_assets.evaluation_assets)
    weighted_path = repo_root / WEIGHTED_ASSET_REPO_PATH
    weighted_sidecar = repo_root / WEIGHTED_ASSET_SIDECAR_REPO_PATH
    weighted_asset = load_calibration_asset(weighted_path, weighted_sidecar)
    checkpoint = Path(config["syncseal_checkpoint"])
    if _sha256_file(checkpoint) != config["syncseal_checkpoint_sha256"]:
        raise ValueError("SyncSeal checkpoint SHA-256 differs")
    geometry = SyncSealTorchScript.from_file(checkpoint, device=config["device"])
    return BlindProductionAssets(content_assets, weighted_asset, geometry)


class ThresholdFreezeBlocked(RuntimeError):
    """Carry every attempted fixed-denominator row when threshold output is blocked."""

    def __init__(self, cause: Exception, calibration_rows, replay_rows) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.calibration_rows = tuple(calibration_rows)
        self.replay_rows = tuple(replay_rows)
        self.status = "METHOD_FAILED" if "0/256" in str(cause) else "OPERATIONAL_BLOCKED"


def evaluate_threshold_with_runtime(
    roster: BlindCalibrationRoster,
    key: bytes,
    public_assets: BlindProductionAssets,
    image_loader,
    *,
    producer_exact: str,
    replay_image_loader=None,
):
    """Evaluate fixed calibration and fresh replay without writing any artifact."""

    if type(public_assets) is not BlindProductionAssets:
        raise TypeError("threshold freeze requires BlindProductionAssets")
    if public_assets.threshold_asset is not None:
        raise ValueError("threshold calibration runtime must not contain an earlier threshold")
    rows = run_development_calibration(roster, key, public_assets, image_loader)
    try:
        tau_blind = candidate_tau_blind(rows, roster)
    except Exception as error:
        raise ThresholdFreezeBlocked(error, rows, ()) from error
    replay = run_development_full_system_replay(
        roster,
        key,
        public_assets,
        image_loader if replay_image_loader is None else replay_image_loader,
        tau_blind,
    )
    try:
        asset = build_threshold_asset(
            rows, roster, replay, producer_exact=producer_exact,
            calibration_key_digest=public_key_digest(key),
        )
    except Exception as error:
        raise ThresholdFreezeBlocked(error, rows, replay) from error
    return tuple(rows), tuple(replay), asset


def freeze_threshold_with_runtime(
    roster: BlindCalibrationRoster,
    key: bytes,
    public_assets: BlindProductionAssets,
    image_loader,
    output_path: str | Path,
    *,
    producer_exact: str,
) -> Path:
    """Run fresh calibration and full-system replay before create-only output."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError("blind threshold output is create-only")
    _, _, asset = evaluate_threshold_with_runtime(
        roster, key, public_assets, image_loader, producer_exact=producer_exact
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as sink:
        sink.write(asset.json_bytes)
    return output


def _verify_producer_checkout(producer_exact: str) -> None:
    if not isinstance(producer_exact, str) or len(producer_exact) != 40 or any(
        character not in "0123456789abcdef" for character in producer_exact
    ):
        raise ValueError("producer exact must be lowercase 40-hex")
    head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != producer_exact:
        raise RuntimeError("calibration checkout does not match producer exact")
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
    )
    if status:
        raise RuntimeError("calibration checkout must be clean")


def _calibration_rows_payload(rows) -> list[dict[str, Any]]:
    payload = []
    for row in rows:
        z_hex = None
        if row.method_complete and row.pre_score is not None:
            try:
                z_hex = encode_binary64(row.z, "z")
            except (TypeError, ValueError):
                z_hex = None
        payload.append(
            {
                "geometry_status": row.geometry_outcome,
                "image_digest": row.image_digest,
                "method_complete": row.method_complete,
                "operational_error": row.operational_error,
                "post_m_be_hex": (
                    None if row.post_score is None else encode_binary64(row.post_score, "post_m")
                ),
                "pre_m_be_hex": (
                    None if row.pre_score is None else encode_binary64(row.pre_score, "pre_m")
                ),
                "roster_index": row.roster_index,
                "source_stratum": row.source_stratum,
                "unit_id": row.unit_id,
                "z_be_hex": z_hex,
            }
        )
    return payload


def _replay_rows_payload(rows) -> list[dict[str, Any]]:
    return [
        {
            "image_digest": row.image_digest,
            "method_complete": row.method_complete,
            "operational_error": row.operational_error,
            "positive": row.positive,
            "post_m_be_hex": (
                None if row.post_score is None else encode_binary64(row.post_score, "post_m")
            ),
            "pre_m_be_hex": (
                None if row.pre_score is None else encode_binary64(row.pre_score, "pre_m")
            ),
            "recovered": row.recovered,
            "roster_index": row.roster_index,
            "route": row.route,
            "source_stratum": row.source_stratum,
            "unit_id": row.unit_id,
        }
        for row in rows
    ]


def _input_summary(
    roster: BlindCalibrationRoster,
    evidence: dict[str, str],
    roster_file_sha256: str,
    runtime_config_file_sha256: str,
    key: bytes,
) -> dict[str, Any]:
    return {
        "calibration_key_digest": public_key_digest(key),
        "disjoint_evidence_digest": hashlib.sha256(
            stable_json_bytes(dict(sorted(evidence.items())))
        ).hexdigest(),
        "disjoint_from": list(roster.disjoint_from),
        "roster_digest": roster.digest,
        "roster_file_sha256": roster_file_sha256,
        "runtime_config_file_sha256": runtime_config_file_sha256,
        "source_strata": dict(sorted(Counter(unit.source_stratum for unit in roster.units).items())),
    }


def _config_summary(config: dict[str, str]) -> dict[str, Any]:
    return {
        "automatic_retries": 0,
        "content_model_id": config["content_model_id"],
        "decision_rule": "positive_iff_m_strictly_greater_than_tau_blind",
        "device": config["device"],
        "geometry_route": "Geometry-Direct_once_per_current_RGB",
        "preprocess_id": BLIND_PREPROCESS_ID,
        "production_runtime_id": BLIND_PRODUCTION_RUNTIME_ID,
        "scorer_id": BLIND_SCORER_ID,
        "statistic_id": BLIND_STATISTIC_ID,
        "syncseal_checkpoint": config["syncseal_checkpoint"],
        "syncseal_checkpoint_sha256": config["syncseal_checkpoint_sha256"],
        "weighted_asset_repo_path": str(WEIGHTED_ASSET_REPO_PATH),
        "weighted_asset_sha256": _sha256_file(REPO_ROOT / WEIGHTED_ASSET_REPO_PATH),
        "weighted_asset_sidecar_repo_path": str(WEIGHTED_ASSET_SIDECAR_REPO_PATH),
        "weighted_asset_sidecar_sha256": _sha256_file(
            REPO_ROOT / WEIGHTED_ASSET_SIDECAR_REPO_PATH
        ),
    }


def _base_calibration_result(*, producer_exact: str) -> dict[str, Any]:
    return {
        "calibration_rows": [],
        "candidate_tau_blind_be_hex": None,
        "claim_ceiling": CALIBRATION_CLAIM_CEILING,
        "config_summary": None,
        "denominator": BLIND_DEV_DENOMINATOR,
        "error": None,
        "fresh_replay_false_positives": None,
        "fresh_replay_rows": [],
        "fresh_replay_zero_of_256": False,
        "frozen_tau_blind_be_hex": None,
        "input_summary": None,
        "producer_exact": producer_exact,
        "schema_version": CALIBRATION_RESULT_SCHEMA,
        "science_denominator": 0,
        "status": "OPERATIONAL_BLOCKED",
        "threshold_candidate_ready": False,
        "threshold_candidate_sha256": None,
    }


def calibrate_and_record(
    roster_path: str | Path,
    key_path: str | Path,
    runtime_config_path: str | Path,
    threshold_candidate_path: str | Path,
    result_output_path: str | Path,
    *,
    producer_exact: str,
) -> tuple[Path, Path | None, str]:
    """Run one formal calibration attempt and retain success or failure create-only."""

    threshold_candidate = Path(threshold_candidate_path)
    result_output = Path(result_output_path)
    if threshold_candidate.exists():
        raise FileExistsError("blind threshold candidate is create-only")
    if result_output.exists():
        raise FileExistsError("blind calibration result is create-only")
    result = _base_calibration_result(producer_exact=producer_exact)
    rows = ()
    replay = ()
    try:
        config, runtime_config_file_sha256 = load_runtime_config(runtime_config_path)
        result["config_summary"] = _config_summary(config)
        roster, evidence, roster_file_sha256 = load_roster_inputs(roster_path)
        key = normalize_detection_key(Path(key_path).read_bytes())
        result["input_summary"] = _input_summary(
            roster, evidence, roster_file_sha256, runtime_config_file_sha256, key
        )
        _verify_producer_checkout(producer_exact)

        def disk_image_loader(image_ref: str) -> Image.Image:
            with Image.open(image_ref) as opened:
                return opened.copy()

        cached_images = validate_unique_current_images(roster, disk_image_loader)
        public_assets = build_production_runtime(
            REPO_ROOT, config, hf_token=os.environ.get("HF_TOKEN", "")
        )

        def calibration_image_loader(image_ref: str) -> Image.Image:
            return cached_images[image_ref].copy()

        rows, replay, asset = evaluate_threshold_with_runtime(
            roster,
            key,
            public_assets,
            calibration_image_loader,
            producer_exact=producer_exact,
            replay_image_loader=disk_image_loader,
        )
        result["candidate_tau_blind_be_hex"] = asset.payload["tau_blind_be_hex"]
        result["frozen_tau_blind_be_hex"] = asset.payload["tau_blind_be_hex"]
        result["fresh_replay_false_positives"] = 0
        result["fresh_replay_zero_of_256"] = True
        threshold_candidate.parent.mkdir(parents=True, exist_ok=True)
        with threshold_candidate.open("xb") as sink:
            sink.write(asset.json_bytes)
        if _sha256_file(threshold_candidate) != hashlib.sha256(asset.json_bytes).hexdigest():
            raise RuntimeError("threshold candidate durable readback differs")
        result["status"] = "CALIBRATION_COMPLETE_THRESHOLD_CANDIDATE_READY"
        result["threshold_candidate_sha256"] = hashlib.sha256(asset.json_bytes).hexdigest()
        result["threshold_candidate_ready"] = True
    except ThresholdFreezeBlocked as blocked:
        rows = blocked.calibration_rows
        replay = blocked.replay_rows
        result["status"] = blocked.status
        result["error"] = str(blocked)
        if rows and all(row.method_complete for row in rows):
            try:
                result["candidate_tau_blind_be_hex"] = encode_binary64(
                    max(row.z for row in rows), "candidate_tau_blind"
                )
            except (TypeError, ValueError):
                pass
        if replay:
            result["fresh_replay_false_positives"] = sum(row.positive for row in replay)
    except Exception as error:
        result["status"] = "OPERATIONAL_BLOCKED"
        result["error"] = f"{type(error).__name__}: {error}"
    result["calibration_rows"] = _calibration_rows_payload(rows)
    result["fresh_replay_rows"] = _replay_rows_payload(replay)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    with result_output.open("xb") as sink:
        sink.write(stable_json_bytes(result))
    return (
        result_output,
        threshold_candidate if result["threshold_candidate_ready"] else None,
        result["status"],
    )


def _walk_weighted_scores(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "weighted_joint" and isinstance(child, (int, float)) and not isinstance(child, bool):
                scalar = float(child)
                if math.isfinite(scalar):
                    yield scalar
            yield from _walk_weighted_scores(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_weighted_scores(child)


def diagnose_existing_artifacts(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Read-only compact diagnostics; these are not calibration or paper evidence."""

    records = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            records.append({"path": str(path), "present": False})
            continue
        payload = _read_json(path)
        scores = tuple(_walk_weighted_scores(payload))
        records.append(
            {
                "path": str(path),
                "present": True,
                "status": payload.get("status") if isinstance(payload, dict) else None,
                "finite_weighted_joint_count": len(scores),
                "finite_weighted_joint_min": min(scores) if scores else None,
                "finite_weighted_joint_max": max(scores) if scores else None,
            }
        )
    return {
        "classification": "read_only_engineering_diagnostic_not_calibration_or_paper_evidence",
        "artifacts": records,
    }


def _load_factory(spec: str):
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("runtime factory must be module:callable")
    module_name, attribute = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise TypeError("runtime factory must resolve to a callable")
    return factory


def run_callback(
    manifest_path: str | Path,
    key_path: str | Path,
    threshold_path: str | Path,
    runtime_factory: str,
    output_path: str | Path,
) -> tuple[Path, str, tuple[str, ...]]:
    """Run the fixed image-only N=4 callback once with an injected real runtime."""

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != {"cases", "denominator"}:
        raise ValueError("callback manifest fields differ")
    cases = manifest["cases"]
    if manifest["denominator"] != 4 or not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("callback requires a fixed N=4 manifest")
    required = {
        "direct_positive",
        "geometry_recovered_positive",
        "unwatermarked_geometry_negative",
    }
    labels = {case.get("coverage") for case in cases if isinstance(case, dict)}
    if not required.issubset(labels) or not labels.issubset(required):
        raise ValueError("callback coverage differs")
    threshold = load_threshold_asset(threshold_path)
    factory = _load_factory(runtime_factory)
    public_assets = factory(REPO_ROOT)
    if type(public_assets) is not BlindProductionAssets:
        raise TypeError("runtime factory must return BlindProductionAssets")
    if public_assets.threshold_asset is not None:
        raise ValueError("callback runtime factory must not prebind a threshold")
    public_assets = replace(public_assets, threshold_asset=threshold)
    detection_key = Path(key_path).read_bytes()
    records = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"case_id", "coverage", "image_path"}:
            raise ValueError("callback case fields differ")
        try:
            with Image.open(case["image_path"]) as opened:
                current_rgb = opened.copy()
            record = detect_watermark(current_rgb, detection_key, public_assets)
            records.append(
                {
                    "case_id": case["case_id"],
                    "coverage": case["coverage"],
                    "method_complete": record.method_complete,
                    "operational_error": record.operational_error,
                    "positive": record.positive,
                    "post_m": None if record.post is None else record.post.value,
                    "pre_m": None if record.pre is None else record.pre.value,
                    "recovered": record.recovered,
                    "route": record.route,
                }
            )
        except Exception as error:
            records.append(
                {
                    "case_id": case["case_id"],
                    "coverage": case["coverage"],
                    "method_complete": False,
                    "operational_error": f"{type(error).__name__}: {error}",
                    "positive": False,
                    "post_m": None,
                    "pre_m": None,
                    "recovered": False,
                    "route": "ERROR_FAIL_CLOSED",
                }
            )
    expected = {
        "direct_positive": ("DIRECT_POSITIVE", True, False),
        "geometry_recovered_positive": ("GEOMETRY_RECOVERED", True, True),
        "unwatermarked_geometry_negative": ("GEOMETRY_RECOVERED", False, True),
    }
    operational = tuple(
        case["case_id"]
        for case, record in zip(cases, records, strict=True)
        if not record["method_complete"]
    )
    mismatches = tuple(
        case["case_id"]
        for case, record in zip(cases, records, strict=True)
        if record["method_complete"]
        and (record["route"], record["positive"], record["recovered"])
            != expected[case["coverage"]]
    )
    status = (
        "OPERATIONAL_BLOCKED" if operational
        else "METHOD_FAILED" if mismatches
        else "CALLBACK_N4_PASSED"
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as sink:
        sink.write(
            stable_json_bytes(
                {
                    "claim_ceiling": "engineering_image_only_callback_n4_science_denominator_0",
                    "denominator": 4,
                    "mismatched_case_ids": list(mismatches),
                    "operational_case_ids": list(operational),
                    "records": records,
                    "status": status,
                }
            )
        )
    return output, status, mismatches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("calibrate-and-freeze")
    freeze.add_argument("--roster", required=True)
    freeze.add_argument("--key-file", required=True)
    freeze.add_argument("--runtime-config", required=True)
    freeze.add_argument("--producer-exact", required=True)
    freeze.add_argument("--candidate-output", required=True)
    freeze.add_argument("--result-output", required=True)
    diagnose = sub.add_parser("diagnose-existing")
    diagnose.add_argument("artifacts", nargs="+")
    callback = sub.add_parser("callback")
    callback.add_argument("--manifest", required=True)
    callback.add_argument("--key-file", required=True)
    callback.add_argument("--threshold", required=True)
    callback.add_argument("--runtime-factory", required=True)
    callback.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "calibrate-and-freeze":
        result, threshold, status = calibrate_and_record(
            args.roster,
            args.key_file,
            args.runtime_config,
            args.candidate_output,
            args.result_output,
            producer_exact=args.producer_exact,
        )
        print(
            "CEGWM_BLIND_DETECTION_V1 "
            + stable_json_bytes(
                {
                    "denominator": 256,
                    "disjoint_from": list(BLIND_DEV_DISJOINT_FROM),
                    "result_output": str(result),
                    "status": status,
                    "threshold_candidate": None if threshold is None else str(threshold),
                }
            ).decode("ascii")
        )
        return 0 if threshold is not None else 2 if status == "METHOD_FAILED" else 3
    if args.command == "callback":
        output, status, mismatches = run_callback(
            args.manifest,
            args.key_file,
            args.threshold,
            args.runtime_factory,
            args.output,
        )
        print(
            "CEGWM_BLIND_DETECTION_V1 "
            + stable_json_bytes(
                {
                    "denominator": 4,
                    "mismatched_case_ids": list(mismatches),
                    "output": str(output),
                    "status": status,
                }
            ).decode("ascii")
        )
        return 0 if status == "CALLBACK_N4_PASSED" else 2 if status == "METHOD_FAILED" else 3
    diagnostic = diagnose_existing_artifacts(args.artifacts)
    print("CEGWM_BLIND_DETECTION_V1 " + stable_json_bytes(diagnostic).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
