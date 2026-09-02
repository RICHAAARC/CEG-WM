#!/usr/bin/env python3
"""Prepare/freeze BlindDetection-V1 assets without silently running models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cegwm.method.blind_detection import (  # noqa: E402
    BLIND_DEV_DISJOINT_FROM,
    BlindCalibrationRoster,
    BlindCalibrationUnit,
    build_threshold_asset,
    candidate_tau_blind,
    load_threshold_asset,
    stable_json_bytes,
)
from cegwm.runtime.blind_detection import (  # noqa: E402
    BlindProductionAssets,
    detect_watermark,
    run_development_calibration,
    run_development_full_system_replay,
)
from cegwm.shared.keys import public_key_digest  # noqa: E402


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def load_roster(path: str | Path) -> BlindCalibrationRoster:
    payload = _read_json(path)
    if not isinstance(payload, dict) or set(payload) != {"disjoint_from", "units"}:
        raise ValueError("blind development roster fields differ")
    units = tuple(BlindCalibrationUnit(**unit) for unit in payload["units"])
    return BlindCalibrationRoster(units, tuple(payload["disjoint_from"]))


class ThresholdFreezeBlocked(RuntimeError):
    """Carry every attempted fixed-denominator row when threshold output is blocked."""

    def __init__(self, cause: Exception, calibration_rows, replay_rows) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.calibration_rows = tuple(calibration_rows)
        self.replay_rows = tuple(replay_rows)
        self.status = "METHOD_FAILED" if "0/256" in str(cause) else "OPERATIONAL_BLOCKED"


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

    if type(public_assets) is not BlindProductionAssets:
        raise TypeError("threshold freeze requires BlindProductionAssets")
    if public_assets.threshold_asset is not None:
        raise ValueError("threshold calibration runtime must not contain an earlier threshold")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("blind threshold output is create-only")
    rows = run_development_calibration(roster, key, public_assets, image_loader)
    try:
        tau_blind = candidate_tau_blind(rows, roster)
    except Exception as error:
        raise ThresholdFreezeBlocked(error, rows, ()) from error
    replay = run_development_full_system_replay(
        roster, key, public_assets, image_loader, tau_blind
    )
    try:
        asset = build_threshold_asset(
            rows, roster, replay, producer_exact=producer_exact,
            calibration_key_digest=public_key_digest(key),
        )
    except Exception as error:
        raise ThresholdFreezeBlocked(error, rows, replay) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as sink:
        sink.write(asset.json_bytes)
    return output


def calibrate_and_freeze(
    roster_path: str | Path,
    key_path: str | Path,
    runtime_factory: str,
    output_path: str | Path,
    *,
    producer_exact: str,
) -> Path:
    roster = load_roster(roster_path)
    key = Path(key_path).read_bytes()
    factory = _load_factory(runtime_factory)
    public_assets = factory(REPO_ROOT)
    if type(public_assets) is not BlindProductionAssets:
        raise TypeError("runtime factory must return BlindProductionAssets")

    def image_loader(image_ref: str) -> Image.Image:
        with Image.open(image_ref) as opened:
            return opened.copy()

    return freeze_threshold_with_runtime(
        roster, key, public_assets, image_loader, output_path,
        producer_exact=producer_exact,
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
    freeze.add_argument("--runtime-factory", required=True)
    freeze.add_argument("--producer-exact", required=True)
    freeze.add_argument("--output", required=True)
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
        try:
            output = calibrate_and_freeze(
                args.roster,
                args.key_file,
                args.runtime_factory,
                args.output,
                producer_exact=args.producer_exact,
            )
        except ThresholdFreezeBlocked as blocked:
            print(
                "CEGWM_BLIND_DETECTION_V1 "
                + stable_json_bytes(
                    {
                        "calibration_rows": [asdict(row) for row in blocked.calibration_rows],
                        "error": str(blocked),
                        "replay_rows": [asdict(row) for row in blocked.replay_rows],
                        "status": blocked.status,
                        "threshold_written": False,
                    }
                ).decode("ascii")
            )
            return 2 if blocked.status == "METHOD_FAILED" else 3
        print(
            "CEGWM_BLIND_DETECTION_V1 "
            + stable_json_bytes(
                {
                    "denominator": 256,
                    "disjoint_from": list(BLIND_DEV_DISJOINT_FROM),
                    "output": str(output),
                    "status": "THRESHOLD_FROZEN_AFTER_0_OF_256_REPLAY",
                }
            ).decode("ascii")
        )
        return 0
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
