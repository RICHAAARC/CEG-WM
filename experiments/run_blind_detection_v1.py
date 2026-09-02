#!/usr/bin/env python3
"""Prepare/freeze BlindDetection-V1 assets without silently running models."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    BlindCalibrationRow,
    BlindCalibrationUnit,
    build_threshold_asset,
    load_threshold_asset,
    stable_json_bytes,
)
from cegwm.runtime.blind_detection import (  # noqa: E402
    BlindPublicAssets,
    detect_watermark,
)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def load_roster(path: str | Path) -> BlindCalibrationRoster:
    payload = _read_json(path)
    if not isinstance(payload, dict) or set(payload) != {"disjoint_from", "units"}:
        raise ValueError("blind development roster fields differ")
    units = tuple(BlindCalibrationUnit(**unit) for unit in payload["units"])
    return BlindCalibrationRoster(units, tuple(payload["disjoint_from"]))


def load_rows(path: str | Path) -> tuple[BlindCalibrationRow, ...]:
    rows: list[BlindCalibrationRow] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank calibration row at line {line_number}")
            rows.append(BlindCalibrationRow(**json.loads(line)))
    return tuple(rows)


def freeze_threshold(
    roster_path: str | Path,
    rows_path: str | Path,
    output_path: str | Path,
    *,
    producer_exact: str,
) -> Path:
    """Create the production asset only after complete 256-row validation/replay."""

    roster = load_roster(roster_path)
    rows = load_rows(rows_path)
    asset = build_threshold_asset(rows, roster, producer_exact=producer_exact)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as sink:
        sink.write(asset.json_bytes)
    return output


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
) -> Path:
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
    if not required.issubset(labels):
        raise ValueError("callback coverage differs")
    threshold = load_threshold_asset(threshold_path)
    factory = _load_factory(runtime_factory)
    public_assets = factory(REPO_ROOT)
    if not isinstance(public_assets, BlindPublicAssets):
        raise TypeError("runtime factory must return BlindPublicAssets")
    public_assets = replace(public_assets, threshold_asset=threshold)
    detection_key = Path(key_path).read_bytes()
    records = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"case_id", "coverage", "image_path"}:
            raise ValueError("callback case fields differ")
        with Image.open(case["image_path"]) as opened:
            current_rgb = opened.copy()
        record = detect_watermark(current_rgb, detection_key, public_assets)
        records.append(
            {
                "case_id": case["case_id"],
                "coverage": case["coverage"],
                "error": record.error,
                "positive": record.positive,
                "post_m": None if record.post is None else record.post.value,
                "pre_m": None if record.pre is None else record.pre.value,
                "recovered": record.recovered,
                "route": record.route,
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as sink:
        sink.write(
            stable_json_bytes(
                {
                    "claim_ceiling": "engineering_image_only_callback_n4_science_denominator_0",
                    "denominator": 4,
                    "records": records,
                }
            )
        )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-threshold")
    freeze.add_argument("--roster", required=True)
    freeze.add_argument("--rows", required=True)
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
    if args.command == "freeze-threshold":
        output = freeze_threshold(
            args.roster, args.rows, args.output, producer_exact=args.producer_exact
        )
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
        output = run_callback(
            args.manifest,
            args.key_file,
            args.threshold,
            args.runtime_factory,
            args.output,
        )
        print(
            "CEGWM_BLIND_DETECTION_V1 "
            + stable_json_bytes(
                {"denominator": 4, "output": str(output), "status": "CALLBACK_N4_RECORDED"}
            ).decode("ascii")
        )
        return 0
    diagnostic = diagnose_existing_artifacts(args.artifacts)
    print("CEGWM_BLIND_DETECTION_V1 " + stable_json_bytes(diagnostic).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
