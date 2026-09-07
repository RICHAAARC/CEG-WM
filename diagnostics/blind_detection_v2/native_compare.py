"""Same-input official SyncSeal / production adapter comparison on CPU or CUDA.

Historical images are diagnostic-only. This does not evaluate content detection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from cegwm.geometry_v7.syncseal import SyncSealTorchScript, _to_tensor
from cegwm.geometry_v7.r1b import rectify_attacked_rgb


def compare(image, adapter, result=None):
    result = {} if result is None else result
    # Construct the native input independently of the adapter conversion.
    native = torch.tensor(np.asarray(image).copy(), device=adapter.device).permute(2, 0, 1)[None].float() / 255
    current = _to_tensor(image, adapter.device)
    start = time.monotonic()
    with torch.no_grad():
        raw = adapter.model.detect(native)
        geometry = adapter.detect_geometry(image)
        result.update(input_max_abs=float((native-current).abs().max()),
                      raw_preds=raw["preds"][0].detach().cpu().tolist(),
                      geometry=asdict(geometry))
        result["raw_corner_max_abs"] = float(np.max(np.abs(raw["preds_pts"][0].detach().cpu().numpy().reshape(4,2) - np.asarray(geometry.raw_syncseal_corners))))
        endpoints = (raw["preds_pts"][0].reshape(4,2)*128+128).round()* (511/255)
        result["native_integer_endpoints"] = endpoints.long().cpu().tolist()
        result["public_float_endpoints"] = ((np.asarray(geometry.observed_corners_in_canonical_normalized)+1)*511/2).tolist()
        unwarped = adapter.model.unwarp(native, raw["preds_pts"], (512, 512))
    public = rectify_attacked_rgb(image, geometry.homography_observed_to_canonical)
    official = unwarped[0].detach().cpu().permute(1, 2, 0).numpy() * 255
    delta = np.asarray(public, dtype=float) - official
    result.update({
        "input_max_abs": float((native-current).abs().max()),
        "raw_preds": raw["preds"][0].detach().cpu().tolist(),
        "raw_corner_max_abs": float(np.max(np.abs(raw["preds_pts"][0].detach().cpu().numpy().reshape(4,2) - np.asarray(geometry.raw_syncseal_corners)))),
        "geometry": asdict(geometry),
        "native_vs_adapter_warp_mae_255": float(np.abs(delta).mean()),
        "native_vs_adapter_warp_rmse_255": float(np.sqrt(np.mean(delta**2))),
        "elapsed_seconds": time.monotonic()-start,
    })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--diagnostic-code", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.diagnostic_code))
    from diagnostic import apply_attack, CONDITIONS, oracle_geometry, transform_points
    adapter = SyncSealTorchScript.from_file(args.checkpoint, device=args.device)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "native_detect_code.txt").write_text(adapter.model.detect.code)
    (args.output / "native_detector_code.txt").write_text(adapter.model.detector.code)
    (args.output / "native_unwarp_code.txt").write_text(adapter.model.unwarp.code)
    rows = []
    for entry in manifest["entries"]:
        unit = entry["sample_id"]
        for arm in ("clean", "watermarked"):
            for condition in CONDITIONS:
                row = {"sample_id": unit, "arm": arm, "condition": condition, "error": None}
                try:
                    with Image.open(args.inputs / arm / (unit+"__"+arm+".png")) as source:
                        image = source.convert("RGB")
                    compare(apply_attack(image, condition), adapter, row)
                    truth = oracle_geometry(condition)["truth_corners"]
                    predicted = row["geometry"]["observed_corners_in_canonical_normalized"]
                    row["corner_rmse_pixels"] = float(np.sqrt(np.mean(np.sum((np.asarray(predicted)-truth)**2, axis=1)))*511/2)
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                with (args.output / "rows.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
        print(f"completed {unit}: {len(rows)}/400", flush=True)
    summary = {"status": "POSTHOC_NATIVE_ADAPTER_DIAGNOSTIC_ONLY", "science_denominator": 0,
               "planned_rows": 400, "rows": len(rows), "errors": sum(r["error"] is not None for r in rows),
               "device": args.device, "torch": torch.__version__, "content_scoring_executed": False}
    for key in ("input_max_abs", "raw_corner_max_abs", "native_vs_adapter_warp_mae_255", "native_vs_adapter_warp_rmse_255"):
        values = [r[key] for r in rows if key in r]
        summary[key] = {"median": float(np.median(values)), "max": max(values)} if values else None
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
