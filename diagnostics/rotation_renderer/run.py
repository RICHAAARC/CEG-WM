"""Paired, geometry-only renderer diagnostic on the eight historical R0 CG PNGs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED
from cegwm.geometry_v7.r1a import (
    _pixel_output_to_source, condition_by_id, corner_rmse, truth_correspondences,
)

CONDITIONS = ("core_rotation_neg15", "core_rotation_pos15")
FACTORS = (("black", "bilinear"), ("reflect", "bilinear"),
           ("black", "bicubic"), ("reflect", "bicubic"))
PIXEL_RMSE_FACTOR = 255.5 * math.sqrt(2.0)


def render(image, condition, fill, interpolation):
    """One historical Pillow perspective sample; only padding/kernel vary."""
    if image.mode != "RGB" or image.size != (512, 512):
        raise ValueError("requires original 512x512 RGB")
    if condition not in CONDITIONS or (fill, interpolation) not in FACTORS:
        raise ValueError("outside fixed diagnostic matrix")
    spec = condition_by_id(condition)
    coefficients = list(_pixel_output_to_source(spec.truth_observed_to_canonical))
    source = image
    if fill == "reflect":
        # +/-15deg source extent is within [-58,570]; 128 covers cubic support.
        # Padding copies/reflections only, with no interpolation. Perspective
        # denominator is one for these rotations, so adding pad to c,f maps
        # the SAME source coordinates into the padded array.
        source = Image.fromarray(np.pad(np.asarray(image),
            ((128, 128), (128, 128), (0, 0)), mode="reflect"))
        coefficients[2] += 128
        coefficients[5] += 128
    kernel = {"bilinear": Image.Resampling.BILINEAR,
              "bicubic": Image.Resampling.BICUBIC}[interpolation]
    return source.transform((512, 512), Image.Transform.PERSPECTIVE,
        coefficients, resample=kernel, fillcolor=(0, 0, 0))


def load_inputs(history_path, r0_root):
    history = json.loads(history_path.read_text(encoding="utf-8-sig"))
    inputs = [dict(item) for item in history["r0_input"]["ordered_evaluation_cg_inputs"]]
    if len(inputs) != 8 or len({x["unit_id"] for x in inputs}) != 8:
        raise ValueError("historical eight-image mapping required")
    if history["r0_input"]["selected_residual_strength_multiplier"] != .75:
        raise ValueError("requires historical .75 CG mapping")
    baseline = {}
    for row in history["raw_records"]:
        if row["condition_id"] in CONDITIONS:
            key = (row["unit_id"], row["condition_id"])
            if key in baseline:
                raise ValueError("duplicate historical row")
            baseline[key] = row
    if set(baseline) != {(x["unit_id"], c) for x in inputs for c in CONDITIONS}:
        raise ValueError("historical rotation rows incomplete")
    mapping_path = r0_root / "source_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig")) if mapping_path.exists() else None
    if mapping and mapping["r0_input"]["ordered_evaluation_cg_inputs"] != inputs:
        raise ValueError("local download mapping differs from original R1A mapping")
    for item in inputs:
        local = [x for x in mapping["images"] if x["unit_id"] == item["unit_id"]] if mapping else []
        if mapping and len(local) != 1:
            raise ValueError("local original image mapping missing or duplicate")
        item["local_file"] = local[0]["local_filename"] if mapping else item["path"]
        item["drive_id"] = local[0].get("drive_id") if mapping else None
        path = (r0_root / item["local_file"]).resolve()
        if not path.is_relative_to(r0_root.resolve()):
            raise ValueError("input escaped original artifact root")
        with Image.open(path) as source:
            if source.format != "PNG" or source.mode != "RGB" or source.size != (512, 512):
                raise ValueError("requires saved original RGB PNG")
            source.load()
    return history, inputs, baseline


def is_identity(homography):
    if homography is None:
        return None
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or matrix[2,2] == 0:
        return None
    return bool(np.allclose(matrix/matrix[2,2], np.eye(3), rtol=0, atol=1e-6))


def measure(detector, image, unit, condition, fill, interpolation, historical):
    row = dict(unit_id=unit["unit_id"], original_image_file=unit["path"],
        local_image_file=unit.get("local_file"), drive_id=unit.get("drive_id"),
        condition_id=condition, angle_deg=-15 if condition == CONDITIONS[0] else 15,
        fill=fill, interpolation=interpolation, science_denominator=0,
        errors=[], geometry=None, corner_rmse_px=None, identity=None,
        historical_corner_rmse_px=None, difference_from_history_px=None)
    truth = truth_correspondences(condition_by_id(condition))
    row["truth_observed_corners_in_canonical_normalized"] = truth
    row["identity_baseline_rmse_px"] = corner_rmse(CANONICAL_CORNERS_NORMALIZED, truth) * PIXEL_RMSE_FACTOR
    old = historical.get("prediction_rmse")
    if old is not None:
        row["historical_corner_rmse_px"] = old * PIXEL_RMSE_FACTOR
    row["historical_errors"] = historical.get("errors", [])
    start = time.perf_counter()
    try:
        attacked = render(image, condition, fill, interpolation)
        geometry = detector(attacked)  # exactly one real detection, no truth inputs
        row["geometry"] = asdict(geometry)
        row["identity"] = is_identity(geometry.homography_observed_to_canonical)
        points = geometry.observed_corners_in_canonical_normalized
        if points is not None:
            # Raw coordinate error is informative even for an illegal H.
            row["corner_rmse_px"] = corner_rmse(points, truth) * PIXEL_RMSE_FACTOR
            if row["historical_corner_rmse_px"] is not None:
                row["difference_from_history_px"] = row["corner_rmse_px"] - row["historical_corner_rmse_px"]
        if geometry.error is not None or not geometry.legal:
            row["errors"].append("geometry:" + str(geometry.error or geometry.status.value))
    except Exception as error:
        row["errors"].append(type(error).__name__ + ": " + str(error))
    row["seconds"] = time.perf_counter() - start
    return row


def summarize(rows):
    groups = []
    for fill, interpolation in FACTORS:
        for condition in CONDITIONS:
            chosen = [x for x in rows if (x["fill"], x["interpolation"], x["condition_id"]) == (fill, interpolation, condition)]
            if not chosen:
                continue
            values = [x["corner_rmse_px"] for x in chosen if x["corner_rmse_px"] is not None]
            groups.append(dict(fill=fill, interpolation=interpolation, condition_id=condition,
                denominator=8, recorded=len(chosen), errors=sum(bool(x["errors"]) for x in chosen),
                identity_count=sum(x["identity"] is True for x in chosen),
                median_corner_rmse_px=float(np.median(values)) if len(values) == 8 else None,
                max_corner_rmse_px=max(values) if len(values) == 8 else None))
    return dict(recorded=len(rows), planned=64, science_denominator=0, groups=groups)


def run(args):
    import torch
    from cegwm.geometry_v7.syncseal import SyncSealTorchScript
    history, inputs, baseline = load_inputs(args.history, args.r0_root)
    output = args.output
    phase_path = output / (args.phase + ".jsonl")
    if args.phase == "baseline":
        output.mkdir(parents=True, exist_ok=False)
        prior = []
        factors = FACTORS[:1]
    else:
        prior = [json.loads(x) for x in (output / "baseline.jsonl").read_text().splitlines()]
        expected = {(x["unit_id"], c, x["path"], x["local_file"]) for x in inputs for c in CONDITIONS}
        actual = {(x["unit_id"], x["condition_id"], x["original_image_file"], x["local_image_file"]) for x in prior}
        if len(prior) != 16 or actual != expected or any((x["fill"],x["interpolation"]) != FACTORS[0] for x in prior):
            raise ValueError("preserved first sixteen rows required; never auto retry")
        factors = FACTORS[1:]
    # Exclusive output precedes model initialization; no resume or hidden rerun.
    with phase_path.open("x") as sink:
        torch.set_num_threads(args.threads)
        environment = dict(torch=torch.__version__, pillow=pillow_version,
            device="cpu", threads=args.threads, model_path=str(args.model),
            historical_result=str(args.history), r0_root=str(args.r0_root),
            historical_exact=history.get("exact"), r0_input=history["r0_input"],
            local_inputs=inputs,
            identity_definition="normalized H allclose identity, rtol=0 atol=1e-6; absent H is null",
            center_pixel_convention=255.5, output_size=[512,512], samples_per_image=1,
            center_note="historical matrix passed unchanged to Pillow edge-coordinate sampler; effective pixel-center rotation center 255.0; all four conditions share this convention",
            reflection="numpy reflect, 128px, denominator-one coordinate offset",
            claim="paired geometry diagnostic only; no content scoring")
        with (output / (args.phase + "_environment.json")).open("x") as file:
            json.dump(environment, file, indent=2)
        model = SyncSealTorchScript.from_file(args.model, device="cpu")
        rows = list(prior)
        for fill, interpolation in factors:
            for condition in CONDITIONS:
                for unit in inputs:
                    with Image.open(args.r0_root / unit["local_file"]) as source:
                        image = source.copy()
                    row = measure(model.detect_geometry, image, unit, condition,
                        fill, interpolation, baseline[unit["unit_id"], condition])
                    sink.write(json.dumps(row, allow_nan=False) + "\n")
                    sink.flush()
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ("unit_id", "angle_deg", "fill", "interpolation", "corner_rmse_px", "difference_from_history_px", "errors", "seconds")}), flush=True)
    with (output / (args.phase + "_summary.json")).open("x") as file:
        json.dump(summarize(rows), file, indent=2)
    fields = ["unit_id", "condition_id", "historical_corner_rmse_px"]
    fields += [f"{f}_{i}_rmse_px" for f, i in FACTORS]
    fields += [f"{f}_{i}_errors" for f, i in FACTORS]
    with (output / (args.phase + "_paired.csv")).open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for unit in inputs:
            for condition in CONDITIONS:
                paired = dict(unit_id=unit["unit_id"], condition_id=condition)
                for row in rows:
                    if (row["unit_id"], row["condition_id"]) == (unit["unit_id"], condition):
                        paired["historical_corner_rmse_px"] = row["historical_corner_rmse_px"]
                        prefix = row["fill"] + "_" + row["interpolation"]
                        paired[prefix + "_rmse_px"] = row["corner_rmse_px"]
                        paired[prefix + "_errors"] = json.dumps(row["errors"])
                writer.writerow(paired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "r0-root", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("baseline", "remaining"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
