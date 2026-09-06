"""Posthoc rotation diagnostics. No production imports until explicitly requested."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

STATUS = "POSTHOC_DIAGNOSTIC_ONLY"
BASELINE = "e12c7eae91cc36edc5d1a1d96249780a3925eccb"
REPO = Path(__file__).resolve().parents[2]
PRODUCER = "9ec454055c74cf4ed89001387c9f700e9ba5aef0"
TAU = 1.2657276026437319
CONDITIONS = ("clean_no_attack", "rotation_10_bicubic_reflect_center_crop_v1")
STRATA = {"rotation_success": 4, "rotation_near_miss": 24,
          "crop_success_rotation_failure": 24, "rotation_typical_failure": 24,
          "rotation_worst_failure": 24}


def apply_attack(image, condition):
    """Diagnostic replay of the recorded formal attack; not a method change.

    Source: historical formal producer, formal_experiment.py:508-524.
    Reflect padding, center, angle, resampler and crop are unchanged.
    """
    rgb = image.convert("RGB")
    if condition == CONDITIONS[0]:
        return rgb.copy()
    if condition != CONDITIONS[1]:
        raise ValueError("condition outside frozen diagnostic matrix")
    width, height = rgb.size
    theta = math.radians(10.0)
    half_width, half_height = (width-1)/2.0, (height-1)/2.0
    pad_x = max(0, math.ceil(abs(math.cos(theta))*half_width + abs(math.sin(theta))*half_height + 2-half_width))
    pad_y = max(0, math.ceil(abs(math.sin(theta))*half_width + abs(math.cos(theta))*half_height + 2-half_height))
    if pad_x >= width or pad_y >= height:
        raise ValueError("rotation input outside reflect-padding domain")
    padded = np.pad(np.asarray(rgb, dtype=np.uint8), ((pad_y,pad_y),(pad_x,pad_x),(0,0)), mode="reflect")
    rotated = Image.fromarray(padded).rotate(10.0, resample=Image.Resampling.BICUBIC,
        center=(pad_x+half_width,pad_y+half_height), fillcolor=(0,0,0))
    return rotated.crop((pad_x,pad_y,pad_x+width,pad_y+height))


def load_source_rows(root):
    # External, read-only extraction of historical formal records; never Git data.
    return json.loads((Path(root)/"implementation/source_rows.json").read_text())


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(obj, out, indent=2, allow_nan=False)
        out.write("\n")


def transform_points(matrix, points):
    p = np.c_[np.asarray(points, dtype=float), np.ones(len(points))]
    q = p @ np.asarray(matrix, dtype=float).T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-12):
        raise ValueError("nonfinite or infinite projected point")
    return q[:, :2] / q[:, 2, None]


def translation(x, y):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


def oracle_geometry(condition, size=(512, 512)):
    """Actual frozen Pillow rotation, including edge/center coordinate distinction.

    D maps attacked output -> original source in Pillow edge coordinates.
    Pixel-center truth B = T(-.5) D T(.5). The frozen rectifier internally
    treats its pixel coefficients as Pillow coefficients, so its oracle input
    is N D inv(N), while geometric error must use N B inv(N).
    This adaptation is diagnostic-only; no production transform is changed.
    """
    w, h = size
    if size != (512, 512):
        raise ValueError("frozen runtime requires RGB 512x512")
    if condition not in CONDITIONS:
        raise ValueError("condition outside frozen diagnostic matrix")
    theta = math.radians(10.) if condition == CONDITIONS[1] else 0.
    c, s = round(math.cos(theta), 15), round(math.sin(theta), 15)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    d = translation(cx, cy) @ rotation @ translation(-cx, -cy)
    b = translation(-.5, -.5) @ d @ translation(.5, .5)
    n = np.array([[2/(w-1), 0., -1.], [0., 2/(h-1), -1.], [0., 0., 1.]])
    h_truth, h_sampler = n @ b @ np.linalg.inv(n), n @ d @ np.linalg.inv(n)
    corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
    yy, xx = np.mgrid[:h, :w]
    canonical = np.c_[xx.ravel(), yy.ravel()]
    observed = transform_points(np.linalg.inv(b), canonical)
    # Conservative support: bilinear inverse footprint in the cropped image
    # and bicubic forward footprint in the original, unreflected image.
    support = ((observed[:, 0] >= 2) & (observed[:, 0] <= w-3)
               & (observed[:, 1] >= 2) & (observed[:, 1] <= h-3)
               & (canonical[:, 0] >= 3) & (canonical[:, 0] <= w-4)
               & (canonical[:, 1] >= 3) & (canonical[:, 1] <= h-4))
    return {"H_truth_pixel_centers_normalized": h_truth.tolist(),
            "H_oracle_sampler_observed_to_canonical": h_sampler.tolist(),
            "pillow_observed_to_original_edge_coordinates": d.tolist(),
            "truth_corners": transform_points(h_truth, corners).tolist(),
            "support_mask": support.reshape(h, w)}


def audit_inputs(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    index = json.loads((root / "drive_index.json").read_text())
    entries, pairs = manifest["entries"], index["pairs"]
    ids = [e["sample_id"] for e in entries]
    if manifest["status"] != STATUS or manifest["science_denominator"] != 0:
        raise ValueError("diagnostic boundary differs")
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("fixed 100-pair manifest differs")
    if any(not re.fullmatch(r"formal_evaluation_pairs-\d{4}", unit) for unit in ids):
        raise ValueError("unexpected sample identity")
    if Counter(e["selection_stratum"] for e in entries) != STRATA:
        raise ValueError("fixed strata differ")
    if len(pairs) != 100 or {p["sample_id"] for p in pairs} != set(ids):
        raise ValueError("manifest/index pairing differs")
    if manifest["source"]["threshold"] != TAU:
        raise ValueError("formal threshold differs")
    reference = json.loads((Path(__file__).parent/"input_reference.json").read_text())
    if [(e["sample_id"], e["selection_stratum"]) for e in entries] != [(e["sample_id"], e["selection_stratum"]) for e in reference["entries"]]:
        raise ValueError("external roster differs from frozen diagnostic reference")
    source = load_source_rows(root)
    expected_rows = {(unit, condition, role) for unit in ids for condition in CONDITIONS for role in ("negative", "positive")}
    observed_rows = {(r["physical_unit_id"], r["condition"], r["truth_role"]) for r in source["records"]}
    if (source["source_file_id"] != manifest["source"]["result_file_id"] or
        source["identity"]["expected_exact"] != PRODUCER or len(source["records"]) != 400 or
        expected_rows != observed_rows or any(r["threshold"] != TAU for r in source["records"])):
        raise ValueError("original formal reference rows differ")
    rows = []
    for pair in pairs:
        for arm in ("clean", "watermarked"):
            name = pair[arm]["title"]
            if name != pair["sample_id"] + "__" + arm + ".png":
                raise ValueError("unexpected pair-arm mapping")
            path = root / arm / name
            row = {"sample_id": pair["sample_id"], "arm": arm, "status": "DECODED"}
            try:
                with Image.open(path) as im:
                    im.load()
                    row.update(mode=im.mode, size=list(im.size), format=im.format)
                    if im.mode != "RGB" or im.size != (512, 512) or im.format != "PNG":
                        raise ValueError("expected ordinary RGB PNG 512x512")
            except Exception as err:
                row.update(status="FAILED", error=f"{type(err).__name__}: {err}")
            rows.append(row)
    extras = {}
    for arm in ("clean", "watermarked"):
        expected = {p[arm]["title"] for p in pairs}
        extras[arm] = sorted(p.name for p in (root/arm).iterdir() if p.name not in expected)
    failed = sum(r["status"] != "DECODED" for r in rows)
    return {"status": STATUS, "science_denominator": 0, "planned_pairs": 100,
            "planned_images": 200, "decoded_images": 200-failed, "failed_images": failed,
            "extra_files": extras, "rows": rows,
            "original_reference_rows": 400,
            "input_usable": failed == 0 and not any(extras.values()),
            "selection_rule_status": "ROSTER_FROZEN_CENTRAL_RANK_WINDOW_RECONSTRUCTED_ORIGINAL_ALGORITHM_UNVERIFIED"}


class FrozenRuntime:
    """Only image + fixed key/assets enter the real scorer and geometry backend."""
    def __init__(self, runtime_root):
        repo = REPO
        check = subprocess.run(["git", "diff", "--quiet", BASELINE, "--", "src", "configs",
                                "experiments"], cwd=repo)
        if check.returncode != 0:
            raise ValueError("production source differs from authorized main baseline")
        sys.path[:0] = [str(repo/"src"), str(repo)]
        from experiments.run_blind_detection_v1 import build_production_runtime, load_runtime_config
        from cegwm.shared.keys import normalize_detection_key, public_key_digest
        from cegwm.protocol.content_chain import CONTENT_CHAIN_PUBLIC_KEY_DIGEST
        from cegwm.runtime.blind_detection import _detect_core, _score_current_rgb, _raw_h, _geometry_disposition
        from cegwm.geometry_v7.r1b import rectify_attacked_rgb
        root_key, token = os.environ.get("CEG_WM_ROOT_KEY", ""), os.environ.get("HF_TOKEN", "")
        if not root_key or not token:
            raise RuntimeError("CEG_WM_ROOT_KEY and HF_TOKEN are required")
        key = normalize_detection_key(root_key)
        if public_key_digest(key) != CONTENT_CHAIN_PUBLIC_KEY_DIGEST:
            raise ValueError("detection key identity differs")
        config = load_runtime_config(repo)
        pipeline, assets = build_production_runtime(repo, config, hf_token=token, runtime_root=Path(runtime_root))
        self.runtime = {"pipeline": pipeline, "assets": assets, "key": key}
        self._detect, self._score, self._raw_h = _detect_core, _score_current_rgb, _raw_h
        self._disposition = _geometry_disposition
        self.rectify, self.attack = rectify_attacked_rgb, apply_attack

    def score(self, image):
        return float(self._score(image, self.runtime["key"], self.runtime["assets"]).value)

    def detect(self, image):
        return self._detect(image, self.runtime["key"], self.runtime["assets"], TAU)

    def geometry(self, image):
        return self.runtime["assets"].geometry_backend.detect_geometry(image)

    def matrix(self, geometry):
        disposition, error = self._disposition(geometry)
        if disposition != "RAW_H":
            raise ValueError(error or disposition)
        return self._raw_h(geometry)


def finite_score(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite score")
    return value


def diagnose_image(image, condition, backend):
    """Keep production decisions independent of forced/oracle diagnostic scores."""
    truth = oracle_geometry(condition, image.size)
    row = {"pre_score": None, "syncseal_post_score": None, "oracle_post_score": None,
           "reference_score": None, "production": None, "predicted_H": None,
           "predicted_corners": None, "corner_rmse_px": None, "corner_max_error_px": None,
           "forced_post_after_direct_positive": False, "errors": {},
           **{k: v for k, v in truth.items() if k != "support_mask"},
           "valid_support_fraction": float(truth["support_mask"].mean()),
           "diagnostic_adjudication": "UNADJUDICATED"}
    def attempt(name, fn):
        try:
            return fn()
        except Exception as err:
            row["errors"][name] = f"{type(err).__name__}: {err}"
            return None
    row["reference_score"] = attempt("reference", lambda: finite_score(backend.score(image)))
    attacked = attempt("attack", lambda: backend.attack(image, condition))
    if attacked is None:
        return row
    record = attempt("production", lambda: backend.detect(attacked))
    geometry = None
    if record is not None:
        row["pre_score"] = None if record.pre is None else finite_score(record.pre.value)
        row["syncseal_post_score"] = None if record.post is None else finite_score(record.post.value)
        row["production"] = {"route": record.route, "positive": record.positive,
                             "method_complete": record.method_complete,
                             "operational_error": record.operational_error,
                             "normalized_score": (None if not record.method_complete else
                                 max(s.value for s in (record.pre, record.post) if s is not None))}
        geometry = record.geometry
        if not record.method_complete:
            row["errors"]["production_incomplete"] = record.operational_error or "incomplete"
        if record.route == "DIRECT_POSITIVE":
            row["forced_post_after_direct_positive"] = True
            geometry = attempt("forced_geometry", lambda: backend.geometry(attacked))
    if geometry is not None:
        row["geometry_record"] = asdict(geometry)
        matrix = attempt("geometry_matrix", lambda: backend.matrix(geometry))
        if matrix is not None:
            row["predicted_H"] = np.asarray(matrix).tolist()
            predicted = attempt("project_corners", lambda: transform_points(matrix,
                                [(-1, -1), (1, -1), (1, 1), (-1, 1)]))
            if predicted is not None:
                row["predicted_corners"] = predicted.tolist()
                distances = np.linalg.norm((predicted - truth["truth_corners"]) * 255.5, axis=1)
                row["corner_rmse_px"] = float(np.sqrt(np.mean(distances**2)))
                row["corner_max_error_px"] = float(distances.max())
            if row["forced_post_after_direct_positive"]:
                row["syncseal_post_score"] = attempt("forced_post", lambda: finite_score(
                    backend.score(backend.rectify(attacked, matrix))))
    oracle_rgb = attempt("oracle_rectification", lambda: backend.rectify(
        attacked, truth["H_oracle_sampler_observed_to_canonical"]))
    if oracle_rgb is not None:
        row["oracle_post_score"] = attempt("oracle_score", lambda: finite_score(backend.score(oracle_rgb)))
        mask = truth["support_mask"]
        error = np.asarray(oracle_rgb, dtype=float)-np.asarray(image, dtype=float)
        row["oracle_rgb_mae_full"] = float(np.abs(error).mean())
        row["oracle_rgb_mae_valid_support"] = float(np.abs(error[mask]).mean())
    for name in ("reference_score", "pre_score", "syncseal_post_score", "oracle_post_score"):
        value = row[name]
        row[name+"_margin"] = None if value is None else value-TAU
    row["oracle_minus_syncseal"] = (None if row["oracle_post_score"] is None or
        row["syncseal_post_score"] is None else row["oracle_post_score"]-row["syncseal_post_score"])
    return row


def run(root, output, backend):
    root, output = Path(root), Path(output)
    audit = audit_inputs(root)
    if not audit["input_usable"]:
        raise ValueError("input decode/identity audit incomplete; no model diagnostics started")
    if (output/"summary.json").exists():
        raise FileExistsError("completed diagnostic package already exists")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/"manifest.json").read_text())
    sources = load_source_rows(root)["records"]
    source_map = {(r["physical_unit_id"], r["condition"], r["truth_role"]): r for r in sources}
    statuses = Counter()
    retained = []
    execution_kind = "FROZEN_REAL_RUNTIME" if type(backend) is FrozenRuntime else "INJECTED_TEST_BACKEND"
    for e in manifest["entries"]:
        for condition in CONDITIONS:
            for arm in ("clean", "watermarked"):
                identity = dict(sample_id=e["sample_id"], selection_stratum=e["selection_stratum"],
                                condition=condition, arm=arm, truth_role="negative" if arm == "clean" else "positive")
                path = output/(e["sample_id"]+"__"+condition+"__"+arm+".json")
                if path.exists():
                    previous = json.loads(path.read_text())
                    if any(previous.get(k) != v for k,v in identity.items()) or previous.get("threshold") != TAU or previous.get("science_denominator") != 0 or previous.get("producer") != PRODUCER or previous.get("execution_kind") != execution_kind:
                        raise ValueError("existing diagnostic row identity differs")
                    statuses[previous["unit_status"]] += 1
                    retained.append(previous)
                    continue
                try:
                    with Image.open(root/arm/(e["sample_id"]+"__"+arm+".png")) as im:
                        row = diagnose_image(im.copy(), condition, backend)
                    status = "COMPLETE" if not row["errors"] else "PARTIAL_DIAGNOSTIC"
                except Exception as err:
                    row = {"errors": {"unit": f"{type(err).__name__}: {err}"}}
                    status = "FAILED"
                source = source_map[(e["sample_id"], condition, identity["truth_role"])]
                row["original_formal_record"] = source
                production = row.get("production")
                row["replay_matches_original"] = (None if not production or not production["method_complete"] else
                    production["positive"] == source["decision"] and production["route"] == source["route"] and
                    math.isclose(production["normalized_score"], source["normalized_score"], rel_tol=1e-5, abs_tol=1e-5))
                statuses[status] += 1
                retained.append({**identity, **row, "unit_status": status})
                write_new(path,
                          {**identity, **row, "status": STATUS, "unit_status": status,
                           "science_denominator": 0, "threshold": TAU, "producer": PRODUCER,
                           "baseline_main": BASELINE,
                           "execution_kind": execution_kind})
    # Paired gaps are explanatory measurements, never an input to detection.
    paired = []
    for e in manifest["entries"]:
        for condition in CONDITIONS:
            arms = {r["arm"]:r for r in retained if r["sample_id"] == e["sample_id"] and r["condition"] == condition}
            comparison = {"sample_id": e["sample_id"], "selection_stratum": e["selection_stratum"], "condition": condition}
            for key in ("reference_score", "pre_score", "syncseal_post_score", "oracle_post_score"):
                pos, neg = arms["watermarked"].get(key), arms["clean"].get(key)
                comparison[key+"_positive_minus_negative"] = None if pos is None or neg is None else pos-neg
            paired.append(comparison)
    # May exist if interrupted after this derived file but before final summary.
    paired_path = output/"paired_differences.json"
    paired_payload = {"status": STATUS, "science_denominator": 0, "planned_pair_conditions": 200, "rows": paired}
    if paired_path.exists():
        if json.loads(paired_path.read_text()) != paired_payload:
            raise ValueError("existing derived paired table differs")
    else:
        write_new(paired_path, paired_payload)
    write_new(output/"summary.json", {"status": STATUS, "science_denominator": 0,
        "planned_pairs": 100, "planned_image_condition_rows": 400, "unit_status_counts": dict(statuses),
        "conditions": CONDITIONS, "strata": STRATA,
        "claim_ceiling": "posthoc diagnostic only; no population effect or formal metric",
        "producer": PRODUCER, "baseline_main": BASELINE, "threshold": TAU, "execution_kind": execution_kind})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="external Drive diagnostic data/cache directory")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-inputs")
    audit.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run-diagnostic")
    execute.add_argument("--runtime-root", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--execute-authorized-diagnostic", action="store_true", required=True,
                         help="use only after separate real-runtime execution authorization")
    args = parser.parse_args()
    if args.command == "audit-inputs":
        result = audit_inputs(args.root)
        write_new(args.output, result)
        print(json.dumps({k:v for k,v in result.items() if k != "rows"}))
    else:
        if not audit_inputs(args.root)["input_usable"] or (args.output/"summary.json").exists():
            raise ValueError("inputs incomplete or package already complete; runtime not loaded")
        for path in (args.output, args.runtime_root, args.root):
            if path.resolve().is_relative_to(REPO):
                raise ValueError("data, runtime assets and output must remain outside Git worktree")
        run(args.root, args.output, FrozenRuntime(args.runtime_root))


if __name__ == "__main__":
    main()
