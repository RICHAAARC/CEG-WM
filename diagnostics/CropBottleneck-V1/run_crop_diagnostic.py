"""Eight-source crop diagnosis. Reuses the pinned paper writer/reader unchanged."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

EXACT = "d9ef4adb4f8424f1102f577bb76347fbba112bd9"
ROUTES = ("clean", "crop_direct", "predicted_h", "oracle_h")
ROLES = ("positive", "negative")
MODELS = ("stabilityai/stable-diffusion-3.5-medium", "facebook/dinov2-small")
TAU = 1.5545224858704942
TERMINAL = ("SCORED", "NO_VALID_H")


def public_error(error):
    message = "".join(traceback.format_exception(error))
    for name in ("HF_TOKEN", "CEG_WM_ROOT_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return {"type": type(error).__name__, "traceback_redacted": message}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def bind(path, value):
    """Reject mixed protocol/assets/environment instead of silently resuming."""
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError(f"Resume identity differs: {Path(path).name}; use a new RUN_ID")
    else:
        write(path, value)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)
    return {"path": str(path), "sha256": digest(path)}


def verify_file(record):
    if digest(record["path"]) != record["sha256"]:
        raise RuntimeError("Saved artifact hash differs; refusing mixed resume")


def crop_geometry(size=512):
    """Physical pixel centers and PIL-edge sampler coordinates, separately."""
    import numpy as np
    extent = round(size * math.sqrt(0.80))
    left = (size - extent) // 2
    scale = extent / size
    # PIL resize: original center = left + (observed_center+.5)*scale-.5.
    center = np.array([[scale, 0., left + .5 * scale - .5],
                       [0., scale, left + .5 * scale - .5], [0., 0., 1.]])
    edge = np.array([[scale, 0., left], [0., scale, left], [0., 0., 1.]])
    norm = np.array([[2 / (size - 1), 0., -1.],
                     [0., 2 / (size - 1), -1.], [0., 0., 1.]])
    return {
        "crop_box": [left, left, left + extent, left + extent],
        "retained_area_fraction": (extent / size) ** 2,
        "physical_h_observed_to_canonical": (norm @ center @ np.linalg.inv(norm)).tolist(),
        # Frozen renderer passes pixel coefficients straight to PIL. Its sampler
        # evaluates pixel-edge coordinates; this H yields exactly x -> (x-left)/s.
        "sampler_h_observed_to_canonical": (norm @ edge @ np.linalg.inv(norm)).tolist(),
        "oracle_policy": "analytic_crop_inverse_black_bilinear_no_original_pixels",
    }


def corner_error(predicted, truth):
    import numpy as np
    q = np.array([[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]]).T
    a, b = np.asarray(predicted) @ q, np.asarray(truth) @ q
    a, b = a[:2] / a[2], b[:2] / b[2]
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=0))) * 511 / 2)


def summarize(root, roster):
    rows = []
    for unit in roster:
        for role in ROLES:
            for route in ROUTES:
                path = root / "units" / unit["unit_id"] / role / route / "result.json"
                row = read(path) if path.exists() else {"status": "PENDING"}
                rows.append({**row, "unit_id": unit["unit_id"], "role": role, "route": route})
    counts = {}
    for role in ROLES:
        for route in ROUTES:
            subset = [r for r in rows if r["role"] == role and r["route"] == route]
            counts[f"{role}/{route}"] = {
                "planned": len(roster),
                "scored": sum(r["status"] == "SCORED" for r in subset),
                "above_historical_reference_tau": sum(r.get("above_reference_tau", False) for r in subset),
                "no_valid_h": sum(r["status"] == "NO_VALID_H" for r in subset),
                "operational_error": sum(r["status"] == "OPERATIONAL_ERROR" for r in subset),
                "pending": sum(r["status"] == "PENDING" for r in subset),
            }
    complete = all(r["status"] in TERMINAL for r in rows)
    result = {"status": "COMPLETE_DIAGNOSTIC" if complete else "INCOMPLETE",
              "planned_source_pairs": 8, "planned_route_records": 64,
              "claim_ceiling": "engineering_diagnosis_only_not_formal_TPR_or_FPR",
              "tau_role": "historical_reference_only_no_recalibration",
              "counts": counts, "records": rows}
    write(root / "summary.json", result)
    csv_path = root / "readout.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        columns = ("unit_id", "role", "route", "status", "score", "margin_to_reference_tau",
                   "registered_weighted_joint", "wrong_key_max", "lf_registered", "hf_registered",
                   "corner_rmse_physical_pixels")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            stats = row.get("statistic", {})
            branches = row.get("branches", {})
            writer.writerow({**{key: row.get(key) for key in columns}, "score": stats.get("value"),
                             "registered_weighted_joint": stats.get("registered_weighted_joint"),
                             "wrong_key_max": stats.get("wrong_key_max"),
                             "lf_registered": branches.get("lf", {}).get("registered"),
                             "hf_registered": branches.get("hf", {}).get("registered")})
    os.replace(tmp, csv_path)
    return result


def prepare_assets(root):
    """Pin current public model revisions for THIS rerun, not claim old weights."""
    from huggingface_hub import HfApi, snapshot_download
    path = root / "model_revisions.json"
    token = os.environ["HF_TOKEN"]
    if path.exists():
        revisions = read(path)
    else:
        revisions = {name: HfApi().model_info(name, token=token).sha for name in MODELS}
        write(path, revisions)
    cache = Path(os.environ["HF_HUB_CACHE"])
    for name in MODELS:
        patterns = (["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                     "text_encoder_3/*", "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
                     "transformer/*", "vae/*"] if name == MODELS[0] else ["*.json", "*.safetensors"])
        snapshot_download(name, revision=revisions[name], token=token, cache_dir=str(cache),
                          allow_patterns=patterns,
                          ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.ot"])
        # Isolated cache only. The unchanged historical loader requests main.
        # Resolve it to the explicitly downloaded SHA before the OFFLINE child.
        ref = cache / ("models--" + name.replace("/", "--")) / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revisions[name], encoding="ascii")


def score_image(runtime, image):
    from cegwm.runtime.blind_scoring_v2 import score_branches_v2
    from cegwm.method.blind_detection import statistic_from_weighted_scores
    branches, mode = score_branches_v2(image, runtime["key"], runtime["assets"], reuse_observation=True)
    score = statistic_from_weighted_scores(branches["weighted_joint"])
    return {"branches": branches, "statistic": asdict(score), "observation_mode": mode,
            "above_reference_tau": score.value > TAU, "margin_to_reference_tau": score.value - TAU}


def execute(root, repo, runtime_root):
    import torch
    from PIL import Image
    from experiments.run_paper_main_worker_v2 import _build_runtime, _main_pair
    from cegwm.formal_experiment_v2 import load_formal_config, expand_rosters, apply_attack
    from cegwm.runtime.blind_detection import _geometry_disposition, _raw_h
    from cegwm.geometry_v7.r1b import rectify_attacked_rgb
    from cegwm.shared.keys import normalize_detection_key, public_key_digest

    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != EXACT:
        raise RuntimeError("Historical code revision differs")
    if subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--"], text=True):
        raise RuntimeError("Historical tracked source is modified")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no model was loaded")
    config = load_formal_config(repo / "configs/paper_experiment/formal_experiment_v2.json")
    roster = [asdict(u) for u in expand_rosters(repo, config)["formal_evaluation_pairs"][:8]]
    key_digest = public_key_digest(normalize_detection_key(os.environ["CEG_WM_ROOT_KEY"]))
    protocol = {"schema": "crop_bottleneck_v1", "source_exact": EXACT, "roster": roster,
                "script_sha256": digest(__file__), "roles": list(ROLES), "routes": list(ROUTES),
                "reference_tau": TAU, "key_digest": key_digest, "crop": crop_geometry(),
                "negative_role": "same_seed_primary_null_without_syncseal",
                "prediction_policy": "single_continuous_H_original_renderer_no_search",
                "C_is_post_only": True, "original_paper_decision_reconstructible_from_B_C": True}
    bind(root / "protocol.json", protocol)
    summarize(root, roster)
    versions = {name: version(name) for name in
                ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow", "huggingface_hub", "safetensors", "sentencepiece")}
    env = {"packages": versions, "python": platform.python_version(), "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "model_revisions": read(root / "model_revisions.json")}
    bind(root / "environment.json", env)
    if summarize(root, roster)["status"] == "COMPLETE_DIAGNOSTIC":
        for unit in roster:
            unit_root = root / "units" / unit["unit_id"]
            pair = read(unit_root / "pair.json")
            for artifact in pair.values():
                verify_file(artifact)
            for role in ROLES:
                for route in ROUTES:
                    row = read(unit_root / role / route / "result.json")
                    for name in ("image", "valid_support"):
                        if name in row:
                            verify_file(row[name])
        print("Already complete; no generation or scoring repeated.", flush=True)
        return
    runtime = _build_runtime(runtime_root)
    checkpoints = sorted(runtime_root.glob("syncseal-*/syncmodel.jit.pt"), key=lambda p: p.stat().st_mtime)
    bind(root / "syncseal_identity.json", {"sha256": digest(checkpoints[-1])})
    # Copy of the exact executable script for provenance; never store credentials.
    saved_script = root / "run_crop_diagnostic.py"
    if not saved_script.exists():
        saved_script.write_bytes(Path(__file__).read_bytes())
    truth = crop_geometry()
    for unit in roster:
        unit_root = root / "units" / unit["unit_id"]
        pair_path = unit_root / "pair.json"
        print(f"SOURCE {unit['unit_id']}", flush=True)
        if not pair_path.exists():
            attempt = unit_root / "generation_attempts" / uuid.uuid4().hex
            write(attempt / "state.json", {"status": "STARTED", "time": time.time()})
            try:
                clean, marked = _main_pair(runtime, unit["prompt"], unit["seed"])
                pair = {"negative": save_image(attempt / "clean.png", clean),
                        "positive": save_image(attempt / "watermarked.png", marked)}
                write(attempt / "state.json", {"status": "COMPLETE", "time": time.time()})
                write(pair_path, pair)
            except Exception as error:
                write(attempt / "state.json", {"status": "OPERATIONAL_ERROR", **public_error(error)})
                for role in ROLES:
                    for route in ROUTES:
                        write(unit_root / role / route / "result.json",
                              {"status": "OPERATIONAL_ERROR", "stage": "generation", **public_error(error)})
                summarize(root, roster)
                raise
        pair = read(pair_path)
        for role in ROLES:
            verify_file(pair[role])
            with Image.open(pair[role]["path"]) as source:
                original = source.convert("RGB").copy()
            attacked = apply_attack(original, "center_crop_80_restore")
            for route in ROUTES:
                folder = unit_root / role / route
                result_path = folder / "result.json"
                if result_path.exists() and read(result_path)["status"] in TERMINAL:
                    old = read(result_path)
                    for name in ("image", "valid_support"):
                        if name in old:
                            verify_file(old[name])
                    continue
                attempt = folder / "attempts" / uuid.uuid4().hex
                write(attempt / "state.json", {"status": "STARTED", "time": time.time()})
                try:
                    metadata = {}
                    if route == "clean":
                        image = original
                    elif route == "crop_direct":
                        image = attacked
                    else:
                        if route == "predicted_h":
                            geometry = runtime["assets"].geometry_backend.detect_geometry_continuous(attacked)
                            disposition, detail = _geometry_disposition(geometry)
                            metadata["geometry"] = asdict(geometry)
                            write(attempt / "geometry.json", metadata)
                            if disposition == "OPERATIONAL":
                                raise RuntimeError("SyncSeal operational failure")
                            try:
                                if disposition == "INVALID_H":
                                    raise LookupError("No legal predicted H")
                                matrix = _raw_h(geometry)
                            except (LookupError, TypeError, ValueError):
                                result = {"status": "NO_VALID_H", **metadata}
                                write(attempt / "state.json", result)
                                write(result_path, result)
                                summarize(root, roster)
                                continue
                            metadata["corner_rmse_physical_pixels"] = corner_error(
                                matrix, truth["physical_h_observed_to_canonical"])
                        else:
                            matrix = truth["sampler_h_observed_to_canonical"]
                            metadata["truth"] = truth
                        metadata["h_passed_to_frozen_renderer"] = matrix
                        image = rectify_attacked_rgb(attacked, matrix)
                        support = rectify_attacked_rgb(Image.new("RGB", (512, 512), "white"), matrix)
                        metadata["valid_support"] = save_image(attempt / "support.png", support)
                    metadata["image"] = save_image(attempt / "input.png", image)
                    result = {"status": "SCORED", **metadata, **score_image(runtime, image)}
                    write(attempt / "state.json", result)
                    write(result_path, result)
                    print(f"  {role}/{route}: margin={result['margin_to_reference_tau']:.5f}", flush=True)
                except Exception as error:
                    result = {"status": "OPERATIONAL_ERROR", "stage": route, **public_error(error)}
                    write(attempt / "state.json", result)
                    write(result_path, result)
                    summarize(root, roster)
                    raise
                summarize(root, roster)
    print(json.dumps(summarize(root, roster)["counts"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--prepare-assets", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.repo), str(args.repo / "src")]
    try:
        if args.prepare_assets:
            prepare_assets(args.output)
        else:
            execute(args.output, args.repo, args.runtime)
    except Exception as error:
        write(args.output / "last_failure.json", {**public_error(error), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
