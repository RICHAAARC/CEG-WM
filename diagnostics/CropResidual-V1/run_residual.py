"""Read-only saved-image diagnostic. No generator, denoiser, or SyncSeal inference."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

EXACT = "d9ef4adb4f8424f1102f577bb76347fbba112bd9"
MODEL = "stabilityai/stable-diffusion-3.5-medium"
TAU = 1.5545224858704942
ROLES = ("positive", "negative")
IDS = tuple(f"v2-formal_evaluation_pairs-{i:04d}" for i in range(1, 9))
ATOL, RTOL = 1e-5, 1e-4


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bind(path, data):
    if Path(path).exists():
        if read(path) != data:
            raise RuntimeError(f"Resume identity mismatch: {Path(path).name}; do not mix runs")
    else:
        write(path, data)


def error_data(error):
    trace = "".join(traceback.format_exception(error))
    for name in ("HF_TOKEN", "CEG_WM_ROOT_KEY"):
        if os.environ.get(name):
            trace = trace.replace(os.environ[name], "[REDACTED]")
    return {"error_type": type(error).__name__, "traceback_redacted": trace}


def cases():
    result = [{"id": "oracle_replay", "kind": "oracle"},
              {"id": "predicted_replay", "kind": "predicted"},
              {"id": "oracle_uncorrected_pixel_convention", "kind": "convention"},
              {"id": "half_prediction_residual", "kind": "half_residual"}]
    for axis in ("x", "y"):
        for value in (-4., -2., -.5, .5, 2., 4.):
            result.append({"id": f"translation_{axis}_{value:+g}px", "kind": "translation",
                           "axis": axis, "value": value})
    for value in (-.04, -.02, -.005, .005, .02, .04):
        result.append({"id": f"scale_{value * 100:+g}pct", "kind": "scale", "value": value})
    return result


def normalization():
    import numpy as np
    return np.array([[2 / 511, 0., -1.], [0., 2 / 511, -1.], [0., 0., 1.]])


def matrix_for(case, crop, predicted):
    """All perturbations compose BEFORE one sampling from saved attacked RGB."""
    import numpy as np
    base = np.asarray(crop["sampler_h_observed_to_canonical"], dtype=float)
    kind = case["kind"]
    if kind == "oracle":
        return base
    if kind == "predicted":
        return np.asarray(predicted, dtype=float)
    if kind == "convention":
        return np.asarray(crop["physical_h_observed_to_canonical"], dtype=float)
    if kind == "half_residual":
        # A single predeclared diagnostic point along the actual predicted-H error.
        # No interpolation weight is selected using detection scores.
        p = np.asarray(predicted, dtype=float)
        return .5 * base / base[2, 2] + .5 * p / p[2, 2]
    residual = np.eye(3)
    if kind == "translation":
        residual[0 if case["axis"] == "x" else 1, 2] = case["value"]
    elif kind == "scale":
        scale = 1 + case["value"]
        residual[0, 0] = residual[1, 1] = scale
        residual[0, 2] = residual[1, 2] = 256 * (1 - scale)
    else:
        raise ValueError("Unknown fixed diagnostic case")
    n = normalization()
    return n @ residual @ np.linalg.inv(n) @ base


def geometry_metrics(matrix, crop):
    import numpy as np
    n = normalization()
    edge = np.linalg.inv(n) @ matrix @ n
    plus = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    center = np.linalg.inv(plus) @ edge @ plus
    actual = n @ center @ np.linalg.inv(n)
    truth = np.asarray(crop["physical_h_observed_to_canonical"])
    points = np.array([[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.], [0., 0., 1.]]).T
    a, b = actual @ points, truth @ points
    delta = (a[:2] / a[2] - b[:2] / b[2]) * 511 / 2
    return {"effective_corner_rmse_px": float(np.sqrt(np.mean(np.sum(delta[:, :4] ** 2, axis=0)))),
            "effective_center_shift_xy_px": delta[:, 4].tolist()}


def load_saved_image(source, artifact):
    from PIL import Image
    path = Path(artifact["path"]).resolve()
    if not path.is_relative_to(source.resolve()):
        raise RuntimeError("Saved image points outside SOURCE directory")
    if sha(path) != artifact["sha256"]:
        raise RuntimeError(f"Saved image hash differs: {path.name}")
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (512, 512):
            raise ValueError("Saved input must be RGB 512x512")
        return image.copy()


def inspect_source(source):
    protocol, summary = read(source / "protocol.json"), read(source / "summary.json")
    if summary["status"] != "COMPLETE_DIAGNOSTIC" or protocol["source_exact"] != EXACT:
        raise ValueError("Completed original crop diagnostic required")
    if tuple(u["unit_id"] for u in protocol["roster"]) != IDS or protocol["reference_tau"] != TAU:
        raise ValueError("Source roster or threshold differs")
    indexed = {(r["unit_id"], r["role"], r["route"]): r for r in summary["records"]}
    if len(indexed) != 64 or len(summary["records"]) != 64:
        raise ValueError("Expected 64 unique source route records")
    for uid in IDS:
        for role in ROLES:
            for route in ("clean", "crop_direct", "oracle_h", "predicted_h"):
                row = indexed[uid, role, route]
                if row["status"] != "SCORED":
                    raise ValueError("This fixed diagnostic requires all saved baseline routes")
                load_saved_image(source, row["image"])
    return protocol, indexed


class Reader:
    """Only the VAE plus the unchanged LF/HF arithmetic from the pinned source."""
    def __init__(self, repo, source):
        import torch
        from diffusers import AutoencoderKL
        from diffusers.image_processor import VaeImageProcessor
        from cegwm.method.lf import (FrozenLFPublicAssets, LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
            LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID)
        from cegwm.method.hf import FrozenHFPublicAssets
        from cegwm.method.content_whitening import FrozenContentWhiteningLFPublicAssets
        from experiments.run_blind_detection_v1 import load_whitening_asset_semantic, load_weighted_asset_semantic
        from cegwm.shared.keys import normalize_detection_key
        from cegwm.runtime.content_weighted_joint_sd35 import derive_stability_wrong_keys
        revision = read(source / "model_revisions.json")[MODEL]
        self.vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", revision=revision,
            torch_dtype=torch.float16, token=os.environ["HF_TOKEN"]).to("cuda").eval()
        # Exactly StableDiffusion3Pipeline v0.39.0's image_processor construction.
        self.processor = VaeImageProcessor(vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1))
        lf = FrozenLFPublicAssets(self.vae, self.processor, MODEL + ":image_processor",
            LF_BALANCED_BLOCKS_CARRIER_METHOD_ID, LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
            LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID)
        self.lf = FrozenContentWhiteningLFPublicAssets(lf, load_whitening_asset_semantic(repo))
        self.hf = FrozenHFPublicAssets(self.vae, self.processor, MODEL + ":image_processor")
        self.weighted = load_weighted_asset_semantic(repo)
        self.key = normalize_detection_key(os.environ["CEG_WM_ROOT_KEY"])
        self.keys = (self.key, *derive_stability_wrong_keys(self.key))

    def observe(self, image):
        from cegwm.runtime.observation import encode_final_rgb_image
        return encode_final_rgb_image(image, self.processor, self.vae)

    def score(self, observation):
        from cegwm.runtime.blind_scoring_v2 import _lf, _hf
        from cegwm.method.content_weighted_joint import weighted_joint_score
        from cegwm.method.blind_detection import statistic_from_weighted_scores
        labels = ("registered", *(f"wrong_{i:02d}" for i in range(16)))
        pairs = [(_lf(observation, key, self.lf), _hf(observation, key, self.hf)) for key in self.keys]
        branches = {"lf": {label: p[0] for label, p in zip(labels, pairs)},
                    "hf": {label: p[1] for label, p in zip(labels, pairs)},
                    "weighted_joint": {label: weighted_joint_score(*p, self.weighted) for label, p in zip(labels, pairs)}}
        stat = statistic_from_weighted_scores(branches["weighted_joint"])
        return {"branches": branches, "statistic": asdict(stat), "margin": stat.value - TAU,
                "above_reference_tau": stat.value > TAU}


def replay_comparison(current, old):
    import numpy as np
    diffs, passed = {}, True
    for branch in ("lf", "hf", "weighted_joint"):
        a = list(current["branches"][branch].values())
        b = list(old["branches"][branch].values())
        diffs[branch] = float(np.max(np.abs(np.asarray(a) - b)))
        passed &= bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))
    passed &= bool(np.isclose(current["statistic"]["value"], old["statistic"]["value"], atol=ATOL, rtol=RTOL))
    return {"passed": passed, "max_abs_branch_differences": diffs,
            "statistic_difference": current["statistic"]["value"] - old["statistic"]["value"],
            "atol": ATOL, "rtol": RTOL}


def observation_metrics(image, oracle_image, observation, oracle_observation):
    import numpy as np
    a, b = np.asarray(image, dtype=float) / 255, np.asarray(oracle_image, dtype=float) / 255
    z, z0 = observation.detach().float().cpu().numpy(), oracle_observation.detach().float().cpu().numpy()
    def relative(x, y):
        return float(np.linalg.norm(x - y) / max(float(np.linalg.norm(y)), 1e-12))
    return {"rgb_rmse_full": float(np.sqrt(np.mean((a-b)**2))),
            "rgb_rmse_fixed_inner": float(np.sqrt(np.mean((a[64:448,64:448]-b[64:448,64:448])**2))),
            "vae_relative_l2_full": relative(z, z0),
            "vae_relative_l2_fixed_inner": relative(z[...,8:56,8:56], z0[...,8:56,8:56])}


def summarize(output):
    rows, counts = [], {}
    for uid in IDS:
        for role in ROLES:
            for case in cases():
                path = output / "records" / uid / role / (case["id"] + ".json")
                row = read(path) if path.exists() else {"status": "PENDING"}
                rows.append({**row, "unit_id": uid, "role": role, "case": case["id"]})
    for role in ROLES:
        for case in cases():
            subset = [r for r in rows if r["role"] == role and r["case"] == case["id"]]
            scored = [r for r in subset if r["status"] == "SCORED"]
            counts[f"{role}/{case['id']}"] = {"planned": 8, "scored": len(scored),
                "above_reference_tau": sum(r.get("above_reference_tau", False) for r in scored),
                "operational_error": sum(r["status"] == "OPERATIONAL_ERROR" for r in subset),
                "pending": sum(r["status"] == "PENDING" for r in subset),
                "mean_margin": sum(r["margin"] for r in scored) / len(scored) if scored else None}
    result = {"status": "COMPLETE_DIAGNOSTIC" if all(r["status"] == "SCORED" for r in rows) else "INCOMPLETE",
              "source_pairs": 8, "cases_per_image": len(cases()), "planned_scores": len(rows),
              "claim_ceiling": "fixed_grid_sensitivity_diagnosis_not_blind_recovery_or_calibrated_FPR",
              "counts": counts, "records": rows}
    write(output / "summary.json", result)
    cols = ["unit_id", "role", "case", "status", "margin", "delta_score_vs_oracle",
            "delta_registered_vs_oracle", "delta_wrongmax_vs_oracle", "delta_hf_vs_oracle",
            "delta_lf_vs_oracle", "oracle_above_reference_tau", "lost_vs_oracle", "effective_corner_rmse_px",
            "rgb_rmse_full", "rgb_rmse_fixed_inner", "vae_relative_l2_full", "vae_relative_l2_fixed_inner"]
    tmp = output / "readout.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in cols} for r in rows)
    os.replace(tmp, output / "readout.csv")
    return result


def run(source, output, repo):
    import numpy as np
    import torch
    from cegwm.geometry_v7.r1b import rectify_attacked_rgb
    from cegwm.shared.keys import normalize_detection_key, public_key_digest
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError("SOURCE and OUTPUT must be separate non-overlapping directories")
    protocol, old = inspect_source(source)  # Read-only verification BEFORE any inference.
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != EXACT:
        raise ValueError("Historical source revision differs")
    if subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--"], text=True):
        raise ValueError("Tracked historical source has modifications")
    if public_key_digest(normalize_detection_key(os.environ["CEG_WM_ROOT_KEY"])) != protocol["key_digest"]:
        raise ValueError("Detection key differs from saved run")
    identity = {"source": str(source.resolve()), "source_protocol_sha": sha(source / "protocol.json"),
                "source_summary_sha": sha(source / "summary.json"), "source_model_revisions_sha": sha(source / "model_revisions.json"),
                "source_environment_sha": sha(source / "environment.json"), "script_sha": sha(__file__),
                "code_exact": EXACT, "cases": cases(), "tau_reference_only": TAU, "atol": ATOL, "rtol": RTOL}
    bind(output / "protocol.json", identity)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA VAE reader required; no image generation fallback")
    env = {"python": platform.python_version(), "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
           "packages": {p: version(p) for p in ("torch", "torchvision", "diffusers", "transformers", "numpy", "Pillow", "huggingface_hub", "safetensors")}}
    bind(output / "environment.json", env)
    if summarize(output)["status"] == "COMPLETE_DIAGNOSTIC":
        print("Already complete; no scoring repeated.", flush=True)
        return
    reader = Reader(repo, source)
    crop = protocol["crop"]
    references = {}

    def process(uid, role, case, attacked, oracle_image, oracle_z, oracle_score):
        path = output / "records" / uid / role / (case["id"] + ".json")
        if path.exists() and read(path)["status"] == "SCORED":
            return read(path)
        attempt = output / "attempts" / uid / role / case["id"] / (uuid.uuid4().hex + ".json")
        write(attempt, {"status": "STARTED", "time": time.time()})
        try:
            predicted = old[uid, role, "predicted_h"]["h_passed_to_frozen_renderer"]
            matrix = matrix_for(case, crop, predicted)
            image = rectify_attacked_rgb(attacked, matrix.tolist())
            if case["kind"] in ("oracle", "predicted"):
                previous = old[uid, role, "oracle_h" if case["kind"] == "oracle" else "predicted_h"]
                if not np.array_equal(np.asarray(image), np.asarray(load_saved_image(source, previous["image"]))):
                    raise RuntimeError("Replay RGB differs from saved route; stop before residual tests")
            z = oracle_z if case["kind"] == "oracle" else reader.observe(image)
            scored = reader.score(z)
            replay = replay_comparison(scored, previous) if case["kind"] in ("oracle", "predicted") else None
            if replay is not None and not replay["passed"]:
                write(attempt, {"status": "REPLAY_MISMATCH", "replay": replay, **scored})
                raise RuntimeError("VAE-only reader replay mismatch; residual diagnosis halted")
            a, b = scored["statistic"], oracle_score["statistic"]
            result = {"status": "SCORED", "h_passed_to_renderer": matrix.tolist(), **scored,
                      **geometry_metrics(matrix, crop), **observation_metrics(image, oracle_image, z, oracle_z),
                      "replay": replay, "delta_score_vs_oracle": a["value"] - b["value"],
                      "delta_registered_vs_oracle": a["registered_weighted_joint"] - b["registered_weighted_joint"],
                      "delta_wrongmax_vs_oracle": a["wrong_key_max"] - b["wrong_key_max"],
                      "delta_hf_vs_oracle": scored["branches"]["hf"]["registered"] - oracle_score["branches"]["hf"]["registered"],
                      "delta_lf_vs_oracle": scored["branches"]["lf"]["registered"] - oracle_score["branches"]["lf"]["registered"],
                      "oracle_above_reference_tau": oracle_score["above_reference_tau"],
                      "lost_vs_oracle": oracle_score["above_reference_tau"] and not scored["above_reference_tau"]}
            write(attempt, result)
            write(path, result)
            print(f"{uid[-4:]} {role} {case['id']}: margin={scored['margin']:.4f}", flush=True)
            return result
        except Exception as error:
            result = {"status": "OPERATIONAL_ERROR", **error_data(error)}
            # Keep detailed replay values if available, alongside the final error.
            if attempt.exists() and read(attempt).get("status") == "REPLAY_MISMATCH":
                result["replay_failure"] = read(attempt)
            write(attempt, result)
            write(path, result)
            summarize(output)
            raise

    # Complete ALL 32 replay controls before starting any perturbation inference.
    for uid in IDS:
        for role in ROLES:
            attacked = load_saved_image(source, old[uid, role, "crop_direct"]["image"])
            oracle_image = load_saved_image(source, old[uid, role, "oracle_h"]["image"])
            oracle_z = reader.observe(oracle_image)
            oracle_score = reader.score(oracle_z)
            check = replay_comparison(oracle_score, old[uid, role, "oracle_h"])
            if not check["passed"]:
                write(output / "replay_gate.json", {"status": "FAILED", "unit_id": uid, "role": role, "check": check})
                raise RuntimeError("Oracle reader replay mismatch; no residual tests started")
            references[uid, role] = (attacked, oracle_image, oracle_z, oracle_score)
            for case in cases()[:2]:
                process(uid, role, case, *references[uid, role])
            summarize(output)
    write(output / "replay_gate.json", {"status": "PASSED", "planned": 32, "atol": ATOL, "rtol": RTOL})
    for uid in IDS:
        for role in ROLES:
            for case in cases()[2:]:
                process(uid, role, case, *references[uid, role])
            summarize(output)
    result = summarize(output)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    args = p.parse_args()
    # Guard this even for error reporting: never create a file inside SOURCE.
    if args.output.resolve().is_relative_to(args.source.resolve()) or args.source.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("SOURCE and OUTPUT must not overlap")
    sys.path[:0] = [str(args.repo), str(args.repo / "src")]
    try:
        run(args.source, args.output, args.repo)
    except Exception as error:
        write(args.output / "last_failure.json", error_data(error))
        raise


if __name__ == "__main__":
    main()
