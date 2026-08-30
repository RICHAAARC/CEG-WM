"""G0/G1 runner: real SD3.5 only, RGB-only detection, retained failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cegwm.method.geometry_v4_generative import (
    build_reused_weighted_joint_content_adapter,
    detect_g1_geometry,
    measure_final_rgb,
    rectify_g1_rgb,
    rgb_only_anchor_score,
)
from cegwm.protocol.geometry_v4_generative import G1_MIN_ANCHOR_SCORE, contract_sha256, load_g0_g1_contract
from experiments.content_iss_engine import _load_pipeline_and_assets
from cegwm.runtime.geometry_v4_sd35 import run_sd35_final_latent_pair


def _rgb(image: Image.Image) -> np.ndarray: return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
def _attack(image: Image.Image, name: str) -> Image.Image:
    if name == "identity": return image.copy()
    if name == "rotation_5": return image.rotate(5, resample=Image.Resampling.BICUBIC)
    if name == "scale_0.9":
        w, h = image.size; small = image.resize((round(w*.9), round(h*.9)), Image.Resampling.BICUBIC); canvas = Image.new("RGB", (w,h)); canvas.paste(small, ((w-small.width)//2,(h-small.height)//2)); return canvas
    if name == "translation_0.08_0":
        w,h=image.size; out=Image.new("RGB",(w,h)); out.paste(image,(round(w*.08),0)); return out
    if name == "crop_0.9":
        w,h=image.size; d=round(min(w,h)*.05); return image.crop((d,d,w-d,h-d)).resize((w,h),Image.Resampling.BICUBIC)
    raise ValueError("unfrozen G1 attack")


def _g1_attacked_record(attacked_rgb: np.ndarray, key: object, wrong_key: object) -> dict[str, object]:
    """Evaluate two independent blind arms from the same current attacked RGB."""
    correct = detect_g1_geometry(attacked_rgb, key)
    wrong = detect_g1_geometry(attacked_rgb, wrong_key)
    rectified_correct = None
    if correct["status"] == "RELIABLE" and correct["H_hat"] is not None:
        rectified = rectify_g1_rgb(attacked_rgb, correct["H_hat"])
        rectified_correct = rgb_only_anchor_score(rectified, key)
    passed = bool(
        correct["status"] == "RELIABLE"
        and correct["H_hat"] is not None
        and len(correct["corners_hat"]) == 4
        and int(correct["support"]) >= 6
        and wrong["status"] == "UNRELIABLE"
        and rectified_correct is not None
        and rectified_correct >= G1_MIN_ANCHOR_SCORE
    )
    return {
        "passed": passed,
        "correct_key_geometry": correct,
        "wrong_key_geometry": wrong,
        "correct_key_self_rectified_anchor": rectified_correct,
        "geometry_role": "coordinate_only_never_positive",
    }


def _record(stage: str, seed: int, prompt: str, attack: str, pipeline: object, key: object, wrong_key: object, detector: object) -> dict[str, object]:
    base = {"seed": seed, "prompt": prompt, "attack": attack, "failure": None}
    try:
        pair = run_sd35_final_latent_pair(pipeline, prompt, key, height=512, width=512, generator=torch.Generator(device="cuda").manual_seed(seed))
        attacked = _attack(pair.marked, attack)
        attacked_rgb = _rgb(attacked)
        if stage == "G0":
            observation = measure_final_rgb(_rgb(pair.clean), _rgb(pair.marked), key, wrong_key, detector)
            final_record: dict[str, object] = {"passed": observation.passed, **asdict(observation)}
            attacked_record = {"correct_key_anchor": rgb_only_anchor_score(attacked_rgb, key), "wrong_key_anchor": rgb_only_anchor_score(attacked_rgb, wrong_key)}
        else:
            attacked_record = _g1_attacked_record(attacked_rgb, key, wrong_key)
            try:
                observation = measure_final_rgb(_rgb(pair.clean), _rgb(pair.marked), key, wrong_key, detector)
                final_record = {"passed": observation.passed, **asdict(observation)}
            except Exception as diagnostic_error:
                final_record = {"passed": False, "diagnostic_failure": f"{type(diagnostic_error).__name__}: {diagnostic_error}"}
        return {**base, "content_detector": detector.identities(), "final_rgb": final_record, "attacked_rgb": attacked_record}
    except Exception as error:
        return {**base, "failure": f"{type(error).__name__}: {error}", "final_rgb": None, "attacked_rgb": None}
def run(stage: str, detection_key: object, wrong_key: object, *, repo_root: str | Path, hf_token: str, artifact_root: str | Path | None = None) -> tuple[dict[str, object], ...]:
    contract = load_g0_g1_contract(repo_root)
    if stage not in {"G0", "G1"}: raise ValueError("stage must be G0 or G1")
    if not torch.cuda.is_available(): raise RuntimeError("real SD3.5 G0/G1 requires a CUDA GPU; no RGB proxy is permitted")
    pipeline, assets = _load_pipeline_and_assets(contract["identity"]["model_id"], hf_token)
    pipeline = pipeline.to("cuda")
    content_detector = build_reused_weighted_joint_content_adapter(assets, repo_root)
    roster = contract[stage.lower()]
    records = tuple(_record(stage, seed, prompt, attack, pipeline, detection_key, wrong_key, content_detector) for seed, prompt in zip(roster["seeds"], roster["prompts"], strict=True) for attack in roster["attacks"])
    root = Path(contract["artifact_root"] if artifact_root is None else artifact_root); root.mkdir(parents=True, exist_ok=False)
    path = root / f"{stage.lower()}-records.json"; payload = {"contract_sha256": contract_sha256(repo_root), "stage": stage, "content_detector": content_detector.identities(), "records": records}; path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="ascii")
    (root / f"{stage.lower()}-records.json.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest()+f"  {path.name}\n", encoding="ascii")
    return records


def _checkout_state(repo_root: str | Path) -> tuple[str, str, bool]:
    root = Path(repo_root)
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "HEAD"), git("branch", "--show-current"), git("status", "--porcelain") == ""


def _secret_environment(environ: dict[str, str]) -> tuple[bytes, str]:
    root_key = environ.pop("CEG_WM_ROOT_KEY", "")
    hf_token = environ.pop("HF_TOKEN", "")
    try:
        if not isinstance(root_key, str) or not root_key.strip() or not isinstance(hf_token, str) or not hf_token.strip():
            raise RuntimeError("required Colab secrets are unavailable")
        from cegwm.runtime.content_weighted_joint_sd35 import derive_stability_wrong_keys
        from cegwm.shared.keys import normalize_detection_key
        normalized = normalize_detection_key(root_key)
        return normalized, hf_token
    finally:
        root_key = ""
        environ.pop("CEG_WM_ROOT_KEY", None)


def main(argv: list[str] | None = None, *, environ: dict[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geometry-V4 detached G0/G1 runner")
    parser.add_argument("--stage", choices=("G0", "G1"), required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    args = parser.parse_args(argv)
    try:
        exact, branch, clean = _checkout_state(args.repo_root)
        if exact != args.expected_exact or branch != "" or not clean:
            raise RuntimeError("detached checkout exact or clean state differs")
        env = os.environ if environ is None else environ
        detection_key, hf_token = _secret_environment(env)
        try:
            from cegwm.runtime.content_weighted_joint_sd35 import derive_stability_wrong_keys
            wrong_key = derive_stability_wrong_keys(detection_key)[0]
            records = run(args.stage, detection_key, wrong_key, repo_root=args.repo_root, hf_token=hf_token, artifact_root=args.artifact_root)
        finally:
            hf_token = ""
            env.pop("HF_TOKEN", None)
        gate_name = "final_rgb" if args.stage == "G0" else "attacked_rgb"
        passed = sum(bool(record.get(gate_name, {}).get("passed")) for record in records if isinstance(record.get(gate_name), dict))
        expected = 4 if args.stage == "G0" else 20
        summary = {"stage": args.stage, "source_exact": exact, "clean": clean, "units": len(records), "passed": passed, "expected": expected, "status": "PASS" if passed == expected else "GATE_FAILED"}
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)
        return 0 if summary["status"] == "PASS" else 2
    except Exception as error:
        print(json.dumps({"stage": args.stage, "status": "STOPPED", "error": type(error).__name__}, sort_keys=True, separators=(",", ":")), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
