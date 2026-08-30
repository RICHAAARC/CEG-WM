"""G0/G1 runner: real SD3.5 only, RGB-only detection, retained failures."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from cegwm.method.geometry_v4_generative import measure_final_rgb, rgb_only_anchor_score
from cegwm.protocol.geometry_v4_generative import contract_sha256, load_g0_g1_contract
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
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
def _default_content_detector(rgb: np.ndarray, key: bytes) -> float:
    del key
    return float(np.asarray(rgb, dtype=np.float64).mean())
def _record(seed: int, prompt: str, attack: str, pipeline: object, key: object, wrong_key: object, detector: Callable[[np.ndarray,bytes],float]) -> dict[str, object]:
    base = {"seed": seed, "prompt": prompt, "attack": attack, "failure": None}
    try:
        pair = run_sd35_final_latent_pair(pipeline, prompt, key, height=512, width=512, generator=torch.Generator(device="cuda").manual_seed(seed))
        observation = measure_final_rgb(_rgb(pair.clean), _rgb(pair.marked), key, wrong_key, detector)
        attacked = _attack(pair.marked, attack)
        return {**base, "final_rgb": {"passed": observation.passed, **asdict(observation)}, "attacked_rgb": {"correct_key_anchor": rgb_only_anchor_score(_rgb(attacked), key), "wrong_key_anchor": rgb_only_anchor_score(_rgb(attacked), wrong_key)}}
    except Exception as error:
        return {**base, "failure": f"{type(error).__name__}: {error}", "final_rgb": None, "attacked_rgb": None}
def run(stage: str, detection_key: object, wrong_key: object, *, repo_root: str | Path, hf_token: str, artifact_root: str | Path | None = None, content_detector: Callable[[np.ndarray,bytes],float] = _default_content_detector) -> tuple[dict[str, object], ...]:
    contract = load_g0_g1_contract(repo_root)
    if stage not in {"G0", "G1"}: raise ValueError("stage must be G0 or G1")
    if not torch.cuda.is_available(): raise RuntimeError("real SD3.5 G0/G1 requires a CUDA GPU; no RGB proxy is permitted")
    pipeline = load_sd35_pipeline(contract["identity"]["model_id"], torch_dtype=torch.float16, token=hf_token).to("cuda")
    roster = contract[stage.lower()]
    records = tuple(_record(seed, prompt, attack, pipeline, detection_key, wrong_key, content_detector) for seed, prompt in zip(roster["seeds"], roster["prompts"], strict=True) for attack in roster["attacks"])
    root = Path(contract["artifact_root"] if artifact_root is None else artifact_root); root.mkdir(parents=True, exist_ok=False)
    path = root / f"{stage.lower()}-records.json"; payload = {"contract_sha256": contract_sha256(repo_root), "stage": stage, "records": records}; path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="ascii")
    (root / f"{stage.lower()}-records.json.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest()+f"  {path.name}\n", encoding="ascii")
    return records
