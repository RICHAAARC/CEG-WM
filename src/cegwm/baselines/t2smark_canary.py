"""Resumable, one-physical-unit T2SMark SD3.5 Colab canary runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

RUN_SCHEMA = "cegwm.t2smark_canary.v1"
RUN_ID_DEFAULT = "t2smark_sd35_one_unit_v1"
CONDITIONS = ("clean_no_attack", "gaussian_noise_sigma_0p05_v1", "jpeg_q50_v1",
              "brightness_1p25_v1", "crop_75_resize_bicubic_v1",
              "rotation_10_bicubic_reflect_center_crop_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        image.save(temporary, format="PNG")
        with open(temporary, "rb") as stream: os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any] | None:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return None


def _identity(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in ("schema", "project_exact", "official_exact", "model_id",
        "model_revision", "prompt", "generation_seed", "watermark_seed", "parameters")}


def establish_contract(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Create once or fail closed when any method/config identity drifts."""
    path = run_dir / "run_config.json"; existing = _read_json(path)
    if existing is None:
        atomic_json(path, config); return config
    if _identity(existing) != _identity(config):
        raise RuntimeError("run_config identity drift: refuse to reuse stable RUN_ID")
    return existing


def valid_file(path: Path, expected_sha256: str | None = None) -> bool:
    return path.is_file() and path.stat().st_size > 0 and (expected_sha256 is None or sha256_file(path) == expected_sha256)


def valid_generation(run_dir: Path, config: dict[str, Any]) -> bool:
    checkpoint = _read_json(run_dir / "generation_checkpoint.json")
    if not checkpoint or checkpoint.get("identity") != _identity(config): return False
    return all(valid_file(run_dir / name, checkpoint.get("files", {}).get(name)) for name in ("clean.png", "watermarked.png"))


def observation_path(run_dir: Path, condition: str, role: str) -> Path:
    return run_dir / "observations" / f"{condition}__{role}.json"


def valid_observation(run_dir: Path, config: dict[str, Any], condition: str, role: str) -> bool:
    record = _read_json(observation_path(run_dir, condition, role))
    if not record or record.get("status") != "ok" or record.get("identity") != _identity(config): return False
    image = run_dir / record.get("image", "")
    score = record.get("score")
    return valid_file(image, record.get("image_sha256")) and isinstance(score, (int, float)) and math.isfinite(score)


def pending_observations(run_dir: Path, config: dict[str, Any], force: bool = False) -> list[tuple[str, str]]:
    return [(condition, role) for condition in CONDITIONS for role in ("clean_negative", "watermarked_positive")
            if force or not valid_observation(run_dir, config, condition, role)]


def _attack(image: Image.Image, condition: str) -> Image.Image:
    rgb = image.convert("RGB")
    if condition == "clean_no_attack": return rgb
    if condition == "jpeg_q50_v1":
        import io
        buffer = io.BytesIO(); rgb.save(buffer, "JPEG", quality=50, subsampling=2, optimize=False, progressive=False)
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    if condition == "brightness_1p25_v1": return ImageEnhance.Brightness(rgb).enhance(1.25)
    if condition == "crop_75_resize_bicubic_v1":
        w,h=rgb.size; cw,ch=max(1,round(w*.75)),max(1,round(h*.75)); left,top=(w-cw)//2,(h-ch)//2
        return rgb.crop((left,top,left+cw,top+ch)).resize((w,h),Image.Resampling.BICUBIC)
    if condition == "gaussian_noise_sigma_0p05_v1":
        array=np.asarray(rgb,dtype=np.float32)/255.; rng=np.random.default_rng(0)
        return Image.fromarray(np.clip((array+rng.normal(0,.05,array.shape))*255,0,255).astype(np.uint8))
    if condition == "rotation_10_bicubic_reflect_center_crop_v1":
        array=np.asarray(rgb,dtype=np.uint8); h,w=array.shape[:2]; theta=math.radians(10)
        a,b=(w-1)/2,(h-1)/2; px=max(0,math.ceil(abs(math.cos(theta))*a+abs(math.sin(theta))*b+2-a)); py=max(0,math.ceil(abs(math.sin(theta))*a+abs(math.cos(theta))*b+2-b))
        padded=np.pad(array,((py,py),(px,px),(0,0)),mode="reflect"); center=(px+(w-1)/2,py+(h-1)/2)
        return Image.fromarray(padded).rotate(10,resample=Image.Resampling.BICUBIC,center=center,fillcolor=(0,0,0)).crop((px,py,px+w,py+h))
    raise ValueError(condition)


def rebuild_partial(run_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows=[]
    for condition in CONDITIONS:
        for role in ("clean_negative", "watermarked_positive"):
            record=_read_json(observation_path(run_dir,condition,role))
            if record and record.get("identity")==_identity(config): rows.append(record)
    atomic_json(run_dir/"partial_result.json", {"identity":_identity(config),"complete":len(rows)==12,"observations":rows})
    csv="condition,role,status,score\n" + "".join(f"{r['condition']},{r['role']},{r['status']},{r.get('score','')}\n" for r in rows)
    temporary=run_dir/".partial_scores.csv.tmp"; temporary.write_text(csv,encoding="utf-8"); os.replace(temporary,run_dir/"partial_scores.csv")
    return rows


def run_canary(run_dir: Path, config: dict[str, Any], execute: Callable[[str, str], tuple[Image.Image, float]], force: bool=False) -> None:
    """Resume valid work, retry failed/corrupt work, and publish final only at 12/12."""
    run_dir.mkdir(parents=True, exist_ok=True); establish_contract(run_dir,config)
    for condition,role in pending_observations(run_dir,config,force):
        attempt={"attempted_at":time.time(),"condition":condition,"role":role,"identity":_identity(config)}
        atomic_json(run_dir/"attempts"/f"{condition}__{role}__{time.time_ns()}.json",attempt)
        try:
            image,score=execute(condition,role)
            if not math.isfinite(score): raise RuntimeError("native score is non-finite")
            image_name=f"images/{condition}__{role}.png"; atomic_png(run_dir/image_name,image)
            record={**attempt,"status":"ok","score":float(score),"image":image_name,"image_sha256":sha256_file(run_dir/image_name)}
        except Exception as exc:
            record={**attempt,"status":"failed","failure":f"{type(exc).__name__}: {exc}"}
        atomic_json(observation_path(run_dir,condition,role),record); rebuild_partial(run_dir,config)
    rows=rebuild_partial(run_dir,config)
    if len(rows)==12 and all(row.get("status")=="ok" for row in rows):
        atomic_json(run_dir/"canary_result.json",{"identity":_identity(config),"engineering_canary_complete":True,"observations":rows})
        os.replace(run_dir/"partial_scores.csv",run_dir/"scores.csv")
    elif (run_dir / "canary_result.json").exists():
        os.replace(run_dir / "canary_result.json", run_dir / f"canary_result.stale.{time.time_ns()}.json")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",required=True); parser.add_argument("--project-exact",required=True)
    parser.add_argument("--run-id",default=RUN_ID_DEFAULT); parser.add_argument("--official-source",required=True); parser.add_argument("--force-rerun-all",action="store_true"); args=parser.parse_args()
    if not os.environ.get("HF_TOKEN"): raise RuntimeError("HF_TOKEN must be supplied only through environment")
    import torch
    sys.path.insert(0, args.official_source)
    from cegwm.baselines.t2smark import embed_t2smark_sd35, score_t2smark_rgb
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    config={"schema":RUN_SCHEMA,"project_exact":args.project_exact,
      "official_exact":"0c1fbfd50fcd1fba135477a2c016e284d5d7914d",
      "model_id":"stabilityai/stable-diffusion-3.5-medium",
      "model_revision":"b940f670f0eda2d07fbb75229e779da1ad11eb80",
      "prompt":"A small red ceramic cube on a pale wooden table, studio photograph",
      "generation_seed":1701,"watermark_seed":9173,
      "parameters":{"height":512,"width":512,"guidance_scale":4.0,"num_inference_steps":40,
                    "num_inversion_steps":10,"key_length":16,"message_length":256,"tau":.674}}
    root=Path(args.run_dir); establish_contract(root,config)
    generation_rebuilt = not valid_generation(root,config) or args.force_rerun_all
    if generation_rebuilt:
        from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline
        pipe=InversionDiffusion3Pipeline.from_pretrained(config["model_id"],revision=config["model_revision"],
            torch_dtype=torch.float16,token=os.environ["HF_TOKEN"]).to("cuda")
        pipe.set_progress_bar_config(disable=True); device=torch.device(pipe._execution_device)
        latent=torch.randn((1,16,64,64),generator=torch.Generator("cuda").manual_seed(config["generation_seed"]),device=device,dtype=torch.float16)
        keygen=torch.Generator("cuda").manual_seed(config["watermark_seed"])
        master=torch.randint(0,2,(16,),generator=keygen,device=device); session=torch.randint(0,2,(16,),generator=keygen,device=device)
        message=torch.randint(0,2,(256,),generator=keygen,device=device)
        common={"prompt":config["prompt"],"height":512,"width":512,"guidance_scale":4.0,"num_inference_steps":40}
        clean=pipe(latents=latent,**common).images[0].convert("RGB")
        watermarked=pipe(latents=embed_t2smark_sd35(latent,master,session,message),**common).images[0].convert("RGB")
        atomic_png(root/"clean.png",clean); atomic_png(root/"watermarked.png",watermarked)
        atomic_json(root/"generation_checkpoint.json",{"identity":_identity(config),"files":{"clean.png":sha256_file(root/"clean.png"),"watermarked.png":sha256_file(root/"watermarked.png")}})
    from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline
    pipe=InversionDiffusion3Pipeline.from_pretrained(config["model_id"],revision=config["model_revision"],torch_dtype=torch.float16,token=os.environ["HF_TOKEN"]).to("cuda")
    keygen=torch.Generator("cuda").manual_seed(config["watermark_seed"]); master=torch.randint(0,2,(16,),generator=keygen,device=pipe._execution_device)
    def execute(condition: str, role: str) -> tuple[Image.Image,float]:
        image=_attack(Image.open(root/("clean.png" if role=="clean_negative" else "watermarked.png")),condition)
        return image,score_t2smark_rgb(np.asarray(image,dtype=np.uint8),pipe,master,10)
    run_canary(root,config,execute,args.force_rerun_all or generation_rebuilt)


if __name__ == "__main__": main()
