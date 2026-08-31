"""Command boundary for the three pinned external SD3.5 engineering canaries.

The shared transaction is imported from :mod:`baseline_canary`; model work is
kept deliberately behind a callable factory so importing and CPU testing this
module neither downloads a model nor creates a CUDA context.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any
import numpy as np

from .baseline_canary import (RUN_ID_DEFAULTS, RUN_SCHEMA, atomic_json, atomic_png, clear_stale_lock, run_transaction, sha256_file, validate_final_publication)
from .t2smark_canary import _attack
from .external_sd35 import GaussianShadingCarrier, SD35_SHAPE, ShallowDiffuseCarrier, TreeRingCarrier, score_rgb

OFFICIAL_EXACTS = {
    "tree_ring": "3015283d9cf82e90b628f02ad2121bd37408ca9a",
    "gaussian_shading": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
    "shallow_diffuse": "c80c553fdf66fda8db735d77a9d56538b7a0ade8",
}
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"


def require_official_source(method: str, source: Path) -> None:
    """Fail closed unless the cloned official source is detached, clean, exact."""
    if not (source / ".git").exists():
        raise RuntimeError("official source is not a git repository")
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    attached = subprocess.run(["git", "-C", str(source), "symbolic-ref", "-q", "--short", "HEAD"], text=True, capture_output=True)
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], text=True, capture_output=True, check=True).stdout.strip()
    if head != OFFICIAL_EXACTS[method] or attached.returncode == 0 or dirty:
        raise RuntimeError("official source identity is not detached, clean, pinned exact")


def main() -> None:
    parser = argparse.ArgumentParser(description="pinned external SD3.5 engineering canary")
    parser.add_argument("--method", choices=sorted(OFFICIAL_EXACTS), required=True)
    parser.add_argument("--run-dir", required=True); parser.add_argument("--project-exact", required=True)
    parser.add_argument("--official-source", required=True); parser.add_argument("--run-id")
    parser.add_argument("--force-rerun-all", action="store_true"); parser.add_argument("--clear-stale-lock", action="store_true"); args = parser.parse_args()
    root=Path(args.run_dir)
    if args.clear_stale_lock:
        print({"cleared_stale_lock":clear_stale_lock(root)}); return
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be supplied only through environment")
    require_official_source(args.method, Path(args.official_source))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this engineering canary")
    if len(args.project_exact)!=40 or any(c not in "0123456789abcdef" for c in args.project_exact): raise RuntimeError("project exact must be a 40-hex commit")
    run_id=args.run_id or RUN_ID_DEFAULTS[args.method]
    if root.name != run_id: raise RuntimeError("run-dir basename must equal stable method RUN_ID")
    params={"height":512,"width":512,"guidance_scale":4.5,"num_inference_steps":20,"num_inversion_steps":20,"carrier":args.method}
    if args.method=="tree_ring": params.update({"watermark_seed":999999,"channel":0,"radius":10,"pattern":"ring"})
    elif args.method=="gaussian_shading": params.update({"watermark_seed":20260622,"channel_copy":1,"hw_copy":8})
    else: params.update({"watermark_seed":42,"channel":0,"radius":10,"mask":"circle","pattern":"complex_rand","injection":"complex","measurement":"l1_complex","edit_fraction":.2,"post_edit_guidance_scale":1.0})
    config={"schema":RUN_SCHEMA,"project_exact":args.project_exact,"official_exact":OFFICIAL_EXACTS[args.method],"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"prompt":"A small red ceramic cube on a pale wooden table, studio photograph","generation_seed":1701,"watermark_seed":params["watermark_seed"],"parameters":params}
    state: dict[str,Any]={}
    def carrier() -> Any:
        return state.setdefault("carrier", {"tree_ring":TreeRingCarrier,"gaussian_shading":GaussianShadingCarrier,"shallow_diffuse":ShallowDiffuseCarrier}[args.method].fixed(seed=params["watermark_seed"],device="cuda"))
    def pipe() -> Any:
        if "pipe" not in state:
            from .sd35_runtime import load_sd3_pipeline
            state["pipe"]=load_sd3_pipeline(MODEL_ID,MODEL_REVISION)
        return state["pipe"]
    def generate() -> None:
        p=pipe(); base=torch.randn(SD35_SHAPE,generator=torch.Generator("cuda").manual_seed(1701),device="cuda",dtype=torch.float16); c=carrier()
        if args.method=="shallow_diffuse":
            edit_index=16; pre=p.denoise_segment(base,prompt=config["prompt"],guidance=4.5,steps=20,start=0,end=edit_index); marked_edit=c.inject(pre.clone()); clean_lat=p.denoise_segment(pre,prompt=config["prompt"],guidance=1.0,steps=20,start=edit_index,end=20); marked_branch=p.denoise_segment(marked_edit,prompt=config["prompt"],guidance=1.0,steps=20,start=edit_index,end=20); marked_lat=clean_lat.clone(); marked_lat[:,0]=marked_branch[:,0]; clean=p.decode_latents(clean_lat); marked=p.decode_latents(marked_lat)
        else:
            marked_lat=c.inject(base) if args.method=="tree_ring" else c.create_strict_paired_latents(base); common={"prompt":config["prompt"],"height":512,"width":512,"guidance_scale":4.5,"num_inference_steps":20}; clean=p(latents=base,**common).images[0].convert("RGB"); marked=p(latents=marked_lat,**common).images[0].convert("RGB")
        atomic_png(root/"clean.png",clean); atomic_png(root/"watermarked.png",marked); atomic_json(root/"generation_checkpoint.json",{"identity":{k:config[k] for k in ("schema","project_exact","official_exact","model_id","model_revision","prompt","generation_seed","watermark_seed","parameters")},"files":{"clean.png":sha256_file(root/"clean.png"),"watermarked.png":sha256_file(root/"watermarked.png")}})
    def execute(condition: str, role: str) -> tuple[Any,float]:
        from PIL import Image
        image=_attack(Image.open(root/("clean.png" if role=="clean_negative" else "watermarked.png")),condition)
        return image,score_rgb(np.asarray(image,dtype=np.uint8),pipe(),carrier(),inversion_steps=20,prompt=config["prompt"])
    run_transaction(root,config,generate,execute,force=args.force_rerun_all)
    if not validate_final_publication(root): raise RuntimeError("final publication hash validation failed")


if __name__ == "__main__":
    main()
