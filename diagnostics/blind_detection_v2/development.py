"""Stage-2 fresh development: residual strength, native warp, alignment tolerance.

No threshold fitting, V2 selection, formal execution, or diagnostic-image reuse.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
STRENGTHS = (0.5, 0.75, 1.0)
ANGLES = (0., -13., 7.)
REFERENCE_TAU = 1.2657276026437319  # descriptive historical reference only
PROMPTS = (
    "A white ceramic teapot on a pale linen tablecloth beside a window, soft daylight, photograph",
    "A bicycle leaning against a red brick wall covered partly in ivy, street photograph",
    "A rocky mountain stream beneath dense pine trees, detailed landscape photograph",
    "Colorful folded fabric stacked on a wooden market stall, close-up photograph",
)
ROSTER = tuple({"unit_id": f"v2-mechanism-{i:02d}", "seed": 2026090700+i, "prompt": p} for i,p in enumerate(PROMPTS))
PERTURBATIONS = tuple([(a, 0., 0.) for a in (0., -.25, .25, -.5, .5, -1., 1., -2., 2.)]
                      + [(0., d, 0.) for d in (-1., 1., -2., 2., -4., 4.)]
                      + [(0., 0., d) for d in (-1., 1., -2., 2., -4., 4.)])


def rotation(image, angle):
    w,h = image.size
    theta = math.radians(angle)
    x,y = (w-1)/2, (h-1)/2
    px = max(0, math.ceil(abs(math.cos(theta))*x+abs(math.sin(theta))*y+2-x))
    py = max(0, math.ceil(abs(math.sin(theta))*x+abs(math.cos(theta))*y+2-y))
    padded = np.pad(np.asarray(image), ((py,py),(px,px),(0,0)), mode="reflect")
    return Image.fromarray(padded).rotate(angle, Image.Resampling.BICUBIC,
        center=(px+x,py+y)).crop((px,py,px+w,py+h)) if angle else image.copy()


def sampler_h(angle, dx=0., dy=0.):
    # Pillow edge-coordinate attack inverse. Truth only enters this diagnostic.
    t = math.radians(angle)
    c,s = math.cos(t),math.sin(t)
    d = np.array([[c,-s,255.5*(1-c+s)+dx],[s,c,255.5*(1-s-c)+dy],[0.,0.,1.]])
    n = np.array([[2/511,0.,-1.],[0.,2/511,-1.],[0.,0.,1.]])
    return n @ d @ np.linalg.inv(n)


def quality(a,b):
    mse = float(np.mean((np.asarray(a,dtype=float)-np.asarray(b,dtype=float))**2))
    return {"mse_rgb_255": mse, "psnr_db": 10*math.log10(255**2/mse) if mse else None}


def run(output):
    from experiments.run_blind_detection_v1 import build_production_runtime, load_runtime_config
    from cegwm.runtime.content_iss_sd35 import run_content_iss_evaluation_pair
    from cegwm.runtime.blind_detection import _detect_core, _score_current_rgb
    from cegwm.geometry_v7.syncseal import _to_tensor, _to_rgb
    from cegwm.geometry_v7.r1b import rectify_attacked_rgb
    from cegwm.shared.keys import normalize_detection_key
    output.mkdir(parents=True, exist_ok=False)
    plan = {"status":"DEVELOPMENT_ONLY", "science_denominator":0, "roster":ROSTER,
            "strengths":STRENGTHS, "angles":ANGLES, "perturbations":PERTURBATIONS,
            "planned_rows":216, "reference_threshold":REFERENCE_TAU,
            "threshold_use":"descriptive_only_no_calibration", "stage4_samples_used":False}
    (output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    key = normalize_detection_key(os.environ["CEG_WM_ROOT_KEY"])
    config = load_runtime_config(REPO)
    runtime = output/"runtime"
    runtime.mkdir()
    pipeline,assets = build_production_runtime(REPO,config,hf_token=os.environ["HF_TOKEN"],runtime_root=runtime)
    adapter = assets.geometry_backend
    rows=[]

    def save(row, compute):
        start=time.monotonic()
        row["error"]=None
        try:
            compute(row)
        except Exception as error:
            row["error"]=f"{type(error).__name__}: {error}"
        row["seconds"]=time.monotonic()-start
        rows.append(row)
        with (output/"rows.jsonl").open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")

    def score(image):
        return float(_score_current_rgb(image,key,assets).value)

    for unit in ROSTER:
        pair=None
        generation_error=None
        try:
            pair=run_content_iss_evaluation_pair(pipeline,unit["prompt"],key,assets.content_assets.iss_assets,
                                                height=512,width=512,seed=unit["seed"])
            pair.image.save(output/(unit["unit_id"]+"__content.png"))
            pair.primary_null.save(output/(unit["unit_id"]+"__clean.png"))
            current=_to_tensor(pair.image,adapter.device)
            with torch.no_grad():
                embedded=adapter.model.embed(current)["imgs_w"]
        except Exception as error:
            generation_error=f"{type(error).__name__}: {error}"

        def ensure():
            if generation_error:
                raise RuntimeError("generation:"+generation_error)

        for strength in STRENGTHS:
            for angle in ANGLES:
                def measure(row):
                    ensure()
                    cg=_to_rgb((current+strength*(embedded-current)).clamp(0,1))
                    observed=rotation(cg,angle)
                    row["quality_vs_content"]=quality(cg,pair.image)
                    row["quality_vs_clean"]=quality(cg,pair.primary_null)
                    result=_detect_core(observed,key,assets,REFERENCE_TAU)
                    row["v1"]=asdict(result)
                    if not result.method_complete:
                        row["error"]=result.operational_error or "incomplete_v1_detection"
                    geometry=adapter.detect_geometry(observed)
                    row["geometry"]=asdict(geometry)
                    row["adapter_warp_score"]=score(rectify_attacked_rgb(observed,geometry.homography_observed_to_canonical))
                    q=np.array([[-1.,-1.,1.],[1.,-1.,1.],[1.,1.,1.],[-1.,1.,1.]])
                    # Exact pixel-center truth differs from the Pillow sampler by T(-.5) D T(.5).
                    t=np.array([[1.,0.,-1/511],[0.,1.,-1/511],[0.,0.,1.]])
                    truth=q@(t@sampler_h(angle)@np.linalg.inv(t)).T
                    prediction=np.asarray(geometry.observed_corners_in_canonical_normalized)
                    row["corner_rmse_pixels"]=float(np.sqrt(np.mean(np.sum((prediction-truth[:,:2]/truth[:,2,None])**2,axis=1)))*511/2)
                    raw=torch.tensor(geometry.raw_syncseal_corners,device=adapter.device,dtype=torch.float32).reshape(1,8)
                    with torch.no_grad():
                        native=adapter.model.unwarp(_to_tensor(observed,adapter.device),raw,(512,512))
                    row["native_warp_score"]=score(_to_rgb(native.clamp(0,1)))
                save({"unit_id":unit["unit_id"],"kind":"strength","arm":"positive","strength":strength,"angle":angle},measure)
        for angle in ANGLES:
            def negative(row):
                ensure()
                row["v1"]=asdict(_detect_core(rotation(pair.primary_null,angle),key,assets,REFERENCE_TAU))
                if not row["v1"]["method_complete"]:
                    row["error"]=row["v1"]["operational_error"] or "incomplete_v1_detection"
            save({"unit_id":unit["unit_id"],"kind":"baseline_negative","arm":"negative","angle":angle},negative)
        for arm in ("positive","negative"):
            for da,dx,dy in PERTURBATIONS:
                def tolerance(row):
                    ensure()
                    image=_to_rgb((current+.75*(embedded-current)).clamp(0,1)) if arm=="positive" else pair.primary_null
                    aligned=rectify_attacked_rgb(rotation(image,7.),sampler_h(7.+da,dx,dy))
                    row["score"]=score(aligned)
                    row["above_reference"]=row["score"]>REFERENCE_TAU
                save({"unit_id":unit["unit_id"],"kind":"tolerance","arm":arm,"angle_error":da,"dx":dx,"dy":dy},tolerance)
        print(f"{unit['unit_id']}: {len(rows)}/216 rows",flush=True)
    summary={**plan,"completed_rows":len(rows),"errors":sum(r["error"] is not None for r in rows),
             "torch":torch.__version__,"device":config["device"],
             "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output)))
