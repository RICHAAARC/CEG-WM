"""Single fixed M1 development roster; initialize, explicitly run, then summarize.

No M2/M3 mode and no automatic retries. Every planned image retains a row.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/latent_landmark_sync_v1/candidate.json"


def dump(path, value):
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def external_key(key, source_id):
    key = key.encode() if isinstance(key,str) else bytes(key)
    return hashlib.sha256(b"CEG-WM/Latent-Landmark-Sync-V1/external-negative\x00"+len(key).to_bytes(8,"big")+key+source_id.encode("ascii")).digest()


def initialize(output):
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    frozen = output / "configuration.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != cfg:
            raise ValueError("existing output configuration differs")
    else:
        dump(frozen, cfg)
    rows_path = output / "rows.json"
    if not rows_path.exists():
        dump(rows_path, [dict(source_id=s["id"], arm=arm, status="NOT_EXECUTED", image=None,
                              correct_key_reader=None, external_wrong_key_reader=None, error=None)
                         for s in cfg["sources"] for arm in cfg["arms"]])
    return cfg, json.loads(rows_path.read_text())


def quality(model, reference, image, device, cfg):
    import torch
    from cegwm.method.latent_landmark_sync import CENTERS
    a = np.asarray(reference,dtype=np.float64)/255
    b = np.asarray(image,dtype=np.float64)/255
    delta = b-a
    mse = float(np.mean(delta*delta))
    def tensor(pixels):
        return torch.from_numpy(pixels.astype(np.float32)).permute(2,0,1)[None].to(device)*2-1
    with torch.no_grad():
        perceptual = float(model(tensor(b),tensor(a)).item())
    local = []
    radius = cfg["quality"]["local_residual_window_rgb"]//2
    for point in CENTERS:
        x,y = np.floor(8*point+3.5+.5).astype(int)
        window = delta[y-radius:y+radius+1,x-radius:x+radius+1]
        local.append(dict(rms=float(np.sqrt(np.mean(window*window))),max_abs=float(np.max(np.abs(window)))))
    return dict(psnr=None if mse==0 else float(-10*np.log10(mse)), identical=mse==0, lpips=perceptual,
                residual_rms=float(np.sqrt(mse)),residual_max_abs=float(np.max(np.abs(delta))),local_residuals=local)


def run(output, runtime_root):
    import torch
    cfg, rows = initialize(output)
    if any(row["status"] != "NOT_EXECUTED" for row in rows):
        raise RuntimeError("this output has execution records; automatic retries are disabled")
    from experiments.run_paper_main_worker_v2 import _build_runtime
    from cegwm.runtime.diffusers_sd35 import run_sd35_plain
    from cegwm.runtime.latent_landmark_sd35 import generate_content_landmark, read_image
    from cegwm.runtime.paper_detection_v2 import score_current_rgb
    from cegwm.method.content_iss import score_content_iss_image, iss_beta
    packages = ["torch","diffusers","transformers","accelerate","lpips","torchmetrics","numpy","Pillow"]
    versions = {}
    for name in packages:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = "NOT_INSTALLED"
    metadata = dict(status="INITIALIZING", versions=versions, planned_images=24, independent_sources=4,
                    cuda_available=torch.cuda.is_available(), setup_error=None,
                    config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                    reference_threshold_sha256=hashlib.sha256((CONFIG.parent/"reference_threshold.json").read_bytes()).hexdigest())
    git = subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],capture_output=True,text=True)
    metadata["source_exact"] = git.stdout.strip() if git.returncode==0 else None
    source_paths = ["src/cegwm/method/latent_landmark_sync.py","src/cegwm/runtime/latent_landmark_sd35.py",
                    "experiments/run_latent_landmark_sync_v1.py","src/cegwm/runtime/content_iss_sd35.py",
                    "src/cegwm/runtime/observation.py","src/cegwm/runtime/paper_detection_v2.py",
                    "src/cegwm/runtime/blind_scoring_v2.py","experiments/run_paper_main_worker_v2.py"]
    metadata["source_sha256"] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths}
    dump(output/"execution.json", metadata)
    try:
        runtime = _build_runtime(runtime_root)
        import lpips
        perceptual = lpips.LPIPS(net=cfg["quality"]["lpips_net"]).to(runtime["device"]).eval()
    except Exception as error:
        metadata.update(status="NOT_EXECUTED",setup_error=type(error).__name__+": "+str(error))
        dump(output/"execution.json", metadata)
        summarize(output)
        return 3
    pipeline, key = runtime["pipeline"],runtime["key"]
    iss = runtime["assets"].content_assets.iss_assets
    public = iss.embed_assets.hf_public_assets
    metadata.update(status="RUNNING", model_config=dict(pipeline.scheduler.config),
                    vae_config=dict(pipeline.vae.config), pipeline_config=dict(getattr(pipeline,"config",{})),
                    processor_class=type(public.image_processor).__module__+"."+type(public.image_processor).__name__,
                    processor_config=dict(getattr(public.image_processor,"config",{})),
                    model_identity_note="main factory model/revision behavior reused; actual config commit hashes recorded where available, missing hashes are not an exact weight identity claim",
                    lpips="LPIPS alex default learned v0.1; RGB [-1,1]")
    dump(output/"execution.json", metadata)
    image_dir = output/"images"; image_dir.mkdir(exist_ok=True)
    for source in cfg["sources"]:
        images, beta = {}, None
        for arm in cfg["arms"]:
            row = next(r for r in rows if r["source_id"]==source["id"] and r["arm"]==arm)
            try:
                trace = None
                if arm == "plain":
                    image = run_sd35_plain(pipeline,source["prompt"],height=512,width=512,
                                           generator=torch.Generator(device=runtime["device"]).manual_seed(source["seed"]))
                elif arm == "main_syncseal":
                    image = runtime["assets"].geometry_backend.embed_final_rgb(images["content_only"],cfg["syncseal_residual_multiplier"])
                    trace = dict(source="paired_content_only_final_RGB", fresh_generation=False)
                else:
                    if beta is None:
                        beta = iss_beta(score_content_iss_image(images["plain"],key,iss.lf_public_assets),iss.iss_asset)
                    rms = 0. if arm=="content_only" else float(arm.split("_")[1])
                    image,trace = generate_content_landmark(pipeline,source["prompt"],key,iss,beta,
                                                          source["seed"],runtime["device"],rms)
                image = image.convert("RGB")
                if image.size != (512,512): raise ValueError("final RGB size differs")
                image_path = image_dir/f"{source['id']}__{arm}.png"
                with image_path.open("xb") as stream: image.save(stream,format="PNG")
                images[arm] = image
                row.update(status="GENERATED",image=str(image_path.relative_to(output)),
                           image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),generation_trace=trace)
                dump(output/"rows.json",rows)
                # The reader sees final RGB8 only. No source, prompt, beta, or truth.
                row["correct_key_reader"] = read_image(image,key,public.image_processor,public.vae)
                dump(output/"rows.json",rows)
                row["external_wrong_key_reader"] = read_image(image,external_key(key,source["id"]),public.image_processor,public.vae)
                # Save all raw blind outputs before evaluating their correspondence.
                dump(output/"rows.json",rows)
                row["content_pre"] = float(score_current_rgb(image,key,runtime["assets"]).value)
                row["quality_vs_plain"] = quality(perceptual,images["plain"],image,runtime["device"],cfg)
                row["status"] = "SCORED"
            except Exception as error:
                row.update(status="ERROR",error=type(error).__name__+": "+str(error))
            dump(output/"rows.json", rows)
        # Retain images only while evaluating their paired source.
        images.clear()
    metadata["status"] = "EXECUTED" if all(r["status"]=="SCORED" for r in rows) else "INCOMPLETE"
    dump(output/"execution.json",metadata)
    summarize(output)
    return 0 if metadata["status"]=="EXECUTED" else 3


def summarize(output):
    from cegwm.method.latent_landmark_sync import CENTERS, solve, transform, latent_to_rgb_h
    cfg,rows = initialize(output)
    report = dict(status="NOT_EXECUTED",independent_sources=4,planned_images=24,
                  status_counts={s:sum(r["status"]==s for r in rows) for s in sorted(set(r["status"] for r in rows))},
                  selected_rms=None,peak_threshold=None,amplitudes=[],claim="M1 clean development only; no FPR or rotation evidence")
    complete = all(r["status"]=="SCORED" for r in rows)
    negatives = [r["external_wrong_key_reader"] for r in rows] + [r["correct_key_reader"] for r in rows if r["arm"] in cfg["arms"][:3]]
    threshold = max([0.]+[p["score"] for query in negatives for p in query["peaks"]]) if all(query is not None for query in negatives) else None
    status="NO_CLEAN_READABLE_WINDOW" if complete else ("INCOMPLETE" if any(r["status"]!="NOT_EXECUTED" for r in rows) else "NOT_EXECUTED")
    report.update(status=status,peak_threshold=threshold,ambiguity_threshold=0.,negative_queries_planned=len(negatives),
                  negative_queries=sum(query is not None for query in negatives))
    negative_evaluated=[]
    for row in rows:
        query_types=["external_wrong_key_reader"]
        if row["arm"] in cfg["arms"][:3]: query_types.append("correct_key_reader")
        for query_type in query_types:
            reader=row[query_type]
            geometry=solve(reader["peaks"],threshold,0.) if reader is not None and threshold is not None else None
            negative_evaluated.append(dict(source_id=row["source_id"],arm=row["arm"],query=query_type,
                status="EVALUATED" if geometry is not None else "NOT_EVALUATED",geometry=geometry))
    dump(output/"negative_evaluated.json",negative_evaluated)
    report["negative_accepted"] = sum(e["geometry"] is not None and e["geometry"]["accepted"] for e in negative_evaluated)
    evaluated=[]
    for row in rows:
        reader=row["correct_key_reader"]
        geometry = solve(reader["peaks"],threshold,0.) if reader is not None and threshold is not None else dict(
            accepted=False,reason="reader_or_complete_negative_threshold_unavailable",H_latent=None,matches=[])
        point_errors = {str(i):None for i in range(8)}
        for p in geometry["matches"]:
            point_errors[str(p["identity"])] = float(np.linalg.norm(np.array(p["xy"])-CENTERS[p["identity"]]))
        correct = sum(e is not None and e<=cfg["correct_identity_max_latent_error"] for e in point_errors.values())
        h = geometry["H_latent"]
        controls=np.asarray(cfg["control_points_rgb"])
        rmse=None if h is None else float(np.sqrt(np.mean(np.sum((transform(latent_to_rgb_h(h),controls)-controls)**2,axis=1))))
        evaluated.append(dict(source_id=row["source_id"],arm=row["arm"],geometry=geometry,point_errors_latent=point_errors,
                              correct_identities=correct,control_rgb_rmse=rmse,
                              readable=geometry["accepted"] and correct>=4 and rmse is not None and rmse<=cfg["clean_control_point_rgb_rmse_max"]))
    dump(output/"evaluated.json",evaluated)
    for rms,arm in zip(cfg["rms"],cfg["arms"][3:]):
        results=[]
        for source in cfg["sources"]:
            selected = {r["arm"]:r for r in rows if r["source_id"]==source["id"]}
            row,main,content = selected[arm],selected["main_syncseal"],selected["content_only"]
            q,m = row.get("quality_vs_plain"),main.get("quality_vs_plain")
            psnr=lambda value: math.inf if value["identical"] else value["psnr"]
            quality_ok=q is not None and m is not None and psnr(q)>=psnr(m)-.5 and q["lpips"]<=1.1*m["lpips"]+1e-4
            content_ok="content_pre" in content and "content_pre" in row and (not content["content_pre"]>cfg["reference_tau"] or row["content_pre"]>cfg["reference_tau"])
            read=next(e for e in evaluated if e["source_id"]==source["id"] and e["arm"]==arm)
            results.append(dict(source_id=source["id"],status=row["status"],readable=read["readable"],quality_ok=quality_ok,content_preserved=content_ok,
                                passed=row["status"]=="SCORED" and read["readable"] and quality_ok and content_ok))
        passed=all(r["passed"] for r in results)
        report["amplitudes"].append(dict(rms=rms,passed=passed,denominator=4,sources=results))
        if complete and passed and report["selected_rms"] is None:
            report.update(selected_rms=rms,status="CLEAN_READABLE_WINDOW")
    dump(output/"summary.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("initialize","run","summarize"),default="initialize")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--runtime-root",type=Path,default=Path("/content/latent-landmark-sync-v1-runtime"))
    args=parser.parse_args(argv)
    if args.mode=="run": return run(args.output,args.runtime_root)
    initialize(args.output)
    print(json.dumps(summarize(args.output),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
