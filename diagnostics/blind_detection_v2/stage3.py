"""Fixed development A/B. Both scorers choose their own top3; no formal decision."""
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

from experiments.run_blind_detection_v1 import build_production_runtime, load_runtime_config
from cegwm.runtime.blind_detection_v2 import _search_candidates, SearchPlan
from cegwm.runtime.blind_scoring_v2 import score_branches_v2
from cegwm.runtime.blind_detection import _detect_core
from cegwm.runtime.content_weighted_joint_sd35 import ContentCalibrationAssets
from cegwm.runtime.blind_detection import BlindProductionAssets
from cegwm.method.blind_detection import statistic_from_weighted_scores
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.shared.keys import normalize_detection_key

REPO=Path(__file__).resolve().parents[2]
TAU_REFERENCE=1.2657276026437319
UNITS=("v2-mechanism-00","v2-mechanism-01")
CONDITIONS=(("identity",0.,0.,0.),("offgrid_rotation",-11.7,0.,0.),("offgrid_joint",18.3,.75,-.75))
ROSTER=tuple((u,c[0],arm) for u in UNITS for c in CONDITIONS for arm in ("positive","negative"))


class CountVAE:
    """Local diagnostic proxy; no mutation of the model or global encode function."""
    def __init__(self,vae): self.vae,self.calls=vae,0
    def __getattr__(self,name): return getattr(self.vae,name)
    def encode(self,pixels):
        self.calls+=1
        return self.vae.encode(pixels)


def counted_assets(assets):
    iss=assets.content_assets.iss_assets
    hf=iss.hf_public_assets
    lf=iss.lf_public_assets
    proxies={}
    def proxy(vae):
        if id(vae) not in proxies: proxies[id(vae)]=CountVAE(vae)
        return proxies[id(vae)]
    hf2=replace(hf,vae=proxy(hf.vae))
    carrier2=replace(lf.carrier_assets,vae=proxy(lf.carrier_assets.vae))
    lf2=replace(lf,carrier_assets=carrier2)
    embed2=replace(iss.embed_assets,hf_public_assets=hf2,lf_public_assets=carrier2)
    iss2=replace(iss,embed_assets=embed2,lf_public_assets=lf2)
    return BlindProductionAssets(ContentCalibrationAssets(iss2),assets.weighted_joint_asset,assets.geometry_backend),list(proxies.values())


def attack_and_truth(image,angle,tx,ty):
    c,s=math.cos(math.radians(angle)),math.sin(math.radians(angle))
    center=255.5
    d=np.array([[c,-s,center*(1-c+s)],[s,c,center*(1-s-c)],[0.,0.,1.]])
    shift=np.array([[1.,0.,-tx],[0.,1.,-ty],[0.,0.,1.]])
    d=d@shift
    n=np.array([[2/511,0.,-1.],[0.,2/511,-1.],[0.,0.,1.]])
    if angle==tx==ty==0:
        return image.copy(),n@d@np.linalg.inv(n)
    pad=math.ceil((abs(c)+abs(s))*center+max(abs(tx),abs(ty))+3-center)
    padded=Image.fromarray(np.pad(np.asarray(image),((pad,pad),(pad,pad),(0,0)),mode="reflect"))
    coeff=d[:2].copy()
    coeff[:,2]+=pad
    attacked=padded.transform((512,512),Image.Transform.AFFINE,tuple(coeff.ravel()),Image.Resampling.BICUBIC)
    return attacked,n@d@np.linalg.inv(n)


class Session:
    def __init__(self,inputs,output):
        self.inputs,self.output=Path(inputs),Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        plan={"status":"STAGE3_DEVELOPMENT_ONLY","science_denominator":0,"roster":ROSTER,
              "search_plan":asdict(SearchPlan()),"source":"stage2 mechanism content-only and primary-null images",
              "conditions":CONDITIONS,"multiplier":.75,"max_unique_search_candidates_per_image":278,
              "reference_tau_descriptive_only":TAU_REFERENCE}
        (self.output/"plan.json").write_text(json.dumps(plan,indent=2))
        self.key=normalize_detection_key(os.environ["CEG_WM_ROOT_KEY"])
        runtime=self.output/"runtime"
        runtime.mkdir()
        self.pipeline,assets=build_production_runtime(REPO,load_runtime_config(REPO),hf_token=os.environ["HF_TOKEN"],runtime_root=runtime)
        self.assets,self.counters=counted_assets(assets)
        self.results=[]
        self.prepared={}

    def _current(self,unit,condition,arm):
        if (unit,arm) not in self.prepared:
            suffix="content" if arm=="positive" else "clean"
            with Image.open(self.inputs/f"{unit}__{suffix}.png") as source:
                image=source.convert("RGB")
            if arm=="positive": image=self.assets.geometry_backend.embed_final_rgb(image,.75)
            self.prepared[unit,arm]=image
        _,angle,tx,ty=next(c for c in CONDITIONS if c[0]==condition)
        return attack_and_truth(self.prepared[unit,arm],angle,tx,ty)

    def run_one(self,index):
        if index!=len(self.results): raise ValueError("execute the fixed roster once in order")
        unit,condition,arm=ROSTER[index]
        started=time.monotonic()
        output=self.output/f"image-{index:02d}"
        output.mkdir()
        result={"index":index,"unit_id":unit,"condition":condition,"arm":arm,"error":None}
        try:
            current,truth=self._current(unit,condition,arm)
            geometry=self.assets.geometry_backend.detect_geometry(current)
            cache={}  # Per-image scalar/branch records only. Never images or latents.
            active=[None]
            def observe(kind,parameters):
                active[0]="raw_h" if kind=="raw_h" else tuple(parameters)
            def pair(candidate,identifier):
                if identifier in cache: return cache[identifier]
                record={"candidate":identifier,"errors":{}}
                order=(("original",False),("optimized",True))
                if len(cache)%2: order=tuple(reversed(order))
                record["execution_order"]=[m for m,_ in order]
                for mode,reuse in order:
                    begin=time.monotonic()
                    before=sum(c.calls for c in self.counters)
                    try:
                        branches,actual=score_branches_v2(candidate,self.key,self.assets,reuse_observation=reuse)
                        stat=statistic_from_weighted_scores(branches["weighted_joint"])
                        record[mode]={"branches":branches,"statistic":asdict(stat),"actual_mode":actual}
                    except Exception as error:
                        record["errors"][mode]=f"{type(error).__name__}: {error}"
                    record[mode+"_seconds"]=time.monotonic()-begin
                    record[mode+"_encodes"]=sum(c.calls for c in self.counters)-before
                if not record["errors"]:
                    record["branch_differences"]={b:{k:record["optimized"]["branches"][b][k]-v for k,v in record["original"]["branches"][b].items()} for b in ("lf","hf","weighted_joint")}
                    record["m_difference"]=record["optimized"]["statistic"]["value"]-record["original"]["statistic"]["value"]
                    record["reference_boundary_changed"]=(record["original"]["statistic"]["value"]>TAU_REFERENCE)!=(record["optimized"]["statistic"]["value"]>TAU_REFERENCE)
                cache[identifier]=record
                with (output/"ab_candidates.jsonl").open("a") as stream:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                return record
            for mode in ("original","optimized"):
                def score(candidate):
                    record=pair(candidate,active[0])
                    if mode in record["errors"]: raise RuntimeError(record["errors"][mode])
                    return statistic_from_weighted_scores(record[mode]["branches"]["weighted_joint"])
                search=_search_candidates(current,score,lambda image:geometry,
                    SearchPlan(),on_candidate=observe)
                result[mode]={k:v for k,v in search.items() if k!="rows"}
                result[mode]["coarse_maximum"]=max((r["score"] for r in search["rows"] if r["kind"] in ("original","coarse") and r["score"] is not None),default=None)
                result[mode]["rank_order"]=[(r["kind"],r["parameters"]) for r in sorted(
                    (r for r in search["rows"] if r["score"] is not None),key=lambda r:-r["score"])]
                (output/f"search_{mode}.json").write_text(json.dumps(search,indent=2,allow_nan=False))
            result["top3_equal"]=result["original"]["selected_seeds"]==result["optimized"]["selected_seeds"]
            result["ranking_equal"]=result["original"]["rank_order"]==result["optimized"]["rank_order"]
            result["max_difference"]=(result["optimized"]["maximum"]-result["original"]["maximum"]
                if result["original"]["maximum"] is not None and result["optimized"]["maximum"] is not None else None)
            result["v1"]=asdict(_detect_core(current,self.key,self.assets,TAU_REFERENCE))
            # Truth is introduced only after both searches and does not select candidates.
            try:
                pair(rectify_attacked_rgb(current,truth),"diagnostic_oracle")
            except Exception as error:
                result["oracle_error"]=f"{type(error).__name__}: {error}"
            _,angle,_,_=next(c for c in CONDITIONS if c[0]==condition)
            nearest=min(SearchPlan().angles,key=lambda a:abs(a-angle))
            result["truth_diagnostic_only"]={"nearest_coarse_angle":nearest,
                "nearest_angle_scores":cache.get((nearest,0.,0.)),"oracle_scores":cache.get("diagnostic_oracle"),
                "top3_contains_nearest":{m:any(p[0]==nearest for p in result[m]["selected_seeds"]) for m in ("original","optimized")}}
            result["unique_search_candidates"]=len(cache)-int("diagnostic_oracle" in cache)
            result["ab_candidate_errors"]=sum(bool(r["errors"]) for r in cache.values())
            result["reference_boundary_changes"]=sum(r.get("reference_boundary_changed",False) for r in cache.values())
            result["max_abs_branch_difference"]=max((abs(x) for r in cache.values() for branch in r.get("branch_differences",{}).values() for x in branch.values()),default=None)
            result["paired_search_cost"]={mode:{"seconds":sum(r[mode+"_seconds"] for k,r in cache.items() if k!="diagnostic_oracle"),
                "encodes":sum(r[mode+"_encodes"] for k,r in cache.items() if k!="diagnostic_oracle")} for mode in ("original","optimized")}
        except Exception as error:
            result["error"]=f"{type(error).__name__}: {error}"
        result["seconds"]=time.monotonic()-started
        self.results.append(result)
        (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        with (self.output/"rows.jsonl").open("a") as stream: stream.write(json.dumps(result,allow_nan=False)+"\n")
        return result

    def pilot(self):
        result=self.run_one(0)
        return {"pilot_index":0,"pilot_error":result["error"],"pilot_seconds":result["seconds"],
                "search_complete":{m:result.get(m,{}).get("complete",False) for m in ("original","optimized")},
                "ab_candidate_errors":result.get("ab_candidate_errors"),"paired_search_cost":result.get("paired_search_cost"),
                "remaining_images":len(ROSTER)-1,
                "remaining_seconds_estimate_from_this_image":result["seconds"]*(len(ROSTER)-1) if result["error"] is None else None,
                "estimate_limit":"first unattacked image; top3 divergence and attacks can increase cost"}

    def remaining(self):
        for index in range(len(self.results),len(ROSTER)):
            result=self.run_one(index)
            print({k:result.get(k) for k in ("index","condition","arm","seconds","error")},flush=True)
        summary={"status":"STAGE3_DEVELOPMENT_ONLY","science_denominator":0,"planned_images":len(ROSTER),
            "written_images":len(self.results),"image_errors":sum(r["error"] is not None for r in self.results),
            "ab_candidate_errors":sum(r.get("ab_candidate_errors",0) for r in self.results),
            "incomplete_search_images":sum(any(not r.get(m,{}).get("complete",False) for m in ("original","optimized")) for r in self.results),
            "v1_incomplete_images":sum(not r.get("v1",{}).get("method_complete",False) for r in self.results),
            "torch":torch.__version__,"gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None}
        (self.output/"summary.json").write_text(json.dumps(summary,indent=2))
        return summary
