"""Uncalibrated BlindDetection-V2 candidate. No formal positive decision.

The scorer must compute the complete registered-minus-16-wrong-max statistic
on the supplied current RGB. No original image, truth or threshold is accepted.
"""
from dataclasses import dataclass, asdict
import math
from typing import Callable

import numpy as np
from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate, GeometryStatus
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.runtime.blind_detection import _raw_h, _geometry_disposition, _score_current_rgb, BlindProductionAssets
from cegwm.method.blind_detection import BlindStatistic
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key


@dataclass(frozen=True)
class SearchPlan:
    angle_limit: float = 30.
    coarse_step: float = .5
    top_k: int = 3
    fine_offsets: tuple = (-.25,0.,.25)
    translation_offsets: tuple = (-1.,0.,1.)

    def __post_init__(self):
        if not (math.isfinite(self.angle_limit) and 0<self.angle_limit<=180):
            raise ValueError("finite positive angular domain required")
        if not (math.isfinite(self.coarse_step) and self.coarse_step>0):
            raise ValueError("positive angular step required")
        if not math.isclose(self.angle_limit/self.coarse_step,round(self.angle_limit/self.coarse_step)):
            raise ValueError("domain must contain an integral number of steps")
        if not isinstance(self.top_k,int) or isinstance(self.top_k,bool) or self.top_k<1:
            raise ValueError("positive top_k required")
        for offsets in (self.fine_offsets,self.translation_offsets):
            if not offsets or len(set(offsets))!=len(offsets) or any(not math.isfinite(x) for x in offsets):
                raise ValueError("finite unique refinement offsets required")

    @property
    def angles(self):
        n=round(self.angle_limit/self.coarse_step)
        return tuple(i*self.coarse_step for i in range(-n,n+1))

    @property
    def score_upper_bound(self):
        # Conservative for configurable plans; the default has 200 calls.
        refinement=len(self.fine_offsets)*len(self.translation_offsets)**2
        overlap=int(0. in self.fine_offsets and 0. in self.translation_offsets)
        return len(self.angles)+1+self.top_k*(refinement-overlap)


def rigid_hypothesis(angle,tx=0.,ty=0.):
    """Public 512-canvas hypothesis in the frozen Pillow sampling convention."""
    c,s=math.cos(math.radians(angle)),math.sin(math.radians(angle))
    center=255.5
    pixel=np.array([[c,-s,center*(1-c+s)+tx],[s,c,center*(1-s-c)+ty],[0.,0.,1.]])
    n=np.array([[2/511,0.,-1.],[0.,2/511,-1.],[0.,0.,1.]])
    return n@pixel@np.linalg.inv(n)


def _search_candidates(image: Image.Image, score_current_rgb: Callable,
           geometry_backend: Callable | None = None, plan: SearchPlan = SearchPlan(),
           *, on_candidate: Callable | None = None):
    current=require_ordinary_rgb_image(image)
    if current.size!=(512,512):
        raise ValueError("public canvas must be 512x512")
    rows=[]
    seen=set()
    score_calls=0

    def evaluate(kind,parameters,matrix):
        nonlocal score_calls
        row={"kind":kind,"parameters":parameters,"score":None,"error":None}
        try:
            # Every warp starts from current. Refinement never warps a prior candidate.
            candidate=current.copy() if matrix is None else rectify_attacked_rgb(current,matrix)
            if on_candidate is not None:
                on_candidate(kind,parameters)
            score_calls+=1
            score=score_current_rgb(candidate)
            if isinstance(score,BlindStatistic):
                row["statistic"]=asdict(score)
                score=score.value
            if isinstance(score,bool) or not isinstance(score,(float,int)) or not math.isfinite(score):
                raise ValueError("candidate score must be the finite complete m statistic")
            row["score"]=float(score)
        except Exception as error:
            row["error"]=f"{type(error).__name__}: {error}"
        rows.append(row)
        return row

    coarse=[evaluate("original",(0.,0.,0.),None)]
    seen.add((0.,0.,0.))
    for angle in plan.angles:
        if angle==0: continue
        parameters=(angle,0.,0.)
        seen.add(parameters)
        coarse.append(evaluate("coarse",parameters,rigid_hypothesis(*parameters)))
    geometry_record={"status":"NOT_REQUESTED","reason":None,"error":None}
    if geometry_backend is not None:
        try:
            geometry=geometry_backend(current.copy())
            if not isinstance(geometry,GeometryEstimate):
                raise TypeError("typed geometry estimate required")
            geometry_record["status"]=geometry.status.value
            disposition,error=_geometry_disposition(geometry)
            geometry_record["reason"]=geometry.error
            if disposition=="OPERATIONAL":
                raise RuntimeError(error)
            if disposition=="RAW_H":
                evaluate("raw_h",None,_raw_h(geometry))
        except Exception as error:
            geometry_record["error"]=f"{type(error).__name__}: {error}"
    ranked=sorted((r for r in coarse if r["score"] is not None),key=lambda r:-r["score"])
    seeds=ranked[:plan.top_k]  # Stable ties; no threshold cutoff or oracle ranking.
    for seed in seeds:
        for offset in plan.fine_offsets:
            angle=seed["parameters"][0]+offset
            if abs(angle)>plan.angle_limit: continue
            for tx in plan.translation_offsets:
                for ty in plan.translation_offsets:
                    parameters=(angle,tx,ty)
                    if parameters in seen: continue
                    seen.add(parameters)
                    evaluate("refinement",parameters,rigid_hypothesis(*parameters))
    finite=[r for r in rows if r["score"] is not None]
    return {"status":"UNCALIBRATED_BLIND_DETECTION_V2_CANDIDATE","science_denominator":0,
            "statistic":"max_over_attempted_complete_per_candidate_m",
            "maximum":max((r["score"] for r in finite),default=None),
            "complete":geometry_record["error"] is None and all(r["error"] is None for r in rows),
            "geometry":geometry_record,"selected_seeds":[r["parameters"] for r in seeds],
            "score_calls":score_calls,"candidate_attempts":len(rows),
            "score_upper_bound":plan.score_upper_bound,"rows":rows}


def detect_watermark_v2(image,key,assets: BlindProductionAssets,*,reuse_observation=False):
    """Image/key/public-assets only. V1 thresholds do not authorize V2 positives."""
    if type(assets) is not BlindProductionAssets:
        raise TypeError("V2 requires real BlindProductionAssets")
    detection_key=normalize_detection_key(key)
    modes={}
    if reuse_observation:
        from cegwm.runtime.blind_scoring_v2 import score_branches_v2
        from cegwm.method.blind_detection import statistic_from_weighted_scores
        def scorer(rgb):
            branches,mode=score_branches_v2(rgb,detection_key,assets,reuse_observation=True)
            modes[mode]=modes.get(mode,0)+1
            return statistic_from_weighted_scores(branches["weighted_joint"])
    else:
        scorer=lambda rgb:_score_current_rgb(rgb,detection_key,assets)
    result=_search_candidates(image,scorer,assets.geometry_backend.detect_geometry,SearchPlan())
    result["reuse_observation_requested"]=bool(reuse_observation)
    result["scoring_modes"]=modes if reuse_observation else {"original":result["score_calls"]}
    return result
