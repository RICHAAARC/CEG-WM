"""Injected execution verifies independent A/B selection, never content efficacy."""
from dataclasses import dataclass
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from diagnostics.blind_detection_v2 import stage3
from cegwm.runtime import blind_detection_v2 as core
from cegwm.geometry_v7.contracts import estimate_geometry


@pytest.mark.quick
@pytest.mark.parametrize("optimized_error",[False,True])
def test_ab_paths_choose_independently_and_reuse_only_same_candidate_records(tmp_path,monkeypatch,optimized_error):
    @dataclass
    class V1:
        method_complete: bool=True
    session=object.__new__(stage3.Session)
    session.output=tmp_path
    session.results=[]
    session.counters=[]
    session.key=b"injected-stage3-key"
    h=estimate_geometry(0.,((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)))
    session.assets=SimpleNamespace(geometry_backend=SimpleNamespace(detect_geometry=lambda image:h))
    session._current=lambda *args:(Image.new("RGB",(512,512),(100,100,100)),np.eye(3))
    monkeypatch.setattr(stage3,"SearchPlan",lambda:core.SearchPlan(angle_limit=1.,coarse_step=.5,top_k=1,translation_offsets=(0.,)))
    def warp(image,matrix):
        value=100+round(np.degrees(np.arctan2(matrix[1][0],matrix[0][0]))*20)
        return Image.new("RGB",(512,512),(value,value,value))
    monkeypatch.setattr(core,"rectify_attacked_rgb",warp)
    calls=[]
    def score(image,key,assets,*,reuse_observation=False):
        value=image.getpixel((0,0))[0]/255
        calls.append((value,reuse_observation))
        if optimized_error and reuse_observation: raise RuntimeError("injected optimized error")
        values={"registered":-value if reuse_observation else value,**{f"wrong_{i:02d}":0. for i in range(16)}}
        return {b:dict(values) for b in ("lf","hf","weighted_joint")},"injected"
    monkeypatch.setattr(stage3,"score_branches_v2",score)
    monkeypatch.setattr(stage3,"_detect_core",lambda *a:V1())
    result=session.run_one(0)
    assert result["error"] is None and result["original"]["complete"]
    records=[json.loads(line) for line in (tmp_path/"image-00/ab_candidates.jsonl").read_text().splitlines()]
    assert len(calls)==2*len(records)
    assert len(records)==result["unique_search_candidates"]+1
    if optimized_error:
        assert not result["optimized"]["complete"] and result["ab_candidate_errors"]==len(records)
    else:
        assert not result["top3_equal"]
        assert result["original"]["selected_seeds"]==[(1.,0.,0.)]
        assert result["optimized"]["selected_seeds"]==[(-1.,0.,0.)]


@pytest.mark.quick
def test_joint_attack_and_truth_are_consistent_without_sequential_image_warps():
    pixels=np.zeros((512,512,3),dtype=np.uint8)
    pixels[110:150,200:240,0]=230
    pixels[310:350,340:365,1]=210
    image,truth=stage3.attack_and_truth(Image.fromarray(pixels),18.3,.75,-.75)
    recovered=np.asarray(stage3.rectify_attacked_rgb(image,truth),dtype=float)
    yy,xx=np.mgrid[:512,:512]
    for channel in (0,1):
        before=pixels[:,:,channel].astype(float)
        after=recovered[:,:,channel]
        for grid in (xx,yy):
            assert abs((before*grid).sum()/before.sum()-(after*grid).sum()/after.sum())<.2
