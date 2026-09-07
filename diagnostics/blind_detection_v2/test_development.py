"""Focused geometry and dataset separation checks; no content success claim."""
import numpy as np
from PIL import Image
import pytest

from diagnostics.blind_detection_v2.development import ROSTER, PERTURBATIONS, ANGLES, STRENGTHS, rotation, sampler_h
from cegwm.geometry_v7.r1b import rectify_attacked_rgb


@pytest.mark.quick
@pytest.mark.parametrize("angle",[-13.,7.,21.])
def test_truth_sampler_recovers_asymmetric_landmarks(angle):
    image=np.zeros((512,512,3),dtype=np.uint8)
    image[140:165,190:210,0]=255
    image[320:340,350:365,1]=255
    original=Image.fromarray(image)
    recovered=np.asarray(rectify_attacked_rgb(rotation(original,angle),sampler_h(angle)))
    yy,xx=np.mgrid[:512,:512]
    for channel in (0,1):
        before=image[:,:,channel].astype(float)
        after=recovered[:,:,channel].astype(float)
        assert abs((before*xx).sum()/before.sum()-(after*xx).sum()/after.sum()) < .2
        assert abs((before*yy).sum()/before.sum()-(after*yy).sum()/after.sum()) < .2


@pytest.mark.quick
def test_development_is_small_disjoint_and_keeps_all_planned_conditions():
    assert len(ROSTER)==4 and len({r['seed'] for r in ROSTER})==4
    assert all(r['seed'] not in range(2027000000,2027040000) for r in ROSTER)
    assert all(r['seed'] not in range(2026101000,2026101004) for r in ROSTER)
    assert len(ROSTER)*(len(STRENGTHS)*len(ANGLES)+len(ANGLES)+2*len(PERTURBATIONS))==216
    assert len(set(PERTURBATIONS))==21


@pytest.mark.quick
@pytest.mark.parametrize("failure",["sync_embed","public_warp"])
def test_independent_diagnostics_continue_after_local_failure(tmp_path,monkeypatch,failure):
    from dataclasses import dataclass
    from types import SimpleNamespace
    import json
    import torch
    import diagnostics.blind_detection_v2.development as dev
    import experiments.run_blind_detection_v1 as production
    import cegwm.runtime.content_iss_sd35 as iss
    import cegwm.runtime.blind_detection as blind
    import cegwm.geometry_v7.r1b as warp
    from cegwm.geometry_v7.contracts import estimate_geometry
    @dataclass
    class Result:
        method_complete: bool = True
        operational_error: str | None = None
    class Model:
        def embed(self,x):
            if failure=="sync_embed": raise RuntimeError("injected sync failure")
            return {"imgs_w":x}
        def unwarp(self,x,raw,size): return x
    geometry=estimate_geometry(0.,((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)),
        raw_syncseal_corners=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)))
    adapter=SimpleNamespace(device=torch.device("cpu"),model=Model(),detect_geometry=lambda image:geometry)
    assets=SimpleNamespace(geometry_backend=adapter,content_assets=SimpleNamespace(iss_assets=None))
    monkeypatch.setenv("CEG_WM_ROOT_KEY","local-test-key-0000")
    monkeypatch.setenv("HF_TOKEN","local-test-token")
    monkeypatch.setattr(production,"load_runtime_config",lambda _: {"device":"cpu"})
    monkeypatch.setattr(production,"build_production_runtime",lambda *a,**k:(None,assets))
    monkeypatch.setattr(iss,"run_content_iss_evaluation_pair",lambda *a,**k:SimpleNamespace(image=Image.new("RGB",(512,512)),primary_null=Image.new("RGB",(512,512))))
    monkeypatch.setattr(blind,"_detect_core",lambda *a:Result())
    monkeypatch.setattr(blind,"_score_current_rgb",lambda *a:SimpleNamespace(value=0.))
    if failure=="public_warp":
        def broken(*a): raise RuntimeError("injected public warp failure")
        monkeypatch.setattr(warp,"rectify_attacked_rgb",broken)
    monkeypatch.setattr(dev,"ROSTER",ROSTER[:1])
    monkeypatch.setattr(dev,"STRENGTHS",(.75,))
    monkeypatch.setattr(dev,"ANGLES",(0.,))
    monkeypatch.setattr(dev,"PERTURBATIONS",((0.,0.,0.),))
    out=tmp_path/"result"
    dev.run(out)
    rows=[json.loads(line) for line in (out/"rows.jsonl").read_text().splitlines()]
    assert len(rows)==4
    if failure=="sync_embed":
        assert all(r["error"] is None for r in rows if r["arm"]=="negative")
        assert all(r["error"] is not None for r in rows if r["arm"]=="positive")
    else:
        for row in rows[:2]:
            assert row["native_warp_score"]==0.
            assert "adapter_warp_score" in row["stage_errors"]
            assert row["error"] is not None
