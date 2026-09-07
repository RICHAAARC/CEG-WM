"""Search mechanics only: injected scores are not watermark evidence."""
import numpy as np
from PIL import Image
import pytest
from diagnostics.blind_detection_v2 import search_prototype as prototype
from diagnostics.blind_detection_v2.development import rotation


@pytest.mark.quick
def test_finite_search_keeps_all_scores_and_never_threshold_shortcuts(monkeypatch):
    source=Image.new("RGB",(512,512),(53,71,99))
    seen=[]
    def warp(image,matrix):
        assert image.getpixel((0,0))==(53,71,99)
        seen.append(matrix)
        return image.copy()
    monkeypatch.setattr(prototype,"rectify_attacked_rgb",warp)
    calls=[]
    def scorer(image):
        image.putpixel((0,0),(0,0,0))
        calls.append(len(calls))
        return 10000. if len(calls)==1 else -float(len(calls))
    plan=prototype.SearchPlan(angle_limit=1.,coarse_step=.5,top_k=1)
    result=prototype.search(source,scorer,plan=plan)
    assert result["maximum"]==10000.
    assert result["score_calls"]==31  # 5 coarse including original + 26 refinements
    assert result["complete"]
    assert source.getpixel((0,0))==(53,71,99)
    assert len(seen)==30


@pytest.mark.quick
def test_failed_and_nonfinite_scores_are_retained_without_a_positive():
    calls=[]
    def scorer(image):
        calls.append(1)
        if len(calls)==1: raise RuntimeError("injected score failure")
        return float("nan")
    result=prototype.search(Image.new("RGB",(512,512)),scorer,
        plan=prototype.SearchPlan(angle_limit=1.,coarse_step=1.,top_k=1))
    assert result["score_calls"]==3
    assert len(result["rows"])==3 and all(r["error"] for r in result["rows"])
    assert result["maximum"] is None and not result["complete"]
    assert "positive" not in result


@pytest.mark.quick
def test_off_grid_asymmetric_image_refinement_is_sampled_from_original():
    reference=np.zeros((512,512,3),dtype=np.uint8)
    reference[100:190,160:183,0]=230
    reference[330:355,350:410,1]=180
    observed=rotation(Image.fromarray(reference),1.3)
    def fixture_score(image):
        return -float(np.mean((np.asarray(image,dtype=float)-reference)**2))
    result=prototype.search(observed,fixture_score,
        plan=prototype.SearchPlan(angle_limit=2.,coarse_step=.5,top_k=2,translation_offsets=(0.,)))
    winner=max(result["rows"],key=lambda r:r["score"])
    assert winner["parameters"]==(1.25,0.,0.)
    assert winner["score"]>max(r["score"] for r in result["rows"] if r["kind"] in ("original","coarse"))


@pytest.mark.quick
def test_default_budget_and_unavailable_h_do_not_remove_rotation_candidates():
    from cegwm.geometry_v7.contracts import estimate_geometry
    unsupported=estimate_geometry(0.,((0.,0.),(0.,0.),(0.,0.),(0.,0.)))
    plan=prototype.SearchPlan(angle_limit=.5,coarse_step=.5,top_k=1,translation_offsets=(0.,))
    result=prototype.search(Image.new("RGB",(512,512)),lambda image:0.,lambda image:unsupported,plan)
    assert result["geometry"]["status"]=="UNSUPPORTED"
    assert result["complete"] and result["score_calls"]==5
    assert prototype.SearchPlan().score_upper_bound==200


@pytest.mark.quick
@pytest.mark.parametrize("status",["UNSUPPORTED","UNRELIABLE"])
def test_inconsistent_geometry_never_claims_complete(status):
    from dataclasses import replace
    from cegwm.geometry_v7.contracts import estimate_geometry, GeometryStatus
    normal=estimate_geometry(0.,((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)))
    inconsistent=replace(normal,status=GeometryStatus(status),error="injected inconsistency")
    result=prototype.search(Image.new("RGB",(512,512)),lambda image:0.,lambda image:inconsistent,
        prototype.SearchPlan(angle_limit=.5,coarse_step=.5,top_k=1,translation_offsets=(0.,)))
    assert not result["complete"]
    assert result["geometry"]["error"]
    assert not any(r["kind"]=="raw_h" for r in result["rows"])
    assert result["score_calls"]==5  # Independent rotation search still runs.
