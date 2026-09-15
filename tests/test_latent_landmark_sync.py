"""M0 engineering tests. No real model, VAE or carrier-survival evidence."""
import inspect
import numpy as np
import pytest
from cegwm.method.latent_landmark_sync import (
    CENTERS, codes, carrier, kernel, find_peaks, responses, solve, transform,
    latent_to_rgb_h, public_h, pil_sampling_public_h,
)
from cegwm.runtime.latent_landmark_sd35 import LandmarkCallback, read_image

pytestmark = pytest.mark.unit


def peaks_for(h, indices=range(8)):
    points = transform(np.linalg.inv(h), CENTERS)
    return [dict(identity=i, rank=0, xy=points[i].tolist(), score=.9, refinement="synthetic") for i in indices]


def rigid(angle, translation):
    t = np.deg2rad(angle)
    h = np.array([[np.cos(t),-np.sin(t),translation[0]], [np.sin(t),np.cos(t),translation[1]],[0,0,1.]])
    return h


def test_kernel_codes_and_total_budget():
    q = codes("m0-fixed-key")
    np.testing.assert_allclose(q.T@q, np.eye(8), atol=1e-14)
    np.testing.assert_array_equal(q, codes("m0-fixed-key"))
    assert not np.allclose(q, codes("external-key"))
    yy,xx = np.mgrid[-3:4,-3:4]
    k = kernel(xx-.3, yy+.2)
    assert abs(k.sum()) < 1e-14
    assert abs(np.linalg.norm(k)-1) < 1e-14
    for rms in (.005,.01,.02):
        z = carrier("m0-fixed-key", rms)
        assert abs(np.sqrt(np.mean(z*z))-rms) < 1e-14
        np.testing.assert_allclose(z.sum(axis=(2,3)), 0, atol=1e-13)


@pytest.mark.parametrize("angle", [-15.,-10.,0.,10.,15.])
def test_exact_signed_rigid_and_missing_points(angle):
    h = rigid(angle, [2.1,-1.7])
    result = solve(peaks_for(h, [0,2,4,6]))
    assert result["accepted"]
    np.testing.assert_allclose(result["H_latent"], h, atol=1e-12)


def test_outliers_wrong_identities_duplicates_and_degeneracy():
    h = rigid(10, [1.,2.])
    p = peaks_for(h)
    p[1]["xy"] = [9.,52.]
    p[3]["xy"] = p[5]["xy"]
    result = solve(p)
    assert result["accepted"] and len(result["matches"]) == 6
    np.testing.assert_allclose(result["H_latent"], h, atol=1e-12)
    assert not solve(peaks_for(h, [0,2,4]))["accepted"]
    assert not solve([dict(p[0], identity=i) for i in range(8)])["accepted"]
    assert not solve([])["accepted"]
    assert not solve(peaks_for(rigid(30,[0,0])))["accepted"]


def test_synthetic_carrier_read_and_no_signal():
    peaks = find_peaks(responses(carrier("m0-fixed-key", .01), "m0-fixed-key"))
    result = solve(peaks)
    assert result["accepted"] and len(result["matches"]) == 8
    errors = [np.linalg.norm(np.array(p["xy"])-CENTERS[p["identity"]]) for p in result["matches"]]
    assert max(errors) < .15
    assert find_peaks(responses(np.zeros((1,16,64,64)), "m0-fixed-key")) == []


def test_pixel_center_conjugation_and_direction():
    a = np.array([[8.,0,3.5],[0,8.,3.5],[0,0,1.]])
    n = np.array([[2/511,0,-1],[0,2/511,-1],[0,0,1.]])
    for angle in (-10,10):
        h = rigid(angle,[.3,-.2])
        observed = transform(np.linalg.inv(h), CENTERS)
        rgb_observed = transform(a, observed)
        rgb_canonical = transform(a, CENTERS)
        np.testing.assert_allclose(transform(latent_to_rgb_h(h), rgb_observed), rgb_canonical, atol=1e-12)
        np.testing.assert_allclose(transform(public_h(h), transform(n, rgb_observed)), transform(n,rgb_canonical), atol=1e-12)
    # PIL uses inverse output-to-input coordinates; a +8 input-to-output shift
    # must put a single integer pixel at x+8, under the main rectifier.
    from PIL import Image
    from cegwm.geometry_v7.r1b import rectify_attacked_rgb
    rgb = np.zeros((512,512,3), dtype=np.uint8); rgb[200,200] = 255
    moved = np.asarray(rectify_attacked_rgb(Image.fromarray(rgb), public_h(rigid(0,[1,0]))))
    np.testing.assert_array_equal(moved[200,208], [255,255,255])


def test_blind_signature_and_callback_order():
    assert list(inspect.signature(read_image).parameters) == ["image","key","image_processor","vae","peak_threshold","ambiguity_threshold"]
    import torch
    from types import SimpleNamespace
    def content(pipe, index, timestep, kw):
        return dict(kw, latents=kw["latents"]+1) if index == 18 else kw
    cb = LandmarkCallback(content,"m0-fixed-key", .01)
    pipe = SimpleNamespace(scheduler=SimpleNamespace(timesteps=torch.arange(20)))
    z = torch.zeros((1,16,64,64), dtype=torch.float64)
    kw = dict(latents=z)
    for index in range(20):
        kw = cb(pipe,index,index,kw)
    np.testing.assert_allclose(kw["latents"].numpy(), 1+carrier("m0-fixed-key", .01), atol=1e-15)
    assert cb.trace()["updates_after_write"] == 1
    with pytest.raises(RuntimeError): cb(pipe,18,18,kw)


def test_candidate_content_object_protocol(monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    import cegwm.runtime.paper_detection_v2 as scoring
    import cegwm.runtime.latent_landmark_sd35 as candidate
    from cegwm.runtime.latent_landmark_sd35 import detect_candidate
    image = Image.new("RGB", (512,512))
    calls = []
    values = iter([2., 0., 2., 0., 0.])
    def score(img,key,assets):
        calls.append(key)
        return SimpleNamespace(value=next(values))
    monkeypatch.setattr(scoring, "score_current_rgb", score)
    yes = dict(accepted=True,H_latent=np.eye(3).tolist())
    no = dict(accepted=False,H_latent=None)
    ambiguous=dict(accepted=False,H_latent=np.eye(3).tolist(),reason="ambiguous")
    reads=[]
    geometries=iter([yes,no,ambiguous])
    def reader(*args):
        reads.append(args); return dict(geometry=next(geometries))
    monkeypatch.setattr(candidate,"read_image",reader)
    public=SimpleNamespace(image_processor=object(),vae=object())
    assets=SimpleNamespace(content_assets=SimpleNamespace(iss_assets=SimpleNamespace(embed_assets=SimpleNamespace(hf_public_assets=public))))
    assert detect_candidate(image,"key",assets,1.,.5,0.)["route"] == "pre_early_return"
    assert len(calls) == 1 and len(reads)==0
    assert detect_candidate(image,"key",assets,1.,.5,0.)["route"] == "single_H_post"
    assert len(calls) == 3 and len(reads)==1
    assert not detect_candidate(image,"key",assets,1.,.5,0.)["accepted"]
    assert len(calls) == 4 and len(reads)==2 and set(calls) == {"key"}
    assert not detect_candidate(image,"key",assets,1.,.5,0.)["accepted"]
    assert len(calls)==5 and len(reads)==3
    assert "geometry" not in inspect.signature(detect_candidate).parameters


def test_distinct_equal_hypotheses_are_ambiguous():
    peaks = peaks_for(np.eye(3)) + [dict(p,rank=1) for p in peaks_for(rigid(0,[2,0]))]
    result=solve(peaks)
    assert not result["accepted"] and result["reason"] == "ambiguous"
    assert result["runner_up"] is not None


def test_m1_fixed_denominator_and_shared_threshold(tmp_path):
    import json
    from experiments.run_latent_landmark_sync_v1 import initialize,summarize,dump
    cfg,rows=initialize(tmp_path)
    assert len(rows)==24 and all(r["status"]=="NOT_EXECUTED" for r in rows)
    assert summarize(tmp_path)["selected_rms"] is None
    for row in rows:
        row.update(status="SCORED",content_pre=2.,quality_vs_plain=dict(psnr=40.,identical=False,lpips=.01))
        signal=peaks_for(np.eye(3)) if row["arm"].startswith("rms") else []
        row["correct_key_reader"]=dict(peaks=signal)
        row["external_wrong_key_reader"]=dict(peaks=[dict(p,score=.2) for p in peaks_for(np.eye(3))])
    dump(tmp_path/"rows.json",rows)
    summary=summarize(tmp_path)
    assert summary["selected_rms"]==.005 and summary["peak_threshold"]==.2
    assert summary["negative_queries"]==36
    # A failed source remains in the four-source denominator and stops selection.
    rows[3]["status"]="ERROR"
    dump(tmp_path/"rows.json",rows)
    incomplete=summarize(tmp_path)
    assert incomplete["selected_rms"] is None and incomplete["planned_images"]==24


def test_mock_m1_real_entry_protocol(tmp_path,monkeypatch):
    """Exercise the real orchestration with synthetic assets; no science rows."""
    import sys
    from types import SimpleNamespace
    from PIL import Image
    import experiments.run_latent_landmark_sync_v1 as runner
    import experiments.run_paper_main_worker_v2 as factory
    import cegwm.runtime.diffusers_sd35 as plain
    import cegwm.runtime.latent_landmark_sd35 as candidate
    import cegwm.runtime.paper_detection_v2 as scoring
    import cegwm.method.content_iss as iss_method
    class Perceptual:
        def to(self,*args): return self
        def eval(self): return self
    monkeypatch.setitem(sys.modules,"lpips",SimpleNamespace(LPIPS=lambda **kw:Perceptual()))
    public=SimpleNamespace(image_processor=object(),vae=object())
    iss=SimpleNamespace(embed_assets=SimpleNamespace(hf_public_assets=public),lf_public_assets=object(),iss_asset=object())
    pipeline=SimpleNamespace(scheduler=SimpleNamespace(config={}),vae=SimpleNamespace(config={}))
    assets=SimpleNamespace(content_assets=SimpleNamespace(iss_assets=iss),geometry_backend=SimpleNamespace(embed_final_rgb=lambda image,rms:image.copy()))
    monkeypatch.setattr(factory,"_build_runtime",lambda path:dict(pipeline=pipeline,assets=assets,key=b"test",device="cpu"))
    calls=[]
    monkeypatch.setattr(plain,"run_sd35_plain",lambda *args,**kwargs:Image.new("RGB",(512,512)))
    def generate(*args):
        calls.append(args[-1]); return Image.new("RGB",(512,512)),dict(mock=True)
    monkeypatch.setattr(candidate,"generate_content_landmark",generate)
    monkeypatch.setattr(candidate,"read_image",lambda *args:dict(peaks=[]))
    monkeypatch.setattr(scoring,"score_current_rgb",lambda *args:SimpleNamespace(value=2.))
    monkeypatch.setattr(iss_method,"score_content_iss_image",lambda *args:0.)
    monkeypatch.setattr(iss_method,"iss_beta",lambda *args:1.)
    monkeypatch.setattr(runner,"quality",lambda *args:dict(psnr=40.,identical=False,lpips=.01))
    assert runner.run(tmp_path,tmp_path/"runtime")==0
    cfg,rows=runner.initialize(tmp_path)
    assert len(rows)==24 and all(r["status"]=="SCORED" for r in rows)
    assert calls==[0.,.005,.01,.02]*4
    assert runner.summarize(tmp_path)["status"]=="NO_CLEAN_READABLE_WINDOW"


@pytest.mark.parametrize("angle",[-10.,10.,90.])
def test_pil_rotated_impulse_uses_half_center_sampling(angle):
    from PIL import Image
    from cegwm.geometry_v7.r1b import rectify_attacked_rgb
    # Rotate about the location of the impulse: a correct mapping leaves it at
    # exactly the same integer pixel for every angle, exposing half-center bias.
    point=np.array([200.,210.])
    h_rgb=rigid(angle,[0,0]); h_rgb[:2,2]=point-h_rgb[:2,:2]@point
    a=np.array([[8.,0,3.5],[0,8.,3.5],[0,0,1.]])
    h_lat=np.linalg.inv(a)@h_rgb@a
    rgb=np.zeros((512,512,3),dtype=np.uint8); rgb[210,200]=255
    corrected=np.asarray(rectify_attacked_rgb(Image.fromarray(rgb),pil_sampling_public_h(h_lat)))[:,:,0].astype(float)
    yy,xx=np.mgrid[:512,:512]
    centroid=np.array([(xx*corrected).sum(),(yy*corrected).sum()])/corrected.sum()
    np.testing.assert_allclose(centroid,point,atol=.01)
    np.testing.assert_allclose(transform(latent_to_rgb_h(h_lat),[point]),[point],atol=1e-12)
    unadapted=np.asarray(rectify_attacked_rgb(Image.fromarray(rgb),public_h(h_lat)))[:,:,0].astype(float)
    wrong=np.array([(xx*unadapted).sum(),(yy*unadapted).sum()])/unadapted.sum()
    assert np.linalg.norm(wrong-point)>.08
