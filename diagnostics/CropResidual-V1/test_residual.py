"""Synthetic CPU engineering tests, no GPU/model download/generation."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
from PIL import Image
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE.parents[1], HERE.parents[1] / "alive" / "CEG-WM") if (p / "src/cegwm").is_dir())
sys.path[:0] = [str(HERE), str(REPO), str(REPO / "src")]
import run_residual as d
from cegwm.geometry_v7.r1b import rectify_attacked_rgb, _pixel_output_to_source
from cegwm.method.blind_detection import statistic_from_weighted_scores
from cegwm.shared.keys import normalize_detection_key, public_key_digest
torch.set_num_threads(1)


def crop():
    return {"sampler_h_observed_to_canonical": [[458/512,0,54/512/511],[0,458/512,54/512/511],[0,0,1]],
            "physical_h_observed_to_canonical": [[458/512,0,0],[0,458/512,0],[0,0,1]]}


def test_fixed_grid_and_analytic_pixel_convention():
    assert len(d.cases()) == 22
    assert len({c["id"] for c in d.cases()}) == 22
    truth = crop()
    for case in d.cases():
        h = d.matrix_for(case, truth, truth["sampler_h_observed_to_canonical"])
        assert np.isfinite(h).all()
        if case["kind"] == "translation":
            result = d.geometry_metrics(h, truth)
            assert result["effective_corner_rmse_px"] == pytest.approx(abs(case["value"]))
            shift = [case["value"],0] if case["axis"] == "x" else [0,case["value"]]
            assert result["effective_center_shift_xy_px"] == pytest.approx(shift)
        if case["kind"] == "scale":
            assert d.geometry_metrics(h, truth)["effective_center_shift_xy_px"] == pytest.approx([0,0], abs=1e-10)
    matrix = d.matrix_for(d.cases()[0], truth, None)
    assert d.geometry_metrics(matrix, truth)["effective_corner_rmse_px"] < 1e-10
    coeff = _pixel_output_to_source(matrix.tolist())
    assert (coeff[0], coeff[2]) == pytest.approx((512/458, -27*512/458))


def test_one_pass_translation_moves_support_not_missing_content():
    image = Image.new("RGB", (512,512), (100,100,100))
    case = {"id":"probe", "kind":"translation", "axis":"x", "value":2.}
    matrix = d.matrix_for(case, crop(), None)
    actual = np.asarray(rectify_attacked_rgb(image, matrix.tolist()))
    expected = np.zeros((512,512,3), dtype=np.uint8)
    expected[27:485,29:487] = 100
    np.testing.assert_array_equal(actual, expected)


def test_replay_gate_checks_wrong_keys_not_only_margin():
    labels = ("registered", *(f"wrong_{i:02d}" for i in range(16)))
    baseline = {"branches": {k: dict.fromkeys(labels, 0.) for k in ("lf","hf","weighted_joint")}, "statistic":{"value":0.}}
    changed = json.loads(json.dumps(baseline))
    changed["branches"]["hf"]["wrong_05"] = .1
    assert not d.replay_comparison(changed, baseline)["passed"]


def test_vae_only_score_matches_original_shared_observation_arithmetic():
    from types import SimpleNamespace
    from cegwm.method.lf import (FrozenLFPublicAssets, LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID)
    from cegwm.method.hf import FrozenHFPublicAssets
    from cegwm.method.content_whitening import FrozenContentWhiteningLFPublicAssets
    from experiments.run_blind_detection_v1 import load_whitening_asset_semantic, load_weighted_asset_semantic
    from cegwm.runtime.content_weighted_joint_sd35 import derive_stability_wrong_keys
    from cegwm.runtime.blind_scoring_v2 import score_branches_v2
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.zeros(1))
            self.config=SimpleNamespace(scaling_factor=1.,shift_factor=0.)
        def encode(self,pixels):
            z=torch.randn((1,16,64,64),generator=torch.Generator().manual_seed(7))
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:z))
    vae=VAE()
    processor=SimpleNamespace(preprocess=lambda image:torch.zeros((1,3,512,512)))
    lf=FrozenLFPublicAssets(vae,processor,d.MODEL+":image_processor",LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
                           LF_BLOCKNORM_DETECTOR_STATISTIC_ID,LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID)
    reader=d.Reader.__new__(d.Reader)
    reader.vae,reader.processor=vae,processor
    reader.lf=FrozenContentWhiteningLFPublicAssets(lf,load_whitening_asset_semantic(REPO))
    reader.hf=FrozenHFPublicAssets(vae,processor,d.MODEL+":image_processor")
    reader.weighted=load_weighted_asset_semantic(REPO)
    reader.key=normalize_detection_key("00"*32)
    reader.keys=(reader.key,*derive_stability_wrong_keys(reader.key))
    image=Image.new("RGB",(512,512))
    actual=reader.score(reader.observe(image))
    assets=SimpleNamespace(content_assets=SimpleNamespace(iss_assets=SimpleNamespace(
        lf_public_assets=reader.lf,hf_public_assets=reader.hf)),weighted_joint_asset=reader.weighted)
    branches,mode=score_branches_v2(image,reader.key,assets,reuse_observation=True)
    assert mode=="shared_candidate_observation"
    assert actual["branches"]==branches
    assert actual["statistic"]==asdict(statistic_from_weighted_scores(branches["weighted_joint"]))


class FakeReader:
    fail_at = None
    calls = 0
    def __init__(self, *args):
        pass
    def observe(self, image):
        arr = np.asarray(image.resize((64,64)), dtype=np.float32).mean(axis=2)/255
        return torch.from_numpy(arr).reshape(1,1,64,64).repeat(1,16,1,1)
    def score(self, observation):
        FakeReader.calls += 1
        if FakeReader.calls == FakeReader.fail_at:
            raise RuntimeError("synthetic interruption")
        value = float(observation.mean())
        vals = {"registered": 2 + value, **{f"wrong_{i:02d}":0. for i in range(16)}}
        branches = {k:dict(vals) for k in ("lf","hf","weighted_joint")}
        stat = statistic_from_weighted_scores(vals)
        return {"branches": branches, "statistic":asdict(stat), "margin":stat.value-d.TAU,
                "above_reference_tau":stat.value>d.TAU}


def make_source(root):
    root.mkdir()
    rows = []
    fake = FakeReader()
    pred = d.matrix_for({"kind":"translation","axis":"x","value":2.}, crop(), None).tolist()
    for uid in d.IDS:
        for role in d.ROLES:
            x = np.tile(np.arange(512,dtype=np.uint16)//2,(512,1)).astype(np.uint8)
            image = Image.fromarray(np.stack([x,x.T,x],axis=-1))
            for route in ("clean","crop_direct","oracle_h","predicted_h"):
                matrix = crop()["sampler_h_observed_to_canonical"] if route=="oracle_h" else pred
                rendered = rectify_attacked_rgb(image,matrix) if route.endswith("_h") else image
                path = root / uid / role / (route+".png")
                path.parent.mkdir(parents=True,exist_ok=True)
                rendered.save(path)
                row={"unit_id":uid,"role":role,"route":route,"status":"SCORED",
                     "image":{"path":str(path),"sha256":d.sha(path)}, **fake.score(fake.observe(rendered))}
                if route=="predicted_h":
                    row["h_passed_to_frozen_renderer"]=pred
                rows.append(row)
    d.write(root/"summary.json",{"status":"COMPLETE_DIAGNOSTIC","records":rows})
    d.write(root/"protocol.json",{"source_exact":d.EXACT,"roster":[{"unit_id":u} for u in d.IDS],
        "reference_tau":d.TAU,"crop":crop(),"key_digest":public_key_digest(normalize_detection_key("00"*32))})
    d.write(root/"model_revisions.json",{d.MODEL:"synthetic"})
    d.write(root/"environment.json",{"synthetic":True})
    FakeReader.calls=0


def setup_run(monkeypatch):
    monkeypatch.setenv("CEG_WM_ROOT_KEY","00"*32)
    monkeypatch.setattr(d,"Reader",FakeReader)
    monkeypatch.setattr(d,"version",lambda name:"synthetic")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _:"synthetic")
    monkeypatch.setattr(d.subprocess,"check_output",lambda cmd,**kw: d.EXACT if "rev-parse" in cmd else "")


def test_interrupt_resume_retains_source_and_completed_scores(monkeypatch,tmp_path):
    source,output=tmp_path/"source",tmp_path/"output"
    make_source(source)
    before={str(p.relative_to(source)):d.sha(p) for p in source.rglob("*") if p.is_file()}
    setup_run(monkeypatch)
    FakeReader.fail_at=49  # ALL 32 replays pass, then first perturbation fails.
    with pytest.raises(RuntimeError,match="synthetic interruption"):
        d.run(source,output,REPO)
    assert d.read(output/"replay_gate.json")["status"]=="PASSED"
    control=output/"records"/d.IDS[0]/"positive"/"oracle_replay.json"
    stamp=control.stat().st_mtime_ns
    FakeReader.fail_at=None
    d.run(source,output,REPO)
    result=d.read(output/"summary.json")
    assert result["status"]=="COMPLETE_DIAGNOSTIC"
    assert result["planned_scores"]==352 and len(result["records"])==352
    assert all(c["planned"]==8 for c in result["counts"].values())
    assert control.stat().st_mtime_ns==stamp
    assert before=={str(p.relative_to(source)):d.sha(p) for p in source.rglob("*") if p.is_file()}
    assert any(d.read(p).get("status")=="OPERATIONAL_ERROR" for p in (output/"attempts").rglob("*.json"))
    count=FakeReader.calls
    d.run(source,output,REPO)
    assert FakeReader.calls==count


def test_missing_source_image_stops_before_reader(monkeypatch,tmp_path):
    source=tmp_path/"source"
    make_source(source)
    path=next(source.rglob("crop_direct.png"))
    path.unlink()
    setup_run(monkeypatch)
    with pytest.raises(FileNotFoundError):
        d.run(source,tmp_path/"output",REPO)
    assert FakeReader.calls==0


def test_replay_mismatch_stops_all_perturbations(monkeypatch,tmp_path):
    source=tmp_path/"source"
    make_source(source)
    result=d.read(source/"summary.json")
    row=next(r for r in result["records"] if r["route"]=="oracle_h")
    row["branches"]["hf"]["registered"]+=1
    d.write(source/"summary.json",result)
    setup_run(monkeypatch)
    with pytest.raises(RuntimeError,match="replay mismatch"):
        d.run(source,tmp_path/"output",REPO)
    assert not list((tmp_path/"output/records").rglob("translation*.json"))
    assert d.read(tmp_path/"output/replay_gate.json")["status"]=="FAILED"


def test_notebook_embedded_script_no_generation_and_visible_subprocess_logs():
    nb=nbformat.read(HERE/"CEG_WM_crop_residual_v1.ipynb",as_version=4)
    nbformat.validate(nb)
    texts=[c.source for c in nb.cells if c.cell_type=="code"]
    for text in texts:
        compile(text,"cell","exec")
    class Capture:
        def write_text(self,text,encoding): self.value=text
    capture=Capture()
    exec(next(t for t in texts if t.startswith("# Self-contained")),{"script":capture})
    assert capture.value==(HERE/"run_residual.py").read_text()
    for forbidden in ("_main_pair(","_build_runtime(","run_sd35_plain(","run_content_iss_evaluation_pair(","StableDiffusion3Pipeline.from_pretrained"):
        assert forbidden not in capture.value
    setup=next(t for t in texts if "def run_logged" in t)
    assert "stdout=subprocess.PIPE" in setup and "print(line" in setup
    # Execute only the actual logging helper, with a real harmless child process.
    import ast, tempfile, subprocess, time
    tree=ast.parse(setup)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_logged")
    with tempfile.TemporaryDirectory() as tmp:
        ns={"output":Path(tmp),"child_env":{"HF_TOKEN":"fake_secret","CEG_WM_ROOT_KEY":"fake_key"},"subprocess":subprocess,"time":time}
        exec(compile(ast.Module(body=[function],type_ignores=[]),"helper","exec"),ns)
        ns["run_logged"]([sys.executable,"-c","print('hello fake_secret')"],"log-test")
        saved=next(Path(tmp).rglob("*.log")).read_text()
        assert "hello [REDACTED]" in saved and "fake_secret" not in saved
        with pytest.raises(RuntimeError,match="exit code 2"):
            ns["run_logged"]([sys.executable,"-c","import sys;print('failure detail');sys.exit(2)"],"failure-test")
