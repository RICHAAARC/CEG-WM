"""Colab control-flow and engineering-profile tests: no network, models or GPU."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cegwm.self_reference.colab_config import GENERATION, roster
from cegwm.self_reference.colab_runner import Controller, initialize, run_logged
from cegwm.self_reference.colab_storage import atomic_json, completed_asset, read_json, write_asset, quarantine
from cegwm.self_reference.colab_worker import checkpoint_plan, context
from cegwm.self_reference.data import Cache
from cegwm.self_reference.engine import train, load_checkpoint, stage_b_report
from cegwm.self_reference.execution import ExecutionProfile, configure_torch

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[2]
_fixture_spec=importlib.util.spec_from_file_location("sr_v1_test_fixtures",Path(__file__).with_name("test_self_reference_v1.py"))
_fixture_module=importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
fixture,assert_nested_equal=_fixture_module.fixture,_fixture_module.assert_nested_equal


def changed_cache(cache, cfg):
    raw = cache.raw.copy()
    raw["config_hash"] = cfg.digest
    atomic_json(cache.path, raw)
    return Cache(cache.path, cfg)


def test_microbatch4_equivalence_and_same_profile_exact_resume(fixture):
    cfg, adapter, cache, root = fixture
    cfg = replace(cfg, batch_size=4, updates=2, checkpoint_every=2)
    cache = changed_cache(cache,cfg)
    profile = ExecutionProfile(recovery_every=1)
    batch = train(cache,cfg,adapter,"C",root/"batch")
    micro = train(cache,cfg,adapter,"C",root/"micro",execution=profile)
    a, b = [load_checkpoint(p,cache,cfg,"C","cpu") for p in (batch,micro)]
    for module in ("writer", "reader"):
        for name in a[module]:
            assert torch.allclose(a[module][name], b[module][name], atol=3e-6, rtol=3e-5), name
    # Adam first moments encode the accumulated gradients; compare them too.
    for index, state in a["optimizer"]["state"].items():
        assert torch.allclose(state["exp_avg"], b["optimizer"]["state"][index]["exp_avg"], atol=2e-7, rtol=3e-4)
    first = train(cache,cfg,adapter,"C",root/"resume",execution=profile,stop_after=1)
    resumed = train(cache,cfg,adapter,"C",root/"resume",execution=profile,resume=first)
    c = load_checkpoint(resumed,cache,cfg,"C","cpu")
    for key in ("writer","reader","optimizer","rng","loss","next_input_signature","last_input_signature"):
        assert_nested_equal(b[key], c[key])
    with pytest.raises(ValueError, match="profile mismatch"):
        train(cache,cfg,adapter,"C",root/"resume",execution=replace(profile,checkpoint_vae=False),resume=first)


def test_corrupt_latest_and_missing_candidate_replay_without_collision(fixture):
    cfg,adapter,cache,root = fixture
    profile = ExecutionProfile(recovery_every=1)
    directory = root/"C"
    last = train(cache,cfg,adapter,"C",directory,execution=profile)
    expected = load_checkpoint(last,cache,cfg,"C","cpu")
    assert checkpoint_plan(directory,cache,cfg,"C",profile)["status"] == "complete"
    last.write_bytes(b"partial corrupt write")
    plan = checkpoint_plan(directory,cache,cfg,"C",profile)
    assert plan["update"] == 3 and len(plan["invalid"]) == 1 and not last.exists()
    rebuilt = train(cache,cfg,adapter,"C",directory,execution=profile,resume=plan["resume"])
    assert_nested_equal(expected["writer"], load_checkpoint(rebuilt,cache,cfg,"C","cpu")["writer"])
    # A final checkpoint alone cannot hide a lost intermediate selection candidate.
    missing = directory/"update_000002.pt"
    quarantine([missing,missing.with_suffix(".json")],directory,"test lost candidate")
    plan = checkpoint_plan(directory,cache,cfg,"C",profile)
    assert plan["update"] == 1 and not last.exists()
    rebuilt = train(cache,cfg,adapter,"C",directory,execution=profile,resume=plan["resume"])
    assert checkpoint_plan(directory,cache,cfg,"C",profile)["status"] == "complete"
    assert_nested_equal(expected["optimizer"], load_checkpoint(rebuilt,cache,cfg,"C","cpu")["optimizer"])
    assert list((directory/"quarantine").glob("*/reason.json"))
    with pytest.raises(ValueError,match="identity"):
        checkpoint_plan(directory,cache,cfg,"C",replace(profile,microbatch=2))


def test_two_images_in_one_family_keep_image_denominator():
    cfg = replace(__import__("cegwm.self_reference.config",fromlist=["Config"]).Config(),bootstrap_samples=20)
    records = [{"source": f"image-{i}", "source_id": f"image-{i}", "source_group":"family", "shift":list(s),
                "top1":i==0,"status":"ok","bit_accuracy":float(i==0),"exact":i==0}
               for i in range(2) for s in ((-2,0),(2,0),(0,-2),(0,2))]
    failed=[{**r,"top1":False,"bit_accuracy":0.,"exact":False} for r in records]
    report = stage_b_report({"B0":failed,"B1":failed,"C":records},cfg)
    assert report["source_denominator"] == 2 and report["cluster_denominator"] == 1
    assert report["arms"]["C"]["all_four_source_rate"] == .5
    assert report["arms"]["C"]["source_bootstrap95"] == [.5,.5]
    for arm in ("B0","B1"):
        assert report["paired"][arm]["rescue_sources"] == 1
        assert report["paired"][arm]["harm_sources"] == 0 and report["paired"][arm]["gain"] == .5


class FakeStages:
    def __init__(self,root,pass_a=False,fail=None):
        self.root,self.pass_a,self.fail = root,pass_a,fail
        self.calls=[]
        self.pending_latent,self.pending_clean=[0,9],[0,9,20]
        self.trained=set()
        self.actual_train=[]

    def __call__(self,task,*args):
        self.calls.append((task,args))
        if task == self.fail:
            raise KeyboardInterrupt("synthetic disconnection")
        if task == "cache_status":
            atomic_json(self.root/"cache_status.json", {"missing_latent":self.pending_latent,"missing_clean":self.pending_clean,"missing_vae":bool(self.pending_latent)})
        elif task == "generate":
            self.pending_latent=[]
        elif task == "clean":
            self.pending_clean=[]
        elif task == "train":
            arm=args[1]
            if arm not in self.trained:
                self.actual_train.append(arm)
                self.trained.add(arm)
        elif task in ("dev","test"):
            atomic_json(self.root/(task+".json"), {"A_gate":{"pass":self.pass_a,"status":"continue_pilot" if self.pass_a else "stop_C_v1"},
                "A":{},"quality":{"status":"not_evaluable"},"research_status":"not_evaluable",
                "B":{"status":"measured" if self.pass_a else "skipped_by_dev_A_gate","source_denominator":64,"cluster_denominator":8}})


def fake_controller(root,backend):
    atomic_json(root/"run.json", {"identity":{"code_commit":"fixture"},"python":sys.executable,"repo":str(REPO)})
    return Controller(root,invoke=backend)


def test_runner_full_rerun_partial_cache_and_A_fail_no_test(tmp_path):
    backend=FakeStages(tmp_path)
    for repetition in range(2):
        control=fake_controller(tmp_path,backend)
        for stage in ("cache","train","evaluate"):
            control.execute(stage)
        summary=control.summary()
        assert summary["status"] == "stopped_by_dev_A" and summary["scientific_status"] == "not_evaluable"
    assert sum(task=="generate" for task,_ in backend.calls) == 1
    assert sum(task=="clean" for task,_ in backend.calls) == 1
    assert sum(task=="prepare" for task,_ in backend.calls) == 1
    assert sum(task=="prepare_quality" for task,_ in backend.calls) == 1
    assert not any(task in ("shift","test") for task,_ in backend.calls)
    assert backend.actual_train == ["B0","B1","C"]
    assert [args for task,args in backend.calls if task=="generate"] == [("--count",640)]


def test_runner_failure_safe_skips_and_success_summary(tmp_path):
    backend=FakeStages(tmp_path,fail="generate")
    control=fake_controller(tmp_path,backend)
    control.execute("cache")
    before=len(backend.calls)
    control.execute("train"); control.execute("evaluate")
    assert len(backend.calls)==before and control.summary()["status"] == "incomplete"
    backend.fail=None; backend.pass_a=True
    control=fake_controller(tmp_path,backend)
    for stage in ("cache","train","evaluate"):
        control.execute(stage)
    summary=control.summary()
    assert summary["status"] == "execution_completed" and summary["scientific_status"] == "not_evaluable"
    assert summary["test"]["B"]["cluster_denominator"] == 8


def test_asset_completion_partial_hash_and_identity(tmp_path):
    directory=tmp_path/"latent"; directory.mkdir()
    (directory/"x.npy").write_bytes(b"orphan")
    assert completed_asset(directory,"x.npy","identity") is None
    assert list((directory/"quarantine").glob("*/x.npy"))
    write_asset(directory,"x.npy","identity",lambda path:path.write_bytes(b"complete"))
    assert completed_asset(directory,"x.npy","identity")["sha256"]
    with pytest.raises(ValueError,match="identity conflict"):
        completed_asset(directory,"x.npy","other")
    (directory/"x.npy").write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash mismatch"):
        completed_asset(directory,"x.npy","identity")


def test_actual_cache_status_reuses_pairs_without_manifest_or_model_load(tmp_path):
    from cegwm.self_reference.colab_worker import cache_status,source_identity,clean_identity
    data,_=roster(REPO)
    data={"sources":data["sources"][:3]}
    run={"identity":{"code":"fixture"}}
    atomic_json(tmp_path/"runtime.json",{"identity":{"torch":"fixture"}})
    atomic_json(tmp_path/"model.json",{"vae_files":{}})
    for index,row in enumerate(data["sources"]):
        if index==2:
            continue
        latent=write_asset(tmp_path/"cache/latent",row["source_id"]+".npy",source_identity(run,row,tmp_path),lambda p:p.write_bytes(b"fixture"))
        if index==0:
            write_asset(tmp_path/"cache/clean",row["source_id"]+".png",clean_identity(run,row,tmp_path,latent),lambda p:p.write_bytes(b"fixture-clean"))
    cache_status(tmp_path,run,data)
    status=read_json(tmp_path/"cache_status.json")
    assert status=={"missing_latent":[2],"missing_clean":[1,2],"missing_vae":False}
    assert not (tmp_path/"cache/manifest.json").exists()


def test_token_never_reaches_tee_or_status(tmp_path,monkeypatch,capsys):
    token="hf_TESTSECRETabcdefghijk"
    monkeypatch.setenv("HF_TOKEN",token)
    run_logged([sys.executable,"-c","import os; print(os.environ['HF_TOKEN'])"],tmp_path/"log.txt")
    assert token not in (tmp_path/"log.txt").read_text()
    assert token not in capsys.readouterr().out
    assert "[REDACTED]" in (tmp_path/"log.txt").read_text()


def test_numerical_settings_common_and_no_real_cuda(monkeypatch):
    calls=[]
    fake=SimpleNamespace(use_deterministic_algorithms=lambda x:calls.append(("det",x)),
                         set_float32_matmul_precision=lambda x:calls.append(("precision",x)),
                         backends=SimpleNamespace(cudnn=SimpleNamespace(),cuda=SimpleNamespace(matmul=SimpleNamespace())))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    configure_torch("cuda",fake)
    assert calls==[("det",True),("precision","highest")]
    assert not fake.backends.cuda.matmul.allow_tf32 and not fake.backends.cudnn.benchmark
    # All actual-model paths explicitly call this same helper before load_adapter/from_pretrained.
    import inspect
    from cegwm.self_reference import colab_worker,engine
    for fn in (colab_worker.generate,colab_worker.clean,colab_worker.research_stage,engine.train):
        assert "configure_torch(" in inspect.getsource(fn)


def test_train_and_eval_stage_settings_precede_adapter_load(fixture,monkeypatch):
    from cegwm.self_reference import colab_worker,adapter as adapters,engine,data
    cfg,fake,cache,root=fixture
    calls=[]
    monkeypatch.setattr(data,"Cache",lambda *args:cache)
    monkeypatch.setattr(colab_worker,"configure_torch",lambda device:calls.append("configure"))
    monkeypatch.setattr(adapters,"load_adapter",lambda *args:(calls.append("load") or fake))
    monkeypatch.setattr(colab_worker,"checkpoint_plan",lambda *args:{"status":"start","resume":None})
    monkeypatch.setattr(engine,"train",lambda *args,**kwargs:calls.append("train"))
    colab_worker.research_stage(root,"train","C")
    assert calls==["configure","load","train"]
    calls.clear()
    monkeypatch.setattr(engine,"validate_selection",lambda *args:{})
    monkeypatch.setattr(engine,"evaluate",lambda *args,**kwargs:(calls.append("evaluate") or {"research_status":"not_evaluable","A_gate":{}}))
    colab_worker.research_stage(root,"dev")
    assert calls==["configure","load","evaluate"]


def test_clean_settings_precede_adapter_load_with_fake_tensor(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from cegwm.self_reference import colab_worker,adapter as adapters
    row=roster(REPO)[0]["sources"][0]
    run={"identity":{"test":"only"}}
    atomic_json(tmp_path/"runtime.json",{"identity":{}})
    write_asset(tmp_path/"cache/latent",row["source_id"]+".npy",colab_worker.source_identity(run,row,tmp_path),lambda p:p.write_bytes(b"fake"))
    calls=[]
    class FakeTensor:
        def __getitem__(self,index): return self
        def to(self,*args): return self
    tensor=FakeTensor()
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(from_numpy=lambda array:tensor,no_grad=nullcontext))
    monkeypatch.setattr(np,"load",lambda *args,**kwargs:None)
    monkeypatch.setattr(colab_worker,"configure_torch",lambda device:calls.append("configure"))
    monkeypatch.setattr(adapters,"load_adapter",lambda *args:(calls.append("load") or SimpleNamespace(decode=lambda z:z)))
    monkeypatch.setattr(adapters,"rgb8",lambda z:z)
    monkeypatch.setattr(adapters,"pil_image",lambda z:SimpleNamespace(save=lambda p,**kwargs:Path(p).write_bytes(b"fake-png")))
    colab_worker.clean(tmp_path,run,[row])
    assert calls==["configure","load"]


def test_frozen_roster_identity_and_notebook_compile(tmp_path,monkeypatch):
    data,checksum=roster(REPO)
    from dataclasses import asdict
    from cegwm.self_reference.colab_config import json_hash,DEPENDENCIES
    frozen=read_json(REPO/"configs/self_reference_v1_colab.json")
    assert frozen=={"generation":GENERATION,"generation_hash":json_hash(GENERATION),"execution":asdict(ExecutionProfile()),
                   "execution_hash":ExecutionProfile().digest,"dependencies":DEPENDENCIES}
    assert len(data["sources"])==640 and len({r["source_group"] for r in data["sources"]})==80
    assert all(len(data["keys"][s])==16 for s in data["keys"])
    import subprocess
    monkeypatch.setattr(subprocess,"check_output",lambda *a,**k:"a"*40+"\n")
    with pytest.raises(ValueError,match="immutable code commit"):
        initialize(REPO,tmp_path,sys.executable,"b"*40)
    atomic_json(tmp_path/"run.json", {"repo":str(REPO),"identity":{}})
    with pytest.raises(ValueError,match="identity mismatch"):
        context(tmp_path)
    spec=importlib.util.spec_from_file_location("notebook_builder",REPO/"scripts/build_self_reference_notebook.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    notebook=module.build("a"*40)
    import nbformat
    nbformat.validate(notebook)
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type=="code":
            compile(cell.source,f"cell-{i}","exec")
            assert cell.execution_count is None and not cell.outputs
    assert "--no-deps" in notebook.cells[4].source
    assert "--system-site-packages" in notebook.cells[4].source
    assert not any("import torch" in c.source for c in notebook.cells if c.cell_type=="code" and "smoke =" not in c.source)


def test_notebook_stage_cells_run_in_order_and_safe_skip(tmp_path):
    spec=importlib.util.spec_from_file_location("notebook_builder",REPO/"scripts/build_self_reference_notebook.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    notebook=module.build("a"*40)
    backend=FakeStages(tmp_path,fail="generate")
    controller=fake_controller(tmp_path,backend)
    # Execute the four actual orchestration/summary cells in order with injected services.
    scope={"BOOT_OK":True,"controller":controller,"json":json,"SETUP_ERROR":None}
    cells=[c for c in notebook.cells if c.cell_type=="code"]
    for cell in cells[2:]:
        exec(compile(cell.source,"notebook-stage","exec"),scope)
    assert not any(task in ("train","select","dev","test") for task,_ in backend.calls)
    assert read_json(tmp_path/"summary.json")["status"] == "incomplete"


def test_notebook_without_ensurepip_repairs_partial_venv_and_targets_local_wheel(tmp_path,monkeypatch):
    """Real CPU subprocesses; no index, downloads, build tools, weights or GPU."""
    import base64
    import hashlib
    import os
    import re
    import subprocess
    import uuid
    import zipfile
    monkeypatch.setenv("PIP_NO_INDEX","1")
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK","1")
    monkeypatch.setenv("PIP_CONFIG_FILE",os.devnull)
    spec=importlib.util.spec_from_file_location("bootstrap_notebook_builder",REPO/"scripts/build_self_reference_notebook.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    notebook=module.build("984955b1049af48f2f93bd32d91c0bd99c37736e")
    cells=[cell for cell in notebook.cells if cell.cell_type=="code"]
    commands=[]
    def logged(command,log):
        commands.append(list(map(str,command)))
        run_logged(command,log)
    scope={"BOOT_OK":False,"Path":Path,"sys":sys,"json":json,"subprocess":subprocess,"re":re,"run_logged":logged}
    # Actual delivered helper code; the rest of setup is safely skipped.
    exec(compile(cells[1].source,"notebook-bootstrap","exec"),scope)
    name="cegwm_bootstrap_probe_"+uuid.uuid4().hex
    distribution=name+"-0.0.1.dist-info"
    files={name+"/__init__.py":b"VALUE = 'target-only'\n",
           distribution+"/METADATA":f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.1\n".encode(),
           distribution+"/WHEEL":b"Wheel-Version: 1.0\nGenerator: local-cpu-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    records=[]
    for path,payload in files.items():
        digest=base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        records.append(f"{path},sha256={digest},{len(payload)}")
    files[distribution+"/RECORD"]=("\n".join(records)+f"\n{distribution}/RECORD,,\n").encode()
    wheel=tmp_path/(name+"-0.0.1-py3-none-any.whl")
    with zipfile.ZipFile(wheel,"w") as archive:
        for path,payload in files.items(): archive.writestr(path,payload)
    venv=tmp_path/"interrupted-venv"
    # Reproduce the important failed-bootstrap state: python exists, local pip does not.
    subprocess.run([sys.executable,"-m","venv","--system-site-packages","--without-pip",str(venv)],check=True)
    target=venv/"bin/python"
    assert target.exists()
    sentinel=venv/"keep-existing-state.txt"; sentinel.write_text("not cleared")
    package_probe="import importlib.metadata as m,json,sys; print(json.dumps({'prefix':sys.prefix,'packages':sorted((d.metadata.get('Name',''),d.version) for d in m.distributions())}))"
    host_before=subprocess.check_output([sys.executable,"-c",package_probe],text=True)
    environment_before=os.environ.copy()
    target_probe=f"import sys,json,sysconfig,{name}; print(json.dumps({{'prefix':sys.prefix,'site':sysconfig.get_path('purelib'),'file':{name}.__file__,'value':{name}.VALUE}}))"
    host_lookup=f"import importlib.util; print(importlib.util.find_spec({name!r}))"
    assert subprocess.check_output([sys.executable,"-c",host_lookup],text=True).strip()=="None"
    for repetition in range(2):
        pip_command,host_identity=scope["prepare_target_venv"](venv,tmp_path/"bootstrap.log")
        assert pip_command==[sys.executable,"-m","pip","--python",str(target)]
        logged([*pip_command,"install","--no-user","--no-index","--no-deps","--force-reinstall",str(wheel)],tmp_path/"bootstrap.log")
        actual=json.loads(subprocess.check_output([str(target),"-c",target_probe],text=True))
        assert Path(actual["prefix"]).resolve()==venv.resolve()
        assert Path(actual["site"]).resolve().is_relative_to(venv.resolve())
        assert Path(actual["file"]).resolve().is_relative_to(Path(actual["site"]).resolve())
        assert actual["value"]=="target-only" and sentinel.read_text()=="not cleared"
        assert not (Path(actual["site"])/"pip").exists()
        assert scope["python_environment"](sys.executable)==host_identity
        assert subprocess.check_output([sys.executable,"-c",host_lookup],text=True).strip()=="None"
    assert subprocess.check_output([sys.executable,"-c",package_probe],text=True)==host_before
    assert os.environ==environment_before
    assert sum("venv" in command and "--without-pip" in command for command in commands)==2
    assert all("--python" in command for command in commands if "install" in command)
    assert "Successfully installed" in (tmp_path/"bootstrap.log").read_text()
