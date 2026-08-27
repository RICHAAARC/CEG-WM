"""CPU/fake D2 contracts."""
from __future__ import annotations
import importlib.util, json, os, zipfile
from pathlib import Path
import pytest
M=Path(__file__).parents[2]/"experiments"/"run_geometry_v1_qk_d2_independent_confirmation_operational.py";S=importlib.util.spec_from_file_location("d2",M);assert S and S.loader;R=importlib.util.module_from_spec(S);S.loader.exec_module(R)
def source(root:Path,leak:str|None=None)->Path:
 root.mkdir();d0={"execution_exact":R.D0_EXACT,"run_id":R.D0_RUN_ID,"protocol":R.D0_PROTOCOL,"plan_digest":R.D0_PLAN_DIGEST,"roster_digest":R.D0_ROSTER_DIGEST,"status":R.D0_STATUS,"science_denominator":0};rec={"run_id":R.SOURCE_RUN_ID,"protocol":R.SOURCE_PROTOCOL,"status":R.SOURCE_STATUS,"science_denominator":0,"runner_execution_identity":{"commit":R.SOURCE_RUNNER_EXACT},"selected_layer_paths":list(R.SOURCE_SELECTED),"declared_unit_count":768,"audited_unit_count":768,"artifact_status":"complete","source_d0_artifact_identity":d0};man={"run_id":R.SOURCE_RUN_ID,"protocol":R.SOURCE_PROTOCOL,"runner_execution_exact":R.SOURCE_RUNNER_EXACT,"status":R.SOURCE_STATUS,"unit_count":768,"source_d0_artifact_identity":d0.copy()};term={"run_id":R.SOURCE_RUN_ID,"status":R.SOURCE_STATUS,"science_denominator":0,"selected_layer_paths":list(R.SOURCE_SELECTED)}
 if leak:rec["audit_note"]=leak
 for n,v in (("receipt.json",rec),("manifest.json",man),("terminal.json",term)):(root/n).write_text(json.dumps(v),encoding="utf-8")
 return root
CROSS_BINDINGS=tuple(("receipt.json",(field,)) for field in ("run_id","protocol","status","science_denominator","declared_unit_count","audited_unit_count","artifact_status"))+(("receipt.json",("runner_execution_identity","commit")),("receipt.json",("selected_layer_paths",)))+tuple(("receipt.json",("source_d0_artifact_identity",field)) for field in ("execution_exact","run_id","protocol","plan_digest","roster_digest","status","science_denominator"))+tuple(("manifest.json",(field,)) for field in ("run_id","protocol","runner_execution_exact","status","unit_count"))+tuple(("manifest.json",("source_d0_artifact_identity",field)) for field in ("execution_exact","run_id","protocol","plan_digest","roster_digest","status","science_denominator"))+tuple(("terminal.json",(field,)) for field in ("run_id","status","science_denominator","selected_layer_paths"))
def _wrong(v):return 1 if isinstance(v,int) else list(reversed(v)) if isinstance(v,list) else "wrong"
@pytest.mark.parametrize("sidecar,path",CROSS_BINDINGS,ids=lambda v:"_".join(v) if isinstance(v,tuple) else v)
def test_cross_sidecar_mismatch_fails_main_before_runtime(monkeypatch,tmp_path,sidecar,path):
 x=source(tmp_path/"s");target=json.loads((x/sidecar).read_text());node=target
 for key in path[:-1]:node=node[key]
 node[path[-1]]=_wrong(node[path[-1]]);(x/sidecar).write_text(json.dumps(target),encoding="utf-8")
 monkeypatch.setattr(R,"run_d2",lambda **_:pytest.fail("loader_or_stats_reached"));monkeypatch.setattr(R,"_package",lambda *a:pytest.fail("package_reached"));rd,wr=os.pipe()
 try:rc=R._main(["--repo-root",str(tmp_path),"--expected-exact","a"*40,"--source-root",str(x),"--output-root",str(tmp_path/"o"),"--control-fd",str(wr)]);line=os.read(rd,R.MAX_CONTROL_BYTES+1)
 finally:os.close(rd);os.close(wr)
 c=json.loads(line[len(R.FAILURE_PREFIX):]);assert rc==1 and c["failure_point"]=="source_validation" and c["error_class"]=="validation_error"
def unit(pid,path,kind,control,good=True):
 t=pid.rsplit("-",1)[1];a,b=(1.,3.) if good else (4.,2.);r=[a,None,a+1] if control=="matched_h" else [b,None,b+1]
 return {"pair_id":pid,"transform_label":t,"control_label":control,"descriptor_kind":kind,"layer_path":path,"reference_grid":[32,32],"attacked_grid":[32,32],"input_identity":{"sha256":"a"*64},"h_identity":{"sha256":"b"*64},"status":"calculated","failure_reason":None,"candidate_correspondences":[],"true_match_ranks":r,"coverage":1.,"ambiguity_gaps":[1.],"fit_residual":.1,"recovery_error":.2}
def units(good=True):return [unit(p,l,k,c,good) for p in R.PAIRS for l in R.ATTENTION_LAYER_PATHS for k in R.KINDS for c in R.CONTROLS]
def test_source_validation_and_leak_precede_runtime(monkeypatch,tmp_path):
 x=source(tmp_path/"s",leak="Bearer opaque-value");monkeypatch.setattr(R,"_exact",lambda e,r:e);monkeypatch.setattr(R,"_stats",lambda u:pytest.fail("stats"));monkeypatch.setattr(R,"_package",lambda *a:pytest.fail("package"));rd,wr=os.pipe()
 try:rc=R._main(["--repo-root",str(tmp_path),"--expected-exact","a"*40,"--source-root",str(x),"--output-root",str(tmp_path/"o"),"--control-fd",str(wr)]);line=os.read(rd,R.MAX_CONTROL_BYTES+1)
 finally:os.close(rd);os.close(wr)
 c=json.loads(line[len(R.FAILURE_PREFIX):]);assert rc==1 and c["failure_point"]=="source_validation" and c["error_class"]=="validation_error"
def test_64_roster_grid_and_independent_paired_direction_math():
 assert len(units())==64 and R._grid_h([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],(32,32),(16,64)).shape==(3,3)
 ok,stats,audit=R._stats(units());assert ok and len(stats)==4 and all(s["statistic"]==-2. for s in stats)
 ok,stats,_=R._stats(units(False));assert not ok and all(s["statistic"]==2. for s in stats)
 assert audit["route_level_transform_instability"] is False
def test_two_create_only_32_unit_shards(monkeypatch,tmp_path):
 monkeypatch.setattr(R,"_exact",lambda e,r:e);monkeypatch.setattr(R,"_validate_source",lambda _: {"source":"ok"});monkeypatch.setattr(R,"_topology",lambda _:None);monkeypatch.setattr(R,"_spec",lambda _:object())
 # Packaging contract is checked directly with retained public records.
 summary={"run_id":"geometry-v1-qk-d2-aaaaaaaaaaaa","runner_execution_identity":{"commit":"a"*40},"source_direction_artifact_identity":{"source":"ok"},"status":"D2_UNRESOLVED","artifact_status":"unavailable"};out=tmp_path/"out";R._package(out,summary,units());assert [p.name for p in sorted((out/"layers").glob("*.zip"))]==["00.zip","01.zip"]
 with pytest.raises(FileExistsError):R._package(out,summary,units())
