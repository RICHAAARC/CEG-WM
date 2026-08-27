"""CPU/fake D2 contracts."""
from __future__ import annotations
import importlib.util, json, os, zipfile
from pathlib import Path
import pytest
M=Path(__file__).parents[2]/"experiments"/"run_geometry_v1_qk_d2_independent_confirmation_operational.py";S=importlib.util.spec_from_file_location("d2",M);assert S and S.loader;R=importlib.util.module_from_spec(S);S.loader.exec_module(R)
from PIL import Image
def public_audit():return {"statistic":-1.,"all_eight_pairs_supported":True,"pair_audit":[],"per_transform_audit":[{"transform_label":t,"two_reference_common_counts":[2,2],"two_reference_medians":[-1.,-1.],"two_reference_equal_weight_median":-1.} for t in R.TRANSFORMS]}
def public_stats():return [{"layer_path":f"transformer_blocks.{i}.attn","block_index":i,"q_stat":-1.,"k_stat":-1.,"q_audit":public_audit(),"k_audit":public_audit(),"eligible":True} for i in range(24)]
def source(root:Path,leak:str|None=None)->Path:
 root.mkdir();d0={"execution_exact":R.D0_EXACT,"run_id":R.D0_RUN_ID,"protocol":R.D0_PROTOCOL,"plan_digest":R.D0_PLAN_DIGEST,"roster_digest":R.D0_ROSTER_DIGEST,"status":R.D0_STATUS,"science_denominator":0};rec={"run_id":R.SOURCE_RUN_ID,"protocol":R.SOURCE_PROTOCOL,"status":R.SOURCE_STATUS,"science_denominator":0,"runner_execution_identity":{"commit":R.SOURCE_RUNNER_EXACT},"selected_layer_paths":list(R.SOURCE_SELECTED),"declared_unit_count":768,"audited_unit_count":768,"artifact_status":"complete","source_d0_artifact_identity":d0,"layer_statistics":public_stats(),"route_audit":{"per_transform":[{"transform_label":t,"finite_stat_count":48,"nonnegative_stat_count":0,"all_layer_nonnegative":False} for t in R.TRANSFORMS],"route_level_transform_instability":False}};man={"run_id":R.SOURCE_RUN_ID,"protocol":R.SOURCE_PROTOCOL,"runner_execution_exact":R.SOURCE_RUNNER_EXACT,"status":R.SOURCE_STATUS,"unit_count":768,"source_d0_artifact_identity":d0.copy()};term={"run_id":R.SOURCE_RUN_ID,"status":R.SOURCE_STATUS,"science_denominator":0,"selected_layer_paths":list(R.SOURCE_SELECTED)}
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
def test_public_direction_statistics_reach_run_d2_with_no_weight_leak(monkeypatch,tmp_path):
 x=source(tmp_path/"s");seen={};rd,wr=os.pipe()
 def stop(**kwargs):seen.update(kwargs);raise RuntimeError()
 monkeypatch.setattr(R,"run_d2",stop);monkeypatch.setattr(R,"_package",lambda *a:pytest.fail("package_reached"))
 try:rc=R._main(["--repo-root",str(tmp_path),"--expected-exact","a"*40,"--source-root",str(x),"--output-root",str(tmp_path/"o"),"--control-fd",str(wr)]);line=os.read(rd,R.MAX_CONTROL_BYTES+1)
 finally:os.close(rd);os.close(wr)
 c=json.loads(line[len(R.FAILURE_PREFIX):]);assert seen["source_identity"]["run_id"]==R.SOURCE_RUN_ID and rc==1 and c["failure_point"]=="run_d2" and c["error_class"]=="runtime_error"
@pytest.mark.parametrize("field,value",(("model_weights","opaque"),("audit_note","model weights"),("audit_note","weight tensor"),("audit_note","raw weights"),("audit_note","raw QK material"),("audit_note","token material"),("audit_note","secret value"),("audit_note","/mnt/private/source")))
def test_sensitive_source_values_fail_before_run_d2(monkeypatch,tmp_path,field,value):
 x=source(tmp_path/"s");receipt=json.loads((x/"receipt.json").read_text());receipt[field]=value;(x/"receipt.json").write_text(json.dumps(receipt),encoding="utf-8");rd,wr=os.pipe()
 monkeypatch.setattr(R,"run_d2",lambda **_:pytest.fail("run_d2_reached"));monkeypatch.setattr(R,"_package",lambda *a:pytest.fail("package_reached"))
 try:rc=R._main(["--repo-root",str(tmp_path),"--expected-exact","a"*40,"--source-root",str(x),"--output-root",str(tmp_path/"o"),"--control-fd",str(wr)]);line=os.read(rd,R.MAX_CONTROL_BYTES+1)
 finally:os.close(rd);os.close(wr)
 c=json.loads(line[len(R.FAILURE_PREFIX):]);assert rc==1 and c["failure_point"]=="source_validation" and c["error_class"]=="validation_error"
def test_d4_actual_pillow_marker_correspondence_rejects_old_mapping():
 image=Image.new("RGB",(512,512));markers=((71,93,(201,17,33)),(211,137,(19,203,47)),(353,271,(41,73,229)),(119,389,(239,181,29)));old=((0.,-1.,512.),(1.,0.,0.))
 for x,y,color in markers:image.putpixel((x,y),color)
 attacked,h=R._attack(image,"d4");assert h==[[0.,1.,0.],[-1.,0.,512.],[0.,0.,1.]] and R.build_fixed_plan()["d4_transform"]=="rotate_90"
 for x,y,color in markers:
  found=next((xx,yy) for yy in range(512) for xx in range(512) if attacked.getpixel((xx,yy))==color);sx,sy=x+.5,y+.5;dx=h[0][0]*sx+h[0][1]*sy+h[0][2];dy=h[1][0]*sx+h[1][1]*sy+h[1][2];assert (dx,dy)==(found[0]+.5,found[1]+.5) and found==(y,511-x) and found!=(int(old[0][0]*sx+old[0][1]*sy+old[0][2]-.5),int(old[1][0]*sx+old[1][1]*sy+old[1][2]-.5))
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
