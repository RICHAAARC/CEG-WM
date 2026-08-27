"""Combined C1-then-fixed-N96 Content V10 paired allocator producer."""
from __future__ import annotations
import argparse, hashlib, math, os, shutil
from pathlib import Path
from typing import Any
from experiments import run_content_v10_calibration as c1
from cegwm.protocol.content_chain_v10 import TEXTURE_N96_MANIFEST_DIGEST, load_content_v10_contract, load_content_v10_n96_paired_contract
from cegwm.shared.keys import public_key_digest

PREFIX="CEGWM_CONTENT_V10_N96_PAIRED_SUMMARY"; RESULT="content_v10_n96_paired_result.json"

def _score(image: Any, key: bytes, assets: Any) -> tuple[float,float]:
 from cegwm.method.content_whitening_v4 import score_content_v4_lf_image
 from cegwm.method.hf import score_hf_image
 from cegwm.runtime.observation import require_ordinary_rgb_image
 image=require_ordinary_rgb_image(image); pair=(float(score_content_v4_lf_image(image,key,assets.lf_public_assets)),float(score_hf_image(image,key,assets.hf_public_assets)))
 if not all(math.isfinite(x) and -1<=x<=1 for x in pair): raise ValueError("N96 branch score differs")
 return pair

def _margins(values: list[tuple[float,float]]) -> dict[str,float]:
 return {"lf_a":values[0][0]-max(x[0] for x in values[1:]),"hf_a":values[0][1]-max(x[1] for x in values[1:])}

def _spearman(texture: list[float], response: list[float]) -> float:
 if len(texture)!=96 or len(response)!=96 or not all(math.isfinite(x) for x in texture+response): raise ValueError("N96 Spearman fixed denominator differs")
 def ranks(values: list[float]) -> list[float]:
  order=sorted(range(96),key=values.__getitem__); out=[0.0]*96; start=0
  while start<96:
   end=start+1
   while end<96 and values[order[end]]==values[order[start]]: end+=1
   rank=(start+1+end)/2
   for index in order[start:end]: out[index]=rank
   start=end
  return out
 x,y=ranks(texture),ranks(response); mx=math.fsum(x)/96; my=math.fsum(y)/96
 dx=[v-mx for v in x]; dy=[v-my for v in y]; den=math.sqrt(math.fsum(v*v for v in dx)*math.fsum(v*v for v in dy))
 if not math.isfinite(den) or den<=0: raise ValueError("N96 Spearman zero rank variance")
 value=math.fsum(a*b for a,b in zip(dx,dy))/den
 if not math.isfinite(value): raise ValueError("N96 Spearman nonfinite")
 return value

def _statistics(rows: list[dict[str,Any]]) -> dict[str,dict[str,Any]]:
 if len(rows)!=96: raise ValueError("N96 statistic denominator differs")
 texture=[float(row["texture_scalar"]) for row in rows]; output={}
 for branch in ("lf","hf"):
  for margin in ("a","b"):
   key=f"{branch}_{margin}"; v9=[float(row["v9"][key]) for row in rows]; v10=[float(row["v10"][key]) for row in rows]; delta=[b-a for a,b in zip(v9,v10)]
   if not all(math.isfinite(value) for value in v9+v10+delta): raise ValueError("N96 statistic nonfinite")
   output[f"v9_{key}"]={"observed_n":96,"texture_spearman":_spearman(texture,v9)}
   output[f"v10_{key}"]={"observed_n":96,"texture_spearman":_spearman(texture,v10)}
   output[f"delta_{key}"]={"observed_n":96,"mean":math.fsum(delta)/96,"positive_count":sum(value>0 for value in delta),"ties_count":sum(value==0 for value in delta),"texture_spearman":_spearman(texture,delta)}
 return output

def _publish_result(path: Path, payload: bytes) -> str:
 sidecar=path.with_name(path.name+".sha256")
 if path.exists() or sidecar.exists(): raise FileExistsError("N96 result destination is create-only")
 digest=hashlib.sha256(payload).hexdigest(); made=False
 try:
  with path.open("xb") as handle: made=True; handle.write(payload)
  with sidecar.open("xb") as handle: handle.write(f"{digest}  {path.name}\n".encode("ascii"))
 except BaseException:
  sidecar.unlink(missing_ok=True)
  if made: path.unlink(missing_ok=True)
  raise
 return digest

def execute(args: argparse.Namespace) -> int:
 secret=os.environ.pop(c1.KEY_ENV,""); token=os.environ.pop(c1.TOKEN_ENV,""); local_run=None; c1_published=False; phase="c1"; failed=[]; c1_asset=c1_sidecar=None; digest=None
 try:
  root=Path(args.repo_root).resolve(); sink=Path(args.artifact_sink).resolve(); local=Path(args.local_work_root).resolve()
  exact=c1._git_exact(root,args.expected_exact); c1_contract=load_content_v10_contract(root); n96_contract=load_content_v10_n96_paired_contract(root); units=c1._units(root)
  if not secret or not token.strip(): raise RuntimeError("C1 and N96 child-only secrets are required")
  key=c1.derive_calibration_key(secret); public=public_key_digest(key); run_id=hashlib.sha256((c1_contract.digest+public).encode("ascii")).hexdigest()[:24]
  local_run=local/run_id; local_run.mkdir(parents=True,exist_ok=False)
  pipeline,assets=c1._load_pipeline_and_assets(token)
  from cegwm.method.content_v10_texture_neutral import load_independent_calibration_asset, weighted_joint_v10
  payload=c1.produce_calibration_payload(pipeline,assets,units,key,exact=exact,protocol_digest=c1_contract.digest,public_digest=public)
  c1_asset,c1_sidecar=c1._paths(sink,run_id); staged,stage_side,digest=c1._stage_and_validate(local_run,payload,exact,c1_contract.digest,public,load_independent_calibration_asset)
  c1._publish_staged(staged,stage_side,c1_asset,c1_sidecar); c1_published=True; v10_asset=load_independent_calibration_asset(c1_asset,c1_sidecar,producer_execution_exact=exact,protocol_digest=c1_contract.digest,calibration_public_key_digest=public)
  from cegwm.method.content_weighted_joint_v9 import load_calibration_asset, weighted_joint_score
  v9_path=root/"configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json"; v9_side=v9_path.with_name(v9_path.name+".sha256")
  if hashlib.sha256(v9_path.read_bytes()).hexdigest()!="63c17e8200a92383b061541fc234dfef36e4b7356954c160ce5f048f820cde96": raise ValueError("frozen V9 asset differs")
  v9_asset=load_calibration_asset(v9_path,v9_side); rows=[]
  import json
  phase="n96"
  for ordinal,line in enumerate((root/"configs/content_chain/content_texture_n96_evaluation_v1.jsonl").read_text(encoding="utf-8").splitlines()):
   unit=json.loads(line)
   try:
    from cegwm.runtime.content_v10_texture_neutral_sd35 import ContentV10PairedArmFailure, derive_v10_calibration_wrong_keys, run_content_v10_paired_evaluation
    output=run_content_v10_paired_evaluation(pipeline,unit["prompt"],key,assets,height=unit["height"],width=unit["width"],seed=unit["seed"])
    keys=(key,*derive_v10_calibration_wrong_keys(key)); plain=[_score(output.primary_null,k,assets) for k in keys]; v9=[_score(output.v9_image,k,assets) for k in keys]; v10=[_score(output.v10_image,k,assets) for k in keys]
    row={"unit_id":unit["unit_id"],"block_id":unit["block_id"],"plain_rgb_sha256":output.plain_rgb_sha256,"texture_scalar":output.texture_scalar,"texture_raw_digest":output.texture_raw_digest,"v9":_margins(v9),"v10":_margins(v10)}
    for name,scores,asset,fn in (("v9",v9,v9_asset,weighted_joint_score),("v10",v10,v10_asset,weighted_joint_v10)):
     row[name]["lf_b"]=scores[0][0]-plain[0][0]; row[name]["hf_b"]=scores[0][1]-plain[0][1]; row[name]["joint_a"]=fn(*scores[0],asset)-max(fn(*x,asset) for x in scores[1:]); row[name]["joint_b"]=fn(*scores[0],asset)-fn(*plain[0],asset); row[name]["joint_interpretation"]="whole_system_descriptive_only"
    rows.append(row)
   except BaseException as error:
    arm=error.arm if 'ContentV10PairedArmFailure' in locals() and isinstance(error,ContentV10PairedArmFailure) else "unknown"; failed.append({"phase":"n96","ordinal":ordinal,"unit_id":unit["unit_id"],"arm":arm,"error_class":type(error).__name__}); raise
  if len(rows)!=96: raise ValueError("fixed N96 denominator differs")
  statistics=_statistics(rows); result=sink/run_id/RESULT; result_payload=c1.stable_json_bytes({"exact":exact,"manifest_digest":TEXTURE_N96_MANIFEST_DIGEST,"c1_asset_sha256":digest,"v9_asset_sha256":"63c17e8200a92383b061541fc234dfef36e4b7356954c160ce5f048f820cde96","fixed_units":96,"rows":rows,"statistics":statistics,"failure_ledger":[],"claim_ceiling":n96_contract.config["claim_ceiling"]}); result_sha=_publish_result(result,result_payload)
  print(PREFIX+" "+c1.stable_json_bytes({"status":"complete","completeness":"complete","scientific_status":"exploratory_evaluable","claim_ceiling":n96_contract.config["claim_ceiling"],"exact":exact,"c1_committed_units":32,"c1_pair_count":1056,"n96_committed_units":96,"n96_failed_units":0,"result_path":str(result),"sidecar_path":str(result)+".sha256","result_sha256":result_sha}).decode("ascii"),flush=True); return 0
 except BaseException as error:
  if not failed: failed.append({"phase":phase,"ordinal":None,"unit_id":None,"arm":"unknown","error_class":type(error).__name__})
  c1_state={"status":"calibration_complete","asset_path":str(c1_asset) if c1_published else None,"sidecar_path":str(c1_sidecar) if c1_published else None,"asset_sha256":digest if c1_published else None}
  print(PREFIX+" "+c1.stable_json_bytes({"status":"incomplete","completeness":"incomplete","scientific_status":"not_evaluable","fixed_units":96,"n96_committed_units":0,"n96_failed_units":len(failed),"failure_ledger":failed,"c1":c1_state,"result_path":None,"sidecar_path":None,"result_sha256":None}).decode("ascii"),flush=True); return 2
 finally:
  secret=""; token=""; os.environ.pop(c1.KEY_ENV,None); os.environ.pop(c1.TOKEN_ENV,None)

def _arguments() -> argparse.Namespace:
 p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--expected-exact",required=True); p.add_argument("--local-work-root",required=True); p.add_argument("--artifact-sink",required=True); return p.parse_args()
if __name__=="__main__": raise SystemExit(execute(_arguments()))
