"""D2 independent confirmation of the two frozen artifact-selected layers."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, re, subprocess, zipfile
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import torch
from PIL import Image, ImageDraw
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.sd35_qk_observation import SD35QKObservation, SD35QKObservationSpec, observe_sd35_image_qk

PROTOCOL="geometry-v1-qk-d2-independent-confirmation-v1"; SCHEMA="geometry-v1-qk-d2-independent-confirmation-operational-v1"; MODEL_ID="stabilityai/stable-diffusion-3.5-medium"
SOURCE_RUN_ID="geometry-v1-qk-direction-all-layer-41742d462d62"; SOURCE_PROTOCOL="geometry-v1-qk-direction-all-layer-selection-v1"; SOURCE_RUNNER_EXACT="41742d462d62525189855c8ebb2ee1995fb9230a"; SOURCE_STATUS="DIRECTION_TWO_CANDIDATES_FROZEN"
SOURCE_SELECTED=("transformer_blocks.23.attn","transformer_blocks.14.attn"); ATTENTION_LAYER_PATHS=SOURCE_SELECTED
D0_EXACT="4732211beefbeface95cb842c117b9719e362f1a"; D0_RUN_ID="geometry-v1-qk-d0-4732211beefb"; D0_PROTOCOL="geometry-v1-qk-d0-all-layer-discovery-v1"; D0_STATUS="D0_UNRESOLVED"; D0_PLAN_DIGEST="96e1e5ae6fb8ae66a545b1b10d6c896176989272c81ef1fd737184dcdfaea7b8"; D0_ROSTER_DIGEST="88850de32ae0783427f86d0a5c82c6272a30811931ca0f883f6888cf8b83ac9e"
REFS=("d2_confirmation_a","d2_confirmation_b"); TRANSFORMS=("identity","d4","similarity","crop_rescale"); KINDS=("q","k"); CONTROLS=("matched_h","shuffled_h"); PAIRS=tuple(f"{r}-{t}" for r in REFS for t in TRANSFORMS)
UNIT_COUNT=64; MAX_CONTROL_BYTES=1024; MAX_ROOT_BYTES=262144; MAX_UNIT_BYTES=16384; MAX_LAYER_UNIT_BYTES=524288; MAX_LAYER_ZIP_BYTES=1048576; MAX_SOURCE_BYTES=50331648
SUCCESS_PREFIX="CEGWM_GEOMETRY_V1_QK_D2 "; FAILURE_PREFIX="CEGWM_GEOMETRY_V1_QK_D2_FAILURE "
UNIT_FIELDS=frozenset(("pair_id","transform_label","control_label","descriptor_kind","layer_path","reference_grid","attacked_grid","input_identity","h_identity","status","failure_reason","candidate_correspondences","true_match_ranks","coverage","ambiguity_gaps","fit_residual","recovery_error"))
_LEAKS=(re.compile(r"\braw\s*(?:q\s*/\s*k|qk|query|key|token(?:\s+material)?)\b",re.I),re.compile(r"\b(?:hf[_ -]?token|access[_ -]?token|auth(?:entication)?[_ -]?token|api[_ -]?key|bearer\s+[a-z0-9._-]+|token\s+(?:material|credential|secret|value|data)|credential(?:s)?(?:\s+(?:material|value|data))?)\b",re.I),re.compile(r"\b(?:raw\s+weights?|model\s+weights?|weight\s+tensors?)\b",re.I))
_hs=importlib.util.spec_from_file_location("d2_harness",Path(__file__).with_name("run_geometry_v1_qk_equivariance_preflight.py")); assert _hs and _hs.loader; HARNESS=importlib.util.module_from_spec(_hs); _hs.loader.exec_module(HARNESS)

def _json(v:Any,m:int)->bytes:
 d=json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode();
 if len(d)>m: raise ValueError("bounded_json_exceeded")
 return d
def _sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def _write(p:Path,b:bytes)->None:
 with p.open("xb") as h:h.write(b)
def _exact(expected:str,root:Path)->str:
 if not re.fullmatch(r"[0-9a-f]{40}",expected):raise ValueError("invalid_expected_exact")
 actual=subprocess.run(["git","rev-parse","HEAD"],cwd=root,check=True,capture_output=True,text=True).stdout.strip(); dirty=subprocess.run(["git","status","--porcelain"],cwd=root,check=True,capture_output=True,text=True).stdout.strip()
 if actual!=expected or dirty:raise RuntimeError("execution_identity_mismatch")
 return actual
def _read(p:Path,m:int)->Mapping[str,Any]:
 if p.is_symlink() or not p.is_file() or p.stat().st_size>m:raise ValueError("invalid_bounded_source_file")
 try:v=json.loads(p.read_bytes())
 except (OSError,UnicodeDecodeError,json.JSONDecodeError) as e:raise ValueError("invalid_source_json") from e
 if not isinstance(v,dict):raise ValueError("invalid_source_json")
 return v
def _reject_leak(v:Any,depth:int=0)->None:
 if depth>64:raise ValueError("public_value_structure_depth_exceeded")
 if isinstance(v,Mapping):
  for k,x in v.items():
   low=k.lower() if isinstance(k,str) else "";bad=any(z in low for z in ("raw","token","prompt","latent","secret","hf_","private","image_bytes")) or ("weight" in low and low!="two_reference_equal_weight_median")
   if not isinstance(k,str) or bad:raise ValueError("forbidden_public_field")
   _reject_leak(x,depth+1)
 elif isinstance(v,list):
  for x in v:_reject_leak(x,depth+1)
 elif isinstance(v,str):
  low=v.lower(); n=low.replace("\\","/"); path=n.startswith("//") or n.startswith("~/") or "file://" in n or bool(re.search(r"\b[a-z]:/",n)) or bool(re.search(r"(?<![:/a-z0-9._-])//[a-z0-9_.-]+/[a-z0-9_.-]+",n)) or any(x.group(0)!="/content/drive" and not x.group(0).startswith("/content/drive/") for x in re.finditer(r"(?<![:/a-z0-9._-])/[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*",n))
  if any(x in low for x in ("hf_","hf token","secret","prompt","latent")) or any(p.search(low) for p in _LEAKS) or path:raise ValueError("forbidden_public_value")

def _validate_source(root:Path)->dict[str,Any]:
 if root.is_symlink() or not root.is_dir() or any(not p.is_file() or p.is_symlink() for p in root.iterdir()) or {p.name for p in root.iterdir()}!={"receipt.json","manifest.json","terminal.json"}:raise ValueError("source_file_roster_mismatch")
 receipt,manifest,terminal=_read(root/"receipt.json",MAX_ROOT_BYTES),_read(root/"manifest.json",MAX_ROOT_BYTES),_read(root/"terminal.json",MAX_CONTROL_BYTES)
 _reject_leak(receipt);_reject_leak(manifest);_reject_leak(terminal)
 if (receipt.get("run_id"),receipt.get("protocol"),receipt.get("status"),receipt.get("science_denominator"),receipt.get("runner_execution_identity",{}).get("commit"),receipt.get("selected_layer_paths"),receipt.get("declared_unit_count"),receipt.get("audited_unit_count"),receipt.get("artifact_status"))!=(SOURCE_RUN_ID,SOURCE_PROTOCOL,SOURCE_STATUS,0,SOURCE_RUNNER_EXACT,list(SOURCE_SELECTED),768,768,"complete"):raise ValueError("source_receipt_identity_mismatch")
 d0=receipt.get("source_d0_artifact_identity",{})
 if (d0.get("execution_exact"),d0.get("run_id"),d0.get("protocol"),d0.get("plan_digest"),d0.get("roster_digest"),d0.get("status"),d0.get("science_denominator"))!=(D0_EXACT,D0_RUN_ID,D0_PROTOCOL,D0_PLAN_DIGEST,D0_ROSTER_DIGEST,D0_STATUS,0):raise ValueError("source_d0_identity_mismatch")
 md0=manifest.get("source_d0_artifact_identity",{})
 if (manifest.get("run_id"),manifest.get("protocol"),manifest.get("runner_execution_exact"),manifest.get("status"),manifest.get("unit_count"),md0)!=(SOURCE_RUN_ID,SOURCE_PROTOCOL,SOURCE_RUNNER_EXACT,SOURCE_STATUS,768,d0):raise ValueError("source_manifest_identity_mismatch")
 if (terminal.get("run_id"),terminal.get("status"),terminal.get("science_denominator"),terminal.get("selected_layer_paths"))!=(SOURCE_RUN_ID,SOURCE_STATUS,0,list(SOURCE_SELECTED)):raise ValueError("source_terminal_identity_mismatch")
 return {"run_id":SOURCE_RUN_ID,"runner_execution_exact":SOURCE_RUNNER_EXACT,"protocol":SOURCE_PROTOCOL,"status":SOURCE_STATUS,"selected_layer_paths":list(SOURCE_SELECTED),"d0_identity":d0,"science_denominator":0}

def _reference(r:str)->Image.Image:
 im=Image.new("RGB",(512,512),(41,22,71));d=ImageDraw.Draw(im)
 if r==REFS[0]:
  for x in range(11,512,37):d.line((x,0,511,(x*7)%512),fill=((x*9)%256,211,84),width=4)
  d.polygon(((57,81),(231,46),(188,393),(42,287)),fill=(236,104,52));d.ellipse((297,169,468,446),fill=(53,155,229))
 elif r==REFS[1]:
  for y in range(13,512,41):d.arc((13,y-91,493,y+147),24,278,fill=(224,(y*13)%256,118),width=5)
  d.rectangle((71,248,245,454),fill=(65,201,157));d.polygon(((331,37),(478,159),(391,355),(269,187)),fill=(181,82,228))
 else:raise ValueError("unknown_d2_reference")
 return im
def _sim_h()->np.ndarray:
 a=np.deg2rad(7.);s=.93;c,si=np.cos(a)*s,np.sin(a)*s;L=np.array([[c,-si],[si,c]]);center=np.array([256.,256.]);off=center+np.array([13.,17.])-L@center;return np.array([[c,-si,off[0]],[si,c,off[1]],[0.,0.,1.]])
def _pillow_inv(h:np.ndarray)->tuple[float,...]:
 inv=np.linalg.inv(h);L,o=inv[:2,:2],inv[:2,2];z=L@np.array([.5,.5])+o-.5;return(float(L[0,0]),float(L[0,1]),float(z[0]),float(L[1,0]),float(L[1,1]),float(z[1]))
def _attack(im:Image.Image,label:str)->tuple[Image.Image,list[list[float]]]:
 if label=="identity":return im.copy(),np.eye(3).tolist()
 if label=="d4":return im.transpose(Image.Transpose.ROTATE_90),[[0.,1.,0.],[-1.,0.,512.],[0.,0.,1.]]
 if label=="similarity":h=_sim_h();return im.transform((512,512),Image.Transform.AFFINE,_pillow_inv(h),resample=Image.Resampling.BICUBIC),h.tolist()
 if label=="crop_rescale":
  l,t,r,b=32,44,476,468;sx,sy=512/(r-l),512/(b-t);return im.crop((l,t,r,b)).resize((512,512),Image.Resampling.BICUBIC),[[sx,0.,-l*sx],[0.,sy,-t*sy],[0.,0.,1.]]
 raise ValueError("unknown_d2_transform")
def build_fixed_plan()->dict[str,Any]:
 pairs=[]
 for r in REFS:
  for i,t in enumerate(TRANSFORMS):
   _,mh=_attack(_reference(r),t);_,sh=_attack(_reference(r),TRANSFORMS[(i+1)%4]);pairs.append({"reference_id":r,"pair_id":f"{r}-{t}","transform_label":t,"matched_h":mh,"shuffled_h":sh,"resampler":"PIL.Image.Resampling.BICUBIC"})
 return {"schema":"geometry-v1-qk-d2-plan-v1","protocol":PROTOCOL,"reference_recipe_ids":{REFS[0]:"d2-procedural-seed-3141",REFS[1]:"d2-procedural-seed-5926"},"public_observation_seed":73,"d4_transform":"rotate_90","attention_layer_paths":list(ATTENTION_LAYER_PATHS),"pairs":pairs,"declared_unit_count":UNIT_COUNT}
def _null(p:Any)->tuple[torch.Tensor,torch.Tensor]:
 x=p.encode_prompt(prompt="",prompt_2="",prompt_3="",do_classifier_free_guidance=False)
 if not isinstance(x,(tuple,list)) or len(x)!=4 or not isinstance(x[0],torch.Tensor) or not isinstance(x[2],torch.Tensor):raise ValueError("invalid_null_conditioning")
 return x[0].detach(),x[2].detach()
def _topology(p:Any)->None:
 g=getattr(getattr(p,"transformer",None),"get_submodule",None)
 if not callable(g):raise ValueError("fixed_path_topology_unavailable")
 for path in ATTENTION_LAYER_PATHS:
  a=g(path);q,k,h=getattr(a,"to_q",None),getattr(a,"to_k",None),getattr(a,"heads",None)
  if not isinstance(q,torch.nn.Module) or not isinstance(k,torch.nn.Module) or isinstance(h,bool) or not isinstance(h,int) or h<1:raise ValueError("fixed_path_topology_unavailable")
def _spec(p:Any)->SD35QKObservationSpec:
 h,po=_null(p);return SD35QKObservationSpec(MODEL_ID,getattr(p,"_commit_hash",None),ATTENTION_LAYER_PATHS,20,7,73,(8,8),h,po)
def _grid_h(h:Any,rg:Any,ag:Any)->np.ndarray:
 if not all(isinstance(x,tuple) and len(x)==2 and all(isinstance(y,int) and not isinstance(y,bool) and y>0 for y in x) for x in (rg,ag)):raise ValueError("invalid_source_grid")
 a=np.asarray(h,dtype=np.float64)
 if a.shape!=(3,3) or not np.isfinite(a).all():raise ValueError("invalid_rgb_h")
 rr,rc=rg;ar,ac=ag;out=np.diag((ac/512.,ar/512.,1.))@a@np.linalg.inv(np.diag((rc/512.,rr/512.,1.)))
 if not np.isfinite(out).all():raise ValueError("invalid_grid_h")
 return out
def _failure(pair:Mapping[str,Any],path:str,kind:str,control:str,reason:str)->dict[str,Any]:return {"pair_id":pair["pair_id"],"transform_label":pair["transform_label"],"control_label":control,"descriptor_kind":kind,"layer_path":path,"reference_grid":None,"attacked_grid":None,"input_identity":None,"h_identity":None,"status":"failed","failure_reason":reason,"candidate_correspondences":[],"true_match_ranks":[],"coverage":None,"ambiguity_gaps":[],"fit_residual":None,"recovery_error":None}
def _layer(o:SD35QKObservation,path:str)->Any:
 for x in o.layers:
  if x.layer_path==path:return x
 raise ValueError("fixed_layer_not_observed")
def _unit(pair:Mapping[str,Any],r:SD35QKObservation,a:SD35QKObservation,path:str,kind:str,control:str)->dict[str,Any]:
 x,y=_layer(r,path),_layer(a,path);name="query" if kind=="q" else "key";return HARNESS.evaluate_unit({"pair_id":pair["pair_id"],"transform_label":pair["transform_label"],"control_label":control,"descriptor_kind":kind,"layer_path":path,"reference_descriptors":getattr(x,name).detach().cpu().numpy(),"attacked_descriptors":getattr(y,name).detach().cpu().numpy(),"reference_source_grid":x.source_grid,"attacked_source_grid":y.source_grid,"reference_sample_indices":x.sample_indices.detach().cpu().numpy(),"attacked_sample_indices":y.sample_indices.detach().cpu().numpy(),"H_reference_to_attacked":_grid_h(pair[control],x.source_grid,y.source_grid)})
def _expand(plan:Mapping[str,Any],reason:str)->tuple[dict[str,Any],...]:return tuple(_failure(p,l,k,c,reason) for p in plan["pairs"] for l in ATTENTION_LAYER_PATHS for k in KINDS for c in CONTROLS)
def _finite(v:Any)->float|None:
 if v is None:return None
 if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)):raise ValueError("invalid_true_match_rank")
 return float(v)
def _stats(units:Sequence[Mapping[str,Any]])->tuple[bool,list[dict[str,Any]],dict[str,Any]]:
 stats=[]
 for path in ATTENTION_LAYER_PATHS:
  for kind in KINDS:
   pairs=[]
   for pid in PAIRS:
    m=[u for u in units if (u["pair_id"],u["layer_path"],u["descriptor_kind"],u["control_label"])==(pid,path,kind,"matched_h")];s=[u for u in units if (u["pair_id"],u["layer_path"],u["descriptor_kind"],u["control_label"])==(pid,path,kind,"shuffled_h")]
    if len(m)!=1 or len(s)!=1 or m[0]["status"]!="calculated" or s[0]["status"]!="calculated" or len(m[0]["true_match_ranks"])!=len(s[0]["true_match_ranks"]):pairs.append({"pair_id":pid,"transform_label":pid.rsplit("-",1)[1],"common_finite_count":0,"pair_median":None});continue
    d=[a-b for a,b in zip((_finite(x) for x in m[0]["true_match_ranks"]),(_finite(x) for x in s[0]["true_match_ranks"])) if a is not None and b is not None];pairs.append({"pair_id":pid,"transform_label":pid.rsplit("-",1)[1],"common_finite_count":len(d),"pair_median":float(median(d)) if d else None})
   support=all(x["pair_median"] is not None for x in pairs);val=float(median([x["pair_median"] for x in pairs])) if support else None;trans=[]
   for t in TRANSFORMS:
    x=[z for z in pairs if z["transform_label"]==t];v=[z["pair_median"] for z in x if z["pair_median"] is not None];trans.append({"transform_label":t,"two_reference_common_counts":[z["common_finite_count"] for z in x],"two_reference_medians":[z["pair_median"] for z in x],"two_reference_equal_weight_median":float(median(v)) if len(v)==2 else None})
   stats.append({"layer_path":path,"descriptor_kind":kind,"statistic":val,"all_eight_pairs_supported":support,"pair_audit":pairs,"per_transform_audit":trans,"strictly_negative":val is not None and val<0})
 route=[]
 for t in TRANSFORMS:
  vals=[s["per_transform_audit"][TRANSFORMS.index(t)]["two_reference_equal_weight_median"] for s in stats];fin=[x for x in vals if x is not None and math.isfinite(float(x))];route.append({"transform_label":t,"finite_stat_count":len(fin),"nonnegative_stat_count":sum(x>=0 for x in fin),"all_layer_nonnegative":len(fin)==4 and all(x>=0 for x in fin)})
 audit={"per_transform":route,"route_level_transform_instability":any(x["transform_label"] in ("d4","crop_rescale") and x["all_layer_nonnegative"] for x in route)};return all(x["strictly_negative"] for x in stats),stats,audit
def run_d2(*,expected_exact:str,repo_root:Path,source_root:Path,hf_token:str,loader:Callable[...,Any]=load_sd35_pipeline,observer:Callable[...,SD35QKObservation]=observe_sd35_image_qk,source_identity:Mapping[str,Any]|None=None)->tuple[dict[str,Any],tuple[dict[str,Any],...]]:
 source=dict(source_identity) if source_identity is not None else _validate_source(source_root);plan=build_fixed_plan();exact=_exact(expected_exact,repo_root);status="D2_STOPPED";reason="model_or_topology_unavailable";failure="model_load";runtime={}
 try:
  p=loader(MODEL_ID,torch_dtype=torch.float16,token=hf_token);p=p.to("cuda" if torch.cuda.is_available() else "cpu") if hasattr(p,"to") else p;_topology(p);spec=_spec(p);status="D2_UNRESOLVED";reason=None;failure=None;runtime={"pipeline_class":f"{type(p).__module__}.{type(p).__qualname__}"}
 except BaseException:spec=None;p=None
 units=[];global_reason=None
 for r in REFS:
  ref=None;ref_reason=None
  if spec is not None:
   try:ref=observer(_reference(r),pipeline=p,spec=spec)
   except BaseException as e:ref_reason="reference_observation_failed";failure="image_observation";global_reason="global_transformer_or_capture_failure" if getattr(e,"geometry_failure_point",None) in ("transformer_call","qk_capture") else global_reason
  for pair in (x for x in plan["pairs"] if x["reference_id"]==r):
   attacked=None;pair_reason=reason if spec is None else None
   if spec is not None:
    try:attacked=observer(_attack(_reference(r),pair["transform_label"])[0],pipeline=p,spec=spec)
    except BaseException as e:pair_reason="attacked_observation_failed";failure="image_observation";global_reason="global_transformer_or_capture_failure" if getattr(e,"geometry_failure_point",None) in ("transformer_call","qk_capture") else global_reason
    if ref_reason is not None:pair_reason=ref_reason
   for l in ATTENTION_LAYER_PATHS:
    for k in KINDS:
     for c in CONTROLS:
      try:units.append(_failure(pair,l,k,c,pair_reason) if pair_reason else _unit(pair,ref,attacked,l,k,c))
      except (AttributeError,KeyError,TypeError,ValueError):units.append(_failure(pair,l,k,c,"layer_observation_or_calculation_failed"))
 if len(units)!=UNIT_COUNT:raise RuntimeError("d2_fixed_unit_expansion_mismatch")
 if spec is None:units=list(_expand(plan,reason))
 if global_reason is not None:units=list(_expand(plan,global_reason));status="D2_STOPPED"
 if status!="D2_STOPPED" and any(u["status"]!="calculated" for u in units):status="D2_STOPPED"
 if status=="D2_STOPPED":stats=[];audit={"per_transform":[],"route_level_transform_instability":False}
 else:
  confirmed,stats,audit=_stats(units);status="D2_CANDIDATES_CONFIRMED" if confirmed else "D2_UNRESOLVED"
 summary={"schema":SCHEMA,"protocol":PROTOCOL,"run_id":f"geometry-v1-qk-d2-{exact[:12]}","runner_execution_identity":{"commit":exact},"source_direction_artifact_identity":source,"plan_digest":_sha(_json(plan,MAX_ROOT_BYTES)),"fixed_layer_paths":list(ATTENTION_LAYER_PATHS),"declared_unit_count":UNIT_COUNT,"calculated_unit_count":sum(u["status"]=="calculated" for u in units),"failed_unit_count":sum(u["status"]=="failed" for u in units),"direction_statistics":stats,"route_audit":audit,"status":status,"science_denominator":0,"operational_failure_point":failure,"runtime":runtime,"artifact_status":"unavailable"}
 return summary,tuple(units)
def _package(root:Path,summary:dict[str,Any],units:Sequence[Mapping[str,Any]])->dict[str,Any]:
 if root.exists():raise FileExistsError("output_root_must_be_create_only")
 root.mkdir(parents=True);d=root/"layers";d.mkdir();shards=[]
 for i,path in enumerate(ATTENTION_LAYER_PATHS):
  us=[u for u in units if u["layer_path"]==path]
  if len(us)!=32:raise ValueError("layer_shard_count_mismatch")
  raw=[]
  for u in us:
   if frozenset(u)!=UNIT_FIELDS:raise ValueError("invalid_public_unit_fields")
   _reject_leak(u);raw.append(_json(u,MAX_UNIT_BYTES))
  if sum(map(len,raw))>MAX_LAYER_UNIT_BYTES:raise ValueError("layer_unit_bound_exceeded")
  target=d/f"{i:02d}.zip"
  with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as z:
   for j,x in enumerate(raw):z.writestr(f"{j:02d}.json",x)
  if target.stat().st_size>MAX_LAYER_ZIP_BYTES:raise ValueError("layer_zip_bound_exceeded")
  shards.append({"layer_path":path,"filename":f"layers/{i:02d}.zip","unit_count":32,"bytes":target.stat().st_size})
 summary["artifact_status"]="complete";summary["layer_shards"]=shards;_write(root/"receipt.json",_json(summary,MAX_ROOT_BYTES));_write(root/"manifest.json",_json({"run_id":summary["run_id"],"protocol":PROTOCOL,"runner_execution_exact":summary["runner_execution_identity"]["commit"],"source_direction_artifact_identity":summary["source_direction_artifact_identity"],"status":summary["status"],"unit_count":UNIT_COUNT,"layer_shards":shards},MAX_ROOT_BYTES));_write(root/"terminal.json",_json({"run_id":summary["run_id"],"status":summary["status"],"fixed_layer_paths":list(ATTENTION_LAYER_PATHS),"science_denominator":0},MAX_CONTROL_BYTES));return {"artifact_status":"complete","receipt_filename":"receipt.json","manifest_filename":"manifest.json"}
def _emit(fd:int,prefix:str,v:Mapping[str,Any])->None:
 line=prefix.encode()+_json(v,MAX_CONTROL_BYTES-len(prefix)-1)+b"\n";
 if len(line)>MAX_CONTROL_BYTES:raise ValueError("control_bound_exceeded")
 os.write(fd,line)
def _error(e:BaseException)->str:
 if isinstance(e,(FileExistsError,FileNotFoundError,PermissionError,OSError)):return "filesystem_error"
 if isinstance(e,(ValueError,TypeError,json.JSONDecodeError,zipfile.BadZipFile)):return "validation_error"
 if isinstance(e,subprocess.SubprocessError):return "subprocess_error"
 if isinstance(e,RuntimeError):return "runtime_error"
 return "unexpected_error"
def _main(argv:list[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument("--repo-root",required=True);p.add_argument("--expected-exact",required=True);p.add_argument("--source-root",required=True);p.add_argument("--output-root",required=True);p.add_argument("--control-fd",required=True,type=int);a=p.parse_args(argv);stage="source_validation";run=f"geometry-v1-qk-d2-{a.expected_exact[:12]}"
 try:
  source=_validate_source(Path(a.source_root));stage="run_d2";summary,units=run_d2(expected_exact=a.expected_exact,repo_root=Path(a.repo_root),source_root=Path(a.source_root),hf_token=os.environ.get("HF_TOKEN",""),source_identity=source);stage="artifact_packaging";package=_package(Path(a.output_root),summary,units);stage="control_channel";_emit(a.control_fd,SUCCESS_PREFIX,{"status":"success","run_id":summary["run_id"],"d2_status":summary["status"],"fixed_layer_paths":list(ATTENTION_LAYER_PATHS),"science_denominator":0,**package});return 0
 except BaseException as e:
  if stage=="control_channel":return 1
  try:_emit(a.control_fd,FAILURE_PREFIX,{"status":"failure","run_id":run,"failure_point":stage,"error_class":_error(e),"artifact_status":"unavailable"})
  except BaseException:pass
  return 1
if __name__=="__main__":raise SystemExit(_main())
