"""Create-only producer for the independent Content V10 calibration asset."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from cegwm.method.content_v10_texture_neutral import load_independent_calibration_asset
from cegwm.method.content_weighted_joint_v9 import fit_weighted_joint_calibration, stable_json_bytes
from cegwm.protocol.content_chain_v10 import CALIBRATION_MANIFEST_DIGEST, METHOD_ID, load_content_v10_contract
from cegwm.runtime.content_v10_texture_neutral_sd35 import run_content_v10_calibration_unit
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes

KEY_ENV="CEG_WM_ROOT_KEY"; TOKEN_ENV="HF_TOKEN"; PREFIX="CEGWM_CONTENT_V10_CALIBRATION_SUMMARY"
ASSET_FILENAME="content_v10_weighted_joint_calibration.json"
KEY_DOMAIN="stage-a/content-v10-texture-neutral-weighted-joint-calibration-key/v1"

@dataclass(frozen=True)
class CalibrationUnit:
    unit_id: str; source_id: str; block_id: str; block_slot: int; prompt: str; seed: int; height: int; width: int

def _git_exact(root: Path, expected: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None: raise ValueError("expected exact must be lowercase 40-hex")
    exact=subprocess.run(["git","rev-parse","HEAD"],cwd=root,check=True,capture_output=True,text=True).stdout.strip()
    dirty=subprocess.run(["git","status","--porcelain"],cwd=root,check=True,capture_output=True,text=True).stdout
    if exact != expected or dirty: raise RuntimeError("Content V10 calibration checkout identity differs")
    return exact

def _units(root: Path) -> tuple[CalibrationUnit,...]:
    path=root/"configs/content_chain/content_v10_calibration_v1.jsonl"; raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CALIBRATION_MANIFEST_DIGEST: raise ValueError("Content V10 calibration manifest digest differs")
    rows=[json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(rows)!=32: raise ValueError("Content V10 calibration requires exactly 32 units")
    units=tuple(CalibrationUnit(**{key: row[key] for key in CalibrationUnit.__annotations__}) for row in rows)
    if len({unit.unit_id for unit in units})!=32 or any(unit.height!=512 or unit.width!=512 for unit in units): raise ValueError("Content V10 calibration roster differs")
    return units

def derive_calibration_key(root_key: str | bytes) -> bytes:
    return prg_bytes(normalize_detection_key(root_key), KEY_DOMAIN, 32)

def _load_pipeline_and_assets(token: str) -> tuple[Any, Any]:
    import torch
    from experiments import run_content_v6_clean as v6_runner
    if not torch.cuda.is_available(): raise RuntimeError("cuda_required_for_real_Content_V10_calibration")
    pipeline, assets=v6_runner._load_pipeline_and_assets("stabilityai/stable-diffusion-3.5-medium",token)
    return pipeline, assets.evaluation_assets

def _paths(sink: Path, run_id: str) -> tuple[Path,Path]:
    root=sink/run_id; return root/ASSET_FILENAME, root/(ASSET_FILENAME+".sha256")

def _publish(asset: Path, sidecar: Path, payload: bytes) -> str:
    if asset.exists() or sidecar.exists(): raise FileExistsError("Content V10 calibration destination is create-only")
    asset.parent.mkdir(parents=True,exist_ok=False); digest=hashlib.sha256(payload).hexdigest(); made_asset=False
    try:
        with asset.open("xb") as handle: handle.write(payload); made_asset=True
        with sidecar.open("xb") as handle: handle.write(f"{digest}  {ASSET_FILENAME}\n".encode("ascii"))
    except BaseException:
        sidecar.unlink(missing_ok=True)
        if made_asset: asset.unlink(missing_ok=True)
        asset.parent.rmdir()
        raise
    return digest

def _asset_payload(exact: str, fit: Any) -> bytes:
    return stable_json_bytes({"schema_version":1,"method_id":METHOD_ID,"asset_role_id":"content_v10_weighted_joint_calibration","lf_weight":.25,"hf_weight":.75,"lf_scorer_id":"content_v4_whitened_lf_dct_matched_cosine_v1","hf_scorer_id":"frozen_hf_final_rgb_public_vae_global_normalized_correlation","calibration_manifest_digest":CALIBRATION_MANIFEST_DIGEST,"mu_lf":fit.mu_lf,"sigma_lf":fit.sigma_lf,"mu_hf":fit.mu_hf,"sigma_hf":fit.sigma_hf,"rho":fit.rho})

def _summary(**values: Any) -> None:
    print(PREFIX+" "+stable_json_bytes(values).decode("ascii"),flush=True)

def execute(args: argparse.Namespace) -> int:
    exact=""; run_id=""; asset_path=None; sidecar_path=None
    try:
        root=Path(args.repo_root).resolve(); sink=Path(args.artifact_sink).resolve(); local=Path(args.local_work_root).resolve()
        exact=_git_exact(root,args.expected_exact); contract=load_content_v10_contract(root); units=_units(root)
        secret=os.environ.pop(KEY_ENV,""); token=os.environ.pop(TOKEN_ENV,"")
        try:
            if not secret or not token.strip(): raise RuntimeError("Content V10 calibration child-only secrets are required")
            key=derive_calibration_key(secret); public_digest=public_key_digest(key); run_id=hashlib.sha256((contract.digest+public_digest).encode("ascii")).hexdigest()[:24]
            local_run=local/run_id
            if local_run.exists(): raise FileExistsError("Content V10 calibration local work root is create-only")
            local_run.mkdir(parents=True,exist_ok=False)
            asset_path,sidecar_path=_paths(sink,run_id)
            if asset_path.exists() or sidecar_path.exists(): raise FileExistsError("Content V10 calibration destination is create-only")
            pipeline,assets=_load_pipeline_and_assets(token)
            pairs=[]
            for unit in units: pairs.extend(run_content_v10_calibration_unit(pipeline,unit,key,assets))
            if len(pairs)!=1056: raise ValueError("Content V10 calibration pair count differs")
            fit=fit_weighted_joint_calibration(pairs); pairs.clear(); payload=_asset_payload(exact,fit)
            digest=_publish(asset_path,sidecar_path,payload); load_independent_calibration_asset(asset_path,sidecar_path)
            _summary(status="complete",completeness="complete",scientific_status="not_adjudicated",claim_ceiling="v10_calibration_asset_generation_only_no_efficacy_claim",exact=exact,manifest_digest=CALIBRATION_MANIFEST_DIGEST,fixed_units=32,committed_units=32,failed_units=0,pair_count=1056,asset_path=str(asset_path),sidecar_path=str(sidecar_path),asset_sha256=digest)
            return 0
        finally:
            secret=""; token=""; os.environ.pop(KEY_ENV,None); os.environ.pop(TOKEN_ENV,None)
    except BaseException:
        _summary(status="incomplete",completeness="incomplete",scientific_status="not_evaluable",claim_ceiling="v10_calibration_asset_generation_only_no_efficacy_claim",exact=exact,manifest_digest=CALIBRATION_MANIFEST_DIGEST,fixed_units=32,committed_units=0,failed_units=32,pair_count=0,asset_path=None,sidecar_path=None,asset_sha256=None)
        return 2

def _arguments() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",required=True)
    parser.add_argument("--expected-exact",required=True)
    parser.add_argument("--local-work-root",required=True)
    parser.add_argument("--artifact-sink",required=True)
    return parser.parse_args()
if __name__=="__main__": raise SystemExit(execute(_arguments()))
