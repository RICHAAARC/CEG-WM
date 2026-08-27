"""V10's sole method delta: validated Texture is neutralized only for V3 allocation."""
from __future__ import annotations
import hashlib, json, math, struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from cegwm.protocol.content_chain_v10 import CALIBRATION_MANIFEST_DIGEST, METHOD_ID

_ROLE = "content_v10_weighted_joint_calibration"
_MAX = 255.0 * math.sqrt(2.0)
_LF_SCORER = "content_v4_whitened_lf_dct_matched_cosine_v1"
_HF_SCORER = "frozen_hf_final_rgb_public_vae_global_normalized_correlation"

@dataclass(frozen=True)
class TextureSummary:
    mean: float; std_ddof0: float; minimum: float; q10: float; q25: float; q50: float; q75: float; q90: float; maximum: float; iqr: float; digest: str

@dataclass(frozen=True)
class TextureNeutralAllocation:
    allocation: Any
    texture_summary: TextureSummary
    texture_contribution: float = .5

@dataclass(frozen=True)
class V10CalibrationAsset:
    mu_lf: float; sigma_lf: float; mu_hf: float; sigma_hf: float; rho: float

def _summary(values: tuple[float, ...]) -> TextureSummary:
    ordered = sorted(values)
    def q(p: float) -> float: return ordered[round((len(ordered)-1)*p)]
    mean = sum(values)/len(values); std = math.sqrt(sum((x-mean)**2 for x in values)/len(values))
    raw = b"".join(struct.pack(">d", x) for x in values)
    return TextureSummary(mean,std,ordered[0],q(.1),q(.25),q(.5),q(.75),q(.9),ordered[-1],q(.75)-q(.25),hashlib.sha256(raw).hexdigest())

def allocate_texture_neutral(signals: Any) -> TextureNeutralAllocation:
    """Use V3 allocator and its measurements, after only Texture is neutralized."""
    raw = tuple(signals.texture_complexity)
    if len(raw) != 16 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in raw):
        raise ValueError("Content V10 Texture diagnostics must be 16 real values")
    texture = tuple(float(x) for x in raw)
    if any(not math.isfinite(x) or x < 0.0 or x > _MAX for x in texture):
        raise ValueError("Content V10 Texture diagnostics must be finite RGB8 4-by-4 values")
    from dataclasses import replace
    from cegwm.method.content_adaptive_v3 import allocate_content
    return TextureNeutralAllocation(allocate_content(replace(signals, texture_complexity=(0.0,)*16)), _summary(texture))

def load_independent_calibration_asset(path: str | Path, sidecar: str | Path, *, producer_execution_exact: str | None = None, protocol_digest: str | None = None, calibration_public_key_digest: str | None = None) -> V10CalibrationAsset:
    raw = Path(path).read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if Path(sidecar).read_bytes() != f"{digest}  {Path(path).name}\n".encode("ascii"): raise ValueError("Content V10 calibration sidecar differs")
    value: Mapping[str, Any] = json.loads(raw)
    required={"schema_version","method_id","asset_role_id","lf_weight","hf_weight","lf_scorer_id","hf_scorer_id","calibration_manifest_digest","producer_execution_exact","protocol_digest","calibration_public_key_digest","mu_lf","sigma_lf","mu_hf","sigma_hf","rho"}
    if not isinstance(value,dict) or set(value)!=required or value.get("schema_version")!=1 or value.get("method_id")!=METHOD_ID or value.get("asset_role_id")!=_ROLE: raise ValueError("Content V10 calibration asset identity differs")
    if (value["lf_weight"],value["hf_weight"],value["lf_scorer_id"],value["hf_scorer_id"],value["calibration_manifest_digest"]) != (.25,.75,_LF_SCORER,_HF_SCORER,CALIBRATION_MANIFEST_DIGEST): raise ValueError("Content V10 calibration bindings differ")
    matcher=__import__("re").fullmatch
    if matcher(r"[0-9a-f]{40}",value["producer_execution_exact"]) is None or any(matcher(r"[0-9a-f]{64}",value[key]) is None for key in ("protocol_digest","calibration_public_key_digest")): raise ValueError("Content V10 calibration provenance differs")
    if (producer_execution_exact is not None and value["producer_execution_exact"] != producer_execution_exact) or (protocol_digest is not None and value["protocol_digest"] != protocol_digest) or (calibration_public_key_digest is not None and value["calibration_public_key_digest"] != calibration_public_key_digest): raise ValueError("Content V10 calibration expected provenance differs")
    values=tuple(float(value[x]) for x in ("mu_lf","sigma_lf","mu_hf","sigma_hf","rho"))
    if not all(math.isfinite(x) for x in values) or values[1]<=0 or values[3]<=0 or not -1<=values[4]<=1: raise ValueError("Content V10 calibration payload differs")
    return V10CalibrationAsset(*values)

def weighted_joint_v10(lf: Any, hf: Any, asset: V10CalibrationAsset) -> float:
    if not isinstance(asset,V10CalibrationAsset): raise TypeError("Content V10 requires validated independent calibration asset")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (lf, hf)):
        raise TypeError("Content V10 branch scores must be real scalars")
    lf, hf = float(lf), float(hf)
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in (lf, hf)):
        raise ValueError("Content V10 branch scores must be finite in [-1,1]")
    denominator=math.sqrt(.25**2+.75**2+2*.25*.75*asset.rho)
    if not math.isfinite(denominator) or denominator <= 0.0: raise ValueError("Content V10 joint denominator differs")
    result=((.25*(lf-asset.mu_lf)/asset.sigma_lf)+(.75*(hf-asset.mu_hf)/asset.sigma_hf))/denominator
    if not math.isfinite(result): raise ValueError("Content V10 joint score must be finite")
    return result
