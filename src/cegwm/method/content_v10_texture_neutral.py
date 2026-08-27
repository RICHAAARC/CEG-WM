"""V10's sole method delta: validated Texture is neutralized only for V3 allocation."""
from __future__ import annotations
import hashlib, json, math, struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from cegwm.protocol.content_chain_v10 import METHOD_ID

_ROLE = "content_v10_weighted_joint_calibration"
_MAX = 255.0 * math.sqrt(2.0)

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
    texture = tuple(float(x) for x in signals.texture_complexity)
    if len(texture) != 16 or any(not math.isfinite(x) or x < 0.0 or x > _MAX for x in texture):
        raise ValueError("Content V10 Texture diagnostics must be finite RGB8 4-by-4 values")
    from dataclasses import replace
    from cegwm.method.content_adaptive_v3 import allocate_content
    return TextureNeutralAllocation(allocate_content(replace(signals, texture_complexity=(0.0,)*16)), _summary(texture))

def load_independent_calibration_asset(path: str | Path, sidecar: str | Path) -> V10CalibrationAsset:
    raw = Path(path).read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if Path(sidecar).read_bytes() != f"{digest}  {Path(path).name}\n".encode("ascii"): raise ValueError("Content V10 calibration sidecar differs")
    value: Mapping[str, Any] = json.loads(raw)
    required={"schema_version","method_id","asset_role_id","lf_weight","hf_weight","lf_scorer_id","hf_scorer_id","calibration_manifest_digest","mu_lf","sigma_lf","mu_hf","sigma_hf","rho"}
    if not isinstance(value,dict) or set(value)!=required or value.get("schema_version")!=1 or value.get("method_id")!=METHOD_ID or value.get("asset_role_id")!=_ROLE: raise ValueError("Content V10 calibration asset identity differs")
    if (value["lf_weight"],value["hf_weight"]) != (.25,.75) or not all(isinstance(value[x],str) and value[x] for x in ("lf_scorer_id","hf_scorer_id","calibration_manifest_digest")): raise ValueError("Content V10 calibration bindings differ")
    values=tuple(float(value[x]) for x in ("mu_lf","sigma_lf","mu_hf","sigma_hf","rho"))
    if not all(math.isfinite(x) for x in values) or values[1]<=0 or values[3]<=0 or not -1<=values[4]<=1: raise ValueError("Content V10 calibration payload differs")
    return V10CalibrationAsset(*values)

def weighted_joint_v10(lf: Any, hf: Any, asset: V10CalibrationAsset) -> float:
    if not isinstance(asset,V10CalibrationAsset): raise TypeError("Content V10 requires validated independent calibration asset")
    zlf=(float(lf)-asset.mu_lf)/asset.sigma_lf; zhf=(float(hf)-asset.mu_hf)/asset.sigma_hf
    return (.25*zlf+.75*zhf)/math.sqrt(.25**2+.75**2+2*.25*.75*asset.rho)
