"""V10 changes only the embed-side Texture contribution to its neutral value."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from cegwm.method.content_adaptive_v3 import ContentSignals, ContentAllocation
from cegwm.method.content_adaptive_v2 import allocate_content
from cegwm.method.content_weighted_joint_v9 import WeightedJointAsset, weighted_joint_score
from cegwm.protocol.content_chain_v10 import METHOD_ID

@dataclass(frozen=True)
class TextureNeutralAllocation:
    allocation: ContentAllocation
    texture_diagnostic: tuple[float, ...]
    texture_contribution: float = .5

def allocate_texture_neutral(signals: ContentSignals) -> TextureNeutralAllocation:
    """Delegate all V9 controls, replacing only Texture with the V2 neutral raw value."""
    texture = tuple(float(x) for x in signals.texture_complexity)
    if len(texture) != 16:
        raise ValueError("Content V10 requires exactly 16 Texture diagnostics")
    neutral = replace(signals, texture_complexity=(0.0,) * 16)
    return TextureNeutralAllocation(allocate_content(neutral), texture)

def load_independent_calibration_asset(path: str | Path, sidecar: str | Path) -> WeightedJointAsset:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if Path(sidecar).read_bytes() != f"{digest}  {Path(path).name}\n".encode("ascii"):
        raise ValueError("Content V10 calibration sidecar differs")
    value: Any = json.loads(raw)
    if not isinstance(value, dict) or value.get("method_id") != METHOD_ID or value.get("asset_role_id") != "content_v10_weighted_joint_calibration":
        raise ValueError("Content V10 independent calibration asset identity differs")
    raise ValueError("Content V10 calibration asset is required but no accepted V10 asset exists")

def weighted_joint_v10(lf: Any, hf: Any, asset: WeightedJointAsset) -> float:
    return weighted_joint_score(lf, hf, asset)
