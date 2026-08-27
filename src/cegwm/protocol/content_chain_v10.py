"""Fail-closed public identity contract for Content V10."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CONFIG_PATH = "configs/content_chain/content_v10_texture_neutral_v1.json"
METHOD_ID = "content_v10_texture_neutral_dual_branch_weighted_joint_v1"

@dataclass(frozen=True)
class ContentV10Contract:
    config: Mapping[str, Any]
    digest: str

def load_content_v10_contract(root: str | Path) -> ContentV10Contract:
    raw = (Path(root) / CONFIG_PATH).read_bytes()
    value = json.loads(raw)
    expected = {"schema_version": 1, "method_id": METHOD_ID,
        "base_method_id": "content_v9_v6_calibrated_weighted_joint_v1",
        "texture_contribution": "fixed_0.5_no_allocation_authority",
        "joint_weights": {"lf": .25, "hf": .75}}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("Content V10 frozen identity differs")
    asset = value.get("calibration_asset")
    if not isinstance(asset, dict) or asset != {"required": True, "method_id": METHOD_ID,
        "asset_role": "content_v10_weighted_joint_calibration", "status": "independent_asset_required",
        "lf_scorer_id": "content_v4_whitened_lf_dct_matched_cosine_v1",
        "hf_scorer_id": "frozen_hf_final_rgb_public_vae_global_normalized_correlation",
        "calibration_manifest_digest_format": "lowercase_64_hex"}:
        raise ValueError("Content V10 independent calibration contract differs")
    return ContentV10Contract(value, hashlib.sha256(raw).hexdigest())
