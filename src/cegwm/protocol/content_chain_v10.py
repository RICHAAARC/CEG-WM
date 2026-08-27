"""Fail-closed public identity contract for Content V10."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CONFIG_PATH = "configs/content_chain/content_v10_texture_neutral_v1.json"
METHOD_ID = "content_v10_texture_neutral_dual_branch_weighted_joint_v1"
CALIBRATION_MANIFEST_DIGEST = "ae35cef6fa997a375987f77456aada0badcd7be3e029fd5115ebf3398dea5af5"
TEXTURE_N96_MANIFEST_DIGEST = "73cdb9d6b840490567dd2a40dbf1bd10140e52ae46a43d00fdc01b24a9bc1fb8"

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
        "calibration_manifest_digest": CALIBRATION_MANIFEST_DIGEST,
        "calibration_manifest_digest_format": "lowercase_64_hex"}:
        raise ValueError("Content V10 independent calibration contract differs")
    calibration = {"fixed_units": 32, "ordered_pairs_per_unit": 33, "required_pairs": 1056,
        "pair_order": ["candidate_wrong_00_to_15", "primary_null_registered", "primary_null_wrong_00_to_15"],
        "key_domain": "stage-a/content-v10-texture-neutral-weighted-joint-calibration-key/v1",
        "wrong_key_domain": "stage-a/content-adaptive-v2-external-wrong-key/v1", "wrong_key_count": 16,
        "fit": {"mean": "binary64_fsum", "sample_sd_ddof": 1, "pearson_rho": "paired"},
        "terminal_failure": "all_or_none_rc2_no_asset",
        "claim_ceiling": "v10_calibration_asset_generation_only_no_efficacy_claim"}
    if value.get("calibration_protocol") != calibration:
        raise ValueError("Content V10 calibration protocol differs")
    texture = {"handoff_exact": "48cc344fb01c099557e620ab7556c731e127d0f2",
        "execution_exact": "7917a7da15fbeee79083b4938362d2bdf202a740",
        "manifest_digest": TEXTURE_N96_MANIFEST_DIGEST}
    if value.get("texture_n96_provenance") != texture:
        raise ValueError("Content V10 Texture provenance differs")
    return ContentV10Contract(value, hashlib.sha256(raw).hexdigest())
