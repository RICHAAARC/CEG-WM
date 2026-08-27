"""Fail-closed public identity contract for Content V10."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CONFIG_PATH = "configs/content_chain/content_v10_texture_neutral_v1.json"
N96_PAIRED_CONFIG_PATH = "configs/content_chain/content_v10_n96_paired_v1.json"
METHOD_ID = "content_v10_texture_neutral_dual_branch_weighted_joint_v1"
CALIBRATION_MANIFEST_DIGEST = "ae35cef6fa997a375987f77456aada0badcd7be3e029fd5115ebf3398dea5af5"
TEXTURE_N96_MANIFEST_DIGEST = "73cdb9d6b840490567dd2a40dbf1bd10140e52ae46a43d00fdc01b24a9bc1fb8"

@dataclass(frozen=True)
class ContentV10Contract:
    config: Mapping[str, Any]
    digest: str

@dataclass(frozen=True)
class ContentV10N96PairedContract:
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

def load_content_v10_n96_paired_contract(root: str | Path) -> ContentV10N96PairedContract:
    root = Path(root); raw = (root / N96_PAIRED_CONFIG_PATH).read_bytes(); value = json.loads(raw)
    manifest = root / "configs/content_chain/content_texture_n96_evaluation_v1.jsonl"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != TEXTURE_N96_MANIFEST_DIGEST: raise ValueError("Content V10 N96 manifest digest differs")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    calibration = [json.loads(line) for line in (root / "configs/content_chain/content_v10_calibration_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 96 or len(calibration) != 32: raise ValueError("Content V10 paired roster denominator differs")
    for field in ("unit_id", "source_id", "seed"):
        if {row[field] for row in rows} & {row[field] for row in calibration}: raise ValueError("Content V10 calibration and N96 rosters overlap")
    if {row["prompt"].encode("utf-8") for row in rows} & {row["prompt"].encode("utf-8") for row in calibration}: raise ValueError("Content V10 calibration and N96 rosters overlap")
    if {(row["prompt"].encode("utf-8"), row["seed"]) for row in rows} & {(row["prompt"].encode("utf-8"), row["seed"]) for row in calibration}: raise ValueError("Content V10 calibration and N96 prompt-seed pairs overlap")
    expected = {"schema_version": 1, "analysis_id": "content_v10_n96_paired_texture_allocator_v1", "claim_ceiling": "exploratory_prospective_paired_texture_allocator_evaluation_no_superiority_claim",
        "n96_manifest": {"path": "configs/content_chain/content_texture_n96_evaluation_v1.jsonl", "sha256": TEXTURE_N96_MANIFEST_DIGEST, "fixed_units": 96, "common_plain_per_unit": 1, "beta_per_unit": 1},
        "c1_precondition": {"accepted_v10_asset_required": True, "calibration_manifest_sha256": CALIBRATION_MANIFEST_DIGEST, "fixed_units": 32, "asset_readback_required": True},
        "v9_precondition": {"asset_sha256": "63c17e8200a92383b061541fc234dfef36e4b7356954c160ce5f048f820cde96", "fitter_forbidden": True},
        "paired_arms": {"v9": "default_allocate_content", "v10": "allocate_texture_neutral", "same_seed": True, "same_registered_and_16_wrong_keys": True, "scorers": ["content_v4_whitened_lf_dct_matched_cosine_v1", "frozen_hf_final_rgb_public_vae_global_normalized_correlation"]},
        "statistics": {"margin_a": "registered_minus_max_wrong", "margin_b": "registered_minus_common_plain_registered", "paired": ["v10_minus_v9_mean", "v10_minus_v9_positive_count"], "texture_spearman": "tie_aware_method_margins_and_deltas", "joint": "whole_system_descriptive_only"},
        "completion": {"fixed_denominator": 96, "any_arm_or_unit_failure": "rc2_incomplete"}, "compute": {"c1_diffusion_calls": 64, "n96_diffusion_calls": 288, "total_diffusion_calls": 352}}
    if value != expected: raise ValueError("Content V10 N96 paired contract differs")
    return ContentV10N96PairedContract(value, hashlib.sha256(raw).hexdigest())
