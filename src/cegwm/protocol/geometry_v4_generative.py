"""Frozen G0/G1 identity and fail-closed input boundaries for Geometry-V4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from cegwm.protocol.geometry_v4 import GEOMETRY_V4_GENERATED_WRITER_ID

CONFIG_NAME = "geometry_v4_g0_g1_v1.json"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
PLACEMENT = "20_step_callback_step_19_final_latent_before_VAE_decode"
CALLBACK_STEP_INDEX = 19
LUMA_RMS_CAP = 2.0 / 255.0
LUMA_PEAK_CAP = 8.0 / 255.0
ENERGY_SHARES = (0.4, 0.6)
G1_ROTATION_COARSE_DEGREES = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)
G1_SCALE_COARSE = (0.88, 0.9, 0.95, 1.0, 1.05, 1.1, 1.12)
G1_ROTATION_FINE_OFFSETS = (-1.25, -0.625, 0.0, 0.625, 1.25)
G1_SCALE_FINE_OFFSETS = (-0.025, -0.0125, 0.0, 0.0125, 0.025)
G1_MIN_ANCHOR_SCORE = 3.0
G1_MIN_TRANSLATION_PSR = 4.0
G1_MIN_TILE_SCORE = 0.05
G1_MIN_SUPPORT = 6
G1_MIN_MACRO_REGIONS = 3


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("ascii")


def load_g0_g1_contract(repo_root: str | Path) -> Mapping[str, Any]:
    raw = (Path(repo_root) / "configs" / "geometry_v4" / CONFIG_NAME).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Geometry-V4 G0/G1 contract must be canonical JSON") from error
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError("Geometry-V4 G0/G1 contract encoding differs")
    identity, budget = value.get("identity"), value.get("residual_budget")
    if not isinstance(identity, dict) or not isinstance(budget, dict):
        raise ValueError("Geometry-V4 G0/G1 contract structure differs")
    if (identity.get("model_id"), identity.get("placement"), identity.get("callback_step_index_zero_based"), identity.get("writer_id")) != (MODEL_ID, PLACEMENT, CALLBACK_STEP_INDEX, GEOMETRY_V4_GENERATED_WRITER_ID):
        raise ValueError("Geometry-V4 G0/G1 runtime identity differs")
    if tuple(budget.get("global_local_energy_shares", ())) != ENERGY_SHARES or budget.get("luma_rms_cap") != LUMA_RMS_CAP or budget.get("luma_peak_cap") != LUMA_PEAK_CAP:
        raise ValueError("Geometry-V4 G0/G1 budget differs")
    detector = value.get("content_detector")
    if not isinstance(detector, dict) or detector != {
        "adapter_id": "geometry_v4_reused_content_v9_weighted_joint_rgb_key_only_v1",
        "calibration_asset": "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json",
        "calibration_sidecar": "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json.sha256",
        "hf_scorer": "score_hf_image", "joint_operator": "weighted_joint_score",
        "lf_scorer": "score_content_whitened_lf_image",
        "loader": "experiments.content_iss_engine._load_pipeline_and_assets",
    }:
        raise ValueError("Geometry-V4 G0/G1 content detector identity differs")
    if tuple(value.get("g0", {}).get("seeds", ())) != (5101, 5102, 5103, 5104):
        raise ValueError("Geometry-V4 G0 seed roster differs")
    g1 = value.get("g1", {})
    if tuple(g1.get("seeds", ())) != (6101, 6102, 6103, 6104) or tuple(g1.get("prompts", ())) != (
        "a red ceramic teapot in a sunlit kitchen",
        "a blue paper kite over a grassy field",
        "a glass terrarium with a tiny fern",
        "a silver compass beside a folded map",
    ) or tuple(g1.get("attacks", ())) != ("identity", "rotation_5", "scale_0.9", "translation_0.08_0", "crop_0.9"):
        raise ValueError("Geometry-V4 G1 4-by-5 roster differs")
    g1_detector = value.get("g1_detector")
    if not isinstance(g1_detector, dict) or g1_detector != {
        "h_direction": "attacked_to_canonical",
        "rotation_coarse_degrees": list(G1_ROTATION_COARSE_DEGREES),
        "scale_coarse": list(G1_SCALE_COARSE),
        "rotation_fine_offsets": list(G1_ROTATION_FINE_OFFSETS),
        "scale_fine_offsets": list(G1_SCALE_FINE_OFFSETS),
        "interpolation": "numpy_bilinear_pixel_center",
        "fill": "current_rgb_channel_median",
        "rotation_center": "normalized_image_center_0.5_0.5",
        "order": "centered_rotation_scale_then_translation_then_attacked_to_canonical_inverse",
        "translation": "valid_mask_hann_normalized_cross_power_phase_correlation",
        "candidate_rank": "signed_combined_anchor_then_translation_psr_then_fixed_lexicographic",
        "min_anchor_score": G1_MIN_ANCHOR_SCORE,
        "min_translation_psr": G1_MIN_TRANSLATION_PSR,
        "min_tile_score": G1_MIN_TILE_SCORE,
        "min_support": G1_MIN_SUPPORT,
        "min_macro_regions": G1_MIN_MACRO_REGIONS,
    }:
        raise ValueError("Geometry-V4 G1 blind detector contract differs")
    return value


def contract_sha256(repo_root: str | Path) -> str:
    return hashlib.sha256((Path(repo_root) / "configs" / "geometry_v4" / CONFIG_NAME).read_bytes()).hexdigest()
