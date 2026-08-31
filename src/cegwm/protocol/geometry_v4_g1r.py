"""Frozen V4-G1R contract and safety-only evaluation definitions."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any, Mapping

from cegwm.protocol.geometry_v4 import derive_geometry_v4_key
from cegwm.shared.keys import normalize_detection_key

CONFIG_NAME = "geometry_v4_g1r_v1.json"
CONFIG_SHA256 = "1231298b12c81140b5482053b9c3a6fdf662ca7bf8272753e5fbeb970aafc713"
PROTOCOL_ID = "cegwm-geometry-v4-g1r-v1"
METHOD_ID = "geometry_v4_keyed_multiscale_sync_anchor_v1"
WRITER_ID = "geometry_v4_g1r_vae_decoder_output_writer_v2"
STAGE = "V4-G1R"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
PLACEMENT = "final_VAE_decoder_output_forward_hook_once_before_RGB_postprocess"
DECODER_HOOK_CALLS_REQUIRED = 1
LUMA_RMS_CAP = 2.0 / 255.0
LUMA_PEAK_CAP = 8.0 / 255.0
ENERGY_SHARES = (0.40, 0.36, 0.24)
WRITER_TARGET_RMS_FRACTION = 0.25
FIT_TILE_IDS = (0, 2, 5, 7, 8, 10, 13, 15)
VALIDATE_TILE_IDS = (1, 3, 4, 6, 9, 11, 12, 14)
ATTACKS = ("identity", "rotation_5", "scale_0.9", "translation_0.08_0", "crop_0.9")
DEVELOPMENT_SEEDS = (6201, 6202, 6203, 6204)
CONFIRMATION_SEEDS = (6301, 6302, 6303, 6304)
LEGACY_SEEDS = (6101, 6102, 6103, 6104)
SEARCH_TOP_K = 5
TRANSLATION_PEAKS_PER_RS = 3
TRANSLATION_NMS_RADIUS_PIXELS = 2
SEARCH_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION = 0.10
SEARCH_MACRO_CYCLES = (14.0, 22.0, 30.0)
SEARCH_DIRECTIONS = (0.0, 45.0, 90.0, 135.0)
SEARCH_ATOM_OFFSETS = ((-2.0, -4.0), (-0.75, -1.5), (0.75, 1.5), (2.0, 4.0))
LOCAL_FREQUENCY_PAIRS = (
    (2, 3), (3, 2), (2, -3), (3, -2), (3, 5), (5, 3),
    (3, -5), (5, -3), (4, 5), (5, 4), (4, -5), (5, -4),
    (4, 7), (7, 4), (4, -7), (7, -4), (5, 7), (7, 5),
    (5, -7), (7, -5), (6, 8), (8, 6), (6, -8), (8, -6),
)
TRANSLATION_PSR_MIN = 8.0
LOCAL_PREPROCESSING = "fixed_cubic_polynomial_detrend_then_narrow_band"
FIT_PATCH_WINDOW_DIVISOR = 20
HOLDOUT_PATCH_WINDOW_DIVISORS = (20, 24)
HOLDOUT_FREQUENCY_RADIUS = (12.0, 31.0)
HOLDOUT_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION = 0.10
DEVELOPMENT_ARTIFACT_FILES = ("g1r-development-records.json", "g1r-development-summary.json", "g1r-development-manifest.json")
DEVELOPMENT_NOTEBOOK_ID = "geometry_v4_g0_g1_colab_v4_g1r_development_v1"
DEVELOPMENT_SOURCE_REQUIRED = 4
DEVELOPMENT_CORRECT_SAFE_REQUIRED = 20
FINAL_RGB_PSNR_MIN = 40.0
FINAL_RGB_SSIM_MIN = 0.98
CONTENT_SCORE_DRIFT_MAX = 0.05
SAFETY_TOLERANCES = {"corner": 0.02, "center": 0.02, "rotation_degrees": 2.0, "log_scale": 0.03}
FIT_GATES = {"support": 6, "coverage": 0.75, "macro_regions": 3, "condition": 1e4, "reprojection": 0.02, "correlation": 0.42, "margin": 0.025}
HOLDOUT_GATES = {"coverage": 0.75, "macro_regions": 3, "correlation": 0.06, "margin": 0.015, "psr": 8.0, "rotation_spread": 2.0, "log_scale_spread": 0.03}
KEY_LABELS = {
    "search": b"CEG-WM/geometry-v4/g1r/search/v1",
    "fit": b"CEG-WM/geometry-v4/g1r/fit/v1",
    "validate": b"CEG-WM/geometry-v4/g1r/validate/v1",
}


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("ascii")


def contract_sha256(repo_root: str | Path) -> str:
    return hashlib.sha256((Path(repo_root) / "configs" / "geometry_v4" / CONFIG_NAME).read_bytes()).hexdigest()


def load_contract(repo_root: str | Path) -> Mapping[str, Any]:
    raw = (Path(repo_root) / "configs" / "geometry_v4" / CONFIG_NAME).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("V4-G1R contract bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V4-G1R contract must be canonical JSON") from error
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError("V4-G1R contract encoding differs")
    identity, anchor, rosters = value.get("identity"), value.get("anchor"), value.get("rosters")
    if not isinstance(identity, dict) or not isinstance(anchor, dict) or not isinstance(rosters, dict):
        raise ValueError("V4-G1R contract structure differs")
    if (identity.get("protocol_id"), identity.get("method_id"), identity.get("writer_id"), identity.get("stage")) != (PROTOCOL_ID, METHOD_ID, WRITER_ID, STAGE):
        raise ValueError("V4-G1R identity differs")
    shares = anchor.get("energy_shares", {})
    if (shares.get("search_global"), shares.get("fit_local"), shares.get("validate_holdout")) != ENERGY_SHARES or tuple(anchor.get("fit_tile_ids", ())) != FIT_TILE_IDS or tuple(anchor.get("validate_tile_ids", ())) != VALIDATE_TILE_IDS:
        raise ValueError("V4-G1R anchor partition differs")
    if anchor.get("luma_rms_cap") != LUMA_RMS_CAP or anchor.get("luma_peak_cap") != LUMA_PEAK_CAP or anchor.get("writer_target_rms_fraction") != WRITER_TARGET_RMS_FRACTION:
        raise ValueError("V4-G1R budget differs")
    if tuple(rosters.get("attacks", ())) != ATTACKS or tuple(rosters.get("development", {}).get("seeds", ())) != DEVELOPMENT_SEEDS or tuple(rosters.get("confirmation", {}).get("seeds", ())) != CONFIRMATION_SEEDS or tuple(rosters.get("forbidden_legacy_seeds", ())) != LEGACY_SEEDS or rosters.get("units_per_split") != 20:
        raise ValueError("V4-G1R roster differs")
    if set(FIT_TILE_IDS) & set(VALIDATE_TILE_IDS) or set(FIT_TILE_IDS) | set(VALIDATE_TILE_IDS) != set(range(16)):
        raise ValueError("V4-G1R tile partitions differ")
    search = value.get("search", {})
    if search.get("top_k") != SEARCH_TOP_K or search.get("translation_peaks_per_rs") != TRANSLATION_PEAKS_PER_RS or search.get("translation_nms_radius_pixels") != TRANSLATION_NMS_RADIUS_PIXELS or search.get("coarse_control") != "keyed_normalized_complex_cross_power_phase_correlation" or search.get("keyed_reference_frequency_support_min_fraction") != SEARCH_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION or search.get("phase_consistency") != "candidate_peak_over_surface_rms" or search.get("translation_psr_min_for_reliable") != TRANSLATION_PSR_MIN or value.get("blind_boundary", {}).get("h_direction") != "attacked_to_canonical" or value.get("blind_boundary", {}).get("geometry_can_form_positive") is not False:
        raise ValueError("V4-G1R blind boundary differs")
    if tuple(search.get("macro_cycles", ())) != SEARCH_MACRO_CYCLES or tuple(search.get("directions_degrees", ())) != SEARCH_DIRECTIONS or tuple(tuple(item) for item in search.get("atom_offsets", ())) != SEARCH_ATOM_OFFSETS or search.get("atoms_per_macro") != len(SEARCH_ATOM_OFFSETS) or search.get("coarse_component_set") != "low_and_mid_8_macros" or search.get("fine_component_set") != "all_12_macros" or search.get("component_consensus") != "trim_one_each_side_then_mean" or search.get("frequency_support") != "per_macro_keyed_reference_derived_near_exact":
        raise ValueError("V4-G1R search constellation differs")
    fit, holdout = value.get("fit", {}), value.get("holdout", {})
    if fit.get("local_preprocessing") != LOCAL_PREPROCESSING or (fit.get("support_min"), fit.get("spatial_coverage_min"), fit.get("macro_regions_min"), fit.get("condition_number_max"), fit.get("reprojection_rms_diagonal_max"), fit.get("masked_normalized_correlation_min"), fit.get("match_margin_min")) != (FIT_GATES["support"], FIT_GATES["coverage"], FIT_GATES["macro_regions"], FIT_GATES["condition"], FIT_GATES["reprojection"], FIT_GATES["correlation"], FIT_GATES["margin"]):
        raise ValueError("V4-G1R local preprocessing differs")
    if fit.get("patch_window_divisor") != FIT_PATCH_WINDOW_DIVISOR:
        raise ValueError("V4-G1R fit patch differs")
    if tuple(tuple(item) for item in fit.get("local_frequency_pairs", ())) != LOCAL_FREQUENCY_PAIRS or fit.get("local_code") != "fixed_keyed_24_atom_spread_spectrum":
        raise ValueError("V4-G1R local code differs")
    if (holdout.get("spatial_coverage_min"), holdout.get("macro_regions_min"), holdout.get("masked_normalized_correlation_min"), holdout.get("match_margin_min"), holdout.get("psr_min"), holdout.get("cross_scale_rotation_spread_degrees_max"), holdout.get("cross_scale_log_scale_spread_max")) != (HOLDOUT_GATES["coverage"], HOLDOUT_GATES["macro_regions"], HOLDOUT_GATES["correlation"], HOLDOUT_GATES["margin"], HOLDOUT_GATES["psr"], HOLDOUT_GATES["rotation_spread"], HOLDOUT_GATES["log_scale_spread"]):
        raise ValueError("V4-G1R holdout gates differ")
    if (holdout.get("primary_patch_window_divisor"), holdout.get("secondary_patch_window_divisor"), tuple(holdout.get("narrow_band_frequency_radius", ())), holdout.get("strong_keyed_frequency_support_min_fraction")) != (*HOLDOUT_PATCH_WINDOW_DIVISORS, HOLDOUT_FREQUENCY_RADIUS, HOLDOUT_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION):
        raise ValueError("V4-G1R holdout preprocessing differs")
    development = value.get("development_runner", {})
    if tuple(development.get("artifact_files", ())) != DEVELOPMENT_ARTIFACT_FILES or development.get("notebook_identity") != DEVELOPMENT_NOTEBOOK_ID or development.get("stage") != "development" or development.get("confirmation_allowed") is not False or development.get("units") != 20 or development.get("source_observability_required") != DEVELOPMENT_SOURCE_REQUIRED or development.get("correct_safe_reliable_required") != DEVELOPMENT_CORRECT_SAFE_REQUIRED or development.get("unsafe_per_arm_max") != 0 or development.get("unit_failures_max") != 0 or development.get("final_rgb_psnr_min_exclusive") != FINAL_RGB_PSNR_MIN or development.get("final_rgb_ssim_min_exclusive") != FINAL_RGB_SSIM_MIN or development.get("final_rgb_luma_rms_max") != LUMA_RMS_CAP or development.get("final_rgb_luma_peak_max") != LUMA_PEAK_CAP or development.get("content_score_drift_max_exclusive") != CONTENT_SCORE_DRIFT_MAX:
        raise ValueError("V4-G1R development runner differs")
    if development.get("truth_probe") != "post_arm_freeze_record_only_noninterfering":
        raise ValueError("V4-G1R truth probe boundary differs")
    runtime = value.get("runtime", {})
    if runtime.get("model_id") != MODEL_ID or runtime.get("placement") != PLACEMENT or runtime.get("hook_module") != "AutoencoderKL.decoder" or runtime.get("decoder_output_hook_calls_required") != DECODER_HOOK_CALLS_REQUIRED or runtime.get("single_fixed_update") is not True:
        raise ValueError("V4-G1R runtime identity differs")
    return value


def derive_g1r_keys(detection_key: str | bytes | bytearray | memoryview) -> Mapping[str, bytes]:
    geometry_key = derive_geometry_v4_key(normalize_detection_key(detection_key))
    answer: dict[str, bytes] = {}
    for name, label in KEY_LABELS.items():
        prk = hmac.new(b"\0" * 32, geometry_key, hashlib.sha256).digest()
        answer[name] = hmac.new(prk, label + b"\x01", hashlib.sha256).digest()
    if len(set(answer.values())) != 3:
        raise RuntimeError("V4-G1R key domains collided")
    return answer


def require_split(contract: Mapping[str, Any], split: str) -> tuple[tuple[int, str, str], ...]:
    if split not in {"development", "confirmation"}:
        raise ValueError("V4-G1R split must be development or confirmation")
    roster = contract["rosters"][split]
    seeds, prompts = tuple(roster["seeds"]), tuple(roster["prompts"])
    if len(seeds) != 4 or len(prompts) != 4 or any(seed in LEGACY_SEEDS for seed in seeds):
        raise ValueError("V4-G1R split roster differs")
    return tuple((int(seed), str(prompt), attack) for seed, prompt in zip(seeds, prompts, strict=True) for attack in ATTACKS)


def unsafe_geometry(status: str, metrics: Mapping[str, object]) -> bool:
    """Truth-only runner evaluation; never callable from detector control flow."""
    if status != "RELIABLE":
        return False
    try:
        values = {name: float(metrics[name]) for name in ("mapped_corner_error", "center_reprojection_error", "rotation_abs_error_degrees", "log_scale_abs_error")}
    except (KeyError, TypeError, ValueError):
        return True
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        return True
    return values["mapped_corner_error"] > SAFETY_TOLERANCES["corner"] or values["center_reprojection_error"] > SAFETY_TOLERANCES["center"] or values["rotation_abs_error_degrees"] > SAFETY_TOLERANCES["rotation_degrees"] or values["log_scale_abs_error"] > SAFETY_TOLERANCES["log_scale"]
