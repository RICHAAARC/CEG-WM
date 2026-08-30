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
    if tuple(value.get("g0", {}).get("seeds", ())) != (5101, 5102, 5103, 5104):
        raise ValueError("Geometry-V4 G0 seed roster differs")
    if set(value.get("g1", {}).get("attacks", ())) != {"identity", "rotation_5", "scale_0.9", "translation_0.08_0", "crop_0.9"}:
        raise ValueError("Geometry-V4 G1 attack roster differs")
    return value


def contract_sha256(repo_root: str | Path) -> str:
    return hashlib.sha256((Path(repo_root) / "configs" / "geometry_v4" / CONFIG_NAME).read_bytes()).hexdigest()
