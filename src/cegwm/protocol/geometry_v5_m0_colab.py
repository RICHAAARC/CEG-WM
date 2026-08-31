"""Byte-bound M0-R0 SD2.1 execution contract; no runtime side effects."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

EXECUTION_PATH = "configs/geometry_v5/geometry_v5_m0_sd21_execution_v1.json"
EXECUTION_SHA256 = "8feeb465325e4bfc52c12cfb1d2bab13558e7cdbd45b2b2acdaa10cbb59a44c9"
SOURCE_M0_EXACT = "5f3ad4908c91a9947a625fb803974ee6ec852985"
CLAIM_CEILING = "M0_SD21_global_RST_development_engineering_only_science_denominator_0"
_HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class M0ExecutionContract:
    config: Mapping[str, Any]
    byte_sha256: str


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def load_m0_execution_contract(repo_root: str | Path) -> M0ExecutionContract:
    raw = (Path(repo_root) / EXECUTION_PATH).read_bytes()
    if _HEX64.fullmatch(EXECUTION_SHA256) is None or hashlib.sha256(raw).hexdigest() != EXECUTION_SHA256:
        raise ValueError("M0 execution contract byte digest differs")
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("M0 execution contract is not canonical JSON")
    _validate(value)
    return M0ExecutionContract(_freeze(value), EXECUTION_SHA256)


def _validate(value: Mapping[str, Any]) -> None:
    if value.get("source_m0_exact") != SOURCE_M0_EXACT or value.get("claim_ceiling") != CLAIM_CEILING:
        raise ValueError("M0 execution identity differs")
    model = value.get("model")
    scheduler = value.get("scheduler")
    generation = value.get("generation")
    inversion = value.get("inversion")
    grid = value.get("spectral_grid")
    artifacts = value.get("artifacts")
    if not all(isinstance(item, dict) for item in (model, scheduler, generation, inversion, grid, artifacts)):
        raise ValueError("M0 execution sections differ")
    if model.get("model_id") != "sd2-community/stable-diffusion-2-1-base" or model.get("model_revision") != "4e63672c03103b6c636b8fb4119ba982469b2955":
        raise ValueError("M0 model binding differs")
    if scheduler.get("class") != "DDIMScheduler" or scheduler.get("steps") != 50 or scheduler.get("eta") != 0:
        raise ValueError("M0 DDIM binding differs")
    if generation.get("guidance_scale") != 7.5 or generation.get("prompt") != "manifest_unit_prompt" or inversion.get("prompt") != "" or inversion.get("guidance_scale") != 1.0:
        raise ValueError("M0 prompt or guidance boundary differs")
    if grid.get("rotation_degrees") != [-15, 15, 1] or grid.get("spatial_scale") != [0.85, 1.15, 0.01] or grid.get("direction") != "attacked_to_canonical_spatial":
        raise ValueError("M0 blind spectral grid differs")
    if artifacts.get("mode") != "create_only" or artifacts.get("records") != 44 or artifacts.get("failure_retention") is not True:
        raise ValueError("M0 artifact retention differs")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
