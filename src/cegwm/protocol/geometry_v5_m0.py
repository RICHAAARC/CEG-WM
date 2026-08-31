"""Byte-bound Geometry-V5 M0 roster and raw-record protocol.

M0 is an engineering development contract. It has no detector decision,
rectification action, or positive watermark evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping



M0_CONFIG_PATH = "configs/geometry_v5/geometry_v5_m0_sd21_v1.json"
M0_MANIFEST_PATH = "configs/geometry_v5/geometry_v5_m0_development_v1.jsonl"
M0_CONFIG_SHA256 = "44180b2da75a161ec2be3768db4a893eca4e7e7445a3f448cd70e64f088338b6"
M0_MANIFEST_SHA256 = "616ac333224e3a4b62b26d0a42db88d6693d150ab3d2d0f77d08450632bb6eba"
M0_METHOD_ID = "geometry_v5_training_free_initial_noise_sync"
M0_CLAIM_CEILING = "M0_SD21_global_RST_development_engineering_only_science_denominator_0"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SEEDS = (7501, 7502, 7503, 7504)
_ATTACK_IDS = (
    "identity", "rotation_-10", "rotation_+10", "scale_0.9", "scale_1.1",
    "translation_x_-0.08", "translation_x_+0.08", "translation_y_-0.08",
    "translation_y_+0.08", "compound_rot+7_scale0.93_tx+0.05_ty-0.04",
    "compound_rot-7_scale1.07_tx-0.05_ty+0.04",
)


class M0RawStatus(str, Enum):
    ESTIMATE_AVAILABLE = "ESTIMATE_AVAILABLE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class M0Unit:
    unit_id: str
    prompt: str
    seed: int


@dataclass(frozen=True, slots=True)
class GeometryV5M0Contract:
    config: Mapping[str, Any]
    units: tuple[M0Unit, ...]
    config_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class GeometryV5M0RawRecord:
    """Raw mechanism output only; truth never enters this type."""

    status: M0RawStatus
    rotation_degrees: float | None
    scale: float | None
    tx: float | None
    ty: float | None
    H_hat: tuple[tuple[float, float, float], ...] | None
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        try:
            status = M0RawStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError("M0 raw status differs") from error
        object.__setattr__(self, "status", status)
        diagnostics = _finite_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))
        if status is M0RawStatus.FAILED:
            if any(value is not None for value in (self.rotation_degrees, self.scale, self.tx, self.ty, self.H_hat)):
                raise ValueError("FAILED must not fabricate an estimate")
            return
        rotation = _finite(self.rotation_degrees, "rotation_degrees")
        scale = _finite(self.scale, "scale")
        tx = _finite(self.tx, "tx")
        ty = _finite(self.ty, "ty")
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        matrix = _attacked_to_canonical_matrix(self.H_hat)
        expected = _assemble_attacked_to_canonical_similarity(rotation, scale, tx, ty)
        if not _matrices_close(matrix, expected):
            raise ValueError("H_hat must match the record attacked_to_canonical R/S/T")
        object.__setattr__(self, "rotation_degrees", rotation)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "tx", tx)
        object.__setattr__(self, "ty", ty)
        object.__setattr__(self, "H_hat", matrix)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def validate_sha256(value: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError("digest must be lowercase 64-hex")
    return value


def load_geometry_v5_m0_contract(repo_root: str | Path) -> GeometryV5M0Contract:
    root = Path(repo_root)
    config_raw = (root / M0_CONFIG_PATH).read_bytes()
    manifest_raw = (root / M0_MANIFEST_PATH).read_bytes()
    config_digest = validate_sha256(M0_CONFIG_SHA256)
    manifest_digest = validate_sha256(M0_MANIFEST_SHA256)
    if hashlib.sha256(config_raw).hexdigest() != config_digest:
        raise ValueError("M0 config byte digest differs")
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_digest:
        raise ValueError("M0 manifest byte digest differs")
    try:
        config = json.loads(config_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("M0 config must be UTF-8 JSON") from error
    if not isinstance(config, dict) or canonical_json_bytes(config) != config_raw:
        raise ValueError("M0 config must use canonical JSON bytes")
    _validate_config(config)
    units = _load_units(manifest_raw)
    return GeometryV5M0Contract(_freeze(config), units, config_digest, manifest_digest)


def _load_units(raw: bytes) -> tuple[M0Unit, ...]:
    if not raw.endswith(b"\n"):
        raise ValueError("M0 manifest must end with newline")
    rows = raw.splitlines()
    if len(rows) != 4:
        raise ValueError("M0 manifest roster differs")
    units: list[M0Unit] = []
    for index, line in enumerate(rows, 1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("M0 manifest must be UTF-8 JSONL") from error
        if not isinstance(row, dict) or tuple(row) != ("unit_id", "prompt", "seed"):
            raise ValueError("M0 manifest fields or order differ")
        if (json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") != line):
            raise ValueError("M0 manifest row bytes differ")
        if row["unit_id"] != f"geometry-v5-m0-dev-{index:04d}" or not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError("M0 manifest unit identity differs")
        if isinstance(row["seed"], bool) or row["seed"] != _SEEDS[index - 1]:
            raise ValueError("M0 manifest seed order differs")
        units.append(M0Unit(**row))
    if len({unit.prompt for unit in units}) != 4:
        raise ValueError("M0 prompts must be distinct")
    return tuple(units)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("method_id") != M0_METHOD_ID or config.get("parent_p0_exact") != "258b823a35e78d07088bddde0baac91d4069123a":
        raise ValueError("M0 identity differs")
    bindings = _mapping(config, "source_bindings")
    if _mapping(bindings, "maxsive").get("exact") != "a9554024aed176e705cc15ca1cbd31b9c7f75bfb" or _mapping(bindings, "tree_ring").get("exact") != "3015283d9cf82e90b628f02ad2121bd37408ca9a":
        raise ValueError("M0 upstream exact differs")
    if "not_copied" not in str(config.get("deliberate_v5_adaptation")) or "not_attributed" not in str(config.get("translation_extension")):
        raise ValueError("M0 adaptation honesty differs")
    runtime = _mapping(config, "runtime")
    if runtime.get("rgb_shape") != [512, 512, 3] or runtime.get("latent_shape") != [4, 64, 64] or runtime.get("steps") != 50 or runtime.get("eta") != 0 or runtime.get("guidance_scale") != 7.5:
        raise ValueError("M0 runtime identity differs")
    template = _mapping(config, "template")
    if template.get("channel") != 3 or template.get("scale") != 5 or template.get("radial_lengths") != [0.2, 0.3, 0.4, 0.5]:
        raise ValueError("M0 template identity differs")
    development = _mapping(config, "development")
    if tuple(development.get("seeds", ())) != _SEEDS or development.get("physical_denominator") != 44 or development.get("replacement_allowed") is not False or development.get("retry_allowed") is not False:
        raise ValueError("M0 fixed denominator differs")
    attacks = development.get("attacks")
    if not isinstance(attacks, list) or tuple(attack.get("attack_id") for attack in attacks if isinstance(attack, dict)) != _ATTACK_IDS:
        raise ValueError("M0 attack order differs")
    scope = _mapping(config, "scope")
    if scope.get("raw_statuses") != ["ESTIMATE_AVAILABLE", "FAILED"] or any(scope.get(name) is not False for name in ("may_emit_RELIABLE", "may_rectify", "may_vote_content")):
        raise ValueError("M0 output boundary differs")
    engineering = _mapping(config, "engineering_evaluation")
    if engineering.get("claim_ceiling") != M0_CLAIM_CEILING:
        raise ValueError("M0 claim ceiling differs")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite non-bool real")
    return float(value)


def _finite_diagnostics(value: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("diagnostics must be a mapping")
    return {str(name): _finite(item, "diagnostic") for name, item in value.items()}


def _attacked_to_canonical_matrix(value: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("H_hat must be 3x3")
    rows = tuple(tuple(_finite(item, "H_hat") for item in row) for row in value)
    if any(len(row) != 3 for row in rows) or rows[2] != (0.0, 0.0, 1.0):
        raise ValueError("H_hat must be normalized attacked_to_canonical similarity")
    a, negative_b, _ = rows[0]
    b, matching_a, _ = rows[1]
    if not _close(a, matching_a) or not _close(negative_b, -b) or a * a + b * b <= 0.0:
        raise ValueError("H_hat must be an orientation-preserving positive-scale similarity")
    return rows


def _matrices_close(
    received: tuple[tuple[float, float, float], ...],
    expected: tuple[tuple[float, float, float], ...],
) -> bool:
    return all(_close(left, right) for left_row, right_row in zip(received, expected, strict=True) for left, right in zip(left_row, right_row, strict=True))


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _assemble_attacked_to_canonical_similarity(
    rotation_degrees: float, scale: float, tx: float, ty: float
) -> tuple[tuple[float, float, float], ...]:
    angle = math.radians(rotation_degrees)
    a, b = scale * math.cos(angle), scale * math.sin(angle)
    return ((a, -b, tx), (b, a, ty), (0.0, 0.0, 1.0))


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be object")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
