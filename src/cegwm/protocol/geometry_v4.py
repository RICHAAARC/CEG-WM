"""Frozen P0 contract for the Geometry-V4 coordination-only route."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

GEOMETRY_V4_METHOD_ID = "geometry_v4_keyed_multiscale_sync_anchor_v1"
GEOMETRY_V4_PROXY_WRITER_ID = "geometry_v4_rgb_proxy_writer_v1"
GEOMETRY_V4_GENERATED_WRITER_ID = "geometry_v4_generated_writer_v1"
GEOMETRY_V4_PROTOCOL_ID = "cegwm-geometry-v4-p0-contract-v1"
GEOMETRY_V4_PROTOCOL_VERSION = 1
GEOMETRY_V4_CONFIG_NAME = "geometry_v4_p0_contract_v1.json"
GEOMETRY_V4_CONFIG_SHA256 = "91a921bc9c768916476f23dee462d6ada3ff6913034e725acd5dd8c3a0c3e4c1"
GEOMETRY_V4_STATUS = ("RELIABLE", "UNRELIABLE", "STOPPED")
GEOMETRY_V4_GEOMETRY_KEY_DOMAIN = (
    b"CEG-WM/geometry-v4/keyed-multiscale-sync-anchor/v1"
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class GeometryV4Observation:
    """Coordinate-only output; it deliberately contains no watermark verdict."""

    H_hat: tuple[float, ...] | None
    corners_hat: tuple[tuple[float, float], ...]
    support: int
    reliability: float
    status: str

    def __post_init__(self) -> None:
        if self.status not in GEOMETRY_V4_STATUS:
            raise ValueError("geometry V4 status differs")
        if isinstance(self.support, bool) or not isinstance(self.support, int) or self.support < 0:
            raise ValueError("geometry V4 support must be a non-negative integer")
        if isinstance(self.reliability, bool) or not isinstance(self.reliability, (int, float)):
            raise ValueError("geometry V4 reliability must be numeric")


def derive_geometry_v4_key(
    detection_key: bytes, *, length: int = 32, salt: bytes = b""
) -> bytes:
    """Derive a versioned geometry subkey without exposing raw key material."""
    if not isinstance(detection_key, bytes) or not detection_key:
        raise TypeError("detection key must be non-empty bytes")
    if not isinstance(salt, bytes):
        raise TypeError("geometry V4 salt must be bytes")
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= 255 * 32:
        raise ValueError("geometry V4 length differs")
    prk = hmac.new(salt or b"\x00" * 32, detection_key, hashlib.sha256).digest()
    output = b""
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(
            prk, previous + GEOMETRY_V4_GEOMETRY_KEY_DOMAIN + bytes((counter,)), hashlib.sha256
        ).digest()
        output += previous
        counter += 1
    return output[:length]


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("ascii")


def load_geometry_v4_p0_contract(repo_root: str | Path) -> Mapping[str, Any]:
    path = Path(repo_root) / "configs" / "geometry_v4" / GEOMETRY_V4_CONFIG_NAME
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != GEOMETRY_V4_CONFIG_SHA256:
        raise ValueError("geometry V4 P0 contract bytes differ")
    try:
        contract = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("geometry V4 P0 contract must be JSON") from error
    if not isinstance(contract, dict) or raw != _canonical_bytes(contract):
        raise ValueError("geometry V4 P0 contract encoding differs")
    if (
        contract.get("p0_protocol_id") != GEOMETRY_V4_PROTOCOL_ID
        or contract.get("p0_protocol_version") != GEOMETRY_V4_PROTOCOL_VERSION
        or contract.get("identities", {}).get("method_id") != GEOMETRY_V4_METHOD_ID
        or contract.get("identities", {}).get("proxy_writer_id") != GEOMETRY_V4_PROXY_WRITER_ID
        or contract.get("identities", {}).get("generated_writer_id") != GEOMETRY_V4_GENERATED_WRITER_ID
    ):
        raise ValueError("geometry V4 P0 identity differs")
    if tuple(contract.get("status_enum", ())) != GEOMETRY_V4_STATUS:
        raise ValueError("geometry V4 status enum differs")
    if contract.get("claim_ceiling") != "P0_local_static_engineering_only_science_denominator_0":
        raise ValueError("geometry V4 P0 claim ceiling differs")
    if contract.get("prohibitions", {}).get("geometry_positive_watermark_decision") is not True:
        raise ValueError("geometry V4 positive-decision prohibition differs")
    return contract


def geometry_v4_contract_digest(contract: Mapping[str, Any]) -> str:
    canonical = _canonical_bytes(contract)
    return hashlib.sha256(canonical).hexdigest()


def require_geometry_v4_contract_digest(digest: str) -> None:
    if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
        raise ValueError("geometry V4 contract digest must be lowercase 64-hex")


__all__ = [name for name in globals() if name.startswith("GEOMETRY_V4_")] + [
    "GeometryV4Observation",
    "derive_geometry_v4_key",
    "geometry_v4_contract_digest",
    "load_geometry_v4_p0_contract",
    "require_geometry_v4_contract_digest",
]
