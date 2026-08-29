"""Canonical P1 RGB-proxy contract loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from cegwm.protocol.geometry_v4 import (
    GEOMETRY_V4_METHOD_ID,
    GEOMETRY_V4_PROTOCOL_ID,
    GEOMETRY_V4_PROXY_WRITER_ID,
)

P1_CONFIG = "geometry_v4_p1_proxy_v1.json"
P1_DIGEST = "7495f741a143d9a21ab39c17fd0d28e4549dcbbfb478bf5da5f321f286b62cc4"
P1_RUNNER_ID = "geometry_v4_p1_proxy_engine_v1"
P1_ATTACKS = (
    "identity",
    "rotation_-10",
    "rotation_-5",
    "rotation_+5",
    "rotation_+10",
    "scale_0.9",
    "scale_1.1",
    "translation_-0.10_0",
    "translation_+0.10_0",
    "translation_0_-0.10",
    "translation_0_+0.10",
    "crop_rescale_0.9",
    "crop_rescale_0.8",
    "crop_rescale_0.7",
    "compound_-7_0.9_+0.05_-0.05",
    "compound_+7_1.1_-0.05_+0.05",
)
P1_SPLITS = {"P1D": tuple(range(4101, 4109)), "P1C": tuple(range(4201, 4209))}


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_p1_proxy(root: str | Path) -> Mapping[str, Any]:
    raw = (Path(root) / "configs" / "geometry_v4" / P1_CONFIG).read_bytes()
    if hashlib.sha256(raw).hexdigest() != P1_DIGEST:
        raise ValueError("P1 proxy config differs")
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError("P1 proxy config is noncanonical")
    identities = value.get("identities", {})
    if identities != {
        "method_id": GEOMETRY_V4_METHOD_ID,
        "protocol_id": GEOMETRY_V4_PROTOCOL_ID,
        "proxy_writer_id": GEOMETRY_V4_PROXY_WRITER_ID,
        "runner_id": P1_RUNNER_ID,
    }:
        raise ValueError("P1 proxy identity differs")
    if tuple(value.get("attacks", ())) != P1_ATTACKS:
        raise ValueError("P1 proxy attack roster differs")
    splits = value.get("splits", {})
    if any(tuple(splits.get(name, {}).get("seeds", ())) != seeds for name, seeds in P1_SPLITS.items()):
        raise ValueError("P1 proxy split differs")
    energy = value.get("energy", {})
    if (
        energy.get("directions_deg") != [0, 45, 90, 135]
        or energy.get("scales_cycles_per_image") != [8, 16, 32]
        or energy.get("tile_centers") != [0.125, 0.375, 0.625, 0.875]
        or energy.get("global_fraction") != 0.4
        or energy.get("local_fraction") != 0.6
        or energy.get("luma_rms_cap") != 2 / 255
        or energy.get("luma_peak_cap") != 8 / 255
    ):
        raise ValueError("P1 proxy anchor or budget identity differs")
    return value
