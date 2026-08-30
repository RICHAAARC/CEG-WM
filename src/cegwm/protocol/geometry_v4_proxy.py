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
P1_DIGEST = "543c71692b6151f7255c41d5df5eca78997a22cddc6a3b27228565c1ee7b05a8"
P1_RUNNER_ID = "geometry_v4_p1_proxy_engine_v1"
P1_SOURCE_ID = "geometry_v4_procedural_rgb_v1"
P1_SOURCE_SHAPE = (64, 64, 3)
P1_H_DIRECTION = "attacked_to_canonical"
P1_SCALE_BOUNDS = (0.65, 1.55)
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
P1_DEVELOPMENT_CANARY_ID = "geometry_v4_p1d_multiscale_sync_matching_canary_v1"
P1_DEVELOPMENT_CANARY_SEEDS = (4101, 4102)
P1_DEVELOPMENT_CANARY_ATTACKS = (
    "identity",
    "rotation_-5",
    "rotation_+5",
    "scale_0.9",
    "scale_1.1",
    "translation_-0.10_0",
    "translation_+0.10_0",
    "crop_rescale_0.9",
)


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
        or energy.get("scales_cycles_per_image") != [8, 16, 24]
        or energy.get("tile_centers") != [0.125, 0.375, 0.625, 0.875]
        or energy.get("global_fraction") != 0.4
        or energy.get("local_fraction") != 0.6
        or energy.get("luma_rms_cap") != 2 / 255
        or energy.get("luma_peak_cap") != 8 / 255
    ):
        raise ValueError("P1 proxy anchor or budget identity differs")
    detector = value.get("detector", {})
    attack_operator = value.get("attack_operator", {})
    source = value.get("source", {})
    modes = value.get("runner_modes", {})
    canary = value.get("development_canary", {})
    if (
        attack_operator.get("public_h_direction") != P1_H_DIRECTION
        or tuple(detector.get("coarse_scale_bounds", ())) != P1_SCALE_BOUNDS
        or tuple(detector.get("rs_refine_scale_bounds", ())) != P1_SCALE_BOUNDS
        or detector.get("cross_scale_reliability_evidence")
        != "unclipped_periodic_raw_rotation_and_raw_log_scale_relative_to_consensus"
        or detector.get("cross_scale_estimation") != "keyed_sparse_constellation_glrt_primary_v1"
        or detector.get("cross_scale_primary_grid")
        != "rotation_deg_-16_to_16_step_0.5_log_scale_log_0.65_to_log_1.55_step_0.01_zero_anchored"
        or detector.get("cross_scale_group_score")
        != "four_component_joint_glrt_with_geometric_mean_completeness"
        or detector.get("cross_scale_endpoint_policy")
        != "primary_endpoint_or_flat_nonfinite_template_degenerate_invalid_fail_closed"
        or detector.get("whole_log_polar_role") != "diagnostic_only"
        or detector.get("local_search_radius_pixels_at_64") != 8
        or detector.get("local_match_min_valid_fraction") != 0.60
        or detector.get("translation_phase_correlation") != "valid_overlap_fixed_hann_normalized_cross_power"
        or P1_SCALE_BOUNDS[1] < 1 / 0.7
        or source.get("generator_id") != P1_SOURCE_ID
        or (source.get("height"), source.get("width"), 3) != P1_SOURCE_SHAPE
        or modes.get("full", {}).get("units_per_split") != 128
        or modes.get("full", {}).get("external_images_allowed") is not False
        or modes.get("full", {}).get("attack_subset_allowed") is not False
        or modes.get("engineering_canary", {}).get("formal_denominator_member") is not False
        or canary.get("id") != P1_DEVELOPMENT_CANARY_ID
        or tuple(canary.get("seeds", ())) != P1_DEVELOPMENT_CANARY_SEEDS
        or tuple(canary.get("attacks", ())) != P1_DEVELOPMENT_CANARY_ATTACKS
        or not set(P1_DEVELOPMENT_CANARY_ATTACKS).issubset(P1_ATTACKS)
    ):
        raise ValueError("P1 proxy H, search, source, or runner-mode identity differs")
    return value
