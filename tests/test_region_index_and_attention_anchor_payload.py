"""
文件目的：region_index_spec 与 attention anchor 摘要回归测试。
Module type: General module
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from main.core import digests
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
    SubspacePlannerImpl,
)
from main.watermarking.geometry_chain.attention_anchor_extractor import AttentionAnchorExtractor


def _build_planner_cfg() -> dict:
    return {
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 6,
                "sample_count": 12,
                "feature_dim": 32,
                "seed": 7,
                "timestep_start": 0,
                "timestep_end": 12,
            }
        }
    }


def _build_planner_inputs() -> dict:
    return {
        "trace_signature": {
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "height": 512,
            "width": 512,
        },
        "mask_summary": {
            "area_ratio": 0.4,
            "downsample_grid_shape": [8, 8],
            "downsample_grid_true_indices": [0, 1, 2, 8, 9, 10],
            "downsample_grid_digest": "a" * 64,
        },
        "routing_digest": "b" * 64,
    }


def test_region_index_spec_is_produced_and_digest_anchored() -> None:
    """
    功能：验证 planner 产出 region_index_spec 且 digest 可复算。

    Verify planner emits region index specs and reproducible region_index_digest.

    Args:
        None.

    Returns:
        None.
    """
    impl_digest = digests.canonical_sha256(
        {"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION}
    )
    planner = SubspacePlannerImpl(SUBSPACE_PLANNER_ID, SUBSPACE_PLANNER_VERSION, impl_digest)

    result = planner.plan(
        _build_planner_cfg(),
        mask_digest="mask_digest_anchor",
        cfg_digest="cfg_digest_anchor",
        inputs=_build_planner_inputs(),
    )

    assert result.status == "ok"
    assert isinstance(result.plan, dict)
    plan = result.plan
    semantics = plan.get("subspace_evidence_semantics")
    assert isinstance(semantics, dict)
    assert isinstance(semantics.get("source"), dict)
    assert isinstance(semantics.get("evidence_level"), str)
    assert isinstance(semantics.get("reason"), str)
    assert semantics.get("version") == "subspace_evidence_semantics_v1"

    hf_spec = plan.get("hf_region_index_spec")
    lf_spec = plan.get("lf_region_index_spec")
    route_basis_bridge = plan.get("route_basis_bridge")
    assert isinstance(hf_spec, dict)
    assert isinstance(lf_spec, dict)
    assert isinstance(route_basis_bridge, dict)
    assert route_basis_bridge.get("bridge_version") == "route_basis_bridge_v1"
    assert route_basis_bridge.get("region_index_digest") == plan.get("region_index_digest")

    recomputed = digests.canonical_sha256(
        {
            "hf_region_index_spec": hf_spec,
            "lf_region_index_spec": lf_spec,
        }
    )
    assert recomputed == plan.get("region_index_digest")


def test_attention_anchor_map_relation_has_digest_and_small_payload() -> None:
    """
    功能：验证 attention 关系摘要存在 digest 且 payload 小体积。

    Verify attention relation summary emits digest and compact payload.

    Args:
        None.

    Returns:
        None.
    """
    impl_digest = digests.canonical_sha256(
        {"impl_id": "geometry_attention_anchor_sd3_v1", "impl_version": "v1"}
    )
    extractor = AttentionAnchorExtractor("geometry_attention_anchor_sd3_v1", "v1", impl_digest)

    cfg = {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "anchor_top_k": 4,
            }
        },
    }

    transformer = SimpleNamespace(config=SimpleNamespace(patch_size=2))
    pipeline = SimpleNamespace(transformer=transformer)
    latents = np.random.default_rng(7).standard_normal((1, 4, 16, 16)).astype(np.float32)

    evidence = extractor.extract(cfg, {"pipeline": pipeline, "latents": latents, "rng": None})
    assert evidence.get("status") == "ok"
    assert isinstance(evidence.get("anchor_digest"), str)
    assert len(evidence.get("anchor_digest")) == 64
    assert evidence.get("anchor_source_semantics") == "token_latent_relation_summary"
    assert evidence.get("anchor_evidence_level") == "real"
    assert "anchor_blocking_reason" not in evidence

    anchor_metrics = evidence.get("anchor_metrics")
    assert isinstance(anchor_metrics, dict)
    payload_size = len(json.dumps(anchor_metrics, ensure_ascii=False))
    assert payload_size < 4096
