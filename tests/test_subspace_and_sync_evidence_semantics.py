"""
File purpose: 子空间与同步质量证据语义回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np

from main.core import digests
from main.registries.geometry_registry import resolve_sync_module
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
    SubspacePlannerImpl,
)
from main.watermarking.geometry_chain.sync import SyncRuntimeContext


def _build_planner_cfg() -> Dict[str, Any]:
    return {
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 6,
                "sample_count": 12,
                "feature_dim": 32,
                "seed": 9,
                "timestep_start": 0,
                "timestep_end": 12,
            }
        }
    }


def _build_planner_inputs() -> Dict[str, Any]:
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


def test_subspace_plan_contains_evidence_semantics_payload() -> None:
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
    semantics = cast(Dict[str, Any], semantics)
    assert semantics.get("version") == "subspace_evidence_semantics_v1"
    assert isinstance(semantics.get("source"), dict)
    assert semantics.get("evidence_level") in {"surrogate", "hybrid"}
    assert isinstance(semantics.get("reason"), str)


def test_sync_quality_semantics_contains_supporting_evidence_level() -> None:
    factory = resolve_sync_module("geometry_latent_sync_sd3_v2")
    sync_module = factory({})
    cfg: Dict[str, Any] = {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "detect": {"geometry": {"enabled": True, "enable_latent_sync": True}},
    }
    latents = np.random.default_rng(20260227).normal(size=(1, 4, 8, 8)).astype(np.float32)
    context = SyncRuntimeContext(pipeline=object(), latents=latents, rng=None, trajectory_evidence=None)
    result = sync_module.sync_with_context(cfg, context, runtime_inputs={"relation_digest": "r" * 64})

    assert isinstance(result, dict)
    output = cast(Dict[str, Any], result)
    semantics = output.get("sync_quality_semantics")
    assert isinstance(semantics, dict)
    semantics = cast(Dict[str, Any], semantics)
    assert semantics.get("score_type") == "heuristic"
    assert semantics.get("trusted_as_primary_geometry_evidence") is False
    assert semantics.get("evidence_level") == "supporting"

    metrics = output.get("sync_quality_metrics")
    if isinstance(metrics, dict):
        metrics = cast(Dict[str, Any], metrics)
        components = metrics.get("quality_components_v2")
        assert isinstance(components, dict)
        components = cast(Dict[str, Any], components)
        assert components.get("version") == "latent_sync_quality_components_v2"
