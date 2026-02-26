"""
File purpose: Runtime smoke tests for new saliency-source policy and mask-conditioned subspace planner impl IDs.
Module type: General module

功能说明：
- 验证新 impl_id 已被 runtime resolver 与 whitelist 同时接受。
- 验证语义显著性策略实现可输出 saliency provenance 锚点。
- 验证 mask-conditioned 子空间规划实现可输出 subspace_conditioning 锚点。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests
from main.policy.runtime_whitelist import assert_impl_allowed, load_runtime_whitelist
from main.registries import runtime_resolver
from main.watermarking.content_chain.semantic_mask_provider import (
    SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID,
)
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SUBSPACE_PLANNER_MASK_CONDITIONED_ID,
)


def _build_runtime_cfg() -> Dict[str, Any]:
    """
    功能：构造用于 runtime smoke 的最小配置。

    Build minimal configuration for runtime smoke tests.

    Args:
        None.

    Returns:
        Minimal config mapping with explicit new impl IDs.
    """
    return {
        "impl": {
            "content_extractor_id": SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID,
            "geometry_extractor_id": "geometry_baseline_identity_v1",
            "fusion_rule_id": "fusion_baseline_identity_v1",
            "subspace_planner_id": SUBSPACE_PLANNER_MASK_CONDITIONED_ID,
            "sync_module_id": "geometry_sync_baseline_v1",
        },
        "evaluate": {
            "target_fpr": 1e-6,
        },
        "allow_threshold_fallback_for_tests": True,
        "enable_mask": True,
        "mask": {
            "saliency_source": "auto_fallback",
            "semantic_model_path": "Z:/path/not_exists/model.safetensors",
            "impl_id": "semantic_saliency_v2",
        },
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 4,
                "sample_count": 8,
                "feature_dim": 16,
                "seed": 7,
            }
        },
    }


def test_runtime_resolver_accepts_new_impl_ids() -> None:
    """
    功能：验证 runtime resolver 与 whitelist 同时接受新 impl_id。

    Validate that the new impl IDs are accepted by both runtime whitelist
    and runtime resolver build pipeline.

    Args:
        None.

    Returns:
        None.
    """
    cfg = _build_runtime_cfg()
    whitelist = load_runtime_whitelist()
    impl_identity = {
        "content_extractor_id": cfg["impl"]["content_extractor_id"],
        "geometry_extractor_id": cfg["impl"]["geometry_extractor_id"],
        "fusion_rule_id": cfg["impl"]["fusion_rule_id"],
        "subspace_planner_id": cfg["impl"]["subspace_planner_id"],
        "sync_module_id": cfg["impl"]["sync_module_id"],
    }
    assert_impl_allowed(whitelist, impl_identity)

    identity, impl_set, _ = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)
    assert identity.content_extractor_id == SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID
    assert identity.subspace_planner_id == SUBSPACE_PLANNER_MASK_CONDITIONED_ID
    assert getattr(impl_set.content_extractor, "impl_id", "") == SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID
    assert getattr(impl_set.subspace_planner, "impl_id", "") == SUBSPACE_PLANNER_MASK_CONDITIONED_ID


def test_runtime_smoke_emits_saliency_and_subspace_conditioning_anchors() -> None:
    """
    功能：验证新 impl 在运行时能产出关键证据锚点。

    Validate that saliency-source policy and mask-conditioned planner
    emit auditable evidence anchors.

    Args:
        None.

    Returns:
        None.
    """
    cfg = _build_runtime_cfg()
    _, impl_set, _ = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)

    content_result = impl_set.content_extractor.extract(
        cfg,
        inputs={
            "image": [1, 2, 3],
            "image_shape": (64, 64, 3),
        },
        cfg_digest=digests.canonical_sha256(cfg),
    )
    assert content_result.status == "ok"
    assert isinstance(content_result.mask_stats, dict)
    assert isinstance(content_result.mask_stats.get("saliency_provenance"), dict)

    plan_result = impl_set.subspace_planner.plan(
        cfg,
        mask_digest=content_result.mask_digest,
        cfg_digest=digests.canonical_sha256(cfg),
        inputs={
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 64,
                "width": 64,
            },
            "mask_summary": content_result.mask_stats,
        },
    )
    assert plan_result.status == "ok"
    assert isinstance(plan_result.plan, dict)
    assert isinstance(plan_result.plan.get("subspace_conditioning"), dict)
    assert isinstance(plan_result.plan_stats, dict)
    assert isinstance(plan_result.plan_stats.get("subspace_conditioning"), dict)
