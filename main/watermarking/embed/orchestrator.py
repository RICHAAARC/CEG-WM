"""
嵌入流程编排

功能说明：
- 定义了嵌入流程的编排器函数，用于协调不同组件的执行流程。
- 每个编排器函数都接受配置和实现集作为输入，并返回包含业务字段的记录映射。
- 实现了输入验证和错误处理，确保接口的健壮性。
"""

from __future__ import annotations

from typing import Any, Dict

from main.registries.runtime_resolver import BuiltImplSet


def run_embed_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet, cfg_digest: str) -> Dict[str, Any]:
    """
    功能：执行嵌入占位流程。

    Execute embed placeholder flow using injected implementations.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.
        cfg_digest: Canonical SHA256 digest of cfg (computed from include_paths).
                   Passed to content_extractor to bind mask_digest to authoritative digest.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If impl_set is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_set, BuiltImplSet):
        # impl_set 类型不合法，必须 fail-fast。
        raise TypeError("impl_set must be BuiltImplSet")
    if not isinstance(cfg_digest, str) or not cfg_digest:
        # cfg_digest 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg_digest must be non-empty str")

    content_result = impl_set.content_extractor.extract(cfg, cfg_digest=cfg_digest)
    
    # 提取 mask_digest 以绑定到规划器。
    mask_digest = None
    if isinstance(content_result, dict):
        mask_digest = content_result.get("mask_digest")
    # 若 content_result 是对象（如 ContentEvidence），提取 mask_digest。
    elif hasattr(content_result, "mask_digest"):
        mask_digest = content_result.mask_digest
    
    # 调用规划器计算 plan_digest，绑定 cfg_digest + mask_digest + planner_params。
    subspace_result = impl_set.subspace_planner.plan(
        cfg,
        mask_digest=mask_digest,
        cfg_digest=cfg_digest
    )
    sync_result = impl_set.sync_module.sync(cfg)

    # 构造返回的业务字段映射。
    # 关键：plan_digest 作为锚点字段必须同时返回，以供后续 detect 侧验证。
    record_fields = {
        "operation": "embed",
        "embed_placeholder": True,
        "image_path": "placeholder_input.png",
        "watermarked_path": "placeholder_output.png",
        "seed": 42,
        "strength": 0.5,
        "content_result": content_result,
        "subspace_plan": subspace_result,
        "sync_result": sync_result
    }
    
    # 若 subspace_result 是对象，提取 plan_digest 和 basis_digest 作为审计锚点。
    if hasattr(subspace_result, "plan_digest"):
        record_fields["plan_digest"] = subspace_result.plan_digest
        record_fields["basis_digest"] = subspace_result.basis_digest
        record_fields["plan_stats"] = subspace_result.plan_stats
    
    return record_fields
