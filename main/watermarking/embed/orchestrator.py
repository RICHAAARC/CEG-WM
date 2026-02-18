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
    
    # 捕获 content_chain 的执行状态（用于 execution_report）。
    # 允许的值：ok / fail / absent
    content_chain_status = "ok"
    if isinstance(content_result, dict):
        content_status = content_result.get("status", "unknown")
        if content_status == "absent":
            content_chain_status = "absent"
        elif content_status != "ok":
            content_chain_status = "fail"
    elif hasattr(content_result, "status"):
        content_status = content_result.status
        if content_status == "absent":
            content_chain_status = "absent"
        elif content_status != "ok":
            content_chain_status = "fail"
    
    # 提取 mask_digest 以绑定到规划器。
    mask_digest = None
    if isinstance(content_result, dict):
        mask_digest = content_result.get("mask_digest")
    # 若 content_result 是对象（如 ContentEvidence），提取 mask_digest。
    elif hasattr(content_result, "mask_digest"):
        mask_digest = content_result.mask_digest
    
    # 调用规划器计算 plan_digest，绑定 cfg_digest + mask_digest + planner_params。
    planner_inputs = _build_planner_inputs_for_runtime(cfg)
    subspace_result = impl_set.subspace_planner.plan(
        cfg,
        mask_digest=mask_digest,
        cfg_digest=cfg_digest,
        inputs=planner_inputs
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
        "subspace_plan": subspace_result.as_dict() if hasattr(subspace_result, "as_dict") else subspace_result,
        "sync_result": sync_result,
        # 添加 execution_report（冻结门禁要求）。
        # 注：embed 阶段未执行融合，fusion_status 置为 "absent"；
        #     geometry 链不参与，geometry_chain_status 置为 "absent"。
        "execution_report": {
            "content_chain_status": content_chain_status,
            "geometry_chain_status": "absent",
            "fusion_status": "absent",
            "audit_obligations_satisfied": True
        }
    }
    
    # 若 subspace_result 是对象，提取 plan_digest 和 basis_digest 作为审计锚点。
    if hasattr(subspace_result, "plan_digest"):
        record_fields["plan_digest"] = subspace_result.plan_digest
        record_fields["basis_digest"] = subspace_result.basis_digest
        record_fields["plan_stats"] = subspace_result.plan_stats
        if isinstance(subspace_result.plan, dict):
            record_fields["subspace_rank"] = subspace_result.plan.get("rank")
            record_fields["subspace_energy_ratio"] = subspace_result.plan.get("energy_ratio")
            record_fields["subspace_planner_impl_identity"] = subspace_result.plan.get("planner_impl_identity")
    
    return record_fields


def _build_planner_inputs_for_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造规划器输入签名。

    Build deterministic planner input signature from cfg runtime fields.

    Args:
        cfg: Configuration mapping.

    Returns:
        Planner input mapping containing trace_signature.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    trace_signature = {
        "num_inference_steps": cfg.get("inference_num_steps", cfg.get("generation", {}).get("num_inference_steps", 16) if isinstance(cfg.get("generation"), dict) else 16),
        "guidance_scale": cfg.get("inference_guidance_scale", cfg.get("generation", {}).get("guidance_scale", 7.0) if isinstance(cfg.get("generation"), dict) else 7.0),
        "height": cfg.get("inference_height", cfg.get("model", {}).get("height", 512) if isinstance(cfg.get("model"), dict) else 512),
        "width": cfg.get("inference_width", cfg.get("model", {}).get("width", 512) if isinstance(cfg.get("model"), dict) else 512),
    }
    return {
        "trace_signature": trace_signature
    }
