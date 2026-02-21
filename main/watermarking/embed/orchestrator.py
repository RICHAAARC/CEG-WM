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
from main.core import digests
from main.watermarking.content_chain.high_freq_embedder import (
    HighFreqEmbedder,
    HIGH_FREQ_EMBEDDER_ID,
    HIGH_FREQ_EMBEDDER_VERSION,
)


def run_embed_orchestrator(
    cfg: Dict[str, Any],
    impl_set: BuiltImplSet,
    cfg_digest: str,
    *,
    trajectory_evidence: Dict[str, Any] | None = None,
    injection_evidence: Dict[str, Any] | None = None,
    content_result_override: Any | None = None,
    subspace_result_override: Any | None = None
) -> Dict[str, Any]:
    """
    功能：执行嵌入占位流程。

    Execute embed placeholder flow using injected implementations.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.
        cfg_digest: Canonical SHA256 digest of cfg (computed from include_paths).
                   Passed to content_extractor to bind mask_digest to authoritative digest.
        trajectory_evidence: Optional trajectory tap evidence mapping.
        injection_evidence: Optional injection evidence mapping.
        content_result_override: Optional precomputed content result.
        subspace_result_override: Optional precomputed subspace plan result.

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
    if trajectory_evidence is not None and not isinstance(trajectory_evidence, dict):
        # trajectory_evidence 类型不符合预期，必须 fail-fast。
        raise TypeError("trajectory_evidence must be dict or None")
    if injection_evidence is not None and not isinstance(injection_evidence, dict):
        # injection_evidence 类型不符合预期，必须 fail-fast。
        raise TypeError("injection_evidence must be dict or None")

    if content_result_override is not None and not isinstance(content_result_override, dict) and not hasattr(content_result_override, "as_dict"):
        # content_result_override 类型不符合预期，必须 fail-fast。
        raise TypeError("content_result_override must be dict, ContentEvidence, or None")
    if subspace_result_override is not None and not isinstance(subspace_result_override, dict) and not hasattr(subspace_result_override, "as_dict"):
        # subspace_result_override 类型不符合预期，必须 fail-fast。
        raise TypeError("subspace_result_override must be dict, SubspacePlan, or None")

    content_result = content_result_override if content_result_override is not None else impl_set.content_extractor.extract(cfg, cfg_digest=cfg_digest)

    content_evidence_payload = None
    if hasattr(content_result, "as_dict") and callable(content_result.as_dict):
        content_evidence_payload = content_result.as_dict()
    elif isinstance(content_result, dict):
        content_evidence_payload = dict(content_result)
    else:
        # content_result 类型不符合预期，必须 fail-fast。
        raise TypeError(f"content_result must be dict or have as_dict method, got {type(content_result).__name__}")

    if trajectory_evidence is not None:
        if content_evidence_payload is None:
            content_evidence_payload = {}
        content_evidence_payload["trajectory_evidence"] = trajectory_evidence
        _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
    if injection_evidence is not None:
        if content_evidence_payload is None:
            content_evidence_payload = {}
        _merge_injection_evidence(content_evidence_payload, injection_evidence)
    
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
    planner_inputs = _build_planner_inputs_for_runtime(cfg, trajectory_evidence)
    subspace_result = subspace_result_override if subspace_result_override is not None else impl_set.subspace_planner.plan(
        cfg,
        mask_digest=mask_digest,
        cfg_digest=cfg_digest,
        inputs=planner_inputs
    )
    
    sync_result = impl_set.sync_module.sync(cfg)

    hf_evidence = _build_hf_embed_evidence(
        cfg=cfg,
        cfg_digest=cfg_digest,
        subspace_result=subspace_result,
        trajectory_evidence=trajectory_evidence,
    )
    if content_evidence_payload is None:
        content_evidence_payload = {}
    _merge_hf_evidence(content_evidence_payload, hf_evidence)

    # 构造返回的业务字段映射。
    # 关键：plan_digest 作为锚点字段必须同时返回，以供后续 detect 侧验证。
    record_fields = {
        "operation": "embed",
        "embed_placeholder": True,
        "image_path": "placeholder_input.png",
        "watermarked_path": "placeholder_output.png",
        "seed": 42,
        "strength": 0.5,
        "content_result": content_evidence_payload,
        "content_evidence": content_evidence_payload,
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


def _build_hf_embed_evidence(
    cfg: Dict[str, Any],
    cfg_digest: str,
    subspace_result: Any,
    trajectory_evidence: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    功能：构造 embed 侧 HF 证据。

    Build embed-side HF evidence bound to planner-defined subspace.

    Args:
        cfg: Configuration mapping.
        cfg_digest: Config digest.
        subspace_result: Planner result object or mapping.
        trajectory_evidence: Optional trajectory evidence mapping.

    Returns:
        HF evidence mapping.

    Raises:
        TypeError: If cfg/cfg_digest types are invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(cfg_digest, str) or not cfg_digest:
        raise TypeError("cfg_digest must be non-empty str")

    plan_payload = None
    plan_digest = None
    if hasattr(subspace_result, "as_dict") and callable(subspace_result.as_dict):
        plan_payload = subspace_result.as_dict()
    elif isinstance(subspace_result, dict):
        plan_payload = dict(subspace_result)

    if hasattr(subspace_result, "plan_digest"):
        plan_digest = subspace_result.plan_digest
    elif isinstance(plan_payload, dict):
        plan_digest = plan_payload.get("plan_digest")

    embedder = HighFreqEmbedder(
        impl_id=HIGH_FREQ_EMBEDDER_ID,
        impl_version=HIGH_FREQ_EMBEDDER_VERSION,
        impl_digest=digests.canonical_sha256(
            {
                "impl_id": HIGH_FREQ_EMBEDDER_ID,
                "impl_version": HIGH_FREQ_EMBEDDER_VERSION,
            }
        ),
    )
    return embedder.embed(
        latents=trajectory_evidence,
        plan=plan_payload,
        cfg=cfg,
        cfg_digest=cfg_digest,
        expected_plan_digest=plan_digest,
    )


def _merge_hf_evidence(content_evidence_payload: Dict[str, Any], hf_evidence: Dict[str, Any]) -> None:
    """
    功能：将 HF 证据写入 content_evidence 兼容字段。

    Merge HF evidence into content evidence payload using append-only compatible fields.

    Args:
        content_evidence_payload: Mutable content evidence mapping.
        hf_evidence: HF evidence mapping.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return
    if not isinstance(hf_evidence, dict):
        return

    content_evidence_payload["hf_trace_digest"] = hf_evidence.get("hf_trace_digest")
    content_evidence_payload["hf_score"] = hf_evidence.get("hf_score")

    score_parts = content_evidence_payload.get("score_parts")
    if not isinstance(score_parts, dict):
        score_parts = {}
        content_evidence_payload["score_parts"] = score_parts

    score_parts["hf_status"] = hf_evidence.get("status")
    summary = hf_evidence.get("hf_evidence_summary")
    if isinstance(summary, dict):
        if "hf_absent_reason" in summary:
            score_parts["hf_absent_reason"] = summary.get("hf_absent_reason")
        if "hf_failure_reason" in summary:
            score_parts["hf_failure_reason"] = summary.get("hf_failure_reason")
        score_parts["hf_metrics"] = summary
    score_parts["content_score_rule_version"] = hf_evidence.get("content_score_rule_version")


def _merge_injection_evidence(content_evidence_payload: Dict[str, Any], injection_evidence: Dict[str, Any]) -> None:
    """
    功能：将注入证据写入 content_evidence 兼容字段。
    
    Merge injection evidence into content evidence payload using append-only fields.

    Args:
        content_evidence_payload: Mutable content evidence mapping.
        injection_evidence: Injection evidence mapping.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return
    if not isinstance(injection_evidence, dict):
        return

    content_evidence_payload["injection_status"] = injection_evidence.get("status")
    content_evidence_payload["injection_absent_reason"] = injection_evidence.get("injection_absent_reason")
    content_evidence_payload["injection_failure_reason"] = injection_evidence.get("injection_failure_reason")
    content_evidence_payload["injection_trace_digest"] = injection_evidence.get("injection_trace_digest")
    content_evidence_payload["injection_params_digest"] = injection_evidence.get("injection_params_digest")
    content_evidence_payload["injection_metrics"] = injection_evidence.get("injection_metrics")
    content_evidence_payload["subspace_binding_digest"] = injection_evidence.get("subspace_binding_digest")


def _inject_trajectory_audit_fields(
    content_evidence_payload: Dict[str, Any],
    trajectory_evidence: Dict[str, Any]
) -> None:
    """
    功能：将轨迹 tap 子状态写入 content_evidence.audit（兼容新旧字段）。

    Inject trajectory tap status fields into content_evidence.audit.

    Args:
        content_evidence_payload: Content evidence payload mapping.
        trajectory_evidence: Trajectory evidence mapping.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return
    if not isinstance(trajectory_evidence, dict):
        return

    audit = content_evidence_payload.get("audit")
    if not isinstance(audit, dict):
        audit = {}
        content_evidence_payload["audit"] = audit

    tap_audit = trajectory_evidence.get("audit")
    tap_status = None
    tap_absent_reason = None
    if isinstance(tap_audit, dict):
        tap_status = tap_audit.get("trajectory_tap_status")
        tap_absent_reason = tap_audit.get("trajectory_absent_reason")

    if not isinstance(tap_status, str) or not tap_status:
        status_value = trajectory_evidence.get("status")
        if isinstance(status_value, str) and status_value:
            tap_status = status_value

    if not isinstance(tap_absent_reason, str) or not tap_absent_reason:
        reason_value = trajectory_evidence.get("trajectory_absent_reason")
        if isinstance(reason_value, str) and reason_value:
            tap_absent_reason = reason_value

    if isinstance(tap_status, str) and tap_status:
        audit["trajectory_tap_status"] = tap_status
    if isinstance(tap_absent_reason, str) and tap_absent_reason:
        audit["trajectory_absent_reason"] = tap_absent_reason


def _build_planner_inputs_for_runtime(
    cfg: Dict[str, Any],
    trajectory_evidence: Dict[str, Any] | None
) -> Dict[str, Any]:
    """
    功能：构造规划器输入签名。

    Build deterministic planner input signature from cfg runtime fields.

    Args:
        cfg: Configuration mapping.
        trajectory_evidence: Optional trajectory tap evidence.

    Returns:
        Planner input mapping containing trace_signature.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if trajectory_evidence is not None and not isinstance(trajectory_evidence, dict):
        raise TypeError("trajectory_evidence must be dict or None")

    trace_signature = {
        "num_inference_steps": cfg.get("inference_num_steps", cfg.get("generation", {}).get("num_inference_steps", 16) if isinstance(cfg.get("generation"), dict) else 16),
        "guidance_scale": cfg.get("inference_guidance_scale", cfg.get("generation", {}).get("guidance_scale", 7.0) if isinstance(cfg.get("generation"), dict) else 7.0),
        "height": cfg.get("inference_height", cfg.get("model", {}).get("height", 512) if isinstance(cfg.get("model"), dict) else 512),
        "width": cfg.get("inference_width", cfg.get("model", {}).get("width", 512) if isinstance(cfg.get("model"), dict) else 512),
    }
    inputs = {"trace_signature": trace_signature}
    if trajectory_evidence is not None:
        inputs["trajectory_evidence"] = trajectory_evidence
    return inputs
