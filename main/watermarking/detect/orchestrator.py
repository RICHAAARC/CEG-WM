"""
检测、评估与校准编排

功能说明：
- 定义了检测、评估与校准的编排器函数，用于协调不同组件的执行流程。
- 每个编排器函数都接受配置和实现集作为输入，并返回包含业务字段的记录映射。
- 实现了输入验证和错误处理，确保接口的健壮性。
- 内容证据与几何证据数据类转换为适配字典，确保向下兼容性。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from main.registries.runtime_resolver import BuiltImplSet
from main.core import digests
from main.watermarking.fusion.interfaces import FusionDecision
from main.watermarking.fusion import neyman_pearson


def run_detect_orchestrator(
    cfg: Dict[str, Any],
    impl_set: BuiltImplSet,
    input_record: Optional[Dict[str, Any]] = None,
    cfg_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：执行检测占位流程，包括 plan_digest 一致性验证。

    Execute detect placeholder flow using injected implementations.
    Validates plan_digest consistency with embed-time plan_digest when available.

    Args:
        cfg: Config mapping (may differ from embed-time cfg).
        impl_set: Built implementation set.
        input_record: Optional input record mapping (contains embed-time plan_digest).
        cfg_digest: Optional cfg digest for detect-time cfg.
                   If None, plan_digest validation is skipped.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_set, BuiltImplSet):
        # impl_set 类型不合法，必须 fail-fast。
        raise TypeError("impl_set must be BuiltImplSet")
    if input_record is not None and not isinstance(input_record, dict):
        # input_record 类型不合法，必须 fail-fast。
        raise TypeError("input_record must be dict or None")
    if cfg_digest is not None and not isinstance(cfg_digest, str):
        # cfg_digest 类型不合法，必须 fail-fast。
        raise TypeError("cfg_digest must be str or None")

    content_result = impl_set.content_extractor.extract(cfg)
    geometry_result = impl_set.geometry_extractor.extract(cfg)
    
    # (1) 统一转换 ContentEvidence / GeometryEvidence 数据类为 dict。
    # 优先使用 .as_dict() 方法；若不存在则直接使用数据类或字典。
    content_evidence_payload = None
    if hasattr(content_result, "as_dict") and callable(content_result.as_dict):
        content_evidence_payload = content_result.as_dict()
    elif isinstance(content_result, dict):
        content_evidence_payload = content_result
    
    geometry_evidence_payload = None
    if hasattr(geometry_result, "as_dict") and callable(geometry_result.as_dict):
        geometry_evidence_payload = geometry_result.as_dict()
    elif isinstance(geometry_result, dict):
        geometry_evidence_payload = geometry_result
    
    # (2) 构造融合输入适配 dict，兼容 FusionBaselineIdentity 的旧字段读取逻辑。
    # 优先从 .as_dict() 结果中读取，但为向后兼容也检查数据类属性。
    content_evidence_adapted = _adapt_content_evidence_for_fusion(content_result)
    geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
    
    planner_inputs = _build_planner_inputs_for_runtime(cfg)
    mask_digest = None
    if isinstance(content_evidence_payload, dict):
        mask_digest = content_evidence_payload.get("mask_digest")

    detect_plan_result = impl_set.subspace_planner.plan(
        cfg,
        mask_digest=mask_digest,
        cfg_digest=cfg_digest,
        inputs=planner_inputs
    )

    embed_time_plan_digest = None
    embed_time_basis_digest = None
    embed_time_planner_impl_identity = None
    if isinstance(input_record, dict):
        embed_time_plan_digest = input_record.get("plan_digest")
        embed_time_basis_digest = input_record.get("basis_digest")
        embed_time_planner_impl_identity = input_record.get("subspace_planner_impl_identity")

    detect_time_plan_digest = getattr(detect_plan_result, "plan_digest", None)
    detect_time_basis_digest = getattr(detect_plan_result, "basis_digest", None)
    detect_time_planner_impl_identity = None
    if hasattr(detect_plan_result, "plan") and isinstance(detect_plan_result.plan, dict):
        detect_time_planner_impl_identity = detect_plan_result.plan.get("planner_impl_identity")

    mismatch_reasons = _collect_plan_mismatch_reasons(
        embed_time_plan_digest=embed_time_plan_digest,
        detect_time_plan_digest=detect_time_plan_digest,
        embed_time_basis_digest=embed_time_basis_digest,
        detect_time_basis_digest=detect_time_basis_digest,
        embed_time_planner_impl_identity=embed_time_planner_impl_identity,
        detect_time_planner_impl_identity=detect_time_planner_impl_identity
    )
    primary_mismatch_reason, primary_mismatch_field_path = _resolve_primary_mismatch(
        mismatch_reasons
    )

    forced_mismatch = len(mismatch_reasons) > 0
    if forced_mismatch:
        content_evidence_payload = {
            "status": "mismatch",
            "score": None,
            "plan_digest": detect_time_plan_digest,
            "basis_digest": detect_time_basis_digest,
            "content_failure_reason": "detector_plan_mismatch",
            "content_mismatch_reason": primary_mismatch_reason,
            "content_mismatch_field_path": primary_mismatch_field_path,
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "audit": {
                "impl_identity": "detect_orchestrator",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"impl_id": "detect_orchestrator", "impl_version": "v1"}),
                "trace_digest": digests.canonical_sha256({
                    "mismatch_reasons": mismatch_reasons,
                    "primary_mismatch_reason": primary_mismatch_reason,
                    "primary_mismatch_field_path": primary_mismatch_field_path
                })
            }
        }
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_mismatch_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    else:
        fusion_result = impl_set.fusion_rule.fuse(cfg, content_evidence_adapted, geometry_evidence_adapted)
    input_fields = len(input_record or {})

    # plan_digest 一致性验证。
    plan_digest_status = "not_validated"
    plan_digest_mismatch_reason = None
    
    # 仅当 input_record 包含 plan_digest 且 cfg_digest 可用时，才进行验证。
    if input_record and cfg_digest:
        embed_time_plan_digest = input_record.get("plan_digest")
        if embed_time_plan_digest is not None:
            if detect_time_plan_digest is not None:
                if detect_time_plan_digest == embed_time_plan_digest:
                    plan_digest_status = "ok"
                else:
                    plan_digest_status = "mismatch"
                    plan_digest_mismatch_reason = "plan_digest_mismatch"
            else:
                plan_digest_status = "compute_failed"
    
    record: Dict[str, Any] = {
        "operation": "detect",
        "detect_placeholder": True,
        "image_path": "placeholder_test.png",
        "score": getattr(fusion_result, "evidence_summary", {}).get("content_score"),
        "execution_report": {
            "content_chain_status": "fail" if forced_mismatch else "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "fail" if forced_mismatch else "ok",
            "audit_obligations_satisfied": True
        },
        "input_record_fields": input_fields,
        "plan_digest_validation_status": plan_digest_status,
        "plan_digest_mismatch_reason": primary_mismatch_reason if forced_mismatch else plan_digest_mismatch_reason,
        # (append-only) 保留完整的 payload，供后续升级 fusion 规则时直接消费冻结字段。
        "content_evidence_payload": content_evidence_payload,
        "geometry_evidence_payload": geometry_evidence_payload,
        "content_result": content_result,
        "geometry_result": geometry_result,
        "fusion_result": fusion_result
    }
    return record


def _collect_plan_mismatch_reasons(
    embed_time_plan_digest: Any,
    detect_time_plan_digest: Any,
    embed_time_basis_digest: Any,
    detect_time_basis_digest: Any,
    embed_time_planner_impl_identity: Any,
    detect_time_planner_impl_identity: Any
) -> list[str]:
    """
    功能：收集计划锚点不一致原因。

    Collect mismatch reasons for plan/basis/impl identity anchors.

    Args:
        embed_time_plan_digest: Embed-time plan digest.
        detect_time_plan_digest: Detect-time recomputed plan digest.
        embed_time_basis_digest: Embed-time basis digest.
        detect_time_basis_digest: Detect-time recomputed basis digest.
        embed_time_planner_impl_identity: Embed-time planner impl identity payload.
        detect_time_planner_impl_identity: Detect-time planner impl identity payload.

    Returns:
        List of mismatch reason tokens.
    """
    reasons: list[str] = []
    if isinstance(embed_time_plan_digest, str) and isinstance(detect_time_plan_digest, str):
        if embed_time_plan_digest != detect_time_plan_digest:
            reasons.append("plan_digest_mismatch")
    if isinstance(embed_time_basis_digest, str) and isinstance(detect_time_basis_digest, str):
        if embed_time_basis_digest != detect_time_basis_digest:
            reasons.append("basis_digest_mismatch")
    if isinstance(embed_time_planner_impl_identity, dict) and isinstance(detect_time_planner_impl_identity, dict):
        if embed_time_planner_impl_identity != detect_time_planner_impl_identity:
            reasons.append("planner_impl_identity_mismatch")
    return reasons


def _resolve_primary_mismatch(mismatch_reasons: list[str]) -> tuple[str, str]:
    """
    功能：选择单一主 mismatch 原因并返回对应字段路径。

    Resolve a single primary mismatch reason and its field path.

    Args:
        mismatch_reasons: Collected mismatch reason tokens.

    Returns:
        Tuple of (primary_reason, field_path).
    """
    reason_to_field_path = {
        "plan_digest_mismatch": "content_evidence.plan_digest",
        "basis_digest_mismatch": "content_evidence.basis_digest",
        "planner_impl_identity_mismatch": "content_evidence.planner_impl_identity"
    }
    for token in [
        "plan_digest_mismatch",
        "basis_digest_mismatch",
        "planner_impl_identity_mismatch"
    ]:
        if token in mismatch_reasons:
            return token, reason_to_field_path[token]
    return "unknown_mismatch", "content_evidence"


def _build_mismatch_fusion_decision(
    cfg: Dict[str, Any],
    content_evidence_adapted: Dict[str, Any],
    geometry_evidence_adapted: Dict[str, Any]
) -> FusionDecision:
    """
    功能：构造 mismatch 的融合失败判决。

    Build a single-path FusionDecision for mismatch failures.

    Args:
        cfg: Configuration mapping.
        content_evidence_adapted: Adapted content evidence mapping.
        geometry_evidence_adapted: Adapted geometry evidence mapping.

    Returns:
        FusionDecision with decision_status="error" and score-free evidence summary.
    """
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)

    evidence_summary = {
        "content_score": None,
        "geometry_score": geometry_evidence_adapted.get("geo_score"),
        "content_status": "mismatch",
        "geometry_status": geometry_evidence_adapted.get("status", "absent"),
        "fusion_rule_id": "detect_mismatch_guard_v1"
    }
    audit = {
        "guard": "plan_anchor_consistency",
        "reason": content_evidence_adapted.get("content_failure_reason", "detector_plan_mismatch")
    }
    return FusionDecision(
        is_watermarked=None,
        decision_status="error",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary,
        audit=audit
    )


def _build_planner_inputs_for_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造规划器输入签名。

    Build deterministic planner input signature from runtime cfg.

    Args:
        cfg: Configuration mapping.

    Returns:
        Planner input mapping.
    """
    trace_signature = {
        "num_inference_steps": cfg.get("inference_num_steps", cfg.get("generation", {}).get("num_inference_steps", 16) if isinstance(cfg.get("generation"), dict) else 16),
        "guidance_scale": cfg.get("inference_guidance_scale", cfg.get("generation", {}).get("guidance_scale", 7.0) if isinstance(cfg.get("generation"), dict) else 7.0),
        "height": cfg.get("inference_height", cfg.get("model", {}).get("height", 512) if isinstance(cfg.get("model"), dict) else 512),
        "width": cfg.get("inference_width", cfg.get("model", {}).get("width", 512) if isinstance(cfg.get("model"), dict) else 512),
    }
    return {
        "trace_signature": trace_signature
    }


def run_calibrate_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet) -> Dict[str, Any]:
    """
    功能：执行校准占位流程。

    Execute calibration placeholder flow using injected implementations.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_set, BuiltImplSet):
        # impl_set 类型不合法，必须 fail-fast。
        raise TypeError("impl_set must be BuiltImplSet")

    subspace_result = impl_set.subspace_planner.plan(cfg)

    record: Dict[str, Any] = {
        "operation": "calibrate",
        "calibration_placeholder": True,
        "protocol": "neyman_pearson",
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
        "calibration_samples": 1000,
        "subspace_plan": subspace_result
    }
    return record


def run_evaluate_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet) -> Dict[str, Any]:
    """
    功能：执行评估占位流程。

    Execute evaluation placeholder flow using injected implementations.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_set, BuiltImplSet):
        # impl_set 类型不合法，必须 fail-fast。
        raise TypeError("impl_set must be BuiltImplSet")

    content_result = impl_set.content_extractor.extract(cfg)
    geometry_result = impl_set.geometry_extractor.extract(cfg)
    
    # (1) 统一转换 ContentEvidence / GeometryEvidence 数据类为 dict。
    content_evidence_payload = None
    if hasattr(content_result, "as_dict") and callable(content_result.as_dict):
        content_evidence_payload = content_result.as_dict()
    elif isinstance(content_result, dict):
        content_evidence_payload = content_result
    
    geometry_evidence_payload = None
    if hasattr(geometry_result, "as_dict") and callable(geometry_result.as_dict):
        geometry_evidence_payload = geometry_result.as_dict()
    elif isinstance(geometry_result, dict):
        geometry_evidence_payload = geometry_result
    
    # (2) 构造融合输入适配 dict。
    content_evidence_adapted = _adapt_content_evidence_for_fusion(content_result)
    geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
    
    fusion_result = impl_set.fusion_rule.fuse(cfg, content_evidence_adapted, geometry_evidence_adapted)

    record: Dict[str, Any] = {
        "operation": "evaluate",
        "evaluation_placeholder": True,
        "metrics": {
            "tpr": 0.95,
            "fpr": 0.01,
            "accuracy": 0.97
        },
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
        "test_samples": 500,
        # (append-only) 保留完整的 payload。
        "content_evidence_payload": content_evidence_payload,
        "geometry_evidence_payload": geometry_evidence_payload,
        "content_result": content_result,
        "geometry_result": geometry_result,
        "fusion_result": fusion_result
    }
    return record


def _adapt_content_evidence_for_fusion(content_evidence: Any) -> Dict[str, Any]:
    """
    功能：将 ContentEvidence 数据类适配为融合规则期望的字典格式。

    Adapt ContentEvidence (dataclass or dict) to fusion rule expected format.
    Prioritizes .as_dict() method; falls back to direct dict or attribute extraction.

    Args:
        content_evidence: ContentEvidence dataclass, dict, or compatible object.

    Returns:
        Dict with fields expected by fusion rule (status, score, etc.).

    Raises:
        TypeError: If content_evidence type is unrecognized.
    """
    if isinstance(content_evidence, dict):
        # 已是字典，直接返回。
        return content_evidence
    
    # 尝试用 .as_dict() 方法。
    if hasattr(content_evidence, "as_dict") and callable(content_evidence.as_dict):
        try:
            return content_evidence.as_dict()
        except Exception:
            # 如果 .as_dict() 失败，继续尝试属性提取。
            pass
    
    # 从数据类属性直接构造。
    adapted = {}
    
    # 提取关键字段（来自 ContentEvidence 冻结结构）。
    for field_name in ["status", "score", "audit", "mask_digest", "mask_stats",
                       "plan_digest", "basis_digest", "lf_trace_digest", "hf_trace_digest",
                       "lf_score", "hf_score", "score_parts", "content_failure_reason"]:
        if hasattr(content_evidence, field_name):
            adapted[field_name] = getattr(content_evidence, field_name)
    
    return adapted if adapted else {"status": "unknown"}


def _adapt_geometry_evidence_for_fusion(geometry_evidence: Any) -> Dict[str, Any]:
    """
    功能：将 GeometryEvidence 数据类适配为融合规则期望的字典格式。

    Adapt GeometryEvidence (dataclass or dict) to fusion rule expected format.
    Prioritizes .as_dict() method; falls back to direct dict or attribute extraction.

    Args:
        geometry_evidence: GeometryEvidence dataclass, dict, or compatible object.

    Returns:
        Dict with fields expected by fusion rule (status, geo_score, etc.).

    Raises:
        TypeError: If geometry_evidence type is unrecognized.
    """
    if isinstance(geometry_evidence, dict):
        # 已是字典，直接返回。
        return geometry_evidence
    
    # 尝试用 .as_dict() 方法。
    if hasattr(geometry_evidence, "as_dict") and callable(geometry_evidence.as_dict):
        try:
            return geometry_evidence.as_dict()
        except Exception:
            # 如果 .as_dict() 失败，继续尝试属性提取。
            pass
    
    # 从数据类属性直接构造。
    adapted = {}
    
    # 提取关键字段（来自 GeometryEvidence 冻结结构）。
    for field_name in ["status", "geo_score", "audit", "anchor_digest", "align_trace_digest",
                       "align_residuals", "stability_metrics", "geometry_failure_reason"]:
        if hasattr(geometry_evidence, field_name):
            adapted[field_name] = getattr(geometry_evidence, field_name)
    
    return adapted if adapted else {"status": "unknown"}
