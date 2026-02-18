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
    
    fusion_result = impl_set.fusion_rule.fuse(cfg, content_evidence_adapted, geometry_evidence_adapted)
    input_fields = len(input_record or {})

    # plan_digest 一致性验证（可选）。
    plan_digest_status = "not_validated"
    plan_digest_mismatch_reason = None
    
    # 仅当 input_record 包含 plan_digest 且 cfg_digest 可用时，才进行验证。
    if input_record and cfg_digest:
        embed_time_plan_digest = input_record.get("plan_digest")
        if embed_time_plan_digest is not None:
            # 在 detect 侧重新计算 plan_digest（使用相同的 subspace_planner）。
            # 提取 mask_digest（detect 时重新计算的 mask）。
            mask_digest = None
            if content_evidence_payload:
                mask_digest = content_evidence_payload.get("mask_digest")
            
            detect_time_plan_result = impl_set.subspace_planner.plan(
                cfg,
                mask_digest=mask_digest,
                cfg_digest=cfg_digest
            )
            
            # 对比两个 plan_digest。
            detect_time_plan_digest = None
            if hasattr(detect_time_plan_result, "plan_digest"):
                detect_time_plan_digest = detect_time_plan_result.plan_digest
            
            if detect_time_plan_digest is not None:
                if detect_time_plan_digest == embed_time_plan_digest:
                    plan_digest_status = "ok"
                else:
                    # plan_digest 不一致：可能是 cfg_digest 变化或其他参数变化。
                    plan_digest_status = "mismatch"
                    plan_digest_mismatch_reason = "plan_digest_mismatch"
            else:
                # Detect 侧计算失败，无法进行对比。
                plan_digest_status = "compute_failed"
    
    record: Dict[str, Any] = {
        "operation": "detect",
        "detect_placeholder": True,
        "image_path": "placeholder_test.png",
        "score": getattr(fusion_result, "evidence_summary", {}).get("content_score"),
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
        "input_record_fields": input_fields,
        "plan_digest_validation_status": plan_digest_status,
        "plan_digest_mismatch_reason": plan_digest_mismatch_reason,
        # (append-only) 保留完整的 payload，供后续升级 fusion 规则时直接消费冻结字段。
        "content_evidence_payload": content_evidence_payload,
        "geometry_evidence_payload": geometry_evidence_payload,
        "content_result": content_result,
        "geometry_result": geometry_result,
        "fusion_result": fusion_result
    }
    return record


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
