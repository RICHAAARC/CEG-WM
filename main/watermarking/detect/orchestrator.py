"""
全流程运行期编排

功能说明：
- 检测流程的运行期编排，负责调用注入的实现来完成业务流程。
- 评估流程的运行期编排，负责调用注入的实现来完成业务流程。
- 校准流程的运行期编排，负责调用注入的实现来完成业务流程。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from main.registries.runtime_resolver import BuiltImplSet


def run_detect_orchestrator(
    cfg: Dict[str, Any],
    impl_set: BuiltImplSet,
    input_record: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    功能：执行检测占位流程。

    Execute detect placeholder flow using injected implementations.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.
        input_record: Optional input record mapping.

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

    content_result = impl_set.content_extractor.extract(cfg)
    geometry_result = impl_set.geometry_extractor.extract(cfg)
    fusion_result = impl_set.fusion_rule.fuse(content_result, geometry_result, cfg)
    input_fields = len(input_record or {})

    record: Dict[str, Any] = {
        "operation": "detect",
        "detect_placeholder": True,
        "image_path": "placeholder_test.png",
        "score": fusion_result.get("score", 0.0),
        "decision": {
            "is_watermarked": False
        },
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
        "input_record_fields": input_fields,
        "content_result": content_result,
        "geometry_result": geometry_result,
        "fusion_result": fusion_result
    }
    threshold_value = fusion_result.get("threshold")
    if threshold_value is not None:
        record["threshold"] = threshold_value
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
        "decision": {
            "is_watermarked": False
        },
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
    fusion_result = impl_set.fusion_rule.fuse(content_result, geometry_result, cfg)

    record: Dict[str, Any] = {
        "operation": "evaluate",
        "evaluation_placeholder": True,
        "decision": {
            "is_watermarked": False
        },
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
        "content_result": content_result,
        "geometry_result": geometry_result,
        "fusion_result": fusion_result
    }
    return record
