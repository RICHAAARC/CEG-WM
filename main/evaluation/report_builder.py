"""
File purpose: 评测报告生成与锚点字段组装。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from main.core import digests
from main.core import records_io


def build_evaluation_report(
    cfg_digest: Optional[str],
    plan_digest: Optional[str],
    thresholds_digest: Optional[str],
    threshold_metadata_digest: Optional[str],
    impl_digest: Optional[str],
    fusion_rule_version: Optional[str],
    attack_protocol_version: Optional[str],
    attack_protocol_digest: Optional[str],
    policy_path: Optional[str],
    metrics_overall: Dict[str, Any],
    metrics_by_attack_condition: List[Dict[str, Any]],
    thresholds_artifact: Optional[Dict[str, Any]] = None,
    attack_protocol_spec: Optional[Dict[str, Any]] = None,
    strict_anchor_validation: bool = True,
) -> Dict[str, Any]:
    """
    功能：组装完整的评测报告对象（包含所有锚点字段）。强制校验锚点字段完整性。

    Build complete evaluation report with all anchor fields required for auditability.
    Report contains overall metrics and per-condition grouped metrics.
    Enforces that all required anchor fields are present and non-empty.

    Args:
        cfg_digest: Config digest (canonical SHA256). Must be non-empty string.
        plan_digest: Generate-side or detect-side plan digest. Must be non-empty string.
        thresholds_digest: Thresholds artifact digest. Must be non-empty string.
        threshold_metadata_digest: Threshold metadata artifact digest. Must be non-empty string.
        impl_digest: Implementation digest (content/geometry/fusion). Must be non-empty string.
        fusion_rule_version: Fusion rule version string. Must be non-empty string.
        attack_protocol_version: Attack protocol version. Must be non-empty string.
        attack_protocol_digest: Attack protocol canonical digest. Must be non-empty string.
        policy_path: Policy path identifier. Must be non-empty string.
        metrics_overall: Overall metrics dict.
        metrics_by_attack_condition: List of per-condition metric dicts.
        thresholds_artifact: Optional thresholds artifact for reference.
        attack_protocol_spec: Optional attack protocol spec (for backward compatibility).
        strict_anchor_validation: If True, FAIL on missing anchor fields. If False, use "<absent>" defaults.

    Returns:
        Report dict with structure:
        {
            "evaluation_version": ...,
            "cfg_digest": ...,
            "plan_digest": ...,
            "thresholds_digest": ...,
            "threshold_metadata_digest": ...,
            "impl_digest": ...,
            "fusion_rule_version": ...,
            "attack_protocol_version": ...,
            "attack_protocol_digest": ...,
            "policy_path": ...,
            "metrics": {...},
            "metrics_by_attack_condition": [...],
            "anchors": {...}
        }

    Raises:
        TypeError: If required inputs are invalid.
        RuntimeError: If strict_anchor_validation=True and any anchor field is missing or empty.
    """
    if not isinstance(metrics_overall, dict):
        raise TypeError("metrics_overall must be dict")
    if not isinstance(metrics_by_attack_condition, list):
        raise TypeError("metrics_by_attack_condition must be list")

    # 当启用严格验证时，检查所有锚点字段必须非空。
    if strict_anchor_validation:
        missing_anchors = []
        
        if not isinstance(cfg_digest, str) or not cfg_digest.strip():
            missing_anchors.append(("cfg_digest", "config digest from cfg compute or passthrough"))
        if not isinstance(plan_digest, str) or not plan_digest.strip():
            missing_anchors.append(("plan_digest", "attack plan digest from plan generation"))
        if not isinstance(thresholds_digest, str) or not thresholds_digest.strip():
            missing_anchors.append(("thresholds_digest", "thresholds artifact digest"))
        if not isinstance(threshold_metadata_digest, str) or not threshold_metadata_digest.strip():
            missing_anchors.append(("threshold_metadata_digest", "threshold metadata artifact digest"))
        if not isinstance(impl_digest, str) or not impl_digest.strip():
            missing_anchors.append(("impl_digest", "implementation digest from config/orchestrator"))
        if not isinstance(fusion_rule_version, str) or not fusion_rule_version.strip():
            missing_anchors.append(("fusion_rule_version", "fusion rule version from config"))
        if not isinstance(attack_protocol_version, str) or not attack_protocol_version.strip():
            missing_anchors.append(("attack_protocol_version", "attack protocol version from configs/attack_protocol.yaml"))
        if not isinstance(attack_protocol_digest, str) or not attack_protocol_digest.strip():
            missing_anchors.append(("attack_protocol_digest", "attack protocol canonical digest"))
        if not isinstance(policy_path, str) or not policy_path.strip():
            missing_anchors.append(("policy_path", "policy path identifier"))
        
        if missing_anchors:
            error_msg_parts = ["Evaluation report anchor field validation FAILED:"]
            for field_name, source_description in missing_anchors:
                error_msg_parts.append(f"  - {field_name} (source: {source_description}) is missing or empty")
            raise RuntimeError("\n".join(error_msg_parts))

    # 当禁用严格验证时，使用 <absent> 作为默认值（向后兼容）。
    safe_cfg_digest = cfg_digest if isinstance(cfg_digest, str) and cfg_digest.strip() else "<absent>"
    safe_plan_digest = plan_digest if isinstance(plan_digest, str) and plan_digest.strip() else "<absent>"
    safe_thresholds_digest = (
        thresholds_digest if isinstance(thresholds_digest, str) and thresholds_digest.strip() else "<absent>"
    )
    safe_threshold_metadata_digest = (
        threshold_metadata_digest
        if isinstance(threshold_metadata_digest, str) and threshold_metadata_digest.strip()
        else "<absent>"
    )
    safe_impl_digest = impl_digest if isinstance(impl_digest, str) and impl_digest.strip() else "<absent>"
    safe_fusion_rule_version = (
        fusion_rule_version if isinstance(fusion_rule_version, str) and fusion_rule_version.strip() else "<absent>"
    )
    safe_attack_protocol_version = (
        attack_protocol_version if isinstance(attack_protocol_version, str) and attack_protocol_version.strip() else "<absent>"
    )
    safe_attack_protocol_digest = (
        attack_protocol_digest if isinstance(attack_protocol_digest, str) and attack_protocol_digest.strip() else "<absent>"
    )
    safe_policy_path = (
        policy_path if isinstance(policy_path, str) and policy_path.strip() else "<absent>"
    )

    # 计算 metrics 和条件指标的 digest（用于审计）。
    metrics_digest = digests.canonical_sha256(metrics_overall)
    metrics_by_condition_digest = digests.canonical_sha256(
        sorted(metrics_by_attack_condition, key=lambda x: x.get("group_key", ""))
    )

    # 组装报告（append-only 扩展原则）。
    report = {
        "evaluation_version": "eval_v1",
        # 锚点字段（必须存在）。
        "cfg_digest": safe_cfg_digest,
        "plan_digest": safe_plan_digest,
        "thresholds_digest": safe_thresholds_digest,
        "threshold_metadata_digest": safe_threshold_metadata_digest,
        "impl_digest": safe_impl_digest,
        "fusion_rule_version": safe_fusion_rule_version,
        "attack_protocol_version": safe_attack_protocol_version,
        "attack_protocol_digest": safe_attack_protocol_digest,
        "policy_path": safe_policy_path,
        # 指标数据。
        "metrics": metrics_overall,
        "metrics_by_attack_condition": metrics_by_attack_condition,
        # 审计用 digest。
        "metrics_digest": metrics_digest,
        "metrics_by_condition_digest": metrics_by_condition_digest,
    }

    # 构造锚点物业摘要（用于 run_closure）。
    anchors = {
        "cfg_digest": safe_cfg_digest,
        "plan_digest": safe_plan_digest,
        "thresholds_digest": safe_thresholds_digest,
        "threshold_metadata_digest": safe_threshold_metadata_digest,
        "attack_protocol_version": safe_attack_protocol_version,
        "attack_protocol_digest": safe_attack_protocol_digest,
        "policy_path": safe_policy_path,
        "impl_digest": safe_impl_digest,
        "fusion_rule_version": safe_fusion_rule_version,
    }
    report["anchors"] = anchors

    # 如果提供了 thresholds_artifact，将关键字段持久化（用于后续复核）。
    if isinstance(thresholds_artifact, dict):
        report["thresholds_artifact_metadata"] = {
            "threshold_id": thresholds_artifact.get("threshold_id", "<absent>"),
            "score_name": thresholds_artifact.get("score_name", "<absent>"),
            "target_fpr": thresholds_artifact.get("target_fpr", "<absent>"),
            "threshold_value": thresholds_artifact.get("threshold_value", "<absent>"),
            "threshold_key_used": thresholds_artifact.get("threshold_key_used", "<absent>"),
        }

    # (向后兼容) 若提供了 attack_protocol_spec，保留整个对象供老代码使用。
    if isinstance(attack_protocol_spec, dict):
        report["attack_protocol"] = attack_protocol_spec

    # append-only: 路由摘要字段（若可从指标中推断则保留）。
    report["routing_decisions"] = {
        "attack_protocol_version": safe_attack_protocol_version,
        "policy_path": safe_policy_path,
        "rescue_rate": metrics_overall.get("rescue_rate") if isinstance(metrics_overall, dict) else None,
        "geo_available_rate": metrics_overall.get("geo_available_rate") if isinstance(metrics_overall, dict) else None,
    }
    report["routing_digest"] = digests.canonical_sha256(report["routing_decisions"])

    # append-only: impl 锚点容器（兼容 impl_digest 单字段）。
    report["impl_anchors"] = {
        "content": {
            "impl_digest": safe_impl_digest,
        },
        "geometry": {
            "impl_digest": safe_impl_digest,
        },
        "fusion": {
            "impl_digest": safe_impl_digest,
            "fusion_rule_version": safe_fusion_rule_version,
        },
    }

    return report


def build_eval_report(**kwargs: Any) -> Dict[str, Any]:
    """
    功能：构造 evaluate 报告（build_evaluation_report 的别名入口）。

    Build evaluation report via canonical report builder.

    Args:
        **kwargs: Same keyword arguments accepted by build_evaluation_report.

    Returns:
        Evaluation report mapping.
    """
    return build_evaluation_report(**kwargs)


def write_eval_report_via_records_io(
    eval_report: Dict[str, Any],
    output_path: str,
) -> None:
    """
    功能：通过 records_io 写入评测报告工件。

    Write evaluation report artifact through records_io controlled path.

    Args:
        eval_report: Evaluation report mapping.
        output_path: Artifact path under artifacts directory.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(eval_report, dict):
        raise TypeError("eval_report must be dict")
    if not isinstance(output_path, str) or not output_path:
        raise TypeError("output_path must be non-empty str")

    # artifacts 语义旁路防护：将报告作为业务载荷嵌套，避免顶层 records anchor 冲突。
    payload = {
        "evaluation_report": eval_report,
        "report_type": "eval_report_v1",
        "report_digest": digests.canonical_sha256(eval_report),
    }
    records_io.write_artifact_json(output_path, payload)


def build_conditional_metrics_container(
    attack_protocol_version: str,
    attack_group_metrics: List[Dict[str, Any]],
    additional_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    功能：为条件指标构造标准化容器（向后兼容旧字段）。

    Build container for conditional metrics with version and audit info.

    Args:
        attack_protocol_version: Attack protocol version.
        attack_group_metrics: List of grouped metric dicts.
        additional_items: Optional list of additional items for backward compatibility.

    Returns:
        Conditional metrics container dict.
    """
    if additional_items is None:
        additional_items = []
    
    return {
        "version": "conditional_eval_v1",
        "attack_protocol_version": attack_protocol_version,
        "attack_group_metrics": attack_group_metrics,
        "items": additional_items,  # 保留用于向后兼容。
    }
