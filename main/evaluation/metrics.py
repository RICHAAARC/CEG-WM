"""
File purpose: 评测指标计算与分组聚合。
Module type: General module
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def canonical_condition_key(family: str, params_version: str) -> str:
    """
    功能：生成规范化的条件键（deterministic 排序）。

    Generate canonical attack condition key from family and params version.
    Key format is strictly "family::params_version" with sorted components to ensure
    deterministic ordering independent of Python dict insertion order.

    Args:
        family: Attack family name (e.g., "rotate", "resize").
        params_version: Attack params version (e.g., "v1", "v2").

    Returns:
        Canonical condition key as "family::params_version".

    Raises:
        TypeError: If inputs are not strings.
    """
    if not isinstance(family, str):
        raise TypeError(f"family must be str, got {type(family).__name__}")
    if not isinstance(params_version, str):
        raise TypeError(f"params_version must be str, got {type(params_version).__name__}")
    
    # 规范化：返回格式统一的键，确保 deterministic 排序。
    canonical = f"{family}::{params_version}"
    return canonical


def extract_nested_value(payload: Dict[str, Any], dotted_path: str) -> Optional[Any]:
    """
    功能：从嵌套 dict 中按点分路径提取值。

    Extract value from nested dict using dot-separated path.

    Args:
        payload: Nested dict.
        dotted_path: Path like "attack.family" or "attack_family".

    Returns:
        Extracted value or None.
    """
    if not isinstance(payload, dict) or not isinstance(dotted_path, str):
        return None

    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def extract_first_string_value(
    record: Dict[str, Any],
    candidate_paths: List[str],
    default_value: str = "unknown",
) -> str:
    """
    功能：按候选路径列表提取第一个非空字符串值。

    Extract first non-empty string value from record using candidate paths.

    Args:
        record: Record dict.
        candidate_paths: List of dotted path candidates.
        default_value: Fallback value if no path matches.

    Returns:
        First matching non-empty string or default_value.
    """
    if not isinstance(record, dict) or not isinstance(candidate_paths, list):
        return default_value

    for candidate in candidate_paths:
        if not isinstance(candidate, str) or not candidate:
            continue
        value = extract_nested_value(record, candidate)
        if isinstance(value, str) and value:
            return value
    return default_value


def build_attack_group_key(
    record: Dict[str, Any],
    protocol_spec: Dict[str, Any],
) -> str:
    """
    功能：为记录构造规范化的 family::params_version 分组键。

    Build canonical attack group key in format "family::params_version" from record.
    Uses canonical_condition_key() to ensure deterministic ordering.

    Args:
        record: Detection record dict.
        protocol_spec: Attack protocol spec dict.

    Returns:
        Canonical group key string (e.g., "rotate::v1").
    """
    if not isinstance(record, dict):
        return canonical_condition_key("unknown_attack", "unknown_params")
    if not isinstance(protocol_spec, dict):
        protocol_spec = {}

    family_fields = protocol_spec.get("family_field_candidates", [])
    params_fields = protocol_spec.get("params_version_field_candidates", [])

    family_value = extract_first_string_value(record, family_fields, "unknown_attack")
    params_value = extract_first_string_value(record, params_fields, "unknown_params")

    return canonical_condition_key(family_value, params_value)


def extract_ground_truth_label(record: Dict[str, Any]) -> Optional[bool]:
    """
    功能：从记录中安全提取地真标签。

    Extract ground truth label from record.

    Args:
        record: Detection record dict.

    Returns:
        True/False if label exists and is bool, None otherwise.
    """
    if not isinstance(record, dict):
        return None

    for key in ["label", "ground_truth", "is_watermarked"]:
        value = record.get(key)
        if isinstance(value, bool):
            return value
    return None


def extract_geometry_score(record: Dict[str, Any]) -> Optional[float]:
    """
    功能：从记录中提取 geometry score。

    Extract geometry score from record if available.

    Args:
        record: Detection record dict.

    Returns:
        Float score or None if unavailable.
    """
    if not isinstance(record, dict):
        return None

    geometry_payload = record.get("geometry_evidence_payload")
    if isinstance(geometry_payload, dict):
        score = geometry_payload.get("geo_score")
        if isinstance(score, (int, float)) and np.isfinite(float(score)):
            return float(score)

    return None


def extract_rescue_triggered(record: Dict[str, Any]) -> bool:
    """
    功能：检测是否触发 rescue 机制。

    Check if rescue mechanism was triggered in record.

    Args:
        record: Detection record dict.

    Returns:
        True if rescue was triggered, False otherwise.
    """
    if not isinstance(record, dict):
        return False

    decision = record.get("decision")
    if isinstance(decision, dict):
        routing = decision.get("routing_decisions")
        if isinstance(routing, dict):
            return bool(routing.get("rescue_triggered", False))

    return False


def extract_hf_failure_decision(record: Dict[str, Any]) -> Optional[bool]:
    """
    功能：检测 HF 模块是否报告失败。

    Check if HF module reported failure in record.

    Args:
        record: Detection record dict.

    Returns:
        True if HF failed, False if ok, None if absent.
    """
    if not isinstance(record, dict):
        return None

    decision = record.get("decision")
    if isinstance(decision, dict):
        routing = decision.get("routing_decisions")
        if isinstance(routing, dict):
            hf_failed = routing.get("hf_failure_decision")
            if isinstance(hf_failed, bool):
                return hf_failed

    return None


def compute_overall_metrics(
    records: List[Dict[str, Any]],
    threshold_value: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    功能：计算 overall 指标（未分组）。

    Compute overall metrics across all records using threshold.

    Args:
        records: List of detection record dicts.
        threshold_value: Decision threshold.

    Returns:
        Tuple of (metrics_dict, breakdown_dict).
    """
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")

    threshold_float = float(threshold_value)

    # 初始化计数器。
    n_total = 0
    n_reject = 0
    n_pos = 0
    n_neg = 0
    tp = 0
    fp = 0
    accepted = 0
    geo_available_accepted = 0
    rescue_triggered_count = 0
    rescue_positive_count = 0

    reject_count_by_reason: Dict[str, int] = {
        "invalid_record": 0,
        "missing_content_payload": 0,
        "status_not_ok": 0,
        "invalid_score": 0,
        "missing_ground_truth": 0,
    }

    # 逐记录处理。
    for item in records:
        if not isinstance(item, dict):
            reject_count_by_reason["invalid_record"] += 1
            n_total += 1
            continue

        n_total += 1

        content_payload = item.get("content_evidence_payload")
        if not isinstance(content_payload, dict):
            n_reject += 1
            reject_count_by_reason["missing_content_payload"] += 1
            continue

        if content_payload.get("status") != "ok":
            n_reject += 1
            reject_count_by_reason["status_not_ok"] += 1
            continue

        score_value = content_payload.get("score")
        if not isinstance(score_value, (int, float)):
            n_reject += 1
            reject_count_by_reason["invalid_score"] += 1
            continue

        score_float = float(score_value)
        if not np.isfinite(score_float):
            n_reject += 1
            reject_count_by_reason["invalid_score"] += 1
            continue

        gt_value = extract_ground_truth_label(item)
        if gt_value is None:
            n_reject += 1
            reject_count_by_reason["missing_ground_truth"] += 1
            continue

        # 记录被接受（通过所有验证）。
        accepted += 1
        pred_positive = score_float >= threshold_float

        if extract_geometry_score(item) is not None:
            geo_available_accepted += 1

        if extract_rescue_triggered(item):
            rescue_triggered_count += 1
            if pred_positive:
                rescue_positive_count += 1

        if gt_value:
            n_pos += 1
            if pred_positive:
                tp += 1
        else:
            n_neg += 1
            if pred_positive:
                fp += 1

    # 计算率。
    reject_rate = float(n_reject / n_total) if n_total > 0 else 1.0
    tpr_value = float(tp / n_pos) if n_pos > 0 else None
    fpr_empirical = float(fp / n_neg) if n_neg > 0 else None
    geo_available_rate = float(geo_available_accepted / accepted) if accepted > 0 else None
    rescue_rate = float(rescue_triggered_count / accepted) if accepted > 0 else None
    rescue_gain_rate = float(rescue_positive_count / accepted) if accepted > 0 else None

    reject_rate_by_reason = {
        key_name: (float(count_value / n_total) if n_total > 0 else 0.0)
        for key_name, count_value in reject_count_by_reason.items()
    }

    metrics = {
        "tpr_at_fpr": tpr_value,
        "tpr_at_fpr_primary": tpr_value,
        "fpr_empirical": fpr_empirical,
        "reject_rate": reject_rate,
        "geo_available_rate": geo_available_rate,
        "rescue_rate": rescue_rate,
        "rescue_gain_rate": rescue_gain_rate,
        "reject_rate_by_reason": reject_rate_by_reason,
        "n_total": n_total,
        "n_accepted": accepted,
        "n_rejected": n_reject,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }

    breakdown = {
        "confusion": {
            "tp": tp,
            "fp": fp,
            "fn": max(0, n_pos - tp),
            "tn": max(0, n_neg - fp),
        }
    }

    return metrics, breakdown


def compute_attack_group_metrics(
    records: List[Dict[str, Any]],
    threshold_value: float,
    protocol_spec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    功能：按 family::params_version 分组计算指标。

    Compute metrics grouped by attack condition (family::params_version).

    Args:
        records: List of detection record dicts.
        threshold_value: Decision threshold.
        protocol_spec: Attack protocol spec dict.

    Returns:
        List of grouped metric dicts, one per unique attack condition.
    """
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")
    if not isinstance(protocol_spec, dict):
        protocol_spec = {}

    threshold_float = float(threshold_value)

    # 按分组键聚合统计。
    attack_groups: Dict[str, Dict[str, Any]] = {}

    for item in records:
        group_key = build_attack_group_key(item, protocol_spec)
        if group_key not in attack_groups:
            attack_groups[group_key] = {
                "n_total": 0,
                "n_accepted": 0,
                "n_pos": 0,
                "n_neg": 0,
                "tp": 0,
                "fp": 0,
                "geo_available_accepted": 0,
                "rescue_triggered": 0,
                "rescue_positive": 0,
                "reject_count_by_reason": {
                    "invalid_record": 0,
                    "missing_content_payload": 0,
                    "status_not_ok": 0,
                    "invalid_score": 0,
                    "missing_ground_truth": 0,
                },
            }

        group_stats = attack_groups[group_key]
        group_stats["n_total"] += 1

        if not isinstance(item, dict):
            group_stats["reject_count_by_reason"]["invalid_record"] += 1
            continue

        content_payload = item.get("content_evidence_payload")
        if not isinstance(content_payload, dict):
            group_stats["reject_count_by_reason"]["missing_content_payload"] += 1
            continue

        if content_payload.get("status") != "ok":
            group_stats["reject_count_by_reason"]["status_not_ok"] += 1
            continue

        score_value = content_payload.get("score")
        if not isinstance(score_value, (int, float)) or not np.isfinite(float(score_value)):
            group_stats["reject_count_by_reason"]["invalid_score"] += 1
            continue

        score_float = float(score_value)
        gt_value = extract_ground_truth_label(item)
        if gt_value is None:
            group_stats["reject_count_by_reason"]["missing_ground_truth"] += 1
            continue

        # 被接受。
        group_stats["n_accepted"] += 1
        pred_positive = score_float >= threshold_float

        if extract_geometry_score(item) is not None:
            group_stats["geo_available_accepted"] += 1

        if extract_rescue_triggered(item):
            group_stats["rescue_triggered"] += 1
            if pred_positive:
                group_stats["rescue_positive"] += 1

        if gt_value:
            group_stats["n_pos"] += 1
            if pred_positive:
                group_stats["tp"] += 1
        else:
            group_stats["n_neg"] += 1
            if pred_positive:
                group_stats["fp"] += 1

    # 生成输出列表（按分组键排序）。
    result: List[Dict[str, Any]] = []
    for group_key in sorted(attack_groups.keys()):
        group = attack_groups[group_key]

        group_n_pos = int(group.get("n_pos", 0))
        group_n_neg = int(group.get("n_neg", 0))
        group_tp = int(group.get("tp", 0))
        group_fp = int(group.get("fp", 0))
        group_n_accepted = int(group.get("n_accepted", 0))
        group_n_total = int(group.get("n_total", 0))

        group_tpr = float(group_tp / group_n_pos) if group_n_pos > 0 else None
        group_fpr = float(group_fp / group_n_neg) if group_n_neg > 0 else None
        group_geo_available_rate = (
            float(group.get("geo_available_accepted", 0) / group_n_accepted)
            if group_n_accepted > 0
            else None
        )
        group_rescue_rate = (
            float(group.get("rescue_triggered", 0) / group_n_accepted)
            if group_n_accepted > 0
            else None
        )
        group_rescue_gain_rate = (
            float(group.get("rescue_positive", 0) / group_n_accepted)
            if group_n_accepted > 0
            else None
        )

        group_reject_rate_by_reason = {
            key: (float(count / group_n_total) if group_n_total > 0 else 0.0)
            for key, count in group.get("reject_count_by_reason", {}).items()
        }

        result.append({
            "group_key": group_key,
            "n_total": group_n_total,
            "n_accepted": group_n_accepted,
            "n_pos": group_n_pos,
            "n_neg": group_n_neg,
            "tp": group_tp,
            "fp": group_fp,
            "tpr_at_fpr_primary": group_tpr,
            "fpr_empirical": group_fpr,
            "geo_available_rate": group_geo_available_rate,
            "rescue_rate": group_rescue_rate,
            "rescue_gain_rate": group_rescue_gain_rate,
            "reject_rate_by_reason": group_reject_rate_by_reason,
        })

    return result
