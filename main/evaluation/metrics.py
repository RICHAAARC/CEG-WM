"""
File purpose: 评测指标计算与分组聚合。
Module type: General module
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from main.evaluation.image_quality import compute_quality_metrics_batch


EVENT_ATTESTATION_SCORE_NAME = "event_attestation_score"
EVENT_ATTESTATION_SCORE_ALIAS_NAME = "event_attestation_statistics_score"
LEGACY_EVENT_ATTESTATION_ALIAS_RERUN_REASON = (
    "legacy_event_attestation_statistics_score_artifact_requires_rerun"
)
LEGACY_EVENT_ATTESTATION_ALIAS_RERUN_PROFILES = (
    "paper_full_cuda_mini_real_validation",
    "paper_full_cuda",
)


def is_legacy_event_attestation_alias_score_name(score_name: str) -> bool:
    """
    功能：判定 score_name 是否为旧 event_attestation_statistics_score alias。

    Determine whether the requested score name is the legacy
    event_attestation_statistics_score compatibility alias.

    Args:
        score_name: Candidate score name.

    Returns:
        True when the score name is the legacy alias; otherwise False.
    """
    if not isinstance(score_name, str):
        raise TypeError("score_name must be str")
    return score_name == EVENT_ATTESTATION_SCORE_ALIAS_NAME


def build_legacy_event_attestation_alias_rerun_guidance() -> Dict[str, Any]:
    """
    功能：构造旧 event attestation alias 触发时的统一重跑指引。

    Build the canonical rerun guidance for legacy event-attestation alias
    artifacts that are no longer accepted by the current formal mainline.

    Args:
        None.

    Returns:
        Structured rerun-guidance mapping.
    """
    return {
        "reason": LEGACY_EVENT_ATTESTATION_ALIAS_RERUN_REASON,
        "legacy_score_name": EVENT_ATTESTATION_SCORE_ALIAS_NAME,
        "canonical_score_name": EVENT_ATTESTATION_SCORE_NAME,
        "recommended_rerun_profiles": list(LEGACY_EVENT_ATTESTATION_ALIAS_RERUN_PROFILES),
    }


def raise_if_legacy_event_attestation_alias_requested(score_name: str, consumer: str) -> None:
    """
    功能：在 formal 主线请求旧 alias 时抛出统一重跑错误。

    Raise a stable rerun-required error when the formal mainline attempts to
    consume the legacy event_attestation_statistics_score alias.

    Args:
        score_name: Requested score name.
        consumer: Consumer identifier for diagnostics.

    Returns:
        None.

    Raises:
        ValueError: If the legacy alias is requested.
    """
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")
    if not isinstance(consumer, str) or not consumer:
        raise TypeError("consumer must be non-empty str")
    if not is_legacy_event_attestation_alias_score_name(score_name):
        return

    rerun_guidance = build_legacy_event_attestation_alias_rerun_guidance()
    rerun_profiles = ",".join(rerun_guidance["recommended_rerun_profiles"])
    raise ValueError(
        f"{consumer} requires rerun with current formal event-attestation artifacts; "
        f"reason={rerun_guidance['reason']}; "
        f"requested_score_name={rerun_guidance['legacy_score_name']}; "
        f"canonical_score_name={rerun_guidance['canonical_score_name']}; "
        f"recommended_rerun_profiles={rerun_profiles}"
    )


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


def _extract_chain_status(record: Dict[str, Any], chain_name: str) -> Optional[str]:
    """
    功能：提取链路状态（content/geometry）。

    Extract chain status from record payloads.

    Args:
        record: Detection record dict.
        chain_name: Chain name, one of "content" or "geometry".

    Returns:
        Status string if available, otherwise None.
    """
    if not isinstance(record, dict):
        return None
    if chain_name == "content":
        payload = record.get("content_evidence_payload")
    elif chain_name == "geometry":
        payload = record.get("geometry_evidence_payload")
    else:
        return None
    if not isinstance(payload, dict):
        return None
    status_value = payload.get("status")
    if isinstance(status_value, str) and status_value:
        return status_value
    return None


def compute_overall_metrics(
    records: List[Dict[str, Any]],
    threshold_value: float,
    score_name: str = "content_score",
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
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")

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
        "content_absent": 0,
        "content_mismatch": 0,
        "content_fail": 0,
        "geometry_absent": 0,
        "geometry_mismatch": 0,
        "geometry_fail": 0,
    }

    # 逐记录处理。
    for item in records:
        if not isinstance(item, dict):
            reject_count_by_reason["invalid_record"] += 1
            n_total += 1
            continue

        n_total += 1

        score_value, score_status = _extract_score_value_for_metrics(item, score_name)
        if score_status == "missing_content_payload":
            n_reject += 1
            reject_count_by_reason["missing_content_payload"] += 1
            continue

        if score_status == "status_not_ok":
            n_reject += 1
            reject_count_by_reason["status_not_ok"] += 1
            content_status = _extract_chain_status(item, "content")
            if content_status == "absent":
                reject_count_by_reason["content_absent"] += 1
            elif content_status == "mismatch":
                reject_count_by_reason["content_mismatch"] += 1
            elif content_status == "fail":
                reject_count_by_reason["content_fail"] += 1
            continue

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
        else:
            geometry_status = _extract_chain_status(item, "geometry")
            if geometry_status == "absent":
                reject_count_by_reason["geometry_absent"] += 1
            elif geometry_status == "mismatch":
                reject_count_by_reason["geometry_mismatch"] += 1
            elif geometry_status == "fail":
                reject_count_by_reason["geometry_fail"] += 1

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
        "score_name": score_name,
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
    score_name: str = "content_score",
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
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")
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
                    "content_absent": 0,
                    "content_mismatch": 0,
                    "content_fail": 0,
                    "geometry_absent": 0,
                    "geometry_mismatch": 0,
                    "geometry_fail": 0,
                },
            }

        group_stats = attack_groups[group_key]
        group_stats["n_total"] += 1

        if not isinstance(item, dict):
            group_stats["reject_count_by_reason"]["invalid_record"] += 1
            continue

        score_value, score_status = _extract_score_value_for_metrics(item, score_name)
        if score_status == "missing_content_payload":
            group_stats["reject_count_by_reason"]["missing_content_payload"] += 1
            continue

        if score_status == "status_not_ok":
            group_stats["reject_count_by_reason"]["status_not_ok"] += 1
            content_status = _extract_chain_status(item, "content")
            if content_status == "absent":
                group_stats["reject_count_by_reason"]["content_absent"] += 1
            elif content_status == "mismatch":
                group_stats["reject_count_by_reason"]["content_mismatch"] += 1
            elif content_status == "fail":
                group_stats["reject_count_by_reason"]["content_fail"] += 1
            continue

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
        else:
            geometry_status = _extract_chain_status(item, "geometry")
            if geometry_status == "absent":
                group_stats["reject_count_by_reason"]["geometry_absent"] += 1
            elif geometry_status == "mismatch":
                group_stats["reject_count_by_reason"]["geometry_mismatch"] += 1
            elif geometry_status == "fail":
                group_stats["reject_count_by_reason"]["geometry_fail"] += 1

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


def compute_roc_curve(records: List[Dict[str, Any]], score_name: str = "content_score") -> Tuple[List[float], List[float], List[float]]:
    """
    功能：计算 ROC 曲线（多阈值扫描）。

    Compute ROC curve by sweeping thresholds over all available scores.

    Args:
        records: Detection record list.

    Returns:
        Tuple of (fpr_list, tpr_list, thresholds).

    Raises:
        TypeError: If records type is invalid.
    """
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")

    score_label_pairs: List[Tuple[float, bool]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        if score_name == "content_score":
            content_payload = item.get("content_evidence_payload")
            if not isinstance(content_payload, dict):
                continue
            score_value = content_payload.get("score")
        else:
            score_value, score_status = _extract_score_value_for_metrics(item, score_name)
            if score_status is not None:
                continue
        if not isinstance(score_value, (int, float)):
            continue
        score_float = float(score_value)
        if not np.isfinite(score_float):
            continue
        label_value = extract_ground_truth_label(item)
        if label_value is None:
            continue
        score_label_pairs.append((score_float, bool(label_value)))

    if not score_label_pairs:
        return [], [], []

    unique_thresholds = sorted({pair[0] for pair in score_label_pairs}, reverse=True)
    unique_thresholds.append(float("-inf"))

    fpr_list: List[float] = []
    tpr_list: List[float] = []
    thresholds: List[float] = []

    pos_total = sum(1 for _, label in score_label_pairs if label)
    neg_total = sum(1 for _, label in score_label_pairs if not label)

    for threshold in unique_thresholds:
        true_positive = 0
        false_positive = 0
        for score_float, label_value in score_label_pairs:
            pred_positive = score_float >= threshold
            if pred_positive and label_value:
                true_positive += 1
            elif pred_positive and not label_value:
                false_positive += 1

        tpr_value = float(true_positive / pos_total) if pos_total > 0 else 0.0
        fpr_value = float(false_positive / neg_total) if neg_total > 0 else 0.0
        tpr_list.append(tpr_value)
        fpr_list.append(fpr_value)
        thresholds.append(float(threshold) if np.isfinite(threshold) else -1e12)

    pairs = sorted(zip(fpr_list, tpr_list, thresholds), key=lambda item: item[0])
    return [item[0] for item in pairs], [item[1] for item in pairs], [item[2] for item in pairs]


def compute_auc(fpr_list: List[float], tpr_list: List[float]) -> float:
    """
    功能：使用梯形法计算 AUC。

    Compute AUC value from ROC points using trapezoidal integration.

    Args:
        fpr_list: False positive rates.
        tpr_list: True positive rates.

    Returns:
        AUC value in [0, 1].

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(fpr_list, list) or not isinstance(tpr_list, list):
        raise ValueError("fpr_list and tpr_list must be lists")
    if len(fpr_list) != len(tpr_list):
        raise ValueError("fpr_list and tpr_list must have same length")
    if len(fpr_list) == 0:
        raise ValueError("fpr_list and tpr_list must be non-empty")

    auc_value = 0.0
    for index in range(len(fpr_list) - 1):
        delta_x = float(fpr_list[index + 1] - fpr_list[index])
        auc_value += delta_x * float(tpr_list[index + 1] + tpr_list[index]) * 0.5
    return float(max(0.0, min(1.0, auc_value)))


def aggregate_metrics(
    records_manifest: Any,
    thresholds_artifact: Dict[str, Any],
    attack_protocol: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：聚合评测指标并返回结构化结果。

    Aggregate evaluation metrics from records and threshold artifact.

    Args:
        records_manifest: Record list or manifest-like mapping containing records.
        thresholds_artifact: Threshold artifact mapping with threshold_value.
        attack_protocol: Attack protocol spec mapping.

    Returns:
        Dict containing overall metrics, breakdown and grouped metrics.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If threshold value is absent.
    """
    if isinstance(records_manifest, list):
        records = records_manifest
    elif isinstance(records_manifest, dict):
        candidate_records = records_manifest.get("records")
        if isinstance(candidate_records, list):
            records = candidate_records
        else:
            records = records_manifest.get("items") if isinstance(records_manifest.get("items"), list) else []
    else:
        raise TypeError("records_manifest must be list or dict")

    if not isinstance(thresholds_artifact, dict):
        raise TypeError("thresholds_artifact must be dict")
    if not isinstance(attack_protocol, dict):
        raise TypeError("attack_protocol must be dict")

    threshold_value = thresholds_artifact.get("threshold_value")
    if not isinstance(threshold_value, (int, float)):
        raise ValueError("thresholds_artifact.threshold_value must be number")
    score_name = thresholds_artifact.get("score_name", "content_score")
    if not isinstance(score_name, str) or not score_name:
        raise ValueError("thresholds_artifact.score_name must be non-empty str")

    metrics_overall, breakdown = compute_overall_metrics(records, float(threshold_value), score_name=score_name)
    metrics_by_attack_condition = compute_attack_group_metrics(
        records,
        float(threshold_value),
        attack_protocol,
        score_name=score_name,
    )
    result = {
        "metrics_overall": metrics_overall,
        "breakdown": breakdown,
        "metrics_by_attack_condition": metrics_by_attack_condition,
    }

    roc_fpr, roc_tpr, roc_thresholds = compute_roc_curve(records, score_name=score_name)
    auc_value = None
    if roc_fpr and roc_tpr:
        try:
            auc_value = compute_auc(roc_fpr, roc_tpr)
        except ValueError:
            auc_value = None
    result["roc_auc"] = {
        "auc": auc_value,
        "roc_curve_points": len(roc_fpr),
        "fpr": roc_fpr,
        "tpr": roc_tpr,
        "thresholds": roc_thresholds,
    }

    image_pairs = records_manifest.get("image_pairs") if isinstance(records_manifest, dict) else None
    if isinstance(image_pairs, (list, tuple)):
        result["quality_metrics"] = compute_quality_metrics_batch(image_pairs)

    return result


def _extract_score_value_for_metrics(record: Dict[str, Any], score_name: str) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(record, dict):
        return None, "missing_content_payload"
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")

    if score_name == "content_score":
        content_payload = record.get("content_evidence_payload")
        if not isinstance(content_payload, dict):
            return None, "missing_content_payload"
        if content_payload.get("status") != "ok":
            return None, "status_not_ok"
        score_value = content_payload.get("score")
    elif score_name == "content_attestation_score":
        attestation_node = record.get("attestation")
        if not isinstance(attestation_node, dict):
            return None, "status_not_ok"
        image_evidence_result = attestation_node.get("image_evidence_result")
        if not isinstance(image_evidence_result, dict):
            return None, "status_not_ok"
        if image_evidence_result.get("status") != "ok":
            return None, "status_not_ok"
        formal_score_name = image_evidence_result.get("content_attestation_score_name")
        if isinstance(formal_score_name, str) and formal_score_name and formal_score_name != "content_attestation_score":
            return None, "status_not_ok"
        score_value = image_evidence_result.get("content_attestation_score")
    elif score_name == "event_attestation_score":
        attestation_node = record.get("attestation")
        if not isinstance(attestation_node, dict):
            return None, "status_not_ok"
        final_event_decision = attestation_node.get("final_event_attested_decision")
        if not isinstance(final_event_decision, dict):
            return None, "status_not_ok"
        formal_score_name = final_event_decision.get("event_attestation_score_name")
        if isinstance(formal_score_name, str) and formal_score_name and formal_score_name != "event_attestation_score":
            return None, "status_not_ok"
        score_value = final_event_decision.get("event_attestation_score")
    elif score_name == "event_attestation_statistics_score":
        attestation_node = record.get("attestation")
        if not isinstance(attestation_node, dict):
            return None, "status_not_ok"
        final_event_decision = attestation_node.get("final_event_attested_decision")
        if not isinstance(final_event_decision, dict):
            return None, "status_not_ok"
        primary_score_name = final_event_decision.get("event_attestation_score_name")
        primary_score_value = final_event_decision.get("event_attestation_score")
        if (
            (not isinstance(primary_score_name, str) or not primary_score_name or primary_score_name == "event_attestation_score")
            and isinstance(primary_score_value, (int, float))
        ):
            score_value = primary_score_value
        else:
            formal_score_name = final_event_decision.get("event_attestation_statistics_score_name")
            if (
                isinstance(formal_score_name, str)
                and formal_score_name
                and formal_score_name != "event_attestation_statistics_score"
            ):
                return None, "status_not_ok"
            score_value = final_event_decision.get("event_attestation_statistics_score")
    else:
        raise ValueError(f"unsupported score_name: {score_name}")

    if not isinstance(score_value, (int, float)):
        return None, None
    score_float = float(score_value)
    if not np.isfinite(score_float):
        return None, None
    return score_float, None
