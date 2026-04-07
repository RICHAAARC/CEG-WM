"""
File purpose: Build append-only PW02 operating, quality, and payload metric exports.
Module type: Semi-general module
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from scripts.notebook_runtime_common import ensure_directory, normalize_path_value, utc_now_iso, write_json_atomic


PAPER_SCOPE_ORDER: Tuple[str, ...] = ("content_chain", "event_attestation", "system_final")
ANALYSIS_KEY_TO_SCOPE = {
    "content": "content_chain",
    "attestation": "event_attestation",
}
ROC_FILE_NAMES = {
    "content_chain": "roc_curve_content_chain.json",
    "event_attestation": "roc_curve_event_attestation.json",
    "system_final": "roc_curve_system_final.json",
}
TARGET_FPRS: Tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5)
SYSTEM_FINAL_OPERATING_REASON = "missing scalar score chain for threshold sweep; only derived point metrics are available"
QUALITY_SCOPE_UNAVAILABLE_REASON = "quality payload is only defined for clean content-chain image pairs in current workflow"
LPIPS_UNAVAILABLE_REASON = "requires additional model dependency or upstream implementation"
CLIP_UNAVAILABLE_REASON = "requires additional model dependency or upstream implementation"
PAYLOAD_UNAVAILABLE_REASON = "missing upstream decoded bits / reference bits / bit error sidecar"


def _load_required_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    功能：读取必需的 JSON 对象文件。

    Load one required JSON object file.

    Args:
        path_obj: JSON file path.
        label: Human-readable label.

    Returns:
        Parsed JSON mapping.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")
    import json

    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return cast(Dict[str, Any], payload)


def _extract_mapping(node: Any) -> Dict[str, Any]:
    """
    功能：将可选映射节点规范化为 dict。

    Normalize an optional mapping node to dict.

    Args:
        node: Candidate mapping node.

    Returns:
        Normalized dict mapping.
    """
    return dict(cast(Mapping[str, Any], node)) if isinstance(node, Mapping) else {}


def _extract_float(node: Mapping[str, Any], key_name: str) -> float | None:
    """
    功能：提取有限浮点字段。

    Extract one finite floating-point field.

    Args:
        node: Mapping payload.
        key_name: Field name.

    Returns:
        Finite float when available, otherwise None.
    """
    value = node.get(key_name)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
    return None


def _extract_int(node: Mapping[str, Any], key_name: str) -> int | None:
    """
    功能：提取整数字段。

    Extract one integer field.

    Args:
        node: Mapping payload.
        key_name: Field name.

    Returns:
        Integer value when available, otherwise None.
    """
    value = node.get(key_name)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _normalize_float_list(values: Any) -> List[float]:
    """
    功能：把数值序列规范化为浮点列表。

    Normalize a numeric sequence to a finite float list.

    Args:
        values: Candidate numeric sequence.

    Returns:
        Normalized float list.
    """
    if not isinstance(values, list):
        return []
    normalized: List[float] = []
    for value in cast(List[object], values):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            value_float = float(value)
            if math.isfinite(value_float):
                normalized.append(value_float)
    return normalized


def _write_csv_rows(output_path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """
    功能：按固定列顺序写出 CSV。

    Write rows to CSV using a fixed field ordering.

    Args:
        output_path: CSV output path.
        fieldnames: Fixed field ordering.
        rows: Ordered row payloads.

    Returns:
        None.
    """
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be Path")
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: row.get(field_name) for field_name in fieldnames})


def _extract_label_counts(metrics_payload: Mapping[str, Any]) -> Tuple[int | None, int | None]:
    """
    功能：从 metrics payload 提取正负样本数。

    Extract positive and negative sample counts from one metrics payload.

    Args:
        metrics_payload: Metrics mapping.

    Returns:
        Tuple of (positive_count, negative_count).
    """
    positive_count = _extract_int(metrics_payload, "n_pos")
    if positive_count is None:
        positive_count = _extract_int(metrics_payload, "n_positive")
    negative_count = _extract_int(metrics_payload, "n_neg")
    if negative_count is None:
        negative_count = _extract_int(metrics_payload, "n_negative")
    return positive_count, negative_count


def _build_augmented_roc_points(
    fpr_values: Sequence[float],
    tpr_values: Sequence[float],
    threshold_values: Sequence[float],
) -> List[Tuple[float, float, float | None]]:
    """
    功能：为 ROC 序列补齐隐式边界点。

    Build an augmented ROC sequence with implicit origin and terminal points.

    Args:
        fpr_values: False-positive-rate sequence.
        tpr_values: True-positive-rate sequence.
        threshold_values: Threshold sequence.

    Returns:
        Ordered ROC tuples of (fpr, tpr, threshold).
    """
    points = [
        (float(fpr_value), float(tpr_value), float(threshold_values[index]) if index < len(threshold_values) else None)
        for index, (fpr_value, tpr_value) in enumerate(zip(fpr_values, tpr_values))
        if math.isfinite(float(fpr_value)) and math.isfinite(float(tpr_value))
    ]
    points.sort(key=lambda item: item[0])
    if not points:
        return []
    if points[0][0] > 0.0 or points[0][1] > 0.0:
        points.insert(0, (0.0, 0.0, None))
    if points[-1][0] < 1.0 or points[-1][1] < 1.0:
        points.append((1.0, 1.0, None))
    return points


def _compute_empirical_eer(points: Sequence[Tuple[float, float, float | None]]) -> Dict[str, Any]:
    """
    功能：基于经验 ROC 点计算 EER。

    Compute an empirical equal-error-rate estimate from ROC points.

    Args:
        points: Ordered ROC tuples.

    Returns:
        EER summary payload.
    """
    if not isinstance(points, Sequence):
        raise TypeError("points must be Sequence")
    if not points:
        return {
            "status": "not_available",
            "reason": "roc_curve_unavailable",
            "eer": None,
            "selected_fpr": None,
            "selected_fnr": None,
            "selected_threshold": None,
            "method": "empirical_nearest_operating_point",
        }

    best_point = None
    best_delta = None
    for fpr_value, tpr_value, threshold_value in points:
        fnr_value = 1.0 - float(tpr_value)
        delta_value = abs(float(fpr_value) - fnr_value)
        if best_delta is None or delta_value < best_delta:
            best_delta = delta_value
            best_point = (float(fpr_value), fnr_value, threshold_value)

    if best_point is None:
        return {
            "status": "not_available",
            "reason": "roc_curve_unavailable",
            "eer": None,
            "selected_fpr": None,
            "selected_fnr": None,
            "selected_threshold": None,
            "method": "empirical_nearest_operating_point",
        }

    selected_fpr, selected_fnr, selected_threshold = best_point
    return {
        "status": "ok",
        "reason": None,
        "eer": float((selected_fpr + selected_fnr) * 0.5),
        "selected_fpr": selected_fpr,
        "selected_fnr": selected_fnr,
        "selected_threshold": selected_threshold,
        "method": "empirical_nearest_operating_point",
    }


def _compute_tpr_at_target_fpr(
    points: Sequence[Tuple[float, float, float | None]],
    target_fpr: float,
) -> Dict[str, Any]:
    """
    功能：基于经验 ROC 点计算目标 FPR 下的最大 TPR。

    Compute the maximum achievable TPR under one target FPR constraint.

    Args:
        points: Ordered ROC tuples.
        target_fpr: Target false-positive rate.

    Returns:
        TPR summary payload.
    """
    if not isinstance(points, Sequence):
        raise TypeError("points must be Sequence")
    if not isinstance(target_fpr, float):
        raise TypeError("target_fpr must be float")
    if not points:
        return {
            "status": "not_available",
            "reason": "roc_curve_unavailable",
            "target_fpr": target_fpr,
            "tpr": None,
            "selected_fpr": None,
            "selected_threshold": None,
            "method": "empirical_max_tpr_under_fpr_constraint",
        }

    eligible_points = [point for point in points if point[0] <= target_fpr + 1e-15]
    if not eligible_points:
        eligible_points = [(0.0, 0.0, None)]

    selected_point = max(eligible_points, key=lambda item: (item[1], -item[0]))
    return {
        "status": "ok",
        "reason": None,
        "target_fpr": target_fpr,
        "tpr": float(selected_point[1]),
        "selected_fpr": float(selected_point[0]),
        "selected_threshold": selected_point[2],
        "method": "empirical_max_tpr_under_fpr_constraint",
    }


def _build_not_available_roc_payload(*, family_id: str, scope: str, output_path: Path, reason: str) -> Dict[str, Any]:
    """
    功能：构造不可得 scope 的 ROC 占位导出。

    Build a stable not-available ROC export for one scope.

    Args:
        family_id: Family identifier.
        scope: Canonical scope.
        output_path: Output JSON path.
        reason: Not-available reason.

    Returns:
        ROC export payload.
    """
    payload = {
        "artifact_type": "paper_workflow_pw02_operating_curve",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": scope,
        "status": "not_available",
        "reason": reason,
        "score_name": None,
        "positive_count": None,
        "negative_count": None,
        "auc": None,
        "roc_curve_points": [],
        "roc_point_count": 0,
        "source_analysis_path": None,
    }
    write_json_atomic(output_path, payload)
    return payload


def build_pw02_metrics_extensions(
    *,
    family_id: str,
    stage_root: Path,
    clean_score_analysis_exports: Mapping[str, str],
) -> Dict[str, Any]:
    """
    功能：为 PW02 构建独立 operating、quality 与 payload 导出目录。

    Build append-only PW02 operating, quality, and payload export artifacts.

    Args:
        family_id: Family identifier.
        stage_root: PW02 export root.
        clean_score_analysis_exports: Mapping from analysis key to analysis JSON path.

    Returns:
        Summary mapping of generated directories and files.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")
    if not isinstance(clean_score_analysis_exports, Mapping):
        raise TypeError("clean_score_analysis_exports must be Mapping")

    operating_metrics_dir = ensure_directory(stage_root / "operating_metrics")
    quality_dir = ensure_directory(stage_root / "quality")
    payload_dir = ensure_directory(stage_root / "payload")

    analysis_payloads: Dict[str, Dict[str, Any]] = {}
    for analysis_key, scope_name in ANALYSIS_KEY_TO_SCOPE.items():
        analysis_path_value = clean_score_analysis_exports.get(analysis_key)
        if not isinstance(analysis_path_value, str) or not analysis_path_value.strip():
            continue
        analysis_payload = _load_required_json_dict(
            Path(analysis_path_value).expanduser().resolve(),
            f"PW02 clean score analysis {analysis_key}",
        )
        analysis_payloads[scope_name] = analysis_payload

    roc_curve_paths: Dict[str, str] = {}
    auc_rows: List[Dict[str, Any]] = []
    eer_rows: List[Dict[str, Any]] = []
    tpr_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []

    for scope_name in PAPER_SCOPE_ORDER:
        roc_output_path = operating_metrics_dir / ROC_FILE_NAMES[scope_name]
        analysis_payload = analysis_payloads.get(scope_name)
        if analysis_payload is None:
            roc_payload = _build_not_available_roc_payload(
                family_id=family_id,
                scope=scope_name,
                output_path=roc_output_path,
                reason=SYSTEM_FINAL_OPERATING_REASON,
            )
            auc_rows.append(
                {
                    "scope": scope_name,
                    "status": "not_available",
                    "reason": SYSTEM_FINAL_OPERATING_REASON,
                    "auc": None,
                    "roc_point_count": 0,
                    "positive_count": None,
                    "negative_count": None,
                    "source_analysis_path": None,
                }
            )
            eer_rows.append(
                {
                    "scope": scope_name,
                    "status": "not_available",
                    "reason": SYSTEM_FINAL_OPERATING_REASON,
                    "eer": None,
                    "selected_fpr": None,
                    "selected_fnr": None,
                    "selected_threshold": None,
                    "method": "empirical_nearest_operating_point",
                    "source_analysis_path": None,
                }
            )
            for target_fpr in TARGET_FPRS:
                tpr_rows.append(
                    {
                        "scope": scope_name,
                        "target_fpr": target_fpr,
                        "status": "not_available",
                        "reason": SYSTEM_FINAL_OPERATING_REASON,
                        "tpr": None,
                        "selected_fpr": None,
                        "selected_threshold": None,
                        "method": "empirical_max_tpr_under_fpr_constraint",
                        "source_analysis_path": None,
                    }
                )
            quality_rows.append(
                {
                    "scope": scope_name,
                    "status": "not_available",
                    "reason": QUALITY_SCOPE_UNAVAILABLE_REASON,
                    "pair_count": None,
                    "expected_pair_count": None,
                    "missing_count": None,
                    "error_count": None,
                    "mean_psnr": None,
                    "mean_ssim": None,
                    "lpips_status": "not_available",
                    "lpips_reason": LPIPS_UNAVAILABLE_REASON,
                    "clip_status": "not_available",
                    "clip_reason": CLIP_UNAVAILABLE_REASON,
                    "source_analysis_path": None,
                }
            )
            roc_curve_paths[scope_name] = normalize_path_value(roc_output_path)
            continue

        metrics_payload = _extract_mapping(analysis_payload.get("operating_metrics"))
        roc_auc_payload = _extract_mapping(analysis_payload.get("roc_auc"))
        quality_payload = _extract_mapping(analysis_payload.get("clean_positive_quality_metrics"))
        fpr_values = _normalize_float_list(roc_auc_payload.get("fpr"))
        tpr_values = _normalize_float_list(roc_auc_payload.get("tpr"))
        threshold_values = _normalize_float_list(roc_auc_payload.get("thresholds"))
        positive_count, negative_count = _extract_label_counts(metrics_payload)
        augmented_points = _build_augmented_roc_points(fpr_values, tpr_values, threshold_values)
        roc_payload = {
            "artifact_type": "paper_workflow_pw02_operating_curve",
            "schema_version": "pw_stage_02_v1",
            "created_at": utc_now_iso(),
            "family_id": family_id,
            "scope": scope_name,
            "status": "ok" if fpr_values and tpr_values else "not_available",
            "reason": None if fpr_values and tpr_values else "roc_curve_unavailable",
            "score_name": analysis_payload.get("score_name"),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "auc": _extract_float(roc_auc_payload, "auc"),
            "roc_curve_points": [
                {
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "threshold": float(threshold_values[index]) if index < len(threshold_values) else None,
                }
                for index, (fpr_value, tpr_value) in enumerate(zip(fpr_values, tpr_values))
            ],
            "roc_point_count": len(fpr_values),
            "source_analysis_path": normalize_path_value(Path(str(clean_score_analysis_exports[cast(str, next(key for key, value in ANALYSIS_KEY_TO_SCOPE.items() if value == scope_name))])).expanduser().resolve()),
        }
        write_json_atomic(roc_output_path, roc_payload)
        roc_curve_paths[scope_name] = normalize_path_value(roc_output_path)

        auc_rows.append(
            {
                "scope": scope_name,
                "status": roc_payload["status"],
                "reason": roc_payload["reason"],
                "auc": roc_payload["auc"],
                "roc_point_count": roc_payload["roc_point_count"],
                "positive_count": positive_count,
                "negative_count": negative_count,
                "source_analysis_path": roc_payload["source_analysis_path"],
            }
        )

        eer_payload = _compute_empirical_eer(augmented_points)
        eer_rows.append(
            {
                "scope": scope_name,
                "status": eer_payload["status"],
                "reason": eer_payload["reason"],
                "eer": eer_payload["eer"],
                "selected_fpr": eer_payload["selected_fpr"],
                "selected_fnr": eer_payload["selected_fnr"],
                "selected_threshold": eer_payload["selected_threshold"],
                "method": eer_payload["method"],
                "source_analysis_path": roc_payload["source_analysis_path"],
            }
        )

        for target_fpr in TARGET_FPRS:
            tpr_payload = _compute_tpr_at_target_fpr(augmented_points, float(target_fpr))
            tpr_rows.append(
                {
                    "scope": scope_name,
                    "target_fpr": float(target_fpr),
                    "status": tpr_payload["status"],
                    "reason": tpr_payload["reason"],
                    "tpr": tpr_payload["tpr"],
                    "selected_fpr": tpr_payload["selected_fpr"],
                    "selected_threshold": tpr_payload["selected_threshold"],
                    "method": tpr_payload["method"],
                    "source_analysis_path": roc_payload["source_analysis_path"],
                }
            )

        quality_rows.append(
            {
                "scope": scope_name,
                "status": quality_payload.get("status", "not_available"),
                "reason": quality_payload.get("availability_reason"),
                "pair_count": quality_payload.get("count"),
                "expected_pair_count": quality_payload.get("expected_count"),
                "missing_count": quality_payload.get("missing_count"),
                "error_count": quality_payload.get("error_count"),
                "mean_psnr": quality_payload.get("mean_psnr"),
                "mean_ssim": quality_payload.get("mean_ssim"),
                "lpips_status": "not_available",
                "lpips_reason": LPIPS_UNAVAILABLE_REASON,
                "clip_status": "not_available",
                "clip_reason": CLIP_UNAVAILABLE_REASON,
                "source_analysis_path": roc_payload["source_analysis_path"],
            }
        )

    auc_summary_path = operating_metrics_dir / "auc_summary.json"
    eer_summary_path = operating_metrics_dir / "eer_summary.json"
    tpr_summary_path = operating_metrics_dir / "tpr_at_target_fpr_summary.csv"
    quality_summary_csv_path = quality_dir / "quality_metrics_summary.csv"
    quality_summary_json_path = quality_dir / "quality_metrics_summary.json"
    payload_clean_summary_path = payload_dir / "payload_clean_summary.json"

    write_json_atomic(
        auc_summary_path,
        {
            "artifact_type": "paper_workflow_pw02_auc_summary",
            "schema_version": "pw_stage_02_v1",
            "created_at": utc_now_iso(),
            "family_id": family_id,
            "scopes": list(PAPER_SCOPE_ORDER),
            "rows": auc_rows,
        },
    )
    write_json_atomic(
        eer_summary_path,
        {
            "artifact_type": "paper_workflow_pw02_eer_summary",
            "schema_version": "pw_stage_02_v1",
            "created_at": utc_now_iso(),
            "family_id": family_id,
            "scopes": list(PAPER_SCOPE_ORDER),
            "rows": eer_rows,
        },
    )
    _write_csv_rows(
        tpr_summary_path,
        [
            "scope",
            "target_fpr",
            "status",
            "reason",
            "tpr",
            "selected_fpr",
            "selected_threshold",
            "method",
            "source_analysis_path",
        ],
        tpr_rows,
    )
    _write_csv_rows(
        quality_summary_csv_path,
        [
            "scope",
            "status",
            "reason",
            "pair_count",
            "expected_pair_count",
            "missing_count",
            "error_count",
            "mean_psnr",
            "mean_ssim",
            "lpips_status",
            "lpips_reason",
            "clip_status",
            "clip_reason",
            "source_analysis_path",
        ],
        quality_rows,
    )
    write_json_atomic(
        quality_summary_json_path,
        {
            "artifact_type": "paper_workflow_pw02_quality_metrics_summary",
            "schema_version": "pw_stage_02_v1",
            "created_at": utc_now_iso(),
            "family_id": family_id,
            "rows": quality_rows,
        },
    )
    write_json_atomic(
        payload_clean_summary_path,
        {
            "artifact_type": "paper_workflow_pw02_payload_clean_summary",
            "schema_version": "pw_stage_02_v1",
            "created_at": utc_now_iso(),
            "family_id": family_id,
            "status": "not_available",
            "reason": PAYLOAD_UNAVAILABLE_REASON,
            "future_upstream_sidecar_required": True,
        },
    )

    return {
        "operating_metrics_dir": normalize_path_value(operating_metrics_dir),
        "quality_metrics_dir": normalize_path_value(quality_dir),
        "payload_metrics_dir": normalize_path_value(payload_dir),
        "roc_curve_paths": roc_curve_paths,
        "auc_summary_path": normalize_path_value(auc_summary_path),
        "eer_summary_path": normalize_path_value(eer_summary_path),
        "tpr_at_target_fpr_summary_path": normalize_path_value(tpr_summary_path),
        "quality_metrics_summary_csv_path": normalize_path_value(quality_summary_csv_path),
        "quality_metrics_summary_json_path": normalize_path_value(quality_summary_json_path),
        "payload_clean_summary_path": normalize_path_value(payload_clean_summary_path),
    }