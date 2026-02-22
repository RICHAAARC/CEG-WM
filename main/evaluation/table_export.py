"""
File purpose: 评测报告导出为表格格式（JSON→CSV/TSV）。
Module type: General module
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# 导出 CSV 的固定列顺序（不依赖 dict 插入顺序或 Python 版本）。
EXPORT_COLUMN_NAMES = [
    "condition",
    "n_total",
    "n_accepted",
    "n_pos",
    "n_neg",
    "tp",
    "fp",
    "tpr_at_fpr_primary",
    "fpr_empirical",
    "geo_available_rate",
    "rescue_rate",
    "reject_rate",
]
"""
Fixed column ordering for CSV export to ensure stable output across Python versions.
Deterministic ordering prevents dict key ordering issues (Python 3.7+ insertion order guarantee).
"""


def export_metrics_to_csv(
    report: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    功能：将评测报告指标导出为 CSV 格式。

    Export metrics from evaluation report to CSV string or file.
    Includes both overall metrics and per-condition grouped metrics.

    Args:
        report: Evaluation report dict (from report_builder.build_evaluation_report).
        output_path: Optional output file path. If None, returns CSV as string.

    Returns:
        CSV content as string.

    Raises:
        TypeError: If report is invalid.
        IOError: If file write fails.
    """
    if not isinstance(report, dict):
        raise TypeError("report must be dict")

    # 提取数据。
    metrics_overall = report.get("metrics", {})
    metrics_by_condition = report.get("metrics_by_attack_condition", [])

    # 构造行数据：overall + per-condition。
    rows: List[Dict[str, Any]] = []

    # (1) Overall 行。
    overall_row = {
        "condition": "OVERALL",
        "n_total": metrics_overall.get("n_total", 0),
        "n_accepted": metrics_overall.get("n_accepted", 0),
        "n_pos": metrics_overall.get("n_pos", 0),
        "n_neg": metrics_overall.get("n_neg", 0),
        "tp": metrics_overall.get("confusion", {}).get("tp", 0) if isinstance(metrics_overall.get("confusion"), dict) else 0,
        "fp": metrics_overall.get("confusion", {}).get("fp", 0) if isinstance(metrics_overall.get("confusion"), dict) else 0,
        "tpr_at_fpr_primary": _format_metric_value(metrics_overall.get("tpr_at_fpr_primary")),
        "fpr_empirical": _format_metric_value(metrics_overall.get("fpr_empirical")),
        "geo_available_rate": _format_metric_value(metrics_overall.get("geo_available_rate")),
        "rescue_rate": _format_metric_value(metrics_overall.get("rescue_rate")),
        "reject_rate": _format_metric_value(metrics_overall.get("reject_rate")),
    }
    rows.append(overall_row)

    # (2) Per-condition 行（按 group_key 排序以确保稳定性）。
    if isinstance(metrics_by_condition, list):
        # 排序条件指标以确保列顺序稳定
        sorted_conditions = sorted(
            metrics_by_condition,
            key=lambda x: x.get("group_key", "unknown")
        )
        
        for condition_metrics in sorted_conditions:
            if not isinstance(condition_metrics, dict):
                continue
            condition_row = {
                "condition": condition_metrics.get("group_key", "unknown"),
                "n_total": condition_metrics.get("n_total", 0),
                "n_accepted": condition_metrics.get("n_accepted", 0),
                "n_pos": condition_metrics.get("n_pos", 0),
                "n_neg": condition_metrics.get("n_neg", 0),
                "tp": condition_metrics.get("tp", 0),
                "fp": condition_metrics.get("fp", 0),
                "tpr_at_fpr_primary": _format_metric_value(condition_metrics.get("tpr_at_fpr_primary")),
                "fpr_empirical": _format_metric_value(condition_metrics.get("fpr_empirical")),
                "geo_available_rate": _format_metric_value(condition_metrics.get("geo_available_rate")),
                "rescue_rate": _format_metric_value(condition_metrics.get("rescue_rate")),
                "reject_rate": _format_metric_value(
                    sum(condition_metrics.get("reject_rate_by_reason", {}).values())
                    if isinstance(condition_metrics.get("reject_rate_by_reason"), dict)
                    else 0.0
                ),
            }
            rows.append(condition_row)

    # 生成 CSV 内容。
    output = StringIO()
    if len(rows) > 0:
        # 使用固定的列顺序常量（不依赖 dict.keys() 或 Python 版本）。
        writer = csv.DictWriter(output, fieldnames=EXPORT_COLUMN_NAMES)
        writer.writeheader()
        writer.writerows(rows)

    csv_content = output.getvalue()

    # 若指定了输出路径，写盘。
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(csv_content, encoding="utf-8")

    return csv_content


def _format_metric_value(value: Any) -> str:
    """
    功能：格式化 metric 值为字符串（handling None 和浮点）。

    Format metric value for CSV export.

    Args:
        value: Raw metric value (float, int, None, etc).

    Returns:
        Formatted string ("N/A" if None, or f"{value:.6f}" for floats).
    """
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)
    return str(value)


def export_anchors_to_manifest(
    report: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    功能：导出报告锚点信息到 manifest 文件（JSON 格式）。

    Export report anchors to manifest file for auditability.

    Args:
        report: Evaluation report dict.
        output_path: Output JSON file path.

    Returns:
        None.

    Raises:
        TypeError: If report is invalid.
        IOError: If file write fails.
    """
    import json

    if not isinstance(report, dict):
        raise TypeError("report must be dict")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anchors = report.get("anchors", {})
    manifest = {
        "evaluation_version": report.get("evaluation_version", "<absent>"),
        "anchors": anchors,
        "attack_protocol_version": report.get("attack_protocol_version", "<absent>"),
        "attack_protocol_digest": report.get("attack_protocol_digest", "<absent>"),
        "metrics_digest": report.get("metrics_digest", "<absent>"),
        "metrics_by_condition_digest": report.get("metrics_by_condition_digest", "<absent>"),
    }

    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
