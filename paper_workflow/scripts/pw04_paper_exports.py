"""
File purpose: Build PW04 canonical paper-facing exports, tables, figures, bootstrap confidence intervals, and optional tail estimation artifacts.
Module type: General module
"""

from __future__ import annotations

import csv
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

import numpy as np

from paper_workflow.scripts.pw_common import CLEAN_NEGATIVE_SAMPLE_ROLE
from scripts.notebook_runtime_common import ensure_directory, normalize_path_value, utc_now_iso, write_json_atomic


SCHEMA_VERSION = "pw_stage_04_v1"
BOOTSTRAP_RANDOM_SEED = 20260407
BOOTSTRAP_RESAMPLE_COUNT = 2000
TAIL_MIN_NEGATIVE_SAMPLE_COUNT = 20
TAIL_MIN_FIT_SAMPLE_COUNT = 8
TAIL_FIT_QUANTILE = 0.8
TAIL_TARGETS: Tuple[Tuple[str, float], ...] = (("1e4", 1e-4), ("1e5", 1e-5))

PAPER_SCOPE_ORDER: Tuple[str, ...] = ("content_chain", "event_attestation", "system_final")
PAPER_SCOPE_DISPLAY_NAMES: Dict[str, str] = {
    "content_chain": "Content Chain",
    "event_attestation": "Event Attestation",
    "system_final": "System Final",
}
PAPER_SCOPE_SEMANTICS: Dict[str, str] = {
    "content_chain": "Content-primary decision layer without attestation union.",
    "event_attestation": "Event-level attestation decision layer.",
    "system_final": "System final layer. This is the canonical paper-facing final scope; legacy union aliases remain compatibility-only.",
}

CONTENT_CHAIN_METRICS_FILE_NAME = "content_chain_metrics.json"
EVENT_ATTESTATION_METRICS_FILE_NAME = "event_attestation_metrics.json"
SYSTEM_FINAL_METRICS_FILE_NAME = "system_final_metrics.json"
PAPER_METRIC_REGISTRY_FILE_NAME = "paper_metric_registry.json"
MAIN_METRICS_SUMMARY_CSV_FILE_NAME = "main_metrics_summary.csv"
ATTACK_FAMILY_SUMMARY_PAPER_CSV_FILE_NAME = "attack_family_summary_paper.csv"
ATTACK_CONDITION_SUMMARY_PAPER_CSV_FILE_NAME = "attack_condition_summary_paper.csv"
RESCUE_METRICS_SUMMARY_CSV_FILE_NAME = "rescue_metrics_summary.csv"
BOOTSTRAP_CONFIDENCE_INTERVALS_FILE_NAME = "bootstrap_confidence_intervals.json"
BOOTSTRAP_CONFIDENCE_INTERVALS_CSV_FILE_NAME = "bootstrap_confidence_intervals.csv"
ATTACK_TPR_BY_FAMILY_FIGURE_FILE_NAME = "attack_tpr_by_family.png"
CLEAN_VS_ATTACK_SCOPE_OVERVIEW_FIGURE_FILE_NAME = "clean_vs_attack_scope_overview.png"
RESCUE_BREAKDOWN_FIGURE_FILE_NAME = "rescue_breakdown.png"
TAIL_FIT_DIAGNOSTICS_FILE_NAME = "tail_fit_diagnostics.json"
TAIL_FIT_STABILITY_SUMMARY_FILE_NAME = "tail_fit_stability_summary.json"

MAIN_METRICS_SUMMARY_FIELDNAMES: List[str] = [
    "scope",
    "display_name",
    "clean_positive_count",
    "clean_negative_count",
    "clean_tpr",
    "clean_fpr",
    "attack_positive_count",
    "attack_tpr",
    "accepted_count_clean_positive",
    "accepted_count_clean_negative",
    "accepted_count_attack_positive",
    "bootstrap_ci_clean_tpr_lower",
    "bootstrap_ci_clean_tpr_upper",
    "bootstrap_ci_clean_fpr_lower",
    "bootstrap_ci_clean_fpr_upper",
    "bootstrap_ci_attack_tpr_lower",
    "bootstrap_ci_attack_tpr_upper",
    "metric_source_clean",
    "metric_source_attack",
]
ATTACK_FAMILY_PAPER_FIELDNAMES: List[str] = [
    "attack_family",
    "event_count",
    "parent_event_count",
    "content_chain_attack_tpr",
    "event_attestation_attack_tpr",
    "system_final_attack_tpr",
    "content_score_mean",
    "event_attestation_score_mean",
    "attack_quality_pair_count",
    "attack_mean_psnr",
    "attack_mean_ssim",
    "attack_mean_lpips",
]
ATTACK_CONDITION_PAPER_FIELDNAMES: List[str] = [
    "attack_condition_key",
    "attack_family",
    "attack_config_name",
    "event_count",
    "parent_event_count",
    "content_chain_attack_tpr",
    "event_attestation_attack_tpr",
    "system_final_attack_tpr",
    "content_score_mean",
    "event_attestation_score_mean",
    "attack_quality_pair_count",
    "attack_mean_psnr",
    "attack_mean_ssim",
    "attack_mean_lpips",
]
RESCUE_METRICS_SUMMARY_FIELDNAMES: List[str] = [
    "geo_helped_positive_count",
    "geo_not_used_count",
    "rescue_rate",
    "attested_rate",
    "clean_false_accept_count",
    "attack_true_accept_count",
    "attack_true_accept_count_by_family",
    "geo_rescue_eligible_count",
    "geo_rescue_applied_count",
    "geo_not_used_reason_counts",
]
BOOTSTRAP_CSV_FIELDNAMES: List[str] = [
    "scope",
    "metric_name",
    "point_estimate",
    "lower_bound",
    "upper_bound",
    "sample_size",
    "success_count",
    "n_resamples",
    "random_seed",
    "status",
    "reason",
]


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
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return cast(Dict[str, Any], payload)


def _extract_mapping(node: Any) -> Dict[str, Any]:
    """
    功能：将可选映射节点标准化为 dict。

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


def _infer_count_from_rate(rate_value: float | None, total_count: int | None) -> int | None:
    """
    功能：从 rate 与样本数回推计数。

    Infer one integer count from rate and total count.

    Args:
        rate_value: Rate value in [0, 1].
        total_count: Total sample count.

    Returns:
        Rounded integer count when both inputs are estimable.
    """
    if rate_value is None or total_count is None or total_count < 0:
        return None
    inferred_value = int(round(rate_value * total_count))
    return max(0, min(total_count, inferred_value))


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


def _read_csv_rows(input_path: Path) -> List[Dict[str, Any]]:
    """
    功能：读取 CSV 行以供图形导出消费。

    Read CSV rows for figure generation.

    Args:
        input_path: CSV input path.

    Returns:
        Ordered CSV rows.
    """
    if not isinstance(input_path, Path):
        raise TypeError("input_path must be Path")
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(cast(Mapping[str, Any], row)) for row in csv.DictReader(handle)]


def _resolve_clean_attestation_export_path(pw02_summary: Mapping[str, Any], family_root: Path) -> Path:
    """
    功能：解析 PW02 clean attestation evaluate export 路径。

    Resolve the PW02 clean attestation evaluate export path.

    Args:
        pw02_summary: PW02 summary payload.
        family_root: Paper workflow family root.

    Returns:
        Resolved clean attestation export path.
    """
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")

    clean_evaluate_exports = pw02_summary.get("clean_evaluate_exports")
    if isinstance(clean_evaluate_exports, Mapping):
        attestation_path_value = clean_evaluate_exports.get("attestation")
        if isinstance(attestation_path_value, str) and attestation_path_value.strip():
            return Path(attestation_path_value).expanduser().resolve()

    return (family_root / "exports" / "pw02" / "evaluate" / "clean" / "attestation" / "evaluate_record.json").resolve()


def _resolve_clean_metric_bundle(
    *,
    family_root: Path,
    pw02_summary: Mapping[str, Any],
) -> Dict[str, Path]:
    """
    功能：解析 canonical clean 侧来源路径。

    Resolve canonical clean-side source paths used by paper-facing exports.

    Args:
        family_root: Paper workflow family root.
        pw02_summary: PW02 summary payload.

    Returns:
        Mapping of canonical clean-side paths.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")

    content_path_value = pw02_summary.get("formal_final_decision_metrics_artifact_path")
    content_path = (
        Path(str(content_path_value)).expanduser().resolve()
        if isinstance(content_path_value, str) and content_path_value.strip()
        else (family_root / "exports" / "pw02" / "formal_final_decision_metrics.json").resolve()
    )
    system_path_value = pw02_summary.get("derived_system_union_metrics_artifact_path")
    system_path = (
        Path(str(system_path_value)).expanduser().resolve()
        if isinstance(system_path_value, str) and system_path_value.strip()
        else (family_root / "exports" / "pw02" / "derived_system_union_metrics.json").resolve()
    )
    return {
        "content_chain": content_path,
        "event_attestation": _resolve_clean_attestation_export_path(pw02_summary, family_root),
        "system_final": system_path,
    }


def _build_scope_metrics_payload(
    *,
    family_id: str,
    scope: str,
    semantics: str,
    clean_legacy_scope_name: str,
    attack_legacy_scope_name: str,
    clean_source_path: Path,
    attack_source_path: Path,
    clean_positive_count: int | None,
    clean_negative_count: int | None,
    clean_tpr: float | None,
    clean_fpr: float | None,
    attack_positive_count: int | None,
    attack_tpr: float | None,
    accepted_count_clean_positive: int | None,
    accepted_count_clean_negative: int | None,
    accepted_count_attack_positive: int | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    功能：构造一个 canonical scope 的 paper-facing 指标载荷与汇总行。

    Build one canonical scope payload and summary row.

    Args:
        family_id: Family identifier.
        scope: Canonical paper-facing scope.
        semantics: Scope definition text.
        clean_legacy_scope_name: Clean-side legacy scope name.
        attack_legacy_scope_name: Attack-side legacy scope name.
        clean_source_path: Clean-side source artifact path.
        attack_source_path: Attack-side source artifact path.
        clean_positive_count: Clean positive sample count.
        clean_negative_count: Clean negative sample count.
        clean_tpr: Clean true positive rate.
        clean_fpr: Clean false positive rate.
        attack_positive_count: Attack positive sample count.
        attack_tpr: Attack true positive rate.
        accepted_count_clean_positive: Clean-side positive accept count.
        accepted_count_clean_negative: Clean-side negative false accept count.
        accepted_count_attack_positive: Attack-side true accept count.

    Returns:
        Tuple of (canonical_metrics_payload, main_summary_row).
    """
    if scope not in PAPER_SCOPE_ORDER:
        raise ValueError(f"unsupported paper scope: {scope}")

    display_name = PAPER_SCOPE_DISPLAY_NAMES[scope]
    canonical_metrics_payload = {
        "artifact_type": "paper_workflow_pw04_canonical_scope_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": scope,
        "display_name": display_name,
        "legacy_scope_name": attack_legacy_scope_name,
        "legacy_source_artifact_path": normalize_path_value(attack_source_path),
        "semantics": semantics,
        "clean_metrics": {
            "legacy_scope_name": clean_legacy_scope_name,
            "legacy_source_artifact_path": normalize_path_value(clean_source_path),
            "clean_positive_count": clean_positive_count,
            "clean_negative_count": clean_negative_count,
            "clean_tpr": clean_tpr,
            "clean_fpr": clean_fpr,
            "accepted_count_clean_positive": accepted_count_clean_positive,
            "accepted_count_clean_negative": accepted_count_clean_negative,
        },
        "attack_metrics": {
            "legacy_scope_name": attack_legacy_scope_name,
            "legacy_source_artifact_path": normalize_path_value(attack_source_path),
            "attack_positive_count": attack_positive_count,
            "attack_tpr": attack_tpr,
            "accepted_count_attack_positive": accepted_count_attack_positive,
        },
        "compatibility": {
            "paper_facing_scope_name": scope,
            "clean_legacy_scope_name": clean_legacy_scope_name,
            "clean_legacy_source_artifact_path": normalize_path_value(clean_source_path),
            "attack_legacy_scope_name": attack_legacy_scope_name,
            "attack_legacy_source_artifact_path": normalize_path_value(attack_source_path),
            "legacy_aliases": [clean_legacy_scope_name, attack_legacy_scope_name],
        },
    }
    summary_row = {
        "scope": scope,
        "display_name": display_name,
        "clean_positive_count": clean_positive_count,
        "clean_negative_count": clean_negative_count,
        "clean_tpr": clean_tpr,
        "clean_fpr": clean_fpr,
        "attack_positive_count": attack_positive_count,
        "attack_tpr": attack_tpr,
        "accepted_count_clean_positive": accepted_count_clean_positive,
        "accepted_count_clean_negative": accepted_count_clean_negative,
        "accepted_count_attack_positive": accepted_count_attack_positive,
        "bootstrap_ci_clean_tpr_lower": None,
        "bootstrap_ci_clean_tpr_upper": None,
        "bootstrap_ci_clean_fpr_lower": None,
        "bootstrap_ci_clean_fpr_upper": None,
        "bootstrap_ci_attack_tpr_lower": None,
        "bootstrap_ci_attack_tpr_upper": None,
        "metric_source_clean": normalize_path_value(clean_source_path),
        "metric_source_attack": normalize_path_value(attack_source_path),
    }
    return canonical_metrics_payload, summary_row


def _build_content_chain_scope(
    *,
    family_id: str,
    clean_source_path: Path,
    attack_source_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    功能：构造 content_chain canonical 指标映射。

    Build the canonical content_chain scope payload.

    Args:
        family_id: Family identifier.
        clean_source_path: Clean-side source artifact path.
        attack_source_path: Attack-side source artifact path.

    Returns:
        Tuple of (canonical_metrics_payload, main_summary_row).
    """
    clean_payload = _load_required_json_dict(clean_source_path, "PW02 formal_final_decision_metrics")
    attack_payload = _load_required_json_dict(attack_source_path, "PW04 formal_attack_final_decision_metrics")
    clean_metrics = _extract_mapping(clean_payload.get("metrics"))
    attack_metrics = _extract_mapping(attack_payload.get("metrics"))

    clean_positive_count = _extract_int(clean_metrics, "n_positive")
    clean_negative_count = _extract_int(clean_metrics, "n_negative")
    clean_tpr = _extract_float(clean_metrics, "final_decision_tpr")
    clean_fpr = _extract_float(clean_metrics, "final_decision_fpr")
    attack_positive_count = _extract_int(attack_metrics, "attack_positive_count")
    attack_tpr = _extract_float(attack_metrics, "attack_tpr")

    return _build_scope_metrics_payload(
        family_id=family_id,
        scope="content_chain",
        semantics=PAPER_SCOPE_SEMANTICS["content_chain"],
        clean_legacy_scope_name="formal_final_decision",
        attack_legacy_scope_name="formal_attack_final_decision",
        clean_source_path=clean_source_path,
        attack_source_path=attack_source_path,
        clean_positive_count=clean_positive_count,
        clean_negative_count=clean_negative_count,
        clean_tpr=clean_tpr,
        clean_fpr=clean_fpr,
        attack_positive_count=attack_positive_count,
        attack_tpr=attack_tpr,
        accepted_count_clean_positive=_infer_count_from_rate(clean_tpr, clean_positive_count),
        accepted_count_clean_negative=_infer_count_from_rate(clean_fpr, clean_negative_count),
        accepted_count_attack_positive=_extract_int(attack_metrics, "accepted_count"),
    )


def _extract_attestation_clean_metrics(clean_source_path: Path) -> Dict[str, Any]:
    """
    功能：从 PW02 clean attestation evaluate export 提取稳定指标。

    Extract clean event-attestation metrics from the surfaced PW02 evaluate export.

    Args:
        clean_source_path: PW02 clean attestation evaluate export path.

    Returns:
        Stable attestation metric mapping.
    """
    export_payload = _load_required_json_dict(clean_source_path, "PW02 clean attestation evaluate export")
    evaluate_record = _extract_mapping(export_payload.get("evaluate_record"))
    evaluation_report = _extract_mapping(evaluate_record.get("evaluation_report"))
    metrics_payload = _extract_mapping(evaluate_record.get("metrics"))
    if not metrics_payload:
        metrics_payload = _extract_mapping(evaluation_report.get("metrics"))
    breakdown_payload = _extract_mapping(evaluate_record.get("breakdown"))
    if not breakdown_payload:
        breakdown_payload = _extract_mapping(evaluation_report.get("breakdown"))
    confusion_payload = _extract_mapping(breakdown_payload.get("confusion"))

    clean_positive_count = _extract_int(metrics_payload, "n_pos")
    clean_negative_count = _extract_int(metrics_payload, "n_neg")
    clean_tpr = _extract_float(metrics_payload, "tpr_at_fpr_primary")
    if clean_tpr is None:
        clean_tpr = _extract_float(metrics_payload, "tpr_at_fpr")
    clean_fpr = _extract_float(metrics_payload, "fpr_empirical")
    accepted_count_clean_positive = _extract_int(confusion_payload, "tp")
    if accepted_count_clean_positive is None:
        accepted_count_clean_positive = _infer_count_from_rate(clean_tpr, clean_positive_count)
    accepted_count_clean_negative = _extract_int(confusion_payload, "fp")
    if accepted_count_clean_negative is None:
        accepted_count_clean_negative = _infer_count_from_rate(clean_fpr, clean_negative_count)

    return {
        "clean_positive_count": clean_positive_count,
        "clean_negative_count": clean_negative_count,
        "clean_tpr": clean_tpr,
        "clean_fpr": clean_fpr,
        "accepted_count_clean_positive": accepted_count_clean_positive,
        "accepted_count_clean_negative": accepted_count_clean_negative,
    }


def _build_event_attestation_scope(
    *,
    family_id: str,
    clean_source_path: Path,
    attack_source_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    功能：构造 event_attestation canonical 指标映射。

    Build the canonical event_attestation scope payload.

    Args:
        family_id: Family identifier.
        clean_source_path: Clean-side source artifact path.
        attack_source_path: Attack-side source artifact path.

    Returns:
        Tuple of (canonical_metrics_payload, main_summary_row).
    """
    clean_metrics = _extract_attestation_clean_metrics(clean_source_path)
    attack_payload = _load_required_json_dict(attack_source_path, "PW04 formal_attack_attestation_metrics")
    attack_metrics = _extract_mapping(attack_payload.get("metrics"))

    return _build_scope_metrics_payload(
        family_id=family_id,
        scope="event_attestation",
        semantics=PAPER_SCOPE_SEMANTICS["event_attestation"],
        clean_legacy_scope_name="clean_attestation_evaluate_export",
        attack_legacy_scope_name="formal_attack_attestation",
        clean_source_path=clean_source_path,
        attack_source_path=attack_source_path,
        clean_positive_count=cast(int | None, clean_metrics["clean_positive_count"]),
        clean_negative_count=cast(int | None, clean_metrics["clean_negative_count"]),
        clean_tpr=cast(float | None, clean_metrics["clean_tpr"]),
        clean_fpr=cast(float | None, clean_metrics["clean_fpr"]),
        attack_positive_count=_extract_int(attack_metrics, "attack_positive_count"),
        attack_tpr=_extract_float(attack_metrics, "attack_tpr"),
        accepted_count_clean_positive=cast(int | None, clean_metrics["accepted_count_clean_positive"]),
        accepted_count_clean_negative=cast(int | None, clean_metrics["accepted_count_clean_negative"]),
        accepted_count_attack_positive=_extract_int(attack_metrics, "accepted_count"),
    )


def _build_system_final_scope(
    *,
    family_id: str,
    clean_source_path: Path,
    attack_source_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    功能：构造 system_final canonical 指标映射。

    Build the canonical system_final scope payload.

    Args:
        family_id: Family identifier.
        clean_source_path: Clean-side source artifact path.
        attack_source_path: Attack-side source artifact path.

    Returns:
        Tuple of (canonical_metrics_payload, main_summary_row).
    """
    clean_payload = _load_required_json_dict(clean_source_path, "PW02 derived_system_union_metrics")
    attack_payload = _load_required_json_dict(attack_source_path, "PW04 derived_attack_union_metrics")
    clean_metrics = _extract_mapping(clean_payload.get("metrics"))
    attack_metrics = _extract_mapping(attack_payload.get("metrics"))

    clean_positive_count = _extract_int(clean_metrics, "n_positive")
    clean_negative_count = _extract_int(clean_metrics, "n_negative")
    clean_tpr = _extract_float(clean_metrics, "system_tpr")
    clean_fpr = _extract_float(clean_metrics, "system_fpr")
    attack_positive_count = _extract_int(attack_metrics, "attack_positive_count")
    attack_tpr = _extract_float(attack_metrics, "attack_tpr")

    return _build_scope_metrics_payload(
        family_id=family_id,
        scope="system_final",
        semantics=PAPER_SCOPE_SEMANTICS["system_final"],
        clean_legacy_scope_name="derived_system_union",
        attack_legacy_scope_name="derived_attack_union",
        clean_source_path=clean_source_path,
        attack_source_path=attack_source_path,
        clean_positive_count=clean_positive_count,
        clean_negative_count=clean_negative_count,
        clean_tpr=clean_tpr,
        clean_fpr=clean_fpr,
        attack_positive_count=attack_positive_count,
        attack_tpr=attack_tpr,
        accepted_count_clean_positive=_infer_count_from_rate(clean_tpr, clean_positive_count),
        accepted_count_clean_negative=_infer_count_from_rate(clean_fpr, clean_negative_count),
        accepted_count_attack_positive=_extract_int(attack_metrics, "accepted_count"),
    )


def _build_paper_group_rows(
    *,
    grouped_rows: Sequence[Mapping[str, Any]],
    group_key_name: str,
) -> List[Dict[str, Any]]:
    """
    功能：将旧分组指标行映射为 paper-facing canonical 列。

    Remap grouped legacy rows to paper-facing canonical columns.

    Args:
        grouped_rows: Legacy grouped metric rows.
        group_key_name: Grouping field name.

    Returns:
        Ordered paper-facing rows.
    """
    output_rows: List[Dict[str, Any]] = []
    for row in sorted(grouped_rows, key=lambda item: str(item.get(group_key_name, ""))):
        remapped_row = {
            group_key_name: row.get(group_key_name),
            "event_count": row.get("event_count"),
            "parent_event_count": row.get("parent_event_count"),
            "content_chain_attack_tpr": row.get("formal_final_decision_attack_tpr"),
            "event_attestation_attack_tpr": row.get("formal_attestation_attack_tpr"),
            "system_final_attack_tpr": row.get("derived_attack_union_tpr"),
            "content_score_mean": row.get("content_score_mean"),
            "event_attestation_score_mean": row.get("event_attestation_score_mean"),
            "attack_quality_pair_count": row.get("attack_quality_pair_count"),
            "attack_mean_psnr": row.get("attack_mean_psnr"),
            "attack_mean_ssim": row.get("attack_mean_ssim"),
        }
        if group_key_name == "attack_condition_key":
            remapped_row["attack_family"] = row.get("attack_family")
            remapped_row["attack_config_name"] = row.get("attack_config_name")
        output_rows.append(remapped_row)
    return output_rows


def _build_rescue_metrics_summary_row(
    *,
    attack_event_rows: Sequence[Mapping[str, Any]],
    system_final_main_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：从真实 attack formal records 构造 rescue 汇总行。

    Build the rescue summary row from real attack formal records and clean metrics.

    Args:
        attack_event_rows: Materialized PW04 attack event rows.
        system_final_main_row: Canonical system_final main-summary row.

    Returns:
        One rescue summary row.
    """
    geo_helped_positive_count = 0
    geo_not_used_count = 0
    geo_rescue_eligible_count = 0
    geo_rescue_applied_count = 0
    attested_positive_count = 0
    attack_true_accept_count = 0
    geo_not_used_reason_counts: Dict[str, int] = {}
    attack_true_accept_count_by_family: Dict[str, int] = {}

    for attack_event_row in attack_event_rows:
        formal_record = _extract_mapping(attack_event_row.get("formal_record"))
        attestation_payload = _extract_mapping(formal_record.get("attestation"))
        image_evidence_payload = _extract_mapping(attestation_payload.get("image_evidence_result"))
        formal_attestation_payload = _extract_mapping(formal_record.get("formal_event_attestation_decision"))
        attack_family = str(attack_event_row.get("attack_family", "<unknown>"))
        derived_attack_union_positive = bool(formal_record.get("derived_attack_union_positive", False))
        geo_rescue_eligible = bool(image_evidence_payload.get("geo_rescue_eligible", False))
        geo_rescue_applied = bool(image_evidence_payload.get("geo_rescue_applied", False))
        geo_not_used_reason = image_evidence_payload.get("geo_not_used_reason")

        if geo_rescue_eligible:
            geo_rescue_eligible_count += 1
        if geo_rescue_applied:
            geo_rescue_applied_count += 1
        if geo_rescue_applied and derived_attack_union_positive:
            geo_helped_positive_count += 1
        if isinstance(geo_not_used_reason, str) and geo_not_used_reason:
            geo_not_used_count += 1
            geo_not_used_reason_counts[geo_not_used_reason] = geo_not_used_reason_counts.get(geo_not_used_reason, 0) + 1
        if formal_attestation_payload.get("is_watermarked") is True:
            attested_positive_count += 1
        if derived_attack_union_positive:
            attack_true_accept_count += 1
            attack_true_accept_count_by_family[attack_family] = attack_true_accept_count_by_family.get(attack_family, 0) + 1

    attack_positive_count = _extract_int(system_final_main_row, "attack_positive_count")
    clean_false_accept_count = _extract_int(system_final_main_row, "accepted_count_clean_negative")
    rescue_rate = None if attack_positive_count in {None, 0} else float(geo_helped_positive_count / attack_positive_count)
    attested_rate = None if attack_positive_count in {None, 0} else float(attested_positive_count / attack_positive_count)
    return {
        "geo_helped_positive_count": geo_helped_positive_count,
        "geo_not_used_count": geo_not_used_count,
        "rescue_rate": rescue_rate,
        "attested_rate": attested_rate,
        "clean_false_accept_count": clean_false_accept_count,
        "attack_true_accept_count": attack_true_accept_count,
        "attack_true_accept_count_by_family": json.dumps(dict(sorted(attack_true_accept_count_by_family.items())), ensure_ascii=False, sort_keys=True),
        "geo_rescue_eligible_count": geo_rescue_eligible_count,
        "geo_rescue_applied_count": geo_rescue_applied_count,
        "geo_not_used_reason_counts": json.dumps(dict(sorted(geo_not_used_reason_counts.items())), ensure_ascii=False, sort_keys=True),
    }


def _bootstrap_binary_rate_interval(
    *,
    scope: str,
    metric_name: str,
    success_count: int | None,
    sample_size: int | None,
    point_estimate: float | None,
    random_seed: int,
    n_resamples: int,
) -> Dict[str, Any]:
    """
    功能：对二元率指标执行确定性 bootstrap 区间估计。

    Compute a deterministic bootstrap confidence interval for one binary rate.

    Args:
        scope: Canonical scope name.
        metric_name: Metric name.
        success_count: Number of successes.
        sample_size: Number of trials.
        point_estimate: Observed point estimate.
        random_seed: Random seed.
        n_resamples: Number of bootstrap resamples.

    Returns:
        Interval payload with explicit status.
    """
    interval_payload = {
        "scope": scope,
        "metric_name": metric_name,
        "point_estimate": point_estimate,
        "sample_size": sample_size,
        "success_count": success_count,
        "n_resamples": n_resamples,
        "random_seed": random_seed,
        "status": "ok",
        "reason": None,
        "lower_bound": None,
        "upper_bound": None,
    }
    if sample_size is None or sample_size <= 0:
        interval_payload["status"] = "not_estimable"
        interval_payload["reason"] = "sample_size_non_positive"
        return interval_payload
    if success_count is None or success_count < 0 or success_count > sample_size:
        interval_payload["status"] = "not_estimable"
        interval_payload["reason"] = "success_count_invalid"
        return interval_payload

    probability = float(success_count / sample_size)
    rng = np.random.default_rng(random_seed)
    bootstrap_samples = rng.binomial(sample_size, probability, size=n_resamples) / float(sample_size)
    lower_bound, upper_bound = np.quantile(bootstrap_samples, [0.025, 0.975])
    interval_payload["lower_bound"] = float(lower_bound)
    interval_payload["upper_bound"] = float(upper_bound)
    return interval_payload


def _build_bootstrap_payload(summary_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, float | None]]]:
    """
    功能：构造 canonical scope 的 bootstrap CI 导出。

    Build bootstrap confidence intervals for the canonical summary rows.

    Args:
        summary_rows: Canonical main-summary rows.

    Returns:
        Tuple of (json_payload, csv_rows, per_scope_interval_lookup).
    """
    payload = {
        "artifact_type": "paper_workflow_pw04_bootstrap_confidence_intervals",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "random_seed": BOOTSTRAP_RANDOM_SEED,
        "n_resamples": BOOTSTRAP_RESAMPLE_COUNT,
        "scopes": {},
    }
    csv_rows: List[Dict[str, Any]] = []
    interval_lookup: Dict[str, Dict[str, float | None]] = {}
    for scope_index, summary_row in enumerate(summary_rows):
        scope = str(summary_row.get("scope"))
        metric_specs = [
            ("clean_tpr", _extract_int(summary_row, "accepted_count_clean_positive"), _extract_int(summary_row, "clean_positive_count"), _extract_float(summary_row, "clean_tpr")),
            ("clean_fpr", _extract_int(summary_row, "accepted_count_clean_negative"), _extract_int(summary_row, "clean_negative_count"), _extract_float(summary_row, "clean_fpr")),
            ("attack_tpr", _extract_int(summary_row, "accepted_count_attack_positive"), _extract_int(summary_row, "attack_positive_count"), _extract_float(summary_row, "attack_tpr")),
        ]
        scope_payload: Dict[str, Any] = {}
        merged_bounds: Dict[str, float | None] = {
            "bootstrap_ci_clean_tpr_lower": None,
            "bootstrap_ci_clean_tpr_upper": None,
            "bootstrap_ci_clean_fpr_lower": None,
            "bootstrap_ci_clean_fpr_upper": None,
            "bootstrap_ci_attack_tpr_lower": None,
            "bootstrap_ci_attack_tpr_upper": None,
        }
        for metric_offset, metric_spec in enumerate(metric_specs):
            metric_name, success_count, sample_size, point_estimate = metric_spec
            interval_payload = _bootstrap_binary_rate_interval(
                scope=scope,
                metric_name=metric_name,
                success_count=success_count,
                sample_size=sample_size,
                point_estimate=point_estimate,
                random_seed=BOOTSTRAP_RANDOM_SEED + scope_index * 10 + metric_offset,
                n_resamples=BOOTSTRAP_RESAMPLE_COUNT,
            )
            scope_payload[metric_name] = interval_payload
            csv_rows.append(interval_payload)
            if metric_name == "clean_tpr":
                merged_bounds["bootstrap_ci_clean_tpr_lower"] = cast(float | None, interval_payload["lower_bound"])
                merged_bounds["bootstrap_ci_clean_tpr_upper"] = cast(float | None, interval_payload["upper_bound"])
            elif metric_name == "clean_fpr":
                merged_bounds["bootstrap_ci_clean_fpr_lower"] = cast(float | None, interval_payload["lower_bound"])
                merged_bounds["bootstrap_ci_clean_fpr_upper"] = cast(float | None, interval_payload["upper_bound"])
            elif metric_name == "attack_tpr":
                merged_bounds["bootstrap_ci_attack_tpr_lower"] = cast(float | None, interval_payload["lower_bound"])
                merged_bounds["bootstrap_ci_attack_tpr_upper"] = cast(float | None, interval_payload["upper_bound"])
        cast(Dict[str, Any], payload["scopes"])[scope] = scope_payload
        interval_lookup[scope] = merged_bounds
    return payload, csv_rows, interval_lookup


def _extract_clean_negative_score_samples(
    *,
    pw02_summary: Mapping[str, Any],
    score_name: str,
) -> List[float]:
    """
    功能：从 PW02 summary.score_runs 中提取 clean negative score 样本。

    Extract clean-negative score samples from PW02 score-run summaries.

    Args:
        pw02_summary: PW02 summary payload.
        score_name: Canonical score name.

    Returns:
        Ordered clean-negative score sample list.
    """
    score_runs = _extract_mapping(pw02_summary.get("score_runs"))
    score_run = _extract_mapping(score_runs.get(score_name))
    evaluate_inputs = _extract_mapping(score_run.get("evaluate_inputs"))
    records = evaluate_inputs.get("records")
    if not isinstance(records, list):
        return []

    samples: List[float] = []
    for record_node in records:
        record = _extract_mapping(record_node)
        if record.get("sample_role") != CLEAN_NEGATIVE_SAMPLE_ROLE:
            continue
        score_value = _extract_float(record, "score_value")
        if score_value is not None:
            samples.append(score_value)
    return samples


def _estimate_tail_fit(
    *,
    scope: str,
    score_name: str,
    threshold_value: float | None,
    score_samples: Sequence[float],
) -> Dict[str, Any]:
    """
    功能：执行确定性 log-CCDF tail fit。

    Run a deterministic log-CCDF tail fit on clean-negative score samples.

    Args:
        scope: Canonical scope name.
        score_name: Source score name.
        threshold_value: Observed empirical decision threshold.
        score_samples: Clean-negative score samples.

    Returns:
        Tail-fit diagnostic payload.
    """
    if threshold_value is None:
        return {
            "scope": scope,
            "score_name": score_name,
            "status": "not_estimable",
            "reason": "threshold_value_absent",
        }
    if len(score_samples) < TAIL_MIN_NEGATIVE_SAMPLE_COUNT:
        return {
            "scope": scope,
            "score_name": score_name,
            "status": "insufficient_negative_tail_samples",
            "reason": f"negative_sample_count={len(score_samples)} < {TAIL_MIN_NEGATIVE_SAMPLE_COUNT}",
            "negative_sample_count": len(score_samples),
        }

    sorted_scores = np.sort(np.asarray([float(value) for value in score_samples], dtype=float))
    tail_start_index = min(len(sorted_scores) - 1, max(0, int(math.floor(TAIL_FIT_QUANTILE * len(sorted_scores)))))
    tail_scores = sorted_scores[tail_start_index:]
    if tail_scores.size < TAIL_MIN_FIT_SAMPLE_COUNT:
        return {
            "scope": scope,
            "score_name": score_name,
            "status": "insufficient_negative_tail_samples",
            "reason": f"tail_sample_count={int(tail_scores.size)} < {TAIL_MIN_FIT_SAMPLE_COUNT}",
            "negative_sample_count": int(sorted_scores.size),
            "tail_sample_count": int(tail_scores.size),
        }

    ranks = np.arange(1, tail_scores.size + 1, dtype=float)
    survival_probabilities = (tail_scores.size - ranks + 1.0) / float(sorted_scores.size)
    log_probabilities = np.log(np.clip(survival_probabilities, 1e-12, 1.0))
    slope, intercept = np.polyfit(tail_scores, log_probabilities, 1)
    predicted_log_probabilities = slope * tail_scores + intercept
    ss_res = float(np.sum((log_probabilities - predicted_log_probabilities) ** 2))
    ss_tot = float(np.sum((log_probabilities - np.mean(log_probabilities)) ** 2))
    fit_r2 = None if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)

    if not math.isfinite(float(slope)) or not math.isfinite(float(intercept)) or float(slope) >= 0.0:
        return {
            "scope": scope,
            "score_name": score_name,
            "status": "not_estimable",
            "reason": "non_negative_or_non_finite_tail_slope",
            "negative_sample_count": int(sorted_scores.size),
            "tail_sample_count": int(tail_scores.size),
            "tail_start_score": float(tail_scores[0]),
        }

    estimated_fpr_at_threshold = float(math.exp(intercept + slope * float(threshold_value)))
    estimated_fpr_at_threshold = float(max(0.0, min(1.0, estimated_fpr_at_threshold)))
    return {
        "scope": scope,
        "score_name": score_name,
        "status": "ok",
        "reason": None,
        "negative_sample_count": int(sorted_scores.size),
        "tail_sample_count": int(tail_scores.size),
        "tail_start_score": float(tail_scores[0]),
        "tail_quantile": TAIL_FIT_QUANTILE,
        "observed_threshold_value": float(threshold_value),
        "estimated_fpr_at_observed_threshold": estimated_fpr_at_threshold,
        "fit_model": "log_linear_ccdf",
        "fit_parameters": {
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "fit_r2": fit_r2,
    }


def _build_tail_estimation_payloads(
    *,
    family_root: Path,
    pw02_summary: Mapping[str, Any],
    enable_tail_estimation: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    功能：构造可选 tail estimation 产物负载。

    Build optional tail-estimation payloads while keeping them separate from empirical clean FPR.

    Args:
        family_root: Paper workflow family root.
        pw02_summary: PW02 summary payload.
        enable_tail_estimation: Whether tail estimation is enabled.

    Returns:
        Mapping of tail artifact names to JSON payloads.
    """
    content_threshold_export = _load_required_json_dict(
        (family_root / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json").resolve(),
        "PW02 content threshold export",
    )
    attestation_threshold_export = _load_required_json_dict(
        (family_root / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json").resolve(),
        "PW02 attestation threshold export",
    )
    content_threshold_value = _extract_float(_extract_mapping(content_threshold_export.get("thresholds_artifact")), "threshold_value")
    attestation_threshold_value = _extract_float(_extract_mapping(attestation_threshold_export.get("thresholds_artifact")), "threshold_value")

    base_scope_payloads: Dict[str, Dict[str, Any]] = {
        "content_chain": {
            "score_name": "content_chain_score",
            "threshold_value": content_threshold_value,
            "samples": _extract_clean_negative_score_samples(pw02_summary=pw02_summary, score_name="content_chain_score"),
            "source_threshold_export_path": normalize_path_value((family_root / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json").resolve()),
        },
        "event_attestation": {
            "score_name": "event_attestation_score",
            "threshold_value": attestation_threshold_value,
            "samples": _extract_clean_negative_score_samples(pw02_summary=pw02_summary, score_name="event_attestation_score"),
            "source_threshold_export_path": normalize_path_value((family_root / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json").resolve()),
        },
        "system_final": {
            "score_name": None,
            "threshold_value": None,
            "samples": [],
            "source_threshold_export_path": None,
        },
    }

    diagnostics_payload = {
        "artifact_type": "paper_workflow_pw04_tail_fit_diagnostics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "tail_estimation_enabled": enable_tail_estimation,
        "empirical_fpr_separation": "estimated tail FPR artifacts are separated from empirical clean FPR and are never written into main_metrics_summary.csv",
        "scope_diagnostics": {},
    }
    stability_payload = {
        "artifact_type": "paper_workflow_pw04_tail_fit_stability_summary",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "tail_estimation_enabled": enable_tail_estimation,
        "scopes": {},
    }

    target_payloads: Dict[str, Dict[str, Any]] = {
        target_key: {
            "artifact_type": "paper_workflow_pw04_tail_estimation",
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "target_fpr": target_value,
            "tail_estimation_enabled": enable_tail_estimation,
            "empirical_fpr_separation": "estimated tail FPR artifacts are separated from empirical clean FPR and are not mixed into empirical metrics fields",
            "scope_estimates": {},
        }
        for target_key, target_value in TAIL_TARGETS
    }

    if not enable_tail_estimation:
        disabled_reason = "tail_estimation_flag_not_enabled"
        for scope in PAPER_SCOPE_ORDER:
            scope_payload = {
                "status": "not_applicable" if scope == "system_final" else "disabled",
                "reason": "system_final_is_decision_union_without_scalar_score" if scope == "system_final" else disabled_reason,
                "scope": scope,
            }
            cast(Dict[str, Any], diagnostics_payload["scope_diagnostics"])[scope] = scope_payload
            cast(Dict[str, Any], stability_payload["scopes"])[scope] = scope_payload
            for target_key, _ in TAIL_TARGETS:
                cast(Dict[str, Any], target_payloads[target_key]["scope_estimates"])[scope] = scope_payload
        return {
            "estimated_tail_fpr_1e4": target_payloads["1e4"],
            "estimated_tail_fpr_1e5": target_payloads["1e5"],
            "tail_fit_diagnostics": diagnostics_payload,
            "tail_fit_stability_summary": stability_payload,
        }

    fitted_scopes: Dict[str, Dict[str, Any]] = {}
    for scope in PAPER_SCOPE_ORDER:
        if scope == "system_final":
            fitted_scope_payload = {
                "scope": scope,
                "status": "not_applicable",
                "reason": "system_final_is_decision_union_without_scalar_score",
            }
        else:
            scope_spec = base_scope_payloads[scope]
            fitted_scope_payload = _estimate_tail_fit(
                scope=scope,
                score_name=cast(str, scope_spec["score_name"]),
                threshold_value=cast(float | None, scope_spec["threshold_value"]),
                score_samples=cast(Sequence[float], scope_spec["samples"]),
            )
            fitted_scope_payload["source_threshold_export_path"] = scope_spec["source_threshold_export_path"]
        fitted_scopes[scope] = fitted_scope_payload
        cast(Dict[str, Any], diagnostics_payload["scope_diagnostics"])[scope] = fitted_scope_payload

    for target_key, target_value in TAIL_TARGETS:
        target_payload = target_payloads[target_key]
        for scope in PAPER_SCOPE_ORDER:
            fitted_scope_payload = fitted_scopes[scope]
            if fitted_scope_payload.get("status") != "ok":
                cast(Dict[str, Any], target_payload["scope_estimates"])[scope] = dict(fitted_scope_payload)
                continue
            fit_parameters = _extract_mapping(fitted_scope_payload.get("fit_parameters"))
            slope = _extract_float(fit_parameters, "slope")
            intercept = _extract_float(fit_parameters, "intercept")
            if slope is None or intercept is None or slope >= 0.0:
                cast(Dict[str, Any], target_payload["scope_estimates"])[scope] = {
                    "scope": scope,
                    "status": "not_estimable",
                    "reason": "tail_fit_parameters_invalid",
                }
                continue
            estimated_threshold_value = float((math.log(target_value) - intercept) / slope)
            cast(Dict[str, Any], target_payload["scope_estimates"])[scope] = {
                "scope": scope,
                "status": "ok",
                "reason": None,
                "score_name": fitted_scope_payload.get("score_name"),
                "target_fpr": target_value,
                "estimated_threshold_value": estimated_threshold_value,
                "observed_threshold_value": fitted_scope_payload.get("observed_threshold_value"),
                "estimated_fpr_at_observed_threshold": fitted_scope_payload.get("estimated_fpr_at_observed_threshold"),
                "negative_sample_count": fitted_scope_payload.get("negative_sample_count"),
                "tail_sample_count": fitted_scope_payload.get("tail_sample_count"),
                "fit_r2": fitted_scope_payload.get("fit_r2"),
                "source_threshold_export_path": fitted_scope_payload.get("source_threshold_export_path"),
            }

    for scope in PAPER_SCOPE_ORDER:
        threshold_1e4 = _extract_mapping(target_payloads["1e4"]["scope_estimates"].get(scope)).get("estimated_threshold_value")
        threshold_1e5 = _extract_mapping(target_payloads["1e5"]["scope_estimates"].get(scope)).get("estimated_threshold_value")
        cast(Dict[str, Any], stability_payload["scopes"])[scope] = {
            "scope": scope,
            "status": fitted_scopes[scope].get("status"),
            "reason": fitted_scopes[scope].get("reason"),
            "fit_r2": fitted_scopes[scope].get("fit_r2"),
            "tail_sample_count": fitted_scopes[scope].get("tail_sample_count"),
            "threshold_monotonicity_ok": (
                isinstance(threshold_1e4, (int, float))
                and isinstance(threshold_1e5, (int, float))
                and float(threshold_1e5) >= float(threshold_1e4)
            ) if fitted_scopes[scope].get("status") == "ok" else None,
        }

    return {
        "estimated_tail_fpr_1e4": target_payloads["1e4"],
        "estimated_tail_fpr_1e5": target_payloads["1e5"],
        "tail_fit_diagnostics": diagnostics_payload,
        "tail_fit_stability_summary": stability_payload,
    }


def _parse_csv_float(row: Mapping[str, Any], key_name: str) -> float:
    """
    功能：从 CSV 行解析浮点值。

    Parse one floating-point value from a CSV row.

    Args:
        row: CSV row mapping.
        key_name: Field name.

    Returns:
        Parsed float or NaN when absent.
    """
    value = row.get(key_name)
    if value in {None, "", "None"}:
        return float("nan")
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")


def _parse_csv_int(row: Mapping[str, Any], key_name: str) -> int:
    """
    功能：从 CSV 行解析整数值。

    Parse one integer value from a CSV row.

    Args:
        row: CSV row mapping.
        key_name: Field name.

    Returns:
        Parsed integer or zero when absent.
    """
    value = row.get(key_name)
    if value in {None, "", "None"}:
        return 0
    try:
        return int(float(cast(Any, value)))
    except (TypeError, ValueError):
        return 0


def _write_png_image(output_path: Path, width: int, height: int, pixels: bytes) -> None:
    """
    功能：将 RGBA 像素缓冲区编码为 PNG 文件。

    Encode one RGBA pixel buffer into a PNG file.

    Args:
        output_path: PNG output path.
        width: Image width.
        height: Image height.
        pixels: RGBA pixel buffer.

    Returns:
        None.
    """
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    if len(pixels) != width * height * 4:
        raise ValueError("RGBA buffer size mismatch")

    ensure_directory(output_path.parent)
    scanlines = bytearray()
    row_stride = width * 4
    for row_index in range(height):
        scanlines.append(0)
        row_offset = row_index * row_stride
        scanlines.extend(pixels[row_offset : row_offset + row_stride])

    def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    png_bytes = bytearray(b"\x89PNG\r\n\x1a\n")
    png_bytes.extend(
        _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0),
        )
    )
    png_bytes.extend(_png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9)))
    png_bytes.extend(_png_chunk(b"IEND", b""))
    output_path.write_bytes(bytes(png_bytes))


def _create_canvas(width: int, height: int, background_color: Tuple[int, int, int, int]) -> bytearray:
    """
    功能：创建纯色 RGBA 画布。

    Create one solid-color RGBA canvas.

    Args:
        width: Image width.
        height: Image height.
        background_color: RGBA background color.

    Returns:
        Mutable RGBA canvas buffer.
    """
    return bytearray(background_color * (width * height))


def _fill_rect(
    canvas: bytearray,
    *,
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: Tuple[int, int, int, int],
) -> None:
    """
    功能：在 RGBA 画布中填充矩形。

    Fill one rectangle in the RGBA canvas.

    Args:
        canvas: Mutable RGBA canvas buffer.
        width: Canvas width.
        height: Canvas height.
        left: Left x coordinate.
        top: Top y coordinate.
        right: Right x coordinate.
        bottom: Bottom y coordinate.
        color: RGBA fill color.

    Returns:
        None.
    """
    clipped_left = max(0, min(width, left))
    clipped_right = max(0, min(width, right))
    clipped_top = max(0, min(height, top))
    clipped_bottom = max(0, min(height, bottom))
    if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
        return
    for y_index in range(clipped_top, clipped_bottom):
        for x_index in range(clipped_left, clipped_right):
            offset = (y_index * width + x_index) * 4
            canvas[offset : offset + 4] = bytes(color)


def _draw_grouped_bar_chart_png(
    *,
    output_path: Path,
    series_values: Sequence[Sequence[float]],
    colors: Sequence[Tuple[int, int, int, int]],
    y_max: float,
) -> None:
    """
    功能：使用标准库回退绘制分组柱状 PNG 图。

    Draw a grouped bar-chart PNG using only the Python standard library.

    Args:
        output_path: PNG output path.
        series_values: Per-series numeric values.
        colors: Per-series RGBA colors.
        y_max: Maximum y-axis value.

    Returns:
        None.
    """
    if y_max <= 0.0:
        y_max = 1.0
    category_count = len(series_values[0]) if series_values else 0
    canvas_width = 960
    canvas_height = 540
    left_margin = 64
    right_margin = 36
    top_margin = 32
    bottom_margin = 52
    plot_width = canvas_width - left_margin - right_margin
    plot_height = canvas_height - top_margin - bottom_margin

    canvas = _create_canvas(canvas_width, canvas_height, (255, 255, 255, 255))
    _fill_rect(
        canvas,
        width=canvas_width,
        height=canvas_height,
        left=left_margin,
        top=top_margin,
        right=left_margin + plot_width,
        bottom=top_margin + plot_height,
        color=(248, 248, 248, 255),
    )
    _fill_rect(
        canvas,
        width=canvas_width,
        height=canvas_height,
        left=left_margin,
        top=top_margin + plot_height,
        right=left_margin + plot_width,
        bottom=top_margin + plot_height + 2,
        color=(80, 80, 80, 255),
    )
    _fill_rect(
        canvas,
        width=canvas_width,
        height=canvas_height,
        left=left_margin - 2,
        top=top_margin,
        right=left_margin,
        bottom=top_margin + plot_height,
        color=(80, 80, 80, 255),
    )

    if category_count > 0 and series_values:
        group_width = plot_width / float(category_count)
        bar_width = max(6.0, group_width / float(len(series_values) + 1))
        for category_index in range(category_count):
            group_left = left_margin + category_index * group_width
            for series_index, values in enumerate(series_values):
                value = values[category_index]
                if not math.isfinite(value):
                    continue
                normalized_height = max(0.0, min(1.0, float(value / y_max)))
                bar_height = int(round(normalized_height * plot_height))
                bar_left = int(round(group_left + (series_index + 0.5) * bar_width))
                bar_right = int(round(bar_left + bar_width * 0.75))
                bar_bottom = top_margin + plot_height
                bar_top = bar_bottom - bar_height
                _fill_rect(
                    canvas,
                    width=canvas_width,
                    height=canvas_height,
                    left=bar_left,
                    top=bar_top,
                    right=bar_right,
                    bottom=bar_bottom,
                    color=colors[series_index],
                )

    _write_png_image(output_path, canvas_width, canvas_height, bytes(canvas))


def _draw_single_bar_chart_png(
    *,
    output_path: Path,
    values: Sequence[float],
    colors: Sequence[Tuple[int, int, int, int]],
) -> None:
    """
    功能：使用标准库回退绘制单序列柱状 PNG 图。

    Draw one single-series bar-chart PNG using only the Python standard library.

    Args:
        output_path: PNG output path.
        values: Numeric bar heights.
        colors: Per-bar RGBA colors.

    Returns:
        None.
    """
    y_max = max([float(value) for value in values if math.isfinite(float(value))] + [1.0])
    _draw_grouped_bar_chart_png(
        output_path=output_path,
        series_values=[list(values)],
        colors=[colors[0] if colors else (31, 119, 180, 255)],
        y_max=y_max,
    )


def _build_attack_tpr_by_family_figure(input_csv_path: Path, output_path: Path) -> None:
    """
    功能：从已写盘 family CSV 导出 attack TPR by family 图。

    Export the attack TPR by family figure from the paper-facing CSV.

    Args:
        input_csv_path: Paper-facing family CSV path.
        output_path: PNG output path.

    Returns:
        None.
    """
    rows = _read_csv_rows(input_csv_path)
    family_labels = [str(row.get("attack_family", "<unknown>")) for row in rows]
    x_axis = np.arange(len(rows), dtype=float)
    width = 0.24

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _draw_grouped_bar_chart_png(
            output_path=output_path,
            series_values=[
                [_parse_csv_float(row, "content_chain_attack_tpr") for row in rows],
                [_parse_csv_float(row, "event_attestation_attack_tpr") for row in rows],
                [_parse_csv_float(row, "system_final_attack_tpr") for row in rows],
            ],
            colors=[
                (31, 119, 180, 255),
                (255, 127, 14, 255),
                (44, 160, 44, 255),
            ],
            y_max=1.0,
        )
        return

    plt.figure(figsize=(10, 4.8))
    for offset_index, (column_name, color_name, label_name) in enumerate(
        [
            ("content_chain_attack_tpr", "#1f77b4", "content_chain"),
            ("event_attestation_attack_tpr", "#ff7f0e", "event_attestation"),
            ("system_final_attack_tpr", "#2ca02c", "system_final"),
        ]
    ):
        values = [_parse_csv_float(row, column_name) for row in rows]
        plt.bar(x_axis + (offset_index - 1) * width, values, width=width, color=color_name, label=label_name)

    plt.xticks(x_axis, family_labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Attack TPR")
    plt.title("Attack TPR by Family")
    plt.legend()
    plt.tight_layout()
    ensure_directory(output_path.parent)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _build_clean_vs_attack_scope_figure(input_csv_path: Path, output_path: Path) -> None:
    """
    功能：从主指标总表导出 clean vs attack scope overview 图。

    Export the clean-vs-attack scope overview figure from the main summary CSV.

    Args:
        input_csv_path: Main metrics summary CSV path.
        output_path: PNG output path.

    Returns:
        None.
    """
    rows = _read_csv_rows(input_csv_path)
    scope_labels = [str(row.get("scope", "<unknown>")) for row in rows]
    x_axis = np.arange(len(rows), dtype=float)
    width = 0.24

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _draw_grouped_bar_chart_png(
            output_path=output_path,
            series_values=[
                [_parse_csv_float(row, "clean_tpr") for row in rows],
                [_parse_csv_float(row, "clean_fpr") for row in rows],
                [_parse_csv_float(row, "attack_tpr") for row in rows],
            ],
            colors=[
                (31, 119, 180, 255),
                (214, 39, 40, 255),
                (44, 160, 44, 255),
            ],
            y_max=1.0,
        )
        return

    plt.figure(figsize=(8.8, 4.8))
    plotted_columns = [
        ("clean_tpr", "#1f77b4", "clean_tpr"),
        ("clean_fpr", "#d62728", "clean_fpr"),
        ("attack_tpr", "#2ca02c", "attack_tpr"),
    ]
    for offset_index, (column_name, color_name, label_name) in enumerate(plotted_columns):
        values = [_parse_csv_float(row, column_name) for row in rows]
        plt.bar(x_axis + (offset_index - 1) * width, values, width=width, color=color_name, label=label_name)

    plt.xticks(x_axis, scope_labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Rate")
    plt.title("Clean vs Attack Scope Overview")
    plt.legend()
    plt.tight_layout()
    ensure_directory(output_path.parent)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _build_rescue_breakdown_figure(input_csv_path: Path, output_path: Path) -> None:
    """
    功能：从 rescue 汇总表导出 rescue breakdown 图。

    Export the rescue breakdown figure from the rescue summary CSV.

    Args:
        input_csv_path: Rescue summary CSV path.
        output_path: PNG output path.

    Returns:
        None.
    """
    rows = _read_csv_rows(input_csv_path)
    if not rows:
        raise ValueError("rescue summary CSV must contain at least one row")
    row = rows[0]
    labels = [
        "geo_helped_positive_count",
        "geo_not_used_count",
        "clean_false_accept_count",
        "attack_true_accept_count",
    ]
    values = [_parse_csv_int(row, label_name) for label_name in labels]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _draw_single_bar_chart_png(
            output_path=output_path,
            values=[float(value) for value in values],
            colors=[
                (44, 160, 44, 255),
                (255, 127, 14, 255),
                (214, 39, 40, 255),
                (31, 119, 180, 255),
            ],
        )
        return

    plt.figure(figsize=(8.2, 4.6))
    plt.bar(labels, values, color=["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"])
    plt.ylabel("Count")
    plt.title("Rescue Breakdown")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    ensure_directory(output_path.parent)
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_pw04_paper_exports(
    *,
    family_id: str,
    family_root: Path,
    pw02_summary: Mapping[str, Any],
    pw04_paths: Mapping[str, Path],
    attack_event_rows: Sequence[Mapping[str, Any]],
    per_attack_family_metrics_payload: Mapping[str, Any],
    per_attack_condition_metrics_payload: Mapping[str, Any],
    attack_quality_metrics_payload: Mapping[str, Any],
    enable_tail_estimation: bool,
) -> Dict[str, Any]:
    """
    功能：生成 PW04 的 canonical paper-facing 导出层产物。

    Generate canonical PW04 paper-facing exports while preserving all legacy artifacts.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        pw02_summary: PW02 summary payload.
        pw04_paths: Resolved PW04 path mapping.
        attack_event_rows: Materialized PW04 attack event rows.
        per_attack_family_metrics_payload: Legacy per-family metrics payload.
        per_attack_condition_metrics_payload: Legacy per-condition metrics payload.
        enable_tail_estimation: Whether optional tail estimation is enabled.

    Returns:
        Summary mapping of the newly generated paper-facing artifacts.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")
    if not isinstance(pw04_paths, Mapping):
        raise TypeError("pw04_paths must be Mapping")
    if not isinstance(attack_quality_metrics_payload, Mapping):
        raise TypeError("attack_quality_metrics_payload must be Mapping")
    if not isinstance(enable_tail_estimation, bool):
        raise TypeError("enable_tail_estimation must be bool")

    metrics_root = cast(Path, pw04_paths["metrics_root"])
    tables_root = cast(Path, pw04_paths["tables_root"])
    figures_root = cast(Path, pw04_paths["figures_root"])
    tail_root = cast(Path, pw04_paths["tail_root"])
    ensure_directory(metrics_root)
    ensure_directory(tables_root)
    ensure_directory(figures_root)
    ensure_directory(tail_root)

    clean_paths = _resolve_clean_metric_bundle(family_root=family_root, pw02_summary=pw02_summary)
    canonical_metric_paths = {
        "content_chain": metrics_root / CONTENT_CHAIN_METRICS_FILE_NAME,
        "event_attestation": metrics_root / EVENT_ATTESTATION_METRICS_FILE_NAME,
        "system_final": metrics_root / SYSTEM_FINAL_METRICS_FILE_NAME,
    }
    bootstrap_json_path = metrics_root / BOOTSTRAP_CONFIDENCE_INTERVALS_FILE_NAME
    bootstrap_csv_path = tables_root / BOOTSTRAP_CONFIDENCE_INTERVALS_CSV_FILE_NAME
    main_metrics_summary_path = tables_root / MAIN_METRICS_SUMMARY_CSV_FILE_NAME
    attack_family_summary_paper_path = tables_root / ATTACK_FAMILY_SUMMARY_PAPER_CSV_FILE_NAME
    attack_condition_summary_paper_path = tables_root / ATTACK_CONDITION_SUMMARY_PAPER_CSV_FILE_NAME
    rescue_metrics_summary_path = tables_root / RESCUE_METRICS_SUMMARY_CSV_FILE_NAME
    tail_paths = {
        "estimated_tail_fpr_1e4_path": tail_root / "estimated_tail_fpr_1e4.json",
        "estimated_tail_fpr_1e5_path": tail_root / "estimated_tail_fpr_1e5.json",
        "tail_fit_diagnostics_path": tail_root / TAIL_FIT_DIAGNOSTICS_FILE_NAME,
        "tail_fit_stability_summary_path": tail_root / TAIL_FIT_STABILITY_SUMMARY_FILE_NAME,
    }
    figure_paths = {
        "attack_tpr_by_family_png_path": figures_root / ATTACK_TPR_BY_FAMILY_FIGURE_FILE_NAME,
        "clean_vs_attack_scope_overview_png_path": figures_root / CLEAN_VS_ATTACK_SCOPE_OVERVIEW_FIGURE_FILE_NAME,
        "rescue_breakdown_png_path": figures_root / RESCUE_BREAKDOWN_FIGURE_FILE_NAME,
    }

    scope_payloads_and_rows = [
        _build_content_chain_scope(
            family_id=family_id,
            clean_source_path=clean_paths["content_chain"],
            attack_source_path=cast(Path, pw04_paths["formal_attack_final_decision_metrics_path"]),
        ),
        _build_event_attestation_scope(
            family_id=family_id,
            clean_source_path=clean_paths["event_attestation"],
            attack_source_path=cast(Path, pw04_paths["formal_attack_attestation_metrics_path"]),
        ),
        _build_system_final_scope(
            family_id=family_id,
            clean_source_path=clean_paths["system_final"],
            attack_source_path=cast(Path, pw04_paths["derived_attack_union_metrics_path"]),
        ),
    ]

    scope_payload_map: Dict[str, Dict[str, Any]] = {}
    main_summary_rows: List[Dict[str, Any]] = []
    for scope_payload, summary_row in scope_payloads_and_rows:
        scope_name = str(scope_payload["scope"])
        scope_payload_map[scope_name] = scope_payload
        main_summary_rows.append(summary_row)

    bootstrap_payload, bootstrap_csv_rows, bootstrap_lookup = _build_bootstrap_payload(main_summary_rows)
    for summary_row in main_summary_rows:
        scope_name = str(summary_row["scope"])
        summary_row.update(bootstrap_lookup[scope_name])

    for scope_name in PAPER_SCOPE_ORDER:
        write_json_atomic(canonical_metric_paths[scope_name], scope_payload_map[scope_name])

    write_json_atomic(bootstrap_json_path, bootstrap_payload)
    _write_csv_rows(bootstrap_csv_path, BOOTSTRAP_CSV_FIELDNAMES, bootstrap_csv_rows)

    ordered_main_rows = sorted(main_summary_rows, key=lambda row: PAPER_SCOPE_ORDER.index(str(row["scope"])))
    _write_csv_rows(main_metrics_summary_path, MAIN_METRICS_SUMMARY_FIELDNAMES, ordered_main_rows)

    family_rows = _build_paper_group_rows(
        grouped_rows=cast(List[Mapping[str, Any]], per_attack_family_metrics_payload.get("rows", [])),
        group_key_name="attack_family",
    )
    _write_csv_rows(attack_family_summary_paper_path, ATTACK_FAMILY_PAPER_FIELDNAMES, family_rows)

    condition_rows = _build_paper_group_rows(
        grouped_rows=cast(List[Mapping[str, Any]], per_attack_condition_metrics_payload.get("rows", [])),
        group_key_name="attack_condition_key",
    )
    _write_csv_rows(attack_condition_summary_paper_path, ATTACK_CONDITION_PAPER_FIELDNAMES, condition_rows)

    system_final_main_row = next(row for row in ordered_main_rows if row["scope"] == "system_final")
    rescue_summary_row = _build_rescue_metrics_summary_row(
        attack_event_rows=attack_event_rows,
        system_final_main_row=system_final_main_row,
    )
    _write_csv_rows(rescue_metrics_summary_path, RESCUE_METRICS_SUMMARY_FIELDNAMES, [rescue_summary_row])

    tail_payloads = _build_tail_estimation_payloads(
        family_root=family_root,
        pw02_summary=pw02_summary,
        enable_tail_estimation=enable_tail_estimation,
    )
    write_json_atomic(cast(Path, tail_paths["estimated_tail_fpr_1e4_path"]), tail_payloads["estimated_tail_fpr_1e4"])
    write_json_atomic(cast(Path, tail_paths["estimated_tail_fpr_1e5_path"]), tail_payloads["estimated_tail_fpr_1e5"])
    write_json_atomic(cast(Path, tail_paths["tail_fit_diagnostics_path"]), tail_payloads["tail_fit_diagnostics"])
    write_json_atomic(cast(Path, tail_paths["tail_fit_stability_summary_path"]), tail_payloads["tail_fit_stability_summary"])

    _build_attack_tpr_by_family_figure(attack_family_summary_paper_path, cast(Path, figure_paths["attack_tpr_by_family_png_path"]))
    _build_clean_vs_attack_scope_figure(main_metrics_summary_path, cast(Path, figure_paths["clean_vs_attack_scope_overview_png_path"]))
    _build_rescue_breakdown_figure(rescue_metrics_summary_path, cast(Path, figure_paths["rescue_breakdown_png_path"]))

    paper_metric_registry_payload = {
        "artifact_type": "paper_workflow_pw04_paper_metric_registry",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "canonical_scopes": list(PAPER_SCOPE_ORDER),
        "legacy_scope_mapping": {
            scope_name: {
                "clean": {
                    "legacy_scope_name": _extract_mapping(scope_payload_map[scope_name].get("compatibility")).get("clean_legacy_scope_name"),
                    "artifact_path": _extract_mapping(scope_payload_map[scope_name].get("compatibility")).get("clean_legacy_source_artifact_path"),
                },
                "attack": {
                    "legacy_scope_name": _extract_mapping(scope_payload_map[scope_name].get("compatibility")).get("attack_legacy_scope_name"),
                    "artifact_path": _extract_mapping(scope_payload_map[scope_name].get("compatibility")).get("attack_legacy_source_artifact_path"),
                },
            }
            for scope_name in PAPER_SCOPE_ORDER
        },
        "artifact_paths": {
            "canonical_metrics": {scope_name: normalize_path_value(path_obj) for scope_name, path_obj in canonical_metric_paths.items()},
            "supplemental_metrics": {
                "attack_quality_metrics_path": normalize_path_value(cast(Path, pw04_paths["attack_quality_metrics_path"])),
            },
            "tables": {
                "main_metrics_summary_csv_path": normalize_path_value(main_metrics_summary_path),
                "attack_family_summary_paper_csv_path": normalize_path_value(attack_family_summary_paper_path),
                "attack_condition_summary_paper_csv_path": normalize_path_value(attack_condition_summary_paper_path),
                "rescue_metrics_summary_csv_path": normalize_path_value(rescue_metrics_summary_path),
                "bootstrap_confidence_intervals_csv_path": normalize_path_value(bootstrap_csv_path),
            },
            "figures": {key_name: normalize_path_value(path_obj) for key_name, path_obj in figure_paths.items()},
            "bootstrap": {
                "bootstrap_confidence_intervals_path": normalize_path_value(bootstrap_json_path),
                "bootstrap_confidence_intervals_csv_path": normalize_path_value(bootstrap_csv_path),
            },
            "tail": {key_name: normalize_path_value(path_obj) for key_name, path_obj in tail_paths.items()},
        },
        "semantics": dict(PAPER_SCOPE_SEMANTICS),
        "clean_source_paths": {scope_name: normalize_path_value(path_obj) for scope_name, path_obj in clean_paths.items()},
        "attack_source_paths": {
            "content_chain": normalize_path_value(cast(Path, pw04_paths["formal_attack_final_decision_metrics_path"])),
            "event_attestation": normalize_path_value(cast(Path, pw04_paths["formal_attack_attestation_metrics_path"])),
            "system_final": normalize_path_value(cast(Path, pw04_paths["derived_attack_union_metrics_path"])),
        },
        "compatibility_policy": {
            "append_only": True,
            "legacy_artifacts_preserved": True,
            "paper_facing_scope_names": list(PAPER_SCOPE_ORDER),
            "legacy_scope_names_visibility": "legacy scope names are restricted to registry and compatibility fields; paper-facing tables and figures use canonical scope names only",
        },
    }
    paper_metric_registry_path = metrics_root / PAPER_METRIC_REGISTRY_FILE_NAME
    write_json_atomic(paper_metric_registry_path, paper_metric_registry_payload)

    return {
        "paper_scope_registry_path": normalize_path_value(paper_metric_registry_path),
        "canonical_metrics_paths": {scope_name: normalize_path_value(path_obj) for scope_name, path_obj in canonical_metric_paths.items()},
        "attack_quality_metrics_path": normalize_path_value(cast(Path, pw04_paths["attack_quality_metrics_path"])),
        "paper_tables_paths": {
            "main_metrics_summary_csv_path": normalize_path_value(main_metrics_summary_path),
            "attack_family_summary_paper_csv_path": normalize_path_value(attack_family_summary_paper_path),
            "attack_condition_summary_paper_csv_path": normalize_path_value(attack_condition_summary_paper_path),
            "rescue_metrics_summary_csv_path": normalize_path_value(rescue_metrics_summary_path),
        },
        "paper_figures_paths": {key_name: normalize_path_value(path_obj) for key_name, path_obj in figure_paths.items()},
        "bootstrap_confidence_intervals_path": normalize_path_value(bootstrap_json_path),
        "bootstrap_confidence_intervals_csv_path": normalize_path_value(bootstrap_csv_path),
        "tail_estimation_paths": {key_name: normalize_path_value(path_obj) for key_name, path_obj in tail_paths.items()},
        "paper_exports_completed": True,
    }