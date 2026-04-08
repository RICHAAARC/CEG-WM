"""
File purpose: Build append-only PW02 operating, quality, and payload metric exports.
Module type: Semi-general module
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from main.evaluation import workflow_inputs as eval_workflow_inputs
from scripts.notebook_runtime_common import ensure_directory, normalize_path_value, utc_now_iso, write_json_atomic


PAPER_SCOPE_ORDER: Tuple[str, ...] = ("content_chain", "event_attestation", "system_final")
OPERATING_SCOPE_ORDER: Tuple[str, ...] = PAPER_SCOPE_ORDER + ("system_final_auxiliary",)
ANALYSIS_KEY_TO_SCOPE = {
    "content": "content_chain",
    "attestation": "event_attestation",
}
ROC_FILE_NAMES = {
    "content_chain": "roc_curve_content_chain.json",
    "event_attestation": "roc_curve_event_attestation.json",
    "system_final": "roc_curve_system_final.json",
    "system_final_auxiliary": "roc_curve_system_final_auxiliary.json",
}
TARGET_FPRS: Tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5)
CONTENT_SCORE_NAME = "content_chain_score"
EVENT_ATTESTATION_SCORE_NAME = "event_attestation_score"
SYSTEM_FINAL_AUXILIARY_SCOPE = "system_final_auxiliary"
SYSTEM_FINAL_AUXILIARY_SCORE_NAME = "system_final_auxiliary_score"
SYSTEM_FINAL_AUXILIARY_DECISION_THRESHOLD = 0.0
SYSTEM_FINAL_AUXILIARY_SEMANTICS_FILE_NAME = "system_final_auxiliary_operating_semantics.json"
SYSTEM_FINAL_OPERATING_REASON = "missing scalar score chain for threshold sweep; only derived point metrics are available"
QUALITY_SCOPE_UNAVAILABLE_REASON = "quality payload is only defined for clean content-chain image pairs in current workflow"
LPIPS_UNAVAILABLE_REASON = "requires additional model dependency or upstream implementation"
CLIP_UNAVAILABLE_REASON = "requires frozen quality model identity and bootstrap contract"
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


def _coerce_finite_float(value: Any) -> float | None:
    """
    功能：将候选值解析为有限浮点数。

    Coerce a candidate value into a finite float.

    Args:
        value: Candidate scalar value.

    Returns:
        Finite float when available; otherwise None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    if isinstance(value, str) and value.strip():
        try:
            value_float = float(value.strip())
        except ValueError:
            return None
        return value_float if math.isfinite(value_float) else None
    return None


def _load_records_from_records_summary(records_summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    功能：从 records summary 装载已写出的 prepare record。

    Load prepared record payloads from one records-summary mapping.

    Args:
        records_summary: Records summary mapping.

    Returns:
        Loaded prepared record payloads.
    """
    if not isinstance(records_summary, Mapping):
        raise TypeError("records_summary must be Mapping")

    records_node = records_summary.get("records")
    if not isinstance(records_node, list):
        raise ValueError("records_summary.records must be list")

    payloads: List[Dict[str, Any]] = []
    for record_node in cast(List[object], records_node):
        if not isinstance(record_node, Mapping):
            raise ValueError("records_summary.records items must be mappings")
        record_path_value = record_node.get("record_path")
        if not isinstance(record_path_value, str) or not record_path_value.strip():
            raise ValueError("records_summary.records record_path must be non-empty str")
        payloads.append(
            _load_required_json_dict(
                Path(record_path_value).expanduser().resolve(),
                "PW02 prepared evaluate record",
            )
        )
    return payloads


def _resolve_threshold_export_payload(stage_root: Path, threshold_dir_name: str) -> Dict[str, Any]:
    """
    功能：读取 PW02 顶层 threshold export。

    Load one top-level PW02 threshold export payload.

    Args:
        stage_root: PW02 export root.
        threshold_dir_name: Threshold export directory token.

    Returns:
        Threshold export payload.
    """
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")
    if not isinstance(threshold_dir_name, str) or not threshold_dir_name:
        raise TypeError("threshold_dir_name must be non-empty str")

    threshold_export_path = stage_root / "thresholds" / threshold_dir_name / "thresholds.json"
    return _load_required_json_dict(threshold_export_path, f"PW02 threshold export {threshold_dir_name}")


def _compute_auc_from_points(points: Sequence[Tuple[float, float, float | None]]) -> float | None:
    """
    功能：基于经验 ROC 点计算梯形 AUC。

    Compute trapezoidal AUC from ordered ROC points.

    Args:
        points: Ordered ROC points.

    Returns:
        AUC value when points are available; otherwise None.
    """
    if not isinstance(points, Sequence):
        raise TypeError("points must be Sequence")
    if len(points) < 2:
        return None

    auc_value = 0.0
    for index in range(1, len(points)):
        previous_fpr, previous_tpr, _ = points[index - 1]
        current_fpr, current_tpr, _ = points[index]
        auc_value += (current_fpr - previous_fpr) * (previous_tpr + current_tpr) * 0.5
    return float(auc_value)


def _build_system_final_auxiliary_operating_payload(
    *,
    family_id: str,
    stage_root: Path,
    operating_metrics_dir: Path,
    score_runs: Mapping[str, Mapping[str, Any]],
) -> Tuple[Path, Dict[str, Any]]:
    """
    功能：构造 system_final auxiliary 的 clean-side operating 语义与证明工件。

    Build the clean-side operating semantics artifact for the auxiliary
    system_final scalar chain.

    Args:
        family_id: Family identifier.
        stage_root: PW02 export root.
        operating_metrics_dir: PW02 operating metrics directory.
        score_runs: Score-run summary mapping keyed by score name.

    Returns:
        Tuple of (artifact_path, semantics_payload).
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")
    if not isinstance(operating_metrics_dir, Path):
        raise TypeError("operating_metrics_dir must be Path")
    if not isinstance(score_runs, Mapping):
        raise TypeError("score_runs must be Mapping")

    content_score_run = score_runs.get(CONTENT_SCORE_NAME)
    if not isinstance(content_score_run, Mapping):
        raise ValueError("PW02 score_runs missing content_chain_score run")

    content_threshold_export = _resolve_threshold_export_payload(stage_root, "content")
    attestation_threshold_export = _resolve_threshold_export_payload(stage_root, "attestation")
    content_thresholds_artifact = _extract_mapping(content_threshold_export.get("thresholds_artifact"))
    attestation_thresholds_artifact = _extract_mapping(attestation_threshold_export.get("thresholds_artifact"))
    content_threshold_value = _extract_float(content_thresholds_artifact, "threshold_value")
    attestation_threshold_value = _extract_float(attestation_thresholds_artifact, "threshold_value")
    if content_threshold_value is None:
        raise ValueError("PW02 content threshold export missing thresholds_artifact.threshold_value")
    if attestation_threshold_value is None:
        raise ValueError("PW02 attestation threshold export missing thresholds_artifact.threshold_value")

    records_summary = content_score_run.get("evaluate_inputs")
    if not isinstance(records_summary, Mapping):
        raise ValueError("PW02 content score run missing evaluate_inputs records summary")
    prepared_records = _load_records_from_records_summary(cast(Mapping[str, Any], records_summary))

    score_rows: List[Dict[str, Any]] = []
    mismatch_count = 0
    auxiliary_positive_count = 0
    auxiliary_positive_positive_count = 0
    auxiliary_positive_negative_count = 0
    positive_count = 0
    negative_count = 0

    for record in prepared_records:
        label_value = record.get("label")
        if not isinstance(label_value, bool):
            raise ValueError("PW02 prepared record missing bool label for system_final auxiliary analysis")

        content_score, content_score_source = eval_workflow_inputs._resolve_content_score_source(record)
        event_attestation_score, event_attestation_score_source = eval_workflow_inputs._resolve_event_attestation_score_source(record)
        if content_score is None:
            raise ValueError(
                f"PW02 auxiliary system_final missing content score: event_id={record.get('paper_workflow_event_id')}"
            )
        if event_attestation_score is None:
            raise ValueError(
                "PW02 auxiliary system_final missing event attestation score: "
                f"event_id={record.get('paper_workflow_event_id')}"
            )

        content_margin = float(content_score - content_threshold_value)
        event_attestation_margin = float(event_attestation_score - attestation_threshold_value)
        auxiliary_score = float(max(content_margin, event_attestation_margin))
        derived_union_positive = bool(content_margin >= 0.0 or event_attestation_margin >= 0.0)
        auxiliary_positive = bool(auxiliary_score >= SYSTEM_FINAL_AUXILIARY_DECISION_THRESHOLD)
        if auxiliary_positive != derived_union_positive:
            mismatch_count += 1

        if label_value:
            positive_count += 1
        else:
            negative_count += 1
        if auxiliary_positive:
            auxiliary_positive_count += 1
            if label_value:
                auxiliary_positive_positive_count += 1
            else:
                auxiliary_positive_negative_count += 1

        score_rows.append(
            {
                "event_id": record.get("paper_workflow_event_id"),
                "sample_role": record.get("sample_role"),
                "label": label_value,
                "content_score": float(content_score),
                "content_score_source": content_score_source,
                "event_attestation_score": float(event_attestation_score),
                "event_attestation_score_source": event_attestation_score_source,
                "content_margin": content_margin,
                "event_attestation_margin": event_attestation_margin,
                "system_final_auxiliary_score": auxiliary_score,
                "derived_union_positive": derived_union_positive,
                "auxiliary_positive": auxiliary_positive,
            }
        )

    if mismatch_count > 0:
        raise ValueError(
            "PW02 auxiliary system_final score does not preserve derived union semantics: "
            f"mismatch_count={mismatch_count}"
        )

    ordered_scores = sorted({float(row[SYSTEM_FINAL_AUXILIARY_SCORE_NAME]) for row in score_rows}, reverse=True)
    roc_fpr: List[float] = []
    roc_tpr: List[float] = []
    roc_thresholds: List[float] = []
    for threshold_value in ordered_scores:
        predicted_positive_rows = [
            row for row in score_rows if float(row[SYSTEM_FINAL_AUXILIARY_SCORE_NAME]) >= float(threshold_value)
        ]
        tp_count = sum(1 for row in predicted_positive_rows if bool(row["label"]))
        fp_count = sum(1 for row in predicted_positive_rows if not bool(row["label"]))
        roc_thresholds.append(float(threshold_value))
        roc_tpr.append(float(tp_count / positive_count) if positive_count > 0 else 0.0)
        roc_fpr.append(float(fp_count / negative_count) if negative_count > 0 else 0.0)

    augmented_points = _build_augmented_roc_points(roc_fpr, roc_tpr, roc_thresholds)
    auc_value = _compute_auc_from_points(augmented_points)
    tpr_at_zero = float(auxiliary_positive_positive_count / positive_count) if positive_count > 0 else None
    fpr_at_zero = float(auxiliary_positive_negative_count / negative_count) if negative_count > 0 else None
    semantics_path = operating_metrics_dir / SYSTEM_FINAL_AUXILIARY_SEMANTICS_FILE_NAME
    payload = {
        "artifact_type": "paper_workflow_pw02_system_final_auxiliary_operating_semantics",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": SYSTEM_FINAL_AUXILIARY_SCOPE,
        "score_name": SYSTEM_FINAL_AUXILIARY_SCORE_NAME,
        "canonical": False,
        "analysis_only": True,
        "status": "ok",
        "score_definition": "max(content_chain_score - content_threshold, event_attestation_score - event_attestation_threshold)",
        "score_directionality": "larger margin indicates stronger evidence that at least one canonical branch exceeds its bound threshold",
        "decision_operator": ">=",
        "decision_threshold": SYSTEM_FINAL_AUXILIARY_DECISION_THRESHOLD,
        "threshold_binding": {
            "content_threshold_export_path": normalize_path_value(stage_root / "thresholds" / "content" / "thresholds.json"),
            "event_attestation_threshold_export_path": normalize_path_value(stage_root / "thresholds" / "attestation" / "thresholds.json"),
            "content_threshold_id": content_thresholds_artifact.get("threshold_id"),
            "event_attestation_threshold_id": attestation_thresholds_artifact.get("threshold_id"),
            "content_threshold_value": content_threshold_value,
            "event_attestation_threshold_value": attestation_threshold_value,
        },
        "input_scores": {
            "content_score_name": CONTENT_SCORE_NAME,
            "event_attestation_score_name": EVENT_ATTESTATION_SCORE_NAME,
        },
        "decision_equivalence": {
            "target_scope": "system_final",
            "target_semantics": "derived_system_union_metrics",
            "verified_record_count": len(score_rows),
            "mismatch_count": mismatch_count,
            "status": "exact_match",
            "proof_statement": "auxiliary_score >= 0 iff content_chain_score >= content_threshold or event_attestation_score >= event_attestation_threshold",
        },
        "source_records_glob": records_summary.get("records_glob"),
        "operating_metrics": {
            "score_name": SYSTEM_FINAL_AUXILIARY_SCORE_NAME,
            "n_pos": positive_count,
            "n_neg": negative_count,
            "accepted_count": len(score_rows),
            "threshold_value": SYSTEM_FINAL_AUXILIARY_DECISION_THRESHOLD,
            "tpr_at_threshold_zero": tpr_at_zero,
            "fpr_at_threshold_zero": fpr_at_zero,
            "auxiliary_positive_count": auxiliary_positive_count,
            "auxiliary_positive_positive_count": auxiliary_positive_positive_count,
            "auxiliary_positive_negative_count": auxiliary_positive_negative_count,
        },
        "roc_auc": {
            "auc": auc_value,
            "roc_curve_points": len(roc_fpr),
            "fpr": roc_fpr,
            "tpr": roc_tpr,
            "thresholds": roc_thresholds,
        },
    }
    write_json_atomic(semantics_path, payload)
    return semantics_path, payload


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
    score_runs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：为 PW02 构建独立 operating、quality 与 payload 导出目录。

    Build append-only PW02 operating, quality, and payload export artifacts.

    Args:
        family_id: Family identifier.
        stage_root: PW02 export root.
        clean_score_analysis_exports: Mapping from analysis key to analysis JSON path.
        score_runs: In-memory PW02 score-run summaries keyed by score name.

    Returns:
        Summary mapping of generated directories and files.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")
    if not isinstance(clean_score_analysis_exports, Mapping):
        raise TypeError("clean_score_analysis_exports must be Mapping")
    if not isinstance(score_runs, Mapping):
        raise TypeError("score_runs must be Mapping")

    operating_metrics_dir = ensure_directory(stage_root / "operating_metrics")
    quality_dir = ensure_directory(stage_root / "quality")
    payload_dir = ensure_directory(stage_root / "payload")

    analysis_payloads: Dict[str, Dict[str, Any]] = {}
    analysis_source_paths: Dict[str, str] = {}
    for analysis_key, scope_name in ANALYSIS_KEY_TO_SCOPE.items():
        analysis_path_value = clean_score_analysis_exports.get(analysis_key)
        if not isinstance(analysis_path_value, str) or not analysis_path_value.strip():
            continue
        analysis_payload = _load_required_json_dict(
            Path(analysis_path_value).expanduser().resolve(),
            f"PW02 clean score analysis {analysis_key}",
        )
        analysis_payloads[scope_name] = analysis_payload
        analysis_source_paths[scope_name] = normalize_path_value(Path(analysis_path_value).expanduser().resolve())

    system_final_auxiliary_semantics_path, system_final_auxiliary_payload = _build_system_final_auxiliary_operating_payload(
        family_id=family_id,
        stage_root=stage_root,
        operating_metrics_dir=operating_metrics_dir,
        score_runs=score_runs,
    )
    analysis_payloads[SYSTEM_FINAL_AUXILIARY_SCOPE] = system_final_auxiliary_payload
    analysis_source_paths[SYSTEM_FINAL_AUXILIARY_SCOPE] = normalize_path_value(system_final_auxiliary_semantics_path)

    roc_curve_paths: Dict[str, str] = {}
    auc_rows: List[Dict[str, Any]] = []
    eer_rows: List[Dict[str, Any]] = []
    tpr_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []

    for scope_name in OPERATING_SCOPE_ORDER:
        roc_output_path = operating_metrics_dir / ROC_FILE_NAMES[scope_name]
        analysis_payload = analysis_payloads.get(scope_name)
        if analysis_payload is None:
            if scope_name != "system_final":
                raise ValueError(f"PW02 missing operating analysis payload for scope: {scope_name}")
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
            if scope_name in PAPER_SCOPE_ORDER:
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
                        "mean_lpips": None,
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
            "source_analysis_path": analysis_source_paths.get(scope_name),
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

        if scope_name in PAPER_SCOPE_ORDER:
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
                    "mean_lpips": quality_payload.get("mean_lpips"),
                    "lpips_status": quality_payload.get("lpips_status", "not_available"),
                    "lpips_reason": quality_payload.get("lpips_reason", LPIPS_UNAVAILABLE_REASON),
                    "clip_status": quality_payload.get("clip_status", "not_available"),
                    "clip_reason": quality_payload.get("clip_reason", CLIP_UNAVAILABLE_REASON),
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
            "scopes": list(OPERATING_SCOPE_ORDER),
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
            "scopes": list(OPERATING_SCOPE_ORDER),
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
            "mean_lpips",
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

    analysis_only_artifact_paths = {
        "pw02_system_final_auxiliary_operating_semantics": normalize_path_value(system_final_auxiliary_semantics_path),
        "pw02_system_final_auxiliary_roc_curve": normalize_path_value(operating_metrics_dir / ROC_FILE_NAMES[SYSTEM_FINAL_AUXILIARY_SCOPE]),
        "pw02_operating_auc_summary": normalize_path_value(auc_summary_path),
        "pw02_operating_eer_summary": normalize_path_value(eer_summary_path),
        "pw02_operating_tpr_at_target_fpr_summary": normalize_path_value(tpr_summary_path),
    }

    return {
        "operating_metrics_dir": normalize_path_value(operating_metrics_dir),
        "quality_metrics_dir": normalize_path_value(quality_dir),
        "payload_metrics_dir": normalize_path_value(payload_dir),
        "roc_curve_paths": roc_curve_paths,
        "system_final_auxiliary_operating_semantics_path": normalize_path_value(system_final_auxiliary_semantics_path),
        "auc_summary_path": normalize_path_value(auc_summary_path),
        "eer_summary_path": normalize_path_value(eer_summary_path),
        "tpr_at_target_fpr_summary_path": normalize_path_value(tpr_summary_path),
        "quality_metrics_summary_csv_path": normalize_path_value(quality_summary_csv_path),
        "quality_metrics_summary_json_path": normalize_path_value(quality_summary_json_path),
        "payload_clean_summary_path": normalize_path_value(payload_clean_summary_path),
        "analysis_only_artifact_paths": analysis_only_artifact_paths,
    }