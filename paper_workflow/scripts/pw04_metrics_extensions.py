"""
File purpose: Build append-only PW04 robustness, geometry, payload, and tradeoff exports.
Module type: Semi-general module
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from PIL import Image, ImageDraw

from scripts.notebook_runtime_common import ensure_directory, normalize_path_value, utc_now_iso, write_json_atomic


PAPER_SCOPE_ORDER: Tuple[str, ...] = ("content_chain", "event_attestation", "system_final")
SYSTEM_FINAL_AUXILIARY_SCOPE = "system_final_auxiliary"
SYSTEM_FINAL_AUXILIARY_SCORE_NAME = "system_final_auxiliary_score"
SYSTEM_FINAL_AUXILIARY_DECISION_THRESHOLD = 0.0
SCOPE_COLUMN_NAMES = {
    "content_chain": "content_chain_attack_tpr",
    "event_attestation": "event_attestation_attack_tpr",
    "system_final": "system_final_attack_tpr",
}
SEVERITY_NOT_AVAILABLE_REASON = "attack condition has no frozen family-local scalar severity rule in current protocol"
DEEP_GEOMETRY_NOT_AVAILABLE_REASON = "PW03 geometry sidecar is missing required diagnostics"
PAYLOAD_UNAVAILABLE_REASON = "missing upstream decoded bits / bit error sidecar"
SYSTEM_FINAL_AUXILIARY_ATTACK_SUMMARY_FILE_NAME = "system_final_auxiliary_attack_summary.json"
SYSTEM_FINAL_AUXILIARY_ATTACK_BY_FAMILY_FILE_NAME = "system_final_auxiliary_attack_by_family.csv"
SYSTEM_FINAL_AUXILIARY_ATTACK_BY_CONDITION_FILE_NAME = "system_final_auxiliary_attack_by_condition.csv"


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
    功能：将可选映射节点规范化为 dict。

    Normalize an optional mapping node to dict.

    Args:
        node: Candidate mapping node.

    Returns:
        Normalized dict mapping.
    """
    return dict(cast(Mapping[str, Any], node)) if isinstance(node, Mapping) else {}


def _parse_csv_float(row: Mapping[str, Any], key_name: str) -> float | None:
    """
    功能：从 CSV 行解析有限浮点值。

    Parse one finite floating-point value from a CSV row.

    Args:
        row: CSV row mapping.
        key_name: Field name.

    Returns:
        Parsed finite float or None.
    """
    value = row.get(key_name)
    if value in {None, "", "None"}:
        return None
    try:
        value_float = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def _parse_csv_int(row: Mapping[str, Any], key_name: str) -> int | None:
    """
    功能：从 CSV 行解析整数值。

    Parse one integer value from a CSV row.

    Args:
        row: CSV row mapping.
        key_name: Field name.

    Returns:
        Parsed integer or None.
    """
    value = row.get(key_name)
    if value in {None, "", "None"}:
        return None
    try:
        return int(float(cast(Any, value)))
    except (TypeError, ValueError):
        return None


def _read_csv_rows(input_path: Path) -> List[Dict[str, Any]]:
    """
    功能：读取 CSV 行。

    Read one CSV file into ordered rows.

    Args:
        input_path: CSV input path.

    Returns:
        Ordered CSV rows.
    """
    if not isinstance(input_path, Path):
        raise TypeError("input_path must be Path")
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(cast(Mapping[str, Any], row)) for row in csv.DictReader(handle)]


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


def _safe_mean(values: Sequence[float]) -> float | None:
    """
    功能：计算有限数值序列的均值。

    Compute the mean of a finite numeric sequence.

    Args:
        values: Numeric sequence.

    Returns:
        Mean value or None when empty.
    """
    if not isinstance(values, Sequence):
        raise TypeError("values must be Sequence")
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


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


def _build_condition_severity_lookup(
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    功能：从 attack event 行构造 condition 级 severity 元数据索引。

    Build one per-condition severity metadata lookup from attack event rows.

    Args:
        attack_event_rows: Materialized attack event rows.

    Returns:
        Condition-keyed severity metadata mapping.
    """
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    field_names = [
        "severity_status",
        "severity_reason",
        "severity_rule_version",
        "severity_axis_kind",
        "severity_directionality",
        "severity_source_param",
        "severity_scalarization",
        "severity_value",
        "severity_sort_value",
        "severity_label",
        "severity_level_index",
    ]
    lookup: Dict[str, Dict[str, Any]] = {}
    for attack_event_row in attack_event_rows:
        attack_condition_key = attack_event_row.get("attack_condition_key")
        if not isinstance(attack_condition_key, str) or not attack_condition_key:
            raise ValueError("attack_event_row missing attack_condition_key")
        severity_metadata = {
            field_name: attack_event_row.get(field_name)
            for field_name in field_names
        }
        existing_metadata = lookup.get(attack_condition_key)
        if existing_metadata is None:
            lookup[attack_condition_key] = severity_metadata
            continue
        if existing_metadata != severity_metadata:
            raise ValueError(
                "attack condition severity metadata must be identical across PW04 rows: "
                f"attack_condition_key={attack_condition_key}"
            )
    return lookup


def _summarize_severity_metadata(
    severity_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：汇总 family-local severity 元数据状态。

    Summarize family-local severity metadata for one row collection.

    Args:
        severity_rows: Condition-level severity metadata rows.

    Returns:
        Severity summary mapping.
    """
    if not isinstance(severity_rows, Sequence):
        raise TypeError("severity_rows must be Sequence")

    total_count = len(severity_rows)
    ok_rows = [row for row in severity_rows if row.get("severity_status") == "ok"]
    axis_kinds = sorted(
        {
            str(row.get("severity_axis_kind"))
            for row in ok_rows
            if isinstance(row.get("severity_axis_kind"), str) and row.get("severity_axis_kind")
        }
    )
    rule_versions = sorted(
        {
            str(row.get("severity_rule_version"))
            for row in ok_rows
            if isinstance(row.get("severity_rule_version"), str) and row.get("severity_rule_version")
        }
    )
    directionality_values = sorted(
        {
            str(row.get("severity_directionality"))
            for row in ok_rows
            if isinstance(row.get("severity_directionality"), str) and row.get("severity_directionality")
        }
    )
    label_pairs: List[Tuple[float, str]] = []
    for row in ok_rows:
        sort_value = _coerce_finite_float(row.get("severity_sort_value"))
        label_value = row.get("severity_label")
        if sort_value is None:
            continue
        if not isinstance(label_value, str) or not label_value:
            label_value = str(sort_value)
        label_pairs.append((sort_value, label_value))
    ordered_labels = [label for _, label in sorted(label_pairs, key=lambda item: item[0])]
    severity_values = [item[0] for item in label_pairs]

    if not ok_rows:
        reasons = [
            str(row.get("severity_reason"))
            for row in severity_rows
            if isinstance(row.get("severity_reason"), str) and row.get("severity_reason")
        ]
        return {
            "severity_level_status": "not_available",
            "severity_level_reason": reasons[0] if reasons else SEVERITY_NOT_AVAILABLE_REASON,
            "severity_axis_kind": None,
            "severity_rule_version": None,
            "severity_directionality": None,
            "severity_available_condition_count": 0,
            "severity_total_condition_count": total_count,
            "severity_min_sort_value": None,
            "severity_max_sort_value": None,
            "severity_label_sequence": None,
            "severity_canonical_global_axis": False,
        }

    summary_reason = None
    summary_status = "ok"
    if len(ok_rows) < total_count:
        summary_status = "partial"
        summary_reason = f"family-local severity frozen for {len(ok_rows)}/{total_count} attack conditions"

    return {
        "severity_level_status": summary_status,
        "severity_level_reason": summary_reason,
        "severity_axis_kind": axis_kinds[0] if len(axis_kinds) == 1 else json.dumps(axis_kinds, ensure_ascii=False),
        "severity_rule_version": rule_versions[0] if len(rule_versions) == 1 else json.dumps(rule_versions, ensure_ascii=False),
        "severity_directionality": (
            directionality_values[0]
            if len(directionality_values) == 1
            else json.dumps(directionality_values, ensure_ascii=False)
        ),
        "severity_available_condition_count": len(ok_rows),
        "severity_total_condition_count": total_count,
        "severity_min_sort_value": min(severity_values) if severity_values else None,
        "severity_max_sort_value": max(severity_values) if severity_values else None,
        "severity_label_sequence": json.dumps(ordered_labels, ensure_ascii=False) if ordered_labels else None,
        "severity_canonical_global_axis": False,
    }


def _summarize_boolean_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    key_name: str,
    unavailable_reason: str,
) -> Dict[str, Any]:
    """
    功能：汇总布尔型几何诊断字段。

    Summarize one boolean diagnostic signal across rows.

    Args:
        rows: Attack event rows.
        key_name: Boolean field name.
        unavailable_reason: Fallback reason when the signal is absent.

    Returns:
        Summary mapping with counts, rate, status, and reason.
    """
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be Sequence")
    if not isinstance(key_name, str) or not key_name:
        raise TypeError("key_name must be non-empty str")
    if not isinstance(unavailable_reason, str) or not unavailable_reason:
        raise TypeError("unavailable_reason must be non-empty str")

    available_values = [row.get(key_name) for row in rows if isinstance(row.get(key_name), bool)]
    explicit_reasons = sorted(
        {
            str(row.get(f"{key_name}_reason")).strip()
            for row in rows
            if isinstance(row.get(f"{key_name}_reason"), str) and str(row.get(f"{key_name}_reason")).strip()
        }
    )
    available_count = len(available_values)
    true_count = sum(1 for value in available_values if value is True)
    if available_count <= 0:
        return {
            f"{key_name}_available_count": 0,
            f"{key_name}_true_count": 0,
            f"{key_name}_true_rate": None,
            f"{key_name}_status": "not_available",
            f"{key_name}_reason": (
                explicit_reasons[0]
                if len(explicit_reasons) == 1
                else (json.dumps(explicit_reasons, ensure_ascii=False) if explicit_reasons else unavailable_reason)
            ),
        }

    status_value = "ok" if available_count == len(rows) else "partial"
    if status_value == "ok":
        reason_value = None
    else:
        reason_value = f"available for {available_count}/{len(rows)} attack events"
        if explicit_reasons:
            explicit_reason_text = (
                explicit_reasons[0]
                if len(explicit_reasons) == 1
                else json.dumps(explicit_reasons, ensure_ascii=False)
            )
            reason_value = f"{reason_value}; unavailable_reasons={explicit_reason_text}"
    return {
        f"{key_name}_available_count": available_count,
        f"{key_name}_true_count": true_count,
        f"{key_name}_true_rate": float(true_count / available_count),
        f"{key_name}_status": status_value,
        f"{key_name}_reason": reason_value,
    }


def _resolve_pw02_threshold_export_path(
    pw02_summary: Mapping[str, Any],
    family_root: Path,
    threshold_key: str,
) -> Path:
    """
    功能：解析 PW02 threshold export 路径。

    Resolve one PW02 threshold export path.

    Args:
        pw02_summary: PW02 summary payload.
        family_root: Paper workflow family root.
        threshold_key: Threshold key, one of {"content", "attestation"}.

    Returns:
        Resolved threshold export path.
    """
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(threshold_key, str) or threshold_key not in {"content", "attestation"}:
        raise ValueError("threshold_key must be one of {'content', 'attestation'}")

    threshold_exports = pw02_summary.get("threshold_exports")
    if isinstance(threshold_exports, Mapping):
        path_value = threshold_exports.get(threshold_key)
        if isinstance(path_value, str) and path_value.strip():
            return Path(path_value).expanduser().resolve()
    return (family_root / "exports" / "pw02" / "thresholds" / threshold_key / "thresholds.json").resolve()


def _build_system_final_auxiliary_attack_exports(
    *,
    family_id: str,
    family_root: Path,
    pw02_summary: Mapping[str, Any],
    attack_event_rows: Sequence[Mapping[str, Any]],
    robustness_dir: Path,
) -> Dict[str, Any]:
    """
    功能：构造 system_final auxiliary 的 attack-side analysis 导出。

    Build attack-side auxiliary analysis exports for the synthetic
    system_final scalar chain.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        pw02_summary: PW02 summary payload.
        attack_event_rows: Materialized attack event rows.
        robustness_dir: PW04 robustness directory.

    Returns:
        Mapping with summary payload, CSV rows, and written paths.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")
    if not isinstance(robustness_dir, Path):
        raise TypeError("robustness_dir must be Path")

    content_threshold_export_path = _resolve_pw02_threshold_export_path(pw02_summary, family_root, "content")
    attestation_threshold_export_path = _resolve_pw02_threshold_export_path(pw02_summary, family_root, "attestation")
    content_threshold_export = _load_required_json_dict(content_threshold_export_path, "PW02 content threshold export")
    attestation_threshold_export = _load_required_json_dict(attestation_threshold_export_path, "PW02 attestation threshold export")
    content_thresholds_artifact = _extract_mapping(content_threshold_export.get("thresholds_artifact"))
    attestation_thresholds_artifact = _extract_mapping(attestation_threshold_export.get("thresholds_artifact"))
    content_threshold_value = _coerce_finite_float(content_thresholds_artifact.get("threshold_value"))
    attestation_threshold_value = _coerce_finite_float(attestation_thresholds_artifact.get("threshold_value"))
    if content_threshold_value is None:
        raise ValueError("PW02 content threshold export missing thresholds_artifact.threshold_value")
    if attestation_threshold_value is None:
        raise ValueError("PW02 attestation threshold export missing thresholds_artifact.threshold_value")

    detailed_rows: List[Dict[str, Any]] = []
    mismatch_count = 0
    missing_content_score_count = 0
    missing_event_attestation_score_count = 0
    for attack_event_row in attack_event_rows:
        content_score = _coerce_finite_float(attack_event_row.get("content_score"))
        event_attestation_score = _coerce_finite_float(attack_event_row.get("event_attestation_score"))
        if content_score is None:
            missing_content_score_count += 1
        if event_attestation_score is None:
            missing_event_attestation_score_count += 1
        if content_score is None and event_attestation_score is None:
            raise ValueError(
                "PW04 auxiliary system_final missing both content_score and event_attestation_score: "
                f"attack_event_id={attack_event_row.get('attack_event_id')}"
            )

        content_margin = (
            float(content_score - content_threshold_value)
            if content_score is not None
            else float("-inf")
        )
        event_attestation_margin = (
            float(event_attestation_score - attestation_threshold_value)
            if event_attestation_score is not None
            else float("-inf")
        )
        auxiliary_score = float(max(content_margin, event_attestation_margin))
        auxiliary_positive = bool(auxiliary_score >= SYSTEM_FINAL_AUXILIARY_DECISION_THRESHOLD)
        formal_record = _extract_mapping(attack_event_row.get("formal_record"))
        derived_attack_union_positive = bool(formal_record.get("derived_attack_union_positive", False))
        if auxiliary_positive != derived_attack_union_positive:
            mismatch_count += 1

        detailed_rows.append(
            {
                "attack_event_id": attack_event_row.get("attack_event_id"),
                "parent_event_id": attack_event_row.get("parent_event_id"),
                "attack_family": attack_event_row.get("attack_family"),
                "attack_condition_key": attack_event_row.get("attack_condition_key"),
                "attack_config_name": attack_event_row.get("attack_config_name"),
                "content_score": content_score,
                "event_attestation_score": event_attestation_score,
                "content_margin": content_margin,
                "event_attestation_margin": event_attestation_margin,
                SYSTEM_FINAL_AUXILIARY_SCORE_NAME: auxiliary_score,
                "auxiliary_positive": auxiliary_positive,
                "derived_attack_union_positive": derived_attack_union_positive,
            }
        )

    if mismatch_count > 0:
        raise ValueError(
            "PW04 auxiliary system_final score does not preserve derived attack union semantics: "
            f"mismatch_count={mismatch_count}"
        )

    def _build_group_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        event_count = len(rows)
        parent_event_count = len(
            {
                str(row.get("parent_event_id"))
                for row in rows
                if isinstance(row.get("parent_event_id"), str) and row.get("parent_event_id")
            }
        )
        auxiliary_scores = [
            float(cast(float, row.get(SYSTEM_FINAL_AUXILIARY_SCORE_NAME)))
            for row in rows
            if isinstance(row.get(SYSTEM_FINAL_AUXILIARY_SCORE_NAME), (int, float))
        ]
        auxiliary_positive_count = sum(1 for row in rows if bool(row.get("auxiliary_positive")))
        derived_positive_count = sum(1 for row in rows if bool(row.get("derived_attack_union_positive")))
        return {
            "event_count": event_count,
            "parent_event_count": parent_event_count,
            "system_final_auxiliary_attack_tpr": float(auxiliary_positive_count / event_count) if event_count > 0 else None,
            "system_final_auxiliary_score_mean": _safe_mean(auxiliary_scores),
            "system_final_auxiliary_score_min": min(auxiliary_scores) if auxiliary_scores else None,
            "system_final_auxiliary_score_max": max(auxiliary_scores) if auxiliary_scores else None,
            "auxiliary_positive_count": auxiliary_positive_count,
            "derived_attack_union_positive_count": derived_positive_count,
            "consistency_mismatch_count": sum(
                1 for row in rows if bool(row.get("auxiliary_positive")) != bool(row.get("derived_attack_union_positive"))
            ),
            "decision_threshold": SYSTEM_FINAL_AUXILIARY_DECISION_THRESHOLD,
        }

    family_rows: List[Dict[str, Any]] = []
    grouped_family_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for row in detailed_rows:
        attack_family = row.get("attack_family")
        if not isinstance(attack_family, str) or not attack_family:
            raise ValueError("system_final auxiliary attack row missing attack_family")
        grouped_family_rows.setdefault(attack_family, []).append(row)
    for attack_family in sorted(grouped_family_rows):
        family_rows.append(
            {
                "attack_family": attack_family,
                **_build_group_summary(grouped_family_rows[attack_family]),
            }
        )

    condition_rows: List[Dict[str, Any]] = []
    grouped_condition_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for row in detailed_rows:
        attack_condition_key = row.get("attack_condition_key")
        if not isinstance(attack_condition_key, str) or not attack_condition_key:
            raise ValueError("system_final auxiliary attack row missing attack_condition_key")
        grouped_condition_rows.setdefault(attack_condition_key, []).append(row)
    for attack_condition_key in sorted(grouped_condition_rows):
        first_row = grouped_condition_rows[attack_condition_key][0]
        condition_rows.append(
            {
                "attack_condition_key": attack_condition_key,
                "attack_family": first_row.get("attack_family"),
                "attack_config_name": first_row.get("attack_config_name"),
                **_build_group_summary(grouped_condition_rows[attack_condition_key]),
            }
        )

    summary_payload = {
        "artifact_type": "paper_workflow_pw04_system_final_auxiliary_attack_summary",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": SYSTEM_FINAL_AUXILIARY_SCOPE,
        "score_name": SYSTEM_FINAL_AUXILIARY_SCORE_NAME,
        "canonical": False,
        "analysis_only": True,
        "score_definition": "max(content_chain_score - content_threshold, event_attestation_score - event_attestation_threshold)",
        "decision_operator": ">=",
        "decision_threshold": SYSTEM_FINAL_AUXILIARY_DECISION_THRESHOLD,
        "threshold_binding": {
            "content_threshold_export_path": normalize_path_value(content_threshold_export_path),
            "event_attestation_threshold_export_path": normalize_path_value(attestation_threshold_export_path),
            "content_threshold_id": content_thresholds_artifact.get("threshold_id"),
            "event_attestation_threshold_id": attestation_thresholds_artifact.get("threshold_id"),
            "content_threshold_value": content_threshold_value,
            "event_attestation_threshold_value": attestation_threshold_value,
        },
        "overall": {
            **_build_group_summary(detailed_rows),
            "missing_content_score_count": missing_content_score_count,
            "missing_event_attestation_score_count": missing_event_attestation_score_count,
            "consistency_status": "exact_match",
            "verified_record_count": len(detailed_rows),
        },
        "by_family": family_rows,
        "by_condition": condition_rows,
    }

    summary_path = robustness_dir / SYSTEM_FINAL_AUXILIARY_ATTACK_SUMMARY_FILE_NAME
    family_path = robustness_dir / SYSTEM_FINAL_AUXILIARY_ATTACK_BY_FAMILY_FILE_NAME
    condition_path = robustness_dir / SYSTEM_FINAL_AUXILIARY_ATTACK_BY_CONDITION_FILE_NAME
    write_json_atomic(summary_path, summary_payload)
    _write_csv_rows(
        family_path,
        [
            "attack_family",
            "event_count",
            "parent_event_count",
            "system_final_auxiliary_attack_tpr",
            "system_final_auxiliary_score_mean",
            "system_final_auxiliary_score_min",
            "system_final_auxiliary_score_max",
            "auxiliary_positive_count",
            "derived_attack_union_positive_count",
            "consistency_mismatch_count",
            "decision_threshold",
        ],
        family_rows,
    )
    _write_csv_rows(
        condition_path,
        [
            "attack_condition_key",
            "attack_family",
            "attack_config_name",
            "event_count",
            "parent_event_count",
            "system_final_auxiliary_attack_tpr",
            "system_final_auxiliary_score_mean",
            "system_final_auxiliary_score_min",
            "system_final_auxiliary_score_max",
            "auxiliary_positive_count",
            "derived_attack_union_positive_count",
            "consistency_mismatch_count",
            "decision_threshold",
        ],
        condition_rows,
    )
    return {
        "system_final_auxiliary_attack_summary_path": normalize_path_value(summary_path),
        "system_final_auxiliary_attack_by_family_path": normalize_path_value(family_path),
        "system_final_auxiliary_attack_by_condition_path": normalize_path_value(condition_path),
        "system_final_auxiliary_attack_summary": summary_payload,
    }


def _resolve_pw02_quality_summary_paths(
    family_root: Path,
    pw02_summary: Mapping[str, Any],
) -> Tuple[Path, Path]:
    """
    功能：解析 PW02 clean quality summary 路径。

    Resolve the PW02 clean quality summary CSV and JSON paths.

    Args:
        family_root: Family root path.
        pw02_summary: PW02 summary payload.

    Returns:
        Tuple of (csv_path, json_path).
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")

    csv_path_value = pw02_summary.get("quality_metrics_summary_csv_path")
    json_path_value = pw02_summary.get("quality_metrics_summary_json_path")
    csv_path = (
        Path(str(csv_path_value)).expanduser().resolve()
        if isinstance(csv_path_value, str) and csv_path_value.strip()
        else (family_root / "exports" / "pw02" / "quality" / "quality_metrics_summary.csv").resolve()
    )
    json_path = (
        Path(str(json_path_value)).expanduser().resolve()
        if isinstance(json_path_value, str) and json_path_value.strip()
        else (family_root / "exports" / "pw02" / "quality" / "quality_metrics_summary.json").resolve()
    )
    return csv_path, json_path


def _build_robustness_rows(
    main_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    功能：构造 robustness 三类 CSV 行。

    Build robustness curve, macro summary, and worst-case rows.

    Args:
        main_rows: Canonical main summary rows.
        family_rows: Paper-facing family rows.
        condition_rows: Paper-facing condition rows.
        attack_event_rows: Materialized attack event rows.

    Returns:
        Tuple of curve rows, macro rows, and worst-case rows.
    """
    curve_rows: List[Dict[str, Any]] = []
    macro_rows: List[Dict[str, Any]] = []
    worst_case_rows: List[Dict[str, Any]] = []
    condition_severity_lookup = _build_condition_severity_lookup(attack_event_rows)

    family_rows_by_name: Dict[str, Dict[str, Any]] = {
        str(row.get("attack_family")): dict(cast(Mapping[str, Any], row))
        for row in family_rows
        if isinstance(row.get("attack_family"), str) and row.get("attack_family")
    }
    main_rows_by_scope = {
        str(row.get("scope")): dict(cast(Mapping[str, Any], row))
        for row in main_rows
        if isinstance(row.get("scope"), str) and row.get("scope")
    }

    for scope_name in PAPER_SCOPE_ORDER:
        column_name = SCOPE_COLUMN_NAMES[scope_name]
        family_scope_values: List[float] = []
        condition_scope_rows: List[Tuple[float, Dict[str, Any]]] = []
        for family_name in sorted(family_rows_by_name):
            family_row = family_rows_by_name[family_name]
            mean_attack_tpr = _parse_csv_float(family_row, column_name)
            if mean_attack_tpr is not None:
                family_scope_values.append(mean_attack_tpr)
            matching_condition_rows = [
                dict(cast(Mapping[str, Any], row))
                for row in condition_rows
                if str(row.get("attack_family")) == family_name
            ]
            severity_rows = [
                condition_severity_lookup[str(row.get("attack_condition_key"))]
                for row in matching_condition_rows
                if isinstance(row.get("attack_condition_key"), str)
                and row.get("attack_condition_key") in condition_severity_lookup
            ]
            severity_summary = _summarize_severity_metadata(severity_rows)
            condition_values = [
                cast(float, _parse_csv_float(row, column_name))
                for row in matching_condition_rows
                if _parse_csv_float(row, column_name) is not None
            ]
            min_attack_tpr = min(condition_values) if condition_values else mean_attack_tpr
            curve_rows.append(
                {
                    "scope": scope_name,
                    "attack_family": family_name,
                    "event_count": _parse_csv_int(family_row, "event_count"),
                    "parent_event_count": _parse_csv_int(family_row, "parent_event_count"),
                    "mean_attack_tpr": mean_attack_tpr,
                    "min_attack_tpr": min_attack_tpr,
                    "macro_avg_attack_tpr": None,
                    **severity_summary,
                }
            )
            for matching_condition_row in matching_condition_rows:
                condition_value = _parse_csv_float(matching_condition_row, column_name)
                if condition_value is not None:
                    condition_scope_rows.append((condition_value, matching_condition_row))

        macro_avg_attack_tpr = _safe_mean(family_scope_values)
        overall_attack_tpr = _parse_csv_float(main_rows_by_scope.get(scope_name, {}), "attack_tpr")
        worst_case_attack_tpr = min((value for value, _ in condition_scope_rows), default=None)
        macro_severity_summary = _summarize_severity_metadata(list(condition_severity_lookup.values()))
        macro_rows.append(
            {
                "scope": scope_name,
                "overall_attack_tpr": overall_attack_tpr,
                "macro_avg_attack_tpr": macro_avg_attack_tpr,
                "worst_case_attack_tpr": worst_case_attack_tpr,
                "family_count": len(family_rows_by_name),
                "condition_count": len(condition_scope_rows),
                **macro_severity_summary,
            }
        )
        if condition_scope_rows:
            worst_case_value, worst_case_row = min(condition_scope_rows, key=lambda item: item[0])
            worst_case_condition_key = worst_case_row.get("attack_condition_key")
            worst_case_severity = _summarize_severity_metadata(
                [condition_severity_lookup[str(worst_case_condition_key)]]
                if isinstance(worst_case_condition_key, str) and worst_case_condition_key in condition_severity_lookup
                else []
            )
            worst_case_rows.append(
                {
                    "scope": scope_name,
                    "attack_family": worst_case_row.get("attack_family"),
                    "attack_condition_key": worst_case_row.get("attack_condition_key"),
                    "attack_config_name": worst_case_row.get("attack_config_name"),
                    "event_count": _parse_csv_int(worst_case_row, "event_count"),
                    "parent_event_count": _parse_csv_int(worst_case_row, "parent_event_count"),
                    "worst_case_attack_tpr": worst_case_value,
                    **worst_case_severity,
                    "severity_label": (
                        condition_severity_lookup[str(worst_case_condition_key)].get("severity_label")
                        if isinstance(worst_case_condition_key, str) and worst_case_condition_key in condition_severity_lookup
                        else None
                    ),
                }
            )
        else:
            worst_case_rows.append(
                {
                    "scope": scope_name,
                    "attack_family": None,
                    "attack_condition_key": None,
                    "attack_config_name": None,
                    "event_count": None,
                    "parent_event_count": None,
                    "worst_case_attack_tpr": None,
                    **_summarize_severity_metadata([]),
                    "severity_label": None,
                }
            )

    macro_by_scope = {row["scope"]: row for row in macro_rows}
    for curve_row in curve_rows:
        scope_name = str(curve_row["scope"])
        curve_row["macro_avg_attack_tpr"] = macro_by_scope[scope_name]["macro_avg_attack_tpr"]
    return curve_rows, macro_rows, worst_case_rows


def _collect_geometry_rows(
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    功能：从 formal record 聚合稳定几何诊断字段。

    Aggregate stable geometry diagnostic fields from materialized formal records.

    Args:
        attack_event_rows: Materialized attack event rows.

    Returns:
        Tuple of per-family rows and overall summary row.
    """
    grouped_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for attack_event_row in attack_event_rows:
        attack_family = attack_event_row.get("attack_family")
        if not isinstance(attack_family, str) or not attack_family:
            raise ValueError("attack_event_row missing attack_family")
        grouped_rows.setdefault(attack_family, []).append(attack_event_row)

    def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        geo_rescue_eligible_count = 0
        geo_rescue_applied_count = 0
        geo_helped_positive_count = 0
        geo_not_used_count = 0
        geo_not_used_reason_counts: Dict[str, int] = {}
        sync_status_counts: Dict[str, int] = {}
        geometry_failure_reason_counts: Dict[str, int] = {}
        for row in rows:
            formal_record = _extract_mapping(row.get("formal_record"))
            attestation_payload = _extract_mapping(formal_record.get("attestation"))
            image_evidence_payload = _extract_mapping(attestation_payload.get("image_evidence_result"))
            geometry_diagnostics = _extract_mapping(row.get("geometry_diagnostics"))
            geo_rescue_eligible = bool(image_evidence_payload.get("geo_rescue_eligible", False))
            geo_rescue_applied = bool(image_evidence_payload.get("geo_rescue_applied", False))
            geo_not_used_reason = image_evidence_payload.get("geo_not_used_reason")
            sync_status = geometry_diagnostics.get("sync_status")
            geometry_failure_reason = geometry_diagnostics.get("geometry_failure_reason")
            if geo_rescue_eligible:
                geo_rescue_eligible_count += 1
            if geo_rescue_applied:
                geo_rescue_applied_count += 1
            if geo_rescue_applied and bool(formal_record.get("derived_attack_union_positive", False)):
                geo_helped_positive_count += 1
            if isinstance(geo_not_used_reason, str) and geo_not_used_reason:
                geo_not_used_count += 1
                geo_not_used_reason_counts[geo_not_used_reason] = geo_not_used_reason_counts.get(geo_not_used_reason, 0) + 1
            if isinstance(sync_status, str) and sync_status:
                sync_status_counts[sync_status] = sync_status_counts.get(sync_status, 0) + 1
            if isinstance(geometry_failure_reason, str) and geometry_failure_reason:
                geometry_failure_reason_counts[geometry_failure_reason] = (
                    geometry_failure_reason_counts.get(geometry_failure_reason, 0) + 1
                )

        sync_success_summary = _summarize_boolean_diagnostic(rows, "sync_success", DEEP_GEOMETRY_NOT_AVAILABLE_REASON)
        inverse_summary = _summarize_boolean_diagnostic(
            rows,
            "inverse_transform_success",
            DEEP_GEOMETRY_NOT_AVAILABLE_REASON,
        )
        anchor_summary = _summarize_boolean_diagnostic(
            rows,
            "attention_anchor_available",
            DEEP_GEOMETRY_NOT_AVAILABLE_REASON,
        )
        return {
            "geo_rescue_eligible_count": geo_rescue_eligible_count,
            "geo_rescue_applied_count": geo_rescue_applied_count,
            "geo_helped_positive_count": geo_helped_positive_count,
            "geo_not_used_count": geo_not_used_count,
            "geo_not_used_reason_counts": json.dumps(dict(sorted(geo_not_used_reason_counts.items())), ensure_ascii=False, sort_keys=True),
            "sync_status_counts": json.dumps(dict(sorted(sync_status_counts.items())), ensure_ascii=False, sort_keys=True),
            "geometry_failure_reason_counts": json.dumps(
                dict(sorted(geometry_failure_reason_counts.items())),
                ensure_ascii=False,
                sort_keys=True,
            ),
            **sync_success_summary,
            **inverse_summary,
            **anchor_summary,
            "future_upstream_sidecar_required": any(
                summary.get(field_name) != "ok"
                for summary, field_name in [
                    (sync_success_summary, "sync_success_status"),
                    (inverse_summary, "inverse_transform_success_status"),
                    (anchor_summary, "attention_anchor_available_status"),
                ]
            ),
        }

    family_rows: List[Dict[str, Any]] = []
    for attack_family in sorted(grouped_rows):
        rows = grouped_rows[attack_family]
        summary = _summarize(rows)
        family_rows.append(
            {
                "attack_family": attack_family,
                "event_count": len(rows),
                **summary,
            }
        )

    overall_summary = {
        "family_id": None,
        "attack_family_count": len(grouped_rows),
        "event_count": len(attack_event_rows),
        **_summarize(attack_event_rows),
    }
    return family_rows, overall_summary


def _draw_tradeoff_frontier_png(output_path: Path, tradeoff_rows: Sequence[Mapping[str, Any]]) -> None:
    """
    功能：根据真实 tradeoff 表绘制 frontier PNG。

    Draw a tradeoff frontier PNG from the real tradeoff rows.

    Args:
        output_path: PNG output path.
        tradeoff_rows: Tradeoff row payloads.

    Returns:
        None.
    """
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be Path")

    width = 960
    height = 540
    left_margin = 80
    right_margin = 40
    top_margin = 40
    bottom_margin = 60
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [left_margin, top_margin, left_margin + plot_width, top_margin + plot_height],
        fill=(248, 248, 248, 255),
        outline=(96, 96, 96, 255),
        width=2,
    )
    draw.line(
        [(left_margin, top_margin + plot_height), (left_margin + plot_width, top_margin + plot_height)],
        fill=(64, 64, 64, 255),
        width=2,
    )
    draw.line(
        [(left_margin, top_margin), (left_margin, top_margin + plot_height)],
        fill=(64, 64, 64, 255),
        width=2,
    )

    point_rows: List[Tuple[str, float, float]] = []
    for row in tradeoff_rows:
        if row.get("clean_quality_status") != "ok":
            continue
        clean_mean_psnr = row.get("clean_mean_psnr")
        attack_macro_avg_tpr = row.get("attack_macro_avg_tpr")
        if isinstance(clean_mean_psnr, (int, float)) and isinstance(attack_macro_avg_tpr, (int, float)):
            clean_mean_psnr_float = float(clean_mean_psnr)
            attack_macro_avg_tpr_float = float(attack_macro_avg_tpr)
            if math.isfinite(clean_mean_psnr_float) and math.isfinite(attack_macro_avg_tpr_float):
                point_rows.append((str(row.get("scope")), clean_mean_psnr_float, attack_macro_avg_tpr_float))

    if point_rows:
        x_values = [item[1] for item in point_rows]
        y_values = [item[2] for item in point_rows]
        x_min = min(x_values)
        x_max = max(x_values)
        if math.isclose(x_min, x_max):
            x_min -= 1.0
            x_max += 1.0
        y_min = 0.0
        y_max = max(1.0, max(y_values))
        color_map = {
            "content_chain": (31, 119, 180, 255),
            "event_attestation": (255, 127, 14, 255),
            "system_final": (44, 160, 44, 255),
        }
        for scope_name, clean_mean_psnr, attack_macro_avg_tpr in point_rows:
            normalized_x = (clean_mean_psnr - x_min) / max(x_max - x_min, 1e-12)
            normalized_y = (attack_macro_avg_tpr - y_min) / max(y_max - y_min, 1e-12)
            x_coord = int(round(left_margin + normalized_x * plot_width))
            y_coord = int(round(top_margin + plot_height - normalized_y * plot_height))
            color = color_map.get(scope_name, (96, 96, 96, 255))
            draw.ellipse([x_coord - 7, y_coord - 7, x_coord + 7, y_coord + 7], fill=color, outline=(32, 32, 32, 255), width=1)

    ensure_directory(output_path.parent)
    image.save(output_path)


def _update_registry_supplemental_paths(registry_path: Path, extension_paths: Mapping[str, str]) -> None:
    """
    功能：向 paper metric registry 追加 supplemental 路径。

    Append supplemental export paths to the paper metric registry.

    Args:
        registry_path: Registry JSON path.
        extension_paths: New path bindings.

    Returns:
        None.
    """
    if not isinstance(registry_path, Path):
        raise TypeError("registry_path must be Path")
    if not isinstance(extension_paths, Mapping):
        raise TypeError("extension_paths must be Mapping")
    registry_payload = _load_required_json_dict(registry_path, "PW04 paper metric registry")
    artifact_paths = _extract_mapping(registry_payload.get("artifact_paths"))
    supplemental_metrics = _extract_mapping(artifact_paths.get("supplemental_metrics"))
    supplemental_metrics.update(dict(extension_paths))
    artifact_paths["supplemental_metrics"] = supplemental_metrics
    registry_payload["artifact_paths"] = artifact_paths
    write_json_atomic(registry_path, registry_payload)


def build_pw04_metrics_extensions(
    *,
    family_id: str,
    family_root: Path,
    export_root: Path,
    pw02_summary: Mapping[str, Any],
    attack_event_rows: Sequence[Mapping[str, Any]],
    main_metrics_summary_csv_path: Path,
    attack_family_summary_paper_csv_path: Path,
    attack_condition_summary_paper_csv_path: Path,
    paper_metric_registry_path: Path | None = None,
) -> Dict[str, Any]:
    """
    功能：为 PW04 构建 robustness、geometry、payload 与 tradeoff 收口导出。

    Build append-only PW04 robustness, geometry, payload, and tradeoff export artifacts.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        export_root: PW04 export root.
        pw02_summary: PW02 summary payload.
        attack_event_rows: Materialized attack event rows.
        main_metrics_summary_csv_path: Canonical main summary CSV path.
        attack_family_summary_paper_csv_path: Paper-facing family summary CSV path.
        attack_condition_summary_paper_csv_path: Paper-facing condition summary CSV path.
        paper_metric_registry_path: Optional paper metric registry path.

    Returns:
        Summary mapping of generated directories and files.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(export_root, Path):
        raise TypeError("export_root must be Path")
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")
    if not isinstance(main_metrics_summary_csv_path, Path):
        raise TypeError("main_metrics_summary_csv_path must be Path")
    if not isinstance(attack_family_summary_paper_csv_path, Path):
        raise TypeError("attack_family_summary_paper_csv_path must be Path")
    if not isinstance(attack_condition_summary_paper_csv_path, Path):
        raise TypeError("attack_condition_summary_paper_csv_path must be Path")
    if paper_metric_registry_path is not None and not isinstance(paper_metric_registry_path, Path):
        raise TypeError("paper_metric_registry_path must be Path or None")

    robustness_dir = ensure_directory(export_root / "robustness")
    geometry_dir = ensure_directory(export_root / "geometry_diagnostics")
    payload_dir = ensure_directory(export_root / "payload_robustness")
    tradeoff_dir = ensure_directory(export_root / "tradeoff")
    auxiliary_attack_exports = _build_system_final_auxiliary_attack_exports(
        family_id=family_id,
        family_root=family_root,
        pw02_summary=pw02_summary,
        attack_event_rows=attack_event_rows,
        robustness_dir=robustness_dir,
    )

    main_rows = _read_csv_rows(main_metrics_summary_csv_path)
    family_rows = _read_csv_rows(attack_family_summary_paper_csv_path)
    condition_rows = _read_csv_rows(attack_condition_summary_paper_csv_path)
    robustness_curve_rows, robustness_macro_rows, worst_case_rows = _build_robustness_rows(
        main_rows=main_rows,
        family_rows=family_rows,
        condition_rows=condition_rows,
        attack_event_rows=attack_event_rows,
    )
    geometry_family_rows, geometry_summary_row = _collect_geometry_rows(attack_event_rows)

    robustness_curve_by_family_path = robustness_dir / "robustness_curve_by_family.csv"
    robustness_macro_summary_path = robustness_dir / "robustness_macro_summary.csv"
    worst_case_attack_summary_path = robustness_dir / "worst_case_attack_summary.csv"
    geo_chain_usage_by_family_path = geometry_dir / "geo_chain_usage_by_family.csv"
    geo_diagnostics_summary_path = geometry_dir / "geo_diagnostics_summary.csv"
    payload_attack_summary_path = payload_dir / "payload_attack_summary.json"
    quality_robustness_tradeoff_path = tradeoff_dir / "quality_robustness_tradeoff.csv"
    quality_robustness_frontier_path = tradeoff_dir / "quality_robustness_frontier.png"

    _write_csv_rows(
        robustness_curve_by_family_path,
        [
            "scope",
            "attack_family",
            "event_count",
            "parent_event_count",
            "mean_attack_tpr",
            "min_attack_tpr",
            "macro_avg_attack_tpr",
            "severity_level_status",
            "severity_level_reason",
            "severity_axis_kind",
            "severity_rule_version",
            "severity_directionality",
            "severity_available_condition_count",
            "severity_total_condition_count",
            "severity_min_sort_value",
            "severity_max_sort_value",
            "severity_label_sequence",
            "severity_canonical_global_axis",
        ],
        robustness_curve_rows,
    )
    _write_csv_rows(
        robustness_macro_summary_path,
        [
            "scope",
            "overall_attack_tpr",
            "macro_avg_attack_tpr",
            "worst_case_attack_tpr",
            "family_count",
            "condition_count",
            "severity_level_status",
            "severity_level_reason",
            "severity_axis_kind",
            "severity_rule_version",
            "severity_directionality",
            "severity_available_condition_count",
            "severity_total_condition_count",
            "severity_min_sort_value",
            "severity_max_sort_value",
            "severity_label_sequence",
            "severity_canonical_global_axis",
        ],
        robustness_macro_rows,
    )
    _write_csv_rows(
        worst_case_attack_summary_path,
        [
            "scope",
            "attack_family",
            "attack_condition_key",
            "attack_config_name",
            "event_count",
            "parent_event_count",
            "worst_case_attack_tpr",
            "severity_level_status",
            "severity_level_reason",
            "severity_axis_kind",
            "severity_rule_version",
            "severity_directionality",
            "severity_available_condition_count",
            "severity_total_condition_count",
            "severity_min_sort_value",
            "severity_max_sort_value",
            "severity_label_sequence",
            "severity_canonical_global_axis",
            "severity_label",
        ],
        worst_case_rows,
    )
    _write_csv_rows(
        geo_chain_usage_by_family_path,
        [
            "attack_family",
            "event_count",
            "geo_rescue_eligible_count",
            "geo_rescue_applied_count",
            "geo_helped_positive_count",
            "geo_not_used_count",
            "geo_not_used_reason_counts",
            "sync_status_counts",
            "geometry_failure_reason_counts",
            "sync_success_available_count",
            "sync_success_true_count",
            "sync_success_true_rate",
            "sync_success_status",
            "sync_success_reason",
            "inverse_transform_success_available_count",
            "inverse_transform_success_true_count",
            "inverse_transform_success_true_rate",
            "inverse_transform_success_status",
            "inverse_transform_success_reason",
            "attention_anchor_available_available_count",
            "attention_anchor_available_true_count",
            "attention_anchor_available_true_rate",
            "attention_anchor_available_status",
            "attention_anchor_available_reason",
            "future_upstream_sidecar_required",
        ],
        geometry_family_rows,
    )
    _write_csv_rows(
        geo_diagnostics_summary_path,
        [
            "family_id",
            "attack_family_count",
            "event_count",
            "geo_rescue_eligible_count",
            "geo_rescue_applied_count",
            "geo_helped_positive_count",
            "geo_not_used_count",
            "geo_not_used_reason_counts",
            "sync_status_counts",
            "geometry_failure_reason_counts",
            "sync_success_available_count",
            "sync_success_true_count",
            "sync_success_true_rate",
            "sync_success_status",
            "sync_success_reason",
            "inverse_transform_success_available_count",
            "inverse_transform_success_true_count",
            "inverse_transform_success_true_rate",
            "inverse_transform_success_status",
            "inverse_transform_success_reason",
            "attention_anchor_available_available_count",
            "attention_anchor_available_true_count",
            "attention_anchor_available_true_rate",
            "attention_anchor_available_status",
            "attention_anchor_available_reason",
            "future_upstream_sidecar_required",
        ],
        [{"family_id": family_id, **geometry_summary_row}],
    )
    write_json_atomic(
        payload_attack_summary_path,
        {
            "artifact_type": "paper_workflow_pw04_payload_attack_summary",
            "schema_version": "pw_stage_04_v1",
            "created_at": utc_now_iso(),
            "family_id": family_id,
            "status": "not_available",
            "reason": PAYLOAD_UNAVAILABLE_REASON,
            "future_upstream_sidecar_required": True,
        },
    )

    quality_summary_csv_path, quality_summary_json_path = _resolve_pw02_quality_summary_paths(
        family_root=family_root,
        pw02_summary=pw02_summary,
    )
    quality_summary_payload = _load_required_json_dict(quality_summary_json_path, "PW02 quality metrics summary")
    quality_rows = cast(List[Mapping[str, Any]], quality_summary_payload.get("rows", []))
    clean_quality_row = next((row for row in quality_rows if row.get("scope") == "content_chain"), {})
    attack_lpips_values = [
        value
        for value in (
            _coerce_finite_float(row.get("attack_quality_lpips"))
            for row in attack_event_rows
        )
        if value is not None
    ]
    attack_clip_values = [
        value
        for value in (
            _coerce_finite_float(row.get("attack_quality_clip_text_similarity"))
            for row in attack_event_rows
        )
        if value is not None
    ]
    attack_quality_ok_count = sum(1 for row in attack_event_rows if row.get("attack_quality_status") == "ok")
    attack_clip_sample_count = len(attack_clip_values)
    attack_clip_model_name = next(
        (
            str(row.get("attack_quality_clip_model_name"))
            for row in attack_event_rows
            if isinstance(row.get("attack_quality_clip_model_name"), str) and str(row.get("attack_quality_clip_model_name")).strip()
        ),
        clean_quality_row.get("clip_model_name"),
    )
    if attack_quality_ok_count <= 0:
        attack_lpips_status = "not_available"
        attack_lpips_reason = "PW04 attack quality rows do not contain LPIPS values"
    elif attack_quality_ok_count == len(attack_event_rows):
        attack_lpips_status = "ok"
        attack_lpips_reason = None
    else:
        attack_lpips_status = "partial"
        attack_lpips_reason = f"LPIPS available for {attack_quality_ok_count}/{len(attack_event_rows)} attack events"

    if attack_clip_sample_count <= 0:
        attack_clip_status = "not_available"
        attack_clip_reason = "PW04 attack quality rows do not contain CLIP text similarity values"
    elif attack_clip_sample_count == len(attack_event_rows):
        attack_clip_status = "ok"
        attack_clip_reason = None
    else:
        attack_clip_status = "partial"
        attack_clip_reason = f"CLIP available for {attack_clip_sample_count}/{len(attack_event_rows)} attack events"

    tradeoff_rows: List[Dict[str, Any]] = []
    for robustness_row in robustness_macro_rows:
        tradeoff_rows.append(
            {
                "scope": robustness_row.get("scope"),
                "clean_quality_scope": clean_quality_row.get("scope"),
                "clean_quality_status": clean_quality_row.get("status", "not_available"),
                "clean_mean_psnr": clean_quality_row.get("mean_psnr"),
                "clean_mean_ssim": clean_quality_row.get("mean_ssim"),
                "clean_mean_lpips": clean_quality_row.get("mean_lpips"),
                "attack_macro_avg_tpr": robustness_row.get("macro_avg_attack_tpr"),
                "overall_attack_tpr": robustness_row.get("overall_attack_tpr"),
                "attack_mean_lpips": _safe_mean(attack_lpips_values),
                "attack_lpips_status": attack_lpips_status,
                "attack_lpips_reason": attack_lpips_reason,
                "attack_mean_clip_text_similarity": _safe_mean(attack_clip_values),
                "attack_clip_model_name": attack_clip_model_name,
                "attack_clip_status": attack_clip_status,
                "attack_clip_reason": attack_clip_reason,
                "attack_clip_sample_count": attack_clip_sample_count,
                "quality_metrics_summary_csv_path": normalize_path_value(quality_summary_csv_path),
                "quality_metrics_summary_json_path": normalize_path_value(quality_summary_json_path),
                "robustness_macro_summary_path": normalize_path_value(robustness_macro_summary_path),
                "lpips_status": clean_quality_row.get("lpips_status", "not_available"),
                "lpips_reason": clean_quality_row.get("lpips_reason"),
                "clip_status": clean_quality_row.get("clip_status", "not_available"),
                "clip_reason": clean_quality_row.get("clip_reason"),
            }
        )
    _write_csv_rows(
        quality_robustness_tradeoff_path,
        [
            "scope",
            "clean_quality_scope",
            "clean_quality_status",
            "clean_mean_psnr",
            "clean_mean_ssim",
            "clean_mean_lpips",
            "attack_macro_avg_tpr",
            "overall_attack_tpr",
            "attack_mean_lpips",
            "attack_lpips_status",
            "attack_lpips_reason",
            "attack_mean_clip_text_similarity",
            "attack_clip_model_name",
            "attack_clip_status",
            "attack_clip_reason",
            "attack_clip_sample_count",
            "quality_metrics_summary_csv_path",
            "quality_metrics_summary_json_path",
            "robustness_macro_summary_path",
            "lpips_status",
            "lpips_reason",
            "clip_status",
            "clip_reason",
        ],
        tradeoff_rows,
    )
    _draw_tradeoff_frontier_png(quality_robustness_frontier_path, tradeoff_rows)

    extension_paths = {
        "robustness_curve_by_family_path": normalize_path_value(robustness_curve_by_family_path),
        "robustness_macro_summary_path": normalize_path_value(robustness_macro_summary_path),
        "worst_case_attack_summary_path": normalize_path_value(worst_case_attack_summary_path),
        "geo_chain_usage_by_family_path": normalize_path_value(geo_chain_usage_by_family_path),
        "geo_diagnostics_summary_path": normalize_path_value(geo_diagnostics_summary_path),
        "payload_attack_summary_path": normalize_path_value(payload_attack_summary_path),
        "quality_robustness_tradeoff_path": normalize_path_value(quality_robustness_tradeoff_path),
        "quality_robustness_frontier_path": normalize_path_value(quality_robustness_frontier_path),
        "system_final_auxiliary_attack_summary_path": auxiliary_attack_exports["system_final_auxiliary_attack_summary_path"],
        "system_final_auxiliary_attack_by_family_path": auxiliary_attack_exports["system_final_auxiliary_attack_by_family_path"],
        "system_final_auxiliary_attack_by_condition_path": auxiliary_attack_exports["system_final_auxiliary_attack_by_condition_path"],
    }
    if paper_metric_registry_path is not None:
        _update_registry_supplemental_paths(paper_metric_registry_path, extension_paths)

    analysis_only_artifact_paths: Dict[str, str] = {}
    summary_analysis_only_paths = pw02_summary.get("analysis_only_artifact_paths")
    if isinstance(summary_analysis_only_paths, Mapping):
        for label, path_value in summary_analysis_only_paths.items():
            if not isinstance(label, str) or not label:
                continue
            if not isinstance(path_value, str) or not path_value.strip():
                continue
            resolved_path = Path(path_value).expanduser().resolve()
            if resolved_path.exists() and resolved_path.is_file():
                analysis_only_artifact_paths[label] = normalize_path_value(resolved_path)
    analysis_only_artifact_paths.update(
        {
            "pw04_system_final_auxiliary_attack_summary": auxiliary_attack_exports["system_final_auxiliary_attack_summary_path"],
            "pw04_system_final_auxiliary_attack_by_family": auxiliary_attack_exports["system_final_auxiliary_attack_by_family_path"],
            "pw04_system_final_auxiliary_attack_by_condition": auxiliary_attack_exports["system_final_auxiliary_attack_by_condition_path"],
        }
    )
    analysis_only_artifact_annotations = {
        label: {"canonical": False, "analysis_only": True}
        for label in analysis_only_artifact_paths.keys()
    }

    return {
        "robustness_dir": normalize_path_value(robustness_dir),
        "geometry_diagnostics_dir": normalize_path_value(geometry_dir),
        "payload_robustness_dir": normalize_path_value(payload_dir),
        "tradeoff_dir": normalize_path_value(tradeoff_dir),
        "analysis_only_artifact_paths": analysis_only_artifact_paths,
        "analysis_only_artifact_annotations": analysis_only_artifact_annotations,
        **extension_paths,
    }