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

from paper_workflow.scripts.pw_common import (
    extract_payload_metrics_from_decode_sidecar,
    summarize_payload_probe_rows,
)
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


def _load_optional_wrong_event_attestation_challenge_record(
    event_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：从 PW03 event manifest 读取可选 wrong-event challenge record。

    Load the optional PW03 wrong-event challenge record referenced by one event manifest.

    Args:
        event_manifest: PW03 event manifest payload.

    Returns:
        Challenge record payload when available, otherwise an empty mapping.
    """
    inline_record = event_manifest.get("wrong_event_attestation_challenge_record")
    if isinstance(inline_record, Mapping):
        return dict(cast(Mapping[str, Any], inline_record))

    record_path_value = event_manifest.get("wrong_event_attestation_challenge_record_path")
    if not isinstance(record_path_value, str) or not record_path_value:
        return {}
    record_path = Path(record_path_value).expanduser().resolve()
    if not record_path.exists() or not record_path.is_file():
        return {}
    try:
        return _load_required_json_dict(record_path, "PW04 wrong-event attestation challenge record")
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {}


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


def _load_optional_payload_decode_sidecar_metrics(
    attack_event_row: Mapping[str, Any],
    label: str,
) -> Dict[str, Any]:
    """
    功能：读取可选 attack payload decode sidecar 并提取 summary 指标。

    Load one optional attack payload decode sidecar and extract summary-ready metrics.

    Args:
        attack_event_row: Attack event row payload.
        label: Human-readable label.

    Returns:
        Extracted decode-sidecar metrics, or an empty mapping when absent.
    """
    if not isinstance(attack_event_row, Mapping):
        raise TypeError("attack_event_row must be Mapping")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")

    sidecar_path_value = attack_event_row.get("payload_decode_sidecar_path")
    if not isinstance(sidecar_path_value, str) or not sidecar_path_value.strip():
        return {}
    sidecar_payload = _load_required_json_dict(
        Path(sidecar_path_value).expanduser().resolve(),
        label,
    )
    return extract_payload_metrics_from_decode_sidecar(sidecar_payload)


def _extract_int_list(value: Any) -> List[int]:
    """
    功能：把整型序列规范化为 int 列表。

    Normalize one integer sequence to a list of ints.

    Args:
        value: Candidate sequence.

    Returns:
        Normalized integer list.
    """
    if not isinstance(value, list):
        return []
    normalized: List[int] = []
    for item in cast(List[object], value):
        if isinstance(item, int) and not isinstance(item, bool):
            normalized.append(int(item))
    return normalized


def _derive_payload_primary_metrics(lf_trace: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：从 LF trace 派生 payload 主指标。

    Derive primary payload metrics from one LF detect trace.

    Args:
        lf_trace: LF detect trace mapping.

    Returns:
        Derived payload metric mapping.
    """
    if not isinstance(lf_trace, Mapping):
        raise TypeError("lf_trace must be Mapping")

    agreement_count = _parse_csv_int(lf_trace, "agreement_count")
    n_bits_compared = _parse_csv_int(lf_trace, "n_bits_compared")
    mismatch_indices = _extract_int_list(lf_trace.get("mismatch_indices"))
    mismatch_count = len(mismatch_indices) if mismatch_indices else None
    bit_accuracy = _coerce_finite_float(lf_trace.get("codeword_agreement"))
    if bit_accuracy is None and agreement_count is not None and isinstance(n_bits_compared, int) and n_bits_compared > 0:
        bit_accuracy = float(agreement_count / n_bits_compared)
    if mismatch_count is None and agreement_count is not None and isinstance(n_bits_compared, int) and n_bits_compared >= agreement_count:
        mismatch_count = int(n_bits_compared - agreement_count)
    bit_error_rate = None if bit_accuracy is None else float(max(0.0, min(1.0, 1.0 - bit_accuracy)))
    message_success = None
    if isinstance(n_bits_compared, int) and n_bits_compared > 0 and bit_accuracy is not None:
        message_success = bool(bit_accuracy >= 1.0 - 1e-12)

    if agreement_count is not None and isinstance(n_bits_compared, int) and n_bits_compared > 0:
        primary_metric_source = "agreement_count_and_n_bits_compared"
    elif bit_accuracy is not None and isinstance(n_bits_compared, int) and n_bits_compared > 0:
        primary_metric_source = "codeword_agreement_and_n_bits_compared"
    elif bit_accuracy is not None:
        primary_metric_source = "codeword_agreement_only"
    else:
        primary_metric_source = None

    return {
        "agreement_count": agreement_count,
        "mismatch_count": mismatch_count,
        "bit_accuracy": bit_accuracy,
        "bit_error_rate": bit_error_rate,
        "message_success": message_success,
        "primary_metric_source": primary_metric_source,
    }


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


def _resolve_geometry_condition_value(value: Any) -> str:
    """
    功能：将 geometry 条件值归一化为 true / false / missing。

    Normalize one geometry-condition value to true / false / missing.

    Args:
        value: Candidate condition value.

    Returns:
        Normalized condition bucket.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return "missing"


def _collect_geometry_conditional_rows(
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    功能：按几何诊断条件分桶，构造 attack TPR 条件指标行。

    Build attack TPR conditional rows stratified by frozen geometry diagnostics.

    Args:
        attack_event_rows: Materialized attack event rows.

    Returns:
        Conditional geometry metric rows.
    """
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    grouped_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    ordered_condition_names = [
        "sync_success",
        "inverse_transform_success",
        "attention_anchor_available",
        "geo_rescue_eligible",
        "geo_rescue_applied",
    ]
    for attack_event_row in attack_event_rows:
        formal_record = _extract_mapping(attack_event_row.get("formal_record"))
        formal_final_decision = _extract_mapping(formal_record.get("formal_final_decision"))
        formal_event_attestation_decision = _extract_mapping(
            formal_record.get("formal_event_attestation_decision")
        )
        attestation_payload = _extract_mapping(formal_record.get("attestation"))
        image_evidence_payload = _extract_mapping(attestation_payload.get("image_evidence_result"))
        geometry_diagnostics = _extract_mapping(attack_event_row.get("geometry_diagnostics"))
        bucket_payload = {
            "parent_event_id": attack_event_row.get("parent_event_id"),
            "attack_family": attack_event_row.get("attack_family"),
            "formal_final_positive": formal_final_decision.get("is_watermarked") is True,
            "formal_attestation_positive": formal_event_attestation_decision.get("is_watermarked") is True,
            "derived_attack_union_positive": bool(formal_record.get("derived_attack_union_positive", False)),
            "geo_helped_positive": bool(image_evidence_payload.get("geo_rescue_applied", False))
            and bool(formal_record.get("derived_attack_union_positive", False)),
        }
        condition_specs = [
            ("sync_success", geometry_diagnostics.get("sync_success")),
            ("inverse_transform_success", geometry_diagnostics.get("inverse_transform_success")),
            ("attention_anchor_available", geometry_diagnostics.get("attention_anchor_available")),
            ("geo_rescue_eligible", image_evidence_payload.get("geo_rescue_eligible")),
            ("geo_rescue_applied", image_evidence_payload.get("geo_rescue_applied")),
        ]
        for condition_name, condition_raw_value in condition_specs:
            condition_value = _resolve_geometry_condition_value(condition_raw_value)
            grouped_rows.setdefault((condition_name, condition_value), []).append(dict(bucket_payload))

    conditional_rows: List[Dict[str, Any]] = []
    for condition_name in ordered_condition_names:
        for condition_value in ["true", "false", "missing"]:
            rows = grouped_rows.get((condition_name, condition_value), [])
            event_count = len(rows)
            formal_final_positive_count = sum(1 for row in rows if bool(row.get("formal_final_positive")))
            formal_attestation_positive_count = sum(
                1 for row in rows if bool(row.get("formal_attestation_positive"))
            )
            derived_attack_union_positive_count = sum(
                1 for row in rows if bool(row.get("derived_attack_union_positive"))
            )
            geo_helped_positive_count = sum(1 for row in rows if bool(row.get("geo_helped_positive")))
            conditional_rows.append(
                {
                    "geometry_condition_name": condition_name,
                    "geometry_condition_value": condition_value,
                    "event_count": event_count,
                    "parent_event_count": len(
                        {
                            str(row.get("parent_event_id"))
                            for row in rows
                            if isinstance(row.get("parent_event_id"), str) and str(row.get("parent_event_id"))
                        }
                    ),
                    "attack_family_count": len(
                        {
                            str(row.get("attack_family"))
                            for row in rows
                            if isinstance(row.get("attack_family"), str) and str(row.get("attack_family"))
                        }
                    ),
                    "formal_final_positive_count": formal_final_positive_count,
                    "formal_final_decision_attack_tpr": (
                        float(formal_final_positive_count / event_count) if event_count > 0 else None
                    ),
                    "formal_attestation_positive_count": formal_attestation_positive_count,
                    "formal_attestation_attack_tpr": (
                        float(formal_attestation_positive_count / event_count) if event_count > 0 else None
                    ),
                    "derived_attack_union_positive_count": derived_attack_union_positive_count,
                    "derived_attack_union_attack_tpr": (
                        float(derived_attack_union_positive_count / event_count) if event_count > 0 else None
                    ),
                    "geo_helped_positive_count": geo_helped_positive_count,
                    "geo_helped_positive_rate": (
                        float(geo_helped_positive_count / event_count) if event_count > 0 else None
                    ),
                }
            )
    return conditional_rows


def _build_geometry_conditional_rescue_summary(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：在 content-only 失败子集上汇总 geometry conditional rescue 指标。

    Summarize geometry conditional rescue metrics on the content-failed subset.

    Args:
        rows: Materialized attack event rows for one grouping.

    Returns:
        Conditional rescue metric summary.
    """
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be Sequence")

    event_count = len(rows)
    formal_final_positive_count = 0
    formal_attestation_positive_count = 0
    content_failed_subset_event_count = 0
    geo_only_positive_count = 0
    geo_rescue_eligible_on_content_failed_count = 0
    geo_rescue_applied_on_content_failed_count = 0

    for row in rows:
        formal_record = _extract_mapping(row.get("formal_record"))
        formal_final_decision = _extract_mapping(formal_record.get("formal_final_decision"))
        formal_event_attestation_decision = _extract_mapping(
            formal_record.get("formal_event_attestation_decision")
        )
        attestation_payload = _extract_mapping(formal_record.get("attestation"))
        image_evidence_payload = _extract_mapping(attestation_payload.get("image_evidence_result"))

        formal_final_positive = formal_final_decision.get("is_watermarked") is True
        formal_attestation_positive = formal_event_attestation_decision.get("is_watermarked") is True
        geo_rescue_eligible = image_evidence_payload.get("geo_rescue_eligible") is True
        geo_rescue_applied = image_evidence_payload.get("geo_rescue_applied") is True

        if formal_final_positive:
            formal_final_positive_count += 1
        if formal_attestation_positive:
            formal_attestation_positive_count += 1
        if not formal_final_positive:
            content_failed_subset_event_count += 1
            if geo_rescue_eligible:
                geo_rescue_eligible_on_content_failed_count += 1
            if geo_rescue_applied:
                geo_rescue_applied_on_content_failed_count += 1
            if formal_attestation_positive:
                geo_only_positive_count += 1

    content_only_positive_rate = (
        float(formal_final_positive_count / event_count) if event_count > 0 else None
    )
    attestation_positive_rate = (
        float(formal_attestation_positive_count / event_count) if event_count > 0 else None
    )
    conditional_rescue_rate = (
        float(geo_only_positive_count / content_failed_subset_event_count)
        if content_failed_subset_event_count > 0
        else None
    )
    rescue_precision = (
        float(geo_only_positive_count / geo_rescue_applied_on_content_failed_count)
        if geo_rescue_applied_on_content_failed_count > 0
        else None
    )
    rescue_lift_over_content_only = (
        float(attestation_positive_rate - content_only_positive_rate)
        if attestation_positive_rate is not None and content_only_positive_rate is not None
        else None
    )
    if event_count <= 0:
        status_value = "not_available"
        reason_value = "no_attack_events_available_for_conditional_rescue_metrics"
    elif content_failed_subset_event_count <= 0:
        status_value = "not_available"
        reason_value = "no_content_failed_subset_available"
    else:
        status_value = "ok"
        reason_value = None

    return {
        "event_count": event_count,
        "formal_final_positive_count": formal_final_positive_count,
        "formal_attestation_positive_count": formal_attestation_positive_count,
        "content_only_positive_rate": content_only_positive_rate,
        "attestation_positive_rate": attestation_positive_rate,
        "content_failed_subset_event_count": content_failed_subset_event_count,
        "geo_rescue_eligible_on_content_failed_count": geo_rescue_eligible_on_content_failed_count,
        "geo_rescue_applied_on_content_failed_count": geo_rescue_applied_on_content_failed_count,
        "geo_only_positive_on_content_failed_subset": geo_only_positive_count,
        "geo_only_positive_on_content_failed_subset_rate": conditional_rescue_rate,
        "conditional_rescue_rate": conditional_rescue_rate,
        "rescue_precision": rescue_precision,
        "rescue_lift_over_content_only": rescue_lift_over_content_only,
        "status": status_value,
        "reason": reason_value,
    }


def _build_geometry_conditional_rescue_metrics_payload(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：构造 geometry conditional rescue 的显式导出。 

    Build the explicit geometry conditional rescue metrics export.

    Args:
        family_id: Family identifier.
        attack_event_rows: Materialized attack event rows.

    Returns:
        Geometry conditional rescue metrics payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    grouped_by_family: Dict[str, List[Dict[str, Any]]] = {}
    grouped_by_condition: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for attack_event_row in attack_event_rows:
        attack_family = str(attack_event_row.get("attack_family") or "<unknown>")
        attack_condition_key = str(attack_event_row.get("attack_condition_key") or "<unknown>")
        grouped_by_family.setdefault(attack_family, []).append(dict(cast(Mapping[str, Any], attack_event_row)))
        grouped_by_condition.setdefault((attack_family, attack_condition_key), []).append(
            dict(cast(Mapping[str, Any], attack_event_row))
        )

    overall_summary = _build_geometry_conditional_rescue_summary(attack_event_rows)
    if overall_summary["status"] == "ok":
        readiness_status = "ready"
        readiness_reason = None
    else:
        readiness_status = "not_ready"
        readiness_reason = overall_summary["reason"]
    return {
        "artifact_type": "paper_workflow_pw04_geometry_conditional_rescue_metrics_export",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "metric_name": "geometry_conditional_rescue_metrics",
        "canonical": False,
        "analysis_only": True,
        "status": overall_summary["status"],
        "reason": overall_summary["reason"],
        "definition": {
            "conditional_rescue_rate": "geo_only_positive_on_content_failed_subset divided by content_failed_subset_event_count",
            "rescue_precision": "geo_only_positive_on_content_failed_subset divided by geo_rescue_applied_on_content_failed_count",
            "rescue_lift_over_content_only": "attestation_positive_rate minus content_only_positive_rate within the same grouping",
            "geo_only_positive_on_content_failed_subset": "count of rows with formal_final_decision negative and formal_event_attestation_decision positive",
        },
        "overall": overall_summary,
        "readiness": {
            "status": readiness_status,
            "reason": readiness_reason,
            "required_for_formal_release": False,
            "blocking": False,
            "claim_scope": "geometry_conditional_rescue_optional",
        },
        "by_attack_family": [
            {
                "attack_family": attack_family,
                **_build_geometry_conditional_rescue_summary(rows),
            }
            for attack_family, rows in sorted(grouped_by_family.items())
        ],
        "by_attack_condition": [
            {
                "attack_family": attack_family,
                "attack_condition_key": attack_condition_key,
                "attack_config_name": next(
                    (
                        row.get("attack_config_name")
                        for row in rows
                        if isinstance(row.get("attack_config_name"), str) and row.get("attack_config_name")
                    ),
                    None,
                ),
                **_build_geometry_conditional_rescue_summary(rows),
            }
            for (attack_family, attack_condition_key), rows in sorted(grouped_by_condition.items())
        ],
    }


def _load_geometry_optional_claim_evidence(
    attack_event_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：读取一条 attack row 绑定的 geometry optional claim evidence。 

    Load geometry optional-claim evidence from one attack event row.

    Args:
        attack_event_row: Materialized attack event row.

    Returns:
        Optional-claim evidence mapping.
    """
    if not isinstance(attack_event_row, Mapping):
        raise TypeError("attack_event_row must be Mapping")

    event_manifest = _extract_mapping(attack_event_row.get("event_manifest"))
    evidence_payload = _extract_mapping(event_manifest.get("geometry_optional_claim_evidence"))
    if not evidence_payload:
        detect_payload = _extract_mapping(attack_event_row.get("detect_payload"))
        evidence_payload = _extract_mapping(
            detect_payload.get("paper_workflow_geometry_optional_claim_evidence")
        )

    sample_role = str(
        attack_event_row.get("sample_role")
        or event_manifest.get("sample_role")
        or ""
    )
    formal_record = _extract_mapping(attack_event_row.get("formal_record"))
    attestation_payload = _extract_mapping(formal_record.get("attestation"))
    image_evidence_payload = _extract_mapping(attestation_payload.get("image_evidence_result"))
    geometry_diagnostics = _extract_mapping(attack_event_row.get("geometry_diagnostics"))
    supporting_signal_count = sum(
        1
        for field_name in ["sync_success", "inverse_transform_success", "attention_anchor_available"]
        if geometry_diagnostics.get(field_name) is True
    )

    if evidence_payload:
        normalized_payload = dict(cast(Mapping[str, Any], evidence_payload))
    elif sample_role == "attacked_positive":
        normalized_payload = {
            "status": "not_available",
            "reason": "missing_geometry_optional_claim_evidence",
        }
    else:
        normalized_payload = {
            "status": "not_applicable",
            "reason": "sample_role_not_attacked_positive",
        }
    supporting_evidence_available = normalized_payload.get("supporting_evidence_available")
    if not isinstance(supporting_evidence_available, bool):
        supporting_evidence_available = supporting_signal_count > 0
    content_score = _coerce_finite_float(attack_event_row.get("content_score"))
    content_threshold_value = _coerce_finite_float(normalized_payload.get("content_threshold_value"))
    parent_boundary_hit = normalized_payload.get("parent_boundary_hit")
    if not isinstance(parent_boundary_hit, bool):
        parent_boundary_hit = normalized_payload.get("eligible_for_optional_claim") is True
    attacked_content_failed = normalized_payload.get("attacked_content_failed")
    if not isinstance(attacked_content_failed, bool):
        if content_score is not None and content_threshold_value is not None:
            attacked_content_failed = content_score < content_threshold_value
        else:
            attacked_content_failed = None
    geo_rescue_candidate_family = normalized_payload.get("geo_rescue_candidate_family")
    if not isinstance(geo_rescue_candidate_family, bool):
        geo_rescue_candidate_family = (
            normalized_payload.get("geometry_rescue_candidate") is True
            or attack_event_row.get("geometry_rescue_candidate") is True
        )
    geo_rescue_eligible = normalized_payload.get("geo_rescue_eligible")
    if not isinstance(geo_rescue_eligible, bool):
        geo_rescue_eligible = bool(
            parent_boundary_hit and attacked_content_failed is True and geo_rescue_candidate_family
        )
    geo_rescue_applied = normalized_payload.get("geo_rescue_applied")
    if not isinstance(geo_rescue_applied, bool):
        geo_rescue_applied = bool(
            geo_rescue_eligible and image_evidence_payload.get("geo_rescue_applied") is True
        )
    return {
        **normalized_payload,
        "status": normalized_payload.get("status", "not_available"),
        "reason": normalized_payload.get("reason"),
        "plan_status": normalized_payload.get("plan_status", "not_available"),
        "plan_reason": normalized_payload.get("plan_reason"),
        "claim_mode": normalized_payload.get("claim_mode", "optional_geometry_rescue_evidence_only"),
        "claim_scope": normalized_payload.get("claim_scope", "attacked_positive_content_failed_subset"),
        "content_positive_veto_allowed": False,
        "rescue_directionality": normalized_payload.get("rescue_directionality", "one_way_positive_only"),
        "eligible_for_optional_claim": normalized_payload.get("eligible_for_optional_claim") is True,
        "parent_boundary_hit": parent_boundary_hit,
        "attacked_content_failed": attacked_content_failed,
        "geo_rescue_candidate_family": geo_rescue_candidate_family,
        "geo_rescue_eligible": geo_rescue_eligible,
        "geo_rescue_applied": geo_rescue_applied,
        "protocol_version": normalized_payload.get("protocol_version"),
        "boundary_metric": normalized_payload.get("boundary_metric"),
        "boundary_abs_margin_max": normalized_payload.get("boundary_abs_margin_max"),
        "boundary_metric_value": normalized_payload.get("boundary_metric_value"),
        "boundary_resolution_status": normalized_payload.get("boundary_resolution_status"),
        "boundary_resolution_reason": normalized_payload.get("boundary_resolution_reason"),
        "parent_content_margin": normalized_payload.get("parent_content_margin"),
        "parent_event_attestation_margin": normalized_payload.get("parent_event_attestation_margin"),
        "parent_source_detect_record_path": normalized_payload.get("parent_source_detect_record_path"),
        "content_threshold_export_path": normalized_payload.get("content_threshold_export_path"),
        "content_threshold_value": normalized_payload.get("content_threshold_value"),
        "event_attestation_threshold_export_path": normalized_payload.get("event_attestation_threshold_export_path"),
        "event_attestation_threshold_value": normalized_payload.get("event_attestation_threshold_value"),
        "sync_status": geometry_diagnostics.get("sync_status"),
        "sync_success": geometry_diagnostics.get("sync_success"),
        "inverse_transform_success": geometry_diagnostics.get("inverse_transform_success"),
        "attention_anchor_available": geometry_diagnostics.get("attention_anchor_available"),
        "geometry_failure_reason": geometry_diagnostics.get("geometry_failure_reason"),
        "supporting_signal_count": supporting_signal_count,
        "supporting_evidence_available": supporting_evidence_available,
    }


def _build_geometry_optional_claim_summary(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：汇总 geometry optional claim 的辅助证据。 

    Summarize geometry optional-claim evidence for one grouping.

    Args:
        rows: Materialized attack event rows.

    Returns:
        Optional-claim evidence summary.
    """
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be Sequence")

    evidence_rows = [_load_geometry_optional_claim_evidence(row) for row in rows]
    available_evidence = [
        row
        for row in evidence_rows
        if row.get("boundary_resolution_status") == "ok" or row.get("status") in {"ok", "not_applicable"}
    ]
    boundary_hit_event_count = sum(1 for row in evidence_rows if row.get("parent_boundary_hit") is True)
    content_failed_event_count = sum(1 for row in evidence_rows if row.get("attacked_content_failed") is True)
    content_failed_subset_boundary_event_count = sum(
        1
        for row in evidence_rows
        if row.get("parent_boundary_hit") is True and row.get("attacked_content_failed") is True
    )
    eligible_rows = [row for row in evidence_rows if row.get("geo_rescue_eligible") is True]
    boundary_subset_eligible_event_count = len(eligible_rows)
    evidence_scope_rescue_applied_event_count = sum(
        1 for row in evidence_rows if row.get("geo_rescue_applied") is True
    )
    boundary_subset_rescue_applied_event_count = sum(
        1 for row in eligible_rows if row.get("geo_rescue_applied") is True
    )
    boundary_excluded_event_count = sum(
        1
        for row in evidence_rows
        if row.get("boundary_resolution_status") == "ok" and row.get("parent_boundary_hit") is not True
    )
    boundary_resolution_failed_event_count = sum(
        1
        for row in evidence_rows
        if row.get("boundary_resolution_status") == "failed" or row.get("status") == "not_available"
    )
    supporting_evidence_event_count = sum(
        1 for row in eligible_rows if row.get("supporting_evidence_available") is True
    )
    sync_success_support_count = sum(1 for row in eligible_rows if row.get("sync_success") is True)
    inverse_transform_support_count = sum(
        1 for row in eligible_rows if row.get("inverse_transform_success") is True
    )
    attention_anchor_support_count = sum(
        1 for row in eligible_rows if row.get("attention_anchor_available") is True
    )
    plan_ready_event_count = sum(1 for row in evidence_rows if row.get("plan_status") == "ready")
    claim_modes = sorted(
        {
            str(row.get("claim_mode"))
            for row in evidence_rows
            if isinstance(row.get("claim_mode"), str) and str(row.get("claim_mode"))
        }
    )
    protocol_versions = sorted(
        {
            str(row.get("protocol_version"))
            for row in evidence_rows
            if isinstance(row.get("protocol_version"), str) and str(row.get("protocol_version"))
        }
    )

    if not rows:
        status_value = "not_available"
        reason_value = "no_geometry_optional_claim_attack_rows"
    elif not available_evidence:
        status_value = "not_available"
        reason_value = "no_geometry_optional_claim_boundary_resolution_available"
    elif boundary_hit_event_count <= 0 and boundary_resolution_failed_event_count <= 0:
        status_value = "not_applicable"
        reason_value = "no_attack_events_within_content_margin_boundary_subset"
    elif boundary_resolution_failed_event_count <= 0:
        status_value = "ok"
        reason_value = None
    else:
        status_value = "partial"
        reason_value = (
            f"geometry optional claim boundary resolution available for {len(available_evidence)}/{len(rows)} attack events"
        )

    return {
        "status": status_value,
        "reason": reason_value,
        "event_count": len(rows),
        "boundary_hit_event_count": boundary_hit_event_count,
        "content_failed_event_count": content_failed_event_count,
        "content_failed_subset_boundary_event_count": content_failed_subset_boundary_event_count,
        "boundary_subset_eligible_event_count": boundary_subset_eligible_event_count,
        "boundary_subset_rescue_applied_event_count": boundary_subset_rescue_applied_event_count,
        "evidence_scope_rescue_applied_event_count": evidence_scope_rescue_applied_event_count,
        "boundary_resolved_event_count": len(available_evidence),
        "boundary_excluded_event_count": boundary_excluded_event_count,
        "boundary_resolution_failed_event_count": boundary_resolution_failed_event_count,
        "plan_ready_event_count": plan_ready_event_count,
        "evidence_event_count": len(available_evidence),
        "missing_evidence_event_count": len(rows) - len(available_evidence),
        "supporting_evidence_event_count": supporting_evidence_event_count,
        "supporting_evidence_rate": (
            float(supporting_evidence_event_count / len(eligible_rows))
            if eligible_rows
            else None
        ),
        "sync_success_support_count": sync_success_support_count,
        "inverse_transform_support_count": inverse_transform_support_count,
        "attention_anchor_support_count": attention_anchor_support_count,
        "claim_modes": claim_modes,
        "protocol_versions": protocol_versions,
    }


def _build_geometry_optional_claim_summary_payload(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：构造 geometry optional claim 的 analysis-only 导出。 

    Build the analysis-only geometry optional-claim evidence export.

    Args:
        family_id: Family identifier.
        attack_event_rows: Materialized attack event rows.

    Returns:
        Optional-claim evidence payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    grouped_by_family: Dict[str, List[Dict[str, Any]]] = {}
    grouped_by_condition: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    evidence_rows = [_load_geometry_optional_claim_evidence(row) for row in attack_event_rows]
    claim_scopes = sorted(
        {
            str(row.get("claim_scope"))
            for row in evidence_rows
            if isinstance(row.get("claim_scope"), str) and str(row.get("claim_scope"))
        }
    )
    protocol_versions = sorted(
        {
            str(row.get("protocol_version"))
            for row in evidence_rows
            if isinstance(row.get("protocol_version"), str) and str(row.get("protocol_version"))
        }
    )
    boundary_metrics = sorted(
        {
            str(row.get("boundary_metric"))
            for row in evidence_rows
            if isinstance(row.get("boundary_metric"), str) and str(row.get("boundary_metric"))
        }
    )
    boundary_abs_margin_values = sorted(
        {
            float(cast(float, row["boundary_abs_margin_max"]))
            for row in evidence_rows
            if isinstance(row.get("boundary_abs_margin_max"), (int, float)) and not isinstance(row.get("boundary_abs_margin_max"), bool)
        }
    )
    for attack_event_row in attack_event_rows:
        attack_family = str(attack_event_row.get("attack_family") or "<unknown>")
        attack_condition_key = str(attack_event_row.get("attack_condition_key") or "<unknown>")
        grouped_by_family.setdefault(attack_family, []).append(dict(cast(Mapping[str, Any], attack_event_row)))
        grouped_by_condition.setdefault((attack_family, attack_condition_key), []).append(
            dict(cast(Mapping[str, Any], attack_event_row))
        )

    overall_summary = _build_geometry_optional_claim_summary(attack_event_rows)
    conditional_rescue_reference = _build_geometry_conditional_rescue_summary(attack_event_rows)
    readiness_status = "ready" if overall_summary["status"] in {"ok", "partial", "not_applicable"} else "not_ready"
    readiness_reason = None if readiness_status == "ready" else overall_summary["reason"]

    return {
        "artifact_type": "paper_workflow_pw04_geometry_optional_claim_summary",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "metric_name": "geometry_optional_claim_summary",
        "canonical": False,
        "analysis_only": True,
        "status": overall_summary["status"],
        "reason": overall_summary["reason"],
        "field_semantics": {
            "overall_scope": (
                "overall contains boundary / optional-claim evidence counts only; these fields are analysis-only "
                "and are not the formal paper rescue metrics"
            ),
            "conditional_rescue_reference_scope": (
                "conditional_rescue_reference mirrors the formal content_failed subset conditional rescue summary "
                "for reference only and is not flattened into overall"
            ),
            "boundary_subset_eligible_event_count": (
                "count of rows eligible for the optional claim; this is eligible-scope and not an evidence count"
            ),
            "boundary_subset_rescue_applied_event_count": (
                "count of geo_rescue_applied == True within eligible_rows; this is the eligible-scope applied count"
            ),
            "evidence_scope_rescue_applied_event_count": (
                "count of geo_rescue_applied == True within evidence_rows; this is the evidence-scope applied count"
            ),
            "supporting_evidence_event_count": (
                "count of rows with supporting_evidence_available == True; this is not a rescue-applied count"
            ),
            "geo_rescue_applied_on_content_failed_count": (
                "formal conditional rescue count on the content_failed subset from conditional_rescue_metrics"
            ),
            "geo_only_positive_on_content_failed_subset": (
                "formal rescue success count on the content_failed subset from conditional_rescue_metrics"
            ),
        },
        "claim_contract": {
            "claim_scope": claim_scopes[0] if len(claim_scopes) == 1 else claim_scopes,
            "claim_modes": overall_summary["claim_modes"],
            "content_positive_veto_allowed": False,
            "rescue_directionality": "one_way_positive_only",
            "protocol_version": protocol_versions[0] if len(protocol_versions) == 1 else protocol_versions,
            "boundary_metric": boundary_metrics[0] if len(boundary_metrics) == 1 else boundary_metrics,
            "boundary_abs_margin_max": (
                boundary_abs_margin_values[0]
                if len(boundary_abs_margin_values) == 1
                else boundary_abs_margin_values
            ),
            "eligibility_is_boundary_subset_only": True,
        },
        "readiness": {
            "status": readiness_status,
            "reason": readiness_reason,
            "required_for_formal_release": False,
            "blocking": False,
            "claim_scope": "geometry_optional_claim_optional",
        },
        "overall": overall_summary,
        "conditional_rescue_reference": conditional_rescue_reference,
        "by_attack_family": [
            {
                "attack_family": attack_family,
                **_build_geometry_optional_claim_summary(rows),
            }
            for attack_family, rows in sorted(grouped_by_family.items())
        ],
        "by_attack_condition": [
            {
                "attack_family": attack_family,
                "attack_condition_key": attack_condition_key,
                "attack_config_name": next(
                    (
                        row.get("attack_config_name")
                        for row in rows
                        if isinstance(row.get("attack_config_name"), str) and row.get("attack_config_name")
                    ),
                    None,
                ),
                "severity_label": next(
                    (
                        row.get("severity_label")
                        for row in rows
                        if isinstance(row.get("severity_label"), str) and row.get("severity_label")
                    ),
                    None,
                ),
                "severity_level_index": next(
                    (
                        row.get("severity_level_index")
                        for row in rows
                        if isinstance(row.get("severity_level_index"), int)
                    ),
                    None,
                ),
                **_build_geometry_optional_claim_summary(rows),
            }
            for (attack_family, attack_condition_key), rows in sorted(grouped_by_condition.items())
        ],
    }


def _normalize_geometry_optional_claim_csv_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：把 geometry optional claim 汇总行规范化为 CSV 友好格式。 

    Normalize one geometry optional-claim summary row for CSV export.

    Args:
        row: Summary row mapping.

    Returns:
        CSV-friendly row mapping.
    """
    if not isinstance(row, Mapping):
        raise TypeError("row must be Mapping")

    normalized_row = dict(cast(Mapping[str, Any], row))
    for field_name in ["claim_modes", "protocol_versions"]:
        field_value = normalized_row.get(field_name)
        if isinstance(field_value, list):
            normalized_row[field_name] = json.dumps(field_value, ensure_ascii=False)
    return normalized_row


def _build_geometry_optional_claim_export_rows(
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    功能：构造 geometry optional claim 的逐事件导出行。 

    Build per-event geometry optional-claim export rows.

    Args:
        attack_event_rows: Materialized attack event rows.

    Returns:
        Event-level export rows.
    """
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    export_rows: List[Dict[str, Any]] = []
    for attack_event_row in attack_event_rows:
        evidence_payload = _load_geometry_optional_claim_evidence(attack_event_row)
        boundary_subset_rescue_applied_count = int(
            evidence_payload.get("geo_rescue_eligible") is True
            and evidence_payload.get("geo_rescue_applied") is True
        )
        evidence_scope_rescue_applied_count = int(evidence_payload.get("geo_rescue_applied") is True)
        export_rows.append(
            {
                "attack_event_id": attack_event_row.get("attack_event_id"),
                "parent_event_id": attack_event_row.get("parent_event_id"),
                "attack_family": attack_event_row.get("attack_family"),
                "attack_condition_key": attack_event_row.get("attack_condition_key"),
                "attack_config_name": attack_event_row.get("attack_config_name"),
                "severity_label": attack_event_row.get("severity_label"),
                "severity_level_index": attack_event_row.get("severity_level_index"),
                "status": evidence_payload.get("status"),
                "reason": evidence_payload.get("reason"),
                "plan_status": evidence_payload.get("plan_status"),
                "plan_reason": evidence_payload.get("plan_reason"),
                "claim_mode": evidence_payload.get("claim_mode"),
                "claim_scope": evidence_payload.get("claim_scope"),
                "eligible_for_optional_claim": evidence_payload.get("eligible_for_optional_claim"),
                "geo_rescue_eligible": evidence_payload.get("geo_rescue_eligible"),
                "geo_rescue_applied": evidence_payload.get("geo_rescue_applied"),
                "boundary_subset_rescue_applied_count": boundary_subset_rescue_applied_count,
                "evidence_scope_rescue_applied_count": evidence_scope_rescue_applied_count,
                "supporting_evidence_available": evidence_payload.get("supporting_evidence_available"),
                "supporting_signal_count": evidence_payload.get("supporting_signal_count"),
                "sync_success": evidence_payload.get("sync_success"),
                "inverse_transform_success": evidence_payload.get("inverse_transform_success"),
                "attention_anchor_available": evidence_payload.get("attention_anchor_available"),
                "protocol_version": evidence_payload.get("protocol_version"),
                "boundary_metric": evidence_payload.get("boundary_metric"),
                "boundary_abs_margin_max": evidence_payload.get("boundary_abs_margin_max"),
                "boundary_metric_value": evidence_payload.get("boundary_metric_value"),
                "boundary_resolution_status": evidence_payload.get("boundary_resolution_status"),
                "boundary_resolution_reason": evidence_payload.get("boundary_resolution_reason"),
                "parent_content_margin": evidence_payload.get("parent_content_margin"),
                "parent_event_attestation_margin": evidence_payload.get("parent_event_attestation_margin"),
                "parent_source_detect_record_path": evidence_payload.get("parent_source_detect_record_path"),
                "content_threshold_export_path": evidence_payload.get("content_threshold_export_path"),
                "event_attestation_threshold_export_path": evidence_payload.get("event_attestation_threshold_export_path"),
            }
        )
    return export_rows


def _build_geometry_optional_claim_by_severity_rows(
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    功能：按 severity 聚合 geometry optional claim 汇总。 

    Build severity-aggregated geometry optional-claim summary rows.

    Args:
        attack_event_rows: Materialized attack event rows.

    Returns:
        Severity-level summary rows.
    """
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    grouped_rows: Dict[Tuple[int | None, str | None], List[Dict[str, Any]]] = {}
    for attack_event_row in attack_event_rows:
        severity_level_index = (
            int(cast(int, attack_event_row["severity_level_index"]))
            if isinstance(attack_event_row.get("severity_level_index"), int)
            else None
        )
        severity_label = (
            str(attack_event_row.get("severity_label"))
            if isinstance(attack_event_row.get("severity_label"), str) and str(attack_event_row.get("severity_label"))
            else None
        )
        grouped_rows.setdefault((severity_level_index, severity_label), []).append(
            dict(cast(Mapping[str, Any], attack_event_row))
        )

    severity_rows: List[Dict[str, Any]] = []
    for (severity_level_index, severity_label), rows in sorted(
        grouped_rows.items(),
        key=lambda item: (
            item[0][0] is None,
            item[0][0] if item[0][0] is not None else 10**9,
            item[0][1] or "",
        ),
    ):
        summary_row = _build_geometry_optional_claim_summary(rows)
        severity_rows.append(
            _normalize_geometry_optional_claim_csv_row(
                {
                    "severity_level_index": severity_level_index,
                    "severity_label": severity_label,
                    "attack_family_count": len(
                        {
                            str(row.get("attack_family"))
                            for row in rows
                            if isinstance(row.get("attack_family"), str) and str(row.get("attack_family"))
                        }
                    ),
                    "attack_condition_count": len(
                        {
                            str(row.get("attack_condition_key"))
                            for row in rows
                            if isinstance(row.get("attack_condition_key"), str) and str(row.get("attack_condition_key"))
                        }
                    ),
                    **summary_row,
                }
            )
        )
    return severity_rows


def _build_geometry_optional_claim_example_manifest(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：构造 geometry optional claim 示例清单。 

    Build the geometry optional-claim example manifest.

    Args:
        family_id: Family identifier.
        attack_event_rows: Materialized attack event rows.

    Returns:
        Example manifest payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    export_rows = _build_geometry_optional_claim_export_rows(attack_event_rows)
    eligible_rows = [row for row in export_rows if row.get("eligible_for_optional_claim") is True]
    ordered_rows = sorted(
        eligible_rows,
        key=lambda row: (
            0 if row.get("supporting_evidence_available") is True else 1,
            _coerce_finite_float(row.get("boundary_metric_value")) if _coerce_finite_float(row.get("boundary_metric_value")) is not None else float("inf"),
            str(row.get("attack_family") or ""),
            str(row.get("attack_event_id") or ""),
        ),
    )
    selected_rows: List[Dict[str, Any]] = []
    selected_families: set[str] = set()
    for row in ordered_rows:
        attack_family = str(row.get("attack_family") or "<unknown>")
        if attack_family in selected_families:
            continue
        selected_rows.append(dict(row))
        selected_families.add(attack_family)

    if selected_rows:
        status_value = "ok"
        reason_value = None
    else:
        status_value = "not_available"
        reason_value = "no_geometry_optional_claim_examples_within_boundary_subset"

    return {
        "artifact_type": "paper_workflow_pw04_geometry_optional_claim_example_manifest",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "status": status_value,
        "reason": reason_value,
        "selection_protocol": {
            "eligible_required": True,
            "max_examples_per_attack_family": 1,
            "sort_order": [
                "supporting_evidence_available_desc",
                "boundary_metric_value_asc",
                "attack_event_id_asc",
            ],
        },
        "boundary_subset_eligible_event_count": len(eligible_rows),
        "example_count": len(selected_rows),
        "rows": selected_rows,
    }


def _summarize_wrong_event_outcomes(
    rows: Sequence[Mapping[str, Any]],
    *,
    evaluation_mode: str,
) -> Dict[str, Any]:
    """
    功能：按 clean 或 attack 视角汇总 wrong-event FAR。 

    Summarize wrong-event rejection and false-accept outcomes for one view.

    Args:
        rows: Wrong-event challenge rows.
        evaluation_mode: Either clean or attack.

    Returns:
        FAR summary mapping.
    """
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be Sequence")
    if evaluation_mode not in {"clean", "attack"}:
        raise ValueError(f"unsupported evaluation_mode: {evaluation_mode}")

    attempted_event_count = 0
    wrong_event_rejected_count = 0
    wrong_event_false_accept_count = 0
    for row in rows:
        if evaluation_mode == "clean":
            verify_status = row.get("clean_parent_verify_status")
            rejected_flag = row.get("clean_parent_wrong_event_rejected")
            if not isinstance(rejected_flag, bool):
                rejected_flag = row.get("wrong_event_rejected")
        else:
            verify_status = row.get("attack_verify_status")
            rejected_flag = row.get("attack_wrong_event_rejected")

        verify_token = str(verify_status) if isinstance(verify_status, str) and verify_status else None
        if isinstance(rejected_flag, bool) or verify_token in {"rejected", "false_accept"}:
            attempted_event_count += 1
            if rejected_flag is True or verify_token == "rejected":
                wrong_event_rejected_count += 1
            elif rejected_flag is False or verify_token == "false_accept":
                wrong_event_false_accept_count += 1

    wrong_event_far = (
        float(wrong_event_false_accept_count / attempted_event_count)
        if attempted_event_count > 0
        else None
    )
    return {
        "attempted_event_count": attempted_event_count,
        "wrong_event_rejected_count": wrong_event_rejected_count,
        "wrong_event_false_accept_count": wrong_event_false_accept_count,
        "wrong_event_far": wrong_event_far,
    }


def _build_wrong_event_far_metric_payload(
    *,
    family_id: str,
    metric_name: str,
    summary_path: Path,
    metric_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：构造显式 wrong-event FAR 导出。 

    Build one explicit wrong-event FAR export payload.

    Args:
        family_id: Family identifier.
        metric_name: Export metric name.
        summary_path: Source summary path.
        metric_summary: FAR summary mapping.

    Returns:
        Explicit FAR export payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(metric_name, str) or not metric_name:
        raise TypeError("metric_name must be non-empty str")
    if not isinstance(summary_path, Path):
        raise TypeError("summary_path must be Path")
    if not isinstance(metric_summary, Mapping):
        raise TypeError("metric_summary must be Mapping")

    attempted_event_count = int(metric_summary.get("attempted_event_count", 0) or 0)
    status_value = "ok" if attempted_event_count > 0 else "not_available"
    reason_value = None if attempted_event_count > 0 else f"no_{metric_name}_attempt_available"
    metric_value = _coerce_finite_float(metric_summary.get("wrong_event_far"))
    payload = {
        "artifact_type": "paper_workflow_pw04_wrong_event_far_export",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "metric_name": metric_name,
        "status": status_value,
        "reason": reason_value,
        "canonical": False,
        "analysis_only": True,
        "source_artifacts": {
            "wrong_event_attestation_challenge_summary_path": normalize_path_value(summary_path),
        },
        "attempted_event_count": attempted_event_count,
        "wrong_event_rejected_count": int(metric_summary.get("wrong_event_rejected_count", 0) or 0),
        "wrong_event_false_accept_count": int(metric_summary.get("wrong_event_false_accept_count", 0) or 0),
        "metric_value": metric_value,
    }
    payload[metric_name] = metric_value
    return payload


def _build_wrong_event_far_by_challenge_type_payload(
    *,
    family_id: str,
    summary_path: Path,
    challenge_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：按 challenge_type 构造 wrong-event FAR 导出。 

    Build the explicit wrong-event FAR-by-challenge-type export.

    Args:
        family_id: Family identifier.
        summary_path: Source summary path.
        challenge_rows: Wrong-event challenge rows.

    Returns:
        Challenge-type FAR export payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(summary_path, Path):
        raise TypeError("summary_path must be Path")
    if not isinstance(challenge_rows, Sequence):
        raise TypeError("challenge_rows must be Sequence")

    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for challenge_row in challenge_rows:
        challenge_type = str(challenge_row.get("challenge_type") or "wrong_statement")
        grouped_rows.setdefault(challenge_type, []).append(dict(cast(Mapping[str, Any], challenge_row)))

    output_rows: List[Dict[str, Any]] = []
    for challenge_type, rows in sorted(grouped_rows.items()):
        clean_summary = _summarize_wrong_event_outcomes(rows, evaluation_mode="clean")
        attack_summary = _summarize_wrong_event_outcomes(rows, evaluation_mode="attack")
        output_rows.append(
            {
                "challenge_type": challenge_type,
                "event_count": len(rows),
                "clean_attempted_event_count": clean_summary["attempted_event_count"],
                "clean_wrong_event_rejected_count": clean_summary["wrong_event_rejected_count"],
                "clean_wrong_event_false_accept_count": clean_summary["wrong_event_false_accept_count"],
                "wrong_event_far_clean": clean_summary["wrong_event_far"],
                "attack_attempted_event_count": attack_summary["attempted_event_count"],
                "attack_wrong_event_rejected_count": attack_summary["wrong_event_rejected_count"],
                "attack_wrong_event_false_accept_count": attack_summary["wrong_event_false_accept_count"],
                "wrong_event_far_attack": attack_summary["wrong_event_far"],
            }
        )

    return {
        "artifact_type": "paper_workflow_pw04_wrong_event_far_by_challenge_type_export",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "metric_name": "wrong_event_far_by_challenge_type",
        "status": "ok" if output_rows else "not_available",
        "reason": None if output_rows else "no_wrong_event_rows_available",
        "canonical": False,
        "analysis_only": True,
        "source_artifacts": {
            "wrong_event_attestation_challenge_summary_path": normalize_path_value(summary_path),
        },
        "wrong_event_far_by_challenge_type": output_rows,
        "rows": output_rows,
    }


def _build_payload_attack_summary_payload(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：基于 LF detect trace 构造 payload robustness 汇总。

    Build the append-only payload robustness summary from LF detect-side traces.

    Args:
        family_id: Family identifier.
        attack_event_rows: Positive attacked-event rows.

    Returns:
        Payload robustness summary payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    row_metrics: List[Dict[str, Any]] = []
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for attack_event_row in attack_event_rows:
        formal_record = _extract_mapping(attack_event_row.get("formal_record"))
        content_payload = _extract_mapping(formal_record.get("content_evidence_payload"))
        score_parts = _extract_mapping(content_payload.get("score_parts"))
        lf_trace = _extract_mapping(score_parts.get("lf_trajectory_detect_trace"))
        if not lf_trace:
            lf_trace = _extract_mapping(content_payload.get("lf_evidence_summary"))
        if not lf_trace:
            lf_trace = _extract_mapping(score_parts.get("lf_metrics"))
        decode_sidecar_metrics = _load_optional_payload_decode_sidecar_metrics(
            attack_event_row,
            f"PW04 payload decode sidecar {attack_event_row.get('attack_event_id')}",
        )
        codeword_agreement = (
            float(cast(float, decode_sidecar_metrics["codeword_agreement"]))
            if isinstance(decode_sidecar_metrics.get("codeword_agreement"), float)
            else _coerce_finite_float(lf_trace.get("codeword_agreement"))
        )
        n_bits_compared = (
            int(cast(int, decode_sidecar_metrics["n_bits_compared"]))
            if isinstance(decode_sidecar_metrics.get("n_bits_compared"), int)
            else _parse_csv_int(lf_trace, "n_bits_compared")
        )
        payload_primary_metrics = _derive_payload_primary_metrics(
            {
                "codeword_agreement": codeword_agreement,
                "n_bits_compared": n_bits_compared,
            }
        )
        if isinstance(decode_sidecar_metrics.get("message_decode_success"), bool):
            payload_primary_metrics["message_success"] = bool(decode_sidecar_metrics["message_decode_success"])
        attestation_payload = _extract_mapping(formal_record.get("attestation"))
        attested_decision = _extract_mapping(attestation_payload.get("final_event_attested_decision"))
        attack_family = str(attack_event_row.get("attack_family") or "<unknown>")
        metric_row = {
            "attack_event_id": attack_event_row.get("attack_event_id"),
            "attack_family": attack_family,
            "attack_condition_key": attack_event_row.get("attack_condition_key"),
            "codeword_agreement": codeword_agreement,
            "n_bits_compared": n_bits_compared,
            **payload_primary_metrics,
            "event_attestation_score": _coerce_finite_float(attested_decision.get("event_attestation_score")),
            "is_event_attested": attested_decision.get("is_event_attested") if isinstance(attested_decision.get("is_event_attested"), bool) else None,
            "lf_detect_variant": (
                str(decode_sidecar_metrics.get("lf_detect_variant"))
                if isinstance(decode_sidecar_metrics.get("lf_detect_variant"), str) and str(decode_sidecar_metrics.get("lf_detect_variant"))
                else (lf_trace.get("detect_variant") if isinstance(lf_trace.get("detect_variant"), str) else attack_event_row.get("lf_detect_variant"))
            ),
            "message_source": (
                str(decode_sidecar_metrics.get("message_source"))
                if isinstance(decode_sidecar_metrics.get("message_source"), str) and str(decode_sidecar_metrics.get("message_source"))
                else (lf_trace.get("message_source") if isinstance(lf_trace.get("message_source"), str) else None)
            ),
            "payload_reference_sidecar_path": (
                attack_event_row.get("payload_reference_sidecar_path")
                if isinstance(attack_event_row.get("payload_reference_sidecar_path"), str)
                else None
            ),
            "payload_decode_sidecar_path": (
                attack_event_row.get("payload_decode_sidecar_path")
                if isinstance(attack_event_row.get("payload_decode_sidecar_path"), str)
                else None
            ),
            "payload_probe_mode": decode_sidecar_metrics.get("payload_probe_mode"),
            "payload_probe_available": decode_sidecar_metrics.get("payload_probe_available"),
            "payload_probe_status": decode_sidecar_metrics.get("payload_probe_status"),
            "payload_probe_reason": decode_sidecar_metrics.get("payload_probe_reason"),
            "payload_probe_source": decode_sidecar_metrics.get("payload_probe_source"),
            "payload_probe_reconstruction_applied": decode_sidecar_metrics.get("payload_probe_reconstruction_applied"),
            "payload_probe_alignment_signal_available": decode_sidecar_metrics.get("payload_probe_alignment_signal_available"),
            "payload_probe_consistency_score": decode_sidecar_metrics.get("payload_probe_consistency_score"),
            "payload_probe_bp_converged": decode_sidecar_metrics.get("payload_probe_bp_converged"),
            "probe_margin_threshold": decode_sidecar_metrics.get("probe_margin_threshold"),
            "probe_reference_n_bits": decode_sidecar_metrics.get("probe_reference_n_bits"),
            "probe_effective_n_bits": decode_sidecar_metrics.get("probe_effective_n_bits"),
            "probe_agreement_count": decode_sidecar_metrics.get("probe_agreement_count"),
            "probe_bit_accuracy": decode_sidecar_metrics.get("probe_bit_accuracy"),
            "probe_support_rate": decode_sidecar_metrics.get("probe_support_rate"),
        }
        row_metrics.append(metric_row)
        grouped_rows.setdefault(attack_family, []).append(metric_row)

    available_rows = [row for row in row_metrics if isinstance(row.get("codeword_agreement"), float)]

    def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        available = [row for row in rows if isinstance(row.get("codeword_agreement"), float)]
        agreement_values = [float(row["codeword_agreement"]) for row in available]
        bits_compared_values = [
            int(cast(int, row["n_bits_compared"]))
            for row in available
            if isinstance(row.get("n_bits_compared"), int)
        ]
        bit_accuracy_values = [float(row["bit_accuracy"]) for row in rows if isinstance(row.get("bit_accuracy"), float)]
        bit_error_rate_values = [float(row["bit_error_rate"]) for row in rows if isinstance(row.get("bit_error_rate"), float)]
        weighted_bit_accuracy_numerator = sum(
            float(cast(float, row["bit_accuracy"])) * int(cast(int, row["n_bits_compared"]))
            for row in rows
            if isinstance(row.get("bit_accuracy"), float) and isinstance(row.get("n_bits_compared"), int)
        )
        weighted_bit_accuracy_denominator = sum(
            int(cast(int, row["n_bits_compared"]))
            for row in rows
            if isinstance(row.get("bit_accuracy"), float) and isinstance(row.get("n_bits_compared"), int)
        )
        attestation_values = [
            float(row["event_attestation_score"])
            for row in rows
            if isinstance(row.get("event_attestation_score"), float)
        ]
        message_success_values = [row["message_success"] for row in rows if isinstance(row.get("message_success"), bool)]
        detect_variants = sorted(
            {
                str(row.get("lf_detect_variant"))
                for row in available
                if isinstance(row.get("lf_detect_variant"), str) and str(row.get("lf_detect_variant"))
            }
        )
        message_sources = sorted(
            {
                str(row.get("message_source"))
                for row in available
                if isinstance(row.get("message_source"), str) and str(row.get("message_source"))
            }
        )
        return {
            "event_count": len(rows),
            "available_payload_event_count": len(available),
            "missing_payload_event_count": len(rows) - len(available),
            "mean_codeword_agreement": _safe_mean(agreement_values),
            "min_codeword_agreement": min(agreement_values) if agreement_values else None,
            "max_codeword_agreement": max(agreement_values) if agreement_values else None,
            "mean_n_bits_compared": _safe_mean(bits_compared_values),
            "mean_bit_accuracy": _safe_mean(bit_accuracy_values),
            "weighted_bit_accuracy": (
                float(weighted_bit_accuracy_numerator / weighted_bit_accuracy_denominator)
                if weighted_bit_accuracy_denominator > 0
                else None
            ),
            "mean_bit_error_rate": _safe_mean(bit_error_rate_values),
            "weighted_bit_error_rate": (
                float(1.0 - (weighted_bit_accuracy_numerator / weighted_bit_accuracy_denominator))
                if weighted_bit_accuracy_denominator > 0
                else None
            ),
            "message_success_count": sum(1 for value in message_success_values if value is True),
            "message_success_rate": (
                float(sum(1 for value in message_success_values if value is True) / len(message_success_values))
                if message_success_values
                else None
            ),
            "payload_primary_metric_sources": sorted(
                {
                    str(row.get("primary_metric_source"))
                    for row in rows
                    if isinstance(row.get("primary_metric_source"), str) and str(row.get("primary_metric_source"))
                }
            ),
            "attested_event_count": sum(1 for row in rows if row.get("is_event_attested") is True),
            "mean_event_attestation_score": _safe_mean(attestation_values),
            "lf_detect_variants": detect_variants,
            "message_sources": message_sources,
        }

    overall_summary = _summarize(row_metrics)
    if not available_rows:
        status_value = "not_available"
        reason_value = PAYLOAD_UNAVAILABLE_REASON
    elif len(available_rows) == len(row_metrics):
        status_value = "ok"
        reason_value = None
    else:
        status_value = "partial"
        reason_value = f"payload trace available for {len(available_rows)}/{len(row_metrics)} attack events"

    if status_value == "ok":
        readiness_status = "ready"
        readiness_reason = None
    elif status_value == "partial":
        readiness_status = "partial"
        readiness_reason = reason_value
    else:
        readiness_status = "not_ready"
        readiness_reason = reason_value or PAYLOAD_UNAVAILABLE_REASON

    return {
        "artifact_type": "paper_workflow_pw04_payload_attack_summary",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "status": status_value,
        "reason": reason_value,
        "future_upstream_sidecar_required": status_value != "ok",
        "readiness": {
            "status": readiness_status,
            "reason": readiness_reason,
            "required_for_formal_release": True,
            "blocking": readiness_status != "ready",
            "gap_classification": (
                "partial_upstream_result" if readiness_status == "partial" else (
                    "upstream_result_unavailable" if readiness_status == "not_ready" else None
                )
            ),
        },
        "probe_overall": summarize_payload_probe_rows(
            rows=row_metrics,
            unavailable_reason=PAYLOAD_UNAVAILABLE_REASON,
        ),
        "overall": overall_summary,
        "by_attack_family": [
            {
                "attack_family": attack_family,
                "probe_overall": summarize_payload_probe_rows(
                    rows=rows,
                    unavailable_reason=PAYLOAD_UNAVAILABLE_REASON,
                ),
                **_summarize(rows),
            }
            for attack_family, rows in sorted(grouped_rows.items())
        ],
        "probe_by_attack_family": [
            {
                "attack_family": attack_family,
                **summarize_payload_probe_rows(
                    rows=rows,
                    unavailable_reason=PAYLOAD_UNAVAILABLE_REASON,
                ),
            }
            for attack_family, rows in sorted(grouped_rows.items())
        ],
    }


def _build_wrong_event_attestation_challenge_summary_payload(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    功能：构造 wrong-event attestation challenge 汇总。

    Build the wrong-event attestation challenge summary using PW03
    per-event challenge records.

    Args:
        family_id: Family identifier.
        attack_event_rows: Positive attacked-event rows.

    Returns:
        Challenge summary payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    challenge_rows: List[Dict[str, Any]] = []
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for attack_event_row in attack_event_rows:
        attack_family = str(attack_event_row.get("attack_family") or "<unknown>")
        attack_event_id = attack_event_row.get("attack_event_id")
        parent_event_id = attack_event_row.get("parent_event_id")
        event_manifest = _extract_mapping(attack_event_row.get("event_manifest"))
        challenge_record = _load_optional_wrong_event_attestation_challenge_record(event_manifest)
        if not challenge_record:
            row_payload = {
                "attack_event_id": attack_event_id,
                "attack_family": attack_family,
                "attack_condition_key": attack_event_row.get("attack_condition_key"),
                "parent_event_id": parent_event_id,
                "source_event_id": parent_event_id,
                "challenged_event_id": None,
                "challenge_parent_event_id": None,
                "challenge_type": "wrong_statement",
                "status": "missing_upstream_challenge_record",
                "reason": "PW03 wrong-event attestation challenge record missing",
                "plan_status": None,
                "plan_reason": None,
                "binding_status": None,
                "verify_status": None,
                "bundle_verification_status": None,
                "bundle_verification_mismatch_reasons": [],
                "wrong_statement_digest": None,
                "bundle_attestation_digest": None,
                "wrong_event_rejected": None,
                "clean_parent_binding_status": None,
                "clean_parent_verify_status": None,
                "clean_parent_wrong_event_rejected": None,
                "attack_binding_status": None,
                "attack_bundle_verification_status": None,
                "attack_attestation_digest": None,
                "attack_verify_status": None,
                "attack_wrong_event_rejected": None,
            }
            challenge_rows.append(row_payload)
            grouped_rows.setdefault(attack_family, []).append(row_payload)
            continue

        row_payload = {
            "attack_event_id": challenge_record.get("attack_event_id", attack_event_id),
            "attack_family": attack_family,
            "attack_condition_key": attack_event_row.get("attack_condition_key"),
            "parent_event_id": challenge_record.get("parent_event_id", parent_event_id),
            "source_event_id": challenge_record.get("source_event_id", parent_event_id),
            "challenged_event_id": challenge_record.get("challenged_event_id"),
            "challenge_parent_event_id": challenge_record.get("challenge_parent_event_id"),
            "challenge_type": challenge_record.get("challenge_type", "wrong_statement"),
            "status": challenge_record.get("status", "missing_upstream_challenge_record"),
            "reason": challenge_record.get("reason"),
            "plan_status": challenge_record.get("plan_status"),
            "plan_reason": challenge_record.get("plan_reason"),
            "binding_status": challenge_record.get("binding_status"),
            "verify_status": challenge_record.get("verify_status"),
            "bundle_verification_status": challenge_record.get("bundle_verification_status"),
            "bundle_verification_mismatch_reasons": list(
                challenge_record.get("bundle_verification_mismatch_reasons") or []
            ),
            "wrong_statement_digest": challenge_record.get("wrong_statement_digest"),
            "bundle_attestation_digest": challenge_record.get("bundle_attestation_digest"),
            "wrong_event_rejected": challenge_record.get("wrong_event_rejected"),
            "clean_parent_binding_status": challenge_record.get("clean_parent_binding_status"),
            "clean_parent_verify_status": challenge_record.get("clean_parent_verify_status"),
            "clean_parent_wrong_event_rejected": challenge_record.get("clean_parent_wrong_event_rejected"),
            "attack_binding_status": challenge_record.get("attack_binding_status"),
            "attack_bundle_verification_status": challenge_record.get("attack_bundle_verification_status"),
            "attack_attestation_digest": challenge_record.get("attack_attestation_digest"),
            "attack_verify_status": challenge_record.get("attack_verify_status"),
            "attack_wrong_event_rejected": challenge_record.get("attack_wrong_event_rejected"),
        }
        challenge_rows.append(row_payload)
        grouped_rows.setdefault(attack_family, []).append(row_payload)

    def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        clean_summary = _summarize_wrong_event_outcomes(rows, evaluation_mode="clean")
        attack_summary = _summarize_wrong_event_outcomes(rows, evaluation_mode="attack")
        return {
            "event_count": len(rows),
            "attempted_event_count": clean_summary["attempted_event_count"],
            "bundle_verified_count": clean_summary["attempted_event_count"],
            "wrong_event_rejected_count": clean_summary["wrong_event_rejected_count"],
            "wrong_event_false_accept_count": clean_summary["wrong_event_false_accept_count"],
            "wrong_event_rejection_rate": (
                float(
                    clean_summary["wrong_event_rejected_count"]
                    / clean_summary["attempted_event_count"]
                )
                if clean_summary["attempted_event_count"] > 0
                else None
            ),
            "wrong_event_far_clean": clean_summary["wrong_event_far"],
            "attack_attempted_event_count": attack_summary["attempted_event_count"],
            "attack_wrong_event_rejected_count": attack_summary["wrong_event_rejected_count"],
            "attack_wrong_event_false_accept_count": attack_summary["wrong_event_false_accept_count"],
            "wrong_event_far_attack": attack_summary["wrong_event_far"],
        }

    overall_summary = _summarize(challenge_rows)
    challenge_type_grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for challenge_row in challenge_rows:
        challenge_type = str(challenge_row.get("challenge_type") or "wrong_statement")
        challenge_type_grouped_rows.setdefault(challenge_type, []).append(
            dict(cast(Mapping[str, Any], challenge_row))
        )
    wrong_event_far_by_challenge_type = [
        {
            "challenge_type": challenge_type,
            "event_count": len(rows),
            "clean_attempted_event_count": clean_summary["attempted_event_count"],
            "clean_wrong_event_rejected_count": clean_summary["wrong_event_rejected_count"],
            "clean_wrong_event_false_accept_count": clean_summary["wrong_event_false_accept_count"],
            "wrong_event_far_clean": clean_summary["wrong_event_far"],
            "attack_attempted_event_count": attack_summary["attempted_event_count"],
            "attack_wrong_event_rejected_count": attack_summary["wrong_event_rejected_count"],
            "attack_wrong_event_false_accept_count": attack_summary["wrong_event_false_accept_count"],
            "wrong_event_far_attack": attack_summary["wrong_event_far"],
        }
        for challenge_type, rows in sorted(challenge_type_grouped_rows.items())
        for clean_summary, attack_summary in [
            (
                _summarize_wrong_event_outcomes(rows, evaluation_mode="clean"),
                _summarize_wrong_event_outcomes(rows, evaluation_mode="attack"),
            )
        ]
    ]
    if overall_summary["event_count"] <= 0:
        status_value = "not_available"
        reason_value = "no_positive_attack_events_available_for_wrong_event_challenge"
    elif overall_summary["attempted_event_count"] <= 0:
        status_value = "not_available"
        distinct_reasons = sorted(
            {
                str(row.get("reason"))
                for row in challenge_rows
                if isinstance(row.get("reason"), str) and str(row.get("reason"))
            }
        )
        reason_value = (
            distinct_reasons[0]
            if len(distinct_reasons) == 1
            else "no_valid_upstream_wrong_event_challenge_attempt_available"
        )
    elif overall_summary["wrong_event_false_accept_count"] > 0:
        status_value = "mismatch"
        reason_value = (
            f"wrong-event bundle binding failed for "
            f"{overall_summary['wrong_event_false_accept_count']}/{overall_summary['attempted_event_count']} attempts"
        )
    elif overall_summary["attempted_event_count"] < len(challenge_rows):
        status_value = "partial"
        reason_value = (
            f"wrong-event challenge executed for {overall_summary['attempted_event_count']}/{len(challenge_rows)} attack events"
        )
    else:
        status_value = "ok"
        reason_value = None

    if status_value == "ok":
        readiness_status = "ready"
        readiness_reason = None
    elif status_value == "partial":
        readiness_status = "partial"
        readiness_reason = reason_value
    else:
        readiness_status = "not_ready"
        readiness_reason = reason_value

    return {
        "artifact_type": "paper_workflow_pw04_wrong_event_attestation_challenge_summary",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "status": status_value,
        "reason": reason_value,
        "future_upstream_sidecar_required": False,
        "readiness": {
            "status": readiness_status,
            "reason": readiness_reason,
            "required_for_formal_release": True,
            "blocking": readiness_status != "ready",
            "gap_classification": (
                "partial_upstream_result" if readiness_status == "partial" else (
                    "upstream_result_unavailable" if readiness_status == "not_ready" else None
                )
            ),
        },
        "overall": overall_summary,
        "by_attack_family": [
            {
                "attack_family": attack_family,
                **_summarize(rows),
            }
            for attack_family, rows in sorted(grouped_rows.items())
        ],
        "wrong_event_far_by_challenge_type": wrong_event_far_by_challenge_type,
        "rows": challenge_rows,
    }


def _build_clean_imperceptibility_payload(
    *,
    family_id: str,
    clean_quality_metrics_payload: Mapping[str, Any],
    clean_quality_metrics_path: Path,
) -> Dict[str, Any]:
    """
    功能：显式导出 clean imperceptibility 语义。 

    Build the explicit clean imperceptibility export from the legacy clean
    quality metrics payload.

    Args:
        family_id: Family identifier.
        clean_quality_metrics_payload: PW04 clean quality payload.
        clean_quality_metrics_path: PW04 clean quality JSON path.

    Returns:
        Clean imperceptibility export payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(clean_quality_metrics_payload, Mapping):
        raise TypeError("clean_quality_metrics_payload must be Mapping")
    if not isinstance(clean_quality_metrics_path, Path):
        raise TypeError("clean_quality_metrics_path must be Path")

    overall_payload = _extract_mapping(clean_quality_metrics_payload.get("overall"))
    pair_count = _parse_csv_int(overall_payload, "count")
    status_value = overall_payload.get("status")
    reason_value = overall_payload.get("availability_reason")
    if not isinstance(status_value, str) or not status_value:
        status_value = "not_available"
    if not isinstance(reason_value, str) or not reason_value:
        reason_value = None

    return {
        "artifact_type": "paper_workflow_pw04_clean_imperceptibility_export",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "metric_name": "clean_imperceptibility",
        "status": status_value,
        "reason": reason_value,
        "canonical": False,
        "analysis_only": True,
        "fallback_source": "pw04_clean_quality_metrics.overall",
        "legacy_scope_name": "content_chain",
        "pair_semantics": {
            "scene": "clean",
            "reference_artifact": "plain_preview_image",
            "candidate_artifact": "watermarked_output_image",
            "semantic_definition": "imperceptibility",
            "directionality": "larger psnr/ssim and smaller lpips indicate stronger clean imperceptibility",
        },
        "source_artifacts": {
            "clean_quality_metrics_path": normalize_path_value(clean_quality_metrics_path),
        },
        "metrics": {
            "pair_count": pair_count,
            "mean_psnr": _coerce_finite_float(overall_payload.get("mean_psnr")),
            "mean_ssim": _coerce_finite_float(overall_payload.get("mean_ssim")),
            "mean_lpips": _coerce_finite_float(overall_payload.get("mean_lpips")),
            "mean_clip_text_similarity": _coerce_finite_float(overall_payload.get("mean_clip_text_similarity")),
            "clip_model_name": overall_payload.get("clip_model_name"),
            "lpips_status": overall_payload.get("lpips_status"),
            "lpips_reason": overall_payload.get("lpips_reason"),
            "clip_status": overall_payload.get("clip_status"),
            "clip_reason": overall_payload.get("clip_reason"),
        },
    }


def _build_attack_distortion_payload(
    *,
    family_id: str,
    attack_quality_metrics_payload: Mapping[str, Any],
    attack_quality_metrics_path: Path,
) -> Dict[str, Any]:
    """
    功能：显式导出 attack distortion 语义。 

    Build the explicit attack distortion export from the legacy PW04 attack
    quality summary payload.

    Args:
        family_id: Family identifier.
        attack_quality_metrics_payload: PW04 attack quality payload.
        attack_quality_metrics_path: PW04 attack quality JSON path.

    Returns:
        Attack distortion export payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_quality_metrics_payload, Mapping):
        raise TypeError("attack_quality_metrics_payload must be Mapping")
    if not isinstance(attack_quality_metrics_path, Path):
        raise TypeError("attack_quality_metrics_path must be Path")

    overall_payload = _extract_mapping(attack_quality_metrics_payload.get("overall"))
    status_value = overall_payload.get("status")
    reason_value = overall_payload.get("reason")
    if not isinstance(status_value, str) or not status_value:
        status_value = "not_available"
    if not isinstance(reason_value, str) or not reason_value:
        reason_value = None

    return {
        "artifact_type": "paper_workflow_pw04_attack_distortion_export",
        "schema_version": "pw_stage_04_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "metric_name": "attack_distortion",
        "status": status_value,
        "reason": reason_value,
        "canonical": False,
        "analysis_only": True,
        "fallback_source": "pw04_attack_quality_metrics.overall",
        "pair_semantics": {
            "scene": "attack",
            "reference_artifact": "watermarked_parent_image",
            "candidate_artifact": "attacked_image",
            "semantic_definition": "distortion",
            "directionality": "larger psnr/ssim and smaller lpips indicate smaller attack distortion",
        },
        "source_artifacts": {
            "attack_quality_metrics_path": normalize_path_value(attack_quality_metrics_path),
        },
        "metrics": {
            "pair_count": _parse_csv_int(overall_payload, "pair_count"),
            "mean_psnr": _coerce_finite_float(overall_payload.get("mean_psnr")),
            "mean_ssim": _coerce_finite_float(overall_payload.get("mean_ssim")),
            "mean_lpips": _coerce_finite_float(overall_payload.get("mean_lpips")),
            "mean_clip_text_similarity": _coerce_finite_float(overall_payload.get("mean_clip_text_similarity")),
            "clip_model_name": overall_payload.get("clip_model_name"),
            "lpips_status": overall_payload.get("lpips_status"),
            "lpips_reason": overall_payload.get("lpips_reason"),
            "clip_status": overall_payload.get("clip_status"),
            "clip_reason": overall_payload.get("clip_reason"),
        },
    }


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
    clean_quality_metrics_payload: Mapping[str, Any],
    clean_quality_metrics_path: Path,
    attack_quality_metrics_payload: Mapping[str, Any],
    attack_quality_metrics_path: Path,
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
        clean_quality_metrics_payload: PW04 clean quality payload.
        clean_quality_metrics_path: PW04 clean quality path.
        attack_quality_metrics_payload: PW04 attack quality payload.
        attack_quality_metrics_path: PW04 attack quality path.
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
    if not isinstance(clean_quality_metrics_payload, Mapping):
        raise TypeError("clean_quality_metrics_payload must be Mapping")
    if not isinstance(clean_quality_metrics_path, Path):
        raise TypeError("clean_quality_metrics_path must be Path")
    if not isinstance(attack_quality_metrics_payload, Mapping):
        raise TypeError("attack_quality_metrics_payload must be Mapping")
    if not isinstance(attack_quality_metrics_path, Path):
        raise TypeError("attack_quality_metrics_path must be Path")
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
    geometry_conditional_rows = _collect_geometry_conditional_rows(attack_event_rows)

    robustness_curve_by_family_path = robustness_dir / "robustness_curve_by_family.csv"
    robustness_macro_summary_path = robustness_dir / "robustness_macro_summary.csv"
    worst_case_attack_summary_path = robustness_dir / "worst_case_attack_summary.csv"
    geo_chain_usage_by_family_path = geometry_dir / "geo_chain_usage_by_family.csv"
    geo_diagnostics_summary_path = geometry_dir / "geo_diagnostics_summary.csv"
    geo_diagnostics_conditional_metrics_path = geometry_dir / "geo_diagnostics_conditional_metrics.csv"
    conditional_rescue_metrics_path = geometry_dir / "conditional_rescue_metrics.json"
    geometry_optional_claim_summary_path = geometry_dir / "geometry_optional_claim_summary.json"
    geometry_optional_claim_by_family_path = geometry_dir / "geometry_optional_claim_by_family.csv"
    geometry_optional_claim_by_severity_path = geometry_dir / "geometry_optional_claim_by_severity.csv"
    geometry_optional_claim_example_manifest_path = geometry_dir / "geometry_optional_claim_example_manifest.json"
    payload_attack_summary_path = payload_dir / "payload_attack_summary.json"
    wrong_event_attestation_challenge_summary_path = payload_dir / "wrong_event_attestation_challenge_summary.json"
    wrong_event_far_clean_path = payload_dir / "wrong_event_far_clean.json"
    wrong_event_far_attack_path = payload_dir / "wrong_event_far_attack.json"
    wrong_event_far_by_challenge_type_path = payload_dir / "wrong_event_far_by_challenge_type.json"
    quality_robustness_tradeoff_path = tradeoff_dir / "quality_robustness_tradeoff.csv"
    quality_robustness_frontier_path = tradeoff_dir / "quality_robustness_frontier.png"
    clean_imperceptibility_path = tradeoff_dir / "clean_imperceptibility.json"
    attack_distortion_path = tradeoff_dir / "attack_distortion.json"

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
    _write_csv_rows(
        geo_diagnostics_conditional_metrics_path,
        [
            "geometry_condition_name",
            "geometry_condition_value",
            "event_count",
            "parent_event_count",
            "attack_family_count",
            "formal_final_positive_count",
            "formal_final_decision_attack_tpr",
            "formal_attestation_positive_count",
            "formal_attestation_attack_tpr",
            "derived_attack_union_positive_count",
            "derived_attack_union_attack_tpr",
            "geo_helped_positive_count",
            "geo_helped_positive_rate",
        ],
        geometry_conditional_rows,
    )
    write_json_atomic(
        conditional_rescue_metrics_path,
        _build_geometry_conditional_rescue_metrics_payload(
            family_id=family_id,
            attack_event_rows=attack_event_rows,
        ),
    )
    geometry_optional_claim_summary_payload = _build_geometry_optional_claim_summary_payload(
        family_id=family_id,
        attack_event_rows=attack_event_rows,
    )
    write_json_atomic(
        geometry_optional_claim_summary_path,
        geometry_optional_claim_summary_payload,
    )
    _write_csv_rows(
        geometry_optional_claim_by_family_path,
        [
            "attack_family",
            "status",
            "reason",
            "event_count",
            "boundary_hit_event_count",
            "content_failed_event_count",
            "content_failed_subset_boundary_event_count",
            "boundary_subset_eligible_event_count",
            "boundary_subset_rescue_applied_event_count",
            "evidence_scope_rescue_applied_event_count",
            "boundary_resolved_event_count",
            "boundary_excluded_event_count",
            "boundary_resolution_failed_event_count",
            "plan_ready_event_count",
            "evidence_event_count",
            "missing_evidence_event_count",
            "supporting_evidence_event_count",
            "supporting_evidence_rate",
            "sync_success_support_count",
            "inverse_transform_support_count",
            "attention_anchor_support_count",
            "claim_modes",
            "protocol_versions",
        ],
        [
            _normalize_geometry_optional_claim_csv_row(row)
            for row in cast(Sequence[Mapping[str, Any]], geometry_optional_claim_summary_payload.get("by_attack_family", []))
        ],
    )
    _write_csv_rows(
        geometry_optional_claim_by_severity_path,
        [
            "severity_level_index",
            "severity_label",
            "attack_family_count",
            "attack_condition_count",
            "status",
            "reason",
            "event_count",
            "boundary_hit_event_count",
            "content_failed_event_count",
            "content_failed_subset_boundary_event_count",
            "boundary_subset_eligible_event_count",
            "boundary_subset_rescue_applied_event_count",
            "evidence_scope_rescue_applied_event_count",
            "boundary_resolved_event_count",
            "boundary_excluded_event_count",
            "boundary_resolution_failed_event_count",
            "plan_ready_event_count",
            "evidence_event_count",
            "missing_evidence_event_count",
            "supporting_evidence_event_count",
            "supporting_evidence_rate",
            "sync_success_support_count",
            "inverse_transform_support_count",
            "attention_anchor_support_count",
            "claim_modes",
            "protocol_versions",
        ],
        _build_geometry_optional_claim_by_severity_rows(attack_event_rows),
    )
    write_json_atomic(
        geometry_optional_claim_example_manifest_path,
        _build_geometry_optional_claim_example_manifest(
            family_id=family_id,
            attack_event_rows=attack_event_rows,
        ),
    )
    write_json_atomic(
        payload_attack_summary_path,
        _build_payload_attack_summary_payload(
            family_id=family_id,
            attack_event_rows=attack_event_rows,
        ),
    )
    wrong_event_attestation_challenge_summary_payload = _build_wrong_event_attestation_challenge_summary_payload(
        family_id=family_id,
        attack_event_rows=attack_event_rows,
    )
    write_json_atomic(
        wrong_event_attestation_challenge_summary_path,
        wrong_event_attestation_challenge_summary_payload,
    )
    write_json_atomic(
        wrong_event_far_clean_path,
        _build_wrong_event_far_metric_payload(
            family_id=family_id,
            metric_name="wrong_event_far_clean",
            summary_path=wrong_event_attestation_challenge_summary_path,
            metric_summary={
                "attempted_event_count": _extract_mapping(
                    wrong_event_attestation_challenge_summary_payload.get("overall")
                ).get("attempted_event_count"),
                "wrong_event_rejected_count": _extract_mapping(
                    wrong_event_attestation_challenge_summary_payload.get("overall")
                ).get("wrong_event_rejected_count"),
                "wrong_event_false_accept_count": _extract_mapping(
                    wrong_event_attestation_challenge_summary_payload.get("overall")
                ).get("wrong_event_false_accept_count"),
                "wrong_event_far": _extract_mapping(
                    wrong_event_attestation_challenge_summary_payload.get("overall")
                ).get("wrong_event_far_clean"),
            },
        ),
    )
    write_json_atomic(
        wrong_event_far_attack_path,
        _build_wrong_event_far_metric_payload(
            family_id=family_id,
            metric_name="wrong_event_far_attack",
            summary_path=wrong_event_attestation_challenge_summary_path,
            metric_summary={
                "attempted_event_count": _extract_mapping(
                    wrong_event_attestation_challenge_summary_payload.get("overall")
                ).get("attack_attempted_event_count"),
                "wrong_event_rejected_count": _extract_mapping(
                    wrong_event_attestation_challenge_summary_payload.get("overall")
                ).get("attack_wrong_event_rejected_count"),
                "wrong_event_false_accept_count": _extract_mapping(
                    wrong_event_attestation_challenge_summary_payload.get("overall")
                ).get("attack_wrong_event_false_accept_count"),
                "wrong_event_far": _extract_mapping(
                    wrong_event_attestation_challenge_summary_payload.get("overall")
                ).get("wrong_event_far_attack"),
            },
        ),
    )
    write_json_atomic(
        wrong_event_far_by_challenge_type_path,
        _build_wrong_event_far_by_challenge_type_payload(
            family_id=family_id,
            summary_path=wrong_event_attestation_challenge_summary_path,
            challenge_rows=cast(
                Sequence[Mapping[str, Any]],
                wrong_event_attestation_challenge_summary_payload.get("rows", []),
            ),
        ),
    )

    clean_quality_overall = _extract_mapping(clean_quality_metrics_payload.get("overall"))
    attack_quality_overall = _extract_mapping(attack_quality_metrics_payload.get("overall"))
    write_json_atomic(
        clean_imperceptibility_path,
        _build_clean_imperceptibility_payload(
            family_id=family_id,
            clean_quality_metrics_payload=clean_quality_metrics_payload,
            clean_quality_metrics_path=clean_quality_metrics_path,
        ),
    )
    write_json_atomic(
        attack_distortion_path,
        _build_attack_distortion_payload(
            family_id=family_id,
            attack_quality_metrics_payload=attack_quality_metrics_payload,
            attack_quality_metrics_path=attack_quality_metrics_path,
        ),
    )

    tradeoff_rows: List[Dict[str, Any]] = []
    for robustness_row in robustness_macro_rows:
        tradeoff_rows.append(
            {
                "scope": robustness_row.get("scope"),
                "clean_quality_scope": "content_chain",
                "clean_quality_status": clean_quality_overall.get("status", "not_available"),
                "clean_mean_psnr": clean_quality_overall.get("mean_psnr"),
                "clean_mean_ssim": clean_quality_overall.get("mean_ssim"),
                "clean_mean_lpips": clean_quality_overall.get("mean_lpips"),
                "attack_macro_avg_tpr": robustness_row.get("macro_avg_attack_tpr"),
                "overall_attack_tpr": robustness_row.get("overall_attack_tpr"),
                "attack_mean_lpips": attack_quality_overall.get("mean_lpips"),
                "attack_lpips_status": attack_quality_overall.get("lpips_status", "not_available"),
                "attack_lpips_reason": attack_quality_overall.get("lpips_reason"),
                "attack_mean_clip_text_similarity": attack_quality_overall.get("mean_clip_text_similarity"),
                "attack_clip_model_name": attack_quality_overall.get("clip_model_name"),
                "attack_clip_status": attack_quality_overall.get("clip_status", "not_available"),
                "attack_clip_reason": attack_quality_overall.get("clip_reason"),
                "attack_clip_sample_count": attack_quality_overall.get("clip_sample_count"),
                "clean_quality_metrics_path": normalize_path_value(clean_quality_metrics_path),
                "attack_quality_metrics_path": normalize_path_value(attack_quality_metrics_path),
                "robustness_macro_summary_path": normalize_path_value(robustness_macro_summary_path),
                "lpips_status": clean_quality_overall.get("lpips_status", "not_available"),
                "lpips_reason": clean_quality_overall.get("lpips_reason"),
                "clip_status": clean_quality_overall.get("clip_status", "not_available"),
                "clip_reason": clean_quality_overall.get("clip_reason"),
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
            "clean_quality_metrics_path",
            "attack_quality_metrics_path",
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
        "geo_diagnostics_conditional_metrics_path": normalize_path_value(
            geo_diagnostics_conditional_metrics_path
        ),
        "conditional_rescue_metrics_path": normalize_path_value(conditional_rescue_metrics_path),
        "geometry_optional_claim_summary_path": normalize_path_value(geometry_optional_claim_summary_path),
        "geometry_optional_claim_by_family_path": normalize_path_value(geometry_optional_claim_by_family_path),
        "geometry_optional_claim_by_severity_path": normalize_path_value(geometry_optional_claim_by_severity_path),
        "geometry_optional_claim_example_manifest_path": normalize_path_value(geometry_optional_claim_example_manifest_path),
        "payload_attack_summary_path": normalize_path_value(payload_attack_summary_path),
        "wrong_event_attestation_challenge_summary_path": normalize_path_value(wrong_event_attestation_challenge_summary_path),
        "wrong_event_far_clean_path": normalize_path_value(wrong_event_far_clean_path),
        "wrong_event_far_attack_path": normalize_path_value(wrong_event_far_attack_path),
        "wrong_event_far_by_challenge_type_path": normalize_path_value(wrong_event_far_by_challenge_type_path),
        "clean_quality_metrics_path": normalize_path_value(clean_quality_metrics_path),
        "clean_imperceptibility_path": normalize_path_value(clean_imperceptibility_path),
        "attack_distortion_path": normalize_path_value(attack_distortion_path),
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
            "pw04_conditional_rescue_metrics": normalize_path_value(conditional_rescue_metrics_path),
            "pw04_geometry_optional_claim_summary": normalize_path_value(geometry_optional_claim_summary_path),
            "pw04_geometry_optional_claim_by_family": normalize_path_value(geometry_optional_claim_by_family_path),
            "pw04_geometry_optional_claim_by_severity": normalize_path_value(geometry_optional_claim_by_severity_path),
            "pw04_geometry_optional_claim_example_manifest": normalize_path_value(geometry_optional_claim_example_manifest_path),
            "pw04_payload_attack_summary": normalize_path_value(payload_attack_summary_path),
            "pw04_wrong_event_attestation_challenge_summary": normalize_path_value(wrong_event_attestation_challenge_summary_path),
            "pw04_wrong_event_far_clean": normalize_path_value(wrong_event_far_clean_path),
            "pw04_wrong_event_far_attack": normalize_path_value(wrong_event_far_attack_path),
            "pw04_wrong_event_far_by_challenge_type": normalize_path_value(wrong_event_far_by_challenge_type_path),
            "pw04_clean_imperceptibility": normalize_path_value(clean_imperceptibility_path),
            "pw04_attack_distortion": normalize_path_value(attack_distortion_path),
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