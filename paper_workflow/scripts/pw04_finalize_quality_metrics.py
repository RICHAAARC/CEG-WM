"""
File purpose: Finalize PW04 clean and attack quality metrics from shard outputs.
Module type: Semi-general module
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from scripts.notebook_runtime_common import ensure_directory, normalize_path_value, utc_now_iso, write_json_atomic


SCHEMA_VERSION = "pw_stage_04_v1"
QUALITY_FINALIZE_MANIFEST_FILE_NAME = "quality_finalize_manifest.json"


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

    Normalize one optional mapping node to dict.

    Args:
        node: Candidate mapping node.

    Returns:
        Plain dict payload.
    """
    return dict(cast(Mapping[str, Any], node)) if isinstance(node, Mapping) else {}


def _coerce_finite_float(value: Any) -> float | None:
    """
    功能：把候选值解析为有限浮点数。

    Coerce one candidate value into a finite float.

    Args:
        value: Candidate scalar value.

    Returns:
        Finite float when available, otherwise None.
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


def _coerce_non_negative_int(value: Any) -> int | None:
    """
    功能：把输入解析为非负整数。

    Parse one optional non-negative integer value.

    Args:
        value: Candidate numeric value.

    Returns:
        Parsed non-negative integer when available, otherwise None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0.0:
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed_value = int(float(value))
        except ValueError:
            return None
        return parsed_value if parsed_value >= 0 else None
    return None


def _safe_mean(values: Sequence[float]) -> float | None:
    """
    功能：计算有限数值序列的均值。

    Compute the mean of a finite numeric sequence.

    Args:
        values: Numeric sequence.

    Returns:
        Mean value when available, otherwise None.
    """
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


def _build_grouped_attack_quality_rows(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    group_key_name: str,
) -> List[Dict[str, Any]]:
    """
    功能：按攻击族或条件聚合 attack quality 行。

    Build grouped attack quality rows for one categorical key.

    Args:
        pair_rows: Per-pair attack quality rows.
        group_key_name: Grouping field name.

    Returns:
        Grouped attack quality rows.
    """
    grouped_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for pair_row in pair_rows:
        if not isinstance(pair_row, Mapping):
            raise TypeError("pair_rows items must be mappings")
        if pair_row.get("status") != "ok":
            continue
        group_value = pair_row.get(group_key_name)
        if not isinstance(group_value, str) or not group_value:
            raise ValueError(f"pair row missing {group_key_name}")
        grouped_rows.setdefault(group_value, []).append(pair_row)

    output_rows: List[Dict[str, Any]] = []
    for group_value in sorted(grouped_rows):
        rows = grouped_rows[group_value]
        psnr_values = [
            float(row["psnr"])
            for row in rows
            if _coerce_finite_float(row.get("psnr")) is not None
        ]
        ssim_values = [
            float(row["ssim"])
            for row in rows
            if _coerce_finite_float(row.get("ssim")) is not None
        ]
        lpips_values = [
            float(row["lpips"])
            for row in rows
            if _coerce_finite_float(row.get("lpips")) is not None
        ]
        clip_values = [
            float(row["clip_text_similarity"])
            for row in rows
            if _coerce_finite_float(row.get("clip_text_similarity")) is not None
        ]
        grouped_row: Dict[str, Any] = {
            group_key_name: group_value,
            "quality_pair_count": len(psnr_values),
            "mean_psnr": _safe_mean(psnr_values),
            "mean_ssim": _safe_mean(ssim_values),
            "mean_lpips": _safe_mean(lpips_values),
            "mean_clip_text_similarity": _safe_mean(clip_values),
            "clip_sample_count": len(clip_values),
        }
        if group_key_name == "attack_condition_key":
            grouped_row["attack_family"] = rows[0].get("attack_family")
            grouped_row["attack_config_name"] = rows[0].get("attack_config_name")
        output_rows.append(grouped_row)
    return output_rows


def _merge_quality_runtime(shard_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    """
    功能：合并并校验 quality runtime 配置一致性。

    Merge and validate quality runtime settings across shard summaries.

    Args:
        shard_summaries: Shard-local quality summaries.

    Returns:
        Merged runtime payload when available, otherwise None.
    """
    runtime_payloads = [
        _extract_mapping(summary.get("quality_runtime"))
        for summary in shard_summaries
        if isinstance(summary, Mapping) and _extract_mapping(summary.get("quality_runtime"))
    ]
    if not runtime_payloads:
        return None
    canonical_text = json.dumps(runtime_payloads[0], ensure_ascii=False, sort_keys=True)
    for runtime_payload in runtime_payloads[1:]:
        if json.dumps(runtime_payload, ensure_ascii=False, sort_keys=True) != canonical_text:
            raise ValueError("PW04 quality shard runtime options must be identical across shard outputs")
    return runtime_payloads[0]


def _aggregate_quality_summary(
    *,
    shard_summaries: Sequence[Mapping[str, Any]],
    expected_pair_count: int,
) -> Dict[str, Any]:
    """
    功能：把多个 shard 质量摘要合并为统一总体摘要。

    Aggregate multiple shard quality summaries into one global summary.

    Args:
        shard_summaries: Shard-local quality summaries.
        expected_pair_count: Expected global pair count.

    Returns:
        Aggregated quality summary.
    """
    if not isinstance(expected_pair_count, int) or isinstance(expected_pair_count, bool) or expected_pair_count < 0:
        raise TypeError("expected_pair_count must be non-negative int")

    pair_rows: List[Dict[str, Any]] = []
    lpips_reasons: List[str] = []
    clip_reasons: List[str] = []
    for shard_summary in shard_summaries:
        if not isinstance(shard_summary, Mapping):
            raise TypeError("shard_summaries items must be mappings")
        pair_rows.extend(
            dict(cast(Mapping[str, Any], pair_row))
            for pair_row in cast(List[Any], shard_summary.get("pair_rows", []))
            if isinstance(pair_row, Mapping)
        )
        lpips_reason = shard_summary.get("lpips_reason")
        clip_reason = shard_summary.get("clip_reason")
        if isinstance(lpips_reason, str) and lpips_reason.strip():
            lpips_reasons.append(lpips_reason)
        if isinstance(clip_reason, str) and clip_reason.strip():
            clip_reasons.append(clip_reason)

    successful_rows = [row for row in pair_rows if row.get("status") == "ok"]
    successful_count = len(successful_rows)
    missing_count = sum(1 for row in pair_rows if str(row.get("status", "")).startswith("missing"))
    error_count = sum(1 for row in pair_rows if row.get("status") == "error")
    psnr_values = [float(row["psnr"]) for row in successful_rows if _coerce_finite_float(row.get("psnr")) is not None]
    ssim_values = [float(row["ssim"]) for row in successful_rows if _coerce_finite_float(row.get("ssim")) is not None]
    lpips_values = [float(row["lpips"]) for row in successful_rows if _coerce_finite_float(row.get("lpips")) is not None]
    clip_values = [
        float(row["clip_text_similarity"])
        for row in successful_rows
        if _coerce_finite_float(row.get("clip_text_similarity")) is not None
    ]
    clip_missing_text_count = sum(
        1
        for row in successful_rows
        if _coerce_finite_float(row.get("clip_text_similarity")) is None
        and not (isinstance(row.get("failure_reason"), str) and row.get("failure_reason"))
    )
    runtime_payload = _merge_quality_runtime(shard_summaries)
    prompt_text_expected = any(bool(summary.get("prompt_text_expected")) for summary in shard_summaries)
    prompt_text_available_count = (
        sum(_coerce_non_negative_int(summary.get("prompt_text_available_count")) or 0 for summary in shard_summaries)
        if prompt_text_expected
        else None
    )
    prompt_text_missing_count = (
        sum(_coerce_non_negative_int(summary.get("prompt_text_missing_count")) or 0 for summary in shard_summaries)
        if prompt_text_expected
        else None
    )

    status_value = "ok" if successful_count > 0 else "unavailable"
    availability_reason = None if successful_count > 0 else "no_valid_image_pairs"
    lpips_status = "ok" if lpips_values else "not_available"
    lpips_reason = None if lpips_status == "ok" else (lpips_reasons[0] if lpips_reasons else "LPIPS model unavailable")
    clip_sample_count = len(clip_values)
    clip_model_name = next(
        (
            summary.get("clip_model_name")
            for summary in shard_summaries
            if isinstance(summary.get("clip_model_name"), str) and str(summary.get("clip_model_name")).strip()
        ),
        None,
    )
    if not prompt_text_expected:
        prompt_text_coverage_status = "not_configured"
        prompt_text_coverage_reason = "prompt text key not configured for CLIP quality metric"
    elif successful_count <= 0:
        prompt_text_coverage_status = "not_available"
        prompt_text_coverage_reason = "no_valid_image_pairs"
    elif (prompt_text_missing_count or 0) <= 0:
        prompt_text_coverage_status = "ok"
        prompt_text_coverage_reason = None
    elif (prompt_text_missing_count or 0) < successful_count:
        prompt_text_coverage_status = "partial"
        prompt_text_coverage_reason = (
            f"prompt text unavailable for {prompt_text_missing_count}/{successful_count} valid image pairs"
        )
    else:
        prompt_text_coverage_status = "not_available"
        prompt_text_coverage_reason = (
            f"prompt text unavailable for all {prompt_text_missing_count} valid image pairs"
        )

    if not prompt_text_expected:
        clip_status = "not_available"
        clip_reason = "prompt text key not configured for CLIP quality metric"
        clip_model_name = None
    elif clip_sample_count > 0 and (prompt_text_missing_count or 0) == 0 and not clip_reasons:
        clip_status = "ok"
        clip_reason = None
    elif clip_sample_count > 0:
        clip_status = "partial"
        partial_reasons: List[str] = []
        if (prompt_text_missing_count or 0) > 0:
            partial_reasons.append(
                f"prompt text unavailable for {prompt_text_missing_count}/{successful_count} valid image pairs"
            )
        partial_reasons.extend(reason for reason in clip_reasons if reason)
        clip_reason = "; ".join(dict.fromkeys(partial_reasons)) if partial_reasons else None
    elif successful_count <= 0:
        clip_status = "not_available"
        clip_reason = "no_valid_image_pairs"
    elif (prompt_text_missing_count or 0) > 0:
        clip_status = "not_available"
        clip_reason = f"prompt text unavailable for all {prompt_text_missing_count} valid image pairs"
    else:
        clip_status = "not_available"
        clip_reason = clip_reasons[0] if clip_reasons else "CLIP model unavailable"

    quality_readiness_reasons: List[str] = []
    if successful_count <= 0:
        quality_readiness_status = "not_ready"
        quality_readiness_reason = "no_valid_image_pairs"
    else:
        if successful_count != expected_pair_count:
            quality_readiness_reasons.append(
                f"valid image pairs available for {successful_count}/{expected_pair_count} expected bindings"
            )
        if lpips_status != "ok":
            quality_readiness_reasons.append(lpips_reason or "LPIPS model unavailable")
        if prompt_text_expected and prompt_text_coverage_status not in {"ok", "not_configured"}:
            quality_readiness_reasons.append(
                prompt_text_coverage_reason or "prompt text coverage incomplete"
            )
        if prompt_text_expected and clip_status != "ok":
            quality_readiness_reasons.append(clip_reason or "CLIP model unavailable")
        if quality_readiness_reasons:
            quality_readiness_status = "partial"
            quality_readiness_reason = "; ".join(dict.fromkeys(quality_readiness_reasons))
        else:
            quality_readiness_status = "ready"
            quality_readiness_reason = None

    return {
        "status": status_value,
        "availability_reason": availability_reason,
        "expected_count": expected_pair_count,
        "count": successful_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "mean_psnr": _safe_mean(psnr_values),
        "mean_ssim": _safe_mean(ssim_values),
        "lpips_status": lpips_status,
        "lpips_reason": lpips_reason,
        "mean_lpips": _safe_mean(lpips_values),
        "mean_clip_text_similarity": _safe_mean(clip_values),
        "clip_model_name": clip_model_name,
        "clip_sample_count": clip_sample_count,
        "clip_status": clip_status,
        "clip_reason": clip_reason,
        "quality_runtime": runtime_payload,
        "prompt_text_expected": prompt_text_expected,
        "prompt_text_available_count": prompt_text_available_count,
        "prompt_text_missing_count": prompt_text_missing_count,
        "prompt_text_coverage_status": prompt_text_coverage_status,
        "prompt_text_coverage_reason": prompt_text_coverage_reason,
        "quality_readiness_status": quality_readiness_status,
        "quality_readiness_reason": quality_readiness_reason,
        "quality_readiness_blocking": quality_readiness_status != "ready",
        "quality_readiness_required_for_formal_release": True,
        "pair_rows": pair_rows,
    }


def _validate_expected_pair_ids(
    *,
    planned_pairs: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    id_key: str,
) -> None:
    """
    功能：校验 finalize 后的 pair id 集与 plan 完全一致。

    Validate that finalized pair ids exactly match the plan.

    Args:
        planned_pairs: Planned pair specifications.
        pair_rows: Finalized pair rows.
        id_key: Stable pair identifier key.

    Returns:
        None.
    """
    expected_ids = {
        str(pair_row[id_key])
        for pair_row in planned_pairs
        if isinstance(pair_row, Mapping) and isinstance(pair_row.get(id_key), str) and pair_row.get(id_key)
    }
    actual_ids = {
        str(pair_row["pair_id"])
        for pair_row in pair_rows
        if isinstance(pair_row, Mapping) and isinstance(pair_row.get("pair_id"), str) and pair_row.get("pair_id")
    }
    if actual_ids != expected_ids:
        raise ValueError(
            f"PW04 finalized quality pair ids do not match plan for {id_key}: expected={len(expected_ids)}, actual={len(actual_ids)}"
        )


def _build_clean_quality_metrics_payload(
    *,
    family_id: str,
    quality_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：构建 PW04 clean quality metrics 导出。

    Build the PW04 clean quality metrics export payload.

    Args:
        family_id: Family identifier.
        quality_summary: Aggregated clean quality summary.

    Returns:
        Clean quality metrics payload.
    """
    return {
        "artifact_type": "paper_workflow_pw04_clean_quality_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "reference_semantics": "preview_generation_persisted_artifact_vs_watermarked_output_image",
        "overall": {
            key_name: quality_summary.get(key_name)
            for key_name in [
                "status",
                "availability_reason",
                "expected_count",
                "count",
                "missing_count",
                "error_count",
                "mean_psnr",
                "mean_ssim",
                "lpips_status",
                "lpips_reason",
                "mean_lpips",
                "mean_clip_text_similarity",
                "clip_model_name",
                "clip_sample_count",
                "clip_status",
                "clip_reason",
                "quality_runtime",
                "prompt_text_expected",
                "prompt_text_available_count",
                "prompt_text_missing_count",
                "prompt_text_coverage_status",
                "prompt_text_coverage_reason",
                "quality_readiness_status",
                "quality_readiness_reason",
                "quality_readiness_blocking",
                "quality_readiness_required_for_formal_release",
            ]
        },
        "pair_rows": list(cast(List[Any], quality_summary.get("pair_rows", []))),
    }


def _build_attack_quality_metrics_payload(
    *,
    family_id: str,
    quality_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：构建 PW04 attack quality metrics 导出。

    Build the PW04 attack quality metrics export payload.

    Args:
        family_id: Family identifier.
        quality_summary: Aggregated attack quality summary.

    Returns:
        Attack quality metrics payload.
    """
    pair_rows = cast(List[Mapping[str, Any]], quality_summary.get("pair_rows", []))
    return {
        "artifact_type": "paper_workflow_pw04_attack_quality_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "reference_semantics": "parent_source_image_vs_attacked_image",
        "overall": {
            key_name: quality_summary.get(key_name)
            for key_name in [
                "status",
                "availability_reason",
                "expected_count",
                "count",
                "missing_count",
                "error_count",
                "mean_psnr",
                "mean_ssim",
                "lpips_status",
                "lpips_reason",
                "mean_lpips",
                "mean_clip_text_similarity",
                "clip_model_name",
                "clip_sample_count",
                "clip_status",
                "clip_reason",
                "quality_runtime",
                "prompt_text_expected",
                "prompt_text_available_count",
                "prompt_text_missing_count",
                "prompt_text_coverage_status",
                "prompt_text_coverage_reason",
                "quality_readiness_status",
                "quality_readiness_reason",
                "quality_readiness_blocking",
                "quality_readiness_required_for_formal_release",
            ]
        },
        "by_attack_family": _build_grouped_attack_quality_rows(
            pair_rows=pair_rows,
            group_key_name="attack_family",
        ),
        "by_attack_condition": _build_grouped_attack_quality_rows(
            pair_rows=pair_rows,
            group_key_name="attack_condition_key",
        ),
        "pair_rows": list(pair_rows),
    }


def run_pw04_finalize_quality_metrics(
    *,
    family_id: str,
    quality_pair_plan_path: Path,
    clean_quality_metrics_path: Path,
    attack_quality_metrics_path: Path,
    quality_shard_paths: Sequence[Path],
) -> Dict[str, Any]:
    """
    功能：从 PW04 quality shard 输出合并 clean 与 attack 质量工件。

    Finalize the PW04 clean and attack quality artifacts from shard outputs.

    Args:
        family_id: Family identifier.
        quality_pair_plan_path: Quality pair plan path.
        clean_quality_metrics_path: Final clean quality metrics path.
        attack_quality_metrics_path: Final attack quality metrics path.
        quality_shard_paths: Ordered shard output paths.

    Returns:
        Finalized clean/attack quality export summary.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(quality_pair_plan_path, Path):
        raise TypeError("quality_pair_plan_path must be Path")
    if not isinstance(clean_quality_metrics_path, Path):
        raise TypeError("clean_quality_metrics_path must be Path")
    if not isinstance(attack_quality_metrics_path, Path):
        raise TypeError("attack_quality_metrics_path must be Path")
    if not isinstance(quality_shard_paths, Sequence):
        raise TypeError("quality_shard_paths must be Sequence")

    quality_pair_plan = _load_required_json_dict(quality_pair_plan_path, "PW04 quality pair plan")
    if quality_pair_plan.get("family_id") != family_id:
        raise ValueError(
            f"PW04 quality pair plan family_id mismatch: expected={family_id}, actual={quality_pair_plan.get('family_id')}"
        )
    clean_pairs = cast(List[Mapping[str, Any]], quality_pair_plan.get("clean_pairs", []))
    attack_pairs = cast(List[Mapping[str, Any]], quality_pair_plan.get("attack_pairs", []))
    clean_shard_summaries: List[Dict[str, Any]] = []
    attack_shard_summaries: List[Dict[str, Any]] = []
    for shard_path in quality_shard_paths:
        if not isinstance(shard_path, Path):
            raise TypeError("quality_shard_paths items must be Path")
        shard_payload = _load_required_json_dict(shard_path, f"PW04 quality shard {shard_path.name}")
        if shard_payload.get("family_id") != family_id:
            raise ValueError(
                f"PW04 quality shard family_id mismatch: expected={family_id}, actual={shard_payload.get('family_id')}"
            )
        clean_shard_summaries.append(_extract_mapping(shard_payload.get("clean_quality_summary")))
        attack_shard_summaries.append(_extract_mapping(shard_payload.get("attack_quality_summary")))

    clean_quality_summary = _aggregate_quality_summary(
        shard_summaries=clean_shard_summaries,
        expected_pair_count=len(clean_pairs),
    )
    attack_quality_summary = _aggregate_quality_summary(
        shard_summaries=attack_shard_summaries,
        expected_pair_count=len(attack_pairs),
    )
    _validate_expected_pair_ids(
        planned_pairs=clean_pairs,
        pair_rows=cast(List[Mapping[str, Any]], clean_quality_summary.get("pair_rows", [])),
        id_key="event_id",
    )
    _validate_expected_pair_ids(
        planned_pairs=attack_pairs,
        pair_rows=cast(List[Mapping[str, Any]], attack_quality_summary.get("pair_rows", [])),
        id_key="attack_event_id",
    )

    clean_quality_payload = _build_clean_quality_metrics_payload(
        family_id=family_id,
        quality_summary=clean_quality_summary,
    )
    attack_quality_payload = _build_attack_quality_metrics_payload(
        family_id=family_id,
        quality_summary=attack_quality_summary,
    )
    ensure_directory(clean_quality_metrics_path.parent)
    ensure_directory(attack_quality_metrics_path.parent)
    write_json_atomic(clean_quality_metrics_path, clean_quality_payload)
    write_json_atomic(attack_quality_metrics_path, attack_quality_payload)

    quality_finalize_manifest_path = quality_pair_plan_path.parent / QUALITY_FINALIZE_MANIFEST_FILE_NAME
    finalize_manifest_payload = {
        "artifact_type": "paper_workflow_pw04_quality_finalize_manifest",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "status": "completed",
        "quality_pair_plan_path": normalize_path_value(quality_pair_plan_path),
        "quality_shard_paths": [normalize_path_value(path_obj) for path_obj in quality_shard_paths],
        "clean_quality_metrics_path": normalize_path_value(clean_quality_metrics_path),
        "attack_quality_metrics_path": normalize_path_value(attack_quality_metrics_path),
        "clean_expected_pair_count": len(clean_pairs),
        "attack_expected_pair_count": len(attack_pairs),
    }
    write_json_atomic(quality_finalize_manifest_path, finalize_manifest_payload)

    return {
        "clean_quality_metrics_path": normalize_path_value(clean_quality_metrics_path),
        "attack_quality_metrics_path": normalize_path_value(attack_quality_metrics_path),
        "quality_finalize_manifest_path": normalize_path_value(quality_finalize_manifest_path),
        "clean_quality_payload": clean_quality_payload,
        "attack_quality_payload": attack_quality_payload,
        "clean_pair_lookup": {
            str(pair_row["pair_id"]): dict(pair_row)
            for pair_row in cast(List[Mapping[str, Any]], clean_quality_summary.get("pair_rows", []))
            if isinstance(pair_row.get("pair_id"), str) and pair_row.get("pair_id")
        },
        "attack_pair_lookup": {
            str(pair_row["pair_id"]): dict(pair_row)
            for pair_row in cast(List[Mapping[str, Any]], attack_quality_summary.get("pair_rows", []))
            if isinstance(pair_row.get("pair_id"), str) and pair_row.get("pair_id")
        },
    }