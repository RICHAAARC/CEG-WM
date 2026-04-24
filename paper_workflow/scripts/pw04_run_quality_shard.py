"""
File purpose: Execute one PW04 quality shard over clean and attack pair plans.
Module type: Semi-general module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast

from paper_workflow.scripts.pw_quality_metrics import build_quality_metrics_from_pairs
from scripts.notebook_runtime_common import (
    ensure_directory,
    normalize_path_value,
    utc_now_iso,
    write_json_atomic_compact,
)


SCHEMA_VERSION = "pw_stage_04_v1"


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


def _build_clean_quality_summary(
    pair_specs: Sequence[Mapping[str, Any]],
    *,
    phase_profile_output_path: Path | None = None,
    phase_profile_label: str | None = None,
) -> Dict[str, Any]:
    """
    功能：计算一个 quality shard 的 clean 质量摘要。

    Compute one clean quality summary for a shard-local pair collection.

    Args:
        pair_specs: Clean pair specifications.

    Returns:
        Raw quality summary payload.
    """
    return build_quality_metrics_from_pairs(
        pair_specs=pair_specs,
        reference_path_key="reference_image_path",
        candidate_path_key="candidate_image_path",
        pair_id_key="event_id",
        text_key="prompt_text",
        extra_metadata_keys=["pair_namespace", "sample_role", "plain_preview_image_path", "watermarked_output_image_path"],
        enable_phase_profiler=True,
        phase_profile_output_path=phase_profile_output_path,
        phase_profile_label=phase_profile_label,
    )


def _build_attack_quality_summary(
    pair_specs: Sequence[Mapping[str, Any]],
    *,
    phase_profile_output_path: Path | None = None,
    phase_profile_label: str | None = None,
) -> Dict[str, Any]:
    """
    功能：计算一个 quality shard 的 attack 质量摘要。

    Compute one attack quality summary for a shard-local pair collection.

    Args:
        pair_specs: Attack pair specifications.

    Returns:
        Raw quality summary payload.
    """
    return build_quality_metrics_from_pairs(
        pair_specs=pair_specs,
        reference_path_key="reference_image_path",
        candidate_path_key="candidate_image_path",
        pair_id_key="attack_event_id",
        text_key="prompt_text",
        extra_metadata_keys=["pair_namespace", "parent_event_id", "attack_family", "attack_condition_key", "attack_config_name"],
        enable_phase_profiler=True,
        phase_profile_output_path=phase_profile_output_path,
        phase_profile_label=phase_profile_label,
    )


def run_pw04_quality_shard(
    *,
    family_id: str,
    quality_pair_plan_path: Path,
    quality_shard_index: int,
) -> Dict[str, Any]:
    """
    功能：执行单个 PW04 quality shard。

    Execute one PW04 quality shard over the prepared clean and attack pairs.

    Args:
        family_id: Family identifier.
        quality_pair_plan_path: Prepared quality pair plan path.
        quality_shard_index: Zero-based shard index.

    Returns:
        Quality shard export summary.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(quality_pair_plan_path, Path):
        raise TypeError("quality_pair_plan_path must be Path")
    if not isinstance(quality_shard_index, int) or isinstance(quality_shard_index, bool) or quality_shard_index < 0:
        raise TypeError("quality_shard_index must be non-negative int")

    quality_pair_plan = _load_required_json_dict(quality_pair_plan_path, "PW04 quality pair plan")
    if quality_pair_plan.get("family_id") != family_id:
        raise ValueError(
            f"PW04 quality pair plan family_id mismatch: expected={family_id}, actual={quality_pair_plan.get('family_id')}"
        )
    shard_rows = cast(List[Mapping[str, Any]], quality_pair_plan.get("shards", []))
    shard_payload = next(
        (row for row in shard_rows if row.get("quality_shard_index") == quality_shard_index),
        None,
    )
    if not isinstance(shard_payload, Mapping):
        raise ValueError(f"PW04 quality pair plan missing shard index: {quality_shard_index}")

    clean_pair_id_set = {
        pair_id for pair_id in cast(List[Any], shard_payload.get("clean_pair_ids", [])) if isinstance(pair_id, str) and pair_id
    }
    attack_pair_id_set = {
        pair_id for pair_id in cast(List[Any], shard_payload.get("attack_pair_ids", [])) if isinstance(pair_id, str) and pair_id
    }
    clean_pairs = [
        dict(cast(Mapping[str, Any], pair_row))
        for pair_row in cast(List[Any], quality_pair_plan.get("clean_pairs", []))
        if isinstance(pair_row, Mapping) and pair_row.get("event_id") in clean_pair_id_set
    ]
    attack_pairs = [
        dict(cast(Mapping[str, Any], pair_row))
        for pair_row in cast(List[Any], quality_pair_plan.get("attack_pairs", []))
        if isinstance(pair_row, Mapping) and pair_row.get("attack_event_id") in attack_pair_id_set
    ]

    output_path = quality_pair_plan_path.parent / "shards" / f"quality_shard_{quality_shard_index:04d}.json"
    clean_phase_profile_path = output_path.with_name(f"quality_shard_{quality_shard_index:04d}.clean_phase_profile.json")
    attack_phase_profile_path = output_path.with_name(f"quality_shard_{quality_shard_index:04d}.attack_phase_profile.json")
    clean_quality_summary = _build_clean_quality_summary(
        clean_pairs,
        phase_profile_output_path=clean_phase_profile_path,
        phase_profile_label=f"{family_id}:quality_shard_{quality_shard_index:04d}:clean",
    )
    attack_quality_summary = _build_attack_quality_summary(
        attack_pairs,
        phase_profile_output_path=attack_phase_profile_path,
        phase_profile_label=f"{family_id}:quality_shard_{quality_shard_index:04d}:attack",
    )
    ensure_directory(output_path.parent)
    payload = {
        "artifact_type": "paper_workflow_pw04_quality_shard",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "quality_pair_plan_path": normalize_path_value(quality_pair_plan_path),
        "quality_shard_index": quality_shard_index,
        "clean_pair_count": len(clean_pairs),
        "attack_pair_count": len(attack_pairs),
        "clean_quality_phase_profile_path": normalize_path_value(clean_phase_profile_path),
        "attack_quality_phase_profile_path": normalize_path_value(attack_phase_profile_path),
        "clean_quality_summary": clean_quality_summary,
        "attack_quality_summary": attack_quality_summary,
    }
    write_json_atomic_compact(output_path, payload)
    return {
        "path": normalize_path_value(output_path),
        "payload": payload,
    }