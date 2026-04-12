"""
File purpose: Prepare reusable PW04 clean and attack quality pair plans.
Module type: Semi-general module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast

from scripts.notebook_runtime_common import ensure_directory, normalize_path_value, utc_now_iso, validate_path_within_base, write_json_atomic


SCHEMA_VERSION = "pw_stage_04_v1"
QUALITY_PAIR_PLAN_FILE_NAME = "quality_pair_plan.json"


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


def _resolve_clean_quality_pair_manifest_path(
    *,
    family_root: Path,
    pw02_summary: Mapping[str, Any],
) -> Path:
    """
    功能：解析 PW02 clean pair manifest 路径。

    Resolve the PW02 clean pair manifest path.

    Args:
        family_root: Family root path.
        pw02_summary: PW02 summary payload.

    Returns:
        Resolved clean pair manifest path.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")

    path_value = pw02_summary.get("clean_quality_pair_manifest_path")
    if isinstance(path_value, str) and path_value.strip():
        resolved_path = Path(path_value).expanduser().resolve()
    else:
        resolved_path = (family_root / "exports" / "pw02" / "quality" / "clean_quality_pair_manifest.json").resolve()
    validate_path_within_base(family_root, resolved_path, "PW02 clean quality pair manifest")
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(
            f"PW02 clean quality pair manifest not found: {normalize_path_value(resolved_path)}"
        )
    return resolved_path


def _build_attack_quality_pair_rows(
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    功能：把 PW04 attacked positive 行转换为 attack quality pair rows。

    Convert attacked-positive rows into attack quality pair rows.

    Args:
        attack_event_rows: Materialized attacked-positive rows.

    Returns:
        Attack quality pair rows.
    """
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    pair_rows: List[Dict[str, Any]] = []
    for attack_event_row in attack_event_rows:
        if not isinstance(attack_event_row, Mapping):
            raise TypeError("attack_event_rows items must be mappings")
        pair_rows.append(
            {
                "pair_namespace": "attack",
                "attack_event_id": attack_event_row.get("attack_event_id"),
                "parent_event_id": attack_event_row.get("parent_event_id"),
                "attack_family": attack_event_row.get("attack_family"),
                "attack_condition_key": attack_event_row.get("attack_condition_key"),
                "attack_config_name": attack_event_row.get("attack_config_name"),
                "reference_image_path": attack_event_row.get("parent_source_image_path"),
                "candidate_image_path": attack_event_row.get("attacked_image_path"),
                "prompt_text": attack_event_row.get("parent_prompt_text"),
            }
        )
    return pair_rows


def _derive_quality_shard_count(
    *,
    attack_event_rows: Sequence[Mapping[str, Any]],
    planned_shard_count: int | None,
) -> int:
    """
    功能：解析 quality shard 数量。

    Derive the quality shard count for the PW04 quality pipeline.

    Args:
        attack_event_rows: Materialized attacked-positive rows.
        planned_shard_count: Optional explicit shard count.

    Returns:
        Stable shard count.
    """
    if planned_shard_count is not None:
        if not isinstance(planned_shard_count, int) or isinstance(planned_shard_count, bool) or planned_shard_count <= 0:
            raise ValueError("planned_shard_count must be positive int when provided")
        return planned_shard_count

    shard_indices = {
        int(attack_event_row["attack_shard_index"])
        for attack_event_row in attack_event_rows
        if isinstance(attack_event_row, Mapping)
        and isinstance(attack_event_row.get("attack_shard_index"), int)
        and not isinstance(attack_event_row.get("attack_shard_index"), bool)
    }
    return max(len(shard_indices), 1)


def run_pw04_prepare_quality_pairs(
    *,
    family_id: str,
    family_root: Path,
    pw02_summary: Mapping[str, Any],
    attack_event_rows: Sequence[Mapping[str, Any]],
    quality_root: Path,
    planned_shard_count: int | None = None,
) -> Dict[str, Any]:
    """
    功能：构建 PW04 clean 与 attack quality 的统一 pair plan。

    Build the unified PW04 clean and attack quality pair plan.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        pw02_summary: PW02 summary payload.
        attack_event_rows: Materialized attacked-positive rows.
        quality_root: PW04 quality working directory.
        planned_shard_count: Optional explicit shard count.

    Returns:
        Quality pair plan export summary.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw02_summary, Mapping):
        raise TypeError("pw02_summary must be Mapping")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")
    if not isinstance(quality_root, Path):
        raise TypeError("quality_root must be Path")

    ensure_directory(quality_root)
    validate_path_within_base(family_root, quality_root, "PW04 quality root")

    clean_pair_manifest_path = _resolve_clean_quality_pair_manifest_path(
        family_root=family_root,
        pw02_summary=pw02_summary,
    )
    clean_pair_manifest_payload = _load_required_json_dict(
        clean_pair_manifest_path,
        "PW02 clean quality pair manifest",
    )
    clean_pairs = cast(
        List[Dict[str, Any]],
        [dict(cast(Mapping[str, Any], row)) for row in clean_pair_manifest_payload.get("pair_rows", []) if isinstance(row, Mapping)],
    )
    for clean_pair in clean_pairs:
        clean_pair["pair_namespace"] = "clean"
    attack_pairs = _build_attack_quality_pair_rows(attack_event_rows)

    shard_count = _derive_quality_shard_count(
        attack_event_rows=attack_event_rows,
        planned_shard_count=planned_shard_count,
    )
    shard_assignments: List[Dict[str, Any]] = [
        {
            "quality_shard_index": shard_index,
            "clean_pair_ids": [],
            "attack_pair_ids": [],
            "clean_pair_count": 0,
            "attack_pair_count": 0,
            "total_pair_count": 0,
        }
        for shard_index in range(shard_count)
    ]
    for pair_index, clean_pair in enumerate(clean_pairs):
        shard_payload = shard_assignments[pair_index % shard_count]
        shard_payload["clean_pair_ids"].append(clean_pair.get("event_id"))
        shard_payload["clean_pair_count"] += 1
        shard_payload["total_pair_count"] += 1
    for pair_index, attack_pair in enumerate(attack_pairs):
        shard_payload = shard_assignments[pair_index % shard_count]
        shard_payload["attack_pair_ids"].append(attack_pair.get("attack_event_id"))
        shard_payload["attack_pair_count"] += 1
        shard_payload["total_pair_count"] += 1

    output_path = quality_root / QUALITY_PAIR_PLAN_FILE_NAME
    validate_path_within_base(family_root, output_path, "PW04 quality pair plan")
    payload = {
        "artifact_type": "paper_workflow_pw04_quality_pair_plan",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "status": "ok",
        "reason": None,
        "clean_quality_pair_manifest_path": normalize_path_value(clean_pair_manifest_path),
        "clean_expected_pair_count": len(clean_pairs),
        "attack_expected_pair_count": len(attack_pairs),
        "quality_shard_count": shard_count,
        "clean_pairs": clean_pairs,
        "attack_pairs": attack_pairs,
        "shards": shard_assignments,
    }
    write_json_atomic(output_path, payload)
    return {
        "path": normalize_path_value(output_path),
        "payload": payload,
    }