"""
File purpose: Build paper workflow family manifest and source shard plan.
Module type: General module
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Sequence, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    normalize_path_value,
    utc_now_iso,
    write_json_atomic,
    write_yaml_mapping,
)

from paper_workflow.scripts.pw_common import (
    ACTIVE_SAMPLE_ROLE,
    DEFAULT_CONFIG_RELATIVE_PATH,
    DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH,
    RESERVED_SAMPLE_ROLES,
    SOURCE_TRUTH_STAGE,
    build_family_root,
    build_method_identity_snapshot,
    build_positive_source_event_grid,
    build_source_shard_plan,
    ensure_family_layout,
    load_default_config_snapshot,
    load_prompt_lines,
    parse_seed_list,
    resolve_family_layout_paths,
    write_jsonl,
)


def _load_pw_base_cfg() -> Dict[str, Any]:
    """
    Load pw_base config mapping.

    Args:
        None.

    Returns:
        Parsed pw_base config mapping.
    """
    pw_base_path = (REPO_ROOT / DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH).resolve()
    from scripts.notebook_runtime_common import load_yaml_mapping

    return load_yaml_mapping(pw_base_path)


def _resolve_source_alignment_reference_files(pw_base_cfg: Dict[str, Any]) -> list[str]:
    """
    Resolve source alignment file list for method identity snapshot.

    Args:
        pw_base_cfg: Parsed pw_base config mapping.

    Returns:
        Source alignment file list.
    """
    candidates = pw_base_cfg.get("source_alignment_reference_files")
    if isinstance(candidates, list):
        output: list[str] = []
        for item in cast(list[object], candidates):
            if isinstance(item, str) and item:
                output.append(item)
            else:
                output = []
                break
        if output:
            return output
    return [
        "scripts/01_Paper_Full_Cuda_Parallel.py",
        "scripts/01_run_paper_full_cuda_parallel.py",
        "scripts/01_run_paper_full_cuda_parallel_worker.py",
        "scripts/01_run_paper_full_cuda.py",
        "scripts/notebook_runtime_common.py",
        "notebook/00_main/01_Paper_Full_Cuda_Parallel.ipynb",
        "configs/default.yaml",
    ]


def run_pw00_build_family_manifest(
    *,
    drive_project_root: Path,
    family_id: str,
    prompt_file: str,
    seed_list: Sequence[int] | str,
    source_shard_count: int,
) -> Dict[str, Any]:
    """
    Build PW00 family manifest outputs.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        prompt_file: Prompt file path.
        seed_list: Seed list or seed-list text.
        source_shard_count: Source shard count.

    Returns:
        PW00 summary payload.
    """
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not prompt_file.strip():
        raise TypeError("prompt_file must be non-empty str")
    if source_shard_count <= 0:
        raise TypeError("source_shard_count must be positive int")

    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    layout = ensure_family_layout(family_root)
    _ = resolve_family_layout_paths(family_root)

    pw_base_cfg = _load_pw_base_cfg()
    normalized_seeds = parse_seed_list(seed_list)
    prompt_path, prompt_lines = load_prompt_lines(prompt_file)
    prompt_file_normalized = normalize_path_value(prompt_path)

    event_grid = build_positive_source_event_grid(
        family_id=family_id,
        prompt_lines=prompt_lines,
        seeds=normalized_seeds,
        prompt_file=prompt_file_normalized,
    )
    source_shard_plan = build_source_shard_plan(
        family_id=family_id,
        source_shard_count=source_shard_count,
        events=event_grid,
    )

    default_cfg_path = (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).resolve()
    default_cfg_obj = load_default_config_snapshot(REPO_ROOT)
    source_alignment_reference_files = _resolve_source_alignment_reference_files(pw_base_cfg)
    method_identity_snapshot = build_method_identity_snapshot(
        default_cfg_obj=default_cfg_obj,
        default_cfg_path=default_cfg_path,
        source_alignment_reference_files=source_alignment_reference_files,
    )

    config_snapshot = copy.deepcopy(default_cfg_obj)
    family_manifest: Dict[str, Any] = {
        "artifact_type": "paper_eval_family_manifest",
        "schema_version": "pw_stage_01_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "drive_project_root": normalize_path_value(normalized_drive_root),
        "family_root_relative": "paper_workflow/families",
        "source_truth_stage": SOURCE_TRUTH_STAGE,
        "stage_boundary": {
            "implemented": ["PW00", "PW01"],
            "excluded": [
                "PW02_Source_Merge_And_Global_Thresholds",
                "clean_negative_source_shards",
                "attack_event_shards",
                "release_signoff",
            ],
        },
        "sample_roles": {
            "active": [ACTIVE_SAMPLE_ROLE],
            "reserved": list(RESERVED_SAMPLE_ROLES),
        },
        "event_identity": {
            "unit_definition": "sample_event = (prompt_index, prompt_text, seed, sample_role)",
            "event_id_fields": [
                "family_id",
                "sample_role",
                "prompt_index",
                "prompt_sha256",
                "seed",
            ],
            "event_id_digest": "sha256",
        },
        "source_parameters": {
            "prompt_file": prompt_file_normalized,
            "prompt_count": len(prompt_lines),
            "seed_list": list(normalized_seeds),
            "seed_count": len(normalized_seeds),
            "source_shard_count": source_shard_count,
        },
        "counts": {
            "positive_source_event_count": len(event_grid),
            "total_event_count": len(event_grid),
        },
        "paths": {
            "paper_eval_family_manifest": normalize_path_value(layout["family_manifest_path"]),
            "source_event_grid": normalize_path_value(layout["source_event_grid_path"]),
            "source_shard_plan": normalize_path_value(layout["source_shard_plan_path"]),
            "prompt_snapshot": normalize_path_value(layout["prompt_snapshot_path"]),
            "method_identity_snapshot": normalize_path_value(layout["method_identity_snapshot_path"]),
            "config_snapshot": normalize_path_value(layout["config_snapshot_path"]),
        },
        "default_config_path": normalize_path_value(default_cfg_path),
        "pw_base_config_path": normalize_path_value((REPO_ROOT / DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH).resolve()),
    }

    layout["prompt_snapshot_path"].write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    write_json_atomic(layout["method_identity_snapshot_path"], method_identity_snapshot)
    write_yaml_mapping(layout["config_snapshot_path"], config_snapshot)
    write_jsonl(layout["source_event_grid_path"], event_grid)
    write_json_atomic(layout["source_shard_plan_path"], source_shard_plan)
    write_json_atomic(layout["family_manifest_path"], family_manifest)

    summary_path = layout["runtime_state_root"] / "pw00_summary.json"
    summary: Dict[str, Any] = {
        "status": "ok",
        "stage_name": "PW00_Paper_Eval_Family_Manifest",
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "summary_path": normalize_path_value(summary_path),
        "paper_eval_family_manifest_path": normalize_path_value(layout["family_manifest_path"]),
        "source_event_grid_path": normalize_path_value(layout["source_event_grid_path"]),
        "source_shard_plan_path": normalize_path_value(layout["source_shard_plan_path"]),
        "event_count": len(event_grid),
        "source_shard_count": source_shard_count,
    }
    write_json_atomic(summary_path, summary)
    return summary
