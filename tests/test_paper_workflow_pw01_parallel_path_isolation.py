"""
File purpose: Validate PW01 shard-local parallel path isolation across shards.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, cast

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
import paper_workflow.scripts.pw01_run_source_event_shard as pw01_module
from paper_workflow.scripts.pw_common import read_jsonl
from scripts.notebook_runtime_common import ensure_directory


def _build_pw00_family(tmp_path: Path, family_id: str) -> Dict[str, Any]:
    """
    Build a minimal PW00 fixture family for PW01 path-isolation tests.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / "pw01_parallel_prompts.txt"
    prompt_file.write_text("prompt one\nprompt two\n", encoding="utf-8")
    return run_pw00_build_family_manifest(
        drive_project_root=tmp_path / "drive",
        family_id=family_id,
        prompt_file=str(prompt_file),
        seed_list=[3, 9],
        source_shard_count=2,
    )


def _load_shard_assigned_events(summary: Dict[str, Any], shard_index: int) -> List[Dict[str, Any]]:
    """
    Load the ordered assigned events for one shard from the PW00 fixture outputs.

    Args:
        summary: PW00 summary payload.
        shard_index: Shard index.

    Returns:
        Ordered shard-assigned events.
    """
    shard_plan = json.loads(Path(str(summary["source_shard_plan_path"])).read_text(encoding="utf-8"))
    shard_assignment = pw01_module.resolve_positive_shard_assignment(
        shard_plan,
        shard_index=shard_index,
        shard_count=2,
    )
    event_lookup = {
        row["event_id"]: row
        for row in read_jsonl(Path(str(summary["source_event_grid_path"])))
    }
    return [
        cast(Dict[str, Any], event_lookup[event_id])
        for event_id in cast(List[str], shard_assignment["assigned_event_ids"])
    ]


def test_pw01_parallel_worker_paths_are_isolated_between_shards(tmp_path: Path) -> None:
    """
    Keep worker plans, results, and logs isolated across concurrent shard roots.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_parallel_isolation")
    family_manifest = json.loads(
        Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8")
    )
    default_config_path = pw01_module._resolve_default_config_path(family_manifest)
    family_root = Path(str(summary["family_root"]))

    shard_00_root = ensure_directory(pw01_module._build_shard_root(family_root, 0))
    shard_01_root = ensure_directory(pw01_module._build_shard_root(family_root, 1))

    shard_00_plans = pw01_module._prepare_local_worker_plans(
        drive_project_root=tmp_path / "drive",
        family_id="family_parallel_isolation",
        shard_index=0,
        shard_count=2,
        stage_01_worker_count=2,
        shard_root=shard_00_root,
        default_config_path=default_config_path,
        assigned_events=_load_shard_assigned_events(summary, 0),
    )
    shard_01_plans = pw01_module._prepare_local_worker_plans(
        drive_project_root=tmp_path / "drive",
        family_id="family_parallel_isolation",
        shard_index=1,
        shard_count=2,
        stage_01_worker_count=2,
        shard_root=shard_01_root,
        default_config_path=default_config_path,
        assigned_events=_load_shard_assigned_events(summary, 1),
    )

    observed_paths = set()
    for shard_root, worker_plans in [(shard_00_root, shard_00_plans), (shard_01_root, shard_01_plans)]:
        for worker_plan in worker_plans:
            for key_name in [
                "worker_root",
                "worker_plan_path",
                "worker_result_path",
                "stdout_log_path",
                "stderr_log_path",
            ]:
                path_obj = Path(str(worker_plan[key_name])).resolve()
                path_obj.relative_to(shard_root.resolve())
                normalized_path = path_obj.as_posix()
                assert normalized_path not in observed_paths
                observed_paths.add(normalized_path)

    assert shard_00_root != shard_01_root
    assert {plan["local_worker_index"] for plan in shard_00_plans} == {0, 1}
    assert {plan["local_worker_index"] for plan in shard_01_plans} == {0, 1}