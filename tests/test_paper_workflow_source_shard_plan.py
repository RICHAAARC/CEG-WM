"""
File purpose: Unit tests for PW00 source shard plan coverage and validation.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw01_run_source_event_shard import resolve_positive_shard_assignment
from paper_workflow.scripts.pw_common import read_jsonl


def _build_pw00_fixture(tmp_path: Path) -> Dict[str, Any]:
    """
    Build a PW00 fixture family and return summary.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / "fixture_prompts.txt"
    prompt_file.write_text("prompt 0\nprompt 1\nprompt 2\n", encoding="utf-8")
    return run_pw00_build_family_manifest(
        drive_project_root=tmp_path / "drive",
        family_id="family_plan_fixture",
        prompt_file=str(prompt_file),
        seed_list=[1, 2],
        source_shard_count=4,
    )


def test_source_shard_plan_covers_each_event_exactly_once(tmp_path: Path) -> None:
    """
    Verify shard plan coverage has no duplicates and no omissions.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    summary = _build_pw00_fixture(tmp_path)
    event_rows = read_jsonl(Path(str(summary["source_event_grid_path"])))
    shard_plan = json.loads(Path(str(summary["source_shard_plan_path"])).read_text(encoding="utf-8"))

    expected_event_ids = {row["event_id"] for row in event_rows}
    positive_plan = cast(Dict[str, Any], shard_plan["sample_role_plans"]["positive_source"])
    covered_event_ids: List[str] = []
    for shard_row in cast(List[Dict[str, Any]], positive_plan["shards"]):
        covered_event_ids.extend(cast(List[str], shard_row["assigned_event_ids"]))

    assert len(covered_event_ids) == len(expected_event_ids)
    assert len(set(covered_event_ids)) == len(expected_event_ids)
    assert set(covered_event_ids) == expected_event_ids


def test_source_shard_plan_rejects_shard_count_mismatch(tmp_path: Path) -> None:
    """
    Verify shard assignment resolver fails on mismatched shard count.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    summary = _build_pw00_fixture(tmp_path)
    shard_plan = json.loads(Path(str(summary["source_shard_plan_path"])).read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="shard_count mismatch"):
        resolve_positive_shard_assignment(
            shard_plan,
            shard_index=0,
            shard_count=5,
        )
