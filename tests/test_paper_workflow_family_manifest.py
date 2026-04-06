"""
File purpose: Unit tests for PW00 family manifest generation stability.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw_common import read_jsonl


def test_pw00_builds_stable_event_grid_and_shard_plan(tmp_path: Path) -> None:
    """
    Build PW00 twice and verify deterministic outputs.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    first_summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="family_stage01_demo",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
    )
    second_summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="family_stage01_demo",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
    )

    first_event_grid_path = Path(str(first_summary["source_event_grid_path"]))
    first_shard_plan_path = Path(str(first_summary["source_shard_plan_path"]))
    first_split_plan_path = Path(str(first_summary["source_split_plan_path"]))
    second_event_grid_path = Path(str(second_summary["source_event_grid_path"]))
    second_shard_plan_path = Path(str(second_summary["source_shard_plan_path"]))
    second_split_plan_path = Path(str(second_summary["source_split_plan_path"]))

    assert first_event_grid_path.read_text(encoding="utf-8") == second_event_grid_path.read_text(encoding="utf-8")
    assert json.loads(first_shard_plan_path.read_text(encoding="utf-8")) == json.loads(
        second_shard_plan_path.read_text(encoding="utf-8")
    )
    assert json.loads(first_split_plan_path.read_text(encoding="utf-8")) == json.loads(
        second_split_plan_path.read_text(encoding="utf-8")
    )

    event_rows = read_jsonl(first_event_grid_path)
    assert len(event_rows) == 12
    assert [row["event_index"] for row in event_rows] == list(range(12))
    assert [row["sample_role"] for row in event_rows] == [
        "positive_source",
        "positive_source",
        "positive_source",
        "positive_source",
        "clean_negative",
        "clean_negative",
        "clean_negative",
        "clean_negative",
        "planner_conditioned_control_negative",
        "planner_conditioned_control_negative",
        "planner_conditioned_control_negative",
        "planner_conditioned_control_negative",
    ]
    assert len({row["event_id"] for row in event_rows}) == len(event_rows)

    family_manifest_path = Path(str(first_summary["paper_eval_family_manifest_path"]))
    family_manifest = json.loads(family_manifest_path.read_text(encoding="utf-8"))
    assert family_manifest["sample_roles"]["active"] == [
        "positive_source",
        "clean_negative",
        "planner_conditioned_control_negative",
    ]
    assert family_manifest["sample_roles"]["reserved"] == ["attacked_positive"]
    assert family_manifest["source_parameters"]["seed_list"] == [0, 7]
    assert family_manifest["source_parameters"]["calibration_fraction"] == 0.5
    assert family_manifest["source_parameters"]["source_shard_count"] == 3
    assert family_manifest["attack_parameters"]["attack_shard_count"] == 3
    assert family_manifest["attack_plan"]["attack_shard_count"] == 3
    assert family_manifest["counts"]["positive_source_event_count"] == 4
    assert family_manifest["counts"]["clean_negative_event_count"] == 4
    assert family_manifest["counts"]["planner_conditioned_control_negative_event_count"] == 4
    assert family_manifest["counts"]["calibration_event_count"] == 4
    assert family_manifest["counts"]["evaluate_event_count"] == 4
    assert family_manifest["counts"]["control_calibration_event_count"] == 2
    assert family_manifest["counts"]["control_evaluate_event_count"] == 2
    assert first_summary["source_shard_count"] == 3
    assert first_summary["attack_shard_count"] == 3


def test_pw00_can_freeze_independent_attack_shard_count(tmp_path: Path) -> None:
    """
    Verify PW00 can freeze an attack shard count independent from source shards.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    drive_project_root = tmp_path / "drive_root"
    prompt_file = tmp_path / "paper_prompts.txt"
    prompt_file.write_text("prompt alpha\nprompt beta\n", encoding="utf-8")

    summary = run_pw00_build_family_manifest(
        drive_project_root=drive_project_root,
        family_id="family_stage01_attack_decoupled",
        prompt_file=str(prompt_file),
        seed_list=[0, 7],
        source_shard_count=3,
        attack_shard_count=5,
    )

    family_manifest_path = Path(str(summary["paper_eval_family_manifest_path"]))
    attack_shard_plan_path = Path(str(summary["attack_shard_plan_path"]))
    source_shard_plan_path = Path(str(summary["source_shard_plan_path"]))

    family_manifest = json.loads(family_manifest_path.read_text(encoding="utf-8"))
    attack_shard_plan = json.loads(attack_shard_plan_path.read_text(encoding="utf-8"))
    source_shard_plan = json.loads(source_shard_plan_path.read_text(encoding="utf-8"))

    assert family_manifest["source_parameters"]["source_shard_count"] == 3
    assert family_manifest["attack_parameters"]["attack_shard_count"] == 5
    assert family_manifest["attack_plan"]["attack_shard_count"] == 5
    assert attack_shard_plan["attack_shard_count"] == 5
    assert len(attack_shard_plan["shards"]) == 5
    assert source_shard_plan["source_shard_count"] == 3
    assert summary["source_shard_count"] == 3
    assert summary["attack_shard_count"] == 5
