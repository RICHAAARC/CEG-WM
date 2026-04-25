"""
File purpose: Validate PW01 shard-local parallel worker assignment and merge behavior.
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

import paper_workflow.scripts.pw01_run_source_event_shard as pw01_module
from paper_workflow.scripts import pw01_stage_runtime_helpers


def _make_event(
    *,
    event_id: str,
    event_index: int,
    source_prompt_index: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Build a minimal assigned-event payload for PW01 worker tests.

    Args:
        event_id: Event identifier.
        event_index: Global event index.
        source_prompt_index: Source prompt index.
        seed: Event seed.

    Returns:
        Minimal event payload.
    """
    return {
        "event_id": event_id,
        "event_index": event_index,
        "sample_role": "positive_source",
        "source_prompt_index": source_prompt_index,
        "prompt_text": f"prompt-{source_prompt_index}",
        "prompt_sha256": f"sha256-{source_prompt_index}",
        "seed": seed,
        "prompt_file": "prompts/paper_small.txt",
    }


def _make_event_manifest(event_id: str, event_index: int) -> Dict[str, Any]:
    """
    Build a minimal event-manifest payload for merge tests.

    Args:
        event_id: Event identifier.
        event_index: Global event index.

    Returns:
        Minimal event-manifest payload.
    """
    return {
        "event_id": event_id,
        "event_index": event_index,
        "event_manifest_path": f"/tmp/{event_id}.json",
    }


def test_pw01_local_worker_assignments_use_local_event_ordinal() -> None:
    """
    Split shard-assigned events by local ordinal rather than global event index.

    Args:
        None.

    Returns:
        None.
    """
    assignments = pw01_module._build_local_worker_assignments(
        assigned_events=[
            _make_event(event_id="evt_005", event_index=5, source_prompt_index=0, seed=3),
            _make_event(event_id="evt_006", event_index=6, source_prompt_index=0, seed=9),
            _make_event(event_id="evt_008", event_index=8, source_prompt_index=1, seed=3),
        ],
        sample_role="positive_source",
        pw01_worker_count=2,
    )

    assert assignments == [
        {
            "local_worker_index": 0,
            "local_event_ordinals": [0, 2],
            "assigned_event_ids": ["evt_005", "evt_008"],
            "assigned_event_indices": [5, 8],
            "assigned_events": [
                {
                    "event_id": "evt_005",
                    "event_index": 5,
                    "sample_role": "positive_source",
                    "source_prompt_index": 0,
                    "prompt_text": "prompt-0",
                    "prompt_sha256": "sha256-0",
                    "seed": 3,
                    "prompt_file": "prompts/paper_small.txt",
                    "local_event_ordinal": 0,
                },
                {
                    "event_id": "evt_008",
                    "event_index": 8,
                    "sample_role": "positive_source",
                    "source_prompt_index": 1,
                    "prompt_text": "prompt-1",
                    "prompt_sha256": "sha256-1",
                    "seed": 3,
                    "prompt_file": "prompts/paper_small.txt",
                    "local_event_ordinal": 2,
                },
            ],
        },
        {
            "local_worker_index": 1,
            "local_event_ordinals": [1],
            "assigned_event_ids": ["evt_006"],
            "assigned_event_indices": [6],
            "assigned_events": [
                {
                    "event_id": "evt_006",
                    "event_index": 6,
                    "sample_role": "positive_source",
                    "source_prompt_index": 0,
                    "prompt_text": "prompt-0",
                    "prompt_sha256": "sha256-0",
                    "seed": 9,
                    "prompt_file": "prompts/paper_small.txt",
                    "local_event_ordinal": 1,
                }
            ],
        },
    ]


def test_pw01_stage_command_keeps_cli_override_tokens() -> None:
    """
    Keep subprocess stage commands on CLI-style override tokens while exposing raw items separately.

    Args:
        None.

    Returns:
        None.
    """
    raw_items = pw01_stage_runtime_helpers._build_stage_override_items("detect")
    command = pw01_stage_runtime_helpers._build_stage_command(
        "detect",
        Path("/tmp/runtime_config.yaml"),
        Path("/tmp/run_root"),
    )

    assert raw_items == [
        "run_root_reuse_allowed=true",
        "run_root_reuse_reason=\"paper_workflow_pw01_detect\"",
    ]
    assert "--override" not in raw_items
    assert command[-4:] == [
        "--override",
        "run_root_reuse_allowed=true",
        "--override",
        "run_root_reuse_reason=\"paper_workflow_pw01_detect\"",
    ]


def test_pw01_merge_worker_results_restores_assigned_event_order(tmp_path: Path) -> None:
    """
    Merge worker results back into shard-assigned event order with full coverage.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    worker_root_00 = tmp_path / "worker_00"
    worker_root_01 = tmp_path / "worker_01"
    worker_result_00 = pw01_module._build_worker_result_payload(
        family_id="family_merge",
        sample_role="positive_source",
        shard_index=0,
        shard_count=2,
        pw01_worker_count=2,
        local_worker_index=0,
        worker_root=worker_root_00,
        worker_plan_path=worker_root_00 / "worker_plan.json",
        assigned_event_ids=["evt_a", "evt_c"],
        assigned_event_indices=[0, 2],
        events=[_make_event_manifest("evt_c", 2), _make_event_manifest("evt_a", 0)],
        status="completed",
    )
    worker_result_01 = pw01_module._build_worker_result_payload(
        family_id="family_merge",
        sample_role="positive_source",
        shard_index=0,
        shard_count=2,
        pw01_worker_count=2,
        local_worker_index=1,
        worker_root=worker_root_01,
        worker_plan_path=worker_root_01 / "worker_plan.json",
        assigned_event_ids=["evt_b"],
        assigned_event_indices=[1],
        events=[_make_event_manifest("evt_b", 1)],
        status="completed",
    )

    merged_events = pw01_module._merge_completed_worker_events(
        worker_results=[worker_result_01, worker_result_00],
        assigned_event_ids=["evt_a", "evt_b", "evt_c"],
    )

    assert [event["event_id"] for event in merged_events] == ["evt_a", "evt_b", "evt_c"]


def test_pw01_merge_rejects_overlapping_completed_events(tmp_path: Path) -> None:
    """
    Reject overlapping completed-event payloads across shard-local workers.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    worker_root_00 = tmp_path / "worker_00"
    worker_root_01 = tmp_path / "worker_01"
    worker_result_00 = pw01_module._build_worker_result_payload(
        family_id="family_overlap",
        sample_role="positive_source",
        shard_index=0,
        shard_count=2,
        pw01_worker_count=2,
        local_worker_index=0,
        worker_root=worker_root_00,
        worker_plan_path=worker_root_00 / "worker_plan.json",
        assigned_event_ids=["evt_a"],
        assigned_event_indices=[0],
        events=[_make_event_manifest("evt_a", 0)],
        status="completed",
    )
    worker_result_01 = pw01_module._build_worker_result_payload(
        family_id="family_overlap",
        sample_role="positive_source",
        shard_index=0,
        shard_count=2,
        pw01_worker_count=2,
        local_worker_index=1,
        worker_root=worker_root_01,
        worker_plan_path=worker_root_01 / "worker_plan.json",
        assigned_event_ids=["evt_a"],
        assigned_event_indices=[0],
        events=[_make_event_manifest("evt_a", 0)],
        status="completed",
    )

    with pytest.raises(ValueError, match="worker assigned events overlap"):
        pw01_module._merge_completed_worker_events(
            worker_results=[worker_result_00, worker_result_01],
            assigned_event_ids=["evt_a"],
        )


def test_pw01_local_worker_assignments_support_clean_negative_role() -> None:
    """
    Verify shard-local worker assignment preserves clean_negative sample_role.

    Args:
        None.

    Returns:
        None.
    """
    negative_event = _make_event(event_id="evt_neg", event_index=4, source_prompt_index=0, seed=5)
    negative_event["sample_role"] = "clean_negative"

    assignments = pw01_module._build_local_worker_assignments(
        assigned_events=[negative_event],
        sample_role="clean_negative",
        pw01_worker_count=1,
    )

    assert assignments[0]["assigned_events"][0]["sample_role"] == "clean_negative"