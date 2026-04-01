"""
File purpose: Validate PW01 stage_01_worker_count guard behavior.
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

import pytest

import paper_workflow.scripts.pw01_run_source_event_shard as pw01_module


@pytest.mark.parametrize("stage_01_worker_count", [0, 3, -1])
def test_pw01_rejects_invalid_stage_01_worker_count(
    tmp_path: Path,
    stage_01_worker_count: int,
) -> None:
    """
    Reject invalid shard-local worker counts before reading family artifacts.

    Args:
        tmp_path: Pytest temporary directory.
        stage_01_worker_count: Invalid worker-count candidate.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="stage_01_worker_count must be 1 or 2"):
        pw01_module.run_pw01_source_event_shard(
            drive_project_root=tmp_path / "drive",
            family_id="family_worker_count_guard",
            shard_index=0,
            shard_count=1,
            stage_01_worker_count=stage_01_worker_count,
        )


def test_pw01_accepts_stage_01_worker_count_one_and_two() -> None:
    """
    Accept the only two supported shard-local worker counts.

    Args:
        None.

    Returns:
        None.
    """
    pw01_module._validate_stage_01_worker_count(1)
    pw01_module._validate_stage_01_worker_count(2)