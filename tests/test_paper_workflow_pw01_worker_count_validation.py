"""
File purpose: Validate PW01 pw01_worker_count guard behavior.
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

import pytest

import paper_workflow.scripts.pw01_run_source_event_shard as pw01_module


@pytest.mark.parametrize("pw01_worker_count", [0, 3, -1])
def test_pw01_rejects_invalid_pw01_worker_count(
    tmp_path: Path,
    pw01_worker_count: int,
) -> None:
    """
    Reject invalid shard-local worker counts before reading family artifacts.

    Args:
        tmp_path: Pytest temporary directory.
        pw01_worker_count: Invalid worker-count candidate.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="pw01_worker_count must be 1 or 2"):
        pw01_module.run_pw01_source_event_shard(
            drive_project_root=tmp_path / "drive",
            family_id="family_worker_count_guard",
            shard_index=0,
            shard_count=1,
            pw01_worker_count=pw01_worker_count,
        )


def test_pw01_accepts_pw01_worker_count_one_and_two() -> None:
    """
    Accept the only two supported shard-local worker counts.

    Args:
        None.

    Returns:
        None.
    """
    pw01_module._validate_pw01_worker_count(1)
    pw01_module._validate_pw01_worker_count(2)