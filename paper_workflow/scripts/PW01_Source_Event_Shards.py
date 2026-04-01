"""
File purpose: CLI entrypoint for PW01 positive source event shards.
Module type: General module
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_workflow.scripts.pw01_run_source_event_shard import run_pw01_source_event_shard


def main() -> int:
    """
    Execute PW01 shard runner entrypoint.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(description="Run one PW01 positive_source shard.")
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    parser.add_argument("--shard-index", required=True, type=int, help="Zero-based shard index.")
    parser.add_argument("--shard-count", required=True, type=int, help="Total shard count.")
    parser.add_argument("--force-rerun", action="store_true", help="Clear completed shard root before rerun.")
    args = parser.parse_args()

    summary = run_pw01_source_event_shard(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
        force_rerun=bool(args.force_rerun),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
