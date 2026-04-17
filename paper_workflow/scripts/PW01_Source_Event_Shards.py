"""
File purpose: CLI wrapper entrypoint for PW01 source event shards.
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
    parser = argparse.ArgumentParser(
        description=(
            "Run one PW01 source-role shard. "
            "Formal mainline roles are positive_source and clean_negative; "
            "planner_conditioned_control_negative is an optional diagnostic cohort."
        )
    )
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    parser.add_argument(
        "--sample-role",
        default="positive_source",
        help=(
            "Source sample role for this shard. Formal mainline values: positive_source, clean_negative. "
            "Optional advanced diagnostic value: planner_conditioned_control_negative."
        ),
    )
    parser.add_argument("--shard-index", required=True, type=int, help="Zero-based shard index.")
    parser.add_argument("--shard-count", required=True, type=int, help="Total shard count.")
    parser.add_argument(
        "--pw01-worker-count",
        default=1,
        type=int,
        help="Shard-local PW01 worker count. Only 1 or 2 is allowed.",
    )
    parser.add_argument(
        "--bound-config-path",
        required=True,
        help="Notebook-bound runtime config snapshot path.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun even if shard outputs already exist.",
    )
    args = parser.parse_args()

    summary = run_pw01_source_event_shard(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        sample_role=str(args.sample_role),
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
        pw01_worker_count=int(args.pw01_worker_count),
        bound_config_path=Path(args.bound_config_path),
        force_rerun=bool(args.force_rerun),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
