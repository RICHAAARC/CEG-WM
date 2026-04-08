"""
File purpose: Execute one shard-local PW01 worker plan.
Module type: General module
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_workflow.scripts.pw01_run_source_event_shard import run_pw01_source_event_shard_worker


def main() -> int:
    """
    Execute the PW01 shard-local worker entrypoint.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(description="Run one PW01 shard-local worker plan.")
    parser.add_argument("--drive-project-root", required=True, help="Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    parser.add_argument("--shard-index", required=True, type=int, help="Zero-based shard index.")
    parser.add_argument(
        "--pw01-worker-count",
        required=True,
        type=int,
        help="Shard-local PW01 worker count. Only 1 or 2 is allowed.",
    )
    parser.add_argument("--local-worker-index", required=True, type=int, help="Zero-based local worker index.")
    parser.add_argument("--worker-plan-path", required=True, help="Worker plan JSON path.")
    args = parser.parse_args()

    summary = run_pw01_source_event_shard_worker(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        shard_index=int(args.shard_index),
        pw01_worker_count=int(args.pw01_worker_count),
        local_worker_index=int(args.local_worker_index),
        worker_plan_path=Path(args.worker_plan_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())