"""
File purpose: CLI entrypoint for PW02 source merge and global thresholds.
Module type: General module
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_workflow.scripts.pw02_merge_source_event_shards import run_pw02_merge_source_event_shards


def main() -> int:
    """
    Execute PW02 merge and threshold workflow.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run PW02 source merge and global thresholds workflow. "
            "The formal mainline requires positive_source and clean_negative only; "
            "planner_conditioned_control_negative is an optional diagnostic cohort that may be absent, "
            "but if partially provided PW02 will fail fast."
        )
    )
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    args = parser.parse_args()

    summary = run_pw02_merge_source_event_shards(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())