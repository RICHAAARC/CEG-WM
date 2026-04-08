"""
File purpose: CLI entrypoint for PW04 attack merge and metrics.
Module type: General module
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_workflow.scripts.pw04_merge_attack_event_shards import run_pw04_merge_attack_event_shards


def main() -> int:
    """
    Execute PW04 merge and metrics entrypoint.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge completed PW03 attack shards and materialize PW04 attack metrics, "
            "with optional tail estimation exports."
        )
    )
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    parser.add_argument("--force-rerun", action="store_true", help="Clear completed PW04 outputs before rerun.")
    parser.add_argument(
        "--enable-tail-estimation",
        action="store_true",
        help="Enable optional tail estimation exports.",
    )
    args = parser.parse_args()

    summary = run_pw04_merge_attack_event_shards(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        force_rerun=bool(args.force_rerun),
        enable_tail_estimation=bool(args.enable_tail_estimation),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())