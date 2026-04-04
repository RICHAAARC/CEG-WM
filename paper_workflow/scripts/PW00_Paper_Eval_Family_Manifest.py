"""
File purpose: CLI entrypoint for PW00 family manifest generation.
Module type: General module
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest


def main() -> int:
    """
    Execute PW00 manifest generation entrypoint.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the PW00 family manifest, source shard plan, and frozen split plan for formal roles "
            "plus the optional planner_conditioned_control_negative diagnostic cohort."
        )
    )
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    parser.add_argument("--prompt-file", required=True, help="Prompt file path.")
    parser.add_argument(
        "--seed-list",
        required=True,
        help="Seed list JSON (for example: [0,1,2]) or comma-separated text (for example: 0,1,2).",
    )
    parser.add_argument("--source-shard-count", required=True, type=int, help="Source shard count.")
    args = parser.parse_args()

    summary = run_pw00_build_family_manifest(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        prompt_file=str(args.prompt_file),
        seed_list=args.seed_list,
        source_shard_count=int(args.source_shard_count),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
