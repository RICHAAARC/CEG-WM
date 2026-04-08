"""
File purpose: CLI entrypoint for PW05 release and signoff.
Module type: General module
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_workflow.scripts.pw05_release_signoff import run_pw05_release_signoff


def main() -> int:
    """
    Execute PW05 release and signoff entrypoint.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the PW05 release package and signoff outputs for one finalized "
            "paper_workflow family."
        )
    )
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    parser.add_argument("--stage-run-id", default=None, help="Optional fixed stage run identifier.")
    parser.add_argument(
        "--notebook-name",
        default="PW05_Release_And_Signoff",
        help="Notebook display name recorded into the release package.",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Clear completed PW05 outputs before rerun.")
    args = parser.parse_args()

    summary = run_pw05_release_signoff(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        stage_run_id=(str(args.stage_run_id) if isinstance(args.stage_run_id, str) and args.stage_run_id.strip() else None),
        notebook_name=str(args.notebook_name),
        force_rerun=bool(args.force_rerun),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())