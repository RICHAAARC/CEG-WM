"""
File purpose: CLI entrypoint for PW03 attack event shards.
Module type: General module
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_workflow.scripts.pw03_run_attack_event_shard import run_pw03_attack_event_shard


def main() -> int:
    """
    Execute PW03 attack shard entrypoint.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run one PW03 attacked-positive shard from the finalized positive source pool. "
            "This wrapper exposes the top-level shard runner only."
        )
    )
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    parser.add_argument("--attack-shard-index", required=True, type=int, help="Zero-based attack shard index.")
    parser.add_argument("--attack-shard-count", required=True, type=int, help="Total attack shard count.")
    parser.add_argument(
        "--attack-local-worker-count",
        default=1,
        type=int,
        help="Shard-local PW03 worker count. Allowed values: 1, 2, 3, or 4.",
    )
    parser.add_argument(
        "--attack-family-allowlist",
        default=None,
        help="Optional JSON list or comma-separated attack-family allowlist.",
    )
    parser.add_argument(
        "--bound-config-path",
        required=True,
        help="Notebook-bound runtime config snapshot path.",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Clear completed shard root before rerun.")
    args = parser.parse_args()

    summary = run_pw03_attack_event_shard(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        attack_shard_index=int(args.attack_shard_index),
        attack_shard_count=int(args.attack_shard_count),
        attack_local_worker_count=int(args.attack_local_worker_count),
        attack_family_allowlist=args.attack_family_allowlist,
        bound_config_path=Path(str(args.bound_config_path)),
        force_rerun=bool(args.force_rerun),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())