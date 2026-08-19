"""Thin soft-route confirmation bootstrap using the fixed overlay boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


_BOOTSTRAP_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
_package_root_text = str(_BOOTSTRAP_PACKAGE_ROOT)
sys.path[:] = [_package_root_text, *(entry for entry in sys.path if entry != _package_root_text)]

from scripts.experiment_execution.semantic_texture_soft_route_candidate_selection_bootstrap import (
    bootstrap_soft_route_mechanism_candidate_selection,
)


def bootstrap_soft_route_mechanism_untouched_confirmation(**kwargs: object) -> tuple[int, dict[str, object]]:
    return bootstrap_soft_route_mechanism_candidate_selection(
        **kwargs,
        entrypoint_path="scripts/experiment_execution/semantic_texture_soft_route_untouched_confirmation_entrypoint.py",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--execution-root", required=True)
    parser.add_argument("--entrypoint-args", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    code, receipt = bootstrap_soft_route_mechanism_untouched_confirmation(
        repository_root=arguments.repository_root,
        checkpoint=arguments.checkpoint,
        execution_root=arguments.execution_root,
        entrypoint_args=arguments.entrypoint_args,
    )
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
