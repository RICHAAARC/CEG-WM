"""Thin soft-route confirmation bootstrap using the fixed overlay boundary."""

from __future__ import annotations

from pathlib import Path
import sys


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
