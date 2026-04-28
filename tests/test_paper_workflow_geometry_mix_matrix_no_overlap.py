"""
File purpose: Verify the geometry_mix matrix has no exact overlap with legacy paper workflow matrices.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set, Tuple

from paper_workflow.scripts.pw_common import build_attack_condition_catalog
from scripts.notebook_runtime_common import REPO_ROOT, load_yaml_mapping


PILOT_MATRIX_PATH = REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_pilot.yaml"
RESCUE_MATRIX_PATH = REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_rescue_v1.yaml"
INTERVAL_DISCOVERY_V2_MATRIX_PATH = (
    REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_interval_discovery_v2.yaml"
)
GEOMETRY_MIX_MATRIX_PATH = REPO_ROOT / "paper_workflow" / "configs" / "pw_matrix_geometry_mix.yaml"


def _build_exact_item_set(matrix_path: Path) -> Set[Tuple[str, str]]:
    """
    Build the exact comparable item set for one matrix.

    Args:
        matrix_path: Matrix config path.

    Returns:
        Set of canonical (attack_family, params_json) tuples.
    """
    matrix_cfg = load_yaml_mapping(matrix_path)
    attack_condition_catalog = build_attack_condition_catalog(matrix_cfg=matrix_cfg)
    return {
        (
            str(row["attack_family"]),
            json.dumps(row["attack_params"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        for row in attack_condition_catalog
    }


def test_geometry_mix_matrix_has_no_exact_overlap_with_legacy_matrices() -> None:
    """
    Verify geometry_mix introduces 18 exact items with zero exact overlap against the legacy families.

    Args:
        None.

    Returns:
        None.
    """
    geometry_mix_items = _build_exact_item_set(GEOMETRY_MIX_MATRIX_PATH)
    legacy_item_sets: Dict[str, Set[Tuple[str, str]]] = {
        "pilot_v1": _build_exact_item_set(PILOT_MATRIX_PATH),
        "geometry_rescue_v1": _build_exact_item_set(RESCUE_MATRIX_PATH),
        "geometry_interval_discovery_v2": _build_exact_item_set(INTERVAL_DISCOVERY_V2_MATRIX_PATH),
    }

    assert len(geometry_mix_items) == 18
    for legacy_name, legacy_items in legacy_item_sets.items():
        intersection = geometry_mix_items & legacy_items
        assert not intersection, f"geometry_mix exact overlap with {legacy_name}: {sorted(intersection)}"