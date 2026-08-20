"""Neutral inventory check for the retained thin Colab notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_colab_notebook_inventory_authority_and_outputs_are_exact() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook_root = root / "notebooks/colab"
    expected = {
        "experiment_execution.ipynb",
        "hf_only_detector_directional_validation.ipynb",
        "hf_transmission_diagnostic.ipynb",
        "lf_transmission_diagnostic.ipynb",
        "lf_whitened_directional_validation.ipynb",
        "lf_whitened_score_screening.ipynb",
        "qk_synchronization_write_diagnostic.ipynb",
        "semantic_texture_soft_detector_asset_preparation.ipynb",
        "semantic_texture_operational_preflight.ipynb",
        "semantic_texture_soft_route_candidate_selection.ipynb",
        "semantic_texture_soft_route_untouched_confirmation.ipynb",
        "contrastive_lf_branch_attribution_candidate_selection.ipynb",
    }
    assert {path.name for path in notebook_root.glob("*.ipynb")} == expected
    active_notebooks = {
        "contrastive_lf_branch_attribution_candidate_selection.ipynb",
    }
    retained_outside_current_delta = {
        "semantic_texture_operational_preflight.ipynb",
        "semantic_texture_soft_detector_asset_preparation.ipynb",
        "semantic_texture_soft_route_candidate_selection.ipynb",
    }
    paused_notebooks = expected - active_notebooks - retained_outside_current_delta
    banner = "**PAUSED / NOT AUTHORIZED:** Do not use **Run all**."
    current_run_all_phrase = "contrastive-lf-branch-attribution-"
    forbidden_authority_phrases = (
        "only currently authorized",
        "current authorized Colab entrypoint",
        "currently authorized entrypoint is",
    )
    sources: dict[str, str] = {}
    for path in sorted(notebook_root.glob("*.ipynb")):
        notebook = json.loads(path.read_text("utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        sources[path.name] = source
        if path.name in paused_notebooks:
            assert notebook["cells"][0]["cell_type"] == "markdown"
            assert "".join(notebook["cells"][0]["source"]).startswith(
                f"{banner}"
            )
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
        assert all(
            cell.get("execution_count") is None
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
    assert active_notebooks <= {
        name for name, source in sources.items() if current_run_all_phrase in source
    }
    asset_source = sources["semantic_texture_soft_detector_asset_preparation.ipynb"]
    assert "semantic_texture_soft_detector_asset_preparation_entrypoint.py" in asset_source
    assert "semantic_texture_soft_detector_assets" in asset_source
    assert "--entrypoint-path" in asset_source
    assert "latest-bundle" not in asset_source
    assert "retry" not in asset_source
    selection_source = sources["contrastive_lf_branch_attribution_candidate_selection.ipynb"]
    assert selection_source.index("drive.mount") < selection_source.index("userdata.get")
    assert "CEG_WM_CONTRASTIVE_LF_REVISION" not in selection_source
    assert "https://github.com/RICHAAARC/CEG-WM.git" in selection_source
    assert "'--branch', 'main'" in selection_source
    assert "build_contrastive_lf_branch_attribution_package.py" in selection_source
    assert "colab_handoffs" not in selection_source
    assert "capture_output=True" not in selection_source.split("command =", 1)[1]
    assert "--runs-root" in selection_source
    assert "--session-id" in selection_source
    assert "contrastive_lf_branch_attribution_bootstrap.py" in selection_source
    assert "check=False" in selection_source
    assert "confirmation" not in selection_source.lower()
    assert "stage_b" not in selection_source.lower()
    assert "latest-bundle" not in selection_source
    assert "latest_run" not in selection_source
    confirmation_source = sources[
        "semantic_texture_soft_route_untouched_confirmation.ipynb"
    ]
    assert "CEG_WM_SOFT_ROUTE_MECHANISM_VALIDATION_SELECTION_ARTIFACT_SHA256" in confirmation_source
    assert "--selection-artifact-sha256" in confirmation_source
    assert "latest" in confirmation_source  # only the explicit prohibition in the paused banner
    for source in sources.values():
        for phrase in forbidden_authority_phrases:
            assert phrase not in source
