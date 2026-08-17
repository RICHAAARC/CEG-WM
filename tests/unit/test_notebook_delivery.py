"""Neutral inventory check for the retained thin Colab notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_colab_notebook_inventory_is_exact_and_output_free() -> None:
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
        "semantic_texture_operational_preflight.ipynb",
    }
    assert {path.name for path in notebook_root.glob("*.ipynb")} == expected
    for path in sorted(notebook_root.glob("*.ipynb")):
        notebook = json.loads(path.read_text("utf-8"))
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
        assert all(
            cell.get("execution_count") is None
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
