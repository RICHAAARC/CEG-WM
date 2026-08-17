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
        "semantic_texture_operational_preflight.ipynb",
    }
    assert {path.name for path in notebook_root.glob("*.ipynb")} == expected
    paused_notebooks = expected - {"semantic_texture_operational_preflight.ipynb"}
    banner = (
        "**PAUSED / NOT AUTHORIZED:** Do not use **Run all**. The sole current "
        "Run-all entrypoint is `semantic_texture_operational_preflight.ipynb`."
    )
    current_run_all_phrase = "Run all once in a fresh GPU runtime"
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
                f"{banner}\n\n"
            )
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
        assert all(
            cell.get("execution_count") is None
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
    assert {
        name for name, source in sources.items() if current_run_all_phrase in source
    } == {"semantic_texture_operational_preflight.ipynb"}
    for source in sources.values():
        for phrase in forbidden_authority_phrases:
            assert phrase not in source
