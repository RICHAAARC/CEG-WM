"""Static contracts for the all-layer direction selection handoff."""
from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "geometry_v1_qk_direction_all_layer_selection_colab.ipynb"


def _code(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_notebook_is_valid_unexecuted_and_mounts_drive_first() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); cells = notebook["cells"]
    assert notebook["nbformat"] == 4
    assert cells[0]["source"] == ["from google.colab import drive\n", "\n", "drive.mount('/content/drive')\n"]
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in cells if cell["cell_type"] == "code")


def test_notebook_binds_source_and_detached_runner_with_one_child() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); source = _code(notebook); ast.parse(source)
    assert "DIRECTION_RUNNER_EXACT = '41742d462d62525189855c8ebb2ee1995fb9230a'" in source
    assert "SOURCE_D0_EXACT = '4732211beefbeface95cb842c117b9719e362f1a'" in source
    assert "geometry-v1-qk-d0-4732211beefb" in source and "geometry-v1-qk-d0-all-layer-discovery-v1" in source
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V1/D0/Geometry-V1-QK-D0-4732211beefb-20260827T064555Z" in source
    assert "git', 'clone', '--no-checkout'" in source and "git', 'checkout', '--detach', DIRECTION_RUNNER_EXACT" in source
    assert "execution_commit != DIRECTION_RUNNER_EXACT or not checkout_clean" in source and "runner_path = repo / RUNNER_PATH" in source
    assert source.count("subprocess.Popen(") == 1 and "env=" not in source and "timeout=7200" in source
    assert "--control-fd" in source and "/content/drive/MyDrive/CEG-WM/Geometry-V1/DIRECTION_ALL_LAYER" in source
    assert "source_d0_artifact_identity" in source and "runner_execution_identity" in source and "science_denominator': 0" in source


def test_notebook_has_no_external_method_surface() -> None:
    source = _code(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    for forbidden in ("force_remount", "torch", "diffusers", "HF_TOKEN", "token", "retry", "fallback", "zipfile", "ZipFile", "cuda"):
        assert forbidden not in source
    assert "stdout=subprocess.DEVNULL" in source and "stderr=subprocess.DEVNULL" in source
