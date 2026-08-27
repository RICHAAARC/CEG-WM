"""Static contracts for the thin D1 Drive handoff notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "geometry_v1_qk_d1_independent_confirmation_colab.ipynb"


def test_d1_notebook_is_unexecuted_and_mounts_drive_first() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); cells = notebook["cells"]
    assert notebook["nbformat"] == 4 and cells[0]["cell_type"] == "code"
    assert cells[0]["source"] == ["from google.colab import drive\n", "\n", "drive.mount('/content/drive')\n"]
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in cells if cell["cell_type"] == "code")


def test_d1_notebook_binds_runner_and_source_identities_with_one_child() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "SOURCE_D01_ARTIFACT_EXACT = 'ccfb7bcefbb18f9812a4e800bbea18b91b031ebb'" in source
    assert "D1_RUNNER_EXACT = '906478a04334118c1fd71996e38ab905bea6d35a'" in source
    assert "geometry-v1-qk-d01-ccfb7bcefbb1" in source and "geometry-v1-qk-d01-artifact-selection-v1" in source
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V1/D01/Geometry-V1-QK-D01-ccfb7bcefbb1-20260827T083601Z" in source
    assert "git', 'clone', '--no-checkout'" in source and "git', 'checkout', '--detach', D1_RUNNER_EXACT" in source
    assert "execution_commit != D1_RUNNER_EXACT or not checkout_clean" in source
    assert "RUNNER_PATH = 'experiments/run_geometry_v1_qk_d1_independent_confirmation_operational.py'" in source and "runner_path = repo / RUNNER_PATH" in source
    assert source.count("subprocess.Popen(") == 1 and "'--control-fd', str(control_write)" in source and "timeout=7200" in source
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V1/D1" in source and "source_d01_artifact_identity" in source and "runner_execution_identity" in source


def test_d1_notebook_has_no_model_or_retry_surface() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)
    for forbidden in ("torch", "diffusers", "HF_TOKEN", "retry", "force_remount", "zipfile", "ZipFile"):
        assert forbidden not in source
    assert "stdout=subprocess.DEVNULL" in source and "stderr=subprocess.DEVNULL" in source
