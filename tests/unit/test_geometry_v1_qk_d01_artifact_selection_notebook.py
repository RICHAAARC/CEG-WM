"""Narrow static contract for the D0.1 Drive-only Colab handoff."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "geometry_v1_qk_d01_artifact_selection_colab.ipynb"


def test_d01_notebook_is_unexecuted_drive_first_single_runner_cpu_handoff() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); cells = notebook["cells"]
    assert notebook["nbformat"] == 4 and cells[0]["cell_type"] == "code"
    assert cells[0]["source"] == ["from google.colab import drive\n", "drive.mount('/content/drive')\n"]
    assert all(cell.get("execution_count") is None and not cell.get("outputs", []) for cell in cells if cell["cell_type"] == "code")
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    assert source.count("subprocess.Popen(") == 1
    assert "RUNNER_PATH = 'experiments/run_geometry_v1_qk_d01_artifact_selection_operational.py'" in source
    assert "'--expected-exact', execution_commit" in source and "'--source-root', str(SOURCE_ROOT)" in source and "'--control-fd', str(control_write)" in source
    assert "4732211beefbeface95cb842c117b9719e362f1a" in source
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V1/D0/Geometry-V1-QK-D0-4732211beefb-20260827T064555Z" in source
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V1/D01" in source and "timeout=7200" in source
    assert "MAX_CONTROL_BYTES = 1024" in source and "stderr=subprocess.DEVNULL" in source
    for prohibited in ("HF_TOKEN", "diffusers", "torch", "cuda", "zipfile", "read_bytes", "hashlib"):
        assert prohibited not in source
    assert "does not display source ZIP contents, retry, fall back, switch layers, tune, or choose per sample" in source
