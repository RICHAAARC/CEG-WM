"""Narrow static contract for the directly runnable D0 Colab handoff."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "geometry_v1_qk_all_layer_discovery_colab.ipynb"


def test_d0_notebook_is_unexecuted_drive_first_and_single_runner_handoff() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); cells = notebook["cells"]
    assert notebook["nbformat"] == 4 and cells[0]["cell_type"] == "code"
    assert cells[0]["source"] == ["from google.colab import drive\n", "drive.mount('/content/drive')\n"]
    assert all(cell.get("execution_count") is None and not cell.get("outputs", []) for cell in cells if cell["cell_type"] == "code")
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    assert source.count("subprocess.Popen(") == 1 and "RUNNER_PATH = 'experiments/run_geometry_v1_qk_all_layer_discovery_operational.py'" in source
    assert "'--expected-exact', execution_commit" in source and "'--control-fd', str(control_write)" in source
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V1/D0" in source and "timeout=7200" in source
    assert "HF_TOKEN" in source and "print(hf_token)" not in source
    for prohibited in ("zipfile", "read_bytes", "hashlib", "REFERENCES ="):
        assert prohibited not in source
    assert "science_denominator=0" in source and "Geometry-V1 D0 all-layer discovery handoff" in source
