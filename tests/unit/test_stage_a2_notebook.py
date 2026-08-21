from __future__ import annotations

import json
from pathlib import Path

import pytest

_NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "stage_a2_hf_colab.ipynb"


@pytest.mark.unit
def test_stage_a2_notebook_is_output_free_exact_bound_and_thin() -> None:
    document = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    first = "".join(code_cells[0]["source"])
    last = "".join(code_cells[-1]["source"])

    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    assert "from google.colab import drive" in first
    assert "drive.mount('/content/drive')" in first
    assert "'fetch', '--depth', '1', 'origin', APPROVED_EXECUTION_EXACT" in source
    assert "'checkout', '--detach', 'FETCH_HEAD'" in source
    assert "getpass.getpass" in source
    assert "experiments.stage_a.run_hf_a2_colab" in source
    assert "--resume-zip" in source and "--resume-checksum" in source
    runner_cell = "".join(code_cells[-2]["source"])
    assert "if line.startswith('CEGWM_PROGRESS ')" in runner_cell
    assert "CEGWM_SUMMARY " not in runner_cell and "CEGWM_FATAL " not in runner_cell
    assert "zipfile.ZipFile" in last and "archived_receipt != receipt" in last
    assert "hashlib.sha256" in last and "shutil.copy2" in last
    assert last.index("Drive checksum copy mismatch") < last.index("summary =")
