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
    assert "CEG_WM_ROOT_KEY" in first and "HF_TOKEN" in first
    assert "userdata.get(name)" in first
    assert "input(" not in source and "getpass" not in source
    assert "CEGWM_STAGE_A_DETECTION_KEY" not in source
    assert "'fetch', '--depth', '1', 'origin', 'refs/heads/stage-a-method-feasibility'" in source
    assert "'checkout', '--detach', 'FETCH_HEAD'" in source
    assert "experiments.stage_a.run_hf_a2_colab" in source
    assert "--run-store-root" in source
    assert "--model-revision" not in source and "--checkpoint-interval-hours" not in source
    runner_cell = "".join(code_cells[-2]["source"])
    assert "if line.startswith('CEGWM_PROGRESS ')" in runner_cell
    assert "CEGWM_SUMMARY " not in runner_cell and "CEGWM_FATAL " not in runner_cell
    assert "zipfile.ZipFile" in last and "hashlib.sha256" in last
    assert last.index("final fixed denominator mismatch") < last.index("summary =")
