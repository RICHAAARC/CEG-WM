from __future__ import annotations

import json
from pathlib import Path

import pytest


BOOKS = (
    "paper_main_worker_colab.ipynb",
    "paper_reconstruction_worker_colab.ipynb",
    "paper_results_finalize_colab.ipynb",
)
PAPER_PRODUCER_EXACT = "9ec454055c74cf4ed89001387c9f700e9ba5aef0"
BASELINE_PRODUCER_EXACT = "e4cf4ed2738cb91204695efbf9fb6ce35858b5f7"


@pytest.mark.unit
def test_formal_notebooks_are_clean_thin_fixed_entries() -> None:
    for name in BOOKS:
        payload = json.loads((Path("notebooks") / name).read_text(encoding="utf-8"))
        code = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
        assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
        text = "\n".join("".join(cell["source"]) for cell in code)
        assert PAPER_PRODUCER_EXACT in text
        assert "--engineering-canary" in text
        assert "PaperFormal-V1-EngineeringCanary" in text
        assert "canary_final.json" in text
        assert "'PYTHONPATH'" in text
        assert "'pip','install','-e'" not in text
        if name == "paper_results_finalize_colab.ipynb":
            assert BASELINE_PRODUCER_EXACT in text
        assert "force_remount" not in text
        assert "force-rerun-all" not in text
        for cell in code:
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.unit
def test_reconstruction_notebook_surfaces_worker_error_state() -> None:
    payload = json.loads(
        (Path("notebooks") / "paper_reconstruction_worker_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    assert "completed.returncode" in text
    assert "job_state.json" in text
    assert "error_code=" in text and "error=" in text
    compile(text, "paper_reconstruction_worker_colab.ipynb", "exec")
