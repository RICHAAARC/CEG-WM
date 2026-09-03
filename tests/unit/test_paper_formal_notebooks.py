from __future__ import annotations

import json
from pathlib import Path

import pytest


BOOKS = (
    "paper_main_worker_colab.ipynb",
    "paper_reconstruction_worker_colab.ipynb",
    "paper_results_finalize_colab.ipynb",
)
PAPER_PRODUCER_EXACT = "e0deb60d3796a59891cd669fe6f071589897885d"
BASELINE_PRODUCER_EXACT = "23862e0c47411d67e66a617cf35dbd54bbdc0435"


@pytest.mark.unit
def test_formal_notebooks_are_clean_thin_fixed_entries() -> None:
    for name in BOOKS:
        payload = json.loads((Path("notebooks") / name).read_text(encoding="utf-8"))
        code = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
        assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
        text = "\n".join("".join(cell["source"]) for cell in code)
        assert PAPER_PRODUCER_EXACT in text
        if name == "paper_results_finalize_colab.ipynb":
            assert BASELINE_PRODUCER_EXACT in text
        assert "force_remount" not in text
        assert "force-rerun-all" not in text
        for cell in code:
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
