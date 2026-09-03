from __future__ import annotations

import json
from pathlib import Path

import pytest


BOOKS = (
    "paper_baseline_worker_colab-t2smark.ipynb",
    "paper_baseline_worker_colab-treering.ipynb",
    "paper_baseline_worker_colab-gaussian-shading.ipynb",
    "paper_baseline_worker_colab-shallow-diffuse.ipynb",
)

PRODUCER_EXACT = "490c7133f98270d126c4dfa5ef60fdf55cc79e0a"


@pytest.mark.unit
def test_formal_baseline_notebooks_are_clean_thin_fixed_entries() -> None:
    for name in BOOKS:
        payload = json.loads((Path("notebooks") / name).read_text(encoding="utf-8"))
        code = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
        assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
        text = "\n".join("".join(cell["source"]) for cell in code)
        assert PRODUCER_EXACT in text
        assert "experiments.run_paper_baseline_worker" in text
        assert "--job-id" in text and "--expected-exact" in text
        assert "force_remount" not in text
        assert "force-rerun-all" not in text
        for cell in code:
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
