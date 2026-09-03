from __future__ import annotations

import json
from pathlib import Path

import pytest


BOOKS = (
    "paper_main_worker_colab.ipynb",
    "paper_reconstruction_worker_colab.ipynb",
    "paper_results_finalize_colab.ipynb",
)
PAPER_PRODUCER_EXACT = "93fc45a03ed3c15b1fde768316ba8db9dcff25e5"
BASELINE_PRODUCER_EXACT = "004b73dd1ebcceae73f05adb76159788414fb43f"


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
