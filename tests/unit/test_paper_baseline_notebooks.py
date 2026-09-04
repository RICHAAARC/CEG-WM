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

PRODUCER_EXACT = "e4cf4ed2738cb91204695efbf9fb6ce35858b5f7"


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
        assert "--engineering-canary" in text
        assert "PaperFormal-V1-EngineeringCanary" in text
        assert "canary_final.json" in text
        assert "'PYTHONPATH'" in text
        assert "'pip', 'install', '-e'" not in text
        assert "force_remount" not in text
        assert "force-rerun-all" not in text
        for cell in code:
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.unit
def test_t2smark_notebook_uses_proven_runtime_profile_and_surfaces_worker_error() -> None:
    payload = json.loads(
        (Path("notebooks") / "paper_baseline_worker_colab-t2smark.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    for dependency in (
        "diffusers==0.32.0",
        "transformers==4.45.2",
        "accelerate==1.1.1",
        "huggingface_hub==0.26.2",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
    ):
        assert dependency in text
    assert "completed.returncode" in text
    assert "job_state.json" in text
    assert "error_code=" in text and "error=" in text
    compile(text, "paper_baseline_worker_colab-t2smark.ipynb", "exec")
