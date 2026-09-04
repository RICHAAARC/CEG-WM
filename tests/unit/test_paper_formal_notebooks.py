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
FORMAL_DRIVE_ROOT = "/content/drive/MyDrive/CEG-WM/PaperFormal-V1"
FORMAL_JOB_IDS = {
    "paper_main_worker_colab.ipynb": "paper-main-v1",
    "paper_reconstruction_worker_colab.ipynb": "paper-main-reconstruction-v1",
}


@pytest.mark.unit
def test_formal_notebooks_are_clean_thin_fixed_entries() -> None:
    for name in BOOKS:
        payload = json.loads((Path("notebooks") / name).read_text(encoding="utf-8"))
        code = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
        assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
        text = "\n".join("".join(cell["source"]) for cell in code)
        assert PAPER_PRODUCER_EXACT in text
        assert FORMAL_DRIVE_ROOT in text
        assert "--engineering-canary" not in text
        assert "--finalize-incomplete" not in text
        assert "PaperFormal-V1-EngineeringCanary" not in text
        assert "canary_final.json" not in text
        assert "'PYTHONPATH'" in text
        assert "'pip','install','-e'" not in text
        assert "'checkout','--detach',EXPECTED_EXACT" in text
        assert "head==EXPECTED_EXACT and not dirty" in text
        if name == "paper_results_finalize_colab.ipynb":
            assert BASELINE_PRODUCER_EXACT in text
            assert "'--drive-root',str(drive_root)" in text
            assert "output_root=drive_root/'finalized'/'paper-formal-v1'" in text
            assert "unified_result_package.json" in text
            assert "job_state.json" in text
        else:
            assert FORMAL_JOB_IDS[name] in text
            expected_final = (
                "reconstruction_final.json"
                if name == "paper_reconstruction_worker_colab.ipynb"
                else "method_final.json"
            )
            assert expected_final in text
            if name == "paper_reconstruction_worker_colab.ipynb":
                assert "'--drive-root',str(drive_root)" in text
                assert "output_root=drive_root/'reconstruction'/JOB_ID" in text
                assert "job_state.json" in text
                assert "result_package_produced':False" in text
            else:
                assert "'--drive-root',str(drive_root/'main')" in text
                assert "final_path=drive_root/'main'/JOB_ID/'method_final.json'" in text
        assert "force_remount" not in text
        assert "force-rerun-all" not in text
        compile(text, name, "exec")
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
