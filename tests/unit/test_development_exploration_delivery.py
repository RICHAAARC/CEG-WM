"""Static delivery checks for the thin development exploration Notebook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks/colab/development_exploration.ipynb"
EXECUTION_REVISION = "5b5f4bb0b47e8153cdb603225141a911d61bb725"


@pytest.mark.quick
def test_development_exploration_notebook_is_thin_and_output_free() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = tuple(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert 4 <= len(code_cells) <= 6
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert "https://github.com/RICHAAARC/CEG-WM.git" in source
    assert f"EXECUTION_REVISION = '{EXECUTION_REVISION}'" in source
    assert "RUN_ID = 'ceg-wm-development-exploration'" in source
    assert "drive.mount('/content/drive')" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in source
    assert "git', '-C', str(CHECKOUT_ROOT), 'fetch'" in source
    assert "checkout', '--detach', 'FETCH_HEAD'" in source
    assert "status', '--porcelain'" in source
    assert "development_exploration_server.py" in source
    assert "subprocess.Popen" in source and "stderr=subprocess.STDOUT" in source
    assert "server_receipts' / SESSION_ID / 'execution_receipt.json'" in source
    assert "server_failures' / SESSION_ID" in source
    assert "execution_failure_receipt_*.json" in source
    assert "SHA256SUMS" in source
    assert "copy_to_drive_export" in source
    assert "Drive export SHA-256 mismatch" in source
    assert source.index("process = subprocess.Popen") < source.index(
        "EXPORT_ROOT.mkdir"
    )
    assert source.index("copy_to_drive_export(artifact_source") < source.index(
        "if server_exit_code != 0"
    )
    assert "mutable branch must never replace" in source
    assert "scientific completion is determined only" in source
    assert "COMMITTED" in source
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "DevelopmentExplorationRunner(",
        "DevelopmentScientificRecord(",
        "execute_development_exploration_session(",
        "hf_only_threshold_fit",
        "4096",
        "--skip-dependency-install",
        "zipfile",
    ):
        assert forbidden not in source
