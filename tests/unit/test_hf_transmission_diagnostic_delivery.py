"""Static delivery boundary for the thin HF transmission Colab entrypoint."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.protocol.hf_transmission_diagnostic import (
    load_hf_transmission_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/colab/hf_transmission_diagnostic.ipynb"
SERVER = ROOT / "scripts/experiment_execution/hf_transmission_diagnostic_server.py"
HISTORICAL_DEVELOPMENT_NOTEBOOK = (
    ROOT / "notebooks/colab/development_exploration.ipynb"
)
PROTOCOL = ROOT / "configs/experiments/hf_transmission_diagnostic.json"
EXECUTION_REVISION = "af1eea8f55086b583e3e5e4a02586959983db70b"
RUN_ID = "ceg_wm_hf_transmission_diagnostic_server_execution"


def _constant(notebook: dict[str, object], name: str) -> object:
    values: list[object] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            ):
                assert isinstance(statement.value, ast.Constant)
                values.append(statement.value.value)
    assert len(values) == 1
    return values[0]


@pytest.mark.quick
def test_hf_transmission_server_help_imports_from_an_isolated_cwd(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        (sys.executable, "-I", str(SERVER), "--help"),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr


@pytest.mark.quick
def test_hf_transmission_notebook_is_thin_historical_and_scientific_only() -> None:
    notebook = json.loads(NOTEBOOK.read_text("utf-8"))
    code_cells = tuple(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    code_source = "\n".join(
        "".join(cell.get("source", [])) for cell in code_cells
    )

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(code_cells) == 5
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert _constant(notebook, "EXECUTION_REVISION") == EXECUTION_REVISION
    assert _constant(notebook, "RUN_ID") == RUN_ID
    assert "https://github.com/RICHAAARC/CEG-WM.git" in code_source
    assert "drive.mount('/content/drive')" in code_source
    assert "userdata.get('HF_TOKEN')" in code_source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in code_source
    assert "checkout', '--detach', 'FETCH_HEAD'" in code_source
    assert "status', '--porcelain'" in code_source
    assert "hf_transmission_diagnostic_server.py" in code_source
    assert "PERSISTENT_ROOT = DRIVE_ROOT / 'persistent'" in code_source
    assert "CACHE_ROOT = DRIVE_ROOT / 'cache'" in code_source
    assert "EXPORT_BASE = DRIVE_ROOT / 'exports'" in code_source
    assert "copy_to_drive_export" in code_source
    assert "execution_receipt.json" in code_source
    assert "SHA256SUMS" in code_source
    assert "hf_transmission_diagnostic_result" in code_source
    assert "hf_transmission_diagnostic_failure" in code_source
    assert source.count("eight-cluster development HF transmission diagnostic") == 1
    assert "fitted no threshold" in source
    assert "paused and is not authorized to run" in source
    assert "hf_only_detector_directional_validation.ipynb" in source
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "HfTransmissionDiagnosticRunner(",
        "DevelopmentScientificRecord(",
        "execute_hf_transmission_diagnostic_session(",
        "qk_geometry_sync",
        "geometry_synchronization_write",
        "lf_carrier",
        "content_router",
        "hf_only_threshold_fit",
        "4096",
        "--skip-dependency-install",
    ):
        assert forbidden not in code_source

    protocol, manifest = load_hf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )
    assert protocol.operational_unit_count == 0
    assert protocol.scientific_cluster_count == 8
    assert protocol.maximum_total_units == 8
    assert tuple(unit.unit_index for unit in protocol.unit_roster) == tuple(range(8))
    assert len(manifest.entries) == 8

    for path in (
        ROOT / "notebooks/colab/README.md",
        ROOT / "scripts/experiment_execution/README.md",
    ):
        readme = path.read_text("utf-8")
        assert "hf_transmission_diagnostic.ipynb" in readme
        assert EXECUTION_REVISION in readme
        assert RUN_ID in readme
        assert "paused" in readme.lower() or "暂停" in readme
        assert "not authorized" in readme.lower() or "不得运行" in readme


@pytest.mark.quick
def test_historical_development_notebook_is_explicitly_paused() -> None:
    notebook = json.loads(HISTORICAL_DEVELOPMENT_NOTEBOOK.read_text("utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "only development entrypoint currently authorized" not in source
    assert "historical development entrypoint" in source
    assert "paused and is not authorized to run" in source
    assert "do not use **Run all**" in source
    assert "hf_transmission_diagnostic.ipynb" in source
