"""Delivery boundary for the thin HF-only detector directional Notebook."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from experiments.protocol.hf_only_detector_directional_validation import (
    load_hf_only_detector_directional_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/colab/hf_only_detector_directional_validation.ipynb"
SERVER_RELATIVE = Path(
    "scripts/experiment_execution/hf_only_detector_directional_validation_server.py"
)
SERVER = ROOT / SERVER_RELATIVE
PROTOCOL = ROOT / "configs/experiments/hf_only_detector_directional_validation.json"
EXECUTION_REVISION = "13000756d7af9111d129f5867ac06f816dfe326a"
RUN_ID = "ceg_wm_hf_only_detector_directional_validation_initial_gate"
REQUIRED_PACKAGE_MEMBERS = {
    "configs/experiments/hf_only_detector_directional_validation.json",
    "configs/experiments/hf_only_detector_directional_validation_manifest.json",
    "experiments/metrics/hf_only_detector_directional_validation.py",
    "experiments/protocol/hf_only_detector_directional_validation.py",
    "experiments/runners/hf_only_detector_directional_validation.py",
    "scripts/experiment_execution/hf_only_detector_directional_validation_entrypoint.py",
    SERVER_RELATIVE.as_posix(),
    "requirements_development_exploration_gpu_execution.txt",
}


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
def test_hf_detector_directional_server_help_imports_from_isolated_cwd(
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
    assert "--authorized-scientific-unit-count" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr


@pytest.mark.quick
def test_hf_detector_directional_notebook_is_thin_and_initial_gate_only() -> None:
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
    assert _constant(notebook, "AUTHORIZED_SCIENTIFIC_UNIT_COUNT") == 8
    assert "https://github.com/RICHAAARC/CEG-WM.git" in code_source
    assert "drive.mount('/content/drive')" in code_source
    assert "userdata.get('HF_TOKEN')" in code_source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in code_source
    assert "checkout', '--detach', 'FETCH_HEAD'" in code_source
    assert SERVER_RELATIVE.name in code_source
    assert "--authorized-scientific-unit-count" in code_source
    assert "execution_receipt.json" in code_source
    assert "SHA256SUMS" in code_source
    assert "two non-scientific operational checks" in source
    assert "eight HF-only detector directional scientific units" in source
    assert "fits no threshold" in source
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "HfOnlyDetectorDirectionalRunner(",
        "DevelopmentScientificRecord(",
        "execute_hf_only_detector_directional_validation_session(",
        "--skip-dependency-install",
        "qk_geometry_sync",
        "lf_carrier",
        "content_router",
        "hf_only_threshold_fit",
    ):
        assert forbidden not in code_source
    protocol, manifest = load_hf_only_detector_directional_protocol(
        PROTOCOL, repository_root=ROOT
    )
    assert protocol.operational_unit_count == 2
    assert protocol.initial_gpu_gate_scientific_unit_count == 8
    assert protocol.maximum_attempts_per_unit == 2
    assert len(manifest.operational_entries) == 2
    assert len(manifest.scientific_entries) == 32


@pytest.mark.quick
def test_hf_detector_directional_exact_checkout_builds_importable_execution_package(
    tmp_path: Path,
) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("detached research copy lacks exact Git checkout capability")
    checkout = tmp_path / "exact_checkout"
    subprocess.run(
        ("git", "clone", "--no-checkout", "--quiet", str(ROOT), str(checkout)),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(checkout), "checkout", "--detach", "--quiet", EXECUTION_REVISION),
        check=True,
    )
    package_root = tmp_path / "package_persistent"
    build_script = (
        "from pathlib import Path; import sys; "
        "sys.path.insert(0, str(Path('.').resolve())); "
        "from scripts.experiment_execution.development_exploration_entrypoint "
        "import _build_or_verify_package; "
        f"print(_build_or_verify_package(Path('.').resolve(), Path({str(package_root)!r}), {EXECUTION_REVISION!r}))"
    )
    built = subprocess.run(
        (sys.executable, "-I", "-c", build_script),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    package = Path(built.stdout.strip())
    with ZipFile(package) as archive:
        names = set(archive.namelist())
        assert REQUIRED_PACKAGE_MEMBERS <= names
        assert archive.testzip() is None
    isolated_help = subprocess.run(
        (sys.executable, "-I", str(checkout / SERVER_RELATIVE), "--help"),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert isolated_help.returncode == 0, isolated_help.stderr
    assert sha256(package.read_bytes()).hexdigest()


@pytest.mark.quick
def test_hf_detector_directional_readmes_expose_only_current_notebook() -> None:
    for path in (
        ROOT / "notebooks/colab/README.md",
        ROOT / "scripts/experiment_execution/README.md",
    ):
        source = path.read_text("utf-8")
        assert NOTEBOOK.name in source
        assert EXECUTION_REVISION in source
        assert RUN_ID in source
        assert "hf_transmission_diagnostic.ipynb" in source
        assert "paused / not authorized" in source
