"""Delivery boundary for the thin LF transmission diagnostic Notebook."""

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

from experiments.protocol.lf_transmission_diagnostic import (
    load_lf_transmission_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/colab/lf_transmission_diagnostic.ipynb"
SERVER_RELATIVE = Path(
    "scripts/experiment_execution/lf_transmission_diagnostic_server.py"
)
SERVER = ROOT / SERVER_RELATIVE
PROTOCOL = ROOT / "configs/experiments/lf_transmission_diagnostic.json"
EXECUTION_REVISION = "2337f9d7c773a6054d558108e31d07d35fbee42f"
RUN_ID = "ceg_wm_lf_carrier_to_detector_transmission_diagnostic"
REQUIRED_PACKAGE_MEMBERS = {
    "configs/experiments/lf_transmission_diagnostic.json",
    "configs/experiments/lf_transmission_diagnostic_manifest.json",
    "experiments/metrics/lf_transmission_diagnostic.py",
    "experiments/protocol/lf_transmission_diagnostic.py",
    "experiments/runners/lf_transmission_diagnostic.py",
    "scripts/experiment_execution/lf_transmission_diagnostic_entrypoint.py",
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
def test_lf_transmission_server_help_imports_from_an_isolated_cwd(
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
    assert "server_support" in SERVER.read_text("utf-8")


@pytest.mark.quick
def test_lf_transmission_notebook_is_thin_and_scientific_only() -> None:
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
    assert SERVER_RELATIVE.name in code_source
    assert "DRIVE_ROOT = DRIVE_MOUNT / 'MyDrive' / 'CEG-WM' / 'lf_transmission_diagnostic'" in code_source
    assert "PERSISTENT_ROOT = DRIVE_ROOT / 'persistent'" in code_source
    assert "CACHE_ROOT = DRIVE_ROOT / 'cache'" in code_source
    assert "EXPORT_BASE = DRIVE_ROOT / 'exports'" in code_source
    assert "execution_receipt.json" in code_source
    assert "SHA256SUMS" in code_source
    assert "lf_transmission_diagnostic_result" in code_source
    assert "lf_transmission_diagnostic_failure" in code_source
    assert "zero operational units and eight scientific units" in source
    assert "2700 seconds per unit" in source
    assert "fits no threshold" in source
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "LfTransmissionDiagnosticRunner(",
        "DevelopmentScientificRecord(",
        "execute_lf_transmission_diagnostic_session(",
        "--skip-dependency-install",
        "qk_geometry_sync",
        "geometry_synchronization_write",
        "content_router",
        "hf_only_threshold_fit",
        "4096",
    ):
        assert forbidden not in code_source

    protocol, manifest = load_lf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )
    assert protocol.operational_unit_count == 0
    assert protocol.scientific_cluster_count == 8
    assert protocol.maximum_total_units == 8
    assert protocol.maximum_attempts_per_unit == 2
    assert protocol.maximum_duration_seconds_per_unit == 2700
    assert tuple(unit.unit_index for unit in protocol.unit_roster) == tuple(range(8))
    assert len(manifest.entries) == 8


@pytest.mark.quick
def test_lf_transmission_exact_checkout_builds_importable_execution_package(
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
        (
            "git",
            "-C",
            str(checkout),
            "checkout",
            "--detach",
            "--quiet",
            EXECUTION_REVISION,
        ),
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
def test_lf_transmission_readmes_preserve_paused_historical_boundary() -> None:
    for path in (
        ROOT / "notebooks/colab/README.md",
        ROOT / "scripts/experiment_execution/README.md",
    ):
        source = path.read_text("utf-8")
        assert NOTEBOOK.name in source
        assert EXECUTION_REVISION in source
        assert RUN_ID in source
        if path.parent.name == "colab":
            assert "当前授权的用户流程固定为两步：先在" in source
            assert "source 完成只读认证并以该 exact bundle identity 绑定后" in source
            assert "不允许 retry、fallback 或动态 latest-bundle 选择" in source
            assert "不产生 threshold、FPR、tau、promotion 或 scientific claim" in source
        else:
            assert "The current user-only sequence has exactly two entrypoints:" in source
            assert "only after source read-only authentication and exact bundle-identity binding" in source
            assert "Neither step allows retry, fallback, or dynamic latest-bundle selection" in source
            assert "threshold, FPR, tau, promotion, or scientific claim" in source
        assert "semantic_texture_soft_detector_asset_preparation.ipynb" in source
        assert "lf_whitened_score_screening.ipynb" in source
        assert "a78c47184cf83ad351bb4442ebd31c218726de25" in source
        assert "ceg_wm_lf_whitening_asset_fit_and_score_screening" in source
        assert "hf_only_detector_directional_validation.ipynb" in source
        assert "paused / not authorized" in source
