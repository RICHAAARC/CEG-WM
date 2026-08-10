"""Delivery boundary for LF whitened directional validation."""

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

from experiments.protocol.lf_whitened_directional_validation import (
    load_lf_whitened_directional_validation_protocol,
)
from scripts.experiment_execution.development_exploration_entrypoint import (
    _build_or_verify_package,
)
from scripts.experiment_execution import lf_whitened_directional_validation_server as server


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/colab/lf_whitened_directional_validation.ipynb"
PROTOCOL = ROOT / "configs/experiments/lf_whitened_directional_validation.json"
SERVER_RELATIVE = Path("scripts/experiment_execution/lf_whitened_directional_validation_server.py")
EXECUTION_REVISION = "194eccdd1f16c295528a4d9e1d7c75c2748f061a"
RUN_ID = "ceg_wm_lf_whitened_directional_validation_prepared_feature_execution"
WHITENING_ASSET_FIT_RUN_ID = "ceg_wm_lf_whitening_asset_fit_and_score_screening"
REQUIRED_PACKAGE_MEMBERS = {
    "configs/experiments/lf_whitened_directional_validation.json",
    "configs/experiments/lf_whitened_directional_validation_manifest.json",
    "experiments/metrics/lf_whitened_directional_validation.py",
    "experiments/protocol/lf_whitened_directional_validation.py",
    "experiments/runners/lf_whitened_directional_validation.py",
    "scripts/experiment_execution/lf_whitened_directional_validation_entrypoint.py",
    SERVER_RELATIVE.as_posix(),
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


@pytest.fixture(scope="module")
def lf_whitened_directional_exact_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    package_root = tmp_path_factory.mktemp("lf_whitened_directional_exact_package")
    return _build_or_verify_package(ROOT, package_root, "a" * 40)


@pytest.mark.quick
def test_lf_whitened_directional_server_help_imports_from_isolated_cwd(
    tmp_path: Path,
    lf_whitened_directional_exact_package: Path,
) -> None:
    extracted = tmp_path / "package"
    with ZipFile(lf_whitened_directional_exact_package) as archive:
        archive.extractall(extracted)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        (sys.executable, "-I", str(extracted / SERVER_RELATIVE), "--help"),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--whitening-asset-persistent-root" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr


@pytest.mark.quick
def test_lf_whitened_directional_generic_package_contains_complete_execution_chain(
    lf_whitened_directional_exact_package: Path,
) -> None:
    with ZipFile(lf_whitened_directional_exact_package) as archive:
        names = set(archive.namelist())
        assert REQUIRED_PACKAGE_MEMBERS <= names
        assert archive.testzip() is None
    assert sha256(lf_whitened_directional_exact_package.read_bytes()).hexdigest()


@pytest.mark.quick
def test_lf_whitened_directional_server_writes_safe_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lf_whitened_directional_exact_package: Path,
) -> None:
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        PROTOCOL, repository_root=ROOT
    )
    persistent = tmp_path / "persistent"
    fit_persistent = tmp_path / "fit_persistent"
    cache = tmp_path / "cache"
    for path in (persistent, fit_persistent, cache):
        path.mkdir()
    artifact = persistent / "worker.zip"
    with ZipFile(artifact, "x") as archive:
        archive.writestr("result.json", "{}")
    root_secret = "directional-root-secret"
    hf_secret = "directional-hf-secret"
    monkeypatch.setattr(server, "_verify_repository", lambda *_args: None)
    monkeypatch.setattr(server, "_probe_resources", lambda **_kwargs: {"gpu": "test"})
    monkeypatch.setattr(server, "_download_configured_model", lambda **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_build_or_verify_package",
        lambda *_args: lf_whitened_directional_exact_package,
    )
    worker = {
        "result_zip": str(artifact),
        "protocol_digest": protocol.digest(),
        "input_manifest_digest": manifest.digest(),
        "candidate_config_digest": "c" * 64,
        "whitening_asset_digest": "d" * 64,
        "whitening_asset_fit_producer_revision": protocol.whitening_asset_fit_producer_revision,
        "unit_roster_digest": protocol.unit_roster_digest,
        "source_cluster_deny_list_digest": protocol.source_cluster_deny_list_digest,
        "committed_unit_count": 33,
        "session_committed_unit_count": 33,
        "termination_reason": "frozen_roster_complete",
        "directional_aggregate": None,
        "formal_tau_created": False,
        "fpr_estimated": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
    }
    monkeypatch.setattr(
        server,
        "execute_lf_whitened_directional_validation_session",
        lambda **_kwargs: (0, worker),
    )

    code, receipt = server.execute_lf_whitened_directional_validation_server_session(
        repository_root=ROOT,
        expected_revision="a" * 40,
        persistent_root=persistent,
        whitening_asset_persistent_root=fit_persistent,
        cache_root=cache,
        run_id=protocol.run_id,
        session_id="lf_whitened_directional_receipt_session",
        environment={"HF_TOKEN": hf_secret, "CEG_WM_ROOT_KEY": root_secret},
        install_dependencies=False,
    )

    receipt_bytes = Path(receipt["receipt_path"]).read_bytes()
    assert code == 0
    assert receipt["committed_revision"] == "a" * 40
    assert receipt["operational_unit_count"] == 1
    assert receipt["scientific_unit_count"] == 32
    assert receipt["formal_tau_created"] is False
    assert receipt["fpr_estimated"] is False
    assert receipt["candidate_promoted"] is False
    assert root_secret.encode() not in receipt_bytes
    assert hf_secret.encode() not in receipt_bytes
    assert json.loads(receipt_bytes)["development_claim_boundary"] == protocol.claim_boundary


@pytest.mark.quick
def test_lf_whitened_directional_notebook_is_thin_and_output_free() -> None:
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
    assert (
        _constant(notebook, "WHITENING_ASSET_FIT_RUN_ID")
        == WHITENING_ASSET_FIT_RUN_ID
    )
    assert "https://github.com/RICHAAARC/CEG-WM.git" in code_source
    assert "drive.mount('/content/drive')" in code_source
    assert "userdata.get('HF_TOKEN')" in code_source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in code_source
    assert "checkout', '--detach', 'FETCH_HEAD'" in code_source
    assert SERVER_RELATIVE.name in code_source
    assert (
        "DRIVE_ROOT = DRIVE_MOUNT / 'MyDrive' / 'CEG-WM' / "
        "'lf_whitened_directional_validation'"
    ) in code_source
    assert "PERSISTENT_ROOT = DRIVE_ROOT / 'persistent'" in code_source
    assert (
        "WHITENING_ASSET_PERSISTENT_ROOT = DRIVE_MOUNT / 'MyDrive' / "
        "'CEG-WM' / 'lf_whitened_score_screening' / 'persistent'"
    ) in code_source
    assert "--whitening-asset-persistent-root" in code_source
    assert "execution_receipt.json" in code_source
    assert "SHA256SUMS" in code_source
    assert "one non-scientific public-endpoint smoke unit" in source
    assert "thirty-two frozen LF whitened directional scientific units" in source
    assert "fits no threshold" in source
    assert "estimates no FPR" in source
    assert "promotes no candidate" in source
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "LfWhitenedDirectionalValidationRunner(",
        "DevelopmentScientificRecord(",
        "execute_lf_whitened_directional_validation_session(",
        "--skip-dependency-install",
        "content_router",
        "qk_geometry_sync",
        "hf_only_threshold_fit",
        "4096",
    ):
        assert forbidden not in code_source

    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        PROTOCOL,
        repository_root=ROOT,
    )
    assert protocol.run_id == RUN_ID
    assert protocol.whitening_asset_fit_run_id == WHITENING_ASSET_FIT_RUN_ID
    assert protocol.operational_unit_count == 1
    assert protocol.scientific_cluster_count == 32
    assert protocol.maximum_total_units == 33
    assert protocol.maximum_attempts_per_unit == 2
    assert protocol.maximum_duration_seconds_per_unit == 2700
    assert len(manifest.entries) == 32
