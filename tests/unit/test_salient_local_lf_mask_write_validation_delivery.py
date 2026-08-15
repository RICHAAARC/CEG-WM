"""Delivery boundary for the salient-local-LF mask/write validation."""

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

from experiments.protocol.salient_local_lf_mask_write_validation import (
    load_salient_local_lf_mask_write_validation_protocol,
)
from scripts.experiment_execution.build_salient_local_lf_mask_write_validation_package import (
    build_salient_local_lf_mask_write_validation_package,
)
from scripts.experiment_execution import salient_local_lf_mask_write_validation_server as server


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/colab/salient_local_lf_mask_write_validation.ipynb"
PROTOCOL_PATH = ROOT / "configs/experiments/salient_local_lf_mask_write_validation.json"
EXECUTION_REVISION = "b2aea883eff21c959c1684bd86a4af1890ca9f15"
RUN_ID = "ceg_wm_salient_local_lf_mask_write_validation"
EXPECTED_PACKAGE_SHA256 = "f6bfa1a2acb64fca0ebc9a667101e89b14e7de27359d1742b4b9ab6811016bd1"
EXPECTED_PACKAGE_SIZE_BYTES = 4132169
SERVER_RELATIVE = Path("scripts/experiment_execution/salient_local_lf_mask_write_validation_server.py")


def _constant(notebook: dict[str, object], name: str) -> object:
    values: list[object] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
                assert isinstance(statement.value, ast.Constant)
                values.append(statement.value.value)
    assert len(values) == 1
    return values[0]


@pytest.fixture(scope="module")
def salient_local_lf_exact_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    package_root = tmp_path_factory.mktemp("salient_local_lf_exact_package")
    package = package_root / "execution.zip"
    result = build_salient_local_lf_mask_write_validation_package(
        ROOT, package, EXECUTION_REVISION
    )
    assert result["package_sha256"] == EXPECTED_PACKAGE_SHA256
    assert result["package_size_bytes"] == EXPECTED_PACKAGE_SIZE_BYTES
    return package


def test_salient_local_lf_exact_package_is_deterministic_and_importable(
    tmp_path: Path,
    salient_local_lf_exact_package: Path,
) -> None:
    repeated = tmp_path / "repeated.zip"
    result = build_salient_local_lf_mask_write_validation_package(
        ROOT, repeated, EXECUTION_REVISION
    )
    assert repeated.read_bytes() == salient_local_lf_exact_package.read_bytes()
    assert result["package_sha256"] == EXPECTED_PACKAGE_SHA256
    extracted = tmp_path / "extracted"
    with ZipFile(salient_local_lf_exact_package) as archive:
        assert archive.testzip() is None
        archive.extractall(extracted)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    server_module = ".".join(SERVER_RELATIVE.with_suffix("").parts)
    isolated_server_help = (
        "import sys; "
        f"sys.path.insert(0, {str(extracted)!r}); "
        f"from {server_module} import main; raise SystemExit(main(['--help']))"
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-c", isolated_server_help),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-revision" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr


@pytest.mark.parametrize("worker_failed", (False, True))
def test_salient_local_lf_server_writes_bounded_success_or_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    salient_local_lf_exact_package: Path,
    worker_failed: bool,
) -> None:
    protocol = load_salient_local_lf_mask_write_validation_protocol(
        PROTOCOL_PATH, repository_root=ROOT
    )
    persistent = tmp_path / "persistent"
    cache = tmp_path / "cache"
    persistent.mkdir()
    cache.mkdir()
    artifact = persistent / ("diagnostic.zip" if worker_failed else "result.zip")
    with ZipFile(artifact, "x") as archive:
        archive.writestr("diagnostic.json" if worker_failed else "result.json", "{}")
    worker = {
        "artifact_kind": (
            "salient_local_lf_mask_write_validation_failure"
            if worker_failed
            else "salient_local_lf_mask_write_validation_result"
        ),
        "diagnostic_zip" if worker_failed else "result_zip": str(artifact),
        "protocol_digest": protocol.digest(),
        "input_manifest_digest": protocol.manifest.digest(),
        "candidate_config_digest": "c" * 64,
        "unit_roster_digest": protocol.unit_roster_digest,
        "package_sha256": EXPECTED_PACKAGE_SHA256,
        "committed_unit_count": 0 if worker_failed else 10,
        "session_committed_unit_count": 0 if worker_failed else 10,
        "termination_reason": "operational_failure" if worker_failed else "frozen_roster_complete",
        "salient_local_lf_mask_write_aggregate": None if worker_failed else {
            "successful_observation_count": 8,
            "identity_failure_count": 0,
            "integrity_failure_count": 0,
            "implementation_failure_count": 0,
            "resource_failure_count": 0,
            "environment_failure_count": 0,
            "module_outcome": "mechanism_signal_not_observed",
        },
        "formal_tau_created": False,
        "fpr_estimated": False,
        "candidate_promoted": False,
        "scientific_claims_supported": not worker_failed,
    }
    monkeypatch.setattr(server, "_verify_repository", lambda *_args: None)
    monkeypatch.setattr(server, "_probe_resources", lambda **_kwargs: {"gpu": "public-test-gpu"})
    monkeypatch.setattr(server, "_verify_locked_dependencies", lambda _repository: "d" * 64)
    monkeypatch.setattr(server, "_download_configured_model", lambda **_kwargs: None)
    monkeypatch.setattr(
        server,
        "build_salient_local_lf_mask_write_validation_package",
        lambda _root, destination, _revision: (
            destination.write_bytes(salient_local_lf_exact_package.read_bytes())
            and {"package_sha256": EXPECTED_PACKAGE_SHA256}
        ),
    )
    monkeypatch.setattr(
        server,
        "_execute_worker",
        lambda **_kwargs: (3 if worker_failed else 0, worker),
    )
    secret_environment = {
        "HF_TOKEN": "salient-hf-secret",
        "CEG_WM_ROOT_KEY": "salient-root-secret",
        "CEG_WM_INSPYRENET_CHECKPOINT_PATH": "/bounded/checkpoint/ckpt_base.pth",
    }
    code, receipt = server.execute_salient_local_lf_mask_write_validation_server_session(
        repository_root=ROOT,
        expected_revision=EXECUTION_REVISION,
        persistent_root=persistent,
        cache_root=cache,
        run_id=RUN_ID,
        session_id="salient_local_lf_delivery_session",
        environment=secret_environment,
        install_dependencies=False,
    )
    receipt_bytes = Path(receipt["receipt_path"]).read_bytes()
    assert code == (3 if worker_failed else 0)
    assert receipt["execution_package_sha256"] == EXPECTED_PACKAGE_SHA256
    assert receipt["operational_unit_count"] == 2
    assert receipt["scientific_unit_count"] == 8
    assert receipt["total_unit_count"] == 10
    assert receipt["maximum_attempts_per_unit"] == 1
    assert (receipt["salient_local_lf_mask_write_aggregate"] is None) is worker_failed
    for secret in secret_environment.values():
        assert secret.encode() not in receipt_bytes


def test_salient_local_lf_notebook_is_thin_exact_and_exports_before_failure() -> None:
    notebook = json.loads(NOTEBOOK.read_text("utf-8"))
    code_cells = tuple(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert len(notebook["cells"]) == 6
    assert len(code_cells) == 5
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert _constant(notebook, "EXECUTION_REVISION") == EXECUTION_REVISION
    assert _constant(notebook, "RUN_ID") == RUN_ID
    assert _constant(notebook, "EXPECTED_PACKAGE_SHA256") == EXPECTED_PACKAGE_SHA256
    assert _constant(notebook, "EXPECTED_PACKAGE_SIZE_BYTES") == EXPECTED_PACKAGE_SIZE_BYTES
    assert "drive.mount('/content/drive')" in code_source
    assert "userdata.get('HF_TOKEN')" in code_source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in code_source
    assert "CEG_WM_INSPYRENET_CHECKPOINT_PATH" in code_source
    assert "checkout', '--detach', 'FETCH_HEAD'" in code_source
    assert "SHA256SUMS" in code_source
    assert code_source.index("copy_create_only(artifact_source") < code_source.index("if server_exit_code != 0")
    assert "two operational preflights and eight fixed scientific" in source
    assert "does not fit the masked-LF whitening asset" in source
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "SalientLocalLfMaskWriteValidationRunner(",
        "DevelopmentScientificRecord(",
        "execute_salient_local_lf_mask_write_validation_session(",
        "masked_null_whitened",
        "qk_",
        "tau =",
    ):
        assert forbidden not in code_source


def test_salient_local_lf_delivery_status_is_protocol_and_quality_only() -> None:
    state = json.loads(
        (ROOT / ".codex/research_state/salient_local_lf_candidate_readiness.yaml").read_text("utf-8")
    )
    assert state["source_cpu_api_implementation_ready"] is True
    assert state["experiment_protocol_admitted"] is True
    assert state["rgb_quality_gate_defined"] is True
    for field in (
        "candidate_runtime_qualified",
        "masked_lf_whitening_asset_ready",
        "scientific_mechanism_validated",
        "promoted",
        "formal_detector",
    ):
        assert state[field] is False
    assert state["diagnostic_only"] is True
    assert sha256(NOTEBOOK.read_bytes()).hexdigest()
