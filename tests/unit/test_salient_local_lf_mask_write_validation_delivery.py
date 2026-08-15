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
    resolve_required_git_authority_revisions,
)
from scripts.experiment_execution import salient_local_lf_mask_write_validation_server as server


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/colab/salient_local_lf_mask_write_validation.ipynb"
PROTOCOL_PATH = ROOT / "configs/experiments/salient_local_lf_mask_write_validation.json"
EXECUTION_REVISION = "bbf66617fec64842260066afdfec1169a8cf1688"
RUN_ID = "ceg_wm_salient_local_lf_mask_write_remote_authority_correction_validation"
EXPECTED_PACKAGE_SHA256 = "d0814b2ac907391a9213cbe108d2be1916f3ff71b835fcb298ec32f49cca6f4c"
EXPECTED_PACKAGE_SIZE_BYTES = 4134799
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


@pytest.mark.parametrize(
    ("process_case", "expected_return_code", "expected_failure_class"),
    (
        ("import_before_main", 1, "implementation_blocked"),
        ("unexpected_return_code", 7, "implementation_blocked"),
        ("signal_termination", -15, "resource_blocked"),
        ("missing_result", 0, "integrity_blocked"),
    ),
)
def test_real_worker_process_failures_are_exported_as_bounded_server_evidence(
    tmp_path: Path,
    process_case: str,
    expected_return_code: int,
    expected_failure_class: str,
) -> None:
    repository = tmp_path / "worker-repository"
    module_root = repository / "scripts/experiment_execution"
    module_root.mkdir(parents=True)
    (repository / "scripts/__init__.py").write_text("", encoding="utf-8")
    (module_root / "__init__.py").write_text("", encoding="utf-8")
    entrypoint = module_root / "salient_local_lf_mask_write_validation_entrypoint.py"
    if process_case == "import_before_main":
        entrypoint.write_text("def broken(:\n", encoding="utf-8")
    elif process_case == "unexpected_return_code":
        entrypoint.write_text(
            "import sys\n"
            "sys.stderr.write('secret-token /content/drive/private prompt key tensor\\n')\n"
            "raise SystemExit(7)\n",
            encoding="utf-8",
        )
    elif process_case == "signal_termination":
        entrypoint.write_text(
            "import os, signal\n"
            "os.kill(os.getpid(), signal.SIGTERM)\n",
            encoding="utf-8",
        )
    else:
        entrypoint.write_text("raise SystemExit(0)\n", encoding="utf-8")
    persistent = tmp_path / "persistent"
    cache = tmp_path / "cache"
    persistent.mkdir()
    cache.mkdir()
    package = persistent / "execution.zip"
    with ZipFile(package, "x") as archive:
        archive.writestr("member.txt", "bounded package")
    package_sha = sha256(package.read_bytes()).hexdigest()
    environment = {
        **os.environ,
        "HF_TOKEN": "server-process-hf-secret",
        "CEG_WM_ROOT_KEY": "server-process-root-secret",
        "CEG_WM_INSPYRENET_CHECKPOINT_PATH": (
            "/content/drive/private/checkpoint/ckpt_base.pth"
        ),
    }
    with pytest.raises(server.SalientLocalLfWorkerProcessError) as captured:
        server._execute_worker(
            repository=repository,
            expected_revision="f" * 40,
            persistent=persistent,
            cache=cache,
            run_id=RUN_ID,
            session_id=f"worker_process_{process_case}",
            package_sha256=package_sha,
            environment=environment,
        )
    error = captured.value
    assert error.return_code == expected_return_code
    code, receipt = server._write_startup_failure_evidence(
        error=error,
        repository=repository,
        persistent=persistent,
        expected_revision="f" * 40,
        run_id=RUN_ID,
        session_id=f"worker_process_{process_case}",
        operation_identity="salient_local_lf_worker_execution",
        completed_steps=(
            "repository_identity_verified",
            "required_git_authority_revisions_resolved",
            "required_git_authority_objects_hydrated",
            "protocol_authority_loaded",
            "execution_inputs_verified",
            "resource_preflight_completed",
            "dependency_lock_verified",
            "model_asset_prepared",
            "execution_package_verified",
            "packaged_protocol_verified",
        ),
        package_path=package,
        package_sha256=package_sha,
        failure_stage="worker_process",
        return_code=error.return_code,
        artifact_kind="salient_local_lf_mask_write_validation_failure",
        commit_authority_status="unavailable",
    )
    assert code == 3
    assert receipt["failure_class"] == expected_failure_class
    assert receipt["failure_stage"] == "worker_process"
    assert receipt["committed_unit_count"] is None
    assert receipt["session_committed_unit_count"] is None
    assert receipt["commit_authority_status"] == "unavailable"
    assert receipt["salient_local_lf_mask_write_aggregate"] is None
    assert receipt["scientific_claims_supported"] is False
    diagnostic_zip = persistent / str(receipt["diagnostic_zip_relative_path"])
    receipt_path = persistent / str(receipt["receipt_relative_path"])
    with ZipFile(diagnostic_zip) as archive:
        diagnostic = json.loads(archive.read("diagnostic.json"))
        protected = diagnostic_zip.read_bytes() + receipt_path.read_bytes()
    assert diagnostic["return_code"] == expected_return_code
    assert diagnostic["commit_authority_status"] == "unavailable"
    assert diagnostic["worker_signal_number"] == (
        15 if process_case == "signal_termination" else None
    )
    assert diagnostic["sanitized_stdout"].startswith("redacted_worker_stream:")
    assert diagnostic["sanitized_stderr"].startswith("redacted_worker_stream:")
    assert len(diagnostic["sanitized_stdout"].encode("utf-8")) <= 4096
    assert len(diagnostic["sanitized_stderr"].encode("utf-8")) <= 4096
    for forbidden in (
        "server-process-hf-secret",
        "server-process-root-secret",
        "/content/drive",
        "secret-token",
        " prompt ",
        " key ",
        " tensor",
    ):
        assert forbidden.encode("utf-8") not in protected


def test_server_session_converts_missing_worker_payload_to_downloadable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    salient_local_lf_exact_package: Path,
) -> None:
    protocol = load_salient_local_lf_mask_write_validation_protocol(
        PROTOCOL_PATH, repository_root=ROOT
    )
    persistent = tmp_path / "persistent"
    cache = tmp_path / "cache"
    persistent.mkdir()
    cache.mkdir()
    monkeypatch.setattr(server, "_verify_repository", lambda *_args: None)
    monkeypatch.setattr(server, "resolve_required_git_authority_revisions", lambda **_kwargs: (EXECUTION_REVISION,))
    monkeypatch.setattr(server, "hydrate_required_git_authority_revisions", lambda *_args: (EXECUTION_REVISION,))
    monkeypatch.setattr(server, "_probe_resources", lambda **_kwargs: {"gpu": "bounded-test-gpu"})
    monkeypatch.setattr(server, "_verify_locked_dependencies", lambda _repository: "d" * 64)
    monkeypatch.setattr(server, "_download_configured_model", lambda **_kwargs: None)
    monkeypatch.setattr(server, "_extract_verified_execution_package", lambda _package, _destination: ROOT)
    monkeypatch.setattr(server, "verify_extracted_salient_local_lf_mask_write_validation_package", lambda *_args: None)
    monkeypatch.setattr(
        server,
        "build_salient_local_lf_mask_write_validation_package",
        lambda _root, destination, _revision: (
            destination.write_bytes(salient_local_lf_exact_package.read_bytes())
            and {"package_sha256": EXPECTED_PACKAGE_SHA256}
        ),
    )
    process_error = server.SalientLocalLfWorkerProcessError(
        "worker_result_missing",
        return_code=1,
        stdout="secret stdout /content/drive/private prompt key tensor",
        stderr="secret stderr /content/drive/private prompt key tensor",
    )
    monkeypatch.setattr(
        server,
        "_execute_worker",
        lambda **_kwargs: (_ for _ in ()).throw(process_error),
    )
    code, receipt = server.execute_salient_local_lf_mask_write_validation_server_session(
        repository_root=ROOT,
        expected_revision=EXECUTION_REVISION,
        persistent_root=persistent,
        cache_root=cache,
        run_id=protocol.run_id,
        session_id="missing_worker_payload_receipt",
        environment={
            "HF_TOKEN": "missing-payload-hf-secret",
            "CEG_WM_ROOT_KEY": "missing-payload-root-secret",
            "CEG_WM_INSPYRENET_CHECKPOINT_PATH": "/content/drive/private/ckpt.pth",
        },
        install_dependencies=False,
    )
    assert code == 3
    assert receipt["failure_stage"] == "worker_process"
    assert receipt["failure_class"] == "integrity_blocked"
    assert receipt["committed_unit_count"] is None
    assert receipt["session_committed_unit_count"] is None
    assert receipt["commit_authority_status"] == "unavailable"
    assert receipt["salient_local_lf_mask_write_aggregate"] is None
    assert receipt["scientific_claims_supported"] is False
    assert (persistent / str(receipt["diagnostic_zip_relative_path"])).is_file()
    assert (persistent / str(receipt["receipt_relative_path"])).is_file()


def test_remote_authority_fetch_failure_exports_zero_unit_startup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("local Git authority objects unavailable")
    execution_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    required = resolve_required_git_authority_revisions(
        execution_revision=execution_revision,
        config_payload=PROTOCOL_PATH.read_bytes(),
    )
    remote = tmp_path / "exact-authority-remote.git"
    subprocess.run(("git", "init", "--bare", str(remote)), check=True, capture_output=True)
    subprocess.run(
        (
            "git",
            "push",
            f"file://{remote}",
            f"{execution_revision}:refs/heads/execution",
            *(f"{revision}:refs/heads/authority-{index}"
              for index, revision in enumerate(required[1:], start=1)),
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    shallow = tmp_path / "shallow-repository"
    subprocess.run(
        (
            "git",
            "clone",
            "--no-local",
            "--no-hardlinks",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            "execution",
            f"file://{remote}",
            str(shallow),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    observed_execution_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=shallow,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert observed_execution_revision == execution_revision
    unavailable_origin = tmp_path / "unavailable-origin.git"
    subprocess.run(
        ("git", "remote", "set-url", "origin", f"file://{unavailable_origin}"),
        cwd=shallow,
        check=True,
    )
    forbidden_calls = []

    def reject_later_stage(*_args: object, **_kwargs: object) -> None:
        forbidden_calls.append(True)
        raise AssertionError("startup failure advanced past Git authority hydration")

    monkeypatch.setattr(server, "_probe_resources", reject_later_stage)
    monkeypatch.setattr(server, "_install_dependencies", reject_later_stage)
    monkeypatch.setattr(server, "_verify_locked_dependencies", reject_later_stage)
    monkeypatch.setattr(server, "_download_configured_model", reject_later_stage)
    monkeypatch.setattr(server, "_execute_worker", reject_later_stage)
    persistent = tmp_path / "persistent"
    cache = tmp_path / "cache"
    persistent.mkdir()
    cache.mkdir()
    secret_environment = {
        "HF_TOKEN": "startup-hf-secret",
        "CEG_WM_ROOT_KEY": "startup-root-secret",
        "CEG_WM_INSPYRENET_CHECKPOINT_PATH": (
            "/content/drive/private/checkpoint/ckpt_base.pth"
        ),
    }
    code, receipt = server.execute_salient_local_lf_mask_write_validation_server_session(
        repository_root=shallow,
        expected_revision=execution_revision,
        persistent_root=persistent,
        cache_root=cache,
        run_id=RUN_ID,
        session_id="salient_local_lf_remote_authority_failure",
        environment=secret_environment,
        install_dependencies=False,
    )
    assert forbidden_calls == []
    assert code == 3
    assert receipt["failure_class"] == "identity_blocked"
    assert receipt["failure_operation_identity"] == (
        "salient_local_lf_required_git_authority_hydration"
    )
    assert receipt["execution_package_available"] is False
    assert receipt["execution_package_relative_path"] is None
    assert receipt["execution_package_sha256"] is None
    assert receipt["committed_unit_count"] == 0
    assert receipt["session_committed_unit_count"] == 0
    assert receipt["salient_local_lf_mask_write_aggregate"] is None
    assert receipt["scientific_claims_supported"] is False
    assert receipt["completed_steps"] == (
        "repository_identity_verified",
        "required_git_authority_revisions_resolved",
    )
    assert "required_git_authority_objects_hydrated" in receipt["not_executed_steps"]
    assert "worker_execution" in receipt["not_executed_steps"]
    assert "operational_units" in receipt["not_executed_steps"]
    assert "scientific_units" in receipt["not_executed_steps"]

    receipt_path = persistent / str(receipt["receipt_relative_path"])
    diagnostic_zip = persistent / str(receipt["diagnostic_zip_relative_path"])
    assert receipt_path.is_file()
    assert diagnostic_zip.is_file()
    persisted = receipt_path.read_bytes()
    with ZipFile(diagnostic_zip) as archive:
        assert set(archive.namelist()) == {
            "diagnostic.json",
            "execution_receipt.json",
            "SHA256SUMS",
        }
        diagnostic = json.loads(archive.read("diagnostic.json"))
        artifact_receipt = json.loads(archive.read("execution_receipt.json"))
        assert archive.read("SHA256SUMS")
    assert diagnostic["failure_class"] == "identity_blocked"
    assert diagnostic["return_code"] == 3
    assert len(diagnostic["failure_message_redacted"].encode("utf-8")) <= 512
    assert len(diagnostic["package_relative_frames"]) <= 8
    assert artifact_receipt["committed_unit_count"] == 0
    assert artifact_receipt["scientific_claims_supported"] is False
    protected = persisted + diagnostic_zip.read_bytes()
    for secret in (*secret_environment.values(), str(tmp_path), "/content/drive"):
        assert secret.encode("utf-8") not in protected


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
    assert "receipt.get('execution_package_available', True)" in code_source
    assert "receipt['diagnostic_zip_relative_path']" in code_source
    assert "salient_local_lf_mask_write_validation_startup_failure" in code_source
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
