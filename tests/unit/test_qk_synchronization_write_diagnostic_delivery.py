"""Delivery boundary tests for Q/K synchronization-write diagnosis."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from experiments.protocol.qk_synchronization_write_diagnostic import (
    CLAIM_BOUNDARY,
    load_qk_synchronization_write_protocol,
)
from scripts.experiment_execution import qk_synchronization_write_diagnostic_server as server_module
from scripts.experiment_execution.qk_synchronization_write_diagnostic_server import (
    QkSynchronizationWriteServerError,
    execute_qk_synchronization_write_diagnostic_server_session,
    main,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/qk_synchronization_write_diagnostic.json"
ENTRYPOINT = ROOT / "scripts/experiment_execution/qk_synchronization_write_diagnostic_entrypoint.py"
SERVER = ROOT / "scripts/experiment_execution/qk_synchronization_write_diagnostic_server.py"
SERVER_RELATIVE = SERVER.relative_to(ROOT)
NOTEBOOK = ROOT / "notebooks/colab/qk_synchronization_write_diagnostic.ipynb"
EXECUTION_REVISION = "da2e89862ecab5401a75dfac9e3e01049cb963e8"
RUN_ID = "ceg_wm_qk_differentiable_vae_decode_cause_localization"
NOTEBOOK_RUN_ID = RUN_ID
HISTORICAL_LOCALIZATION_RUN_ID = "ceg_wm_qk_runtime_failure_localization"
HISTORICAL_RUN_ID = "ceg_wm_qk_synchronization_write_public_rgb8_diagnosis"
REQUIRED_PACKAGE_MEMBERS = {
    "configs/experiments/qk_synchronization_write_diagnostic.json",
    "configs/experiments/qk_synchronization_write_diagnostic_manifest.json",
    "experiments/metrics/qk_synchronization_write_diagnostic.py",
    "experiments/protocol/qk_synchronization_write_diagnostic.py",
    "experiments/runners/qk_synchronization_write_diagnostic.py",
    "runtime/sd35_backend.py",
    "scripts/experiment_execution/qk_synchronization_write_diagnostic_entrypoint.py",
    SERVER_RELATIVE.as_posix(),
}


def _notebook_source() -> tuple[dict[str, object], str]:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return document, "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )


def _constant(source: str, name: str):
    tree = ast.parse(source)
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
        and isinstance(node.value, ast.Constant)
    ]
    assert len(matches) == 1
    return matches[0].value


@pytest.mark.quick
def test_qk_diagnosis_server_help_imports_from_isolated_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SERVER), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-revision" in completed.stdout
    assert "--persistent-root" in completed.stdout
    assert "--cache-root" in completed.stdout


@pytest.mark.quick
def test_qk_failure_localization_delivery_binds_only_operational_entrypoint() -> None:
    protocol, _manifest = load_qk_synchronization_write_protocol(
        PROTOCOL, repository_root=ROOT
    )
    source = ENTRYPOINT.read_text("utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert "execute_qk_synchronization_write_diagnostic_session" in source
    assert 'environment.get("CUDA_LAUNCH_BLOCKING") == "1"' in source
    assert "create_session_intent" in calls
    assert "commit_session_unit" in calls
    assert "_authorized_persistence_bindings" in calls
    assert "verified_terminal_scientific_evidence" not in calls
    assert "execute_scientific_unit" not in calls
    assert "create_dependency_blocked_record" not in calls
    assert "_selected_rgb8" not in calls
    assert protocol.run_id == RUN_ID
    assert protocol.authorized_operational_unit_count == 1
    assert protocol.authorized_scientific_unit_count == 0
    assert protocol.authorized_total_unit_count == 1
    assert protocol.authorized_maximum_attempts_per_unit == 1
    assert len(protocol.authorized_unit_roster) == 1
    assert protocol.operational_unit_count == 1
    assert protocol.scientific_unit_count == 28
    assert protocol.maximum_total_units == 29
    assert protocol.ratio_probe_unit_count == 12
    assert protocol.transform_probe_unit_count == 16
    assert protocol.claim_boundary == CLAIM_BOUNDARY
    for forbidden in (
        "content_router",
        "reference_image",
        "private_qk_cache",
        "synthetic_gradient",
        "precomputed_score",
    ):
        assert forbidden not in source


@pytest.mark.quick
def test_qk_diagnosis_server_receipt_boundary_is_non_scientific(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "--run-id" in output
    assert "--session-id" in output
    server_source = SERVER.read_text("utf-8")
    assert '"scientific_claims_supported": False' in server_source
    assert '"formal_tau_created": False' in server_source
    assert '"fpr_estimated": False' in server_source
    assert '"candidate_promoted": False' in server_source


def _patch_server_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    exit_code: int,
    roster_digest: str,
    cuda_launch_blocking_identity: object = "cuda_launch_blocking_enabled",
) -> tuple[Path, str]:
    protocol, manifest = load_qk_synchronization_write_protocol(
        PROTOCOL, repository_root=ROOT
    )
    persistent = tmp_path / "persistent"
    cache = tmp_path / "cache"
    persistent.mkdir()
    cache.mkdir()
    package = persistent / "execution_package.zip"
    with ZipFile(package, "x") as archive:
        archive.writestr("package_identity.txt", "qk localization")
    artifact = persistent / (
        "diagnostic.zip" if exit_code else "result.zip"
    )
    with ZipFile(artifact, "x") as archive:
        archive.writestr("diagnostic.json", "{}")
    secret = "must-not-enter-qk-localization-receipt"
    worker = {
        "artifact_kind": (
            "qk_synchronization_write_diagnostic_failure"
            if exit_code
            else "qk_synchronization_write_diagnostic_result"
        ),
        "diagnostic_zip" if exit_code else "result_zip": str(artifact),
        "protocol_digest": protocol.digest(),
        "input_manifest_digest": manifest.digest(),
        "candidate_config_digest": "a" * 64,
        "unit_roster_digest": roster_digest,
        "source_cluster_deny_list_digest": (
            protocol.source_cluster_deny_list_digest
        ),
        "package_sha256": "b" * 64,
        "committed_unit_count": 0 if exit_code else 1,
        "session_committed_unit_count": 0 if exit_code else 1,
        "termination_reason": (
            "operational_failure_localization_failed"
            if exit_code
            else "operational_failure_localization_complete"
        ),
        "qk_synchronization_diagnosis_aggregate": None,
        "formal_tau_created": False,
        "fpr_estimated": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
        "cuda_launch_blocking_identity": cuda_launch_blocking_identity,
    }
    monkeypatch.setattr(server_module, "_verify_repository", lambda *_args: None)
    monkeypatch.setattr(
        server_module,
        "_probe_resources",
        lambda **_kwargs: {"gpu_model": "test"},
    )
    monkeypatch.setattr(
        server_module,
        "_download_configured_model",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        server_module,
        "_build_or_verify_package",
        lambda *_args: package,
    )

    def execute_worker(**kwargs):
        assert kwargs["environment"]["HF_TOKEN"] == secret
        assert kwargs["environment"]["CEG_WM_ROOT_KEY"] == secret
        return exit_code, worker

    monkeypatch.setattr(
        server_module,
        "_execute_worker_process",
        execute_worker,
    )
    return persistent, secret


@pytest.mark.quick
def test_qk_worker_process_enables_launch_blocking_before_module_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    worker = {
        "cuda_launch_blocking_identity": "cuda_launch_blocking_enabled",
    }

    def run_worker(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        assert kwargs["cwd"] == ROOT
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            command,
            3,
            stdout=(
                server_module.WORKER_RESULT_PREFIX
                + json.dumps(worker, sort_keys=True)
                + "\n"
            ),
            stderr="worker-private-diagnostic-must-not-be-persisted",
        )

    monkeypatch.setattr(server_module.subprocess, "run", run_worker)
    exit_code, result = server_module._execute_worker_process(
        repository=ROOT,
        expected_revision="1" * 40,
        persistent=tmp_path / "persistent",
        cache=tmp_path / "cache",
        run_id=RUN_ID,
        session_id="qk_launch_blocking_process",
        package_sha256="2" * 64,
        environment={
            "HF_TOKEN": "test-hf-token",
            "CEG_WM_ROOT_KEY": "test-root-key",
        },
    )

    command = observed["command"]
    environment = observed["environment"]
    assert command[:3] == (
        sys.executable,
        "-m",
        "scripts.experiment_execution.qk_synchronization_write_diagnostic_entrypoint",
    )
    assert environment["CUDA_LAUNCH_BLOCKING"] == "1"
    assert exit_code == 3
    assert result == worker


@pytest.mark.quick
@pytest.mark.parametrize("exit_code", (0, 3))
def test_qk_localization_server_accepts_authorized_roster_and_writes_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exit_code: int,
) -> None:
    protocol, _manifest = load_qk_synchronization_write_protocol(
        PROTOCOL, repository_root=ROOT
    )
    persistent, secret = _patch_server_execution(
        monkeypatch,
        tmp_path,
        exit_code=exit_code,
        roster_digest=protocol.authorized_unit_roster_digest,
    )
    observed_exit_code, receipt = (
        execute_qk_synchronization_write_diagnostic_server_session(
            repository_root=ROOT,
            expected_revision="1" * 40,
            persistent_root=persistent,
            cache_root=tmp_path / "cache",
            run_id=protocol.run_id,
            session_id=f"qk_localization_{exit_code}",
            environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
            install_dependencies=False,
        )
    )

    assert observed_exit_code == exit_code
    assert receipt["unit_roster_digest"] == (
        protocol.authorized_unit_roster_digest
    )
    assert receipt["operational_unit_count"] == 1
    assert receipt["scientific_unit_count"] == 0
    assert receipt["total_unit_count"] == 1
    assert receipt["maximum_attempts_per_unit"] == 1
    assert receipt["qk_synchronization_diagnosis_aggregate"] is None
    assert receipt["scientific_claims_supported"] is False
    assert receipt["termination_reason"] != "frozen_roster_complete"
    receipt_bytes = Path(receipt["receipt_path"]).read_text("utf-8")
    assert secret not in receipt_bytes
    assert "traceback" not in receipt_bytes.lower()
    assert "exception_message" not in receipt_bytes


@pytest.mark.quick
def test_qk_localization_server_rejects_dormant_scientific_roster_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, _manifest = load_qk_synchronization_write_protocol(
        PROTOCOL, repository_root=ROOT
    )
    persistent, secret = _patch_server_execution(
        monkeypatch,
        tmp_path,
        exit_code=0,
        roster_digest=protocol.unit_roster_digest,
    )
    with pytest.raises(
        QkSynchronizationWriteServerError,
        match="worker frozen identity drifted",
    ):
        execute_qk_synchronization_write_diagnostic_server_session(
            repository_root=ROOT,
            expected_revision="1" * 40,
            persistent_root=persistent,
            cache_root=tmp_path / "cache",
            run_id=protocol.run_id,
            session_id="qk_localization_full_roster_rejected",
            environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
            install_dependencies=False,
        )
    assert not (
        persistent
        / protocol.run_id
        / "server_receipts"
        / "qk_localization_full_roster_rejected"
        / "execution_receipt.json"
    ).exists()


@pytest.mark.quick
def test_qk_localization_server_rejects_missing_launch_blocking_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, _manifest = load_qk_synchronization_write_protocol(
        PROTOCOL, repository_root=ROOT
    )
    persistent, secret = _patch_server_execution(
        monkeypatch,
        tmp_path,
        exit_code=3,
        roster_digest=protocol.authorized_unit_roster_digest,
        cuda_launch_blocking_identity=None,
    )
    with pytest.raises(
        QkSynchronizationWriteServerError,
        match="worker frozen identity drifted",
    ):
        execute_qk_synchronization_write_diagnostic_server_session(
            repository_root=ROOT,
            expected_revision="1" * 40,
            persistent_root=persistent,
            cache_root=tmp_path / "cache",
            run_id=protocol.run_id,
            session_id="qk_launch_blocking_identity_rejected",
            environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
            install_dependencies=False,
        )


@pytest.mark.quick
def test_qk_diagnosis_notebook_is_thin_exact_and_output_free() -> None:
    document, source = _notebook_source()
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )

    assert len(document["cells"]) == 6
    assert all(
        cell.get("execution_count") is None
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(cell.get("outputs", []) == [] for cell in document["cells"])
    assert _constant(code_source, "EXECUTION_REVISION") == EXECUTION_REVISION
    assert _constant(code_source, "RUN_ID") == NOTEBOOK_RUN_ID
    assert "qk_synchronization_write_diagnostic_server.py" in code_source
    assert "HF_TOKEN" in code_source and "CEG_WM_ROOT_KEY" in code_source
    assert "/content/drive" in code_source
    assert "--expected-revision" in code_source
    assert "--persistent-root" in code_source
    assert "--cache-root" in code_source
    assert "--run-id" in code_source
    assert "--session-id" in code_source
    assert "qk_synchronization_diagnosis_aggregate" in code_source
    assert "receipt['scientific_unit_count'] == 0" in code_source
    assert "receipt['total_unit_count'] == 1" in code_source
    assert "receipt['maximum_attempts_per_unit'] == 1" in code_source
    assert "receipt['qk_synchronization_diagnosis_aggregate'] is None" in code_source
    assert "twenty-eight scientific units" not in source
    for forbidden in (
        "geometry_synchronization_write(",
        "create_qk_ratio_probe_observation(",
        "DevelopmentScientificRecord(",
        "commit_session_unit(",
        "replay_synchronization_diagnosis_aggregate(",
        "fit_threshold(",
        "evaluate_qk_synchronization_write_diagnosis(",
    ):
        assert forbidden not in source


@pytest.mark.quick
def test_qk_localization_exact_checkout_builds_importable_execution_package(
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
        extracted = tmp_path / "exact_package"
        archive.extractall(extracted)
    isolated_help = subprocess.run(
        (sys.executable, "-I", str(extracted / SERVER_RELATIVE), "--help"),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert isolated_help.returncode == 0, isolated_help.stderr
    assert sha256(package.read_bytes()).hexdigest()


@pytest.mark.quick
def test_qk_diagnosis_readmes_preserve_historical_run_boundary() -> None:
    for path in (
        ROOT / "notebooks/colab/README.md",
        ROOT / "scripts/experiment_execution/README.md",
    ):
        source = path.read_text("utf-8")
        normalized_source = " ".join(source.split())
        assert NOTEBOOK.name in source
        assert EXECUTION_REVISION in source
        assert NOTEBOOK_RUN_ID in source
        assert HISTORICAL_LOCALIZATION_RUN_ID in source
        assert HISTORICAL_RUN_ID in source
        assert "records、diagnostics 与 intents 保持不可变" in normalized_source
        assert "不读取、迁移、覆盖或混入" in normalized_source
        assert "1 operational / 0 scientific / 1 total / 1 attempt" in source
