"""Delivery boundary tests for the content-routing directional diagnosis."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from experiments.protocol.content_routing_directional_diagnosis import (
    CLAIM_BOUNDARY,
    canonical_digest,
    load_content_routing_directional_protocol,
)
from scripts.experiment_execution import (
    content_routing_directional_diagnosis_entrypoint as entrypoint_module,
    content_routing_directional_diagnosis_server as server_module,
)
from scripts.experiment_execution.content_routing_directional_diagnosis_server import (
    ContentRoutingDirectionalServerError,
    execute_content_routing_directional_diagnosis_server_session,
)
from scripts.experiment_execution.content_routing_directional_diagnosis_entrypoint import (
    ContentRoutingDirectionalStartupError,
)
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/content_routing_directional_diagnosis.json"
SERVER = ROOT / "scripts/experiment_execution/content_routing_directional_diagnosis_server.py"
NOTEBOOK = ROOT / "notebooks/colab/content_routing_directional_diagnosis.ipynb"
RUN_ID = "ceg_wm_content_routing_backend_binding_correction_diagnosis"
EXECUTION_REVISION = "fc6f8ffff8cae8098b38e5c6ee5cb207923df1d9"
PAUSED_RUN_ID = "ceg_wm_content_routing_registered_key_correction_diagnosis"
REQUIRED_PACKAGE_MEMBERS = {
    "configs/experiments/content_routing_directional_diagnosis.json",
    "configs/experiments/content_routing_reference_fit_manifest.json",
    "configs/experiments/content_routing_directional_diagnosis_manifest.json",
    "experiments/metrics/content_routing_directional_diagnosis.py",
    "experiments/protocol/content_routing_directional_diagnosis.py",
    "experiments/runners/content_routing_directional_diagnosis.py",
    "scripts/experiment_execution/content_routing_directional_diagnosis_entrypoint.py",
    "scripts/experiment_execution/content_routing_directional_diagnosis_server.py",
}


@pytest.mark.quick
def test_content_routing_server_help_imports_from_isolated_cwd(tmp_path: Path) -> None:
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


def _patch_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, str]:
    protocol, reference, probes = load_content_routing_directional_protocol(
        CONFIG,
        repository_root=ROOT,
    )
    persistent = tmp_path / "persistent"
    cache = tmp_path / "cache"
    persistent.mkdir()
    cache.mkdir()
    package = persistent / "execution_package.zip"
    artifact = persistent / "result.zip"
    with ZipFile(package, "x") as archive:
        archive.writestr("package_identity.txt", "routing diagnosis")
    with ZipFile(artifact, "x") as archive:
        archive.writestr("directional_aggregate.json", "{}")
    secret = "must-not-enter-routing-receipt"
    worker = {
        "artifact_kind": "content_routing_directional_diagnosis_result",
        "result_zip": str(artifact),
        "protocol_digest": protocol.digest(),
        "reference_manifest_digest": canonical_digest(asdict(reference)),
        "probe_manifest_digest": canonical_digest(asdict(probes)),
        "input_manifest_digest": "a" * 64,
        "candidate_config_digest": "b" * 64,
        "unit_roster_digest": protocol.unit_roster_digest,
        "package_sha256": "c" * 64,
        "committed_unit_count": 42,
        "session_committed_unit_count": 42,
        "termination_reason": "frozen_roster_complete",
        "content_routing_directional_aggregate": {
            "formal_scientific_claims_supported": False,
        },
        "formal_tau_created": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    monkeypatch.setattr(server_module, "_verify_repository", lambda *_args: None)
    monkeypatch.setattr(
        server_module,
        "_probe_resources",
        lambda **_kwargs: {"gpu_model": "test"},
    )
    monkeypatch.setattr(server_module, "_download_configured_model", lambda **_kwargs: None)
    monkeypatch.setattr(server_module, "_build_or_verify_package", lambda *_args: package)

    def execute_worker(**kwargs):
        assert kwargs["environment"]["HF_TOKEN"] == secret
        assert kwargs["environment"]["CEG_WM_ROOT_KEY"] == secret
        return 0, worker

    monkeypatch.setattr(
        server_module,
        "execute_content_routing_directional_diagnosis_session",
        execute_worker,
    )
    return persistent, secret


@pytest.mark.quick
@pytest.mark.unit
def test_content_routing_server_writes_safe_fixed_roster_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    persistent, secret = _patch_server(monkeypatch, tmp_path)
    exit_code, receipt = execute_content_routing_directional_diagnosis_server_session(
        repository_root=ROOT,
        expected_revision="4" * 40,
        persistent_root=persistent,
        cache_root=tmp_path / "cache",
        run_id=RUN_ID,
        session_id="routing_delivery_session",
        environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
        install_dependencies=False,
    )
    assert exit_code == 0
    assert receipt["operational_unit_count"] == 2
    assert receipt["reference_fit_cluster_count"] == 32
    assert receipt["directional_probe_cluster_count"] == 8
    assert receipt["total_unit_count"] == 42
    assert receipt["maximum_attempts_per_unit"] == 1
    assert receipt["scientific_claims_supported"] is False
    assert receipt["formal_tau_created"] is False
    assert receipt["fpr_estimated"] is False
    assert receipt["candidate_promoted"] is False
    assert receipt["claim_boundary"] == CLAIM_BOUNDARY
    assert "fixed_half_mixing_only" in receipt["claim_boundary"]
    assert "no_alpha_generalization" in receipt["claim_boundary"]
    receipt_bytes = Path(receipt["receipt_path"]).read_bytes()
    assert secret.encode("utf-8") not in receipt_bytes


@pytest.mark.unit
def test_content_routing_entrypoint_preserves_fixed_execution_boundaries() -> None:
    source = inspect.getsource(
        __import__(
            "scripts.experiment_execution.content_routing_directional_diagnosis_entrypoint",
            fromlist=["execute_content_routing_directional_diagnosis_session"],
        )
    )
    assert "cursor.routing_reference_records" in source
    assert "cursor.terminal_routing_reference_records" in source
    assert "_commit_dependency_blocked_probe_records" in source
    assert "verified_terminal_scientific_evidence" in source
    assert "aggregate_content_routing_directional_diagnosis" in source
    startup_source = inspect.getsource(entrypoint_module._initialize_routing_resources)
    for required_startup_boundary in (
        "Sd35PipelineBackend(",
        "Sd35RuntimeAdapter(",
        'runtime.initialize("cuda")',
        "DevelopmentSemanticObservationProducer(",
        "raise ContentRoutingDirectionalStartupError",
    ):
        assert required_startup_boundary in startup_source
    for forbidden_startup_boundary in (
        "load_content_routing_directional_protocol",
        "_registered_experiment_root",
        "ContentRoutingDirectionalDiagnosisRunner",
        "_build_or_verify_package",
        "DevelopmentPersistentStore",
        "create_session_intent",
    ):
        assert forbidden_startup_boundary not in startup_source
    assert "attempt_index" not in source or "create_session_intent" in source
    for forbidden in (
        "detect_lf(",
        "candidate_worth_further_selection",
        "mechanism_signal_observed",
        "formal_threshold",
    ):
        assert forbidden not in source


@pytest.mark.unit
def test_content_routing_exact_package_contains_execution_closure(tmp_path: Path) -> None:
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
        f"print(_build_or_verify_package(Path('.').resolve(), "
        f"Path({str(package_root)!r}), {EXECUTION_REVISION!r}))"
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
        assert archive.testzip() is None
    assert REQUIRED_PACKAGE_MEMBERS <= names
    extracted = tmp_path / "extracted"
    with ZipFile(package) as archive:
        archive.extractall(extracted)
    isolated_import = (
        "from pathlib import Path; import sys; "
        f"sys.path.insert(0, {str(extracted)!r}); "
        "import experiments.runners.content_routing_directional_diagnosis; "
        "import scripts.experiment_execution.content_routing_directional_diagnosis_server"
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-c", isolated_import),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    isolated_help = subprocess.run(
        (
            sys.executable,
            "-I",
            str(
                extracted
                / "scripts/experiment_execution/content_routing_directional_diagnosis_server.py"
            ),
            "--help",
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert isolated_help.returncode == 0, isolated_help.stderr


@pytest.mark.unit
def test_content_routing_server_source_has_no_secret_or_scientific_promotion() -> None:
    source = SERVER.read_text("utf-8")
    assert '"scientific_claims_supported": False' in source
    assert '"formal_tau_created": False' in source
    assert '"fpr_estimated": False' in source
    assert '"candidate_promoted": False' in source
    assert "traceback" not in source
    assert "repr(" not in source
    assert "raw_secret" not in source


def _notebook_source() -> tuple[dict[str, object], str, str]:
    document = json.loads(NOTEBOOK.read_text("utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )
    return document, source, code_source


def _constant(source: str, name: str):
    tree = ast.parse(source)
    matches = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
        and isinstance(node.value, ast.Constant)
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.unit
def test_content_routing_notebook_is_thin_exact_and_output_free() -> None:
    document, source, code_source = _notebook_source()
    code_cells = tuple(
        cell for cell in document["cells"] if cell["cell_type"] == "code"
    )
    assert document["metadata"]["accelerator"] == "GPU"
    assert len(code_cells) == 5
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell.get("outputs", []) == [] for cell in document["cells"])
    assert _constant(code_source, "EXECUTION_REVISION") == EXECUTION_REVISION
    assert _constant(code_source, "RUN_ID") == RUN_ID
    for required in (
        "drive.mount('/content/drive')",
        "userdata.get('HF_TOKEN')",
        "userdata.get('CEG_WM_ROOT_KEY')",
        "checkout', '--detach', 'FETCH_HEAD'",
        "content_routing_directional_diagnosis_server.py",
        "execution_receipt.json",
        "SHA256SUMS",
        "content_routing_directional_aggregate",
        "reference_fit_cluster_count",
        "directional_probe_cluster_count",
        "frozen_roster_complete",
    ):
        assert required in code_source
    assert "two non-scientific operational units" in source
    assert "thirty-two routing-reference fit units" in source
    assert "eight paired routed-versus-uniform scientific probes" in source
    assert "a = 0.50" in source
    assert "causal routing control" in source
    assert "not alpha selection" in source
    assert "does not generalize across alpha values" in source
    assert "fixed-half routing directional validation" in source
    assert PAUSED_RUN_ID not in code_source
    export_order = tuple(
        code_source.index(fragment)
        for fragment in (
            "assert file_sha256(artifact_source) == receipt['artifact_sha256']",
            "artifact_export = copy_to_drive_export(",
            "receipt_export = copy_to_drive_export(",
            "checksums_path = EXPORT_ROOT / 'SHA256SUMS'",
            "print(json.dumps(summary, indent=2, sort_keys=True))",
            "if server_exit_code != 0:",
        )
    )
    assert export_order == tuple(sorted(export_order))
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "ContentRoutingDirectionalDiagnosisRunner(",
        "DevelopmentScientificRecord(",
        "aggregate_content_routing_directional_diagnosis(",
        "create_content_routing_directional_observation(",
        "FormalHfContentDetectionOperation(",
        "--skip-dependency-install",
    ):
        assert forbidden not in code_source


@pytest.mark.unit
def test_content_routing_server_exports_failure_receipt_without_scientific_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    persistent, secret = _patch_server(monkeypatch, tmp_path)
    protocol, reference, probes = load_content_routing_directional_protocol(
        CONFIG,
        repository_root=ROOT,
    )
    artifact = persistent / "result.zip"
    worker = {
        "artifact_kind": "content_routing_directional_diagnosis_failure",
        "diagnostic_zip": str(artifact),
        "protocol_digest": protocol.digest(),
        "reference_manifest_digest": canonical_digest(asdict(reference)),
        "probe_manifest_digest": canonical_digest(asdict(probes)),
        "input_manifest_digest": "a" * 64,
        "candidate_config_digest": "b" * 64,
        "unit_roster_digest": protocol.unit_roster_digest,
        "package_sha256": "c" * 64,
        "committed_unit_count": 0,
        "session_committed_unit_count": 0,
        "termination_reason": "worker_execution_failure",
        "content_routing_directional_aggregate": None,
        "formal_tau_created": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    monkeypatch.setattr(
        server_module,
        "execute_content_routing_directional_diagnosis_session",
        lambda **_kwargs: (3, worker),
    )
    exit_code, receipt = execute_content_routing_directional_diagnosis_server_session(
        repository_root=ROOT,
        expected_revision="4" * 40,
        persistent_root=persistent,
        cache_root=tmp_path / "cache",
        run_id=RUN_ID,
        session_id="routing_failure_export_session",
        environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
        install_dependencies=False,
    )
    assert exit_code == 3
    assert receipt["artifact_kind"] == worker["artifact_kind"]
    assert receipt["content_routing_directional_aggregate"] is None
    assert receipt["scientific_claims_supported"] is False
    assert receipt["claim_boundary"] == CLAIM_BOUNDARY
    assert receipt["termination_reason"] == "worker_execution_failure"
    assert Path(receipt["receipt_path"]).is_file()


@pytest.mark.unit
def test_content_routing_server_exports_safe_fresh_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    persistent, secret = _patch_server(monkeypatch, tmp_path)

    def fail_before_store(**_kwargs):
        raise ContentRoutingDirectionalStartupError(
            failure_type="main.shared.key_schedule.KeyScheduleError",
            failure_class="implementation_failure",
        )

    monkeypatch.setattr(
        server_module,
        "execute_content_routing_directional_diagnosis_session",
        fail_before_store,
    )
    exit_code, receipt = execute_content_routing_directional_diagnosis_server_session(
        repository_root=ROOT,
        expected_revision="5" * 40,
        persistent_root=persistent,
        cache_root=tmp_path / "cache",
        run_id=RUN_ID,
        session_id="routing_startup_failure_session",
        environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
        install_dependencies=False,
    )
    assert exit_code == 3
    assert receipt["committed_unit_count"] == 0
    assert receipt["session_committed_unit_count"] == 0
    assert receipt["content_routing_directional_aggregate"] is None
    assert receipt["failure_class"] == "implementation_failure"
    assert receipt["failure_type"] == "main.shared.key_schedule.KeyScheduleError"
    assert receipt["termination_reason"] == "worker_startup_failure"
    run_root = persistent / RUN_ID
    for forbidden_evidence in ("intents", "bundles", "markers", "module_outcomes"):
        assert not (run_root / forbidden_evidence).exists()
    artifact = Path(receipt["artifact_path"])
    with ZipFile(artifact) as archive:
        assert set(archive.namelist()) == {
            "committed_unit_ids.json",
            "diagnostic.json",
        }
        assert json.loads(archive.read("committed_unit_ids.json")) == []
        diagnostic = json.loads(archive.read("diagnostic.json"))
    assert diagnostic == {
        "failure_class": "implementation_failure",
        "failure_type": "main.shared.key_schedule.KeyScheduleError",
        "scientific_claims_supported": False,
        "stage": "content_routing_directional_diagnosis_startup",
    }
    evidence = artifact.read_bytes() + Path(receipt["receipt_path"]).read_bytes()
    assert secret.encode("utf-8") not in evidence
    for forbidden_text in (
        b"message",
        b"traceback",
        b"prompt",
        b"root_key",
        b"tensor",
        b"repr",
    ):
        assert forbidden_text not in evidence


@pytest.mark.unit
def test_content_routing_startup_diagnostic_rejects_subclass_and_existing_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    persistent, secret = _patch_server(monkeypatch, tmp_path)

    class DerivedStartupError(ContentRoutingDirectionalStartupError):
        pass

    def fail_with_subclass(**_kwargs):
        raise DerivedStartupError(
            failure_type="builtins.RuntimeError",
            failure_class="implementation_failure",
        )

    monkeypatch.setattr(
        server_module,
        "execute_content_routing_directional_diagnosis_session",
        fail_with_subclass,
    )
    with pytest.raises(DerivedStartupError):
        execute_content_routing_directional_diagnosis_server_session(
            repository_root=ROOT,
            expected_revision="6" * 40,
            persistent_root=persistent,
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="routing_startup_subclass_session",
            environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
            install_dependencies=False,
        )
    assert not (persistent / RUN_ID).exists()

    (persistent / RUN_ID / "intents").mkdir(parents=True)

    def fail_exact(**_kwargs):
        raise ContentRoutingDirectionalStartupError(
            failure_type="builtins.RuntimeError",
            failure_class="implementation_failure",
        )

    monkeypatch.setattr(
        server_module,
        "execute_content_routing_directional_diagnosis_session",
        fail_exact,
    )
    with pytest.raises(
        ContentRoutingDirectionalServerError,
        match="fresh run root",
    ):
        execute_content_routing_directional_diagnosis_server_session(
            repository_root=ROOT,
            expected_revision="6" * 40,
            persistent_root=persistent,
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="routing_existing_run_session",
            environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
            install_dependencies=False,
        )


@pytest.mark.unit
def test_content_routing_server_does_not_capture_plain_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    persistent, secret = _patch_server(monkeypatch, tmp_path)
    failure = RuntimeError("ordinary worker failure")

    def fail_plain(**_kwargs):
        raise failure

    monkeypatch.setattr(
        server_module,
        "execute_content_routing_directional_diagnosis_session",
        fail_plain,
    )
    with pytest.raises(RuntimeError) as caught:
        execute_content_routing_directional_diagnosis_server_session(
            repository_root=ROOT,
            expected_revision="7" * 40,
            persistent_root=persistent,
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="routing_plain_runtime_failure_session",
            environment={"HF_TOKEN": secret, "CEG_WM_ROOT_KEY": secret},
            install_dependencies=False,
        )
    assert caught.value is failure
    assert not (persistent / RUN_ID).exists()
