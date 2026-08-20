"""Server delivery checks for LF whitening fit and score screening."""

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

from experiments.protocol.lf_whitened_score_screening import (
    RUN_ID,
    load_lf_whitened_score_screening_protocol,
)
from scripts.experiment_execution import lf_whitened_score_screening_server as server


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/colab/lf_whitened_score_screening.ipynb"
SERVER_RELATIVE = Path(
    "scripts/experiment_execution/lf_whitened_score_screening_server.py"
)
SERVER = ROOT / SERVER_RELATIVE
PROTOCOL = ROOT / "configs/experiments/lf_whitened_score_screening.json"
EXECUTION_REVISION = "a78c47184cf83ad351bb4442ebd31c218726de25"
REQUIRED_PACKAGE_MEMBERS = {
    "configs/experiments/lf_whitened_score_screening.json",
    "configs/experiments/lf_whitened_score_screening_manifest.json",
    "configs/experiments/lf_whitening_null_fit_manifest.json",
    "experiments/metrics/lf_whitened_score_screening.py",
    "experiments/protocol/lf_whitened_score_screening.py",
    "experiments/runners/lf_whitened_score_screening.py",
    "scripts/experiment_execution/lf_whitened_score_screening_entrypoint.py",
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
def test_lf_whitened_screening_server_help_imports_from_isolated_cwd(
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
def test_lf_whitened_screening_server_builds_exact_package_and_safe_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("detached research copy lacks exact Git checkout capability")
    checkout = tmp_path / "exact_checkout"
    subprocess.run(
        ("git", "clone", "--quiet", str(ROOT), str(checkout)),
        check=True,
    )
    revision = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    protocol, null_fit_manifest, screening_manifest = (
        load_lf_whitened_score_screening_protocol(
            checkout / PROTOCOL.relative_to(ROOT),
            repository_root=checkout,
        )
    )
    persistent = tmp_path / "persistent"
    cache = tmp_path / "cache"
    artifact = persistent / RUN_ID / "session_results" / "session.zip"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(artifact, "x") as archive:
        archive.writestr("screening_decision.json", "{}\n")

    monkeypatch.setattr(
        server,
        "_probe_resources",
        lambda **_kwargs: {
            "cuda_device_name": "Test GPU",
            "cuda_total_memory_bytes": 1,
            "free_disk_bytes": {},
        },
    )
    monkeypatch.setattr(
        server,
        "_download_configured_model",
        lambda **_kwargs: cache / "snapshot",
    )
    worker_calls: list[dict[str, object]] = []

    def execute_worker(**kwargs: object) -> tuple[int, dict[str, object]]:
        worker_calls.append(kwargs)
        return 0, {
            "artifact_kind": "lf_whitened_score_screening_result",
            "result_zip": str(artifact),
            "protocol_digest": protocol.digest(),
            "null_fit_manifest_digest": null_fit_manifest.digest(),
            "screening_manifest_digest": screening_manifest.digest(),
            "candidate_config_digest": "a" * 64,
            "unit_roster_digest": protocol.unit_roster_digest,
            "committed_unit_count": 41,
            "session_committed_unit_count": 41,
            "termination_reason": "frozen_roster_complete",
            "screening_decision": {
                "allow_request_for_lf_whitened_directional_validation": False
            },
            "formal_tau_created": False,
            "candidate_promoted": False,
            "scientific_claims_supported": False,
        }

    monkeypatch.setattr(
        server,
        "execute_lf_whitened_score_screening_session",
        execute_worker,
    )
    exit_code, receipt = (
        server.execute_lf_whitened_score_screening_server_session(
            repository_root=checkout,
            expected_revision=revision,
            persistent_root=persistent,
            cache_root=cache,
            run_id=RUN_ID,
            session_id="lf_whitened_screening_server_session",
            environment={
                "HF_TOKEN": "private-hf-token",
                "CEG_WM_ROOT_KEY": "private-root-key",
            },
            install_dependencies=False,
        )
    )

    assert exit_code == 0
    assert len(worker_calls) == 1
    package = Path(str(receipt["execution_package_path"]))
    package_sha256 = sha256(package.read_bytes()).hexdigest()
    assert worker_calls[0]["execution_package_sha256"] == package_sha256
    assert receipt["execution_package_sha256"] == package_sha256
    assert receipt["committed_revision"] == revision
    assert receipt["operational_unit_count"] == 1
    assert receipt["scientific_unit_count"] == 40
    assert receipt["development_claim_boundary"] == protocol.claim_boundary
    assert receipt["formal_tau_created"] is False
    assert receipt["candidate_promoted"] is False
    receipt_text = Path(str(receipt["receipt_path"])).read_text("utf-8")
    assert "private-hf-token" not in receipt_text
    assert "private-root-key" not in receipt_text
    with ZipFile(package) as archive:
        names = set(archive.namelist())
        assert SERVER_RELATIVE.as_posix() in names
        assert "experiments/runners/lf_whitened_score_screening.py" in names
        assert "scripts/experiment_execution/lf_whitened_score_screening_entrypoint.py" in names
        assert archive.testzip() is None
        extracted = tmp_path / "extracted_package"
        archive.extractall(extracted)
    imported = subprocess.run(
        (sys.executable, "-I", str(extracted / SERVER_RELATIVE), "--help"),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    assert json.loads(receipt_text)["execution_package_sha256"] == package_sha256


@pytest.mark.quick
def test_lf_whitened_screening_notebook_is_thin_and_output_free() -> None:
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
    assert (
        "DRIVE_ROOT = DRIVE_MOUNT / 'MyDrive' / 'CEG-WM' / "
        "'lf_whitened_score_screening'"
    ) in code_source
    assert "PERSISTENT_ROOT = DRIVE_ROOT / 'persistent'" in code_source
    assert "CACHE_ROOT = DRIVE_ROOT / 'cache'" in code_source
    assert "EXPORT_BASE = DRIVE_ROOT / 'exports'" in code_source
    assert "execution_receipt.json" in code_source
    assert "SHA256SUMS" in code_source
    assert "lf_whitened_score_screening_result" in code_source
    assert "lf_whitened_score_screening_failure" in code_source
    assert "one non-scientific operational smoke unit" in source
    assert "thirty-two clean null-fit units" in source
    assert "eight paired raw-versus-whitened score screening units" in source
    assert "fits no threshold" in source
    assert "estimates no FPR" in source
    assert "promotes no candidate" in source
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "LfWhitenedScoreScreeningRunner(",
        "DevelopmentScientificRecord(",
        "execute_lf_whitened_score_screening_session(",
        "--skip-dependency-install",
        "qk_geometry_sync",
        "content_router",
        "hf_only_threshold_fit",
        "4096",
    ):
        assert forbidden not in code_source

    protocol, null_fit_manifest, screening_manifest = (
        load_lf_whitened_score_screening_protocol(
            PROTOCOL,
            repository_root=ROOT,
        )
    )
    assert protocol.operational_unit_count == 1
    assert protocol.null_fit_cluster_count == 32
    assert protocol.screening_cluster_count == 8
    assert protocol.maximum_total_units == 41
    assert protocol.maximum_attempts_per_unit == 2
    assert protocol.maximum_duration_seconds_per_unit == 2700
    assert len(null_fit_manifest.entries) == 32
    assert len(screening_manifest.entries) == 8


@pytest.mark.quick
def test_lf_whitened_screening_exact_checkout_builds_importable_package(
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
def test_lf_whitened_screening_readmes_preserve_producer_bound_history() -> None:
    for path in (
        ROOT / "notebooks/colab/README.md",
        ROOT / "scripts/experiment_execution/README.md",
    ):
        source = path.read_text("utf-8")
        normalized_source = " ".join(source.split())
        assert NOTEBOOK.name in source
        assert EXECUTION_REVISION in source
        assert RUN_ID in source
        assert "1 个 non-scientific operational" in normalized_source
        assert "32 个 clean null-fit" in normalized_source
        assert "8 个 paired raw-vs-whitened screening" in normalized_source
        assert "development-only" in source
        assert "threshold" in source
        assert "FPR" in source
        assert "candidate promotion" in source
        assert "lf_transmission_diagnostic.ipynb" in source
        assert "hf_only_detector_directional_validation.ipynb" in source
