"""End-to-end CPU wiring check for the A3b execution package."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from scripts.experiment_execution import experiment_execution_bootstrap
from scripts.experiment_execution.build_experiment_execution_package import (
    build_experiment_execution_package,
)
from scripts.experiment_execution.experiment_execution_entrypoint import (
    prepare_synthetic_wiring,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOTS = (
    "main",
    "runtime",
    "experiments",
    "configs",
    "infrastructure",
)
PACKAGE_EXTRAS = (
    "pyproject.toml",
    "templates/release_readmes/experiment_execution_package.md",
    "scripts/experiment_execution/__init__.py",
    "scripts/experiment_execution/experiment_execution_entrypoint.py",
    "tests/integration/__init__.py",
    "tests/integration/test_packaged_experiment_execution.py",
    "tests/smoke/test_packaged_experiment_execution.py",
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _copy_package_source(destination: Path) -> str:
    for relative in PACKAGE_ROOTS:
        shutil.copytree(
            ROOT / relative,
            destination / relative,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
            ),
        )
    for relative in PACKAGE_EXTRAS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    _git(destination, "init")
    _git(destination, "config", "user.email", "test@example.invalid")
    _git(destination, "config", "user.name", "Test")
    _git(destination, "add", ".")
    _git(destination, "commit", "-m", "integration package fixture")
    return _git(destination, "rev-parse", "HEAD")


@pytest.mark.integration
def test_verified_package_executes_real_a3a_synthetic_wiring(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    revision = _copy_package_source(repository)
    preparation = prepare_synthetic_wiring(
        package_root=repository,
        records_root=tmp_path / "preparation_records",
        workspace_root=tmp_path / "preparation_workspace",
        committed_revision=revision,
        run_id="integration-preparation",
    )
    package = tmp_path / "experiment_execution.zip"
    build_result = build_experiment_execution_package(
        root=repository,
        output_zip=package,
        committed_revision=revision,
        candidate_config_digest=preparation.candidate_config_digest,
        execution_config_digest=preparation.execution_config_digest,
        input_manifest_digest=preparation.input_manifest_digest,
    )

    exit_code, outcome = experiment_execution_bootstrap.run_bootstrap(
        package_zip=package,
        expected_archive_sha256=build_result["archive_sha256"],
        expected_bootstrap_identity=(
            experiment_execution_bootstrap.BOOTSTRAP_IDENTITY
        ),
        expected_bootstrap_schema_version=(
            experiment_execution_bootstrap.BOOTSTRAP_SCHEMA_VERSION
        ),
        expected_bootstrap_sha256=sha256(
            Path(experiment_execution_bootstrap.__file__).read_bytes()
        ).hexdigest(),
        expected_revision=revision,
        expected_candidate_config_digest=(
            preparation.candidate_config_digest
        ),
        expected_execution_config_digest=(
            preparation.execution_config_digest
        ),
        expected_input_manifest_digest=preparation.input_manifest_digest,
        ephemeral_root=(tmp_path / "ephemeral").resolve(),
        persistent_root=(tmp_path / "persistent").resolve(),
        run_id="integration-package",
    )

    assert exit_code == 0
    assert outcome["artifact_kind"] == "experiment_execution_result"
    assert outcome["scientific_claims_supported"] is False
    result_zip = Path(outcome["result_zip"])
    assert result_zip.is_file()
    with zipfile.ZipFile(package) as archive:
        package_names = set(archive.namelist())
    assert (
        "scripts/experiment_execution/experiment_execution_entrypoint.py"
        in package_names
    )
    assert "experiments/runners/formal_operations.py" in package_names
    assert {
        name
        for name in package_names
        if name.startswith(("tests/integration/", "tests/smoke/"))
    } == {
        "tests/integration/__init__.py",
        "tests/integration/test_packaged_experiment_execution.py",
        "tests/smoke/test_packaged_experiment_execution.py",
    }
    assert not any(
        name.startswith(
            (
                ".agents/",
                ".codex/",
                "governance/",
                "notebooks/",
                "outputs/",
                "paper_artifacts/",
            )
        )
        for name in package_names
    )
    assert (
        "scripts/experiment_execution/"
        "experiment_execution_bootstrap.py"
        not in package_names
    )
    with zipfile.ZipFile(result_zip) as archive:
        result_names = set(archive.namelist())
        summary = json.loads(
            archive.read("execution_summary.json")
        )
    assert summary["execution_scope"] == "cpu_synthetic_wiring_only"
    assert summary["scientific_claims_supported"] is False
    assert summary["gpu_executed"] is False
    assert summary["held_out_evaluation_accessed"] is False
    assert summary["success_count"] == 1
    assert summary["record_count"] == 1
    assert summary["record_collection_relative_path"] in result_names

    extracted_package = (
        tmp_path
        / "ephemeral"
        / "experiment_bootstrap_integration-package"
        / "package"
    )
    contained = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "-m",
            "integration or smoke",
            "tests/integration/test_packaged_experiment_execution.py",
            "tests/smoke/test_packaged_experiment_execution.py",
        ),
        cwd=extracted_package,
        check=False,
        capture_output=True,
        text=True,
    )
    assert contained.returncode == 0, (
        contained.stdout + "\n" + contained.stderr
    )
    assert "2 passed" in contained.stdout
