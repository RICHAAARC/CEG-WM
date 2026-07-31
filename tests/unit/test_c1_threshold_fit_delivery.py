"""CPU trust-chain tests for the C1 threshold-fit delivery path."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from scripts.experiment_execution import experiment_execution_bootstrap as bootstrap
from scripts.experiment_execution import experiment_execution_entrypoint as entrypoint
from scripts.experiment_execution.build_experiment_execution_package import (
    EXACT_FILES,
    build_experiment_execution_package,
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
    "requirements_runtime_qualification.txt",
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


def _copy_package_repository(destination: Path) -> str:
    for relative in PACKAGE_ROOTS:
        shutil.copytree(
            ROOT / relative,
            destination / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    for relative in PACKAGE_EXTRAS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    _git(destination, "init")
    _git(destination, "config", "user.email", "test@example.invalid")
    _git(destination, "config", "user.name", "Threshold Fit Test")
    _git(destination, "add", ".")
    _git(destination, "commit", "-m", "threshold fit package fixture")
    return _git(destination, "rev-parse", "HEAD")


def _authority_digests(repository: Path) -> dict[str, str]:
    specification = json.loads(
        (repository / "configs/experiments/c1_hf_reference_run.json").read_text(
            encoding="utf-8"
        )
    )
    execution = json.loads(
        (
            repository
            / "configs/experiments/c1_hf_threshold_fit_execution.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "candidate_config_digest": specification["candidate_binding"][
            "candidate_binding_digest"
        ],
        "execution_config_digest": execution["execution_config_digest"],
        "input_manifest_digest": execution["fit_manifest_digest"],
    }


@pytest.fixture(scope="module")
def threshold_package(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("threshold_package") / "repository"
    root.mkdir()
    revision = _copy_package_repository(root)
    digests = _authority_digests(root)
    package = root.parent / "threshold-fit.zip"
    build = build_experiment_execution_package(
        root=root,
        output_zip=package,
        committed_revision=revision,
    )
    sidecar = Path(build["delivery_manifest_path"])
    return {
        "build": build,
        "digests": digests,
        "package": package,
        "repository": root,
        "revision": revision,
        "sidecar": sidecar,
    }


@pytest.mark.quick
def test_builder_uses_exact_threshold_fit_allowlist_and_is_deterministic(
    threshold_package: dict[str, object],
    tmp_path: Path,
) -> None:
    second = tmp_path / "threshold-fit.zip"
    second_build = build_experiment_execution_package(
        root=threshold_package["repository"],
        output_zip=second,
        committed_revision=threshold_package["revision"],
    )
    assert second.read_bytes() == Path(threshold_package["package"]).read_bytes()
    assert second_build["archive_sha256"] == threshold_package["build"][
        "archive_sha256"
    ]
    sidecar = Path(second_build["delivery_manifest_path"])
    assert sha256(sidecar.read_bytes()).hexdigest() == sha256(
        Path(threshold_package["sidecar"]).read_bytes()
    ).hexdigest()
    with zipfile.ZipFile(second) as archive:
        names = set(archive.namelist())
        manifest = json.loads(
            archive.read("experiment_execution_manifest.json")
        )
        package_readme = archive.read("README.md").decode("utf-8")
        entrypoint_source = archive.read(
            "scripts/experiment_execution/experiment_execution_entrypoint.py"
        ).decode("utf-8")
    assert {
        "configs/experiments/c1_hf_content_threshold_fit_manifest.json",
        "configs/experiments/c1_hf_threshold_fit_execution.json",
        "experiments/runners/c1_hf_threshold_fit.py",
        "scripts/experiment_execution/experiment_execution_entrypoint.py",
    } <= names
    assert names == {
        *EXACT_FILES,
        "README.md",
        "experiment_execution_manifest.json",
    }
    assert not any(
        forbidden in name
        for name in names
        for forbidden in (
            "untouched_confirmation_manifest",
            "experiments/methods/baselines/",
            "experiments/protocol/comparison.py",
            "experiments/runners/synthetic_runtime.py",
            "configs/baselines/",
            "tests/integration/test_packaged_experiment_execution.py",
            "tests/smoke/test_packaged_experiment_execution.py",
        )
    )
    assert manifest["candidate_config_digest"] == threshold_package["digests"][
        "candidate_config_digest"
    ]
    assert manifest["execution_config_digest"] == threshold_package["digests"][
        "execution_config_digest"
    ]
    assert manifest["input_manifest_digest"] == threshold_package["digests"][
        "input_manifest_digest"
    ]
    assert "C1 HF threshold-fit execution package" in package_readme
    assert "untouched-confirmation manifest" in package_readme
    assert "CPU/synthetic development wiring" not in package_readme
    assert "prepare_synthetic_wiring" not in entrypoint_source
    assert "run_synthetic_wiring" not in entrypoint_source


def _bootstrap_command(
    fixture: dict[str, object],
    *,
    ephemeral_root: Path,
    persistent_root: Path,
    run_id: str,
    package: Path | None = None,
    sidecar: Path | None = None,
    archive_sha256: str | None = None,
    delivery_manifest_sha256: str | None = None,
    embedded_manifest_sha256: str | None = None,
) -> tuple[int, dict[str, object], subprocess.CompletedProcess[str]]:
    build = fixture["build"]
    package_path = package or fixture["package"]
    sidecar_path = sidecar or fixture["sidecar"]
    command = (
        sys.executable,
        str(ROOT / "scripts/experiment_execution/experiment_execution_bootstrap.py"),
        "--package-zip",
        str(package_path),
        "--delivery-manifest-path",
        str(sidecar_path),
        "--expected-archive-sha256",
        archive_sha256 or build["archive_sha256"],
        "--expected-delivery-manifest-sha256",
        delivery_manifest_sha256
        or sha256(sidecar_path.read_bytes()).hexdigest(),
        "--expected-embedded-manifest-sha256",
        embedded_manifest_sha256 or build["embedded_manifest_sha256"],
        "--expected-bootstrap-identity",
        bootstrap.BOOTSTRAP_IDENTITY,
        "--expected-bootstrap-schema-version",
        str(bootstrap.BOOTSTRAP_SCHEMA_VERSION),
        "--expected-bootstrap-sha256",
        sha256(Path(bootstrap.__file__).read_bytes()).hexdigest(),
        "--expected-revision",
        fixture["revision"],
        "--ephemeral-root",
        str(ephemeral_root),
        "--persistent-root",
        str(persistent_root),
        "--run-id",
        run_id,
        "--shard-index",
        "0",
    )
    environment = dict(os.environ)
    environment["CEG_WM_ROOT_KEY"] = "cpu-test-registered-detection-key"
    environment["PIP_NO_INDEX"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    fake_site = ephemeral_root.parent / "verified_dependency_metadata"
    fake_site.mkdir(exist_ok=True)
    for requirement in (
        ROOT / "requirements_runtime_qualification.txt"
    ).read_text(encoding="utf-8").splitlines():
        distribution, version = requirement.split("==", 1)
        metadata_root = fake_site / (
            f"{distribution.replace('-', '_')}-{version}.dist-info"
        )
        metadata_root.mkdir(exist_ok=True)
        (metadata_root / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
            encoding="utf-8",
        )
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(fake_site), environment.get("PYTHONPATH")))
    )
    (fake_site / "sitecustomize.py").write_text(
        "import torch\ntorch.__version__ = '2.11.0'\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        command,
        cwd=ephemeral_root.parent,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    outcome = json.loads(completed.stdout)
    return completed.returncode, outcome, completed


def _rewrite_archive(
    source: Path,
    destination: Path,
    replacements: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(source) as archive:
        blobs = {name: archive.read(name) for name in archive.namelist()}
    blobs.update(replacements)
    with zipfile.ZipFile(destination, mode="w") as archive:
        for name, blob in sorted(blobs.items()):
            archive.writestr(name, blob)


def _trusted_sidecar_for_archive(
    fixture: dict[str, object],
    *,
    package: Path,
    embedded_manifest_sha256: str,
    destination: Path,
) -> None:
    raw = json.loads(Path(fixture["sidecar"]).read_text(encoding="utf-8"))
    raw["package_filename"] = package.name
    raw["archive_sha256"] = sha256(package.read_bytes()).hexdigest()
    raw["embedded_manifest_sha256"] = embedded_manifest_sha256
    destination.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.quick
def test_dynamic_import_rejects_preloaded_repository_namespaces(
    threshold_package: dict[str, object],
    tmp_path: Path,
) -> None:
    import experiments

    assert experiments.__file__
    package_root = tmp_path / "package"
    bootstrap._safe_extract(threshold_package["package"], package_root)
    manifest = bootstrap._load_and_verify_manifest(
        package_root,
        expected_revision=threshold_package["revision"],
        expected_candidate_config_digest=threshold_package["digests"][
            "candidate_config_digest"
        ],
        expected_execution_config_digest=threshold_package["digests"][
            "execution_config_digest"
        ],
        expected_input_manifest_digest=threshold_package["digests"][
            "input_manifest_digest"
        ],
    )
    assert "experiments" in sys.modules
    with pytest.raises(
        bootstrap.ExperimentBootstrapError,
        match="loaded before external verification",
    ):
        bootstrap._load_verified_threshold_fit_entrypoint(package_root, manifest)


@pytest.mark.quick
def test_resource_failure_and_second_resume_produce_distinct_artifacts(
    threshold_package: dict[str, object],
    tmp_path: Path,
) -> None:
    persistent = (tmp_path / "persistent").resolve()
    ephemeral = (tmp_path / "ephemeral").resolve()
    first_code, first, first_process = _bootstrap_command(
        threshold_package,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        run_id="resource-resume",
    )
    assert first_code == 0, first_process.stderr
    assert first["artifact_kind"] == "c1_threshold_fit_shard_diagnostic"
    assert first["run_status"] == "diagnostic"
    assert first["failure_class"] == "resource_failure"
    assert first["scientific_claims_supported"] is False
    first_zip = Path(first["result_zip"])
    assert first_zip.is_file()

    second_code, second, second_process = _bootstrap_command(
        threshold_package,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        run_id="resource-resume",
    )
    assert second_code == 0, second_process.stderr
    second_zip = Path(second["result_zip"])
    assert second_zip.is_file()
    assert second_zip != first_zip
    assert first_zip.is_file()
    with zipfile.ZipFile(second_zip) as archive:
        outcome = json.loads(archive.read("threshold_fit_outcome.json"))
        record_name = outcome["record_files"][0]["path"]
        collection = json.loads(archive.read(record_name))
    assert outcome["failure_class"] == "resource_failure"
    assert len(collection["attempts"]) == 2
    expected_dependencies = dict(
        line.split("==", 1)
        for line in (
            ROOT / "requirements_runtime_qualification.txt"
        ).read_text(encoding="utf-8").splitlines()
    )
    environment_facts = outcome["execution_facts"]["environment"]
    assert environment_facts["dependency_versions"] == expected_dependencies
    assert environment_facts["torch_import_version"].split("+", 1)[0] == (
        expected_dependencies["torch"]
    )
    record_root = (
        persistent
        / "threshold_fit_records"
        / threshold_package["revision"]
        / "resource-resume"
        / "threshold_fit"
        / "shard_00"
    )
    assert record_root.is_dir()
    assert not (
        persistent
        / "threshold_fit_records"
        / threshold_package["revision"]
        / "resource-resume"
        / "resource-resume"
    ).exists()
    tampered = json.loads(json.dumps(collection))
    tampered["identity"]["analysis_unit_identity"]["prompt_digest"] = "f" * 64
    tampered_path = tmp_path / "tampered_record.json"
    tampered_path.write_bytes(
        json.dumps(
            tampered,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(
        entrypoint.ExperimentExecutionEntrypointError,
        match="failed typed replay",
    ):
        entrypoint._load_and_replay_threshold_fit_record(tampered_path)


@pytest.mark.quick
def test_preregistered_exclusion_is_always_diagnostic() -> None:
    assert entrypoint._classify_threshold_fit_failure(
        failure_classes=set(),
        retry_pending_count=0,
        excluded_count=1,
        complete_shard=True,
    ) == "excluded"


@pytest.mark.quick
@pytest.mark.parametrize("tamper_kind", ("archive", "manifest", "entrypoint"))
def test_package_tamper_is_bootstrap_failure_before_import(
    threshold_package: dict[str, object],
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    original = Path(threshold_package["package"])
    package = tmp_path / f"{tamper_kind}.zip"
    sidecar = tmp_path / f"{tamper_kind}.manifest.json"
    expected_archive = threshold_package["build"]["archive_sha256"]
    expected_embedded = threshold_package["build"]["embedded_manifest_sha256"]
    if tamper_kind == "archive":
        shutil.copy2(original, package)
        package.write_bytes(package.read_bytes() + b"tamper")
        shutil.copy2(threshold_package["sidecar"], sidecar)
    elif tamper_kind == "manifest":
        with zipfile.ZipFile(original) as archive:
            raw = json.loads(archive.read("experiment_execution_manifest.json"))
        raw["committed_revision"] = "f" * 40
        manifest_blob = (
            json.dumps(raw, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _rewrite_archive(
            original,
            package,
            {"experiment_execution_manifest.json": manifest_blob},
        )
        expected_archive = sha256(package.read_bytes()).hexdigest()
        expected_embedded = sha256(manifest_blob).hexdigest()
        _trusted_sidecar_for_archive(
            threshold_package,
            package=package,
            embedded_manifest_sha256=expected_embedded,
            destination=sidecar,
        )
    else:
        _rewrite_archive(
            original,
            package,
            {
                "scripts/experiment_execution/experiment_execution_entrypoint.py": (
                    b"raise AssertionError('must not import')\n"
                )
            },
        )
        expected_archive = sha256(package.read_bytes()).hexdigest()
        _trusted_sidecar_for_archive(
            threshold_package,
            package=package,
            embedded_manifest_sha256=expected_embedded,
            destination=sidecar,
        )
    code, outcome, _process = _bootstrap_command(
        threshold_package,
        ephemeral_root=(tmp_path / f"ephemeral_{tamper_kind}").resolve(),
        persistent_root=(tmp_path / f"persistent_{tamper_kind}").resolve(),
        run_id=f"tamper-{tamper_kind}",
        package=package,
        sidecar=sidecar,
        archive_sha256=expected_archive,
        embedded_manifest_sha256=expected_embedded,
    )
    assert code == 3
    assert outcome["artifact_kind"] == "bootstrap_failure"
    assert outcome["failure_stage"] in {
        "archive_digest",
        "manifest",
    }


@pytest.mark.quick
def test_sidecar_sha_mismatch_is_bootstrap_failure(
    threshold_package: dict[str, object],
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "tampered.manifest.json"
    original = Path(threshold_package["sidecar"])
    sidecar.write_bytes(original.read_bytes() + b" ")
    code, outcome, _process = _bootstrap_command(
        threshold_package,
        ephemeral_root=(tmp_path / "ephemeral_sidecar").resolve(),
        persistent_root=(tmp_path / "persistent_sidecar").resolve(),
        run_id="tamper-sidecar",
        sidecar=sidecar,
        delivery_manifest_sha256=sha256(original.read_bytes()).hexdigest(),
    )
    assert code == 3
    assert outcome["artifact_kind"] == "bootstrap_failure"
    assert outcome["failure_stage"] == "delivery_manifest"


@pytest.mark.quick
def test_dependency_install_failure_precedes_package_import(
    threshold_package: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[tuple[str, ...], dict[str, str]]] = []
    imported = False

    def missing_version(_distribution: str) -> str:
        raise bootstrap.metadata.PackageNotFoundError

    def failed_install(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append((command, kwargs["env"]))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(command, 1, "", "offline")

    def forbidden_import(*_args: object, **_kwargs: object) -> object:
        nonlocal imported
        imported = True
        raise AssertionError("package import must not occur")

    monkeypatch.setattr(bootstrap.metadata, "version", missing_version)
    monkeypatch.setattr(bootstrap.subprocess, "run", failed_install)
    monkeypatch.setattr(
        bootstrap,
        "_load_verified_threshold_fit_entrypoint",
        forbidden_import,
    )
    code, outcome = bootstrap.run_bootstrap(
        package_zip=threshold_package["package"],
        delivery_manifest_path=threshold_package["sidecar"],
        expected_archive_sha256=threshold_package["build"]["archive_sha256"],
        expected_delivery_manifest_sha256=sha256(
            Path(threshold_package["sidecar"]).read_bytes()
        ).hexdigest(),
        expected_embedded_manifest_sha256=threshold_package["build"][
            "embedded_manifest_sha256"
        ],
        expected_bootstrap_identity=bootstrap.BOOTSTRAP_IDENTITY,
        expected_bootstrap_schema_version=bootstrap.BOOTSTRAP_SCHEMA_VERSION,
        expected_bootstrap_sha256=sha256(
            Path(bootstrap.__file__).read_bytes()
        ).hexdigest(),
        expected_revision=threshold_package["revision"],
        ephemeral_root=(tmp_path / "ephemeral_dependency").resolve(),
        persistent_root=(tmp_path / "persistent_dependency").resolve(),
        run_id="dependency-failure",
        shard_index=0,
        environment={
            "CEG_WM_ROOT_KEY": "must-not-reach-pip",
            "HF_TOKEN": "must-not-reach-pip",
        },
    )
    assert code == 3
    assert outcome["failure_stage"] == "dependency_install"
    assert imported is False
    assert len(commands) == 1
    command, pip_environment = commands[0]
    assert command[:5] == (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    )
    assert "--requirement" in command
    assert "--target" in command
    assert "--cache-dir" in command
    assert "CEG_WM_ROOT_KEY" not in pip_environment
    assert "HF_TOKEN" not in pip_environment


@pytest.mark.quick
def test_threshold_fit_notebook_is_thin_and_output_free() -> None:
    notebook = json.loads(
        (
            ROOT / "notebooks/colab/experiment_execution.ipynb"
        ).read_text(encoding="utf-8")
    )
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "userdata.get('HF_TOKEN')" in source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in source
    assert "--shard-index" in source
    assert "--expected-delivery-manifest-sha256" in source
    assert "--expected-embedded-manifest-sha256" in source
    assert "--expected-candidate-config-digest" not in source
    assert "--expected-execution-config-digest" not in source
    assert "--expected-input-manifest-digest" not in source
    assert "c1_hf_untouched_confirmation_manifest.json" not in source
    assert "prepare_synthetic_wiring" not in source


@pytest.mark.quick
def test_entrypoint_cli_rejects_direct_invocation() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "scripts.experiment_execution.experiment_execution_entrypoint",
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "direct entrypoint invocation is forbidden" in completed.stderr
