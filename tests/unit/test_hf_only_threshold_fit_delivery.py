"""CPU trust-chain tests for the HF-only threshold-fit GPU execution delivery path."""

from __future__ import annotations

from hashlib import sha256
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
import zipfile

import pytest

from scripts.experiment_execution import experiment_execution_entrypoint as entrypoint


ROOT = Path(__file__).resolve().parents[2]
HF_REFERENCE_PRODUCER_REVISION = "cc9af5df0d9a63d349402d56ddd6bb81d117d1e8"
HF_THRESHOLD_DELIVERY_PRODUCER_REVISION = (
    "7797e78a4da11ee39d5554772b299821ea0019b3"
)
DELIVERY_ROOT = ROOT
bootstrap: ModuleType
builder_module: ModuleType
EXACT_FILES: frozenset[str]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_historical_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load historical delivery module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def _historical_threshold_delivery_root(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    global DELIVERY_ROOT, EXACT_FILES, bootstrap, builder_module
    root = tmp_path_factory.mktemp("threshold_delivery_producer") / "repository"
    subprocess.run(
        ("git", "clone", "--no-checkout", "--quiet", str(ROOT), str(root)),
        check=True,
        capture_output=True,
        text=True,
    )
    _git(root, "checkout", "--detach", HF_THRESHOLD_DELIVERY_PRODUCER_REVISION)
    assert _git(root, "rev-parse", "HEAD") == HF_THRESHOLD_DELIVERY_PRODUCER_REVISION
    assert _git(root, "status", "--porcelain") == ""
    DELIVERY_ROOT = root
    builder_module = _load_historical_module(
        "ceg_wm_historical_threshold_package_builder",
        root / "scripts/experiment_execution/build_experiment_execution_package.py",
    )
    bootstrap = _load_historical_module(
        "ceg_wm_historical_threshold_bootstrap",
        root / "scripts/experiment_execution/experiment_execution_bootstrap.py",
    )
    EXACT_FILES = builder_module.EXACT_FILES
    yield root
    DELIVERY_ROOT = ROOT


def _authority_digests(repository: Path) -> dict[str, str]:
    specification = json.loads(
        (repository / "configs/experiments/hf_only_reference_validation.json").read_text(
            encoding="utf-8"
        )
    )
    execution = json.loads(
        (
            repository
            / "configs/experiments/hf_only_threshold_fit_gpu_execution.json"
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
    root = DELIVERY_ROOT
    revision = HF_THRESHOLD_DELIVERY_PRODUCER_REVISION
    digests = _authority_digests(root)
    package = tmp_path_factory.mktemp("threshold_package") / "threshold-fit.zip"
    build = builder_module.build_experiment_execution_package(
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
    second_build = builder_module.build_experiment_execution_package(
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
        "configs/experiments/hf_only_content_threshold_fit_manifest.json",
        "configs/experiments/hf_only_threshold_fit_gpu_execution.json",
        "experiments/runners/hf_only_threshold_fit_gpu_execution.py",
        "requirements_hf_only_threshold_fit_gpu_execution.txt",
        "scripts/experiment_execution/experiment_execution_entrypoint.py",
    } <= names
    assert "requirements_runtime_qualification.txt" not in names
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
    assert "HF-only threshold-fit GPU execution execution package" in package_readme
    assert "untouched-confirmation manifest" in package_readme
    assert "CPU/synthetic development wiring" not in package_readme
    assert "prepare_synthetic_wiring" not in entrypoint_source
    assert "run_synthetic_wiring" not in entrypoint_source
    with pytest.raises(
        builder_module.ExperimentPackageBuildError,
        match="does not equal HEAD",
    ):
        builder_module.build_experiment_execution_package(
            root=threshold_package["repository"],
            output_zip=tmp_path / "cross-signed.zip",
            committed_revision=HF_REFERENCE_PRODUCER_REVISION,
        )


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
        str(DELIVERY_ROOT / "scripts/experiment_execution/experiment_execution_bootstrap.py"),
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
        DELIVERY_ROOT / "requirements_hf_only_threshold_fit_gpu_execution.txt"
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
    expected_torch = next(
        requirement.split("==", 1)[1]
        for requirement in (
            DELIVERY_ROOT / "requirements_hf_only_threshold_fit_gpu_execution.txt"
        ).read_text(encoding="utf-8").splitlines()
        if requirement.startswith("torch==")
    )
    (fake_site / "sitecustomize.py").write_text(
        f"import torch\ntorch.__version__ = {expected_torch!r}\n",
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
    assert first["artifact_kind"] == "hf_only_threshold_fit_shard_diagnostic"
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
            DELIVERY_ROOT / "requirements_hf_only_threshold_fit_gpu_execution.txt"
        ).read_text(encoding="utf-8").splitlines()
    )
    environment_facts = outcome["execution_facts"]["environment"]
    assert environment_facts["dependency_versions"] == expected_dependencies
    assert environment_facts["torch_import_version"] == expected_dependencies["torch"]
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
            "PIP_EXTRA_INDEX_URL": "https://untrusted.invalid/simple",
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
    assert "--no-deps" in command
    assert bootstrap.HF_ONLY_THRESHOLD_FIT_PYPI_INDEX_URL in command
    assert bootstrap.HF_ONLY_THRESHOLD_FIT_PYTORCH_INDEX_URL in command
    assert bootstrap.HF_ONLY_THRESHOLD_FIT_NVIDIA_INDEX_URL in command
    assert "CEG_WM_ROOT_KEY" not in pip_environment
    assert "HF_TOKEN" not in pip_environment
    assert "PIP_EXTRA_INDEX_URL" not in pip_environment
    assert pip_environment["PIP_CONFIG_FILE"] == os.devnull


@pytest.mark.quick
def test_frozen_model_download_failure_produces_bootstrap_diagnostic(
    threshold_package: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "_prepare_verified_dependencies",
        lambda **_kwargs: None,
    )

    def fail_download(**_kwargs: object) -> object:
        raise bootstrap.ExperimentBootstrapError(
            "model_download",
            "offline",
        )

    monkeypatch.setattr(
        bootstrap,
        "_prepare_frozen_model_snapshot",
        fail_download,
    )
    persistent = (tmp_path / "persistent").resolve()
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
        ephemeral_root=(tmp_path / "ephemeral").resolve(),
        persistent_root=persistent,
        model_cache_root=(tmp_path / "cache").resolve(),
        prepare_frozen_model=True,
        run_id="model-download-failure",
        shard_index=0,
        environment={
            "CEG_WM_ROOT_KEY": "test-root-key",
            "HF_TOKEN": "test-hf-token",
        },
    )
    assert code == 3
    assert outcome["artifact_kind"] == "bootstrap_failure"
    assert outcome["failure_stage"] == "model_download"
    assert Path(outcome["diagnostic_zip"]).is_file()


@pytest.mark.quick
@pytest.mark.parametrize("tamper_kind", ("missing", "extra", "version_drift"))
def test_hf_only_reference_dependency_lock_rejects_incomplete_or_drifted_closure(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    lines = (
        DELIVERY_ROOT / "requirements_hf_only_threshold_fit_gpu_execution.txt"
    ).read_text(encoding="utf-8").splitlines()
    if tamper_kind == "missing":
        lines.pop()
    elif tamper_kind == "extra":
        lines.append("unexpected-distribution==1.0.0")
    else:
        lines[0] = "accelerate==1.14.1"
    lock_path = tmp_path / "requirements_hf_only_threshold_fit_gpu_execution.txt"
    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(
        bootstrap.ExperimentBootstrapError,
        match="lock identity drifted",
    ):
        bootstrap._load_verified_dependency_lock(lock_path)


@pytest.mark.quick
@pytest.mark.parametrize("tamper_kind", ("missing", "extra", "version_drift"))
def test_dependency_target_requires_exact_distribution_set_and_versions(
    tamper_kind: str,
) -> None:
    expected = {"alpha": "1.0", "beta": "2.0"}
    observed = dict(expected)
    if tamper_kind == "missing":
        observed.pop("beta")
    elif tamper_kind == "extra":
        observed["gamma"] = "3.0"
    else:
        observed["beta"] = "2.1"
    with pytest.raises(
        bootstrap.ExperimentBootstrapError,
        match="distribution set or versions differ",
    ):
        bootstrap._require_exact_target_distribution_versions(
            observed,
            expected,
            target_kind="test",
        )


@pytest.mark.quick
def test_experiment_execution_readme_excludes_non_hf_only_threshold_fit_gpu_execution_routes() -> None:
    readme = (
        ROOT / "scripts/experiment_execution/README.md"
    ).read_text(encoding="utf-8")
    assert "HF-only threshold-fit GPU execution" in readme
    assert "schema-v2" in readme
    assert "requirements_hf_only_threshold_fit_gpu_execution.txt" in readme
    assert "complete transitive" in readme
    assert "--no-deps" in readme
    for forbidden in (
        "schema-v1",
        "runtime_qualification_bootstrap.py",
        "build_runtime_qualification_package.py",
        "python -m pip",
        "runtime_qualification_runner",
    ):
        assert forbidden not in readme


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
    assert "https://github.com/RICHAAARC/CEG-WM.git" in source
    assert "7797e78a4da11ee39d5554772b299821ea0019b3" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in source
    assert "git', 'clone'" in source
    assert "rev-parse" in source
    assert "status', '--porcelain'" in source
    assert "hf_only_threshold_fit_server.py" in source
    assert "--shard-index" in source
    assert "atomic_copy_verified" in source
    assert "temporary.replace(destination)" in source
    assert "files.download" in source
    for forbidden in (
        "pip install",
        "snapshot_download",
        "from_pretrained",
        "build_experiment_execution_package",
        "experiment_execution_bootstrap.py",
        "experiments.runners",
        "record_writer",
        "records_root",
    ):
        assert forbidden not in source
    assert "--expected-candidate-config-digest" not in source
    assert "--expected-execution-config-digest" not in source
    assert "--expected-input-manifest-digest" not in source
    assert "hf_only_reference_untouched_confirmation_manifest.json" not in source
    assert "prepare_synthetic_wiring" not in source


@pytest.mark.quick
def test_threshold_fit_notebook_drive_copy_fails_closed_on_hash_mismatch(
    tmp_path: Path,
) -> None:
    notebook = json.loads(
        (
            ROOT / "notebooks/colab/experiment_execution.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"file_sha256", "atomic_copy_verified"}
    ]
    assert {node.name for node in functions} == {
        "file_sha256",
        "atomic_copy_verified",
    }
    namespace = {
        "Path": Path,
        "sha256": sha256,
        "tempfile": __import__("tempfile"),
        "shutil": shutil,
        "os": os,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook", "exec"), namespace)
    source_path = tmp_path / "result.zip"
    destination = tmp_path / "drive/result.zip"
    source_path.write_bytes(b"result")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        namespace["atomic_copy_verified"](
            source_path,
            destination,
            "0" * 64,
        )
    assert not destination.exists()


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
