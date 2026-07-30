"""CPU checks for the A3b package, bootstrap, and formal operations."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import zipfile

import pytest
import torch

import experiments.methods.ceg_wm as ceg_wm_adapter_module
import main
import main.content_chain as main_content_chain
import main.shared as main_shared
from scripts.experiment_execution import (
    build_experiment_execution_package as package_builder_module,
)
from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.runners import (
    FormalHfContentDetectionOperation,
    FormalOperationError,
    FormalRuntimeGeometryEstimationOperation,
    InternalRunnerError,
    create_formal_content_detector_binding,
    execute_internal_case,
    formal_operation_config_digest,
)
from runtime import RuntimeBackendIdentity, RuntimeDeviceCapabilities
from runtime import create_runtime_adapter
from scripts.experiment_execution import (
    experiment_execution_bootstrap as bootstrap,
)
from scripts.experiment_execution.build_experiment_execution_package import (
    ExperimentPackageBuildError,
    LOCAL_PATH,
    SENSITIVE_COLAB_PATH,
    build_experiment_execution_package,
)
from scripts.experiment_execution.experiment_execution_entrypoint import (
    prepare_synthetic_wiring,
)


ROOT = Path(__file__).resolve().parents[2]
COMPONENT_CONFIG = (
    ROOT / "configs/experiments/internal_execution_components.json"
)
RUNTIME_CONFIG = ROOT / "configs/runtime/runtime_sd35_flowmatch.json"
DIGESTS = {
    "candidate_config_digest": "1" * 64,
    "execution_config_digest": "2" * 64,
    "input_manifest_digest": "3" * 64,
}


class _IdentityOnlyBackend:
    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )

    def prepare(self, configuration, selected_device):
        return RuntimeBackendIdentity(
            candidate_id=configuration.candidate_id,
            runtime_config_digest=configuration.runtime_config_digest,
            runtime_backend_name="identity_only_synthetic_backend",
            selected_device=selected_device,
            model_id=configuration.model_id,
            model_revision=configuration.model_revision,
            pipeline_class=configuration.pipeline_class,
            scheduler_class=configuration.scheduler_class,
            inference_steps=configuration.inference_steps,
            guidance_scale=configuration.guidance_scale,
            image_height=configuration.image_height,
            image_width=configuration.image_width,
            generation_seed_device=configuration.generation_seed_device,
            latent_dtype=configuration.latent_dtype,
            template_dtype=configuration.template_dtype,
            score_dtype=configuration.score_dtype,
            callback_index=configuration.callback_index,
            callback_hold_scheduler_intervals=(
                configuration.callback_hold_scheduler_intervals
            ),
            vae_decode_protocol=configuration.vae_decode_protocol,
            vae_encode_protocol=configuration.vae_encode_protocol,
            vae_scaling_factor_source=(
                configuration.vae_scaling_factor_source
            ),
            vae_shift_factor_source=(
                configuration.vae_shift_factor_source
            ),
            detection_schedule_index=(
                configuration.detection_schedule_index
            ),
            detection_conditioning_protocol=(
                configuration.detection_conditioning_protocol
            ),
            qk_layer_names=configuration.qk_layer_names,
            dependency_lock=configuration.dependency_lock,
        )

    def close(self) -> None:
        return None


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _minimal_repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repository"
    _write(
        root / "templates/release_readmes/experiment_execution_package.md",
        "# package\n",
    )
    _write(
        root / "pyproject.toml",
        "[project]\nname='fixture'\nversion='0.0.0'\n",
    )
    _write(root / "main/__init__.py", "VALUE = 1\n")
    _write(root / "runtime/__init__.py", "VALUE = 2\n")
    _write(root / "experiments/__init__.py", "VALUE = 3\n")
    _write(root / "configs/README.md", "configuration\n")
    _write(root / "infrastructure/README.md", "infrastructure\n")
    _write(
        root / "tests/integration/__init__.py",
        '"""fixture integration tests."""\n',
    )
    _write(
        root
        / "tests/integration/test_packaged_experiment_execution.py",
        "def test_package():\n    assert True\n",
    )
    _write(
        root / "tests/smoke/test_packaged_experiment_execution.py",
        "def test_entrypoint():\n    assert True\n",
    )
    _write(
        root / "scripts/experiment_execution/__init__.py",
        '"""fixture"""\n',
    )
    _write(
        root
        / "scripts/experiment_execution/experiment_execution_entrypoint.py",
        "def main():\n    return 0\n",
    )
    _write(
        root
        / "scripts/experiment_execution/experiment_execution_bootstrap.py",
        "raise AssertionError('must not be packaged')\n",
    )
    _write(root / "notebooks/forbidden.ipynb", "{}\n")
    _write(root / "governance/forbidden.py", "raise AssertionError\n")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def _build(
    root: Path,
    revision: str,
    output: Path,
) -> dict[str, object]:
    return build_experiment_execution_package(
        root=root,
        output_zip=output,
        committed_revision=revision,
        **DIGESTS,
    )


@pytest.mark.quick
def test_package_build_is_deterministic_and_allowlisted(
    tmp_path: Path,
) -> None:
    root, revision = _minimal_repository(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = _build(root, revision, first)
    second_result = _build(root, revision, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result["archive_sha256"] == second_result["archive_sha256"]
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        manifest = json.loads(
            archive.read("experiment_execution_manifest.json")
        )
    assert manifest["committed_revision"] == revision
    assert manifest["candidate_config_digest"] == "1" * 64
    assert manifest["execution_config_digest"] == "2" * 64
    assert manifest["input_manifest_digest"] == "3" * 64
    assert manifest["entrypoint_identity"].endswith(":main")
    assert {
        name
        for name in names
        if name.startswith(("tests/integration/", "tests/smoke/"))
    } == {
        "tests/integration/__init__.py",
        "tests/integration/test_packaged_experiment_execution.py",
        "tests/smoke/test_packaged_experiment_execution.py",
    }
    assert (
        "scripts/experiment_execution/experiment_execution_bootstrap.py"
        not in names
    )
    assert not any(
        name.startswith(("governance/", "notebooks/", ".codex/", ".agents/"))
        for name in names
    )


@pytest.mark.constraint
def test_builder_path_scanners_preserve_behavior_without_source_local_paths(
) -> None:
    root_components = (
        "home",
        "Users",
        "mnt",
        "content",
        "tmp",
        "var",
        "opt",
        "root",
    )
    portability_scan = re.compile(
        r"(?<![A-Za-z0-9_])(?:/(?:"
        + "|".join(root_components)
        + r")/|[A-Za-z]:[\\/])"
    )
    builder_source = Path(
        package_builder_module.__file__
    ).read_text(encoding="utf-8")
    assert portability_scan.search(builder_source) is None

    local_roots = tuple(
        component.encode("ascii")
        for component in root_components
        if component != "content"
    )
    for root_component in local_roots:
        assert LOCAL_PATH.search(
            b"/" + root_component + b"/project/file.py"
        )
    assert LOCAL_PATH.search(b"C:\\project\\file.py")
    assert LOCAL_PATH.search(b"D:/project/file.py")
    assert LOCAL_PATH.search(b"/" + b"content" + b"/workspace") is None

    sensitive_colab = (
        b"private",
        b"secret",
        b"credentials",
        b"model-weights",
        b"checkpoint",
    )
    for sensitive_component in sensitive_colab:
        assert SENSITIVE_COLAB_PATH.search(
            b"/"
            + b"content"
            + b"/nested/"
            + sensitive_component
            + b"/asset.bin"
        )
    assert SENSITIVE_COLAB_PATH.search(
        b"/" + b"content" + b"/ceg_wm_experiment_execution"
    ) is None


@pytest.mark.constraint
def test_formal_operations_use_identity_preserving_main_facade() -> None:
    assert main.ContentDetectorError is (
        main_content_chain.ContentDetectorError
    )
    assert main.validate_content_detection_result is (
        main_content_chain.validate_content_detection_result
    )
    assert main.rgb8_image_digest is main_shared.rgb8_image_digest
    assert {
        "ContentDetectorError",
        "validate_content_detection_result",
        "rgb8_image_digest",
    }.issubset(main.__all__)

    formal_source = (
        ROOT / "experiments/runners/formal_operations.py"
    ).read_text(encoding="utf-8")
    assert "from main.content_chain import" not in formal_source
    assert "from main.shared import" not in formal_source


@pytest.mark.quick
def test_package_build_rejects_dirty_or_mismatched_revision(
    tmp_path: Path,
) -> None:
    root, revision = _minimal_repository(tmp_path)
    _write(root / "main/__init__.py", "VALUE = 9\n")
    with pytest.raises(
        ExperimentPackageBuildError,
        match="clean",
    ):
        _build(root, revision, tmp_path / "dirty.zip")
    _git(root, "restore", "main/__init__.py")
    with pytest.raises(
        ExperimentPackageBuildError,
        match="does not equal HEAD",
    ):
        _build(root, "f" * 40, tmp_path / "wrong.zip")


@pytest.mark.quick
def test_package_build_rejects_sensitive_colab_paths_but_allows_orchestration(
    tmp_path: Path,
) -> None:
    sensitive_root, _revision = _minimal_repository(
        tmp_path / "sensitive"
    )
    _write(
        sensitive_root / "main/leak.py",
        "MODEL_PATH = '/content/private/model.bin'\n",
    )
    _git(sensitive_root, "add", ".")
    _git(sensitive_root, "commit", "-m", "sensitive path")
    sensitive_revision = _git(sensitive_root, "rev-parse", "HEAD")
    with pytest.raises(
        ExperimentPackageBuildError,
        match="sensitive Colab absolute path",
    ):
        _build(
            sensitive_root,
            sensitive_revision,
            tmp_path / "sensitive.zip",
        )

    allowed_root, _revision = _minimal_repository(tmp_path / "allowed")
    _write(
        allowed_root / "main/orchestration.py",
        (
            "WORKSPACE = '/content/ceg_wm_experiment_execution'\n"
            "DRIVE = '/content/drive/MyDrive/CEG-WM'\n"
        ),
    )
    _git(allowed_root, "add", ".")
    _git(allowed_root, "commit", "-m", "sanctioned orchestration paths")
    allowed_revision = _git(allowed_root, "rev-parse", "HEAD")
    result = _build(
        allowed_root,
        allowed_revision,
        tmp_path / "allowed.zip",
    )
    assert result["archive_sha256"]


def _bootstrap_call(
    *,
    tmp_path: Path,
    package: Path,
    archive_digest: str,
    revision: str,
    run_id: str,
    command_runner,
    bootstrap_digest: str | None = None,
) -> tuple[int, dict[str, object]]:
    return bootstrap.run_bootstrap(
        package_zip=package.resolve(),
        expected_archive_sha256=archive_digest,
        expected_bootstrap_identity=bootstrap.BOOTSTRAP_IDENTITY,
        expected_bootstrap_schema_version=bootstrap.BOOTSTRAP_SCHEMA_VERSION,
        expected_bootstrap_sha256=(
            bootstrap_digest
            if bootstrap_digest is not None
            else sha256(Path(bootstrap.__file__).read_bytes()).hexdigest()
        ),
        expected_revision=revision,
        expected_candidate_config_digest="1" * 64,
        expected_execution_config_digest="2" * 64,
        expected_input_manifest_digest="3" * 64,
        ephemeral_root=(tmp_path / f"ephemeral_{run_id}").resolve(),
        persistent_root=(tmp_path / f"persistent_{run_id}").resolve(),
        run_id=run_id,
        command_runner=command_runner,
    )


@pytest.mark.quick
def test_bootstrap_rejects_archive_tamper_before_package_execution(
    tmp_path: Path,
) -> None:
    root, revision = _minimal_repository(tmp_path)
    package = tmp_path / "package.zip"
    result = _build(root, revision, package)
    package.write_bytes(package.read_bytes() + b"tamper")
    calls = []

    exit_code, outcome = _bootstrap_call(
        tmp_path=tmp_path,
        package=package,
        archive_digest=result["archive_sha256"],
        revision=revision,
        run_id="archive-tamper",
        command_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert exit_code == 3
    assert outcome["artifact_kind"] == "bootstrap_failure"
    assert outcome["failure_stage"] == "archive_digest"
    assert calls == []


@pytest.mark.quick
def test_bootstrap_rejects_unsafe_archive_path_before_package_execution(
    tmp_path: Path,
) -> None:
    package = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(package, mode="w") as archive:
        archive.writestr("../escape.py", "forbidden")
        archive.writestr("experiment_execution_manifest.json", "{}")
    calls = []

    exit_code, outcome = _bootstrap_call(
        tmp_path=tmp_path,
        package=package,
        archive_digest=sha256(package.read_bytes()).hexdigest(),
        revision="a" * 40,
        run_id="unsafe-path",
        command_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert exit_code == 3
    assert outcome["failure_stage"] == "archive_safety"
    assert calls == []


@pytest.mark.quick
def test_bootstrap_rejects_manifest_revision_before_package_execution(
    tmp_path: Path,
) -> None:
    root, revision = _minimal_repository(tmp_path)
    original = tmp_path / "original.zip"
    _build(root, revision, original)
    package = tmp_path / "revision-tampered.zip"
    with zipfile.ZipFile(original) as source:
        blobs = {
            name: source.read(name)
            for name in source.namelist()
        }
    manifest = json.loads(blobs["experiment_execution_manifest.json"])
    manifest["committed_revision"] = "b" * 40
    blobs["experiment_execution_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with zipfile.ZipFile(package, mode="w") as archive:
        for name, blob in sorted(blobs.items()):
            archive.writestr(name, blob)
    calls = []

    exit_code, outcome = _bootstrap_call(
        tmp_path=tmp_path,
        package=package,
        archive_digest=sha256(package.read_bytes()).hexdigest(),
        revision=revision,
        run_id="revision-tamper",
        command_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert exit_code == 3
    assert outcome["failure_stage"] == "manifest"
    assert calls == []


@pytest.mark.quick
def test_bootstrap_rejects_its_own_identity_before_package_execution(
    tmp_path: Path,
) -> None:
    package = tmp_path / "unused.zip"
    package.write_bytes(b"unused")
    calls = []

    exit_code, outcome = _bootstrap_call(
        tmp_path=tmp_path,
        package=package,
        archive_digest=sha256(package.read_bytes()).hexdigest(),
        revision="a" * 40,
        run_id="bootstrap-tamper",
        bootstrap_digest="0" * 64,
        command_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert exit_code == 3
    assert outcome["failure_stage"] == "bootstrap_identity"
    assert calls == []


@pytest.mark.quick
def test_bootstrap_rejects_unsafe_run_id_without_path_escape(
    tmp_path: Path,
) -> None:
    package = tmp_path / "unused-run-id.zip"
    package.write_bytes(b"unused")
    calls = []

    exit_code, outcome = _bootstrap_call(
        tmp_path=tmp_path,
        package=package,
        archive_digest=sha256(package.read_bytes()).hexdigest(),
        revision="a" * 40,
        run_id="../escape",
        command_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert exit_code == 3
    assert outcome["failure_stage"] == "arguments"
    assert outcome["run_id"].startswith("invalid-run-id-")
    assert not (tmp_path / "escape").exists()
    assert calls == []


@pytest.mark.quick
def test_formal_operations_declare_complete_configuration_and_drift(
) -> None:
    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG
    )
    content_operation = FormalHfContentDetectionOperation(
        CegWmExperimentAdapter(configuration)
    )
    image = torch.arange(
        3 * 9 * 9,
        dtype=torch.uint8,
    ).reshape(1, 3, 9, 9)
    binding, _score = create_formal_content_detector_binding(
        content_operation,
        prototype_image=image,
        detection_key="formal-operation-test-key",
    )
    content_declaration = (
        content_operation.formal_runner_semantic_declaration()
    )
    assert set(content_declaration) == {
        "adapter_configuration",
        "adapter_config_digest",
        "adapter_method_anchors",
        "content_detector_public_callable",
        "formal_mode",
        "hf_detector_public_callable",
        "image_encoding",
        "pixel_conversion",
        "semantic_version",
    }
    assert binding.detector_identity
    assert formal_operation_config_digest(
        content_operation,
        operation_role="content_detection",
    )

    runtime_adapter = create_runtime_adapter(
        _IdentityOnlyBackend(),
        RUNTIME_CONFIG,
    )
    geometry_operation = FormalRuntimeGeometryEstimationOperation(
        runtime_adapter=runtime_adapter,
        adapter_configuration=configuration,
        epsilon_inlier=0.8,
        execution_scope="cpu_synthetic_wiring_only",
    )
    geometry_declaration = (
        geometry_operation.formal_runner_semantic_declaration()
    )
    assert geometry_declaration["runtime_state"] == "created"
    assert geometry_declaration["runtime_session"] is None
    assert geometry_declaration["adapter_configuration"][
        "config_digest"
    ] == configuration.config_digest
    assert geometry_declaration["runtime_config_digest"] == (
        runtime_adapter.configuration.runtime_config_digest
    )
    assert geometry_declaration["runtime_configuration"]["model_id"] == (
        runtime_adapter.configuration.model_id
    )
    assert geometry_declaration["runtime_qk_layer_names"] == [
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    ]
    object.__setattr__(geometry_operation, "epsilon_inlier", 0.7)
    with pytest.raises(FormalOperationError, match="drifted"):
        geometry_operation.formal_runner_semantic_declaration()


@pytest.mark.quick
def test_ready_geometry_declaration_is_canonical_and_rejects_backend_drift(
) -> None:
    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG
    )
    runtime_adapter = create_runtime_adapter(
        _IdentityOnlyBackend(),
        RUNTIME_CONFIG,
    )
    runtime_adapter.initialize("cpu")
    geometry_operation = FormalRuntimeGeometryEstimationOperation(
        runtime_adapter=runtime_adapter,
        adapter_configuration=configuration,
        epsilon_inlier=0.8,
        execution_scope="cpu_synthetic_wiring_only",
    )

    declaration = (
        geometry_operation.formal_runner_semantic_declaration()
    )
    assert json.loads(json.dumps(declaration)) == declaration
    assert declaration["runtime_state"] == "ready"
    assert declaration["runtime_session"]["qk_layer_names"] == [
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    ]
    assert formal_operation_config_digest(
        geometry_operation,
        operation_role="geometry_estimation",
    )

    runtime_adapter._backend = _IdentityOnlyBackend()
    with pytest.raises(
        FormalOperationError,
        match="runtime execution identity drifted",
    ):
        geometry_operation.formal_runner_semantic_declaration()
    with pytest.raises(
        FormalOperationError,
        match="runtime execution identity drifted",
    ):
        geometry_operation(
            torch.zeros((1, 3, 9, 9), dtype=torch.uint8),
            "registered-key",
        )


@pytest.mark.quick
def test_content_operation_rejects_method_shadow_before_writer(
    tmp_path: Path,
) -> None:
    preparation = prepare_synthetic_wiring(
        package_root=ROOT,
        records_root=tmp_path / "records",
        workspace_root=tmp_path / "workspace",
        committed_revision="a" * 40,
        run_id="content-shadow",
    )
    preparation.context.adapter.detect_hf = lambda *_args: None

    with pytest.raises(
        InternalRunnerError,
        match="semantic declaration",
    ):
        execute_internal_case(
            preparation.context,
            unit_id=(
                preparation.payload.source_artifact
                .analysis_unit_identity.unit_id
            ),
            payload=preparation.payload,
        )

    assert not preparation.context.writer.path.exists()


@pytest.mark.quick
def test_content_operation_rejects_class_method_implementation_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG
    )
    operation = FormalHfContentDetectionOperation(
        CegWmExperimentAdapter(configuration)
    )
    monkeypatch.setattr(
        CegWmExperimentAdapter,
        "detect_hf",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        FormalOperationError,
        match="implementation or binding drifted",
    ):
        operation.formal_runner_semantic_declaration()


@pytest.mark.quick
@pytest.mark.parametrize(
    "shadowed_method",
    ("detect_hf", "detect_content"),
)
def test_content_operation_rejects_mid_call_method_shadow_before_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shadowed_method: str,
) -> None:
    preparation = prepare_synthetic_wiring(
        package_root=ROOT,
        records_root=tmp_path / "records",
        workspace_root=tmp_path / "workspace",
        committed_revision="b" * 40,
        run_id="content-mid-call-shadow",
    )
    original_hf_detector = ceg_wm_adapter_module.hf_detector

    def _mutating_hf_detector(*args, **kwargs):
        result = original_hf_detector(*args, **kwargs)
        setattr(
            preparation.context.adapter,
            shadowed_method,
            lambda *_args, **_kwargs: None
        )
        return result

    monkeypatch.setattr(
        ceg_wm_adapter_module,
        "hf_detector",
        _mutating_hf_detector,
    )

    with pytest.raises(
        InternalRunnerError,
        match="semantic declaration",
    ):
        execute_internal_case(
            preparation.context,
            unit_id=(
                preparation.payload.source_artifact
                .analysis_unit_identity.unit_id
            ),
            payload=preparation.payload,
        )

    assert not preparation.context.writer.path.exists()


@pytest.mark.constraint
def test_experiment_notebook_is_thin_clean_and_external_bootstrap_only() -> None:
    notebook = ROOT / "notebooks/colab/experiment_execution.ipynb"
    document = json.loads(notebook.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in document["cells"] if cell["cell_type"] == "code"
    ]
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in document["cells"]
    )
    assert 4 <= len(code_cells) <= 6
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs", []) == [] for cell in document["cells"])
    assert "experiment_execution_bootstrap.py" in sources
    assert "--expected-archive-sha256" in sources
    assert "--expected-bootstrap-sha256" in sources
    assert "files.download" in sources
    for forbidden in (
        "experiment_execution_manifest.json",
        "FrozenCaseInputManifest",
        "execute_internal_case",
        "GovernedRecordWriter",
        "zipfile",
        "extractall",
        "hf_detector",
        "geometric_transform_estimator",
        "tau_rescue",
    ):
        assert forbidden not in sources
