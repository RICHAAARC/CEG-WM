"""CPU checks for the experiment_execution_delivery package, bootstrap, and formal operations."""

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
from scripts.experiment_execution import experiment_execution_entrypoint


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
def test_retired_synthetic_entrypoint_helpers_are_removed() -> None:
    assert not hasattr(experiment_execution_entrypoint, "prepare_synthetic_wiring")
    assert not hasattr(experiment_execution_entrypoint, "run_synthetic_wiring")


@pytest.mark.quick
def test_experiment_notebook_is_thin_clean_and_calls_unified_server_only() -> None:
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
    assert "hf_only_threshold_fit_server.py" in sources
    assert "files.download" in sources
    for forbidden in (
        "experiment_execution_bootstrap.py",
        "experiment_execution_entrypoint.py",
        "pip install",
        "StableDiffusion3Pipeline",
        "from diffusers",
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
