from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import types
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional

from runtime import (
    RuntimeAdapterError,
    RuntimeContentExecutionError,
    RuntimeDetectionConditioning,
    RuntimeQkObservationError,
    Sd35BackendError,
    Sd35PipelineBackend,
    create_runtime_adapter,
    load_runtime_configuration,
)
from main import ContentEmbedderError, content_embedder, hf_carrier
from scripts.experiment_execution import runtime_qualification_runner as runner
from scripts.experiment_execution.build_runtime_qualification_package import (
    PackageBuildError,
    build_runtime_qualification_package,
)


pytestmark = pytest.mark.unit


def _runner_storage(
    tmp_path: Path,
    label: str,
) -> tuple[Path, Path, Path]:
    ephemeral = tmp_path / f"{label}-ephemeral"
    persistent = tmp_path / f"{label}-persistent"
    return ephemeral, persistent, ephemeral / f"{label}.zip"


def test_sd35_backend_is_lazy_and_accepts_disjoint_roots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "runtime.sd35_backend.importlib.import_module",
        lambda name: calls.append(name),
    )
    Sd35PipelineBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token=None,
        prompt="probe",
    )
    assert calls == []


@pytest.mark.parametrize(
    ("cache_relative", "persistent_relative"),
    (
        ("shared", "shared"),
        ("persistent/cache", "persistent"),
        ("cache", "cache/persistent"),
    ),
)
def test_sd35_backend_rejects_equal_or_nested_storage_roots(
    tmp_path: Path,
    cache_relative: str,
    persistent_relative: str,
) -> None:
    with pytest.raises(Sd35BackendError, match="bidirectionally disjoint"):
        Sd35PipelineBackend(
            cache_root=tmp_path / cache_relative,
            persistent_root=tmp_path / persistent_relative,
            hf_token=None,
            prompt="probe",
        )


def test_sd35_backend_preparation_binds_frozen_identity(monkeypatch, tmp_path: Path) -> None:
    class Scheduler:
        __module__ = "diffusers"

        def __init__(self):
            self.config = {"frozen": True}
            self.timesteps = torch.arange(20, dtype=torch.float32)

        @classmethod
        def from_config(cls, _config):
            return cls()

        def set_timesteps(self, count, device):
            assert str(device) == "cpu"
            self.timesteps = torch.arange(count, dtype=torch.float32)

        def scale_noise(self, sample, timestep, noise):
            assert timestep.numel() == 1
            assert not torch.is_grad_enabled()
            return sample + noise

    Scheduler.__name__ = "FlowMatchEulerDiscreteScheduler"

    class Posterior:
        def __init__(self, value):
            self.value = value

        def mode(self):
            return self.value

    class Vae:
        config = types.SimpleNamespace(scaling_factor=1.5, shift_factor=0.25)

        def decode(self, latent, return_dict):
            assert return_dict is True
            assert not torch.is_grad_enabled()
            return types.SimpleNamespace(
                sample=torch_functional.interpolate(
                    latent[:, :3],
                    size=(512, 512),
                    mode="nearest",
                )
            )

        def encode(self, image, return_dict):
            assert return_dict is True
            assert not torch.is_grad_enabled()
            mode = torch_functional.interpolate(
                image,
                size=(64, 64),
                mode="nearest",
            )
            mode = mode.repeat(1, 6, 1, 1)[:, :16].to(torch.float16)
            return types.SimpleNamespace(latent_dist=Posterior(mode))

    class ImageProcessor:
        def postprocess(self, value, output_type):
            assert output_type == "pt"
            assert not torch.is_grad_enabled()
            return value

        def preprocess(self, value, height, width):
            assert (height, width) == (512, 512)
            assert not torch.is_grad_enabled()
            return value

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = 1
            self.to_q = torch.nn.Identity()
            self.to_k = torch.nn.Identity()
            self.norm_q = None
            self.norm_k = None

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                Block() for _ in range(24)
            )

        def forward(self, **kwargs):
            assert kwargs["return_dict"] is False
            assert not torch.is_grad_enabled()
            hidden = kwargs["hidden_states"].flatten(2).transpose(1, 2)
            for index in (0, 23):
                attention = self.transformer_blocks[index].attn
                attention.to_q(hidden)
                attention.to_k(hidden)
            return (kwargs["hidden_states"],)

    class Pipeline:
        __module__ = "diffusers"

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            value = cls()
            value.scheduler = Scheduler()
            value.vae = Vae()
            value.transformer = Transformer()
            value.image_processor = ImageProcessor()
            return value

        def to(self, _device):
            return self

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            callback = kwargs["callback_on_step_end"]
            state = {"latents": kwargs["latents"]}
            for index in range(kwargs["num_inference_steps"]):
                state = callback(self, index, torch.tensor(index), state)
            return types.SimpleNamespace(images=state["latents"])

        def encode_prompt(self, **kwargs):
            assert kwargs["do_classifier_free_guidance"] is False
            assert not torch.is_grad_enabled()
            empty = torch.zeros((1, 2, 2), dtype=torch.float16)
            pooled = torch.zeros((1, 2), dtype=torch.float16)
            return empty, None, pooled, None

    Pipeline.__name__ = "StableDiffusion3Pipeline"
    module = types.SimpleNamespace(
        StableDiffusion3Pipeline=Pipeline,
        FlowMatchEulerDiscreteScheduler=Scheduler,
    )
    monkeypatch.setattr(
        "runtime.sd35_backend.importlib.import_module",
        lambda name: module,
    )
    class CpuTestBackend(Sd35PipelineBackend):
        def prepare(self, configuration, selected_device):
            assert selected_device == "cpu"
            identity = super().prepare(configuration, "cuda:0")
            self._device = torch.device("cpu")
            return replace(identity, selected_device="cpu")

    backend = CpuTestBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token="memory-only",
        prompt="probe",
    )
    configuration = load_runtime_configuration()
    adapter = create_runtime_adapter(backend=backend)
    identity = adapter.initialize(requested_device="cpu")
    assert identity.runtime_config_digest == configuration.runtime_config_digest
    assert identity.runtime_backend_name == "diffusers_sd35_pipeline"
    assert identity.callback_index == 18
    latent = torch.ones((1, 16, 64, 64), dtype=torch.float16)
    callback_indices: list[int] = []
    assert backend.run_generation(
        latent,
        lambda index, value: callback_indices.append(index) or value,
    ).shape == latent.shape
    assert callback_indices == list(range(20))
    assert backend.vae_factors().scaling_factor == 1.5
    image = backend.vae_decode(latent)
    assert backend.vae_encode(image).mode().shape == latent.shape
    schedule = backend.create_detection_schedule(20)
    assert schedule.detection_schedule_index == 7
    assert torch.equal(
        backend.scale_detection_noise(
            latent,
            torch.ones_like(latent),
            schedule.detection_timestep,
        ),
        latent + 1,
    )
    assert backend.attention_module("transformer_blocks.23.attn").heads == 1
    forward = backend.run_qk_detection_forward(
        latent,
        schedule.detection_timestep,
        RuntimeDetectionConditioning(
            prompt="",
            prompt_2="",
            prompt_3="",
            do_classifier_free_guidance=False,
            detection_conditioning_protocol=(
                "sd3_empty_text_triplet_without_cfg"
            ),
        ),
    )
    assert forward.qk_layer_names == configuration.qk_layer_names

    carrier = hf_carrier("delivery-e2e-key", tuple(latent.shape))
    paired = adapter.execute_content_write_and_vae(
        latent,
        lambda baseline: content_embedder(baseline, carrier),
    )
    observed = adapter.observe_detection_qk(paired.watermarked_image)
    assert paired.content_materialization_result.budget_status == "accepted"
    assert tuple(
        item.layer_name for item in observed.qk_layer_observations
    ) == configuration.qk_layer_names
    adapter.close()


def _package_manifest(root: Path, revision: str) -> None:
    configuration = root / "configs/runtime/runtime_sd35_flowmatch.json"
    configuration.parent.mkdir(parents=True)
    lock = [
        {"package_name": "python", "version_specifier": ">=3.12"},
        {"package_name": "diffusers", "version_specifier": "0.38.0"},
        {"package_name": "torch", "version_specifier": "2.11.0"},
        {"package_name": "transformers", "version_specifier": "5.12.1"},
        {"package_name": "accelerate", "version_specifier": "1.14.0"},
        {"package_name": "numpy", "version_specifier": "2.0.2"},
        {"package_name": "Pillow", "version_specifier": "11.3.0"},
        {"package_name": "safetensors", "version_specifier": "0.8.0"},
        {"package_name": "huggingface-hub", "version_specifier": "1.20.1"},
    ]
    configuration.write_text(
        json.dumps({"dependency_lock": lock}),
        encoding="utf-8",
    )
    package_files = (
        root / "README.md",
        root / "main/__init__.py",
        root / "runtime/__init__.py",
        root / "pyproject.toml",
        root / "requirements_runtime_qualification.txt",
        root / "scripts/experiment_execution/__init__.py",
        root / "scripts/experiment_execution/runtime_qualification_runner.py",
    )
    for path in package_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    (root / "requirements_runtime_qualification.txt").write_text(
        "\n".join(
            f"{item['package_name']}=={item['version_specifier']}"
            for item in lock
            if item["package_name"] != "python"
        )
        + "\n",
        encoding="utf-8",
    )
    copied = []
    for path in (configuration, *package_files):
        value = path.read_bytes()
        copied.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        )
    (root / "runtime_execution_manifest.json").write_text(
        json.dumps(
            {
                "package_schema_version": 1,
                "profile_name": "experiment_execution_package",
                "package_ready": True,
                "runtime_candidate_revision": revision,
                "copied_files": copied,
                "excluded_parts": sorted(runner.PACKAGE_EXCLUDED_PARTS),
            }
        ),
        encoding="utf-8",
    )


def _versions() -> dict[str, str]:
    return {
        "python": "3.12.9",
        "diffusers": "0.38.0",
        "torch": "2.11.0",
        "transformers": "5.12.1",
        "accelerate": "1.14.0",
        "numpy": "2.0.2",
        "Pillow": "11.3.0",
        "safetensors": "0.8.0",
        "huggingface-hub": "1.20.1",
    }


def _record(
    key_control: str = "registered",
    *,
    run_id: str = "run-001",
    revision: str = "1" * 40,
    seed: int = 20260728,
    prompt_identity: str = runner.PROMPT_IDENTITY,
    prompt_sha256: str = hashlib.sha256(b"probe").hexdigest(),
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "runtime_candidate_revision": revision,
        "runtime_config_digest": "0" * 64,
        "runtime_backend_name": "fake",
        "cuda_available": True,
        "cuda_runtime": "12.8",
        "gpu_name": "Fake GPU",
        "key_control": key_control,
        "key_public_digest": (
            "1" * 64 if key_control == "registered" else "2" * 64
        ),
        "selected_device": "cuda:0",
        "model_id": "model",
        "model_revision": "3" * 40,
        "seed": seed,
        "prompt_identity": prompt_identity,
        "prompt_sha256": prompt_sha256,
        "callback_index": 18,
        "callback_status": "passed",
        "content_relative_l2_nominal": 0.012,
        "content_relative_l2_limit": 0.012,
        "realized_total_l2": 0.5,
        "realized_relative_l2": 0.011,
        "budget_utilization": 0.916,
        "materialization_scale": 1.0,
        "materialization_attempt_count": 1,
        "integrity_status": "passed",
        "budget_status": "accepted",
        "materialization_replay_identity": "4" * 64,
        "paired_base_latent_digest": "5" * 64,
        "vae_scaling_factor_actual": 1.5305,
        "vae_shift_factor_actual": 0.0609,
        "vae_status": "passed",
        "clean_image_sha256": "6" * 64,
        "watermarked_image_sha256": "7" * 64,
        "detection_latent_sha256": "8" * 64,
        "qk_actual_dtype": "float16",
        "qk_status": "passed",
        "qk_layer_names": list(runner.REGISTERED_QK_LAYERS),
        "qk_operator_identities": ["operator-0", "operator-23"],
        "qk_layer_value_digests": [
            {
                "layer_name": runner.REGISTERED_QK_LAYERS[0],
                "query_sha256": "9" * 64,
                "attention_key_sha256": "a" * 64,
            },
            {
                "layer_name": runner.REGISTERED_QK_LAYERS[1],
                "query_sha256": "c" * 64,
                "attention_key_sha256": "d" * 64,
            },
        ],
        "public_noise_domain_digest": "b" * 64,
        "public_noise_values_float32_be_sha256": "b" * 64,
    }


def test_runner_profiles_create_minimal_result_zip(monkeypatch, tmp_path: Path) -> None:
    revision = "1" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    def execute(**kwargs):
        assert kwargs["cache_root"] == ephemeral / "cache"
        assert kwargs["persistent_root"] == persistent
        return _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
        )

    monkeypatch.setattr(runner, "_execute_once", execute)
    ephemeral, persistent, output = _runner_storage(tmp_path, "result")
    result = runner.run_runtime_qualification(
        profile="qualification",
        run_id="run-001",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="qualification-key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "passed"
    assert result["run_id"] == "run-001"
    assert result["result_zip_filename"] == "result.zip"
    assert result["seed"] == 20260728
    assert result["prompt_sha256"] == hashlib.sha256(b"probe").hexdigest()
    assert result["key_controls"] == [
        "registered",
        "registered",
        "negative_identity",
    ]
    assert {
        result["callback_status"],
        result["actual_dtype_status"],
        result["vae_status"],
        result["qk_status"],
        result["determinism_status"],
        result["package_status"],
        result["dependency_status"],
    } == {"passed", "verified"}
    assert result["repetition_count"] == 3
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "environment_summary.json",
            "failures.jsonl",
            "run_summary.json",
            "runtime_checks.jsonl",
        }
        summary = json.loads(archive.read("run_summary.json"))
        environment = json.loads(archive.read("environment_summary.json"))
        records = [
            json.loads(line)
            for line in archive.read("runtime_checks.jsonl").decode().splitlines()
        ]
        assert summary["run_id"] == "run-001"
        assert summary["result_zip_filename"] == "result.zip"
        assert environment["result_schema_version"] == 2
        assert environment["profile"] == "qualification"
        assert environment["run_id"] == summary["run_id"]
        assert (
            environment["runtime_candidate_revision"]
            == summary["runtime_candidate_revision"]
        )
        assert environment["seed"] == summary["seed"]
        assert environment["prompt_identity"] == summary["prompt_identity"]
        assert environment["prompt_sha256"] == summary["prompt_sha256"]
        assert environment["record_digests"] == summary["record_digests"]
        assert environment["key_controls"] == summary["key_controls"]
        assert all(
            record["gpu_name"] == environment["gpu_name"]
            and record["cuda_runtime"] == environment["cuda_runtime"]
            and record["cuda_available"] == environment["cuda_available"]
            for record in records
        )
        assert archive.read("failures.jsonl") == b""


def test_runner_packages_classified_failure(monkeypatch, tmp_path: Path) -> None:
    revision = "2" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)

    def fail(**_kwargs):
        try:
            raise Sd35BackendError("CUDA out of memory")
        except Sd35BackendError as cause:
            raise RuntimeAdapterError("wrapped") from cause

    monkeypatch.setattr(runner, "_execute_once", fail)
    ephemeral, persistent, output = _runner_storage(tmp_path, "failure")
    result = runner.run_runtime_qualification(
        profile="smoke",
        run_id="run-002",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="qualification-key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "failed"
    with zipfile.ZipFile(output) as archive:
        failures = [
            json.loads(line)
            for line in archive.read("failures.jsonl").decode().splitlines()
        ]
    assert failures[0]["failure_class"] == "resource_failure"


def test_runner_packages_preflight_manifest_failure(tmp_path: Path) -> None:
    ephemeral, persistent, output = _runner_storage(
        tmp_path,
        "preflight-failure",
    )
    result = runner.run_runtime_qualification(
        profile="smoke",
        run_id="run-003",
        package_root=tmp_path / "missing-package",
        runtime_candidate_revision="3" * 40,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="qualification-key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "failed"
    assert output.is_file()


@pytest.mark.parametrize(
    ("ephemeral_relative", "persistent_relative"),
    (
        ("shared", "shared"),
        ("persistent/ephemeral", "persistent"),
        ("ephemeral", "ephemeral/persistent"),
    ),
)
def test_runner_rejects_equal_or_nested_storage_roots(
    tmp_path: Path,
    ephemeral_relative: str,
    persistent_relative: str,
) -> None:
    ephemeral = tmp_path / ephemeral_relative
    persistent = tmp_path / persistent_relative
    with pytest.raises(
        runner.QualificationRunnerError,
        match="bidirectionally disjoint",
    ):
        runner.run_runtime_qualification(
            profile="smoke",
            run_id="storage-overlap",
            package_root=tmp_path / "package",
            runtime_candidate_revision="a" * 40,
            result_zip=ephemeral / "result.zip",
            ephemeral_root=ephemeral,
            persistent_root=persistent,
            hf_token=None,
            root_key="key",
            prompt="probe",
        )


def test_runner_rejects_result_outside_ephemeral_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runner.QualificationRunnerError,
        match="strictly within ephemeral_root",
    ):
        runner.run_runtime_qualification(
            profile="smoke",
            run_id="result-outside",
            package_root=tmp_path / "package",
            runtime_candidate_revision="a" * 40,
            result_zip=tmp_path / "outside.zip",
            ephemeral_root=tmp_path / "ephemeral",
            persistent_root=tmp_path / "persistent",
            hf_token=None,
            root_key="key",
            prompt="probe",
        )


@pytest.mark.parametrize(
    ("profile", "source_location", "message"),
    (
        ("replay", None, "replay source is required"),
        ("smoke", "persistent/source.zip", "only allowed for replay"),
        ("qualification", "persistent/source.zip", "only allowed for replay"),
        ("replay", "outside/source.zip", "strictly within persistent_root"),
        ("replay", "persistent", "strictly within persistent_root"),
    ),
)
def test_runner_enforces_profile_specific_replay_source_boundary(
    tmp_path: Path,
    profile: str,
    source_location: str | None,
    message: str,
) -> None:
    ephemeral = tmp_path / "ephemeral"
    persistent = tmp_path / "persistent"
    replay_source = (
        None if source_location is None else tmp_path / source_location
    )
    with pytest.raises(runner.QualificationRunnerError, match=message):
        runner.run_runtime_qualification(
            profile=profile,
            run_id="replay-boundary",
            package_root=tmp_path / "package",
            runtime_candidate_revision="a" * 40,
            result_zip=ephemeral / "result.zip",
            ephemeral_root=ephemeral,
            persistent_root=persistent,
            hf_token=None,
            root_key="key",
            prompt="probe",
            replay_source=replay_source,
        )


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (ContentEmbedderError("hard budget failed"), "budget_failure"),
        (RuntimeContentExecutionError("bitwise replay failed"), "integrity_failure"),
        (RuntimeQkObservationError("missing hook"), "qk_failure"),
        (Sd35BackendError("CUDA out of memory"), "resource_failure"),
    ],
)
def test_failure_classification_follows_adapter_cause(
    cause: BaseException,
    expected: str,
) -> None:
    try:
        raise cause
    except BaseException as inner:
        try:
            raise RuntimeAdapterError("adapter wrapped failure") from inner
        except RuntimeAdapterError as outer:
            assert runner._classify_failure(outer) == expected


def test_failure_classification_follows_implicit_context_chain() -> None:
    try:
        raise RuntimeQkObservationError("registered hook missing")
    except RuntimeQkObservationError:
        try:
            raise RuntimeAdapterError("adapter wrapped without explicit cause")
        except RuntimeAdapterError as outer:
            assert runner._classify_failure(outer) == "qk_failure"


def test_runner_rejects_incomplete_success_record(monkeypatch, tmp_path: Path) -> None:
    revision = "4" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    monkeypatch.setattr(runner, "_execute_once", lambda **_kwargs: {"budget_status": "accepted"})
    ephemeral, persistent, output = _runner_storage(tmp_path, "schema")
    result = runner.run_runtime_qualification(
        profile="smoke",
        run_id="schema-failure",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "failed"
    assert result["failure_classes"] == ["incomplete"]


def test_qualification_classifies_independent_repetition_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    revision = "e" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    calls = 0

    def execute(**kwargs):
        nonlocal calls
        record = _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
        )
        if calls == 1:
            record["gpu_name"] = "Drift GPU"
        calls += 1
        return record

    monkeypatch.setattr(runner, "_execute_once", execute)
    ephemeral, persistent, output = _runner_storage(
        tmp_path,
        "determinism",
    )
    result = runner.run_runtime_qualification(
        profile="qualification",
        run_id="determinism-drift",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=output,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "failed"
    assert result["failure_classes"] == ["determinism_failure"]
    assert result["determinism_status"] == "failed"


def test_dependency_lock_drift_fails_closed(tmp_path: Path) -> None:
    revision = "5" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    versions = _versions()
    versions["torch"] = "0.0.0"
    with pytest.raises(runner.QualificationRunnerError, match="dependency lock drifted"):
        runner.verify_dependency_lock(
            package,
            versions,
        )


def test_dependency_lock_uses_metadata_for_every_frozen_package(
    monkeypatch,
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    calls: list[str] = []
    versions = _versions()

    def version(name: str) -> str:
        calls.append(name)
        return versions[name]

    monkeypatch.setattr(runner.platform, "python_version", lambda: versions["python"])
    monkeypatch.setattr(runner.importlib.metadata, "version", version)
    evidence = runner.verify_dependency_lock(package)
    assert calls == [
        "diffusers",
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "Pillow",
        "safetensors",
        "huggingface-hub",
    ]
    assert {item["package_name"] for item in evidence} == set(versions)


def test_requirements_must_exactly_match_complete_dependency_lock(
    tmp_path: Path,
) -> None:
    revision = "b" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    requirements = package / "requirements_runtime_qualification.txt"
    requirements.write_text(
        requirements.read_text(encoding="utf-8")
        + "unregistered-package==1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(
        runner.QualificationRunnerError,
        match="requirements do not exactly match",
    ):
        runner.verify_dependency_lock(package, _versions())


def test_repository_requirements_match_frozen_dependency_lock() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = runner.verify_dependency_lock(root, _versions())
    assert tuple(item["package_name"] for item in evidence) == tuple(
        package_name for package_name, _version in runner.REGISTERED_DEPENDENCY_LOCK
    )


def test_requirements_reject_same_dependency_set_in_wrong_order(
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    requirements = package / "requirements_runtime_qualification.txt"
    lines = requirements.read_text(encoding="utf-8").splitlines()
    reordered = (lines[1], lines[0], *lines[2:])
    assert set(reordered) == set(lines)
    requirements.write_text("\n".join(reordered) + "\n", encoding="utf-8")
    with pytest.raises(
        runner.QualificationRunnerError,
        match="requirements do not exactly match",
    ):
        runner.verify_dependency_lock(package, _versions())


def test_dependency_lock_must_include_every_frozen_entry(
    tmp_path: Path,
) -> None:
    revision = "f" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    configuration = package / "configs/runtime/runtime_sd35_flowmatch.json"
    payload = json.loads(configuration.read_text(encoding="utf-8"))
    payload["dependency_lock"].pop()
    configuration.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        runner.QualificationRunnerError,
        match="frozen complete lock",
    ):
        runner.verify_dependency_lock(package, _versions())


def test_manifest_rejects_extra_and_tampered_files(tmp_path: Path) -> None:
    revision = "6" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    runner.verify_execution_package(package, revision)
    (package / "extra.py").write_text("extra\n")
    with pytest.raises(
        runner.QualificationRunnerError,
        match="unallowlisted|file set",
    ):
        runner.verify_execution_package(package, revision)
    (package / "extra.py").unlink()
    (package / "README.md").write_text("tampered\n")
    with pytest.raises(runner.QualificationRunnerError, match="identity drifted"):
        runner.verify_execution_package(package, revision)


@pytest.mark.parametrize(
    "manifest_mutation",
    (
        lambda manifest: manifest.update(runtime_candidate_revision="f" * 40),
        lambda manifest: manifest.update(package_ready=False),
        lambda manifest: manifest["copied_files"][0].update(path="/absolute.py"),
        lambda manifest: manifest["copied_files"][0].update(path="C:\\escape.py"),
        lambda manifest: manifest["copied_files"][0].update(path=".env"),
        lambda manifest: manifest["copied_files"][0].update(path="unallowlisted.py"),
    ),
)
def test_manifest_rejects_revision_readiness_and_unsafe_paths(
    manifest_mutation,
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    manifest_path = package / "runtime_execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.QualificationRunnerError):
        runner.verify_execution_package(package, revision)


def test_replay_validates_source_then_reruns(monkeypatch, tmp_path: Path) -> None:
    revision = "7" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)
    calls: list[str] = []

    def execute(**kwargs):
        calls.append(kwargs["key_control"])
        return _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
        )

    monkeypatch.setattr(runner, "_execute_once", execute)
    qualification_ephemeral, persistent, qualification_zip = _runner_storage(
        tmp_path,
        "qualification",
    )
    qualification = runner.run_runtime_qualification(
        profile="qualification",
        run_id="qualification-source",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=qualification_zip,
        ephemeral_root=qualification_ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    replay_source = persistent / "runs" / revision / qualification_zip.name
    replay_source.parent.mkdir(parents=True)
    shutil.copy2(qualification_zip, replay_source)
    calls.clear()
    replay_ephemeral = tmp_path / "replay-ephemeral"
    replay = runner.run_runtime_qualification(
        profile="replay",
        run_id="replay-run",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=replay_ephemeral / "replay.zip",
        ephemeral_root=replay_ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        replay_source=replay_source,
        supplied_dependency_versions=_versions(),
    )
    assert qualification["run_status"] == replay["run_status"] == "passed"
    assert calls == ["registered", "registered", "negative_identity"]
    assert replay["replay_source_record_digests"] == qualification["record_digests"]


def _rewrite_result_zip(
    source: Path,
    target: Path,
    mutations: dict[str, bytes],
) -> None:
    target.parent.mkdir(parents=True)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as output:
        for info in original.infolist():
            output.writestr(
                info,
                mutations.get(info.filename, original.read(info.filename)),
            )


def _qualification_source(
    monkeypatch,
    tmp_path: Path,
    revision: str,
) -> Path:
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)

    def execute(**kwargs):
        return _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
            seed=kwargs["seed"],
            prompt_sha256=hashlib.sha256(
                kwargs["prompt"].encode("utf-8")
            ).hexdigest(),
        )

    monkeypatch.setattr(runner, "_execute_once", execute)
    ephemeral, persistent, source = _runner_storage(tmp_path, "source")
    result = runner.run_runtime_qualification(
        profile="qualification",
        run_id="source-run",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=source,
        ephemeral_root=ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert result["run_status"] == "passed"
    return source


@pytest.mark.parametrize(
    ("member_name", "mutation", "message"),
    (
        (
            "environment_summary.json",
            lambda value: {**value, "gpu_name": "tampered-gpu"},
            "environment or failures drifted",
        ),
        (
            "failures.jsonl",
            lambda _value: {
                "failure_class": "runtime_failure",
                "exception_type": "Injected",
                "message": "injected",
            },
            "environment or failures drifted",
        ),
    ),
)
def test_replay_rejects_environment_tamper_and_failure_injection(
    monkeypatch,
    tmp_path: Path,
    member_name: str,
    mutation,
    message: str,
) -> None:
    revision = "8" * 40
    source = _qualification_source(monkeypatch, tmp_path, revision)
    with zipfile.ZipFile(source) as archive:
        if member_name.endswith(".json"):
            original = json.loads(archive.read(member_name))
            replacement = (
                json.dumps(mutation(original), sort_keys=True) + "\n"
            ).encode()
        else:
            replacement = (
                json.dumps(mutation(None), sort_keys=True) + "\n"
            ).encode()
    tampered = tmp_path / "tampered" / source.name
    _rewrite_result_zip(source, tampered, {member_name: replacement})
    with pytest.raises(runner.QualificationRunnerError, match=message):
        runner._load_replay_source(
            tampered,
            revision,
            hashlib.sha256(b"probe").hexdigest(),
            20260728,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", 20260729),
        ("prompt_identity", "runtime_qualification_prompt_tampered"),
        ("prompt_sha256", "f" * 64),
    ),
)
def test_replay_rejects_summary_record_request_identity_mismatch(
    monkeypatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    revision = "9" * 40
    source = _qualification_source(monkeypatch, tmp_path, revision)
    with zipfile.ZipFile(source) as archive:
        summary = json.loads(archive.read("run_summary.json"))
        environment = json.loads(archive.read("environment_summary.json"))
        records = [
            json.loads(line)
            for line in archive.read("runtime_checks.jsonl").decode().splitlines()
        ]
    records[0][field] = value
    digests = [runner._record_digest(record) for record in records]
    summary["checks"] = records
    summary["record_digests"] = digests
    environment["record_digests"] = digests
    mutations = {
        "run_summary.json": (
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "environment_summary.json": (
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "runtime_checks.jsonl": "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in records
        ).encode(),
    }
    tampered = tmp_path / field / source.name
    _rewrite_result_zip(source, tampered, mutations)
    with pytest.raises(
        runner.QualificationRunnerError,
        match="required success semantics",
    ):
        runner._load_replay_source(
            tampered,
            revision,
            hashlib.sha256(b"probe").hexdigest(),
            20260728,
        )


def test_replay_rejects_record_bytes_that_drift_from_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    revision = "d" * 40
    package = tmp_path / "package"
    package.mkdir()
    _package_manifest(package, revision)

    def execute(**kwargs):
        return _record(
            kwargs["key_control"],
            run_id=kwargs["run_id"],
            revision=kwargs["runtime_candidate_revision"],
        )

    monkeypatch.setattr(runner, "_execute_once", execute)
    source_ephemeral, persistent, source = _runner_storage(
        tmp_path,
        "record-source",
    )
    source_result = runner.run_runtime_qualification(
        profile="qualification",
        run_id="source-run",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=source,
        ephemeral_root=source_ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        supplied_dependency_versions=_versions(),
    )
    assert source_result["run_status"] == "passed"
    tampered = persistent / "tampered" / source.name
    with zipfile.ZipFile(source) as original:
        record_payload = original.read("runtime_checks.jsonl").replace(
            b'"gpu_name": "Fake GPU"',
            b'"gpu_name": "Drift GPU"',
            1,
        )
    _rewrite_result_zip(
        source,
        tampered,
        {"runtime_checks.jsonl": record_payload},
    )
    replay_ephemeral = tmp_path / "record-replay-ephemeral"
    replay = runner.run_runtime_qualification(
        profile="replay",
        run_id="replay-run",
        package_root=package,
        runtime_candidate_revision=revision,
        result_zip=replay_ephemeral / "replay.zip",
        ephemeral_root=replay_ephemeral,
        persistent_root=persistent,
        hf_token=None,
        root_key="key",
        prompt="probe",
        replay_source=tampered,
        supplied_dependency_versions=_versions(),
    )
    assert replay["run_status"] == "failed"
    assert replay["failure_classes"] == ["incomplete"]


@pytest.mark.parametrize(
    ("result", "expected"),
    (
        ({"run_status": "passed", "failure_classes": []}, 0),
        ({"run_status": "failed", "failure_classes": ["runtime_failure"]}, 1),
        ({"run_status": "failed", "failure_classes": ["incomplete"]}, 2),
    ),
)
def test_cli_exit_code_matches_result_status(
    monkeypatch,
    tmp_path: Path,
    result: dict[str, object],
    expected: int,
) -> None:
    monkeypatch.setattr(
        runner,
        "run_runtime_qualification",
        lambda **_kwargs: result,
    )
    ephemeral = tmp_path / "cli-ephemeral"
    assert runner.main(
        [
            "--run-id",
            "cli-run",
            "--result-zip",
            str(ephemeral / "result.zip"),
            "--ephemeral-root",
            str(ephemeral),
            "--persistent-root",
            str(tmp_path / "cli-persistent"),
        ]
    ) == expected


def test_cli_requires_run_id() -> None:
    with pytest.raises(SystemExit) as exc:
        runner.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "missing_option",
    ("--result-zip", "--ephemeral-root", "--persistent-root"),
)
def test_cli_requires_explicit_storage_arguments(
    tmp_path: Path,
    missing_option: str,
) -> None:
    ephemeral = tmp_path / "cli-required-ephemeral"
    arguments = [
        "--run-id",
        "cli-required",
        "--result-zip",
        str(ephemeral / "result.zip"),
        "--ephemeral-root",
        str(ephemeral),
        "--persistent-root",
        str(tmp_path / "cli-required-persistent"),
    ]
    option_index = arguments.index(missing_option)
    del arguments[option_index : option_index + 2]
    with pytest.raises(SystemExit) as exc:
        runner.main(arguments)
    assert exc.value.code == 2


def test_execute_once_passes_persistent_root_to_backend_factory(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FactoryStop(RuntimeError):
        pass

    def factory(**kwargs):
        captured.update(kwargs)
        raise FactoryStop

    with pytest.raises(FactoryStop):
        runner._execute_once(
            backend_factory=factory,
            cache_root=tmp_path / "cache",
            persistent_root=tmp_path / "persistent",
            hf_token=None,
            root_key="key",
            prompt="probe",
            seed=1,
            key_control="registered",
            run_id="factory-boundary",
            runtime_candidate_revision="a" * 40,
        )
    assert captured["cache_root"] == tmp_path / "cache"
    assert captured["persistent_root"] == tmp_path / "persistent"


def test_delivery_python_sources_have_no_scanned_local_absolute_path() -> None:
    root = Path(__file__).resolve().parents[2]
    scanned_paths = (
        root / "runtime/sd35_backend.py",
        root / "scripts/experiment_execution/runtime_qualification_runner.py",
    )
    local_prefixes = (
        "/home/",
        "/Users/",
        "/mnt/",
        "/content/",
        "/tmp/",
        "/var/",
        "/opt/",
        "/root/",
    )
    for path in scanned_paths:
        source = path.read_text(encoding="utf-8")
        assert not any(prefix in source for prefix in local_prefixes)


def _write_package_fixture(repo: Path) -> None:
    dependency_lock = [
        {"package_name": package_name, "version_specifier": version}
        for package_name, version in runner.REGISTERED_DEPENDENCY_LOCK
    ]
    for relative in (
        "main/__init__.py",
        "runtime/__init__.py",
        "configs/runtime/runtime_sd35_flowmatch.json",
        "scripts/experiment_execution/__init__.py",
        "scripts/experiment_execution/README.md",
        "scripts/experiment_execution/runtime_qualification_runner.py",
        "pyproject.toml",
        "requirements_runtime_qualification.txt",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "configs/runtime/runtime_sd35_flowmatch.json":
            path.write_text(
                json.dumps({"dependency_lock": dependency_lock}) + "\n",
                encoding="utf-8",
            )
        elif relative == "requirements_runtime_qualification.txt":
            path.write_text(
                "\n".join(
                    f"{package_name}=={version}"
                    for package_name, version in runner.REGISTERED_DEPENDENCY_LOCK
                    if package_name != "python"
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text("# fixture\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_package_builder_requires_clean_exact_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_package_fixture(repo)
    (
        repo
        / "scripts/experiment_execution/runtime_qualification_bootstrap.py"
    ).write_text("# package-external bootstrap\n", encoding="utf-8")
    (repo / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=CEG-WM Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = _git(repo, "rev-parse", "HEAD")
    ignored = repo / "runtime/ignored.pyc"
    ignored.write_bytes(b"ignored working-tree cache")
    assert _git(repo, "status", "--porcelain") == ""
    output = tmp_path / "package.zip"
    result = build_runtime_qualification_package(
        root=repo,
        output_zip=output,
        runtime_candidate_revision=revision,
    )
    assert result["package_ready"] is True
    assert result["runtime_candidate_revision"] == revision
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("runtime_execution_manifest.json"))
        assert manifest["package_ready"] is True
        assert "README.md" in archive.namelist()
        assert "runtime/ignored.pyc" not in archive.namelist()
        assert (
            "scripts/experiment_execution/build_runtime_qualification_package.py"
            not in archive.namelist()
        )
        assert (
            "scripts/experiment_execution/runtime_qualification_bootstrap.py"
            not in archive.namelist()
        )
        assert not any(name.startswith(".codex/") for name in archive.namelist())
        for entry in manifest["copied_files"]:
            payload = archive.read(entry["path"])
            assert len(payload) == entry["size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    with pytest.raises(PackageBuildError, match="does not equal HEAD"):
        build_runtime_qualification_package(
            root=repo,
            output_zip=tmp_path / "wrong-revision.zip",
            runtime_candidate_revision="0" * 40,
        )
    (repo / "runtime/__init__.py").write_text("# drift\n")
    with pytest.raises(PackageBuildError, match="clean"):
        build_runtime_qualification_package(
            root=repo,
            output_zip=tmp_path / "dirty.zip",
            runtime_candidate_revision=revision,
        )


def test_package_builder_rejects_wrong_requirement_order_without_output(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_package_fixture(repo)
    requirements = repo / "requirements_runtime_qualification.txt"
    lines = requirements.read_text(encoding="utf-8").splitlines()
    reordered = (lines[1], lines[0], *lines[2:])
    assert set(reordered) == set(lines)
    requirements.write_text("\n".join(reordered) + "\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=CEG-WM Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = _git(repo, "rev-parse", "HEAD")
    output = tmp_path / "delivery" / "package.zip"
    with pytest.raises(
        PackageBuildError,
        match="requirements do not exactly match",
    ):
        build_runtime_qualification_package(
            root=repo,
            output_zip=output,
            runtime_candidate_revision=revision,
        )
    assert not output.exists()
    assert not output.parent.exists()


def test_package_builder_rejects_local_absolute_path_blob(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_package_fixture(repo)
    (repo / "runtime/local_path.py").write_text(
        'PATH = "/home/example/private"\n',
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=CEG-WM Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(PackageBuildError, match="local absolute path"):
        build_runtime_qualification_package(
            root=repo,
            output_zip=tmp_path / "package.zip",
            runtime_candidate_revision=revision,
        )


def test_built_package_unpacks_and_runs_independently(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_package_fixture(repo)
    (repo / "scripts/experiment_execution/runtime_qualification_runner.py").write_bytes(
        (
            source_root
            / "scripts/experiment_execution/runtime_qualification_runner.py"
        ).read_bytes()
    )
    (repo / "main/__init__.py").write_text(
        'PACKAGE_TEST_VALUE = "main"\n',
        encoding="utf-8",
    )
    (repo / "runtime/__init__.py").write_text(
        'PACKAGE_TEST_VALUE = "runtime"\n',
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=CEG-WM Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = _git(repo, "rev-parse", "HEAD")
    package_zip = tmp_path / "package.zip"
    build_runtime_qualification_package(
        root=repo,
        output_zip=package_zip,
        runtime_candidate_revision=revision,
    )
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(package_zip) as archive:
        archive.extractall(unpacked)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import main, runtime; "
                "assert main.PACKAGE_TEST_VALUE == 'main'; "
                "assert runtime.PACKAGE_TEST_VALUE == 'runtime'"
            ),
        ],
        cwd=unpacked,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    invoked = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.experiment_execution.runtime_qualification_runner",
            "--help",
        ],
        cwd=unpacked,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invoked.returncode == 0, invoked.stderr
    assert "--run-id" in invoked.stdout
    assert not list(unpacked.rglob("__pycache__"))

    (unpacked / "main/__init__.py").write_text(
        "raise AssertionError('main imported before package verification')\n",
        encoding="utf-8",
    )
    preimport_ephemeral = tmp_path / "preimport-ephemeral"
    preimport_persistent = tmp_path / "preimport-persistent"
    failure_zip = preimport_ephemeral / "preimport-failure.zip"
    preimport = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts.experiment_execution.runtime_qualification_runner "
                "import run_runtime_qualification; "
                "result = run_runtime_qualification("
                "profile='smoke', run_id='preimport-check', package_root='.', "
                f"runtime_candidate_revision='{revision}', "
                f"result_zip={str(failure_zip)!r}, "
                f"ephemeral_root={str(preimport_ephemeral)!r}, "
                f"persistent_root={str(preimport_persistent)!r}, "
                "hf_token=None, root_key='test-key', prompt='probe', "
                "supplied_dependency_versions={}); "
                "assert result['run_status'] == 'failed'; "
                "assert result['failure_classes'] == ['incomplete']"
            ),
        ],
        cwd=unpacked,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert preimport.returncode == 0, preimport.stderr
    assert failure_zip.is_file()


def test_notebook_is_unique_thin_and_output_free() -> None:
    root = Path(__file__).resolve().parents[2]
    notebooks = list((root / "notebooks").rglob("*.ipynb"))
    assert notebooks == [root / "notebooks/colab/runtime_qualification.ipynb"]
    document = json.loads(notebooks[0].read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in document["cells"]
    )
    assert all(cell.get("execution_count") is None for cell in document["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("outputs", []) == [] for cell in document["cells"])
    assert 5 <= len(document["cells"]) <= 7
    assert "runtime_qualification_bootstrap.py" in sources
    assert "HF_TOKEN" in sources
    assert "CEG_WM_ROOT_KEY" in sources
    assert "/content/drive/MyDrive/CEG-WM/runtime_qualification" in sources
    assert "EXPECTED_PACKAGE_SHA256 = input(" in sources
    assert "PROFILE = (input(" in sources
    assert "REPLAY_SOURCE = input(" in sources
    assert '"--expected-package-sha256"' in sources
    assert '"--persistent-root"' in sources
    assert "completed.returncode" in sources
    assert "completed.returncode not in (0, 1, 2, 3)" in sources
    assert "EXPECTED_BOOTSTRAP_SHA256" in sources
    assert "hashlib.sha256" in sources
    assert 'pathlib.Path(BOOTSTRAP).read_bytes()' in sources
    assert 'open("xb")' in sources
    assert "str(TRUSTED_BOOTSTRAP)" in sources
    assert sources.count("pathlib.Path(BOOTSTRAP).read_bytes()") == 1
    bootstrap_path = (
        root
        / "scripts/experiment_execution/runtime_qualification_bootstrap.py"
    )
    bootstrap_digest = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
    assert f'EXPECTED_BOOTSTRAP_SHA256 = "{bootstrap_digest}"' in sources
    assert sources.index("hashlib.sha256") < sources.index(
        "subprocess.run(command"
    )
    assert sources.index("TRUSTED_BOOTSTRAP =") < sources.index(
        "subprocess.run(command"
    )
    for forbidden in (
        "runtime_qualification_runner",
        "runtime_execution_manifest",
        "package_schema_version",
        "result_schema_version",
        "zipfile",
        "extractall",
        "pip install",
        "content_embedder",
        "hf_carrier",
        "to_q",
        "to_k",
        "tau_actual_budget",
        "from_pretrained",
    ):
        assert forbidden not in sources


def test_notebook_runtime_inputs_do_not_require_source_changes() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook = root / "notebooks/colab/runtime_qualification.ipynb"
    before = hashlib.sha256(notebook.read_bytes()).hexdigest()
    document = json.loads(notebook.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )
    assert 'input("Profile [smoke]:' in sources
    assert 'input(f"Execution package path [' in sources
    assert 'input("Expected package SHA-256:' in sources
    assert 'input("Replay source path [none]:' in sources
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == before


def _notebook_cell_source(marker: str) -> str:
    root = Path(__file__).resolve().parents[2]
    document = json.loads(
        (
            root / "notebooks/colab/runtime_qualification.ipynb"
        ).read_text(encoding="utf-8")
    )
    return next(
        "".join(cell.get("source", []))
        for cell in document["cells"]
        if marker in "".join(cell.get("source", []))
    )


def _install_fake_colab_userdata(monkeypatch) -> None:
    google_module = types.ModuleType("google")
    colab_module = types.ModuleType("google.colab")
    colab_module.userdata = types.SimpleNamespace(get=lambda _name: "memory-only")
    google_module.colab = colab_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)


def test_notebook_bootstrap_digest_mismatch_runs_no_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_colab_userdata(monkeypatch)
    source = tmp_path / "bootstrap.py"
    source.write_bytes(b"untrusted")
    subprocess_calls: list[object] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )
    namespace = {
        "BOOTSTRAP": str(source),
        "EXPECTED_BOOTSTRAP_SHA256": "0" * 64,
        "PROFILE": "smoke",
        "CONTENT_ROOT": str(tmp_path),
    }
    with pytest.raises(RuntimeError, match="bootstrap SHA-256 mismatch"):
        exec(_notebook_cell_source("bootstrap_payload ="), namespace)
    assert subprocess_calls == []
    assert not list(tmp_path.glob("ceg_wm_trusted_bootstrap_*"))


def test_notebook_executes_verified_local_snapshot_after_drive_source_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_colab_userdata(monkeypatch)
    drive_source = tmp_path / "drive_bootstrap.py"
    trusted_bytes = b"print('trusted bootstrap')\n"
    drive_source.write_bytes(trusted_bytes)
    digest = hashlib.sha256(trusted_bytes).hexdigest()

    def gpu_check(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, "Fake GPU, 1 MiB\n", "")

    monkeypatch.setattr(subprocess, "run", gpu_check)
    namespace = {
        "BOOTSTRAP": str(drive_source),
        "EXPECTED_BOOTSTRAP_SHA256": digest,
        "PROFILE": "smoke",
        "CONTENT_ROOT": str(tmp_path),
    }
    exec(_notebook_cell_source("bootstrap_payload ="), namespace)
    trusted_snapshot = Path(namespace["TRUSTED_BOOTSTRAP"])
    assert trusted_snapshot.is_file()
    assert trusted_snapshot.read_bytes() == trusted_bytes
    assert trusted_snapshot.parent != drive_source.parent

    drive_source.write_bytes(b"replacement after verification\n")
    observed_commands: list[tuple[str, ...]] = []

    def bootstrap_call(command, **_kwargs):
        observed_commands.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "artifact_kind": "qualification_result",
                    "profile": "smoke",
                    "run_status": "passed",
                    "result_zip": "/persistent/result.zip",
                }
            ),
            "",
        )

    monkeypatch.setattr(subprocess, "run", bootstrap_call)
    namespace.update(
        {
            "PACKAGE_ZIP": "/persistent/package.zip",
            "EXPECTED_PACKAGE_SHA256": "1" * 64,
            "REPLAY_SOURCE": None,
            "DRIVE_ROOT": "/persistent",
        }
    )
    exec(_notebook_cell_source("command = [sys.executable"), namespace)
    assert observed_commands[0][1] == str(trusted_snapshot)
    assert observed_commands[0][1] != str(drive_source)
    assert trusted_snapshot.read_bytes() == trusted_bytes
