"""Git-less and persistence tests for semantic-texture Phase A delivery."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Mapping
import zipfile

import pytest

from scripts.experiment_execution import (
    build_semantic_texture_operational_preflight_package as builder,
)
from scripts.experiment_execution import (
    semantic_texture_operational_preflight_bootstrap as bootstrap,
)
from scripts.experiment_execution import (
    semantic_texture_operational_preflight_entrypoint as entrypoint,
)
from scripts.experiment_execution import (
    semantic_texture_operational_preflight_server as server,
)
from experiments.runners import semantic_texture_operational_preflight as preflight


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _committed_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    for relative in sorted(builder.EXACT_SOURCE_FILES):
        source = ROOT / relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "CEG-WM Phase A Test")
    _git(repository, "config", "user.email", "phase-a@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "phase-a fixture")
    revision = _git(repository, "rev-parse", "HEAD")
    assert len(revision) == 40
    assert _git(repository, "status", "--porcelain=v1") == ""
    return repository, revision


def _built_package(tmp_path: Path) -> dict[str, object]:
    repository, revision = _committed_repository(tmp_path)
    output = tmp_path / "delivery" / "semantic-texture-phase-a.zip"
    result = builder.build_semantic_texture_operational_preflight_package(
        repository_root=repository,
        source_revision=revision,
        output=output,
    )
    return {
        "build": result,
        "manifest": Path(result["delivery_manifest_path"]),
        "output": output,
        "repository": repository,
        "revision": revision,
    }


def test_semantic_texture_preflight_package_is_exact_gitless_and_excludes_outer_layers(
    tmp_path: Path,
) -> None:
    fixture = _built_package(tmp_path)
    with zipfile.ZipFile(fixture["output"]) as archive:
        names = set(archive.namelist())
        embedded = json.loads(archive.read(builder.EMBEDDED_MANIFEST_PATH))
    assert names == {
        *(builder.SOURCE_TO_ARCHIVE_PATH.get(path, path) for path in builder.EXACT_SOURCE_FILES),
        builder.EMBEDDED_MANIFEST_PATH,
    }
    assert builder.ENTRYPOINT_PATH in names
    assert builder.SERVER_PATH in names
    assert "scripts/experiment_execution/build_semantic_texture_operational_preflight_package.py" not in names
    assert "scripts/experiment_execution/semantic_texture_operational_preflight_bootstrap.py" not in names
    assert not any(
        forbidden in PurePosixPath(name).parts
        for name in names
        for forbidden in (
            ".agents",
            ".codex",
            "governance",
            "notebooks",
            "outputs",
            "tests",
        )
    )
    assert embedded["package_ready"] is True
    with zipfile.ZipFile(fixture["output"]) as archive:
        requirements_blob = archive.read(
            "requirements_semantic_texture_operational_preflight.txt"
        )
    assert len(requirements_blob.decode("utf-8").splitlines()) == 62
    assert sha256(requirements_blob).hexdigest() == (
        "07a4c1bbe6fc5e7e6b38334c5a9919a8565b810a9aae7820b61c24cee91270de"
    )
    extract_root = tmp_path / "gitless-package"
    code, result = bootstrap.run_semantic_texture_operational_preflight_bootstrap(
        archive=fixture["output"],
        manifest=fixture["manifest"],
        expected_sha256=fixture["build"]["archive_sha256"],
        expected_size=fixture["build"]["archive_size_bytes"],
        extract_root=extract_root,
        entrypoint_args=("--help",),
        environment={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert code == 0
    assert result["status"] == "passed"
    assert not (extract_root / ".git").exists()


def test_semantic_texture_preflight_package_rebuild_is_deterministic(
    tmp_path: Path,
) -> None:
    fixture = _built_package(tmp_path)
    second = tmp_path / "second" / "semantic-texture-phase-a.zip"
    rebuilt = builder.build_semantic_texture_operational_preflight_package(
        repository_root=fixture["repository"],
        source_revision=fixture["revision"],
        output=second,
    )
    assert second.read_bytes() == Path(fixture["output"]).read_bytes()
    assert rebuilt["archive_sha256"] == fixture["build"]["archive_sha256"]
    assert Path(rebuilt["delivery_manifest_path"]).read_bytes() == Path(
        fixture["manifest"]
    ).read_bytes()


@pytest.mark.parametrize(
    "failure_stage",
    ("pre_trust_integrity", "post_extract_environment"),
)
def test_semantic_texture_preflight_bootstrap_persists_result_before_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    fixture = _built_package(tmp_path)
    extract_root = tmp_path / "blocked-gitless-package"
    expected_sha256 = fixture["build"]["archive_sha256"]
    entrypoint_arguments = (
        "--execute",
        "--source-revision",
        fixture["revision"],
        "--run-id",
        "dependency-blocked",
        "--package-identity",
        fixture["build"]["package_identity"],
        "--output-root",
        str(tmp_path / "operational-output"),
    )
    if failure_stage == "pre_trust_integrity":
        expected_sha256 = "0" * 64
        entrypoint_arguments = ("--help",)
    else:
        def fail_dependency_preparation(
            package_destination: Path,
            execution_environment: Mapping[str, str],
        ) -> dict[str, str]:
            assert package_destination.is_dir()
            assert execution_environment["PYTHONDONTWRITEBYTECODE"] == "1"
            raise bootstrap.SemanticTextureBootstrapError(
                "dependency_identity",
                "test-only dependency failure",
            )

        monkeypatch.setattr(
            bootstrap,
            "_prepare_production_environment",
            fail_dependency_preparation,
        )
    code, result = bootstrap.run_semantic_texture_operational_preflight_bootstrap(
        archive=fixture["output"],
        manifest=fixture["manifest"],
        expected_sha256=expected_sha256,
        expected_size=fixture["build"]["archive_size_bytes"],
        extract_root=extract_root,
        entrypoint_args=entrypoint_arguments,
        environment={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert code != 0
    assert result["status"] == "blocked"
    assert result["blocked_class"] == (
        "integrity_blocked"
        if failure_stage == "pre_trust_integrity"
        else "environment_blocked"
    )
    delivery_root = extract_root.with_name(extract_root.name + ".transport")
    assert {path.name for path in delivery_root.iterdir()} == {
        bootstrap.DELIVERY_COMPLETION_CHECKSUMS_FILENAME,
        bootstrap.TRANSPORT_ARCHIVE_FILENAME,
        bootstrap.TRANSPORT_RECEIPT_FILENAME,
        bootstrap.TRANSPORT_RESULT_FILENAME,
    }
    with zipfile.ZipFile(delivery_root / bootstrap.TRANSPORT_ARCHIVE_FILENAME) as archive:
        assert archive.namelist() == [bootstrap.TRANSPORT_RESULT_FILENAME]
    sums = (
        delivery_root / bootstrap.DELIVERY_COMPLETION_CHECKSUMS_FILENAME
    ).read_text(
        encoding="ascii"
    ).splitlines()
    assert [line.split("  ", 1)[1] for line in sums] == [
        bootstrap.TRANSPORT_RESULT_FILENAME,
        bootstrap.TRANSPORT_ARCHIVE_FILENAME,
        bootstrap.TRANSPORT_RECEIPT_FILENAME,
    ]
    for line in sums:
        expected_digest, artifact_name = line.split("  ", 1)
        assert sha256((delivery_root / artifact_name).read_bytes()).hexdigest() == (
            expected_digest
        )
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in delivery_root.iterdir()
        if path.suffix != ".zip"
    )
    assert "unit_outcomes" not in persisted
    assert "sanitized_error_message" not in persisted


@dataclass(frozen=True)
class _BlockedResult:
    value: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return dict(self.value)


class _DeliveredBlocked(RuntimeError):
    pass


def test_semantic_texture_preflight_server_finalizes_result_zip_receipt_before_raise(
    tmp_path: Path,
) -> None:
    run_id = "semantic-texture-phase-a"
    value = {
        "aggregate": None,
        "asset_authority_status": "identity_blocked",
        "blocked_class": "identity_blocked",
        "candidate_promoted": False,
        "configuration_digest": "1" * 64,
        "formal_tau_created": False,
        "inspyrenet_checkpoint_revision": (
            "d94c2baaa4d023ab018c6f97be6ef37548e3bd1f"
        ),
        "inspyrenet_checkpoint_sha256": (
            "0a6fe2a73ab0532d6d0b8d82849a9760a226df719e3063d09b4149ece6f80fcd"
        ),
        "inspyrenet_checkpoint_size_bytes": 367520613,
        "inspyrenet_source_revision": (
            "f0fa91701a98cfc8e955c554e84522f365ec6da3"
        ),
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "package_identity": "2" * 64,
        "profile_id": "semantic_texture_operational_preflight",
        "result_identity": "3" * 64,
        "run_id": run_id,
        "schema_version": 1,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "source_revision": "4" * 40,
        "status": "blocked",
        "unit_outcomes": [
            {
                "blocked_class": None,
                "elapsed_seconds": 0.5,
                "public_result_identity": "5" * 64,
                "sanitized_error_category": None,
                "sanitized_error_message": None,
                "sanitized_trace_tail": [],
                "started": True,
                "status": "passed",
                "unit_id": "semantic_texture_write_operational",
                "witness_identity": "6" * 64,
            },
            {
                "blocked_class": "identity_blocked",
                "elapsed_seconds": 0.25,
                "public_result_identity": None,
                "sanitized_error_category": "identity_blocked",
                "sanitized_error_message": None,
                "sanitized_trace_tail": [],
                "started": True,
                "status": "blocked",
                "unit_id": "semantic_texture_blind_detection_operational",
                "witness_identity": None,
            },
        ],
    }
    assert "diagnostics" not in inspect.signature(
        server.finalize_semantic_texture_operational_preflight_delivery
    ).parameters
    output_root = tmp_path / "delivery"
    receipt: dict[str, object] = {}
    with pytest.raises(_DeliveredBlocked):
        exit_code, receipt = server.finalize_semantic_texture_operational_preflight_delivery(
            _BlockedResult(value),
            output_root=output_root,
        )
        assert exit_code != 0
        raise _DeliveredBlocked("raise only after complete delivery")
    result_path = output_root / server.RESULT_FILENAME
    archive_path = output_root / receipt["archive_filename"]
    receipt_path = output_root / server.RECEIPT_FILENAME
    delivery_completion_checksums_path = (
        output_root / server.DELIVERY_COMPLETION_CHECKSUMS_FILENAME
    )
    assert all(
        path.is_file()
        for path in (
            result_path,
            archive_path,
            receipt_path,
            delivery_completion_checksums_path,
        )
    )
    assert receipt["archive_sha256"] == sha256(archive_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == [server.RESULT_FILENAME]
    persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted_receipt["archive_sha256"] == receipt["archive_sha256"]
    assert persisted_receipt["model_revision"] == value["model_revision"]
    assert persisted_receipt["inspyrenet_source_revision"] == value[
        "inspyrenet_source_revision"
    ]
    assert persisted_receipt["inspyrenet_checkpoint_sha256"] == value[
        "inspyrenet_checkpoint_sha256"
    ]
    assert not any(
        str(tmp_path) in json.dumps(document)
        for document in (
            json.loads(result_path.read_text(encoding="utf-8")),
            persisted_receipt,
        )
    )
    sums = delivery_completion_checksums_path.read_text(
        encoding="ascii"
    ).splitlines()
    assert [line.split("  ", 1)[1] for line in sums] == [
        result_path.name,
        archive_path.name,
        receipt_path.name,
    ]
    invalid_values: list[dict[str, object]] = []
    unknown_top_level = json.loads(json.dumps(value))
    unknown_top_level["private_error_text"] = "Drive token"
    invalid_values.append(unknown_top_level)
    missing_top_level = json.loads(json.dumps(value))
    del missing_top_level["configuration_digest"]
    invalid_values.append(missing_top_level)
    unknown_nested = json.loads(json.dumps(value))
    unknown_nested["unit_outcomes"][0]["private_state"] = "secret"
    invalid_values.append(unknown_nested)
    persisted_message = json.loads(json.dumps(value))
    persisted_message["unit_outcomes"][1]["sanitized_error_message"] = (
        "local failure"
    )
    invalid_values.append(persisted_message)
    persisted_trace = json.loads(json.dumps(value))
    persisted_trace["unit_outcomes"][1]["sanitized_trace_tail"] = [
        "chained secret"
    ]
    invalid_values.append(persisted_trace)
    mismatched_category = json.loads(json.dumps(value))
    mismatched_category["unit_outcomes"][1]["sanitized_error_category"] = (
        "implementation_blocked"
    )
    invalid_values.append(mismatched_category)
    malformed_top_identity = json.loads(json.dumps(value))
    malformed_top_identity["configuration_digest"] = "A" * 64
    invalid_values.append(malformed_top_identity)
    malformed_unit_identity = json.loads(json.dumps(value))
    malformed_unit_identity["unit_outcomes"][0]["witness_identity"] = "6" * 63
    invalid_values.append(malformed_unit_identity)
    malformed_status = json.loads(json.dumps(value))
    malformed_status["unit_outcomes"][1]["status"] = "passed"
    invalid_values.append(malformed_status)
    malformed_roster = json.loads(json.dumps(value))
    malformed_roster["unit_outcomes"].reverse()
    invalid_values.append(malformed_roster)
    malformed_timing = json.loads(json.dumps(value))
    malformed_timing["unit_outcomes"][0]["elapsed_seconds"] = -0.1
    invalid_values.append(malformed_timing)
    for invalid_index, invalid_value in enumerate(invalid_values):
        invalid_output_root = tmp_path / f"invalid-delivery-{invalid_index}"
        with pytest.raises(server.SemanticTextureOperationalServerError):
            server.finalize_semantic_texture_operational_preflight_delivery(
                _BlockedResult(invalid_value),
                output_root=invalid_output_root,
            )
        assert not invalid_output_root.exists()


def test_semantic_texture_operational_entrypoint_constructs_registered_public_runtime_stack_and_two_unit_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[tuple[str, object]] = []

    class FakeTensor:
        def to(self, *, device: str, dtype: object):
            constructor_calls.append(("latent_to", (device, dtype)))
            return self

    class FakeGenerator:
        def __init__(self, *, device: str) -> None:
            constructor_calls.append(("generator_device", device))

        def manual_seed(self, seed: int) -> None:
            constructor_calls.append(("generation_seed", seed))

    class FakeRuntimeAdapter:
        def initialize(self, requested_device: str):
            constructor_calls.append(("runtime_device", requested_device))
            return SimpleNamespace(image_height=512, image_width=512)

    class FakeExperimentAdapter:
        def execute_semantic_texture_content_write_and_vae(
            self,
            base_latent: object,
            detection_key: str,
            semantic_runtime: object,
        ) -> object:
            constructor_calls.append(("public_write", detection_key))
            return SimpleNamespace(
                result_identity="7" * 64,
                result=SimpleNamespace(
                    witness=SimpleNamespace(witness_identity="8" * 64)
                ),
            )

        def detect_semantic_texture_candidate(
            self,
            detection_image_rgb8: object,
            detection_key: str,
            semantic_runtime: object,
            whitening_asset: object,
            *,
            hf_null: object,
            lf_null: object,
        ) -> object:
            raise AssertionError("detector must remain identity-blocked")

    monkeypatch.setattr(
        entrypoint,
        "Sd35PipelineBackend",
        lambda **kwargs: constructor_calls.append(("backend", kwargs)) or object(),
    )
    monkeypatch.setattr(
        entrypoint,
        "create_runtime_adapter",
        lambda backend, config_path: FakeRuntimeAdapter(),
    )
    monkeypatch.setattr(
        entrypoint,
        "InspyrenetSemanticRuntime",
        lambda checkpoint_path, selected_device: constructor_calls.append(
            ("semantic_runtime", (checkpoint_path, selected_device))
        )
        or object(),
    )
    monkeypatch.setattr(
        entrypoint,
        "load_ceg_wm_experiment_adapter_configuration",
        lambda path: object(),
    )
    monkeypatch.setattr(
        entrypoint,
        "CegWmExperimentAdapter",
        lambda configuration, runtime_adapter: FakeExperimentAdapter(),
    )
    monkeypatch.setattr(
        entrypoint,
        "torch",
        SimpleNamespace(
            Generator=FakeGenerator,
            float16="float16",
            float32="float32",
            randn=lambda shape, **kwargs: constructor_calls.append(
                ("latent_shape", shape)
            )
            or FakeTensor(),
        ),
    )
    for name, value in {
        "HF_TOKEN": "memory-only-token",
        "CEG_WM_ROOT_KEY": "memory-only-root-key",
        "CEG_WM_INSPYRENET_SOURCE_ROOT": str(tmp_path / "source"),
        "CEG_WM_INSPYRENET_CHECKPOINT_PATH": str(tmp_path / "ckpt_base.pth"),
        "CEG_WM_CACHE_ROOT": str(tmp_path / "cache"),
        "CEG_WM_PERSISTENT_ROOT": str(tmp_path / "persistent"),
    }.items():
        monkeypatch.setenv(name, value)
    output_root = tmp_path / "operational-delivery"
    exit_code, _receipt = (
        entrypoint.execute_semantic_texture_operational_preflight_entrypoint(
            source_revision="4" * 40,
            run_id="semantic-texture-operational",
            package_identity="5" * 64,
            output_root=output_root,
        )
    )
    assert exit_code == 2
    assert ("runtime_device", "cuda:0") in constructor_calls
    assert ("generation_seed", 2026081701) in constructor_calls
    assert ("latent_shape", (1, 16, 64, 64)) in constructor_calls
    assert ("latent_to", ("cuda:0", "float16")) in constructor_calls
    assert ("public_write", "memory-only-root-key") in constructor_calls
    backend_call = next(
        value for label, value in constructor_calls if label == "backend"
    )
    assert backend_call == {
        "cache_root": (tmp_path / "cache").resolve(),
        "persistent_root": (tmp_path / "persistent").resolve(),
        "hf_token": "memory-only-token",
        "prompt": "a red cube",
        "negative_prompt": "",
    }
    assert sum(label == "public_write" for label, _ in constructor_calls) == 1
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output_root.iterdir()
        if path.suffix != ".zip"
    )
    archive_path = next(output_root.glob("*.zip"))
    with zipfile.ZipFile(archive_path) as archive:
        persisted += archive.read(server.RESULT_FILENAME).decode("utf-8")
    assert "a red cube" not in persisted
    assert "2026081701" not in persisted
    assert "memory-only-token" not in persisted
    assert "memory-only-root-key" not in persisted


@pytest.mark.parametrize("blocked_class", sorted(preflight.ALLOWED_BLOCKED_CLASSES))
def test_semantic_texture_operational_pre_execution_fault_classes_persist_zero_science_delivery(
    tmp_path: Path,
    blocked_class: str,
) -> None:
    configuration = preflight.load_semantic_texture_operational_configuration(
        ROOT / "configs/experiments/semantic_texture_operational_preflight.json"
    )
    result = preflight.create_semantic_texture_operational_pre_execution_failure(
        configuration,
        source_revision="4" * 40,
        run_id=f"pre-execution-{blocked_class}",
        package_identity="5" * 64,
        blocked_class=blocked_class,
    )
    output_root = tmp_path / blocked_class
    exit_code, _receipt = server.finalize_semantic_texture_operational_preflight_delivery(
        result,
        output_root=output_root,
    )
    assert exit_code == 2
    persisted = result.as_dict()
    assert persisted["aggregate"] is None
    assert persisted["science_started"] is False
    assert persisted["scientific_unit_count"] == 0
    assert all(
        outcome["started"] is False
        and outcome["blocked_class"] == blocked_class
        and outcome["sanitized_error_category"] == blocked_class
        and outcome["sanitized_error_message"] is None
        and outcome["sanitized_trace_tail"] == []
        for outcome in persisted["unit_outcomes"]
    )
    assert {path.name for path in output_root.iterdir()} == {
        server.RESULT_FILENAME,
        server.RECEIPT_FILENAME,
        server.DELIVERY_COMPLETION_CHECKSUMS_FILENAME,
        f"semantic_texture_operational_pre-execution-{blocked_class}.zip",
    }
