"""Lightweight tests for the development exploration server launcher."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import ModuleType
import sys
import zipfile

import pytest

from scripts.experiment_execution import development_exploration_server as server


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    runtime_path = repository / server.RUNTIME_CONFIG_PATH
    protocol_path = repository / server.PROTOCOL_CONFIG_PATH
    dependency_path = repository / server.DEPENDENCY_LOCK_PATH
    runtime_path.parent.mkdir(parents=True)
    protocol_path.parent.mkdir(parents=True)
    runtime_path.write_text(
        json.dumps(
            {
                "model_id": "owner/development-model",
                "model_revision": "1" * 40,
            }
        ),
        encoding="utf-8",
    )
    protocol_path.write_text(
        json.dumps(
            {
                "protocol_id": "development_exploration",
                "protocol_version": "1.0.0",
                "study_budget": {"unit_roster_digest": "2" * 64},
            }
        ),
        encoding="utf-8",
    )
    dependency_path.write_text("example-package==1.0.0\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Development Server Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    return repository.resolve(), _git(repository, "rev-parse", "HEAD")


@pytest.mark.quick
def test_launcher_is_server_colab_neutral_and_does_not_write_records() -> None:
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert "google.colab" not in source
    assert "drive.mount" not in source
    assert "MyDrive" not in source
    assert "hf_only_threshold_fit" not in source
    assert "DevelopmentScientificRecord" not in source
    assert "DevelopmentExplorationRunner" not in source


@pytest.mark.quick
def test_repository_identity_rejects_wrong_revision_and_dirty_tree(
    tmp_path: Path,
) -> None:
    repository, revision = _repository(tmp_path)
    with pytest.raises(
        server.DevelopmentExplorationServerError,
        match="HEAD differs",
    ):
        server._verify_repository(repository, "f" * 40)

    (repository / server.RUNTIME_CONFIG_PATH).write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        server.DevelopmentExplorationServerError,
        match="worktree must be clean",
    ):
        server._verify_repository(repository, revision)


@pytest.mark.quick
def test_dependency_install_uses_development_lock_without_hash_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _revision = _repository(tmp_path)
    calls: list[tuple[tuple[str, ...], Path]] = []

    def run(arguments: tuple[str, ...], **kwargs: object) -> object:
        calls.append((arguments, Path(str(kwargs["cwd"]))))
        return object()

    monkeypatch.setattr(server.subprocess, "run", run)
    server._install_frozen_dependencies(repository)
    assert len(calls) == 1
    arguments, cwd = calls[0]
    assert cwd == repository
    assert arguments[-2:] == (
        "-r",
        str(repository / server.DEPENDENCY_LOCK_PATH),
    )
    assert "--require-hashes" not in arguments


@pytest.mark.quick
def test_model_download_uses_configured_revision_without_snapshot_hash_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[dict[str, object]] = []
    module = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    observed = server._download_configured_model(
        model_id="owner/development-model",
        model_revision="1" * 40,
        cache_root=tmp_path / "cache",
        hf_token="private-token",
    )
    assert observed == snapshot.resolve()
    assert calls == [
        {
            "repo_id": "owner/development-model",
            "revision": "1" * 40,
            "cache_dir": str(tmp_path / "cache/huggingface"),
            "token": "private-token",
        }
    ]


@pytest.mark.quick
def test_server_delegates_to_formal_entrypoint_and_writes_secret_free_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, revision = _repository(tmp_path)
    persistent = (tmp_path / "persistent").resolve()
    cache = (tmp_path / "cache").resolve()
    persistent.mkdir()
    cache.mkdir()
    artifact = persistent / "run" / "worker" / "development_result.zip"
    artifact.parent.mkdir(parents=True)
    with zipfile.ZipFile(artifact, mode="x") as archive:
        archive.writestr("result.json", "{}\n")
    protocol_digest = "7" * 64
    environment = {
        "HF_TOKEN": "private-hf-token",
        "CEG_WM_ROOT_KEY": "private-root-key",
        "UNRELATED_SECRET": "must-not-cross-worker-boundary",
    }
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        server,
        "_probe_resources",
        lambda **_kwargs: {
            "cuda_device_name": "Test GPU",
            "cuda_total_memory_bytes": 24 * 1024**3,
            "free_disk_bytes": {str(persistent): 1, str(cache): 1},
        },
    )
    monkeypatch.setattr(
        server,
        "_load_frozen_bindings",
        lambda _root: {
            "model_id": "owner/development-model",
            "model_revision": "1" * 40,
            "protocol_id": "development_exploration",
            "protocol_version": "1.0.0",
            "protocol_digest": protocol_digest,
            "runtime_config_digest": "8" * 64,
            "unit_roster_digest": "2" * 64,
        },
    )
    monkeypatch.setattr(
        server,
        "_download_configured_model",
        lambda **_kwargs: cache / "snapshot",
    )

    def execute(**kwargs: object) -> tuple[int, dict[str, object]]:
        calls.append(kwargs)
        return 0, {
            "artifact_kind": "development_exploration_result",
            "result_zip": str(artifact),
            "protocol_digest": protocol_digest,
            "execution_intent_authority_digest": "3" * 64,
            "input_manifest_digest": "4" * 64,
            "candidate_config_digest": "9" * 64,
            "unit_roster_digest": "2" * 64,
            "package_sha256": "5" * 64,
            "bootstrap_sha256": "6" * 64,
            "committed_unit_count": 7,
            "termination_reason": "session_soft_stop",
            "environment": environment,
        }

    monkeypatch.setattr(server, "_execute_development_entrypoint", execute)
    exit_code, receipt = server.execute_development_exploration_server_session(
        repository_root=repository,
        expected_revision=revision,
        persistent_root=persistent,
        cache_root=cache,
        run_id="run",
        session_id="session",
        environment=environment,
        install_dependencies=False,
    )
    assert exit_code == 0
    assert calls == [
        {
            "repository_root": repository,
            "expected_revision": revision,
            "persistent_root": persistent,
            "cache_root": cache,
            "run_id": "run",
            "session_id": "session",
            "environment": {
                "HF_TOKEN": "private-hf-token",
                "CEG_WM_ROOT_KEY": "private-root-key",
            },
        }
    ]
    assert receipt["artifact_sha256"] == sha256(artifact.read_bytes()).hexdigest()
    receipt_text = Path(str(receipt["receipt_path"])).read_text(encoding="utf-8")
    assert "private-hf-token" not in receipt_text
    assert "private-root-key" not in receipt_text
    assert "must-not-cross-worker-boundary" not in receipt_text
    assert receipt["scientific_claims_supported"] is False
    assert receipt["formal_tau_created"] is False
    assert receipt["calibration_locked"] is False


@pytest.mark.quick
def test_missing_secrets_produces_secret_free_create_only_diagnostic(
    tmp_path: Path,
) -> None:
    error = server.DevelopmentExplorationServerError(
        "secrets",
        "HF_TOKEN and CEG_WM_ROOT_KEY are required",
    )
    receipt = server._failure_artifacts(
        persistent_root=tmp_path,
        expected_revision="1" * 40,
        run_id="run",
        session_id="session",
        error=error,
    )
    artifact = Path(str(receipt["artifact_path"]))
    assert zipfile.is_zipfile(artifact)
    assert receipt["artifact_sha256"] == sha256(artifact.read_bytes()).hexdigest()
    combined = artifact.read_bytes() + Path(str(receipt["receipt_path"])).read_bytes()
    assert b"Traceback" not in combined
    assert b"private" not in combined
    assert b"failure_message" not in combined
    assert receipt["failure_stage"] == "secrets"
    assert receipt["responsibility_id"] is None
    assert receipt["unit_index"] is None
