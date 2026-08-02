"""CPU/fake tests for the unified HF-only server execution entrypoint."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import ModuleType
import sys

import pytest

from scripts.experiment_execution import experiment_execution_bootstrap as bootstrap
from scripts.experiment_execution import hf_only_threshold_fit_server as server


ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.quick
def test_server_entrypoint_has_no_colab_or_drive_dependency() -> None:
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert "google.colab" not in source
    assert "drive.mount" not in source
    assert "MyDrive" not in source
    assert "notebooks" not in source
    assert "experiments.runners" not in source
    assert "record_writer" not in source


@pytest.mark.quick
def test_server_rejects_wrong_revision_and_dirty_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Server Test")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "fixture")
    revision = _git(repository, "rev-parse", "HEAD")

    with pytest.raises(
        server.HfOnlyThresholdFitServerError,
        match="HEAD differs",
    ):
        server._verify_repository(repository, "f" * 40)

    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(
        server.HfOnlyThresholdFitServerError,
        match="worktree must be clean",
    ):
        server._verify_repository(repository, revision)


@pytest.mark.quick
def test_frozen_model_download_uses_configured_identity_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    config = package / bootstrap.HF_ONLY_THRESHOLD_FIT_RUNTIME_CONFIG_PATH
    config.parent.mkdir(parents=True)
    model_revision = "1" * 40
    config.write_text(
        json.dumps(
            {
                "model_id": "owner/frozen-model",
                "model_revision": model_revision,
            }
        ),
        encoding="utf-8",
    )
    snapshot = tmp_path / "downloaded"
    snapshot.mkdir()
    calls: list[dict[str, object]] = []
    module = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    observed = bootstrap._prepare_frozen_model_snapshot(
        package_root=package,
        model_cache_root=tmp_path / "cache",
        environment={"HF_TOKEN": "private-token"},
    )
    assert observed == ("owner/frozen-model", model_revision, snapshot)
    assert calls == [
        {
            "repo_id": "owner/frozen-model",
            "revision": model_revision,
            "cache_dir": str(tmp_path / "cache/hf_cache/huggingface"),
            "token": "private-token",
        }
    ]


@pytest.mark.quick
def test_server_delegates_to_bootstrap_and_emits_complete_secret_free_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    bootstrap_path = (
        repository
        / "scripts/experiment_execution/experiment_execution_bootstrap.py"
    )
    bootstrap_path.parent.mkdir(parents=True)
    bootstrap_path.write_text("# fixture bootstrap\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    scratch.mkdir()
    cache.mkdir()
    output.mkdir()
    revision = "2" * 40
    artifact = output / "artifacts/result.zip"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"formal-runner-artifact")
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("{}\n", encoding="utf-8")
    observed: dict[str, object] = {}

    monkeypatch.setattr(server, "_verify_repository", lambda *_args: None)
    monkeypatch.setattr(
        server,
        "_load_execution_bindings",
        lambda _root: ("owner/model", "3" * 40, 1),
    )
    monkeypatch.setattr(
        server,
        "_probe_resources",
        lambda **_kwargs: {
            "cuda_device_name": "Fake GPU",
            "cuda_total_memory_bytes": 24 * 1024**3,
            "free_disk_bytes": {},
        },
    )

    def build_package(**_kwargs: object) -> dict[str, object]:
        return {
            "archive_sha256": "4" * 64,
            "candidate_config_digest": "5" * 64,
            "delivery_manifest_path": str(sidecar),
            "embedded_manifest_sha256": "6" * 64,
            "execution_config_digest": "7" * 64,
            "input_manifest_digest": "8" * 64,
        }

    def run_bootstrap(**kwargs: object) -> tuple[int, dict[str, object]]:
        observed.update(kwargs)
        return 0, {
            "artifact_kind": "hf_only_threshold_fit_shard_result",
            "result_zip": str(artifact),
        }

    monkeypatch.setattr(server, "build_experiment_execution_package", build_package)
    monkeypatch.setattr(server.bootstrap, "run_bootstrap", run_bootstrap)
    exit_code, receipt = server.execute_server_threshold_fit_shard(
        repository_root=repository,
        expected_revision=revision,
        scratch_root=scratch,
        cache_root=cache,
        output_root=output,
        run_id="server-run",
        shard_index=0,
        environment={
            "HF_TOKEN": "do-not-persist-hf-token",
            "CEG_WM_ROOT_KEY": "do-not-persist-root-key",
        },
    )
    assert exit_code == 0
    assert observed["prepare_frozen_model"] is True
    assert observed["model_cache_root"] == cache
    assert observed["environment"] == {
        "HF_TOKEN": "do-not-persist-hf-token",
        "CEG_WM_ROOT_KEY": "do-not-persist-root-key",
    }
    assert receipt["artifact_sha256"] == sha256(artifact.read_bytes()).hexdigest()
    assert receipt["committed_revision"] == revision
    assert receipt["run_id"] == "server-run"
    assert receipt["shard_index"] == 0
    assert receipt["model_id"] == "owner/model"
    assert receipt["model_revision"] == "3" * 40
    assert receipt["scientific_claims_supported"] is False
    assert receipt["tau_approval"] is False
    assert receipt["confirmation_unlock"] is False
    persisted = Path(str(receipt["receipt_path"]))
    persisted_text = persisted.read_text(encoding="utf-8")
    assert "do-not-persist" not in persisted_text
    assert "artifact_sha256" in persisted_text
