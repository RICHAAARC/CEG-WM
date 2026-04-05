"""
File purpose: Validate notebook model cache bootstrap helpers for persistent Drive reuse and local runtime sync.
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from scripts.notebook_runtime_common import (
    bootstrap_notebook_model_cache,
    resolve_notebook_model_cache_layout,
)


def _build_cfg_obj(*, model_id: str = "acme/demo-model", revision: str = "rev-1") -> Dict[str, Any]:
    """
    功能：构造最小 notebook 模型缓存配置。 

    Build the minimal runtime config required by notebook cache bootstrap.

    Args:
        model_id: Hugging Face model identifier.
        revision: Requested model revision.

    Returns:
        Runtime configuration mapping.
    """
    return {
        "model": {
            "model_id": model_id,
            "revision": revision,
        },
        "mask": {
            "semantic_model_path": "models/inspyrenet/ckpt_base.pth",
            "semantic_model_source": "inspyrenet",
        },
    }


def _write_snapshot_tree(snapshot_root: Path, *, marker: str) -> None:
    """
    功能：写出最小模型快照目录。 

    Write a minimal model snapshot directory used by tests.

    Args:
        snapshot_root: Snapshot directory path.
        marker: Stable text marker written into the snapshot.

    Returns:
        None.
    """
    snapshot_root.mkdir(parents=True, exist_ok=True)
    (snapshot_root / "model_index.json").write_text(f'{{"marker": "{marker}"}}', encoding="utf-8")
    (snapshot_root / "weights.bin").write_bytes(marker.encode("utf-8"))


def test_resolve_notebook_model_cache_layout_returns_expected_paths(tmp_path: Path) -> None:
    """
    Verify notebook cache layout resolves the required Drive and local paths.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    drive_mount_root = tmp_path / "drive"
    repo_root = tmp_path / "repo"

    layout = resolve_notebook_model_cache_layout(drive_mount_root, repo_root, create_directories=True)

    assert layout["drive_models_root"] == drive_mount_root / "MyDrive" / "Models"
    assert layout["persistent_inspyrenet_root"] == drive_mount_root / "MyDrive" / "Models" / "inspyrenet"
    assert layout["persistent_hf_root"] == drive_mount_root / "MyDrive" / "Models" / "Huggingface"
    assert layout["local_hf_home"] == repo_root / "huggingface_cache"
    assert layout["local_hf_hub_cache"] == repo_root / "huggingface_cache" / "hub"
    assert layout["local_transformers_cache"] == repo_root / "huggingface_cache" / "transformers"
    assert layout["local_runtime_snapshot_root"] == repo_root / "huggingface_cache" / "runtime_snapshots"
    assert layout["persistent_inspyrenet_path"] == drive_mount_root / "MyDrive" / "Models" / "inspyrenet" / "ckpt_base.pth"

    for path_key in [
        "drive_models_root",
        "persistent_inspyrenet_root",
        "persistent_hf_root",
        "local_hf_home",
        "local_hf_hub_cache",
        "local_transformers_cache",
        "local_runtime_snapshot_root",
    ]:
        assert layout[path_key].exists()
        assert layout[path_key].is_dir()


def test_bootstrap_notebook_model_cache_reuses_persistent_snapshot_and_syncs_local(tmp_path: Path) -> None:
    """
    Verify helper reuses an existing persistent snapshot and syncs a local runtime copy.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    drive_mount_root = tmp_path / "drive"
    repo_root = tmp_path / "repo"
    layout = resolve_notebook_model_cache_layout(drive_mount_root, repo_root, create_directories=True)
    persistent_snapshot_path = layout["persistent_hf_root"] / "models--acme--demo-model" / "snapshots" / "snapshot-001"
    persistent_weight_path = layout["persistent_inspyrenet_path"]
    snapshot_calls: List[Dict[str, Any]] = []

    _write_snapshot_tree(persistent_snapshot_path, marker="persistent-reused")
    persistent_weight_path.write_bytes(b"persistent-weight")

    def fake_snapshot_download(**kwargs: Any) -> str:
        snapshot_calls.append(dict(kwargs))
        return persistent_snapshot_path.as_posix()

    def fail_file_download(_url: str, _target_path: Path) -> None:
        raise AssertionError("file download should not be used when persistent weight exists")

    summary = bootstrap_notebook_model_cache(
        _build_cfg_obj(),
        drive_mount_root,
        repo_root,
        semantic_model_source_urls={"inspyrenet": "https://example.invalid/ckpt_base.pth"},
        snapshot_download_fn=fake_snapshot_download,
        file_download_fn=fail_file_download,
    )

    local_snapshot_path = Path(str(summary["local_snapshot_path"]))
    repo_weight_path = Path(str(summary["repo_inspyrenet_path"]))
    assert summary["cache_reuse_mode"] == "persistent_reused_and_local_synced"
    assert summary["weight_cache_mode"] == "persistent_reused"
    assert snapshot_calls == [
        {
            "repo_id": "acme/demo-model",
            "revision": "rev-1",
            "cache_dir": str(layout["persistent_hf_root"]),
            "local_files_only": True,
        }
    ]
    assert local_snapshot_path.exists() and local_snapshot_path.is_dir()
    assert (local_snapshot_path / "model_index.json").read_text(encoding="utf-8") == '{"marker": "persistent-reused"}'
    assert repo_weight_path.exists() and repo_weight_path.read_bytes() == b"persistent-weight"

    second_summary = bootstrap_notebook_model_cache(
        _build_cfg_obj(),
        drive_mount_root,
        repo_root,
        semantic_model_source_urls={"inspyrenet": "https://example.invalid/ckpt_base.pth"},
        snapshot_download_fn=fake_snapshot_download,
        file_download_fn=fail_file_download,
    )
    assert second_summary["cache_reuse_mode"] == "persistent_reused_and_local_reused"


def test_bootstrap_notebook_model_cache_seeds_persistent_weight_from_repo(tmp_path: Path) -> None:
    """
    Verify helper promotes an existing repo-local InSPyReNet weight into the persistent path.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    drive_mount_root = tmp_path / "drive"
    repo_root = tmp_path / "repo"
    layout = resolve_notebook_model_cache_layout(drive_mount_root, repo_root, create_directories=True)
    persistent_snapshot_path = layout["persistent_hf_root"] / "models--acme--demo-model" / "snapshots" / "snapshot-001"
    repo_weight_path = repo_root / "models" / "inspyrenet" / "ckpt_base.pth"

    _write_snapshot_tree(persistent_snapshot_path, marker="repo-seed")
    repo_weight_path.parent.mkdir(parents=True, exist_ok=True)
    repo_weight_path.write_bytes(b"repo-seeded-weight")

    def fake_snapshot_download(**kwargs: Any) -> str:
        return persistent_snapshot_path.as_posix()

    summary = bootstrap_notebook_model_cache(
        _build_cfg_obj(),
        drive_mount_root,
        repo_root,
        semantic_model_source_urls={"inspyrenet": "https://example.invalid/ckpt_base.pth"},
        snapshot_download_fn=fake_snapshot_download,
        file_download_fn=lambda _url, _target_path: (_ for _ in ()).throw(AssertionError("unexpected weight download")),
    )

    persistent_weight_path = Path(str(summary["persistent_inspyrenet_path"]))
    assert summary["weight_cache_mode"] == "persistent_seeded_from_repo"
    assert persistent_weight_path.exists()
    assert persistent_weight_path.read_bytes() == b"repo-seeded-weight"


def test_bootstrap_notebook_model_cache_downloads_missing_persistent_snapshot(tmp_path: Path) -> None:
    """
    Verify helper downloads the persistent snapshot once and then binds the local copy.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    drive_mount_root = tmp_path / "drive"
    repo_root = tmp_path / "repo"
    layout = resolve_notebook_model_cache_layout(drive_mount_root, repo_root, create_directories=True)
    persistent_snapshot_path = layout["persistent_hf_root"] / "models--acme--demo-model" / "snapshots" / "snapshot-002"
    layout["persistent_inspyrenet_path"].write_bytes(b"existing-weight")
    snapshot_calls: List[Dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        snapshot_calls.append(dict(kwargs))
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("persistent snapshot missing")
        _write_snapshot_tree(persistent_snapshot_path, marker="downloaded")
        return persistent_snapshot_path.as_posix()

    summary = bootstrap_notebook_model_cache(
        _build_cfg_obj(revision="rev-2"),
        drive_mount_root,
        repo_root,
        semantic_model_source_urls={"inspyrenet": "https://example.invalid/ckpt_base.pth"},
        snapshot_download_fn=fake_snapshot_download,
        file_download_fn=lambda _url, _target_path: (_ for _ in ()).throw(AssertionError("unexpected weight download")),
    )

    local_snapshot_path = Path(str(summary["local_snapshot_path"]))
    assert summary["cache_reuse_mode"] == "persistent_downloaded_and_local_synced"
    assert snapshot_calls[0]["local_files_only"] is True
    assert "local_files_only" not in snapshot_calls[1]
    assert local_snapshot_path.exists()
    assert (local_snapshot_path / "model_index.json").read_text(encoding="utf-8") == '{"marker": "downloaded"}'


def test_bootstrap_notebook_model_cache_downloads_missing_persistent_weight(tmp_path: Path) -> None:
    """
    Verify helper downloads the persistent InSPyReNet weight when neither cache nor repo copy exists.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    drive_mount_root = tmp_path / "drive"
    repo_root = tmp_path / "repo"
    layout = resolve_notebook_model_cache_layout(drive_mount_root, repo_root, create_directories=True)
    persistent_snapshot_path = layout["persistent_hf_root"] / "models--acme--demo-model" / "snapshots" / "snapshot-003"

    _write_snapshot_tree(persistent_snapshot_path, marker="weight-download")

    def fake_snapshot_download(**kwargs: Any) -> str:
        return persistent_snapshot_path.as_posix()

    def fake_file_download(url: str, target_path: Path) -> None:
        assert url == "https://example.invalid/ckpt_base.pth"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"downloaded-weight")

    summary = bootstrap_notebook_model_cache(
        _build_cfg_obj(),
        drive_mount_root,
        repo_root,
        semantic_model_source_urls={"inspyrenet": "https://example.invalid/ckpt_base.pth"},
        snapshot_download_fn=fake_snapshot_download,
        file_download_fn=fake_file_download,
    )

    persistent_weight_path = Path(str(summary["persistent_inspyrenet_path"]))
    repo_weight_path = Path(str(summary["repo_inspyrenet_path"]))
    assert summary["weight_cache_mode"] == "persistent_downloaded"
    assert persistent_weight_path.exists() and persistent_weight_path.read_bytes() == b"downloaded-weight"
    assert repo_weight_path.exists() and repo_weight_path.read_bytes() == b"downloaded-weight"