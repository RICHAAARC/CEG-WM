"""Shared, neutral server helpers for diagnostic execution delivery."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping


REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RUNTIME_CONFIG_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")
DEPENDENCY_LOCK_PATH = Path("requirements_development_exploration_gpu_execution.txt")
PYPI_INDEX_URL = "https://pypi.org/simple"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
NVIDIA_INDEX_URL = "https://pypi.nvidia.com"


class ExperimentExecutionServerError(RuntimeError):
    """A reusable server preparation boundary failed."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExperimentExecutionServerError("repository identity is unavailable") from exc
    return completed.stdout.strip()


def _absolute_directory(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ExperimentExecutionServerError(f"{role} must be absolute")
    resolved = path.resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExperimentExecutionServerError(f"{role} is unavailable") from exc
    if not resolved.is_dir():
        raise ExperimentExecutionServerError(f"{role} is not a directory")
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _verify_repository(root: Path, expected_revision: str) -> None:
    if not root.is_absolute() or not root.is_dir():
        raise ExperimentExecutionServerError("repository root is invalid")
    if REVISION_PATTERN.fullmatch(expected_revision) is None:
        raise ExperimentExecutionServerError("expected revision is invalid")
    if _git(root, "rev-parse", "HEAD") != expected_revision:
        raise ExperimentExecutionServerError("repository HEAD differs from expected revision")
    if _git(root, "status", "--porcelain"):
        raise ExperimentExecutionServerError("repository worktree must be clean")


def _probe_resources(*, persistent_root: Path, cache_root: Path) -> dict[str, object]:
    free_disk_bytes: dict[str, int] = {}
    try:
        for root in (persistent_root, cache_root):
            observed = int(shutil.disk_usage(root).free)
            if observed <= 0:
                raise ExperimentExecutionServerError("execution storage has no available space")
            free_disk_bytes[str(root)] = observed
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        gpu_name, memory_mib_text = (part.strip() for part in rows[0].rsplit(",", 1))
        gpu_memory_bytes = int(memory_mib_text) * 1024 * 1024
    except ExperimentExecutionServerError:
        raise
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as exc:
        raise ExperimentExecutionServerError("CUDA device identity is unavailable") from exc
    if not gpu_name or gpu_memory_bytes <= 0:
        raise ExperimentExecutionServerError("CUDA device identity is invalid")
    return {
        "cuda_device_name": gpu_name,
        "cuda_total_memory_bytes": gpu_memory_bytes,
        "free_disk_bytes": free_disk_bytes,
    }


def _install_frozen_dependencies(root: Path) -> None:
    lock_path = root / DEPENDENCY_LOCK_PATH
    if not lock_path.is_file():
        raise ExperimentExecutionServerError("development dependency lock is unavailable")
    try:
        subprocess.run(
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--index-url",
                PYPI_INDEX_URL,
                "--extra-index-url",
                PYTORCH_INDEX_URL,
                "--extra-index-url",
                NVIDIA_INDEX_URL,
                "-r",
                str(lock_path),
            ),
            cwd=root,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExperimentExecutionServerError("frozen dependency installation failed") from exc


def _download_configured_model(
    *,
    model_id: str,
    model_revision: str,
    cache_root: Path,
    hf_token: str,
) -> Path:
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=model_revision,
                cache_dir=str(cache_root / "huggingface"),
                token=hf_token,
            )
        ).resolve()
    except Exception as exc:
        raise ExperimentExecutionServerError("configured model revision download failed") from exc
    if not snapshot.is_dir():
        raise ExperimentExecutionServerError("configured model revision is unavailable after download")
    return snapshot


def _write_json_create_only(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with path.open("xb") as target:
            target.write(encoded)
    except OSError as exc:
        raise ExperimentExecutionServerError("execution receipt could not be created") from exc
