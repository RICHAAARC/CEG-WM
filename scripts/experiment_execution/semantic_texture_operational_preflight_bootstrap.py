"""Repository-checkout bootstrap for the semantic-texture operational preflight."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Mapping, Sequence
from zipfile import ZIP_STORED, ZipFile, ZipInfo


PROJECT_REPOSITORY_URL = "https://github.com/RICHAAARC/CEG-WM.git"
PROJECT_BRANCH = "main"
CHECKPOINT_BASENAME = "ckpt_base.pth"
_PYPI_INDEX_URL = "https://pypi.org/simple"
_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
_NVIDIA_INDEX_URL = "https://pypi.nvidia.com"
_OVERLAY_PACKAGE_VERSIONS = {
    "kornia": "0.8.3",
    "kornia-rs": "0.1.14",
    "opencv-python-headless": "4.12.0.88",
    "timm": "1.0.28",
}
TRANSPORT_CHECKSUMS_FILENAME = "SHA256SUMS"
REVISION = re.compile(r"^[0-9a-f]{40}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_ENTRYPOINT_PATHS = frozenset(
    {
        "scripts/experiment_execution/semantic_texture_operational_preflight_entrypoint.py",
        "scripts/experiment_execution/semantic_texture_soft_detector_asset_preparation_entrypoint.py",
    }
)
_TRANSPORT_DELIVERY = {
    "scripts/experiment_execution/semantic_texture_operational_preflight_entrypoint.py": {
        "archive_prefix": "semantic_texture_transport",
        "profile_id": "semantic_texture_operational_preflight_transport",
        "receipt_filename": "semantic_texture_operational_transport_receipt.json",
        "result_filename": "semantic_texture_operational_transport_result.json",
    },
    "scripts/experiment_execution/semantic_texture_soft_detector_asset_preparation_entrypoint.py": {
        "archive_prefix": "semantic_texture_soft_detector_assets",
        "profile_id": "semantic_texture_soft_detector_asset_preparation_transport",
        "receipt_filename": "semantic_texture_soft_detector_asset_transport_receipt.json",
        "result_filename": "semantic_texture_soft_detector_asset_transport_result.json",
    },
}
TRANSPORT_RESULT_FILENAME = _TRANSPORT_DELIVERY[
    "scripts/experiment_execution/semantic_texture_operational_preflight_entrypoint.py"
]["result_filename"]
TRANSPORT_RECEIPT_FILENAME = _TRANSPORT_DELIVERY[
    "scripts/experiment_execution/semantic_texture_operational_preflight_entrypoint.py"
]["receipt_filename"]


class SemanticTextureOperationalBootstrapError(RuntimeError):
    """A repository, environment, or transport boundary failed closed."""

    def __init__(self, blocked_class: str) -> None:
        if blocked_class not in {
            "environment_blocked",
            "resource_blocked",
            "implementation_blocked",
            "identity_blocked",
            "integrity_blocked",
        }:
            raise ValueError("bootstrap blocked class is not registered")
        super().__init__(blocked_class)
        self.blocked_class = blocked_class


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _persist_transport_failure(
    delivery_root: Path,
    *,
    run_id: str,
    blocked_class: str,
    observed_repository_revision: str | None,
    entrypoint_path: str = (
        "scripts/experiment_execution/"
        "semantic_texture_operational_preflight_entrypoint.py"
    ),
) -> dict[str, object]:
    """Persist a bounded transport-only quartet without method result authority."""

    delivery = _TRANSPORT_DELIVERY.get(entrypoint_path)
    if delivery is None or delivery_root.exists():
        raise SemanticTextureOperationalBootstrapError("integrity_blocked")
    delivery_root.mkdir(parents=True)
    result_path = delivery_root / str(delivery["result_filename"])
    archive_path = delivery_root / f"{delivery['archive_prefix']}_{run_id}.zip"
    receipt_path = delivery_root / str(delivery["receipt_filename"])
    checksums_path = delivery_root / TRANSPORT_CHECKSUMS_FILENAME
    result = {
        "aggregate": None,
        "blocked_class": blocked_class,
        "candidate_promoted": False,
        "formal_tau_created": False,
        "observed_repository_revision": observed_repository_revision,
        "profile_id": delivery["profile_id"],
        "run_id": run_id,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "status": "blocked",
        "transport_kind": "repository_bootstrap_failure",
    }
    result_blob = _canonical_bytes(result)
    with result_path.open("xb") as handle:
        handle.write(result_blob)
    with ZipFile(archive_path, mode="x", compression=ZIP_STORED) as archive:
        archive.writestr(_zip_info(result_path.name), result_blob)
    receipt = {
        "archive_filename": archive_path.name,
        "archive_sha256": _sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "blocked_class": blocked_class,
        "observed_repository_revision": observed_repository_revision,
        "profile_id": delivery["profile_id"],
        "result_filename": result_path.name,
        "result_sha256": sha256(result_blob).hexdigest(),
        "run_id": run_id,
        "status": "blocked",
    }
    receipt_blob = _canonical_bytes(receipt)
    with receipt_path.open("xb") as handle:
        handle.write(receipt_blob)
    checksum_blob = (
        f"{_sha256_file(result_path)}  {result_path.name}\n"
        f"{_sha256_file(archive_path)}  {archive_path.name}\n"
        f"{_sha256_file(receipt_path)}  {receipt_path.name}\n"
    ).encode("ascii")
    with checksums_path.open("xb") as handle:
        handle.write(checksum_blob)
    return {
        **receipt,
        "receipt_filename": receipt_path.name,
        "receipt_sha256": _sha256_file(receipt_path),
        "sha256sums_filename": checksums_path.name,
        "sha256sums_sha256": sha256(checksum_blob).hexdigest(),
    }


def _regular_checkpoint(path: Path) -> Path:
    try:
        path_stat = path.lstat()
    except OSError:
        raise SemanticTextureOperationalBootstrapError("environment_blocked") from None
    if path.name != CHECKPOINT_BASENAME or not stat.S_ISREG(path_stat.st_mode):
        raise SemanticTextureOperationalBootstrapError("integrity_blocked")
    return path.resolve()


def _run_checked(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]) -> None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            check=False,
        )
    except OSError:
        raise SemanticTextureOperationalBootstrapError("environment_blocked") from None
    if completed.returncode != 0:
        raise SemanticTextureOperationalBootstrapError("implementation_blocked")


def _require_overlay_imports(repository_root: Path, dependency_root: Path) -> None:
    try:
        import_roots = (str(dependency_root), str(repository_root))
        sys.path[:] = [entry for entry in sys.path if entry not in import_roots]
        sys.path[:0] = import_roots
        if any(metadata.version(name) != version for name, version in _OVERLAY_PACKAGE_VERSIONS.items()):
            raise ValueError
        importlib.import_module("cv2")
        importlib.import_module("kornia")
        importlib.import_module("timm.layers")
        module = importlib.import_module(
            "runtime._vendor.transparent_background.InSPyReNet"
        )
        if not callable(getattr(module, "InSPyReNet_SwinB", None)):
            raise ValueError
    except Exception:
        raise SemanticTextureOperationalBootstrapError("environment_blocked") from None


def _repository_revision(repository_root: Path) -> str:
    if not (repository_root / ".git").is_dir():
        raise SemanticTextureOperationalBootstrapError("integrity_blocked")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        origin = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        raise SemanticTextureOperationalBootstrapError("integrity_blocked") from None
    if (
        REVISION.fullmatch(revision) is None
        or branch != PROJECT_BRANCH
        or origin != PROJECT_REPOSITORY_URL
        or dirty
    ):
        raise SemanticTextureOperationalBootstrapError("integrity_blocked")
    return revision


def _execution_environment(
    repository_root: Path,
    execution_root: Path,
    checkpoint_path: Path,
) -> dict[str, str]:
    environment = dict(os.environ)
    if not environment.get("HF_TOKEN") or not environment.get("CEG_WM_ROOT_KEY"):
        raise SemanticTextureOperationalBootstrapError("environment_blocked")
    cache_root = execution_root / "cache"
    temp_root = execution_root / "tmp"
    dependency_root = execution_root / "dependencies"
    environment.update(
        {
            "CEG_WM_CACHE_ROOT": str(cache_root),
            "CEG_WM_INSPYRENET_CHECKPOINT_PATH": str(checkpoint_path),
            "CEG_WM_PERSISTENT_ROOT": str(execution_root / "persistent"),
            "DIFFUSERS_CACHE": str(cache_root / "diffusers"),
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "PIP_CACHE_DIR": str(cache_root / "pip"),
            "PYTHONPATH": os.pathsep.join((str(dependency_root), str(repository_root))),
            "TMP": str(temp_root),
            "TEMP": str(temp_root),
            "TMPDIR": str(temp_root),
            "TORCH_HOME": str(cache_root / "torch"),
            "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
        }
    )
    return environment


def _entrypoint_run_id(entrypoint_args: Sequence[str]) -> str:
    positions = tuple(
        index
        for index, value in enumerate(entrypoint_args)
        if value == "--run-id"
    )
    if len(positions) != 1 or positions[0] + 1 >= len(entrypoint_args):
        raise SemanticTextureOperationalBootstrapError("integrity_blocked")
    run_id = entrypoint_args[positions[0] + 1]
    if RUN_ID.fullmatch(run_id) is None:
        raise SemanticTextureOperationalBootstrapError("integrity_blocked")
    return run_id


def bootstrap_semantic_texture_operational_preflight(
    *,
    repository_root: str | Path,
    checkpoint: str | Path,
    execution_root: str | Path,
    entrypoint_args: Sequence[str],
    entrypoint_path: str = (
        "scripts/experiment_execution/"
        "semantic_texture_operational_preflight_entrypoint.py"
    ),
) -> tuple[int, dict[str, object]]:
    """Validate a fresh checkout and invoke its public operational entrypoint."""

    repository = Path(repository_root).resolve()
    root = Path(execution_root).resolve()
    run_id = "bootstrap-input-invalid"
    observed_revision: str | None = None
    try:
        bounded_args = tuple(entrypoint_args)
        if not bounded_args or len(bounded_args) > 16:
            raise SemanticTextureOperationalBootstrapError("integrity_blocked")
        if entrypoint_path not in _ENTRYPOINT_PATHS:
            raise SemanticTextureOperationalBootstrapError("integrity_blocked")
        run_id = _entrypoint_run_id(bounded_args)
        if root.exists():
            raise SemanticTextureOperationalBootstrapError("integrity_blocked")
        root.mkdir(parents=True)
        for relative in ("cache", "dependencies", "persistent", "tmp"):
            (root / relative).mkdir()
        observed_revision = _repository_revision(repository)
        checkpoint_path = _regular_checkpoint(Path(checkpoint))
        environment = _execution_environment(
            repository,
            root,
            checkpoint_path,
        )
        if "--execute" in bounded_args:
            _run_checked(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--index-url",
                    _PYPI_INDEX_URL,
                    "--extra-index-url",
                    _PYTORCH_INDEX_URL,
                    "--extra-index-url",
                    _NVIDIA_INDEX_URL,
                    "--requirement",
                    str(
                        repository
                        / "requirements_semantic_texture_operational_preflight_overlay.txt"
                    ),
                    "--no-deps",
                    "--target",
                    str(root / "dependencies"),
                ],
                cwd=repository,
                environment=environment,
            )
            _require_overlay_imports(repository, root / "dependencies")
        command = [
            sys.executable,
            str(
                repository
                / entrypoint_path
            ),
            *bounded_args,
        ]
        completed = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            check=False,
        )
        return completed.returncode, {
            "aggregate": None,
            "entrypoint_exit_code": completed.returncode,
            "observed_repository_revision": observed_revision,
            "science_started": False,
            "scientific_unit_count": 0,
            "stage": "entrypoint",
            "status": "passed" if completed.returncode == 0 else "blocked",
        }
    except Exception as error:
        blocked_class = (
            error.blocked_class
            if isinstance(error, SemanticTextureOperationalBootstrapError)
            else "resource_blocked"
            if isinstance(error, (MemoryError, OSError))
            else "implementation_blocked"
        )
        failure_root = root.with_name(root.name + ".transport")
        receipt = _persist_transport_failure(
            failure_root,
            run_id=run_id,
            blocked_class=blocked_class,
            observed_repository_revision=observed_revision,
            entrypoint_path=entrypoint_path,
        )
        return 2, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--execution-root", required=True)
    parser.add_argument(
        "--entrypoint-path",
        choices=tuple(sorted(_ENTRYPOINT_PATHS)),
        default=(
            "scripts/experiment_execution/"
            "semantic_texture_operational_preflight_entrypoint.py"
        ),
    )
    parser.add_argument("--entrypoint-args", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    exit_code, receipt = bootstrap_semantic_texture_operational_preflight(
        repository_root=arguments.repository_root,
        checkpoint=arguments.checkpoint,
        execution_root=arguments.execution_root,
        entrypoint_args=arguments.entrypoint_args,
        entrypoint_path=arguments.entrypoint_path,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
