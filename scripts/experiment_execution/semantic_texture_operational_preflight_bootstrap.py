"""Repository-checkout bootstrap for the semantic-texture operational preflight."""

from __future__ import annotations

import argparse
from hashlib import sha256
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
TRANSPARENT_BACKGROUND_REPOSITORY_URL = (
    "https://github.com/plemeri/transparent-background.git"
)
TRANSPORT_RESULT_FILENAME = "semantic_texture_operational_transport_result.json"
TRANSPORT_RECEIPT_FILENAME = "semantic_texture_operational_transport_receipt.json"
TRANSPORT_CHECKSUMS_FILENAME = "SHA256SUMS"
REVISION = re.compile(r"^[0-9a-f]{40}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


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
) -> dict[str, object]:
    """Persist a bounded transport-only quartet without method result authority."""

    if delivery_root.exists():
        raise SemanticTextureOperationalBootstrapError("integrity_blocked")
    delivery_root.mkdir(parents=True)
    result_path = delivery_root / TRANSPORT_RESULT_FILENAME
    archive_path = delivery_root / f"semantic_texture_transport_{run_id}.zip"
    receipt_path = delivery_root / TRANSPORT_RECEIPT_FILENAME
    checksums_path = delivery_root / TRANSPORT_CHECKSUMS_FILENAME
    result = {
        "aggregate": None,
        "blocked_class": blocked_class,
        "candidate_promoted": False,
        "formal_tau_created": False,
        "observed_repository_revision": observed_repository_revision,
        "profile_id": "semantic_texture_operational_preflight_transport",
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
        "profile_id": "semantic_texture_operational_preflight_transport",
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
    source_root: Path,
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
            "CEG_WM_INSPYRENET_SOURCE_ROOT": str(source_root),
            "CEG_WM_PERSISTENT_ROOT": str(execution_root / "persistent"),
            "DIFFUSERS_CACHE": str(cache_root / "diffusers"),
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "PIP_CACHE_DIR": str(cache_root / "pip"),
            "PYTHONPATH": os.pathsep.join(
                (str(dependency_root), str(repository_root), str(source_root))
            ),
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
        run_id = _entrypoint_run_id(bounded_args)
        if root.exists():
            raise SemanticTextureOperationalBootstrapError("integrity_blocked")
        root.mkdir(parents=True)
        for relative in ("cache", "dependencies", "persistent", "tmp"):
            (root / relative).mkdir()
        observed_revision = _repository_revision(repository)
        checkpoint_path = _regular_checkpoint(Path(checkpoint))
        source_root = root / "source" / "transparent-background"
        environment = _execution_environment(
            repository,
            root,
            checkpoint_path,
            source_root,
        )
        if "--execute" in bounded_args:
            _run_checked(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(root / "dependencies"),
                    "--requirement",
                    str(repository / "requirements_semantic_texture_operational_preflight.txt"),
                ],
                cwd=repository,
                environment=environment,
            )
            _run_checked(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    PROJECT_BRANCH,
                    TRANSPARENT_BACKGROUND_REPOSITORY_URL,
                    str(source_root),
                ],
                cwd=repository,
                environment=environment,
            )
        command = [
            sys.executable,
            str(
                repository
                / "scripts/experiment_execution/semantic_texture_operational_preflight_entrypoint.py"
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
        )
        return 2, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--execution-root", required=True)
    parser.add_argument("--entrypoint-args", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    exit_code, receipt = bootstrap_semantic_texture_operational_preflight(
        repository_root=arguments.repository_root,
        checkpoint=arguments.checkpoint,
        execution_root=arguments.execution_root,
        entrypoint_args=arguments.entrypoint_args,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
