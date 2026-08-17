"""Shared, neutral helpers for diagnostic execution delivery."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import torch

from experiments.runners.development_persistence import canonical_digest


class DeliverySupportError(RuntimeError):
    """A reusable execution-delivery boundary could not be satisfied."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build_or_verify_package(repository: Path, persistent_root: Path, revision: str) -> Path:
    package_root = persistent_root / "development_execution_packages"
    package_root.mkdir(parents=True, exist_ok=True)
    package = package_root / f"ceg_wm_development_{revision}.zip"
    tracked = tuple(line for line in _git(repository, "ls-files").splitlines() if line)
    if not tracked:
        raise DeliverySupportError("repository has no tracked execution files")
    temporary = package.with_suffix(".building.zip")
    if not package.exists():
        if temporary.exists():
            temporary.unlink()
        with ZipFile(temporary, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in tracked:
                source = repository / relative
                if not source.is_file() or source.is_symlink():
                    raise DeliverySupportError("tracked package member is unavailable")
                info = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes(), compress_type=ZIP_DEFLATED)
        try:
            try:
                target = package.open("xb")
            except FileExistsError:
                pass
            else:
                try:
                    with temporary.open("rb") as source, target:
                        shutil.copyfileobj(source, target)
                except Exception:
                    package.unlink(missing_ok=True)
                    raise
        finally:
            temporary.unlink(missing_ok=True)
    if not package.is_file():
        raise DeliverySupportError("development execution package is invalid")
    with ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise DeliverySupportError("development execution package is invalid")
    return package


def _environment_digest() -> str:
    return canonical_digest(
        {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": tuple(os.sys.version_info[:3]),
            "torch": torch.__version__,
        }
    )


def _session_runtime_identity(*, role: str, display_value: str) -> str:
    """Convert runtime display metadata into a persisted stable identity."""

    normalized = re.sub(r"[^a-z0-9]+", "_", display_value.strip().lower()).strip("_")
    if role not in {"gpu", "cuda"} or not normalized:
        raise DeliverySupportError("session runtime identity is unavailable")
    return f"{role}_{normalized}"


def _base_latent(seed: int, *, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        (1, 16, height // 8, width // 8),
        generator=generator,
        dtype=torch.float32,
    ).to(device="cuda:0", dtype=torch.float16)
