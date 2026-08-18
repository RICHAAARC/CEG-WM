"""Build the exact semantic-texture operational preflight package."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Sequence
import zipfile


PACKAGE_SCHEMA_VERSION = 1
DELIVERY_MANIFEST_SCHEMA_VERSION = 1
PACKAGE_PROFILE = "semantic_texture_operational_preflight_package"
EMBEDDED_MANIFEST_PATH = "semantic_texture_operational_preflight_manifest.json"
ENTRYPOINT_PATH = (
    "scripts/experiment_execution/"
    "semantic_texture_operational_preflight_entrypoint.py"
)
SERVER_PATH = (
    "scripts/experiment_execution/semantic_texture_operational_preflight_server.py"
)
CONFIG_PATH = (
    "configs/experiments/semantic_texture_operational_preflight.json"
)
REQUIREMENTS_PATH = "requirements_semantic_texture_operational_preflight.txt"
OVERLAY_REQUIREMENTS_PATH = "requirements_semantic_texture_operational_preflight_overlay.txt"
RUNNER_PATH = "experiments/runners/semantic_texture_operational_preflight.py"
SOURCE_TO_ARCHIVE_PATH = {
    "templates/release_readmes/semantic_texture_operational_preflight_package.md": (
        "README.md"
    )
}
EXACT_SOURCE_FILES = frozenset(
    {
        CONFIG_PATH,
        "configs/experiments/internal_execution_components.json",
        "configs/runtime/runtime_sd35_flowmatch.json",
        "experiments/__init__.py",
        "experiments/methods/__init__.py",
        "experiments/methods/ceg_wm.py",
        RUNNER_PATH,
        "main/__init__.py",
        "main/content_chain/__init__.py",
        "main/content_chain/detector.py",
        "main/content_chain/embedder.py",
        "main/content_chain/hf_carrier.py",
        "main/content_chain/hf_detector.py",
        "main/content_chain/lf_carrier.py",
        "main/content_chain/lf_detector.py",
        "main/content_chain/lf_whitening.py",
        "main/content_chain/routing.py",
        "main/geometry_chain/__init__.py",
        "main/geometry_chain/qk_sync.py",
        "main/geometry_chain/rectifier.py",
        "main/geometry_chain/reliability.py",
        "main/geometry_chain/transform_estimator.py",
        "main/joint_decision/__init__.py",
        "main/joint_decision/detector.py",
        "main/shared/__init__.py",
        "main/shared/key_schedule.py",
        "main/shared/normal_quantile_table20_float32_be.txt",
        "main/shared/rgb8.py",
        REQUIREMENTS_PATH,
        OVERLAY_REQUIREMENTS_PATH,
        "runtime/__init__.py",
        "runtime/adapter.py",
        "runtime/backend.py",
        "runtime/configuration.py",
        "runtime/content_write.py",
        "runtime/geometry_synchronization.py",
        "runtime/qk_observation.py",
        "runtime/routing_observation.py",
        "runtime/_vendor/transparent_background/InSPyReNet.py",
        "runtime/_vendor/transparent_background/LICENSE",
        "runtime/_vendor/transparent_background/SOURCE.json",
        "runtime/_vendor/transparent_background/__init__.py",
        "runtime/_vendor/transparent_background/backbones/SwinTransformer.py",
        "runtime/_vendor/transparent_background/backbones/__init__.py",
        "runtime/_vendor/transparent_background/modules/__init__.py",
        "runtime/_vendor/transparent_background/modules/attention_module.py",
        "runtime/_vendor/transparent_background/modules/context_module.py",
        "runtime/_vendor/transparent_background/modules/decoder_module.py",
        "runtime/_vendor/transparent_background/modules/layers.py",
        "runtime/sd35_backend.py",
        "scripts/experiment_execution/__init__.py",
        ENTRYPOINT_PATH,
        SERVER_PATH,
        *SOURCE_TO_ARCHIVE_PATH,
    }
)
REQUIRED_ARCHIVE_FILES = frozenset(
    {
        *(
            SOURCE_TO_ARCHIVE_PATH.get(path, path)
            for path in EXACT_SOURCE_FILES
        ),
        "README.md",
    }
)
EXCLUDED_PARTS = frozenset(
    {
        ".agents",
        ".codex",
        ".git",
        ".pytest_cache",
        "__pycache__",
        "audit_reports",
        "governance",
        "notebooks",
        "outputs",
        "paper_artifacts",
        "tests",
    }
)
SENSITIVE_PARTS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "id_rsa",
    "id_ed25519",
)
REVISION = re.compile(r"^[0-9a-f]{40}$")
LOCAL_ABSOLUTE_PATH = re.compile(
    rb"(?<![A-Za-z0-9_])(?:/(?:home|Users|mnt|tmp|var|opt|root)/|[A-Za-z]:[\\/])"
)


class SemanticTexturePackageBuildError(RuntimeError):
    """The package is unsafe, mutable, or not exact-revision-bound."""


def _git(
    root: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SemanticTexturePackageBuildError(
            "source repository identity is unavailable"
        ) from exc
    return completed.stdout.strip() if text else completed.stdout


def _validate_revision(repository_root: Path, source_revision: str) -> None:
    if REVISION.fullmatch(source_revision) is None:
        raise SemanticTexturePackageBuildError(
            "source_revision must be an exact Git revision"
        )
    if _git(repository_root, "rev-parse", "HEAD") != source_revision:
        raise SemanticTexturePackageBuildError(
            "source_revision does not equal HEAD"
        )
    if _git(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        raise SemanticTexturePackageBuildError(
            "source worktree must be clean"
        )


def _safe_relative(path_text: str) -> PurePosixPath:
    if (
        not path_text
        or "\\" in path_text
        or "\x00" in path_text
        or re.match(r"^[A-Za-z]:", path_text)
    ):
        raise SemanticTexturePackageBuildError("unsafe package path")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SemanticTexturePackageBuildError("unsafe package path")
    if any(part in EXCLUDED_PARTS for part in path.parts):
        raise SemanticTexturePackageBuildError("excluded package path")
    if any(
        marker in part.lower()
        for part in path.parts
        for marker in SENSITIVE_PARTS
    ):
        raise SemanticTexturePackageBuildError("sensitive package path")
    return path


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


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
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _tracked_blobs(
    repository_root: Path,
    source_revision: str,
) -> tuple[tuple[str, bytes], ...]:
    selected: list[tuple[str, bytes]] = []
    archive_paths: set[str] = set()
    for source_path in sorted(EXACT_SOURCE_FILES):
        archive_path = SOURCE_TO_ARCHIVE_PATH.get(source_path, source_path)
        _safe_relative(archive_path)
        if archive_path in archive_paths:
            raise SemanticTexturePackageBuildError(
                "duplicate package path"
            )
        raw = _git(
            repository_root,
            "ls-tree",
            source_revision,
            source_path,
            text=False,
        )
        if not isinstance(raw, bytes) or not raw.strip():
            raise SemanticTexturePackageBuildError(
                f"required package source is missing: {source_path}"
            )
        metadata, _separator, listed_path = raw.rstrip(b"\n").partition(b"\t")
        fields = metadata.split()
        if (
            len(fields) != 3
            or fields[0] not in {b"100644", b"100755"}
            or fields[1] != b"blob"
            or listed_path.decode("utf-8") != source_path
        ):
            raise SemanticTexturePackageBuildError(
                f"package source is not a regular tracked blob: {source_path}"
            )
        blob = _git(
            repository_root,
            "show",
            f"{source_revision}:{source_path}",
            text=False,
        )
        if not isinstance(blob, bytes):
            raise SemanticTexturePackageBuildError(
                f"package source blob is invalid: {source_path}"
            )
        if LOCAL_ABSOLUTE_PATH.search(blob):
            raise SemanticTexturePackageBuildError(
                f"local absolute path is forbidden: {source_path}"
            )
        archive_paths.add(archive_path)
        selected.append((archive_path, blob))
    if archive_paths != set(REQUIRED_ARCHIVE_FILES):
        raise SemanticTexturePackageBuildError(
            "package tracked allowlist identity drifted"
        )
    return tuple(selected)


def _zip_info(path_text: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path_text, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_semantic_texture_operational_preflight_package(
    *,
    repository_root: str | Path,
    source_revision: str,
    output: str | Path,
) -> dict[str, object]:
    """Write a create-only deterministic archive and adjacent manifest."""

    root = Path(repository_root).resolve()
    output_path = Path(output).resolve()
    delivery_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    _validate_revision(root, source_revision)
    if output_path == root or root in output_path.parents:
        raise SemanticTexturePackageBuildError(
            "package output must be outside the repository"
        )
    if output_path.exists() or delivery_path.exists():
        raise SemanticTexturePackageBuildError(
            "package output targets must be absent"
        )
    entries = _tracked_blobs(root, source_revision)
    copied_files = [
        {
            "path": path_text,
            "sha256": _sha256_bytes(blob),
            "size_bytes": len(blob),
        }
        for path_text, blob in entries
    ]
    configuration_blob = dict(entries)[CONFIG_PATH]
    package_identity = _sha256_bytes(
        _canonical_bytes(
            {
                "copied_files": copied_files,
                "profile_name": PACKAGE_PROFILE,
                "source_revision": source_revision,
            }
        )
    )
    embedded_manifest = {
        "configuration_sha256": _sha256_bytes(configuration_blob),
        "copied_files": copied_files,
        "entrypoint_path": ENTRYPOINT_PATH,
        "excluded_parts": sorted(EXCLUDED_PARTS),
        "package_identity": package_identity,
        "package_ready": True,
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "profile_name": PACKAGE_PROFILE,
        "server_path": SERVER_PATH,
        "source_revision": source_revision,
    }
    embedded_blob = _canonical_bytes(embedded_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            for path_text, blob in entries:
                archive.writestr(_zip_info(path_text), blob)
            archive.writestr(
                _zip_info(EMBEDDED_MANIFEST_PATH),
                embedded_blob,
            )
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise SemanticTexturePackageBuildError(
                "package output target must remain absent"
            ) from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    delivery_manifest = {
        "archive_filename": output_path.name,
        "archive_sha256": _sha256_file(output_path),
        "archive_size_bytes": output_path.stat().st_size,
        "configuration_sha256": _sha256_bytes(configuration_blob),
        "delivery_manifest_schema_version": DELIVERY_MANIFEST_SCHEMA_VERSION,
        "embedded_manifest_sha256": _sha256_bytes(embedded_blob),
        "package_identity": package_identity,
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "profile_name": PACKAGE_PROFILE,
        "source_revision": source_revision,
    }
    try:
        with delivery_path.open("xb") as handle:
            handle.write(_canonical_bytes(delivery_manifest))
    except FileExistsError as exc:
        raise SemanticTexturePackageBuildError(
            "delivery manifest target must remain absent"
        ) from exc
    return {
        **delivery_manifest,
        "delivery_manifest_path": str(delivery_path),
        "copied_file_count": len(copied_files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    result = build_semantic_texture_operational_preflight_package(
        repository_root=arguments.repository_root,
        source_revision=arguments.source_revision,
        output=arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
