"""Build one exact-revision-bound runtime qualification execution package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


PACKAGE_PROFILE = "experiment_execution_package"
REQUIRED_FILES = {
    "configs/runtime/runtime_sd35_flowmatch.json",
    "pyproject.toml",
    "requirements_runtime_qualification.txt",
    "scripts/experiment_execution/README.md",
    "scripts/experiment_execution/__init__.py",
    "scripts/experiment_execution/runtime_qualification_runner.py",
}
INCLUDE_ROOTS = ("main/", "runtime/")
EXCLUDED_PARTS = {
    ".agents",
    ".codex",
    ".git",
    "governance",
    "notebooks",
    "outputs",
    "__pycache__",
    ".pytest_cache",
}
SENSITIVE_PARTS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "id_rsa",
    "id_ed25519",
)
LOCAL_PATH = re.compile(
    rb"(?<![A-Za-z0-9_])(?:/(?:home|Users|mnt|tmp|var|opt|root)/|[A-Za-z]:[\\/])"
)
REVISION = re.compile(r"^[0-9a-f]{40}$")


class PackageBuildError(RuntimeError):
    """The package is not safe or not bound to an exact clean revision."""


def _git(root: Path, *arguments: str, text: bool = True):
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageBuildError("source repository identity is unavailable") from exc
    return completed.stdout.strip() if text else completed.stdout


def _validate_revision(root: Path, revision: str) -> None:
    if not REVISION.fullmatch(revision):
        raise PackageBuildError("runtime_candidate_revision must be exact SHA-1")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise PackageBuildError("runtime_candidate_revision does not equal HEAD")
    if _git(root, "status", "--porcelain"):
        raise PackageBuildError("source worktree must be clean")


def _safe_relative(path_text: str) -> PurePosixPath:
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PackageBuildError(f"unsafe package path: {path_text}")
    if any(part in EXCLUDED_PARTS for part in path.parts):
        raise PackageBuildError(f"excluded package path: {path_text}")
    if any(
        marker in part.lower()
        for part in path.parts
        for marker in SENSITIVE_PARTS
    ):
        raise PackageBuildError(f"sensitive filename is forbidden: {path_text}")
    return path


def _included(path_text: str) -> bool:
    return path_text in REQUIRED_FILES or path_text.startswith(INCLUDE_ROOTS)


def _tree_blobs(root: Path, revision: str) -> tuple[tuple[str, bytes], ...]:
    tracked = _git(root, "ls-tree", "-r", "--name-only", revision).splitlines()
    selected: list[tuple[str, bytes]] = []
    for path_text in tracked:
        if not _included(path_text):
            continue
        path = _safe_relative(path_text)
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        blob = _git(root, "show", f"{revision}:{path_text}", text=False)
        if LOCAL_PATH.search(blob):
            raise PackageBuildError(
                f"local absolute path is forbidden: {path_text}"
            )
        selected.append((path_text, blob))
    selected_names = {path for path, _blob in selected}
    missing = sorted(REQUIRED_FILES - selected_names)
    if missing:
        raise PackageBuildError(
            f"required package paths are missing from HEAD: {', '.join(missing)}"
        )
    if not any(path.startswith("main/") for path in selected_names):
        raise PackageBuildError("HEAD package lacks main implementation")
    if not any(path.startswith("runtime/") for path in selected_names):
        raise PackageBuildError("HEAD package lacks runtime implementation")
    return tuple(sorted(selected))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _validate_dependency_lock_pair(
    blobs: tuple[tuple[str, bytes], ...],
) -> None:
    blobs_by_path = dict(blobs)
    try:
        configuration = json.loads(
            blobs_by_path[
                "configs/runtime/runtime_sd35_flowmatch.json"
            ].decode("utf-8")
        )
        lock = configuration["dependency_lock"]
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise PackageBuildError("runtime dependency lock is invalid") from exc
    if (
        not isinstance(lock, list)
        or not lock
        or any(
            not isinstance(item, dict)
            or set(item) != {"package_name", "version_specifier"}
            or not isinstance(item["package_name"], str)
            or not item["package_name"]
            or not isinstance(item["version_specifier"], str)
            or not item["version_specifier"]
            for item in lock
        )
    ):
        raise PackageBuildError("runtime dependency lock is invalid")
    expected_requirements = tuple(
        f"{item['package_name']}=={item['version_specifier']}"
        for item in lock
        if item["package_name"] != "python"
    )
    try:
        requirement_lines = tuple(
            line.strip()
            for line in blobs_by_path[
                "requirements_runtime_qualification.txt"
            ].decode("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except (KeyError, UnicodeDecodeError) as exc:
        raise PackageBuildError(
            "runtime requirements lock is invalid"
        ) from exc
    if (
        not expected_requirements
        or not requirement_lines
        or requirement_lines != expected_requirements
    ):
        raise PackageBuildError(
            "runtime requirements do not exactly match dependency lock"
        )


def build_runtime_qualification_package(
    *,
    root: str | Path,
    output_zip: str | Path,
    runtime_candidate_revision: str,
) -> dict[str, object]:
    """Create a zip exclusively from tracked blobs at a clean exact HEAD."""

    root_path = Path(root).resolve()
    output_path = Path(output_zip).resolve()
    _validate_revision(root_path, runtime_candidate_revision)
    blobs = _tree_blobs(root_path, runtime_candidate_revision)
    if output_path == root_path or root_path in output_path.parents:
        raise PackageBuildError("execution package must be outside repository")
    if output_path.exists():
        raise PackageBuildError("package output target must not already exist")
    _validate_dependency_lock_pair(blobs)

    copied: list[dict[str, object]] = []
    archive_blobs: dict[str, bytes] = {}
    for source_path, blob in blobs:
        archive_path = (
            "README.md"
            if source_path == "scripts/experiment_execution/README.md"
            else source_path
        )
        if archive_path in archive_blobs:
            raise PackageBuildError(f"duplicate package path: {archive_path}")
        archive_blobs[archive_path] = blob
        copied.append(
            {
                "path": archive_path,
                "sha256": _sha256_bytes(blob),
                "size_bytes": len(blob),
            }
        )
    copied.sort(key=lambda item: str(item["path"]))
    manifest: dict[str, object] = {
        "package_schema_version": 1,
        "profile_name": PACKAGE_PROFILE,
        "runtime_candidate_revision": runtime_candidate_revision,
        "copied_files": copied,
        "excluded_parts": sorted(EXCLUDED_PARTS),
        "package_ready": True,
    }
    manifest_blob = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
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
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path_text, blob in sorted(archive_blobs.items()):
                archive.writestr(path_text, blob)
            archive.writestr("runtime_execution_manifest.json", manifest_blob)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {
        **manifest,
        "package_filename": output_path.name,
        "package_sha256": _sha256(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--runtime-candidate-revision", required=True)
    arguments = parser.parse_args()
    result = build_runtime_qualification_package(
        root=arguments.root,
        output_zip=arguments.output_zip,
        runtime_candidate_revision=arguments.runtime_candidate_revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
