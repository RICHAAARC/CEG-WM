"""Build one deterministic exact-revision experiment-execution package."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence
import zipfile


PACKAGE_SCHEMA_VERSION = 1
DELIVERY_MANIFEST_SCHEMA_VERSION = 1
PACKAGE_PROFILE = "experiment_execution_package"
ENTRYPOINT_IDENTITY = (
    "scripts.experiment_execution.experiment_execution_entrypoint:main"
)
ENTRYPOINT_MODULE = (
    "scripts.experiment_execution.experiment_execution_entrypoint"
)
ENTRYPOINT_PATH = (
    "scripts/experiment_execution/experiment_execution_entrypoint.py"
)
PACKAGE_TEST_FILES = frozenset(
    {
        "tests/integration/__init__.py",
        "tests/integration/test_packaged_experiment_execution.py",
        "tests/smoke/test_packaged_experiment_execution.py",
    }
)
EVIDENCE_SCOPE = (
    "infrastructure_synthetic_wiring_not_scientific_experiment_evidence"
)
REQUIRED_FILES = {
    "README.md",
    "pyproject.toml",
    "scripts/experiment_execution/__init__.py",
    ENTRYPOINT_PATH,
    *PACKAGE_TEST_FILES,
}
SOURCE_TO_ARCHIVE_PATH = {
    "templates/release_readmes/experiment_execution_package.md": "README.md",
}
INCLUDE_ROOTS = (
    "main/",
    "runtime/",
    "experiments/",
    "configs/",
    "infrastructure/",
    "tests/integration/",
    "tests/smoke/",
)
REQUIRED_ROOTS = INCLUDE_ROOTS
EXCLUDED_PARTS = {
    ".agents",
    ".codex",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "audit_reports",
    "baseline_results",
    "governance",
    "notebooks",
    "outputs",
    "paper_artifacts",
}
SENSITIVE_PARTS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "id_rsa",
    "id_ed25519",
)
_LOCAL_PATH_ROOT_COMPONENTS = (
    b"home",
    b"Users",
    b"mnt",
    b"tmp",
    b"var",
    b"opt",
    b"root",
)
LOCAL_PATH = re.compile(
    rb"(?<![A-Za-z0-9_])(?:/(?:"
    + b"|".join(_LOCAL_PATH_ROOT_COMPONENTS)
    + rb")/|[A-Za-z]:[\\/])"
)
_COLAB_ROOT_COMPONENT = b"content"
SENSITIVE_COLAB_PATH = re.compile(
    rb"(?<![A-Za-z0-9_])/"
    + _COLAB_ROOT_COMPONENT
    + rb"/"
    rb"(?:[^\s'\"`\\/]+/)*"
    rb"(?:private|secrets?|credentials?|model[-_]?weights?|"
    rb"weights?|checkpoints?)"
    rb"(?:/|(?=[\s'\"`]|$))",
    re.IGNORECASE,
)
REVISION = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ExperimentPackageBuildError(RuntimeError):
    """The requested package is unsafe or not bound to a clean revision."""


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
        raise ExperimentPackageBuildError(
            "source repository identity is unavailable"
        ) from exc
    if text:
        return completed.stdout.strip()
    return completed.stdout


def _validate_revision(root: Path, revision: str) -> None:
    if not REVISION.fullmatch(revision):
        raise ExperimentPackageBuildError(
            "committed_revision must be an exact Git revision"
        )
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ExperimentPackageBuildError(
            "committed_revision does not equal HEAD"
        )
    if _git(root, "status", "--porcelain"):
        raise ExperimentPackageBuildError(
            "source worktree must be clean"
        )


def _safe_relative(path_text: str) -> PurePosixPath:
    if (
        not path_text
        or "\\" in path_text
        or "\x00" in path_text
        or re.match(r"^[A-Za-z]:", path_text)
    ):
        raise ExperimentPackageBuildError(
            f"unsafe package path: {path_text}"
        )
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ExperimentPackageBuildError(
            f"unsafe package path: {path_text}"
        )
    if any(part in EXCLUDED_PARTS for part in path.parts):
        raise ExperimentPackageBuildError(
            f"excluded package path: {path_text}"
        )
    if any(
        marker in part.lower()
        for part in path.parts
        for marker in SENSITIVE_PARTS
    ):
        raise ExperimentPackageBuildError(
            f"sensitive package path: {path_text}"
        )
    return path


def _included_source(path_text: str) -> bool:
    if path_text.startswith(("tests/integration/", "tests/smoke/")):
        return path_text in PACKAGE_TEST_FILES
    return (
        path_text in SOURCE_TO_ARCHIVE_PATH
        or path_text == "pyproject.toml"
        or path_text
        in {
            "scripts/experiment_execution/__init__.py",
            ENTRYPOINT_PATH,
        }
        or path_text.startswith(INCLUDE_ROOTS[:5])
    )


def _tree_entries(
    root: Path,
    revision: str,
) -> tuple[tuple[str, bytes], ...]:
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "-z",
        revision,
        text=False,
    )
    if not isinstance(raw, bytes):
        raise ExperimentPackageBuildError(
            "Git tree listing returned invalid data"
        )
    selected: list[tuple[str, bytes]] = []
    archive_names: set[str] = set()
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        try:
            metadata, path_bytes = entry.split(b"\t", 1)
            mode, object_type, _object_id = metadata.split(b" ", 2)
            source_path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ExperimentPackageBuildError(
                "Git tree entry is invalid"
            ) from exc
        if not _included_source(source_path):
            continue
        if mode not in {b"100644", b"100755"} or object_type != b"blob":
            raise ExperimentPackageBuildError(
                f"non-regular package source is forbidden: {source_path}"
            )
        archive_path = SOURCE_TO_ARCHIVE_PATH.get(
            source_path,
            source_path,
        )
        _safe_relative(archive_path)
        if archive_path in archive_names:
            raise ExperimentPackageBuildError(
                f"duplicate package path: {archive_path}"
            )
        blob = _git(
            root,
            "show",
            f"{revision}:{source_path}",
            text=False,
        )
        if not isinstance(blob, bytes):
            raise ExperimentPackageBuildError(
                f"Git blob is invalid: {source_path}"
            )
        if LOCAL_PATH.search(blob):
            raise ExperimentPackageBuildError(
                f"local absolute path is forbidden: {source_path}"
            )
        if SENSITIVE_COLAB_PATH.search(blob):
            raise ExperimentPackageBuildError(
                "sensitive Colab absolute path is forbidden: "
                f"{source_path}"
            )
        archive_names.add(archive_path)
        selected.append((archive_path, blob))
    selected_names = {path for path, _blob in selected}
    missing_files = sorted(REQUIRED_FILES - selected_names)
    if missing_files:
        raise ExperimentPackageBuildError(
            "required package paths are missing from HEAD: "
            + ", ".join(missing_files)
        )
    missing_roots = [
        root_name
        for root_name in REQUIRED_ROOTS
        if not any(path.startswith(root_name) for path in selected_names)
    ]
    if missing_roots:
        raise ExperimentPackageBuildError(
            "required package roots are missing from HEAD: "
            + ", ".join(missing_roots)
        )
    return tuple(sorted(selected))


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _zip_info(path_text: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(
        path_text,
        date_time=(1980, 1, 1, 0, 0, 0),
    )
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_experiment_execution_package(
    *,
    root: str | Path,
    output_zip: str | Path,
    committed_revision: str,
    candidate_config_digest: str,
    execution_config_digest: str,
    input_manifest_digest: str,
) -> dict[str, object]:
    """Build a deterministic archive and adjacent delivery manifest."""

    root_path = Path(root).resolve()
    output_path = Path(output_zip).resolve()
    delivery_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    _validate_revision(root_path, committed_revision)
    for role, digest in (
        ("candidate_config_digest", candidate_config_digest),
        ("execution_config_digest", execution_config_digest),
        ("input_manifest_digest", input_manifest_digest),
    ):
        if not DIGEST.fullmatch(digest):
            raise ExperimentPackageBuildError(
                f"{role} must be a SHA-256 digest"
            )
    if output_path == root_path or root_path in output_path.parents:
        raise ExperimentPackageBuildError(
            "execution package must be outside repository"
        )
    if output_path.exists() or delivery_path.exists():
        raise ExperimentPackageBuildError(
            "package output targets must not already exist"
        )

    entries = _tree_entries(root_path, committed_revision)
    copied_files = [
        {
            "path": path_text,
            "sha256": _sha256_bytes(blob),
            "size_bytes": len(blob),
        }
        for path_text, blob in entries
    ]
    manifest = {
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "profile_name": PACKAGE_PROFILE,
        "committed_revision": committed_revision,
        "candidate_config_digest": candidate_config_digest,
        "execution_config_digest": execution_config_digest,
        "input_manifest_digest": input_manifest_digest,
        "entrypoint_identity": ENTRYPOINT_IDENTITY,
        "entrypoint_module": ENTRYPOINT_MODULE,
        "entrypoint_path": ENTRYPOINT_PATH,
        "evidence_scope": EVIDENCE_SCOPE,
        "copied_files": copied_files,
        "excluded_parts": sorted(EXCLUDED_PARTS),
        "package_ready": True,
    }
    manifest_blob = _canonical_json_bytes(manifest)
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
                _zip_info("experiment_execution_manifest.json"),
                manifest_blob,
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    delivery_manifest = {
        "delivery_manifest_schema_version": (
            DELIVERY_MANIFEST_SCHEMA_VERSION
        ),
        "package_filename": output_path.name,
        "archive_sha256": _sha256_file(output_path),
        "embedded_manifest_sha256": _sha256_bytes(manifest_blob),
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "profile_name": PACKAGE_PROFILE,
        "committed_revision": committed_revision,
        "candidate_config_digest": candidate_config_digest,
        "execution_config_digest": execution_config_digest,
        "input_manifest_digest": input_manifest_digest,
        "entrypoint_identity": ENTRYPOINT_IDENTITY,
        "evidence_scope": EVIDENCE_SCOPE,
        "package_ready": True,
    }
    delivery_path.write_bytes(_canonical_json_bytes(delivery_manifest))
    return {
        **delivery_manifest,
        "delivery_manifest_path": str(delivery_path),
        "copied_file_count": len(copied_files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--committed-revision", required=True)
    parser.add_argument("--candidate-config-digest", required=True)
    parser.add_argument("--execution-config-digest", required=True)
    parser.add_argument("--input-manifest-digest", required=True)
    arguments = parser.parse_args(argv)
    result = build_experiment_execution_package(
        root=arguments.root,
        output_zip=arguments.output_zip,
        committed_revision=arguments.committed_revision,
        candidate_config_digest=arguments.candidate_config_digest,
        execution_config_digest=arguments.execution_config_digest,
        input_manifest_digest=arguments.input_manifest_digest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
