"""Package-external trust bootstrap for semantic-texture Phase A."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence
import zipfile


PACKAGE_SCHEMA_VERSION = 1
PACKAGE_PROFILE = "semantic_texture_operational_preflight_package"
EMBEDDED_MANIFEST_PATH = "semantic_texture_operational_preflight_manifest.json"
ENTRYPOINT_PATH = (
    "scripts/experiment_execution/"
    "semantic_texture_operational_preflight_entrypoint.py"
)
MAX_ARCHIVE_MEMBERS = 128
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_ENTRYPOINT_ARGUMENTS = 32
MAX_ENTRYPOINT_ARGUMENT_BYTES = 256
REVISION = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PATH_PARTS = frozenset(
    {
        ".agents",
        ".codex",
        ".git",
        ".pytest_cache",
        "__pycache__",
        "governance",
        "notebooks",
        "outputs",
        "paper_artifacts",
        "tests",
    }
)
SENSITIVE_MARKERS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "id_rsa",
    "id_ed25519",
)
DELIVERY_FIELDS = {
    "archive_filename",
    "archive_sha256",
    "archive_size_bytes",
    "configuration_sha256",
    "delivery_manifest_schema_version",
    "embedded_manifest_sha256",
    "package_identity",
    "package_schema_version",
    "profile_name",
    "source_revision",
}
EMBEDDED_FIELDS = {
    "configuration_sha256",
    "copied_files",
    "entrypoint_path",
    "excluded_parts",
    "package_identity",
    "package_ready",
    "package_schema_version",
    "profile_name",
    "server_path",
    "source_revision",
}


class SemanticTextureBootstrapError(RuntimeError):
    """A package trust or Git-less execution boundary failed closed."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


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


def _safe_relative(path_text: str, *, stage: str) -> PurePosixPath:
    if (
        not path_text
        or "\\" in path_text
        or "\x00" in path_text
        or re.match(r"^[A-Za-z]:", path_text)
    ):
        raise SemanticTextureBootstrapError(stage, "unsafe archive path")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SemanticTextureBootstrapError(stage, "unsafe archive path")
    if any(part in FORBIDDEN_PATH_PARTS for part in path.parts):
        raise SemanticTextureBootstrapError(stage, "excluded archive path")
    if any(
        marker in part.lower()
        for part in path.parts
        for marker in SENSITIVE_MARKERS
    ):
        raise SemanticTextureBootstrapError(stage, "sensitive archive path")
    return path


def _load_json(path: Path, *, stage: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticTextureBootstrapError(stage, "JSON is unreadable") from exc
    if type(value) is not dict:
        raise SemanticTextureBootstrapError(stage, "JSON object is required")
    return value


def _validate_delivery_manifest(
    manifest: dict[str, object],
    *,
    archive_path: Path,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if (
        set(manifest) != DELIVERY_FIELDS
        or manifest.get("delivery_manifest_schema_version") != 1
        or manifest.get("package_schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("profile_name") != PACKAGE_PROFILE
        or manifest.get("archive_filename") != archive_path.name
        or manifest.get("archive_sha256") != expected_sha256
        or manifest.get("archive_size_bytes") != expected_size
        or DIGEST.fullmatch(str(manifest.get("configuration_sha256", "")))
        is None
        or DIGEST.fullmatch(str(manifest.get("embedded_manifest_sha256", "")))
        is None
        or DIGEST.fullmatch(str(manifest.get("package_identity", ""))) is None
        or REVISION.fullmatch(str(manifest.get("source_revision", ""))) is None
    ):
        raise SemanticTextureBootstrapError(
            "delivery_manifest",
            "delivery manifest identity drifted",
        )


def _validated_members(
    archive: zipfile.ZipFile,
) -> tuple[zipfile.ZipInfo, ...]:
    members = tuple(archive.infolist())
    if not members or len(members) > MAX_ARCHIVE_MEMBERS:
        raise SemanticTextureBootstrapError(
            "archive_safety",
            "archive member count is invalid",
        )
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise SemanticTextureBootstrapError(
            "archive_safety",
            "duplicate archive member",
        )
    total_size = 0
    for member in members:
        _safe_relative(member.filename, stage="archive_safety")
        if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
            raise SemanticTextureBootstrapError(
                "archive_safety",
                "non-regular archive member",
            )
        if member.file_size < 0 or member.file_size > MAX_MEMBER_BYTES:
            raise SemanticTextureBootstrapError(
                "archive_safety",
                "archive member size is invalid",
            )
        total_size += member.file_size
        if total_size > MAX_TOTAL_BYTES:
            raise SemanticTextureBootstrapError(
                "archive_safety",
                "archive total size is invalid",
            )
    if EMBEDDED_MANIFEST_PATH not in names:
        raise SemanticTextureBootstrapError(
            "archive_safety",
            "embedded manifest is missing",
        )
    return members


def _validate_embedded_manifest(
    archive: zipfile.ZipFile,
    members: tuple[zipfile.ZipInfo, ...],
    delivery: dict[str, object],
) -> dict[str, object]:
    try:
        embedded_blob = archive.read(EMBEDDED_MANIFEST_PATH)
        embedded = json.loads(embedded_blob)
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticTextureBootstrapError(
            "embedded_manifest",
            "embedded manifest is unreadable",
        ) from exc
    if (
        type(embedded) is not dict
        or set(embedded) != EMBEDDED_FIELDS
        or embedded.get("package_schema_version") != PACKAGE_SCHEMA_VERSION
        or embedded.get("profile_name") != PACKAGE_PROFILE
        or embedded.get("package_ready") is not True
        or embedded.get("entrypoint_path") != ENTRYPOINT_PATH
        or embedded.get("package_identity") != delivery["package_identity"]
        or embedded.get("source_revision") != delivery["source_revision"]
        or embedded.get("configuration_sha256")
        != delivery["configuration_sha256"]
        or sha256(embedded_blob).hexdigest()
        != delivery["embedded_manifest_sha256"]
    ):
        raise SemanticTextureBootstrapError(
            "embedded_manifest",
            "embedded manifest identity drifted",
        )
    copied_files = embedded.get("copied_files")
    if type(copied_files) is not list or not copied_files:
        raise SemanticTextureBootstrapError(
            "embedded_manifest",
            "embedded file roster is invalid",
        )
    expected: dict[str, tuple[int, str]] = {}
    for item in copied_files:
        if type(item) is not dict or set(item) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise SemanticTextureBootstrapError(
                "embedded_manifest",
                "embedded file entry is invalid",
            )
        path_text = item["path"]
        digest = item["sha256"]
        size = item["size_bytes"]
        if (
            type(path_text) is not str
            or type(digest) is not str
            or DIGEST.fullmatch(digest) is None
            or type(size) is not int
            or size < 0
            or path_text in expected
        ):
            raise SemanticTextureBootstrapError(
                "embedded_manifest",
                "embedded file entry identity drifted",
            )
        _safe_relative(path_text, stage="embedded_manifest")
        expected[path_text] = (size, digest)
    archive_names = {member.filename for member in members}
    if archive_names != {*expected, EMBEDDED_MANIFEST_PATH}:
        raise SemanticTextureBootstrapError(
            "embedded_manifest",
            "archive member roster differs from manifest",
        )
    for path_text, (size, digest) in expected.items():
        blob = archive.read(path_text)
        if len(blob) != size or sha256(blob).hexdigest() != digest:
            raise SemanticTextureBootstrapError(
                "embedded_manifest",
                "archive member content drifted",
            )
    return embedded


def _safe_extract(
    snapshot_path: Path,
    extract_root: Path,
    members: tuple[zipfile.ZipInfo, ...],
) -> None:
    if extract_root.exists():
        raise SemanticTextureBootstrapError(
            "extraction",
            "extract root must be absent",
        )
    extract_root.mkdir(parents=True)
    try:
        with zipfile.ZipFile(snapshot_path) as archive:
            for member in members:
                relative = _safe_relative(member.filename, stage="extraction")
                target = (extract_root / Path(*relative.parts)).resolve()
                if target == extract_root or extract_root not in target.parents:
                    raise SemanticTextureBootstrapError(
                        "extraction",
                        "archive target escapes extraction root",
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SemanticTextureBootstrapError(
            "extraction",
            "verified archive extraction failed",
        ) from exc


def _bounded_entrypoint_arguments(values: Sequence[str]) -> tuple[str, ...]:
    arguments = tuple(values)
    if (
        not arguments
        or len(arguments) > MAX_ENTRYPOINT_ARGUMENTS
        or any(
            not value
            or len(value.encode("utf-8")) > MAX_ENTRYPOINT_ARGUMENT_BYTES
            or "\x00" in value
            for value in arguments
        )
    ):
        raise SemanticTextureBootstrapError(
            "entrypoint_arguments",
            "entrypoint arguments are not bounded",
        )
    forbidden = {"--asset", "--whitening", "--cdf", "--route", "--mask"}
    if any(
        value == marker or value.startswith(marker + "=")
        for value in arguments
        for marker in forbidden
    ):
        raise SemanticTextureBootstrapError(
            "entrypoint_arguments",
            "private or unauthorized detector input is forbidden",
        )
    return arguments


def _result_path(extract_root: Path) -> Path:
    return extract_root.with_name(extract_root.name + ".bootstrap-result.json")


def _write_result_create_only(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_canonical_bytes(dict(value)))
    except FileExistsError as exc:
        raise SemanticTextureBootstrapError(
            "bootstrap_result",
            "bootstrap result target already exists",
        ) from exc


def run_semantic_texture_operational_preflight_bootstrap(
    *,
    archive: str | Path,
    manifest: str | Path,
    expected_sha256: str,
    expected_size: int,
    extract_root: str | Path,
    entrypoint_args: Sequence[str],
    environment: Mapping[str, str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[int, dict[str, object]]:
    """Authenticate, extract, and invoke the Git-less packaged entrypoint."""

    archive_path = Path(archive).resolve()
    manifest_path = Path(manifest).resolve()
    destination = Path(extract_root).resolve()
    result_path = _result_path(destination)
    if result_path.exists():
        raise SemanticTextureBootstrapError(
            "bootstrap_result",
            "bootstrap result target already exists",
        )
    stage = "arguments"
    source_revision: str | None = None
    package_identity: str | None = None
    try:
        if DIGEST.fullmatch(expected_sha256) is None:
            raise SemanticTextureBootstrapError(
                stage,
                "expected archive SHA-256 is invalid",
            )
        if type(expected_size) is not int or expected_size <= 0:
            raise SemanticTextureBootstrapError(
                stage,
                "expected archive size is invalid",
            )
        bounded_args = _bounded_entrypoint_arguments(entrypoint_args)
        if not archive_path.is_file() or archive_path.stat().st_size != expected_size:
            raise SemanticTextureBootstrapError(
                "archive_identity",
                "archive size drifted",
            )
        if _sha256_file(archive_path) != expected_sha256:
            raise SemanticTextureBootstrapError(
                "archive_identity",
                "archive SHA-256 drifted",
            )
        delivery = _load_json(manifest_path, stage="delivery_manifest")
        _validate_delivery_manifest(
            delivery,
            archive_path=archive_path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        source_revision = str(delivery["source_revision"])
        package_identity = str(delivery["package_identity"])
        stage = "archive_snapshot"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".verified.zip",
            delete=False,
        ) as handle:
            snapshot_path = Path(handle.name)
            with archive_path.open("rb") as source:
                shutil.copyfileobj(source, handle)
        try:
            if (
                snapshot_path.stat().st_size != expected_size
                or _sha256_file(snapshot_path) != expected_sha256
            ):
                raise SemanticTextureBootstrapError(
                    stage,
                    "verified archive snapshot drifted",
                )
            try:
                with zipfile.ZipFile(snapshot_path) as verified_archive:
                    members = _validated_members(verified_archive)
                    _validate_embedded_manifest(
                        verified_archive,
                        members,
                        delivery,
                    )
            except zipfile.BadZipFile as exc:
                raise SemanticTextureBootstrapError(
                    "archive_safety",
                    "archive structure is invalid",
                ) from exc
            stage = "extraction"
            _safe_extract(snapshot_path, destination, members)
        finally:
            snapshot_path.unlink(missing_ok=True)
        if (destination / ".git").exists():
            raise SemanticTextureBootstrapError(
                "gitless_boundary",
                "extracted package unexpectedly contains Git state",
            )
        stage = "entrypoint"
        command = (
            sys.executable,
            str(destination / ENTRYPOINT_PATH),
            *bounded_args,
        )
        completed = command_runner(
            command,
            cwd=destination,
            env=dict(os.environ if environment is None else environment),
            capture_output=True,
            text=True,
            check=False,
        )
        if type(completed.returncode) is not int:
            raise SemanticTextureBootstrapError(
                stage,
                "entrypoint return code is invalid",
            )
        result = {
            "aggregate": None,
            "blocked_class": (
                "implementation_blocked"
                if completed.returncode != 0
                else None
            ),
            "candidate_promoted": False,
            "entrypoint_exit_code": completed.returncode,
            "formal_tau_created": False,
            "package_identity": package_identity,
            "profile_id": "semantic_texture_operational_preflight",
            "science_started": False,
            "scientific_claims_supported": False,
            "scientific_unit_count": 0,
            "source_revision": source_revision,
            "stage": stage,
            "status": "passed" if completed.returncode == 0 else "blocked",
        }
        _write_result_create_only(result_path, result)
        return completed.returncode, {
            **result,
            "bootstrap_result_name": result_path.name,
        }
    except SemanticTextureBootstrapError as exc:
        failure = {
            "aggregate": None,
            "blocked_class": "integrity_blocked",
            "candidate_promoted": False,
            "formal_tau_created": False,
            "package_identity": package_identity,
            "profile_id": "semantic_texture_operational_preflight",
            "sanitized_error_category": type(exc).__name__,
            "sanitized_error_message": " ".join(str(exc).split())[:240],
            "science_started": False,
            "scientific_claims_supported": False,
            "scientific_unit_count": 0,
            "source_revision": source_revision,
            "stage": exc.stage,
            "status": "blocked",
        }
        _write_result_create_only(result_path, failure)
        return 3, {**failure, "bootstrap_result_name": result_path.name}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--entrypoint-args", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    exit_code, result = run_semantic_texture_operational_preflight_bootstrap(
        archive=arguments.archive,
        manifest=arguments.manifest,
        expected_sha256=arguments.expected_sha256,
        expected_size=arguments.expected_size,
        extract_root=arguments.extract_root,
        entrypoint_args=arguments.entrypoint_args,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
