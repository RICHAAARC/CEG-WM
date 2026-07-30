"""Package-external trust anchor for experiment package schema version 1."""

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
from datetime import datetime, timezone
from typing import Any, Callable, Sequence
import zipfile


BOOTSTRAP_IDENTITY = "ceg_wm_experiment_execution_bootstrap"
BOOTSTRAP_SCHEMA_VERSION = 1
PACKAGE_SCHEMA_VERSION = 1
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
INCLUDE_ROOTS = (
    "main/",
    "runtime/",
    "experiments/",
    "configs/",
    "infrastructure/",
    "tests/integration/",
    "tests/smoke/",
)
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
MANIFEST_FIELDS = {
    "candidate_config_digest",
    "committed_revision",
    "copied_files",
    "entrypoint_identity",
    "entrypoint_module",
    "entrypoint_path",
    "evidence_scope",
    "excluded_parts",
    "execution_config_digest",
    "input_manifest_digest",
    "package_ready",
    "package_schema_version",
    "profile_name",
}
SUMMARY_FIELDS = {
    "artifact_kind",
    "candidate_config_digest",
    "committed_revision",
    "entrypoint_identity",
    "entrypoint_schema_version",
    "evidence_scope",
    "excluded_count",
    "execution_config_digest",
    "execution_failure_count",
    "execution_scope",
    "gpu_executed",
    "held_out_evaluation_accessed",
    "input_manifest_digest",
    "record_collection_relative_path",
    "record_collection_sha256",
    "record_count",
    "replay_digest",
    "resource_failure_count",
    "run_id",
    "run_status",
    "scientific_claims_supported",
    "scientific_failure_count",
    "success_count",
}
REVISION = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
MAX_ARCHIVE_MEMBERS = 512
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


class ExperimentBootstrapError(RuntimeError):
    """A package-external trust check failed before package execution."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class ExperimentEntrypointError(RuntimeError):
    """The verified package entrypoint failed without becoming science."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _absolute(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ExperimentBootstrapError(
            "arguments",
            f"{role} must be an absolute path",
        )
    return path.resolve()


def _strictly_within(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _safe_relative(path_text: str, *, stage: str) -> PurePosixPath:
    if (
        not path_text
        or "\\" in path_text
        or "\x00" in path_text
        or re.match(r"^[A-Za-z]:", path_text)
    ):
        raise ExperimentBootstrapError(stage, "unsafe package path")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ExperimentBootstrapError(stage, "unsafe package path")
    if any(part in EXCLUDED_PARTS for part in path.parts):
        raise ExperimentBootstrapError(stage, "excluded package path")
    if any(
        marker in part.lower()
        for part in path.parts
        for marker in SENSITIVE_PARTS
    ):
        raise ExperimentBootstrapError(stage, "sensitive package path")
    return path


def _included(path_text: str) -> bool:
    if path_text.startswith(("tests/integration/", "tests/smoke/")):
        return path_text in PACKAGE_TEST_FILES
    return (
        path_text in REQUIRED_FILES
        or path_text.startswith(INCLUDE_ROOTS[:5])
    )


def _verify_bootstrap_identity(
    *,
    expected_bootstrap_identity: str,
    expected_bootstrap_schema_version: int,
    expected_bootstrap_sha256: str,
) -> None:
    if (
        expected_bootstrap_identity != BOOTSTRAP_IDENTITY
        or expected_bootstrap_schema_version != BOOTSTRAP_SCHEMA_VERSION
        or not DIGEST.fullmatch(expected_bootstrap_sha256)
        or _bootstrap_sha256() != expected_bootstrap_sha256
    ):
        raise ExperimentBootstrapError(
            "bootstrap_identity",
            "bootstrap identity, version, or SHA-256 differs from trust input",
        )


def _snapshot_archive(
    source_path: Path,
    snapshot_path: Path,
    expected_archive_sha256: str,
) -> str:
    if not DIGEST.fullmatch(expected_archive_sha256):
        raise ExperimentBootstrapError(
            "archive_digest",
            "expected archive SHA-256 is invalid",
        )
    if not source_path.is_file():
        raise ExperimentBootstrapError(
            "archive_digest",
            "execution package is unavailable",
        )
    digest = sha256()
    try:
        with source_path.open("rb") as source, snapshot_path.open("xb") as sink:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                sink.write(block)
    except OSError as exc:
        if snapshot_path.exists():
            snapshot_path.unlink()
        raise ExperimentBootstrapError(
            "archive_digest",
            "execution package snapshot failed",
        ) from exc
    observed = digest.hexdigest()
    if observed != expected_archive_sha256:
        snapshot_path.unlink()
        raise ExperimentBootstrapError(
            "archive_digest",
            "execution package SHA-256 differs from trust input",
        )
    return observed


def _archive_members(
    archive: zipfile.ZipFile,
) -> tuple[zipfile.ZipInfo, ...]:
    members = tuple(archive.infolist())
    if not members or len(members) > MAX_ARCHIVE_MEMBERS:
        raise ExperimentBootstrapError(
            "archive_safety",
            "archive member count is invalid",
        )
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ExperimentBootstrapError(
            "archive_safety",
            "duplicate archive member",
        )
    total_size = 0
    for member in members:
        _safe_relative(member.filename, stage="archive_safety")
        if member.is_dir():
            raise ExperimentBootstrapError(
                "archive_safety",
                "archive directory members are forbidden",
            )
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ExperimentBootstrapError(
                "archive_safety",
                "archive symlink is forbidden",
            )
        if member.file_size < 0 or member.file_size > MAX_MEMBER_BYTES:
            raise ExperimentBootstrapError(
                "archive_safety",
                "archive member size is invalid",
            )
        total_size += member.file_size
        if total_size > MAX_TOTAL_BYTES:
            raise ExperimentBootstrapError(
                "archive_safety",
                "archive total size is invalid",
            )
    if "experiment_execution_manifest.json" not in names:
        raise ExperimentBootstrapError(
            "archive_safety",
            "package manifest is missing",
        )
    return members


def _safe_extract(source_path: Path, destination: Path) -> None:
    if destination.exists():
        raise ExperimentBootstrapError(
            "archive_safety",
            "package extraction target already exists",
        )
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(source_path) as archive:
            for member in _archive_members(archive):
                relative = _safe_relative(
                    member.filename,
                    stage="archive_safety",
                )
                target = (
                    destination / Path(*relative.parts)
                ).resolve()
                if not _strictly_within(target, destination):
                    raise ExperimentBootstrapError(
                        "archive_safety",
                        "archive target escapes destination",
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ExperimentBootstrapError(
            "archive_safety",
            "execution package is invalid",
        ) from exc


def _load_and_verify_manifest(
    package_root: Path,
    *,
    expected_revision: str,
    expected_candidate_config_digest: str,
    expected_execution_config_digest: str,
    expected_input_manifest_digest: str,
) -> dict[str, Any]:
    try:
        manifest = json.loads(
            (
                package_root / "experiment_execution_manifest.json"
            ).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentBootstrapError(
            "manifest",
            "package manifest is invalid",
        ) from exc
    expected_identity = {
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "profile_name": PACKAGE_PROFILE,
        "committed_revision": expected_revision,
        "candidate_config_digest": expected_candidate_config_digest,
        "execution_config_digest": expected_execution_config_digest,
        "input_manifest_digest": expected_input_manifest_digest,
        "entrypoint_identity": ENTRYPOINT_IDENTITY,
        "entrypoint_module": ENTRYPOINT_MODULE,
        "entrypoint_path": ENTRYPOINT_PATH,
        "evidence_scope": EVIDENCE_SCOPE,
        "excluded_parts": sorted(EXCLUDED_PARTS),
        "package_ready": True,
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != MANIFEST_FIELDS
        or any(
            manifest.get(field) != value
            for field, value in expected_identity.items()
        )
    ):
        raise ExperimentBootstrapError(
            "manifest",
            "package manifest identity differs from trust inputs",
        )
    entries = manifest.get("copied_files")
    if not isinstance(entries, list) or not entries:
        raise ExperimentBootstrapError(
            "manifest",
            "package manifest file list is invalid",
        )
    expected_files: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ExperimentBootstrapError(
                "manifest",
                "package manifest entry is invalid",
            )
        path_text = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            type(path_text) is not str
            or type(digest) is not str
            or not DIGEST.fullmatch(digest)
            or type(size) is not int
            or isinstance(size, bool)
            or size < 0
        ):
            raise ExperimentBootstrapError(
                "manifest",
                "package manifest entry is invalid",
            )
        _safe_relative(path_text, stage="manifest")
        if not _included(path_text):
            raise ExperimentBootstrapError(
                "manifest",
                "unallowlisted package path",
            )
        if path_text in expected_files:
            raise ExperimentBootstrapError(
                "manifest",
                "duplicate manifest path",
            )
        expected_files[path_text] = (size, digest)
    if not REQUIRED_FILES <= set(expected_files):
        raise ExperimentBootstrapError(
            "manifest",
            "required package file is missing",
        )
    if any(
        not any(path.startswith(root) for path in expected_files)
        for root in INCLUDE_ROOTS
    ):
        raise ExperimentBootstrapError(
            "manifest",
            "required package root is missing",
        )
    actual_files: dict[str, Path] = {}
    for candidate in package_root.rglob("*"):
        if candidate.is_symlink():
            raise ExperimentBootstrapError(
                "manifest",
                "unpacked package symlink is forbidden",
            )
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(package_root).as_posix()
        if relative == "experiment_execution_manifest.json":
            continue
        _safe_relative(relative, stage="manifest")
        actual_files[relative] = candidate
    if set(actual_files) != set(expected_files):
        raise ExperimentBootstrapError(
            "manifest",
            "package file set differs from manifest",
        )
    for path_text, candidate in actual_files.items():
        size, digest = expected_files[path_text]
        if (
            candidate.stat().st_size != size
            or _sha256_file(candidate) != digest
        ):
            raise ExperimentBootstrapError(
                "manifest",
                f"package file identity drifted: {path_text}",
            )
    return manifest


def _validate_summary(
    summary_path: Path,
    *,
    result_root: Path,
    run_id: str,
    revision: str,
    candidate_digest: str,
    execution_digest: str,
    input_digest: str,
) -> dict[str, Any]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "entrypoint summary is invalid",
        ) from exc
    if (
        not isinstance(summary, dict)
        or set(summary) != SUMMARY_FIELDS
        or summary.get("entrypoint_schema_version") != 1
        or summary.get("entrypoint_identity") != ENTRYPOINT_IDENTITY
        or summary.get("artifact_kind")
        != "experiment_execution_result"
        or summary.get("execution_scope")
        != "cpu_synthetic_wiring_only"
        or summary.get("evidence_scope") != EVIDENCE_SCOPE
        or summary.get("run_status") != "completed"
        or summary.get("run_id") != run_id
        or summary.get("committed_revision") != revision
        or summary.get("candidate_config_digest") != candidate_digest
        or summary.get("execution_config_digest") != execution_digest
        or summary.get("input_manifest_digest") != input_digest
        or summary.get("scientific_claims_supported") is not False
        or summary.get("gpu_executed") is not False
        or summary.get("held_out_evaluation_accessed") is not False
    ):
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "entrypoint result identity drifted",
        )
    relative_record_path_text = summary.get(
        "record_collection_relative_path"
    )
    if type(relative_record_path_text) is not str:
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "record collection path is invalid",
        )
    try:
        relative_record_path = _safe_relative(
            relative_record_path_text,
            stage="entrypoint_result",
        )
    except ExperimentBootstrapError as exc:
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "record collection path is unsafe",
        ) from exc
    record_path = (
        result_root / Path(*relative_record_path.parts)
    ).resolve()
    if (
        not _strictly_within(record_path, result_root)
        or not record_path.is_file()
        or summary.get("record_collection_sha256")
        != _sha256_file(record_path)
    ):
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "record collection identity drifted",
        )
    for field in (
        "record_count",
        "success_count",
        "resource_failure_count",
        "scientific_failure_count",
        "execution_failure_count",
        "excluded_count",
    ):
        value = summary.get(field)
        if (
            type(value) is not int
            or isinstance(value, bool)
            or value < 0
        ):
            raise ExperimentEntrypointError(
                "entrypoint_result",
                "entrypoint result count is invalid",
            )
    if (
        summary["record_count"]
        != summary["success_count"]
        + summary["resource_failure_count"]
        + summary["scientific_failure_count"]
        + summary["execution_failure_count"]
        + summary["excluded_count"]
        or not DIGEST.fullmatch(summary.get("replay_digest", ""))
    ):
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "entrypoint result counts or replay digest drifted",
        )
    return summary


def _zip_info(path_text: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(
        path_text,
        date_time=(1980, 1, 1, 0, 0, 0),
    )
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _result_archive(
    result_root: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=False)
    files = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file()
    )
    with zipfile.ZipFile(
        destination,
        mode="x",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for source in files:
            relative = source.relative_to(result_root).as_posix()
            _safe_relative(relative, stage="result_copy")
            archive.writestr(_zip_info(relative), source.read_bytes())
    return _sha256_file(destination)


def _diagnostic_archive(
    *,
    persistent_root: Path,
    run_id: str,
    artifact_kind: str,
    expected_archive_sha256: str,
    stage: str,
    error: BaseException,
) -> Path:
    if artifact_kind == "bootstrap_failure":
        directory = persistent_root / "bootstrap_failures" / run_id
        filename = f"ceg_wm_experiment_bootstrap_failure_{run_id}.zip"
    else:
        directory = persistent_root / "entrypoint_failures" / run_id
        filename = f"ceg_wm_experiment_entrypoint_failure_{run_id}.zip"
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / filename
    payload = {
        "diagnostic_schema_version": 1,
        "bootstrap_identity": BOOTSTRAP_IDENTITY,
        "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "bootstrap_sha256": _bootstrap_sha256(),
        "artifact_kind": artifact_kind,
        "run_id": run_id,
        "status": "failed",
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
        "expected_archive_sha256": expected_archive_sha256,
        "scientific_claims_supported": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with zipfile.ZipFile(
        destination,
        mode="x",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr(
            _zip_info("diagnostic.json"),
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    return destination


def run_bootstrap(
    *,
    package_zip: str | Path,
    expected_archive_sha256: str,
    expected_bootstrap_identity: str,
    expected_bootstrap_schema_version: int,
    expected_bootstrap_sha256: str,
    expected_revision: str,
    expected_candidate_config_digest: str,
    expected_execution_config_digest: str,
    expected_input_manifest_digest: str,
    ephemeral_root: str | Path,
    persistent_root: str | Path,
    run_id: str,
    environment: dict[str, str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = (
        subprocess.run
    ),
) -> tuple[int, dict[str, object]]:
    """Verify external trust inputs before importing or running package code."""

    persistent = _absolute(persistent_root, "persistent_root")
    diagnostic_run_id = (
        run_id
        if type(run_id) is str and SAFE_ID.fullmatch(run_id)
        else "invalid-run-id-"
        + sha256(str(run_id).encode("utf-8")).hexdigest()[:16]
    )
    try:
        _verify_bootstrap_identity(
            expected_bootstrap_identity=expected_bootstrap_identity,
            expected_bootstrap_schema_version=(
                expected_bootstrap_schema_version
            ),
            expected_bootstrap_sha256=expected_bootstrap_sha256,
        )
        if (
            not REVISION.fullmatch(expected_revision)
            or not DIGEST.fullmatch(expected_candidate_config_digest)
            or not DIGEST.fullmatch(expected_execution_config_digest)
            or not DIGEST.fullmatch(expected_input_manifest_digest)
            or not SAFE_ID.fullmatch(run_id)
        ):
            raise ExperimentBootstrapError(
                "arguments",
                "revision, digest, or run identity is invalid",
            )
        archive_path = _absolute(package_zip, "package_zip")
        ephemeral = _absolute(ephemeral_root, "ephemeral_root")
        if _overlap(ephemeral, persistent):
            raise ExperimentBootstrapError(
                "arguments",
                "ephemeral and persistent roots must be disjoint",
            )
        workspace = ephemeral / f"experiment_bootstrap_{run_id}"
        workspace.mkdir(parents=True, exist_ok=False)
        archive_snapshot = workspace / "execution_package.zip"
        archive_digest = _snapshot_archive(
            archive_path,
            archive_snapshot,
            expected_archive_sha256,
        )
        package_root = workspace / "package"
        _safe_extract(archive_snapshot, package_root)
        _load_and_verify_manifest(
            package_root,
            expected_revision=expected_revision,
            expected_candidate_config_digest=(
                expected_candidate_config_digest
            ),
            expected_execution_config_digest=(
                expected_execution_config_digest
            ),
            expected_input_manifest_digest=(
                expected_input_manifest_digest
            ),
        )
        result_root = workspace / "result"
        runtime_workspace = workspace / "runtime_workspace"
        command = (
            sys.executable,
            "-m",
            ENTRYPOINT_MODULE,
            "--package-root",
            str(package_root),
            "--output-root",
            str(result_root),
            "--workspace-root",
            str(runtime_workspace),
            "--committed-revision",
            expected_revision,
            "--expected-candidate-config-digest",
            expected_candidate_config_digest,
            "--expected-execution-config-digest",
            expected_execution_config_digest,
            "--expected-input-manifest-digest",
            expected_input_manifest_digest,
            "--run-id",
            run_id,
        )
        runtime_environment = dict(
            os.environ if environment is None else environment
        )
        runtime_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = command_runner(
                command,
                cwd=package_root,
                check=False,
                capture_output=True,
                text=True,
                env=runtime_environment,
            )
        except OSError as exc:
            raise ExperimentBootstrapError(
                "entrypoint_start",
                "verified package entrypoint could not start",
            ) from exc
        if completed.returncode != 0:
            raise ExperimentEntrypointError(
                "entrypoint_execution",
                "verified package entrypoint failed",
            )
        summary = _validate_summary(
            result_root / "execution_summary.json",
            result_root=result_root,
            run_id=run_id,
            revision=expected_revision,
            candidate_digest=expected_candidate_config_digest,
            execution_digest=expected_execution_config_digest,
            input_digest=expected_input_manifest_digest,
        )
        destination = (
            persistent
            / "runs"
            / expected_revision
            / run_id
            / f"ceg_wm_experiment_execution_{run_id}.zip"
        )
        result_digest = _result_archive(result_root, destination)
        return 0, {
            "bootstrap_identity": BOOTSTRAP_IDENTITY,
            "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "artifact_kind": "experiment_execution_result",
            "execution_scope": summary["execution_scope"],
            "evidence_scope": EVIDENCE_SCOPE,
            "run_id": run_id,
            "run_status": "completed",
            "committed_revision": expected_revision,
            "archive_sha256": archive_digest,
            "result_zip": str(destination),
            "result_zip_sha256": result_digest,
            "scientific_claims_supported": False,
        }
    except ExperimentBootstrapError as error:
        failure_path = _diagnostic_archive(
            persistent_root=persistent,
            run_id=diagnostic_run_id,
            artifact_kind="bootstrap_failure",
            expected_archive_sha256=expected_archive_sha256,
            stage=error.stage,
            error=error,
        )
        return 3, {
            "bootstrap_identity": BOOTSTRAP_IDENTITY,
            "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "artifact_kind": "bootstrap_failure",
            "run_id": diagnostic_run_id,
            "run_status": "failed",
            "failure_stage": error.stage,
            "diagnostic_zip": str(failure_path),
            "scientific_claims_supported": False,
        }
    except ExperimentEntrypointError as error:
        failure_path = _diagnostic_archive(
            persistent_root=persistent,
            run_id=diagnostic_run_id,
            artifact_kind="execution_entrypoint_failure",
            expected_archive_sha256=expected_archive_sha256,
            stage=error.stage,
            error=error,
        )
        return 4, {
            "bootstrap_identity": BOOTSTRAP_IDENTITY,
            "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "artifact_kind": "execution_entrypoint_failure",
            "run_id": diagnostic_run_id,
            "run_status": "failed",
            "failure_stage": error.stage,
            "diagnostic_zip": str(failure_path),
            "scientific_claims_supported": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-zip", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-bootstrap-identity", required=True)
    parser.add_argument(
        "--expected-bootstrap-schema-version",
        required=True,
        type=int,
    )
    parser.add_argument("--expected-bootstrap-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument(
        "--expected-candidate-config-digest",
        required=True,
    )
    parser.add_argument(
        "--expected-execution-config-digest",
        required=True,
    )
    parser.add_argument(
        "--expected-input-manifest-digest",
        required=True,
    )
    parser.add_argument("--ephemeral-root", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args(argv)
    exit_code, result = run_bootstrap(
        package_zip=arguments.package_zip,
        expected_archive_sha256=arguments.expected_archive_sha256,
        expected_bootstrap_identity=arguments.expected_bootstrap_identity,
        expected_bootstrap_schema_version=(
            arguments.expected_bootstrap_schema_version
        ),
        expected_bootstrap_sha256=arguments.expected_bootstrap_sha256,
        expected_revision=arguments.expected_revision,
        expected_candidate_config_digest=(
            arguments.expected_candidate_config_digest
        ),
        expected_execution_config_digest=(
            arguments.expected_execution_config_digest
        ),
        expected_input_manifest_digest=(
            arguments.expected_input_manifest_digest
        ),
        ephemeral_root=arguments.ephemeral_root,
        persistent_root=arguments.persistent_root,
        run_id=arguments.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
