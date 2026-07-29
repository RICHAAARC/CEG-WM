"""Trusted bootstrap for runtime qualification package schema version 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence


BOOTSTRAP_SCHEMA_VERSION = 1
PACKAGE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 2
SUPPORTED_PROFILES = ("smoke", "qualification", "replay")
MAX_ARCHIVE_MEMBERS = 512
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
REVISION = re.compile(r"^[0-9a-f]{40}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIREMENT = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9._+-]*$"
)
PACKAGE_REQUIRED_FILES = {
    "README.md",
    "configs/runtime/runtime_sd35_flowmatch.json",
    "pyproject.toml",
    "requirements_runtime_qualification.txt",
    "scripts/experiment_execution/__init__.py",
    "scripts/experiment_execution/runtime_qualification_runner.py",
}
PACKAGE_INCLUDE_ROOTS = ("main/", "runtime/")
PACKAGE_EXCLUDED_PARTS = {
    ".agents",
    ".codex",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "governance",
    "notebooks",
    "outputs",
}
SENSITIVE_PARTS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "id_rsa",
    "id_ed25519",
)
RESULT_FILES = {
    "environment_summary.json",
    "failures.jsonl",
    "run_summary.json",
    "runtime_checks.jsonl",
}
MANIFEST_FIELDS = {
    "copied_files",
    "excluded_parts",
    "package_ready",
    "package_schema_version",
    "profile_name",
    "runtime_candidate_revision",
}
SUMMARY_FIELDS = {
    "actual_dtype_status",
    "callback_status",
    "checks",
    "dependency_lock_evidence",
    "dependency_status",
    "determinism_status",
    "failure_classes",
    "failure_count",
    "finished_at_utc",
    "key_controls",
    "package_status",
    "profile",
    "prompt_identity",
    "prompt_sha256",
    "qk_status",
    "record_digests",
    "repetition_count",
    "replay_source_record_digests",
    "replay_source_revision",
    "replay_source_run_id",
    "result_schema_version",
    "result_zip_filename",
    "run_id",
    "run_status",
    "runtime_candidate_revision",
    "seed",
    "started_at_utc",
    "vae_status",
}


class BootstrapError(RuntimeError):
    """A fail-closed bootstrap error carrying its control-plane stage."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_absolute_path(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise BootstrapError("arguments", f"{name} must be an absolute path")
    return path.resolve()


def _strictly_within(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _safe_relative(path_text: str, *, stage: str) -> PurePosixPath:
    if (
        not path_text
        or "\\" in path_text
        or "\x00" in path_text
        or re.match(r"^[A-Za-z]:", path_text)
    ):
        raise BootstrapError(stage, "unsafe package path")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise BootstrapError(stage, "unsafe package path")
    if any(part in PACKAGE_EXCLUDED_PARTS for part in path.parts):
        raise BootstrapError(stage, "excluded package path")
    if any(
        marker in part.lower()
        for part in path.parts
        for marker in SENSITIVE_PARTS
    ):
        raise BootstrapError(stage, "sensitive package path")
    return path


def _included(path_text: str) -> bool:
    return path_text in PACKAGE_REQUIRED_FILES or path_text.startswith(
        PACKAGE_INCLUDE_ROOTS
    )


def _snapshot_validated_archive(
    archive_path: Path,
    snapshot_path: Path,
    expected_archive_sha256: str,
) -> str:
    if not SHA256.fullmatch(expected_archive_sha256):
        raise BootstrapError(
            "archive_digest",
            "expected package SHA-256 must be 64 lowercase hex characters",
    )
    if not archive_path.is_file():
        raise BootstrapError("archive_digest", "execution package is unavailable")
    if snapshot_path.exists():
        raise BootstrapError(
            "archive_digest",
            "ephemeral archive snapshot must not already exist",
        )
    digest = hashlib.sha256()
    try:
        with archive_path.open("rb") as source, snapshot_path.open("xb") as sink:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                sink.write(block)
    except OSError as exc:
        if snapshot_path.exists():
            snapshot_path.unlink()
        raise BootstrapError(
            "archive_digest",
            "execution package snapshot could not be created",
        ) from exc
    actual = digest.hexdigest()
    if actual != expected_archive_sha256:
        snapshot_path.unlink()
        raise BootstrapError(
            "archive_digest",
            "execution package SHA-256 differs from independent expected digest",
        )
    return actual


def _validated_archive_members(
    archive: zipfile.ZipFile,
) -> tuple[zipfile.ZipInfo, ...]:
    members = tuple(archive.infolist())
    if not members or len(members) > MAX_ARCHIVE_MEMBERS:
        raise BootstrapError("archive_safety", "archive member count is invalid")
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise BootstrapError("archive_safety", "duplicate archive member")
    total_size = 0
    for member in members:
        _safe_relative(member.filename, stage="archive_safety")
        if member.is_dir():
            raise BootstrapError(
                "archive_safety",
                "archive directory members are forbidden",
            )
        if stat.S_ISLNK(member.external_attr >> 16):
            raise BootstrapError("archive_safety", "archive symlink is forbidden")
        if member.file_size < 0 or member.file_size > MAX_MEMBER_BYTES:
            raise BootstrapError("archive_safety", "archive member size is invalid")
        total_size += member.file_size
        if total_size > MAX_TOTAL_BYTES:
            raise BootstrapError("archive_safety", "archive total size is invalid")
    if "runtime_execution_manifest.json" not in names:
        raise BootstrapError("archive_safety", "package manifest is missing")
    return members


def _safe_extract(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        raise BootstrapError(
            "archive_safety",
            "ephemeral package destination must not already exist",
        )
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _validated_archive_members(archive)
            for member in members:
                path = _safe_relative(
                    member.filename,
                    stage="archive_safety",
                )
                target = (destination / Path(*path.parts)).resolve()
                if not _strictly_within(target, destination):
                    raise BootstrapError(
                        "archive_safety",
                        "archive target escapes destination",
                    )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
    except (OSError, zipfile.BadZipFile) as exc:
        raise BootstrapError("archive_safety", "execution package is invalid") from exc


def _load_manifest(package_root: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            (package_root / "runtime_execution_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("manifest", "package manifest is invalid") from exc
    if (
        not isinstance(value, dict)
        or set(value) != MANIFEST_FIELDS
        or value.get("package_schema_version") != PACKAGE_SCHEMA_VERSION
        or value.get("profile_name") != "experiment_execution_package"
        or value.get("package_ready") is not True
        or value.get("excluded_parts") != sorted(PACKAGE_EXCLUDED_PARTS)
        or not REVISION.fullmatch(value.get("runtime_candidate_revision", ""))
    ):
        raise BootstrapError("manifest", "package manifest identity drifted")
    return value


def _verify_manifest_files(
    package_root: Path,
    manifest: dict[str, Any],
) -> None:
    entries = manifest.get("copied_files")
    if not isinstance(entries, list) or not entries:
        raise BootstrapError("manifest", "package manifest file list is invalid")
    expected: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise BootstrapError("manifest", "package manifest entry is invalid")
        path_text = entry["path"]
        digest = entry["sha256"]
        size = entry["size_bytes"]
        if (
            not isinstance(path_text, str)
            or not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
            or type(size) is not int
            or size < 0
        ):
            raise BootstrapError("manifest", "package manifest entry is invalid")
        _safe_relative(path_text, stage="manifest")
        if not _included(path_text):
            raise BootstrapError("manifest", "unallowlisted package path")
        if path_text in expected:
            raise BootstrapError("manifest", "duplicate manifest path")
        expected[path_text] = (size, digest)
    if not PACKAGE_REQUIRED_FILES <= set(expected):
        raise BootstrapError("manifest", "required package file is missing")
    if not any(name.startswith("main/") for name in expected) or not any(
        name.startswith("runtime/") for name in expected
    ):
        raise BootstrapError("manifest", "method or runtime package root is missing")
    actual: dict[str, Path] = {}
    for candidate in package_root.rglob("*"):
        if candidate.is_symlink():
            raise BootstrapError("manifest", "unpacked package symlink is forbidden")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(package_root).as_posix()
        if relative == "runtime_execution_manifest.json":
            continue
        _safe_relative(relative, stage="manifest")
        actual[relative] = candidate
    if set(actual) != set(expected):
        raise BootstrapError("manifest", "package file set differs from manifest")
    for path_text, candidate in actual.items():
        size, digest = expected[path_text]
        if candidate.stat().st_size != size or _sha256(candidate) != digest:
            raise BootstrapError(
                "manifest",
                f"package file identity drifted: {path_text}",
            )


def _verify_frozen_requirements(package_root: Path) -> None:
    requirements_path = package_root / "requirements_runtime_qualification.txt"
    configuration_path = (
        package_root / "configs/runtime/runtime_sd35_flowmatch.json"
    )
    try:
        lines = tuple(
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        lock = json.loads(configuration_path.read_text(encoding="utf-8"))[
            "dependency_lock"
        ]
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as exc:
        raise BootstrapError(
            "manifest",
            "frozen dependency inputs are invalid",
        ) from exc
    if not lines or any(not REQUIREMENT.fullmatch(line) for line in lines):
        raise BootstrapError(
            "manifest",
            "requirements must contain only exact distribution pins",
        )
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
        raise BootstrapError("manifest", "dependency lock is invalid")
    expected = tuple(
        f"{item['package_name']}=={item['version_specifier']}"
        for item in lock
        if item["package_name"] != "python"
    )
    if lines != expected:
        raise BootstrapError(
            "manifest",
            "requirements differ from the package dependency lock",
        )


def _install_requirements(
    package_root: Path,
    pip_cache: Path,
    environment: dict[str, str],
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    pip_cache.mkdir(parents=True, exist_ok=True)
    try:
        command_runner(
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--cache-dir",
                str(pip_cache),
                "--requirement",
                str(package_root / "requirements_runtime_qualification.txt"),
            ),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BootstrapError("dependency_install", "dependency installation failed") from exc


def _validate_result(
    result_zip: Path,
    *,
    profile: str,
    run_id: str,
    revision: str,
    runner_exit_code: int,
) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(result_zip) as archive:
            members = _validated_archive_members_for_result(archive)
            summary = json.loads(archive.read("run_summary.json"))
            json.loads(archive.read("environment_summary.json"))
            archive.read("runtime_checks.jsonl").decode("utf-8")
            archive.read("failures.jsonl").decode("utf-8")
    except (
        OSError,
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise BootstrapError("runner_result", "runner result is invalid") from exc
    if set(members) != RESULT_FILES:
        raise BootstrapError("runner_result", "runner result file set drifted")
    if (
        not isinstance(summary, dict)
        or set(summary) != SUMMARY_FIELDS
        or summary.get("result_schema_version") != RESULT_SCHEMA_VERSION
        or summary.get("profile") != profile
        or summary.get("run_id") != run_id
        or summary.get("runtime_candidate_revision") != revision
        or summary.get("result_zip_filename") != result_zip.name
        or summary.get("run_status") not in {"passed", "failed"}
        or type(summary.get("failure_count")) is not int
        or summary["failure_count"] < 0
        or not isinstance(summary.get("failure_classes"), list)
        or len(summary["failure_classes"]) != summary["failure_count"]
        or not isinstance(summary.get("checks"), list)
    ):
        raise BootstrapError("runner_result", "runner result identity drifted")
    passed = summary["run_status"] == "passed"
    incomplete = "incomplete" in summary["failure_classes"]
    if (
        runner_exit_code not in (0, 1, 2)
        or (runner_exit_code == 0) != passed
        or (runner_exit_code == 2) != (not passed and incomplete)
        or (runner_exit_code == 1 and (passed or incomplete))
        or (passed and summary["failure_count"] != 0)
        or (not passed and summary["failure_count"] == 0)
    ):
        raise BootstrapError("runner_result", "runner exit and result status drifted")
    return summary


def _validated_archive_members_for_result(
    archive: zipfile.ZipFile,
) -> tuple[str, ...]:
    members = tuple(archive.infolist())
    names = tuple(member.filename for member in members)
    if len(names) != len(set(names)):
        raise BootstrapError("runner_result", "duplicate result member")
    for member in members:
        path = PurePosixPath(member.filename)
        if (
            not member.filename
            or "\\" in member.filename
            or "\x00" in member.filename
            or path.is_absolute()
            or ".." in path.parts
            or stat.S_ISLNK(member.external_attr >> 16)
            or member.file_size > MAX_MEMBER_BYTES
        ):
            raise BootstrapError("runner_result", "unsafe result member")
    return names


def _atomic_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        source_digest = _sha256(source)
        if _sha256(temporary) != source_digest:
            raise BootstrapError("result_copy", "result copy identity drifted")
        temporary.replace(destination)
        return source_digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _bootstrap_digest() -> str:
    return _sha256(Path(__file__).resolve())


def _failure_artifact(
    *,
    persistent_root: Path,
    run_id: str,
    profile: str,
    expected_package_sha256: str,
    error: BootstrapError,
) -> Path:
    directory = persistent_root / "bootstrap_failures" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / f"ceg_wm_runtime_bootstrap_failure_{run_id}.zip"
    payload = {
        "bootstrap_failure_schema_version": 1,
        "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "bootstrap_sha256": _bootstrap_digest(),
        "artifact_kind": "bootstrap_failure",
        "run_id": run_id,
        "profile": profile,
        "status": "failed",
        "failure_stage": error.stage,
        "exception_type": type(error).__name__,
        "message": str(error),
        "expected_package_sha256": expected_package_sha256,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with zipfile.ZipFile(
        destination,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "bootstrap_failure.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    return destination


def run_bootstrap(
    *,
    profile: str,
    package_zip: str | Path,
    expected_package_sha256: str,
    ephemeral_root: str | Path,
    persistent_root: str | Path,
    replay_source: str | Path | None = None,
    run_id: str | None = None,
    environment: dict[str, str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[int, dict[str, Any]]:
    """Verify, install, execute, validate, and persist one qualification run."""

    chosen_run_id = run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    persistent = _explicit_absolute_path(persistent_root, "persistent_root")
    try:
        if profile not in SUPPORTED_PROFILES:
            raise BootstrapError("arguments", "unsupported profile")
        if not RUN_ID.fullmatch(chosen_run_id):
            raise BootstrapError("arguments", "run_id is invalid")
        archive_path = _explicit_absolute_path(package_zip, "package_zip")
        ephemeral = _explicit_absolute_path(ephemeral_root, "ephemeral_root")
        if _paths_overlap(ephemeral, persistent):
            raise BootstrapError(
                "arguments",
                "ephemeral and persistent roots must be disjoint",
            )
        source: Path | None = None
        if profile == "replay":
            if replay_source is None:
                raise BootstrapError("arguments", "replay source is required")
            source = _explicit_absolute_path(replay_source, "replay_source")
            if not _strictly_within(source, persistent) or not source.is_file():
                raise BootstrapError(
                    "arguments",
                    "replay source must be a file inside persistent root",
                )
        elif replay_source is not None:
            raise BootstrapError(
                "arguments",
                "replay source is only allowed for replay",
            )
        runtime_environment = dict(os.environ if environment is None else environment)
        if not runtime_environment.get("HF_TOKEN") or not runtime_environment.get(
            "CEG_WM_ROOT_KEY"
        ):
            raise BootstrapError(
                "secrets",
                "HF_TOKEN and CEG_WM_ROOT_KEY environment secrets are required",
            )
        workspace = ephemeral / f"bootstrap_{chosen_run_id}"
        package_root = workspace / "package"
        workspace.mkdir(parents=True, exist_ok=False)
        archive_snapshot = workspace / "execution_package.zip"
        archive_digest = _snapshot_validated_archive(
            archive_path,
            archive_snapshot,
            expected_package_sha256,
        )
        _safe_extract(archive_snapshot, package_root)
        manifest = _load_manifest(package_root)
        _verify_manifest_files(package_root, manifest)
        _verify_frozen_requirements(package_root)
        revision = manifest["runtime_candidate_revision"]
        pip_cache = ephemeral / "pip_cache"
        runtime_environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HOME": str(ephemeral / "hf_cache"),
                "PIP_CACHE_DIR": str(pip_cache),
            }
        )
        install_environment = dict(runtime_environment)
        install_environment.pop("HF_TOKEN", None)
        install_environment.pop("CEG_WM_ROOT_KEY", None)
        _install_requirements(
            package_root,
            pip_cache,
            install_environment,
            command_runner,
        )
        result_zip = (
            workspace
            / f"ceg_wm_runtime_qualification_{chosen_run_id}.zip"
        )
        command: list[str] = [
            sys.executable,
            "-m",
            "scripts.experiment_execution.runtime_qualification_runner",
            "--profile",
            profile,
            "--run-id",
            chosen_run_id,
            "--package-root",
            str(package_root),
            "--runtime-candidate-revision",
            revision,
            "--result-zip",
            str(result_zip),
            "--ephemeral-root",
            str(ephemeral),
            "--persistent-root",
            str(persistent),
        ]
        if source is not None:
            command.extend(("--replay-source", str(source)))
        try:
            completed = command_runner(
                tuple(command),
                cwd=package_root,
                check=False,
                capture_output=True,
                text=True,
                env=runtime_environment,
            )
        except OSError as exc:
            raise BootstrapError("runner_start", "runner could not start") from exc
        if not result_zip.is_file():
            raise BootstrapError(
                "runner_result",
                "runner did not produce a formal result archive",
            )
        summary = _validate_result(
            result_zip,
            profile=profile,
            run_id=chosen_run_id,
            revision=revision,
            runner_exit_code=completed.returncode,
        )
        destination = (
            persistent
            / "runs"
            / revision
            / chosen_run_id
            / result_zip.name
        )
        result_digest = _atomic_copy(result_zip, destination)
        return completed.returncode, {
            "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "artifact_kind": "qualification_result",
            "profile": profile,
            "run_id": chosen_run_id,
            "run_status": summary["run_status"],
            "runtime_candidate_revision": revision,
            "package_sha256": archive_digest,
            "result_zip": str(destination),
            "result_zip_sha256": result_digest,
            "runner_exit_code": completed.returncode,
        }
    except BootstrapError as error:
        failure_path = _failure_artifact(
            persistent_root=persistent,
            run_id=chosen_run_id,
            profile=profile,
            expected_package_sha256=expected_package_sha256,
            error=error,
        )
        return 3, {
            "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "artifact_kind": "bootstrap_failure",
            "profile": profile,
            "run_id": chosen_run_id,
            "run_status": "failed",
            "failure_stage": error.stage,
            "diagnostic_zip": str(failure_path),
            "bootstrap_exit_code": 3,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=SUPPORTED_PROFILES)
    parser.add_argument("--package-zip", required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--ephemeral-root", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--replay-source")
    parser.add_argument("--run-id")
    arguments = parser.parse_args(argv)
    exit_code, result = run_bootstrap(
        profile=arguments.profile,
        package_zip=arguments.package_zip,
        expected_package_sha256=arguments.expected_package_sha256,
        ephemeral_root=arguments.ephemeral_root,
        persistent_root=arguments.persistent_root,
        replay_source=arguments.replay_source,
        run_id=arguments.run_id,
    )
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
