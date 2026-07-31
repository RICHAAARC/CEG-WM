"""Package-external trust anchor for the C1 threshold-fit package."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import metadata
import importlib.util
import inspect
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
from typing import Any, Callable, Mapping, Sequence
import zipfile


TRUSTED_BOOTSTRAP_MODULE_ALIAS = "ceg_wm_verified_experiment_bootstrap"
sys.modules[TRUSTED_BOOTSTRAP_MODULE_ALIAS] = sys.modules[__name__]
BOOTSTRAP_IDENTITY = "ceg_wm_experiment_execution_bootstrap"
BOOTSTRAP_SCHEMA_VERSION = 2
PACKAGE_SCHEMA_VERSION = 2
DELIVERY_MANIFEST_SCHEMA_VERSION = 2
PACKAGE_PROFILE = "c1_hf_threshold_fit_execution_package"
ENTRYPOINT_IDENTITY = (
    "scripts.experiment_execution.experiment_execution_entrypoint:"
    "execute_verified_threshold_fit_shard"
)
ENTRYPOINT_MODULE = (
    "scripts.experiment_execution.experiment_execution_entrypoint"
)
ENTRYPOINT_PATH = (
    "scripts/experiment_execution/experiment_execution_entrypoint.py"
)
EVIDENCE_SCOPE = (
    "c1_hf_threshold_fit_execution_only_no_tau_approval_no_confirmation_access"
)
C1_DEPENDENCY_LOCK_PATH = "requirements_c1_threshold_fit.txt"
C1_DEPENDENCY_LOCK_SHA256 = (
    "07a4c1bbe6fc5e7e6b38334c5a9919a8565b810a9aae7820b61c24cee91270de"
)
C1_PYPI_INDEX_URL = "https://pypi.org/simple"
C1_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
C1_NVIDIA_INDEX_URL = "https://pypi.nvidia.com"
EXACT_FILES = frozenset(
    {
        "configs/experiments/assets/parti_prompts_dataset_snapshot.txt",
        "configs/experiments/c1_hf_content_threshold_fit_manifest.json",
        "configs/experiments/c1_hf_metric_implementation.json",
        "configs/experiments/c1_hf_prompt_roster.json",
        "configs/experiments/c1_hf_reference_run.json",
        "configs/experiments/c1_hf_threshold_fit_execution.json",
        "configs/experiments/internal_execution_components.json",
        "configs/runtime/runtime_sd35_flowmatch.json",
        "experiments/__init__.py",
        "experiments/attacks/__init__.py",
        "experiments/attacks/geometric.py",
        "experiments/methods/__init__.py",
        "experiments/methods/ceg_wm.py",
        "experiments/metrics/__init__.py",
        "experiments/metrics/binomial.py",
        "experiments/metrics/c1_hf_reference.py",
        "experiments/metrics/internal.py",
        "experiments/protocol/__init__.py",
        "experiments/protocol/c1_hf_reference.py",
        "experiments/protocol/c1_hf_threshold_fit_records.py",
        "experiments/protocol/internal_case.py",
        "experiments/protocol/internal_matrix.py",
        "experiments/protocol/internal_record_registry.py",
        "experiments/protocol/internal_records.py",
        "experiments/protocol/internal_splits.py",
        "experiments/protocol/internal_validation.py",
        "experiments/runners/__init__.py",
        "experiments/runners/c1_hf_threshold_fit.py",
        "experiments/runners/formal_operations.py",
        "experiments/runners/internal.py",
        "experiments/runners/record_writer.py",
        "main/__init__.py",
        "main/content_chain/__init__.py",
        "main/content_chain/detector.py",
        "main/content_chain/embedder.py",
        "main/content_chain/hf_carrier.py",
        "main/content_chain/hf_detector.py",
        "main/content_chain/lf_carrier.py",
        "main/content_chain/lf_detector.py",
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
        "pyproject.toml",
        C1_DEPENDENCY_LOCK_PATH,
        "runtime/__init__.py",
        "runtime/adapter.py",
        "runtime/backend.py",
        "runtime/configuration.py",
        "runtime/content_write.py",
        "runtime/qk_observation.py",
        "runtime/sd35_backend.py",
        "scripts/experiment_execution/__init__.py",
        ENTRYPOINT_PATH,
    }
)
REQUIRED_FILES = {
    "README.md",
    *EXACT_FILES,
}
REQUIRED_ROOTS = (
    "main/",
    "runtime/",
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
REVISION = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
MAX_ARCHIVE_MEMBERS = 512
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
DELIVERY_MANIFEST_FIELDS = {
    "archive_sha256",
    "candidate_config_digest",
    "committed_revision",
    "delivery_manifest_schema_version",
    "embedded_manifest_sha256",
    "entrypoint_identity",
    "evidence_scope",
    "execution_config_digest",
    "input_manifest_digest",
    "package_filename",
    "package_ready",
    "package_schema_version",
    "profile_name",
}


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


_CAPABILITY_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class VerifiedThresholdFitPackageCapability:
    """Single-use authority issued only after package-external verification."""

    package_root: Path
    bootstrap_sha256: str
    committed_revision: str
    archive_sha256: str
    embedded_manifest_sha256: str
    copied_file_set_digest: str
    entrypoint_path: str
    entrypoint_sha256: str
    execution_config_digest: str
    input_manifest_digest: str
    candidate_config_digest: str
    _construction_token: object = field(repr=False)
    _issuer_module: object = field(repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _CAPABILITY_CONSTRUCTION_TOKEN:
            raise ExperimentBootstrapError(
                "package_capability",
                "verified package capability construction is forbidden",
            )

    def consume_for_threshold_fit_runner(
        self,
        *,
        expected_package_root: Path,
        expected_entrypoint_path: str,
    ) -> dict[str, object]:
        if self._consumed:
            raise ExperimentBootstrapError(
                "package_capability",
                "verified package capability was already consumed",
            )
        if (
            expected_package_root.resolve() != self.package_root
            or expected_entrypoint_path != self.entrypoint_path
        ):
            raise ExperimentBootstrapError(
                "package_capability",
                "verified package capability consumer identity drifted",
            )
        manifest_path = self.package_root / "experiment_execution_manifest.json"
        if manifest_path.is_symlink():
            raise ExperimentBootstrapError(
                "package_capability",
                "verified package manifest became a symlink",
            )
        try:
            manifest_blob = manifest_path.read_bytes()
            manifest = json.loads(manifest_blob)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExperimentBootstrapError(
                "package_capability",
                "verified package manifest became unreadable",
            ) from exc
        entries = manifest.get("copied_files") if type(manifest) is dict else None
        if (
            sha256(manifest_blob).hexdigest() != self.embedded_manifest_sha256
            or type(entries) is not list
            or _canonical_digest(entries) != self.copied_file_set_digest
        ):
            raise ExperimentBootstrapError(
                "package_capability",
                "verified package manifest changed after verification",
            )
        expected_files: dict[str, tuple[int, str]] = {}
        for entry in entries:
            if type(entry) is not dict or set(entry) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise ExperimentBootstrapError(
                    "package_capability",
                    "verified package file identity became invalid",
                )
            path_text = entry["path"]
            digest = entry["sha256"]
            size = entry["size_bytes"]
            if (
                type(path_text) is not str
                or type(digest) is not str
                or DIGEST.fullmatch(digest) is None
                or type(size) is not int
                or isinstance(size, bool)
                or size < 0
                or path_text in expected_files
            ):
                raise ExperimentBootstrapError(
                    "package_capability",
                    "verified package file identity became invalid",
                )
            _safe_relative(path_text, stage="package_capability")
            expected_files[path_text] = (size, digest)
        actual_files: dict[str, Path] = {}
        for candidate in self.package_root.rglob("*"):
            if candidate.is_symlink():
                raise ExperimentBootstrapError(
                    "package_capability",
                    "verified package gained a symlink after verification",
                )
            if candidate.is_file() and candidate != manifest_path:
                actual_files[
                    candidate.relative_to(self.package_root).as_posix()
                ] = candidate
        if set(actual_files) != set(expected_files):
            raise ExperimentBootstrapError(
                "package_capability",
                "verified package file set changed after verification",
            )
        for path_text, candidate in actual_files.items():
            expected_size, expected_digest = expected_files[path_text]
            if (
                candidate.stat().st_size != expected_size
                or _sha256_file(candidate) != expected_digest
            ):
                raise ExperimentBootstrapError(
                    "package_capability",
                    "verified package file changed after verification",
                )
        object.__setattr__(self, "_consumed", True)
        return {
            "archive_sha256": self.archive_sha256,
            "bootstrap_sha256": self.bootstrap_sha256,
            "candidate_config_digest": self.candidate_config_digest,
            "committed_revision": self.committed_revision,
            "copied_file_set_digest": self.copied_file_set_digest,
            "embedded_manifest_sha256": self.embedded_manifest_sha256,
            "entrypoint_path": self.entrypoint_path,
            "entrypoint_sha256": self.entrypoint_sha256,
            "execution_config_digest": self.execution_config_digest,
            "input_manifest_digest": self.input_manifest_digest,
            "package_root": str(self.package_root),
        }


def _issue_threshold_fit_package_capability(
    *,
    package_root: Path,
    committed_revision: str,
    archive_sha256: str,
    embedded_manifest_sha256: str,
    copied_file_set_digest: str,
    entrypoint_sha256: str,
    execution_config_digest: str,
    input_manifest_digest: str,
    candidate_config_digest: str,
) -> VerifiedThresholdFitPackageCapability:
    """Issue the runner lease after all external trust checks succeed."""

    return VerifiedThresholdFitPackageCapability(
        package_root=package_root.resolve(),
        bootstrap_sha256=_bootstrap_sha256(),
        committed_revision=committed_revision,
        archive_sha256=archive_sha256,
        embedded_manifest_sha256=embedded_manifest_sha256,
        copied_file_set_digest=copied_file_set_digest,
        entrypoint_path=ENTRYPOINT_PATH,
        entrypoint_sha256=entrypoint_sha256,
        execution_config_digest=execution_config_digest,
        input_manifest_digest=input_manifest_digest,
        candidate_config_digest=candidate_config_digest,
        _construction_token=_CAPABILITY_CONSTRUCTION_TOKEN,
        _issuer_module=sys.modules[__name__],
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


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
    return path_text in REQUIRED_FILES


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


def _verify_delivery_manifest(
    path: Path,
    *,
    expected_delivery_manifest_sha256: str,
    package_filename: str,
    archive_sha256: str,
    embedded_manifest_sha256: str,
    revision: str,
) -> dict[str, Any]:
    if (
        DIGEST.fullmatch(expected_delivery_manifest_sha256) is None
        or not path.is_file()
        or path.is_symlink()
    ):
        raise ExperimentBootstrapError(
            "delivery_manifest",
            "delivery manifest trust input is invalid",
        )
    try:
        blob = path.read_bytes()
        raw = json.loads(blob)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentBootstrapError(
            "delivery_manifest",
            "delivery manifest is unreadable",
        ) from exc
    if (
        sha256(blob).hexdigest() != expected_delivery_manifest_sha256
        or type(raw) is not dict
        or set(raw) != DELIVERY_MANIFEST_FIELDS
        or raw["delivery_manifest_schema_version"]
        != DELIVERY_MANIFEST_SCHEMA_VERSION
        or raw["package_filename"] != package_filename
        or raw["archive_sha256"] != archive_sha256
        or raw["embedded_manifest_sha256"] != embedded_manifest_sha256
        or raw["package_schema_version"] != PACKAGE_SCHEMA_VERSION
        or raw["profile_name"] != PACKAGE_PROFILE
        or raw["committed_revision"] != revision
        or type(raw["candidate_config_digest"]) is not str
        or DIGEST.fullmatch(raw["candidate_config_digest"]) is None
        or type(raw["execution_config_digest"]) is not str
        or DIGEST.fullmatch(raw["execution_config_digest"]) is None
        or type(raw["input_manifest_digest"]) is not str
        or DIGEST.fullmatch(raw["input_manifest_digest"]) is None
        or raw["entrypoint_identity"] != ENTRYPOINT_IDENTITY
        or raw["evidence_scope"] != EVIDENCE_SCOPE
        or raw["package_ready"] is not True
    ):
        raise ExperimentBootstrapError(
            "delivery_manifest",
            "delivery manifest identity differs from trust inputs",
        )
    return raw


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
        for root in REQUIRED_ROOTS
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


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _load_verified_dependency_lock(path: Path) -> dict[str, tuple[str, str]]:
    try:
        blob = path.read_bytes()
        lines = blob.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ExperimentBootstrapError(
            "dependency_install",
            "verified dependency lock is unreadable",
        ) from exc
    if sha256(blob).hexdigest() != C1_DEPENDENCY_LOCK_SHA256:
        raise ExperimentBootstrapError(
            "dependency_install",
            "verified C1 dependency lock identity drifted",
        )
    requirements: dict[str, tuple[str, str]] = {}
    for line in lines:
        match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+_-]*)",
            line,
        )
        if match is None:
            raise ExperimentBootstrapError(
                "dependency_install",
                "verified dependency lock contains a non-exact requirement",
            )
        distribution, version = match.groups()
        normalized = _normalized_distribution_name(distribution)
        if normalized in requirements:
            raise ExperimentBootstrapError(
                "dependency_install",
                "verified dependency lock contains a duplicate distribution",
            )
        requirements[normalized] = (distribution, version)
    if len(requirements) != 62 or requirements.get("torch") != (
        "torch",
        "2.11.0+cu128",
    ):
        raise ExperimentBootstrapError(
            "dependency_install",
            "verified C1 dependency lock closure drifted",
        )
    return requirements


def _target_distribution_versions(target: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in metadata.distributions(path=[str(target)]):
        name = distribution.metadata.get("Name")
        if type(name) is str:
            normalized = _normalized_distribution_name(name)
            if normalized in versions:
                raise ExperimentBootstrapError(
                    "dependency_install",
                    "dependency target contains duplicate distribution metadata",
                )
            versions[normalized] = distribution.version
    return versions


def _require_exact_target_distribution_versions(
    observed: Mapping[str, str],
    expected: Mapping[str, str],
    *,
    target_kind: str,
) -> None:
    if dict(observed) != dict(expected):
        raise ExperimentBootstrapError(
            "dependency_install",
            f"{target_kind} dependency distribution set or versions differ "
            "from the verified lock",
        )


def _verify_imported_torch_version(expected_version: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ExperimentBootstrapError(
            "dependency_install",
            "verified torch runtime cannot be imported",
        ) from exc
    imported_version = str(getattr(torch, "__version__", ""))
    if imported_version != expected_version:
        raise ExperimentBootstrapError(
            "dependency_install",
            "imported torch version differs from the verified lock",
        )


def _prepare_verified_dependencies(
    *,
    package_root: Path,
    ephemeral_root: Path,
    environment: Mapping[str, str],
) -> Path | None:
    """Strictly reuse the lock or install it into ephemeral storage."""

    lock_path = package_root / C1_DEPENDENCY_LOCK_PATH
    requirements = _load_verified_dependency_lock(lock_path)
    expected_versions = {
        normalized: version
        for normalized, (_distribution, version) in requirements.items()
    }
    reusable = True
    for distribution, version in requirements.values():
        try:
            observed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            reusable = False
            break
        if observed != version:
            reusable = False
            break
    if reusable:
        _verify_imported_torch_version(requirements["torch"][1])
        return None
    lock_digest = _sha256_file(lock_path)
    dependency_cache_root = ephemeral_root / "verified_dependencies"
    cache_root = ephemeral_root / "pip_cache"
    dependency_cache_root.mkdir(exist_ok=True)
    cache_root.mkdir(exist_ok=True)
    dependency_root = dependency_cache_root / lock_digest
    if dependency_root.exists():
        installed = _target_distribution_versions(dependency_root)
        _require_exact_target_distribution_versions(
            installed,
            expected_versions,
            target_kind="cached",
        )
        sys.path.insert(0, str(dependency_root))
        try:
            _verify_imported_torch_version(requirements["torch"][1])
        finally:
            sys.path.remove(str(dependency_root))
        return dependency_root
    dependency_root.mkdir()
    command = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-deps",
        "--index-url",
        C1_PYPI_INDEX_URL,
        "--extra-index-url",
        C1_PYTORCH_INDEX_URL,
        "--extra-index-url",
        C1_NVIDIA_INDEX_URL,
        "--requirement",
        str(lock_path),
        "--target",
        str(dependency_root),
        "--cache-dir",
        str(cache_root),
    )
    pip_environment = {
        key: value
        for key, value in environment.items()
        if key not in {"CEG_WM_ROOT_KEY", "HF_TOKEN"}
        and (not key.startswith("PIP_") or key == "PIP_NO_INDEX")
    }
    pip_environment["PIP_CACHE_DIR"] = str(cache_root)
    pip_environment["PIP_CONFIG_FILE"] = os.devnull
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=pip_environment,
        )
    except OSError as exc:
        raise ExperimentBootstrapError(
            "dependency_install",
            "verified dependency installation could not start",
        ) from exc
    if completed.returncode != 0:
        raise ExperimentBootstrapError(
            "dependency_install",
            "verified dependency installation failed",
        )
    installed = _target_distribution_versions(dependency_root)
    _require_exact_target_distribution_versions(
        installed,
        expected_versions,
        target_kind="installed",
    )
    sys.path.insert(0, str(dependency_root))
    try:
        _verify_imported_torch_version(requirements["torch"][1])
    finally:
        sys.path.remove(str(dependency_root))
    return dependency_root


def _load_verified_threshold_fit_entrypoint(
    package_root: Path,
    manifest: dict[str, Any],
) -> tuple[Callable[..., Mapping[str, object]], str]:
    """Import only the already manifest-verified threshold-fit entrypoint."""

    entrypoint = (package_root / ENTRYPOINT_PATH).resolve()
    if not _strictly_within(entrypoint, package_root) or entrypoint.is_symlink():
        raise ExperimentBootstrapError(
            "entrypoint_import",
            "verified entrypoint path is unsafe",
        )
    entries = {
        entry["path"]: entry
        for entry in manifest["copied_files"]
        if type(entry) is dict and "path" in entry
    }
    package_namespaces = (
        "main",
        "runtime",
        "experiments",
        "infrastructure",
    )
    preloaded = sorted(
        name
        for name in sys.modules
        if any(
            name == namespace or name.startswith(f"{namespace}.")
            for namespace in package_namespaces
        )
    )
    if preloaded:
        raise ExperimentBootstrapError(
            "entrypoint_import",
            "package namespace was loaded before external verification",
        )
    entry = entries.get(ENTRYPOINT_PATH)
    if (
        type(entry) is not dict
        or entrypoint.stat().st_size != entry["size_bytes"]
        or _sha256_file(entrypoint) != entry["sha256"]
    ):
        raise ExperimentBootstrapError(
            "entrypoint_import",
            "verified entrypoint identity drifted before import",
        )
    module_name = "_ceg_wm_verified_threshold_fit_entrypoint"
    specification = importlib.util.spec_from_file_location(
        module_name,
        entrypoint,
    )
    if specification is None or specification.loader is None:
        raise ExperimentBootstrapError(
            "entrypoint_import",
            "verified entrypoint loader is unavailable",
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    original_path = tuple(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    sys.path.insert(0, str(package_root))
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ExperimentBootstrapError(
            "entrypoint_import",
            "verified entrypoint import failed",
        ) from exc
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_dont_write_bytecode
    callable_entrypoint = getattr(
        module,
        "execute_verified_threshold_fit_shard",
        None,
    )
    source_file = (
        inspect.getsourcefile(callable_entrypoint)
        if callable(callable_entrypoint)
        else None
    )
    if (
        type(getattr(module, "__file__", None)) is not str
        or Path(module.__file__).resolve() != entrypoint
        or source_file is None
        or Path(source_file).resolve() != entrypoint
        or _sha256_file(entrypoint) != entry["sha256"]
    ):
        raise ExperimentBootstrapError(
            "entrypoint_import",
            "verified entrypoint callable provenance drifted",
        )
    loaded_modules = [
        module,
        *(
            loaded
            for name, loaded in sys.modules.items()
            if any(
                name == namespace or name.startswith(f"{namespace}.")
                for namespace in package_namespaces
            )
        ),
    ]
    for loaded in loaded_modules:
        source_value = getattr(loaded, "__file__", None)
        if source_value is None:
            continue
        source = Path(source_value).resolve()
        if (
            not _strictly_within(source, package_root)
            or source.is_symlink()
            or not source.is_file()
        ):
            raise ExperimentBootstrapError(
                "entrypoint_import",
                "loaded package module escaped the verified extraction root",
            )
        relative = source.relative_to(package_root).as_posix()
        bound = entries.get(relative)
        if (
            type(bound) is not dict
            or source.stat().st_size != bound["size_bytes"]
            or _sha256_file(source) != bound["sha256"]
        ):
            raise ExperimentBootstrapError(
                "entrypoint_import",
                "loaded package module differs from the verified manifest",
            )
    return callable_entrypoint, entry["sha256"]


def _validate_threshold_fit_outcome(
    outcome_path: Path,
    *,
    result_root: Path,
    run_id: str,
    revision: str,
    shard_index: int,
) -> dict[str, Any]:
    try:
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "threshold-fit outcome is invalid",
        ) from exc
    required = {
        "artifact_kind",
        "committed_revision",
        "confirmation_unlock",
        "entrypoint_identity",
        "entrypoint_schema_version",
        "environment_digest",
        "execution_facts",
        "execution_scope",
        "failure_class",
        "held_out_evaluation_accessed",
        "planned_shard_count",
        "planned_units_per_shard",
        "record_files",
        "records_digest",
        "resource_identity_digest",
        "run_id",
        "runner_summary",
        "scientific_claims_supported",
        "shard_index",
        "tau_approval",
    }
    if (
        type(outcome) is not dict
        or set(outcome) != required
        or outcome["artifact_kind"]
        not in {
            "c1_threshold_fit_shard_result",
            "c1_threshold_fit_shard_diagnostic",
        }
        or outcome["execution_scope"] != "c1_hf_threshold_fit_only"
        or outcome["run_id"] != run_id
        or outcome["committed_revision"] != revision
        or outcome["shard_index"] != shard_index
        or outcome["planned_shard_count"] != 16
        or outcome["planned_units_per_shard"] != 256
        or outcome["scientific_claims_supported"] is not False
        or outcome["tau_approval"] is not False
        or outcome["confirmation_unlock"] is not False
        or outcome["held_out_evaluation_accessed"] is not False
        or DIGEST.fullmatch(outcome.get("environment_digest", "")) is None
        or DIGEST.fullmatch(outcome.get("resource_identity_digest", "")) is None
    ):
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "threshold-fit outcome identity drifted",
        )
    if (
        outcome["artifact_kind"] == "c1_threshold_fit_shard_result"
        and outcome["failure_class"] is not None
    ) or (
        outcome["artifact_kind"] == "c1_threshold_fit_shard_diagnostic"
        and outcome["failure_class"]
        not in {
            "resource_failure",
            "execution_failure",
            "excluded",
            "scientific_failure",
            "incomplete",
        }
    ):
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "threshold-fit outcome taxonomy drifted",
        )
    record_files = outcome["record_files"]
    if type(record_files) is not list or outcome["records_digest"] != _canonical_digest(
        record_files
    ):
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "threshold-fit record digest drifted",
        )
    expected_paths = {"threshold_fit_outcome.json"}
    for entry in record_files:
        if type(entry) is not dict or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ExperimentEntrypointError(
                "entrypoint_result",
                "threshold-fit record identity is invalid",
            )
        relative = _safe_relative(entry["path"], stage="entrypoint_result")
        candidate = (result_root / Path(*relative.parts)).resolve()
        if (
            not _strictly_within(candidate, result_root)
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != entry["size_bytes"]
            or _sha256_file(candidate) != entry["sha256"]
        ):
            raise ExperimentEntrypointError(
                "entrypoint_result",
                "threshold-fit record file drifted",
            )
        expected_paths.add(entry["path"])
    actual_paths = {
        path.relative_to(result_root).as_posix()
        for path in result_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ExperimentEntrypointError(
            "entrypoint_result",
            "threshold-fit result file set drifted",
        )
    return outcome


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
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    finished_identity = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    if artifact_kind == "bootstrap_failure":
        directory = persistent_root / "bootstrap_failures" / run_id
        filename = (
            f"ceg_wm_experiment_bootstrap_failure_{run_id}_"
            f"{finished_identity}.zip"
        )
    else:
        directory = persistent_root / "entrypoint_failures" / run_id
        filename = (
            f"ceg_wm_experiment_entrypoint_failure_{run_id}_"
            f"{finished_identity}.zip"
        )
    directory.mkdir(parents=True, exist_ok=True)
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
    delivery_manifest_path: str | Path,
    expected_archive_sha256: str,
    expected_delivery_manifest_sha256: str,
    expected_embedded_manifest_sha256: str,
    expected_bootstrap_identity: str,
    expected_bootstrap_schema_version: int,
    expected_bootstrap_sha256: str,
    expected_revision: str,
    ephemeral_root: str | Path,
    persistent_root: str | Path,
    run_id: str,
    shard_index: int,
    environment: dict[str, str] | None = None,
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
            or not DIGEST.fullmatch(expected_embedded_manifest_sha256)
            or not SAFE_ID.fullmatch(run_id)
            or type(shard_index) is not int
            or isinstance(shard_index, bool)
            or not 0 <= shard_index < 16
        ):
            raise ExperimentBootstrapError(
                "arguments",
                "revision, digest, or run identity is invalid",
            )
        archive_path = _absolute(package_zip, "package_zip")
        delivery_path = _absolute(
            delivery_manifest_path,
            "delivery_manifest_path",
        )
        ephemeral = _absolute(ephemeral_root, "ephemeral_root")
        if _overlap(ephemeral, persistent):
            raise ExperimentBootstrapError(
                "arguments",
                "ephemeral and persistent roots must be disjoint",
            )
        ephemeral.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(
                dir=ephemeral,
                prefix=f"experiment_bootstrap_{run_id}_",
            )
        )
        archive_snapshot = workspace / "execution_package.zip"
        archive_digest = _snapshot_archive(
            archive_path,
            archive_snapshot,
            expected_archive_sha256,
        )
        delivery_manifest = _verify_delivery_manifest(
            delivery_path,
            expected_delivery_manifest_sha256=(
                expected_delivery_manifest_sha256
            ),
            package_filename=archive_path.name,
            archive_sha256=archive_digest,
            embedded_manifest_sha256=expected_embedded_manifest_sha256,
            revision=expected_revision,
        )
        candidate_config_digest = delivery_manifest[
            "candidate_config_digest"
        ]
        execution_config_digest = delivery_manifest[
            "execution_config_digest"
        ]
        input_manifest_digest = delivery_manifest["input_manifest_digest"]
        package_root = workspace / "package"
        _safe_extract(archive_snapshot, package_root)
        embedded_manifest_path = (
            package_root / "experiment_execution_manifest.json"
        )
        if (
            _sha256_file(embedded_manifest_path)
            != expected_embedded_manifest_sha256
        ):
            raise ExperimentBootstrapError(
                "manifest",
                "embedded manifest SHA-256 differs from trust input",
            )
        manifest = _load_and_verify_manifest(
            package_root,
            expected_revision=expected_revision,
            expected_candidate_config_digest=candidate_config_digest,
            expected_execution_config_digest=execution_config_digest,
            expected_input_manifest_digest=input_manifest_digest,
        )
        result_root = workspace / "result"
        runtime_environment = dict(
            os.environ if environment is None else environment
        )
        runtime_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        runtime_workspace = workspace / "runtime_workspace"
        runtime_workspace.mkdir()
        model_cache_root = runtime_workspace / "ephemeral"
        model_persistent_root = persistent / "model_runtime"
        model_cache_root.mkdir()
        model_persistent_root.mkdir(parents=True, exist_ok=True)
        runtime_environment["CEG_WM_EPHEMERAL_ROOT"] = str(model_cache_root)
        runtime_environment["CEG_WM_PERSISTENT_ROOT"] = str(
            model_persistent_root
        )
        registered_detection_key = runtime_environment.get("CEG_WM_ROOT_KEY")
        if not registered_detection_key:
            raise ExperimentBootstrapError(
                "secrets",
                "CEG_WM_ROOT_KEY is unavailable",
            )
        dependency_root = _prepare_verified_dependencies(
            package_root=package_root,
            ephemeral_root=ephemeral,
            environment=runtime_environment,
        )
        if dependency_root is not None:
            sys.path.insert(0, str(dependency_root))
        callable_entrypoint, entrypoint_sha256 = (
            _load_verified_threshold_fit_entrypoint(package_root, manifest)
        )
        manifest_path = package_root / "experiment_execution_manifest.json"
        capability = _issue_threshold_fit_package_capability(
            package_root=package_root,
            committed_revision=expected_revision,
            archive_sha256=archive_digest,
            embedded_manifest_sha256=_sha256_file(manifest_path),
            copied_file_set_digest=_canonical_digest(
                manifest["copied_files"]
            ),
            entrypoint_sha256=entrypoint_sha256,
            execution_config_digest=execution_config_digest,
            input_manifest_digest=input_manifest_digest,
            candidate_config_digest=candidate_config_digest,
        )
        records_root = (
            persistent
            / "threshold_fit_records"
            / expected_revision
        )
        scoped_environment_keys = (
            "CEG_WM_EPHEMERAL_ROOT",
            "CEG_WM_PERSISTENT_ROOT",
            "HF_TOKEN",
        )
        prior_environment = {
            key: os.environ.get(key) for key in scoped_environment_keys
        }
        for key in scoped_environment_keys:
            if key in runtime_environment:
                os.environ[key] = runtime_environment[key]
            else:
                os.environ.pop(key, None)
        try:
            callable_entrypoint(
                package_revision_authority=capability,
                package_root=package_root,
                output_root=result_root,
                records_root=records_root,
                shard_index=shard_index,
                run_id=run_id,
                registered_detection_key=registered_detection_key,
            )
        except Exception as exc:
            raise ExperimentEntrypointError(
                "entrypoint_execution",
                "verified package entrypoint failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if dependency_root is not None:
                dependency_path = str(dependency_root)
                if dependency_path in sys.path:
                    sys.path.remove(dependency_path)
            for key, value in prior_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        summary = _validate_threshold_fit_outcome(
            result_root / "threshold_fit_outcome.json",
            result_root=result_root,
            run_id=run_id,
            revision=expected_revision,
            shard_index=shard_index,
        )
        finished_identity = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        destination = (
            persistent
            / "threshold_fit_artifacts"
            / expected_revision
            / run_id
            / f"shard_{shard_index:02d}"
            / (
                f"ceg_wm_{summary['artifact_kind']}_{run_id}_"
                f"{finished_identity}.zip"
            )
        )
        result_digest = _result_archive(result_root, destination)
        return 0, {
            "bootstrap_identity": BOOTSTRAP_IDENTITY,
            "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "artifact_kind": summary["artifact_kind"],
            "execution_scope": summary["execution_scope"],
            "evidence_scope": EVIDENCE_SCOPE,
            "run_id": run_id,
            "run_status": (
                "shard_complete"
                if summary["failure_class"] is None
                else "diagnostic"
            ),
            "failure_class": summary["failure_class"],
            "shard_index": shard_index,
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
    parser.add_argument("--delivery-manifest-path", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument(
        "--expected-delivery-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-embedded-manifest-sha256",
        required=True,
    )
    parser.add_argument("--expected-bootstrap-identity", required=True)
    parser.add_argument(
        "--expected-bootstrap-schema-version",
        required=True,
        type=int,
    )
    parser.add_argument("--expected-bootstrap-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--ephemeral-root", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    arguments = parser.parse_args(argv)
    exit_code, result = run_bootstrap(
        package_zip=arguments.package_zip,
        delivery_manifest_path=arguments.delivery_manifest_path,
        expected_archive_sha256=arguments.expected_archive_sha256,
        expected_delivery_manifest_sha256=(
            arguments.expected_delivery_manifest_sha256
        ),
        expected_embedded_manifest_sha256=(
            arguments.expected_embedded_manifest_sha256
        ),
        expected_bootstrap_identity=arguments.expected_bootstrap_identity,
        expected_bootstrap_schema_version=(
            arguments.expected_bootstrap_schema_version
        ),
        expected_bootstrap_sha256=arguments.expected_bootstrap_sha256,
        expected_revision=arguments.expected_revision,
        ephemeral_root=arguments.ephemeral_root,
        persistent_root=arguments.persistent_root,
        run_id=arguments.run_id,
        shard_index=arguments.shard_index,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
