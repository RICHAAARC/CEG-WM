"""Package-external trust bootstrap for semantic-texture Phase A."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
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
REQUIREMENTS_LOCK_SHA256 = (
    "07a4c1bbe6fc5e7e6b38334c5a9919a8565b810a9aae7820b61c24cee91270de"
)
PYPI_INDEX_URL = "https://pypi.org/simple"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
NVIDIA_INDEX_URL = "https://pypi.nvidia.com"
INSPYRENET_SOURCE_REVISION = "f0fa91701a98cfc8e955c554e84522f365ec6da3"
INSPYRENET_CHECKPOINT_REVISION = "d94c2baaa4d023ab018c6f97be6ef37548e3bd1f"
INSPYRENET_CHECKPOINT_SHA256 = (
    "0a6fe2a73ab0532d6d0b8d82849a9760a226df719e3063d09b4149ece6f80fcd"
)
INSPYRENET_CHECKPOINT_SIZE = 367520613
MINIMUM_CUDA_VRAM_BYTES = 23622320128
MINIMUM_FREE_EPHEMERAL_BYTES = 34359738368
TRANSPORT_RESULT_FILENAME = "semantic_texture_transport_result.json"
TRANSPORT_ARCHIVE_FILENAME = "semantic_texture_transport_result.zip"
TRANSPORT_RECEIPT_FILENAME = "semantic_texture_transport_receipt.json"
DELIVERY_COMPLETION_CHECKSUMS_FILENAME = "SHA256SUMS"
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


def _validate_production_entrypoint_arguments(
    arguments: tuple[str, ...],
    *,
    source_revision: str,
    package_identity: str,
) -> None:
    if (
        len(arguments) != 9
        or arguments[0] != "--execute"
        or arguments[1] != "--source-revision"
        or arguments[2] != source_revision
        or arguments[3] != "--run-id"
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", arguments[4])
        is None
        or arguments[5] != "--package-identity"
        or arguments[6] != package_identity
        or arguments[7] != "--output-root"
        or not Path(arguments[8]).is_absolute()
    ):
        raise SemanticTextureBootstrapError(
            "entrypoint_arguments",
            "production entrypoint authority drifted",
        )


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


def _transport_zip_info(path_text: str) -> zipfile.ZipInfo:
    information = zipfile.ZipInfo(path_text, date_time=(1980, 1, 1, 0, 0, 0))
    information.compress_type = zipfile.ZIP_STORED
    information.create_system = 3
    information.external_attr = 0o100644 << 16
    return information


def _persist_transport_failure(
    extract_root: Path,
    *,
    blocked_class: str,
    stage: str,
    source_revision: str | None,
    package_identity: str | None,
) -> dict[str, object]:
    """Persist package-transport authority without constructing method results."""

    delivery_root = extract_root.with_name(extract_root.name + ".transport")
    if delivery_root.exists():
        raise SemanticTextureBootstrapError(
            "transport_delivery",
            "transport delivery root must be absent",
        )
    delivery_root.mkdir(parents=True)
    result_path = delivery_root / TRANSPORT_RESULT_FILENAME
    archive_path = delivery_root / TRANSPORT_ARCHIVE_FILENAME
    receipt_path = delivery_root / TRANSPORT_RECEIPT_FILENAME
    delivery_completion_checksums_path = (
        delivery_root / DELIVERY_COMPLETION_CHECKSUMS_FILENAME
    )
    result = {
        "aggregate": None,
        "blocked_class": blocked_class,
        "candidate_promoted": False,
        "formal_tau_created": False,
        "package_identity": package_identity,
        "profile_id": "semantic_texture_operational_preflight_transport",
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "source_revision": source_revision,
        "stage": stage,
        "status": "blocked",
        "transport_kind": "package_external_trust_failure",
    }
    result_blob = _canonical_bytes(result)
    with result_path.open("xb") as handle:
        handle.write(result_blob)
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr(
            _transport_zip_info(TRANSPORT_RESULT_FILENAME),
            result_blob,
        )
    receipt = {
        "archive_filename": archive_path.name,
        "archive_sha256": _sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "blocked_class": blocked_class,
        "package_identity": package_identity,
        "profile_id": "semantic_texture_operational_preflight_transport",
        "result_filename": result_path.name,
        "result_sha256": sha256(result_blob).hexdigest(),
        "source_revision": source_revision,
        "stage": stage,
        "status": "blocked",
        "transport_kind": "package_external_trust_failure",
    }
    receipt_blob = _canonical_bytes(receipt)
    with receipt_path.open("xb") as handle:
        handle.write(receipt_blob)
    delivery_completion_checksums_blob = (
        f"{sha256(result_blob).hexdigest()}  {result_path.name}\n"
        f"{_sha256_file(archive_path)}  {archive_path.name}\n"
        f"{sha256(receipt_blob).hexdigest()}  {receipt_path.name}\n"
    ).encode("ascii")
    with delivery_completion_checksums_path.open("xb") as handle:
        handle.write(delivery_completion_checksums_blob)
    return {
        **result,
        "archive_filename": archive_path.name,
        "receipt_filename": receipt_path.name,
        "result_filename": result_path.name,
        "sha256sums_filename": delivery_completion_checksums_path.name,
    }


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _load_exact_dependency_lock(path: Path) -> dict[str, tuple[str, str]]:
    try:
        blob = path.read_bytes()
        lines = blob.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise SemanticTextureBootstrapError(
            "dependency_identity",
            "registered dependency lock is unreadable",
        ) from exc
    requirements: dict[str, tuple[str, str]] = {}
    for line in lines:
        match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+_-]*)",
            line,
        )
        if match is None:
            raise SemanticTextureBootstrapError(
                "dependency_identity",
                "registered dependency lock is not exact",
            )
        distribution, version = match.groups()
        normalized = _normalized_distribution_name(distribution)
        if normalized in requirements:
            raise SemanticTextureBootstrapError(
                "dependency_identity",
                "registered dependency lock contains duplicate identity",
            )
        requirements[normalized] = (distribution, version)
    if (
        sha256(blob).hexdigest() != REQUIREMENTS_LOCK_SHA256
        or len(requirements) != 62
        or requirements.get("torch") != ("torch", "2.11.0+cu128")
    ):
        raise SemanticTextureBootstrapError(
            "dependency_identity",
            "registered dependency lock identity drifted",
        )
    return requirements


def _target_distribution_versions(target: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in metadata.distributions(path=[str(target)]):
        distribution_name = distribution.metadata.get("Name")
        if type(distribution_name) is str:
            normalized = _normalized_distribution_name(distribution_name)
            if normalized in versions:
                raise SemanticTextureBootstrapError(
                    "dependency_identity",
                    "dependency target contains duplicate identity",
                )
            versions[normalized] = distribution.version
    return versions


def _prepare_exact_dependencies(
    destination: Path,
    environment: Mapping[str, str],
) -> Path | None:
    lock_path = (
        destination / "requirements_semantic_texture_operational_preflight.txt"
    )
    requirements = _load_exact_dependency_lock(lock_path)
    expected_versions = {
        normalized: version
        for normalized, (_distribution, version) in requirements.items()
    }
    reusable = True
    for distribution, version in requirements.values():
        try:
            observed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            reusable = False
            break
        if observed_version != version:
            reusable = False
            break
    if reusable:
        return None
    dependency_root = destination.parent / "semantic-texture-operational-dependencies"
    pip_cache_root = destination.parent / "semantic-texture-operational-pip-cache"
    dependency_root.mkdir()
    pip_cache_root.mkdir()
    pip_environment = {
        key: value
        for key, value in environment.items()
        if key not in {"CEG_WM_ROOT_KEY", "HF_TOKEN"}
        and (not key.startswith("PIP_") or key == "PIP_NO_INDEX")
    }
    pip_environment["PIP_CACHE_DIR"] = str(pip_cache_root)
    pip_environment["PIP_CONFIG_FILE"] = os.devnull
    try:
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--no-deps",
                "--index-url",
                PYPI_INDEX_URL,
                "--extra-index-url",
                PYTORCH_INDEX_URL,
                "--extra-index-url",
                NVIDIA_INDEX_URL,
                "--requirement",
                str(lock_path),
                "--target",
                str(dependency_root),
                "--cache-dir",
                str(pip_cache_root),
            ),
            check=False,
            capture_output=True,
            text=True,
            env=pip_environment,
        )
    except OSError as exc:
        raise SemanticTextureBootstrapError(
            "dependency_identity",
            "registered dependency installation could not start",
        ) from exc
    if completed.returncode != 0:
        raise SemanticTextureBootstrapError(
            "dependency_identity",
            "registered dependency installation failed",
        )
    if _target_distribution_versions(dependency_root) != expected_versions:
        raise SemanticTextureBootstrapError(
            "dependency_identity",
            "installed dependency closure drifted",
        )
    return dependency_root


def _prepare_production_environment(
    destination: Path,
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Hydrate exact public assets only for an authorized Colab execution."""

    if sys.version_info[:2] != (3, 12):
        raise SemanticTextureBootstrapError(
            "python_identity",
            "registered Python identity is unavailable",
        )
    if shutil.disk_usage(destination.parent).free < MINIMUM_FREE_EPHEMERAL_BYTES:
        raise SemanticTextureBootstrapError(
            "resource_identity",
            "registered ephemeral capacity is unavailable",
        )
    execution_environment = dict(environment)
    if not execution_environment.get("HF_TOKEN") or not execution_environment.get(
        "CEG_WM_ROOT_KEY"
    ):
        raise SemanticTextureBootstrapError(
            "secret_environment",
            "required environment authority is unavailable",
        )
    dependency_root = _prepare_exact_dependencies(
        destination,
        execution_environment,
    )
    if dependency_root is not None:
        execution_environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(dependency_root),
                execution_environment.get("PYTHONPATH", ""),
            )
        ).rstrip(os.pathsep)
        sys.path.insert(0, str(dependency_root))
    try:
        import torch
    except ImportError as exc:
        raise SemanticTextureBootstrapError(
            "dependency_identity",
            "registered torch dependency is unavailable",
        ) from exc
    if (
        str(torch.__version__) != "2.11.0+cu128"
        or torch.version.cuda != "12.8"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory
        < MINIMUM_CUDA_VRAM_BYTES
    ):
        raise SemanticTextureBootstrapError(
            "resource_identity",
            "registered CUDA resource identity is unavailable",
        )
    asset_root = destination.parent / "semantic-texture-operational-assets"
    source_root = asset_root / "transparent-background"
    checkpoint_root = asset_root / "inspyrenet-checkpoint"
    checkpoint_path = checkpoint_root / "ckpt_base.pth"
    cache_root = destination.parent / "semantic-texture-operational-cache"
    persistent_root = destination.parent / "semantic-texture-operational-persistent"
    try:
        asset_root.mkdir(parents=True, exist_ok=True)
        if not source_root.exists():
            subprocess.run(
                (
                    "git",
                    "clone",
                    "https://github.com/plemeri/transparent-background.git",
                    str(source_root),
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ("git", "checkout", "--detach", INSPYRENET_SOURCE_REVISION),
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
            )
        source_revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_remote = subprocess.run(
            ("git", "remote", "get-url", "origin"),
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise SemanticTextureBootstrapError(
            "environment_identity",
            "registered source hydration is unavailable",
        ) from exc
    except (MemoryError, OSError) as exc:
        raise SemanticTextureBootstrapError(
            "resource_identity",
            "registered source storage is unavailable",
        ) from exc
    if (
        source_revision != INSPYRENET_SOURCE_REVISION
        or source_remote
        != "https://github.com/plemeri/transparent-background.git"
        or source_status
    ):
        raise SemanticTextureBootstrapError(
            "source_identity",
            "registered InSPyReNet source identity drifted",
        )
    if not checkpoint_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id="plemeri/InSPyReNet",
                filename="ckpt_base.pth",
                revision=INSPYRENET_CHECKPOINT_REVISION,
                token=execution_environment["HF_TOKEN"],
                local_dir=checkpoint_root,
            )
        except ImportError as exc:
            raise SemanticTextureBootstrapError(
                "dependency_identity",
                "registered checkpoint dependency is unavailable",
            ) from exc
        except (MemoryError, OSError) as exc:
            raise SemanticTextureBootstrapError(
                "resource_identity",
                "registered checkpoint storage is unavailable",
            ) from exc
        except Exception as exc:
            raise SemanticTextureBootstrapError(
                "environment_identity",
                "registered checkpoint hydration is unavailable",
            ) from exc
        if Path(downloaded).resolve() != checkpoint_path.resolve():
            raise SemanticTextureBootstrapError(
                "checkpoint_identity",
                "registered checkpoint filename drifted",
            )
    if (
        checkpoint_path.stat().st_size != INSPYRENET_CHECKPOINT_SIZE
        or _sha256_file(checkpoint_path) != INSPYRENET_CHECKPOINT_SHA256
    ):
        raise SemanticTextureBootstrapError(
            "checkpoint_identity",
            "registered checkpoint content drifted",
        )
    execution_environment.update(
        {
            "CEG_WM_CACHE_ROOT": str(cache_root),
            "CEG_WM_INSPYRENET_CHECKPOINT_PATH": str(checkpoint_path),
            "CEG_WM_INSPYRENET_SOURCE_ROOT": str(source_root),
            "CEG_WM_PERSISTENT_ROOT": str(persistent_root),
        }
    )
    return execution_environment


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
        execution_environment = dict(
            os.environ if environment is None else environment
        )
        production_execution = "--execute" in bounded_args
        if production_execution:
            _validate_production_entrypoint_arguments(
                bounded_args,
                source_revision=source_revision,
                package_identity=package_identity,
            )
            execution_environment = _prepare_production_environment(
                destination,
                execution_environment,
            )
        command = (
            sys.executable,
            str(destination / ENTRYPOINT_PATH),
            *bounded_args,
        )
        completed = command_runner(
            command,
            cwd=destination,
            env=execution_environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if type(completed.returncode) is not int:
            raise SemanticTextureBootstrapError(
                stage,
                "entrypoint return code is invalid",
            )
        if completed.returncode == 2 and production_execution:
            return 2, {
                "aggregate": None,
                "candidate_promoted": False,
                "entrypoint_exit_code": 2,
                "formal_tau_created": False,
                "package_identity": package_identity,
                "profile_id": "semantic_texture_operational_preflight_transport",
                "science_started": False,
                "scientific_claims_supported": False,
                "scientific_unit_count": 0,
                "source_revision": source_revision,
                "stage": "entrypoint",
                "status": "trusted_operational_delivery_completed",
                "transport_kind": "package_external_trust_passed",
            }
        if completed.returncode != 0:
            failure = _persist_transport_failure(
                destination,
                blocked_class="environment_blocked",
                stage="entrypoint",
                source_revision=source_revision,
                package_identity=package_identity,
            )
            return 3, failure
        result = {
            "aggregate": None,
            "blocked_class": None,
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
            "status": "passed",
        }
        _write_result_create_only(result_path, result)
        return completed.returncode, {
            **result,
            "bootstrap_result_name": result_path.name,
        }
    except SemanticTextureBootstrapError as exc:
        blocked_class = (
            "resource_blocked"
            if exc.stage == "resource_identity"
            else "environment_blocked"
            if exc.stage
            in {
                "dependency_identity",
                "entrypoint",
                "environment_identity",
                "python_identity",
                "secret_environment",
            }
            else "integrity_blocked"
        )
        failure = _persist_transport_failure(
            destination,
            blocked_class=blocked_class,
            stage=exc.stage,
            source_revision=source_revision,
            package_identity=package_identity,
        )
        return 3, failure
    except (MemoryError, OSError):
        failure = _persist_transport_failure(
            destination,
            blocked_class="resource_blocked",
            stage=stage,
            source_revision=source_revision,
            package_identity=package_identity,
        )
        return 3, failure
    except Exception:
        failure = _persist_transport_failure(
            destination,
            blocked_class="implementation_blocked",
            stage=stage,
            source_revision=source_revision,
            package_identity=package_identity,
        )
        return 3, failure


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
