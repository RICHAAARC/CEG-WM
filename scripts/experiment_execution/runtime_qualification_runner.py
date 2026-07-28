"""Revision-bound SD3.5 runtime qualification runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


# The caller must also set PYTHONDONTWRITEBYTECODE=1 before interpreter startup.
# This assignment prevents later lazy project imports from mutating the package.
sys.dont_write_bytecode = True

SUPPORTED_PROFILES = ("smoke", "qualification", "replay")
REVISION = re.compile(r"^[0-9a-f]{40}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REGISTERED_QK_LAYERS = (
    "transformer_blocks.0.attn",
    "transformer_blocks.23.attn",
)
REGISTERED_DEPENDENCY_LOCK = (
    ("python", ">=3.12"),
    ("diffusers", "0.38.0"),
    ("torch", "2.11.0"),
    ("transformers", "5.12.1"),
    ("accelerate", "1.14.0"),
    ("numpy", "2.0.2"),
    ("Pillow", "11.3.0"),
    ("safetensors", "0.8.0"),
    ("huggingface-hub", "1.20.1"),
)
SENSITIVE_PARTS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "id_rsa",
    "id_ed25519",
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
RESULT_FILES = {
    "environment_summary.json",
    "failures.jsonl",
    "run_summary.json",
    "runtime_checks.jsonl",
}
PROMPT_IDENTITY = "runtime_qualification_prompt_v1"
ENVIRONMENT_FIELDS = {
    "accelerate",
    "cuda_available",
    "cuda_runtime",
    "diffusers",
    "gpu_name",
    "huggingface_hub",
    "key_controls",
    "numpy",
    "pillow",
    "profile",
    "prompt_identity",
    "prompt_sha256",
    "python",
    "record_digests",
    "result_schema_version",
    "run_id",
    "runtime_candidate_revision",
    "safetensors",
    "seed",
    "torch",
    "transformers",
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
REQUIRED_RECORD_FIELDS = {
    "budget_status",
    "callback_index",
    "callback_status",
    "clean_image_sha256",
    "content_relative_l2_limit",
    "content_relative_l2_nominal",
    "cuda_available",
    "cuda_runtime",
    "detection_latent_sha256",
    "gpu_name",
    "integrity_status",
    "key_control",
    "key_public_digest",
    "materialization_replay_identity",
    "materialization_attempt_count",
    "materialization_scale",
    "model_id",
    "model_revision",
    "paired_base_latent_digest",
    "prompt_identity",
    "prompt_sha256",
    "public_noise_domain_digest",
    "public_noise_values_float32_be_sha256",
    "qk_actual_dtype",
    "qk_layer_names",
    "qk_layer_value_digests",
    "qk_operator_identities",
    "qk_status",
    "realized_relative_l2",
    "realized_total_l2",
    "run_id",
    "runtime_backend_name",
    "runtime_candidate_revision",
    "runtime_config_digest",
    "seed",
    "selected_device",
    "vae_scaling_factor_actual",
    "vae_shift_factor_actual",
    "vae_status",
    "watermarked_image_sha256",
    "budget_utilization",
}


class QualificationRunnerError(RuntimeError):
    """The qualification request or delivery surface is invalid."""


class DeterminismQualificationError(QualificationRunnerError):
    """Independent runtime repetitions or replay did not reproduce."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _safe_relative(path_text: str) -> PurePosixPath:
    if (
        "\x00" in path_text
        or "\\" in path_text
        or re.match(r"^[A-Za-z]:", path_text)
    ):
        raise QualificationRunnerError(f"unsafe package path: {path_text}")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise QualificationRunnerError(f"unsafe package path: {path_text}")
    if any(
        marker in part.lower()
        for part in path.parts
        for marker in SENSITIVE_PARTS
    ):
        raise QualificationRunnerError(f"sensitive package path: {path_text}")
    if any(part in PACKAGE_EXCLUDED_PARTS for part in path.parts):
        raise QualificationRunnerError(f"excluded package path: {path_text}")
    if path_text not in PACKAGE_REQUIRED_FILES and not path_text.startswith(
        PACKAGE_INCLUDE_ROOTS
    ):
        raise QualificationRunnerError(f"unallowlisted package path: {path_text}")
    return path


def _explicit_absolute_path(value: str | Path, field_name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise QualificationRunnerError(
            f"{field_name} must be an explicit absolute path"
        )
    return path.resolve()


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _strictly_within(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _validate_storage_boundaries(
    *,
    profile: str,
    result_zip: str | Path,
    ephemeral_root: str | Path,
    persistent_root: str | Path,
    replay_source: str | Path | None,
) -> tuple[Path, Path, Path, Path | None]:
    output_path = _explicit_absolute_path(result_zip, "result_zip")
    ephemeral = _explicit_absolute_path(ephemeral_root, "ephemeral_root")
    persistent = _explicit_absolute_path(persistent_root, "persistent_root")
    if _paths_overlap(ephemeral, persistent):
        raise QualificationRunnerError(
            "ephemeral_root and persistent_root must be bidirectionally disjoint"
        )
    if not _strictly_within(output_path, ephemeral):
        raise QualificationRunnerError(
            "result_zip must be strictly within ephemeral_root"
        )
    if profile == "replay":
        if replay_source is None:
            raise QualificationRunnerError("replay source is required")
        source = _explicit_absolute_path(replay_source, "replay_source")
        if not _strictly_within(source, persistent):
            raise QualificationRunnerError(
                "replay_source must be strictly within persistent_root"
            )
    else:
        if replay_source is not None:
            raise QualificationRunnerError(
                "replay_source is only allowed for replay profile"
            )
        source = None
    return output_path, ephemeral, persistent, source


def verify_execution_package(
    package_root: str | Path,
    runtime_candidate_revision: str,
) -> dict[str, Any]:
    """Verify the complete unpacked package before importing project code."""

    root = Path(package_root).resolve()
    manifest_path = root / "runtime_execution_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationRunnerError(
            "runtime execution manifest is unavailable"
        ) from exc
    if (
        set(manifest)
        != {
            "copied_files",
            "excluded_parts",
            "package_ready",
            "package_schema_version",
            "profile_name",
            "runtime_candidate_revision",
        }
        or manifest.get("package_schema_version") != 1
        or manifest.get("profile_name") != "experiment_execution_package"
        or manifest.get("package_ready") is not True
        or manifest.get("runtime_candidate_revision")
        != runtime_candidate_revision
        or not REVISION.fullmatch(runtime_candidate_revision)
        or manifest.get("excluded_parts") != sorted(PACKAGE_EXCLUDED_PARTS)
    ):
        raise QualificationRunnerError("package manifest identity drifted")
    entries = manifest.get("copied_files")
    if not isinstance(entries, list) or not entries:
        raise QualificationRunnerError("package manifest file list is invalid")
    expected: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise QualificationRunnerError("package manifest entry is invalid")
        path_text = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(path_text, str)
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise QualificationRunnerError("package manifest entry is invalid")
        _safe_relative(path_text)
        if path_text in expected:
            raise QualificationRunnerError("package manifest path is duplicated")
        expected[path_text] = (size, digest)
    actual: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise QualificationRunnerError("package symlinks are forbidden")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative == "runtime_execution_manifest.json":
            continue
        _safe_relative(relative)
        actual[relative] = candidate
    if set(actual) != set(expected):
        raise QualificationRunnerError(
            "package file set differs from the signed manifest"
        )
    if not PACKAGE_REQUIRED_FILES <= set(expected):
        raise QualificationRunnerError(
            "package manifest omits required execution files"
        )
    if not any(path.startswith("main/") for path in expected) or not any(
        path.startswith("runtime/") for path in expected
    ):
        raise QualificationRunnerError(
            "package manifest omits method or runtime implementation"
        )
    for path_text, candidate in actual.items():
        size, digest = expected[path_text]
        if candidate.stat().st_size != size or _sha256(candidate) != digest:
            raise QualificationRunnerError(
                f"package file identity drifted: {path_text}"
            )
    return manifest


def _dependency_versions(
    lock: list[dict[str, str]],
    supplied: dict[str, str] | None,
) -> tuple[dict[str, str], ...]:
    evidence: list[dict[str, str]] = []
    for item in lock:
        name = item["package_name"]
        expected = item["version_specifier"]
        if supplied is not None:
            actual = supplied.get(name)
        elif name == "python":
            actual = platform.python_version()
        else:
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = None
        if actual is None:
            raise QualificationRunnerError(
                f"required dependency is unavailable: {name}"
            )
        accepted = (
            tuple(map(int, actual.split(".")[:2])) >= (3, 12)
            if name == "python" and expected == ">=3.12"
            else actual == expected
        )
        if not accepted:
            raise QualificationRunnerError(
                f"dependency lock drifted: {name} expected {expected}, actual {actual}"
            )
        evidence.append(
            {
                "package_name": name,
                "expected_version": expected,
                "actual_version": actual,
            }
        )
    return tuple(evidence)


def verify_dependency_lock(
    package_root: str | Path,
    supplied_versions: dict[str, str] | None = None,
) -> tuple[dict[str, str], ...]:
    path = Path(package_root) / "configs/runtime/runtime_sd35_flowmatch.json"
    try:
        configuration = json.loads(path.read_text(encoding="utf-8"))
        lock = configuration["dependency_lock"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise QualificationRunnerError("runtime dependency lock is invalid") from exc
    if (
        not isinstance(lock, list)
        or not lock
        or any(
            not isinstance(item, dict)
            or set(item) != {"package_name", "version_specifier"}
            or not isinstance(item["package_name"], str)
            or not isinstance(item["version_specifier"], str)
            for item in lock
        )
    ):
        raise QualificationRunnerError("runtime dependency lock is invalid")
    actual_lock = tuple(
        (item["package_name"], item["version_specifier"]) for item in lock
    )
    if actual_lock != REGISTERED_DEPENDENCY_LOCK:
        raise QualificationRunnerError(
            "runtime dependency lock differs from the frozen complete lock"
        )
    requirements_path = (
        Path(package_root) / "requirements_runtime_qualification.txt"
    )
    try:
        requirement_lines = tuple(
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError as exc:
        raise QualificationRunnerError(
            "runtime requirements lock is unavailable"
        ) from exc
    expected_requirements = tuple(
        f"{item['package_name']}=={item['version_specifier']}"
        for item in lock
        if item["package_name"] != "python"
    )
    if requirement_lines != expected_requirements:
        raise QualificationRunnerError(
            "runtime requirements do not exactly match dependency lock"
        )
    return _dependency_versions(lock, supplied_versions)


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _classify_failure(exc: BaseException) -> str:
    chain = tuple(_exception_chain(exc))
    names = {type(item).__name__ for item in chain}
    text = " ".join(str(item).lower() for item in chain)
    if any(
        marker in text
        for marker in (
            "out of memory",
            "cuda is unavailable",
            "no cuda device",
            "disk quota",
            "no space left",
            "resource exhausted",
        )
    ):
        return "resource_failure"
    if names & {"KeyboardInterrupt", "SystemExit", "GeneratorExit"}:
        return "incomplete"
    if "RuntimeQkObservationError" in names:
        return "qk_failure"
    if "ContentEmbedderError" in names and (
        "budget" in text or "hard budget" in text
    ):
        return "budget_failure"
    if names & {"RuntimeContentExecutionError", "ContentEmbedderError"}:
        return "integrity_failure"
    if names & {"Sd35BackendError", "RuntimeAdapterError"}:
        return "runtime_failure"
    if "DeterminismQualificationError" in names:
        return "determinism_failure"
    if isinstance(exc, QualificationRunnerError):
        return "incomplete"
    return "runtime_failure"


def _record_digest(record: dict[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    )


def _replay_comparison_digest(record: dict[str, Any]) -> str:
    return _record_digest(
        {key: value for key, value in record.items() if key != "run_id"}
    )


def _validate_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _validate_record(
    record: object,
    *,
    expected_key_control: str,
    expected_seed: int,
    expected_prompt_identity: str,
    expected_prompt_sha256: str,
    run_id: str,
    revision: str,
) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != REQUIRED_RECORD_FIELDS:
        raise QualificationRunnerError("qualification record is incomplete")
    if (
        record["key_control"] != expected_key_control
        or record["seed"] != expected_seed
        or record["prompt_identity"] != expected_prompt_identity
        or record["prompt_sha256"] != expected_prompt_sha256
        or record["run_id"] != run_id
        or record["runtime_candidate_revision"] != revision
        or record["integrity_status"] != "passed"
        or record["budget_status"] != "accepted"
        or record["callback_status"] != "passed"
        or record["vae_status"] != "passed"
        or record["qk_status"] != "passed"
        or record["callback_index"] != 18
        or record["qk_actual_dtype"] != "float16"
        or record["selected_device"] != "cuda:0"
        or record["cuda_available"] is not True
        or not isinstance(record["cuda_runtime"], str)
        or not record["cuda_runtime"]
        or not isinstance(record["gpu_name"], str)
        or not record["gpu_name"]
        or not isinstance(record["seed"], int)
        or isinstance(record["seed"], bool)
        or not isinstance(record["model_id"], str)
        or not record["model_id"]
        or not isinstance(record["model_revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", record["model_revision"])
        or not isinstance(record["runtime_backend_name"], str)
        or not record["runtime_backend_name"]
        or not isinstance(record["prompt_identity"], str)
        or not record["prompt_identity"]
        or not isinstance(record["vae_scaling_factor_actual"], (int, float))
        or isinstance(record["vae_scaling_factor_actual"], bool)
        or not isinstance(record["vae_shift_factor_actual"], (int, float))
        or isinstance(record["vae_shift_factor_actual"], bool)
        or not math.isfinite(float(record["vae_scaling_factor_actual"]))
        or float(record["vae_scaling_factor_actual"]) <= 0.0
        or not math.isfinite(float(record["vae_shift_factor_actual"]))
        or any(
            isinstance(record[field], bool)
            or not isinstance(record[field], (int, float))
            or not math.isfinite(float(record[field]))
            for field in (
                "budget_utilization",
                "content_relative_l2_limit",
                "content_relative_l2_nominal",
                "materialization_scale",
                "realized_relative_l2",
                "realized_total_l2",
            )
        )
        or float(record["content_relative_l2_nominal"]) <= 0.0
        or record["content_relative_l2_nominal"]
        != record["content_relative_l2_limit"]
        or not 0.0 < float(record["materialization_scale"]) <= 1.0
        or float(record["realized_total_l2"]) <= 0.0
        or float(record["realized_relative_l2"]) <= 0.0
        or not 0.0 < float(record["budget_utilization"]) <= 1.0
        or type(record["materialization_attempt_count"]) is not int
        or record["materialization_attempt_count"] <= 0
        or not isinstance(record["qk_layer_names"], list)
        or tuple(record["qk_layer_names"]) != REGISTERED_QK_LAYERS
        or not isinstance(record["qk_operator_identities"], list)
        or len(record["qk_operator_identities"]) != len(record["qk_layer_names"])
        or any(
            not isinstance(identity, str) or not identity
            for identity in record["qk_operator_identities"]
        )
        or not isinstance(record["qk_layer_value_digests"], list)
        or len(record["qk_layer_value_digests"]) != len(record["qk_layer_names"])
        or any(
            not isinstance(item, dict)
            or set(item)
            != {"attention_key_sha256", "layer_name", "query_sha256"}
            or item["layer_name"] != layer_name
            or not _validate_digest(item["query_sha256"])
            or not _validate_digest(item["attention_key_sha256"])
            for layer_name, item in zip(
                record["qk_layer_names"],
                record["qk_layer_value_digests"],
                strict=True,
            )
        )
        or any(
            not _validate_digest(record[field])
            for field in (
                "clean_image_sha256",
                "detection_latent_sha256",
                "key_public_digest",
                "materialization_replay_identity",
                "paired_base_latent_digest",
                "prompt_sha256",
                "public_noise_domain_digest",
                "public_noise_values_float32_be_sha256",
                "runtime_config_digest",
                "watermarked_image_sha256",
            )
        )
    ):
        raise QualificationRunnerError(
            "qualification record failed required success semantics"
        )
    return record


def _load_replay_source(
    path: str | Path,
    revision: str,
    expected_prompt_sha256: str,
    expected_seed: int,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                len(names) != len(RESULT_FILES)
                or set(names) != RESULT_FILES
                or any(
                    ((member.external_attr >> 16) & 0o170000) == 0o120000
                    for member in members
                )
            ):
                raise QualificationRunnerError(
                    "replay source result file set drifted"
                )
            summary = json.loads(archive.read("run_summary.json"))
            environment = json.loads(
                archive.read("environment_summary.json")
            )
            record_lines = archive.read("runtime_checks.jsonl").decode("utf-8")
            failure_bytes = archive.read("failures.jsonl")
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise QualificationRunnerError("replay source artifact is invalid") from exc
    except UnicodeDecodeError as exc:
        raise QualificationRunnerError("replay source records are invalid") from exc
    if not isinstance(summary, dict):
        raise QualificationRunnerError("replay source identity drifted")
    try:
        source_records = tuple(
            json.loads(line) for line in record_lines.splitlines()
        )
    except json.JSONDecodeError as exc:
        raise QualificationRunnerError("replay source records are invalid") from exc
    digests = summary.get("record_digests")
    source_run_id = summary.get("run_id")
    expected_prompt_identity = PROMPT_IDENTITY
    dependency_evidence = summary.get("dependency_lock_evidence")
    if (
        set(summary) != SUMMARY_FIELDS
        or summary.get("profile") != "qualification"
        or summary.get("result_schema_version") != 2
        or summary.get("run_status") != "passed"
        or summary.get("runtime_candidate_revision") != revision
        or summary.get("replay_source_revision") is not None
        or summary.get("replay_source_run_id") is not None
        or summary.get("replay_source_record_digests") != []
        or summary.get("prompt_sha256") != expected_prompt_sha256
        or summary.get("prompt_identity") != expected_prompt_identity
        or summary.get("seed") != expected_seed
        or summary.get("result_zip_filename") != Path(path).name
        or summary.get("repetition_count") != 3
        or summary.get("failure_count") != 0
        or summary.get("failure_classes") != []
        or not isinstance(summary.get("started_at_utc"), str)
        or not summary.get("started_at_utc")
        or not isinstance(summary.get("finished_at_utc"), str)
        or not summary.get("finished_at_utc")
        or summary.get("callback_status") != "passed"
        or summary.get("actual_dtype_status") != "passed"
        or summary.get("vae_status") != "passed"
        or summary.get("qk_status") != "passed"
        or summary.get("determinism_status") != "passed"
        or summary.get("package_status") != "verified"
        or summary.get("dependency_status") != "verified"
        or summary.get("key_controls")
        != ["registered", "registered", "negative_identity"]
        or summary.get("checks") != list(source_records)
        or not isinstance(source_run_id, str)
        or not RUN_ID.fullmatch(source_run_id)
        or not isinstance(digests, list)
        or len(digests) != 3
        or any(not re.fullmatch(r"[0-9a-f]{64}", item or "") for item in digests)
        or not isinstance(dependency_evidence, list)
        or len(dependency_evidence) != len(REGISTERED_DEPENDENCY_LOCK)
        or any(
            not isinstance(item, dict)
            or set(item)
            != {"actual_version", "expected_version", "package_name"}
            or (
                item["package_name"],
                item["expected_version"],
            )
            != expected
            or not isinstance(item["actual_version"], str)
            or not item["actual_version"]
            for item, expected in zip(
                dependency_evidence,
                REGISTERED_DEPENDENCY_LOCK,
                strict=True,
            )
        )
    ):
        raise QualificationRunnerError("replay source identity drifted")
    controls = ("registered", "registered", "negative_identity")
    if len(source_records) != len(controls):
        raise QualificationRunnerError("replay source record count drifted")
    validated = tuple(
        _validate_record(
            record,
            expected_key_control=control,
            expected_seed=expected_seed,
            expected_prompt_identity=expected_prompt_identity,
            expected_prompt_sha256=expected_prompt_sha256,
            run_id=source_run_id,
            revision=revision,
        )
        for record, control in zip(source_records, controls, strict=True)
    )
    actual_digests = tuple(_record_digest(record) for record in validated)
    if tuple(digests) != actual_digests:
        raise QualificationRunnerError("replay source record digests drifted")
    dependency_versions = {
        item["package_name"]: item["actual_version"]
        for item in dependency_evidence
    }
    expected_environment = {
        "result_schema_version": 2,
        "profile": "qualification",
        "run_id": source_run_id,
        "runtime_candidate_revision": revision,
        "seed": expected_seed,
        "prompt_identity": expected_prompt_identity,
        "prompt_sha256": expected_prompt_sha256,
        "record_digests": list(actual_digests),
        "key_controls": list(controls),
        "python": dependency_versions.get("python"),
        "torch": dependency_versions.get("torch"),
        "diffusers": dependency_versions.get("diffusers"),
        "transformers": dependency_versions.get("transformers"),
        "accelerate": dependency_versions.get("accelerate"),
        "numpy": dependency_versions.get("numpy"),
        "pillow": dependency_versions.get("Pillow"),
        "safetensors": dependency_versions.get("safetensors"),
        "huggingface_hub": dependency_versions.get("huggingface-hub"),
        "cuda_available": validated[0]["cuda_available"],
        "cuda_runtime": validated[0]["cuda_runtime"],
        "gpu_name": validated[0]["gpu_name"],
    }
    if (
        failure_bytes != b""
        or not isinstance(environment, dict)
        or set(environment) != ENVIRONMENT_FIELDS
        or environment != expected_environment
        or any(
            (
                record["cuda_available"],
                record["cuda_runtime"],
                record["gpu_name"],
            )
            != (
                environment["cuda_available"],
                environment["cuda_runtime"],
                environment["gpu_name"],
            )
            for record in validated
        )
    ):
        raise QualificationRunnerError(
            "replay source environment or failures drifted"
        )
    comparison_digests = tuple(
        _replay_comparison_digest(record) for record in validated
    )
    return source_run_id, actual_digests, comparison_digests


def _tensor_sha256(torch: Any, value: Any) -> str:
    contiguous = value.detach().contiguous().to(device="cpu")
    return _sha256_bytes(
        bytes(contiguous.view(torch.uint8).reshape(-1).tolist())
    )


def _execute_once(
    *,
    backend_factory: Any,
    cache_root: Path,
    persistent_root: Path,
    hf_token: str | None,
    root_key: str,
    prompt: str,
    seed: int,
    key_control: str,
    run_id: str,
    runtime_candidate_revision: str,
) -> dict[str, Any]:
    import torch
    from main import content_embedder, hf_carrier
    from runtime import Sd35PipelineBackend, create_runtime_adapter

    factory = backend_factory or Sd35PipelineBackend
    backend = factory(
        cache_root=cache_root,
        persistent_root=persistent_root,
        hf_token=hf_token,
        prompt=prompt,
    )
    adapter = create_runtime_adapter(backend=backend)
    try:
        with torch.inference_mode():
            session = adapter.initialize(requested_device="cuda")
            configuration = adapter.configuration
            generator = torch.Generator(device="cpu").manual_seed(seed)
            latent = torch.randn(
                (
                    1,
                    16,
                    configuration.image_height // 8,
                    configuration.image_width // 8,
                ),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=session.selected_device, dtype=torch.float16)
            carrier = hf_carrier(root_key, tuple(latent.shape))

            def operation(baseline_values: tuple[float, ...]):
                return content_embedder(baseline_values, carrier)

            content = adapter.execute_content_write_and_vae(latent, operation)
            qk = adapter.observe_detection_qk(content.watermarked_image)
        materialization = content.content_materialization_result
        return {
            "run_id": run_id,
            "runtime_candidate_revision": runtime_candidate_revision,
            "runtime_config_digest": session.runtime_config_digest,
            "runtime_backend_name": session.runtime_backend_name,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "key_control": key_control,
            "key_public_digest": carrier.root_key_public_digest,
            "selected_device": session.selected_device,
            "model_id": session.model_id,
            "model_revision": session.model_revision,
            "seed": seed,
            "prompt_identity": PROMPT_IDENTITY,
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "callback_index": content.content_materialization.callback_index,
            "callback_status": "passed",
            "content_relative_l2_nominal": (
                materialization.content_relative_l2_nominal
            ),
            "content_relative_l2_limit": materialization.content_relative_l2_limit,
            "realized_total_l2": materialization.realized_total_l2,
            "realized_relative_l2": materialization.realized_relative_l2,
            "budget_utilization": materialization.budget_utilization,
            "materialization_scale": materialization.materialization_scale,
            "materialization_attempt_count": materialization.attempt_count,
            "integrity_status": materialization.integrity_status,
            "budget_status": materialization.budget_status,
            "materialization_replay_identity": (
                materialization.observation.materialization_replay_identity
            ),
            "paired_base_latent_digest": content.paired_base_latent_digest,
            "vae_scaling_factor_actual": content.vae_scaling_factor_actual,
            "vae_shift_factor_actual": content.vae_shift_factor_actual,
            "vae_status": "passed",
            "clean_image_sha256": _tensor_sha256(torch, content.clean_image),
            "watermarked_image_sha256": _tensor_sha256(
                torch, content.watermarked_image
            ),
            "detection_latent_sha256": _tensor_sha256(
                torch, content.watermarked_detection_latent
            ),
            "qk_actual_dtype": qk.qk_actual_dtype,
            "qk_status": "passed",
            "qk_layer_names": [
                observation.layer_name
                for observation in qk.qk_layer_observations
            ],
            "qk_operator_identities": [
                observation.operator_identity
                for observation in qk.qk_layer_observations
            ],
            "qk_layer_value_digests": [
                {
                    "layer_name": observation.layer_name,
                    "query_sha256": _tensor_sha256(torch, observation.query),
                    "attention_key_sha256": _tensor_sha256(
                        torch, observation.attention_key
                    ),
                }
                for observation in qk.qk_layer_observations
            ],
            "public_noise_domain_digest": qk.public_noise_domain_digest,
            "public_noise_values_float32_be_sha256": (
                qk.public_noise_values_float32_be_sha256
            ),
        }
    finally:
        adapter.close()


def run_runtime_qualification(
    *,
    profile: str,
    run_id: str,
    package_root: str | Path,
    runtime_candidate_revision: str,
    result_zip: str | Path,
    ephemeral_root: str | Path,
    persistent_root: str | Path,
    hf_token: str | None,
    root_key: str,
    prompt: str,
    seed: int = 20260728,
    replay_source: str | Path | None = None,
    backend_factory: Any = None,
    supplied_dependency_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the package, execute the selected profile, and always zip status."""

    output_path, root, persistent, replay_path = _validate_storage_boundaries(
        profile=profile,
        result_zip=result_zip,
        ephemeral_root=ephemeral_root,
        persistent_root=persistent_root,
        replay_source=replay_source,
    )
    root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise QualificationRunnerError("result zip target already exists")
    started = datetime.now(timezone.utc).isoformat()
    prompt_identity = PROMPT_IDENTITY
    prompt_sha256 = _sha256_bytes(prompt.encode("utf-8"))
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    dependency_evidence: tuple[dict[str, str], ...] = ()
    package_verified = False
    dependency_verified = False
    replay_source_run_id: str | None = None
    replay_digests: tuple[str, ...] = ()
    replay_comparison_digests: tuple[str, ...] = ()
    status = "failed"
    try:
        if profile not in SUPPORTED_PROFILES:
            raise QualificationRunnerError("unsupported qualification profile")
        if not RUN_ID.fullmatch(run_id):
            raise QualificationRunnerError("run_id is invalid")
        if not root_key:
            raise QualificationRunnerError("qualification root key is required")
        verify_execution_package(package_root, runtime_candidate_revision)
        package_verified = True
        dependency_evidence = verify_dependency_lock(
            package_root,
            supplied_dependency_versions,
        )
        dependency_verified = True
        if profile == "replay":
            assert replay_path is not None
            (
                replay_source_run_id,
                replay_digests,
                replay_comparison_digests,
            ) = _load_replay_source(
                replay_path,
                runtime_candidate_revision,
                prompt_sha256,
                seed,
            )
        repetitions = 1 if profile == "smoke" else 2
        for _index in range(repetitions):
            records.append(
                _validate_record(
                    _execute_once(
                        backend_factory=backend_factory,
                        cache_root=root / "cache",
                        persistent_root=persistent,
                        hf_token=hf_token,
                        root_key=root_key,
                        prompt=prompt,
                        seed=seed,
                        key_control="registered",
                        run_id=run_id,
                        runtime_candidate_revision=runtime_candidate_revision,
                    ),
                    expected_key_control="registered",
                    expected_seed=seed,
                    expected_prompt_identity=prompt_identity,
                    expected_prompt_sha256=prompt_sha256,
                    run_id=run_id,
                    revision=runtime_candidate_revision,
                )
            )
        if repetitions == 2 and records[0] != records[1]:
            raise DeterminismQualificationError(
                "independent registered-key repetitions drifted"
            )
        if profile != "smoke":
            negative_key = hashlib.sha256(
                (
                    "ceg-wm-runtime-qualification-negative-key-v1\0" + root_key
                ).encode()
            ).hexdigest()
            negative = _validate_record(
                _execute_once(
                    backend_factory=backend_factory,
                    cache_root=root / "cache",
                    persistent_root=persistent,
                    hf_token=hf_token,
                    root_key=negative_key,
                    prompt=prompt,
                    seed=seed,
                    key_control="negative_identity",
                    run_id=run_id,
                    runtime_candidate_revision=runtime_candidate_revision,
                ),
                expected_key_control="negative_identity",
                expected_seed=seed,
                expected_prompt_identity=prompt_identity,
                expected_prompt_sha256=prompt_sha256,
                run_id=run_id,
                revision=runtime_candidate_revision,
            )
            if negative["key_public_digest"] == records[0]["key_public_digest"]:
                raise QualificationRunnerError(
                    "negative key identity did not diverge"
                )
            records.append(negative)
        record_digests = tuple(_record_digest(record) for record in records)
        if profile == "replay" and tuple(
            _replay_comparison_digest(record) for record in records
        ) != replay_comparison_digests:
            raise DeterminismQualificationError(
                "replay record digests differ from qualification source"
            )
        status = "passed"
    except BaseException as exc:
        failures.append(
            {
                "failure_class": _classify_failure(exc),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        record_digests = tuple(_record_digest(record) for record in records)
    summary = {
        "result_schema_version": 2,
        "profile": profile,
        "run_id": run_id,
        "result_zip_filename": output_path.name,
        "run_status": status,
        "runtime_candidate_revision": runtime_candidate_revision,
        "replay_source_revision": runtime_candidate_revision if replay_digests else None,
        "replay_source_run_id": replay_source_run_id,
        "replay_source_record_digests": list(replay_digests),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "prompt_identity": prompt_identity,
        "prompt_sha256": prompt_sha256,
        "repetition_count": len(records),
        "failure_count": len(failures),
        "failure_classes": [
            failure["failure_class"] for failure in failures
        ],
        "callback_status": (
            "passed"
            if records and all(record["callback_status"] == "passed" for record in records)
            else "failed"
        ),
        "actual_dtype_status": (
            "passed"
            if records
            and all(
                record["integrity_status"] == "passed"
                and record["budget_status"] == "accepted"
                for record in records
            )
            else "failed"
        ),
        "vae_status": (
            "passed"
            if records and all(record["vae_status"] == "passed" for record in records)
            else "failed"
        ),
        "qk_status": (
            "passed"
            if records and all(record["qk_status"] == "passed" for record in records)
            else "failed"
        ),
        "determinism_status": (
            "not_evaluated"
            if profile == "smoke" and status == "passed"
            else ("passed" if status == "passed" else "failed")
        ),
        "package_status": "verified" if package_verified else "failed",
        "dependency_status": "verified" if dependency_verified else "failed",
        "record_digests": list(record_digests),
        "key_controls": [record["key_control"] for record in records],
        "dependency_lock_evidence": list(dependency_evidence),
        "checks": records,
    }
    dependency_versions = {
        item["package_name"]: item["actual_version"]
        for item in dependency_evidence
    }
    environment = {
        "result_schema_version": 2,
        "profile": profile,
        "run_id": run_id,
        "runtime_candidate_revision": runtime_candidate_revision,
        "seed": seed,
        "prompt_identity": prompt_identity,
        "prompt_sha256": prompt_sha256,
        "record_digests": list(record_digests),
        "key_controls": [record["key_control"] for record in records],
        "python": dependency_versions.get("python"),
        "torch": dependency_versions.get("torch"),
        "diffusers": dependency_versions.get("diffusers"),
        "transformers": dependency_versions.get("transformers"),
        "accelerate": dependency_versions.get("accelerate"),
        "numpy": dependency_versions.get("numpy"),
        "pillow": dependency_versions.get("Pillow"),
        "safetensors": dependency_versions.get("safetensors"),
        "huggingface_hub": dependency_versions.get("huggingface-hub"),
        "cuda_available": (
            records[0]["cuda_available"] if records else False
        ),
        "cuda_runtime": records[0]["cuda_runtime"] if records else None,
        "gpu_name": records[0]["gpu_name"] if records else None,
    }
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        temp = Path(temporary)
        payloads = {
            "run_summary.json": json.dumps(
                summary, indent=2, sort_keys=True
            ) + "\n",
            "environment_summary.json": json.dumps(
                environment, indent=2, sort_keys=True
            ) + "\n",
            "runtime_checks.jsonl": "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in records
            ),
            "failures.jsonl": "".join(
                json.dumps(failure, sort_keys=True) + "\n"
                for failure in failures
            ),
        }
        with zipfile.ZipFile(
            output_path,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
    return {
        **summary,
        "result_zip": output_path.name,
        "result_zip_sha256": _sha256(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--runtime-candidate-revision", default="")
    parser.add_argument("--result-zip", required=True)
    parser.add_argument("--ephemeral-root", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--replay-source")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--root-key-env", default="CEG_WM_ROOT_KEY")
    parser.add_argument(
        "--prompt",
        default="A quiet mountain lake at sunrise, high detail.",
    )
    arguments = parser.parse_args(argv)
    result = run_runtime_qualification(
        profile=arguments.profile,
        run_id=arguments.run_id,
        package_root=arguments.package_root,
        runtime_candidate_revision=arguments.runtime_candidate_revision,
        result_zip=arguments.result_zip,
        ephemeral_root=arguments.ephemeral_root,
        persistent_root=arguments.persistent_root,
        hf_token=os.environ.get(arguments.hf_token_env),
        root_key=os.environ.get(arguments.root_key_env, ""),
        prompt=arguments.prompt,
        replay_source=arguments.replay_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["run_status"] == "passed":
        return 0
    failure_classes = set(result["failure_classes"])
    return 2 if "incomplete" in failure_classes else 1


if __name__ == "__main__":
    raise SystemExit(main())
