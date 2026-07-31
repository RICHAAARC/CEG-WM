"""Verified package entrypoint for one C1 HF threshold-fit shard."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import re
from typing import Mapping, Sequence

import torch

from experiments.protocol.c1_hf_threshold_fit_records import (
    C1_HF_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE,
    canonical_c1_hf_threshold_fit_record_bytes,
    load_c1_hf_threshold_fit_record_collection,
    replay_c1_hf_threshold_fit_record_collection,
)
from experiments.runners.c1_hf_threshold_fit import (
    load_c1_hf_threshold_fit_package_authority,
    run_c1_hf_threshold_fit_verified_package_shard,
)
from experiments.runners.formal_operations import (
    FormalHfContentDetectionOperation,
    create_formal_content_detector_binding,
)


THRESHOLD_FIT_ENTRYPOINT_IDENTITY = (
    "scripts.experiment_execution.experiment_execution_entrypoint:"
    "execute_verified_threshold_fit_shard"
)
THRESHOLD_FIT_ENTRYPOINT_SCHEMA_VERSION = 1
THRESHOLD_FIT_EXECUTION_SCOPE = "c1_hf_threshold_fit_only"
THRESHOLD_FIT_FAILURE_CLASSES = frozenset(
    {
        "resource_failure",
        "execution_failure",
        "excluded",
        "scientific_failure",
        "incomplete",
    }
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
C1_DEPENDENCY_LOCK_PATH = "requirements_c1_threshold_fit.txt"
C1_DEPENDENCY_LOCK_SHA256 = (
    "07a4c1bbe6fc5e7e6b38334c5a9919a8565b810a9aae7820b61c24cee91270de"
)


class ExperimentExecutionEntrypointError(ValueError):
    """The verified threshold-fit package invocation failed closed."""


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _classify_threshold_fit_failure(
    *,
    failure_classes: set[str],
    retry_pending_count: int,
    excluded_count: int,
    complete_shard: bool,
) -> str | None:
    if "scientific_failure" in failure_classes:
        return "scientific_failure"
    if "execution_failure" in failure_classes:
        return "execution_failure"
    if "resource_failure" in failure_classes or retry_pending_count:
        return "resource_failure"
    if excluded_count:
        return "excluded"
    if not complete_shard:
        return "incomplete"
    return None


def _load_and_replay_threshold_fit_record(source: Path) -> tuple[object, object, bytes]:
    try:
        collection = load_c1_hf_threshold_fit_record_collection(source)
        last = replay_c1_hf_threshold_fit_record_collection(
            collection,
            expected_identity=collection.identity,
        )
        blob = source.read_bytes()
        if blob != canonical_c1_hf_threshold_fit_record_bytes(collection):
            raise ValueError("record changed after typed validation")
    except Exception as exc:
        raise ExperimentExecutionEntrypointError(
            "threshold-fit record collection failed typed replay"
        ) from exc
    return collection, last, blob


def _verified_dependency_versions(package_root: Path) -> tuple[dict[str, str], str]:
    lock_path = package_root / C1_DEPENDENCY_LOCK_PATH
    try:
        blob = lock_path.read_bytes()
        requirements = [
            line.split("==", 1)
            for line in blob.decode("utf-8").splitlines()
        ]
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ExperimentExecutionEntrypointError(
            "verified dependency lock is unreadable"
        ) from exc
    if (
        sha256(blob).hexdigest() != C1_DEPENDENCY_LOCK_SHA256
        or len(requirements) != 62
        or any(len(item) != 2 for item in requirements)
    ):
        raise ExperimentExecutionEntrypointError(
            "verified C1 dependency lock identity or closure drifted"
        )
    dependency_versions: dict[str, str] = {}
    for distribution, expected_version in requirements:
        normalized = re.sub(r"[-_.]+", "-", distribution).lower()
        if (
            re.fullmatch(
                r"[a-z0-9][a-z0-9-]*",
                distribution,
            )
            is None
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9.+_-]*",
                expected_version,
            )
            is None
            or normalized in dependency_versions
        ):
            raise ExperimentExecutionEntrypointError(
                "verified C1 dependency lock is not exact and unique"
            )
        try:
            observed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise ExperimentExecutionEntrypointError(
                f"verified dependency is unavailable: {distribution}"
            ) from exc
        if observed_version != expected_version:
            raise ExperimentExecutionEntrypointError(
                f"verified dependency version drifted: {distribution}"
            )
        dependency_versions[normalized] = observed_version
    if dependency_versions.get("torch") != "2.11.0+cu128":
        raise ExperimentExecutionEntrypointError(
            "verified C1 torch dependency drifted"
        )
    torch_import_version = str(torch.__version__)
    if torch_import_version != dependency_versions["torch"]:
        raise ExperimentExecutionEntrypointError(
            "imported torch version differs from package metadata"
        )
    return dependency_versions, torch_import_version


def execute_verified_threshold_fit_shard(
    *,
    package_revision_authority: object,
    package_root: str | Path,
    output_root: str | Path,
    records_root: str | Path,
    shard_index: int,
    run_id: str,
    registered_detection_key: str,
) -> Mapping[str, object]:
    """Run one verified formal shard without deriving tau or unlocking data."""

    root = Path(package_root).resolve()
    output = Path(output_root).resolve()
    records = Path(records_root).resolve()
    if (
        not 0 <= shard_index < 16
        or SAFE_ID.fullmatch(run_id) is None
        or not registered_detection_key
        or output.exists()
        or output == records
        or output in records.parents
        or records in output.parents
    ):
        raise ExperimentExecutionEntrypointError(
            "threshold-fit package invocation is invalid"
        )
    output.mkdir(parents=True)
    authority = load_c1_hf_threshold_fit_package_authority(root)
    detector_binding, _ = create_formal_content_detector_binding(
        FormalHfContentDetectionOperation(authority.adapter),
        prototype_image=torch.arange(12, dtype=torch.uint8).reshape(1, 3, 2, 2),
        detection_key=registered_detection_key,
    )
    environment_digest, resource_identity_digest, execution_facts = (
        _derive_threshold_fit_execution_identity(
            package_revision_authority=package_revision_authority,
            authority=authority,
        )
    )
    runner_summary = run_c1_hf_threshold_fit_verified_package_shard(
        authority=authority,
        shard_index=shard_index,
        run_id=run_id,
        registered_detection_key=registered_detection_key,
        environment_digest=environment_digest,
        resource_identity_digest=resource_identity_digest,
        records_root=records,
        package_revision_authority=package_revision_authority,
    )
    record_files = []
    failure_classes: set[str] = set()
    terminal_count = 0
    shard_root = (
        records
        / run_id
        / "threshold_fit"
        / f"shard_{shard_index:02d}"
    )
    expected_units = authority.assignments[
        shard_index * 256 : (shard_index + 1) * 256
    ]
    sources = (
        sorted(
            path for path in shard_root.glob("unit_*.json") if path.is_file()
        )
        if shard_root.is_dir()
        else []
    )
    for source in sources:
        relative = source.relative_to(shard_root).as_posix()
        collection, last, blob = _load_and_replay_threshold_fit_record(source)
        identity = collection.identity
        unit_index = identity.unit_index
        expected_unit = (
            expected_units[unit_index - shard_index * 256]
            if type(unit_index) is int
            and shard_index * 256 <= unit_index < (shard_index + 1) * 256
            else None
        )
        if (
            expected_unit is None
            or identity.run_id != run_id
            or identity.committed_revision
            != package_revision_authority.committed_revision
            or identity.execution_evidence_kind
            != C1_HF_THRESHOLD_FIT_REAL_EXECUTION_EVIDENCE
            or identity.c1_specification_digest
            != authority.metric_binding.c1_specification_digest
            or identity.protocol_id != authority.protocol_id
            or identity.protocol_version != authority.protocol_version
            or identity.protocol_digest != authority.metric_binding.protocol_digest
            or identity.shard_index != shard_index
            or identity.execution_config_digest
            != authority.configuration.execution_config_digest
            or identity.fit_manifest_digest
            != authority.metric_binding.fit_manifest_digest
            or identity.metric_binding_digest
            != authority.metric_binding.binding_digest
            or identity.metric_registry_digest
            != authority.metric_binding.metric_registry_digest
            or identity.candidate_config_digest != authority.candidate_config_digest
            or identity.method_config_digest != authority.method_config_digest
            or identity.runtime_config_digest != authority.runtime_config_digest
            or identity.model_revision != authority.model_revision
            or identity.detector_identity != detector_binding.detector_identity
            or identity.detector_config_digest
            != detector_binding.content_config_digest
            or identity.preprocessing_identity
            != detector_binding.preprocessing_identity
            or identity.registered_key_family_digest
            != authority.metric_binding.registered_key_family_digest
            or identity.registered_key_public_digest
            != detector_binding.root_key_public_digest
            or identity.environment_digest != environment_digest
            or identity.analysis_unit_identity != expected_unit
            or source.name != f"unit_{unit_index:04d}.json"
        ):
            raise ExperimentExecutionEntrypointError(
                "threshold-fit record is outside the frozen shard identity"
            )
        if last.status in {"success", "excluded"} or (
            last.status == "failed"
            and last.failure_class
            in {"execution_failure", "scientific_failure", "resource_failure"}
        ):
            terminal_count += 1
        failure_class = last.failure_class
        if failure_class is not None:
            failure_classes.add(failure_class)
        destination = output / "records" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        record_files.append(
            {
                "path": f"records/{relative}",
                "sha256": sha256(blob).hexdigest(),
                "size_bytes": len(blob),
            }
        )
    if len({entry["path"] for entry in record_files}) != len(record_files):
        raise ExperimentExecutionEntrypointError(
            "threshold-fit record identity is duplicated"
        )
    if (
        runner_summary["planned_unit_count"] != 256
        or runner_summary["recorded_unit_count"] != len(record_files)
        or runner_summary["success_count"]
        + runner_summary["excluded_count"]
        + runner_summary["failed_count"]
        + runner_summary["retry_pending_count"]
        != len(record_files)
    ):
        raise ExperimentExecutionEntrypointError(
            "threshold-fit runner counts differ from persisted records"
        )
    complete_shard = len(record_files) == 256 and terminal_count == 256
    failure_class = _classify_threshold_fit_failure(
        failure_classes=failure_classes,
        retry_pending_count=runner_summary["retry_pending_count"],
        excluded_count=runner_summary["excluded_count"],
        complete_shard=complete_shard,
    )
    if (
        failure_class is not None
        and failure_class not in THRESHOLD_FIT_FAILURE_CLASSES
    ):
        raise ExperimentExecutionEntrypointError(
            "threshold-fit failure taxonomy drifted"
        )
    records_digest = _canonical_digest(record_files)
    outcome = {
        "artifact_kind": (
            "c1_threshold_fit_shard_result"
            if failure_class is None
            else "c1_threshold_fit_shard_diagnostic"
        ),
        "entrypoint_identity": THRESHOLD_FIT_ENTRYPOINT_IDENTITY,
        "entrypoint_schema_version": THRESHOLD_FIT_ENTRYPOINT_SCHEMA_VERSION,
        "execution_scope": THRESHOLD_FIT_EXECUTION_SCOPE,
        "run_id": run_id,
        "shard_index": shard_index,
        "planned_shard_count": 16,
        "planned_units_per_shard": 256,
        "committed_revision": package_revision_authority.committed_revision,
        "environment_digest": environment_digest,
        "resource_identity_digest": resource_identity_digest,
        "execution_facts": execution_facts,
        "failure_class": failure_class,
        "runner_summary": dict(runner_summary),
        "record_files": record_files,
        "records_digest": records_digest,
        "scientific_claims_supported": False,
        "tau_approval": False,
        "confirmation_unlock": False,
        "held_out_evaluation_accessed": False,
    }
    (output / "threshold_fit_outcome.json").write_text(
        json.dumps(outcome, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outcome


def _derive_threshold_fit_execution_identity(
    *,
    package_revision_authority: object,
    authority: object,
) -> tuple[str, str, Mapping[str, object]]:
    """Derive identities from installed, model, CUDA, and package facts."""

    dependency_versions, torch_import_version = _verified_dependency_versions(
        authority.repository_root
    )
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if cuda_device_count:
        properties = torch.cuda.get_device_properties(0)
        cuda_device_name = str(properties.name)
        cuda_total_memory_bytes = int(properties.total_memory)
        cuda_capability = list(torch.cuda.get_device_capability(0))
    else:
        cuda_device_name = "unavailable"
        cuda_total_memory_bytes = 0
        cuda_capability = []
    package_facts = {
        "archive_sha256": package_revision_authority.archive_sha256,
        "bootstrap_sha256": package_revision_authority.bootstrap_sha256,
        "committed_revision": package_revision_authority.committed_revision,
        "copied_file_set_digest": (
            package_revision_authority.copied_file_set_digest
        ),
        "embedded_manifest_sha256": (
            package_revision_authority.embedded_manifest_sha256
        ),
    }
    environment_facts = {
        "dependency_versions": dependency_versions,
        "model_revision": authority.model_revision,
        "package_facts": package_facts,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "runtime_config_digest": authority.runtime_config_digest,
        "torch_cuda_version": torch.version.cuda,
        "torch_import_version": torch_import_version,
    }
    resource_facts = {
        "cuda_available": cuda_available,
        "cuda_capability": cuda_capability,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "cuda_total_memory_bytes": cuda_total_memory_bytes,
        "package_archive_sha256": package_revision_authority.archive_sha256,
        "selected_device": "cuda:0",
    }
    return (
        _canonical_digest(environment_facts),
        _canonical_digest(resource_facts),
        {
            "environment": environment_facts,
            "resource": resource_facts,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    parser.error(
        "direct entrypoint invocation is forbidden; use the verified external bootstrap"
    )


if __name__ == "__main__":
    raise SystemExit(main())
