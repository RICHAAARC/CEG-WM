"""Server launcher for the frozen Q/K synchronization-write diagnosis."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Sequence
from zipfile import is_zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.protocol.qk_synchronization_write_diagnostic import (
    CLAIM_BOUNDARY,
    load_qk_synchronization_write_protocol,
)
from scripts.experiment_execution.development_exploration_entrypoint import _build_or_verify_package
from scripts.experiment_execution.development_exploration_server import (
    RUNTIME_CONFIG_PATH,
    _absolute_directory,
    _download_configured_model,
    _file_sha256,
    _install_frozen_dependencies,
    _paths_overlap,
    _probe_resources,
    _verify_repository,
    _write_json_create_only,
)
PROTOCOL_PATH = Path("configs/experiments/qk_synchronization_write_diagnostic.json")
SAFE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
WORKER_RESULT_PREFIX = "CEG_WM_QK_WORKER_RESULT="


class QkSynchronizationWriteServerError(RuntimeError):
    """The server could not start the exact Q/K diagnosis worker."""


def _execute_worker_process(
    *,
    repository: Path,
    expected_revision: str,
    persistent: Path,
    cache: Path,
    run_id: str,
    session_id: str,
    package_sha256: str,
    environment: Mapping[str, str],
) -> tuple[int, Mapping[str, object]]:
    worker_environment = dict(environment)
    worker_environment["CUDA_LAUNCH_BLOCKING"] = "1"
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "scripts.experiment_execution.qk_synchronization_write_diagnostic_entrypoint",
            "--repository-root",
            str(repository),
            "--expected-revision",
            expected_revision,
            "--persistent-root",
            str(persistent),
            "--cache-root",
            str(cache),
            "--run-id",
            run_id,
            "--session-id",
            session_id,
            "--execution-package-sha256",
            package_sha256,
        ),
        cwd=repository,
        env=worker_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in {0, 3}:
        raise QkSynchronizationWriteServerError(
            "Q/K diagnosis worker did not return a bounded result"
        )
    result_lines = tuple(
        line.removeprefix(WORKER_RESULT_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    )
    if len(result_lines) != 1:
        raise QkSynchronizationWriteServerError(
            "Q/K diagnosis worker result identity is unavailable"
        )
    try:
        result = json.loads(result_lines[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise QkSynchronizationWriteServerError(
            "Q/K diagnosis worker result identity is invalid"
        ) from exc
    if type(result) is not dict:
        raise QkSynchronizationWriteServerError(
            "Q/K diagnosis worker result identity is invalid"
        )
    return completed.returncode, result


def _validated_artifact(worker: Mapping[str, object], *, persistent: Path, exit_code: int) -> Path:
    value = worker.get("diagnostic_zip" if exit_code else "result_zip")
    if type(value) is not str:
        raise QkSynchronizationWriteServerError("worker returned no result artifact")
    artifact = Path(value).resolve()
    if not artifact.is_file() or persistent not in artifact.parents or not is_zipfile(artifact):
        raise QkSynchronizationWriteServerError("worker result artifact is invalid")
    return artifact


def execute_qk_synchronization_write_diagnostic_server_session(
    *,
    repository_root: str | Path,
    expected_revision: str,
    persistent_root: str | Path,
    cache_root: str | Path,
    run_id: str,
    session_id: str,
    environment: Mapping[str, str] | None = None,
    install_dependencies: bool = True,
) -> tuple[int, dict[str, object]]:
    repository = Path(repository_root).resolve()
    persistent = _absolute_directory(persistent_root, "persistent_root")
    cache = _absolute_directory(cache_root, "cache_root")
    if any((_paths_overlap(repository, persistent), _paths_overlap(repository, cache), _paths_overlap(persistent, cache))):
        raise QkSynchronizationWriteServerError("execution roots must be disjoint")
    if SAFE_ID_PATTERN.fullmatch(run_id) is None or SAFE_ID_PATTERN.fullmatch(session_id) is None:
        raise QkSynchronizationWriteServerError("run or session identity is invalid")
    _verify_repository(repository, expected_revision)
    protocol, manifest = load_qk_synchronization_write_protocol(repository / PROTOCOL_PATH, repository_root=repository)
    if run_id != protocol.run_id:
        raise QkSynchronizationWriteServerError("run identity drifted")
    runtime_document = json.loads((repository / RUNTIME_CONFIG_PATH).read_text("utf-8"))
    env = dict(os.environ if environment is None else environment)
    hf_token = env.get("HF_TOKEN")
    root_key = env.get("CEG_WM_ROOT_KEY")
    if not hf_token or not root_key:
        raise QkSynchronizationWriteServerError("HF_TOKEN and CEG_WM_ROOT_KEY are required")
    resources = _probe_resources(persistent_root=persistent, cache_root=cache)
    if install_dependencies:
        _install_frozen_dependencies(repository)
    _download_configured_model(model_id=runtime_document["model_id"], model_revision=runtime_document["model_revision"], cache_root=cache, hf_token=hf_token)
    package = _build_or_verify_package(repository, persistent, expected_revision)
    package_sha = _file_sha256(package)
    exit_code, worker = _execute_worker_process(
        repository=repository,
        expected_revision=expected_revision,
        persistent=persistent,
        cache=cache,
        run_id=run_id,
        session_id=session_id,
        package_sha256=package_sha,
        environment={
            **env,
            "HF_TOKEN": hf_token,
            "CEG_WM_ROOT_KEY": root_key,
        },
    )
    if type(exit_code) is not int or isinstance(exit_code, bool):
        raise QkSynchronizationWriteServerError("worker exit code is invalid")
    artifact = _validated_artifact(worker, persistent=persistent, exit_code=exit_code)
    if worker.get("protocol_digest") != protocol.digest() or worker.get("input_manifest_digest") != manifest.digest() or worker.get("unit_roster_digest") != protocol.authorized_unit_roster_digest or worker.get("source_cluster_deny_list_digest") != protocol.source_cluster_deny_list_digest or worker.get("cuda_launch_blocking_identity") != "cuda_launch_blocking_enabled":
        raise QkSynchronizationWriteServerError("worker frozen identity drifted")
    if (
        worker.get("qk_synchronization_diagnosis_aggregate") is not None
        or worker.get("termination_reason")
        not in {
            "operational_failure_localization_complete",
            "operational_failure_localization_failed",
        }
    ):
        raise QkSynchronizationWriteServerError(
            "worker operational localization boundary drifted"
        )
    receipt_path = persistent / run_id / "server_receipts" / session_id / "execution_receipt.json"
    receipt = {
        **worker,
        "artifact_path": str(artifact),
        "artifact_sha256": _file_sha256(artifact),
        "committed_revision": expected_revision,
        "execution_package_path": str(package),
        "execution_package_sha256": package_sha,
        "exit_code": exit_code,
        "model_id": runtime_document["model_id"],
        "model_revision": runtime_document["model_revision"],
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "operational_unit_count": protocol.authorized_operational_unit_count,
        "scientific_unit_count": protocol.authorized_scientific_unit_count,
        "total_unit_count": protocol.authorized_total_unit_count,
        "maximum_attempts_per_unit": (
            protocol.authorized_maximum_attempts_per_unit
        ),
        "resource_facts": resources,
        "run_id": run_id,
        "session_id": session_id,
        "development_claim_boundary": CLAIM_BOUNDARY,
        "scientific_claims_supported": False,
        "formal_tau_created": False,
        "fpr_estimated": False,
        "candidate_promoted": False,
    }
    _write_json_create_only(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = sha256(receipt_path.read_bytes()).hexdigest()
    return exit_code, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--skip-dependency-install", action="store_true")
    arguments = parser.parse_args(argv)
    exit_code, receipt = execute_qk_synchronization_write_diagnostic_server_session(repository_root=arguments.repository_root, expected_revision=arguments.expected_revision, persistent_root=arguments.persistent_root, cache_root=arguments.cache_root, run_id=arguments.run_id, session_id=arguments.session_id, install_dependencies=not arguments.skip_dependency_install)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
