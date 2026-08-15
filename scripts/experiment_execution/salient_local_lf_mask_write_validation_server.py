"""Server boundary for the exact salient-local-LF mask/write worker."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Sequence
from zipfile import is_zipfile

from experiments.protocol.salient_local_lf_mask_write_validation import (
    load_salient_local_lf_mask_write_validation_protocol,
)
from scripts.experiment_execution.build_salient_local_lf_mask_write_validation_package import (
    build_salient_local_lf_mask_write_validation_package,
)
from scripts.experiment_execution.development_exploration_server import (
    RUNTIME_CONFIG_PATH, _absolute_directory, _download_configured_model,
    _file_sha256, _paths_overlap, _probe_resources, _verify_repository,
    _write_json_create_only,
)
from scripts.experiment_execution.salient_local_lf_mask_write_validation_entrypoint import WORKER_RESULT_PREFIX


PROTOCOL_PATH = Path("configs/experiments/salient_local_lf_mask_write_validation.json")
SAFE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


class SalientLocalLfMaskWriteServerError(RuntimeError):
    """The server could not preserve exact package and worker authority."""


def _install_dependencies(repository: Path) -> None:
    completed = subprocess.run(
        (sys.executable, "-m", "pip", "install", "--requirement",
         "requirements_inspyrenet_salient_local_lf_gpu_execution.txt",
         "--index-url", "https://pypi.org/simple",
         "--extra-index-url", "https://download.pytorch.org/whl/cu128",
         "--extra-index-url", "https://pypi.nvidia.com"),
        cwd=repository, check=False,
    )
    if completed.returncode:
        raise SalientLocalLfMaskWriteServerError("frozen dependency installation failed")


def _verify_locked_dependencies(repository: Path) -> str:
    lock_path = repository / "requirements_inspyrenet_salient_local_lf_gpu_execution.txt"
    locked = []
    for line in lock_path.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise SalientLocalLfMaskWriteServerError("dependency lock entry is invalid")
        name, version = line.split("==", 1)
        locked.append((name, version))
    if len(locked) != 67 or len({name.lower().replace("_", "-") for name, _ in locked}) != 67:
        raise SalientLocalLfMaskWriteServerError("dependency lock coverage drifted")
    try:
        observed = tuple((name, metadata.version(name)) for name, _ in locked)
    except metadata.PackageNotFoundError as exc:
        raise SalientLocalLfMaskWriteServerError("locked dependency is unavailable") from exc
    if observed != tuple(locked):
        raise SalientLocalLfMaskWriteServerError("installed dependency identity drifted")
    return sha256(lock_path.read_bytes()).hexdigest()


def _execute_worker(*, repository: Path, expected_revision: str, persistent: Path,
                    cache: Path, run_id: str, session_id: str,
                    package_sha256: str, environment: Mapping[str, str]) -> tuple[int, dict[str, object]]:
    completed = subprocess.run(
        (sys.executable, "-m", "scripts.experiment_execution.salient_local_lf_mask_write_validation_entrypoint",
         "--repository-root", str(repository), "--expected-revision", expected_revision,
         "--persistent-root", str(persistent), "--cache-root", str(cache),
         "--run-id", run_id, "--session-id", session_id,
         "--execution-package-sha256", package_sha256),
        cwd=repository, env=dict(environment), capture_output=True, text=True, check=False,
    )
    if completed.returncode not in {0, 3}:
        raise SalientLocalLfMaskWriteServerError("worker did not return a bounded result")
    payloads = [line.removeprefix(WORKER_RESULT_PREFIX) for line in completed.stdout.splitlines()
                if line.startswith(WORKER_RESULT_PREFIX)]
    if len(payloads) != 1:
        raise SalientLocalLfMaskWriteServerError("worker result identity is unavailable")
    try:
        value = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise SalientLocalLfMaskWriteServerError("worker result identity is invalid") from exc
    if type(value) is not dict:
        raise SalientLocalLfMaskWriteServerError("worker result identity is invalid")
    return completed.returncode, value


def execute_salient_local_lf_mask_write_validation_server_session(
    *, repository_root: str | Path, expected_revision: str,
    persistent_root: str | Path, cache_root: str | Path,
    run_id: str, session_id: str, environment: Mapping[str, str] | None = None,
    install_dependencies: bool = True,
) -> tuple[int, dict[str, object]]:
    repository = Path(repository_root).resolve()
    persistent = _absolute_directory(persistent_root, "persistent_root")
    cache = _absolute_directory(cache_root, "cache_root")
    if any((_paths_overlap(repository, persistent), _paths_overlap(repository, cache), _paths_overlap(persistent, cache))):
        raise SalientLocalLfMaskWriteServerError("execution roots must be disjoint")
    if SAFE_ID_PATTERN.fullmatch(run_id) is None or SAFE_ID_PATTERN.fullmatch(session_id) is None:
        raise SalientLocalLfMaskWriteServerError("run or session identity is invalid")
    _verify_repository(repository, expected_revision)
    protocol = load_salient_local_lf_mask_write_validation_protocol(
        repository / PROTOCOL_PATH, repository_root=repository,
    )
    if run_id != protocol.run_id:
        raise SalientLocalLfMaskWriteServerError("run identity drifted")
    env = dict(os.environ if environment is None else environment)
    if not env.get("HF_TOKEN") or not env.get("CEG_WM_ROOT_KEY") or not env.get("CEG_WM_INSPYRENET_CHECKPOINT_PATH"):
        raise SalientLocalLfMaskWriteServerError("required execution input is unavailable")
    resources = _probe_resources(persistent_root=persistent, cache_root=cache)
    if install_dependencies:
        _install_dependencies(repository)
    dependency_lock_identity = _verify_locked_dependencies(repository)
    runtime_document = json.loads((repository / RUNTIME_CONFIG_PATH).read_text("utf-8"))
    _download_configured_model(model_id=runtime_document["model_id"],
                               model_revision=runtime_document["model_revision"],
                               cache_root=cache, hf_token=env["HF_TOKEN"])
    package_path = persistent / run_id / "execution_packages" / f"{expected_revision}.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    if package_path.exists():
        package_sha = _file_sha256(package_path)
    else:
        package_sha = str(build_salient_local_lf_mask_write_validation_package(
            repository, package_path, expected_revision)["package_sha256"])
    exit_code, worker = _execute_worker(
        repository=repository, expected_revision=expected_revision,
        persistent=persistent, cache=cache, run_id=run_id, session_id=session_id,
        package_sha256=package_sha, environment=env,
    )
    artifact_key = "diagnostic_zip" if exit_code else "result_zip"
    artifact_value = worker.get(artifact_key)
    if type(artifact_value) is not str:
        raise SalientLocalLfMaskWriteServerError("worker artifact identity is missing")
    artifact = Path(artifact_value).resolve()
    if not artifact.is_file() or persistent not in artifact.parents or not is_zipfile(artifact):
        raise SalientLocalLfMaskWriteServerError("worker artifact is invalid")
    if (worker.get("protocol_digest") != protocol.digest()
            or worker.get("input_manifest_digest") != protocol.manifest.digest()
            or worker.get("unit_roster_digest") != protocol.unit_roster_digest
            or worker.get("package_sha256") != package_sha):
        raise SalientLocalLfMaskWriteServerError("worker frozen authority drifted")
    aggregate = worker.get("salient_local_lf_mask_write_aggregate")
    if exit_code and aggregate is not None:
        raise SalientLocalLfMaskWriteServerError("failed worker cannot forge an aggregate")
    receipt_path = persistent / run_id / "server_receipts" / session_id / "execution_receipt.json"
    receipt = {
        **worker, "artifact_path": str(artifact), "artifact_sha256": _file_sha256(artifact),
        "committed_revision": expected_revision, "execution_package_path": str(package_path),
        "execution_package_sha256": package_sha, "exit_code": exit_code,
        "model_id": runtime_document["model_id"], "model_revision": runtime_document["model_revision"],
        "protocol_id": protocol.protocol_id, "protocol_version": protocol.protocol_version,
        "operational_unit_count": 2, "scientific_unit_count": 8, "total_unit_count": 10,
        "maximum_attempts_per_unit": 1, "resource_facts": resources,
        "dependency_lock_identity": dependency_lock_identity,
        "run_id": run_id, "session_id": session_id,
        "development_claim_boundary": protocol.raw["claim_boundary"],
        "formal_tau_created": False, "fpr_estimated": False, "candidate_promoted": False,
    }
    _write_json_create_only(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = sha256(receipt_path.read_bytes()).hexdigest()
    return exit_code, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("repository-root", "expected-revision", "persistent-root", "cache-root", "run-id", "session-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--skip-dependency-install", action="store_true")
    args = parser.parse_args(argv)
    code, receipt = execute_salient_local_lf_mask_write_validation_server_session(
        repository_root=args.repository_root, expected_revision=args.expected_revision,
        persistent_root=args.persistent_root, cache_root=args.cache_root,
        run_id=args.run_id, session_id=args.session_id,
        install_dependencies=not args.skip_dependency_install,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
