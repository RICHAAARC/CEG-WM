"""Server boundary for the exact salient-local-LF mask/write worker."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, is_zipfile, ZipFile, ZipInfo

from experiments.protocol.salient_local_lf_mask_write_validation import (
    load_salient_local_lf_mask_write_validation_protocol,
)
from scripts.experiment_execution.build_salient_local_lf_mask_write_validation_package import (
    SalientLocalLfPackageBuildError,
    build_salient_local_lf_mask_write_validation_package,
    resolve_required_git_authority_revisions,
    verify_extracted_salient_local_lf_mask_write_validation_package,
    verify_salient_local_lf_mask_write_validation_package,
)
from scripts.experiment_execution.development_exploration_server import (
    RUNTIME_CONFIG_PATH, _absolute_directory, _download_configured_model,
    _file_sha256, _paths_overlap, _probe_resources, _verify_repository,
    _write_json_create_only,
)
from scripts.experiment_execution.salient_local_lf_mask_write_validation_entrypoint import (
    WORKER_RESULT_PREFIX,
    _safe_failure,
)


PROTOCOL_PATH = Path("configs/experiments/salient_local_lf_mask_write_validation.json")
SAFE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
STARTUP_STEP_ORDER = (
    "repository_identity_verified",
    "required_git_authority_revisions_resolved",
    "required_git_authority_objects_hydrated",
    "protocol_authority_loaded",
    "execution_inputs_verified",
    "resource_preflight_completed",
    "dependency_lock_verified",
    "model_asset_prepared",
    "execution_package_verified",
    "packaged_protocol_verified",
    "worker_execution",
    "operational_units",
    "scientific_units",
)


class SalientLocalLfMaskWriteServerError(RuntimeError):
    """The server could not preserve exact package and worker authority."""


class SalientLocalLfRemoteAuthorityError(SalientLocalLfMaskWriteServerError):
    """A required exact Git authority could not be hydrated safely."""

    def __init__(self, failure_class: str) -> None:
        if failure_class not in {"identity_blocked", "environment_blocked"}:
            raise ValueError("remote authority failure class is invalid")
        super().__init__("required exact Git authority is unavailable")
        self.failure_class = failure_class


class SalientLocalLfWorkerProcessError(SalientLocalLfMaskWriteServerError):
    """The worker process ended without one canonical bounded result."""

    def __init__(
        self, reason_identity: str, *, return_code: int,
        stdout: str, stderr: str,
    ) -> None:
        if reason_identity not in {
            "unexpected_return_code",
            "worker_result_missing",
            "worker_result_duplicated",
            "worker_result_invalid_json",
            "worker_result_invalid_type",
        }:
            raise ValueError("worker process reason identity is invalid")
        super().__init__("worker did not return one bounded result")
        self.reason_identity = reason_identity
        self.return_code = return_code
        self.signal_number = -return_code if return_code < 0 else None
        self.failure_class = (
            "resource_blocked"
            if self.signal_number is not None
            else (
                "integrity_blocked"
                if reason_identity != "unexpected_return_code"
                else "implementation_blocked"
            )
        )
        self.stdout_summary = _bounded_stream_summary(stdout)
        self.stderr_summary = _bounded_stream_summary(stderr)


def _bounded_stream_summary(value: str) -> str:
    payload = value.encode("utf-8", errors="replace")
    return (
        "redacted_worker_stream:"
        f"bytes={len(payload)}:sha256={sha256(payload).hexdigest()}"
    )[:4096]


def _run_git_authority_command(
    repository: Path, arguments: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SalientLocalLfRemoteAuthorityError("environment_blocked") from exc


def _exact_commit_is_available(repository: Path, revision: str) -> bool:
    probe = _run_git_authority_command(
        repository,
        ("cat-file", "-e", f"{revision}^{{commit}}"),
    )
    if probe.returncode != 0:
        return False
    identity = _run_git_authority_command(
        repository,
        ("rev-parse", "--verify", f"{revision}^{{commit}}"),
    )
    return identity.returncode == 0 and identity.stdout.strip() == revision


def hydrate_required_git_authority_revisions(
    repository: str | Path, revisions: Sequence[str],
) -> tuple[str, ...]:
    """Hydrate only missing exact commits via one bounded direct-SHA fetch each."""

    root = Path(repository).resolve()
    requested = tuple(revisions)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None
               for value in requested)
    ):
        raise SalientLocalLfRemoteAuthorityError("identity_blocked")
    for revision in requested:
        if _exact_commit_is_available(root, revision):
            continue
        fetched = _run_git_authority_command(
            root,
            ("fetch", "--no-tags", "--depth", "1", "origin", revision),
        )
        if fetched.returncode != 0 or not _exact_commit_is_available(root, revision):
            raise SalientLocalLfRemoteAuthorityError("identity_blocked")
    return requested


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _startup_failure_class(error: BaseException) -> str:
    if type(error) is SalientLocalLfWorkerProcessError:
        return error.failure_class
    if type(error) is SalientLocalLfRemoteAuthorityError:
        return error.failure_class
    if type(error) is SalientLocalLfPackageBuildError:
        return "integrity_blocked"
    if isinstance(error, MemoryError):
        return "resource_blocked"
    if type(error) is OSError:
        return "environment_blocked"
    return "implementation_blocked"


def _write_startup_failure_evidence(
    *, error: BaseException, repository: Path, persistent: Path,
    expected_revision: str, run_id: str, session_id: str,
    operation_identity: str, completed_steps: Sequence[str],
    package_path: Path | None, package_sha256: str | None,
    failure_stage: str = "server_startup",
    return_code: int = 3,
    artifact_kind: str = "salient_local_lf_mask_write_validation_startup_failure",
) -> tuple[int, dict[str, object]]:
    failure_class = _startup_failure_class(error)
    completed = tuple(completed_steps)
    if any(step not in STARTUP_STEP_ORDER for step in completed):
        raise SalientLocalLfMaskWriteServerError("startup completion identity is invalid")
    not_executed = tuple(step for step in STARTUP_STEP_ORDER if step not in completed)
    diagnostic = _safe_failure(
        error,
        repository=repository,
        operation_identity=operation_identity,
        unit_index=None,
    )
    diagnostic.update(
        {
            "failure_class": failure_class,
            "failure_stage": failure_stage,
            "return_code": return_code,
            "completed_steps": completed,
            "not_executed_steps": not_executed,
        }
    )
    if type(error) is SalientLocalLfWorkerProcessError:
        diagnostic.update(
            {
                "worker_process_reason_identity": error.reason_identity,
                "worker_signal_number": error.signal_number,
                "sanitized_stdout": error.stdout_summary,
                "sanitized_stderr": error.stderr_summary,
            }
        )
    package_relative = None
    if (
        package_path is not None
        and package_sha256 is not None
        and package_path.is_file()
    ):
        try:
            package_relative = package_path.resolve().relative_to(persistent).as_posix()
        except ValueError as exc:
            raise SalientLocalLfMaskWriteServerError(
                "startup package path is outside persistent root"
            ) from exc
    receipt_base: dict[str, object] = {
        "artifact_kind": artifact_kind,
        "committed_revision": expected_revision,
        "run_id": run_id,
        "session_id": session_id,
        "exit_code": 3,
        "failure_class": failure_class,
        "failure_stage": failure_stage,
        "failure_operation_identity": operation_identity,
        "completed_steps": completed,
        "not_executed_steps": not_executed,
        "execution_package_available": package_relative is not None,
        "execution_package_relative_path": package_relative,
        "execution_package_sha256": package_sha256,
        "protocol_digest": None,
        "input_manifest_digest": None,
        "unit_roster_digest": None,
        "committed_unit_count": 0,
        "session_committed_unit_count": 0,
        "operational_unit_count": 2,
        "scientific_unit_count": 8,
        "total_unit_count": 10,
        "maximum_attempts_per_unit": 1,
        "salient_local_lf_mask_write_aggregate": None,
        "scientific_claims_supported": False,
        "formal_tau_created": False,
        "fpr_estimated": False,
        "candidate_promoted": False,
    }
    diagnostic_bytes = _canonical_bytes(diagnostic)
    receipt_bytes = _canonical_bytes(receipt_base)
    checksum_bytes = (
        f"{sha256(diagnostic_bytes).hexdigest()}  diagnostic.json\n"
        f"{sha256(receipt_bytes).hexdigest()}  execution_receipt.json\n"
    ).encode("ascii")
    artifact = persistent / run_id / "startup_failures" / session_id / "diagnostic.zip"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(artifact, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in (
            ("diagnostic.json", diagnostic_bytes),
            ("execution_receipt.json", receipt_bytes),
            ("SHA256SUMS", checksum_bytes),
        ):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    artifact_relative = artifact.relative_to(persistent).as_posix()
    receipt = {
        **receipt_base,
        "diagnostic_zip_relative_path": artifact_relative,
        "artifact_sha256": _file_sha256(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
    }
    if artifact_kind == "salient_local_lf_mask_write_validation_failure":
        receipt["artifact_path"] = str(artifact)
        receipt["execution_package_path"] = (
            str(package_path) if package_relative is not None else None
        )
    receipt_path = (
        persistent / run_id / "server_receipts" / session_id / "execution_receipt.json"
    )
    _write_json_create_only(receipt_path, receipt)
    receipt["receipt_relative_path"] = receipt_path.relative_to(persistent).as_posix()
    receipt["receipt_sha256"] = sha256(receipt_path.read_bytes()).hexdigest()
    return 3, receipt


def _extract_verified_execution_package(package_path: Path, destination: Path) -> Path:
    """Materialize the already exact-verified package without consulting Git."""

    if destination.exists():
        raise SalientLocalLfMaskWriteServerError("package extraction destination already exists")
    destination.mkdir(parents=True)
    with ZipFile(package_path, "r") as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if len(names) != len(set(names)):
            raise SalientLocalLfMaskWriteServerError("package member identity is duplicated")
        for info in infos:
            pure = PurePosixPath(info.filename)
            if (pure.is_absolute() or not pure.parts or ".." in pure.parts
                    or "." in pure.parts or "\\" in info.filename
                    or info.is_dir() or info.external_attr >> 16 != 0o100644):
                raise SalientLocalLfMaskWriteServerError("package member identity is unsafe")
            target = destination.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(archive.read(info))
    if (destination / ".git").exists():
        raise SalientLocalLfMaskWriteServerError("execution package cannot contain Git authority")
    return destination


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
        raise SalientLocalLfWorkerProcessError(
            "unexpected_return_code",
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    payloads = [line.removeprefix(WORKER_RESULT_PREFIX) for line in completed.stdout.splitlines()
                if line.startswith(WORKER_RESULT_PREFIX)]
    if not payloads:
        raise SalientLocalLfWorkerProcessError(
            "worker_result_missing",
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    if len(payloads) != 1:
        raise SalientLocalLfWorkerProcessError(
            "worker_result_duplicated",
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    try:
        value = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise SalientLocalLfWorkerProcessError(
            "worker_result_invalid_json",
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        ) from exc
    if type(value) is not dict:
        raise SalientLocalLfWorkerProcessError(
            "worker_result_invalid_type",
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return completed.returncode, value


def _validate_worker_result(
    *, worker: Mapping[str, object], exit_code: int, persistent: Path,
    protocol: object, package_sha256: str,
) -> tuple[Path, object]:
    artifact_key = "diagnostic_zip" if exit_code else "result_zip"
    artifact_value = worker.get(artifact_key)
    if type(artifact_value) is not str:
        raise SalientLocalLfMaskWriteServerError("worker artifact identity is missing")
    artifact = Path(artifact_value).resolve()
    if not artifact.is_file() or persistent not in artifact.parents or not is_zipfile(artifact):
        raise SalientLocalLfMaskWriteServerError("worker artifact is invalid")
    aggregate = worker.get("salient_local_lf_mask_write_aggregate")
    bootstrap_failure = worker.get("worker_failure_stage") == "worker_bootstrap"
    if bootstrap_failure:
        observed_authority = (
            worker.get("protocol_digest"),
            worker.get("input_manifest_digest"),
            worker.get("unit_roster_digest"),
        )
        expected_authority = (
            protocol.digest(), protocol.manifest.digest(), protocol.unit_roster_digest,
        )
        if (
            exit_code != 3
            or worker.get("package_sha256") != package_sha256
            or observed_authority not in {expected_authority, (None, None, None)}
            or worker.get("committed_unit_count") != 0
            or worker.get("session_committed_unit_count") != 0
            or aggregate is not None
            or worker.get("scientific_claims_supported") is not False
        ):
            raise SalientLocalLfMaskWriteServerError(
                "worker bootstrap failure authority drifted"
            )
        return artifact, aggregate
    if (
        worker.get("protocol_digest") != protocol.digest()
        or worker.get("input_manifest_digest") != protocol.manifest.digest()
        or worker.get("unit_roster_digest") != protocol.unit_roster_digest
        or worker.get("package_sha256") != package_sha256
    ):
        raise SalientLocalLfMaskWriteServerError("worker frozen authority drifted")
    if exit_code and aggregate is not None:
        raise SalientLocalLfMaskWriteServerError("failed worker cannot forge an aggregate")
    expected_claim = (
        type(aggregate) is dict
        and aggregate.get("successful_observation_count") == 8
        and all(aggregate.get(key) == 0 for key in (
            "identity_failure_count", "integrity_failure_count",
            "implementation_failure_count", "resource_failure_count",
            "environment_failure_count",
        ))
    )
    if worker.get("scientific_claims_supported") is not expected_claim:
        raise SalientLocalLfMaskWriteServerError(
            "worker scientific claim authority drifted"
        )
    return artifact, aggregate


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
    completed_steps: list[str] = []
    operation_identity = "salient_local_lf_repository_identity_verification"
    package_path: Path | None = None
    package_sha: str | None = None
    worker_started = False
    try:
        _verify_repository(repository, expected_revision)
        completed_steps.append("repository_identity_verified")

        operation_identity = "salient_local_lf_required_git_authority_resolution"
        config_payload = (repository / PROTOCOL_PATH).read_bytes()
        required_revisions = resolve_required_git_authority_revisions(
            execution_revision=expected_revision,
            config_payload=config_payload,
        )
        completed_steps.append("required_git_authority_revisions_resolved")

        operation_identity = "salient_local_lf_required_git_authority_hydration"
        hydrate_required_git_authority_revisions(repository, required_revisions)
        completed_steps.append("required_git_authority_objects_hydrated")

        operation_identity = "salient_local_lf_protocol_authority_load"
        protocol = load_salient_local_lf_mask_write_validation_protocol(
            repository / PROTOCOL_PATH, repository_root=repository,
        )
        if run_id != protocol.run_id:
            raise SalientLocalLfMaskWriteServerError("run identity drifted")
        completed_steps.append("protocol_authority_loaded")

        operation_identity = "salient_local_lf_execution_input_verification"
        env = dict(os.environ if environment is None else environment)
        if (
            not env.get("HF_TOKEN")
            or not env.get("CEG_WM_ROOT_KEY")
            or not env.get("CEG_WM_INSPYRENET_CHECKPOINT_PATH")
        ):
            raise SalientLocalLfMaskWriteServerError(
                "required execution input is unavailable"
            )
        completed_steps.append("execution_inputs_verified")

        operation_identity = "salient_local_lf_resource_preflight"
        resources = _probe_resources(persistent_root=persistent, cache_root=cache)
        completed_steps.append("resource_preflight_completed")

        operation_identity = "salient_local_lf_dependency_lock_verification"
        if install_dependencies:
            _install_dependencies(repository)
        dependency_lock_identity = _verify_locked_dependencies(repository)
        completed_steps.append("dependency_lock_verified")

        operation_identity = "salient_local_lf_model_asset_preparation"
        runtime_document = json.loads(
            (repository / RUNTIME_CONFIG_PATH).read_text("utf-8")
        )
        _download_configured_model(
            model_id=runtime_document["model_id"],
            model_revision=runtime_document["model_revision"],
            cache_root=cache,
            hf_token=env["HF_TOKEN"],
        )
        completed_steps.append("model_asset_prepared")

        operation_identity = "salient_local_lf_execution_package_verification"
        package_path = (
            persistent / run_id / "execution_packages" / f"{expected_revision}.zip"
        )
        package_path.parent.mkdir(parents=True, exist_ok=True)
        if package_path.exists():
            package_sha = str(
                verify_salient_local_lf_mask_write_validation_package(
                    repository, package_path, expected_revision,
                )["package_sha256"]
            )
        else:
            package_sha = str(
                build_salient_local_lf_mask_write_validation_package(
                    repository, package_path, expected_revision,
                )["package_sha256"]
            )
        completed_steps.append("execution_package_verified")

        with TemporaryDirectory(
            prefix="ceg-wm-salient-local-lf-package-"
        ) as temporary:
            operation_identity = "salient_local_lf_packaged_protocol_verification"
            execution_repository = _extract_verified_execution_package(
                package_path, Path(temporary) / "repository",
            )
            verify_extracted_salient_local_lf_mask_write_validation_package(
                execution_repository, expected_revision,
            )
            packaged_protocol = load_salient_local_lf_mask_write_validation_protocol(
                execution_repository / PROTOCOL_PATH,
                repository_root=execution_repository,
            )
            if (
                packaged_protocol.digest() != protocol.digest()
                or packaged_protocol.manifest.digest() != protocol.manifest.digest()
                or packaged_protocol.unit_roster_digest != protocol.unit_roster_digest
            ):
                raise SalientLocalLfMaskWriteServerError(
                    "packaged protocol authority drifted"
                )
            completed_steps.append("packaged_protocol_verified")
            operation_identity = "salient_local_lf_worker_execution"
            worker_started = True
            exit_code, worker = _execute_worker(
                repository=execution_repository,
                expected_revision=expected_revision,
                persistent=persistent,
                cache=cache,
                run_id=run_id,
                session_id=session_id,
                package_sha256=package_sha,
                environment=env,
            )
            completed_steps.append("worker_execution")
            artifact, aggregate = _validate_worker_result(
                worker=worker,
                exit_code=exit_code,
                persistent=persistent,
                protocol=protocol,
                package_sha256=package_sha,
            )
    except Exception as exc:
        return _write_startup_failure_evidence(
            error=exc,
            repository=repository,
            persistent=persistent,
            expected_revision=expected_revision,
            run_id=run_id,
            session_id=session_id,
            operation_identity=operation_identity,
            completed_steps=completed_steps,
            package_path=package_path,
            package_sha256=package_sha,
            failure_stage=("worker_process" if worker_started else "server_startup"),
            return_code=(
                exc.return_code
                if type(exc) is SalientLocalLfWorkerProcessError
                else 3
            ),
            artifact_kind=(
                "salient_local_lf_mask_write_validation_failure"
                if worker_started
                else "salient_local_lf_mask_write_validation_startup_failure"
            ),
        )
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
