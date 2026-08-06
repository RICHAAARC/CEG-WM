"""Server and Colab-neutral launcher for development module exploration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping, Sequence
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
RUNTIME_CONFIG_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")
PROTOCOL_CONFIG_PATH = Path(
    "configs/experiments/thirteen_module_mechanism_screening.json"
)
DEPENDENCY_LOCK_PATH = Path(
    "requirements_development_exploration_gpu_execution.txt"
)
PYPI_INDEX_URL = "https://pypi.org/simple"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
NVIDIA_INDEX_URL = "https://pypi.nvidia.com"
SERVER_FAILURE_STAGES = frozenset(
    {
        "arguments",
        "configuration",
        "dependency_install",
        "diagnostic",
        "model_download",
        "receipt",
        "repository",
        "resource_preflight",
        "secrets",
        "worker_execution",
        "worker_import",
        "worker_result",
    }
)


class DevelopmentExplorationServerError(RuntimeError):
    """A server preparation or development worker boundary failed."""

    def __init__(self, stage: str, message: str, *, failure_type: str | None = None) -> None:
        if stage not in SERVER_FAILURE_STAGES:
            raise ValueError("server failure stage is unregistered")
        super().__init__(message)
        self.stage = stage
        self.safe_message = message
        self.failure_type = failure_type or type(self).__name__


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DevelopmentExplorationServerError(
            "repository",
            "repository identity is unavailable",
            failure_type=type(exc).__name__,
        ) from exc
    return completed.stdout.strip()


def _absolute_directory(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise DevelopmentExplorationServerError(
            "arguments",
            f"{role} must be absolute",
        )
    resolved = path.resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DevelopmentExplorationServerError(
            "resource_preflight",
            f"{role} is unavailable",
            failure_type=type(exc).__name__,
        ) from exc
    if not resolved.is_dir():
        raise DevelopmentExplorationServerError(
            "resource_preflight",
            f"{role} is not a directory",
        )
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _verify_repository(root: Path, expected_revision: str) -> None:
    if not root.is_absolute() or not root.is_dir():
        raise DevelopmentExplorationServerError(
            "repository",
            "repository root is invalid",
        )
    if REVISION_PATTERN.fullmatch(expected_revision) is None:
        raise DevelopmentExplorationServerError(
            "repository",
            "expected revision is invalid",
        )
    if _git(root, "rev-parse", "HEAD") != expected_revision:
        raise DevelopmentExplorationServerError(
            "repository",
            "repository HEAD differs from expected revision",
        )
    if _git(root, "status", "--porcelain"):
        raise DevelopmentExplorationServerError(
            "repository",
            "repository worktree must be clean",
        )


def _load_frozen_bindings(root: Path) -> dict[str, str]:
    try:
        runtime_blob = (root / RUNTIME_CONFIG_PATH).read_bytes()
        runtime = json.loads(runtime_blob)
        protocol_path = root / PROTOCOL_CONFIG_PATH
        protocol = json.loads(protocol_path.read_bytes())
        model_id = runtime["model_id"]
        model_revision = runtime["model_revision"]
        protocol_id = protocol["protocol_id"]
        protocol_version = protocol["protocol_version"]
        unit_roster_digest = protocol["study_budget"]["unit_roster_digest"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DevelopmentExplorationServerError(
            "configuration",
            "frozen runtime or development protocol is unreadable",
            failure_type=type(exc).__name__,
        ) from exc
    if (
        type(model_id) is not str
        or not model_id
        or type(model_revision) is not str
        or REVISION_PATTERN.fullmatch(model_revision) is None
        or type(protocol_id) is not str
        or not protocol_id
        or type(protocol_version) is not str
        or not protocol_version
        or type(unit_roster_digest) is not str
        or DIGEST_PATTERN.fullmatch(unit_roster_digest) is None
    ):
        raise DevelopmentExplorationServerError(
            "configuration",
            "frozen runtime or development protocol identity is invalid",
        )
    try:
        from experiments.protocol.development_exploration import (
            load_frozen_development_exploration_protocol,
        )

        loaded_protocol = load_frozen_development_exploration_protocol(protocol_path)
    except Exception as exc:
        raise DevelopmentExplorationServerError(
            "configuration",
            "frozen development protocol failed typed loading",
            failure_type=type(exc).__name__,
        ) from exc
    if (
        loaded_protocol.protocol_id != protocol_id
        or loaded_protocol.protocol_version != protocol_version
        or loaded_protocol.study_budget.unit_roster_digest != unit_roster_digest
    ):
        raise DevelopmentExplorationServerError(
            "configuration",
            "typed development protocol differs from its checked-in identity",
        )
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "protocol_digest": loaded_protocol.digest(),
        "runtime_config_digest": sha256(
            json.dumps(
                runtime,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "unit_roster_digest": unit_roster_digest,
    }


def _probe_resources(*, persistent_root: Path, cache_root: Path) -> dict[str, object]:
    free_disk_bytes: dict[str, int] = {}
    try:
        for root in (persistent_root, cache_root):
            observed = int(shutil.disk_usage(root).free)
            if observed <= 0:
                raise DevelopmentExplorationServerError(
                    "resource_preflight",
                    "execution storage has no available space",
                )
            free_disk_bytes[str(root)] = observed
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        gpu_name, memory_mib_text = (
            part.strip() for part in rows[0].rsplit(",", 1)
        )
        gpu_memory_bytes = int(memory_mib_text) * 1024 * 1024
    except DevelopmentExplorationServerError:
        raise
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as exc:
        raise DevelopmentExplorationServerError(
            "resource_preflight",
            "CUDA device identity is unavailable",
            failure_type=type(exc).__name__,
        ) from exc
    if not gpu_name or gpu_memory_bytes <= 0:
        raise DevelopmentExplorationServerError(
            "resource_preflight",
            "CUDA device identity is invalid",
        )
    return {
        "cuda_device_name": gpu_name,
        "cuda_total_memory_bytes": gpu_memory_bytes,
        "free_disk_bytes": free_disk_bytes,
    }


def _install_frozen_dependencies(root: Path) -> None:
    lock_path = root / DEPENDENCY_LOCK_PATH
    if not lock_path.is_file():
        raise DevelopmentExplorationServerError(
            "dependency_install",
            "development dependency lock is unavailable",
        )
    try:
        subprocess.run(
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--index-url",
                PYPI_INDEX_URL,
                "--extra-index-url",
                PYTORCH_INDEX_URL,
                "--extra-index-url",
                NVIDIA_INDEX_URL,
                "-r",
                str(lock_path),
            ),
            cwd=root,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DevelopmentExplorationServerError(
            "dependency_install",
            "frozen dependency installation failed",
            failure_type=type(exc).__name__,
        ) from exc


def _download_configured_model(
    *,
    model_id: str,
    model_revision: str,
    cache_root: Path,
    hf_token: str,
) -> Path:
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=model_revision,
                cache_dir=str(cache_root / "huggingface"),
                token=hf_token,
            )
        ).resolve()
    except Exception as exc:
        raise DevelopmentExplorationServerError(
            "model_download",
            "configured model revision download failed",
            failure_type=type(exc).__name__,
        ) from exc
    if not snapshot.is_dir():
        raise DevelopmentExplorationServerError(
            "model_download",
            "configured model revision is unavailable after download",
        )
    return snapshot


def _execute_development_entrypoint(
    *,
    repository_root: Path,
    expected_revision: str,
    persistent_root: Path,
    cache_root: Path,
    run_id: str,
    session_id: str,
    environment: Mapping[str, str],
    maximum_wiring_clusters: int | None,
    stop_before_scientific_units: bool,
) -> tuple[int, Mapping[str, object]]:
    try:
        from scripts.experiment_execution.development_exploration_entrypoint import (
            execute_development_exploration_session,
        )
    except ImportError as exc:
        raise DevelopmentExplorationServerError(
            "worker_import",
            "development exploration worker entrypoint is unavailable",
            failure_type=type(exc).__name__,
        ) from exc
    try:
        return execute_development_exploration_session(
            repository_root=repository_root,
            expected_revision=expected_revision,
            persistent_root=persistent_root,
            cache_root=cache_root,
            run_id=run_id,
            session_id=session_id,
            environment=environment,
            maximum_wiring_clusters=maximum_wiring_clusters,
            stop_before_scientific_units=stop_before_scientific_units,
        )
    except Exception as exc:
        raise DevelopmentExplorationServerError(
            "worker_execution",
            "development exploration worker execution failed",
            failure_type=type(exc).__name__,
        ) from exc


def _write_json_create_only(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as target:
            target.write(encoded)
    except OSError as exc:
        raise DevelopmentExplorationServerError(
            "receipt",
            "execution receipt could not be created",
            failure_type=type(exc).__name__,
        ) from exc


def _validated_worker_result(
    result: Mapping[str, object],
    *,
    persistent_root: Path,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(result, Mapping):
        raise DevelopmentExplorationServerError(
            "worker_result",
            "development worker result is invalid",
        )
    artifact_values = [
        result[name]
        for name in ("result_zip", "diagnostic_zip")
        if type(result.get(name)) is str
    ]
    if len(artifact_values) != 1:
        raise DevelopmentExplorationServerError(
            "worker_result",
            "development worker returned no unique result or diagnostic ZIP",
        )
    artifact_source = Path(str(artifact_values[0]))
    artifact = artifact_source.resolve()
    if (
        artifact_source.is_symlink()
        or not artifact.is_file()
        or persistent_root not in artifact.parents
        or not zipfile.is_zipfile(artifact)
    ):
        raise DevelopmentExplorationServerError(
            "worker_result",
            "development worker artifact is unavailable or invalid",
        )
    digest_fields = (
        "protocol_digest",
        "execution_intent_authority_digest",
        "input_manifest_digest",
        "candidate_config_digest",
        "unit_roster_digest",
        "package_sha256",
    )
    safe_result: dict[str, object] = {}
    for name in digest_fields:
        value = result.get(name)
        if type(value) is not str or DIGEST_PATTERN.fullmatch(value) is None:
            raise DevelopmentExplorationServerError(
                "worker_result",
                f"development worker {name} is invalid",
            )
        safe_result[name] = value
    committed_unit_count = result.get("committed_unit_count")
    termination_reason = result.get("termination_reason")
    artifact_kind = result.get("artifact_kind")
    if (
        type(committed_unit_count) is not int
        or isinstance(committed_unit_count, bool)
        or committed_unit_count < 0
        or type(termination_reason) is not str
        or not termination_reason
        or artifact_kind not in {
            "development_exploration_result",
            "development_exploration_diagnostic",
        }
    ):
        raise DevelopmentExplorationServerError(
            "worker_result",
            "development worker completion summary is invalid",
        )
    safe_result.update(
        {
            "artifact_kind": artifact_kind,
            "committed_unit_count": committed_unit_count,
            "termination_reason": termination_reason,
        }
    )
    return artifact, safe_result


def execute_development_exploration_server_session(
    *,
    repository_root: str | Path,
    expected_revision: str,
    persistent_root: str | Path,
    cache_root: str | Path,
    run_id: str,
    session_id: str,
    environment: Mapping[str, str] | None = None,
    install_dependencies: bool = True,
    maximum_wiring_clusters: int | None = None,
    stop_before_scientific_units: bool = False,
) -> tuple[int, dict[str, object]]:
    """Prepare a worker environment, run one session, and write a safe receipt."""

    repository = Path(repository_root).resolve()
    persistent = _absolute_directory(persistent_root, "persistent_root")
    cache = _absolute_directory(cache_root, "cache_root")
    if (
        _paths_overlap(repository, persistent)
        or _paths_overlap(repository, cache)
        or _paths_overlap(persistent, cache)
    ):
        raise DevelopmentExplorationServerError(
            "arguments",
            "repository, persistent, and cache roots must be disjoint",
        )
    if SAFE_ID_PATTERN.fullmatch(run_id) is None or SAFE_ID_PATTERN.fullmatch(session_id) is None:
        raise DevelopmentExplorationServerError(
            "arguments",
            "run_id or session_id is invalid",
        )
    if type(stop_before_scientific_units) is not bool:
        raise DevelopmentExplorationServerError(
            "arguments",
            "stop before scientific units flag must be boolean",
        )
    _verify_repository(repository, expected_revision)
    bindings = _load_frozen_bindings(repository)
    runtime_environment = dict(os.environ if environment is None else environment)
    hf_token = runtime_environment.get("HF_TOKEN")
    root_key = runtime_environment.get("CEG_WM_ROOT_KEY")
    if not hf_token or not root_key:
        raise DevelopmentExplorationServerError(
            "secrets",
            "HF_TOKEN and CEG_WM_ROOT_KEY are required",
        )
    resources = _probe_resources(
        persistent_root=persistent,
        cache_root=cache,
    )
    if install_dependencies:
        _install_frozen_dependencies(repository)
    _download_configured_model(
        model_id=bindings["model_id"],
        model_revision=bindings["model_revision"],
        cache_root=cache,
        hf_token=hf_token,
    )
    worker_environment = {
        "HF_TOKEN": hf_token,
        "CEG_WM_ROOT_KEY": root_key,
    }
    exit_code, worker_result = _execute_development_entrypoint(
        repository_root=repository,
        expected_revision=expected_revision,
        persistent_root=persistent,
        cache_root=cache,
        run_id=run_id,
        session_id=session_id,
        environment=worker_environment,
        maximum_wiring_clusters=maximum_wiring_clusters,
        stop_before_scientific_units=stop_before_scientific_units,
    )
    if type(exit_code) is not int or isinstance(exit_code, bool):
        raise DevelopmentExplorationServerError(
            "worker_result",
            "development worker exit code is invalid",
        )
    artifact, safe_worker_result = _validated_worker_result(
        worker_result,
        persistent_root=persistent,
    )
    if safe_worker_result["protocol_digest"] != bindings["protocol_digest"]:
        raise DevelopmentExplorationServerError(
            "worker_result",
            "development worker protocol digest differs from checked-in protocol",
        )
    if safe_worker_result["unit_roster_digest"] != bindings["unit_roster_digest"]:
        raise DevelopmentExplorationServerError(
            "worker_result",
            "development worker unit roster differs from checked-in protocol",
        )
    receipt_path = (
        persistent
        / run_id
        / "server_receipts"
        / session_id
        / "execution_receipt.json"
    )
    receipt = {
        **safe_worker_result,
        "artifact_path": str(artifact),
        "artifact_sha256": _file_sha256(artifact),
        "committed_revision": expected_revision,
        "exit_code": exit_code,
        "model_id": bindings["model_id"],
        "model_revision": bindings["model_revision"],
        "protocol_id": bindings["protocol_id"],
        "protocol_version": bindings["protocol_version"],
        "resource_facts": resources,
        "run_id": run_id,
        "session_id": session_id,
        "scientific_claims_supported": False,
        "formal_tau_created": False,
        "calibration_locked": False,
    }
    _write_json_create_only(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return exit_code, receipt


def _failure_artifacts(
    *,
    persistent_root: str | Path,
    expected_revision: str,
    run_id: str,
    session_id: str,
    error: DevelopmentExplorationServerError,
) -> dict[str, object]:
    persistent = Path(persistent_root).resolve()
    safe_run_id = run_id if SAFE_ID_PATTERN.fullmatch(run_id) else "invalid_run"
    safe_session_id = (
        session_id if SAFE_ID_PATTERN.fullmatch(session_id) else "invalid_session"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = persistent / safe_run_id / "server_failures" / safe_session_id
    diagnostic_zip = root / f"development_diagnostic_{timestamp}.zip"
    diagnostic_zip.parent.mkdir(parents=True, exist_ok=True)
    diagnostic = {
        "artifact_kind": "development_exploration_diagnostic",
        "committed_revision": expected_revision,
        "failure_stage": error.stage,
        "failure_type": error.failure_type,
        "responsibility_id": None,
        "unit_index": None,
        "run_id": run_id,
        "session_id": session_id,
        "scientific_claims_supported": False,
        "formal_tau_created": False,
        "calibration_locked": False,
    }
    try:
        with zipfile.ZipFile(
            diagnostic_zip,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                "development_server_diagnostic.json",
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
            )
    except OSError as exc:
        raise DevelopmentExplorationServerError(
            "diagnostic",
            "development diagnostic ZIP could not be created",
            failure_type=type(exc).__name__,
        ) from exc
    receipt_path = root / f"execution_failure_receipt_{timestamp}.json"
    receipt = {
        **diagnostic,
        "artifact_path": str(diagnostic_zip),
        "artifact_sha256": _file_sha256(diagnostic_zip),
        "exit_code": 3,
    }
    _write_json_create_only(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--skip-dependency-install", action="store_true")
    parser.add_argument("--maximum-wiring-clusters", type=int)
    parser.add_argument("--stop-before-scientific-units", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        exit_code, receipt = execute_development_exploration_server_session(
            repository_root=arguments.repository_root,
            expected_revision=arguments.expected_revision,
            persistent_root=arguments.persistent_root,
            cache_root=arguments.cache_root,
            run_id=arguments.run_id,
            session_id=arguments.session_id,
            install_dependencies=not arguments.skip_dependency_install,
            maximum_wiring_clusters=arguments.maximum_wiring_clusters,
            stop_before_scientific_units=arguments.stop_before_scientific_units,
        )
    except DevelopmentExplorationServerError as error:
        receipt = _failure_artifacts(
            persistent_root=arguments.persistent_root,
            expected_revision=arguments.expected_revision,
            run_id=arguments.run_id,
            session_id=arguments.session_id,
            error=error,
        )
        exit_code = 3
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
