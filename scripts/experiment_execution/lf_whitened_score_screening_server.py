"""Server launcher for the frozen LF whitening fit and score screening."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence
from zipfile import is_zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from experiments.protocol.lf_whitened_score_screening import (
    CLAIM_BOUNDARY,
    load_lf_whitened_score_screening_protocol,
)
from scripts.experiment_execution.delivery_support import (
    _build_or_verify_package,
)
from scripts.experiment_execution.server_support import (
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
from scripts.experiment_execution.lf_whitened_score_screening_entrypoint import (
    execute_lf_whitened_score_screening_session,
)


PROTOCOL_PATH = Path("configs/experiments/lf_whitened_score_screening.json")
SAFE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


class LfWhitenedScoreScreeningServerError(RuntimeError):
    """The server could not start or package the LF whitening worker."""


def _validated_worker_artifact(
    worker: Mapping[str, object],
    *,
    persistent_root: Path,
    exit_code: int,
) -> Path:
    artifact_key = "diagnostic_zip" if exit_code else "result_zip"
    artifact_value = worker.get(artifact_key)
    if type(artifact_value) is not str:
        raise LfWhitenedScoreScreeningServerError(
            "worker returned no result artifact"
        )
    artifact = Path(artifact_value).resolve()
    if (
        not artifact.is_file()
        or persistent_root not in artifact.parents
        or not is_zipfile(artifact)
    ):
        raise LfWhitenedScoreScreeningServerError(
            "worker result artifact is unavailable or invalid"
        )
    return artifact


def execute_lf_whitened_score_screening_server_session(
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
    """Prepare one exact package and execute or resume the frozen worker."""

    repository = Path(repository_root).resolve()
    persistent = _absolute_directory(persistent_root, "persistent_root")
    cache = _absolute_directory(cache_root, "cache_root")
    if (
        _paths_overlap(repository, persistent)
        or _paths_overlap(repository, cache)
        or _paths_overlap(persistent, cache)
    ):
        raise LfWhitenedScoreScreeningServerError(
            "execution roots must be disjoint"
        )
    if (
        SAFE_ID_PATTERN.fullmatch(run_id) is None
        or SAFE_ID_PATTERN.fullmatch(session_id) is None
    ):
        raise LfWhitenedScoreScreeningServerError(
            "run or session identity is invalid"
        )
    _verify_repository(repository, expected_revision)
    protocol, null_fit_manifest, screening_manifest = (
        load_lf_whitened_score_screening_protocol(
            repository / PROTOCOL_PATH,
            repository_root=repository,
        )
    )
    if run_id != protocol.run_id:
        raise LfWhitenedScoreScreeningServerError("run identity drifted")
    runtime = json.loads((repository / RUNTIME_CONFIG_PATH).read_text("utf-8"))
    runtime_environment = dict(os.environ if environment is None else environment)
    hf_token = runtime_environment.get("HF_TOKEN")
    root_key = runtime_environment.get("CEG_WM_ROOT_KEY")
    if not hf_token or not root_key:
        raise LfWhitenedScoreScreeningServerError(
            "HF_TOKEN and CEG_WM_ROOT_KEY are required"
        )
    resources = _probe_resources(persistent_root=persistent, cache_root=cache)
    if install_dependencies:
        _install_frozen_dependencies(repository)
    _download_configured_model(
        model_id=runtime["model_id"],
        model_revision=runtime["model_revision"],
        cache_root=cache,
        hf_token=hf_token,
    )
    package = _build_or_verify_package(
        repository, persistent, expected_revision
    )
    package_sha256 = _file_sha256(package)
    exit_code, worker = execute_lf_whitened_score_screening_session(
        repository_root=repository,
        expected_revision=expected_revision,
        persistent_root=persistent,
        cache_root=cache,
        run_id=run_id,
        session_id=session_id,
        execution_package_sha256=package_sha256,
        environment={"HF_TOKEN": hf_token, "CEG_WM_ROOT_KEY": root_key},
    )
    if type(exit_code) is not int or isinstance(exit_code, bool):
        raise LfWhitenedScoreScreeningServerError("worker exit code is invalid")
    artifact = _validated_worker_artifact(
        worker,
        persistent_root=persistent,
        exit_code=exit_code,
    )
    if (
        worker.get("protocol_digest") != protocol.digest()
        or worker.get("null_fit_manifest_digest") != null_fit_manifest.digest()
        or worker.get("screening_manifest_digest") != screening_manifest.digest()
        or worker.get("unit_roster_digest") != protocol.unit_roster_digest
    ):
        raise LfWhitenedScoreScreeningServerError(
            "worker frozen identity differs from checked-in protocol"
        )
    receipt_path = (
        persistent
        / run_id
        / "server_receipts"
        / session_id
        / "execution_receipt.json"
    )
    receipt = {
        **worker,
        "artifact_path": str(artifact),
        "artifact_sha256": _file_sha256(artifact),
        "committed_revision": expected_revision,
        "execution_package_path": str(package),
        "execution_package_sha256": package_sha256,
        "exit_code": exit_code,
        "model_id": runtime["model_id"],
        "model_revision": runtime["model_revision"],
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "operational_unit_count": protocol.operational_unit_count,
        "scientific_unit_count": (
            protocol.null_fit_cluster_count + protocol.screening_cluster_count
        ),
        "resource_facts": resources,
        "run_id": run_id,
        "session_id": session_id,
        "development_claim_boundary": CLAIM_BOUNDARY,
        "scientific_claims_supported": False,
        "formal_tau_created": False,
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
    exit_code, receipt = execute_lf_whitened_score_screening_server_session(
        repository_root=arguments.repository_root,
        expected_revision=arguments.expected_revision,
        persistent_root=arguments.persistent_root,
        cache_root=arguments.cache_root,
        run_id=arguments.run_id,
        session_id=arguments.session_id,
        install_dependencies=not arguments.skip_dependency_install,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
