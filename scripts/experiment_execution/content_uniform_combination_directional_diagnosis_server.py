"""Server launcher for the frozen uniform-combination directional diagnosis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.protocol.content_uniform_combination_directional_diagnosis import (
    CLAIM_BOUNDARY,
    canonical_digest,
    load_content_uniform_combination_directional_protocol,
)
from scripts.experiment_execution.content_uniform_combination_directional_diagnosis_entrypoint import (
    ContentUniformCombinationDirectionalStartupError,
    execute_content_uniform_combination_directional_diagnosis_session,
)
from scripts.experiment_execution.development_exploration_entrypoint import (
    _build_or_verify_package,
)
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


PROTOCOL_PATH = Path("configs/experiments/content_uniform_combination_directional_diagnosis.json")
SAFE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
SAFE_FAILURE_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,255}$")


class ContentUniformCombinationDirectionalServerError(RuntimeError):
    """The server could not start or verify the combination diagnosis worker."""


def _startup_failure_worker(
    *,
    error: ContentUniformCombinationDirectionalStartupError,
    persistent_root: Path,
    run_id: str,
    session_id: str,
    protocol,
    reference_manifest,
    probe_manifest,
    package_sha256: str,
) -> dict[str, object]:
    """Create one safe diagnostic only when the frozen run has no evidence."""

    if type(error) is not ContentUniformCombinationDirectionalStartupError:
        raise ContentUniformCombinationDirectionalServerError(
            "startup failure exact type is required"
        )
    if (
        error.failure_class not in {"implementation_failure", "resource_failure"}
        or type(error.failure_type) is not str
        or SAFE_FAILURE_TYPE_PATTERN.fullmatch(error.failure_type) is None
    ):
        raise ContentUniformCombinationDirectionalServerError(
            "startup failure identity is invalid"
        )
    run_root = persistent_root / run_id
    if run_root.exists():
        raise ContentUniformCombinationDirectionalServerError(
            "startup diagnostic requires a fresh run root"
        )
    result_root = run_root / "session_results"
    result_root.mkdir(parents=True, exist_ok=False)
    archive = result_root / f"{session_id}.zip"
    diagnostic = {
        "failure_class": error.failure_class,
        "failure_type": error.failure_type,
        "stage": "content_uniform_combination_directional_diagnosis_startup",
        "scientific_claims_supported": False,
    }
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as target:
        target.writestr(
            "committed_unit_ids.json",
            json.dumps([], separators=(",", ":"), sort_keys=True).encode("utf-8"),
        )
        target.writestr(
            "diagnostic.json",
            json.dumps(diagnostic, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            ),
        )
    return {
        "artifact_kind": "content_uniform_combination_directional_diagnosis_failure",
        "diagnostic_zip": str(archive),
        "protocol_digest": protocol.digest(),
        "reference_manifest_digest": canonical_digest(asdict(reference_manifest)),
        "probe_manifest_digest": canonical_digest(asdict(probe_manifest)),
        "input_manifest_digest": canonical_digest(
            {
                "probe": canonical_digest(asdict(probe_manifest)),
                "reference": canonical_digest(asdict(reference_manifest)),
            }
        ),
        "unit_roster_digest": protocol.unit_roster_digest,
        "package_sha256": package_sha256,
        "committed_unit_count": 0,
        "session_committed_unit_count": 0,
        "termination_reason": "worker_startup_failure",
        "content_uniform_combination_directional_aggregate": None,
        "failure_class": error.failure_class,
        "failure_type": error.failure_type,
        "formal_tau_created": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _validated_artifact(
    worker: Mapping[str, object],
    *,
    persistent_root: Path,
    exit_code: int,
) -> Path:
    value = worker.get("diagnostic_zip" if exit_code else "result_zip")
    if type(value) is not str:
        raise ContentUniformCombinationDirectionalServerError("worker returned no artifact")
    artifact = Path(value).resolve()
    if (
        not artifact.is_file()
        or persistent_root not in artifact.parents
        or not is_zipfile(artifact)
    ):
        raise ContentUniformCombinationDirectionalServerError("worker artifact is invalid")
    return artifact


def execute_content_uniform_combination_directional_diagnosis_server_session(
    *,
    repository_root: str | Path,
    expected_revision: str,
    persistent_root: str | Path,
    whitening_asset_persistent_root: str | Path,
    cache_root: str | Path,
    run_id: str,
    session_id: str,
    environment: Mapping[str, str] | None = None,
    install_dependencies: bool = True,
) -> tuple[int, dict[str, object]]:
    repository = Path(repository_root).resolve()
    persistent = _absolute_directory(persistent_root, "persistent_root")
    whitening_asset_persistent = _absolute_directory(
        whitening_asset_persistent_root, "whitening_asset_persistent_root"
    )
    cache = _absolute_directory(cache_root, "cache_root")
    if (
        _paths_overlap(repository, persistent)
        or _paths_overlap(repository, cache)
        or _paths_overlap(persistent, cache)
        or _paths_overlap(repository, whitening_asset_persistent)
        or _paths_overlap(cache, whitening_asset_persistent)
    ):
        raise ContentUniformCombinationDirectionalServerError("execution roots must be disjoint")
    if (
        SAFE_ID_PATTERN.fullmatch(run_id) is None
        or SAFE_ID_PATTERN.fullmatch(session_id) is None
    ):
        raise ContentUniformCombinationDirectionalServerError("run or session identity is invalid")
    _verify_repository(repository, expected_revision)
    protocol, reference_manifest, probe_manifest = (
        load_content_uniform_combination_directional_protocol(
            repository / PROTOCOL_PATH,
            repository_root=repository,
        )
    )
    if run_id != protocol.run_id:
        raise ContentUniformCombinationDirectionalServerError("run identity drifted")
    runtime = json.loads((repository / RUNTIME_CONFIG_PATH).read_text("utf-8"))
    runtime_environment = dict(os.environ if environment is None else environment)
    hf_token = runtime_environment.get("HF_TOKEN")
    root_key = runtime_environment.get("CEG_WM_ROOT_KEY")
    if not hf_token or not root_key:
        raise ContentUniformCombinationDirectionalServerError(
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
    package = _build_or_verify_package(repository, persistent, expected_revision)
    package_sha256 = _file_sha256(package)
    try:
        exit_code, worker = execute_content_uniform_combination_directional_diagnosis_session(
            repository_root=repository,
            expected_revision=expected_revision,
            persistent_root=persistent,
            whitening_asset_persistent_root=whitening_asset_persistent,
            cache_root=cache,
            run_id=run_id,
            session_id=session_id,
            execution_package_sha256=package_sha256,
            environment={"HF_TOKEN": hf_token, "CEG_WM_ROOT_KEY": root_key},
        )
    except ContentUniformCombinationDirectionalStartupError as exc:
        if type(exc) is not ContentUniformCombinationDirectionalStartupError:
            raise
        exit_code = 3
        worker = _startup_failure_worker(
            error=exc,
            persistent_root=persistent,
            run_id=run_id,
            session_id=session_id,
            protocol=protocol,
            reference_manifest=reference_manifest,
            probe_manifest=probe_manifest,
            package_sha256=package_sha256,
        )
    if type(exit_code) is not int or isinstance(exit_code, bool):
        raise ContentUniformCombinationDirectionalServerError("worker exit code is invalid")
    artifact = _validated_artifact(
        worker,
        persistent_root=persistent,
        exit_code=exit_code,
    )
    if (
        worker.get("protocol_digest") != protocol.digest()
        or worker.get("reference_manifest_digest")
        != canonical_digest(asdict(reference_manifest))
        or worker.get("probe_manifest_digest")
        != canonical_digest(asdict(probe_manifest))
        or worker.get("unit_roster_digest") != protocol.unit_roster_digest
        or worker.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ContentUniformCombinationDirectionalServerError("worker frozen identity drifted")
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
        "reference_fit_cluster_count": protocol.reference_fit_cluster_count,
        "directional_probe_cluster_count": protocol.directional_probe_cluster_count,
        "total_unit_count": protocol.maximum_total_units,
        "maximum_attempts_per_unit": protocol.maximum_attempts_per_unit,
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
    parser.add_argument("--whitening-asset-persistent-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--skip-dependency-install", action="store_true")
    arguments = parser.parse_args(argv)
    exit_code, receipt = execute_content_uniform_combination_directional_diagnosis_server_session(
        repository_root=arguments.repository_root,
        expected_revision=arguments.expected_revision,
        persistent_root=arguments.persistent_root,
        whitening_asset_persistent_root=arguments.whitening_asset_persistent_root,
        cache_root=arguments.cache_root,
        run_id=arguments.run_id,
        session_id=arguments.session_id,
        install_dependencies=not arguments.skip_dependency_install,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
