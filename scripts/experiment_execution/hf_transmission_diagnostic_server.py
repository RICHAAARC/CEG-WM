"""Colab/server launcher for the frozen HF transport diagnostic."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from experiments.protocol.hf_transmission_diagnostic import (
    load_hf_transmission_protocol,
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
from scripts.experiment_execution.hf_transmission_diagnostic_entrypoint import (
    execute_hf_transmission_diagnostic_session,
)


PROTOCOL_PATH = Path("configs/experiments/hf_transmission_diagnostic.json")
SAFE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


class HfTransmissionServerError(RuntimeError):
    """The server could not start or package the HF transport worker."""


def execute_hf_transmission_diagnostic_server_session(
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
    if (
        _paths_overlap(repository, persistent)
        or _paths_overlap(repository, cache)
        or _paths_overlap(persistent, cache)
    ):
        raise HfTransmissionServerError("execution roots must be disjoint")
    if (
        SAFE_ID_PATTERN.fullmatch(run_id) is None
        or SAFE_ID_PATTERN.fullmatch(session_id) is None
    ):
        raise HfTransmissionServerError("run or session identity is invalid")
    _verify_repository(repository, expected_revision)
    protocol, manifest = load_hf_transmission_protocol(
        repository / PROTOCOL_PATH,
        repository_root=repository,
    )
    runtime = json.loads((repository / RUNTIME_CONFIG_PATH).read_text("utf-8"))
    runtime_environment = dict(os.environ if environment is None else environment)
    hf_token = runtime_environment.get("HF_TOKEN")
    root_key = runtime_environment.get("CEG_WM_ROOT_KEY")
    if not hf_token or not root_key:
        raise HfTransmissionServerError("HF_TOKEN and CEG_WM_ROOT_KEY are required")
    resources = _probe_resources(persistent_root=persistent, cache_root=cache)
    if install_dependencies:
        _install_frozen_dependencies(repository)
    _download_configured_model(
        model_id=runtime["model_id"],
        model_revision=runtime["model_revision"],
        cache_root=cache,
        hf_token=hf_token,
    )
    exit_code, worker = execute_hf_transmission_diagnostic_session(
        repository_root=repository,
        expected_revision=expected_revision,
        persistent_root=persistent,
        cache_root=cache,
        run_id=run_id,
        session_id=session_id,
        environment={"HF_TOKEN": hf_token, "CEG_WM_ROOT_KEY": root_key},
    )
    artifact_key = "diagnostic_zip" if exit_code else "result_zip"
    artifact = Path(worker[artifact_key]).resolve()
    receipt_path = (
        persistent / run_id / "server_receipts" / session_id / "execution_receipt.json"
    )
    receipt = {
        **worker,
        "artifact_path": str(artifact),
        "artifact_sha256": _file_sha256(artifact),
        "committed_revision": expected_revision,
        "exit_code": exit_code,
        "model_id": runtime["model_id"],
        "model_revision": runtime["model_revision"],
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "manifest_id": manifest.manifest_id,
        "resource_facts": resources,
        "run_id": run_id,
        "session_id": session_id,
        "formal_tau_created": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
    }
    _write_json_create_only(receipt_path, receipt)
    return exit_code, {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path.read_bytes()).hexdigest(),
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--skip-dependency-install", action="store_true")
    args = parser.parse_args()
    exit_code, result = execute_hf_transmission_diagnostic_server_session(
        repository_root=args.repository_root,
        expected_revision=args.expected_revision,
        persistent_root=args.persistent_root,
        cache_root=args.cache_root,
        run_id=args.run_id,
        session_id=args.session_id,
        install_dependencies=not args.skip_dependency_install,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
