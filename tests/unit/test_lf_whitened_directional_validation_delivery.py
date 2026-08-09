"""Delivery boundary for LF whitened directional validation."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from experiments.protocol.lf_whitened_directional_validation import (
    load_lf_whitened_directional_validation_protocol,
)
from scripts.experiment_execution.development_exploration_entrypoint import (
    _build_or_verify_package,
)
from scripts.experiment_execution import lf_whitened_directional_validation_server as server


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/lf_whitened_directional_validation.json"
SERVER_RELATIVE = Path("scripts/experiment_execution/lf_whitened_directional_validation_server.py")
SERVER = ROOT / SERVER_RELATIVE
REQUIRED_PACKAGE_MEMBERS = {
    "configs/experiments/lf_whitened_directional_validation.json",
    "configs/experiments/lf_whitened_directional_validation_manifest.json",
    "experiments/metrics/lf_whitened_directional_validation.py",
    "experiments/protocol/lf_whitened_directional_validation.py",
    "experiments/runners/lf_whitened_directional_validation.py",
    "scripts/experiment_execution/lf_whitened_directional_validation_entrypoint.py",
    SERVER_RELATIVE.as_posix(),
}


@pytest.mark.quick
def test_lf_whitened_directional_server_help_imports_from_isolated_cwd(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        (sys.executable, "-I", str(SERVER), "--help"),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--whitening-asset-persistent-root" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr


@pytest.mark.quick
def test_lf_whitened_directional_generic_package_contains_complete_execution_chain(tmp_path: Path) -> None:
    package = _build_or_verify_package(ROOT, tmp_path, "a" * 40)
    with ZipFile(package) as archive:
        names = set(archive.namelist())
        assert REQUIRED_PACKAGE_MEMBERS <= names
        assert archive.testzip() is None
    assert sha256(package.read_bytes()).hexdigest()


@pytest.mark.quick
def test_lf_whitened_directional_server_writes_safe_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        PROTOCOL, repository_root=ROOT
    )
    persistent = tmp_path / "persistent"
    fit_persistent = tmp_path / "fit_persistent"
    cache = tmp_path / "cache"
    for path in (persistent, fit_persistent, cache):
        path.mkdir()
    artifact = persistent / "worker.zip"
    with ZipFile(artifact, "x") as archive:
        archive.writestr("result.json", "{}")
    package = persistent / "package.zip"
    package.write_bytes(b"package")
    root_secret = "directional-root-secret"
    hf_secret = "directional-hf-secret"
    monkeypatch.setattr(server, "_verify_repository", lambda *_args: None)
    monkeypatch.setattr(server, "_probe_resources", lambda **_kwargs: {"gpu": "test"})
    monkeypatch.setattr(server, "_download_configured_model", lambda **_kwargs: None)
    monkeypatch.setattr(server, "_build_or_verify_package", lambda *_args: package)
    worker = {
        "result_zip": str(artifact),
        "protocol_digest": protocol.digest(),
        "input_manifest_digest": manifest.digest(),
        "candidate_config_digest": "c" * 64,
        "whitening_asset_digest": "d" * 64,
        "whitening_asset_fit_producer_revision": protocol.whitening_asset_fit_producer_revision,
        "unit_roster_digest": protocol.unit_roster_digest,
        "source_cluster_deny_list_digest": protocol.source_cluster_deny_list_digest,
        "committed_unit_count": 33,
        "session_committed_unit_count": 33,
        "termination_reason": "frozen_roster_complete",
        "directional_aggregate": None,
        "formal_tau_created": False,
        "fpr_estimated": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
    }
    monkeypatch.setattr(
        server,
        "execute_lf_whitened_directional_validation_session",
        lambda **_kwargs: (0, worker),
    )

    code, receipt = server.execute_lf_whitened_directional_validation_server_session(
        repository_root=ROOT,
        expected_revision="a" * 40,
        persistent_root=persistent,
        whitening_asset_persistent_root=fit_persistent,
        cache_root=cache,
        run_id=protocol.run_id,
        session_id="lf_whitened_directional_receipt_session",
        environment={"HF_TOKEN": hf_secret, "CEG_WM_ROOT_KEY": root_secret},
        install_dependencies=False,
    )

    receipt_bytes = Path(receipt["receipt_path"]).read_bytes()
    assert code == 0
    assert receipt["committed_revision"] == "a" * 40
    assert receipt["operational_unit_count"] == 1
    assert receipt["scientific_unit_count"] == 32
    assert receipt["formal_tau_created"] is False
    assert receipt["fpr_estimated"] is False
    assert receipt["candidate_promoted"] is False
    assert root_secret.encode() not in receipt_bytes
    assert hf_secret.encode() not in receipt_bytes
    assert json.loads(receipt_bytes)["development_claim_boundary"] == protocol.claim_boundary
