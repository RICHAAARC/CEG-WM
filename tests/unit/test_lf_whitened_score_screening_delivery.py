"""Server delivery checks for LF whitening fit and score screening."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from experiments.protocol.lf_whitened_score_screening import (
    RUN_ID,
    load_lf_whitened_score_screening_protocol,
)
from scripts.experiment_execution import lf_whitened_score_screening_server as server


ROOT = Path(__file__).resolve().parents[2]
SERVER_RELATIVE = Path(
    "scripts/experiment_execution/lf_whitened_score_screening_server.py"
)
SERVER = ROOT / SERVER_RELATIVE
PROTOCOL = ROOT / "configs/experiments/lf_whitened_score_screening.json"


@pytest.mark.quick
def test_lf_whitened_screening_server_help_imports_from_isolated_cwd(
    tmp_path: Path,
) -> None:
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
    assert "ModuleNotFoundError" not in completed.stderr


@pytest.mark.quick
def test_lf_whitened_screening_server_builds_exact_package_and_safe_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "exact_checkout"
    subprocess.run(
        ("git", "clone", "--quiet", str(ROOT), str(checkout)),
        check=True,
    )
    revision = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    protocol, null_fit_manifest, screening_manifest = (
        load_lf_whitened_score_screening_protocol(
            checkout / PROTOCOL.relative_to(ROOT),
            repository_root=checkout,
        )
    )
    persistent = tmp_path / "persistent"
    cache = tmp_path / "cache"
    artifact = persistent / RUN_ID / "session_results" / "session.zip"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(artifact, "x") as archive:
        archive.writestr("screening_decision.json", "{}\n")

    monkeypatch.setattr(
        server,
        "_probe_resources",
        lambda **_kwargs: {
            "cuda_device_name": "Test GPU",
            "cuda_total_memory_bytes": 1,
            "free_disk_bytes": {},
        },
    )
    monkeypatch.setattr(
        server,
        "_download_configured_model",
        lambda **_kwargs: cache / "snapshot",
    )
    worker_calls: list[dict[str, object]] = []

    def execute_worker(**kwargs: object) -> tuple[int, dict[str, object]]:
        worker_calls.append(kwargs)
        return 0, {
            "artifact_kind": "lf_whitened_score_screening_result",
            "result_zip": str(artifact),
            "protocol_digest": protocol.digest(),
            "null_fit_manifest_digest": null_fit_manifest.digest(),
            "screening_manifest_digest": screening_manifest.digest(),
            "candidate_config_digest": "a" * 64,
            "unit_roster_digest": protocol.unit_roster_digest,
            "committed_unit_count": 41,
            "session_committed_unit_count": 41,
            "termination_reason": "frozen_roster_complete",
            "screening_decision": {
                "allow_request_for_lf_whitened_directional_validation": False
            },
            "formal_tau_created": False,
            "candidate_promoted": False,
            "scientific_claims_supported": False,
        }

    monkeypatch.setattr(
        server,
        "execute_lf_whitened_score_screening_session",
        execute_worker,
    )
    exit_code, receipt = (
        server.execute_lf_whitened_score_screening_server_session(
            repository_root=checkout,
            expected_revision=revision,
            persistent_root=persistent,
            cache_root=cache,
            run_id=RUN_ID,
            session_id="lf_whitened_screening_server_session",
            environment={
                "HF_TOKEN": "private-hf-token",
                "CEG_WM_ROOT_KEY": "private-root-key",
            },
            install_dependencies=False,
        )
    )

    assert exit_code == 0
    assert len(worker_calls) == 1
    package = Path(str(receipt["execution_package_path"]))
    package_sha256 = sha256(package.read_bytes()).hexdigest()
    assert worker_calls[0]["execution_package_sha256"] == package_sha256
    assert receipt["execution_package_sha256"] == package_sha256
    assert receipt["committed_revision"] == revision
    assert receipt["operational_unit_count"] == 1
    assert receipt["scientific_unit_count"] == 40
    assert receipt["development_claim_boundary"] == protocol.claim_boundary
    assert receipt["formal_tau_created"] is False
    assert receipt["candidate_promoted"] is False
    receipt_text = Path(str(receipt["receipt_path"])).read_text("utf-8")
    assert "private-hf-token" not in receipt_text
    assert "private-root-key" not in receipt_text
    with ZipFile(package) as archive:
        names = set(archive.namelist())
        assert SERVER_RELATIVE.as_posix() in names
        assert "experiments/runners/lf_whitened_score_screening.py" in names
        assert "scripts/experiment_execution/lf_whitened_score_screening_entrypoint.py" in names
        assert archive.testzip() is None
        extracted = tmp_path / "extracted_package"
        archive.extractall(extracted)
    imported = subprocess.run(
        (sys.executable, "-I", str(extracted / SERVER_RELATIVE), "--help"),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    assert json.loads(receipt_text)["execution_package_sha256"] == package_sha256
