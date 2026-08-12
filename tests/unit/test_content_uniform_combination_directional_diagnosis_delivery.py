from __future__ import annotations

from pathlib import Path
import inspect
import subprocess
import sys

import pytest

from experiments.protocol.content_uniform_combination_directional_diagnosis import (
    load_content_uniform_combination_directional_protocol,
)
from scripts.experiment_execution import content_uniform_combination_directional_diagnosis_entrypoint as entrypoint
from scripts.experiment_execution import content_uniform_combination_directional_diagnosis_server as server


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/content_uniform_combination_directional_diagnosis.json"


def test_server_help_imports_from_isolated_working_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/experiment_execution/content_uniform_combination_directional_diagnosis_server.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--whitening-asset-persistent-root" in result.stdout


def test_execution_chain_freezes_real_public_surfaces_and_fixed_denominator() -> None:
    protocol, _reference, _probes = load_content_uniform_combination_directional_protocol(CONFIG, repository_root=ROOT)
    assert len(protocol.unit_roster) == 41
    source = inspect.getsource(entrypoint.execute_content_uniform_combination_directional_diagnosis_session)
    assert "_replay_verified_whitening_asset" in source
    assert "runner.execute_operational_unit" in source
    assert "runner.execute_reference_fit_unit" in source
    assert "runner.execute_probe_unit" in source
    assert "runner.replay_aggregate" in source
    assert "successful_references" in source
    assert "cursor.routing_reference_records" not in source


def test_server_receipt_contract_preserves_one_thirty_two_eight(monkeypatch, tmp_path: Path) -> None:
    protocol, reference, probes = load_content_uniform_combination_directional_protocol(CONFIG, repository_root=ROOT)
    monkeypatch.setattr(server, "_verify_repository", lambda *_: None)
    monkeypatch.setattr(server, "_probe_resources", lambda **_: {"gpu": "bounded"})
    monkeypatch.setattr(server, "_install_frozen_dependencies", lambda *_: None)
    monkeypatch.setattr(server, "_download_configured_model", lambda **_: None)
    package = tmp_path / "package.zip"
    package.write_bytes(b"package")
    monkeypatch.setattr(server, "_build_or_verify_package", lambda *_: package)
    artifact = tmp_path / "persistent" / protocol.run_id / "session_results" / "session.zip"
    artifact.parent.mkdir(parents=True)
    from zipfile import ZipFile
    with ZipFile(artifact, "w") as target:
        target.writestr("committed_unit_ids.json", b"[]")
    worker = {
        "artifact_kind": "content_uniform_combination_directional_diagnosis_result",
        "diagnostic_zip": str(artifact),
        "protocol_digest": protocol.digest(),
        "reference_manifest_digest": server.canonical_digest(server.asdict(reference)),
        "probe_manifest_digest": server.canonical_digest(server.asdict(probes)),
        "unit_roster_digest": protocol.unit_roster_digest,
        "claim_boundary": protocol.claim_boundary,
        "content_uniform_combination_directional_aggregate": None,
        "termination_reason": "worker_execution_failure",
    }
    monkeypatch.setattr(server, "execute_content_uniform_combination_directional_diagnosis_session", lambda **_: (3, worker))
    runtime = ROOT / "configs/runtime/runtime_sd35_flowmatch.json"
    monkeypatch.setattr(server, "RUNTIME_CONFIG_PATH", runtime.relative_to(ROOT))
    code, receipt = server.execute_content_uniform_combination_directional_diagnosis_server_session(
        repository_root=ROOT,
        expected_revision="9" * 40,
        persistent_root=tmp_path / "persistent",
        whitening_asset_persistent_root=tmp_path / "fit",
        cache_root=tmp_path / "cache",
        run_id=protocol.run_id,
        session_id="combination_delivery_session",
        environment={"HF_TOKEN": "hf_secret", "CEG_WM_ROOT_KEY": "root_secret"},
        install_dependencies=False,
    )
    assert code == 3
    assert (receipt["operational_unit_count"], receipt["reference_fit_cluster_count"], receipt["directional_probe_cluster_count"], receipt["total_unit_count"]) == (1, 32, 8, 41)
    assert receipt["maximum_attempts_per_unit"] == 1
    serialized = str(receipt)
    assert "hf_secret" not in serialized and "root_secret" not in serialized


def test_server_and_worker_do_not_claim_selection_or_formal_threshold() -> None:
    source = inspect.getsource(server) + inspect.getsource(entrypoint)
    assert '"formal_tau_created": False' in source
    assert '"candidate_promoted": False' in source
    assert '"scientific_claims_supported": False' in source
    assert "content_uniform_combination_directional_aggregate" in source
