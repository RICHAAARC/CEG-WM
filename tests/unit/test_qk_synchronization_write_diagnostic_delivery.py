"""Delivery boundary tests for Q/K synchronization-write diagnosis."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.protocol.qk_synchronization_write_diagnostic import (
    CLAIM_BOUNDARY,
    load_qk_synchronization_write_protocol,
)
from experiments.runners.qk_synchronization_write_diagnostic import RGB8_MEMBER_PATH
from scripts.experiment_execution.qk_synchronization_write_diagnostic_server import main


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/qk_synchronization_write_diagnostic.json"
ENTRYPOINT = ROOT / "scripts/experiment_execution/qk_synchronization_write_diagnostic_entrypoint.py"
SERVER = ROOT / "scripts/experiment_execution/qk_synchronization_write_diagnostic_server.py"


@pytest.mark.quick
def test_qk_diagnosis_server_help_imports_from_isolated_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SERVER), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-revision" in completed.stdout
    assert "--persistent-root" in completed.stdout
    assert "--cache-root" in completed.stdout


@pytest.mark.quick
def test_qk_diagnosis_delivery_binds_production_entrypoint_and_public_rgb8_member() -> None:
    protocol, _manifest = load_qk_synchronization_write_protocol(
        PROTOCOL, repository_root=ROOT
    )
    source = ENTRYPOINT.read_text("utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert "execute_qk_synchronization_write_diagnostic_session" in source
    assert "create_session_intent" in calls
    assert "commit_session_unit" in calls
    assert "verified_terminal_scientific_evidence" in calls
    assert "execute_scientific_unit" in calls
    assert "create_dependency_blocked_record" in calls
    assert "RGB8_MEMBER_PATH" in source
    assert protocol.maximum_total_units == 29
    assert protocol.ratio_probe_unit_count == 12
    assert protocol.transform_probe_unit_count == 16
    assert protocol.claim_boundary == CLAIM_BOUNDARY
    for forbidden in (
        "content_router",
        "reference_image",
        "private_qk_cache",
        "synthetic_gradient",
        "precomputed_score",
    ):
        assert forbidden not in source


@pytest.mark.quick
def test_qk_diagnosis_server_receipt_boundary_is_non_scientific(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "--run-id" in output
    assert "--session-id" in output
    server_source = SERVER.read_text("utf-8")
    assert '"scientific_claims_supported": False' in server_source
    assert '"formal_tau_created": False' in server_source
    assert '"fpr_estimated": False' in server_source
    assert '"candidate_promoted": False' in server_source
