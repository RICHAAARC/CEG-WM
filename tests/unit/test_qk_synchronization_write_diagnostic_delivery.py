"""Delivery boundary tests for Q/K synchronization-write diagnosis."""

from __future__ import annotations

import ast
import json
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
NOTEBOOK = ROOT / "notebooks/colab/qk_synchronization_write_diagnostic.ipynb"
EXECUTION_REVISION = "1c80ee84cadfc73744ddbcdb48b45787ee7c44e2"
RUN_ID = "ceg_wm_qk_synchronization_write_public_rgb8_diagnosis"
HISTORICAL_RUN_ID = "ceg_wm_qk_synchronization_write_diagnosis"


def _notebook_source() -> tuple[dict[str, object], str]:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return document, "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )


def _constant(source: str, name: str):
    tree = ast.parse(source)
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
        and isinstance(node.value, ast.Constant)
    ]
    assert len(matches) == 1
    return matches[0].value


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
    assert protocol.run_id == RUN_ID
    assert protocol.operational_unit_count == 1
    assert protocol.scientific_unit_count == 28
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


@pytest.mark.quick
def test_qk_diagnosis_notebook_is_thin_exact_and_output_free() -> None:
    document, source = _notebook_source()
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )

    assert len(document["cells"]) == 6
    assert all(
        cell.get("execution_count") is None
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(cell.get("outputs", []) == [] for cell in document["cells"])
    assert _constant(code_source, "EXECUTION_REVISION") == EXECUTION_REVISION
    assert _constant(code_source, "RUN_ID") == RUN_ID
    assert "qk_synchronization_write_diagnostic_server.py" in code_source
    assert "HF_TOKEN" in code_source and "CEG_WM_ROOT_KEY" in code_source
    assert "/content/drive" in code_source
    assert "--expected-revision" in code_source
    assert "--persistent-root" in code_source
    assert "--cache-root" in code_source
    assert "--run-id" in code_source
    assert "--session-id" in code_source
    assert "qk_synchronization_diagnosis_aggregate" in code_source
    for forbidden in (
        "geometry_synchronization_write(",
        "create_qk_ratio_probe_observation(",
        "DevelopmentScientificRecord(",
        "commit_session_unit(",
        "replay_synchronization_diagnosis_aggregate(",
        "fit_threshold(",
        "evaluate_qk_synchronization_write_diagnosis(",
    ):
        assert forbidden not in source


@pytest.mark.quick
def test_qk_diagnosis_readmes_preserve_historical_run_boundary() -> None:
    for path in (
        ROOT / "notebooks/colab/README.md",
        ROOT / "scripts/experiment_execution/README.md",
    ):
        source = path.read_text("utf-8")
        normalized_source = " ".join(source.split())
        assert NOTEBOOK.name in source
        assert EXECUTION_REVISION in source
        assert RUN_ID in source
        assert HISTORICAL_RUN_ID in source
        assert "records、diagnostics 与 intents 保持不可变" in normalized_source
        assert "不读取、迁移、覆盖或混入" in normalized_source
