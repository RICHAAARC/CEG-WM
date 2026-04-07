"""
File purpose: Contract tests for the PW05 notebook entrypoint and parameter wiring.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, cast

from scripts.notebook_runtime_common import REPO_ROOT


NOTEBOOK_PW05_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW05_Release_And_Signoff.ipynb"


def _load_notebook(notebook_path: Path) -> Dict[str, Any]:
    """
    Load one notebook JSON object.

    Args:
        notebook_path: Notebook path.

    Returns:
        Parsed notebook payload.
    """
    notebook_payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not isinstance(notebook_payload, dict):
        raise AssertionError(f"notebook root must be object: {notebook_path}")
    return cast(Dict[str, Any], notebook_payload)


def _find_cell_sources(notebook_path: Path, marker: str, cell_type: str) -> List[str]:
    """
    Find notebook cell sources by marker text and cell type.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.
        cell_type: Notebook cell type.

    Returns:
        Matching cell source texts.
    """
    notebook_payload = _load_notebook(notebook_path)
    cells = notebook_payload.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"cells must be list: {notebook_path}")
    matches: List[str] = []
    for cell_node in cast(List[object], cells):
        if not isinstance(cell_node, dict):
            continue
        if cell_node.get("cell_type") != cell_type:
            continue
        source_node = cell_node.get("source")
        if not isinstance(source_node, list):
            continue
        source_text = "\n".join(str(line) for line in cast(List[object], source_node))
        while "\n\n" in source_text:
            source_text = source_text.replace("\n\n", "\n")
        if marker in source_text:
            matches.append(source_text)
    return matches


def _find_code_cell_source(notebook_path: Path, marker: str) -> str:
    """
    Find one code cell by marker text.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.

    Returns:
        Joined code source text.
    """
    matches = _find_cell_sources(notebook_path, marker, "code")
    if not matches:
        raise AssertionError(f"code cell marker not found: {marker}")
    return matches[0]


def test_pw05_notebook_binds_expected_script_and_parameters() -> None:
    """
    Verify the PW05 notebook binds the real script and required parameters.

    Args:
        None.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(NOTEBOOK_PW05_PATH, "SCRIPT_PATH = REPO_ROOT")
    execute_source = _find_code_cell_source(NOTEBOOK_PW05_PATH, "COMMAND = [")

    assert '"pw05_release_signoff.py"' in constants_source
    assert 'NOTEBOOK_NAME = "PW05_Release_And_Signoff"' in constants_source
    assert 'FORCE_RERUN = False' in constants_source

    assert '"--drive-project-root"' in execute_source
    assert '"--family-id"' in execute_source
    assert 'COMMAND.append("--force-rerun")' in execute_source


def test_pw05_notebook_reads_pw04_inputs_and_pw05_outputs() -> None:
    """
    Verify the PW05 notebook precheck and summary cells read the expected artifacts.

    Args:
        None.

    Returns:
        None.
    """
    precheck_source = _find_code_cell_source(NOTEBOOK_PW05_PATH, "PRECHECK_RESULTS = []")
    summary_source = _find_code_cell_source(NOTEBOOK_PW05_PATH, "PW05_RESULT_SUMMARY = {")

    assert 'FAMILY_MANIFEST_PATH = FAMILY_ROOT / "manifests" / "paper_eval_family_manifest.json"' in precheck_source
    assert 'CONFIG_SNAPSHOT_PATH = FAMILY_ROOT / "snapshots" / "config_snapshot.yaml"' in precheck_source
    assert 'PW04_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw04_summary.json"' in precheck_source
    assert 'PW02_FINALIZE_MANIFEST_PATH = FAMILY_ROOT / "exports" / "pw02" / "paper_source_finalize_manifest.json"' in precheck_source
    assert 'CONTENT_THRESHOLD_EXPORT_PATH = FAMILY_ROOT / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json"' in precheck_source
    assert 'ATTESTATION_THRESHOLD_EXPORT_PATH = FAMILY_ROOT / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json"' in precheck_source
    assert 'PW04 status == completed' in precheck_source
    assert 'PW04 paper exports completed' in precheck_source
    assert 'canonical metrics 输出存在' in precheck_source
    assert 'paper tables 输出存在' in precheck_source
    assert 'paper figures 输出存在' in precheck_source
    assert 'tail estimation 输出存在' in precheck_source
    assert 'PW05_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw05_summary.json"' in precheck_source

    assert 'SIGNOFF_REPORT_PATH = Path(str(PW05_SUMMARY["signoff_report_path"]))' in summary_source
    assert 'RELEASE_MANIFEST_PATH = Path(str(PW05_SUMMARY["release_manifest_path"]))' in summary_source
    assert 'WORKFLOW_SUMMARY_PATH = Path(str(PW05_SUMMARY["workflow_summary_path"]))' in summary_source
    assert 'STAGE_MANIFEST_PATH = Path(str(PW05_SUMMARY["stage_manifest_path"]))' in summary_source
    assert 'PACKAGE_MANIFEST_PATH = Path(str(PW05_SUMMARY["package_manifest_path"]))' in summary_source
    assert 'PACKAGE_PATH = Path(str(PW05_SUMMARY["package_path"]))' in summary_source
    assert '"signoff_report": json.loads(SIGNOFF_REPORT_PATH.read_text(encoding="utf-8"))' in summary_source
    assert '"release_manifest": json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))' in summary_source
    assert '"package_exists": PACKAGE_PATH.exists()' in summary_source