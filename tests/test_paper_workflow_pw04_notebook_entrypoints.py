"""
File purpose: Contract tests for the PW04 notebook entrypoint and parameter wiring.
Module type: General module
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, cast

import pytest

from scripts.notebook_runtime_common import REPO_ROOT


NOTEBOOK_PW04_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW04_Attack_Merge_And_Metrics.ipynb"


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


def test_pw04_notebook_binds_expected_script_and_parameters() -> None:
    """
    Verify the PW04 notebook binds the real script and required parameters.

    Args:
        None.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(NOTEBOOK_PW04_PATH, "SCRIPT_PATH = REPO_ROOT")
    bootstrap_source = _find_code_cell_source(NOTEBOOK_PW04_PATH, "from scripts.notebook_runtime_common import resolve_notebook_model_cache_layout")
    execute_source = _find_code_cell_source(NOTEBOOK_PW04_PATH, "COMMAND = [")

    assert '"PW04_Attack_Merge_And_Metrics.py"' in constants_source
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in constants_source
    assert 'LOCAL_HF_HUB_CACHE = LOCAL_HF_HOME / "hub"' in constants_source
    assert 'LOCAL_TRANSFORMERS_CACHE = LOCAL_HF_HOME / "transformers"' in constants_source
    assert 'FORCE_RERUN = False' in constants_source
    assert 'ENABLE_TAIL_ESTIMATION = False' in constants_source

    assert 'os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_HUB_CACHE)' in bootstrap_source
    assert 'snapshot_download(' not in bootstrap_source
    assert 'bootstrap_notebook_model_cache(' not in bootstrap_source

    assert '"--drive-project-root"' in execute_source
    assert '"--family-id"' in execute_source
    assert '"--force-rerun"' in execute_source
    assert '"--enable-tail-estimation"' in execute_source


def test_pw04_notebook_reads_pw02_inputs_and_pw04_outputs() -> None:
    """
    Verify the PW04 notebook precheck and summary cells read the expected artifacts.

    Args:
        None.

    Returns:
        None.
    """
    precheck_source = _find_code_cell_source(NOTEBOOK_PW04_PATH, "PRECHECK_RESULTS = []")
    summary_source = _find_code_cell_source(NOTEBOOK_PW04_PATH, "PW04_RESULT_SUMMARY = {")

    assert 'PW02_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw02_summary.json"' in precheck_source
    assert 'FINALIZE_MANIFEST_PATH = FAMILY_ROOT / "exports" / "pw02" / "paper_source_finalize_manifest.json"' in precheck_source
    assert 'ATTACK_SHARD_PLAN_PATH = FAMILY_ROOT / "manifests" / "attack_shard_plan.json"' in precheck_source
    assert 'CONTENT_THRESHOLD_EXPORT_PATH = FAMILY_ROOT / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json"' in precheck_source
    assert 'ATTESTATION_THRESHOLD_EXPORT_PATH = FAMILY_ROOT / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json"' in precheck_source
    assert '所有计划内 PW03 shard manifest 存在且 completed' in precheck_source
    assert 'expected_attack_event_count == discovered_attack_event_count' in precheck_source

    assert 'PW04_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw04_summary.json"' in precheck_source
    assert 'FORMAL_ATTACK_FINAL_DECISION_METRICS_PATH = FAMILY_ROOT / "exports" / "pw04" / "formal_attack_final_decision_metrics.json"' in summary_source
    assert 'FORMAL_ATTACK_ATTESTATION_METRICS_PATH = FAMILY_ROOT / "exports" / "pw04" / "formal_attack_attestation_metrics.json"' in summary_source
    assert 'DERIVED_ATTACK_UNION_METRICS_PATH = FAMILY_ROOT / "exports" / "pw04" / "derived_attack_union_metrics.json"' in summary_source
    assert 'CLEAN_ATTACK_OVERVIEW_PATH = FAMILY_ROOT / "exports" / "pw04" / "clean_attack_overview.json"' in summary_source
    assert 'PAPER_SCOPE_REGISTRY_PATH = Path(str(PW04_SUMMARY["paper_scope_registry_path"]))' in summary_source
    assert 'CANONICAL_METRICS_PATHS = dict(PW04_SUMMARY.get("canonical_metrics_paths", {}))' in summary_source
    assert 'PAPER_TABLE_PATHS = dict(PW04_SUMMARY.get("paper_tables_paths", {}))' in summary_source
    assert 'PAPER_FIGURE_PATHS = dict(PW04_SUMMARY.get("paper_figures_paths", {}))' in summary_source
    assert 'TAIL_ESTIMATION_PATHS = dict(PW04_SUMMARY.get("tail_estimation_paths", {}))' in summary_source
    assert 'BOOTSTRAP_CONFIDENCE_INTERVALS_PATH = Path(str(PW04_SUMMARY["bootstrap_confidence_intervals_path"]))' in summary_source
    assert '"paper_metric_registry": json.loads(PAPER_SCOPE_REGISTRY_PATH.read_text(encoding="utf-8"))' in summary_source
    assert '"content_chain_metrics": json.loads(Path(str(CANONICAL_METRICS_PATHS["content_chain"])).read_text(encoding="utf-8"))' in summary_source
    assert '"paper_figures_paths": {' in summary_source
    assert '"estimated_tail_fpr_1e4": json.loads(Path(str(TAIL_ESTIMATION_PATHS["estimated_tail_fpr_1e4_path"])).read_text(encoding="utf-8"))' in summary_source
    assert 'matplotlib' not in summary_source
    assert 'np.random' not in summary_source


def test_pw04_wrapper_delegates_to_run_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Verify the PW04 wrapper only parses args and delegates to the run function.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    wrapper_module = importlib.import_module("paper_workflow.scripts.PW04_Attack_Merge_And_Metrics")
    captured: Dict[str, Any] = {}

    def fake_run_pw04_merge_attack_event_shards(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(wrapper_module, "run_pw04_merge_attack_event_shards", fake_run_pw04_merge_attack_event_shards)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "PW04_Attack_Merge_And_Metrics.py",
            "--drive-project-root",
            str(tmp_path / "drive_root"),
            "--family-id",
            "family_pw04_demo",
            "--force-rerun",
            "--enable-tail-estimation",
        ],
    )

    assert wrapper_module.main() == 0
    assert captured["drive_project_root"] == tmp_path / "drive_root"
    assert captured["family_id"] == "family_pw04_demo"
    assert captured["force_rerun"] is True
    assert captured["enable_tail_estimation"] is True
