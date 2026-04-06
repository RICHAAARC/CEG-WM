"""
File purpose: Contract tests for the PW04 notebook entrypoint and parameter wiring.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, cast

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

    assert '"pw04_merge_attack_event_shards.py"' in constants_source
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in constants_source
    assert 'LOCAL_HF_HUB_CACHE = LOCAL_HF_HOME / "hub"' in constants_source
    assert 'LOCAL_TRANSFORMERS_CACHE = LOCAL_HF_HOME / "transformers"' in constants_source
    assert 'FORCE_RERUN = False' in constants_source

    assert 'os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_HUB_CACHE)' in bootstrap_source
    assert 'snapshot_download(' not in bootstrap_source
    assert 'bootstrap_notebook_model_cache(' not in bootstrap_source

    assert '"--drive-project-root"' in execute_source
    assert '"--family-id"' in execute_source
    assert '"--force-rerun"' in execute_source


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
