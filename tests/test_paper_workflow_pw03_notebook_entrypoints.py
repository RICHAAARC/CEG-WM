"""
File purpose: Contract tests for the PW03 notebook entrypoint and parameter wiring.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, cast

from scripts.notebook_runtime_common import REPO_ROOT


NOTEBOOK_PW03_PATH = REPO_ROOT / "paper_workflow" / "notebook" / "PW03_Attack_Event_Shards.ipynb"


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


def _find_markdown_cell_source(notebook_path: Path, marker: str) -> str:
    """
    Find one markdown cell by marker text.

    Args:
        notebook_path: Notebook path.
        marker: Marker text.

    Returns:
        Joined markdown source text.
    """
    matches = _find_cell_sources(notebook_path, marker, "markdown")
    if not matches:
        raise AssertionError(f"markdown cell marker not found: {marker}")
    return matches[0]


def test_pw03_notebook_binds_expected_script_and_parameters() -> None:
    """
    Verify the PW03 notebook binds the real script and required parameters.

    Args:
        None.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(NOTEBOOK_PW03_PATH, "SCRIPT_PATH = REPO_ROOT")
    execute_source = _find_code_cell_source(NOTEBOOK_PW03_PATH, "COMMAND = [")

    assert '"pw03_run_attack_event_shard.py"' in constants_source
    assert 'ATTACK_SHARD_INDEX = 0' in constants_source
    assert 'ATTACK_SHARD_COUNT = 2' in constants_source
    assert 'ATTACK_LOCAL_WORKER_COUNT = 2' in constants_source
    assert 'ATTACK_FAMILY_ALLOWLIST = None' in constants_source

    assert '"--drive-project-root"' in execute_source
    assert '"--family-id"' in execute_source
    assert '"--attack-shard-index"' in execute_source
    assert '"--attack-shard-count"' in execute_source
    assert '"--attack-local-worker-count"' in execute_source
    assert '"--bound-config-path"' in execute_source
    assert 'str(PW03_BOUND_CONFIG_PATH)' in execute_source
    assert 'ATTACK_FAMILY_ALLOWLIST' in execute_source
    assert '"--force-rerun"' in execute_source


def test_pw03_notebook_reads_pw02_finalize_inputs_and_shard_outputs() -> None:
    """
    Verify the PW03 notebook precheck and execute cells read the expected artifacts.

    Args:
        None.

    Returns:
        None.
    """
    precheck_source = _find_code_cell_source(NOTEBOOK_PW03_PATH, "PRECHECK_RESULTS = []")
    execute_source = _find_code_cell_source(NOTEBOOK_PW03_PATH, "COMMAND = [")
    summary_source = _find_code_cell_source(NOTEBOOK_PW03_PATH, "PW03_RESULT_SUMMARY = {")

    assert 'PW02_SUMMARY_PATH = FAMILY_ROOT / "runtime_state" / "pw02_summary.json"' in precheck_source
    assert 'FINALIZE_MANIFEST_PATH = None' in precheck_source
    assert 'PW03_BOUND_CONFIG_PATH = PRECHECK_BOUND_CONFIG_PATH' in precheck_source
    assert 'write_yaml_mapping(PW03_BOUND_CONFIG_PATH, PRECHECK_BOUND_CFG)' in precheck_source
    assert 'STAGE_01_PREFLIGHT = detect_stage_01_preflight(PW03_BOUND_CONFIG_PATH)' in precheck_source
    assert 'CURRENT_SHARD_EVENT_IDS = []' in precheck_source
    assert 'CURRENT_SHARD_PLAN = next(row for row in ATTACK_SHARD_PLAN["shards"] if row["attack_shard_index"] == ATTACK_SHARD_INDEX)' in precheck_source

    assert 'SHARD_ROOT = FAMILY_ROOT / "attack_shards" / f"shard_{ATTACK_SHARD_INDEX:04d}"' in execute_source
    assert 'PW03_SHARD_MANIFEST_PATH = SHARD_ROOT / "shard_manifest.json"' in execute_source
    assert 'GPU_PEAK_SUMMARY_PATH = SHARD_ROOT / "artifacts" / "gpu_session_peak.json"' in execute_source
    assert 'PW03_SUMMARY = json.loads(PW03_SHARD_MANIFEST_PATH.read_text(encoding="utf-8"))' in execute_source
    assert 'GPU_PEAK_SUMMARY = json.loads(GPU_PEAK_SUMMARY_PATH.read_text(encoding="utf-8"))' in execute_source
    assert 'print_json("gpu_session_peak_summary", GPU_PEAK_NOTEBOOK_SUMMARY)' in execute_source

    assert '"total_event_count": PW03_SUMMARY.get("event_count")' in summary_source
    assert '"completed_event_count": PW03_SUMMARY.get("completed_event_count")' in summary_source
    assert '"failed_event_count": PW03_SUMMARY.get("failed_event_count")' in summary_source
    assert '"gpu_peak_memory_mib": GPU_PEAK_SUMMARY.get("peak_memory_mib")' in summary_source
    assert '"wrapped_command_count": GPU_PEAK_SUMMARY.get("wrapped_command_count")' in summary_source


def test_pw03_notebook_parallel_plan_explains_isolation_and_worker_layout() -> None:
    """
    Verify the PW03 notebook explains shard isolation and worker-local layout.

    Args:
        None.

    Returns:
        None.
    """
    intro_markdown = _find_markdown_cell_source(NOTEBOOK_PW03_PATH, "用途：")
    parallel_markdown = _find_markdown_cell_source(NOTEBOOK_PW03_PATH, "扩展规则：")
    parallel_source = _find_code_cell_source(NOTEBOOK_PW03_PATH, "parallel_plan = []")

    assert "只消费 PW02 finalized positive source pool" in intro_markdown
    assert "不重新生成 clean 正样本" in intro_markdown
    assert "每个 shard 只写入 attack_shards/shard_xxxx" in parallel_markdown
    assert "双 worker 的独立日志与结果只写入当前 shard 的 workers/worker_XX" in parallel_markdown

    assert '"worker_mode": "single_process" if ATTACK_LOCAL_WORKER_COUNT == 1 else "shard_local_subprocess_parallel"' in parallel_source
    assert '"--bound-config-path"' in parallel_source
    assert 'str(PW03_BOUND_CONFIG_PATH)' in parallel_source
    assert 'str(shard_root / "workers" / f"worker_{local_worker_index:02d}")' in parallel_source