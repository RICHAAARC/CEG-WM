"""
File purpose: Contract tests for the PW03 notebook entrypoint and parameter wiring.
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
    bootstrap_source = _find_code_cell_source(
        NOTEBOOK_PW03_PATH,
        'MODEL_CACHE_BOOTSTRAP = bootstrap_notebook_model_cache(',
    )
    execute_source = _find_code_cell_source(NOTEBOOK_PW03_PATH, "COMMAND = [")

    assert '"PW03_Attack_Event_Shards.py"' in constants_source
    assert 'DRIVE_MODELS_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "Models"' in constants_source
    assert 'PERSISTENT_INSPYRENET_ROOT = DRIVE_MODELS_ROOT / "inspyrenet"' in constants_source
    assert 'PERSISTENT_HF_ROOT = DRIVE_MODELS_ROOT / "Huggingface"' in constants_source
    assert 'LOCAL_HF_HOME = REPO_ROOT / "huggingface_cache"' in constants_source
    assert 'ATTACK_SHARD_INDEX = 0' in constants_source
    assert 'ATTACK_SHARD_COUNT = 16' in constants_source
    assert 'ATTACK_LOCAL_WORKER_COUNT =' in constants_source
    assert 'ATTACK_FAMILY_ALLOWLIST = None' in constants_source
    assert 'PW00 family 冻结的 attack_shard_count' in constants_source
    assert '当前允许 1、2、3 或 4' in constants_source
    assert 'LOCAL_RUNTIME_ENABLED = True' in constants_source
    assert 'LOCAL_PROJECT_ROOT = Path("/content/CEG_WM_PaperWorkflow")' in constants_source
    assert 'DRIVE_BUNDLE_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_PaperWorkflow_Bundles"' in constants_source
    assert '必须与 PW00 和 PW01 一致' not in constants_source
    assert '当前只允许 1 或 2' not in constants_source

    assert 'from huggingface_hub import HfApi' in bootstrap_source
    assert 'bootstrap_notebook_model_cache' in bootstrap_source
    assert 'os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_HUB_CACHE)' in bootstrap_source
    assert 'MODEL_SNAPSHOT_PATH = str(MODEL_CACHE_BOOTSTRAP["local_snapshot_path"])' in bootstrap_source
    assert 'MODEL_DOWNLOAD_SUMMARY = dict(MODEL_CACHE_BOOTSTRAP["model_audit_summary"])' in bootstrap_source
    assert '"snapshot_source": MODEL_CACHE_BOOTSTRAP["snapshot_source"]' in bootstrap_source
    assert '"model_source_binding": MODEL_CACHE_BOOTSTRAP["model_source_binding"]' in bootstrap_source
    assert 'print_json("model_cache_bootstrap", MODEL_CACHE_BOOTSTRAP)' in bootstrap_source

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
    assert 'PW03_PREFLIGHT = detect_pw03_preflight(PW03_BOUND_CONFIG_PATH)' in precheck_source
    assert 'record_precheck("PW03 preflight"' in precheck_source
    assert '"pw03_preflight": PW03_PREFLIGHT' in precheck_source
    assert 'detect_stage_01_preflight' not in precheck_source
    assert 'persistent Huggingface 路径仅兼容保留' in precheck_source
    assert '模型 snapshot 来源为本地会话缓存' in precheck_source
    assert 'str(Path(str(MODEL_SNAPSHOT_PATH)).resolve()).startswith(str(LOCAL_HF_HOME.resolve()))' in precheck_source
    assert 'ATTACK_LOCAL_WORKER_COUNT in {1, 2, 3, 4}' in precheck_source
    assert 'family 冻结 attack_shard_count 与 ATTACK_SHARD_COUNT 一致' in precheck_source
    assert 'CURRENT_SHARD_EVENT_IDS = []' in precheck_source
    assert 'CURRENT_SHARD_PLAN = next(row for row in ATTACK_SHARD_PLAN["shards"] if row["attack_shard_index"] == ATTACK_SHARD_INDEX)' in precheck_source
    assert 'MODEL_DOWNLOAD_SUMMARY["binding_status"] = PRECHECK_MODEL_SOURCE_BINDING.get("binding_status", "<absent>")' in precheck_source

    assert 'SHARD_ROOT = FAMILY_ROOT / "attack_shards" / f"shard_{ATTACK_SHARD_INDEX:04d}"' in execute_source
    assert 'PW03_SHARD_MANIFEST_PATH = SHARD_ROOT / "shard_manifest.json"' in execute_source
    assert 'GPU_PEAK_SUMMARY_PATH = SHARD_ROOT / "artifacts" / "gpu_session_peak.json"' in execute_source
    assert 'PW03_SUMMARY = json.loads(PW03_SHARD_MANIFEST_PATH.read_text(encoding="utf-8"))' in execute_source
    assert 'GPU_PEAK_SUMMARY = json.loads(GPU_PEAK_SUMMARY_PATH.read_text(encoding="utf-8"))' in execute_source
    assert 'print_json("gpu_session_peak_summary", GPU_PEAK_NOTEBOOK_SUMMARY)' in execute_source

    assert '"total_event_count": PW03_SUMMARY.get("event_count")' in summary_source
    assert '"completed_event_count": PW03_SUMMARY.get("completed_event_count")' in summary_source
    assert '"failed_event_count": PW03_SUMMARY.get("failed_event_count")' in summary_source
    assert '"model_snapshot_path": str(MODEL_SNAPSHOT_PATH)' in summary_source
    assert '"persistent_inspyrenet_path": str(PERSISTENT_WEIGHT_PATH)' in summary_source
    assert '"repo_inspyrenet_path": str(WEIGHT_PATH)' in summary_source
    assert '"snapshot_source": MODEL_CACHE_BOOTSTRAP.get("snapshot_source", "<absent>")' in summary_source
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
    assert "控制 1/2/3/4 路 local worker" in parallel_markdown
    assert "每个 local worker 的独立日志与结果只写入当前 shard 的 workers/worker_XX" in parallel_markdown

    assert '"worker_mode": "single_process" if ATTACK_LOCAL_WORKER_COUNT == 1 else "shard_local_subprocess_parallel"' in parallel_source
    assert '"--bound-config-path"' in parallel_source
    assert 'str(PW03_BOUND_CONFIG_PATH)' in parallel_source
    assert 'str(shard_root / "workers" / f"worker_{local_worker_index:02d}")' in parallel_source


def test_pw03_wrapper_delegates_to_run_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Verify the PW03 wrapper only parses args and delegates to the run function.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    wrapper_module = importlib.import_module("paper_workflow.scripts.PW03_Attack_Event_Shards")
    captured: Dict[str, Any] = {}

    def fake_run_pw03_attack_event_shard(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(wrapper_module, "run_pw03_attack_event_shard", fake_run_pw03_attack_event_shard)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "PW03_Attack_Event_Shards.py",
            "--drive-project-root",
            str(tmp_path / "drive_root"),
            "--family-id",
            "family_pw03_demo",
            "--attack-shard-index",
            "1",
            "--attack-shard-count",
            "4",
            "--attack-local-worker-count",
            "2",
            "--attack-family-allowlist",
            '["jpeg"]',
            "--bound-config-path",
            str(tmp_path / "bound_config.yaml"),
            "--force-rerun",
        ],
    )

    assert wrapper_module.main() == 0
    assert captured["drive_project_root"] == tmp_path / "drive_root"
    assert captured["family_id"] == "family_pw03_demo"
    assert captured["attack_shard_index"] == 1
    assert captured["attack_shard_count"] == 4
    assert captured["attack_local_worker_count"] == 2
    assert captured["attack_family_allowlist"] == '["jpeg"]'
    assert captured["bound_config_path"] == tmp_path / "bound_config.yaml"
    assert captured["force_rerun"] is True