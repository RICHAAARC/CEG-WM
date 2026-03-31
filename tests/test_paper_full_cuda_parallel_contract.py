"""
文件目的：验证 stage 01 prompt-level 并行路径的最小合同。
Module type: General module

职责边界：
1. 仅覆盖稳定分片、worker 结果载荷与父调度聚合顺序。
2. 不执行真实 GPU workflow，不依赖 Colab 或 Google Drive。
3. 不修改既有单路线 formal contract，只验证新增并行入口的边界行为。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PARALLEL_RUNNER_PATH = REPO_ROOT / "scripts" / "01_run_paper_full_cuda_parallel.py"
PARALLEL_WORKER_PATH = REPO_ROOT / "scripts" / "01_run_paper_full_cuda_parallel_worker.py"


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    """
    功能：按文件路径加载测试目标模块。 

    Load one target module from a file path.

    Args:
        module_path: Module file path.
        module_name: Synthetic module name used by importlib.

    Returns:
        Loaded module object.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARALLEL_RUNNER = _load_module(PARALLEL_RUNNER_PATH, "test_parallel_stage_01_runner")
PARALLEL_WORKER = _load_module(PARALLEL_WORKER_PATH, "test_parallel_stage_01_worker")


def _make_direct_entry(prompt_index: int) -> dict:
    """
    功能：构造最小 direct entry 测试载荷。 

    Build the minimal direct-entry payload used by the contract tests.

    Args:
        prompt_index: Prompt index.

    Returns:
        Minimal direct-entry mapping.
    """
    return {
        "prompt_index": prompt_index,
        "prompt_text": f"prompt-{prompt_index}",
        "package_relative_path": f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json",
    }


def _make_stage_result(prompt_index: int) -> dict:
    """
    功能：构造最小 stage result 测试载荷。 

    Build the minimal stage-result payload used by the contract tests.

    Args:
        prompt_index: Prompt index.

    Returns:
        Minimal stage-result mapping.
    """
    return {
        "prompt_index": prompt_index,
        "prompt_text": f"prompt-{prompt_index}",
        "stage_results": {"embed": {"return_code": 0}, "detect": {"return_code": 0}},
        "package_relative_path": f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json",
    }


def test_parallel_worker_assignments_use_stable_modulo_rule() -> None:
    """
    功能：验证并行分片严格遵循 prompt_index 取模规则。 

    Verify that the parallel shard plan follows the stable modulo rule.

    Args:
        None.

    Returns:
        None.
    """
    assignments = PARALLEL_RUNNER._build_worker_assignments(prompt_count=7, worker_count=2)

    assert assignments == [
        {"worker_index": 0, "assigned_prompt_indices": [0, 2, 4, 6]},
        {"worker_index": 1, "assigned_prompt_indices": [1, 3, 5]},
    ]


def test_parallel_worker_payload_records_shard_identity() -> None:
    """
    功能：验证 worker 结果载荷显式记录分片身份与完成索引。 

    Verify that one worker result payload records shard identity and completed
    prompt indices explicitly.

    Args:
        None.

    Returns:
        None.
    """
    payload = PARALLEL_WORKER._build_worker_result_payload(
        stage_run_id="stage_01_parallel_test",
        prompt_file_path="prompts/paper_full.txt",
        worker_index=1,
        worker_count=2,
        assigned_prompt_indices=[1, 3],
        direct_entries=[_make_direct_entry(1), _make_direct_entry(3)],
        source_pool_stage_results=[_make_stage_result(1), _make_stage_result(3)],
        status="ok",
    )

    assert payload["stage_name"] == "01_Paper_Full_Cuda_Parallel_mainline"
    assert payload["worker_index"] == 1
    assert payload["worker_count"] == 2
    assert payload["assigned_prompt_indices"] == [1, 3]
    assert payload["completed_prompt_indices"] == [1, 3]
    assert payload["direct_entry_count"] == 2
    assert payload["source_pool_stage_result_count"] == 2


def test_parallel_merge_worker_results_restores_prompt_order() -> None:
    """
    功能：验证父调度聚合后恢复原始 prompt 索引顺序。 

    Verify that the parent orchestrator restores the original prompt-index
    order after merging all worker results.

    Args:
        None.

    Returns:
        None.
    """
    worker_results = [
        PARALLEL_WORKER._build_worker_result_payload(
            stage_run_id="stage_01_parallel_test",
            prompt_file_path="prompts/paper_full.txt",
            worker_index=1,
            worker_count=2,
            assigned_prompt_indices=[1, 3],
            direct_entries=[_make_direct_entry(3), _make_direct_entry(1)],
            source_pool_stage_results=[_make_stage_result(3), _make_stage_result(1)],
            status="ok",
        ),
        PARALLEL_WORKER._build_worker_result_payload(
            stage_run_id="stage_01_parallel_test",
            prompt_file_path="prompts/paper_full.txt",
            worker_index=0,
            worker_count=2,
            assigned_prompt_indices=[0, 2],
            direct_entries=[_make_direct_entry(2), _make_direct_entry(0)],
            source_pool_stage_results=[_make_stage_result(2), _make_stage_result(0)],
            status="ok",
        ),
    ]

    direct_entries, source_pool_stage_results = PARALLEL_RUNNER._merge_worker_results(
        worker_results,
        prompt_count=4,
    )

    assert [entry["prompt_index"] for entry in direct_entries] == [0, 1, 2, 3]
    assert [entry["prompt_index"] for entry in source_pool_stage_results] == [0, 1, 2, 3]


def test_parallel_merge_rejects_duplicate_prompt_indices() -> None:
    """
    功能：验证聚合阶段会拒绝重复 prompt 索引。 

    Verify that the merge step rejects duplicated prompt indices.

    Args:
        None.

    Returns:
        None.
    """
    duplicate_results = [
        PARALLEL_WORKER._build_worker_result_payload(
            stage_run_id="stage_01_parallel_test",
            prompt_file_path="prompts/paper_full.txt",
            worker_index=0,
            worker_count=2,
            assigned_prompt_indices=[0],
            direct_entries=[_make_direct_entry(0)],
            source_pool_stage_results=[_make_stage_result(0)],
            status="ok",
        ),
        PARALLEL_WORKER._build_worker_result_payload(
            stage_run_id="stage_01_parallel_test",
            prompt_file_path="prompts/paper_full.txt",
            worker_index=1,
            worker_count=2,
            assigned_prompt_indices=[0],
            direct_entries=[_make_direct_entry(0)],
            source_pool_stage_results=[_make_stage_result(0)],
            status="ok",
        ),
    ]

    with pytest.raises(ValueError, match="duplicate direct entry"):
        PARALLEL_RUNNER._merge_worker_results(duplicate_results, prompt_count=1)