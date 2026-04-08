"""
文件目的：执行 stage 01 prompt-level 并行分片 worker。
Module type: General module

职责边界：
1. 只负责按照稳定分片规则执行 source pool prompt 子任务。
2. 不负责 pooled threshold、calibrate、evaluate 或 package 收口。
3. 结果通过独立 worker JSON 交给父调度器聚合，避免污染单路线主链。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence

from scripts.notebook_runtime_common import ensure_directory, load_yaml_mapping, normalize_path_value, resolve_repo_path, write_json_atomic


BASE_RUNNER_SCRIPT_PATH = Path("scripts/01_run_paper_full_cuda.py")
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
WORKER_RESULT_RELATIVE_ROOT = "artifacts/stage_01_parallel_workers"
PARALLEL_STAGE_NAME = "01_Paper_Full_Cuda_Parallel_mainline"


def _load_base_runner_module() -> ModuleType:
    """
    功能：按文件路径加载 baseline stage 01 runner。 

    Load the baseline stage-01 runner module from its file path.

    Args:
        None.

    Returns:
        Loaded module object.

    Raises:
        RuntimeError: If the runner module cannot be loaded.
    """
    runner_path = (Path(__file__).resolve().parents[1] / BASE_RUNNER_SCRIPT_PATH).resolve()
    spec = importlib.util.spec_from_file_location("stage_01_paper_full_cuda_base_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load stage-01 runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_RUNNER_MODULE = _load_base_runner_module()


def _validate_worker_bounds(worker_index: int, worker_count: int) -> None:
    """
    功能：校验 worker 下标与总数边界。 

    Validate the worker index and worker count bounds.

    Args:
        worker_index: Zero-based worker index.
        worker_count: Total worker count.

    Returns:
        None.

    Raises:
        TypeError: If the bounds are invalid.
    """
    if worker_count <= 0:
        raise TypeError("worker_count must be positive int")
    if worker_index < 0 or worker_index >= worker_count:
        raise TypeError("worker_index must satisfy 0 <= worker_index < worker_count")


def _assigned_prompt_indices(prompt_count: int, worker_index: int, worker_count: int) -> List[int]:
    """
    功能：按稳定取模规则计算 worker 分片。 

    Compute the stable modulo-based prompt shard for one worker.

    Args:
        prompt_count: Total prompt count.
        worker_index: Zero-based worker index.
        worker_count: Total worker count.

    Returns:
        Assigned prompt indices in ascending order.
    """
    if prompt_count < 0:
        raise TypeError("prompt_count must be non-negative int")
    _validate_worker_bounds(worker_index, worker_count)
    return [prompt_index for prompt_index in range(prompt_count) if prompt_index % worker_count == worker_index]


def _worker_result_path(run_root: Path, worker_index: int) -> Path:
    """
    功能：构造单个 worker 结果文件路径。 

    Build the result-file path for one worker.

    Args:
        run_root: Stage run root.
        worker_index: Zero-based worker index.

    Returns:
        Result JSON path.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if worker_index < 0:
        raise TypeError("worker_index must be non-negative int")
    return run_root / WORKER_RESULT_RELATIVE_ROOT / f"worker_{worker_index:02d}_result.json"


def _build_worker_result_payload(
    *,
    stage_run_id: Optional[str],
    prompt_file_path: str,
    worker_index: int,
    worker_count: int,
    assigned_prompt_indices: Sequence[int],
    direct_entries: Sequence[Dict[str, Any]],
    source_pool_stage_results: Sequence[Dict[str, Any]],
    status: str,
    failure_reason: Optional[str] = None,
    exception_type: Optional[str] = None,
    exception_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    功能：构造 worker 执行结果载荷。 

    Build the structured result payload for one parallel worker.

    Args:
        stage_run_id: External stage run identifier.
        prompt_file_path: Normalized prompt file path.
        worker_index: Zero-based worker index.
        worker_count: Total worker count.
        assigned_prompt_indices: Prompt indices assigned to the worker.
        direct_entries: Completed direct source entries.
        source_pool_stage_results: Completed source-pool stage results.
        status: Worker status string.
        failure_reason: Optional failure reason.
        exception_type: Optional exception type.
        exception_message: Optional exception message.

    Returns:
        JSON-serializable result payload.
    """
    _validate_worker_bounds(worker_index, worker_count)
    if status not in {"ok", "failed"}:
        raise ValueError("status must be 'ok' or 'failed'")
    return {
        "artifact_type": "stage_01_parallel_worker_result",
        "stage_name": PARALLEL_STAGE_NAME,
        "stage_run_id": stage_run_id,
        "prompt_file": prompt_file_path,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "status": status,
        "assigned_prompt_indices": [int(prompt_index) for prompt_index in assigned_prompt_indices],
        "completed_prompt_indices": [int(entry["prompt_index"]) for entry in direct_entries],
        "direct_entry_count": len(direct_entries),
        "source_pool_stage_result_count": len(source_pool_stage_results),
        "direct_entries": list(direct_entries),
        "source_pool_stage_results": list(source_pool_stage_results),
        "failure_reason": failure_reason,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }


def run_parallel_worker(
    config_path: Path,
    run_root: Path,
    *,
    stage_run_id: Optional[str],
    worker_index: int,
    worker_count: int,
) -> int:
    """
    功能：执行单个 stage 01 并行 worker。 

    Execute one parallel source-pool worker for stage 01.

    Args:
        config_path: Runtime config path.
        run_root: Workflow run root.
        stage_run_id: External stage run identifier.
        worker_index: Zero-based worker index.
        worker_count: Total worker count.

    Returns:
        Process-style exit code.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    _validate_worker_bounds(worker_index, worker_count)

    cfg_obj = load_yaml_mapping(config_path)
    source_pool_cfg = BASE_RUNNER_MODULE._resolve_stage_01_source_pool_cfg(cfg_obj)
    if source_pool_cfg["enabled"] is not True:
        raise ValueError("stage 01 parallel worker requires stage_01_source_pool.enabled=true")

    prompt_pool, prompt_file_path = BASE_RUNNER_MODULE._resolve_stage_01_prompt_pool(cfg_obj)
    assigned_prompt_indices = _assigned_prompt_indices(len(prompt_pool), worker_index, worker_count)
    result_path = _worker_result_path(run_root, worker_index)
    ensure_directory(result_path.parent)

    direct_entries: List[Dict[str, Any]] = []
    source_pool_stage_results: List[Dict[str, Any]] = []
    try:
        for prompt_index in assigned_prompt_indices:
            prompt_text = prompt_pool[prompt_index]
            direct_entry = BASE_RUNNER_MODULE._run_source_pool_subrun(
                index=prompt_index,
                prompt_text=prompt_text,
                prompt_file_path=prompt_file_path,
                cfg_obj=cfg_obj,
                run_root=run_root,
                record_usage=source_pool_cfg["record_usage"],
            )
            direct_entries.append(direct_entry)
            source_pool_stage_results.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_text": prompt_text,
                    "stage_results": direct_entry["stage_results"],
                    "package_relative_path": direct_entry["package_relative_path"],
                }
            )

        result_payload = _build_worker_result_payload(
            stage_run_id=stage_run_id,
            prompt_file_path=prompt_file_path,
            worker_index=worker_index,
            worker_count=worker_count,
            assigned_prompt_indices=assigned_prompt_indices,
            direct_entries=direct_entries,
            source_pool_stage_results=source_pool_stage_results,
            status="ok",
        )
        write_json_atomic(result_path, result_payload)
        return 0
    except Exception as exc:
        result_payload = _build_worker_result_payload(
            stage_run_id=stage_run_id,
            prompt_file_path=prompt_file_path,
            worker_index=worker_index,
            worker_count=worker_count,
            assigned_prompt_indices=assigned_prompt_indices,
            direct_entries=direct_entries,
            source_pool_stage_results=source_pool_stage_results,
            status="failed",
            failure_reason="parallel_worker_exception",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        result_payload["traceback"] = traceback.format_exc()
        write_json_atomic(result_path, result_payload)
        print(json.dumps(result_payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


def main() -> int:
    """
    功能：parallel worker CLI 入口。 

    Entry point for the parallel worker script.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(description="Run one stage-01 prompt-parallel worker.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.as_posix()), help="Runtime config path.")
    parser.add_argument("--run-root", required=True, help="Workflow run root.")
    parser.add_argument("--stage-run-id", default=None, help="Optional external stage run identifier.")
    parser.add_argument("--worker-index", required=True, type=int, help="Zero-based worker index.")
    parser.add_argument("--worker-count", required=True, type=int, help="Total worker count.")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    run_root = resolve_repo_path(args.run_root)
    return run_parallel_worker(
        config_path,
        run_root,
        stage_run_id=args.stage_run_id,
        worker_index=int(args.worker_index),
        worker_count=int(args.worker_count),
    )


if __name__ == "__main__":
    sys.exit(main())