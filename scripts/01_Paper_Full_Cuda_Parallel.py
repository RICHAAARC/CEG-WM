"""
文件目的：01_Paper_Full_Cuda_Parallel notebook 的并行 stage 01 编排入口。
Module type: General module

职责边界：
1. 仅通过薄封装复用正式 stage 01 wrapper，不改动既有单路线实现。
2. 只切换 notebook / stage 命名与 runner 脚本绑定，formal contract 继续沿用既有路径。
3. 所有并行执行细节下沉到独立 parallel runner 与 worker 脚本。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


BASE_WRAPPER_PATH = Path(__file__).resolve().with_name("01_Paper_Full_Cuda.py")
PARALLEL_STAGE_NAME = "01_Paper_Full_Cuda_Parallel"
PARALLEL_RUNNER_SCRIPT_PATH = Path("scripts/01_run_paper_full_cuda_parallel.py")


def _load_base_wrapper_module() -> ModuleType:
    """
    功能：按文件路径加载 baseline stage 01 wrapper。 

    Load the baseline stage-01 wrapper module from its file path.

    Args:
        None.

    Returns:
        Loaded module object.

    Raises:
        RuntimeError: If the wrapper module cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location("stage_01_paper_full_cuda_wrapper", BASE_WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load stage-01 wrapper: {BASE_WRAPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_WRAPPER_MODULE = _load_base_wrapper_module()
_BASE_WRAPPER_MODULE.STAGE_01_NAME = PARALLEL_STAGE_NAME
_BASE_WRAPPER_MODULE.RUNNER_SCRIPT_PATH = PARALLEL_RUNNER_SCRIPT_PATH


def main() -> int:
    """
    功能：并行 stage 01 wrapper CLI 入口。 

    Execute the parallel stage-01 wrapper entry point.

    Args:
        None.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(description="Run the parallel stage-01 Paper_Full_Cuda notebook orchestration.")
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root.")
    parser.add_argument("--config", default=str(_BASE_WRAPPER_MODULE.DEFAULT_CONFIG_PATH.as_posix()), help="Source config path.")
    parser.add_argument("--notebook-name", default=PARALLEL_STAGE_NAME, help="Notebook display name.")
    parser.add_argument("--stage-run-id", default=None, help="Optional fixed stage run identifier.")
    parser.add_argument("--worker-count", default=2, type=int, help="Prompt worker count for the parallel runner.")
    args = parser.parse_args()

    original_run_command_with_logs = _BASE_WRAPPER_MODULE.run_command_with_logs

    def _run_command_with_parallel_worker_count(command, cwd, stdout_log_path, stderr_log_path):
        command_list = [str(part) for part in command]
        command_list.extend(["--worker-count", str(int(args.worker_count))])
        return original_run_command_with_logs(command_list, cwd, stdout_log_path, stderr_log_path)

    _BASE_WRAPPER_MODULE.run_command_with_logs = _run_command_with_parallel_worker_count
    summary = _BASE_WRAPPER_MODULE.run_stage_01(
        drive_project_root=_BASE_WRAPPER_MODULE.resolve_repo_path(args.drive_project_root),
        config_path=_BASE_WRAPPER_MODULE.resolve_repo_path(args.config),
        notebook_name=str(args.notebook_name),
        stage_run_id=args.stage_run_id or _BASE_WRAPPER_MODULE.make_stage_run_id(PARALLEL_STAGE_NAME),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())