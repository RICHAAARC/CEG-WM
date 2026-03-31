"""
文件目的：验证并行 stage 01 notebook 与脚本绑定合同。
Module type: General module

职责边界：
1. 覆盖并行 notebook 的路径绑定、worker 参数绑定与脚本 help 可执行性。
2. 显式验证原 notebook 保持单路线绑定，不被并行入口污染。
3. 不触发真实 notebook 执行，只解析 JSON 与运行 --help。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.notebook_runtime_common import build_repo_import_subprocess_env


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SINGLE_PATH = REPO_ROOT / "notebook" / "00_main" / "01_Paper_Full_Cuda.ipynb"
NOTEBOOK_PARALLEL_PATH = REPO_ROOT / "notebook" / "00_main" / "01_Paper_Full_Cuda_Parallel.ipynb"


def _load_notebook(notebook_path: Path) -> Dict[str, Any]:
    """
    功能：读取 notebook JSON 文档。 

    Load one notebook JSON document.

    Args:
        notebook_path: Notebook path.

    Returns:
        Parsed notebook mapping.
    """
    notebook_obj = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not isinstance(notebook_obj, dict):
        raise AssertionError(f"notebook root must be dict: {notebook_path}")
    return notebook_obj


def _find_code_cell_source(notebook_path: Path, marker: str) -> str:
    """
    功能：按标记文本定位 notebook code cell。 

    Locate one code-cell source block by a stable marker string.

    Args:
        notebook_path: Notebook path.
        marker: Marker text expected inside one code cell.

    Returns:
        Joined code-cell source string.
    """
    notebook_obj = _load_notebook(notebook_path)
    cells = notebook_obj.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"cells must be list: {notebook_path}")

    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if not isinstance(source, list):
            continue
        source_text = "\n".join(str(line) for line in source)
        if marker in source_text:
            return source_text
    raise AssertionError(f"code cell marker not found: {marker}")


def test_parallel_notebook_binds_parallel_stage_paths_and_worker_count() -> None:
    """
    功能：验证并行 notebook 指向新脚本、新输出根与固定 worker 数。 

    Verify that the parallel notebook binds to the new script, the isolated
    Drive root, and the fixed worker count.

    Args:
        None.

    Returns:
        None.
    """
    constants_source = _find_code_cell_source(NOTEBOOK_PARALLEL_PATH, 'NOTEBOOK_NAME = "01_Paper_Full_Cuda_Parallel"')
    execute_source = _find_code_cell_source(NOTEBOOK_PARALLEL_PATH, "COMMAND = [")

    assert 'DRIVE_PROJECT_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_Outputs_Parallel"' in constants_source
    assert 'SCRIPT_PATH = REPO_ROOT / "scripts" / "01_Paper_Full_Cuda_Parallel.py"' in constants_source
    assert "PARALLEL_WORKER_COUNT = 2" in constants_source
    assert '"--worker-count"' in execute_source
    assert 'str(PARALLEL_WORKER_COUNT)' in execute_source


def test_parallel_notebook_does_not_pollute_single_route_notebook() -> None:
    """
    功能：验证新增并行 notebook 不改写原始单路线绑定。 

    Verify that the new parallel notebook does not rewrite the original
    single-route notebook bindings.

    Args:
        None.

    Returns:
        None.
    """
    single_constants = _find_code_cell_source(NOTEBOOK_SINGLE_PATH, 'NOTEBOOK_NAME = "01_Paper_Full_Cuda"')
    parallel_constants = _find_code_cell_source(NOTEBOOK_PARALLEL_PATH, 'NOTEBOOK_NAME = "01_Paper_Full_Cuda_Parallel"')

    assert 'DRIVE_PROJECT_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_Outputs_project_root"' in single_constants
    assert 'SCRIPT_PATH = REPO_ROOT / "scripts" / "01_Paper_Full_Cuda.py"' in single_constants
    assert "PARALLEL_WORKER_COUNT" not in single_constants
    assert 'DRIVE_PROJECT_ROOT = DRIVE_MOUNT_ROOT / "MyDrive" / "CEG_WM_Outputs_Parallel"' in parallel_constants


@pytest.mark.parametrize(
    "script_name",
    [
        "01_Paper_Full_Cuda_Parallel.py",
        "01_run_paper_full_cuda_parallel.py",
        "01_run_paper_full_cuda_parallel_worker.py",
    ],
)
def test_parallel_stage_scripts_support_help(script_name: str) -> None:
    """
    功能：验证并行 stage 脚本在补齐 repo import 上下文后可执行 --help。 

    Verify that each parallel stage script supports --help when the repository
    import context is provided.

    Args:
        script_name: Stage script file name.

    Returns:
        None.
    """
    command = [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=build_repo_import_subprocess_env(repo_root=REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()