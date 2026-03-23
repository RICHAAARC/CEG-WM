"""
文件目的：验证正式路径收口后的清理状态与验收入口。
Module type: General module

Regression tests for repository cleanup and official acceptance entrypoints.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_module(module_name: str, module_path: Path) -> Any:
    """
    功能：按路径动态加载模块。

    Dynamically load a module from a file path.

    Args:
        module_name: Import name used for the dynamic module.
        module_path: Target Python file path.

    Returns:
        Imported module object.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_removed_research_only_files_absent() -> None:
    """
    功能：验证已收口的 research-only 与低价值文件已被删除。

    Verify obsolete research-only and low-value files are absent.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    removed_paths = [
        repo_root / "scripts" / "run_with_heartbeat.py",
        repo_root / "scripts" / "test_experiment_matrix_overrides.py",
        repo_root / "scripts" / "run_multi_protocol_evaluation.py",
        repo_root / "scripts" / "audits" / "audit_protocol_compare_outputs_schema.py",
        repo_root / "tests" / "test_multi_protocol_runner_and_compare.py",
        repo_root / "tests" / "test_multi_protocol_write_path_safety.py",
        repo_root / "tests" / "test_publish_workflow_unchanged_by_multi_protocol.py",
        repo_root / "tests" / "test_run_all_audits_strict_does_not_block_on_research_only_na.py",
    ]

    for path in removed_paths:
        assert not path.exists(), f"obsolete path should be removed: {path}"


def test_run_all_audits_registers_append_only_and_not_protocol_compare() -> None:
    """
    功能：验证聚合审计清单已切换到 append-only 审计。

    Verify run_all_audits registers append-only audit and drops protocol_compare audit.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_module("run_all_audits_cleanup_test", repo_root / "scripts" / "run_all_audits.py")

    assert "audits/audit_records_fields_append_only.py" in module.AUDIT_SCRIPTS
    assert "audits/audit_protocol_compare_outputs_schema.py" not in module.AUDIT_SCRIPTS


def test_cpu_smoke_entrypoint_defaults_to_smoke_profile() -> None:
    """
    功能：验证 CPU smoke 验收入口使用官方 smoke 配置与输出目录。

    Verify CPU smoke entrypoint defaults to the official smoke config and run root.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_module(
        "run_cpu_first_e2e_verification_test",
        repo_root / "scripts" / "run_cpu_first_e2e_verification.py",
    )

    assert module.DEFAULT_CONFIG_PATH.as_posix() == "configs/smoke_cpu.yaml"
    assert module.DEFAULT_RUN_ROOT.as_posix() == "outputs/onefile_cpu_smoke_verify"


def test_paper_full_entrypoint_defaults_to_formal_profile() -> None:
    """
    功能：验证 paper_full 正式验收入口绑定唯一正式配置。

    Verify the formal GPU entrypoint binds the single official paper_full config.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_module(
        "run_paper_full_workflow_verification_test",
        repo_root / "scripts" / "run_paper_full_workflow_verification.py",
    )

    assert module.DEFAULT_CONFIG_PATH.as_posix() == "configs/paper_full_cuda.yaml"
    assert module.DEFAULT_RUN_ROOT.as_posix() == "outputs/onefile_paper_full_cuda_verify"


def test_paper_full_cuda_output_entrypoint_defaults_to_project_outputs_only() -> None:
    """
    功能：验证新的 paper_full_cuda 输出导向入口绑定正式配置且不走 formal 验收。

    Verify the new paper_full_cuda output entrypoint binds the official config
    and uses the output-only run root.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_module(
        "run_paper_full_cuda_test",
        repo_root / "scripts" / "run_paper_full_cuda.py",
    )

    assert module.DEFAULT_CONFIG_PATH.as_posix() == "configs/paper_full_cuda.yaml"
    assert module.DEFAULT_RUN_ROOT.as_posix() == "outputs/colab_run_paper_full_cuda"