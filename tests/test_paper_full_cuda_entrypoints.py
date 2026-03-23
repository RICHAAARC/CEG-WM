"""
文件目的：当前正式 paper_full_cuda 主入口回归测试。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


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


def test_paper_full_cuda_output_entrypoint_defaults_to_project_outputs_only() -> None:
    """
    功能：验证当前正式主入口绑定官方配置与 output-only run_root。

    Verify the official paper_full_cuda entrypoint binds the official config
    and output-only run root.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_module(
        "run_paper_full_cuda_entrypoint_test",
        repo_root / "scripts" / "run_paper_full_cuda.py",
    )

    assert module.DEFAULT_CONFIG_PATH.as_posix() == "configs/paper_full_cuda.yaml"
    assert module.DEFAULT_RUN_ROOT.as_posix() == "outputs/colab_run_paper_full_cuda"


def test_paper_full_cuda_config_declares_dtype_and_parallel_statistics() -> None:
    """
    功能：验证正式配置显式声明 dtype 与并行 attestation 统计链。

    Verify the official paper_full_cuda config declares dtype and the parallel
    attestation statistics chain used by the output-only workflow.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    cfg_obj = yaml.safe_load((repo_root / "configs" / "paper_full_cuda.yaml").read_text(encoding="utf-8"))

    assert isinstance(cfg_obj, dict)
    assert cfg_obj.get("model_source") == "hf"
    assert cfg_obj.get("hf_revision") == "main"
    model_cfg = cfg_obj.get("model") if isinstance(cfg_obj.get("model"), dict) else {}
    assert model_cfg.get("dtype") == "float16"
    parallel_cfg = (
        cfg_obj.get("parallel_attestation_statistics")
        if isinstance(cfg_obj.get("parallel_attestation_statistics"), dict)
        else {}
    )
    assert parallel_cfg.get("enabled") is True
    assert parallel_cfg.get("calibration_score_name") == "event_attestation_score"
    assert parallel_cfg.get("evaluate_score_name") == "event_attestation_score"
    assert isinstance(cfg_obj.get("experiment_matrix"), dict)