"""
文件目的：Notebook 编排阶段的轻量验收与路径共用工具。
Module type: General module

职责边界：
1. 仅提供路径标准化、相对路径视图与 formal GPU 前置检查。
2. 不直接参与 main/ 内部机制执行。
3. 兼容 notebook 运行入口对历史共用模块名的依赖，但不再依赖 archive。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

from scripts.notebook_runtime_common import normalize_path_value, relative_path_under_base


def build_path_views(run_root: Path, raw_paths: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    功能：构建绝对与相对路径视图。

    Build normalized absolute and run-root-relative path views.

    Args:
        run_root: Workflow run root.
        raw_paths: Mapping of path labels to path-like values.

    Returns:
        Mapping with absolute and relative path views.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(raw_paths, dict):
        raise TypeError("raw_paths must be dict")

    normalized_paths: Dict[str, str] = {}
    relative_paths: Dict[str, str] = {}
    for key_name, raw_value in raw_paths.items():
        normalized_paths[key_name] = normalize_path_value(raw_value)
        relative_paths[key_name] = relative_path_under_base(run_root, raw_value)
    return {"paths": normalized_paths, "paths_relative": relative_paths}


def detect_formal_gpu_preflight(cfg_path: Path) -> Dict[str, Any]:
    """
    功能：执行 formal GPU 与 attestation 环境前置检查。

    Execute preflight checks for CUDA availability and attestation environment variables.

    Args:
        cfg_path: Runtime config path.

    Returns:
        Preflight status mapping.
    """
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")
    cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg_obj, dict):
        raise ValueError("config root must be mapping")

    attestation_cfg = cfg_obj.get("attestation") if isinstance(cfg_obj.get("attestation"), dict) else {}
    required_env_vars = []
    for key_name in ("k_master_env_var", "k_prompt_env_var", "k_seed_env_var"):
        value = attestation_cfg.get(key_name)
        if isinstance(value, str) and value:
            required_env_vars.append(value)

    missing_env_vars = [name for name in required_env_vars if not os.environ.get(name)]
    nvidia_smi_path = shutil.which("nvidia-smi")
    return {
        "ok": bool(nvidia_smi_path and not missing_env_vars),
        "gpu_tool_available": bool(nvidia_smi_path),
        "nvidia_smi_path": nvidia_smi_path or "<absent>",
        "missing_attestation_env_vars": missing_env_vars,
    }