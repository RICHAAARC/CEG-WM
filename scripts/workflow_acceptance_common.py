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
from typing import Any, Dict, List, Mapping, Optional, cast

import yaml

from scripts.notebook_runtime_common import (
    collect_attestation_env_summary,
    normalize_path_value,
    relative_path_under_base,
    resolve_attestation_env_var_names,
)


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


def _load_config_mapping(cfg_path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    功能：安全读取 preflight 配置。 

    Safely load the runtime config for preflight evaluation.

    Args:
        cfg_path: Runtime config path.

    Returns:
        Tuple of parsed config mapping or None, and error text or None.
    """
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")
    if not cfg_path.exists() or not cfg_path.is_file():
        return None, f"config_not_found:{normalize_path_value(cfg_path)}"
    try:
        cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"config_load_failed:{type(exc).__name__}:{exc}"
    if not isinstance(cfg_obj, dict):
        return None, "config_root_not_mapping"
    return cast(Dict[str, Any], cfg_obj), None


def _collect_gpu_tool_summary() -> Dict[str, Any]:
    """
    功能：收集 nvidia-smi 可用性摘要。 

    Collect the lightweight nvidia-smi availability summary.

    Args:
        None.

    Returns:
        GPU-tool availability mapping.
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    return {
        "gpu_tool_available": bool(nvidia_smi_path),
        "nvidia_smi_path": nvidia_smi_path or "<absent>",
    }


def _build_missing_path_check(path_obj: Optional[Path], label: str) -> Dict[str, Any]:
    """
    功能：构造路径存在性检查结果。 

    Build a standard path-existence check payload.

    Args:
        path_obj: Candidate path.
        label: Human-readable label.

    Returns:
        Path-check mapping.
    """
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    exists = isinstance(path_obj, Path) and path_obj.exists()
    return {
        "label": label,
        "path": normalize_path_value(path_obj),
        "exists": bool(exists),
    }


def detect_stage_01_preflight(cfg_path: Path) -> Dict[str, Any]:
    """
    功能：执行 stage 01 的 formal preflight。 

    Execute stage-01 preflight checks for GPU, attestation env, and required config gates.

    Args:
        cfg_path: Runtime config path.

    Returns:
        Stage-01 preflight status mapping.
    """
    cfg_obj, config_error = _load_config_mapping(cfg_path)
    gpu_summary = _collect_gpu_tool_summary()
    result: Dict[str, Any] = {
        "stage_name": "01_Paper_Full_Cuda",
        "cfg_path": normalize_path_value(cfg_path),
        "config_error": config_error,
        "gpu_required": True,
        **gpu_summary,
        "attestation_env_required": False,
        "required_attestation_env_vars": [],
        "missing_attestation_env_vars": [],
        "attestation_env_var_bindings_complete": True,
        "stage_01_source_pool_enabled": False,
        "stage_01_source_pool_prompt_file_bound": False,
        "stage_01_pooled_threshold_build_enabled": False,
        "stage_01_pooled_threshold_target_pair_count_valid": False,
        "failed_checks": [],
        "ok": False,
    }
    failed_checks = cast(List[str], result["failed_checks"])
    if config_error is not None or cfg_obj is None:
        failed_checks.append("config_invalid")
        result["ok"] = False
        return result

    attestation_summary = collect_attestation_env_summary(cfg_obj)
    env_var_names = resolve_attestation_env_var_names(cfg_obj)
    missing_env_var_bindings = [
        config_key
        for config_key in ("k_master_env_var", "k_prompt_env_var", "k_seed_env_var")
        if config_key not in env_var_names
    ]
    attestation_cfg = cfg_obj.get("attestation") if isinstance(cfg_obj.get("attestation"), dict) else {}
    source_pool_cfg = cfg_obj.get("stage_01_source_pool") if isinstance(cfg_obj.get("stage_01_source_pool"), dict) else {}
    pooled_cfg = (
        cfg_obj.get("stage_01_pooled_threshold_build")
        if isinstance(cfg_obj.get("stage_01_pooled_threshold_build"), dict)
        else {}
    )

    result["attestation_env_required"] = bool(attestation_cfg.get("enabled", False))
    result["required_attestation_env_vars"] = attestation_summary["required_env_vars"]
    result["missing_attestation_env_vars"] = attestation_summary["missing_env_vars"]
    result["attestation_env_var_bindings_complete"] = len(missing_env_var_bindings) == 0
    result["missing_attestation_env_var_bindings"] = missing_env_var_bindings
    result["stage_01_source_pool_enabled"] = source_pool_cfg.get("enabled") is True
    result["stage_01_source_pool_prompt_file_bound"] = (
        source_pool_cfg.get("use_inference_prompt_file") is True
        and isinstance(cfg_obj.get("inference_prompt_file"), str)
        and bool(str(cfg_obj.get("inference_prompt_file")).strip())
    )
    result["stage_01_pooled_threshold_build_enabled"] = pooled_cfg.get("enabled") is True
    result["stage_01_pooled_threshold_target_pair_count_valid"] = (
        isinstance(pooled_cfg.get("target_pair_count"), int)
        and int(pooled_cfg.get("target_pair_count")) > 0
    )

    if not bool(result["gpu_tool_available"]):
        failed_checks.append("gpu_tool_unavailable")
    if bool(result["attestation_env_required"]) and not bool(result["attestation_env_var_bindings_complete"]):
        failed_checks.append("attestation_env_var_bindings_incomplete")
    if bool(result["attestation_env_required"]) and result["missing_attestation_env_vars"]:
        failed_checks.append("missing_attestation_env_vars")
    if not bool(result["stage_01_source_pool_enabled"]):
        failed_checks.append("stage_01_source_pool_disabled")
    if not bool(result["stage_01_source_pool_prompt_file_bound"]):
        failed_checks.append("stage_01_source_pool_prompt_file_missing")
    if not bool(result["stage_01_pooled_threshold_build_enabled"]):
        failed_checks.append("stage_01_pooled_threshold_build_disabled")
    if not bool(result["stage_01_pooled_threshold_target_pair_count_valid"]):
        failed_checks.append("stage_01_pooled_threshold_target_pair_count_invalid")

    result["ok"] = len(failed_checks) == 0
    return result


def detect_stage_02_preflight(
    cfg_path: Path,
    source_package_path: Path,
    source_contract_path: Path,
) -> Dict[str, Any]:
    """
    功能：执行 stage 02 的 package-only preflight。 

    Execute stage-02 preflight checks for the source package and required source contract.

    Args:
        cfg_path: Runtime config path.
        source_package_path: Source stage-01 package path.
        source_contract_path: Required source contract path resolved from the extracted package.

    Returns:
        Stage-02 preflight status mapping.
    """
    cfg_obj, config_error = _load_config_mapping(cfg_path)
    gpu_summary = _collect_gpu_tool_summary()
    source_package_check = _build_missing_path_check(source_package_path, "source_package")
    source_contract_check = _build_missing_path_check(source_contract_path, "source_contract")
    attestation_summary = collect_attestation_env_summary(cfg_obj) if isinstance(cfg_obj, dict) else {
        "required_env_vars": [],
        "missing_env_vars": [],
    }
    failed_checks: List[str] = []
    if config_error is not None or cfg_obj is None:
        failed_checks.append("config_invalid")
    if not bool(source_package_check["exists"]):
        failed_checks.append("source_package_missing")
    if not bool(source_contract_check["exists"]):
        failed_checks.append("source_contract_missing")

    return {
        "stage_name": "02_Parallel_Attestation_Statistics",
        "cfg_path": normalize_path_value(cfg_path),
        "config_error": config_error,
        "gpu_required": False,
        **gpu_summary,
        "attestation_env_required": False,
        "required_attestation_env_vars": attestation_summary.get("required_env_vars", []),
        "missing_attestation_env_vars": attestation_summary.get("missing_env_vars", []),
        "source_package_path": source_package_check["path"],
        "source_package_exists": source_package_check["exists"],
        "source_contract_path": source_contract_check["path"],
        "source_contract_exists": source_contract_check["exists"],
        "failed_checks": failed_checks,
        "ok": len(failed_checks) == 0,
    }


def detect_stage_03_preflight(
    cfg_path: Path,
    source_package_path: Path,
    source_thresholds_artifact_path: Path,
) -> Dict[str, Any]:
    """
    功能：执行 stage 03 的 experiment-matrix preflight。 

    Execute stage-03 preflight checks for GPU availability and required stage-01 source artifacts.

    Args:
        cfg_path: Runtime config path.
        source_package_path: Source stage-01 package path.
        source_thresholds_artifact_path: Source thresholds artifact path.

    Returns:
        Stage-03 preflight status mapping.
    """
    cfg_obj, config_error = _load_config_mapping(cfg_path)
    gpu_summary = _collect_gpu_tool_summary()
    source_package_check = _build_missing_path_check(source_package_path, "source_package")
    thresholds_check = _build_missing_path_check(source_thresholds_artifact_path, "source_thresholds_artifact")
    attestation_summary = collect_attestation_env_summary(cfg_obj) if isinstance(cfg_obj, dict) else {
        "required_env_vars": [],
        "missing_env_vars": [],
    }
    failed_checks: List[str] = []
    if config_error is not None or cfg_obj is None:
        failed_checks.append("config_invalid")
    if not bool(gpu_summary["gpu_tool_available"]):
        failed_checks.append("gpu_tool_unavailable")
    if not bool(source_package_check["exists"]):
        failed_checks.append("source_package_missing")
    if not bool(thresholds_check["exists"]):
        failed_checks.append("source_thresholds_artifact_missing")

    return {
        "stage_name": "03_Experiment_Matrix_Full",
        "cfg_path": normalize_path_value(cfg_path),
        "config_error": config_error,
        "gpu_required": True,
        **gpu_summary,
        "attestation_env_required": False,
        "required_attestation_env_vars": attestation_summary.get("required_env_vars", []),
        "missing_attestation_env_vars": attestation_summary.get("missing_env_vars", []),
        "source_package_path": source_package_check["path"],
        "source_package_exists": source_package_check["exists"],
        "source_thresholds_artifact_path": thresholds_check["path"],
        "source_thresholds_artifact_exists": thresholds_check["exists"],
        "failed_checks": failed_checks,
        "ok": len(failed_checks) == 0,
    }


def detect_stage_04_preflight(
    cfg_path: Path,
    stage_inputs: Mapping[str, Mapping[str, Any]],
    *,
    require_stage_02: bool,
    require_stage_03: bool,
) -> Dict[str, Any]:
    """
    功能：执行 stage 04 的 signoff-input preflight。 

    Execute stage-04 preflight checks for required stage packages and lineage-ready manifests.

    Args:
        cfg_path: Runtime config path.
        stage_inputs: Prepared stage-input mapping.
        require_stage_02: Whether stage 02 is mandatory.
        require_stage_03: Whether stage 03 is mandatory.

    Returns:
        Stage-04 preflight status mapping.
    """
    if not isinstance(stage_inputs, Mapping):
        raise TypeError("stage_inputs must be Mapping")

    _, config_error = _load_config_mapping(cfg_path)
    gpu_summary = _collect_gpu_tool_summary()
    failed_checks: List[str] = []

    def _stage_ready(stage_key: str, required: bool) -> Dict[str, Any]:
        info = stage_inputs.get(stage_key)
        stage_manifest = info.get("stage_manifest") if isinstance(info, Mapping) else None
        package_manifest = info.get("package_manifest") if isinstance(info, Mapping) else None
        status = info.get("status") if isinstance(info, Mapping) else "not_provided"
        input_path = info.get("input_path") if isinstance(info, Mapping) else "<absent>"
        has_manifests = isinstance(stage_manifest, Mapping) and isinstance(package_manifest, Mapping)
        if required and status != "prepared":
            failed_checks.append(f"{stage_key}_package_not_ready")
        elif status == "prepared" and not has_manifests:
            failed_checks.append(f"{stage_key}_manifest_missing")
        return {
            "required": required,
            "status": status,
            "input_path": input_path,
            "has_manifests": has_manifests,
        }

    stage_01_summary = _stage_ready("stage_01", True)
    stage_02_summary = _stage_ready("stage_02", require_stage_02)
    stage_03_summary = _stage_ready("stage_03", require_stage_03)
    if config_error is not None:
        failed_checks.append("config_invalid")

    return {
        "stage_name": "04_Release_And_Signoff",
        "cfg_path": normalize_path_value(cfg_path),
        "config_error": config_error,
        "gpu_required": False,
        **gpu_summary,
        "attestation_env_required": False,
        "require_stage_02": require_stage_02,
        "require_stage_03": require_stage_03,
        "stage_01": stage_01_summary,
        "stage_02": stage_02_summary,
        "stage_03": stage_03_summary,
        "failed_checks": failed_checks,
        "ok": len(failed_checks) == 0,
    }


def detect_formal_gpu_preflight(cfg_path: Path) -> Dict[str, Any]:
    """
    功能：stage 01 formal preflight 的兼容包装器。 

    Provide a compatibility wrapper that preserves the legacy formal GPU preflight API.

    Args:
        cfg_path: Runtime config path.

    Returns:
        Legacy-compatible preflight status mapping derived from stage 01 semantics.
    """
    stage_01_preflight = detect_stage_01_preflight(cfg_path)

    return {
        "ok": bool(stage_01_preflight.get("ok", False)),
        "gpu_tool_available": bool(stage_01_preflight.get("gpu_tool_available", False)),
        "nvidia_smi_path": stage_01_preflight.get("nvidia_smi_path", "<absent>"),
        "missing_attestation_env_vars": stage_01_preflight.get("missing_attestation_env_vars", []),
        "required_attestation_env_vars": stage_01_preflight.get("required_attestation_env_vars", []),
        "compatibility_wrapper": True,
        "stage_name": stage_01_preflight.get("stage_name"),
        "failed_checks": stage_01_preflight.get("failed_checks", []),
    }