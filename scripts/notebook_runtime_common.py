"""
文件目的：Notebook 与 scripts 编排共用的纯运行时辅助能力。
Module type: General module

职责边界：
1. 仅承载 SHA256、复制、压缩解压、路径校验、Drive 目录组织、运行元数据收集等纯编排能力。
2. 不承载 main/ 下的 watermarking、detection、evaluation 核心机制。
3. 为独立 Colab 会话下的 stage package、lineage 与审计输出提供稳定工具。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, cast

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PW00_STAGE_NAME = "PW00_Paper_Eval_Family_Manifest"
PW01_STAGE_NAME = "PW01_Source_Event_Shards"
PW02_STAGE_NAME = "PW02_Source_Merge_And_Global_Thresholds"
PW03_STAGE_NAME = "PW03_Attack_Event_Shards"
PW04_STAGE_NAME = "PW04_Attack_Merge_And_Metrics"
PW05_STAGE_NAME = "PW05_Release_And_Signoff"
FORMAL_STAGE_PACKAGE_ROLE = "formal_stage_package"
FAILURE_DIAGNOSTICS_PACKAGE_ROLE = "failure_diagnostics_package"
FORMAL_PACKAGE_DISCOVERY_SCOPE = "discoverable_formal_only"
EXCLUDED_PACKAGE_DISCOVERY_SCOPE = "excluded_from_formal_discovery"
ATTESTATION_ENV_FILE_NAME = "attestation_env.json"
ATTESTATION_ENV_INFO_FILE_NAME = "attestation_env_info.json"
NOTEBOOK_MODEL_SNAPSHOT_ENV_VAR = "CEG_WM_MODEL_SNAPSHOT_PATH"
NOTEBOOK_MODEL_SNAPSHOT_BINDING_SOURCE = "notebook_snapshot_download"
RUNTIME_DIAGNOSTICS_SCHEMA_VERSION = "pw_runtime_diagnostics_v1"
RUNTIME_DIAGNOSTICS_STDIO_TAIL_LIMIT = 4000
ATTESTATION_ENV_VAR_LENGTHS = {
    "k_master_env_var": 64,
    "k_prompt_env_var": 32,
    "k_seed_env_var": 32,
}


def utc_now_iso() -> str:
    """
    功能：返回 UTC ISO 8601 时间戳。

    Return the current UTC timestamp in ISO 8601 format.

    Args:
        None.

    Returns:
        UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def _sanitize_identifier(value: str) -> str:
    """
    功能：将标识字符串规范化为安全片段。

    Normalize a free-form identifier into a filesystem-safe token.

    Args:
        value: Raw identifier string.

    Returns:
        Sanitized lowercase identifier token.
    """
    if not isinstance(value, str) or not value.strip():
        raise TypeError("value must be non-empty str")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return normalized.lower() or "stage"


def make_stage_run_id(stage_name: Optional[str] = None) -> str:
    """
    功能：生成阶段运行唯一标识。

    Generate a stable stage run identifier based on UTC time and random suffix.

    Args:
        stage_name: Optional stable stage name.

    Returns:
        Stage run identifier string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    random_suffix = uuid.uuid4().hex[:8]
    if isinstance(stage_name, str) and stage_name.strip():
        return f"{_sanitize_identifier(stage_name)}__{timestamp}__{random_suffix}"
    return f"stage__{timestamp}__{random_suffix}"


def ensure_directory(path_obj: Path) -> Path:
    """
    功能：确保目录存在。

    Ensure that a directory exists.

    Args:
        path_obj: Directory path.

    Returns:
        The same directory path.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def normalize_path_value(path_value: Any) -> str:
    """
    功能：将路径值规范化为 POSIX 字符串。

    Normalize a path-like value into a stable POSIX string.

    Args:
        path_value: Path-like value.

    Returns:
        Normalized POSIX path string, or "<absent>" when unavailable.
    """
    if isinstance(path_value, Path):
        return path_value.resolve().as_posix()
    if isinstance(path_value, str) and path_value.strip():
        candidate = Path(path_value.strip()).expanduser()
        if candidate.is_absolute():
            return candidate.resolve().as_posix()
        return (REPO_ROOT / candidate).resolve().as_posix()
    return "<absent>"


def relative_path_under_base(base_path: Path, path_value: Any) -> str:
    """
    功能：计算相对基础目录的路径。

    Represent a path-like value relative to a base directory when possible.

    Args:
        base_path: Base directory path.
        path_value: Path-like value.

    Returns:
        Relative POSIX path, "." for the base itself, or "<absent>" when not under base.
    """
    if not isinstance(base_path, Path):
        raise TypeError("base_path must be Path")

    base_text = normalize_path_value(base_path).rstrip("/")
    candidate_text = normalize_path_value(path_value)
    if candidate_text == "<absent>":
        return "<absent>"
    if candidate_text == base_text:
        return "."

    prefix = f"{base_text}/"
    if candidate_text.startswith(prefix):
        return candidate_text[len(prefix):]
    return "<absent>"


def resolve_repo_path(path_value: str, repo_root: Path = REPO_ROOT) -> Path:
    """
    功能：按仓库根目录解析路径。

    Resolve a path against the repository root unless it is already absolute.

    Args:
        path_value: Raw path string.
        repo_root: Repository root path.

    Returns:
        Resolved absolute path.
    """
    if not isinstance(path_value, str) or not path_value.strip():
        raise TypeError("path_value must be non-empty str")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    candidate = Path(path_value.strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _normalize_pythonpath_entry(path_text: str) -> str:
    """
    功能：规范化 PYTHONPATH 条目以便稳定比较。

    Normalize one PYTHONPATH entry into a stable comparison token.

    Args:
        path_text: Raw PYTHONPATH entry text.

    Returns:
        Normalized token suitable for equality checks.
    """
    if not isinstance(path_text, str) or not path_text.strip():
        raise TypeError("path_text must be non-empty str")

    expanded_path = os.path.expanduser(path_text.strip())
    normalized_path = os.path.normpath(expanded_path)
    if os.path.isabs(normalized_path):
        return os.path.normcase(str(Path(normalized_path).resolve()))
    return os.path.normcase(normalized_path)


def build_repo_import_subprocess_env(
    base_env: Optional[Mapping[str, str]] = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, str]:
    """
    功能：构造带仓库根目录导入上下文的子进程环境。

    Build a subprocess environment that preserves the source environment while
    ensuring the repository root is available on PYTHONPATH.

    Args:
        base_env: Optional source environment mapping. When omitted, os.environ
            is copied.
        repo_root: Repository root path that must be importable by child
            processes.

    Returns:
        Environment mapping suitable for subprocess.run.

    Raises:
        TypeError: If base_env is not a mapping or repo_root is not Path.
    """
    if base_env is not None and not isinstance(base_env, Mapping):
        raise TypeError("base_env must be Mapping[str, str] or None")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    source_env = os.environ if base_env is None else base_env
    env_mapping = {str(key): str(value) for key, value in source_env.items()}

    repo_root_text = str(repo_root.resolve())
    repo_root_token = _normalize_pythonpath_entry(repo_root_text)
    existing_pythonpath = str(env_mapping.get("PYTHONPATH", "")).strip()
    retained_entries: List[str] = []

    if existing_pythonpath:
        for entry_text in existing_pythonpath.split(os.pathsep):
            normalized_entry = entry_text.strip()
            if not normalized_entry:
                continue
            if _normalize_pythonpath_entry(normalized_entry) == repo_root_token:
                continue
            retained_entries.append(normalized_entry)

    env_mapping["PYTHONPATH"] = os.pathsep.join([repo_root_text, *retained_entries])
    return env_mapping


def _is_nonempty_file(path_obj: Path) -> bool:
    """
    功能：判断文件是否存在且非空。

    Check whether one file exists and has non-zero size.

    Args:
        path_obj: Candidate file path.

    Returns:
        True when the file exists, is regular, and has positive size.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    return bool(path_obj.exists() and path_obj.is_file() and path_obj.stat().st_size > 0)


def _directory_contains_files(path_obj: Path) -> bool:
    """
    功能：判断目录是否包含至少一个文件。

    Check whether one directory contains at least one file entry.

    Args:
        path_obj: Candidate directory path.

    Returns:
        True when the directory exists and contains at least one file.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_dir():
        return False
    for child_path in path_obj.rglob("*"):
        if child_path.is_file():
            return True
    return False


def _sync_file_copy(source_path: Path, destination_path: Path) -> Path:
    """
    功能：把单文件同步到目标路径。

    Synchronize one source file into the destination path.

    Args:
        source_path: Existing source file path.
        destination_path: Destination file path.

    Returns:
        Destination file path.
    """
    if not isinstance(source_path, Path):
        raise TypeError("source_path must be Path")
    if not isinstance(destination_path, Path):
        raise TypeError("destination_path must be Path")
    if source_path.resolve() == destination_path.resolve():
        return destination_path
    return copy_file(source_path, destination_path)


def _replace_directory_copy(source_root: Path, destination_root: Path) -> Path:
    """
    功能：以整目录替换方式同步快照目录。

    Replace one destination directory with a fresh copy of the source tree.

    Args:
        source_root: Existing source directory.
        destination_root: Destination directory.

    Returns:
        Destination directory path.
    """
    if not isinstance(source_root, Path):
        raise TypeError("source_root must be Path")
    if not isinstance(destination_root, Path):
        raise TypeError("destination_root must be Path")
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"source_root not found: {source_root}")
    if destination_root.exists():
        if not destination_root.is_dir():
            raise RuntimeError(f"destination_root must be directory when present: {destination_root}")
        shutil.rmtree(destination_root)
    ensure_directory(destination_root.parent)
    shutil.copytree(source_root, destination_root)
    return destination_root


def resolve_notebook_model_cache_layout(
    drive_mount_root: Path,
    repo_root: Path = REPO_ROOT,
    *,
    create_directories: bool = False,
) -> Dict[str, Path]:
    """
    功能：解析 notebook 使用的本地会话缓存与兼容性 Drive 目录布局。

    Resolve the notebook cache layout for session-local Hugging Face caches,
    while preserving compatibility paths for lightweight Google Drive
    artifacts.

    Args:
        drive_mount_root: Notebook Drive mount root.
        repo_root: Repository root used for the local runtime cache.
        create_directories: Whether to create all managed directories.

    Returns:
        Mapping of cache layout paths.
    """
    if not isinstance(drive_mount_root, Path):
        raise TypeError("drive_mount_root must be Path")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(create_directories, bool):
        raise TypeError("create_directories must be bool")

    drive_models_root = drive_mount_root / "MyDrive" / "Models"
    persistent_inspyrenet_root = drive_models_root / "inspyrenet"
    persistent_hf_root = drive_models_root / "Huggingface"
    local_hf_home = repo_root / "huggingface_cache"
    local_hf_hub_cache = local_hf_home / "hub"
    local_transformers_cache = local_hf_home / "transformers"
    local_runtime_snapshot_root = local_hf_home / "runtime_snapshots"

    layout: Dict[str, Path] = {
        "drive_models_root": drive_models_root,
        "persistent_inspyrenet_root": persistent_inspyrenet_root,
        "persistent_inspyrenet_path": persistent_inspyrenet_root / "ckpt_base.pth",
        "persistent_hf_root": persistent_hf_root,
        "local_hf_home": local_hf_home,
        "local_hf_hub_cache": local_hf_hub_cache,
        "local_transformers_cache": local_transformers_cache,
        "local_runtime_snapshot_root": local_runtime_snapshot_root,
    }

    if create_directories:
        for key in [
            "drive_models_root",
            "persistent_inspyrenet_root",
            "local_hf_home",
            "local_hf_hub_cache",
            "local_transformers_cache",
            "local_runtime_snapshot_root",
        ]:
            ensure_directory(layout[key])
    return layout


def bootstrap_notebook_model_cache(
    cfg_obj: Mapping[str, Any],
    drive_mount_root: Path,
    repo_root: Path = REPO_ROOT,
    *,
    semantic_model_source_urls: Optional[Mapping[str, str]] = None,
    weight_url_override: Optional[str] = None,
    force_inspyrenet_download: bool = False,
    snapshot_download_fn: Optional[Callable[..., str]] = None,
    file_download_fn: Optional[Callable[[str, Path], None]] = None,
) -> Dict[str, Any]:
    """
    功能：准备 notebook 所需的本地会话模型缓存，并同步轻量审计元信息。

    Prepare the notebook model cache by resolving the Stable Diffusion
    snapshot directly from the session-local Hugging Face cache while keeping
    lightweight Drive compatibility only for the InSPyReNet weight file.

    Args:
        cfg_obj: Runtime configuration mapping.
        drive_mount_root: Notebook Drive mount root.
        repo_root: Repository root used for local runtime paths.
        semantic_model_source_urls: Optional mapping from semantic model source
            identifiers to download URLs.
        weight_url_override: Optional explicit semantic model weight URL.
        force_inspyrenet_download: Whether to force a fresh InSPyReNet download
            into the persistent compatibility cache.
        snapshot_download_fn: Optional snapshot download callable used for
            testing.
        file_download_fn: Optional file download callable used for testing.

    Returns:
        Bootstrap summary containing the session-local snapshot path,
        compatibility metadata, and lightweight audit fields.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    if not isinstance(drive_mount_root, Path):
        raise TypeError("drive_mount_root must be Path")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if semantic_model_source_urls is not None and not isinstance(semantic_model_source_urls, Mapping):
        raise TypeError("semantic_model_source_urls must be Mapping[str, str] or None")
    if weight_url_override is not None and not isinstance(weight_url_override, str):
        raise TypeError("weight_url_override must be str or None")
    if not isinstance(force_inspyrenet_download, bool):
        raise TypeError("force_inspyrenet_download must be bool")

    layout = resolve_notebook_model_cache_layout(drive_mount_root, repo_root, create_directories=True)
    model_identity = resolve_model_identity(cfg_obj)
    model_id = str(model_identity["model_id"])
    revision = str(model_identity["revision"])
    if model_id == "<absent>":
        raise ValueError("model_id must be provided in cfg_obj")

    mask_cfg = cast(Dict[str, Any], cfg_obj.get("mask")) if isinstance(cfg_obj.get("mask"), dict) else {}
    semantic_model_path = mask_cfg.get("semantic_model_path")
    if not isinstance(semantic_model_path, str) or not semantic_model_path.strip():
        raise ValueError("mask.semantic_model_path must be non-empty str")
    repo_inspyrenet_path = resolve_repo_path(semantic_model_path, repo_root)
    ensure_directory(repo_inspyrenet_path.parent)

    semantic_model_source = (
        str(mask_cfg.get("semantic_model_source")).strip()
        if isinstance(mask_cfg.get("semantic_model_source"), str)
        else ""
    )
    semantic_source_urls = semantic_model_source_urls or {}
    semantic_weight_url = (
        weight_url_override.strip()
        if isinstance(weight_url_override, str) and weight_url_override.strip()
        else str(semantic_source_urls.get(semantic_model_source, "")).strip()
    )

    persistent_inspyrenet_path = layout["persistent_inspyrenet_path"]
    weight_cache_mode = "persistent_reused"
    if force_inspyrenet_download:
        weight_cache_mode = "persistent_downloaded"
    elif _is_nonempty_file(persistent_inspyrenet_path):
        weight_cache_mode = "persistent_reused"
    elif _is_nonempty_file(repo_inspyrenet_path):
        _sync_file_copy(repo_inspyrenet_path, persistent_inspyrenet_path)
        weight_cache_mode = "persistent_seeded_from_repo"
    else:
        weight_cache_mode = "persistent_downloaded"

    if weight_cache_mode == "persistent_downloaded":
        if not semantic_weight_url:
            raise RuntimeError(
                f"unsupported semantic_model_source for notebook bootstrap: {semantic_model_source or '<absent>'}"
            )
        download_file = file_download_fn
        if download_file is None:
            import urllib.request

            def _default_file_download(url: str, target_path: Path) -> None:
                urllib.request.urlretrieve(url, str(target_path))

            download_file = _default_file_download

        temp_download_path = persistent_inspyrenet_path.with_suffix(persistent_inspyrenet_path.suffix + ".downloading")
        if temp_download_path.exists():
            temp_download_path.unlink()
        download_file(semantic_weight_url, temp_download_path)
        if not _is_nonempty_file(temp_download_path):
            temp_download_path.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded semantic weight is invalid: {temp_download_path}")
        temp_download_path.replace(persistent_inspyrenet_path)

    _sync_file_copy(persistent_inspyrenet_path, repo_inspyrenet_path)

    snapshot_download = snapshot_download_fn
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as huggingface_snapshot_download

        snapshot_download = huggingface_snapshot_download

    requested_revision = None if revision == "<absent>" else revision
    requested_model_source = (
        str(cfg_obj.get("model_source")).strip()
        if isinstance(cfg_obj.get("model_source"), str) and str(cfg_obj.get("model_source")).strip()
        else "<absent>"
    )
    local_snapshot_mode = "local_session_cache"
    try:
        local_snapshot_path = Path(
            str(
                snapshot_download(
                    repo_id=model_id,
                    revision=requested_revision,
                    cache_dir=str(layout["local_hf_hub_cache"]),
                    local_files_only=True,
                )
            )
        ).resolve()
    except Exception:
        local_snapshot_mode = "downloaded_this_session"
        local_snapshot_path = Path(
            str(
                snapshot_download(
                    repo_id=model_id,
                    revision=requested_revision,
                    cache_dir=str(layout["local_hf_hub_cache"]),
                )
            )
        ).resolve()

    if not local_snapshot_path.exists() or not local_snapshot_path.is_dir():
        raise RuntimeError(f"local model snapshot missing or invalid: {local_snapshot_path}")

    model_source_binding: Dict[str, Any] = {
        "binding_source": NOTEBOOK_MODEL_SNAPSHOT_BINDING_SOURCE,
        "binding_env_var": NOTEBOOK_MODEL_SNAPSHOT_ENV_VAR,
        "binding_status": "ready_for_env_binding",
        "binding_reason": "local_session_snapshot_prepared",
        "model_snapshot_path": normalize_path_value(local_snapshot_path),
        "requested_model_id": model_id,
        "requested_model_source": requested_model_source,
        "requested_hf_revision": revision,
    }
    model_audit_summary: Dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "model_snapshot_path": normalize_path_value(local_snapshot_path),
        "snapshot_exists": True,
        "snapshot_source": local_snapshot_mode,
        "snapshot_path_basename": local_snapshot_path.name,
        "local_hf_home": normalize_path_value(layout["local_hf_home"]),
        "local_hf_hub_cache": normalize_path_value(layout["local_hf_hub_cache"]),
        "local_transformers_cache": normalize_path_value(layout["local_transformers_cache"]),
        "cache_reuse_mode": local_snapshot_mode,
        "binding_status": model_source_binding["binding_status"],
        "binding_source": model_source_binding["binding_source"],
        "binding_reason": model_source_binding["binding_reason"],
        "model_source_binding": dict(model_source_binding),
    }

    return {
        "model_id": model_id,
        "revision": revision,
        "semantic_model_source": semantic_model_source or "<absent>",
        "semantic_model_url": semantic_weight_url or "<absent>",
        "persistent_hf_root": normalize_path_value(layout["persistent_hf_root"]),
        "persistent_snapshot_path": "<disabled>",
        "local_hf_home": normalize_path_value(layout["local_hf_home"]),
        "local_hf_hub_cache": normalize_path_value(layout["local_hf_hub_cache"]),
        "local_transformers_cache": normalize_path_value(layout["local_transformers_cache"]),
        "local_snapshot_path": normalize_path_value(local_snapshot_path),
        "persistent_inspyrenet_path": normalize_path_value(persistent_inspyrenet_path),
        "repo_inspyrenet_path": normalize_path_value(repo_inspyrenet_path),
        "weight_cache_mode": weight_cache_mode,
        "persistent_snapshot_mode": "disabled",
        "local_snapshot_mode": local_snapshot_mode,
        "cache_reuse_mode": local_snapshot_mode,
        "snapshot_source": local_snapshot_mode,
        "model_source_binding": model_source_binding,
        "model_audit_summary": model_audit_summary,
    }


def _normalize_model_snapshot_path(path_value: str) -> str:
    """
    功能：规范化 notebook 传入的模型快照目录路径。

    Normalize the model snapshot directory path forwarded by the notebook.

    Args:
        path_value: Raw snapshot path string.

    Returns:
        Normalized absolute path string.
    """
    if not isinstance(path_value, str) or not path_value.strip():
        raise TypeError("path_value must be non-empty str")

    snapshot_path = Path(path_value.strip()).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = (REPO_ROOT / snapshot_path).resolve()
    else:
        snapshot_path = snapshot_path.resolve()
    return normalize_path_value(snapshot_path)


def resolve_notebook_model_snapshot_binding(
    cfg_obj: Mapping[str, Any],
    env_mapping: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """
    功能：解析 notebook bootstrap 提供的模型快照绑定信息。

    Resolve the model snapshot binding forwarded from the notebook bootstrap
    stage.

    Args:
        cfg_obj: Runtime configuration mapping.
        env_mapping: Optional environment mapping. When omitted, os.environ is
            consulted.

    Returns:
        Structured model snapshot binding summary.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    if env_mapping is not None and not isinstance(env_mapping, Mapping):
        raise TypeError("env_mapping must be Mapping[str, str] or None")

    model_identity = resolve_model_identity(cfg_obj)
    source_env = os.environ if env_mapping is None else env_mapping
    raw_snapshot_path = source_env.get(NOTEBOOK_MODEL_SNAPSHOT_ENV_VAR)
    binding_summary: Dict[str, Any] = {
        "binding_source": NOTEBOOK_MODEL_SNAPSHOT_BINDING_SOURCE,
        "binding_env_var": NOTEBOOK_MODEL_SNAPSHOT_ENV_VAR,
        "binding_status": "absent",
        "binding_reason": "model_snapshot_env_var_absent",
        "model_snapshot_path": "<absent>",
        "requested_model_id": model_identity["model_id"],
        "requested_model_source": (
            cfg_obj.get("model_source")
            if isinstance(cfg_obj.get("model_source"), str) and str(cfg_obj.get("model_source")).strip()
            else "<absent>"
        ),
        "requested_hf_revision": model_identity["revision"],
    }
    if not isinstance(raw_snapshot_path, str) or not raw_snapshot_path.strip():
        return binding_summary

    normalized_snapshot_path = _normalize_model_snapshot_path(raw_snapshot_path)
    binding_summary["model_snapshot_path"] = normalized_snapshot_path
    snapshot_path_obj = Path(normalized_snapshot_path)
    if snapshot_path_obj.exists() and snapshot_path_obj.is_dir():
        binding_summary["binding_status"] = "bound"
        binding_summary["binding_reason"] = "model_snapshot_env_var_bound_to_runtime_config"
        return binding_summary

    binding_summary["binding_status"] = "invalid"
    binding_summary["binding_reason"] = "model_snapshot_env_var_path_missing_or_not_directory"
    return binding_summary


def apply_notebook_model_snapshot_binding(
    cfg_obj: Mapping[str, Any],
    env_mapping: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """
    功能：把 notebook 模型快照绑定固化到运行时配置副本。

    Apply the notebook-provided model snapshot binding to a runtime config
    copy.

    Args:
        cfg_obj: Runtime configuration mapping.
        env_mapping: Optional environment mapping. When omitted, os.environ is
            consulted.

    Returns:
        Runtime configuration copy with optional model snapshot binding fields.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")

    cfg_copy = dict(cfg_obj)
    binding_summary = resolve_notebook_model_snapshot_binding(cfg_obj, env_mapping)
    if binding_summary["binding_status"] not in {"bound", "invalid"}:
        return cfg_copy

    cfg_copy["model_snapshot_path"] = binding_summary["model_snapshot_path"]
    cfg_copy["model_source_binding"] = binding_summary
    return cfg_copy


def resolve_attestation_env_var_names(cfg_obj: Mapping[str, Any]) -> Dict[str, str]:
    """
    功能：解析 attestation 配置中的环境变量名。 

    Resolve attestation environment-variable names from the runtime config.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Mapping from attestation config keys to environment-variable names.
    """
    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    env_names: Dict[str, str] = {}
    for config_key in ATTESTATION_ENV_VAR_LENGTHS:
        raw_value = attestation_cfg.get(config_key)
        if isinstance(raw_value, str) and raw_value.strip():
            env_names[config_key] = raw_value.strip()
    return env_names


def _resolve_attestation_env_specs(
    cfg_obj: Mapping[str, Any],
    *,
    require_complete: bool,
) -> List[Dict[str, Any]]:
    """
    功能：解析 attestation secret 的规范集合。 

    Resolve the attestation secret specifications from the runtime config.

    Args:
        cfg_obj: Runtime configuration mapping.
        require_complete: Whether all configured attestation env vars must exist.

    Returns:
        Ordered attestation secret specification list.

    Raises:
        ValueError: If attestation is enabled and required env-var names are absent.
    """
    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    attestation_enabled = bool(attestation_cfg.get("enabled", False))
    env_var_names = resolve_attestation_env_var_names(cfg_obj)
    specs: List[Dict[str, Any]] = []
    missing_config_keys: List[str] = []
    for config_key, hex_length in ATTESTATION_ENV_VAR_LENGTHS.items():
        env_name = env_var_names.get(config_key)
        if env_name is None:
            missing_config_keys.append(config_key)
            continue
        specs.append(
            {
                "config_key": config_key,
                "env_name": env_name,
                "hex_length": hex_length,
            }
        )
    if attestation_enabled and require_complete and missing_config_keys:
        raise ValueError(
            "attestation.enabled requires env-var bindings for all secret roles: "
            f"missing={missing_config_keys}"
        )
    return specs


def validate_attestation_hex_secret(secret_value: str, expected_hex_length: int, label: str) -> str:
    """
    功能：校验 attestation secret 的十六进制格式与长度。 

    Validate the format and length of one attestation secret.

    Args:
        secret_value: Candidate secret string.
        expected_hex_length: Required hexadecimal length.
        label: Human-readable label used in error reporting.

    Returns:
        Canonical lowercase secret string.

    Raises:
        ValueError: If the secret is not valid lowercase/uppercase hexadecimal with the required length.
    """
    if not secret_value.strip():
        raise ValueError(f"{label} must be non-empty hex string")
    if expected_hex_length <= 0:
        raise TypeError("expected_hex_length must be positive int")
    if not label:
        raise TypeError("label must be non-empty str")

    normalized_secret = secret_value.strip().lower()
    if len(normalized_secret) != expected_hex_length:
        raise ValueError(
            f"{label} must be exactly {expected_hex_length} hex chars: actual={len(normalized_secret)}"
        )
    if re.fullmatch(r"[0-9a-f]+", normalized_secret) is None:
        raise ValueError(f"{label} must contain only hexadecimal characters")
    return normalized_secret


def _mask_attestation_secret(secret_value: str) -> str:
    """
    功能：构造不泄漏真实值的 masked secret。 

    Build a masked representation for one attestation secret.

    Args:
        secret_value: Canonical secret string.

    Returns:
        Masked secret string.
    """
    if not secret_value:
        raise TypeError("secret_value must be non-empty str")
    if len(secret_value) <= 8:
        return "*" * len(secret_value)
    return f"{secret_value[:4]}...{secret_value[-4:]}"


def generate_attestation_session_env(cfg_obj: Mapping[str, Any]) -> Dict[str, str]:
    """
    功能：生成本次会话使用的临时 attestation secrets。 

    Generate session-scoped attestation secrets for the configured env vars.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Mapping from env-var name to generated secret value.
    """
    specs = _resolve_attestation_env_specs(cfg_obj, require_complete=True)
    generated_payload: Dict[str, str] = {}
    for spec in specs:
        env_name = str(spec["env_name"])
        hex_length = int(spec["hex_length"])
        generated_payload[env_name] = secrets.token_hex(hex_length // 2)
    return generated_payload


def _resolve_attestation_secret_paths(drive_project_root: Path) -> Dict[str, Path]:
    """
    功能：解析 attestation secret 与 masked info 的固定路径。 

    Resolve the canonical attestation secret and info paths under the Drive project root.

    Args:
        drive_project_root: Google Drive project root.

    Returns:
        Mapping with secrets_root, attestation_env_path, and attestation_env_info_path.
    """
    secrets_root = drive_project_root / "secrets"
    return {
        "secrets_root": secrets_root,
        "attestation_env_path": secrets_root / ATTESTATION_ENV_FILE_NAME,
        "attestation_env_info_path": secrets_root / ATTESTATION_ENV_INFO_FILE_NAME,
    }


def write_attestation_env_file(attestation_env_path: Path, env_payload: Mapping[str, Any]) -> Path:
    """
    功能：写出真实 attestation_env.json。 

    Persist the real attestation secret mapping to the canonical JSON file.

    Args:
        attestation_env_path: Destination JSON path.
        env_payload: Env-var to secret mapping.

    Returns:
        Written JSON path.
    """
    normalized_payload: Dict[str, str] = {}
    for env_name, secret_value in env_payload.items():
        if not isinstance(env_name, str) or not env_name:
            raise ValueError("attestation env payload keys must be non-empty str")
        if not isinstance(secret_value, str) or not secret_value:
            raise ValueError(f"attestation env payload value must be non-empty str: {env_name}")
        normalized_payload[env_name] = secret_value

    write_json_atomic(attestation_env_path, normalized_payload)
    return attestation_env_path


def write_masked_attestation_env_info(
    attestation_env_info_path: Path,
    cfg_obj: Mapping[str, Any],
    env_payload: Mapping[str, Any],
    *,
    status: str,
    reused_existing: bool,
    generated: bool,
    attestation_env_path: Optional[Path] = None,
) -> Path:
    """
    功能：写出不包含真实 secret 的 masked info 文件。 

    Persist the masked attestation info payload without leaking the real secret values.

    Args:
        attestation_env_info_path: Destination JSON path.
        cfg_obj: Runtime configuration mapping.
        env_payload: Valid env-var to secret mapping.
        status: Bootstrap status token.
        reused_existing: Whether the secret file was reused.
        generated: Whether the secret file was generated in the current session.
        attestation_env_path: Optional real secret-file path.

    Returns:
        Written info JSON path.
    """
    if not status:
        raise TypeError("status must be non-empty str")

    required_env_vars = [spec["env_name"] for spec in _resolve_attestation_env_specs(cfg_obj, require_complete=True)]
    masked_values: Dict[str, str] = {}
    for env_name in required_env_vars:
        secret_value = env_payload.get(env_name)
        if not isinstance(secret_value, str) or not secret_value:
            raise ValueError(f"masked attestation info requires non-empty secret for {env_name}")
        masked_values[str(env_name)] = _mask_attestation_secret(secret_value)

    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    info_payload: Dict[str, Any] = {
        "enabled": bool(attestation_cfg.get("enabled", False)),
        "status": status,
        "generated": generated,
        "reused_existing": reused_existing,
        "attestation_env_path": normalize_path_value(attestation_env_path),
        "required_env_vars": required_env_vars,
        "masked_values": masked_values,
    }
    write_json_atomic(attestation_env_info_path, info_payload)
    return attestation_env_info_path


def restore_attestation_env_from_file(cfg_obj: Mapping[str, Any], attestation_env_path: Path) -> Dict[str, str]:
    """
    功能：从 secrets 文件恢复并注入 attestation 环境变量。 

    Restore attestation secrets from the canonical JSON file and inject them into os.environ.

    Args:
        cfg_obj: Runtime configuration mapping.
        attestation_env_path: Canonical secret-file path.

    Returns:
        Validated env-var to secret mapping.

    Raises:
        FileNotFoundError: If the secret file is missing.
        ValueError: If the secret file is malformed or fails validation.
    """
    if not attestation_env_path.exists() or not attestation_env_path.is_file():
        raise FileNotFoundError(f"attestation secret file not found: {normalize_path_value(attestation_env_path)}")

    payload = json.loads(attestation_env_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("attestation secret file must have a JSON object root")
    payload_obj = cast(Dict[str, Any], payload)

    restored_payload: Dict[str, str] = {}
    for spec in _resolve_attestation_env_specs(cfg_obj, require_complete=True):
        env_name = str(spec["env_name"])
        secret_value = payload_obj.get(env_name)
        validated_secret = validate_attestation_hex_secret(
            str(secret_value) if isinstance(secret_value, str) else "",
            int(spec["hex_length"]),
            env_name,
        )
        restored_payload[env_name] = validated_secret
        os.environ[env_name] = validated_secret
    return restored_payload


def ensure_attestation_env_bootstrap(
    cfg_obj: Mapping[str, Any],
    drive_project_root: Path,
    *,
    allow_generate: bool,
    allow_missing: bool = False,
) -> Dict[str, Any]:
    """
    功能：复用或引导 attestation secret，并返回 masked summary。 

    Reuse or bootstrap attestation secrets, inject them into os.environ, and return a masked summary.

    Args:
        cfg_obj: Runtime configuration mapping.
        drive_project_root: Google Drive project root.
        allow_generate: Whether missing secrets may be generated for the current session.
        allow_missing: Whether a missing secret file should return a non-blocking summary.

    Returns:
        Masked attestation environment summary suitable for notebooks and stage wrappers.

    Raises:
        FileNotFoundError: If secrets are missing and generation is not allowed.
        ValueError: If the existing secret file is malformed.
    """
    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    attestation_enabled = bool(attestation_cfg.get("enabled", False))
    env_names = resolve_attestation_env_var_names(cfg_obj)
    path_mapping = _resolve_attestation_secret_paths(drive_project_root)
    attestation_env_path = path_mapping["attestation_env_path"]
    attestation_env_info_path = path_mapping["attestation_env_info_path"]

    summary: Dict[str, Any] = {
        "enabled": attestation_enabled,
        "status": "disabled" if not attestation_enabled else "pending",
        "generated": False,
        "reused_existing": False,
        "required_env_vars": list(env_names.values()),
        "present_env_vars": [],
        "missing_env_vars": list(env_names.values()),
        "masked_values": {},
        "attestation_env_path": normalize_path_value(attestation_env_path),
        "attestation_env_info_path": normalize_path_value(attestation_env_info_path),
    }
    if not attestation_enabled:
        return summary

    env_payload: Dict[str, str]
    if attestation_env_path.exists():
        env_payload = restore_attestation_env_from_file(cfg_obj, attestation_env_path)
        summary["status"] = "reused"
        summary["reused_existing"] = True
    elif allow_generate:
        env_payload = generate_attestation_session_env(cfg_obj)
        ensure_directory(path_mapping["secrets_root"])
        write_attestation_env_file(attestation_env_path, env_payload)
        for env_name, secret_value in env_payload.items():
            os.environ[env_name] = secret_value
        summary["status"] = "generated"
        summary["generated"] = True
    elif allow_missing:
        summary["status"] = "missing"
        return summary
    else:
        raise FileNotFoundError(
            f"attestation secret file not found: {normalize_path_value(attestation_env_path)}"
        )

    write_masked_attestation_env_info(
        attestation_env_info_path,
        cfg_obj,
        env_payload,
        status=str(summary["status"]),
        reused_existing=bool(summary["reused_existing"]),
        generated=bool(summary["generated"]),
        attestation_env_path=attestation_env_path,
    )

    final_summary = collect_attestation_env_summary(cfg_obj)
    final_summary.update(
        {
            "status": summary["status"],
            "generated": summary["generated"],
            "reused_existing": summary["reused_existing"],
            "masked_values": {
                env_name: _mask_attestation_secret(secret_value)
                for env_name, secret_value in env_payload.items()
            },
            "attestation_env_path": normalize_path_value(attestation_env_path),
            "attestation_env_info_path": normalize_path_value(attestation_env_info_path),
        }
    )
    return final_summary


def write_json_atomic(path_obj: Path, payload: Mapping[str, Any]) -> None:
    """
    功能：稳定写出 JSON 文件。

    Persist a mapping as formatted JSON with parent-directory creation.

    Args:
        path_obj: Destination JSON path.
        payload: Mapping payload.

    Returns:
        None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    ensure_directory(path_obj.parent)
    path_obj.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def write_json_atomic_compact(path_obj: Path, payload: Mapping[str, Any]) -> None:
    """
    功能：以紧凑 JSON 文本稳定写出文件。

    Persist a mapping as compact JSON with parent-directory creation.

    Args:
        path_obj: Destination JSON path.
        payload: Mapping payload.

    Returns:
        None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    ensure_directory(path_obj.parent)
    path_obj.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _truncate_output_tail(text: str | None, *, limit: int = RUNTIME_DIAGNOSTICS_STDIO_TAIL_LIMIT) -> str | None:
    """
    功能：截断 stdout / stderr 尾部文本。

    Truncate one stdout or stderr payload to a stable tail section.

    Args:
        text: Full subprocess output text.
        limit: Maximum retained character count.

    Returns:
        Truncated tail text, or None when unavailable.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        raise TypeError("text must be str or None")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise TypeError("limit must be positive int")
    if len(text) <= limit:
        return text
    return f"...\n{text[-limit:]}"


def _normalize_runtime_count_summary(count_summary: Mapping[str, Any]) -> Dict[str, int]:
    """
    功能：规范化 stage runtime diagnostics 的 count_summary。 

    Normalize one runtime diagnostics count summary into non-negative integers.

    Args:
        count_summary: Stage-specific count mapping.

    Returns:
        Normalized count mapping.
    """
    if not isinstance(count_summary, Mapping):
        raise TypeError("count_summary must be Mapping")

    normalized: Dict[str, int] = {}
    for raw_key, raw_value in count_summary.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise TypeError("count_summary keys must be non-empty str")
        if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value < 0:
            raise TypeError(f"count_summary[{raw_key!r}] must be non-negative int")
        normalized[raw_key.strip()] = int(raw_value)

    if not normalized:
        raise ValueError("count_summary must not be empty")
    return normalized


def build_stage_runtime_workload_summary(
    *,
    unit_label: str,
    unit_count: int,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """
    功能：构造 stage runtime diagnostics 的 workload_summary。 

    Build the workload summary for one stage runtime diagnostics payload.

    Args:
        unit_label: Canonical workload unit label.
        unit_count: Number of workload units.
        elapsed_seconds: Stage elapsed seconds.

    Returns:
        Workload summary mapping with per-unit elapsed time.
    """
    if not isinstance(unit_label, str) or not unit_label.strip():
        raise TypeError("unit_label must be non-empty str")
    if not isinstance(unit_count, int) or isinstance(unit_count, bool) or unit_count < 0:
        raise TypeError("unit_count must be non-negative int")
    if not isinstance(elapsed_seconds, (int, float)) or isinstance(elapsed_seconds, bool) or float(elapsed_seconds) < 0:
        raise TypeError("elapsed_seconds must be non-negative float")

    normalized_elapsed_seconds = float(elapsed_seconds)
    elapsed_seconds_per_unit = None
    if unit_count > 0:
        elapsed_seconds_per_unit = normalized_elapsed_seconds / float(unit_count)

    return {
        "unit_label": unit_label.strip(),
        "unit_count": int(unit_count),
        "elapsed_seconds_per_unit": elapsed_seconds_per_unit,
    }


def _normalize_runtime_workload_summary(workload_summary: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：规范化 stage runtime diagnostics 的 workload_summary。 

    Normalize one runtime diagnostics workload summary payload.

    Args:
        workload_summary: Raw workload summary mapping.

    Returns:
        Normalized workload summary mapping.
    """
    if not isinstance(workload_summary, Mapping):
        raise TypeError("workload_summary must be Mapping")

    unit_label = workload_summary.get("unit_label")
    unit_count = workload_summary.get("unit_count")
    elapsed_seconds_per_unit = workload_summary.get("elapsed_seconds_per_unit")

    if not isinstance(unit_label, str) or not unit_label.strip():
        raise TypeError("workload_summary.unit_label must be non-empty str")
    if not isinstance(unit_count, int) or isinstance(unit_count, bool) or unit_count < 0:
        raise TypeError("workload_summary.unit_count must be non-negative int")
    if elapsed_seconds_per_unit is not None and (
        not isinstance(elapsed_seconds_per_unit, (int, float))
        or isinstance(elapsed_seconds_per_unit, bool)
        or float(elapsed_seconds_per_unit) < 0
    ):
        raise TypeError("workload_summary.elapsed_seconds_per_unit must be non-negative float or None")

    normalized = {
        "unit_label": unit_label.strip(),
        "unit_count": int(unit_count),
        "elapsed_seconds_per_unit": (
            None if elapsed_seconds_per_unit is None else float(elapsed_seconds_per_unit)
        ),
    }
    for raw_key, raw_value in workload_summary.items():
        if raw_key in normalized:
            continue
        normalized[str(raw_key)] = raw_value
    return normalized


def build_stage_runtime_diagnostics_payload(
    *,
    stage_name: str,
    family_id: str,
    expected_output_label: str,
    expected_output_path: Path,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
    return_code: int,
    stdout_text: str | None,
    stderr_text: str | None,
    count_summary: Mapping[str, Any],
    workload_summary: Mapping[str, Any],
    shard_index: int | None = None,
    sample_role: str | None = None,
    gpu_session_peak_path: Path | None = None,
    monitor_status: str | None = None,
) -> Dict[str, Any]:
    """
    功能：构造 notebook stage 级运行时诊断 JSON 载荷。

    Build the stage-level runtime diagnostics payload for one notebook stage.

    Args:
        stage_name: Canonical stage name.
        family_id: Family identifier.
        expected_output_label: Human-readable expected output label.
        expected_output_path: Expected output path.
        started_at_utc: Stage start timestamp in UTC.
        finished_at_utc: Stage finish timestamp in UTC.
        elapsed_seconds: Stage elapsed seconds.
        return_code: Subprocess return code.
        stdout_text: Full subprocess stdout text.
        stderr_text: Full subprocess stderr text.
        count_summary: Stage-specific count mapping.
        workload_summary: Stage workload summary mapping.
        shard_index: Optional shard index.
        sample_role: Optional sample role.
        gpu_session_peak_path: Optional GPU peak summary path.
        monitor_status: Optional monitor status.

    Returns:
        Runtime diagnostics mapping ready for JSON persistence.
    """
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(expected_output_label, str) or not expected_output_label.strip():
        raise TypeError("expected_output_label must be non-empty str")
    if not isinstance(expected_output_path, Path):
        raise TypeError("expected_output_path must be Path")
    if not isinstance(started_at_utc, str) or not started_at_utc.strip():
        raise TypeError("started_at_utc must be non-empty str")
    if not isinstance(finished_at_utc, str) or not finished_at_utc.strip():
        raise TypeError("finished_at_utc must be non-empty str")
    if not isinstance(elapsed_seconds, (int, float)) or isinstance(elapsed_seconds, bool) or float(elapsed_seconds) < 0:
        raise TypeError("elapsed_seconds must be non-negative float")
    if not isinstance(return_code, int) or isinstance(return_code, bool):
        raise TypeError("return_code must be int")
    if stdout_text is not None and not isinstance(stdout_text, str):
        raise TypeError("stdout_text must be str or None")
    if stderr_text is not None and not isinstance(stderr_text, str):
        raise TypeError("stderr_text must be str or None")
    normalized_count_summary = _normalize_runtime_count_summary(count_summary)
    normalized_workload_summary = _normalize_runtime_workload_summary(workload_summary)
    if shard_index is not None and (
        not isinstance(shard_index, int)
        or isinstance(shard_index, bool)
        or shard_index < 0
    ):
        raise TypeError("shard_index must be non-negative int or None")
    if sample_role is not None and (not isinstance(sample_role, str) or not sample_role.strip()):
        raise TypeError("sample_role must be non-empty str or None")
    if gpu_session_peak_path is not None and not isinstance(gpu_session_peak_path, Path):
        raise TypeError("gpu_session_peak_path must be Path or None")
    if monitor_status is not None and (not isinstance(monitor_status, str) or not monitor_status.strip()):
        raise TypeError("monitor_status must be non-empty str or None")

    return {
        "artifact_type": "paper_workflow_stage_runtime_diagnostics",
        "schema_version": RUNTIME_DIAGNOSTICS_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "stage_name": stage_name.strip(),
        "family_id": family_id.strip(),
        "sample_role": sample_role.strip() if isinstance(sample_role, str) else None,
        "shard_index": shard_index,
        "expected_output_label": expected_output_label.strip(),
        "expected_output_path": normalize_path_value(expected_output_path),
        "expected_output_exists": expected_output_path.exists(),
        "started_at_utc": started_at_utc.strip(),
        "finished_at_utc": finished_at_utc.strip(),
        "elapsed_seconds": float(elapsed_seconds),
        "count_summary": normalized_count_summary,
        "workload_summary": normalized_workload_summary,
        "return_code": int(return_code),
        "stdout_tail": _truncate_output_tail(stdout_text),
        "stderr_tail": _truncate_output_tail(stderr_text),
        "gpu_session_peak_path": normalize_path_value(gpu_session_peak_path) if isinstance(gpu_session_peak_path, Path) else None,
        "monitor_status": monitor_status.strip() if isinstance(monitor_status, str) else None,
    }


def write_stage_runtime_diagnostics(
    *,
    diagnostics_path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """
    功能：写出 notebook stage 级运行时诊断 JSON。

    Persist one notebook stage runtime diagnostics JSON payload.

    Args:
        diagnostics_path: Runtime diagnostics output path.
        payload: Diagnostics mapping payload.

    Returns:
        Written diagnostics path.
    """
    if not isinstance(diagnostics_path, Path):
        raise TypeError("diagnostics_path must be Path")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    write_json_atomic(diagnostics_path, payload)
    return diagnostics_path


def compute_mapping_sha256(payload: Mapping[str, Any]) -> str:
    """
    功能：计算 JSON 映射对象的规范 SHA256。

    Compute a canonical SHA256 digest for one JSON-like mapping.

    Args:
        payload: Mapping payload.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    serialized = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_json_dict(path_obj: Path) -> Dict[str, Any]:
    """
    功能：读取 JSON 对象文件。

    Read a JSON file and require its root node to be a mapping.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed mapping, or an empty mapping when unavailable or invalid.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        return {}
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return cast(Dict[str, Any], payload) if isinstance(payload, dict) else {}


def read_optional_json(path_obj: Path) -> Dict[str, Any] | None:
    """
    功能：读取可选 JSON 对象文件。

    Read an optional JSON file and return None when unavailable or invalid.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed mapping, or None when missing, not a file, invalid, or not a mapping.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        return None
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception:
        return None
    return cast(Dict[str, Any], payload) if isinstance(payload, dict) else None


def read_required_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    功能：严格读取必需 JSON 对象文件。

    Read one required JSON file and require its root node to be a mapping.

    Args:
        path_obj: JSON file path.
        label: Human-readable label used in error reporting.

    Returns:
        Parsed JSON mapping.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the JSON root is not a mapping.
    """
    if not label:
        raise TypeError("label must be non-empty str")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return cast(Dict[str, Any], payload)


def load_yaml_mapping(path_obj: Path) -> Dict[str, Any]:
    """
    功能：读取 YAML 映射配置。

    Read a YAML file and require its root node to be a mapping.

    Args:
        path_obj: YAML path.

    Returns:
        Parsed mapping.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"yaml file not found: {path_obj}")
    payload = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be mapping: {path_obj}")
    return cast(Dict[str, Any], payload)


def resolve_model_identity(cfg_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：解析配置中的模型标识与 revision。

    Resolve the effective model identifier and revision from the runtime config.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Mapping with model_id and revision fields.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    model_node = cfg_obj.get("model")
    model_cfg = cast(Dict[str, Any], model_node) if isinstance(model_node, dict) else {}
    model_id = model_cfg.get("model_id") if isinstance(model_cfg.get("model_id"), str) else cfg_obj.get("model_id")
    revision = model_cfg.get("revision") if isinstance(model_cfg.get("revision"), str) else cfg_obj.get("hf_revision")
    return {
        "model_id": model_id if isinstance(model_id, str) and model_id.strip() else "<absent>",
        "revision": revision if isinstance(revision, str) and revision.strip() else "<absent>",
    }


def write_yaml_mapping(path_obj: Path, payload: Mapping[str, Any]) -> None:
    """
    功能：写出 YAML 映射配置。

    Persist a mapping as YAML with stable parent-directory creation.

    Args:
        path_obj: Destination YAML path.
        payload: Mapping payload.

    Returns:
        None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    ensure_directory(path_obj.parent)
    path_obj.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def compute_file_sha256(path_obj: Path) -> str:
    """
    功能：计算单文件 SHA256。

    Compute the SHA256 digest of one file.

    Args:
        path_obj: File path.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"file not found for sha256: {path_obj}")

    hasher = hashlib.sha256()
    with path_obj.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_file(source_path: Path, destination_path: Path) -> Path:
    """
    功能：复制文件并保留元数据。

    Copy one file to the destination path with metadata preservation.

    Args:
        source_path: Existing source file.
        destination_path: Destination file path.

    Returns:
        Destination file path.
    """
    if not isinstance(source_path, Path):
        raise TypeError("source_path must be Path")
    if not isinstance(destination_path, Path):
        raise TypeError("destination_path must be Path")
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")
    ensure_directory(destination_path.parent)
    shutil.copy2(source_path, destination_path)
    return destination_path


def validate_path_within_base(base_path: Path, candidate_path: Path, label: str) -> None:
    """
    功能：校验路径位于受控基础目录内。

    Validate that a candidate path is located under a controlled base directory.

    Args:
        base_path: Controlled base directory.
        candidate_path: Candidate path to validate.
        label: Human-readable label for error reporting.

    Returns:
        None.
    """
    if not isinstance(base_path, Path):
        raise TypeError("base_path must be Path")
    if not isinstance(candidate_path, Path):
        raise TypeError("candidate_path must be Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    try:
        candidate_path.resolve().relative_to(base_path.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be under {base_path}, got {candidate_path}") from exc


def extract_zip_archive(zip_path: Path, destination_root: Path) -> List[str]:
    """
    功能：解压 ZIP 包到目标目录。

    Extract a ZIP archive into a clean destination directory.

    Args:
        zip_path: ZIP archive path.
        destination_root: Extraction directory.

    Returns:
        Relative file names extracted from the archive.
    """
    if not isinstance(zip_path, Path):
        raise TypeError("zip_path must be Path")
    if not isinstance(destination_root, Path):
        raise TypeError("destination_root must be Path")
    if not zip_path.exists() or not zip_path.is_file():
        raise FileNotFoundError(f"zip archive not found: {zip_path}")

    if destination_root.exists():
        shutil.rmtree(destination_root)
    ensure_directory(destination_root)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination_root)
        members = [name for name in archive.namelist() if name and not name.endswith("/")]
    return sorted(members)


def create_zip_archive_from_directory(source_root: Path, zip_path: Path) -> Path:
    """
    功能：将目录压缩为 ZIP 包。

    Create a ZIP archive from one directory using relative paths.

    Args:
        source_root: Directory to archive.
        zip_path: Destination ZIP path.

    Returns:
        Created ZIP path.
    """
    if not isinstance(source_root, Path):
        raise TypeError("source_root must be Path")
    if not isinstance(zip_path, Path):
        raise TypeError("zip_path must be Path")
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    ensure_directory(zip_path.parent)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_root.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=file_path.relative_to(source_root).as_posix())
    return zip_path


def resolve_stage_roots(drive_project_root: Path, stage_name: str, stage_run_id: str) -> Dict[str, Path]:
    """
    功能：构造阶段隔离的 Drive 目录布局。

    Build the stage-isolated Google Drive directory layout.

    Args:
        drive_project_root: Google Drive project root.
        stage_name: Stable stage name.
        stage_run_id: Unique stage run identifier.

    Returns:
        Mapping containing run_root, log_root, runtime_state_root, and export_root.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")

    return {
        "run_root": drive_project_root / "runs" / stage_name / stage_run_id,
        "log_root": drive_project_root / "logs" / stage_name / stage_run_id,
        "runtime_state_root": drive_project_root / "runtime_state" / stage_name / stage_run_id,
        "export_root": drive_project_root / "exports" / stage_name / stage_run_id,
    }


def resolve_source_lineage_paths(extracted_root: Path) -> Dict[str, Path]:
    """
    功能：解析解压 source package 中的标准 lineage 路径。

    Resolve the canonical source-lineage paths inside one extracted source
    package.

    Args:
        extracted_root: Extracted source-package root.

    Returns:
        Mapping of canonical source-lineage paths.
    """
    return {
        "source_stage_manifest_path": extracted_root / "artifacts" / "stage_manifest.json",
        "source_package_manifest_path": extracted_root / "artifacts" / "package_manifest.json",
        "source_runtime_config_snapshot_path": extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml",
        "source_thresholds_artifact_path": extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
    }


def resolve_source_prompt_snapshot_path(extracted_root: Path) -> str:
    """
    功能：解析 source package 中首个 prompt snapshot 文件。

    Resolve the first available prompt-snapshot file path under one extracted
    source package.

    Args:
        extracted_root: Extracted source-package root.

    Returns:
        Normalized prompt-snapshot path, or "<absent>" when unavailable.
    """
    prompt_root = extracted_root / "runtime_metadata" / "prompt_snapshot"
    if prompt_root.exists() and prompt_root.is_dir():
        for prompt_path in sorted(prompt_root.rglob("*")):
            if prompt_path.is_file():
                return normalize_path_value(prompt_path)
    return "<absent>"


def build_stage_package_filename(
    stage_name: str,
    stage_run_id: str,
    source_stage_run_id: Optional[str] = None,
) -> str:
    """
    功能：构造带 lineage 的阶段包文件名。

    Build a stage package filename that exposes run lineage in its basename.

    Args:
        stage_name: Stable stage name.
        stage_run_id: Current stage run identifier.
        source_stage_run_id: Optional upstream stage run identifier.

    Returns:
        ZIP file name.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")
    if source_stage_run_id is not None and (not isinstance(source_stage_run_id, str) or not source_stage_run_id):
        raise TypeError("source_stage_run_id must be non-empty str or None")
    if source_stage_run_id:
        return f"{stage_name}__{stage_run_id}__from__{source_stage_run_id}.zip"
    return f"{stage_name}__{stage_run_id}.zip"


def build_failure_diagnostics_filename(stage_name: str, stage_run_id: str) -> str:
    """
    功能：构造 failure diagnostics ZIP 文件名。

    Build the ZIP filename used by one failure-diagnostics package.

    Args:
        stage_name: Stable stage name.
        stage_run_id: Current stage run identifier.

    Returns:
        Failure-diagnostics ZIP file name.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")
    return f"{stage_name}__{stage_run_id}__failure_diagnostics.zip"


def is_discoverable_formal_package_manifest(package_manifest: Any) -> bool:
    """
    功能：判断 manifest 是否属于可发现的正式 stage package。

    Determine whether a manifest belongs to a discoverable formal stage
    package.

    Args:
        package_manifest: Candidate package manifest mapping.

    Returns:
        True when the manifest represents a discoverable formal package.
    """
    if not isinstance(package_manifest, Mapping) or not package_manifest:
        return False

    manifest_obj = cast(Mapping[str, Any], package_manifest)

    stage_name = manifest_obj.get("stage_name")
    stage_run_id = manifest_obj.get("stage_run_id")
    package_filename = manifest_obj.get("package_filename")
    if not isinstance(stage_name, str) or not stage_name:
        return False
    if not isinstance(stage_run_id, str) or not stage_run_id:
        return False
    if not isinstance(package_filename, str) or not package_filename.endswith(".zip"):
        return False

    package_role = manifest_obj.get("package_role")
    if package_role not in {None, FORMAL_STAGE_PACKAGE_ROLE}:
        return False
    package_discovery_scope = manifest_obj.get("package_discovery_scope")
    if package_discovery_scope not in {None, FORMAL_PACKAGE_DISCOVERY_SCOPE}:
        return False
    return True


def _is_discoverable_formal_package_manifest(package_manifest: Mapping[str, Any]) -> bool:
    """
    功能：兼容内部旧调用的正式 package 判定包装。

    Preserve the legacy private helper name for internal callers while routing
    to the shared public helper.

    Args:
        package_manifest: Candidate package manifest mapping.

    Returns:
        True when the manifest represents a discoverable formal package.
    """
    return is_discoverable_formal_package_manifest(package_manifest)


def run_command_with_logs(
    command: Sequence[str],
    cwd: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> Dict[str, Any]:
    """
    功能：执行子进程并写出 stdout 与 stderr 日志。

    Execute a subprocess and persist stdout and stderr to explicit log files.

    Args:
        command: Command argument sequence.
        cwd: Working directory.
        stdout_log_path: Stdout log file path.
        stderr_log_path: Stderr log file path.

    Returns:
        Execution result mapping with return code and log paths.
    """
    if not isinstance(cwd, Path):
        raise TypeError("cwd must be Path")
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    command_list = [str(item) for item in command]
    result = subprocess.run(
        command_list,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_log_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_log_path.write_text(result.stderr or "", encoding="utf-8")
    return {
        "return_code": int(result.returncode),
        "stdout_log_path": stdout_log_path.as_posix(),
        "stderr_log_path": stderr_log_path.as_posix(),
        "command": command_list,
    }


def _run_git_command(repo_root: Path, args: List[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return "<absent>"
    return (result.stdout or "").strip() or "<absent>"


def collect_git_summary(repo_root: Path) -> Dict[str, Any]:
    """
    功能：收集 Git 摘要。

    Collect repository Git metadata for manifest emission.

    Args:
        repo_root: Repository root path.

    Returns:
        Git summary mapping.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    return {
        "remote": _run_git_command(repo_root, ["remote", "get-url", "origin"]),
        "branch": _run_git_command(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run_git_command(repo_root, ["rev-parse", "HEAD"]),
        "commit_short": _run_git_command(repo_root, ["rev-parse", "--short=8", "HEAD"]),
    }


def collect_python_summary() -> Dict[str, Any]:
    """
    功能：收集 Python 运行环境摘要。

    Collect Python runtime metadata.

    Args:
        None.

    Returns:
        Python runtime summary mapping.
    """
    return {
        "python_version": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def collect_cuda_summary() -> Dict[str, Any]:
    """
    功能：收集 CUDA 与 GPU 摘要。

    Collect CUDA and GPU metadata without failing when hardware is absent.

    Args:
        None.

    Returns:
        CUDA and GPU summary mapping.
    """
    summary: Dict[str, Any] = {
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "nvidia_smi_path": shutil.which("nvidia-smi") or "<absent>",
        "gpu_name": "<absent>",
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    if summary["nvidia_smi_available"]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            gpu_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
            summary["gpu_query_lines"] = gpu_lines
            if gpu_lines:
                summary["gpu_name"] = gpu_lines[0]
    try:
        import torch

        summary["cuda_available"] = bool(torch.cuda.is_available())
        summary["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            summary["torch_gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        summary["torch_gpu_name"] = "<unavailable>"
    return summary


def collect_attestation_env_summary(cfg_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：收集 attestation 环境变量状态摘要。

    Collect presence-only attestation environment metadata.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Attestation environment status summary.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    attestation_node = cfg_obj.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    env_names = list(resolve_attestation_env_var_names(cfg_obj).values())
    return {
        "enabled": bool(attestation_cfg.get("enabled", False)),
        "required_env_vars": env_names,
        "present_env_vars": [item for item in env_names if os.environ.get(item)],
        "missing_env_vars": [item for item in env_names if not os.environ.get(item)],
    }


def collect_model_summary(cfg_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：收集模型配置摘要。

    Collect lightweight model metadata without hashing the full cache directory.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Model metadata summary.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    model_identity = resolve_model_identity(cfg_obj)
    summary: Dict[str, Any] = {
        "model_id": model_identity["model_id"],
        "model_source": cfg_obj.get("model_source"),
        "hf_revision": model_identity["revision"],
        "model_snapshot_path": (
            cfg_obj.get("model_snapshot_path")
            if isinstance(cfg_obj.get("model_snapshot_path"), str) and str(cfg_obj.get("model_snapshot_path")).strip()
            else "<absent>"
        ),
        "hf_home": os.environ.get("HF_HOME", "<absent>"),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE", "<absent>"),
        "cache_scan_status": "not_attempted",
    }
    model_source_binding = cfg_obj.get("model_source_binding")
    if isinstance(model_source_binding, Mapping):
        summary["model_source_binding_status"] = model_source_binding.get("binding_status", "<absent>")
        summary["model_source_binding_reason"] = model_source_binding.get("binding_reason", "<absent>")
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        repo_id = summary.get("model_id")
        repos = [repo.repo_id for repo in cache_info.repos]
        summary["cache_scan_status"] = "ok"
        summary["cache_repo_present"] = bool(isinstance(repo_id, str) and repo_id in repos)
    except Exception:
        summary["cache_scan_status"] = "unavailable"
        summary["cache_repo_present"] = False
    return summary


def collect_weight_summary(repo_root: Path, cfg_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """
    功能：收集 InSPyReNet 权重摘要。

    Collect a concise summary for the semantic mask weight file.

    Args:
        repo_root: Repository root path.
        cfg_obj: Runtime configuration mapping.

    Returns:
        Weight metadata summary.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")

    mask_cfg = cast(Dict[str, Any], cfg_obj.get("mask")) if isinstance(cfg_obj.get("mask"), dict) else {}
    semantic_model_path = mask_cfg.get("semantic_model_path")
    summary: Dict[str, Any] = {
        "semantic_model_path": semantic_model_path,
        "exists": False,
        "sha256": "<absent>",
        "size_bytes": None,
    }
    if isinstance(semantic_model_path, str) and semantic_model_path:
        path_obj = (repo_root / semantic_model_path).resolve()
        summary["resolved_path"] = path_obj.as_posix()
        if path_obj.exists() and path_obj.is_file():
            summary["exists"] = True
            summary["size_bytes"] = int(path_obj.stat().st_size)
            summary["sha256"] = compute_file_sha256(path_obj)
    return summary


def build_directory_digest_summary(directory_path: Path, max_entries: int = 24) -> Dict[str, Any]:
    """
    功能：构建目录级清单摘要。

    Build a concise digest summary for a directory tree.

    Args:
        directory_path: Directory path to summarize.
        max_entries: Maximum number of file entries to include.

    Returns:
        Directory digest summary mapping.
    """
    if not isinstance(directory_path, Path):
        raise TypeError("directory_path must be Path")
    if not isinstance(max_entries, int) or max_entries <= 0:
        raise TypeError("max_entries must be positive int")

    summary: Dict[str, Any] = {
        "root_path": normalize_path_value(directory_path),
        "exists": bool(directory_path.exists() and directory_path.is_dir()),
        "file_count": 0,
        "entries": [],
        "manifest_sha256": "<absent>",
    }
    if not directory_path.exists() or not directory_path.is_dir():
        return summary

    entries: List[Dict[str, Any]] = []
    for file_path in sorted(directory_path.rglob("*")):
        if not file_path.is_file():
            continue
        entries.append(
            {
                "relative_path": file_path.relative_to(directory_path).as_posix(),
                "size_bytes": int(file_path.stat().st_size),
                "sha256": compute_file_sha256(file_path),
            }
        )
    summary["file_count"] = len(entries)
    summary["entries"] = entries[:max_entries]
    summary["manifest_sha256"] = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return summary


def copy_prompt_snapshot(repo_root: Path, cfg_obj: Mapping[str, Any], destination_root: Path) -> Dict[str, Any]:
    """
    功能：复制 prompt 文件快照。

    Copy the configured prompt file into runtime metadata for packaging and lineage.

    Args:
        repo_root: Repository root path.
        cfg_obj: Runtime configuration mapping.
        destination_root: Prompt snapshot directory.

    Returns:
        Prompt snapshot summary.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    if not isinstance(destination_root, Path):
        raise TypeError("destination_root must be Path")

    prompt_value = cfg_obj.get("inference_prompt_file")
    summary: Dict[str, Any] = {
        "source_path": "<absent>",
        "snapshot_path": "<absent>",
        "exists": False,
    }
    if not isinstance(prompt_value, str) or not prompt_value:
        return summary

    source_path = (repo_root / prompt_value).resolve()
    summary["source_path"] = source_path.as_posix()
    if not source_path.exists() or not source_path.is_file():
        return summary

    destination_path = ensure_directory(destination_root) / source_path.name
    copy_file(source_path, destination_path)
    summary["snapshot_path"] = destination_path.as_posix()
    summary["exists"] = True
    summary["sha256"] = compute_file_sha256(destination_path)
    return summary


def build_package_index(file_paths: Iterable[Path], package_root: Path) -> Dict[str, Any]:
    """
    功能：构建 package 内部索引。

    Build a package index with relative paths, sizes, and file digests.

    Args:
        file_paths: Files included in the package staging directory.
        package_root: Package staging root.

    Returns:
        Package index mapping.
    """
    if not isinstance(package_root, Path):
        raise TypeError("package_root must be Path")

    entries: List[Dict[str, Any]] = []
    for file_path in sorted(file_paths):
        if not file_path.exists() or not file_path.is_file():
            continue
        entries.append(
            {
                "relative_path": file_path.relative_to(package_root).as_posix(),
                "size_bytes": int(file_path.stat().st_size),
                "sha256": compute_file_sha256(file_path),
            }
        )
    return {
        "created_at": utc_now_iso(),
        "file_count": len(entries),
        "files": entries,
    }


def resolve_export_package_manifest_path(export_root: Path) -> Path:
    """
    功能：返回 stage export 目录中的外部 package_manifest 路径。

    Return the canonical external package_manifest path for one export directory.

    Args:
        export_root: Stage export directory.

    Returns:
        External package_manifest path.
    """
    if not isinstance(export_root, Path):
        raise TypeError("export_root must be Path")
    return export_root / "package_manifest.json"


def resolve_export_package_index_path(export_root: Path) -> Path:
    """
    功能：返回 stage export 目录中的外部 package_index 路径。

    Return the canonical external package_index path for one export directory.

    Args:
        export_root: Stage export directory.

    Returns:
        External package_index path.
    """
    if not isinstance(export_root, Path):
        raise TypeError("export_root must be Path")
    return export_root / "package_index.json"


def read_json_from_zip(zip_path: Path, member_name: str) -> Dict[str, Any]:
    """
    功能：从 ZIP 中读取 JSON 对象。

    Read one JSON object member from a ZIP archive.

    Args:
        zip_path: ZIP archive path.
        member_name: Member path inside the archive.

    Returns:
        Parsed mapping or an empty mapping when unavailable.
    """
    if not isinstance(zip_path, Path):
        raise TypeError("zip_path must be Path")
    if not isinstance(member_name, str) or not member_name:
        raise TypeError("member_name must be non-empty str")
    if not zip_path.exists() or not zip_path.is_file():
        return {}
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(member_name, "r") as handle:
                payload = json.loads(handle.read().decode("utf-8"))
    except Exception:
        return {}
    return cast(Dict[str, Any], payload) if isinstance(payload, dict) else {}


def find_external_package_manifest(source_package_path: Path) -> Path | None:
    """
    功能：定位 ZIP 包同目录的外部 package_manifest。

    Locate the external package_manifest stored alongside a stage package ZIP.

    Args:
        source_package_path: Stage package ZIP path.

    Returns:
        External package_manifest path when present.
    """
    if not isinstance(source_package_path, Path):
        raise TypeError("source_package_path must be Path")
    candidate_paths = [
        source_package_path.parent / "package_manifest.json",
        source_package_path.with_suffix(f"{source_package_path.suffix}.package_manifest.json"),
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path
    return None


def probe_stage_package_policy(package_path: Path) -> Dict[str, Any]:
    """
    功能：探测 stage package 的 formal policy 元数据。

    Probe one stage package for external/internal manifest metadata and formal
    package eligibility.

    Args:
        package_path: Stage package ZIP path.

    Returns:
        Package-policy probe summary.
    """
    external_manifest_path = find_external_package_manifest(package_path)
    external_manifest = read_json_dict(external_manifest_path) if isinstance(external_manifest_path, Path) else {}
    internal_manifest = read_json_from_zip(package_path, "artifacts/package_manifest.json")
    stage_manifest = read_json_from_zip(package_path, "artifacts/stage_manifest.json")
    manifest_for_policy = external_manifest if external_manifest else internal_manifest
    package_role_raw = manifest_for_policy.get("package_role")
    package_discovery_scope_raw = manifest_for_policy.get("package_discovery_scope")
    package_role = package_role_raw.strip() if isinstance(package_role_raw, str) and package_role_raw.strip() else None
    package_discovery_scope = (
        package_discovery_scope_raw.strip()
        if isinstance(package_discovery_scope_raw, str) and package_discovery_scope_raw.strip()
        else None
    )
    diagnostics_like = (
        package_role in {FAILURE_DIAGNOSTICS_PACKAGE_ROLE, "failed_diagnostics_package"}
        or (isinstance(package_role, str) and "diagnostic" in package_role.lower())
        or package_discovery_scope == EXCLUDED_PACKAGE_DISCOVERY_SCOPE
    )
    formal_role_compatible = package_role in {None, FORMAL_STAGE_PACKAGE_ROLE}
    formal_discovery_scope_compatible = package_discovery_scope in {None, FORMAL_PACKAGE_DISCOVERY_SCOPE}
    explicit_non_formal = not (formal_role_compatible and formal_discovery_scope_compatible) and (
        package_role is not None or package_discovery_scope is not None
    )
    diagnostics_reference_paths: List[str] = []
    for candidate_path in [
        manifest_for_policy.get("diagnostics_manifest_path"),
        manifest_for_policy.get("diagnostics_summary_path"),
        manifest_for_policy.get("diagnostics_package_path"),
    ]:
        if isinstance(candidate_path, str) and candidate_path.strip() and candidate_path not in diagnostics_reference_paths:
            diagnostics_reference_paths.append(candidate_path)
    if isinstance(external_manifest_path, Path):
        diagnostics_reference_paths.append(normalize_path_value(external_manifest_path))

    return {
        "external_manifest_path": normalize_path_value(external_manifest_path) if isinstance(external_manifest_path, Path) else "<absent>",
        "external_manifest": external_manifest,
        "internal_manifest": internal_manifest,
        "stage_manifest": stage_manifest,
        "manifest_for_policy": manifest_for_policy,
        "package_policy_source": "external_manifest" if external_manifest else "internal_manifest" if internal_manifest else "absent",
        "package_role": package_role,
        "package_discovery_scope": package_discovery_scope,
        "diagnostics_like": diagnostics_like,
        "formal_role_compatible": formal_role_compatible,
        "formal_discovery_scope_compatible": formal_discovery_scope_compatible,
        "formal_package_eligible": is_discoverable_formal_package_manifest(manifest_for_policy),
        "explicit_non_formal": explicit_non_formal,
        "diagnostics_reference_paths": diagnostics_reference_paths,
    }


def validate_package_manifest_binding(
    package_path: Path,
    package_manifest: Mapping[str, Any],
    *,
    required_sha_match: bool,
) -> Dict[str, Any]:
    """
    功能：校验 package_manifest 与实际 ZIP 的绑定关系。

    Validate one package manifest against the actual ZIP file.

    Args:
        package_path: Stage package ZIP path.
        package_manifest: Candidate package manifest mapping.
        required_sha_match: Whether package_sha256 must match the ZIP digest.

    Returns:
        Validation summary mapping.
    """
    if not isinstance(package_path, Path):
        raise TypeError("package_path must be Path")
    if not isinstance(package_manifest, Mapping):
        raise TypeError("package_manifest must be Mapping")

    actual_sha256 = compute_file_sha256(package_path)
    manifest_package_sha256 = package_manifest.get("package_sha256")
    manifest_package_filename = package_manifest.get("package_filename")
    result = {
        "actual_sha256": actual_sha256,
        "manifest_package_sha256": manifest_package_sha256,
        "manifest_package_filename": manifest_package_filename,
        "sha256_match": bool(isinstance(manifest_package_sha256, str) and manifest_package_sha256 == actual_sha256),
        "filename_match": bool(isinstance(manifest_package_filename, str) and manifest_package_filename == package_path.name),
        "manifest_digest": compute_mapping_sha256(package_manifest),
    }
    if required_sha_match and not bool(result["sha256_match"]):
        raise ValueError(
            f"package sha256 mismatch: expected={manifest_package_sha256} actual={actual_sha256} path={package_path}"
        )
    if isinstance(manifest_package_filename, str) and manifest_package_filename and manifest_package_filename != package_path.name:
        raise ValueError(
            f"package filename mismatch: manifest={manifest_package_filename} actual={package_path.name}"
        )
    return result


def discover_stage_packages(export_stage_root: Path) -> List[Dict[str, Any]]:
    """
    功能：递归发现 stage export 目录下的候选 package。

    Recursively discover candidate stage packages under one export root.

    Args:
        export_stage_root: Stage export root directory.

    Returns:
        Candidate package summary list.
    """
    if not isinstance(export_stage_root, Path):
        raise TypeError("export_stage_root must be Path")
    if not export_stage_root.exists() or not export_stage_root.is_dir():
        return []

    candidates: List[Dict[str, Any]] = []
    for zip_path in sorted(export_stage_root.rglob("*.zip")):
        if not zip_path.is_file():
            continue
        package_policy_probe = probe_stage_package_policy(zip_path)
        external_manifest = cast(Dict[str, Any], package_policy_probe.get("external_manifest", {}))
        internal_manifest = cast(Dict[str, Any], package_policy_probe.get("internal_manifest", {}))
        stage_manifest = cast(Dict[str, Any], package_policy_probe.get("stage_manifest", {}))
        manifest_for_sort = cast(Dict[str, Any], package_policy_probe.get("manifest_for_policy", {}))
        if not bool(package_policy_probe.get("formal_package_eligible", False)):
            continue
        package_created_at = manifest_for_sort.get("package_created_at") if isinstance(manifest_for_sort, dict) else None
        stage_run_id = manifest_for_sort.get("stage_run_id") if isinstance(manifest_for_sort, dict) else None
        validation_error = None
        validation_summary: Dict[str, Any] = {}
        try:
            if external_manifest:
                validation_summary = validate_package_manifest_binding(zip_path, external_manifest, required_sha_match=True)
            elif internal_manifest:
                validation_summary = validate_package_manifest_binding(zip_path, internal_manifest, required_sha_match=False)
        except Exception as exc:
            validation_error = str(exc)
        candidates.append(
            {
                "package_path": zip_path.as_posix(),
                "package_filename": zip_path.name,
                "package_created_at": package_created_at or utc_now_iso(),
                "stage_run_id": stage_run_id or stage_manifest.get("stage_run_id") or "<absent>",
                "external_manifest_path": package_policy_probe.get("external_manifest_path", "<absent>"),
                "external_manifest": external_manifest,
                "internal_manifest": internal_manifest,
                "stage_manifest": stage_manifest,
                "package_role": package_policy_probe.get("package_role") or FORMAL_STAGE_PACKAGE_ROLE,
                "package_discovery_scope": package_policy_probe.get("package_discovery_scope") or FORMAL_PACKAGE_DISCOVERY_SCOPE,
                "validation": validation_summary,
                "validation_error": validation_error,
                "mtime": datetime.fromtimestamp(zip_path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    candidates.sort(
        key=lambda item: (
            item.get("package_created_at") or "",
            item.get("stage_run_id") or "",
            item.get("mtime") or "",
        ),
        reverse=True,
    )
    return candidates


def select_latest_stage_package(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    功能：从候选 package 列表中选择最新有效项。

    Select the latest valid stage package from discovered candidates.

    Args:
        candidates: Candidate package mappings.

    Returns:
        Selected candidate mapping.
    """
    if not isinstance(candidates, Sequence) or not candidates:
        raise FileNotFoundError("no stage package candidates available")

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("validation_error"):
            continue
        validation = candidate.get("validation")
        if isinstance(validation, Mapping) and validation.get("filename_match"):
            selection = dict(candidate)
            selection["selection_reason"] = (
                "selected the latest candidate by package_created_at/stage_run_id with a valid manifest-to-zip binding"
            )
            return selection
    raise FileNotFoundError("no valid stage package candidate passed manifest binding checks")


def _normalize_optional_package_input(package_path_input: Any) -> Path | None:
    """
    功能：把 notebook 中的可空 package 输入规范化为 Path。 

    Normalize one optional notebook package input into a Path or None.

    Args:
        package_path_input: Optional stage package input from notebook cells.

    Returns:
        Normalized Path when a non-empty input is provided, else None.
    """
    if package_path_input is None:
        return None
    if isinstance(package_path_input, Path):
        return package_path_input.expanduser()
    if isinstance(package_path_input, str):
        normalized = package_path_input.strip()
        if not normalized:
            return None
        return Path(normalized).expanduser()
    raise TypeError("package_path_input must be Path, str, or None")


def inspect_stage_package_input(
    package_path: Path | None,
    *,
    expected_stage_name: str,
) -> Dict[str, Any]:
    """
    功能：检查单个 stage package 是否满足 notebook 发现条件。 

    Inspect one stage package against the notebook discovery contract.

    Args:
        package_path: Candidate stage package path.
        expected_stage_name: Expected stage name bound to the notebook stage.

    Returns:
        Inspection summary describing existence, formal-package policy, stage
        name binding, manifest parseability, and package-index availability.
    """
    if package_path is not None and not isinstance(package_path, Path):
        raise TypeError("package_path must be Path or None")
    if not isinstance(expected_stage_name, str) or not expected_stage_name:
        raise TypeError("expected_stage_name must be non-empty str")

    summary: Dict[str, Any] = {
        "input_path": normalize_path_value(package_path) if isinstance(package_path, Path) else "<absent>",
        "package_exists": False,
        "package_policy_source": "absent",
        "package_role": "<absent>",
        "package_discovery_scope": "<absent>",
        "diagnostics_like": False,
        "explicit_non_formal": False,
        "formal_package_eligible": False,
        "manifest_parseable": False,
        "package_index_parseable": False,
        "stage_manifest_stage_name": "<absent>",
        "manifest_stage_name": "<absent>",
        "stage_name_matches": False,
        "validation": {},
        "validation_error": None,
        "formal_package_valid": False,
    }
    if package_path is None:
        return summary
    if not package_path.exists() or not package_path.is_file():
        summary["validation_error"] = f"package file missing: {normalize_path_value(package_path)}"
        return summary

    summary["package_exists"] = True
    package_policy_probe = probe_stage_package_policy(package_path)
    external_manifest = cast(Dict[str, Any], package_policy_probe.get("external_manifest", {}))
    internal_manifest = cast(Dict[str, Any], package_policy_probe.get("internal_manifest", {}))
    stage_manifest = cast(Dict[str, Any], package_policy_probe.get("stage_manifest", {}))
    manifest_for_policy = cast(Dict[str, Any], package_policy_probe.get("manifest_for_policy", {}))
    package_index = read_json_from_zip(package_path, "artifacts/package_index.json")
    discovered_stage_name = stage_manifest.get("stage_name") or manifest_for_policy.get("stage_name")

    summary.update(
        {
            "package_policy_source": package_policy_probe.get("package_policy_source", "absent"),
            "package_role": package_policy_probe.get("package_role") or FORMAL_STAGE_PACKAGE_ROLE,
            "package_discovery_scope": package_policy_probe.get("package_discovery_scope") or FORMAL_PACKAGE_DISCOVERY_SCOPE,
            "diagnostics_like": bool(package_policy_probe.get("diagnostics_like", False)),
            "explicit_non_formal": bool(package_policy_probe.get("explicit_non_formal", False)),
            "formal_package_eligible": bool(package_policy_probe.get("formal_package_eligible", False)),
            "manifest_parseable": bool(stage_manifest) and bool(manifest_for_policy),
            "package_index_parseable": bool(package_index),
            "stage_manifest_stage_name": stage_manifest.get("stage_name", "<absent>"),
            "manifest_stage_name": manifest_for_policy.get("stage_name", "<absent>"),
            "stage_name_matches": discovered_stage_name == expected_stage_name,
        }
    )

    try:
        if external_manifest:
            summary["validation"] = validate_package_manifest_binding(
                package_path,
                external_manifest,
                required_sha_match=True,
            )
        elif internal_manifest:
            summary["validation"] = validate_package_manifest_binding(
                package_path,
                internal_manifest,
                required_sha_match=False,
            )
        else:
            summary["validation_error"] = "package manifest missing"
    except Exception as exc:
        # 中文注释：这里保留原始异常文本，方便 notebook precheck 直接显示失败原因。
        summary["validation_error"] = f"{type(exc).__name__}: {exc}"

    summary["formal_package_valid"] = bool(
        summary["package_exists"]
        and summary["formal_package_eligible"]
        and not summary["explicit_non_formal"]
        and summary["manifest_parseable"]
        and summary["package_index_parseable"]
        and summary["stage_name_matches"]
        and summary["validation_error"] is None
    )
    return summary


def discover_latest_formal_stage_package(
    export_stage_root: Path,
    *,
    expected_stage_name: str,
) -> Dict[str, Any]:
    """
    功能：在指定 exports 根目录中发现最新合法 formal stage package。 

    Discover the latest valid formal stage package under one stage-specific
    export root.

    Args:
        export_stage_root: Stage-specific export root directory.
        expected_stage_name: Expected stage name for the discovered package.

    Returns:
        Discovery summary carrying inspected candidates, the selected package,
        and one stable discovery error when no legal package is available.
    """
    if not isinstance(export_stage_root, Path):
        raise TypeError("export_stage_root must be Path")
    if not isinstance(expected_stage_name, str) or not expected_stage_name:
        raise TypeError("expected_stage_name must be non-empty str")

    discovered_candidates = discover_stage_packages(export_stage_root)
    inspected_candidates: List[Dict[str, Any]] = []
    for candidate in discovered_candidates:
        package_path_value = candidate.get("package_path")
        if not isinstance(package_path_value, str) or not package_path_value:
            continue
        inspected_candidate = dict(candidate)
        inspected_candidate.update(
            inspect_stage_package_input(
                Path(package_path_value),
                expected_stage_name=expected_stage_name,
            )
        )
        inspected_candidates.append(inspected_candidate)

    valid_candidates = [
        candidate
        for candidate in inspected_candidates
        if bool(candidate.get("formal_package_valid", False))
    ]

    selected_candidate: Dict[str, Any] = {}
    discovery_error: str | None = None
    if valid_candidates:
        try:
            selected_candidate = select_latest_stage_package(valid_candidates)
        except Exception as exc:
            # 中文注释：这里不向上抛出，交给 notebook precheck 决定 required/optional 的阻断语义。
            discovery_error = f"{type(exc).__name__}: {exc}"
    else:
        discovery_error = (
            "no valid formal stage package discovered under "
            f"{normalize_path_value(export_stage_root)}"
        )

    selected_package_path = selected_candidate.get("package_path", "<absent>")
    return {
        "export_stage_root": normalize_path_value(export_stage_root),
        "expected_stage_name": expected_stage_name,
        "candidates": inspected_candidates,
        "candidate_count": len(inspected_candidates),
        "selected_candidate": selected_candidate,
        "selected_package_path": selected_package_path,
        "selected_package_valid": bool(selected_candidate),
        "selection_reason": selected_candidate.get("selection_reason", "<absent>"),
        "discovery_error": discovery_error,
    }


def resolve_stage_package_input_or_discover(
    package_path_input: Any,
    export_stage_root: Path,
    *,
    expected_stage_name: str,
) -> Dict[str, Any]:
    """
    功能：按“手工优先，否则自动发现”解析 stage package 输入。 

    Resolve one notebook stage-package input using the stable policy of manual
    override first and automatic discovery otherwise.

    Args:
        package_path_input: Optional manual stage package input.
        export_stage_root: Stage-specific export root directory.
        expected_stage_name: Expected stage name for validation.

    Returns:
        Resolution summary carrying the selected path, whether manual input won,
        inspected discovery candidates, and one reusable validation result.
    """
    manual_package_path = _normalize_optional_package_input(package_path_input)
    resolution: Dict[str, Any] = {
        "manual_input_provided": manual_package_path is not None,
        "manual_input_used": manual_package_path is not None,
        "manual_input_path": normalize_path_value(manual_package_path) if manual_package_path is not None else "<absent>",
        "export_stage_root": normalize_path_value(export_stage_root),
        "expected_stage_name": expected_stage_name,
        "candidates": [],
        "selected_candidate": {},
        "selected_package_path": "<absent>",
        "selected_package_valid": False,
        "selection_reason": "<absent>",
        "auto_discovered_package_path": "<absent>",
        "resolution_error": None,
    }

    if manual_package_path is not None:
        inspection = inspect_stage_package_input(
            manual_package_path,
            expected_stage_name=expected_stage_name,
        )
        resolution.update(inspection)
        resolution.update(
            {
                "selected_candidate": {
                    "package_path": inspection["input_path"],
                    "selection_reason": "manual package path override",
                },
                "selected_package_path": inspection["input_path"],
                "selected_package_valid": bool(inspection.get("formal_package_valid", False)),
                "selection_reason": "manual package path override",
                "auto_discovered_package_path": "<skipped_manual_override>",
                "resolution_error": (
                    None
                    if bool(inspection.get("formal_package_valid", False))
                    else inspection.get("validation_error") or "manual package failed formal stage-package validation"
                ),
            }
        )
        return resolution

    discovery = discover_latest_formal_stage_package(
        export_stage_root,
        expected_stage_name=expected_stage_name,
    )
    resolution.update(
        {
            "manual_input_used": False,
            "candidates": discovery["candidates"],
            "selected_candidate": discovery["selected_candidate"],
            "selected_package_path": discovery["selected_package_path"],
            "selected_package_valid": bool(discovery.get("selected_package_valid", False)),
            "selection_reason": discovery.get("selection_reason", "<absent>"),
            "auto_discovered_package_path": discovery.get("selected_package_path", "<absent>"),
            "resolution_error": discovery.get("discovery_error"),
        }
    )
    if isinstance(discovery.get("selected_package_path"), str) and discovery["selected_package_path"] != "<absent>":
        resolution.update(
            inspect_stage_package_input(
                Path(str(discovery["selected_package_path"])),
                expected_stage_name=expected_stage_name,
            )
        )
        resolution["selected_package_path"] = discovery["selected_package_path"]
        resolution["selected_package_valid"] = bool(resolution.get("formal_package_valid", False))
    return resolution


def tail_text_file(path_obj: Path, max_lines: int = 20) -> List[str]:
    """
    功能：返回文本文件尾部若干行。

    Return the last N lines of one text file.

    Args:
        path_obj: Text file path.
        max_lines: Maximum number of lines.

    Returns:
        Tail lines list.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(max_lines, int) or max_lines <= 0:
        raise TypeError("max_lines must be positive int")
    if not path_obj.exists() or not path_obj.is_file():
        return []
    lines = path_obj.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def collect_missing_file_entries(path_mapping: Mapping[str, Path]) -> List[str]:
    """
    功能：收集缺失文件标签列表。

    Collect labels whose mapped files are missing.

    Args:
        path_mapping: Label to Path mapping.

    Returns:
        Missing label list.
    """
    if not isinstance(path_mapping, Mapping):
        raise TypeError("path_mapping must be Mapping")
    return [str(label) for label, path_obj in path_mapping.items() if not isinstance(path_obj, Path) or not path_obj.exists()]


def summarize_manifest_fields(manifest: Mapping[str, Any], field_names: Sequence[str]) -> Dict[str, Any]:
    """
    功能：提取 manifest 关键字段摘要。

    Extract a small manifest summary using selected field names.

    Args:
        manifest: Manifest mapping.
        field_names: Selected field names.

    Returns:
        Manifest summary mapping.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be Mapping")
    if not isinstance(field_names, Sequence):
        raise TypeError("field_names must be Sequence")
    return {str(field_name): manifest.get(str(field_name)) for field_name in field_names}


def collect_file_index(base_root: Path, path_mapping: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    功能：构造文件索引摘要。

    Build a file-index mapping with existence and relative-path information.

    Args:
        base_root: Base path for relative indexing.
        path_mapping: Label to path-like mapping.

    Returns:
        Indexed path summary mapping.
    """
    if not isinstance(base_root, Path):
        raise TypeError("base_root must be Path")
    if not isinstance(path_mapping, Mapping):
        raise TypeError("path_mapping must be Mapping")

    summary: Dict[str, Dict[str, Any]] = {}
    for label, raw_path in path_mapping.items():
        path_obj = raw_path if isinstance(raw_path, Path) else Path(str(raw_path))
        exists = path_obj.exists() and path_obj.is_file()
        summary[str(label)] = {
            "path": normalize_path_value(path_obj),
            "relative_path": relative_path_under_base(base_root, path_obj),
            "exists": exists,
        }
        if exists:
            summary[str(label)]["sha256"] = compute_file_sha256(path_obj)
    return summary


def stage_relative_copy(source_path: Path, package_root: Path, relative_destination: str) -> Path:
    """
    功能：按相对路径复制文件到 package staging 目录。

    Copy one file into the package staging directory using a relative destination.

    Args:
        source_path: Existing source file.
        package_root: Package staging root.
        relative_destination: Relative destination path inside the package.

    Returns:
        Destination file path.
    """
    if not isinstance(source_path, Path):
        raise TypeError("source_path must be Path")
    if not isinstance(package_root, Path):
        raise TypeError("package_root must be Path")
    if not isinstance(relative_destination, str) or not relative_destination.strip():
        raise TypeError("relative_destination must be non-empty str")

    destination_path = package_root / Path(relative_destination)
    copy_file(source_path, destination_path)
    return destination_path


def finalize_stage_package(
    *,
    stage_name: str,
    stage_run_id: str,
    package_root: Path,
    export_root: Path,
    source_stage_run_id: Optional[str],
    source_stage_package_path: Optional[str],
    package_manifest_path: Path,
    package_role: str = FORMAL_STAGE_PACKAGE_ROLE,
    package_discovery_scope: str = FORMAL_PACKAGE_DISCOVERY_SCOPE,
) -> Dict[str, Any]:
    """
    功能：写出 package index、压缩包与 package_manifest。

    Finalize one stage package by emitting the package index, ZIP archive, and package manifest.

    Args:
        stage_name: Stable stage name.
        stage_run_id: Current stage run identifier.
        package_root: Package staging directory.
        export_root: Stage export directory.
        source_stage_run_id: Optional upstream stage run identifier.
        source_stage_package_path: Optional upstream stage package path.
        package_manifest_path: Destination package manifest path.

    Returns:
        Package manifest mapping.
    """
    if not isinstance(package_role, str) or not package_role:
        raise TypeError("package_role must be non-empty str")
    if not isinstance(package_discovery_scope, str) or not package_discovery_scope:
        raise TypeError("package_discovery_scope must be non-empty str")

    package_index_path = package_root / "artifacts" / "package_index.json"
    package_manifest_internal_path = package_root / "artifacts" / "package_manifest.json"
    package_index = build_package_index(package_root.rglob("*"), package_root)
    write_json_atomic(package_index_path, package_index)

    package_filename = build_stage_package_filename(stage_name, stage_run_id, source_stage_run_id)
    internal_manifest = {
        "package_manifest_version": "v2",
        "stage_name": stage_name,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_stage_run_id,
        "source_stage_package_path": source_stage_package_path,
        "package_filename": package_filename,
        "package_path": "<external_zip>",
        "package_sha256": "<see_external_manifest>",
        "package_created_at": utc_now_iso(),
        "package_contents_index_path": "artifacts/package_index.json",
        "package_manifest_scope": "internal_copy",
        "package_role": package_role,
        "package_discovery_scope": package_discovery_scope,
    }
    write_json_atomic(package_manifest_internal_path, internal_manifest)

    package_path = create_zip_archive_from_directory(package_root, export_root / package_filename)
    package_manifest = {
        "package_manifest_version": "v2",
        "package_path": package_path.as_posix(),
        "package_filename": package_filename,
        "package_sha256": compute_file_sha256(package_path),
        "package_created_at": utc_now_iso(),
        "package_contents_index_path": resolve_export_package_index_path(export_root).as_posix(),
        "stage_name": stage_name,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_stage_run_id,
        "source_stage_package_path": source_stage_package_path,
        "package_role": package_role,
        "package_discovery_scope": package_discovery_scope,
        "package_manifest_digest": "<pending>",
    }
    package_manifest["package_manifest_digest"] = compute_mapping_sha256(package_manifest)
    write_json_atomic(package_manifest_path, package_manifest)
    write_json_atomic(resolve_export_package_manifest_path(export_root), package_manifest)
    write_json_atomic(resolve_export_package_index_path(export_root), package_index)
    return package_manifest


def read_stage_manifest_from_package(extracted_root: Path) -> Dict[str, Any]:
    """
    功能：从解压后的 stage package 中读取 stage_manifest。

    Read the stage_manifest.json file from an extracted stage package.

    Args:
        extracted_root: Extracted package root directory.

    Returns:
        Stage manifest mapping.
    """
    manifest_path = extracted_root / "artifacts" / "stage_manifest.json"
    manifest = read_json_dict(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"stage_manifest.json missing in extracted package: {manifest_path}")
    return manifest


def prepare_source_package(source_package_path: Path, runtime_state_root: Path) -> Dict[str, Any]:
    """
    功能：复制、校验并解压 source package。

    Copy, hash-verify, and extract one source stage package for downstream consumption.

    Args:
        source_package_path: Source ZIP path.
        runtime_state_root: Stage runtime-state root.

    Returns:
        Source package summary mapping.
    """
    package_root = ensure_directory(runtime_state_root / "source_package")
    local_package_path = package_root / source_package_path.name
    copy_file(source_package_path, local_package_path)
    package_sha256 = compute_file_sha256(local_package_path)
    external_manifest_source_path = find_external_package_manifest(source_package_path)
    local_external_manifest_path = package_root / "package_manifest.external.json"
    external_package_manifest: Dict[str, Any] = {}
    if isinstance(external_manifest_source_path, Path):
        copy_file(external_manifest_source_path, local_external_manifest_path)
        external_package_manifest = read_json_dict(local_external_manifest_path)
        validate_package_manifest_binding(local_package_path, external_package_manifest, required_sha_match=True)
        if not _is_discoverable_formal_package_manifest(external_package_manifest):
            raise ValueError(
                "source package must be a discoverable formal stage package: "
                f"path={normalize_path_value(source_package_path)}"
            )
    extracted_root = package_root / "extracted"
    members = extract_zip_archive(local_package_path, extracted_root)
    stage_manifest = read_stage_manifest_from_package(extracted_root)
    internal_package_manifest = read_json_dict(extracted_root / "artifacts" / "package_manifest.json")
    if not internal_package_manifest:
        raise FileNotFoundError("package_manifest.json missing in extracted package")
    package_index = read_json_dict(extracted_root / "artifacts" / "package_index.json")
    package_manifest_for_lineage = external_package_manifest if external_package_manifest else internal_package_manifest
    if not _is_discoverable_formal_package_manifest(package_manifest_for_lineage):
        raise ValueError(
            "source package must be a discoverable formal stage package: "
            f"path={normalize_path_value(source_package_path)}"
        )
    if package_manifest_for_lineage.get("stage_name") not in {None, stage_manifest.get("stage_name")}:
        raise ValueError("source package manifest stage_name does not match source stage_manifest")
    if package_manifest_for_lineage.get("stage_run_id") not in {None, stage_manifest.get("stage_run_id")}:
        raise ValueError("source package manifest stage_run_id does not match source stage_manifest")
    return {
        "source_package_path": local_package_path.as_posix(),
        "source_package_sha256": package_sha256,
        "extracted_root": extracted_root.as_posix(),
        "members": members,
        "stage_manifest": stage_manifest,
        "external_package_manifest_path": normalize_path_value(local_external_manifest_path) if external_package_manifest else "<absent>",
        "external_package_manifest": external_package_manifest,
        "internal_package_manifest": internal_package_manifest,
        "package_manifest": package_manifest_for_lineage,
        "package_manifest_digest": compute_mapping_sha256(package_manifest_for_lineage),
        "package_index": package_index,
    }


def copy_stage_manifest_snapshot(stage_manifest: Mapping[str, Any], destination_path: Path) -> Path:
    """
    功能：写出 source stage_manifest 快照。

    Persist a copied stage manifest snapshot for lineage auditing.

    Args:
        stage_manifest: Source stage manifest mapping.
        destination_path: Destination JSON path.

    Returns:
        Destination path.
    """
    write_json_atomic(destination_path, dict(stage_manifest))
    return destination_path


def persist_source_package_lineage(
    runtime_state_root: Path,
    source_info: Mapping[str, Any],
) -> Dict[str, Path]:
    """
    功能：把 source package lineage 快照写入 runtime_state。

    Persist the canonical source stage/package manifest snapshots under one
    runtime-state lineage directory.

    Args:
        runtime_state_root: Stage runtime-state root.
        source_info: Prepared source-package summary mapping.

    Returns:
        Mapping containing the copied stage and package manifest paths.
    """
    stage_manifest = source_info.get("stage_manifest")
    package_manifest = source_info.get("package_manifest")
    if not isinstance(stage_manifest, Mapping):
        raise TypeError("source_info.stage_manifest must be Mapping")
    if not isinstance(package_manifest, Mapping):
        raise TypeError("source_info.package_manifest must be Mapping")

    lineage_root = ensure_directory(runtime_state_root / "lineage")
    source_stage_manifest_copy_path = lineage_root / "source_stage_manifest.json"
    source_package_manifest_copy_path = lineage_root / "source_package_manifest.json"
    copy_stage_manifest_snapshot(stage_manifest, source_stage_manifest_copy_path)
    write_json_atomic(source_package_manifest_copy_path, dict(package_manifest))
    return {
        "source_stage_manifest_copy_path": source_stage_manifest_copy_path,
        "source_package_manifest_copy_path": source_package_manifest_copy_path,
    }
