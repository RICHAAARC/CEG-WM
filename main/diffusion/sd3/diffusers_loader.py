"""
SD3 pipeline 加载器

功能说明：
- 提供 diffusers/transformers 的受控导入与版本探测。
- 提供 SD3 pipeline 的构造包装，禁止异常向上抛出。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple


def try_import_diffusers() -> Tuple[bool, Dict[str, Any]]:
    """
    功能：尝试导入 diffusers 相关依赖。

    Try importing diffusers and related packages without raising ImportError.

    Args:
        None.

    Returns:
        Tuple of (available, meta) where meta contains version info and errors.
    """
    meta: Dict[str, Any] = {
        "diffusers_available": False,
        "diffusers_version": "<absent>",
        "transformers_version": "<absent>",
        "safetensors_version": "<absent>",
        "import_error": "<absent>"
    }

    try:
        # 静态导入 diffusers 模块（冻结执行）。
        if importlib.util.find_spec("diffusers") is None:
            raise ImportError("diffusers module not found")
        import diffusers
        meta["diffusers_available"] = True
        meta["diffusers_version"] = getattr(diffusers, "__version__", "<absent>")
    except Exception as exc:
        meta["import_error"] = f"{type(exc).__name__}: {exc}"
        return False, meta

    meta["transformers_version"] = _try_import_version("transformers")
    meta["safetensors_version"] = _try_import_version("safetensors")
    return True, meta


def build_sd3_pipeline_from_pretrained(
    model_id: str,
    revision: str | None,
    model_source: str | None,
    extra_kwargs: Dict[str, Any] | None = None,
    local_snapshot_path: str | None = None,
) -> Tuple[Any | None, Dict[str, Any], str | None]:
    """
    功能：构造 SD3 pipeline（受控包装）。

    Build SD3 pipeline in a controlled wrapper. This function must not raise.

    Args:
        model_id: Model identifier string.
        revision: Optional revision string.
        model_source: Optional model source indicator.
        extra_kwargs: Optional extra kwargs for future extension.
        local_snapshot_path: Optional notebook-bound local snapshot directory.

    Returns:
        Tuple of (pipeline_or_none, build_meta, error_or_none).
    """
    if not isinstance(model_id, str) or not model_id:
        # model_id 输入不合法，必须 fail-fast。
        return None, {"status": "invalid_input"}, "model_id must be non-empty str"
    if revision is not None and (not isinstance(revision, str) or not revision):
        # revision 输入不合法，必须 fail-fast。
        return None, {"status": "invalid_input"}, "revision must be non-empty str or None"
    if model_source is not None and (not isinstance(model_source, str) or not model_source):
        # model_source 输入不合法，必须 fail-fast。
        return None, {"status": "invalid_input"}, "model_source must be non-empty str or None"
    if extra_kwargs is not None and not isinstance(extra_kwargs, dict):
        # extra_kwargs 输入不合法，必须 fail-fast。
        return None, {"status": "invalid_input"}, "extra_kwargs must be dict or None"
    if local_snapshot_path is not None and (
        not isinstance(local_snapshot_path, str) or not local_snapshot_path.strip()
    ):
        return None, {"status": "invalid_input"}, "local_snapshot_path must be non-empty str or None"

    resolved_local_snapshot_path, local_snapshot_status, local_snapshot_error = _resolve_local_snapshot_path(
        local_snapshot_path
    )
    local_snapshot_meta_path = resolved_local_snapshot_path
    if local_snapshot_status == "invalid":
        resolved_local_snapshot_path = None

    available, meta = try_import_diffusers()
    build_meta = {
        "status": "skipped",
        "diffusers_available": available,
        "diffusers_version": meta.get("diffusers_version"),
        "transformers_version": meta.get("transformers_version"),
        "safetensors_version": meta.get("safetensors_version"),
        "model_id": model_id,
        "revision": revision or "<absent>",
        "model_source": model_source or "<absent>",
        "requested_model_id": model_id,
        "requested_revision": revision or "<absent>",
        "requested_model_source": model_source or "<absent>",
        "resolved_model_id": model_id,
        "resolved_model_source": model_source or "<absent>",
        "model_source_resolution": "requested_model_source",
        "local_snapshot_requested": local_snapshot_path is not None,
        "local_snapshot_path": (
            local_snapshot_meta_path
            if local_snapshot_meta_path is not None
            else (local_snapshot_path.strip() if isinstance(local_snapshot_path, str) and local_snapshot_path.strip() else "<absent>")
        ),
        "local_snapshot_status": local_snapshot_status,
        "local_snapshot_error": local_snapshot_error or "<absent>",
        "extra_kwargs": extra_kwargs or {},
    }
    if not available:
        return None, build_meta, meta.get("import_error")

    try:
        # 静态导入 diffusers 模块（冻结执行，需确保已在 try_import_diffusers 中验证）。
        if importlib.util.find_spec("diffusers") is None:
            raise ImportError("diffusers module not found")
        import diffusers
    except Exception as exc:
        return None, build_meta, f"{type(exc).__name__}: {exc}"

    # notebook 显式绑定的本地 snapshot 优先于默认 HF source；若绑定无效，则保留原始 source 并记录回退轨迹。
    load_target = model_id
    local_files_only = True  # 默认只使用本地缓存
    if resolved_local_snapshot_path is not None:
        load_target = resolved_local_snapshot_path
        local_files_only = True
        build_meta["resolved_model_id"] = resolved_local_snapshot_path
        build_meta["resolved_model_source"] = "local_path"
        build_meta["model_source_resolution"] = "local_snapshot_priority"
    else:
        if model_source in ("hf", "hf_hub"):
            local_files_only = False
        elif model_source in ("local", "local_path"):
            local_files_only = True
        elif model_source is not None:
            return None, build_meta, f"model_source not allowed: {model_source}"
        if local_snapshot_status == "invalid":
            build_meta["model_source_resolution"] = "fallback_to_requested_model_source"

    # 从 extra_kwargs 中提取设备和精度配置
    device = None
    torch_dtype = None
    if extra_kwargs:
        device = extra_kwargs.get("device")
        dtype_str = extra_kwargs.get("dtype")
        if dtype_str:
            # 转换 dtype 字符串为 torch.dtype
            try:
                import torch
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "fp32": torch.float32,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                }
                torch_dtype = dtype_map.get(dtype_str.lower())
            except Exception:
                pass

    build_kwargs: Dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only
    }
    
    # 添加 torch_dtype 参数（如果指定）
    if torch_dtype is not None:
        build_kwargs["torch_dtype"] = torch_dtype
    
    # 保留其他 extra_kwargs（排除已处理的 device 和 dtype）
    if extra_kwargs:
        for key, value in extra_kwargs.items():
            if key not in ("device", "dtype"):
                build_kwargs[key] = value

    try:
        pipeline = diffusers.DiffusionPipeline.from_pretrained(load_target, **build_kwargs)
        
        # 如果指定了设备，将 pipeline 移动到该设备
        if device is not None and hasattr(pipeline, "to"):
            try:
                import torch
                # 验证设备是否可用
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                    build_meta["device_fallback"] = "cuda_unavailable"
                pipeline = pipeline.to(device)
                build_meta["device"] = device
            except Exception as device_error:
                build_meta["device_error"] = str(device_error)
        
        build_meta["status"] = "built"
        build_meta["local_files_only"] = local_files_only
        build_meta["load_target"] = load_target
        
        # 构造可序列化的 build_kwargs 副本（排除 torch.dtype 对象）
        serializable_kwargs = {}
        for key, value in build_kwargs.items():
            if key == "torch_dtype":
                # 将 torch.dtype 转换为字符串
                serializable_kwargs[key] = str(value) if value is not None else None
            else:
                serializable_kwargs[key] = value
        build_meta["build_kwargs"] = serializable_kwargs
        
        if torch_dtype is not None:
            build_meta["torch_dtype"] = str(torch_dtype)
        return pipeline, build_meta, None
    except Exception as exc:
        build_meta["status"] = "failed"
        build_meta["local_files_only"] = local_files_only
        build_meta["load_target"] = load_target
        
        # 构造可序列化的 build_kwargs 副本
        serializable_kwargs = {}
        for key, value in build_kwargs.items():
            if key == "torch_dtype":
                serializable_kwargs[key] = str(value) if value is not None else None
            else:
                serializable_kwargs[key] = value
        build_meta["build_kwargs"] = serializable_kwargs
        
        if build_meta.get("resolved_model_source") in ("hf", "hf_hub"):
            return None, build_meta, "hf_hub_local_cache_missing_or_unavailable"
        return None, build_meta, f"{type(exc).__name__}: {exc}"


def _resolve_local_snapshot_path(local_snapshot_path: str | None) -> Tuple[str | None, str, str | None]:
    """
    功能：解析可选的本地 snapshot 目录。 

    Resolve the optional local snapshot directory forwarded by notebook
    bootstrap.

    Args:
        local_snapshot_path: Optional raw snapshot directory path.

    Returns:
        Tuple of (resolved_path_or_none, status, error_or_none).
    """
    if local_snapshot_path is None:
        return None, "absent", None

    snapshot_path_obj = Path(local_snapshot_path.strip()).expanduser().resolve()
    snapshot_path_text = snapshot_path_obj.as_posix()
    if not snapshot_path_obj.exists() or not snapshot_path_obj.is_dir():
        return snapshot_path_text, "invalid", "local_snapshot_path_missing_or_not_directory"
    return snapshot_path_text, "bound", None


def _try_import_version(module_name: str) -> str:
    """
    功能：尝试导入模块并返回版本号。

    Try importing a module and returning its __version__ if available.
    Uses find_spec for module discovery instead of import_module (static execution).

    Args:
        module_name: Module name string.

    Returns:
        Version string or "<absent>".
    """
    try:
        # 使用 find_spec 静态探测模块（冻结执行）。
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return "<absent>"
        
        # 仅对已知模块列表加载（白名单 transformers, safetensors）。
        if module_name not in ("transformers", "safetensors"):
            return "<absent>"
        
        if module_name == "transformers":
            try:
                import transformers
                version = getattr(transformers, "__version__", None)
                if isinstance(version, str) and version:
                    return version
            except Exception:
                return "<absent>"
        elif module_name == "safetensors":
            try:
                import safetensors
                version = getattr(safetensors, "__version__", None)
                if isinstance(version, str) and version:
                    return version
            except Exception:
                return "<absent>"
    except Exception:
        return "<absent>"
    return "<absent>"
