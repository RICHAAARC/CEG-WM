"""
SD3 pipeline 加载器

功能说明：
- 提供 diffusers/transformers 的受控导入与版本探测。
- 提供 SD3 pipeline 的构造包装，禁止异常向上抛出。
"""

from __future__ import annotations

import importlib.util
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
    extra_kwargs: Dict[str, Any] | None = None
) -> Tuple[Any | None, Dict[str, Any], str | None]:
    """
    功能：构造 SD3 pipeline（受控包装）。

    Build SD3 pipeline in a controlled wrapper. This function must not raise.

    Args:
        model_id: Model identifier string.
        revision: Optional revision string.
        model_source: Optional model source indicator.
        extra_kwargs: Optional extra kwargs for future extension.

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
        "extra_kwargs": extra_kwargs or {}
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

    # 根据 model_source 决定是否允许从 HF Hub 下载
    local_files_only = True  # 默认只使用本地缓存
    if model_source in ("hf", "hf_hub"):
        # 允许从 HuggingFace Hub 下载模型
        local_files_only = False
    elif model_source in ("local", "local_path"):
        # 仅使用本地文件，不联网下载
        local_files_only = True
    elif model_source is not None:
        # 不支持的 model_source 值
        return None, build_meta, f"model_source not allowed: {model_source}"

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
        pipeline = diffusers.DiffusionPipeline.from_pretrained(model_id, **build_kwargs)
        
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
        build_meta["build_kwargs"] = dict(build_kwargs)
        if torch_dtype is not None:
            build_meta["torch_dtype"] = str(torch_dtype)
        return pipeline, build_meta, None
    except Exception as exc:
        build_meta["status"] = "failed"
        build_meta["local_files_only"] = local_files_only
        build_meta["build_kwargs"] = dict(build_kwargs)
        if model_source == "hf_hub":
            return None, build_meta, "hf_hub_local_cache_missing_or_unavailable"
        return None, build_meta, f"{type(exc).__name__}: {exc}"


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
