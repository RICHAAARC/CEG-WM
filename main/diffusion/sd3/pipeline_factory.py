"""
SD3 pipeline 工厂

功能说明：
- 管道壳构造的唯一入口，禁止旁路构造。
- 返回可序列化的 provenance 与 digest，并显式记录构造状态。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from main.registries import pipeline_registry
from main.diffusion.sd3 import diffusers_loader
from main.diffusion.sd3 import weights_snapshot
from main.diffusion.sd3.provenance import (
    build_pipeline_provenance,
    compute_pipeline_provenance_canon_sha256
)
from main.core import env_fingerprint


PIPELINE_STATUS_BUILT = "built"
PIPELINE_STATUS_FAILED = "failed"
PIPELINE_STATUS_UNBUILT = "unbuilt"


def build_pipeline_shell(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造管道壳并返回溯源信息。

    Build pipeline shell and return provenance, digest, and build status.

    Args:
        cfg: Configuration mapping.

    Returns:
        Mapping with pipeline_impl_id, pipeline_provenance,
        pipeline_provenance_canon_sha256, pipeline_status, pipeline_error.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    pipeline_impl_id, impl_id_error = _resolve_pipeline_impl_id(cfg)
    build_enabled, build_error = _resolve_pipeline_build_enabled(cfg)

    pipeline_status = PIPELINE_STATUS_UNBUILT
    pipeline_error = None
    pipeline_meta: Dict[str, Any] = {}

    pipeline_runtime_meta: Dict[str, Any] = {}
    env_fingerprint_canon_sha256 = "<absent>"
    model_provenance_canon_sha256 = "<absent>"
    diffusers_version = "<absent>"
    transformers_version = "<absent>"
    safetensors_version = "<absent>"
    pipeline_obj = None

    if impl_id_error is not None:
        pipeline_status = PIPELINE_STATUS_FAILED
        pipeline_error = impl_id_error
    elif build_error is not None:
        pipeline_status = PIPELINE_STATUS_FAILED
        pipeline_error = build_error
    elif not build_enabled:
        pipeline_status = PIPELINE_STATUS_UNBUILT
    else:
        if pipeline_impl_id == pipeline_registry.SD3_DIFFUSERS_REAL_ID:
            try:
                available, import_meta = diffusers_loader.try_import_diffusers()
                diffusers_version = import_meta.get("diffusers_version", "<absent>")
                transformers_version = import_meta.get("transformers_version", "<absent>")
                safetensors_version = import_meta.get("safetensors_version", "<absent>")

                model_id = cfg.get("model_id")
                model_source = cfg.get("model_source")
                hf_revision = cfg.get("hf_revision")
                
                # 获取设备和精度配置
                device = cfg.get("device", "cpu")
                model_cfg = cfg.get("model", {})
                dtype_str = model_cfg.get("dtype", "float32") if isinstance(model_cfg, dict) else "float32"
                
                # 构造 extra_kwargs 传递设备和精度信息
                extra_kwargs = {
                    "device": device,
                    "dtype": dtype_str
                }

                if not isinstance(model_id, str) or not model_id:
                    pipeline_status = PIPELINE_STATUS_FAILED
                    pipeline_error = "model_id missing"
                elif not isinstance(model_source, str) or not model_source:
                    pipeline_status = PIPELINE_STATUS_FAILED
                    pipeline_error = "model_source missing"
                elif not isinstance(hf_revision, str) or not hf_revision:
                    pipeline_status = PIPELINE_STATUS_FAILED
                    pipeline_error = "hf_revision missing"
                else:
                    pipeline_obj, build_meta, error = diffusers_loader.build_sd3_pipeline_from_pretrained(
                        model_id=model_id,
                        revision=hf_revision,
                        model_source=model_source,
                        extra_kwargs=extra_kwargs
                    )
                    pipeline_runtime_meta = build_meta if isinstance(build_meta, dict) else {}
                    _attach_resolved_revision(pipeline_obj, pipeline_runtime_meta)
                    pipeline_error = error or "<absent>"
                    pipeline_status = PIPELINE_STATUS_FAILED if error else PIPELINE_STATUS_BUILT

                    if pipeline_status == PIPELINE_STATUS_BUILT:
                        weights_snapshot_sha256, snapshot_meta, snapshot_error = _compute_weights_snapshot(
                            cfg,
                            model_id,
                            model_source,
                            hf_revision,
                            pipeline_runtime_meta
                        )
                        pipeline_runtime_meta["weights_snapshot_sha256"] = weights_snapshot_sha256
                        pipeline_runtime_meta["weights_snapshot_meta"] = snapshot_meta
                        if snapshot_error is not None:
                            pipeline_runtime_meta["weights_snapshot_error"] = snapshot_error
                            pipeline_status = PIPELINE_STATUS_FAILED
                            pipeline_error = _format_weights_snapshot_error(snapshot_error, snapshot_meta)
            except Exception as exc:
                # pipeline 构造失败，必须显式记录错误。
                pipeline_status = PIPELINE_STATUS_FAILED
                pipeline_error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                factory = pipeline_registry.resolve_pipeline_shell(pipeline_impl_id)
                shell_obj = factory(cfg)
                pipeline_meta = _extract_pipeline_meta(shell_obj)
                pipeline_status = PIPELINE_STATUS_BUILT
            except Exception as exc:
                # pipeline 构造失败，必须显式记录错误。
                pipeline_status = PIPELINE_STATUS_FAILED
                pipeline_error = f"{type(exc).__name__}: {exc}"

    cfg_for_provenance = dict(cfg)
    if isinstance(pipeline_runtime_meta, dict):
        weights_snapshot_sha256 = pipeline_runtime_meta.get("weights_snapshot_sha256")
        if isinstance(weights_snapshot_sha256, str) and weights_snapshot_sha256 and weights_snapshot_sha256 != "<absent>":
            cfg_for_provenance.setdefault("model_weights_sha256", weights_snapshot_sha256)
            cfg_for_provenance.setdefault("weights_snapshot_sha256", weights_snapshot_sha256)
        local_files_only = pipeline_runtime_meta.get("local_files_only")
        if isinstance(local_files_only, bool):
            cfg_for_provenance.setdefault("local_files_only", local_files_only)
        resolved_revision = pipeline_runtime_meta.get("resolved_revision")
        if isinstance(resolved_revision, str) and resolved_revision:
            cfg_for_provenance.setdefault("resolved_revision", resolved_revision)

    provenance = build_pipeline_provenance(cfg_for_provenance, pipeline_impl_id, pipeline_meta)
    provenance_digest = compute_pipeline_provenance_canon_sha256(provenance)
    model_provenance_canon_sha256 = compute_pipeline_provenance_canon_sha256(
        _build_model_provenance(provenance)
    )

    env_fp = env_fingerprint.build_env_fingerprint()
    env_fingerprint_canon_sha256 = env_fingerprint.compute_env_fingerprint_canon_sha256(env_fp)

    return {
        "pipeline_impl_id": pipeline_impl_id,
        "pipeline_provenance": provenance,
        "pipeline_provenance_canon_sha256": provenance_digest,
        "pipeline_status": pipeline_status,
        "pipeline_error": _normalize_error(pipeline_error),
        "pipeline_runtime_meta": pipeline_runtime_meta,
        "pipeline_obj": pipeline_obj,
        "env_fingerprint_canon_sha256": env_fingerprint_canon_sha256,
        "diffusers_version": diffusers_version,
        "transformers_version": transformers_version,
        "safetensors_version": safetensors_version,
        "model_provenance_canon_sha256": model_provenance_canon_sha256
    }


def _resolve_pipeline_impl_id(cfg: Dict[str, Any]) -> Tuple[str, str | None]:
    """
    功能：解析 pipeline_impl_id（含默认值）。

    Resolve pipeline_impl_id with defaults and validation.

    Args:
        cfg: Configuration mapping.

    Returns:
        Tuple of (pipeline_impl_id, error_message or None).

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    pipeline_impl_id = cfg.get("pipeline_impl_id")
    if pipeline_impl_id is None:
        pipeline_cfg = cfg.get("pipeline")
        if pipeline_cfg is None:
            return pipeline_registry.SD3_DIFFUSERS_SHELL_ID, None
        if not isinstance(pipeline_cfg, dict):
            return "<absent>", "pipeline section must be dict"
        pipeline_impl_id = pipeline_cfg.get("pipeline_impl_id")
        if pipeline_impl_id is None:
            return pipeline_registry.SD3_DIFFUSERS_SHELL_ID, None

    if not isinstance(pipeline_impl_id, str) or not pipeline_impl_id:
        return "<absent>", "pipeline_impl_id must be non-empty str"

    return pipeline_impl_id, None


def _resolve_pipeline_build_enabled(cfg: Dict[str, Any]) -> Tuple[bool, str | None]:
    """
    功能：解析 pipeline_build_enabled 运行期开关。

    Resolve optional pipeline_build_enabled flag from cfg.
    When paper_faithfulness is enabled, pipeline building is always mandatory.

    Args:
        cfg: Configuration mapping.

    Returns:
        Tuple of (enabled flag, error_message or None).

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    # 检查 paper_faithfulness 是否启用
    # 当启用 paper_faithfulness 时，pipeline 构造必须强制开启
    paper_faithfulness_cfg = cfg.get("paper_faithfulness", {})
    if isinstance(paper_faithfulness_cfg, dict):
        pf_enabled = paper_faithfulness_cfg.get("enabled", False)
        if pf_enabled:
            # paper_faithfulness 启用 → 必须构造 pipeline
            return True, None

    # 否则，读取 pipeline_build_enabled 配置（默认为 True）
    build_enabled = cfg.get("pipeline_build_enabled", True)
    if not isinstance(build_enabled, bool):
        return False, "pipeline_build_enabled must be bool"
    return build_enabled, None


def _extract_pipeline_meta(shell_obj: Any) -> Dict[str, Any]:
    """
    功能：抽取 pipeline meta 字段。

    Extract pipeline metadata from shell object.

    Args:
        shell_obj: Pipeline shell object returned by registry factory.

    Returns:
        Metadata mapping.

    Raises:
        TypeError: If shell_obj cannot be converted to dict.
    """
    if isinstance(shell_obj, dict):
        return shell_obj
    if hasattr(shell_obj, "as_dict"):
        meta = shell_obj.as_dict()
        if not isinstance(meta, dict):
            # meta 类型不合法，必须 fail-fast。
            raise TypeError("pipeline meta must be dict")
        return meta
    # 未知类型，必须 fail-fast。
    raise TypeError("pipeline shell object must provide dict or as_dict")


def _resolve_cfg_str(cfg: Dict[str, Any], field_name: str) -> str:
    """
    功能：解析 cfg 中的字符串字段。

    Resolve a string field from cfg with <absent> fallback.

    Args:
        cfg: Configuration mapping.
        field_name: Field name to resolve.

    Returns:
        Field value or "<absent>".
    """
    value = cfg.get(field_name)
    if isinstance(value, str) and value:
        return value
    return "<absent>"


def _normalize_error(error: str | None) -> str:
    """
    功能：规范化错误字段。

    Normalize error message to non-empty string or <absent>.

    Args:
        error: Error message or None.

    Returns:
        Error string or "<absent>".
    """
    if isinstance(error, str) and error:
        return error
    return "<absent>"


def _build_model_provenance(provenance: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造模型来源溯源对象。

    Build model provenance mapping from pipeline provenance.

    Args:
        provenance: Pipeline provenance mapping.

    Returns:
        Model provenance mapping.
    """
    return {
        "model_id": provenance.get("model_id", "<absent>"),
        "model_source": provenance.get("model_source", "<absent>"),
        "hf_revision": provenance.get("hf_revision", "<absent>"),
        "model_revision": provenance.get("model_revision", "<absent>"),
        "model_weights_sha256": provenance.get("model_weights_sha256", "<absent>")
    }


def _attach_resolved_revision(pipeline_obj: Any, runtime_meta: Dict[str, Any]) -> None:
    """
    功能：解析并写入 resolved_revision。

    Resolve and attach resolved_revision to runtime_meta.

    Args:
        pipeline_obj: Diffusers pipeline object or None.
        runtime_meta: Runtime meta mapping to mutate.

    Returns:
        None.
    """
    if not isinstance(runtime_meta, dict):
        # runtime_meta 类型不合法，必须 fail-fast。
        raise TypeError("runtime_meta must be dict")

    resolved_revision = _extract_resolved_revision(pipeline_obj)
    runtime_meta["resolved_revision"] = resolved_revision


def _extract_resolved_revision(pipeline_obj: Any) -> str:
    """
    功能：抽取 pipeline resolved_revision。

    Extract resolved_revision from pipeline object if available.

    Args:
        pipeline_obj: Diffusers pipeline object or None.

    Returns:
        Resolved revision string or "<absent>".
    """
    if pipeline_obj is None:
        return "<absent>"
    for attr_name in ("_commit_hash", "commit_hash"):
        value = getattr(pipeline_obj, attr_name, None)
        if isinstance(value, str) and value:
            return value
    config_obj = getattr(pipeline_obj, "config", None)
    value = getattr(config_obj, "_commit_hash", None)
    if isinstance(value, str) and value:
        return value
    return "<absent>"


def _compute_weights_snapshot(
    cfg: Dict[str, Any],
    model_id: str,
    model_source: str,
    hf_revision: str,
    runtime_meta: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], str | None]:
    """
    功能：计算权重快照摘要并返回元信息。

    Compute weights snapshot digest and return metadata.

    Args:
        cfg: Configuration mapping.
        model_id: Model identifier string.
        model_source: Model source string.
        hf_revision: Revision string.
        runtime_meta: Runtime meta mapping.

    Returns:
        Tuple of (weights_snapshot_sha256, snapshot_meta, error_or_none).

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(runtime_meta, dict):
        # runtime_meta 类型不合法，必须 fail-fast。
        raise TypeError("runtime_meta must be dict")

    local_files_only = runtime_meta.get("local_files_only")
    if not isinstance(local_files_only, bool):
        local_files_only = None

    cache_dir = _resolve_cache_dir(cfg, runtime_meta)

    return weights_snapshot.compute_weights_snapshot_sha256(
        model_id=model_id,
        model_source=model_source,
        hf_revision=hf_revision,
        local_files_only=local_files_only,
        cache_dir=cache_dir
    )


def _resolve_cache_dir(cfg: Dict[str, Any], runtime_meta: Dict[str, Any]) -> str | None:
    """
    功能：解析可选 cache_dir。

    Resolve optional cache_dir from cfg or runtime_meta.

    Args:
        cfg: Configuration mapping.
        runtime_meta: Runtime meta mapping.

    Returns:
        cache_dir string or None.
    """
    cache_dir = cfg.get("cache_dir")
    if isinstance(cache_dir, str) and cache_dir:
        return cache_dir
    build_kwargs = runtime_meta.get("build_kwargs")
    if isinstance(build_kwargs, dict):
        cache_dir = build_kwargs.get("cache_dir")
        if isinstance(cache_dir, str) and cache_dir:
            return cache_dir
    return None


def _format_weights_snapshot_error(snapshot_error: str, snapshot_meta: Dict[str, Any]) -> str:
    """
    功能：构造结构化 weights_snapshot 错误。

    Build structured weights snapshot error message.

    Args:
        snapshot_error: Error message from snapshot computation.
        snapshot_meta: Snapshot metadata mapping.

    Returns:
        JSON-encoded error string.
    """
    payload = {
        "code": "weights_snapshot_failed",
        "stage": "weights_snapshot",
        "message": snapshot_error,
        "model_id": snapshot_meta.get("model_id", "<absent>"),
        "model_source": snapshot_meta.get("model_source", "<absent>")
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)
