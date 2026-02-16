"""
Pipeline 溯源

功能说明：
- 管道溯源 (pipeline provenance, PP) 结构的构造与规范化摘要计算。
- 缺失字段必须显式标记为 <absent>。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests


def build_pipeline_provenance(
    cfg: Dict[str, Any],
    pipeline_impl_id: str,
    pipeline_meta: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    功能：构造管道溯源对象。

    Build pipeline provenance mapping with explicit <absent> semantics.

    Args:
        cfg: Configuration mapping.
        pipeline_impl_id: Pipeline implementation identifier.
        pipeline_meta: Optional pipeline metadata mapping from registry factory.

    Returns:
        Pipeline provenance mapping with explicit fields.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If pipeline_impl_id is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(pipeline_impl_id, str) or not pipeline_impl_id:
        # pipeline_impl_id 输入不合法，必须 fail-fast。
        raise ValueError("pipeline_impl_id must be non-empty str")
    if pipeline_meta is not None and not isinstance(pipeline_meta, dict):
        # pipeline_meta 类型不合法，必须 fail-fast。
        raise TypeError("pipeline_meta must be dict or None")

    pipeline_meta = pipeline_meta or {}

    provenance = {
        "pipeline_impl_id": pipeline_impl_id,
        "pipeline_impl_version": _resolve_optional_str(pipeline_meta.get("pipeline_impl_version")),
        "pipeline_impl_digest": _resolve_optional_str(pipeline_meta.get("pipeline_impl_digest")),
        "pipeline_factory_entry": "main.diffusion.sd3.pipeline_factory",
        "model_id": _resolve_optional_str(cfg.get("model_id")),
        "hf_revision": _resolve_optional_str(cfg.get("hf_revision")),
        "resolved_revision": _resolve_optional_str(cfg.get("resolved_revision")),
        "model_revision": _resolve_optional_str(cfg.get("model_revision")),
        "model_source": _resolve_optional_str(cfg.get("model_source")),
        "model_weights_sha256": _resolve_optional_str(cfg.get("model_weights_sha256")),
        "weights_snapshot_sha256": _resolve_optional_str(cfg.get("weights_snapshot_sha256")),
        "local_files_only": _resolve_optional_bool(cfg.get("local_files_only"))
    }
    return provenance


def compute_pipeline_provenance_canon_sha256(obj: Dict[str, Any]) -> str:
    """
    功能：计算管道溯源对象的规范化摘要。

    Compute canonical digest for pipeline provenance mapping.

    Args:
        obj: Pipeline provenance mapping.

    Returns:
        Canonical SHA256 digest string.

    Raises:
        TypeError: If obj is invalid.
    """
    if not isinstance(obj, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("obj must be dict")

    return digests.canonical_sha256(obj)


def _resolve_optional_str(value: Any) -> str:
    """
    功能：将可选值规范化为字符串或 <absent>。

    Normalize optional value to non-empty string or <absent>.

    Args:
        value: Optional value to normalize.

    Returns:
        Non-empty string or "<absent>" sentinel.

    Raises:
        None.
    """
    if isinstance(value, str) and value:
        return value
    return "<absent>"


def _resolve_optional_bool(value: Any) -> bool | str:
    """
    功能：将可选布尔值规范化为 bool 或 <absent>。

    Normalize optional boolean value to bool or <absent>.

    Args:
        value: Optional value to normalize.

    Returns:
        Boolean value or "<absent>" sentinel.

    Raises:
        None.
    """
    if isinstance(value, bool):
        return value
    return "<absent>"
