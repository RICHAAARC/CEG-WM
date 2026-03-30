"""
模型权重快照摘要计算

功能说明：
- 计算模型权重快照的可复算摘要并返回结构化元信息。
- 禁止抛异常，所有错误均以 error 字段返回。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from main.core import digests


def resolve_effective_weights_snapshot_inputs(
    *,
    model_id: str | None,
    model_source: str | None,
    resolved_model_id: str | None = None,
    resolved_model_source: str | None = None,
    model_snapshot_path: str | None = None,
    hf_revision: str | None = None,
    local_files_only: bool | None = None,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
    """
    功能：解析权重快照摘要计算所使用的 effective 输入。

    Resolve the effective inputs used by weights snapshot digest computation.

    Priority order:
        1. Use resolved_model_id and resolved_model_source when both are present.
        2. Otherwise, when model_snapshot_path is a valid local directory, use
           it as effective_model_id with effective_model_source="local_path".
        3. Otherwise, fall back to the raw requested model_id and model_source.

    Args:
        model_id: Raw requested model identifier.
        model_source: Raw requested model source.
        resolved_model_id: Resolved model identifier from runtime/build metadata.
        resolved_model_source: Resolved model source from runtime/build metadata.
        model_snapshot_path: Bound local snapshot directory when available.
        hf_revision: Requested HF revision.
        local_files_only: Local-only resolution flag.
        cache_dir: Optional HF cache directory.

    Returns:
        Structured mapping containing requested, resolved, and effective inputs.
    """
    requested_model_id = _normalize_optional_text(model_id)
    requested_model_source = _normalize_optional_text(model_source)
    normalized_resolved_model_id = _normalize_optional_text(resolved_model_id)
    normalized_resolved_model_source = _normalize_optional_text(resolved_model_source)
    normalized_model_snapshot_path = _normalize_valid_local_snapshot_path(model_snapshot_path)
    normalized_hf_revision = _normalize_revision(hf_revision)
    normalized_cache_dir = _normalize_optional_text(cache_dir)

    effective_model_id = requested_model_id
    effective_model_source = requested_model_source
    effective_resolution = "requested_fallback"

    if normalized_resolved_model_id is not None and normalized_resolved_model_source is not None:
        effective_model_id = normalized_resolved_model_id
        effective_model_source = normalized_resolved_model_source
        effective_resolution = "resolved_binding_priority"
    elif normalized_model_snapshot_path is not None:
        effective_model_id = normalized_model_snapshot_path
        effective_model_source = "local_path"
        effective_resolution = "model_snapshot_path_priority"

    return {
        "requested_model_id": requested_model_id,
        "requested_model_source": requested_model_source,
        "resolved_model_id": normalized_resolved_model_id,
        "resolved_model_source": normalized_resolved_model_source,
        "model_snapshot_path": normalized_model_snapshot_path,
        "effective_model_id": effective_model_id,
        "effective_model_source": effective_model_source,
        "effective_hf_revision": normalized_hf_revision,
        "effective_local_files_only": local_files_only if isinstance(local_files_only, bool) else None,
        "effective_cache_dir": normalized_cache_dir,
        "effective_resolution": effective_resolution,
    }


def compute_weights_snapshot_sha256(
    model_id: str,
    model_source: str | None,
    hf_revision: str | None,
    local_files_only: bool | None,
    cache_dir: str | None = None
) -> Tuple[str, Dict[str, Any], str | None]:
    """
    功能：计算权重快照摘要。

    Compute weights snapshot digest with structured metadata.

    Args:
        model_id: Model identifier or local path.
        model_source: Model source indicator.
        hf_revision: Revision string or None.
        local_files_only: Whether to use local-only resolution.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        Tuple of (weights_snapshot_sha256, snapshot_meta, error_or_none).

    Raises:
        None.
    """
    snapshot_meta = _init_snapshot_meta(model_id, model_source, hf_revision, local_files_only, cache_dir)

    if not isinstance(model_id, str) or not model_id:
        snapshot_meta["snapshot_status"] = "failed"
        return "<absent>", snapshot_meta, "model_id must be non-empty str"
    if not isinstance(model_source, str) or not model_source:
        snapshot_meta["snapshot_status"] = "failed"
        return "<absent>", snapshot_meta, "model_source must be non-empty str"

    normalized_revision = _normalize_revision(hf_revision)

    try:
        if model_source == "local_path":
            snapshot_dir, error = _resolve_local_snapshot_dir(model_id)
            if error is not None:
                snapshot_meta["snapshot_status"] = "failed"
                return "<absent>", snapshot_meta, error
            return _compute_snapshot_digest(snapshot_dir, snapshot_meta)

        # 支持 "hf" 和 "hf_hub" 两种标识（向后兼容）
        if model_source in ("hf", "hf_hub"):
            snapshot_dir, resolved_revision, error = _resolve_hf_snapshot_dir(
                model_id,
                normalized_revision,
                local_files_only,
                cache_dir
            )
            if resolved_revision is not None:
                snapshot_meta["resolved_revision"] = resolved_revision
                snapshot_meta["snapshot_provenance_anchors"]["revision_resolved"] = resolved_revision
            if error is not None:
                snapshot_meta["snapshot_status"] = "failed"
                return "<absent>", snapshot_meta, error
            
            digest, snapshot_meta, digest_error = _compute_snapshot_digest(snapshot_dir, snapshot_meta)
            if digest_error is None:
                snapshot_meta["snapshot_provenance_anchors"]["snapshot_digest"] = digest
            return digest, snapshot_meta, digest_error

        snapshot_meta["snapshot_status"] = "failed"
        return "<absent>", snapshot_meta, f"model_source not supported: {model_source}"
    except Exception as exc:
        snapshot_meta["snapshot_status"] = "failed"
        return "<absent>", snapshot_meta, f"{type(exc).__name__}: {exc}"


def _init_snapshot_meta(
    model_id: Any,
    model_source: Any,
    hf_revision: Any,
    local_files_only: Any,
    cache_dir: Any
) -> Dict[str, Any]:
    """
    功能：初始化快照元信息。

    Initialize snapshot metadata with explicit <absent> fields.
    
    增强可复现性锚定（建议项）：
    - 添加 resolved_revision 字段用于记录实际使用的 commit hash
    - 添加 snapshot_provenance_anchors 字段建议在 run_closure 中引用
    - 这些字段降低供应链漂移风险，确保模型权重可追溯

    Args:
        model_id: Model identifier.
        model_source: Model source indicator.
        hf_revision: Revision string or None.
        local_files_only: Local-only flag.
        cache_dir: Cache directory.

    Returns:
        Snapshot metadata mapping.
    """
    return {
        "model_id": model_id if isinstance(model_id, str) and model_id else "<absent>",
        "model_source": model_source if isinstance(model_source, str) and model_source else "<absent>",
        "hf_revision": hf_revision if isinstance(hf_revision, str) and hf_revision else "<absent>",
        "resolved_revision": "<absent>",  # 实际使用的 commit hash（可复现锚定）
        "local_files_only": local_files_only if isinstance(local_files_only, bool) else "<absent>",
        "cache_dir": cache_dir if isinstance(cache_dir, str) and cache_dir else "<absent>",
        "snapshot_dir": "<absent>",
        "file_count": 0,
        "total_bytes": 0,
        "snapshot_status": "unbuilt",
        # 可复现性锚定建议字段（供 run_closure.provenance 引用）
        "snapshot_provenance_anchors": {
            "repo_id": model_id if isinstance(model_id, str) and model_id else "<absent>",
            "revision_requested": hf_revision if isinstance(hf_revision, str) and hf_revision else "<absent>",
            "revision_resolved": "<absent>",  # 将由 _resolve_hf_snapshot_dir 填充
            "snapshot_digest": "<absent>",  # 将由 _compute_snapshot_digest 填充
            "local_files_only_enforced": True  # 当前版本强制离线模式
        }
    }


def _normalize_optional_text(value: Any) -> str | None:
    """
    功能：规范化可选文本字段。

    Normalize an optional textual field.

    Args:
        value: Candidate textual value.

    Returns:
        Stripped text, or None when unavailable.
    """
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    if not normalized_value or normalized_value == "<absent>":
        return None
    return normalized_value


def _normalize_valid_local_snapshot_path(model_snapshot_path: Any) -> str | None:
    """
    功能：规范化并校验本地 snapshot 路径。

    Normalize and validate a local snapshot directory path.

    Args:
        model_snapshot_path: Candidate local snapshot path.

    Returns:
        Resolved local directory path in POSIX form, or None when invalid.
    """
    normalized_path = _normalize_optional_text(model_snapshot_path)
    if normalized_path is None:
        return None

    try:
        path_obj = Path(normalized_path).expanduser().resolve()
    except Exception:
        return None

    if not path_obj.exists() or not path_obj.is_dir():
        return None
    return path_obj.as_posix()


def _normalize_revision(hf_revision: Any) -> str | None:
    """
    功能：规范化 revision 输入。

    Normalize revision value to string or None.

    Args:
        hf_revision: Revision input.

    Returns:
        Normalized revision string or None.
    """
    if isinstance(hf_revision, str) and hf_revision and hf_revision != "<absent>":
        return hf_revision
    return None


def _resolve_local_snapshot_dir(model_id: str) -> Tuple[Path | None, str | None]:
    """
    功能：解析本地路径快照根。

    Resolve local snapshot root path.

    Args:
        model_id: Local path string.

    Returns:
        Tuple of (snapshot_path_or_none, error_or_none).
    """
    path = Path(model_id)
    if not path.exists():
        return None, f"local_path not found: {path}"
    return path.resolve(), None


def _resolve_hf_snapshot_dir(
    model_id: str,
    revision: str | None,
    local_files_only: bool | None,
    cache_dir: str | None
) -> Tuple[Path | None, str | None, str | None]:
    """
    功能：解析 Hugging Face snapshot 目录。

    Resolve Hugging Face snapshot directory via snapshot_download.

    Note: This helper must stay consistent with the actual pipeline build
    semantics. When the build path allows HF Hub downloads, snapshot resolution
    must record and reuse the same local_files_only policy instead of rejecting
    it as a contradictory provenance state.

    Args:
        model_id: HF repo id.
        revision: Optional revision.
        local_files_only: Local-only flag used by snapshot_download.
        cache_dir: Optional cache directory.

    Returns:
        Tuple of (snapshot_path_or_none, resolved_revision_or_none, error_or_none).
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return None, None, f"huggingface_hub import failed: {type(exc).__name__}: {exc}"

    resolved_local_files_only = local_files_only if isinstance(local_files_only, bool) else True

    # snapshot 解析必须与真实 pipeline build 语义一致；若构建阶段允许 HF 下载，
    # 这里也必须接受同一 local_files_only 配置，而不是自相矛盾地拒绝它。
    download_kwargs: Dict[str, Any] = {
        "repo_id": model_id,
        "local_files_only": resolved_local_files_only
    }
    if revision is not None:
        download_kwargs["revision"] = revision
    if isinstance(cache_dir, str) and cache_dir:
        download_kwargs["cache_dir"] = cache_dir

    try:
        snapshot_path = snapshot_download(**download_kwargs)
    except Exception as exc:
        return None, None, f"snapshot_download failed: {type(exc).__name__}: {exc}"

    # 尝试从 snapshot_path 提取 resolved_revision（可复现锚定核查）。
    # Hugging Face 缓存结构通常为：cache_dir/models--{org}--{name}/snapshots/{commit_hash}
    # 提取 commit_hash 作为 resolved_revision 以增强可复现性。
    resolved_snapshot_path = Path(snapshot_path).resolve()
    resolved_revision = None
    
    try:
        # 提取路径中的 commit hash（最后一级目录名）
        if "snapshots" in resolved_snapshot_path.parts:
            snapshots_idx = resolved_snapshot_path.parts.index("snapshots")
            if snapshots_idx + 1 < len(resolved_snapshot_path.parts):
                resolved_revision = resolved_snapshot_path.parts[snapshots_idx + 1]
    except (ValueError, IndexError):
        pass

    return resolved_snapshot_path, resolved_revision, None


def _compute_snapshot_digest(snapshot_path: Path, snapshot_meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str | None]:
    """
    功能：计算快照目录摘要。

    Compute snapshot digest by hashing all files under snapshot_path.

    Args:
        snapshot_path: Snapshot root path.
        snapshot_meta: Snapshot metadata mapping to mutate.

    Returns:
        Tuple of (weights_snapshot_sha256, snapshot_meta, error_or_none).
    """
    if not isinstance(snapshot_path, Path):
        snapshot_meta["snapshot_status"] = "failed"
        return "<absent>", snapshot_meta, "snapshot_path must be Path"

    if snapshot_path.is_file():
        files = [_build_file_entry(snapshot_path, snapshot_path.parent)]
    else:
        files = _collect_snapshot_files(snapshot_path)

    file_count = len(files)
    total_bytes = sum(item.get("size_bytes", 0) for item in files)
    snapshot_meta["snapshot_dir"] = str(snapshot_path)
    snapshot_meta["file_count"] = file_count
    snapshot_meta["total_bytes"] = total_bytes

    digest_value = digests.canonical_sha256({"files": files})
    snapshot_meta["snapshot_status"] = "built"
    return digest_value, snapshot_meta, None


def _collect_snapshot_files(snapshot_path: Path) -> List[Dict[str, Any]]:
    """
    功能：收集快照文件条目。

    Collect snapshot file entries from a directory.

    Args:
        snapshot_path: Snapshot root path.

    Returns:
        List of file entries.
    """
    entries: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(snapshot_path):
        dirs.sort()
        files.sort()
        for filename in files:
            file_path = Path(root) / filename
            if not file_path.is_file():
                continue
            entries.append(_build_file_entry(file_path, snapshot_path))
    entries.sort(key=lambda item: item.get("path", ""))
    return entries


def _build_file_entry(file_path: Path, root_path: Path) -> Dict[str, Any]:
    """
    功能：构造单文件快照条目。

    Build a file entry for snapshot digest.

    Args:
        file_path: File path.
        root_path: Root path to compute relative path.

    Returns:
        File entry mapping.
    """
    relative_path = file_path.relative_to(root_path).as_posix()
    file_sha256 = digests.file_sha256(file_path)
    size_bytes = file_path.stat().st_size
    return {
        "path": relative_path,
        "file_sha256": file_sha256,
        "size_bytes": size_bytes
    }
