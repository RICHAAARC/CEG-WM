"""
File purpose: 生成轨迹承诺（generation trajectory commit），将水印与真实生成事件绑定。
Module type: Core innovation module

创新边界：
    - 从 diffusion 去噪轨迹中采样多个时间步的 latent 摘要，聚合为 trace_summary。
    - 使用 HMAC(k_TR, trace_summary) 计算 trace_commit，绑定生成轨迹。
    - 支持将 trace_commit 与 LF payload 混合（方案 B），使 attestation 绑定生成事件。
    - phi(latent) 为简化摘要函数（均值 + L2 范数）。

依赖假设：
    - latent_snapshots 为时间步 latent 的列表，每个元素是 numpy array 或 torch.Tensor。
    - k_TR 由 key_derivation 模块派生。
    - 仅依赖标准库 hmac / hashlib，numpy 为可选依赖（latent 摘要计算）。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from typing import Any, Dict, List, Sequence, Union


_TRACE_SUMMARY_VERSION = "trajectory_compact_summary_v2"
_DEFAULT_MAX_SNAPSHOTS = 4
_DEFAULT_GROUP_COUNT = 4
_DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    """
    功能：生成 trajectory summary 使用的 canonical JSON bytes。

    Serialize a JSON payload with deterministic ordering and spacing.

    Args:
        payload: JSON-like mapping.

    Returns:
        UTF-8 encoded canonical JSON bytes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _select_snapshot_indices(total_steps: int, max_snapshots: int) -> List[int]:
    """
    功能：为 compact summary 选择关键步索引。

    Select a deterministic set of critical denoising steps for compact summary
    construction. The selection covers the beginning, middle, and terminal phase.

    Args:
        total_steps: Total snapshot count.
        max_snapshots: Maximum number of steps to select.

    Returns:
        Sorted list of selected indices.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if max_snapshots <= 0:
        raise ValueError("max_snapshots must be positive")
    if total_steps <= max_snapshots:
        return list(range(total_steps))

    selected = {0, total_steps - 1}
    while len(selected) < max_snapshots:
        ratio = float(len(selected)) / float(max_snapshots - 1)
        candidate = int(round(ratio * float(total_steps - 1)))
        selected.add(max(0, min(total_steps - 1, candidate)))
        if len(selected) >= max_snapshots:
            break
        midpoint = int(round((min(selected) + max(selected)) / 2.0))
        selected.add(max(0, min(total_steps - 1, midpoint)))
    return sorted(selected)[:max_snapshots]


def _flatten_latent(latent: Any) -> tuple[Any, tuple[int, ...]]:
    """
    功能：将 latent 规范化为一维 float64 向量与 shape。

    Normalize a latent snapshot into a float64 flat vector and its original shape.

    Args:
        latent: Latent snapshot as numpy array or torch.Tensor.

    Returns:
        Tuple of (flat_vector, original_shape).

    Raises:
        TypeError: If latent is unsupported.
        ValueError: If latent is empty.
    """
    if hasattr(latent, "cpu"):
        import torch

        tensor = latent.detach().cpu().to(dtype=torch.float64)
        flat = tensor.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("latent snapshot must not be empty")
        return flat, tuple(int(dim) for dim in tensor.shape)
    if hasattr(latent, "mean") and hasattr(latent, "shape"):
        import numpy as np

        array = np.asarray(latent, dtype=np.float64)
        flat = array.reshape(-1)
        if flat.size == 0:
            raise ValueError("latent snapshot must not be empty")
        return flat, tuple(int(dim) for dim in array.shape)
    raise TypeError(f"latent must be torch.Tensor or numpy.ndarray, got {type(latent).__name__}")


def _phi_latent(latent: Any) -> bytes:
    """
    功能：从单个 latent 张量提取紧凑摘要 u_i。

    Compute a compact sketch from a single latent snapshot. The summary packs
    shape metadata and a fixed set of low-dimensional statistics extracted from
    the latent. Larger values retain stronger evidence of a particular denoising
    state, while remaining compact and reproducible.

    Args:
        latent: Latent snapshot as numpy array or torch.Tensor.
                Must be a numeric array-like with .mean() and linalg.norm().

    Returns:
        Binary compact sketch containing shape metadata and fixed float64 features.

    Raises:
        TypeError: If latent is not a supported array-like type.
        ValueError: If latent is empty.
    """
    flat, shape = _flatten_latent(latent)
    if hasattr(flat, "numel"):
        import torch

        numel = int(flat.numel())
        mean_val = float(flat.mean().item())
        std_val = float(flat.std(unbiased=False).item())
        norm_val = float(torch.linalg.vector_norm(flat).item())
        abs_mean = float(flat.abs().mean().item())
        quantile_values = [float(torch.quantile(flat, q).item()) for q in _DEFAULT_QUANTILES]
        group_means = []
        for group_idx in range(_DEFAULT_GROUP_COUNT):
            group_tensor = flat[group_idx::_DEFAULT_GROUP_COUNT]
            if int(group_tensor.numel()) <= 0:
                group_means.append(0.0)
            else:
                group_means.append(float(group_tensor.mean().item()))
    else:
        import numpy as np

        array = flat
        numel = int(array.size)
        mean_val = float(array.mean())
        std_val = float(array.std())
        norm_val = float(np.linalg.norm(array))
        abs_mean = float(np.mean(np.abs(array)))
        quantile_values = [float(np.quantile(array, q)) for q in _DEFAULT_QUANTILES]
        group_means = []
        for group_idx in range(_DEFAULT_GROUP_COUNT):
            group_array = array[group_idx::_DEFAULT_GROUP_COUNT]
            if group_array.size <= 0:
                group_means.append(0.0)
            else:
                group_means.append(float(group_array.mean()))

    l2_per_element = float(norm_val / max(1.0, float(numel) ** 0.5))
    features = [
        mean_val,
        std_val,
        l2_per_element,
        abs_mean,
        *quantile_values,
        *group_means,
    ]
    shape_bytes = struct.pack("<I", len(shape)) + b"".join(struct.pack("<I", int(dim)) for dim in shape)
    feature_bytes = struct.pack(f"<{len(features)}d", *features)
    return shape_bytes + feature_bytes


def compute_trace_summary(
    latent_snapshots: Sequence[Any],
    summary_config: Dict[str, Any] | None = None,
) -> bytes:
    """
    功能：从多个时间步的 latent 快照聚合 trace_summary。

    Aggregate a trace summary from multiple latent snapshots.
    trace_summary = SHA256(header || phi(latent_t_i1) || ... || phi(latent_t_ik))
    where the selected steps are deterministic critical denoising steps.

    At least one snapshot is required. If the sequence is empty,
    the function raises ValueError.

    Args:
        latent_snapshots: Sequence of latent snapshots (numpy arrays or torch.Tensors).
        summary_config: Optional summary configuration mapping.

    Returns:
        32-byte SHA256 hash of concatenated per-step summaries.

    Raises:
        ValueError: If latent_snapshots is empty.
        TypeError: If any snapshot is of unsupported type.
    """
    if not latent_snapshots:
        raise ValueError("latent_snapshots must be non-empty")

    config = summary_config if isinstance(summary_config, dict) else {}
    max_snapshots = int(config.get("max_snapshots", _DEFAULT_MAX_SNAPSHOTS))
    max_snapshots = max(1, min(max_snapshots, len(latent_snapshots)))
    selected_indices = _select_snapshot_indices(len(latent_snapshots), max_snapshots)
    sketch_dim = 4 + len(_DEFAULT_QUANTILES) + _DEFAULT_GROUP_COUNT
    header_payload = {
        "summary_version": _TRACE_SUMMARY_VERSION,
        "selected_indices": selected_indices,
        "total_snapshots": len(latent_snapshots),
        "sketch_dim": sketch_dim,
    }

    concatenated = _canonical_json_bytes(header_payload)
    for selected_index in selected_indices:
        snapshot = latent_snapshots[selected_index]
        try:
            u_i = _phi_latent(snapshot)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"snapshot[{selected_index}]: {exc}") from exc
        concatenated += u_i

    return hashlib.sha256(concatenated).digest()


def compute_trajectory_commit(
    k_tr: Union[str, bytes],
    latent_snapshots: Sequence[Any],
    *,
    summary_config: Dict[str, Any] | None = None,
) -> str:
    """
    功能：计算生成轨迹承诺 trace_commit = HMAC(k_TR, trace_summary)。

    Compute a trajectory commit binding the generation trajectory to the attestation key.
    This ensures the watermark attests to an actual generation event, not merely a statement.

    Full formula:
        u_i = phi(latent_t_i)                        # per-step compact summary
        trace_summary = SHA256(u_1 || u_2 || ... )   # trajectory aggregation
        trace_commit  = HMAC-SHA256(k_TR, trace_summary)

    Args:
        k_tr: Trajectory commit key from key derivation (hex str or bytes).
        latent_snapshots: Sequence of latent snapshots from diffusion steps.
        summary_config: Optional compact summary configuration mapping.

    Returns:
        Lowercase hex string of HMAC-SHA256 trajectory commit (64 chars).

    Raises:
        TypeError: If k_tr is of invalid type or snapshots are incompatible.
        ValueError: If latent_snapshots is empty or k_tr is empty.
    """
    if isinstance(k_tr, str):
        if not k_tr:
            raise ValueError("k_tr must be non-empty")
        try:
            key_bytes = bytes.fromhex(k_tr)
        except ValueError:
            key_bytes = k_tr.encode("utf-8")
    elif isinstance(k_tr, bytes):
        if not k_tr:
            raise ValueError("k_tr must be non-empty")
        key_bytes = k_tr
    else:
        raise TypeError(f"k_tr must be str or bytes, got {type(k_tr).__name__}")

    trace_summary = compute_trace_summary(latent_snapshots, summary_config=summary_config)
    mac = hmac.new(key_bytes, trace_summary, hashlib.sha256)
    return mac.hexdigest()


def mix_payload_with_trace_commit(
    lf_payload: bytes,
    trace_commit_hex: str,
    mix_bytes: int = 8,
) -> bytes:
    """
    功能：将 trace_commit 的部分字节与 LF payload 混合（方案 B）。

    Mix the leading bytes of trace_commit into the LF attestation payload.
    This binds the embedded payload to both the statement and the generation trajectory.

    Mix formula (XOR of leading bytes):
        payload_final[:mix_bytes] ^= trace_commit_bytes[:mix_bytes]
        payload_final[mix_bytes:] = lf_payload[mix_bytes:]

    Args:
        lf_payload: Original LF attestation payload bytes.
        trace_commit_hex: Hex string of HMAC trajectory commit.
        mix_bytes: Number of leading bytes to XOR (default 8).

    Returns:
        Mixed payload bytes (same length as lf_payload).

    Raises:
        TypeError: If inputs are of invalid type.
        ValueError: If mix_bytes exceeds payload or commit length.
    """
    if not isinstance(lf_payload, bytes):
        raise TypeError(f"lf_payload must be bytes, got {type(lf_payload).__name__}")
    if not isinstance(trace_commit_hex, str) or not trace_commit_hex:
        raise ValueError("trace_commit_hex must be non-empty str")
    if not isinstance(mix_bytes, int) or mix_bytes <= 0:
        raise ValueError("mix_bytes must be positive int")

    try:
        commit_bytes = bytes.fromhex(trace_commit_hex)
    except ValueError:
        raise ValueError("trace_commit_hex must be a valid hex string")

    if mix_bytes > len(lf_payload):
        raise ValueError(
            f"mix_bytes ({mix_bytes}) exceeds lf_payload length ({len(lf_payload)})"
        )
    if mix_bytes > len(commit_bytes):
        raise ValueError(
            f"mix_bytes ({mix_bytes}) exceeds trace_commit length ({len(commit_bytes)})"
        )

    # XOR 混合前 mix_bytes 字节，后续字节保持不变。
    mixed = bytearray(lf_payload)
    for i in range(mix_bytes):
        mixed[i] ^= commit_bytes[i]
    return bytes(mixed)
