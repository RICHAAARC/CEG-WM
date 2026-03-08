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
import struct
from typing import Any, List, Sequence, Union


def _phi_latent(latent: Any) -> bytes:
    """
    功能：从单个 latent 张量提取紧凑摘要 u_i。

    Compute a compact summary u_i from a single latent snapshot.
    phi(latent) = bytes(mean) || bytes(l2_norm)

    Uses double-precision (8 bytes each) for reproducibility.

    Args:
        latent: Latent snapshot as numpy array or torch.Tensor.
                Must be a numeric array-like with .mean() and linalg.norm().

    Returns:
        16-byte compact summary (mean as float64 || l2_norm as float64).

    Raises:
        TypeError: If latent is not a supported array-like type.
        ValueError: If latent is empty.
    """
    # 优先使用 torch（避免不必要的 GPU→CPU 数据搬移）。
    if hasattr(latent, "cpu"):
        import torch
        t = latent.cpu().detach()
        flat = t.reshape(-1).to(dtype=torch.float64)
        if flat.numel() == 0:
            raise ValueError("latent snapshot must not be empty")
        mean_val = float(flat.mean().item())
        norm_val = float(torch.linalg.vector_norm(flat).item())
    elif hasattr(latent, "mean") and hasattr(latent, "flatten"):
        import numpy as np
        arr = np.asarray(latent, dtype=np.float64).flatten()
        if arr.size == 0:
            raise ValueError("latent snapshot must not be empty")
        mean_val = float(arr.mean())
        norm_val = float(np.linalg.norm(arr))
    else:
        raise TypeError(
            f"latent must be torch.Tensor or numpy.ndarray, got {type(latent).__name__}"
        )

    # 打包为固定字节（小端双精度浮点，8 字节 × 2 = 16 字节）。
    return struct.pack("<dd", mean_val, norm_val)


def compute_trace_summary(latent_snapshots: Sequence[Any]) -> bytes:
    """
    功能：从多个时间步的 latent 快照聚合 trace_summary。

    Aggregate a trace summary from multiple latent snapshots.
    trace_summary = SHA256(u_1 || u_2 || ... || u_n)

    At least one snapshot is required. If the sequence is empty,
    the function raises ValueError.

    Args:
        latent_snapshots: Sequence of latent snapshots (numpy arrays or torch.Tensors).
                         Recommended: 3–5 evenly-spaced diffusion steps.

    Returns:
        32-byte SHA256 hash of concatenated per-step summaries.

    Raises:
        ValueError: If latent_snapshots is empty.
        TypeError: If any snapshot is of unsupported type.
    """
    if not latent_snapshots:
        raise ValueError("latent_snapshots must be non-empty")

    concatenated = b""
    for i, snapshot in enumerate(latent_snapshots):
        try:
            u_i = _phi_latent(snapshot)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"snapshot[{i}]: {exc}") from exc
        concatenated += u_i

    return hashlib.sha256(concatenated).digest()


def compute_trajectory_commit(
    k_tr: Union[str, bytes],
    latent_snapshots: Sequence[Any],
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
                         Recommended: 3–5 evenly-spaced steps.

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

    trace_summary = compute_trace_summary(latent_snapshots)
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
