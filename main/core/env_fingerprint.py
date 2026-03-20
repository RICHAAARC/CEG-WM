"""
环境指纹

功能说明：
- 生成可序列化的环境指纹 (environment fingerprint, EF)。
- 缺失字段必须显式标记为 <absent>。
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, cast

from main.core import digests


def build_env_fingerprint() -> Dict[str, Any]:
    """
    功能：构造环境指纹对象。

    Build environment fingerprint mapping with explicit <absent> semantics.

    Args:
        None.

    Returns:
        Environment fingerprint mapping.
    """
    python_version = platform.python_version()
    platform_info = platform.platform()
    sys_platform = sys.platform

    torch_version = "<absent>"
    cuda_available = "<absent>"
    try:
        import torch

        torch_version = getattr(torch, "__version__", "<absent>")
        cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        torch_version = "<absent>"
        cuda_available = "<absent>"

    return {
        "python_version": python_version or "<absent>",
        "platform": platform_info or "<absent>",
        "sys_platform": sys_platform or "<absent>",
        "executable": sys.executable or "<absent>",
        "torch_version": torch_version or "<absent>",
        "cuda_available": cuda_available
    }


def compute_env_fingerprint_canon_sha256(obj: Dict[str, Any]) -> str:
    """
    功能：计算环境指纹的规范化摘要。

    Compute canonical digest for environment fingerprint mapping.

    Args:
        obj: Environment fingerprint mapping.

    Returns:
        Canonical SHA256 digest string.

    Raises:
        TypeError: If obj is invalid.
    """
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("obj must be dict")

    return digests.canonical_sha256(cast(Dict[str, Any], obj_value))
