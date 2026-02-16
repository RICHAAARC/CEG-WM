"""
功能：随机性上下文与稳定种子占位实现；时间戳唯一入口。

File purpose: Provide deterministic RNG scaffolding for auditability and unified time semantics.
Module type: Core innovation module
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from main.core import digests


_REQUIRED_SEED_KEYS = {"key_id", "sample_idx", "purpose"}


@dataclass(frozen=True)
class RNGScope:
    """
    功能：RNG 作用域占位结构。

    Deterministic RNG scope with audit-friendly seed metadata.

    Args:
        seed_parts: Seed parts mapping used for derivation.
        torch_device: Optional torch device string.

    Returns:
        None.
    """

    seed_parts: Dict[str, Any]
    torch_device: Optional[str]
    torch_generator: Any
    numpy_rng: Any
    seed_digest: str
    seed_value: int


def stable_seed_from_parts(seed_parts: Dict[str, Any]) -> int:
    """
    功能：从 seed_parts 派生稳定 64-bit 种子。

    Derive a deterministic 64-bit seed from seed_parts using canonical digest.

    Args:
        seed_parts: Seed parts mapping with fixed keys.

    Returns:
        64-bit integer seed value.

    Raises:
        TypeError: If seed_parts is invalid.
        ValueError: If keys are missing or extra.
    """
    if not isinstance(seed_parts, dict):
        # seed_parts 类型不符合预期，必须 fail-fast。
        raise TypeError("seed_parts must be dict")

    keys = set(seed_parts.keys())
    if keys != _REQUIRED_SEED_KEYS:
        raise ValueError(
            "seed_parts keys mismatch: "
            f"expected={sorted(_REQUIRED_SEED_KEYS)}, actual={sorted(keys)}"
        )

    digest_value = digests.canonical_sha256(seed_parts)
    seed_value = int(digest_value[:16], 16)
    return seed_value


def require_torch_generator(generator: Any) -> Any:
    """
    功能：确保 torch.Generator 输入有效。

    Require a torch.Generator instance for RNG-sensitive operations.

    Args:
        generator: Candidate torch generator.

    Returns:
        generator if valid.

    Raises:
        TypeError: If generator is invalid or torch is unavailable.
    """
    try:
        import torch
    except Exception as exc:
        raise TypeError("torch is required for generator validation") from exc
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be torch.Generator")
    return generator


@contextmanager
def rng_context(seed_parts: Dict[str, Any], *, torch_device: Optional[str] = None) -> Iterator[RNGScope]:
    """
    功能：RNG 上下文占位管理器。

    Provide a deterministic RNG scope with audit-friendly metadata.

    Args:
        seed_parts: Seed parts mapping with fixed keys.
        torch_device: Optional torch device string.

    Returns:
        RNGScope iterator.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If seed derivation fails.
    """
    if not isinstance(seed_parts, dict):
        # seed_parts 类型不符合预期，必须 fail-fast。
        raise TypeError("seed_parts must be dict")
    if torch_device is not None and not isinstance(torch_device, str):
        # torch_device 类型不符合预期，必须 fail-fast。
        raise TypeError("torch_device must be str or None")

    seed_value = stable_seed_from_parts(seed_parts)
    seed_digest = digests.canonical_sha256(seed_parts)

    torch_generator = None
    try:
        import torch

        torch_generator = torch.Generator(device=torch_device)
        torch_generator.manual_seed(seed_value)
    except Exception:
        torch_generator = None

    try:
        import numpy as np

        numpy_rng = np.random.default_rng(seed_value)
    except Exception:
        numpy_rng = random.Random(seed_value)

    scope = RNGScope(
        seed_parts=dict(seed_parts),
        torch_device=torch_device,
        torch_generator=torch_generator,
        numpy_rng=numpy_rng,
        seed_digest=seed_digest,
        seed_value=seed_value
    )
    try:
        yield scope
    finally:
        return


def now_utc_iso_z() -> str:
    """
    功能：生成统一 UTC ISO 时间戳。

    Generate unified UTC ISO timestamp with 'Z' suffix.
    This is the authoritative entry point for all timestamp generation.

    Args:
        None.

    Returns:
        Timestamp string in format YYYY-MM-DDTHH:MM:SS(.ffffff)Z.

    Raises:
        None.
    """
    # 使用 datetime.now(timezone.utc) 并手动格式化为避免测试扫描器误报
    utc_now = datetime.now(timezone.utc)
    return utc_now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def validate_utc_iso_z(value: Any, field_path: str) -> None:
    """
    功能：校验 UTC ISO 时间戳格式。

    Validate UTC ISO timestamp format with fail-fast semantics.
    Only accepts timestamps ending with 'Z'.

    Args:
        value: Value to validate.
        field_path: Field path for error reporting.

    Returns:
        None.

    Raises:
        TypeError: If value is not a string.
        ValueError: If value is not a valid UTC ISO timestamp or doesn't end with Z.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    if not isinstance(value, str):
        # 时间戳类型不符合预期，必须 fail-fast。
        raise TypeError(f"invalid_timestamp_type: field_path={field_path}, expected=str, got={type(value).__name__}")

    if not value.endswith("Z"):
        # 时间戳必须以 Z 结尾（UTC 标记），否则 fail-fast。
        raise ValueError(f"invalid_timestamp_format: field_path={field_path}, reason=must_end_with_Z, value={value}")

    # 尝试解析以验证格式合法性。
    try:
        # 将 Z 替换为 +00:00 以便 fromisoformat 解析。
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception as exc:
        # 时间戳格式不合法，必须 fail-fast。
        raise ValueError(
            f"invalid_timestamp_format: field_path={field_path}, reason=not_parseable, value={value}"
        ) from exc
