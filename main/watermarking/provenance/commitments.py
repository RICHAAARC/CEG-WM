"""
File purpose: HMAC 承诺函数，用于 prompt 和 seed 的密码学承诺。
Module type: Core innovation module

创新边界：
    - 不存储原始 prompt 或 seed，只存储 HMAC 承诺值。
    - 承诺用于绑定生成事件，防止重放伪造。
    - 所有承诺均使用 HMAC-SHA256，符合 RFC 2104。

依赖假设：
    - k_prompt / k_seed 通过安全信道注入，不在此模块管理。
    - prompt 归一化规则固定（去首尾空格 + 合并连续空白）。
"""

from __future__ import annotations

import hashlib
import hmac
import unicodedata
import re
from typing import Union


def _normalize_prompt(prompt: str) -> str:
    """
    功能：规范化 prompt 字符串，用于承诺计算。

    Normalize prompt string before commitment computation.
    Applies Unicode NFC normalization, strips leading/trailing whitespace,
    and collapses internal whitespace runs to single spaces.

    Args:
        prompt: Raw prompt string from user or config.

    Returns:
        Normalized prompt string (deterministic for equivalent inputs).

    Raises:
        TypeError: If prompt is not a str.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    # (1) Unicode NFC 规范化：统一等价字符表示。
    normalized = unicodedata.normalize("NFC", prompt)
    # (2) 去除首尾空白。
    normalized = normalized.strip()
    # (3) 合并内部连续空白为单空格。
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def compute_prompt_commit(k_prompt: Union[str, bytes], prompt: str) -> str:
    """
    功能：使用 HMAC 承诺 prompt，不存储原始 prompt。

    Compute a cryptographic commitment of a prompt string using HMAC-SHA256.
    The raw prompt is never stored; only the commitment is retained.

    Commitment formula:
        prompt_commit = HMAC-SHA256(k_prompt, normalize(prompt))

    Args:
        k_prompt: Commitment key for prompt, str (hex) or raw bytes.
        prompt: Raw prompt string to commit.

    Returns:
        Lowercase hex string of HMAC-SHA256 commitment (64 chars).

    Raises:
        TypeError: If k_prompt or prompt are of invalid type.
        ValueError: If k_prompt is empty.
    """
    if isinstance(k_prompt, str):
        if not k_prompt:
            raise ValueError("k_prompt must be non-empty")
        # 将十六进制字符串解码为字节，若不是合法十六进制则直接 UTF-8 编码。
        try:
            key_bytes = bytes.fromhex(k_prompt)
        except ValueError:
            key_bytes = k_prompt.encode("utf-8")
    elif isinstance(k_prompt, bytes):
        if not k_prompt:
            raise ValueError("k_prompt must be non-empty")
        key_bytes = k_prompt
    else:
        raise TypeError(f"k_prompt must be str or bytes, got {type(k_prompt).__name__}")

    normalized = _normalize_prompt(prompt)
    message_bytes = normalized.encode("utf-8")
    mac = hmac.new(key_bytes, message_bytes, hashlib.sha256)
    return mac.hexdigest()


def compute_seed_commit(k_seed: Union[str, bytes], seed: int) -> str:
    """
    功能：使用 HMAC 承诺生成种子，不存储原始 seed。

    Compute a cryptographic commitment of an integer generation seed using HMAC-SHA256.
    The raw seed is never stored; only the commitment is retained.

    Commitment formula:
        seed_commit = HMAC-SHA256(k_seed, str(seed).encode("utf-8"))

    Args:
        k_seed: Commitment key for seed, str (hex) or raw bytes.
        seed: Integer generation seed.

    Returns:
        Lowercase hex string of HMAC-SHA256 commitment (64 chars).

    Raises:
        TypeError: If k_seed or seed are of invalid type.
        ValueError: If k_seed is empty.
    """
    if isinstance(k_seed, str):
        if not k_seed:
            raise ValueError("k_seed must be non-empty")
        try:
            key_bytes = bytes.fromhex(k_seed)
        except ValueError:
            key_bytes = k_seed.encode("utf-8")
    elif isinstance(k_seed, bytes):
        if not k_seed:
            raise ValueError("k_seed must be non-empty")
        key_bytes = k_seed
    else:
        raise TypeError(f"k_seed must be str or bytes, got {type(k_seed).__name__}")

    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    # 将 seed 序列化为小端 8 字节（带符号），超出范围时回退到 UTF-8 字符串。
    try:
        message_bytes = seed.to_bytes(8, byteorder="little", signed=True)
    except OverflowError:
        message_bytes = str(seed).encode("utf-8")

    mac = hmac.new(key_bytes, message_bytes, hashlib.sha256)
    return mac.hexdigest()
