"""
File purpose: 从 attestation digest 派生四类子密钥（LF/HF/GEO/TR）。
Module type: Core innovation module

创新边界：
    - 实现 HKDF（RFC 5869）：Extract + Expand 两阶段。
    - 四类密钥严格 domain separation（context 标签不同）。
    - K_master 由调用方安全注入，不在此模块存储或生成。
    - 派生密钥输出为 hex 字符串（32 字节 / 64 hex chars）。

依赖假设：
    - d_A 为 32 字节 hex string（attestation digest）。
    - K_master 为 hex string 或 bytes（主密钥）。
    - 仅依赖标准库 hmac / hashlib。
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional, Union

from main.watermarking.provenance.attestation_statement import compute_event_binding_digest


# 四类子密钥的 context 标签（domain separation 核心，不可更改）。
_CONTEXT_LF = b"lf"
_CONTEXT_HF = b"hf"
_CONTEXT_GEO = b"geo"
_CONTEXT_TRACE = b"trace"

# HKDF 输出长度（字节）。
_HKDF_OUTPUT_LEN = 32


@dataclass(frozen=True)
class AttestationKeys:
    """
    功能：四类派生子密钥的不可变容器。

    Immutable container for per-attestation derived cryptographic keys.
    All keys are lowercase hex strings (64 chars = 32 bytes).

    Args:
        k_lf: Key for LF channel attestation payload generation.
        k_hf: Key for HF channel attestation and truncation binding.
        k_geo: Key for geometry chain anchor template derivation.
        k_tr: Key for generation trajectory commit computation.
        attestation_digest: The d_A value used to derive these keys.
        event_binding_digest: The joint event digest used for LF/HF/GEO derivation.
    """
    k_lf: str
    k_hf: str
    k_geo: str
    k_tr: str
    attestation_digest: str
    event_binding_digest: str


def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """
    功能：HKDF-Extract 阶段（RFC 5869 Section 2.2）。

    HKDF-Extract: Compress input keying material into a pseudorandom key.

    Args:
        salt: Salt bytes (K_master encoded as bytes).
        ikm: Input keying material (d_A encoded as bytes).

    Returns:
        32-byte pseudorandom key (PRK).
    """
    # PRK = HMAC-SHA256(salt, IKM)
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """
    功能：HKDF-Expand 阶段（RFC 5869 Section 2.3）。

    HKDF-Expand: Derive output keying material from PRK using domain info.

    Args:
        prk: Pseudorandom key from extract phase (32 bytes).
        info: Context/domain label bytes (enables domain separation).
        length: Desired output length in bytes (must be <= 255 * 32).

    Returns:
        Derived key material of requested length.

    Raises:
        ValueError: If length exceeds HKDF maximum output.
    """
    hash_len = 32  # SHA256 输出长度。
    max_output = 255 * hash_len
    if length > max_output:
        raise ValueError(f"HKDF cannot produce more than {max_output} bytes, requested {length}")

    # T(0) = empty；T(i) = HMAC(PRK, T(i-1) || info || i)。
    output = b""
    t_prev = b""
    counter = 1
    while len(output) < length:
        t_i = hmac.new(prk, t_prev + info + bytes([counter]), hashlib.sha256).digest()
        output += t_i
        t_prev = t_i
        counter += 1

    return output[:length]


def _to_key_bytes(key: Union[str, bytes]) -> bytes:
    """
    功能：将 str 或 bytes 主密钥转换为 bytes。

    Convert a key that is either a hex string or raw bytes to bytes.

    Args:
        key: Key material as hex string or bytes.

    Returns:
        Raw bytes key material.

    Raises:
        TypeError: If key is of unsupported type.
        ValueError: If key is empty.
    """
    if isinstance(key, bytes):
        if not key:
            raise ValueError("key must be non-empty bytes")
        return key
    if isinstance(key, str):
        if not key:
            raise ValueError("key must be non-empty str")
        try:
            return bytes.fromhex(key)
        except ValueError:
            # 若不是合法 hex，直接 UTF-8 编码（允许调用方传入任意字符串主密钥）。
            return key.encode("utf-8")
    raise TypeError(f"key must be str or bytes, got {type(key).__name__}")


def derive_attestation_keys(
    k_master: Union[str, bytes],
    attestation_digest: str,
    *,
    trajectory_commit: Optional[str] = None,
    event_binding_digest: Optional[str] = None,
) -> AttestationKeys:
    """
    功能：从 K_master 和 d_A 派生四类 attestation 子密钥。

    Derive four domain-separated cryptographic keys from K_master, statement
    digest, and event binding digest. The trace key k_TR is derived from the
    statement digest so that trajectory_commit can be computed first. The LF/HF/GEO
    event keys are derived from the joint event binding digest
    compute_event_binding_digest(statement_digest, trajectory_commit).

    Key derivation:
        k_TR   = HKDF(K_master, d_A, context="trace")
        d_E    = compute_event_binding_digest(d_A, trajectory_commit)
        k_LF   = HKDF(K_master, d_E, context="lf")
        k_HF   = HKDF(K_master, d_E, context="hf")
        k_GEO  = HKDF(K_master, d_E, context="geo")

    Args:
        k_master: Master key as hex string or raw bytes.
        attestation_digest: d_A (lowercase hex SHA256 string, 64 chars).
        trajectory_commit: Optional trajectory commit used to bind event keys.
        event_binding_digest: Optional precomputed event binding digest.

    Returns:
        AttestationKeys instance with all four derived keys as hex strings.

    Raises:
        TypeError: If inputs are of invalid type.
        ValueError: If inputs are empty or attestation_digest is not a valid hex string.
    """
    if not isinstance(attestation_digest, str) or not attestation_digest:
        raise ValueError("attestation_digest must be non-empty str")
    # 验证 d_A 是合法 hex（不要求长度固定，支持不同摘要算法输出）。
    try:
        bytes.fromhex(attestation_digest)
    except ValueError:
        raise ValueError(
            f"attestation_digest must be a valid hex string, got: {attestation_digest[:16]}..."
        )
    if event_binding_digest is not None:
        if not isinstance(event_binding_digest, str) or not event_binding_digest:
            raise ValueError("event_binding_digest must be non-empty str when provided")
        try:
            bytes.fromhex(event_binding_digest)
        except ValueError:
            raise ValueError("event_binding_digest must be a valid hex string")
    elif trajectory_commit is not None and not isinstance(trajectory_commit, str):
        raise TypeError("trajectory_commit must be str or None")

    salt = _to_key_bytes(k_master)
    trace_ikm = attestation_digest.encode("utf-8")
    resolved_event_binding_digest = (
        event_binding_digest
        if isinstance(event_binding_digest, str) and event_binding_digest
        else compute_event_binding_digest(attestation_digest, trajectory_commit)
    )
    event_ikm = resolved_event_binding_digest.encode("utf-8")

    # HKDF Extract：trace key 绑定 statement digest，LF/HF/GEO 绑定 event digest。
    trace_prk = _hkdf_extract(salt, trace_ikm)
    event_prk = _hkdf_extract(salt, event_ikm)

    # HKDF Expand：domain separation 通过不同 context 标签实现。
    raw_lf = _hkdf_expand(event_prk, _CONTEXT_LF, _HKDF_OUTPUT_LEN)
    raw_hf = _hkdf_expand(event_prk, _CONTEXT_HF, _HKDF_OUTPUT_LEN)
    raw_geo = _hkdf_expand(event_prk, _CONTEXT_GEO, _HKDF_OUTPUT_LEN)
    raw_tr = _hkdf_expand(trace_prk, _CONTEXT_TRACE, _HKDF_OUTPUT_LEN)

    return AttestationKeys(
        k_lf=raw_lf.hex(),
        k_hf=raw_hf.hex(),
        k_geo=raw_geo.hex(),
        k_tr=raw_tr.hex(),
        attestation_digest=attestation_digest,
        event_binding_digest=resolved_event_binding_digest,
    )


def compute_lf_attestation_payload(
    k_lf: Union[str, bytes],
    attestation_digest: str,
    payload_length: int = 48,
) -> bytes:
    """
    功能：生成 LF 主通道 attestation bits。

    Generate attestation payload bits for embedding in the LF channel.
    Uses HMAC(k_LF, d_A) truncated to the requested length.

    Payload formula:
        bits = HMAC-SHA256(k_LF, d_A)
        payload = truncate(bits, payload_length)
        (If payload_length > 32, HKDF-expand is used for extension.)

    Args:
        k_lf: LF channel derived key as hex str or bytes.
        attestation_digest: d_A hex string.
        payload_length: Desired payload length in bytes (default 48).

    Returns:
        Payload bytes of requested length.

    Raises:
        TypeError: If inputs are of invalid type.
        ValueError: If payload_length <= 0.
    """
    if not isinstance(attestation_digest, str) or not attestation_digest:
        raise ValueError("attestation_digest must be non-empty str")
    if not isinstance(payload_length, int) or payload_length <= 0:
        raise ValueError("payload_length must be positive int")

    key_bytes = _to_key_bytes(k_lf)
    message = attestation_digest.encode("utf-8")

    if payload_length <= 32:
        mac = hmac.new(key_bytes, message, hashlib.sha256).digest()
        return mac[:payload_length]

    # payload_length > 32：使用 HKDF-Expand 延伸。
    prk = hmac.new(key_bytes, message, hashlib.sha256).digest()
    return _hkdf_expand(prk, b"lf_payload", payload_length)


def derive_geo_anchor_seed(
    k_geo: Union[str, bytes],
) -> int:
    """
    功能：从 k_GEO 派生几何链 anchor 的确定性种子。

    Derive a deterministic integer seed for geometry chain anchor selection.
    Controls anchor ordering and sync template generation.

    Args:
        k_geo: GEO channel derived key as hex str or bytes.

    Returns:
        Non-negative integer seed (derived from first 4 bytes of k_GEO bytes).

    Raises:
        TypeError: If k_geo is of invalid type.
        ValueError: If k_geo is empty.
    """
    key_bytes = _to_key_bytes(k_geo)
    # 取前 4 字节转换为无符号整数作为 seed。
    seed_bytes = _hkdf_expand(key_bytes, b"geo_seed", 4)
    return int.from_bytes(seed_bytes, byteorder="little", signed=False)
