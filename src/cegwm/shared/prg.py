"""Domain-separated deterministic key streams for Stage-A carriers."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from cegwm.shared.keys import normalize_detection_key

_PRG_DOMAIN = b"CEG-WM/stage-a/prg/v1\x00"


def _normalize_domain(domain: str) -> bytes:
    if not isinstance(domain, str):
        raise TypeError("PRG domain must be text")
    encoded = domain.encode("utf-8")
    if not encoded or len(encoded) > 1024:
        raise ValueError("PRG domain must contain between 1 and 1024 UTF-8 bytes")
    return encoded


def prg_bytes(
    key: str | bytes | bytearray | memoryview,
    domain: str,
    length: int,
) -> bytes:
    """Expand a key into deterministic bytes under an explicit domain."""

    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("PRG length must be a non-negative integer")
    if length > 64 * 1024 * 1024:
        raise ValueError("one PRG request cannot exceed 64 MiB")
    key_bytes = normalize_detection_key(key)
    domain_bytes = _normalize_domain(domain)
    prefix = _PRG_DOMAIN + len(domain_bytes).to_bytes(4, "big") + domain_bytes
    blocks: list[bytes] = []
    for counter in range((length + 31) // 32):
        blocks.append(
            hmac.new(key_bytes, prefix + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        )
    return b"".join(blocks)[:length]


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(shape)
    if not normalized or len(normalized) > 8:
        raise ValueError("PRG shape must have between 1 and 8 dimensions")
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in normalized):
        raise ValueError("every PRG dimension must be a positive integer")
    if math.prod(normalized) > 16_777_216:
        raise ValueError("one PRG array cannot exceed 16,777,216 elements")
    return normalized


def _array_domain(kind: str, domain: str, shape: tuple[int, ...]) -> str:
    frame = json.dumps(
        {"kind": kind, "domain": domain, "shape": shape},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"array/{frame}"


def prg_rademacher(
    key: str | bytes | bytearray | memoryview,
    domain: str,
    shape: Sequence[int],
    *,
    dtype: np.dtype[object] | type[np.float32] | type[np.float64] = np.float32,
) -> NDArray[np.floating]:
    """Return a deterministic array whose values are exactly -1 or +1."""

    shape_tuple = _shape_tuple(shape)
    output_dtype = np.dtype(dtype)
    if output_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Rademacher output dtype must be float32 or float64")
    count = math.prod(shape_tuple)
    raw = np.frombuffer(
        prg_bytes(key, _array_domain("rademacher", domain, shape_tuple), (count + 7) // 8),
        dtype=np.uint8,
    )
    bits = np.unpackbits(raw, bitorder="little")[:count]
    values = np.where(bits == 0, -1.0, 1.0)
    return values.astype(output_dtype, copy=False).reshape(shape_tuple)


def prg_normal(
    key: str | bytes | bytearray | memoryview,
    domain: str,
    shape: Sequence[int],
    *,
    dtype: np.dtype[object] | type[np.float32] | type[np.float64] = np.float32,
) -> NDArray[np.floating]:
    """Return deterministic approximately standard-normal samples.

    Uniforms come directly from the HMAC stream and Box-Muller is applied in
    float64 before the explicit output cast. No process-global RNG is used.
    """

    shape_tuple = _shape_tuple(shape)
    output_dtype = np.dtype(dtype)
    if output_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("normal output dtype must be float32 or float64")
    count = math.prod(shape_tuple)
    pair_count = (count + 1) // 2
    raw = prg_bytes(
        key,
        _array_domain("normal-box-muller", domain, shape_tuple),
        pair_count * 16,
    )
    integers = np.frombuffer(raw, dtype=">u8").astype(np.float64)
    uniforms = (integers + 0.5) / float(2**64)
    u1 = uniforms[0::2]
    u2 = uniforms[1::2]
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    values = np.empty(pair_count * 2, dtype=np.float64)
    values[0::2] = radius * np.cos(angle)
    values[1::2] = radius * np.sin(angle)
    return values[:count].astype(output_dtype).reshape(shape_tuple)
