"""Detection-key normalization and non-secret public identity."""

from __future__ import annotations

import hashlib
import unicodedata

_DIGEST_DOMAIN = b"CEG-WM/public-detection-key/v1\x00"
_MIN_KEY_BYTES = 16
_MAX_KEY_BYTES = 4096


def normalize_detection_key(key: str | bytes | bytearray | memoryview) -> bytes:
    """Return the canonical bytes used by every keyed content primitive.

    Text keys are NFC-normalized UTF-8. Byte keys are preserved exactly. The
    function deliberately does not trim whitespace because doing so would merge
    distinct key identities.
    """

    if isinstance(key, str):
        normalized = unicodedata.normalize("NFC", key)
        if any(unicodedata.category(character) == "Cc" for character in normalized):
            raise ValueError("text detection keys cannot contain control characters")
        key_bytes = normalized.encode("utf-8")
    elif isinstance(key, (bytes, bytearray, memoryview)):
        key_bytes = bytes(key)
    else:
        raise TypeError("detection key must be text or bytes-like")

    if len(key_bytes) < _MIN_KEY_BYTES:
        raise ValueError(f"detection key must contain at least {_MIN_KEY_BYTES} bytes")
    if len(key_bytes) > _MAX_KEY_BYTES:
        raise ValueError(f"detection key cannot exceed {_MAX_KEY_BYTES} bytes")
    return key_bytes


def public_key_digest(key: str | bytes | bytearray | memoryview) -> str:
    """Return a domain-separated digest suitable for records and comparisons."""

    key_bytes = normalize_detection_key(key)
    framed = len(key_bytes).to_bytes(8, "big") + key_bytes
    return hashlib.sha256(_DIGEST_DOMAIN + framed).hexdigest()
