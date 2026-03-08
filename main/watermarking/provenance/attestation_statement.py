"""
File purpose: 生成事件语句（generation attestation statement）构造器与摘要计算。
Module type: Core innovation module

创新边界：
    - statement 不含任何版本字段，仅含生成事件信息。
    - 支持 canonical JSON 序列化（字段排序固定、UTF-8、无额外空格）。
    - 支持从 statement 计算 d_A = SHA256(CanonicalJSON(statement))。

依赖假设：
    - prompt_commit / seed_commit 由 commitments 模块提供。
    - plan_digest 由 SubspacePlanner 提供。
    - event_nonce 每次生成随机生成（uuid4）。
    - time_bucket 格式为 YYYY-MM-DD（精度保护：不存精确时间）。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional


# statement schema 标识符（固定字符串）。
ATTESTATION_SCHEMA = "gen_attest_v1"

# statement 中允许的字段集合（严格白名单，无版本字段）。
_ALLOWED_FIELDS = frozenset([
    "schema",
    "model_id",
    "prompt_commit",
    "seed_commit",
    "plan_digest",
    "event_nonce",
    "time_bucket",
])


@dataclass(frozen=True)
class AttestationStatement:
    """
    功能：不可变的生成事件语句数据类。

    Immutable generation attestation statement binding all event-specific fields.
    No version fields are allowed; the statement represents a specific generation event.

    Args:
        schema: Fixed identifier string ("gen_attest_v1").
        model_id: Identifier of the generative model used (e.g., "sd3.5").
        prompt_commit: HMAC commitment of the normalized prompt.
        seed_commit: HMAC commitment of the generation seed.
        plan_digest: SubspacePlanner plan digest binding the content plan.
        event_nonce: Per-generation random nonce (UUID4 hex string).
        time_bucket: Date bucket in YYYY-MM-DD format (avoids exact timestamp leakage).
    """
    schema: str
    model_id: str
    prompt_commit: str
    seed_commit: str
    plan_digest: str
    event_nonce: str
    time_bucket: str

    def as_dict(self) -> Dict[str, str]:
        """
        功能：转换为轻量字典。

        Convert statement to ordered dictionary for canonical serialization.

        Returns:
            Ordered dictionary with fixed field ordering (alphabetical by key).
        """
        return {
            "event_nonce": self.event_nonce,
            "model_id": self.model_id,
            "plan_digest": self.plan_digest,
            "prompt_commit": self.prompt_commit,
            "schema": self.schema,
            "seed_commit": self.seed_commit,
            "time_bucket": self.time_bucket,
        }


def _canonical_json_dumps(obj: Dict[str, Any]) -> bytes:
    """
    功能：固定参数 canonical JSON 序列化（字段排序、UTF-8、无额外空格）。

    Serialize a dictionary to canonical JSON bytes.
    Guarantees deterministic output: sorted keys, no extra whitespace, UTF-8 encoding.

    Args:
        obj: Dictionary with string keys and string/int/float/bool/None values.

    Returns:
        UTF-8 encoded canonical JSON bytes.

    Raises:
        TypeError: If obj contains non-JSON-serializable values.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def build_attestation_statement(
    model_id: str,
    prompt_commit: str,
    seed_commit: str,
    plan_digest: str,
    *,
    event_nonce: Optional[str] = None,
    time_bucket: Optional[str] = None,
) -> AttestationStatement:
    """
    功能：构造生成事件语句（不含版本字段）。

    Build a generation attestation statement from event-specific fields.
    No version fields are included; the schema field is a fixed structural identifier.

    Args:
        model_id: Generative model identifier (e.g., "sd3.5").
        prompt_commit: HMAC commitment of the normalized prompt (hex str).
        seed_commit: HMAC commitment of the generation seed (hex str).
        plan_digest: SubspacePlanner plan digest (hex str or canonical string).
        event_nonce: Per-generation random nonce; generated via uuid4() if None.
        time_bucket: Date bucket in YYYY-MM-DD; defaults to today's date if None.

    Returns:
        Immutable AttestationStatement instance.

    Raises:
        TypeError: If any required argument is not a string.
        ValueError: If any required argument is empty.
    """
    # 类型检查。
    for arg_name, arg_val in [
        ("model_id", model_id),
        ("prompt_commit", prompt_commit),
        ("seed_commit", seed_commit),
        ("plan_digest", plan_digest),
    ]:
        if not isinstance(arg_val, str):
            raise TypeError(f"{arg_name} must be str, got {type(arg_val).__name__}")
        if not arg_val:
            raise ValueError(f"{arg_name} must be non-empty")

    # 自动生成 event_nonce（每次生成事件唯一）。
    if event_nonce is None:
        event_nonce = uuid.uuid4().hex
    elif not isinstance(event_nonce, str) or not event_nonce:
        raise ValueError("event_nonce must be non-empty str when provided")

    # 自动填充 time_bucket（精度保护：只存日期，不存精确时间）。
    if time_bucket is None:
        time_bucket = date.today().isoformat()
    elif not isinstance(time_bucket, str) or not time_bucket:
        raise ValueError("time_bucket must be non-empty str when provided")

    return AttestationStatement(
        schema=ATTESTATION_SCHEMA,
        model_id=model_id,
        prompt_commit=prompt_commit,
        seed_commit=seed_commit,
        plan_digest=plan_digest,
        event_nonce=event_nonce,
        time_bucket=time_bucket,
    )


def compute_attestation_digest(statement: AttestationStatement) -> str:
    """
    功能：计算 d_A = SHA256(CanonicalJSON(statement))。

    Compute the attestation digest by SHA256 hashing the canonical JSON serialization
    of the statement. This digest binds all event fields for key derivation.

    Args:
        statement: Attestation statement instance.

    Returns:
        Lowercase hex SHA256 digest string (64 chars).

    Raises:
        TypeError: If statement is not an AttestationStatement.
    """
    if not isinstance(statement, AttestationStatement):
        raise TypeError(
            f"statement must be AttestationStatement, got {type(statement).__name__}"
        )
    canonical_bytes = _canonical_json_dumps(statement.as_dict())
    return hashlib.sha256(canonical_bytes).hexdigest()


def verify_statement_fields(statement_dict: Dict[str, Any]) -> bool:
    """
    功能：验证 statement 字典字段是否合法（不含版本字段、字段完整）。

    Verify that a statement dictionary contains all required fields and no
    disallowed fields (such as version fields).

    Args:
        statement_dict: Raw dictionary representing an attestation statement.

    Returns:
        True when all required fields are present and no disallowed fields exist.

    Raises:
        None.
    """
    if not isinstance(statement_dict, dict):
        return False
    present = set(statement_dict.keys())
    # 所有字段必须在白名单内，且必须包含所有必填字段。
    if not present.issubset(_ALLOWED_FIELDS):
        return False
    if not _ALLOWED_FIELDS.issubset(present):
        return False
    return True


def statement_from_dict(d: Dict[str, Any]) -> AttestationStatement:
    """
    功能：从字典重建 AttestationStatement（用于检测阶段重构 statement）。

    Reconstruct an AttestationStatement from a plain dictionary.
    Used during detection to reconstruct the candidate statement.

    Args:
        d: Dictionary with all required attestation fields.

    Returns:
        AttestationStatement instance.

    Raises:
        TypeError: If d is not a dict or field values are not str.
        ValueError: If required fields are missing or have disallowed extras.
    """
    if not isinstance(d, dict):
        raise TypeError(f"d must be dict, got {type(d).__name__}")
    if not verify_statement_fields(d):
        present = set(d.keys()) if isinstance(d, dict) else set()
        missing = _ALLOWED_FIELDS - present
        extra = present - _ALLOWED_FIELDS
        raise ValueError(
            f"statement dict invalid: missing={missing}, disallowed_extra={extra}"
        )
    return AttestationStatement(
        schema=str(d["schema"]),
        model_id=str(d["model_id"]),
        prompt_commit=str(d["prompt_commit"]),
        seed_commit=str(d["seed_commit"]),
        plan_digest=str(d["plan_digest"]),
        event_nonce=str(d["event_nonce"]),
        time_bucket=str(d["time_bucket"]),
    )
