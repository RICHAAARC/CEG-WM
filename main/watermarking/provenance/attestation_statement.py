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
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Union


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

ATTESTATION_BUNDLE_SCHEMA = "gen_attest_bundle_v1"
ATTESTATION_SIGNER_CERT_SCHEMA = "gen_attest_signer_cert_v1"


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


def _load_ed25519() -> Any:
    """功能：延迟加载 Ed25519 实现。"""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

    return {
        "Ed25519PrivateKey": Ed25519PrivateKey,
        "Ed25519PublicKey": Ed25519PublicKey,
    }


def _to_key_bytes(key: Union[str, bytes]) -> bytes:
    """功能：将密钥材料统一转换为 bytes。"""
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
            return key.encode("utf-8")
    raise TypeError(f"key must be str or bytes, got {type(key).__name__}")


def _derive_attestation_private_key(k_master: Union[str, bytes], label: str) -> Any:
    """功能：从主密钥确定性派生内部签名私钥。"""
    if not isinstance(label, str) or not label:
        raise ValueError("label must be non-empty str")
    ed25519_mod = _load_ed25519()
    seed = hashlib.sha256(label.encode("utf-8") + b":" + _to_key_bytes(k_master)).digest()
    return ed25519_mod["Ed25519PrivateKey"].from_private_bytes(seed)


def _build_signer_certificate_payload(signer_public_key_hex: str) -> Dict[str, Any]:
    """功能：构造内部 signer 证书负载。"""
    return {
        "schema": ATTESTATION_SIGNER_CERT_SCHEMA,
        "algorithm": "ed25519",
        "purpose": "generation_attestation_bundle",
        "signer_public_key_hex": signer_public_key_hex,
    }


def _build_bundle_signature_payload(
    statement_dict: Dict[str, Any],
    attestation_digest: str,
    *,
    lf_payload_hex: Optional[str],
    trace_commit: Optional[str],
    geo_anchor_seed: Optional[int],
) -> Dict[str, Any]:
    """功能：构造 attestation bundle 的签名域。"""
    return {
        "schema": ATTESTATION_BUNDLE_SCHEMA,
        "statement": statement_dict,
        "attestation_digest": attestation_digest,
        "lf_payload_hex": lf_payload_hex,
        "trace_commit": trace_commit,
        "geo_anchor_seed": geo_anchor_seed,
    }


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


def build_signed_attestation_bundle(
    statement: AttestationStatement,
    attestation_digest: str,
    k_master: Union[str, bytes],
    *,
    lf_payload_hex: Optional[str] = None,
    trace_commit: Optional[str] = None,
    geo_anchor_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    功能：构造带内部信任链的 signed attestation bundle。

    Build a signed attestation bundle with an internal trust chain.

    Args:
        statement: Attestation statement instance.
        attestation_digest: Canonical statement digest.
        k_master: Master key material used to derive root and signer keys.
        lf_payload_hex: Optional LF payload hex string.
        trace_commit: Optional trajectory commit hex string.
        geo_anchor_seed: Optional geometry anchor seed.

    Returns:
        Signed attestation bundle mapping.
    """
    if not isinstance(statement, AttestationStatement):
        raise TypeError("statement must be AttestationStatement")
    if not isinstance(attestation_digest, str) or not attestation_digest:
        raise ValueError("attestation_digest must be non-empty str")

    root_private_key = _derive_attestation_private_key(k_master, "attestation_root")
    signer_private_key = _derive_attestation_private_key(k_master, "attestation_signer")
    root_public_key_hex = root_private_key.public_key().public_bytes_raw().hex()
    signer_public_key_hex = signer_private_key.public_key().public_bytes_raw().hex()

    signer_certificate_payload = _build_signer_certificate_payload(signer_public_key_hex)
    signer_certificate_signature_hex = root_private_key.sign(
        _canonical_json_dumps(signer_certificate_payload)
    ).hex()

    statement_dict = statement.as_dict()
    signature_payload = _build_bundle_signature_payload(
        statement_dict,
        attestation_digest,
        lf_payload_hex=lf_payload_hex,
        trace_commit=trace_commit,
        geo_anchor_seed=geo_anchor_seed,
    )
    signature_payload_digest = hashlib.sha256(_canonical_json_dumps(signature_payload)).hexdigest()
    bundle_signature_hex = signer_private_key.sign(_canonical_json_dumps(signature_payload)).hex()

    return {
        "schema": ATTESTATION_BUNDLE_SCHEMA,
        "statement": statement_dict,
        "attestation_digest": attestation_digest,
        "lf_payload_hex": lf_payload_hex,
        "trace_commit": trace_commit,
        "geo_anchor_seed": geo_anchor_seed,
        "signature": {
            "algorithm": "ed25519",
            "signature_hex": bundle_signature_hex,
            "signed_fields_digest": signature_payload_digest,
        },
        "trust_chain": {
            "root_public_key_hex": root_public_key_hex,
            "signer_certificate": {
                "payload": signer_certificate_payload,
                "signature_hex": signer_certificate_signature_hex,
            },
        },
    }


def verify_signed_attestation_bundle(bundle: Dict[str, Any], k_master: Union[str, bytes]) -> Dict[str, Any]:
    """
    功能：验证 signed attestation bundle 的签名与内部信任链。

    Verify a signed attestation bundle including the internal trust chain.

    Args:
        bundle: Signed attestation bundle mapping.
        k_master: Master key material used to derive the expected trust root.

    Returns:
        Verification result mapping.
    """
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be dict")

    ed25519_mod = _load_ed25519()
    mismatch_reasons = []
    if bundle.get("schema") != ATTESTATION_BUNDLE_SCHEMA:
        mismatch_reasons.append("bundle_schema_invalid")
        return {"status": "mismatch", "mismatch_reasons": mismatch_reasons}

    statement_dict = bundle.get("statement")
    if not isinstance(statement_dict, dict) or not verify_statement_fields(statement_dict):
        mismatch_reasons.append("bundle_statement_invalid")
        return {"status": "mismatch", "mismatch_reasons": mismatch_reasons}

    statement = statement_from_dict(statement_dict)
    recomputed_digest = compute_attestation_digest(statement)
    bundle_digest = bundle.get("attestation_digest")
    if not isinstance(bundle_digest, str) or bundle_digest != recomputed_digest:
        mismatch_reasons.append("bundle_digest_mismatch")

    trust_chain = bundle.get("trust_chain") if isinstance(bundle.get("trust_chain"), dict) else {}
    signer_certificate = trust_chain.get("signer_certificate") if isinstance(trust_chain.get("signer_certificate"), dict) else {}
    cert_payload = signer_certificate.get("payload") if isinstance(signer_certificate.get("payload"), dict) else {}
    cert_signature_hex = signer_certificate.get("signature_hex")
    root_public_key_hex = trust_chain.get("root_public_key_hex")
    if not isinstance(root_public_key_hex, str) or not root_public_key_hex:
        mismatch_reasons.append("bundle_root_public_key_absent")
        return {"status": "mismatch", "mismatch_reasons": mismatch_reasons}

    expected_root_private_key = _derive_attestation_private_key(k_master, "attestation_root")
    expected_root_public_key_hex = expected_root_private_key.public_key().public_bytes_raw().hex()
    if root_public_key_hex != expected_root_public_key_hex:
        mismatch_reasons.append("bundle_root_public_key_mismatch")
        return {"status": "mismatch", "mismatch_reasons": mismatch_reasons}

    if not isinstance(cert_signature_hex, str) or not cert_signature_hex:
        mismatch_reasons.append("bundle_signer_certificate_signature_absent")
        return {"status": "mismatch", "mismatch_reasons": mismatch_reasons}
    try:
        expected_root_private_key.public_key().verify(
            bytes.fromhex(cert_signature_hex),
            _canonical_json_dumps(cert_payload),
        )
    except Exception:
        mismatch_reasons.append("bundle_signer_certificate_signature_invalid")
        return {"status": "mismatch", "mismatch_reasons": mismatch_reasons}

    signer_public_key_hex = cert_payload.get("signer_public_key_hex") if isinstance(cert_payload, dict) else None
    if not isinstance(signer_public_key_hex, str) or not signer_public_key_hex:
        mismatch_reasons.append("bundle_signer_public_key_absent")
        return {"status": "mismatch", "mismatch_reasons": mismatch_reasons}

    signature_node = bundle.get("signature") if isinstance(bundle.get("signature"), dict) else {}
    bundle_signature_hex = signature_node.get("signature_hex")
    if not isinstance(bundle_signature_hex, str) or not bundle_signature_hex:
        mismatch_reasons.append("bundle_signature_absent")
        return {"status": "mismatch", "mismatch_reasons": mismatch_reasons}

    signature_payload = _build_bundle_signature_payload(
        statement_dict,
        recomputed_digest,
        lf_payload_hex=bundle.get("lf_payload_hex") if isinstance(bundle.get("lf_payload_hex"), str) else None,
        trace_commit=bundle.get("trace_commit") if isinstance(bundle.get("trace_commit"), str) else None,
        geo_anchor_seed=int(bundle.get("geo_anchor_seed")) if isinstance(bundle.get("geo_anchor_seed"), int) else None,
    )
    expected_signed_fields_digest = hashlib.sha256(_canonical_json_dumps(signature_payload)).hexdigest()
    signed_fields_digest = signature_node.get("signed_fields_digest")
    if not isinstance(signed_fields_digest, str) or signed_fields_digest != expected_signed_fields_digest:
        mismatch_reasons.append("bundle_signed_fields_digest_mismatch")

    signer_public_key = ed25519_mod["Ed25519PublicKey"].from_public_bytes(bytes.fromhex(signer_public_key_hex))
    try:
        signer_public_key.verify(bytes.fromhex(bundle_signature_hex), _canonical_json_dumps(signature_payload))
    except Exception:
        mismatch_reasons.append("bundle_signature_invalid")

    if mismatch_reasons:
        return {
            "status": "mismatch",
            "attestation_digest": recomputed_digest,
            "mismatch_reasons": mismatch_reasons,
        }

    return {
        "status": "ok",
        "attestation_digest": recomputed_digest,
        "mismatch_reasons": [],
        "signer_public_key_hex": signer_public_key_hex,
        "root_public_key_hex": root_public_key_hex,
    }
