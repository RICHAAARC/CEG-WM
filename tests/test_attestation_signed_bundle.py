"""
File purpose: 验证 signed attestation bundle 的构造、验签与篡改失败语义。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

from main.watermarking.provenance.attestation_statement import (
    build_attestation_statement,
    build_signed_attestation_bundle,
    compute_attestation_digest,
    verify_signed_attestation_bundle,
)
from main.watermarking.detect.orchestrator import verify_attestation


def _build_statement() -> Dict[str, Any]:
    statement = build_attestation_statement(
        model_id="stabilityai/stable-diffusion-3-medium",
        prompt_commit="1" * 64,
        seed_commit="2" * 64,
        plan_digest="3" * 64,
        event_nonce="4" * 32,
        time_bucket="2026-03-13",
    )
    digest = compute_attestation_digest(statement)
    bundle = build_signed_attestation_bundle(
        statement,
        digest,
        k_master="5" * 64,
        lf_payload_hex="aa" * 16,
        trace_commit="bb" * 32,
        geo_anchor_seed=17,
    )
    return {
        "statement": statement.as_dict(),
        "attestation_digest": digest,
        "bundle": bundle,
    }


def test_signed_attestation_bundle_roundtrip() -> None:
    """
    功能：signed bundle 在未篡改时必须通过内部 trust chain 验证。

    Signed bundle must verify successfully when untampered.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()

    verification = verify_signed_attestation_bundle(payload["bundle"], "5" * 64)

    assert verification.get("status") == "ok"
    assert verification.get("attestation_digest") == payload["attestation_digest"]
    assert verification.get("mismatch_reasons") == []


def test_signed_attestation_bundle_tamper_fails() -> None:
    """
    功能：bundle 中 statement 被篡改时必须返回 mismatch。

    Tampering a bundle-bound statement must fail verification.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()
    tampered_bundle = dict(payload["bundle"])
    tampered_statement = dict(payload["statement"])
    tampered_statement["plan_digest"] = "6" * 64
    tampered_bundle["statement"] = tampered_statement

    verification = verify_signed_attestation_bundle(tampered_bundle, "5" * 64)

    assert verification.get("status") == "mismatch"
    assert "bundle_digest_mismatch" in list(verification.get("mismatch_reasons") or [])


def test_verify_attestation_rejects_invalid_signed_bundle() -> None:
    """
    功能：detect 侧 attestation 验证必须先拒绝无效 signed bundle。

    Detection attestation verification must reject invalid signed bundles before fusion.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()
    tampered_bundle = dict(payload["bundle"])
    signature_node = dict(tampered_bundle["signature"])
    signature_node["signature_hex"] = "00" * 64
    tampered_bundle["signature"] = signature_node

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=tampered_bundle,
        content_evidence={"lf_score": 1.0},
        geo_score=1.0,
    )

    assert result.get("verdict") == "mismatch"
    bundle_verification = result.get("bundle_verification")
    assert isinstance(bundle_verification, dict)
    assert bundle_verification.get("status") == "mismatch"
    assert "bundle_signature_invalid" in list(bundle_verification.get("mismatch_reasons") or [])