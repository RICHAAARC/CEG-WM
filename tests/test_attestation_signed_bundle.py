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
    compute_event_binding_digest,
    verify_signed_attestation_bundle,
)
from main.watermarking.detect.orchestrator import _build_detect_attestation_result, verify_attestation


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


def test_compute_event_binding_digest_changes_with_trace_commit() -> None:
    """
    功能：事件绑定摘要必须联合绑定 trajectory commit。

    Event binding digest must change when trajectory_commit changes.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()

    digest_without_trace = compute_event_binding_digest(payload["attestation_digest"])
    digest_with_trace = compute_event_binding_digest(payload["attestation_digest"], "cc" * 32)

    assert digest_without_trace != digest_with_trace


def test_verify_attestation_statement_only_cannot_become_event_attested() -> None:
    """
    功能：仅有 statement 而缺少真实性证明时，最终 event-attested 必须为 false。

    Statement-only mode must not produce a final event-attested decision.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        content_evidence={"lf_score": 1.0},
        geo_score=1.0,
    )

    assert result.get("verdict") == "absent"
    assert "bundle_authenticity_absent" in list(result.get("mismatch_reasons") or [])
    authenticity_result = result.get("authenticity_result")
    assert isinstance(authenticity_result, dict)
    assert authenticity_result.get("status") == "statement_only"
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("is_event_attested") is False


def test_verify_attestation_authentic_bundle_can_become_event_attested() -> None:
    """
    功能：真实性通过且图像证据足够时，最终 event-attested 必须为 true。

    Authentic bundles with sufficient image evidence must become event-attested.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={"lf_score": 1.0},
        geo_score=1.0,
    )

    assert result.get("verdict") == "attested"
    assert result.get("event_binding_digest")
    authenticity_result = result.get("authenticity_result")
    assert isinstance(authenticity_result, dict)
    assert authenticity_result.get("status") == "authentic"
    image_evidence_result = result.get("image_evidence_result")
    assert isinstance(image_evidence_result, dict)
    assert image_evidence_result.get("status") == "ok"
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("status") == "attested"
    assert final_decision.get("is_event_attested") is True


def test_build_detect_attestation_result_bridges_hf_attestation_values() -> None:
    """
    功能：验证 detect 侧 attestation 会消费 canonical HF attestation 输入。

    Verify detect-side attestation consumes bridged canonical HF attestation values.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()
    cfg = {
        "__attestation_verify_k_master__": "5" * 64,
        "attestation": {
            "decision_mode": "content_primary_geo_rescue",
            "lf_weight": 0.5,
            "hf_weight": 0.3,
            "geo_weight": 0.2,
            "threshold": 0.65,
        },
    }
    attestation_context = {
        "candidate_statement": payload["statement"],
        "attestation_bundle": payload["bundle"],
        "authenticity_status": "authentic",
        "bundle_verification": {"status": "ok"},
    }
    content_evidence_payload = {
        "score_parts": {
            "hf_attestation_values": [1.0, 0.5, 0.25, 0.125],
        },
        "hf_evidence_summary": {
            "hf_status": "ok",
        },
    }

    result = _build_detect_attestation_result(
        cfg=cfg,
        attestation_context=attestation_context,
        content_evidence_payload=content_evidence_payload,
        geometry_evidence_payload=None,
    )

    image_evidence_result = result.get("image_evidence_result")
    assert isinstance(image_evidence_result, dict)
    assert image_evidence_result.get("status") == "ok"
    assert image_evidence_result.get("decision_mode") == "content_primary_geo_rescue"
    assert image_evidence_result.get("content_attestation_score") is not None
    channel_scores = result.get("channel_scores")
    assert isinstance(channel_scores, dict)
    assert channel_scores.get("hf") is not None
    assert "all_channel_scores_absent" not in list(result.get("mismatch_reasons") or [])


def test_verify_attestation_content_positive_not_dragged_by_low_geometry() -> None:
    """
    功能：当内容侧事件分数已过阈值时，低几何分数不得反向拉低 attestation。

    Low geometry scores must not negate a content-positive attestation decision.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={"lf_score": 0.9},
        geo_score=0.01,
    )

    assert result.get("verdict") == "attested"
    assert result.get("content_attestation_score") == 0.9
    assert result.get("geo_rescue_applied") is False
    assert result.get("geo_not_used_reason") == "content_attestation_threshold_met"


def test_verify_attestation_geometry_can_rescue_borderline_content() -> None:
    """
    功能：当内容侧分数落入 rescue band 且几何满足门槛时，允许单向 rescue。

    Geometry may rescue a borderline content score when the rescue gate is satisfied.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={"lf_score": 0.62},
        geo_score=0.8,
        attested_threshold=0.65,
        geo_rescue_band_delta_low=0.05,
        geo_rescue_min_score=0.3,
    )

    assert result.get("verdict") == "attested"
    assert result.get("content_attestation_score") == 0.62
    assert result.get("geo_rescue_eligible") is True
    assert result.get("geo_rescue_applied") is True


def test_verify_attestation_geometry_cannot_rescue_below_gate() -> None:
    """
    功能：当内容侧落入 rescue band 但几何分数不足时，verdict 必须保持 mismatch。

    Borderline content scores must remain mismatch when geometry fails the rescue gate.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={"lf_score": 0.62},
        geo_score=0.2,
        attested_threshold=0.65,
        geo_rescue_band_delta_low=0.05,
        geo_rescue_min_score=0.3,
    )

    assert result.get("verdict") == "mismatch"
    assert result.get("geo_rescue_eligible") is True
    assert result.get("geo_rescue_applied") is False
    assert result.get("geo_not_used_reason") == "geometry_score_below_rescue_min"


def test_verify_attestation_geometry_cannot_replace_missing_content() -> None:
    """
    功能：当内容主证据缺失时，不允许仅凭几何得到 event-attested。

    Geometry alone must not produce an event-attested verdict when content evidence is absent.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence=None,
        geo_score=1.0,
    )

    assert result.get("verdict") == "absent"
    assert result.get("content_attestation_score") is None
    assert result.get("geo_rescue_applied") is False
    assert result.get("geo_not_used_reason") == "content_attestation_evidence_absent"
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("is_event_attested") is False