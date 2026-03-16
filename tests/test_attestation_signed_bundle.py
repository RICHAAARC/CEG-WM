"""
File purpose: 验证 signed attestation bundle 的构造、验签与篡改失败语义。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, cast

import numpy as np
import pytest

from main.watermarking.provenance.attestation_statement import (
    build_attestation_statement,
    build_signed_attestation_bundle,
    compute_attestation_digest,
    compute_event_binding_digest,
    verify_signed_attestation_bundle,
)
from main.watermarking.provenance.key_derivation import derive_attestation_keys
from main.watermarking.content_chain import channel_hf
from main.watermarking.content_chain.high_freq_embedder import compute_hf_attestation_score
from main.watermarking.content_chain import high_freq_embedder as high_freq_embedder_module
from main.watermarking.detect import orchestrator as detect_orchestrator
from main.cli import run_detect as run_detect_cli

_build_detect_attestation_result = detect_orchestrator._build_detect_attestation_result  # pyright: ignore[reportPrivateUsage]
_build_hf_detect_evidence = detect_orchestrator._build_hf_detect_evidence  # pyright: ignore[reportPrivateUsage]
verify_attestation = detect_orchestrator.verify_attestation
_write_detect_attestation_artifact = run_detect_cli._write_detect_attestation_artifact  # pyright: ignore[reportPrivateUsage]


def _build_hf_runtime_cfg(tail_truncation_ratio: float = 0.1) -> Dict[str, Any]:
    return {
        "watermark": {
            "hf": {
                "enabled": True,
                "tail_truncation_ratio": tail_truncation_ratio,
                "tail_truncation_mode": "projection_tail_truncation",
            }
        }
    }


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
    assert final_decision.get("event_attestation_score") == pytest.approx(0.0)
    assert final_decision.get("event_attestation_score_name") == "event_attestation_score"


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
    assert final_decision.get("event_attestation_score") == pytest.approx(result.get("content_attestation_score"))
    assert final_decision.get("event_attestation_score_name") == "event_attestation_score"


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
    assert result.get("hf_attestation_score") is None
    assert result.get("hf_attestation_decision_score") is None
    assert result.get("geo_rescue_applied") is False
    assert result.get("geo_not_used_reason") == "content_attestation_threshold_met"
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("event_attestation_score") == pytest.approx(0.9)


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
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("event_attestation_score") == pytest.approx(0.62)


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
    assert final_decision.get("event_attestation_score") is None


def test_compute_hf_attestation_score_emits_trace_fields() -> None:
    """
    功能：验证 HF attestation 评分会输出最小可审计 trace 字段。

    Verify HF attestation scoring emits the required minimal trace fields.

    Args:
        None.

    Returns:
        None.
    """
    result = compute_hf_attestation_score(
        hf_values=[-0.4, -0.7, 1.1, -2.5, -0.6, -0.1, 0.2, 1.0],
        k_hf="7" * 64,
        attestation_event_digest="8" * 64,
        plan_digest="9" * 64,
        cfg=_build_hf_runtime_cfg(0.1),
    )

    trace = result.get("hf_attestation_trace")
    trace = cast(Dict[str, Any], result.get("hf_attestation_trace"))
    assert isinstance(trace, dict)
    retained_indices = cast(list[int], trace.get("hf_attestation_retained_indices"))
    assert trace.get("hf_attestation_challenge_digest")
    assert trace.get("hf_attestation_challenge_source") == "attestation_event_digest"
    assert isinstance(result.get("hf_attestation_decision_score"), float)
    assert trace.get("hf_attestation_decision_score") == result.get("hf_attestation_decision_score")
    assert trace.get("hf_attestation_plan_digest_used") == "9" * 64
    assert trace.get("hf_attestation_threshold_percentile_applied") == 90.0
    assert trace.get("hf_attestation_vector_length") == 8
    assert isinstance(retained_indices, list)
    assert trace.get("hf_attestation_retained_count") == len(retained_indices)
    assert trace.get("hf_attestation_weight_vector_digest")
    assert trace.get("hf_attestation_gating_mask_digest")
    assert trace.get("hf_attestation_ordering_digest")
    assert trace.get("hf_attestation_trace_digest") == result.get("hf_attestation_trace_digest")


def test_verify_attestation_records_hf_trace_consistency_match() -> None:
    """
    功能：验证 detect 与 attestation 使用同一 challenge 摘要时会显式记录一致状态。

    Verify trace consistency fields report ok when detect and attestation traces agree.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()
    hf_values = [0.5, -0.25, 1.5, -2.0, 0.25, 0.75, -0.5, 0.1]
    attest_keys = derive_attestation_keys(
        "5" * 64,
        payload["attestation_digest"],
        trajectory_commit=payload["bundle"].get("trace_commit"),
    )
    expected = compute_hf_attestation_score(
        hf_values=hf_values,
        k_hf=attest_keys.k_hf,
        attestation_event_digest=attest_keys.event_binding_digest,
        plan_digest=payload["statement"]["plan_digest"],
        cfg=_build_hf_runtime_cfg(0.1),
    )
    trace = expected.get("hf_attestation_trace")
    trace = cast(Dict[str, Any], expected.get("hf_attestation_trace"))
    assert isinstance(trace, dict)

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={
            "lf_score": 0.8,
            "hf_evidence_summary": {
                "challenge_digest": trace.get("hf_attestation_challenge_digest"),
                "challenge_seed": trace.get("hf_attestation_challenge_seed"),
                "challenge_source": trace.get("hf_attestation_challenge_source"),
                "threshold_percentile_applied": trace.get("hf_attestation_threshold_percentile_applied"),
                "coeffs_retained_count": trace.get("hf_attestation_retained_count"),
            },
        },
        cfg=_build_hf_runtime_cfg(0.1),
        hf_values=hf_values,
        detect_hf_plan_digest_used=payload["statement"]["plan_digest"],
    )

    trace_artifact = result.get("_hf_attestation_trace_artifact")
    trace_artifact = cast(Dict[str, Any], result.get("_hf_attestation_trace_artifact"))
    assert isinstance(trace_artifact, dict)
    assert trace_artifact.get("detect_hf_plan_digest_used") == payload["statement"]["plan_digest"]
    assert trace_artifact.get("hf_attestation_plan_digest_used") == payload["statement"]["plan_digest"]
    assert trace_artifact.get("hf_attestation_challenge_match_status") == "ok"
    assert trace_artifact.get("hf_attestation_threshold_match_status") == "ok"
    assert trace_artifact.get("hf_attestation_retained_count_match_status") == "ok"
    assert trace_artifact.get("hf_attestation_trace_consistency") == "ok"


def test_verify_attestation_records_hf_trace_consistency_mismatch() -> None:
    """
    功能：验证 detect 与 attestation 的 percentile 差异会被显式记录，而不是静默消失。

    Verify percentile/count differences are recorded explicitly when detect and attestation diverge.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()
    hf_values = np.asarray([0.5, -0.25, 1.5, -2.0, 0.25, 0.75, -0.5, 0.1], dtype=np.float32)
    attest_keys = derive_attestation_keys(
        "5" * 64,
        payload["attestation_digest"],
        trajectory_commit=payload["bundle"].get("trace_commit"),
    )
    detect_cfg: Dict[str, Any] = {
        "hf_threshold_percentile": 90.0,
        "hf_attestation_key": attest_keys.k_hf,
        "attestation_event_digest": attest_keys.event_binding_digest,
        "plan_digest": payload["statement"]["plan_digest"],
    }
    _constrained_coeffs, detect_summary = channel_hf.apply_hf_truncation_constraint(hf_values, detect_cfg)
    _ = _constrained_coeffs

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={
            "lf_score": 0.8,
            "hf_evidence_summary": detect_summary,
        },
        cfg=_build_hf_runtime_cfg(0.2),
        hf_values=hf_values,
        detect_hf_plan_digest_used=payload["statement"]["plan_digest"],
    )

    trace_artifact = result.get("_hf_attestation_trace_artifact")
    trace_artifact = cast(Dict[str, Any], result.get("_hf_attestation_trace_artifact"))
    assert isinstance(trace_artifact, dict)
    assert trace_artifact.get("detect_hf_plan_digest_used") == payload["statement"]["plan_digest"]
    assert trace_artifact.get("hf_attestation_plan_digest_used") == payload["statement"]["plan_digest"]
    assert trace_artifact.get("hf_attestation_challenge_match_status") == "ok"
    assert trace_artifact.get("hf_attestation_threshold_match_status") == "mismatch"
    assert trace_artifact.get("hf_attestation_trace_consistency") == "mismatch"
    assert result.get("verdict") in {"attested", "mismatch", "absent"}


def test_compute_hf_attestation_score_follows_canonical_tail_ratio_mapping() -> None:
    """
    功能：验证 attestation 侧 percentile 与 canonical tail_truncation_ratio 同步变化。 

    Verify attestation percentile follows the canonical tail_truncation_ratio mapping.

    Args:
        None.

    Returns:
        None.
    """
    base = compute_hf_attestation_score(
        hf_values=[-0.4, -0.7, 1.1, -2.5, -0.6, -0.1, 0.2, 1.0],
        k_hf="7" * 64,
        attestation_event_digest="8" * 64,
        plan_digest="9" * 64,
        cfg=_build_hf_runtime_cfg(0.1),
    )
    changed = compute_hf_attestation_score(
        hf_values=[-0.4, -0.7, 1.1, -2.5, -0.6, -0.1, 0.2, 1.0],
        k_hf="7" * 64,
        attestation_event_digest="8" * 64,
        plan_digest="9" * 64,
        cfg=_build_hf_runtime_cfg(0.35),
    )

    base_trace = cast(Dict[str, Any], base.get("hf_attestation_trace"))
    changed_trace = cast(Dict[str, Any], changed.get("hf_attestation_trace"))
    assert base_trace.get("hf_attestation_threshold_percentile_applied") == 90.0
    assert changed_trace.get("hf_attestation_threshold_percentile_applied") == 65.0


def test_verify_attestation_exposes_hf_raw_and_decision_scores() -> None:
    """
    功能：验证 attestation 输出同时暴露 HF raw 分数与判决分数。

    Verify attestation output exposes both HF raw and decision scores.

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
        content_evidence={"lf_score": 0.8},
        cfg=_build_hf_runtime_cfg(0.1),
        hf_values=[0.5, -0.25, 1.5, -2.0, 0.25, 0.75, -0.5, 0.1],
    )

    raw_hf_score = result.get("hf_attestation_score")
    decision_hf_score = result.get("hf_attestation_decision_score")
    image_evidence_result = cast(Dict[str, Any], result.get("image_evidence_result"))

    assert isinstance(raw_hf_score, float)
    assert isinstance(decision_hf_score, float)
    assert decision_hf_score >= raw_hf_score
    assert image_evidence_result.get("hf_attestation_score") == raw_hf_score
    assert image_evidence_result.get("hf_attestation_decision_score") == decision_hf_score


def test_verify_attestation_records_hf_trace_consistency_match_with_canonical_percentile() -> None:
    """
    功能：验证 detect 与 attestation 复用同一 canonical HF cfg 时 percentile 一致。 

    Verify detect and attestation share the same canonical HF percentile source.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()
    hf_values = np.asarray([0.5, -0.25, 1.5, -2.0, 0.25, 0.75, -0.5, 0.1], dtype=np.float32)
    attest_keys = derive_attestation_keys(
        "5" * 64,
        payload["attestation_digest"],
        trajectory_commit=payload["bundle"].get("trace_commit"),
    )
    shared_cfg = _build_hf_runtime_cfg(0.35)
    detect_cfg = high_freq_embedder_module._build_hf_channel_cfg(  # pyright: ignore[reportPrivateUsage]
        {
            **shared_cfg,
            "hf_attestation_key": attest_keys.k_hf,
            "attestation_event_digest": attest_keys.event_binding_digest,
            "plan_digest": payload["statement"]["plan_digest"],
        }
    )
    _constrained_coeffs, detect_summary = channel_hf.apply_hf_truncation_constraint(hf_values, detect_cfg)
    _ = _constrained_coeffs

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={
            "lf_score": 0.8,
            "hf_evidence_summary": detect_summary,
        },
        cfg=shared_cfg,
        hf_values=hf_values,
        detect_hf_plan_digest_used=payload["statement"]["plan_digest"],
    )

    trace_artifact = cast(Dict[str, Any], result.get("_hf_attestation_trace_artifact"))
    assert trace_artifact.get("detect_hf_threshold_percentile_applied") == 65.0
    assert trace_artifact.get("hf_attestation_threshold_percentile_applied") == 65.0
    assert trace_artifact.get("hf_attestation_threshold_match_status") == "ok"


def test_write_detect_attestation_artifact_persists_hf_trace(monkeypatch: Any, tmp_path: Path) -> None:
    """
    功能：验证 detect attestation artifact writer 会额外写出 HF attestation trace 工件。

    Verify detect attestation artifact writer emits the standalone HF trace artifact.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    written: Dict[str, Dict[str, Any]] = {}

    def _fake_write_artifact_json(path: str, payload: Dict[str, Any], indent: int = 2, ensure_ascii: bool = False) -> None:
        _ = indent
        _ = ensure_ascii
        written[path.replace("\\", "/")] = payload

    monkeypatch.setattr("main.cli.run_detect.records_io.write_artifact_json", _fake_write_artifact_json)

    record = {
        "attestation": {
            "verdict": "mismatch",
            "fusion_score": 0.53,
        }
    }
    attestation_artifacts = {
        "hf_attestation_trace": {
            "artifact_type": "hf_attestation_trace",
            "hf_attestation_challenge_digest": "a" * 64,
            "hf_attestation_threshold_percentile_applied": 75.0,
            "hf_attestation_retained_count": 2,
        }
    }

    _write_detect_attestation_artifact(record, tmp_path / "artifacts", attestation_artifacts)

    result_path = str((tmp_path / "artifacts" / "attestation" / "attestation_result.json")).replace("\\", "/")
    trace_path = str((tmp_path / "artifacts" / "attestation" / "hf_attestation_trace.json")).replace("\\", "/")
    assert written[result_path] == record["attestation"]
    assert written[trace_path]["artifact_type"] == "hf_attestation_trace"


def test_build_hf_detect_evidence_binds_canonical_plan_digest_locally() -> None:
    """
    功能：验证 detect 侧 HF challenge 仅在局部 runtime cfg 中绑定 canonical plan_digest。 

    Verify detect-side HF evidence binds the canonical plan digest only in the
    local HF runtime cfg.

    Args:
        None.

    Returns:
        None.
    """
    captured_cfg: Dict[str, Any] = {}

    class _TrajectoryCacheStub:
        def is_empty(self) -> bool:
            return False

        def get(self, step_index: int) -> Any:
            if step_index != 0:
                return None
            return np.asarray([1.0], dtype=np.float32)

        def available_steps(self) -> list[int]:
            return [0]

    class _HfEmbedderStub:
        def detect(
            self,
            *,
            latents_or_features: Any,
            plan: Dict[str, Any] | None,
            cfg: Dict[str, Any],
            cfg_digest: str | None,
            expected_plan_digest: str | None,
        ) -> tuple[float, Dict[str, Any]]:
            _ = latents_or_features
            _ = plan
            _ = cfg_digest
            _ = expected_plan_digest
            captured_cfg.update(cfg)
            return 0.5, {
                "status": "ok",
                "hf_score": 0.5,
                "hf_trace_digest": "a" * 64,
                "hf_evidence_summary": {
                    "hf_status": "ok",
                },
            }

    cfg: Dict[str, Any] = {
        "__detect_trajectory_latent_cache__": _TrajectoryCacheStub(),
        "watermark": {
            "hf": {"enabled": True},
            "plan_digest": "1" * 64,
        },
    }
    impl_set = cast(Any, SimpleNamespace(hf_embedder=_HfEmbedderStub()))

    evidence = _build_hf_detect_evidence(
        impl_set,
        cfg,
        cfg_digest=None,
        plan_payload={
            "hf_basis": {
                "trajectory_feature_spec": {"edit_timestep": 0},
            }
        },
        plan_digest="2" * 64,
        embed_time_plan_digest="3" * 64,
        trajectory_evidence=None,
    )

    assert evidence.get("status") == "ok"
    assert captured_cfg.get("plan_digest") == "3" * 64
    watermark_cfg = cast(Dict[str, Any], captured_cfg.get("watermark"))
    assert watermark_cfg.get("plan_digest") == "3" * 64
    assert cast(Dict[str, Any], cfg.get("watermark")).get("plan_digest") == "1" * 64
    assert cfg.get("plan_digest") is None