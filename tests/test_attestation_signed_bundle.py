"""
File purpose: 验证 signed attestation bundle 的构造、验签与篡改失败语义。
Module type: General module
"""

from __future__ import annotations

import hashlib
import hmac
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
    statement_from_dict,
    verify_signed_attestation_bundle,
)
from main.watermarking.provenance.key_derivation import derive_attestation_keys
from main.watermarking.content_chain import channel_hf
from main.watermarking.content_chain import channel_lf
from main.watermarking.content_chain.high_freq_embedder import compute_hf_attestation_score
from main.watermarking.content_chain.low_freq_coder import compute_lf_attestation_score
from main.watermarking.content_chain import high_freq_embedder as high_freq_embedder_module
from main.watermarking.detect import orchestrator as detect_orchestrator
from main.cli import run_detect as run_detect_cli
from main.cli import run_embed as run_embed_cli

_build_detect_attestation_result = detect_orchestrator._build_detect_attestation_result  # pyright: ignore[reportPrivateUsage]
_build_hf_detect_evidence = detect_orchestrator._build_hf_detect_evidence  # pyright: ignore[reportPrivateUsage]
_build_lf_planner_risk_report_artifact = detect_orchestrator._build_lf_planner_risk_report_artifact  # pyright: ignore[reportPrivateUsage]
verify_attestation = detect_orchestrator.verify_attestation
_write_detect_attestation_artifact = run_detect_cli._write_detect_attestation_artifact  # pyright: ignore[reportPrivateUsage]
_write_embed_planner_artifacts = run_embed_cli._write_embed_planner_artifacts  # pyright: ignore[reportPrivateUsage]


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


def _build_matching_lf_attestation_features(
    statement: Dict[str, Any],
    k_master: str,
    trace_commit: str | None = None,
    plan_digest: str | None = None,
    basis_digest: str | None = None,
    event_binding_mode: str = "trajectory_bound",
) -> list[float]:
    attestation_digest = compute_attestation_digest(statement_from_dict(statement))
    attest_keys = derive_attestation_keys(
        k_master,
        attestation_digest,
        trajectory_commit=trace_commit,
        event_binding_mode=event_binding_mode,
    )
    template_bundle = channel_lf.derive_lf_template_bundle(
        {
            "watermark": {
                "lf": {
                    "message_length": 64,
                    "ecc_sparsity": 3,
                }
            },
            "lf_attestation_event_digest": attest_keys.event_binding_digest,
            "lf_attestation_key": attest_keys.k_lf,
            "plan_digest": plan_digest or statement["plan_digest"],
            "lf_basis_digest": basis_digest,
            "basis_digest": basis_digest,
            "event_binding_mode": event_binding_mode,
        },
        48 * 8,
    )
    return [
        3.0 if int(value) > 0 else -3.0
        for value in template_bundle["codeword_bipolar"].tolist()
    ]


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


def test_derive_attestation_keys_statement_only_ignores_trajectory_commit() -> None:
    """
    功能：statement-only 模式下的事件摘要与 k_lf 不得依赖 trajectory_commit。
    """
    attestation_digest = _build_statement()["attestation_digest"]

    without_trace = derive_attestation_keys(
        "5" * 64,
        attestation_digest,
        event_binding_mode="statement_only",
    )
    with_trace = derive_attestation_keys(
        "5" * 64,
        attestation_digest,
        trajectory_commit="cc" * 32,
        event_binding_mode="statement_only",
    )

    assert without_trace.event_binding_mode == "statement_only"
    assert with_trace.event_binding_mode == "statement_only"
    assert without_trace.event_binding_digest == with_trace.event_binding_digest
    assert without_trace.k_lf == with_trace.k_lf


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
    assert authenticity_result.get("bundle_status") is None
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("is_event_attested") is False
    assert final_decision.get("event_attestation_score") == pytest.approx(0.0)
    assert final_decision.get("event_attestation_score_name") == "event_attestation_score"
    assert final_decision.get("event_attestation_statistics_score") == pytest.approx(0.0)
    assert final_decision.get("event_attestation_statistics_score") == pytest.approx(
        final_decision.get("event_attestation_score")
    )
    assert final_decision.get("event_attestation_statistics_score_name") == "event_attestation_statistics_score"
    assert final_decision.get("event_attestation_statistics_score_semantics") == (
        "legacy_alias_of_event_attestation_score_not_an_independent_statistics_semantics"
    )


def test_build_detect_attestation_result_statement_only_provenance_uses_legal_bundle_status() -> None:
    """
    功能：negative branch statement-only provenance 必须生成合法、非空且不可冒充 authentic 的 bundle_status。

    Negative-branch statement-only provenance must emit a legal non-empty
    bundle_status without claiming authentic bundle verification.

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
        "attestation_bundle": None,
        "authenticity_status": "statement_only",
        "attestation_source": "negative_branch_statement_only_provenance",
        "bundle_verification": None,
    }

    result = _build_detect_attestation_result(
        cfg=cfg,
        attestation_context=attestation_context,
        content_evidence_payload={"lf_score": 1.0},
        geometry_evidence_payload=None,
    )

    authenticity_result = result.get("authenticity_result")
    assert isinstance(authenticity_result, dict)
    assert authenticity_result.get("status") == "statement_only"
    assert authenticity_result.get("bundle_status") == "statement_only_provenance_no_bundle"
    assert result.get("attestation_source") == "negative_branch_statement_only_provenance"
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("is_event_attested") is False


def test_build_detect_attestation_result_missing_statement_uses_absent_bundle_status() -> None:
    """
    功能：detect 侧 statement 缺失时必须写出 absent bundle_status，避免 formal gate 因 None 崩溃。

    Detect attestation results with missing statements must emit an explicit
    absent bundle_status so formal gate enforcement does not fail on None.

    Args:
        None.

    Returns:
        None.
    """
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
        "attestation_status": "absent",
        "attestation_absent_reason": "attestation_statement_absent",
        "authenticity_status": "absent",
        "attestation_source": "formal_input_payload",
        "bundle_verification": None,
    }

    result = _build_detect_attestation_result(
        cfg=cfg,
        attestation_context=attestation_context,
        content_evidence_payload=None,
        geometry_evidence_payload=None,
    )

    authenticity_result = result.get("authenticity_result")
    assert isinstance(authenticity_result, dict)
    assert authenticity_result.get("status") == "absent"
    assert authenticity_result.get("bundle_status") == "absent"
    assert result.get("attestation_absent_reason") == "attestation_statement_absent"


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
    assert final_decision.get("event_attestation_statistics_score") == pytest.approx(result.get("content_attestation_score"))
    assert final_decision.get("event_attestation_statistics_score") == pytest.approx(
        final_decision.get("event_attestation_score")
    )
    assert final_decision.get("event_attestation_statistics_score_name") == "event_attestation_statistics_score"
    assert final_decision.get("event_attestation_statistics_score_semantics") == (
        "legacy_alias_of_event_attestation_score_not_an_independent_statistics_semantics"
    )


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


def test_build_detect_attestation_result_prefers_real_lf_attestation_features_over_proxy_score() -> None:
    """
    功能：detect 侧 attestation 必须优先消费真实 LF 主路径系数，而不是退回低 proxy 分数。

    Detect-side attestation must consume the real LF main-path coefficient
    vector before falling back to a low lf_score proxy.

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
        "attestation_source": "formal_input_payload",
        "authenticity_status": "authentic",
        "bundle_verification": {"status": "ok"},
    }
    content_evidence_payload = {
        "status": "ok",
        "lf_score": 0.25011487,
        "score_parts": {
            "hf_attestation_values": [10.0, 0.0, 0.0, 0.0],
        },
    }

    result = _build_detect_attestation_result(
        cfg=cfg,
        attestation_context=attestation_context,
        content_evidence_payload=content_evidence_payload,
        geometry_evidence_payload=None,
        lf_attestation_features=_build_matching_lf_attestation_features(
            payload["statement"],
            "5" * 64,
            trace_commit=cast(str, payload["bundle"].get("trace_commit")),
        ),
    )

    image_evidence_result = result.get("image_evidence_result")
    assert isinstance(image_evidence_result, dict)
    assert image_evidence_result.get("status") == "ok"
    content_attestation_score = image_evidence_result.get("content_attestation_score")
    assert isinstance(content_attestation_score, float)
    channel_scores = image_evidence_result.get("channel_scores")
    assert isinstance(channel_scores, dict)
    assert isinstance(channel_scores.get("lf"), float)
    assert float(channel_scores.get("lf")) > 0.5
    assert float(channel_scores.get("lf")) > content_evidence_payload["lf_score"]
    assert content_attestation_score > 0.65
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("status") == "attested"
    assert final_decision.get("is_event_attested") is True
    assert isinstance(final_decision.get("event_attestation_score"), float)
    assert float(final_decision.get("event_attestation_score")) > 0.65


def test_compute_lf_attestation_score_emits_agreement_fields() -> None:
    """
    功能：LF attestation 评分必须暴露 agreement 与 trace 字段。

    Verify LF attestation scoring exposes agreement_count and trace fields.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_statement()
    statement = statement_from_dict(payload["statement"])
    result = compute_lf_attestation_score(
        latent_features=_build_matching_lf_attestation_features(
            payload["statement"],
            "5" * 64,
            trace_commit=cast(str, payload["bundle"].get("trace_commit")),
            plan_digest="3" * 64,
            basis_digest="6" * 64,
        ),
        k_lf=derive_attestation_keys(
            "5" * 64,
            payload["attestation_digest"],
            trajectory_commit=cast(str, payload["bundle"].get("trace_commit")),
        ).k_lf,
        attestation_digest=compute_event_binding_digest(
            payload["attestation_digest"],
            cast(str, payload["bundle"].get("trace_commit")),
        ),
        lf_params={
            "variance": 1.7,
            "basis_rank": 36,
            "edit_timestep": 12,
            "plan_digest": "3" * 64,
            "lf_basis_digest": "6" * 64,
            "projection_matrix_digest": "7" * 64,
            "trajectory_feature_spec_digest": "8" * 64,
            "trajectory_feature_vector": [0.1, -0.2, 0.3],
            "trajectory_feature_digest": "9" * 64,
            "projected_lf_digest": "a" * 64,
            "projection_seed": 17,
            "trajectory_feature_spec": {
                "feature_operator": "masked_normalized_random_projection",
                "edit_timestep": 12,
            },
        },
    )

    assert result.get("status") == "ok"
    assert result.get("agreement_count") == result.get("n_bits_compared")
    assert result.get("basis_rank") == 36
    assert result.get("variance") == pytest.approx(1.7)
    assert result.get("edit_timestep") == 12
    assert result.get("trajectory_feature_spec") == {
        "feature_operator": "masked_normalized_random_projection",
        "edit_timestep": 12,
    }
    assert result.get("plan_digest") == "3" * 64
    assert result.get("lf_basis_digest") == "6" * 64
    assert result.get("projection_matrix_digest") == "7" * 64
    assert result.get("trajectory_feature_spec_digest") == "8" * 64
    assert result.get("trajectory_feature_vector") == [0.1, -0.2, 0.3]
    assert result.get("trajectory_feature_digest") == "9" * 64
    assert result.get("projected_lf_digest") == "a" * 64
    assert result.get("projection_seed") == 17
    assert result.get("expected_bit_signs")
    expected_template_bundle = channel_lf.derive_lf_template_bundle(
        {
            "watermark": {
                "lf": {
                    "variance": 1.7,
                    "message_length": 64,
                    "ecc_sparsity": 3,
                }
            },
            "lf_attestation_event_digest": compute_event_binding_digest(
                payload["attestation_digest"],
                cast(str, payload["bundle"].get("trace_commit")),
            ),
            "lf_attestation_key": derive_attestation_keys(
                "5" * 64,
                payload["attestation_digest"],
                trajectory_commit=cast(str, payload["bundle"].get("trace_commit")),
            ).k_lf,
            "plan_digest": "3" * 64,
            "lf_basis_digest": "6" * 64,
        },
        int(result.get("n_bits_compared") or 0),
    )
    assert result.get("expected_bit_signs") == [
        int(value) for value in expected_template_bundle["codeword_bipolar"].tolist()
    ]
    assert result.get("posterior_values")
    assert result.get("posterior_signs")
    assert result.get("posterior_margin_values")
    assert result.get("agreement_indices") == list(range(int(result.get("n_bits_compared") or 0)))
    assert result.get("mismatch_indices") == []
    assert result.get("projected_lf_coeffs")
    assert result.get("projected_lf_signs")
    assert result.get("lf_attestation_trace_digest")


def test_build_detect_attestation_result_emits_lf_trace_artifact() -> None:
    """
    功能：detect 侧 attestation 必须输出 LF trace 工件。

    Verify detect-side attestation emits the LF trace artifact.

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
        "attestation_source": "formal_input_payload",
        "authenticity_status": "authentic",
        "bundle_verification": {"status": "ok"},
    }

    result = _build_detect_attestation_result(
        cfg=cfg,
        attestation_context=attestation_context,
        content_evidence_payload={
            "status": "ok",
            "score_parts": {
                "hf_attestation_values": [10.0, 0.0, 0.0, 0.0],
            },
        },
        geometry_evidence_payload=None,
        lf_attestation_features=_build_matching_lf_attestation_features(
            payload["statement"],
            "5" * 64,
            trace_commit=cast(str, payload["bundle"].get("trace_commit")),
                plan_digest="3" * 64,
                basis_digest="6" * 64,
        ),
        lf_attestation_trace_context={
            "variance": 1.7,
            "basis_rank": 36,
            "edit_timestep": 12,
            "plan_digest": "3" * 64,
            "lf_basis_digest": "6" * 64,
            "projection_matrix_digest": "7" * 64,
            "trajectory_feature_spec_digest": "8" * 64,
            "trajectory_feature_vector": [0.1, -0.2, 0.3],
            "trajectory_feature_digest": "9" * 64,
            "projected_lf_digest": "a" * 64,
            "projection_seed": 17,
            "pre_injection_coeffs": [-0.4, 0.2, -0.1],
            "injected_template_coeffs": [0.3, -0.5, 0.4],
            "post_injection_coeffs": [-0.1, -0.3, 0.3],
            "embed_closed_loop_digest": "b" * 64,
            "embed_closed_loop_step_index": 12,
            "embed_closed_loop_selection_rule": "max_lf_delta_norm",
            "planner_rank": 3,
            "basis_digest": "c" * 64,
            "route_basis_bridge": {
                "region_index_digest": "d" * 64,
                "lf_feature_cols": [3, 5, 7],
                "route_layer": {
                    "route_source": "mask_region_index_spec",
                    "feature_routing_mode": "mask_routed_projection",
                    "lf_feature_cols_source": "mask_partition",
                },
                "feature_bridge_layer": {
                    "route_to_feature_bridge": "mask_routed_feature_partition",
                },
            },
            "trajectory_feature_spec": {
                "feature_operator": "masked_normalized_random_projection",
                "edit_timestep": 12,
            },
        },
    )

    trace_artifact = cast(Dict[str, Any], result.get("_lf_attestation_trace_artifact"))
    alignment_artifact = cast(Dict[str, Any], result.get("_lf_alignment_table_artifact"))
    planner_artifact = cast(Dict[str, Any], result.get("_lf_planner_risk_report_artifact"))
    assert trace_artifact.get("artifact_type") == "lf_attestation_trace"
    assert trace_artifact.get("agreement_count") == trace_artifact.get("n_bits_compared")
    assert trace_artifact.get("basis_rank") == 36
    assert trace_artifact.get("variance") == pytest.approx(1.7)
    assert trace_artifact.get("edit_timestep") == 12
    assert trace_artifact.get("trajectory_feature_spec") == {
        "feature_operator": "masked_normalized_random_projection",
        "edit_timestep": 12,
    }
    assert trace_artifact.get("plan_digest") == "3" * 64
    assert trace_artifact.get("lf_basis_digest") == "6" * 64
    assert trace_artifact.get("projection_matrix_digest") == "7" * 64
    assert trace_artifact.get("trajectory_feature_spec_digest") == "8" * 64
    assert trace_artifact.get("trajectory_feature_vector") == [0.1, -0.2, 0.3]
    assert trace_artifact.get("trajectory_feature_digest") == "9" * 64
    assert trace_artifact.get("projected_lf_digest") == "a" * 64
    assert trace_artifact.get("projection_seed") == 17
    assert trace_artifact.get("expected_bit_signs")
    assert trace_artifact.get("posterior_values")
    assert trace_artifact.get("posterior_signs")
    assert trace_artifact.get("posterior_margin_values")
    assert trace_artifact.get("agreement_indices") == list(range(int(trace_artifact.get("n_bits_compared") or 0)))
    assert trace_artifact.get("mismatch_indices") == []
    assert trace_artifact.get("projected_lf_coeffs")
    assert trace_artifact.get("projected_lf_signs")
    assert trace_artifact.get("lf_attestation_trace_digest")
    assert alignment_artifact.get("artifact_type") == "lf_alignment_table"
    assert alignment_artifact.get("pre_injection_coeffs") == [-0.4, 0.2, -0.1]
    assert alignment_artifact.get("post_injection_coeffs") == [-0.1, -0.3, 0.3]
    assert alignment_artifact.get("embed_closed_loop_selection_rule") == "max_lf_delta_norm"
    assert alignment_artifact.get("post_still_negative_count") == 1
    assert alignment_artifact.get("detect_reverted_after_post_positive_count") == 0
    assert planner_artifact.get("artifact_type") == "lf_planner_risk_report"
    assert planner_artifact.get("primary_evidence", {}).get("evidence_type") == "lf_closed_loop_posterior_counts"
    assert isinstance(planner_artifact.get("routing_pattern_summary", {}).get("mismatch_feature_cols"), list)


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


def test_verify_attestation_statistics_score_stays_zero_for_authentic_mismatch() -> None:
    """
    功能：authentic 但未 attested 的样本，其统计链分数必须保持 0。 

    Authentic-but-mismatch samples must keep the event statistics score at zero.
    """
    payload = _build_statement()

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={"lf_score": 0.62},
        geo_score=0.1,
        attested_threshold=0.65,
        geo_rescue_band_delta_low=0.05,
        geo_rescue_min_score=0.3,
    )

    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("status") == "mismatch"
    assert final_decision.get("authenticity_status") == "authentic"
    assert final_decision.get("event_attestation_score") == pytest.approx(0.0)
    assert final_decision.get("event_attestation_statistics_score") == pytest.approx(0.0)


def test_verify_attestation_statement_only_alignment_artifact_matches_embed_signs() -> None:
    """
    功能：statement-only 模式下 embed 与 detect 的 LF expected signs 必须完全对齐。
    """
    payload = _build_statement()
    trace_commit = cast(str, payload["bundle"].get("trace_commit"))
    matching_features = _build_matching_lf_attestation_features(
        payload["statement"],
        "5" * 64,
        trace_commit=trace_commit,
        plan_digest=payload["statement"]["plan_digest"],
        basis_digest="6" * 64,
        event_binding_mode="statement_only",
    )
    attest_keys = derive_attestation_keys(
        "5" * 64,
        payload["attestation_digest"],
        trajectory_commit=trace_commit,
        event_binding_mode="statement_only",
    )
    template_bundle = channel_lf.derive_lf_template_bundle(
        {
            "watermark": {"lf": {"message_length": 64, "ecc_sparsity": 3}},
            "lf_attestation_event_digest": attest_keys.event_binding_digest,
            "lf_attestation_key": attest_keys.k_lf,
            "plan_digest": payload["statement"]["plan_digest"],
            "lf_basis_digest": "6" * 64,
            "basis_digest": "6" * 64,
            "event_binding_mode": "statement_only",
        },
        48 * 8,
    )
    expected_signs = [int(value) for value in template_bundle["codeword_bipolar"].tolist()[:3]]

    result = verify_attestation(
        k_master="5" * 64,
        candidate_statement=payload["statement"],
        attestation_bundle=payload["bundle"],
        content_evidence={"lf_score": 0.95},
        cfg={"attestation": {"use_trajectory_mix": False}},
        lf_latent_features=matching_features[:3],
        lf_params={
            "variance": 1.5,
            "block_length": 3,
            "message_length": 64,
            "ecc_sparsity": 3,
            "plan_digest": payload["statement"]["plan_digest"],
            "lf_basis_digest": "6" * 64,
            "basis_digest": "6" * 64,
            "event_binding_mode": "statement_only",
            "pre_injection_coeffs": [-0.4, 0.2, -0.1],
            "injected_template_coeffs": [0.3, -0.5, 0.4],
            "post_injection_coeffs": [-0.1, -0.3, 0.3],
            "embed_expected_bit_signs": expected_signs,
            "embed_codeword_source": "statement_only_attestation",
            "embed_attestation_event_digest": attest_keys.event_binding_digest,
            "embed_event_binding_mode": "statement_only",
            "embed_basis_digest": "6" * 64,
            "embed_basis_binding_status": "bound",
        },
    )

    alignment_artifact = cast(Dict[str, Any], result.get("_lf_alignment_table_artifact"))
    assert alignment_artifact.get("expected_bit_signs") == expected_signs
    assert alignment_artifact.get("formal_expected_bit_signs") == expected_signs
    assert alignment_artifact.get("embed_expected_bit_signs") == expected_signs
    assert alignment_artifact.get("embed_formal_expected_signs_match") is True
    final_decision = cast(Dict[str, Any], result.get("final_event_attested_decision"))
    assert final_decision.get("event_attestation_score_name") == "event_attestation_score"
    assert final_decision.get("event_attestation_statistics_score_name") == "event_attestation_statistics_score"


def test_lf_planner_risk_report_uses_sign_source_mismatch_guard() -> None:
    """
    功能：当 embed/formal expected signs 不一致时，planner 风险报告只能输出 sign_source_mismatch。
    """
    alignment_table = {
        "expected_bit_signs": [1, -1, 1],
        "embed_expected_bit_signs": [-1, -1, 1],
        "formal_expected_bit_signs": [1, -1, 1],
        "embed_formal_expected_signs_match": False,
        "expected_signs_mismatch_reason": "embed_formal_sign_values_differ",
        "signed_pre_alignment": [-0.4, -0.2, -0.1],
        "signed_template_alignment": [0.3, 0.5, 0.4],
        "signed_post_alignment": [-0.1, 0.3, 0.3],
        "signed_detect_alignment": [-0.2, -0.2, 0.1],
        "detect_side_coeffs": [-0.2, 0.2, 0.1],
        "n_bits_compared": 3,
        "alignment_margin_threshold": 0.15,
        "strong_negative_pre_count": 2,
        "post_still_negative_count": 1,
        "post_crosses_target_halfspace_count": 2,
        "detect_reverted_after_post_positive_count": 1,
        "detect_crosses_target_halfspace_count": 0,
        "plan_digest": "4" * 64,
    }

    report = _build_lf_planner_risk_report_artifact(alignment_table, {"basis_digest": "5" * 64})

    assert isinstance(report, dict)
    assert report.get("risk_classification") == "sign_source_mismatch"
    assert report.get("host_baseline_dominant_flag") is False
    assert report.get("basis_sample_mismatch_flag") is False
    assert report.get("detect_trajectory_shift_flag") is False


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


def test_write_detect_attestation_artifact_persists_attestation_traces(monkeypatch: Any, tmp_path: Path) -> None:
    """
    功能：验证 detect attestation artifact writer 会额外写出 HF/LF attestation trace 工件。

    Verify detect attestation artifact writer emits the standalone HF/LF trace artifacts.

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
        },
        "lf_attestation_trace": {
            "artifact_type": "lf_attestation_trace",
            "lf_attestation_score": 0.6111111111111112,
            "agreement_count": 22,
            "n_bits_compared": 36,
            "basis_rank": 36,
            "variance": 1.7,
            "edit_timestep": 12,
            "trajectory_feature_spec": {
                "feature_operator": "masked_normalized_random_projection",
                "edit_timestep": 12,
            },
            "lf_attestation_trace_digest": "b" * 64,
        },
        "lf_alignment_table": {
            "artifact_type": "lf_alignment_table",
            "attestation_digest": "c" * 64,
            "event_binding_digest": "d" * 64,
            "trace_commit": "e" * 64,
            "n_bits_compared": 2,
            "expected_bit_signs": [1, -1],
            "pre_injection_coeffs": [-0.2, 0.1],
            "injected_template_coeffs": [0.3, -0.2],
            "post_injection_coeffs": [0.1, -0.1],
            "detect_side_coeffs": [-0.1, 0.2],
            "signed_pre_alignment": [-0.2, -0.1],
            "signed_template_alignment": [0.3, 0.2],
            "signed_post_alignment": [0.1, 0.1],
            "signed_detect_alignment": [-0.1, -0.2],
            "alignment_margin_threshold": 0.1,
            "pre_agreement_count": 0,
            "post_agreement_count": 2,
            "detect_agreement_count": 0,
            "strong_negative_pre_count": 2,
            "strong_negative_post_count": 0,
            "strong_negative_detect_count": 2,
            "post_still_negative_count": 1,
            "post_crosses_target_halfspace_count": 2,
            "detect_crosses_target_halfspace_count": 0,
            "detect_reverted_after_post_positive_count": 1,
            "lf_alignment_table_digest": "f" * 64,
        },
        "lf_planner_risk_report": {
            "artifact_type": "lf_planner_risk_report",
            "risk_report_version": "v1",
            "risk_classification": "detect_trajectory_shift",
            "plan_digest": "0" * 64,
            "basis_digest": "1" * 64,
            "primary_evidence": {
                "evidence_type": "lf_closed_loop_posterior_counts",
                "risk_classification_driver": "detect_trajectory_shift",
            },
            "per_dimension_summary": [],
            "high_confidence_mismatch_dimensions": [],
            "routing_pattern_summary": {
                "mismatch_dimension_count": 0,
            },
            "host_baseline_risk_summary": {
                "strong_negative_pre_count": 2,
                "post_still_negative_count": 1,
                "detect_reverted_after_post_positive_count": 1,
            },
        }
    }

    _write_detect_attestation_artifact(record, tmp_path / "artifacts", attestation_artifacts)

    result_path = str((tmp_path / "artifacts" / "attestation" / "attestation_result.json")).replace("\\", "/")
    hf_trace_path = str((tmp_path / "artifacts" / "attestation" / "hf_attestation_trace.json")).replace("\\", "/")
    lf_trace_path = str((tmp_path / "artifacts" / "attestation" / "lf_attestation_trace.json")).replace("\\", "/")
    lf_alignment_path = str((tmp_path / "artifacts" / "attestation" / "lf_alignment_table.json")).replace("\\", "/")
    planner_path = str((tmp_path / "artifacts" / "planner" / "lf_planner_risk_report.json")).replace("\\", "/")
    assert written[result_path] == record["attestation"]
    assert written[hf_trace_path]["artifact_type"] == "hf_attestation_trace"
    assert written[lf_trace_path]["artifact_type"] == "lf_attestation_trace"
    assert written[lf_trace_path]["agreement_count"] == 22
    assert written[lf_alignment_path]["artifact_type"] == "lf_alignment_table"
    assert written[lf_alignment_path]["post_crosses_target_halfspace_count"] == 2
    assert written[planner_path]["artifact_type"] == "lf_planner_risk_report"
    assert written[planner_path]["risk_classification"] == "detect_trajectory_shift"


def test_build_lf_planner_risk_report_artifact_classifies_host_baseline_dominant() -> None:
    artifact = _build_lf_planner_risk_report_artifact(
        {
            "plan_digest": "a" * 64,
            "n_bits_compared": 4,
            "expected_bit_signs": [1, -1, 1, -1],
            "detect_side_coeffs": [-0.9, 0.8, -0.7, 0.6],
            "signed_pre_alignment": [-0.9, -0.8, -0.7, -0.6],
            "signed_template_alignment": [0.4, 0.4, 0.4, 0.4],
            "signed_post_alignment": [-0.5, -0.4, -0.3, 0.2],
            "signed_detect_alignment": [-0.8, -0.7, -0.6, 0.1],
            "alignment_margin_threshold": 0.2,
            "strong_negative_pre_count": 4,
            "post_still_negative_count": 3,
            "post_crosses_target_halfspace_count": 1,
            "detect_crosses_target_halfspace_count": 0,
            "detect_reverted_after_post_positive_count": 0,
        },
        {
            "planner_rank": 4,
            "basis_digest": "b" * 64,
            "lf_feature_count": 4,
            "route_basis_bridge": {
                "lf_feature_cols": [0, 1, 2, 3],
                "route_layer": {"feature_routing_mode": "mask_routed_projection"},
            },
        },
    )

    assert artifact is not None
    assert artifact["risk_classification"] == "host_baseline_dominant"
    assert artifact["host_baseline_dominant_flag"] is True
    assert artifact["basis_sample_mismatch_flag"] is False
    assert artifact["detect_trajectory_shift_flag"] is False
    assert isinstance(artifact["host_baseline_ratio"], float)
    assert isinstance(artifact["sign_stability"], float)
    assert isinstance(artifact["reconstruction_residual_ratio"], float)
    assert isinstance(artifact["top1_energy_ratio"], float)
    assert isinstance(artifact["topk_energy_ratio"], float)
    assert artifact["primary_evidence"]["dominant_signal"] == "host_baseline_counts"
    assert len(artifact["per_dimension_summary"]) == 4
    assert artifact["host_baseline_risk_summary"]["post_still_negative_count"] == 3


def test_build_lf_planner_risk_report_artifact_classifies_basis_sample_mismatch() -> None:
    artifact = _build_lf_planner_risk_report_artifact(
        {
            "plan_digest": "a" * 64,
            "n_bits_compared": 4,
            "expected_bit_signs": [1, -1, 1, -1],
            "detect_side_coeffs": [-0.2, -0.3, 0.1, -0.4],
            "signed_pre_alignment": [-0.1, 0.1, -0.1, 0.1],
            "signed_template_alignment": [0.2, 0.2, 0.2, 0.2],
            "signed_post_alignment": [-0.2, -0.3, -0.1, 0.1],
            "signed_detect_alignment": [-0.2, -0.3, -0.1, 0.1],
            "alignment_margin_threshold": 0.1,
            "strong_negative_pre_count": 0,
            "post_still_negative_count": 3,
            "post_crosses_target_halfspace_count": 1,
            "detect_crosses_target_halfspace_count": 0,
            "detect_reverted_after_post_positive_count": 0,
        },
        {
            "planner_rank": 4,
            "basis_digest": "b" * 64,
            "route_basis_bridge": {
                "lf_feature_cols": [10, 11, 12, 13],
                "route_layer": {"feature_routing_mode": "mask_routed_projection"},
            },
        },
    )

    assert artifact is not None
    assert artifact["risk_classification"] == "basis_sample_mismatch"
    assert artifact["host_baseline_dominant_flag"] is False
    assert artifact["basis_sample_mismatch_flag"] is True
    assert artifact["detect_trajectory_shift_flag"] is False
    assert artifact["host_baseline_risk_summary"]["dominant_signal"] == "post_still_negative_counts"


def test_build_lf_planner_risk_report_artifact_classifies_detect_trajectory_shift() -> None:
    artifact = _build_lf_planner_risk_report_artifact(
        {
            "plan_digest": "a" * 64,
            "n_bits_compared": 4,
            "expected_bit_signs": [1, -1, 1, -1],
            "detect_side_coeffs": [-0.5, -0.4, 0.3, -0.6],
            "signed_pre_alignment": [-0.2, -0.1, -0.2, -0.1],
            "signed_template_alignment": [0.3, 0.3, 0.3, 0.3],
            "signed_post_alignment": [0.4, 0.5, 0.2, -0.2],
            "signed_detect_alignment": [-0.5, -0.4, 0.3, -0.6],
            "alignment_margin_threshold": 0.2,
            "strong_negative_pre_count": 0,
            "post_still_negative_count": 1,
            "post_crosses_target_halfspace_count": 3,
            "detect_crosses_target_halfspace_count": 0,
            "detect_reverted_after_post_positive_count": 2,
        },
        {
            "planner_rank": 4,
            "basis_digest": "b" * 64,
            "route_basis_bridge": {
                "lf_feature_cols": [2, 4, 6, 8],
                "route_layer": {"feature_routing_mode": "mask_routed_projection"},
            },
        },
    )

    assert artifact is not None
    assert artifact["risk_classification"] == "detect_trajectory_shift"
    assert artifact["host_baseline_dominant_flag"] is False
    assert artifact["basis_sample_mismatch_flag"] is False
    assert artifact["detect_trajectory_shift_flag"] is True
    assert artifact["high_confidence_mismatch_dimensions"][0]["detect_reverted_after_post_positive"] is True


def test_build_lf_planner_risk_report_artifact_classifies_mixed() -> None:
    artifact = _build_lf_planner_risk_report_artifact(
        {
            "plan_digest": "a" * 64,
            "n_bits_compared": 8,
            "expected_bit_signs": [1, -1, 1, -1, 1, -1, 1, -1],
            "detect_side_coeffs": [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2],
            "signed_pre_alignment": [-0.3, -0.2, -0.2, 0.1, 0.2, -0.1, 0.1, 0.2],
            "signed_template_alignment": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "signed_post_alignment": [-0.1, -0.2, -0.3, 0.3, 0.4, 0.5, -0.2, 0.4],
            "signed_detect_alignment": [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2],
            "alignment_margin_threshold": 0.15,
            "strong_negative_pre_count": 3,
            "post_still_negative_count": 4,
            "post_crosses_target_halfspace_count": 3,
            "detect_crosses_target_halfspace_count": 0,
                "detect_reverted_after_post_positive_count": 2,
        },
        {
            "planner_rank": 8,
            "basis_digest": "b" * 64,
            "route_basis_bridge": {
                "lf_feature_cols": [1, 3, 5, 7, 9, 11, 13, 15],
                "route_layer": {"feature_routing_mode": "mask_routed_projection"},
            },
        },
    )

    assert artifact is not None
    assert artifact["risk_classification"] == "mixed"
    assert artifact["host_baseline_dominant_flag"] is False
    assert artifact["basis_sample_mismatch_flag"] is False
    assert artifact["detect_trajectory_shift_flag"] is False
    assert "routing_pattern_summary" in artifact


def test_write_embed_planner_artifacts_persists_lf_risk_report(monkeypatch: Any, tmp_path: Path) -> None:
    written: Dict[str, Dict[str, Any]] = {}

    def _fake_write_artifact_json(path: str, payload: Dict[str, Any], indent: int = 2, ensure_ascii: bool = False) -> None:
        _ = indent
        _ = ensure_ascii
        written[path.replace("\\", "/")] = payload

    monkeypatch.setattr("main.cli.run_embed.records_io.write_artifact_json", _fake_write_artifact_json)

    _write_embed_planner_artifacts(
        {
            "lf_planner_risk_report": {
                "artifact_type": "lf_planner_risk_report",
                "risk_report_version": "v1",
                "risk_classification": "host_baseline_dominant",
                "host_baseline_ratio": 1.7,
                "reconstruction_residual_ratio": 0.12,
                "topk_energy_ratio": 0.91,
                "plan_digest": "a" * 64,
                "basis_digest": "b" * 64,
            }
        },
        tmp_path / "artifacts",
    )

    planner_path = str((tmp_path / "artifacts" / "planner" / "lf_planner_risk_report.json")).replace("\\", "/")
    assert written[planner_path]["artifact_type"] == "lf_planner_risk_report"
    assert written[planner_path]["risk_classification"] == "host_baseline_dominant"


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