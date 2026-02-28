"""
File purpose: S3 LDPC 语义与审计字段回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests
from main.watermarking.content_chain.ldpc_codec import build_ldpc_spec
from main.watermarking.content_chain.low_freq_coder import (
    LF_CODER_PRC_ID,
    LF_CODER_PRC_VERSION,
    LFCoderPRC,
)


def _build_cfg(message_length: int, ecc_sparsity: int, bp_iterations: int = 8) -> Dict[str, Any]:
    """
    功能：构造 LF PRC 测试配置。

    Build minimal LF PRC config for unit tests.

    Args:
        message_length: Message length for LDPC.
        ecc_sparsity: Column-weight proxy for H construction.
        bp_iterations: Maximum BP iterations.

    Returns:
        Configuration dictionary.
    """
    return {
        "watermark": {
            "lf": {
                "enabled": True,
                "coding_mode": "latent_space_sign_flipping",
                "decoder": "belief_propagation",
                "variance": 1.5,
                "message_length": message_length,
                "ecc_sparsity": ecc_sparsity,
                "bp_iterations": bp_iterations,
            }
        }
    }


def test_parity_check_digest_changes_with_ecc_sparsity() -> None:
    """
    功能：验证 ecc_sparsity 变化会改变 parity_check_digest。

    Verify parity_check_digest changes when ecc_sparsity changes.

    Args:
        None.

    Returns:
        None.
    """
    spec_a = build_ldpc_spec(message_length=16, ecc_sparsity=2, seed_key="seed_fixed")
    spec_b = build_ldpc_spec(message_length=16, ecc_sparsity=4, seed_key="seed_fixed")

    assert spec_a["parity_check_digest"] != spec_b["parity_check_digest"]


def test_lf_coder_prc_embed_detect_exposes_ldpc_bp_fields() -> None:
    """
    功能：验证 PRC embed/detect 暴露 LDPC 与 BP 审计字段。

    Verify PRC embed/detect path exposes LDPC and BP trace fields.

    Args:
        None.

    Returns:
        None.
    """
    message_length = 8
    cfg = _build_cfg(message_length=message_length, ecc_sparsity=3, bp_iterations=10)
    plan_digest = "plan_digest_for_s3_semantics"
    cfg_digest = digests.canonical_sha256(cfg)

    impl_digest = digests.canonical_sha256({"impl_id": LF_CODER_PRC_ID, "impl_version": LF_CODER_PRC_VERSION})
    coder = LFCoderPRC(LF_CODER_PRC_ID, LF_CODER_PRC_VERSION, impl_digest)

    ldpc_spec = build_ldpc_spec(message_length=message_length, ecc_sparsity=3, seed_key=f"{plan_digest}:{message_length}:3:embed")
    latent_length = int(ldpc_spec.get("n", message_length))
    latent_features = [0.05 * (i + 1) for i in range(latent_length)]

    embed_result = coder.embed_apply(
        cfg=cfg,
        latent_features=latent_features,
        plan_digest=plan_digest,
        cfg_digest=cfg_digest,
    )
    assert embed_result.get("status") == "ok"
    trace_summary = embed_result.get("lf_trace_summary")
    assert isinstance(trace_summary, dict)
    assert trace_summary.get("block_length") == latent_length
    assert isinstance(trace_summary.get("parity_check_digest"), str)

    lf_score, detect_trace = coder.detect_score(
        cfg=cfg,
        latent_features=embed_result.get("latent_features_embedded"),
        plan_digest=plan_digest,
        cfg_digest=cfg_digest,
    )
    assert isinstance(lf_score, float)
    assert isinstance(detect_trace, dict)
    assert detect_trace.get("status") == "ok"
    assert isinstance(detect_trace.get("bp_converged"), bool)
    assert isinstance(detect_trace.get("bp_iteration_count"), int)
    assert isinstance(detect_trace.get("parity_check_digest"), str)
