"""
File purpose: LF 模板编解码（LowFreqTemplateCodec）语义与审计字段回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from main.core import digests
from main.watermarking.content_chain.ldpc_codec import build_ldpc_spec
from main.watermarking.content_chain.low_freq_coder import (
    LOW_FREQ_TEMPLATE_CODEC_ID,
    LOW_FREQ_TEMPLATE_CODEC_VERSION,
    LowFreqTemplateCodec,
)


def _build_cfg(message_length: int, ecc_sparsity: int) -> Dict[str, Any]:
    """
    功能：构造 LowFreqTemplateCodec 测试配置。

    Build minimal LF template codec config for unit tests.

    Args:
        message_length: Message length for LDPC.
        ecc_sparsity: Column-weight proxy for H construction.

    Returns:
        Configuration dictionary.
    """
    return {
        "watermark": {
            "lf": {
                "enabled": True,
                "coding_mode": "pseudogaussian_template_additive",
                "decoder": "matched_correlation",
                "variance": 1.5,
                "message_length": message_length,
                "ecc_sparsity": ecc_sparsity,
            }
        }
    }


def _make_codec() -> LowFreqTemplateCodec:
    """
    功能：构造标准 LowFreqTemplateCodec 实例。

    Build a standard LowFreqTemplateCodec instance for tests.

    Args:
        None.

    Returns:
        LowFreqTemplateCodec instance.
    """
    impl_digest = digests.canonical_sha256(
        {"impl_id": LOW_FREQ_TEMPLATE_CODEC_ID, "impl_version": LOW_FREQ_TEMPLATE_CODEC_VERSION}
    )
    return LowFreqTemplateCodec(LOW_FREQ_TEMPLATE_CODEC_ID, LOW_FREQ_TEMPLATE_CODEC_VERSION, impl_digest)


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


def test_lf_template_codec_embed_detect_exposes_ldpc_fields() -> None:
    """
    功能：验证 LowFreqTemplateCodec embed/detect 暴露 LDPC 审计字段。

    Verify LowFreqTemplateCodec embed/detect path exposes LDPC trace fields and correct
    detect_variant = "correlation_v2", raw_correlation, parity_check_digest.

    Args:
        None.

    Returns:
        None.
    """
    message_length = 8
    cfg = _build_cfg(message_length=message_length, ecc_sparsity=3)
    plan_digest = "plan_digest_for_s3_semantics"
    cfg_digest = digests.canonical_sha256(cfg)
    coder = _make_codec()

    ldpc_spec = build_ldpc_spec(
        message_length=message_length,
        ecc_sparsity=3,
        seed_key=f"{plan_digest}:{message_length}:3:embed",
    )
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

    basis_rank = 8
    rng = np.random.RandomState(7)
    proj_matrix = rng.randn(latent_length, basis_rank).astype(np.float32)
    lf_basis = {
        "projection_matrix": proj_matrix.tolist(),
        "basis_rank": basis_rank,
    }

    lf_score, detect_trace = coder.detect_score(
        cfg=cfg,
        latent_features=embed_result.get("latent_features_embedded"),
        plan_digest=plan_digest,
        cfg_digest=cfg_digest,
        lf_basis=lf_basis,
    )
    assert isinstance(lf_score, float)
    assert isinstance(detect_trace, dict)
    assert detect_trace.get("status") == "ok"
    assert detect_trace.get("detect_variant") == "correlation_v2"
    # higher_is_watermarked 语义：score > 0.5 表示水印存在证据。
    assert detect_trace.get("higher_is_watermarked") is True
    assert isinstance(detect_trace.get("raw_correlation"), float)
    assert isinstance(detect_trace.get("parity_check_digest"), str)


def test_lf_template_codec_detect_accepts_numpy_input_and_reports_basis_fields() -> None:
    """
    功能：验证 detect 支持 numpy.ndarray 输入并输出 basis 审计字段。

    Verify detect supports numpy.ndarray input and emits correlation_dim and basis_rank fields.

    Args:
        None.

    Returns:
        None.
    """
    message_length = 8
    cfg = _build_cfg(message_length=message_length, ecc_sparsity=3)
    plan_digest = "plan_digest_for_numpy_flatten_test"
    coder = _make_codec()

    feature_dim = 128
    basis_rank = 8
    latent_array = np.random.RandomState(11).randn(feature_dim).astype(np.float32)
    proj_matrix = np.random.RandomState(22).randn(feature_dim, basis_rank).astype(np.float32)
    lf_basis = {
        "projection_matrix": proj_matrix.tolist(),
        "basis_rank": basis_rank,
    }

    lf_score, detect_trace = coder.detect_score(
        cfg=cfg,
        latent_features=latent_array,
        plan_digest=plan_digest,
        cfg_digest="cfg_digest_numpy",
        lf_basis=lf_basis,
    )
    assert isinstance(lf_score, float)
    assert detect_trace.get("status") == "ok"
    # correlation_dim = min(len(coeffs_arr), basis_rank) = basis_rank（投影后维度）。
    assert detect_trace.get("correlation_dim") == basis_rank
    assert detect_trace.get("basis_rank") == basis_rank


def test_lf_template_codec_detect_fails_with_absent_lf_basis() -> None:
    """
    功能：验证 detect 在 lf_basis 缺失时显式返回 failed，不静默回退。

    Verify detect returns explicit failure when lf_basis is absent, with no silent fallback.

    Args:
        None.

    Returns:
        None.
    """
    message_length = 16
    cfg = _build_cfg(message_length=message_length, ecc_sparsity=3)
    plan_digest = "plan_digest_for_absent_basis_test"
    coder = _make_codec()

    latent_features = np.random.RandomState(42).randn(32).astype(np.float32)
    lf_score, detect_trace = coder.detect_score(
        cfg=cfg,
        latent_features=latent_features,
        plan_digest=plan_digest,
        cfg_digest="cfg_digest_absent_basis",
    )

    assert lf_score is None
    assert detect_trace.get("status") == "failed"
    assert detect_trace.get("lf_failure_reason") == "lf_basis_required_but_absent"
