"""
File purpose: S3 LDPC 语义与审计字段回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

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

    # 构造 lf_basis：feature_dim=latent_length（直接投影，无需索引降维）。
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
    # correlation_v1 不使用 BP，bp_converged/bp_iteration_count 固定为 None
    assert detect_trace.get("bp_converged") is None
    assert detect_trace.get("bp_iteration_count") is None
    assert detect_trace.get("detect_variant") == "correlation_v1"
    assert isinstance(detect_trace.get("raw_correlation"), float)
    assert isinstance(detect_trace.get("parity_check_digest"), str)


def test_lf_coder_prc_detect_flattens_numpy_array_and_reports_latent_dims() -> None:
    """
    功能：验证 detect 支持 numpy.ndarray 输入并输出维度审计字段。

    Verify detect supports numpy.ndarray input and emits dimension audit fields.

    Args:
        None.

    Returns:
        None.
    """
    message_length = 8
    cfg = _build_cfg(message_length=message_length, ecc_sparsity=3, bp_iterations=10)
    plan_digest = "plan_digest_for_numpy_flatten_test"

    impl_digest = digests.canonical_sha256({"impl_id": LF_CODER_PRC_ID, "impl_version": LF_CODER_PRC_VERSION})
    coder = LFCoderPRC(LF_CODER_PRC_ID, LF_CODER_PRC_VERSION, impl_digest)

    # 构造 128 维 numpy 输入，lf_basis 直接匹配（无需索引降维）。
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
    # available_latent_dim = len(coeffs_arr) = basis_rank（投影后维度）。
    assert detect_trace.get("available_latent_dim") == basis_rank
    assert detect_trace.get("correlation_dim") == basis_rank


def test_lf_coder_prc_detect_accepts_tensor_like_input_without_torch_dependency() -> None:
    """
    功能：验证 detect 在 lf_basis 缺失时显式返回 failed，不静默回退。

    Verify detect returns explicit failure when lf_basis is absent, no silent fallback.

    Args:
        None.

    Returns:
        None.
    """
    class _FakeTensor:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def detach(self) -> "_FakeTensor":
            return self

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._values

    message_length = 16
    cfg = _build_cfg(message_length=message_length, ecc_sparsity=3, bp_iterations=6)
    plan_digest = "plan_digest_for_tensor_like_flatten_test"

    impl_digest = digests.canonical_sha256({"impl_id": LF_CODER_PRC_ID, "impl_version": LF_CODER_PRC_VERSION})
    coder = LFCoderPRC(LF_CODER_PRC_ID, LF_CODER_PRC_VERSION, impl_digest)

    # lf_basis 缺失时应显式失败，不再静默回退。
    fake_tensor = _FakeTensor(np.asarray([[0.1, -0.2, 0.3]], dtype=np.float32))
    lf_score, detect_trace = coder.detect_score(
        cfg=cfg,
        latent_features=fake_tensor,
        plan_digest=plan_digest,
        cfg_digest="cfg_digest_tensor_like",
    )

    assert lf_score is None
    assert detect_trace.get("status") == "failed"
    assert detect_trace.get("lf_failure_reason") == "lf_basis_required_but_absent"
