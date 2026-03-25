"""
测试用例：T1 - LF 参数变化导致 lf_trace_digest 变化

功能说明：
- 验证 plan_digest 和关键 LF 参数（message_length/ecc_sparsity/correlation_scale）
  的变化会导致 lf_trace_digest 变化。
- 确保 V2 detect_score 的 lf_trace_digest 可复算性（相同输入 → 相同摘要）。
"""

from __future__ import annotations

import numpy as np

from typing import Any, Dict

from main.core import digests
from main.watermarking.content_chain.low_freq_coder import (
    LowFreqTemplateCodec,
    LOW_FREQ_TEMPLATE_CODEC_ID,
    LOW_FREQ_TEMPLATE_CODEC_VERSION,
)


def _make_coder() -> LowFreqTemplateCodec:
    impl_digest = digests.canonical_sha256({
        "impl_id": LOW_FREQ_TEMPLATE_CODEC_ID,
        "impl_version": LOW_FREQ_TEMPLATE_CODEC_VERSION,
    })
    return LowFreqTemplateCodec(
        LOW_FREQ_TEMPLATE_CODEC_ID,
        LOW_FREQ_TEMPLATE_CODEC_VERSION,
        impl_digest,
    )


def _build_lf_basis(feature_dim: int = 128, basis_rank: int = 8, seed: int = 42) -> Dict[str, Any]:
    """构造用于测试的最小合法 lf_basis。"""
    rng = np.random.RandomState(seed)
    projection_matrix = rng.randn(feature_dim, basis_rank).astype(np.float32)
    return {
        "projection_matrix": projection_matrix.tolist(),
        "basis_rank": basis_rank,
        "latent_projection_spec": {
            "spec_version": "v1",
            "method": "random_index_selection",
            "feature_dim": feature_dim,
            "seed": seed,
            "edit_timestep": 0,
            "sample_idx": 0,
        },
    }


_BASE_LATENTS = np.random.RandomState(0).randn(128).tolist()


def _run(cfg: Dict[str, Any], plan_digest: str, lf_basis: Dict[str, Any]) -> str:
    """执行 detect_score，返回 lf_trace_digest（ok 时）。"""
    coder = _make_coder()
    score, trace = coder.detect_score(
        cfg=cfg,
        latent_features=_BASE_LATENTS,
        plan_digest=plan_digest,
        lf_basis=lf_basis,
    )
    assert trace["status"] == "ok", f"Expected ok, got {trace['status']} ({trace})"
    return trace["lf_trace_digest"]


def _base_cfg(message_length: int = 32, ecc_sparsity: int = 3, correlation_scale: float = 10.0) -> Dict[str, Any]:
    return {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": message_length,
                "ecc_sparsity": ecc_sparsity,
                "correlation_scale": correlation_scale,
            }
        }
    }


def test_lf_parameter_change_causes_plan_digest_change() -> None:
    """
    功能：验证 plan_digest 或关键 LF 参数变化导致 lf_trace_digest 变化。

    Test that plan_digest change and config changes (message_length, ecc_sparsity,
    correlation_scale) each independently cause lf_trace_digest to change.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_trace_digest does not change when parameters change.
    """
    lf_basis = _build_lf_basis()
    base_pd = "baseline_plan_digest_001"
    base_digest = _run(_base_cfg(), base_pd, lf_basis)

    # T1.1: plan_digest 变化 → lf_trace_digest 必变化。
    variant_pd = _run(_base_cfg(), "different_plan_digest_999", lf_basis)
    assert variant_pd != base_digest, "plan_digest change must cause lf_trace_digest to change"

    # T1.2: message_length 变化 → parity_check_digest 变化 → lf_trace_digest 变化。
    variant_ml = _run(_base_cfg(message_length=64), base_pd, lf_basis)
    assert variant_ml != base_digest, "message_length change must cause lf_trace_digest to change"

    # T1.3: ecc_sparsity 变化 → parity_check_digest 变化 → lf_trace_digest 变化。
    variant_ecc = _run(_base_cfg(ecc_sparsity=5), base_pd, lf_basis)
    assert variant_ecc != base_digest, "ecc_sparsity change must cause lf_trace_digest to change"

    # T1.4: correlation_scale 变化 → lf_trace_digest 变化。
    variant_scale = _run(_base_cfg(correlation_scale=20.0), base_pd, lf_basis)
    assert variant_scale != base_digest, "correlation_scale change must cause lf_trace_digest to change"


def test_lf_trace_digest_is_reproducible() -> None:
    """
    功能：验证 lf_trace_digest 可复算（相同配置 + 相同输入 → 相同摘要）。

    Test that lf_trace_digest is reproducible for identical configuration and input.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_trace_digest is not reproducible.
    """
    cfg = _base_cfg()
    lf_basis = _build_lf_basis()
    plan_digest = "repro_test_plan_digest"
    coder = _make_coder()

    _, trace_1 = coder.detect_score(
        cfg=cfg, latent_features=_BASE_LATENTS, plan_digest=plan_digest, lf_basis=lf_basis,
    )
    _, trace_2 = coder.detect_score(
        cfg=cfg, latent_features=_BASE_LATENTS, plan_digest=plan_digest, lf_basis=lf_basis,
    )

    assert trace_1["status"] == "ok"
    assert trace_2["status"] == "ok"
    assert trace_1["lf_trace_digest"] == trace_2["lf_trace_digest"],         "lf_trace_digest 应在相同配置和输入下保持不变"
