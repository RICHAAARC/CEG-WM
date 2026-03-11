"""
测试用例：T5 - latent_features 不同导致 lf_score 分布不同

功能说明：
- 验证 lf_score 确实依赖 latent_features 的实际数值。
- 验证不同输入产生不同的 lf_score。
- 验证相同输入的 lf_score 可复算。
"""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, List

from main.core import digests
from main.watermarking.content_chain.low_freq_coder import (
    LowFreqTemplateCodecV2,
    LOW_FREQ_TEMPLATE_CODEC_V2_ID,
    LOW_FREQ_TEMPLATE_CODEC_V2_VERSION,
)


def _make_coder() -> LowFreqTemplateCodecV2:
    impl_digest = digests.canonical_sha256({
        "impl_id": LOW_FREQ_TEMPLATE_CODEC_V2_ID,
        "impl_version": LOW_FREQ_TEMPLATE_CODEC_V2_VERSION,
    })
    return LowFreqTemplateCodecV2(
        LOW_FREQ_TEMPLATE_CODEC_V2_ID,
        LOW_FREQ_TEMPLATE_CODEC_V2_VERSION,
        impl_digest,
    )


def _build_lf_basis(feature_dim: int = 64, basis_rank: int = 8, seed: int = 42) -> Dict[str, Any]:
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


_BASE_CFG: Dict[str, Any] = {
    "watermark": {
        "lf": {
            "enabled": True,
            "message_length": 16,
            "ecc_sparsity": 3,
            "correlation_scale": 10.0,
        }
    }
}
_PLAN_DIGEST = "test_score_sensitivity_plan"
_LF_BASIS = _build_lf_basis(feature_dim=64, basis_rank=8, seed=42)


def _score(latent_features: List[float]) -> float:
    coder = _make_coder()
    score, trace = coder.detect_score(
        cfg=_BASE_CFG,
        latent_features=latent_features,
        plan_digest=_PLAN_DIGEST,
        lf_basis=_LF_BASIS,
    )
    assert trace["status"] == "ok", f"Expected ok, got {trace['status']}"
    return score


def test_lf_score_depends_on_latent_features() -> None:
    """
    功能：验证 lf_score 依赖 latent_features。

    Test that lf_score changes when latent_features change.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_score is identical for different inputs.
    """
    rng = np.random.RandomState(1)
    inputs_1 = rng.randn(64).tolist()
    inputs_2 = rng.randn(64).tolist()
    inputs_3 = np.ones(64).tolist()

    score_1 = _score(inputs_1)
    score_2 = _score(inputs_2)
    score_3 = _score(inputs_3)

    # 至少有两个分数不同。
    assert not (score_1 == score_2 == score_3),         f"lf_score should vary with different inputs, got {score_1}, {score_2}, {score_3}"


def test_lf_score_distribution_varies_across_inputs() -> None:
    """
    功能：验证 lf_score 在多个输入上的分布有差异。

    Test that lf_score distribution has sufficient variance across multiple inputs.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_score variance is too low.
    """
    rng = np.random.RandomState(42)
    scores: List[float] = []
    for _ in range(10):
        latent_features = rng.randn(64).tolist()
        scores.append(_score(latent_features))

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    assert variance > 1e-6,         f"lf_score variance 过低（{variance}），输入敏感性不足"


def test_lf_score_same_input_produces_same_score() -> None:
    """
    功能：验证相同输入产生相同的 lf_score（可复算）。

    Test that identical inputs produce identical lf_score (reproducibility).

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_score is not reproducible.
    """
    rng = np.random.RandomState(99)
    latents = rng.randn(64).tolist()
    score_1 = _score(latents)
    score_2 = _score(latents)
    assert score_1 == score_2,         f"相同输入的 lf_score 应完全一致，实际 {score_1} vs {score_2}"
