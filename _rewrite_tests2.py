"""批量重写剩余测试文件为 V2 接口。"""
import pathlib

# --------------------------------------------------------------------------
# test_lf_coder_plan_digest_binding.py
# 测试 plan_digest 和配置参数影响 lf_trace_digest 的可追溯性
# --------------------------------------------------------------------------
plan_binding_content = '''\
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
    assert trace_1["lf_trace_digest"] == trace_2["lf_trace_digest"], \
        "lf_trace_digest 应在相同配置和输入下保持不变"
'''

# --------------------------------------------------------------------------
# test_lf_coder_plan_digest_mismatch.py
# 测试 plan_digest 为 None/空 时 V2 的 TypeError
# --------------------------------------------------------------------------
plan_mismatch_content = '''\
"""
测试用例：T2 - plan_digest 非法时必须抛出 TypeError

功能说明：
- 验证 V2 detect_score 在 plan_digest 为空字符串/None 时抛出 TypeError。
- V2 不再支持"mismatch"状态（plan_digest 直传，非从 cfg 读取），
  非法 plan_digest 应在接口层立即拒绝。
"""

from __future__ import annotations

import pytest

from typing import Any, Dict

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


def test_lf_coder_plan_digest_mismatch_returns_mismatch_status() -> None:
    """
    功能：验证 plan_digest 为空字符串时 detect_score 抛出 TypeError。

    Test that detect_score raises TypeError when plan_digest is an empty string.
    V2 validates plan_digest at entry and raises TypeError for invalid values.

    Args:
        None.

    Returns:
        None.

    Raises:
        pytest.raises(TypeError): Expected when plan_digest is empty.
    """
    cfg: Dict[str, Any] = {"watermark": {"lf": {"enabled": True, "message_length": 16, "ecc_sparsity": 3}}}
    coder = _make_coder()
    # V2 要求 plan_digest 为非空字符串，空字符串触发 TypeError。
    with pytest.raises(TypeError, match="plan_digest must be non-empty str"):
        coder.detect_score(
            cfg=cfg,
            latent_features=[0.1, 0.2, 0.3, 0.4],
            plan_digest="",
            lf_basis=None,
        )


def test_lf_coder_explicit_plan_digest_none_returns_mismatch() -> None:
    """
    功能：验证 plan_digest=None 时 detect_score 抛出 TypeError。

    Test that detect_score raises TypeError when plan_digest is None.

    Args:
        None.

    Returns:
        None.

    Raises:
        pytest.raises(TypeError): Expected when plan_digest is None.
    """
    cfg: Dict[str, Any] = {"watermark": {"lf": {"enabled": True, "message_length": 16, "ecc_sparsity": 3}}}
    coder = _make_coder()
    with pytest.raises(TypeError, match="plan_digest must be non-empty str"):
        coder.detect_score(
            cfg=cfg,
            latent_features=[0.1, 0.2, 0.3, 0.4],
            plan_digest=None,
            lf_basis=None,
        )
'''

# --------------------------------------------------------------------------
# test_lf_score_distribution_sensitivity.py
# 测试不同 latent_features 产生不同的 lf_score
# --------------------------------------------------------------------------
score_distribution_content = '''\
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
    assert not (score_1 == score_2 == score_3), \
        f"lf_score should vary with different inputs, got {score_1}, {score_2}, {score_3}"


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
    assert variance > 1e-6, \
        f"lf_score variance 过低（{variance}），输入敏感性不足"


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
    assert score_1 == score_2, \
        f"相同输入的 lf_score 应完全一致，实际 {score_1} vs {score_2}"
'''

# 写入文件
pathlib.Path("tests/test_lf_coder_plan_digest_binding.py").write_text(plan_binding_content, encoding="utf-8")
pathlib.Path("tests/test_lf_coder_plan_digest_mismatch.py").write_text(plan_mismatch_content, encoding="utf-8")
pathlib.Path("tests/test_lf_score_distribution_sensitivity.py").write_text(score_distribution_content, encoding="utf-8")
print("Done: plan_binding + plan_mismatch + score_distribution")
