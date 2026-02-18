"""
测试用例：T5 - latent_features 不同导致 lf_score 分布不同

功能说明：
- 验证 lf_score 确实依赖 latent_features 的实际数值。
- 验证不同输入产生不同的 lf_score。
- 验证 lf_score 在同一配置下对不同输入的响应。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from main.watermarking.content_chain.low_freq_coder import LowFreqCoder


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
    cfg: Dict[str, Any] = {
        "watermark": {
            "plan_digest": "test_plan_digest_001",
            "lf": {
                "enabled": True,
                "codebook_id": "lf_codebook_v1",
                "ecc": 3,
                "strength": 0.5,
                "delta": 1.0,
                "block_length": 8
            }
        }
    }

    coder = LowFreqCoder(
        impl_id="low_freq_coder_v1",
        impl_version="v1",
        impl_digest="test_impl_digest"
    )

    # 不同的输入向量。
    inputs_1: Dict[str, Any] = {
        "latent_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "latent_shape": (8,)
    }

    inputs_2: Dict[str, Any] = {
        "latent_features": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "latent_shape": (8,)
    }

    inputs_3: Dict[str, Any] = {
        "latent_features": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "latent_shape": (8,)
    }

    # 执行编码。
    result_1 = coder.extract(cfg=cfg, inputs=inputs_1)
    result_2 = coder.extract(cfg=cfg, inputs=inputs_2)
    result_3 = coder.extract(cfg=cfg, inputs=inputs_3)

    assert result_1.status == "ok", "Encoding 1 failed"
    assert result_2.status == "ok", "Encoding 2 failed"
    assert result_3.status == "ok", "Encoding 3 failed"

    score_1 = result_1.lf_score
    score_2 = result_2.lf_score
    score_3 = result_3.lf_score

    # 验证分数不同（至少有两个不同）。
    assert not (score_1 == score_2 == score_3), \
        f"lf_score should vary with different inputs, got {score_1}, {score_2}, {score_3}"


def test_lf_score_distribution_varies_across_inputs() -> None:
    """
    功能：验证 lf_score 在多个输入上的分布有差异。

    Test that lf_score distribution varies across multiple inputs.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_score variance is too low.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "plan_digest": "test_plan_digest_001",
            "lf": {
                "enabled": True,
                "codebook_id": "lf_codebook_v1",
                "ecc": 3,
                "strength": 0.5,
                "delta": 1.0,
                "block_length": 8
            }
        }
    }

    coder = LowFreqCoder(
        impl_id="low_freq_coder_v1",
        impl_version="v1",
        impl_digest="test_impl_digest"
    )

    # 生成多个不同的输入向量。
    import random
    random.seed(42)

    scores: List[float] = []
    for i in range(10):
        latent_features = [random.uniform(-1.0, 1.0) for _ in range(8)]
        inputs = {
            "latent_features": latent_features,
            "latent_shape": (8,)
        }
        result = coder.extract(cfg=cfg, inputs=inputs)
        assert result.status == "ok", f"Encoding {i} failed: {result.content_failure_reason}"
        scores.append(result.lf_score)

    # 验证分数的方差不为零（即有差异）。
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)

    assert variance > 1e-6, \
        f"lf_score variance is too low ({variance}), indicating insufficient input sensitivity"


def test_lf_score_same_input_produces_same_score() -> None:
    """
    功能：验证同一输入产生相同的 lf_score（可复算）。

    Test that the same input produces the same lf_score (reproducibility).

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_score is not reproducible.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "plan_digest": "test_plan_digest_001",
            "lf": {
                "enabled": True,
                "codebook_id": "lf_codebook_v1",
                "ecc": 3,
                "strength": 0.5,
                "delta": 1.0,
                "block_length": 8
            }
        }
    }

    inputs: Dict[str, Any] = {
        "latent_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "latent_shape": (8,)
    }

    coder = LowFreqCoder(
        impl_id="low_freq_coder_v1",
        impl_version="v1",
        impl_digest="test_impl_digest"
    )

    # 执行多次编码。
    result_1 = coder.extract(cfg=cfg, inputs=inputs)
    result_2 = coder.extract(cfg=cfg, inputs=inputs)
    result_3 = coder.extract(cfg=cfg, inputs=inputs)

    assert result_1.status == "ok"
    assert result_2.status == "ok"
    assert result_3.status == "ok"

    # 验证分数完全相同（可复算）。
    assert result_1.lf_score == result_2.lf_score == result_3.lf_score, \
        "lf_score must be reproducible for the same input"
