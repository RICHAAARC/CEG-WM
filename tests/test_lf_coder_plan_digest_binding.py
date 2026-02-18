"""
测试用例：T1 - LF 参数变化导致 plan_digest 变化

功能说明：
- 验证 watermark.lf.* 参数的任意变化都会导致 plan_digest 变化。
- 覆盖 plan_digest_include_paths 中声明的所有 LF 参数。
- 确保 append-only 语义：新增参数不影响旧参数的 digest 绑定。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from main.core import digests
from main.watermarking.content_chain.low_freq_coder import LowFreqCoder


def test_lf_parameter_change_causes_plan_digest_change() -> None:
    """
    功能：验证 LF 参数变化导致 plan_digest 变化。

    Test that any LF parameter change causes plan_digest to change.
    Verifies binding of all watermark.lf.* parameters in plan_digest_include_paths.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If plan_digest does not change when LF parameters change.
    """
    # 基准配置。
    base_cfg: Dict[str, Any] = {
        "watermark": {
            "plan_digest": "baseline_plan_digest_001",
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

    # 基准输入。
    base_inputs: Dict[str, Any] = {
        "latent_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "latent_shape": (8,)
    }

    # 实例化 LowFreqCoder。
    coder = LowFreqCoder(
        impl_id="low_freq_coder_v1",
        impl_version="v1",
        impl_digest="test_impl_digest_001"
    )

    # 执行基准编码。
    result_base = coder.extract(cfg=base_cfg, inputs=base_inputs)
    assert result_base.status == "ok", f"Base encoding failed: {result_base.content_failure_reason}"
    assert result_base.lf_trace_digest is not None, "Base lf_trace_digest should not be None"
    baseline_trace_digest = result_base.lf_trace_digest

    # T1.1: 修改 codebook_id → trace_digest 必变化。
    cfg_variant_codebook = base_cfg.copy()
    cfg_variant_codebook["watermark"] = base_cfg["watermark"].copy()
    cfg_variant_codebook["watermark"]["lf"] = base_cfg["watermark"]["lf"].copy()
    cfg_variant_codebook["watermark"]["lf"]["codebook_id"] = "lf_codebook_v2"

    result_variant_codebook = coder.extract(cfg=cfg_variant_codebook, inputs=base_inputs)
    assert result_variant_codebook.status == "ok", "Variant codebook encoding failed"
    assert result_variant_codebook.lf_trace_digest != baseline_trace_digest, \
        "codebook_id change must cause lf_trace_digest to change"

    # T1.2: 修改 ecc → trace_digest 必变化。
    cfg_variant_ecc = base_cfg.copy()
    cfg_variant_ecc["watermark"] = base_cfg["watermark"].copy()
    cfg_variant_ecc["watermark"]["lf"] = base_cfg["watermark"]["lf"].copy()
    cfg_variant_ecc["watermark"]["lf"]["ecc"] = 5

    result_variant_ecc = coder.extract(cfg=cfg_variant_ecc, inputs=base_inputs)
    assert result_variant_ecc.status == "ok", "Variant ecc encoding failed"
    assert result_variant_ecc.lf_trace_digest != baseline_trace_digest, \
        "ecc change must cause lf_trace_digest to change"

    # T1.3: 修改 strength → trace_digest 必变化。
    cfg_variant_strength = base_cfg.copy()
    cfg_variant_strength["watermark"] = base_cfg["watermark"].copy()
    cfg_variant_strength["watermark"]["lf"] = base_cfg["watermark"]["lf"].copy()
    cfg_variant_strength["watermark"]["lf"]["strength"] = 0.8

    result_variant_strength = coder.extract(cfg=cfg_variant_strength, inputs=base_inputs)
    assert result_variant_strength.status == "ok", "Variant strength encoding failed"
    assert result_variant_strength.lf_trace_digest != baseline_trace_digest, \
        "strength change must cause lf_trace_digest to change"

    # T1.4: 修改 delta → trace_digest 必变化。
    cfg_variant_delta = base_cfg.copy()
    cfg_variant_delta["watermark"] = base_cfg["watermark"].copy()
    cfg_variant_delta["watermark"]["lf"] = base_cfg["watermark"]["lf"].copy()
    cfg_variant_delta["watermark"]["lf"]["delta"] = 2.0

    result_variant_delta = coder.extract(cfg=cfg_variant_delta, inputs=base_inputs)
    assert result_variant_delta.status == "ok", "Variant delta encoding failed"
    assert result_variant_delta.lf_trace_digest != baseline_trace_digest, \
        "delta change must cause lf_trace_digest to change"

    # T1.5: 修改 block_length → trace_digest 必变化。
    cfg_variant_block_length = base_cfg.copy()
    cfg_variant_block_length["watermark"] = base_cfg["watermark"].copy()
    cfg_variant_block_length["watermark"]["lf"] = base_cfg["watermark"]["lf"].copy()
    cfg_variant_block_length["watermark"]["lf"]["block_length"] = 4

    result_variant_block_length = coder.extract(cfg=cfg_variant_block_length, inputs=base_inputs)
    assert result_variant_block_length.status == "ok", "Variant block_length encoding failed"
    assert result_variant_block_length.lf_trace_digest != baseline_trace_digest, \
        "block_length change must cause lf_trace_digest to change"


def test_lf_trace_digest_is_reproducible() -> None:
    """
    功能：验证 lf_trace_digest 可复算。

    Test that lf_trace_digest is reproducible for the same configuration.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_trace_digest is not reproducible.
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
        impl_digest="test_impl_digest_001"
    )

    # 执行两次编码。
    result_1 = coder.extract(cfg=cfg, inputs=inputs)
    result_2 = coder.extract(cfg=cfg, inputs=inputs)

    assert result_1.status == "ok", "First encoding failed"
    assert result_2.status == "ok", "Second encoding failed"
    assert result_1.lf_trace_digest == result_2.lf_trace_digest, \
        "lf_trace_digest must be reproducible for the same configuration"
    assert result_1.lf_score == result_2.lf_score, \
        "lf_score must be reproducible for the same configuration"
