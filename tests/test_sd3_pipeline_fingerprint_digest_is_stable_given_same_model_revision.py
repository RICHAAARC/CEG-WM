"""
File purpose: 测试 SD3 pipeline fingerprint digest 在相同模型配置下的稳定性。
Module type: Core innovation module

Test that SD3 pipeline fingerprint digest is stable and reproducible
for the same model_id+revision+config.
"""

import pytest
from unittest.mock import MagicMock
from main.diffusion.sd3 import pipeline_inspector
from main.core import digests


def test_sd3_pipeline_fingerprint_digest_is_stable_given_same_model_revision():
    """
    功能：测试相同模型版本下 pipeline fingerprint digest 的稳定性。

    Test that pipeline_fingerprint_digest is deterministic for the same
    pipeline structure (same transformer/scheduler/VAE configuration).

    Args:
        None.

    Returns:
        None.
    """
    # 构造 mock pipeline 对象（不依赖外网下载）
    mock_pipeline = _build_mock_sd3_pipeline()
    
    # 第一次调用 inspect_sd3_pipeline
    fingerprint_1, digest_1 = pipeline_inspector.inspect_sd3_pipeline(
        pipeline_obj=mock_pipeline,
        cfg={"model_id": "stabilityai/stable-diffusion-3-medium"}
    )
    
    # 第二次调用 inspect_sd3_pipeline（相同 pipeline 对象）
    fingerprint_2, digest_2 = pipeline_inspector.inspect_sd3_pipeline(
        pipeline_obj=mock_pipeline,
        cfg={"model_id": "stabilityai/stable-diffusion-3-medium"}
    )
    
    # 断言：digest 必须一致（可复算性）
    assert digest_1 == digest_2, f"Expected same digest, got {digest_1} vs {digest_2}"
    
    # 断言：fingerprint 结构必须一致
    assert fingerprint_1 == fingerprint_2, \
        f"Expected same fingerprint, got {fingerprint_1} vs {fingerprint_2}"
    
    # 断言：digest 是有效的 sha256（64 位小写十六进制）
    assert isinstance(digest_1, str), "digest must be str"
    assert len(digest_1) == 64, f"Expected 64-char digest, got {len(digest_1)}"
    assert digest_1.islower(), "digest must be lowercase"
    assert all(c in "0123456789abcdef" for c in digest_1), "digest must be hex"
    
    # 断言：digest 与 canonical_sha256(fingerprint) 一致
    expected_digest = digests.canonical_sha256(fingerprint_1)
    assert digest_1 == expected_digest, \
        f"Expected digest to match canonical_sha256(fingerprint), got {digest_1} vs {expected_digest}"


def test_sd3_pipeline_fingerprint_structure_change_updates_digest():
    """
    功能：测试 pipeline 结构变化导致 digest 变化。

    Test that changing the pipeline structure (e.g., num_blocks or scheduler)
    results in a different digest.

    Args:
        None.

    Returns:
        None.
    """
    # 构造第一个 mock pipeline（24 blocks）
    mock_pipeline_1 = _build_mock_sd3_pipeline(num_blocks=24)
    
    # 构造第二个 mock pipeline（28 blocks）
    mock_pipeline_2 = _build_mock_sd3_pipeline(num_blocks=28)
    
    # 调用 inspect_sd3_pipeline
    fingerprint_1, digest_1 = pipeline_inspector.inspect_sd3_pipeline(
        pipeline_obj=mock_pipeline_1,
        cfg={}
    )
    
    fingerprint_2, digest_2 = pipeline_inspector.inspect_sd3_pipeline(
        pipeline_obj=mock_pipeline_2,
        cfg={}
    )
    
    # 断言：fingerprint 必须不同
    assert fingerprint_1 != fingerprint_2, \
        f"Expected different fingerprints, got {fingerprint_1} vs {fingerprint_2}"
    
    # 断言：digest 必须不同
    assert digest_1 != digest_2, \
        f"Expected different digests, got {digest_1} vs {digest_2}"


def test_sd3_pipeline_fingerprint_handles_absent_modules():
    """
    功能：测试 pipeline 缺失模块时 fingerprint 的处理。

    Test that pipeline_inspector handles missing modules gracefully
    and produces <absent> markers instead of errors.

    Args:
        None.

    Returns:
        None.
    """
    # 构造 mock pipeline：transformer 为 None
    mock_pipeline = MagicMock()
    mock_pipeline.transformer = None  # transformer 缺失
    mock_pipeline.scheduler = MagicMock()
    mock_pipeline.scheduler.__class__.__name__ = "FlowMatchEulerDiscreteScheduler"
    mock_pipeline.scheduler.config = MagicMock()
    mock_pipeline.scheduler.config.num_train_timesteps = 1000
    mock_pipeline.scheduler.config.beta_schedule = "scaled_linear"
    mock_pipeline.scheduler.config.prediction_type = "epsilon"
    mock_pipeline.vae = MagicMock()
    mock_pipeline.vae.config = MagicMock()
    mock_pipeline.vae.config.latent_channels = 16
    mock_pipeline.vae.config.in_channels = 3
    mock_pipeline.vae.config.out_channels = 3
    mock_pipeline.text_encoder = None
    
    # 调用 inspect_sd3_pipeline
    fingerprint, digest = pipeline_inspector.inspect_sd3_pipeline(
        pipeline_obj=mock_pipeline,
        cfg={}
    )
    
    # 断言：transformer_num_blocks 必须为 <absent>（不是 None，不是抛异常）
    assert fingerprint["transformer_num_blocks"] == "<absent>", \
        f"Expected transformer_num_blocks='<absent>', got {fingerprint['transformer_num_blocks']}"
    
    # 断言：digest 仍然有效（不是 None）
    assert isinstance(digest, str), "digest must be str even when modules are absent"
    assert len(digest) == 64, f"Expected 64-char digest, got {len(digest)}"


def _build_mock_sd3_pipeline(num_blocks: int = 24) -> MagicMock:
    """
    功能：构造最小 mock SD3 pipeline 对象。

    Build minimal mock SD3 pipeline for testing without network dependencies.

    Args:
        num_blocks: Transformer block count.

    Returns:
        Mock pipeline object.
    """
    mock_pipeline = MagicMock()
    
    # Mock transformer config: 必须设置所有可能被读取的属性为基本类型
    mock_transformer_config = MagicMock()
    mock_transformer_config.num_blocks = num_blocks
    mock_transformer_config.attention_head_dim = 64
    mock_transformer_config.num_attention_heads = 24
    mock_transformer_config.in_channels = 16
    mock_transformer_config.out_channels = 16
    mock_transformer_config.sample_size = 128  # 必须设置为 int，不能是 MagicMock
    mock_transformer_config.patch_size = 2  # 必须设置为 int
    
    # Mock transformer
    mock_transformer = MagicMock()
    mock_transformer.config = mock_transformer_config
    mock_pipeline.transformer = mock_transformer
    
    # Mock scheduler config: 必须设置所有可能被读取的属性为基本类型
    mock_scheduler_config = MagicMock()
    mock_scheduler_config.num_train_timesteps = 1000
    mock_scheduler_config.beta_schedule = "scaled_linear"
    mock_scheduler_config.prediction_type = "epsilon"
    
    # Mock scheduler
    mock_scheduler = MagicMock()
    mock_scheduler.__class__.__name__ = "FlowMatchEulerDiscreteScheduler"
    mock_scheduler.config = mock_scheduler_config
    mock_pipeline.scheduler = mock_scheduler
    
    # Mock VAE config: 必须设置所有可能被读取的属性为基本类型
    mock_vae_config = MagicMock()
    mock_vae_config.latent_channels = 16
    mock_vae_config.in_channels = 3
    mock_vae_config.out_channels = 3
    
    # Mock VAE
    mock_vae = MagicMock()
    mock_vae.config = mock_vae_config
    mock_pipeline.vae = mock_vae
    
    # Mock text encoders
    mock_text_encoder = MagicMock()
    mock_text_encoder.__class__.__name__ = "CLIPTextModel"
    mock_pipeline.text_encoder = mock_text_encoder
    
    return mock_pipeline

