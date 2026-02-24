"""
File purpose: 验证 embed orchestrator 能够接受 sync_runtime_context 并产出 sync_result
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict
import pytest
import numpy as np

from main.watermarking.geometry_chain.sync import SyncRuntimeContext


class MockPipeline:
    """Mock SD3 pipeline for testing."""
    def __init__(self):
        self.transformer = MockTransformer()
        self.type_name = "StableDiffusion3Pipeline"


class MockTransformer:
    """Mock transformer module."""
    def __init__(self):
        self.config = MockConfig()


class MockConfig:
    """Mock transformer config."""
    def __init__(self):
        self.patch_size = 1


def test_sync_runtime_context_construction_and_validation() -> None:
    """
    功能：SyncRuntimeContext 必须正确接受和验证输入（latents, pipeline, trajectory_evidence）。

    SyncRuntimeContext must correctly accept sync inputs for transmission to sync module.

    Args:
        None

    Returns:
        None

    Raises:
        TypeError: If validation fails.
    """
    # (1) 有效的 SyncRuntimeContext（all fields）
    mock_pipeline = MockPipeline()
    test_latents = np.ones((1, 16, 64, 64), dtype=np.float32)
    trajectory = {"step_0": "data"}
    
    context = SyncRuntimeContext(
        pipeline=mock_pipeline,
        latents=test_latents,
        rng=None,
        trajectory_evidence=trajectory
    )
    
    assert context.latents is test_latents
    assert context.pipeline is mock_pipeline
    assert context.rng is None
    assert context.trajectory_evidence is trajectory
    
    # (2) SyncRuntimeContext with None latents（有效但可吃不开 sync）
    context_no_latents = SyncRuntimeContext(
        pipeline=mock_pipeline,
        latents=None,
        rng=None,
        trajectory_evidence=None
    )
    assert context_no_latents.latents is None
    assert context_no_latents.pipeline is mock_pipeline
    
    # (3) Invalid trajectory_evidence type should raise
    with pytest.raises(TypeError):
        SyncRuntimeContext(
            pipeline=mock_pipeline,
            latents=test_latents,
            rng=None,
            trajectory_evidence="invalid_string"
        )


def test_run_embed_orchestrator_accepts_sync_runtime_context() -> None:
    """
    功能：run_embed_orchestrator 必须接受 sync_runtime_context 参数（signature 检查）。

    Verify that run_embed_orchestrator accepts sync_runtime_context parameter.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If signature check fails.
    """
    # Import after all checks（避免导入问题）
    from main.watermarking.embed import orchestrator
    import inspect
    
    # 检查函数签名是否包含 sync_runtime_context 参数
    sig = inspect.signature(orchestrator.run_embed_orchestrator)
    params = list(sig.parameters.keys())
    
    assert "sync_runtime_context" in params, \
        f"run_embed_orchestrator must have sync_runtime_context parameter. Got: {params}"
    
    # 验证该参数有默认值（optional）
    sync_ctx_param = sig.parameters["sync_runtime_context"]
    assert sync_ctx_param.default is not inspect.Parameter.empty, \
        "sync_runtime_context must be optional with default value"
