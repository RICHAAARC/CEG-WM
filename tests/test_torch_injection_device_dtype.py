"""
File purpose: torch 注入保持 device/dtype 的回归测试。
Module type: General module
"""

from __future__ import annotations

import os
import pytest

if os.environ.get("CEG_WM_ENABLE_TORCH_TESTS", "0") != "1":
    pytest.skip("torch device/dtype test is opt-in (set CEG_WM_ENABLE_TORCH_TESTS=1)", allow_module_level=True)

import torch

from main.watermarking.content_chain.latent_modifier import (
    LATENT_MODIFIER_ID,
    LATENT_MODIFIER_VERSION,
    LatentModifier,
)


def test_torch_injection_keeps_device_and_dtype_on_cuda() -> None:
    """
    功能：CUDA 下注入前后 latents 的 device 与 dtype 必须保持一致。
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    device = torch.device("cuda:0")
    dtype = torch.float16
    latents = torch.randn((1, 4, 8, 8), device=device, dtype=dtype)

    latent_dim = int(latents.numel())
    rank = 8
    lf_projection = torch.zeros((latent_dim, rank), device=device, dtype=torch.float32)
    hf_projection = torch.zeros((latent_dim, rank), device=device, dtype=torch.float32)
    lf_projection[torch.arange(rank, device=device), torch.arange(rank, device=device)] = 1.0
    hf_projection[torch.arange(rank, device=device), torch.arange(rank, device=device)] = 1.0

    plan = {
        "lf_basis": {"projection_matrix": lf_projection},
        "hf_basis": {"hf_projection_matrix": hf_projection},
        "runtime_subspace_binding": {
            "status": "ok",
            "binding_digest": "c" * 64,
        },
    }
    cfg = {
        "lf_enabled": True,
        "hf_enabled": True,
        "lf_strength": 0.5,
        "hf_threshold_percentile": 75.0,
        "watermark_seed": 7,
    }

    modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)
    latents_after, step_evidence = modifier.apply_latent_update(
        latents=latents,
        plan=plan,
        cfg=cfg,
        step_index=0,
        key=None,
    )

    assert isinstance(step_evidence, dict)
    assert latents_after.device == latents.device
    assert latents_after.dtype == latents.dtype
