"""
文件目的：验证 infer_runtime 的 CUDA 显存峰值插桩在 absent 与伪造 cuda 场景下稳定产出。
Module type: General module
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest

from main.diffusion.sd3 import infer_runtime as infer_runtime_module


class _PipelineStub:
    """
    功能：为 infer_runtime 测试提供最小 pipeline 桩。

    Minimal pipeline stub used by infer_runtime observability tests.
    """

    def __init__(self, device_label: str) -> None:
        self.device = device_label
        self.to_calls: list[str] = []

    def to(self, device: Any) -> "_PipelineStub":
        normalized_device = str(device)
        self.to_calls.append(normalized_device)
        self.device = normalized_device
        return self

    def __call__(self, **kwargs: Any) -> Any:
        _ = kwargs
        return SimpleNamespace(images=[object()])


def _build_cfg(device_label: str) -> Dict[str, Any]:
    """
    功能：构造 infer_runtime 最小测试配置。

    Build the minimal config used by infer_runtime memory-profile tests.

    Args:
        device_label: Requested runtime device label.

    Returns:
        Minimal configuration mapping.
    """
    return {
        "inference_enabled": True,
        "inference_prompt": "prompt",
        "inference_num_steps": 4,
        "inference_guidance_scale": 7.0,
        "inference_height": 64,
        "inference_width": 64,
        "device": device_label,
    }


def test_run_sd3_inference_cpu_path_returns_absent_cuda_memory_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：非 cuda 路径必须返回结构化 absent 显存 profile。

    Verify the CPU path preserves a structured absent CUDA memory profile.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch unavailable")

    def _fake_tap_from_pipeline(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        return {
            "output": SimpleNamespace(images=[object()]),
            "trajectory_evidence": {"status": "ok", "trajectory_metrics": {"steps": []}},
            "tap_status": "ok",
            "trajectory_cache_capture_meta": None,
        }

    monkeypatch.setattr(torch.cuda, "is_available", cast(Any, lambda: False))
    monkeypatch.setattr(infer_runtime_module.trajectory_tap, "tap_from_pipeline", _fake_tap_from_pipeline)

    result = infer_runtime_module.run_sd3_inference(
        _build_cfg("cpu"),
        _PipelineStub("cpu"),
        "cpu",
        None,
        runtime_phase_label="preview_generation",
    )

    runtime_meta = cast(Dict[str, Any], result["inference_runtime_meta"])
    cuda_memory_profile = cast(Dict[str, Any], runtime_meta["cuda_memory_profile"])

    assert result["inference_status"] == "ok"
    assert cuda_memory_profile["status"] == "absent"
    assert cuda_memory_profile["reason"] == "cuda_not_available"
    assert cuda_memory_profile["phase_label"] == "preview_generation"
    assert cuda_memory_profile["sample_scope"] == "single_worker_process_local"
    assert cuda_memory_profile["device"] == "cpu"


def test_run_sd3_inference_cuda_path_collects_peak_memory_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：伪造 cuda 路径时必须写出 before after peak 与 delta 显存字段。

    Verify the CUDA sampling path records before, after, peak, and delta fields
    without requiring a real GPU.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch unavailable")

    allocated_values = iter([128, 176])
    reserved_values = iter([256, 320])
    mem_info_values = iter([(2048, 4096), (1984, 4096)])
    reset_calls: list[str] = []

    def _fake_tap_from_pipeline(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        return {
            "output": SimpleNamespace(images=[object()]),
            "trajectory_evidence": {"status": "ok", "trajectory_metrics": {"steps": []}},
            "tap_status": "ok",
            "trajectory_cache_capture_meta": None,
        }

    monkeypatch.setattr(torch.cuda, "is_available", cast(Any, lambda: True))
    monkeypatch.setattr(torch.cuda, "memory_allocated", cast(Any, lambda *args, **kwargs: next(allocated_values)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", cast(Any, lambda *args, **kwargs: next(reserved_values)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", cast(Any, lambda *args, **kwargs: 224))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", cast(Any, lambda *args, **kwargs: 352))
    monkeypatch.setattr(torch.cuda, "mem_get_info", cast(Any, lambda *args, **kwargs: next(mem_info_values)))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        cast(Any, lambda device: reset_calls.append(str(device))),
    )
    monkeypatch.setattr(infer_runtime_module.trajectory_tap, "tap_from_pipeline", _fake_tap_from_pipeline)

    result = infer_runtime_module.run_sd3_inference(
        _build_cfg("cuda"),
        _PipelineStub("cuda:0"),
        "cuda",
        None,
        runtime_phase_label="embed_watermarked_inference",
    )

    runtime_meta = cast(Dict[str, Any], result["inference_runtime_meta"])
    cuda_memory_profile = cast(Dict[str, Any], runtime_meta["cuda_memory_profile"])

    assert result["inference_status"] == "ok"
    assert reset_calls == ["cuda"]
    assert cuda_memory_profile["status"] == "ok"
    assert cuda_memory_profile["phase_label"] == "embed_watermarked_inference"
    assert cuda_memory_profile["sample_scope"] == "single_worker_process_local"
    assert cuda_memory_profile["device"] == "cuda"
    assert cuda_memory_profile["memory_allocated_before_bytes"] == 128
    assert cuda_memory_profile["memory_reserved_before_bytes"] == 256
    assert cuda_memory_profile["memory_allocated_after_bytes"] == 176
    assert cuda_memory_profile["memory_reserved_after_bytes"] == 320
    assert cuda_memory_profile["peak_memory_allocated_bytes"] == 224
    assert cuda_memory_profile["peak_memory_reserved_bytes"] == 352
    assert cuda_memory_profile["allocated_delta_from_before_bytes"] == 48
    assert cuda_memory_profile["reserved_delta_from_before_bytes"] == 64
    assert cuda_memory_profile["mem_get_info_free_before_bytes"] == 2048
    assert cuda_memory_profile["mem_get_info_free_after_bytes"] == 1984
    assert cuda_memory_profile["mem_get_info_total_bytes"] == 4096
    assert cuda_memory_profile["peak_memory_allocated_mib"] == round(224 / (1024.0 * 1024.0), 6)
    assert cuda_memory_profile["peak_memory_reserved_mib"] == round(352 / (1024.0 * 1024.0), 6)