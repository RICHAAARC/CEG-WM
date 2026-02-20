"""
SD3 Diffusion Tracer

功能：
- 在推理 loop 中采样并输出最小轨迹摘要（trajectory digest）。
- 每 K 步记录 timestep/sigma/latent_norm 等关键统计信号。
- 支持可开关（enable_*），关闭时输出 absent 语义。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from main.core import digests


IMPL_ID = "sd3_diffusion_tracer_v1"
IMPL_VERSION = "v1"


def init_tracer(
    cfg: Dict[str, Any],
    enable_tracing: bool = True
) -> Dict[str, Any]:
    """
    功能：初始化 tracer 状态。

    Initialize diffusion tracer state.

    Args:
        cfg: Configuration mapping.
        enable_tracing: Whether to enable tracing.

    Returns:
        Tracer state dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    sample_stride = cfg.get("trajectory_sample_stride", 5)
    if not isinstance(sample_stride, int) or sample_stride < 1:
        sample_stride = 5

    num_inference_steps = cfg.get("num_inference_steps") or cfg.get("inference_num_steps", 10)
    if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
        num_inference_steps = 10

    tracer_state = {
        "enabled": enable_tracing,
        "sample_stride": sample_stride,
        "num_inference_steps": num_inference_steps,
        "trajectory_samples": [],
        "trajectory_spec": {
            "sample_stride": sample_stride,
            "num_inference_steps": num_inference_steps,
            "enabled": enable_tracing
        }
    }

    return tracer_state


def record_trajectory_sample(
    tracer_state: Dict[str, Any],
    step: int,
    timestep: Optional[float],
    latents: Optional[torch.Tensor],
    scheduler: Optional[Any] = None
) -> None:
    """
    功能：记录单个轨迹采样点。

    Record a single trajectory sample point.

    Args:
        tracer_state: Tracer state dict from init_tracer.
        step: Current diffusion step index.
        timestep: Current timestep value.
        latents: Current latent tensor.
        scheduler: Scheduler object for extracting sigma.

    Returns:
        None (modifies tracer_state in-place).
    """
    if not isinstance(tracer_state, dict):
        return

    if not tracer_state.get("enabled", False):
        return

    sample_stride = tracer_state.get("sample_stride", 5)
    if step % sample_stride != 0:
        return

    sample = {
        "step": step
    }

    if timestep is not None:
        if isinstance(timestep, torch.Tensor):
            sample["timestep"] = float(timestep.item())
        else:
            sample["timestep"] = float(timestep)

    # 提取 sigma（若 scheduler 支持）。
    if scheduler is not None and hasattr(scheduler, "sigmas"):
        sigmas = scheduler.sigmas
        if isinstance(sigmas, torch.Tensor) and step < len(sigmas):
            sample["sigma"] = float(sigmas[step].item())

    # 提取 latent norm。
    if latents is not None and isinstance(latents, torch.Tensor):
        with torch.no_grad():
            latent_norm = torch.norm(latents.flatten(), p=2).item()
            sample["latent_norm"] = latent_norm

            # 额外统计：均值、标准差（可选）。
            latent_mean = latents.mean().item()
            latent_std = latents.std().item()
            sample["latent_mean"] = latent_mean
            sample["latent_std"] = latent_std

    tracer_state["trajectory_samples"].append(sample)


def finalize_trajectory(
    tracer_state: Dict[str, Any]
) -> Tuple[Dict[str, Any], str, str]:
    """
    功能：完成轨迹采样并生成摘要。

    Finalize trajectory sampling and generate digests.

    Args:
        tracer_state: Tracer state dict.

    Returns:
        Tuple of (trajectory_evidence dict, trajectory_spec_digest str, trajectory_digest str).
    """
    if not isinstance(tracer_state, dict):
        return _absent_trajectory()

    if not tracer_state.get("enabled", False):
        return _absent_trajectory()

    trajectory_samples = tracer_state.get("trajectory_samples", [])
    trajectory_spec = tracer_state.get("trajectory_spec", {})

    # 生成 trajectory_spec_digest。
    trajectory_spec_digest = digests.canonical_sha256(trajectory_spec)

    # 生成 trajectory_digest（顺序敏感）。
    trajectory_digest = digests.canonical_sha256({
        "trajectory_samples": trajectory_samples,
        "sample_count": len(trajectory_samples)
    })

    # 计算轨迹统计摘要。
    trajectory_metrics = _compute_trajectory_metrics(trajectory_samples)

    trajectory_evidence = {
        "status": "ok",
        "trajectory_spec": trajectory_spec,
        "trajectory_spec_digest": trajectory_spec_digest,
        "trajectory_digest": trajectory_digest,
        "trajectory_metrics": trajectory_metrics,
        "trajectory_tap_version": IMPL_VERSION
    }

    return trajectory_evidence, trajectory_spec_digest, trajectory_digest


def _compute_trajectory_metrics(
    trajectory_samples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    功能：计算轨迹统计摘要。

    Compute trajectory statistics summary.

    Args:
        trajectory_samples: List of trajectory sample dicts.

    Returns:
        Trajectory metrics dict.
    """
    if not trajectory_samples:
        return {
            "sample_count": 0,
            "timestep_samples": [],
            "sigma_samples": [],
            "latent_norm_samples": []
        }

    timestep_samples = [s.get("timestep") for s in trajectory_samples if s.get("timestep") is not None]
    sigma_samples = [s.get("sigma") for s in trajectory_samples if s.get("sigma") is not None]
    latent_norm_samples = [s.get("latent_norm") for s in trajectory_samples if s.get("latent_norm") is not None]

    # 计算分位点（不存大 tensor，只存统计摘要）。
    metrics = {
        "sample_count": len(trajectory_samples),
        "timestep_samples": timestep_samples if len(timestep_samples) <= 20 else _quantiles(timestep_samples),
        "sigma_samples": sigma_samples if len(sigma_samples) <= 20 else _quantiles(sigma_samples),
        "latent_norm_samples": latent_norm_samples if len(latent_norm_samples) <= 20 else _quantiles(latent_norm_samples)
    }

    # 额外统计：latent_norm 的均值与标准差。
    if latent_norm_samples:
        metrics["latent_norm_mean"] = float(np.mean(latent_norm_samples))
        metrics["latent_norm_std"] = float(np.std(latent_norm_samples))

    return metrics


def _quantiles(values: List[float], quantiles: Optional[List[float]] = None) -> Dict[str, float]:
    """
    功能：计算分位点摘要（压缩统计）。

    Compute quantiles summary.

    Args:
        values: List of float values.
        quantiles: List of quantile levels (default: [0, 0.25, 0.5, 0.75, 1.0]).

    Returns:
        Dict mapping quantile level to value.
    """
    if quantiles is None:
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]

    if not values:
        return {f"q{int(q*100)}": None for q in quantiles}

    values_array = np.array(values)
    return {f"q{int(q*100)}": float(np.quantile(values_array, q)) for q in quantiles}


def _absent_trajectory() -> Tuple[Dict[str, Any], str, str]:
    """
    功能：返回 absent 语义的轨迹证据。

    Return absent trajectory evidence.

    Returns:
        Tuple of (trajectory_evidence dict, absent digest, absent digest).
    """
    trajectory_evidence = {
        "status": "absent",
        "trajectory_absent_reason": "tracing_disabled",
        "trajectory_spec": {},
        "trajectory_spec_digest": "<absent>",
        "trajectory_digest": "<absent>",
        "trajectory_metrics": {},
        "trajectory_tap_version": IMPL_VERSION
    }
    return trajectory_evidence, "<absent>", "<absent>"


def get_tracer_impl_identity() -> Dict[str, str]:
    """
    功能：返回 tracer 的实现身份标识。

    Get tracer implementation identity.

    Returns:
        Dict with impl_id, impl_version, impl_digest.
    """
    impl_digest = digests.canonical_sha256({
        "impl_id": IMPL_ID,
        "impl_version": IMPL_VERSION,
        "source_module": "main.diffusion.sd3.diffusion_tracer"
    })
    return {
        "impl_id": IMPL_ID,
        "impl_version": IMPL_VERSION,
        "impl_digest": impl_digest
    }
