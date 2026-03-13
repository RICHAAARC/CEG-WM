"""
推理与证据聚合模块

功能說明：
- 執行 SD3 推理，集成 LatentModifier hook。
- 聚合 step-level 證據成最終的 trace_digest。
- 生成完整的 LF/HF 應邊證據。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from main.watermarking.content_chain import channel_lf
from main.watermarking.content_chain import channel_hf
from main.watermarking.content_chain.latent_modifier import (
    LATENT_MODIFIER_ID,
    LATENT_MODIFIER_VERSION,
    LatentModifier,
)
from main.diffusion.sd3 import hooks
from main.core import digests


def run_inference_with_latent_modifier(
    cfg: Dict[str, Any],
    pipeline_obj: Any,
    plan: Optional[Dict[str, Any]],
    seed: Optional[int] = None,
    device: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    功能：执行推理并收集 LF/HF 证据。
    
    Run SD3 inference with LatentModifier hook integration.
    Collects step-level evidence for later aggregation.
    
    Args:
        cfg: Configuration mapping.
        pipeline_obj: Diffusers pipeline object.
        plan: Subspace plan from SubspacePlanner.
        seed: Random seed for inference.
        device: Device string.
    
    Returns:
        Tuple of (inference_output, evidence_payload).
        evidence_payload contains step_evidence_list and metrics.
    
    Raises:
        TypeError: If inputs types invalid.
        ValueError: If inference failed.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    if pipeline_obj is None:
        return None, {
            "status": "failed",
            "failure_reason": "pipeline_obj_is_none",
            "step_evidence_list": []
        }
    
    # 创建 LatentModifier 实例。
    modifier = LatentModifier(
        impl_id=LATENT_MODIFIER_ID,
        impl_version=LATENT_MODIFIER_VERSION
    )
    
    # 定义 callback。
    callback = hooks.define_on_step_end_callback(
        modifier=modifier,
        plan=plan,
        cfg=cfg
    )
    
    # 准备推理参数。
    infer_kwargs = {
        "prompt": cfg.get("inference_prompt", "a photo of a dog"),
        "num_inference_steps": cfg.get("inference_num_steps", 10),
        "guidance_scale": cfg.get("inference_guidance_scale", 7.0),
        "height": cfg.get("inference_height", 512),
        "width": cfg.get("inference_width", 512),
        "callback_on_step_end": callback
    }
    
    if seed is not None:
        import torch
        generator = torch.Generator(device=device if device else "cpu")
        generator.manual_seed(seed)
        infer_kwargs["generator"] = generator
    
    # 执行推理。
    try:
        output = pipeline_obj(**infer_kwargs)
    except Exception as e:
        return None, {
            "status": "failed",
            "failure_reason": f"inference_error: {type(e).__name__}",
            "error_detail": str(e),
            "step_evidence_list": hooks.collect_step_evidence(callback)
        }
    
    # 收集所有 step 证据。
    step_evidence_list = hooks.collect_step_evidence(callback)
    
    return output, {
        "status": "ok",
        "step_evidence_list": step_evidence_list,
        "output_shape": (
            len(output.images) if hasattr(output, "images") else 0,
            output.images[0].size if hasattr(output, "images") and output.images else None
        )
    }


def aggregate_step_evidence_into_channel_evidence(
    step_evidence_list: list,
    latents_before: Optional[np.ndarray],
    latents_after: Optional[np.ndarray],
    cfg: Dict[str, Any],
    plan_digest: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    功能：聚合 step 证据成最终的 LF/HF 证据。
    
    Aggregate step-level evidence into final LF/HF evidence with digests.
    
    Args:
        step_evidence_list: Ordered list of step evidence dicts.
        latents_before: Initial latents (optional, for metrics).
        latents_after: Final latents (optional, for metrics).
        cfg: Configuration mapping.
        plan_digest: Optional plan digest for binding.
    
    Returns:
        Tuple of (lf_evidence, hf_evidence).
    
    Raises:
        TypeError: If inputs types invalid.
    """
    if not isinstance(step_evidence_list, list):
        raise TypeError("step_evidence_list must be list")
    
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    # 从 step_evidence_list 中提取 LF/HF 子证据。
    lf_trace_components = []
    hf_trace_components = []
    
    for step_ev in step_evidence_list:
        if not isinstance(step_ev, dict):
            continue
        
        lf_ev = step_ev.get("lf_evidence")
        if isinstance(lf_ev, dict) and lf_ev.get("status") == "ok":
            lf_trace_components.append({
                "step_index": step_ev.get("step_index"),
                "lf_status": lf_ev.get("status"),
                "lf_metrics": lf_ev.get("encoding_evidence", {})
            })
        
        hf_ev = step_ev.get("hf_evidence")
        if isinstance(hf_ev, dict) and hf_ev.get("status") == "ok":
            hf_trace_components.append({
                "step_index": step_ev.get("step_index"),
                "hf_status": hf_ev.get("status"),
                "hf_metrics": hf_ev.get("constraint_evidence", {})
            })
    
    # 生成 LF 证据（使用 channel_lf 模块）。
    if latents_before is not None and latents_after is not None:
        latents_before_np = np.asarray(latents_before, dtype=np.float32)
        latents_after_np = np.asarray(latents_after, dtype=np.float32)
    else:
        latents_before_np = np.zeros((1,), dtype=np.float32)
        latents_after_np = np.zeros((1,), dtype=np.float32)
    
    lf_evidence = channel_lf.build_lf_embed_evidence(
        latents_before=latents_before_np,
        latents_after=latents_after_np,
        trace_components=lf_trace_components,
        encoding_evidence={},
        cfg=cfg,
        plan_digest=plan_digest
    )
    
    # 生成 HF 证据（使用 channel_hf 模块）。
    hf_evidence = channel_hf.build_hf_embed_evidence(
        latents_before=latents_before_np,
        latents_after=latents_after_np,
        trace_components=hf_trace_components,
        constraint_evidence={},
        cfg=cfg,
        plan_digest=plan_digest
    )
    
    return lf_evidence, hf_evidence
