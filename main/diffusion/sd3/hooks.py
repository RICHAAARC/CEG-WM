"""
SD3 推理循环 Hook 机制

功能说明：
- 定义 callback_on_step_end hook，用于在推理每步后修改 latents。
- 与 diffusers pipeline 的 callback 协议兼容。
- 集成 LatentModifier，产出可复算证据。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Callable

from main.watermarking.content_chain.latent_modifier import LatentModifier


def define_on_step_end_callback(
    modifier: LatentModifier,
    plan: Optional[Dict[str, Any]],
    cfg: Dict[str, Any]
) -> Callable:
    """
    功能：定义 callback_on_step_end 回调函数。
    
    Define on_step_end callback for diffusers pipeline integration.
    Returns a callback that modifies latents during inference.
    
    Args:
        modifier: LatentModifier instance for applying watermarks.
        plan: Subspace plan mapping from SubspacePlanner.
        cfg: Configuration mapping with channel parameters.
    
    Returns:
        Callback function with signature (pipe, step_index, timestep, callback_kwargs) -> callback_kwargs.
    
    Raises:
        TypeError: If inputs types are invalid.
        ValueError: If modifier or cfg invalid.
    """
    if not isinstance(modifier, LatentModifier):
        raise TypeError("modifier must be LatentModifier instance")
    
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    # 收集 step 级证据到列表（用于终总汇总）。
    step_evidence_list = []
    
    def callback(pipe: Any, step_index: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：在推理每步后修改 latents。
        
        Modify latents after each diffusion step.
        Called by diffusers pipeline after each denoising step.
        
        Args:
            pipe: Diffusers pipeline object.
            step_index: Current step index (0-based).
            timestep: Current timestep value.
            callback_kwargs: Callback argument dict containing latents, etc.
        
        Returns:
            Modified callback_kwargs with updated latents.
        
        Raises:
            None (all errors logged, latents passed through unchanged).
        """
        try:
            # 提取 latents 从 callback_kwargs。
            if "latents" not in callback_kwargs:
                # 无 latents 输入，直接返回。
                return callback_kwargs
            
            latents_input = callback_kwargs["latents"]
            
            # 应用修改。
            latents_modified, step_evidence = modifier.apply_latent_update(
                latents=latents_input,
                plan=plan,
                cfg=cfg,
                step_index=step_index,
                key=None  # 内部衍生
            )
            
            # 更新 callback_kwargs 中的 latents。
            callback_kwargs["latents"] = latents_modified
            
            # 记录 step 证据。
            step_evidence_list.append(step_evidence)
            
        except Exception as e:
            # 错误处理：记录但不抛异常，确保推理不中断。
            import traceback
            traceback.print_exc()
            # latents 保持不变，推理继续。
            pass
        
        return callback_kwargs
    
    # 挂载证据列表到 callback，便于外部收集。
    callback._step_evidence_list = step_evidence_list
    callback._modifier = modifier
    callback._plan = plan
    callback._cfg = cfg
    
    return callback


def collect_step_evidence(callback: Callable) -> list:
    """
    功能：从 callback 中收集所有 step 级证据。
    
    Retrieve accumulated step evidence from callback after inference.
    
    Args:
        callback: Callback function returned by define_on_step_end_callback.
    
    Returns:
        List of step evidence dicts.
    
    Raises:
        AttributeError: If callback does not have _step_evidence_list.
    """
    if not hasattr(callback, "_step_evidence_list"):
        raise AttributeError("callback must have _step_evidence_list attribute")
    
    return callback._step_evidence_list
