"""
SD3 推理循环 Hook 机制

功能说明：
- 定义 callback_on_step_end hook，用于在推理每步后修改 latents。
- 与 diffusers pipeline 的 callback 协议兼容。
- 集成 LatentModifier 与 InjectionContext，产出可复算证据。
- 支持多回调组合（采样 + 注入），通过显式 context 传递计划参数。
- 注入与采样分离，独立审计与消融。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from main.watermarking.content_chain.latent_modifier import LatentModifier
from main.diffusion.sd3.callback_composer import (
    InjectionContext,
    LatentInjectionHook,
    compose_step_end_callbacks,
    extract_injection_context_from_cfg,
)


def define_on_step_end_callback(
    modifier: LatentModifier,
    plan: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
    plan_digest: Optional[str] = None
) -> Callable:
    """
    功能：定义 callback_on_step_end 回调函数，集成注入与可选采样。

    Define on_step_end callback for diffusers pipeline integration.
    Creates a LatentInjectionHook and optionally composes with trajectory sampling callback.
    Returns a callback that modifies latents during inference.

    Args:
        modifier: LatentModifier instance for applying watermarks.
        plan: Subspace plan mapping from SubspacePlanner.
        cfg: Configuration mapping with channel parameters.
        plan_digest: SHA256 digest of plan (optional; if provided, validates consistency).

    Returns:
        Callback function with signature (pipe, step_index, timestep, callback_kwargs) -> callback_kwargs.
        The callback has attributes:
            - _injection_hook: LatentInjectionHook instance (provides collect_evidence() method)
            - _step_evidence_list: List of step evidence dicts from injection

    Raises:
        TypeError: If inputs types are invalid.
        ValueError: If modifier, plan, or cfg invalid.
    """
    if not isinstance(modifier, LatentModifier):
        raise TypeError("modifier must be LatentModifier instance")
    
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    if plan is not None and not isinstance(plan, dict):
        raise TypeError("plan must be dict or None")
    
    # 若 plan 为 None，使用空字典作为基线值。
    if plan is None:
        plan = {}
    
    # 若未提供 plan_digest，则生成基线标识。
    if plan_digest is None:
        plan_digest = "baseline_plan_digest"
    
    # 构建 InjectionContext（显式上下文，避免修改 cfg）。
    try:
        injection_context = extract_injection_context_from_cfg(cfg, plan, plan_digest)
    except (TypeError, ValueError) as e:
        # 上下文构建失败时，创建最小化上下文（都禁用）。
        injection_context = InjectionContext(
            plan_digest=plan_digest,
            plan_ref=plan,
            lf_params_digest="",
            hf_params_digest="",
            enable_lf=False,
            enable_hf=False,
            device=cfg.get("device", "cpu"),
            dtype=cfg.get("dtype", "float32")
        )
    
    # 创建 LatentInjectionHook（专注张量修改）。
    injection_hook = LatentInjectionHook(context=injection_context, modifier=modifier)
    
    # injection_hook 本身就是可调用的，可直接返回。
    # 额外挂载证据列表以兼容旧接口。
    injection_hook._step_evidence_list = injection_hook._step_evidence_list
    
    return injection_hook


def collect_step_evidence(callback: Callable) -> List[Dict[str, Any]]:
    """
    功能：从 callback 中收集所有 step 级证据。

    Retrieve accumulated step evidence from callback after inference.
    Handles both LatentInjectionHook and legacy callback formats.

    Args:
        callback: Callback function returned by define_on_step_end_callback.

    Returns:
        List of step evidence dicts.

    Raises:
        AttributeError: If callback does not have evidence collection capability.
        TypeError: If callback type is invalid.
    """
    if not callable(callback):
        raise TypeError("callback must be callable")
    
    # 若 callback 是 LatentInjectionHook 实例，使用其 collect_evidence 方法。
    if isinstance(callback, LatentInjectionHook):
        return callback.collect_evidence()
    
    # 否则尝试访问 _step_evidence_list 属性（兼容旧接口）。
    if hasattr(callback, "_step_evidence_list"):
        return callback._step_evidence_list
    
    raise AttributeError("callback must have evidence collection capability (LatentInjectionHook or _step_evidence_list)")


def define_composed_callback(
    injection_hook: LatentInjectionHook,
    trajectory_callback: Optional[Callable] = None
) -> Callable:
    """
    功能：将注入回调与可选的采样回调组合。

    Compose injection hook with optional trajectory sampling callback.
    Both callbacks share the same callback_kwargs and are called in sequence.

    Args:
        injection_hook: LatentInjectionHook instance (注入回调).
        trajectory_callback: Optional trajectory sampling callback (采样回调).

    Returns:
        Composed callback function.

    Raises:
        TypeError: If injection_hook type is invalid.
    """
    if not isinstance(injection_hook, LatentInjectionHook):
        raise TypeError("injection_hook must be LatentInjectionHook instance")
    
    if trajectory_callback is not None and not callable(trajectory_callback):
        raise TypeError("trajectory_callback must be callable or None")
    
    # 若无采样回调，直接返回注入回调。
    if trajectory_callback is None:
        return injection_hook
    
    # 否则使用 compose_step_end_callbacks 组合两个回调。
    # 注意：注入在前（优先修改张量），采样在后（观察修改后的张量）。
    composed = compose_step_end_callbacks(injection_hook, trajectory_callback)
    
    # 挂载注入回调的证据列表到组合版本，便于外部收集。
    composed._injection_hook = injection_hook
    
    return composed