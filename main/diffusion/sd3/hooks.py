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


class SelfAttentionCaptureHook:
    """
    功能：捕获 SD3 self-attention 的 Q/K 投影并重建 attention maps。

    Capture Q/K projections from SD3 attention blocks and reconstruct
    self-attention maps after inference.

    Args:
        max_layers: Maximum hooked self-attention layers.
        sample_steps: Optional sampled step indices.
    """

    def __init__(self, max_layers: Optional[int] = 8, sample_steps: Optional[set[int]] = None) -> None:
        self._captures: Dict[tuple[str, int], Dict[str, Any]] = {}
        self._hook_handles: List[Any] = []
        self._step_index = 0
        self._max_layers = max_layers
        self._sample_steps = sample_steps

    def _make_hook_fn(self, module_name: str, projection: str) -> Callable:
        def hook_fn(_module: Any, _input: Any, output: Any) -> None:
            if self._sample_steps is not None and self._step_index not in self._sample_steps:
                return
            if not hasattr(output, "detach"):
                return
            key = (module_name, self._step_index)
            if key not in self._captures:
                self._captures[key] = {}
            self._captures[key][projection] = output.detach().cpu()

        return hook_fn

    def advance_step(self) -> None:
        self._step_index += 1

    def register(self, pipeline: Any) -> None:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            return

        hooked_layers = 0
        for module_name, module in transformer.named_modules():
            if ".attn2." in module_name:
                continue
            if self._max_layers is not None and hooked_layers >= self._max_layers:
                break

            if module_name.endswith(".to_q"):
                parent_name = module_name.rsplit(".", 1)[0]
                self._hook_handles.append(module.register_forward_hook(self._make_hook_fn(parent_name, "q")))
            elif module_name.endswith(".to_k"):
                parent_name = module_name.rsplit(".", 1)[0]
                self._hook_handles.append(module.register_forward_hook(self._make_hook_fn(parent_name, "k")))
                hooked_layers += 1

    def collect(self) -> Optional[Any]:
        import torch

        attention_tensors: List[Any] = []
        for key in sorted(self._captures.keys()):
            pair = self._captures[key]
            q_tensor = pair.get("q")
            k_tensor = pair.get("k")
            if q_tensor is None or k_tensor is None:
                continue

            if q_tensor.ndim == 3:
                q_view = q_tensor
                k_view = k_tensor
            elif q_tensor.ndim == 4:
                q_view = q_tensor.reshape(-1, q_tensor.shape[-2], q_tensor.shape[-1])
                k_view = k_tensor.reshape(-1, k_tensor.shape[-2], k_tensor.shape[-1])
            else:
                continue

            d_k = q_view.shape[-1]
            attention_weights = torch.softmax((q_view @ k_view.transpose(-2, -1)) / (d_k ** 0.5), dim=-1)
            attention_tensors.append(attention_weights)

        self._captures.clear()
        self._step_index = 0

        if not attention_tensors:
            return None

        try:
            return torch.stack(attention_tensors, dim=0)
        except Exception:
            return None

    def remove(self) -> None:
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                continue
        self._hook_handles = []


def register_attention_hooks(pipeline: Any, cfg: Dict[str, Any]) -> SelfAttentionCaptureHook:
    """
    功能：注册 SD3 self-attention 捕获 hooks。

    Register self-attention capture hooks on runtime pipeline.

    Args:
        pipeline: SD3 pipeline object.
        cfg: Configuration mapping.

    Returns:
        SelfAttentionCaptureHook instance.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
    geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
    max_layers = geometry_cfg.get("attention_capture_max_layers", 8)
    if not isinstance(max_layers, int) or max_layers <= 0:
        max_layers = 8

    hook = SelfAttentionCaptureHook(max_layers=max_layers, sample_steps=None)
    hook.register(pipeline)
    return hook


def remove_attention_hooks(hook_handle: Optional[SelfAttentionCaptureHook]) -> None:
    """
    功能：移除 attention capture hooks。

    Remove attention capture hooks safely.

    Args:
        hook_handle: Hook handle instance.

    Returns:
        None.
    """
    if hook_handle is None:
        return
    hook_handle.remove()