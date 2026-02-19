"""
SD3 推理回调组合与注入上下文

功能说明：
- 定义 InjectionContext，承载注入参数的轻量只读容器（可序列化摘要形式）。
- 定义 LatentInjectionHook，专注 latent 张量修改的后链回调（与采样解耦）。
- 实现 compose_step_end_callbacks，支持多个回调按序调用，共享同一个 callback_kwargs。
- 通过显式 context 对象（而非线程上下文或 cfg 修改）传递计划参数，确保可审计与可追溯。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionContext:
    """
    功能：注入回调的只读上下文容器。

    Immutable context object for latent injection callbacks.
    All parameters are digest-bound and must match embed-time and detect-time specifications.

    Attributes:
        plan_digest: SHA256 digest of the subspace plan (str, non-empty).
        plan_ref: Reference to plan mapping (Dict, lightweight subset or full plan).
        lf_params_digest: LF channel parameters digest (str, non-empty, required if enable_lf=True).
        hf_params_digest: HF channel parameters digest (str, non-empty, required if enable_hf=True).
        enable_lf: Flag to enable LF channel injection (bool).
        enable_hf: Flag to enable HF channel injection (bool).
        device: Torch device string ('cpu', 'cuda:0', etc.) for execution (str).
        dtype: Torch dtype string ('float32', 'float16', etc.) for computation (str).

    Raises:
        TypeError: If required fields are missing or wrong type.
        ValueError: If digests are empty when corresponding channels are enabled.
    """

    plan_digest: str
    plan_ref: Dict[str, Any]
    lf_params_digest: str
    hf_params_digest: str
    enable_lf: bool
    enable_hf: bool
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        # 验证必填字段。
        if not isinstance(self.plan_digest, str) or not self.plan_digest:
            raise ValueError("plan_digest must be non-empty str")
        
        if not isinstance(self.plan_ref, dict):
            raise TypeError("plan_ref must be dict")
        
        if not isinstance(self.enable_lf, bool):
            raise TypeError("enable_lf must be bool")
        
        if not isinstance(self.enable_hf, bool):
            raise TypeError("enable_hf must be bool")
        
        # 若启用 LF，则 lf_params_digest 必填。
        if self.enable_lf and (not isinstance(self.lf_params_digest, str) or not self.lf_params_digest):
            raise ValueError("lf_params_digest must be non-empty str when enable_lf=True")
        
        # 若启用 HF，则 hf_params_digest 必填。
        if self.enable_hf and (not isinstance(self.hf_params_digest, str) or not self.hf_params_digest):
            raise ValueError("hf_params_digest must be non-empty str when enable_hf=True")
        
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be non-empty str")
        
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("dtype must be non-empty str")


class LatentInjectionHook:
    """
    功能：专注 latent 张量修改的回调实现。

    Latent injection callback that applies watermark modifications during diffusion inference.
    This callback is decoupled from trajectory sampling and can be composed with other callbacks.

    Attributes:
        context: InjectionContext instance with injection parameters and digests.
        modifier: LatentModifier instance for applying tensor modifications.

    Args:
        context: InjectionContext instance.
        modifier: LatentModifier instance.

    Returns:
        None.

    Raises:
        TypeError: If context or modifier types are invalid.
    """

    def __init__(self, context: InjectionContext, modifier: Any) -> None:
        from main.watermarking.content_chain.latent_modifier import LatentModifier

        if not isinstance(context, InjectionContext):
            raise TypeError("context must be InjectionContext instance")
        
        if not isinstance(modifier, LatentModifier):
            raise TypeError("modifier must be LatentModifier instance")
        
        self.context = context
        self.modifier = modifier
        self._step_evidence_list: List[Dict[str, Any]] = []

    def __call__(
        self,
        pipe: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        功能：在推理每步后修改 latents。

        Apply latent modification during diffusion step callback.
        Modifies callback_kwargs["latents"] in-place and returns updated dict.
        Errors are caught and logged; latents pass through unchanged on failure.

        Args:
            pipe: Diffusers pipeline object.
            step_index: Current step index (0-based).
            timestep: Current timestep value.
            callback_kwargs: Callback arguments dict (must contain 'latents' key).

        Returns:
            Modified callback_kwargs with updated latents (or unchanged if error occurs).

        Raises:
            None (errors are logged, callback does not raise).
        """
        try:
            # 提取 latents 从 callback_kwargs。
            if "latents" not in callback_kwargs:
                # 无 latents 输入，直接返回（兼容不支持张量输入的 pipeline）。
                return callback_kwargs
            
            latents_input = callback_kwargs["latents"]
            
            # 应用修改：使用 context 中的 plan 和 enable 标志。
            latents_modified, step_evidence = self.modifier.apply_latent_update(
                latents=latents_input,
                plan=self.context.plan_ref,
                cfg={
                    "watermark": {
                        "lf": {"enabled": self.context.enable_lf},
                        "hf": {"enabled": self.context.enable_hf},
                    }
                },
                step_index=step_index,
                key=None  # 内部衍生
            )
            
            # 更新 callback_kwargs 中的 latents（真实张量修改）。
            callback_kwargs["latents"] = latents_modified
            
            # 记录 step 证据用于后续汇总。
            self._step_evidence_list.append(step_evidence)
        
        except Exception:
            # 错误处理：记录但不抛异常，确保推理不中断。
            import traceback
            traceback.print_exc()
            # latents 保持不变，推理继续（graceful degradation）。
            pass
        
        return callback_kwargs

    def collect_evidence(self) -> List[Dict[str, Any]]:
        """
        功能：收集所有 step 级注入证据。

        Retrieve accumulated step evidence from injection hook after inference.

        Args:
            None.

        Returns:
            List of step evidence dicts.

        Raises:
            None.
        """
        return self._step_evidence_list


def compose_step_end_callbacks(*callbacks: Callable) -> Callable:
    """
    功能：组合多个 callback_on_step_end 回调，依次调用，共享同一 callback_kwargs。

    Compose multiple step-end callbacks into a single callback.
    Callbacks are invoked in order, each modifying and passing callback_kwargs to the next.
    All callbacks receive the same callback_kwargs dict and can modify it in-place.

    Args:
        *callbacks: Variable number of callback functions with signature
                   (pipe, step_index, timestep, callback_kwargs) -> callback_kwargs.

    Returns:
        Composed callback function with same signature.

    Raises:
        TypeError: If any callback is not callable.
        ValueError: If no callbacks provided.
    """
    if not callbacks:
        raise ValueError("at least one callback must be provided")
    
    for cb in callbacks:
        if not callable(cb):
            raise TypeError("all callbacks must be callable")

    def composed_callback(
        pipe: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        功能：依次调用所有回调，共享 callback_kwargs。

        Call all callbacks in sequence, passing modified callback_kwargs through chain.

        Args:
            pipe: Diffusers pipeline object.
            step_index: Current step index.
            timestep: Current timestep.
            callback_kwargs: Callback arguments dict (shared across all callbacks).

        Returns:
            Final callback_kwargs after all callbacks have processed it.

        Raises:
            None (each callback handles its own errors).
        """
        for cb in callbacks:
            callback_kwargs = cb(pipe, step_index, timestep, callback_kwargs)
        
        return callback_kwargs

    # 挂载原始回调列表到组合版本，便于测试与审计。
    composed_callback._composed_callbacks = list(callbacks)
    
    return composed_callback


def extract_injection_context_from_cfg(
    cfg: Dict[str, Any],
    plan: Dict[str, Any],
    plan_digest: str
) -> InjectionContext:
    """
    功能：从配置和计划中提取 InjectionContext。

    Extract InjectionContext from runtime config and plan digest.
    Validates that all required parameters are present and consistent.

    Args:
        cfg: Configuration mapping.
        plan: Subspace plan mapping from SubspacePlanner.
        plan_digest: SHA256 digest of plan (must be pre-computed).

    Returns:
        InjectionContext instance ready for callback composition.

    Raises:
        TypeError: If cfg or plan types are invalid.
        ValueError: If required fields are missing.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    
    if not isinstance(plan, dict):
        raise TypeError("plan must be dict")
    
    if not isinstance(plan_digest, str) or not plan_digest:
        raise TypeError("plan_digest must be non-empty str")
    
    watermark_cfg = cfg.get("watermark", {})
    if not isinstance(watermark_cfg, dict):
        raise ValueError("cfg.watermark must be dict")
    
    # 提取 LF/HF 启用标志。
    lf_cfg = watermark_cfg.get("lf", {})
    hf_cfg = watermark_cfg.get("hf", {})
    enable_lf = lf_cfg.get("enabled", False)
    enable_hf = hf_cfg.get("enabled", False)
    
    if not isinstance(enable_lf, bool):
        raise TypeError("watermark.lf.enabled must be bool")
    if not isinstance(enable_hf, bool):
        raise TypeError("watermark.hf.enabled must be bool")
    
    # 若启用，则提取对应的参数摘要（应该由执行时计算）。
    lf_params_digest = lf_cfg.get("params_digest", "") if enable_lf else ""
    hf_params_digest = hf_cfg.get("params_digest", "") if enable_hf else ""
    
    if enable_lf and not lf_params_digest:
        raise ValueError("lf.params_digest required when enable_lf=True")
    if enable_hf and not hf_params_digest:
        raise ValueError("hf.params_digest required when enable_hf=True")
    
    # 提取执行策略。
    device = cfg.get("device", "cpu")
    dtype = cfg.get("dtype", "float32")
    
    if not isinstance(device, str) or not device:
        raise ValueError("cfg.device must be non-empty str")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("cfg.dtype must be non-empty str")
    
    # 构建并返回 context。
    return InjectionContext(
        plan_digest=plan_digest,
        plan_ref=plan,
        lf_params_digest=lf_params_digest,
        hf_params_digest=hf_params_digest,
        enable_lf=enable_lf,
        enable_hf=enable_hf,
        device=device,
        dtype=dtype
    )
