"""
Injection Site Binder

功能：
- 把"水印注入位置"显式化为可审计对象（injection_site_spec）。
- 记录 hook 的模块路径、hook 时机、作用张量类型与形状摘要。
- 生成可复算的 injection_site_digest。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from main.core import digests


IMPL_ID = "injection_site_binder_v1"
IMPL_VERSION = "v1"


def build_injection_site_spec(
    hook_type: str,
    target_module_name: Optional[str] = None,
    target_tensor_name: Optional[str] = None,
    hook_timing: Optional[str] = None,
    injection_rule_summary: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    功能：构造注入点规范与摘要。

    Build injection site specification and digest.

    Args:
        hook_type: Hook 类型（例如 "callback_on_step_end"、"hook_transformer_block"）。
        target_module_name: 目标模块名称（例如 "transformer.blocks[0].attn"）。
        target_tensor_name: 作用张量名称（例如 "latents"、"hidden_states"）。
        hook_timing: Hook 时机（例如 "pre"、"post"）。
        injection_rule_summary: 注入规则摘要（投影/截断/编码参数）。
        cfg: Optional configuration mapping for context.

    Returns:
        Tuple of (injection_site_spec dict, injection_site_digest str).

    Raises:
        TypeError: If hook_type is invalid.
    """
    if not isinstance(hook_type, str) or not hook_type:
        raise TypeError("hook_type must be non-empty string")

    injection_site_spec = {
        "hook_type": hook_type
    }

    if isinstance(target_module_name, str) and target_module_name:
        injection_site_spec["target_module_name"] = target_module_name

    if isinstance(target_tensor_name, str) and target_tensor_name:
        injection_site_spec["target_tensor_name"] = target_tensor_name

    if isinstance(hook_timing, str) and hook_timing:
        injection_site_spec["hook_timing"] = hook_timing

    if isinstance(injection_rule_summary, dict):
        injection_site_spec["injection_rule_summary"] = injection_rule_summary
        # 计算 injection_rule_digest（必达，供 alignment check 使用）
        injection_site_spec["injection_rule_digest"] = digests.canonical_sha256(injection_rule_summary)

    # 从 cfg 中提取相关配置（若有）。
    if isinstance(cfg, dict):
        # 记录是否启用 LF/HF 子空间。
        lf_enabled = cfg.get("watermark", {}).get("lf", {}).get("enabled", False)
        hf_enabled = cfg.get("watermark", {}).get("hf", {}).get("enabled", False)
        injection_site_spec["lf_channel_enabled"] = lf_enabled
        injection_site_spec["hf_channel_enabled"] = hf_enabled

        # 记录子空间 frame。
        subspace_frame = cfg.get("watermark", {}).get("subspace", {}).get("frame")
        if isinstance(subspace_frame, str) and subspace_frame:
            injection_site_spec["subspace_frame"] = subspace_frame

    # 生成 injection_site_digest。
    injection_site_digest = digests.canonical_sha256(injection_site_spec)

    return injection_site_spec, injection_site_digest


def build_injection_site_spec_from_callback(
    callback: Any,
    cfg: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    功能：从 callback 对象提取注入点规范。

    Build injection site spec from callback object.

    Args:
        callback: Callback function or object.
        cfg: Optional configuration mapping for context.

    Returns:
        Tuple of (injection_site_spec dict, injection_site_digest str).
    """
    if callback is None:
        return _absent_injection_site()

    # 提取 callback 的基本信息。
    hook_type = "callback_on_step_end"
    target_tensor_name = "latents"
    hook_timing = "post"

    # 尝试从 callback 提取更详细信息（若 callback 是自定义对象）。
    if hasattr(callback, "__name__"):
        callback_name = callback.__name__
    elif hasattr(callback, "__class__"):
        callback_name = callback.__class__.__name__
    else:
        callback_name = "<unknown>"

    injection_rule_summary = {
        "callback_name": callback_name
    }

    # 若 callback 有 modifier 属性（例如 LatentModifier），提取其信息。
    if hasattr(callback, "__self__"):
        callback_self = callback.__self__
        if hasattr(callback_self, "modifier"):
            modifier = callback_self.modifier
            if hasattr(modifier, "impl_id"):
                injection_rule_summary["modifier_impl_id"] = modifier.impl_id
            if hasattr(modifier, "impl_version"):
                injection_rule_summary["modifier_impl_version"] = modifier.impl_version

    return build_injection_site_spec(
        hook_type=hook_type,
        target_module_name=None,
        target_tensor_name=target_tensor_name,
        hook_timing=hook_timing,
        injection_rule_summary=injection_rule_summary,
        cfg=cfg
    )


def _absent_injection_site() -> Tuple[Dict[str, Any], str]:
    """
    功能：返回 absent 语义的注入点规范。

    Return absent injection site spec.

    Returns:
        Tuple of (injection_site_spec dict, absent digest).
    """
    injection_site_spec = {
        "hook_type": "<absent>",
        "status": "absent"
    }
    return injection_site_spec, "<absent>"


def validate_injection_site_digest(
    injection_site_spec: Dict[str, Any],
    expected_digest: str
) -> bool:
    """
    功能：验证 injection_site_digest 是否可复算且一致。

    Validate injection site digest reproducibility.

    Args:
        injection_site_spec: Injection site spec dict.
        expected_digest: Expected digest string.

    Returns:
        True if digest matches, False otherwise.
    """
    if not isinstance(injection_site_spec, dict):
        return False

    if not isinstance(expected_digest, str) or not expected_digest:
        return False

    if expected_digest == "<absent>":
        return injection_site_spec.get("status") == "absent"

    recomputed_digest = digests.canonical_sha256(injection_site_spec)
    return recomputed_digest == expected_digest


def get_binder_impl_identity() -> Dict[str, str]:
    """
    功能：返回 binder 的实现身份标识。

    Get binder implementation identity.

    Returns:
        Dict with impl_id, impl_version, impl_digest.
    """
    impl_digest = digests.canonical_sha256({
        "impl_id": IMPL_ID,
        "impl_version": IMPL_VERSION,
        "source_module": "main.watermarking.paper_faithfulness.injection_site_binder"
    })
    return {
        "impl_id": IMPL_ID,
        "impl_version": IMPL_VERSION,
        "impl_digest": impl_digest
    }
