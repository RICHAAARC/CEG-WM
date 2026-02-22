"""
SD3 推理轨迹构造器

功能说明：
- 构造 infer_trace 对象，保证 digest 输入域稳定。
- 不包含绝对路径，仅包含可复算的推理配置与状态。
"""

from __future__ import annotations

from typing import Dict, Any

from main.core import digests


def build_infer_trace(
    cfg: Dict[str, Any],
    inference_status: str,
    inference_error: str | None,
    inference_runtime_meta: Dict[str, Any] | None,
    trajectory_evidence: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    功能：构造 infer_trace 对象。

    Build inference trace mapping from cfg and inference result.

    Args:
        cfg: Configuration mapping.
        inference_status: Inference status string.
        inference_error: Inference error string or None.
        inference_runtime_meta: Inference runtime meta dict or None.
        trajectory_evidence: Optional trajectory tap evidence mapping.

    Returns:
        Inference trace mapping.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if trajectory_evidence is not None and not isinstance(trajectory_evidence, dict):
        # trajectory_evidence 类型不合法，必须 fail-fast。
        raise TypeError("trajectory_evidence must be dict or None")

    # 提取推理配置（不含绝对路径）
    inference_enabled = cfg.get("inference_enabled", False)
    prompt = cfg.get("inference_prompt", "<absent>")
    num_steps = cfg.get("inference_num_steps", "<absent>")
    guidance_scale = cfg.get("inference_guidance_scale", "<absent>")
    height = cfg.get("inference_height", "<absent>")
    width = cfg.get("inference_width", "<absent>")
    seed = cfg.get("seed", "<absent>")

    trace = {
        "inference_enabled": inference_enabled,
        "inference_prompt": prompt if isinstance(prompt, str) else "<absent>",
        "inference_num_steps": num_steps if isinstance(num_steps, int) else "<absent>",
        "inference_guidance_scale": guidance_scale if isinstance(guidance_scale, (int, float)) else "<absent>",
        "inference_height": height if isinstance(height, int) else "<absent>",
        "inference_width": width if isinstance(width, int) else "<absent>",
        "seed": seed if isinstance(seed, int) else "<absent>",
        "inference_status": inference_status if isinstance(inference_status, str) else "<absent>",
        "inference_error": inference_error if isinstance(inference_error, str) else None
    }

    scheduler_cfg = cfg.get("scheduler")
    if not isinstance(scheduler_cfg, dict):
        scheduler_cfg = {}
    trace["scheduler_config_digest"] = digests.canonical_sha256(scheduler_cfg)

    if trajectory_evidence is None:
        trace["trajectory_tap_status"] = "<absent>"
        trace["trajectory_spec_digest"] = "<absent>"
        trace["trajectory_digest"] = "<absent>"
        trace["trajectory_tap_version"] = "<absent>"
        trace["trajectory_absent_reason"] = "<absent>"
    else:
        trace["trajectory_tap_status"] = trajectory_evidence.get("status", "<absent>")
        trace["trajectory_spec_digest"] = trajectory_evidence.get("trajectory_spec_digest", "<absent>")
        trace["trajectory_digest"] = trajectory_evidence.get("trajectory_digest", "<absent>")
        trace["trajectory_tap_version"] = trajectory_evidence.get("trajectory_tap_version", "<absent>")
        trace["trajectory_absent_reason"] = trajectory_evidence.get("trajectory_absent_reason", "<absent>")

    # 添加 inference_runtime_meta 摘要
    if isinstance(inference_runtime_meta, dict):
        trace["output_image_count"] = inference_runtime_meta.get("output_image_count", 0)
        trace["device"] = inference_runtime_meta.get("device", "<absent>")
    else:
        trace["output_image_count"] = 0
        trace["device"] = "<absent>"

    return trace


def compute_infer_trace_canon_sha256(trace: Dict[str, Any]) -> str:
    """
    功能：计算 infer_trace 的 canonical sha256。

    Compute canonical SHA256 digest for inference trace.

    Args:
        trace: Inference trace mapping.

    Returns:
        SHA256 hex digest string.

    Raises:
        TypeError: If trace is invalid.
    """
    if not isinstance(trace, dict):
        # trace 类型不合法，必须 fail-fast。
        raise TypeError("trace must be dict")

    return digests.canonical_sha256(trace)
