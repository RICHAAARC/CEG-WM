"""
SD3 推理流

功能说明：
- 在不注入真实水印方法的前提下，打通真实 SD3 推理数据流。
- 不抛异常，失败转 inference_status="failed" 并给出结构化错误。
- 返回 inference_runtime_meta（dict）与输出摘要。
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

from main.diffusion.sd3 import trajectory_tap


INFERENCE_STATUS_OK = "ok"
INFERENCE_STATUS_FAILED = "failed"
INFERENCE_STATUS_DISABLED = "disabled"


def run_sd3_inference(
    cfg: Dict[str, Any],
    pipeline_obj: Any,
    device: str | None,
    seed: int | None
) -> Dict[str, Any]:
    """
    功能：执行 SD3 推理并返回 inference_runtime_meta。

    Run SD3 inference with smoke testing only (no real watermarking).

    Args:
        cfg: Configuration mapping.
        pipeline_obj: Pipeline object (may be None if unbuilt).
        device: Device string ("cpu", "cuda", etc.) or None.
        seed: Random seed integer or None.

    Returns:
        Dict with inference_status, inference_error, inference_runtime_meta.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    inference_enabled = cfg.get("inference_enabled", False)
    if not inference_enabled:
        return {
            "inference_status": INFERENCE_STATUS_DISABLED,
            "inference_error": None,
            "inference_runtime_meta": None,
            "trajectory_evidence": trajectory_tap.build_trajectory_evidence(
                cfg,
                INFERENCE_STATUS_DISABLED,
                None,
                seed=seed,
                device=device
            )
        }

    inference_status = INFERENCE_STATUS_OK
    inference_error = None
    inference_runtime_meta: Dict[str, Any] = {}

    # (1) 检查 pipeline_obj 是否可用
    if pipeline_obj is None:
        inference_status = INFERENCE_STATUS_FAILED
        inference_error = "pipeline_obj is None"
        return {
            "inference_status": inference_status,
            "inference_error": inference_error,
            "inference_runtime_meta": None,
            "trajectory_evidence": trajectory_tap.build_trajectory_evidence(
                cfg,
                inference_status,
                None,
                seed=seed,
                device=device
            )
        }

    # (2) 解析推理参数
    try:
        prompt = cfg.get("inference_prompt", "a photo of a dog")
        num_inference_steps = cfg.get("inference_num_steps", 10)
        guidance_scale = cfg.get("inference_guidance_scale", 7.0)
        height = cfg.get("inference_height", 512)
        width = cfg.get("inference_width", 512)

        if not isinstance(prompt, str) or not prompt:
            inference_status = INFERENCE_STATUS_FAILED
            inference_error = "inference_prompt missing or empty"
            return {
                "inference_status": inference_status,
                "inference_error": inference_error,
                "inference_runtime_meta": None,
                "trajectory_evidence": trajectory_tap.build_trajectory_evidence(
                    cfg,
                    inference_status,
                    None,
                    seed=seed,
                    device=device
                )
            }

        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            inference_status = INFERENCE_STATUS_FAILED
            inference_error = f"inference_num_steps invalid: {num_inference_steps}"
            return {
                "inference_status": inference_status,
                "inference_error": inference_error,
                "inference_runtime_meta": None,
                "trajectory_evidence": trajectory_tap.build_trajectory_evidence(
                    cfg,
                    inference_status,
                    None,
                    seed=seed,
                    device=device
                )
            }

        inference_runtime_meta["prompt"] = prompt
        inference_runtime_meta["num_inference_steps"] = num_inference_steps
        inference_runtime_meta["guidance_scale"] = guidance_scale
        inference_runtime_meta["height"] = height
        inference_runtime_meta["width"] = width
        inference_runtime_meta["device"] = device if device else "<absent>"
        inference_runtime_meta["seed"] = seed if seed is not None else "<absent>"

    except Exception as exc:
        inference_status = INFERENCE_STATUS_FAILED
        inference_error = f"param_parse_error: {type(exc).__name__}: {exc}"
        return {
            "inference_status": inference_status,
            "inference_error": inference_error,
            "inference_runtime_meta": inference_runtime_meta,
            "trajectory_evidence": trajectory_tap.build_trajectory_evidence(
                cfg,
                inference_status,
                inference_runtime_meta,
                seed=seed,
                device=device
            )
        }

    # (3) 尝试执行推理
    try:
        # 检查 pipeline_obj 是否有 __call__ 方法
        if not callable(pipeline_obj):
            inference_status = INFERENCE_STATUS_FAILED
            inference_error = "pipeline_obj is not callable"
            return {
                "inference_status": inference_status,
                "inference_error": inference_error,
                "inference_runtime_meta": inference_runtime_meta,
                "trajectory_evidence": trajectory_tap.build_trajectory_evidence(
                    cfg,
                    inference_status,
                    inference_runtime_meta,
                    seed=seed,
                    device=device
                )
            }

        # 构造推理参数
        infer_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width
        }
        if seed is not None:
            import torch
            generator = torch.Generator(device=device if device else "cpu")
            generator.manual_seed(seed)
            infer_kwargs["generator"] = generator

        # 执行推理
        output = pipeline_obj(**infer_kwargs)

        # 提取输出摘要
        if hasattr(output, "images") and output.images is not None and len(output.images) > 0:
            inference_runtime_meta["output_image_count"] = len(output.images)
            first_image = output.images[0]
            if hasattr(first_image, "size"):
                inference_runtime_meta["output_image_size"] = list(first_image.size)
            if hasattr(first_image, "mode"):
                inference_runtime_meta["output_image_mode"] = first_image.mode
        else:
            inference_runtime_meta["output_image_count"] = 0

        inference_status = INFERENCE_STATUS_OK

    except ImportError as exc:
        # torch 不可用
        inference_status = INFERENCE_STATUS_FAILED
        inference_error = f"ImportError: {exc}"
    except AttributeError as exc:
        # pipeline_obj 结构不符合预期
        inference_status = INFERENCE_STATUS_FAILED
        inference_error = f"AttributeError: {exc}"
    except RuntimeError as exc:
        # CUDA/device 错误
        inference_status = INFERENCE_STATUS_FAILED
        inference_error = f"RuntimeError: {exc}"
    except Exception as exc:
        # 其他推理错误
        inference_status = INFERENCE_STATUS_FAILED
        inference_error = f"{type(exc).__name__}: {exc}"

    return {
        "inference_status": inference_status,
        "inference_error": inference_error,
        "inference_runtime_meta": inference_runtime_meta,
        "trajectory_evidence": trajectory_tap.build_trajectory_evidence(
            cfg,
            inference_status,
            inference_runtime_meta,
            seed=seed,
            device=device
        )
    }
