"""
SD3 推理流

功能说明：
- 在不注入真实水印方法的前提下，打通真实 SD3 推理数据流。
- 不抛异常，失败转 inference_status="failed" 并给出结构化错误。
- 返回 inference_runtime_meta（dict）与输出摘要。
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List

from main.diffusion.sd3 import trajectory_tap
from main.diffusion.sd3.callback_composer import InjectionContext
from main.watermarking.content_chain.latent_modifier import LatentModifier
from main.watermarking.content_chain import channel_lf, channel_hf
from main.core import digests


INFERENCE_STATUS_OK = "ok"
INFERENCE_STATUS_FAILED = "failed"
INFERENCE_STATUS_DISABLED = "disabled"


def run_sd3_inference(
    cfg: Dict[str, Any],
    pipeline_obj: Any,
    device: str | None,
    seed: int | None,
    *,
    injection_context: Optional[InjectionContext] = None,
    injection_modifier: Optional[LatentModifier] = None,
    capture_final_latents: bool = False
) -> Dict[str, Any]:
    """
    功能：执行 SD3 推理并返回 inference_runtime_meta。

    Run SD3 inference with smoke testing only (no real watermarking).

    Args:
        cfg: Configuration mapping.
        pipeline_obj: Pipeline object (may be None if unbuilt).
        device: Device string ("cpu", "cuda", etc.) or None.
        seed: Random seed integer or None.
        capture_final_latents: Optional bool to capture final latents from inference (for detect-side scoring).

    Returns:
        Dict with inference_status, inference_error, inference_runtime_meta, and optional final_latents.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if injection_context is not None and not isinstance(injection_context, InjectionContext):
        # injection_context 类型不合法，必须 fail-fast。
        raise TypeError("injection_context must be InjectionContext or None")
    if injection_modifier is not None and not isinstance(injection_modifier, LatentModifier):
        # injection_modifier 类型不合法，必须 fail-fast。
        raise TypeError("injection_modifier must be LatentModifier or None")
    if not isinstance(capture_final_latents, bool):
        # capture_final_latents 类型不合法，必须 fail-fast。
        raise TypeError("capture_final_latents must be bool")

    inference_enabled = cfg.get("inference_enabled", False)
    detect_latents_storage = {"final_latents": None} if capture_final_latents else None

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
            ),
            "injection_evidence": _build_injection_absent_evidence(
                injection_context,
                absent_reason="inference_disabled"
            )
        }

    inference_status = INFERENCE_STATUS_OK
    inference_error = None
    inference_runtime_meta: Dict[str, Any] = {}
    trajectory_evidence: Dict[str, Any] | None = None
    injection_evidence: Dict[str, Any] | None = None

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

        def _capture_latents_callback(
            _pipe: Any,
            step_index: int,
            timestep: Any,
            callback_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            功能：捕获推理过程中的最后一次 latents。

            Capture the latest latents during inference for detect-side scoring.
            """
            if not isinstance(callback_kwargs, dict):
                return callback_kwargs
            latents = callback_kwargs.get("latents")
            if latents is None:
                return callback_kwargs
            if detect_latents_storage is not None:
                detect_latents_storage["final_latents"] = latents
            return callback_kwargs

        # 构造注入回调（闭包捕获 InjectionContext）。
        injection_callback, injection_evidence = _prepare_injection_callback(
            cfg,
            injection_context,
            injection_modifier
        )

        capture_callback = _capture_latents_callback if capture_final_latents else None
        callback_to_use = injection_callback
        if injection_callback is not None and capture_callback is not None:
            def _combined_callback(
                _pipe: Any,
                step_index: int,
                timestep: Any,
                callback_kwargs: Dict[str, Any]
            ) -> Dict[str, Any]:
                callback_kwargs = injection_callback(_pipe, step_index, timestep, callback_kwargs)
                return capture_callback(_pipe, step_index, timestep, callback_kwargs)
            callback_to_use = _combined_callback
        elif injection_callback is None:
            callback_to_use = capture_callback

        if callback_to_use is not None:
            infer_kwargs["callback_on_step_end"] = callback_to_use
            infer_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        # 执行推理并在支持时采样真实 trajectory 摘要。
        tap_call_result = trajectory_tap.tap_from_pipeline(
            cfg,
            pipeline_obj,
            infer_kwargs,
            inference_runtime_meta,
            seed=seed,
            device=device
        )
        output = tap_call_result.get("output")
        trajectory_evidence = tap_call_result.get("trajectory_evidence")
        tap_status = tap_call_result.get("tap_status")

        # 更新注入证据状态（处理不支持 callback 的降级路径）。
        if injection_context is not None:
            injection_evidence = _finalize_injection_evidence(
                injection_context,
                injection_evidence,
                tap_status
            )

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
        "trajectory_evidence": trajectory_evidence if isinstance(trajectory_evidence, dict) else trajectory_tap.build_trajectory_evidence(
            cfg,
            inference_status,
            inference_runtime_meta,
            seed=seed,
            device=device
        ),
        "injection_evidence": injection_evidence if isinstance(injection_evidence, dict) else _build_injection_absent_evidence(
            injection_context,
            absent_reason="injection_not_available"
        ),
        "final_latents": detect_latents_storage.get("final_latents") if detect_latents_storage is not None else None
    }


def _prepare_injection_callback(
    cfg: Dict[str, Any],
    injection_context: Optional[InjectionContext],
    injection_modifier: Optional[LatentModifier]
) -> tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    功能：构造注入回调与初始注入证据。
    
    Prepare injection callback capturing InjectionContext and initialize evidence.

    Args:
        cfg: Configuration mapping.
        injection_context: InjectionContext instance or None.
        injection_modifier: LatentModifier instance or None.
    Returns:
        Tuple of (callback or None, initial injection evidence dict or None).

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if injection_context is None:
        return None, _build_injection_absent_evidence(None, "injection_not_enabled")
    if injection_modifier is None:
        return None, _build_injection_absent_evidence(injection_context, "modifier_missing")

    # 校验 params_digest 与运行期参数的一致性。
    params_status, params_reason, params_payload = _validate_injection_params(cfg, injection_context)
    if params_status != "ok":
        return None, _build_injection_mismatch_evidence(
            injection_context,
            params_reason,
            params_payload
        )

    injection_cfg = _build_injection_cfg(cfg, injection_context)
    step_evidence_list: List[Dict[str, Any]] = []
    plan_cache: Dict[str, Any] | None = None

    def _injection_callback(
        _pipe: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        功能：在每步中执行注入并收集证据。
        
        Apply latent injection and collect step evidence during inference.
        """
        if not isinstance(callback_kwargs, dict):
            return callback_kwargs
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        nonlocal plan_cache
        if plan_cache is None:
            plan_cache = _build_plan_for_injection(
                injection_context.plan_ref,
                injection_context.plan_digest,
                latents,
                injection_cfg
            )

        latents_modified, step_evidence = injection_modifier.apply_latent_update(
            latents=latents,
            plan=plan_cache,
            cfg=injection_cfg,
            step_index=step_index,
            key=None
        )
        callback_kwargs["latents"] = latents_modified
        step_evidence_list.append(step_evidence)
        return callback_kwargs

    initial_evidence = _build_injection_ok_evidence(
        injection_context,
        params_payload,
        step_evidence_list
    )
    return _injection_callback, initial_evidence


def _build_injection_cfg(cfg: Dict[str, Any], context: InjectionContext) -> Dict[str, Any]:
    """
    功能：构造注入配置（不修改原 cfg）。

    Build injection config without mutating cfg.

    Args:
        cfg: Configuration mapping.
        context: InjectionContext instance.

    Returns:
        Injection config mapping for LatentModifier.
    """
    watermark_cfg = cfg.get("watermark", {}) if isinstance(cfg.get("watermark", {}), dict) else {}
    lf_cfg = watermark_cfg.get("lf", {}) if isinstance(watermark_cfg.get("lf", {}), dict) else {}
    hf_cfg = watermark_cfg.get("hf", {}) if isinstance(watermark_cfg.get("hf", {}), dict) else {}

    lf_strength = lf_cfg.get("strength", cfg.get("lf_strength", 1.5))
    hf_threshold_percentile = hf_cfg.get("threshold_percentile", cfg.get("hf_threshold_percentile", 75.0))
    watermark_seed = cfg.get("watermark_seed", cfg.get("seed", 42))

    return {
        "lf_enabled": context.enable_lf,
        "hf_enabled": context.enable_hf,
        "lf_strength": lf_strength,
        "hf_threshold_percentile": hf_threshold_percentile,
        "watermark_seed": watermark_seed
    }


def _validate_injection_params(
    cfg: Dict[str, Any],
    context: InjectionContext
) -> tuple[str, str, Dict[str, Any]]:
    """
    功能：校验注入参数摘要一致性。
    
    Validate injection parameter digests against runtime config.

    Args:
        cfg: Configuration mapping.
        context: InjectionContext instance.

    Returns:
        Tuple of (status, reason, params_payload).
    """
    injection_cfg = _build_injection_cfg(cfg, context)
    lf_params = {
        "impl_id": channel_lf.LF_CHANNEL_IMPL_ID,
        "impl_version": channel_lf.LF_CHANNEL_VERSION,
        "lf_strength": injection_cfg.get("lf_strength"),
        "lf_enabled": bool(context.enable_lf)
    }
    hf_params = {
        "impl_id": channel_hf.HF_CHANNEL_IMPL_ID,
        "impl_version": channel_hf.HF_CHANNEL_VERSION,
        "hf_threshold_percentile": injection_cfg.get("hf_threshold_percentile"),
        "hf_enabled": bool(context.enable_hf)
    }
    lf_params_digest = digests.canonical_sha256(lf_params)
    hf_params_digest = digests.canonical_sha256(hf_params)

    params_payload = {
        "plan_digest": context.plan_digest,
        "lf_params_digest": lf_params_digest,
        "hf_params_digest": hf_params_digest,
        "lf_enabled": bool(context.enable_lf),
        "hf_enabled": bool(context.enable_hf)
    }

    if context.enable_lf and lf_params_digest != context.lf_params_digest:
        return "mismatch", "lf_params_digest_mismatch", params_payload
    if context.enable_hf and hf_params_digest != context.hf_params_digest:
        return "mismatch", "hf_params_digest_mismatch", params_payload
    return "ok", "ok", params_payload


def _build_plan_for_injection(
    plan_ref: Dict[str, Any],
    plan_digest: str,
    latents: Any,
    injection_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    功能：为注入构造包含 basis 的 plan。
    
    Build a runtime plan payload with deterministic tensor-driven basis when missing.

    Args:
        plan_ref: Plan reference mapping from planner.
        plan_digest: Plan digest string.
        latents: Latent tensor or array (used to infer shape).
        injection_cfg: Injection config mapping.

    Returns:
        Plan mapping containing lf_basis/hf_basis or runtime absent binding.
    """
    plan_payload = dict(plan_ref) if isinstance(plan_ref, dict) else {}
    if "lf_basis" in plan_payload and "hf_basis" in plan_payload:
        return plan_payload

    runtime_binding = _build_runtime_subspace_binding(
        plan_payload=plan_payload,
        plan_digest=plan_digest,
        latents=latents,
        injection_cfg=injection_cfg
    )
    plan_payload["runtime_subspace_binding"] = runtime_binding
    if runtime_binding.get("status") != "ok":
        return plan_payload

    if "lf_basis" not in plan_payload:
        plan_payload["lf_basis"] = {
            "projection_matrix": runtime_binding.get("lf_projection_matrix"),
            "basis_source": "trajectory_tensor_stats"
        }
    if "hf_basis" not in plan_payload:
        plan_payload["hf_basis"] = {
            "hf_projection_matrix": runtime_binding.get("hf_projection_matrix"),
            "basis_source": "trajectory_tensor_stats"
        }

    # 避免写入大对象到 evidence 之外的字段。
    runtime_binding.pop("lf_projection_matrix", None)
    runtime_binding.pop("hf_projection_matrix", None)
    return plan_payload


def _to_numpy(latents: Any) -> Optional[Any]:
    """
    功能：将 latents 转换为 numpy 数组（仅用于形状推断）。
    
    Convert latents to numpy array for shape inference.
    """
    try:
        if hasattr(latents, "cpu"):
            return latents.cpu().numpy()
        import numpy as np
        return np.asarray(latents)
    except Exception:
        return None


def _build_runtime_subspace_binding(
    plan_payload: Dict[str, Any],
    plan_digest: str,
    latents: Any,
    injection_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    功能：基于真实推理张量构造确定性子空间绑定摘要。

    Build deterministic runtime subspace binding from real inference tensor summary.

    Args:
        plan_payload: Plan payload mapping.
        plan_digest: Plan digest string.
        latents: Runtime latents tensor.
        injection_cfg: Injection config mapping.

    Returns:
        Runtime subspace binding mapping.
    """
    try:
        import torch
    except Exception:
        return {
            "status": "absent",
            "absent_reason": "torch_unavailable"
        }

    if not torch.is_tensor(latents):
        return {
            "status": "absent",
            "absent_reason": "latents_not_torch_tensor"
        }

    latents_fp32 = latents.reshape(-1).to(dtype=torch.float32)
    latent_dim = int(latents_fp32.shape[0])
    if latent_dim <= 0:
        return {
            "status": "absent",
            "absent_reason": "invalid_latent_dimension"
        }

    planner_params = plan_payload.get("planner_params", {}) if isinstance(plan_payload.get("planner_params"), dict) else {}
    rank_value = planner_params.get("rank", plan_payload.get("rank", 8))
    if not isinstance(rank_value, int):
        try:
            rank_value = int(rank_value)
        except Exception:
            rank_value = 0
    rank = max(1, min(rank_value, latent_dim)) if rank_value > 0 else 0
    if rank <= 0:
        return {
            "status": "absent",
            "absent_reason": "rank_unavailable"
        }

    abs_values = torch.abs(latents_fp32)
    if int(abs_values.shape[0]) < rank:
        return {
            "status": "absent",
            "absent_reason": "rank_exceeds_latent_dimension"
        }

    hf_indices = torch.topk(abs_values, k=rank, largest=True, sorted=True).indices
    lf_indices = torch.topk(abs_values, k=rank, largest=False, sorted=True).indices

    lf_projection_matrix = torch.zeros((latent_dim, rank), dtype=torch.float32, device=latents.device)
    hf_projection_matrix = torch.zeros((latent_dim, rank), dtype=torch.float32, device=latents.device)
    lf_projection_matrix[lf_indices, torch.arange(rank, device=latents.device)] = 1.0
    hf_projection_matrix[hf_indices, torch.arange(rank, device=latents.device)] = 1.0

    l2_total = float(torch.sum(latents_fp32 * latents_fp32).item())
    hf_energy = float(torch.sum((abs_values[hf_indices] ** 2)).item())
    energy_ratio = 0.0 if l2_total <= 0 else hf_energy / l2_total

    trajectory_anchor = {
        "mean": float(torch.mean(latents_fp32).item()),
        "std": float(torch.std(latents_fp32, unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(latents_fp32).item()),
        "latent_dim": latent_dim,
        "rank": rank,
        "lf_enabled": bool(injection_cfg.get("lf_enabled", True)),
        "hf_enabled": bool(injection_cfg.get("hf_enabled", True)),
        "plan_digest": plan_digest,
    }
    trajectory_anchor_digest = digests.canonical_sha256(trajectory_anchor)

    binding_payload = {
        "binding_version": "trajectory_tensor_binding_v1",
        "plan_digest": plan_digest,
        "trajectory_anchor_digest": trajectory_anchor_digest,
        "rank": rank,
        "energy_ratio": round(float(energy_ratio), 8),
        "lf_indices": [int(v) for v in lf_indices.tolist()],
        "hf_indices": [int(v) for v in hf_indices.tolist()],
    }
    binding_digest = digests.canonical_sha256(binding_payload)

    return {
        "status": "ok",
        "absent_reason": None,
        "binding_version": "trajectory_tensor_binding_v1",
        "trajectory_anchor_digest": trajectory_anchor_digest,
        "binding_digest": binding_digest,
        "rank": rank,
        "energy_ratio": round(float(energy_ratio), 8),
        "lf_indices_digest": digests.canonical_sha256([int(v) for v in lf_indices.tolist()]),
        "hf_indices_digest": digests.canonical_sha256([int(v) for v in hf_indices.tolist()]),
        "lf_projection_matrix": lf_projection_matrix,
        "hf_projection_matrix": hf_projection_matrix,
    }


def _build_injection_ok_evidence(
    context: InjectionContext,
    params_payload: Dict[str, Any],
    step_evidence_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    功能：构造注入证据容器（待完成填充）。

    Build injection evidence container (filled after inference).
    """
    return {
        "status": "ok",
        "injection_absent_reason": None,
        "injection_failure_reason": None,
        "injection_trace_digest": None,
        "injection_params_digest": digests.canonical_sha256(params_payload),
        "injection_metrics": None,
        "subspace_binding_digest": None,
        "plan_digest": context.plan_digest,
        "lf_params_digest": context.lf_params_digest if context.enable_lf else None,
        "hf_params_digest": context.hf_params_digest if context.enable_hf else None,
        "_step_evidence_list": step_evidence_list
    }


def _build_injection_absent_evidence(
    context: Optional[InjectionContext],
    absent_reason: str
) -> Dict[str, Any]:
    """
    功能：构造注入 absent 证据。

    Build injection absent evidence mapping.
    """
    return {
        "status": "absent",
        "injection_absent_reason": absent_reason,
        "injection_failure_reason": None,
        "injection_trace_digest": None,
        "injection_params_digest": None,
        "injection_metrics": None,
        "subspace_binding_digest": None,
        "plan_digest": context.plan_digest if isinstance(context, InjectionContext) else None,
        "lf_params_digest": context.lf_params_digest if isinstance(context, InjectionContext) else None,
        "hf_params_digest": context.hf_params_digest if isinstance(context, InjectionContext) else None
    }


def _build_injection_mismatch_evidence(
    context: InjectionContext,
    mismatch_reason: str,
    params_payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    功能：构造注入 mismatch 证据。

    Build injection mismatch evidence mapping.
    """
    return {
        "status": "mismatch",
        "injection_absent_reason": None,
        "injection_failure_reason": mismatch_reason,
        "injection_trace_digest": None,
        "injection_params_digest": digests.canonical_sha256(params_payload),
        "injection_metrics": None,
        "subspace_binding_digest": None,
        "plan_digest": context.plan_digest,
        "lf_params_digest": context.lf_params_digest if context.enable_lf else None,
        "hf_params_digest": context.hf_params_digest if context.enable_hf else None
    }


def _finalize_injection_evidence(
    context: InjectionContext,
    injection_evidence: Optional[Dict[str, Any]],
    tap_status: Optional[str]
) -> Dict[str, Any]:
    """
    功能：根据推理结果收口注入证据。

    Finalize injection evidence with aggregated metrics and digests.
    """
    if injection_evidence is None:
        return _build_injection_absent_evidence(context, "injection_not_available")
    if injection_evidence.get("status") in {"absent", "mismatch"}:
        return injection_evidence
    if tap_status == "unsupported":
        return _build_injection_absent_evidence(context, "unsupported_pipeline")

    step_evidence_list = injection_evidence.pop("_step_evidence_list", [])
    if not isinstance(step_evidence_list, list) or len(step_evidence_list) == 0:
        return _build_injection_absent_evidence(context, "latents_missing")

    metrics = _summarize_injection_metrics(step_evidence_list)
    combined_status_counts = metrics.get("combined_status_counts", {}) if isinstance(metrics, dict) else {}
    if isinstance(combined_status_counts, dict):
        non_absent_count = 0
        for status_key, status_count in combined_status_counts.items():
            if status_key != "absent" and isinstance(status_count, int):
                non_absent_count += status_count
        if non_absent_count == 0:
            return _build_injection_absent_evidence(context, "runtime_subspace_unavailable")

    trace_payload = {
        "plan_digest": injection_evidence.get("plan_digest"),
        "injection_params_digest": injection_evidence.get("injection_params_digest"),
        "metrics": metrics
    }
    trace_digest = digests.canonical_sha256(trace_payload)

    injection_evidence["injection_metrics"] = metrics
    injection_evidence["injection_trace_digest"] = trace_digest
    if isinstance(metrics, dict):
        subspace_binding_digest = metrics.get("subspace_binding_digest")
        if isinstance(subspace_binding_digest, str) and subspace_binding_digest:
            injection_evidence["subspace_binding_digest"] = subspace_binding_digest
    return injection_evidence


def _summarize_injection_metrics(step_evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    功能：汇总注入 step 级指标。
    
    Summarize step evidence metrics without exposing raw tensors.
    """
    if not isinstance(step_evidence_list, list):
        return {
            "step_count": 0,
            "combined_status_counts": {},
            "delta_norm_mean": 0.0
        }

    delta_values = []
    lf_delta_values = []
    hf_delta_values = []
    subspace_binding_digests: List[str] = []
    status_counts: Dict[str, int] = {}
    for step_evidence in step_evidence_list:
        if not isinstance(step_evidence, dict):
            continue
        combined_status = step_evidence.get("combined_status", "unknown")
        if isinstance(combined_status, str):
            status_counts[combined_status] = status_counts.get(combined_status, 0) + 1
        delta_norm = step_evidence.get("modification_delta_norm")
        if isinstance(delta_norm, (int, float)):
            delta_values.append(float(delta_norm))
        lf_delta_norm = step_evidence.get("lf_delta_norm")
        if isinstance(lf_delta_norm, (int, float)):
            lf_delta_values.append(float(lf_delta_norm))
        hf_delta_norm = step_evidence.get("hf_delta_norm")
        if isinstance(hf_delta_norm, (int, float)):
            hf_delta_values.append(float(hf_delta_norm))
        binding_digest = step_evidence.get("runtime_subspace_binding_digest")
        if isinstance(binding_digest, str) and binding_digest:
            subspace_binding_digests.append(binding_digest)

    delta_mean = float(sum(delta_values) / len(delta_values)) if delta_values else 0.0
    lf_delta_mean = float(sum(lf_delta_values) / len(lf_delta_values)) if lf_delta_values else 0.0
    hf_delta_mean = float(sum(hf_delta_values) / len(hf_delta_values)) if hf_delta_values else 0.0
    unique_binding = sorted(list(set(subspace_binding_digests)))
    subspace_binding_digest = unique_binding[0] if len(unique_binding) == 1 else None
    return {
        "step_count": len(step_evidence_list),
        "combined_status_counts": status_counts,
        "delta_norm_mean": delta_mean,
        "lf_delta_norm_mean": lf_delta_mean,
        "hf_delta_norm_mean": hf_delta_mean,
        "subspace_binding_digest": subspace_binding_digest
    }
