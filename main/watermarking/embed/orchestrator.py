"""
嵌入流程编排

功能说明：
- 定义了嵌入流程的编排器函数，用于协调不同组件的执行流程。
- 每个编排器函数都接受配置和实现集作为输入，并返回包含业务字段的记录映射。
- 实现了输入验证和错误处理，确保接口的健壮性。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence, cast

import numpy as np
from PIL import Image

from main.registries.runtime_resolver import BuiltImplSet
from main.core import digests
from main.core import records_io
from main.policy import path_policy
from main.watermarking.common.plan_digest_flow import (
    build_content_plan_and_digest,
    bind_plan_to_record,
)
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    build_runtime_jvp_operator_from_cache,
)
from main.watermarking.content_chain.high_freq_embedder import (
    compute_hf_trace_digest,
)
from main.watermarking.content_chain.low_freq_coder import (
    compute_lf_trace_digest,
)
from main.watermarking.content_chain import high_freq_embedder as high_freq_embedder_module
from main.watermarking.content_chain import low_freq_coder as low_freq_coder_module
from main.watermarking.geometry_chain.sync import SyncRuntimeContext


def _as_dict_payload(value: Any) -> Dict[str, Any] | None:
    """
    功能：将对象规范化为 dict 负载。

    Convert a payload-like object to a dictionary.

    Args:
        value: Candidate payload object.

    Returns:
        Dictionary payload when available; otherwise None.
    """
    if isinstance(value, dict):
        return cast(Dict[str, Any], value)
    as_dict_method = getattr(value, "as_dict", None)
    if callable(as_dict_method):
        converted = as_dict_method()
        if isinstance(converted, dict):
            return cast(Dict[str, Any], converted)
    return None


def _normalize_execution_chain_status(raw_status: Any) -> str:
    """
    功能：将链路状态归一化到 ok/absent/failed 三态。

    Normalize execution-chain status into canonical enum {ok, absent, failed}.

    Args:
        raw_status: Raw status token from runtime payload.

    Returns:
        Canonical status token.
    """
    if not isinstance(raw_status, str) or not raw_status:
        return "failed"
    normalized = raw_status.strip().lower()
    if normalized == "fail":
        return "failed"
    if normalized in {"failed", "error", "mismatch"}:
        return "failed"
    if normalized in {"absent", "none", "disabled", "not_applicable"}:
        return "absent"
    if normalized in {"ok", "synced", "accepted", "rejected", "abstain"}:
        return "ok"
    return "failed"


def _extract_embed_attestation_secrets(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    功能：提取 embed 主链 attestation 所需的临时密钥输入。

    Extract transient attestation secret inputs injected by the CLI layer.

    Args:
        cfg: Configuration mapping.

    Returns:
        Secret input mapping or None.
    """
    secret_node = cfg.get("__attestation_secret_inputs__")
    if isinstance(secret_node, dict):
        return cast(Dict[str, Any], secret_node)
    return None


def _collect_embed_attestation_latent_snapshots(cfg: Dict[str, Any]) -> Sequence[Any] | None:
    """
    功能：从内存 trajectory cache 收集 attestation 所需的 latent 快照。

    Collect latent snapshots for trajectory commit from the in-memory trajectory cache.

    Args:
        cfg: Configuration mapping.

    Returns:
        Ordered latent snapshot sequence when available; otherwise None.
    """
    cache = cfg.get("__embed_trajectory_latent_cache__")
    if cache is None or not hasattr(cache, "available_steps") or not hasattr(cache, "get"):
        return None
    steps = cache.available_steps()
    snapshots: list[Any] = []
    for step_index in steps:
        snapshot = cache.get(step_index)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots or None


def _bind_attestation_runtime_to_cfg(cfg: Dict[str, Any], attestation_payload: Dict[str, Any]) -> None:
    """
    功能：将 attestation 运行时变量绑定到 cfg，供 LF/HF/GEO 主链消费。

    Bind attestation runtime variables into cfg for LF/HF/GEO execution.

    Args:
        cfg: Mutable configuration mapping.
        attestation_payload: Attestation payload mapping.

    Returns:
        None.
    """
    if not isinstance(attestation_payload, dict):
        return
    if attestation_payload.get("attestation_status") != "ok":
        return
    runtime_bindings = attestation_payload.get("runtime_bindings")
    if not isinstance(runtime_bindings, dict):
        return

    cfg["attestation_runtime"] = runtime_bindings
    cfg["attestation_digest"] = runtime_bindings.get("attestation_digest")
    cfg["attestation_event_digest"] = runtime_bindings.get("event_binding_digest")
    cfg["lf_attestation_event_digest"] = runtime_bindings.get("event_binding_digest")
    cfg["hf_attestation_event_digest"] = runtime_bindings.get("event_binding_digest")
    cfg["lf_attestation_key"] = runtime_bindings.get("k_lf")
    cfg["hf_attestation_key"] = runtime_bindings.get("k_hf")
    cfg["k_lf"] = runtime_bindings.get("k_lf")
    cfg["k_hf"] = runtime_bindings.get("k_hf")
    cfg["k_geo"] = runtime_bindings.get("k_geo")
    cfg["geo_anchor_seed"] = runtime_bindings.get("geo_anchor_seed")

    watermark_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    watermark_cfg = dict(watermark_cfg)
    watermark_cfg["attestation_digest"] = runtime_bindings.get("attestation_digest")
    watermark_cfg["attestation_event_digest"] = runtime_bindings.get("event_binding_digest")
    cfg["watermark"] = watermark_cfg


def _clear_attestation_runtime_from_cfg(cfg: Dict[str, Any]) -> None:
    """
    功能：清理仅运行期使用的 attestation 临时字段。

    Remove transient attestation runtime fields after orchestration completes.

    Args:
        cfg: Mutable configuration mapping.

    Returns:
        None.
    """
    for field_name in [
        "attestation_runtime",
        "attestation_digest",
        "attestation_event_digest",
        "lf_attestation_event_digest",
        "hf_attestation_event_digest",
        "lf_attestation_key",
        "hf_attestation_key",
        "k_lf",
        "k_hf",
        "k_geo",
        "geo_anchor_seed",
    ]:
        cfg.pop(field_name, None)
    watermark_node = cfg.get("watermark")
    if isinstance(watermark_node, dict):
        watermark_cfg = dict(cast(Dict[str, Any], watermark_node))
        watermark_cfg.pop("attestation_digest", None)
        watermark_cfg.pop("attestation_event_digest", None)
        cfg["watermark"] = watermark_cfg


def _sanitize_embed_attestation_payload(attestation_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：生成可写入 records 的 attestation 安全载荷。

    Build a record-safe attestation payload that excludes derived secret keys.

    Args:
        attestation_payload: Raw attestation payload mapping.

    Returns:
        Record-safe attestation mapping.
    """
    if not isinstance(attestation_payload, dict):
        return {
            "status": "absent",
            "attestation_absent_reason": "attestation_payload_invalid",
        }
    if attestation_payload.get("attestation_status") != "ok":
        return {
            "status": attestation_payload.get("attestation_status", "absent"),
            "attestation_absent_reason": attestation_payload.get("attestation_absent_reason"),
            "attestation_failure_reason": attestation_payload.get("attestation_failure_reason"),
            "missing_secret_fields": attestation_payload.get("missing_secret_fields"),
        }
    return {
        "status": "ok",
        "statement": attestation_payload.get("statement"),
        "attestation_digest": attestation_payload.get("attestation_digest"),
        "event_binding_digest": attestation_payload.get("event_binding_digest"),
        "lf_payload_hex": attestation_payload.get("lf_payload_hex"),
        "trace_commit": attestation_payload.get("trace_commit"),
        "geo_anchor_seed": attestation_payload.get("geo_anchor_seed"),
        "signed_bundle": attestation_payload.get("signed_bundle"),
        "signed_bundle_present": isinstance(attestation_payload.get("signed_bundle"), dict),
        "trajectory_snapshot_count": attestation_payload.get("trajectory_snapshot_count"),
    }


def _build_embed_attestation_runtime(cfg: Dict[str, Any], plan_digest: str) -> Dict[str, Any]:
    """
    功能：在 embed 主链中构造正式 attestation 运行时载荷。

    Build the formal attestation payload used by the embed main path.

    Args:
        cfg: Configuration mapping.
        plan_digest: Canonical content plan digest.

    Returns:
        Attestation payload mapping with runtime bindings and record-safe fields.
    """
    attestation_node = cfg.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    if not bool(attestation_cfg.get("enabled", False)):
        return {
            "attestation_status": "absent",
            "attestation_absent_reason": "attestation_disabled",
        }

    secret_inputs = _extract_embed_attestation_secrets(cfg)
    required_fields = ["k_master", "k_prompt", "k_seed"]
    missing_secret_fields = [field_name for field_name in required_fields if not isinstance((secret_inputs or {}).get(field_name), str) or not str((secret_inputs or {}).get(field_name)).strip()]
    if missing_secret_fields:
        return {
            "attestation_status": "absent",
            "attestation_absent_reason": "attestation_secret_missing",
            "missing_secret_fields": missing_secret_fields,
        }

    model_id = cfg.get("model_id")
    prompt = cfg.get("inference_prompt")
    seed_value = cfg.get("seed")
    latent_snapshots = _collect_embed_attestation_latent_snapshots(cfg)

    try:
        result = build_embed_attestation(
            k_master=str(secret_inputs.get("k_master")),
            model_id=model_id if isinstance(model_id, str) and model_id else "sd3",
            prompt=prompt if isinstance(prompt, str) else "",
            seed=int(seed_value) if isinstance(seed_value, int) else 0,
            plan_digest=plan_digest,
            k_prompt=str(secret_inputs.get("k_prompt")),
            k_seed=str(secret_inputs.get("k_seed")),
            latent_snapshots=latent_snapshots,
            use_trajectory_mix=bool(attestation_cfg.get("use_trajectory_mix", True)),
        )
    except Exception as exc:
        return {
            "attestation_status": "failed",
            "attestation_failure_reason": f"embed_attestation_build_failed:{type(exc).__name__}",
        }

    result["trajectory_snapshot_count"] = len(latent_snapshots) if latent_snapshots is not None else 0
    result["runtime_bindings"] = {
        "attestation_digest": result.get("attestation_digest"),
        "event_binding_digest": result.get("event_binding_digest"),
        "k_lf": result.get("keys", {}).get("k_lf") if isinstance(result.get("keys"), dict) else None,
        "k_hf": result.get("keys", {}).get("k_hf") if isinstance(result.get("keys"), dict) else None,
        "k_geo": result.get("keys", {}).get("k_geo") if isinstance(result.get("keys"), dict) else None,
        "geo_anchor_seed": result.get("geo_anchor_seed"),
    }
    return result


def run_embed_orchestrator(
    cfg: Dict[str, Any],
    impl_set: BuiltImplSet,
    cfg_digest: str,
    *,
    trajectory_evidence: Dict[str, Any] | None = None,
    injection_evidence: Dict[str, Any] | None = None,
    sync_runtime_context: SyncRuntimeContext | None = None,
    content_result_override: Any | None = None,
    subspace_result_override: Any | None = None
) -> Dict[str, Any]:
    """
    功能：执行嵌入编排流程。

    Execute embed workflow using injected implementations.
    Supports ablation flags: when ablation.normalized.enable_content=false,
    content_extractor returns status="absent" with no failure reason.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.
        cfg_digest: Canonical SHA256 digest of cfg (computed from include_paths).
                   Passed to content_extractor to bind mask_digest to authoritative digest.
        trajectory_evidence: Optional trajectory tap evidence mapping.
        injection_evidence: Optional injection evidence mapping.
        content_result_override: Optional precomputed content result.
        subspace_result_override: Optional precomputed subspace plan result.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If impl_set is invalid.
    """
    if content_result_override is not None and not isinstance(content_result_override, dict) and not hasattr(content_result_override, "as_dict"):
        # content_result_override 类型不符合预期，必须 fail-fast。
        raise TypeError("content_result_override must be dict, ContentEvidence, or None")
    if subspace_result_override is not None and not isinstance(subspace_result_override, dict) and not hasattr(subspace_result_override, "as_dict"):
        # subspace_result_override 类型不符合预期，必须 fail-fast。
        raise TypeError("subspace_result_override must be dict, SubspacePlan, or None")

    # 读取 ablation.normalized 开关（若缺失则默认全启用）。
    ablation_normalized = _get_ablation_normalized(cfg)
    enable_content = ablation_normalized.get("enable_content", True)
    enable_subspace = ablation_normalized.get("enable_subspace", True)
    enable_lf = ablation_normalized.get("enable_lf", True)
    enable_hf = ablation_normalized.get("enable_hf", bool(cfg.get("watermark", {}).get("hf", {}).get("enabled", False)))

    content_inputs = _build_content_inputs_for_embed(cfg)
    if content_inputs is None:
        cfg["__embed_content_input_absent_reason__"] = "embed_input_image_path_missing"
    else:
        cfg.pop("__embed_content_input_absent_reason__", None)
    
    # Ablation: 禁用 content 模块时返回 absent 语义。
    content_result: Any
    if not enable_content:
        content_result = _build_ablation_absent_content_evidence("content_chain_disabled_by_ablation")
    elif content_result_override is not None:
        content_result = cast(Any, content_result_override)
    else:
        content_result = impl_set.content_extractor.extract(
            cfg,
            inputs=content_inputs,
            cfg_digest=cfg_digest
        )

    content_evidence_payload = _as_dict_payload(content_result)
    if content_evidence_payload is None:
        # content_result 类型不符合预期，必须 fail-fast。
        raise TypeError("content_result must be dict or have as_dict method")

    if trajectory_evidence is not None:
        content_evidence_payload["trajectory_evidence"] = trajectory_evidence
        _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
    if injection_evidence is not None:
        _merge_injection_evidence(content_evidence_payload, injection_evidence)
    _normalize_content_evidence_optional_mappings(content_evidence_payload)
    
    # 捕获 content_chain 的执行状态（用于 execution_report）。
    # 统一归一化为 ok / absent / failed，避免写出废弃枚举 fail。
    content_status = content_evidence_payload.get("status", "unknown")
    content_chain_status = _normalize_execution_chain_status(content_status)
    
    # 提取 mask_digest 以绑定到规划器。
    mask_digest = content_evidence_payload.get("mask_digest")
    # 若 content_result 是对象（如 ContentEvidence），提取 mask_digest。
    if not isinstance(mask_digest, str):
        mask_digest = None
    
    # 调用规划器计算 plan_digest，绑定 cfg_digest + mask_digest + planner_params。
    planner_inputs = _build_planner_inputs_for_runtime(cfg, trajectory_evidence, content_evidence_payload)
    subspace_result: Any
    if not enable_subspace:
        subspace_result = _build_ablation_absent_subspace_result("subspace_disabled_by_ablation")
    elif subspace_result_override is not None:
        subspace_result = cast(Any, subspace_result_override)
    else:
        subspace_result = impl_set.subspace_planner.plan(
            cfg,
            mask_digest=mask_digest,
            cfg_digest=cfg_digest,
            inputs=planner_inputs
        )
    
    plan_obj, plan_digest, plan_input_digest, plan_meta = build_content_plan_and_digest(
        cfg,
        subspace_result,
        mask_digest if isinstance(mask_digest, str) else None,
        mask_binding=_extract_mask_binding(content_evidence_payload),
        mask_params_digest=_extract_mask_params_digest(content_evidence_payload),
    )
    attestation_payload = _build_embed_attestation_runtime(cfg, plan_digest)
    _bind_attestation_runtime_to_cfg(cfg, attestation_payload)
    io_anchors = _prepare_embed_real_io_anchors(cfg)
    use_latent_per_step = _should_use_latent_per_step_path(cfg)
    paper_cfg_raw = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_cfg_raw) if isinstance(paper_cfg_raw, dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False))
    if paper_enabled and not use_latent_per_step:
        # paper 模式下必须走 latent per-step 主路径。
        raise ValueError("paper_faithfulness requires latent per-step embed path")
    
    # Paper-faithful mode 强制检查 HF/LF 正式实现（阶段 4）。
    if paper_enabled:
        impl_cfg_raw = cfg.get("impl")
        impl_cfg = cast(Dict[str, Any], impl_cfg_raw) if isinstance(impl_cfg_raw, dict) else {}
        hf_impl_id = impl_cfg.get("hf_embedder_id")
        lf_impl_id = impl_cfg.get("lf_coder_id")
        
        # 检查 HF embedder 必须是 formal truncation codec。
        if hf_impl_id and hf_impl_id != high_freq_embedder_module.HIGH_FREQ_TRUNCATION_CODEC_ID:
            raise ValueError(
                f"paper_faithfulness requires {high_freq_embedder_module.HIGH_FREQ_TRUNCATION_CODEC_ID}, got {hf_impl_id}"
            )
        
        # 检查 LF coder 必须是 low_freq_template_codec
        if lf_impl_id and lf_impl_id != "low_freq_template_codec":
            raise ValueError(f"paper_faithfulness requires low_freq_template_codec, got {lf_impl_id}")
    
    try:
        sync_result = _run_sync_module(cfg, impl_set, sync_runtime_context)

        if use_latent_per_step:
            embed_trace = _build_latent_step_embed_trace(cfg, injection_evidence)
        else:
            embed_trace = {
                "embed_mode": io_anchors["embed_mode"],
                "note": "content_embedding_real_v1",
            }
        embed_trace["attestation_status"] = attestation_payload.get("attestation_status")
        if attestation_payload.get("attestation_status") == "ok":
            embed_trace["attestation_event_digest"] = attestation_payload.get("event_binding_digest")
            embed_trace["signed_bundle_present"] = isinstance(attestation_payload.get("signed_bundle"), dict)
        else:
            embed_trace["attestation_absent_reason"] = attestation_payload.get("attestation_absent_reason")
            embed_trace["attestation_failure_reason"] = attestation_payload.get("attestation_failure_reason")

        if (
            not use_latent_per_step
            and io_anchors["image_path"] != "<absent>"
        ):
            input_image = Image.open(io_anchors["image_path"]).convert("RGB")
            watermarked_image, pipeline_trace = _apply_content_embedding_pipeline(
                impl_set=impl_set,
                image=input_image,
                plan=plan_obj,
                cfg=cfg,
                cfg_digest=cfg_digest,
                plan_digest=plan_digest,
                content_evidence_payload=content_evidence_payload,
                enable_lf=bool(enable_lf),
                enable_hf=bool(enable_hf),
            )
            artifact_rel_path, artifact_sha256, watermarked_path = _write_watermarked_artifact_controlled(
                watermarked_image=watermarked_image,
                run_root=Path(io_anchors["run_root"]),
                artifacts_dir=Path(io_anchors["artifacts_dir"]),
                output_rel=io_anchors["output_rel"],
            )
            io_anchors["watermarked_path"] = watermarked_path
            io_anchors["artifact_rel_path"] = artifact_rel_path
            io_anchors["artifact_sha256"] = artifact_sha256
            embed_trace.update(pipeline_trace)

        # 构造返回的业务字段映射。
        record_fields: Dict[str, Any] = {
            "operation": "embed",
            "embed_mode": embed_trace.get("embed_mode", io_anchors["embed_mode"]),
            "image_path": io_anchors["image_path"],
            "watermarked_path": io_anchors["watermarked_path"],
            "input_sha256": io_anchors["input_sha256"],
            "artifact_sha256": io_anchors["artifact_sha256"],
            "artifact_rel_path": io_anchors["artifact_rel_path"],
            "watermarked_artifact_sha256": io_anchors["artifact_sha256"],
            "watermarked_artifact_rel_path": io_anchors["artifact_rel_path"],
            "seed": 42,
            "strength": 0.5,
            "embed_trace": embed_trace,
            "content_result": content_evidence_payload,
            "content_evidence": content_evidence_payload,
            "sync_result": sync_result,
            "attestation": _sanitize_embed_attestation_payload(attestation_payload),
            # 添加 execution_report（冻结门禁要求）。
            # 注：embed 阶段未执行融合，fusion_status 置为 "absent"；
            #     geometry 链不参与，geometry_chain_status 置为 "absent"。
            "execution_report": {
                "content_chain_status": content_chain_status,
                "geometry_chain_status": _normalize_execution_chain_status(sync_result.get("status", "absent")),
                "fusion_status": "absent",
                "audit_obligations_satisfied": True
            }
        }
        bind_plan_to_record(
            record_fields,
            plan_obj=plan_obj,
            plan_digest=plan_digest,
            plan_input_digest=plan_input_digest,
            plan_meta=plan_meta,
        )
        _bind_mask_and_routing_evidence_to_record(record_fields, content_evidence_payload)
        
        subspace_payload = _as_dict_payload(subspace_result)
        if isinstance(subspace_payload, dict):
            basis_digest = subspace_payload.get("basis_digest")
            if isinstance(basis_digest, str) and basis_digest:
                record_fields["basis_digest"] = basis_digest
            plan_stats = subspace_payload.get("plan_stats")
            if isinstance(plan_stats, dict):
                record_fields["plan_stats"] = plan_stats
            # 写入规划器失败原因（可观测性字段，仅在失败时非 None）。
            plan_failure_reason = subspace_payload.get("plan_failure_reason")
            if plan_failure_reason is not None:
                record_fields["plan_failure_reason"] = plan_failure_reason
            plan_node = subspace_payload.get("plan")
            if isinstance(plan_node, dict):
                plan_node_payload = cast(Dict[str, Any], plan_node)
                record_fields["subspace_rank"] = plan_node_payload.get("rank")
                record_fields["subspace_energy_ratio"] = plan_node_payload.get("energy_ratio")
                record_fields["subspace_planner_impl_identity"] = plan_node_payload.get("planner_impl_identity")
        
        return record_fields
    finally:
        _clear_attestation_runtime_from_cfg(cfg)


def _run_sync_module(
    cfg: Dict[str, Any],
    impl_set: BuiltImplSet,
    sync_runtime_context: SyncRuntimeContext | None,
) -> Dict[str, Any]:
    """
    功能：执行同步模块并优先使用运行期上下文。

    Execute sync module and prefer runtime context when available.

    Args:
        cfg: Configuration mapping.
        impl_set: Built implementation set.
        sync_runtime_context: Optional sync runtime context.

    Returns:
        Sync result mapping.

    Raises:
        TypeError: If inputs are invalid.
    """
    sync_module = impl_set.sync_module
    sync_inject_trace: Dict[str, Any] | None = None

    if sync_runtime_context is not None and sync_runtime_context.latents is not None:
        embed_inject_callable = getattr(sync_module, "embed_inject", None)
        if not callable(embed_inject_callable):
            sync_template_obj = getattr(sync_module, "_sync_template", None)
            embed_inject_callable = getattr(sync_template_obj, "embed_inject", None) if sync_template_obj is not None else None

        if callable(embed_inject_callable):
            seed_value = cfg.get("seed", 42)
            if not isinstance(seed_value, int):
                seed_value = 42
            inject_result = embed_inject_callable(sync_runtime_context.latents, cfg, int(seed_value))
            if isinstance(inject_result, tuple):
                inject_tuple = cast(tuple[Any, ...], inject_result)
                if len(inject_tuple) == 2:
                    injected_latents = inject_tuple[0]
                    sync_inject_trace = _as_dict_payload(inject_tuple[1])
                else:
                    injected_latents = sync_runtime_context.latents
                    sync_inject_trace = None
            else:
                injected_latents = sync_runtime_context.latents
                sync_inject_trace = None
            sync_runtime_context = SyncRuntimeContext(
                pipeline=sync_runtime_context.pipeline,
                latents=injected_latents,
                rng=sync_runtime_context.rng,
                trajectory_evidence=sync_runtime_context.trajectory_evidence,
            )

    if sync_runtime_context is not None:
        sync_with_context = getattr(sync_module, "sync_with_context", None)
        if sync_with_context is not None:
            if not callable(sync_with_context):
                # sync_with_context 类型不符合预期，必须 fail-fast。
                raise TypeError("sync_with_context must be callable")
            if sync_runtime_context.pipeline is not None and sync_runtime_context.latents is not None:
                result_obj = sync_with_context(cfg, sync_runtime_context)
                result = _as_dict_payload(result_obj)
                if not isinstance(result, dict):
                    raise TypeError("sync_with_context must return dict-like payload")
                if isinstance(sync_inject_trace, dict):
                    result["sync_inject_status"] = sync_inject_trace.get("sync_inject_status", sync_inject_trace.get("status"))
                    result["sync_inject_strength"] = sync_inject_trace.get("sync_inject_strength")
                    result["template_digest"] = sync_inject_trace.get("template_digest")
                return result

    if not hasattr(sync_module, "sync") or not callable(sync_module.sync):
        # sync 方法缺失，必须 fail-fast。
        raise TypeError("sync_module must provide sync")
    result_obj = sync_module.sync(cfg)
    result = _as_dict_payload(result_obj)
    if not isinstance(result, dict):
        raise TypeError("sync_module.sync must return dict-like payload")
    if isinstance(sync_inject_trace, dict):
        result["sync_inject_status"] = sync_inject_trace.get("sync_inject_status", sync_inject_trace.get("status"))
        result["sync_inject_strength"] = sync_inject_trace.get("sync_inject_strength")
        result["template_digest"] = sync_inject_trace.get("template_digest")
    return result


def _merge_injection_evidence(content_evidence_payload: Dict[str, Any], injection_evidence: Dict[str, Any]) -> None:
    """
    功能：将注入证据写入 content_evidence 兼容字段。
    
    Merge injection evidence into content evidence payload using append-only fields.

    Args:
        content_evidence_payload: Mutable content evidence mapping.
        injection_evidence: Injection evidence mapping.

    Returns:
        None.
    """
    content_evidence_payload["injection_status"] = injection_evidence.get("status")
    content_evidence_payload["injection_absent_reason"] = injection_evidence.get("injection_absent_reason")
    content_evidence_payload["injection_failure_reason"] = injection_evidence.get("injection_failure_reason")
    content_evidence_payload["injection_trace_digest"] = injection_evidence.get("injection_trace_digest")
    content_evidence_payload["injection_params_digest"] = injection_evidence.get("injection_params_digest")
    content_evidence_payload["step_summary_digest"] = injection_evidence.get("step_summary_digest")
    content_evidence_payload["injection_digest"] = injection_evidence.get("injection_digest")
    content_evidence_payload["injection_metrics"] = injection_evidence.get("injection_metrics")
    content_evidence_payload["subspace_binding_digest"] = injection_evidence.get("subspace_binding_digest")
    content_evidence_payload["lf_impl_binding"] = injection_evidence.get("lf_impl_binding")
    content_evidence_payload["hf_impl_binding"] = injection_evidence.get("hf_impl_binding")


def _inject_trajectory_audit_fields(
    content_evidence_payload: Dict[str, Any],
    trajectory_evidence: Dict[str, Any]
) -> None:
    """
    功能：将轨迹 tap 子状态写入 content_evidence.audit（兼容新旧字段）。

    Inject trajectory tap status fields into content_evidence.audit.

    Args:
        content_evidence_payload: Content evidence payload mapping.
        trajectory_evidence: Trajectory evidence mapping.

    Returns:
        None.
    """
    audit = content_evidence_payload.get("audit")
    if not isinstance(audit, dict):
        audit = {}
        content_evidence_payload["audit"] = audit

    tap_audit = trajectory_evidence.get("audit")
    tap_status = None
    tap_absent_reason = None
    if isinstance(tap_audit, dict):
        tap_audit_payload = cast(Dict[str, Any], tap_audit)
        tap_status = tap_audit_payload.get("trajectory_tap_status")
        tap_absent_reason = tap_audit_payload.get("trajectory_absent_reason")

    if not isinstance(tap_status, str) or not tap_status:
        status_value = trajectory_evidence.get("status")
        if isinstance(status_value, str) and status_value:
            tap_status = status_value

    if not isinstance(tap_absent_reason, str) or not tap_absent_reason:
        reason_value = trajectory_evidence.get("trajectory_absent_reason")
        if isinstance(reason_value, str) and reason_value:
            tap_absent_reason = reason_value

    if isinstance(tap_status, str) and tap_status:
        audit["trajectory_tap_status"] = tap_status
    if isinstance(tap_absent_reason, str) and tap_absent_reason:
        audit["trajectory_absent_reason"] = tap_absent_reason


def _normalize_content_evidence_optional_mappings(content_evidence_payload: Dict[str, Any] | None) -> None:
    """
    功能：将 content_evidence 中可选 mapping 字段的 None 规范化为空映射。

    Normalize optional mapping fields in content_evidence payload from None to {}.

    Args:
        content_evidence_payload: Mutable content evidence mapping.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return

    optional_mapping_keys = [
        "mask_stats",
        "score_parts",
    ]
    for key in optional_mapping_keys:
        current_value = content_evidence_payload.get(key)
        if current_value is None:
            content_evidence_payload[key] = {}


def _build_planner_inputs_for_runtime(
    cfg: Dict[str, Any],
    trajectory_evidence: Dict[str, Any] | None,
    content_evidence_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    功能：构造规划器输入签名。

    Build deterministic planner input signature from cfg runtime fields.

    Args:
        cfg: Configuration mapping.
        trajectory_evidence: Optional trajectory tap evidence.

    Returns:
        Planner input mapping containing trace_signature.

    Raises:
        TypeError: If cfg is invalid.
    """
    trace_signature = {
        "num_inference_steps": cfg.get("inference_num_steps", cfg.get("generation", {}).get("num_inference_steps", 16) if isinstance(cfg.get("generation"), dict) else 16),
        "guidance_scale": cfg.get("inference_guidance_scale", cfg.get("generation", {}).get("guidance_scale", 7.0) if isinstance(cfg.get("generation"), dict) else 7.0),
        "height": cfg.get("inference_height", cfg.get("model", {}).get("height", 512) if isinstance(cfg.get("model"), dict) else 512),
        "width": cfg.get("inference_width", cfg.get("model", {}).get("width", 512) if isinstance(cfg.get("model"), dict) else 512),
    }
    inputs: Dict[str, Any] = {"trace_signature": trace_signature}
    runtime_pipeline = cfg.get("__embed_pipeline_obj__")
    runtime_latents = cfg.get("__embed_final_latents__")
    if runtime_pipeline is not None:
        inputs["pipeline"] = runtime_pipeline
    if runtime_latents is not None:
        inputs["latents"] = runtime_latents
    # 将推理期间捕获的 per-step latent 缓存传递给 planner（内存传递，不写入 records）。
    runtime_traj_cache = cfg.get("__embed_trajectory_latent_cache__")
    if runtime_traj_cache is not None:
        inputs["trajectory_latent_cache"] = runtime_traj_cache
        runtime_jvp_operator = build_runtime_jvp_operator_from_cache(cfg, runtime_traj_cache)
        if callable(runtime_jvp_operator):
            inputs["jvp_operator"] = runtime_jvp_operator
    if trajectory_evidence is not None:
        inputs["trajectory_evidence"] = trajectory_evidence
    if isinstance(content_evidence_payload, dict):
        mask_stats = content_evidence_payload.get("mask_stats")
        if isinstance(mask_stats, dict):
            mask_stats_payload = cast(Dict[str, Any], mask_stats)
            inputs["mask_summary"] = mask_stats_payload
            routing_digest = mask_stats_payload.get("routing_digest")
            if isinstance(routing_digest, str) and routing_digest:
                inputs["routing_digest"] = routing_digest
    return inputs


def _prepare_embed_real_io_anchors(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：执行 embed 阶段真实输入输出锚定。 

    Resolve real input image path and derive controlled output targets.

    Args:
        cfg: Configuration mapping.

    Returns:
        Mapping with image path and digest anchors.

    Raises:
        TypeError: If cfg is invalid.
        ValueError: If runtime IO configuration is invalid.
    """
    result: Dict[str, Any] = {
        "embed_mode": "content_real_v1",
        "image_path": "<absent>",
        "watermarked_path": "<absent>",
        "input_sha256": "<absent>",
        "artifact_sha256": "<absent>",
        "artifact_rel_path": "<absent>",
        "run_root": "<absent>",
        "artifacts_dir": "<absent>",
        "output_rel": "watermarked/watermarked.png",
    }

    input_image_path = _resolve_embed_input_image_path(cfg)
    if input_image_path is None:
        return result

    run_root_raw = cfg.get("__run_root_dir__")
    artifacts_dir_raw = cfg.get("__artifacts_dir__")
    if not isinstance(run_root_raw, str) or not run_root_raw:
        # 缺失 run_root 信息时无法派生受控输出路径，必须 fail-fast。
        raise ValueError("__run_root_dir__ must be non-empty str when input image is configured")
    if not isinstance(artifacts_dir_raw, str) or not artifacts_dir_raw:
        # 缺失 artifacts_dir 信息时无法派生受控输出路径，必须 fail-fast。
        raise ValueError("__artifacts_dir__ must be non-empty str when input image is configured")

    input_path = Path(input_image_path).resolve()
    if not input_path.exists() or not input_path.is_file():
        # 输入图片不存在时必须 fail-fast。
        raise ValueError(f"input image path not found: {input_path}")

    run_root = Path(run_root_raw).resolve()
    artifacts_dir = Path(artifacts_dir_raw).resolve()

    embed_cfg_node = cfg.get("embed")
    embed_cfg = cast(Dict[str, Any], embed_cfg_node) if isinstance(embed_cfg_node, dict) else {}
    output_rel = embed_cfg.get("output_artifact_rel_path", "watermarked/watermarked.png")
    if not isinstance(output_rel, str) or not output_rel:
        # 输出相对路径不合法，必须 fail-fast。
        raise ValueError("embed.output_artifact_rel_path must be non-empty str when provided")

    output_path = (artifacts_dir / output_rel).resolve()
    path_policy.validate_output_target(output_path, "artifact", run_root)

    result["image_path"] = str(input_path)
    result["input_sha256"] = digests.file_sha256(input_path)
    result["run_root"] = str(run_root)
    result["artifacts_dir"] = str(artifacts_dir)
    result["output_rel"] = output_rel
    return result


def _apply_content_embedding_pipeline(
    impl_set: BuiltImplSet,
    image: Image.Image,
    plan: Dict[str, Any],
    cfg: Dict[str, Any],
    cfg_digest: str,
    plan_digest: str | None,
    content_evidence_payload: Dict[str, Any],
    enable_lf: bool = True,
    enable_hf: bool = True,
) -> tuple[Image.Image, Dict[str, Any]]:
    """
    功能：执行 LF/HF 真实嵌入流水线并回填证据。

    Apply real LF/HF embedding pipeline and bind trace evidence.

    Args:
        image: Input PIL image.
        plan: Planner plan mapping.
        cfg: Configuration mapping.
        cfg_digest: Config digest.
        plan_digest: Optional plan digest.
        content_evidence_payload: Mutable content evidence mapping.

    Returns:
        Tuple of (watermarked_image, embed_trace).
    """
    image_array = np.asarray(image, dtype=np.uint8)
    plan_band_spec_node = plan.get("band_spec")
    plan_band_spec = cast(Dict[str, Any], plan_band_spec_node) if isinstance(plan_band_spec_node, dict) else {}
    routing_summary_node = plan_band_spec.get("hf_selector_summary")
    routing_summary = cast(Dict[str, Any], routing_summary_node) if isinstance(routing_summary_node, dict) else {}

    key_material = digests.canonical_sha256(
        {
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "key_id": cfg.get("watermark", {}).get("key_id"),
            "pattern_id": cfg.get("watermark", {}).get("pattern_id"),
        }
    )

    lf_params = _build_lf_image_embed_params(cfg)
    lf_impl_binding: Dict[str, Any] = {
        "impl_selected": getattr(getattr(impl_set, "lf_coder", None), "impl_id", None),
        "evidence_level": "channel_absent",
    }
    lf_watermarked: Any = None
    lf_trace_summary: Dict[str, Any] = {
        "lf_status": "failed",
        "lf_failure_reason": "lf_trace_not_initialized",
    }

    lf_coder = getattr(impl_set, "lf_coder", None)
    can_use_lf_impl = (
        enable_lf
        and
        lf_coder is not None
        and hasattr(lf_coder, "embed_apply")
        and callable(getattr(lf_coder, "embed_apply", None))
        and isinstance(plan_digest, str)
        and bool(plan_digest)
    )
    if can_use_lf_impl and lf_coder is not None:
        try:
            cfg_for_lf = dict(cfg)
            watermark_node = cfg_for_lf.get("watermark")
            watermark_payload = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
            watermark_cfg = dict(watermark_payload)
            watermark_cfg["plan_digest"] = plan_digest
            cfg_for_lf["watermark"] = watermark_cfg
            lf_impl_result_obj = lf_coder.embed_apply(
                cfg=cfg_for_lf,
                latent_features=image_array.reshape(-1).astype(np.float64).tolist(),
                plan_digest=plan_digest,
                cfg_digest=cfg_digest,
            )
            if not isinstance(lf_impl_result_obj, dict):
                raise TypeError("lf_coder.embed_apply must return dict")
            lf_impl_result = cast(Dict[str, Any], lf_impl_result_obj)
            embedded_features = lf_impl_result.get("latent_features_embedded")
            if embedded_features is None:
                raise ValueError("lf_coder embedded features are absent")
            embedded_np = np.asarray(embedded_features, dtype=np.float64)
            if embedded_np.size != image_array.size:
                raise ValueError("lf_coder embedded feature size mismatch for image adapter")
            lf_watermarked = np.clip(np.round(embedded_np.reshape(image_array.shape)), 0, 255).astype(np.uint8)
            lf_trace_summary = {
                "lf_status": "ok",
                "lf_mode": "impl_set_lf_coder_adapter_v2",
                "lf_embedding_digest": lf_impl_result.get("embedding_digest"),
            }
            lf_impl_binding = {
                "impl_selected": getattr(lf_coder, "impl_id", None),
                "evidence_level": "primary",
            }
        except Exception:
            lf_impl_binding = {
                "impl_selected": getattr(lf_coder, "impl_id", None),
                "evidence_level": "channel_failed",
            }
            lf_watermarked = None

    if not enable_lf:
        lf_watermarked = image_array
        lf_trace_summary = {
            "lf_status": "absent",
            "lf_absent_reason": "lf_channel_disabled_by_ablation",
        }
        lf_impl_binding = {
            "impl_selected": getattr(lf_coder, "impl_id", None) if lf_coder is not None else None,
            "evidence_level": "ablation_disabled",
        }
    elif lf_watermarked is None:
        encode_low_freq_dct_fn = getattr(low_freq_coder_module, "encode_low_freq_dct")
        lf_watermarked, lf_trace_summary = encode_low_freq_dct_fn(image_array, plan_band_spec, key_material, lf_params)
    lf_trace_digest = compute_lf_trace_digest(
        {
            "summary": lf_trace_summary,
            "params": lf_params,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
        }
    )
    content_evidence_payload["lf_trace_digest"] = lf_trace_digest
    content_evidence_payload["lf_impl_binding"] = lf_impl_binding
    content_evidence_payload["lf_score"] = None
    score_parts = content_evidence_payload.get("score_parts")
    if not isinstance(score_parts, dict):
        score_parts = {}
        content_evidence_payload["score_parts"] = score_parts
    score_parts = cast(Dict[str, Any], score_parts)
    score_parts["lf_status"] = lf_trace_summary.get("lf_status", "ok")
    score_parts["lf_metrics"] = lf_trace_summary

    hf_enabled = bool(cfg.get("watermark", {}).get("hf", {}).get("enabled", False)) and enable_hf
    embed_trace: Dict[str, Any] = {
        "embed_mode": "content_real_v1",
        "lf_trace_digest": lf_trace_digest,
        "lf_trace_summary": lf_trace_summary,
        "lf_impl_binding": lf_impl_binding,
    }

    if hf_enabled:
        hf_params = _build_hf_image_embed_params(cfg)
        hf_embedder = getattr(impl_set, "hf_embedder", None)
        hf_impl_binding: Dict[str, Any] = {
            "impl_selected": getattr(hf_embedder, "impl_id", None),
            "evidence_level": "channel_absent",
        }
        embed_high_freq_pattern_fn = getattr(high_freq_embedder_module, "embed_high_freq_pattern")
        hf_watermarked, hf_trace_summary = embed_high_freq_pattern_fn(lf_watermarked, routing_summary, key_material, hf_params)
        hf_trace_digest = compute_hf_trace_digest(
            {
                "summary": hf_trace_summary,
                "params": hf_params,
                "plan_digest": plan_digest,
                "cfg_digest": cfg_digest,
            }
        )
        content_evidence_payload["hf_trace_digest"] = hf_trace_digest
        content_evidence_payload["hf_impl_binding"] = hf_impl_binding
        content_evidence_payload["hf_score"] = None
        score_parts["hf_status"] = hf_trace_summary.get("hf_status", "ok")
        score_parts["hf_metrics"] = hf_trace_summary
        embed_trace["hf_trace_digest"] = hf_trace_digest
        embed_trace["hf_trace_summary"] = hf_trace_summary
        embed_trace["hf_impl_binding"] = hf_impl_binding
        watermarked_array = hf_watermarked
    else:
        content_evidence_payload.pop("hf_trace_digest", None)
        content_evidence_payload.pop("hf_score", None)
        content_evidence_payload["hf_impl_binding"] = {
            "impl_selected": getattr(getattr(impl_set, "hf_embedder", None), "impl_id", None),
            "evidence_level": "ablation_disabled" if not enable_hf else "channel_absent",
        }
        score_parts.pop("hf_status", None)
        score_parts.pop("hf_metrics", None)
        score_parts["hf_absent_reason"] = "hf_channel_disabled_by_ablation" if not enable_hf else "hf_disabled_by_config"
        score_parts.pop("hf_failure_reason", None)
        embed_trace["hf_impl_binding"] = content_evidence_payload.get("hf_impl_binding")
        watermarked_array = lf_watermarked

    return Image.fromarray(np.asarray(watermarked_array)), embed_trace


def _write_watermarked_artifact_controlled(
    watermarked_image: Image.Image,
    run_root: Path,
    artifacts_dir: Path,
    output_rel: str,
) -> tuple[str, str, str]:
    """
    功能：受控写入 watermarked artifact 并返回锚点。

    Write watermarked image through controlled records_io path and return anchors.

    Args:
        watermarked_image: Watermarked PIL image.
        run_root: Run root directory.
        artifacts_dir: Artifacts directory.
        output_rel: Relative output path under artifacts.

    Returns:
        Tuple of (artifact_rel_path, artifact_sha256, watermarked_path).
    """
    if not output_rel:
        raise TypeError("output_rel must be non-empty str")

    output_path = (artifacts_dir / output_rel).resolve()
    path_policy.validate_output_target(output_path, "artifact", run_root.resolve())

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        watermarked_image.save(tmp_path, format="PNG")
        records_io.copy_file_controlled(tmp_path, output_path, kind="artifact")
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    artifact_sha256 = digests.file_sha256(output_path)
    artifact_rel_path = str(output_path.relative_to(run_root.resolve()))
    return artifact_rel_path, artifact_sha256, str(output_path)


def _build_lf_image_embed_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    watermark_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    lf_node = watermark_cfg.get("lf")
    lf_cfg = cast(Dict[str, Any], lf_node) if isinstance(lf_node, dict) else {}
    ecc_value = lf_cfg.get("ecc", 1)
    redundancy = ecc_value if isinstance(ecc_value, int) else 1
    return {
        "dct_block_size": int(lf_cfg.get("dct_block_size", 8)),
        "lf_coeff_indices": lf_cfg.get("lf_coeff_indices", [[1, 1], [1, 2], [2, 1]]),
        "alpha": float(lf_cfg.get("strength", 1.5)),
        "redundancy": int(redundancy),
        "variance": float(lf_cfg.get("variance", 1.5)),
    }


def _build_hf_image_embed_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    watermark_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    hf_node = watermark_cfg.get("hf")
    hf_cfg = cast(Dict[str, Any], hf_node) if isinstance(hf_node, dict) else {}
    return {
        "beta": float(hf_cfg.get("tau", 2.0)),
        "tail_truncation_ratio": float(hf_cfg.get("tail_truncation_ratio", 0.1)),
        "tail_truncation_mode": hf_cfg.get("tail_truncation_mode", "projection_tail_truncation"),
        "sampling_stride": int(hf_cfg.get("sampling_stride", 1)),
    }


def _resolve_embed_input_image_path(cfg: Dict[str, Any]) -> str | None:
    """
    功能：解析 embed 输入图像路径。 

    Resolve embed input image path from runtime/CLI/config fields.

    Args:
        cfg: Configuration mapping.

    Returns:
        Input image path string or None.
    """
    candidates = [
        cfg.get("__embed_input_image_path__"),
        cfg.get("input_image_path"),
    ]
    embed_cfg_node = cfg.get("embed")
    embed_cfg = cast(Dict[str, Any], embed_cfg_node) if isinstance(embed_cfg_node, dict) else {}
    if embed_cfg:
        candidates.append(embed_cfg.get("input_image_path"))

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _build_content_inputs_for_embed(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    功能：构造 embed 阶段 content extractor 输入。 

    Build content extractor inputs from embed real image path.

    Args:
        cfg: Configuration mapping.

    Returns:
        Inputs mapping or None when image path is absent.
    """
    image_path = _resolve_embed_input_image_path(cfg)
    if image_path is None:
        return None
    return {
        "image_path": image_path,
    }


def _extract_mask_binding(content_evidence_payload: Any) -> Dict[str, Any] | None:
    """
    功能：提取 mask 分辨率绑定字段。 

    Extract mask resolution binding from content evidence payload.

    Args:
        content_evidence_payload: Content evidence payload.

    Returns:
        Mask binding mapping or None.
    """
    if not isinstance(content_evidence_payload, dict):
        return None
    content_evidence_mapping = cast(Dict[str, Any], content_evidence_payload)
    mask_stats_node = content_evidence_mapping.get("mask_stats")
    if not isinstance(mask_stats_node, dict):
        return None
    mask_stats = cast(Dict[str, Any], mask_stats_node)
    binding = mask_stats.get("mask_resolution_binding")
    if isinstance(binding, dict):
        return cast(Dict[str, Any], binding)
    compat_binding = mask_stats.get("resolution_binding")
    if isinstance(compat_binding, dict):
        return cast(Dict[str, Any], compat_binding)
    return None


def _extract_mask_params_digest(content_evidence_payload: Any) -> str | None:
    """
    功能：提取 mask 参数摘要字段。 

    Extract mask params digest from content evidence payload.

    Args:
        content_evidence_payload: Content evidence payload.

    Returns:
        Mask params digest string or None.
    """
    if not isinstance(content_evidence_payload, dict):
        return None
    content_evidence_mapping = cast(Dict[str, Any], content_evidence_payload)
    mask_stats_node = content_evidence_mapping.get("mask_stats")
    if not isinstance(mask_stats_node, dict):
        return None
    mask_stats = cast(Dict[str, Any], mask_stats_node)
    value = mask_stats.get("mask_params_digest")
    if isinstance(value, str) and value:
        return value
    return None


def _bind_mask_and_routing_evidence_to_record(record_fields: Dict[str, Any], content_evidence_payload: Any) -> None:
    """
    功能：将 mask/routing 证据绑定到 record 顶层字段。 

    Bind mask and routing evidence fields to record for audit closure.

    Args:
        record_fields: Mutable record mapping.
        content_evidence_payload: Content evidence payload mapping.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return
    content_evidence_mapping = cast(Dict[str, Any], content_evidence_payload)
    mask_stats_node = content_evidence_mapping.get("mask_stats")
    if not isinstance(mask_stats_node, dict):
        return
    mask_stats = cast(Dict[str, Any], mask_stats_node)

    mask_binding = _extract_mask_binding(content_evidence_payload)
    if isinstance(mask_binding, dict):
        record_fields["mask_resolution_binding"] = mask_binding

    mask_source_impl_identity = mask_stats.get("mask_source_impl_identity")
    if isinstance(mask_source_impl_identity, dict):
        record_fields["mask_source_impl_identity"] = mask_source_impl_identity

    routing_digest = mask_stats.get("routing_digest")
    if isinstance(routing_digest, str) and routing_digest:
        record_fields["routing_digest"] = routing_digest

    routing_summary = mask_stats.get("routing_summary")
    if isinstance(routing_summary, dict):
        record_fields["routing_summary"] = routing_summary


def _should_use_latent_per_step_path(cfg: Dict[str, Any]) -> bool:
    """
    功能：判定 embed 是否走 latent per-step 注入主路径。

    Determine whether latent per-step path should be selected.

    Args:
        cfg: Configuration mapping.

    Returns:
        True if latent per-step mode is selected.
    """
    paper_spec_node = cfg.get("paper_faithfulness_spec")
    paper_spec_cfg = cast(Dict[str, Any], paper_spec_node) if isinstance(paper_spec_node, dict) else {}
    latent_space_per_step = paper_spec_cfg.get("latent_space_per_step")
    if isinstance(latent_space_per_step, bool):
        return latent_space_per_step
    paper_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_node) if isinstance(paper_node, dict) else {}
    if bool(paper_cfg.get("enabled", False)):
        return True
    embed_node = cfg.get("embed")
    embed_cfg = cast(Dict[str, Any], embed_node) if isinstance(embed_node, dict) else {}
    return bool(embed_cfg.get("use_latent_per_step", False))


def _build_latent_step_embed_trace(
    cfg: Dict[str, Any],
    injection_evidence: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    功能：构造 latent per-step 模式 embed trace。

    Build embed trace payload for latent per-step injection path.

    Args:
        cfg: Configuration mapping.
        injection_evidence: Injection evidence mapping.

    Returns:
        Embed trace mapping.

    Raises:
        None.
    """
    paper_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_node) if isinstance(paper_node, dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False))
    injection_status = injection_evidence.get("status") if isinstance(injection_evidence, dict) else None
    if paper_enabled and injection_status != "ok":
        # paper 模式下 latent 注入证据非 ok 必须立即失败。
        raise ValueError("latent injection evidence not ok under paper mode")

    trace: Dict[str, Any] = {
        "embed_mode": "latent_step_injection_v1",
        "identity_mode": False,
        "identity_reason": None,
        "note": "latent_per_step_injection_primary",
        "injection_status": "<absent>",
        "injection_trace_digest": "<absent>",
        "injection_params_digest": "<absent>",
    }
    if paper_enabled:
        trace["paper_faithfulness_required"] = True
    if isinstance(injection_evidence, dict):
        status_value = injection_evidence.get("status")
        trace_digest = injection_evidence.get("injection_trace_digest")
        params_digest = injection_evidence.get("injection_params_digest")
        absent_reason = injection_evidence.get("injection_absent_reason")
        failure_reason = injection_evidence.get("injection_failure_reason")
        lf_impl_binding = injection_evidence.get("lf_impl_binding")
        hf_impl_binding = injection_evidence.get("hf_impl_binding")
        if isinstance(status_value, str) and status_value:
            trace["injection_status"] = status_value
        if isinstance(trace_digest, str) and trace_digest:
            trace["injection_trace_digest"] = trace_digest
        if isinstance(params_digest, str) and params_digest:
            trace["injection_params_digest"] = params_digest
        if isinstance(absent_reason, str) and absent_reason:
            trace["injection_absent_reason"] = absent_reason
        if isinstance(failure_reason, str) and failure_reason:
            trace["injection_failure_reason"] = failure_reason
        if isinstance(lf_impl_binding, dict):
            trace["lf_impl_binding"] = lf_impl_binding
        if isinstance(hf_impl_binding, dict):
            trace["hf_impl_binding"] = hf_impl_binding
    elif paper_enabled:
        trace["injection_status"] = "absent"
        trace["injection_absent_reason"] = "paper_mode_requires_injection_evidence"
    return trace


def _get_ablation_normalized(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：读取 ablation.normalized 开关段。

    Read ablation.normalized switch settings from cfg.

    Args:
        cfg: Configuration mapping.

    Returns:
        ablation.normalized dict (empty if missing).

    Raises:
        TypeError: If cfg is invalid.
    """
    ablation_node = cfg.get("ablation")
    if not isinstance(ablation_node, dict):
        return {}
    ablation = cast(Dict[str, Any], ablation_node)
    normalized = ablation.get("normalized")
    if not isinstance(normalized, dict):
        return {}
    return cast(Dict[str, Any], normalized)


def _build_ablation_absent_subspace_result(absent_reason: str) -> Dict[str, Any]:
    """
    功能：构造 subspace 关闭时的 absent 规划结果。 

    Build subspace plan payload with explicit absent semantics.

    Args:
        absent_reason: Non-empty absent reason token.

    Returns:
        Mapping compatible with plan digest binding flow.
    """
    if not absent_reason:
        raise TypeError("absent_reason must be non-empty str")
    return {
        "status": "absent",
        "subspace_absent_reason": absent_reason,
        "plan": {},
        "plan_digest": None,
        "basis_digest": None,
        "plan_stats": {
            "planner_status": "absent",
            "planner_absent_reason": absent_reason,
        },
    }


def _build_ablation_absent_content_evidence(absent_reason: str) -> Dict[str, Any]:
    """
    功能：构造 ablation 禁用时的 content_evidence absent 语义。

    Build content_evidence with status="absent" for ablation-disabled modules.

    Args:
        absent_reason: Absence reason string (e.g., "content_chain_disabled_by_ablation").

    Returns:
        ContentEvidence-compatible dict with status="absent", score=None.

    Raises:
        TypeError: If absent_reason is invalid.
    """
    if not absent_reason:
        raise TypeError("absent_reason must be non-empty str")
    return {
        "status": "absent",
        "score": None,
        "audit": {
            "impl_identity": "ablation_switchboard",
            "impl_version": "v1",
            "impl_digest": digests.canonical_sha256({"impl_id": "ablation_switchboard", "impl_version": "v1"}),
            "trace_digest": digests.canonical_sha256({"absent_reason": absent_reason})
        },
        "mask_digest": None,
        "mask_stats": None,
        "plan_digest": None,
        "basis_digest": None,
        "lf_trace_digest": None,
        "hf_trace_digest": None,
        "lf_score": None,
        "hf_score": None,
        "score_parts": {
            "routing_digest": "<absent>",
            "routing_absent_reason": absent_reason,
        },
        "content_failure_reason": None  # absent 状态下无失败原因
    }


# ————————————————————————————
# Cryptographic generation attestation 扩展（附加函数，不修改原有 run_embed_orchestrator）
# ————————————————————————————

def build_embed_attestation(
    k_master: str,
    model_id: str,
    prompt: str,
    seed: int,
    plan_digest: str,
    *,
    k_prompt: str,
    k_seed: str,
    event_nonce: "str | None" = None,
    time_bucket: "str | None" = None,
    latent_snapshots: Sequence[Any] | None = None,
    use_trajectory_mix: bool = True,
) -> Dict[str, Any]:
    """
    功能：构造嵌入阶段的 generation attestation 载荷。

    Build a full generation attestation payload for embedding.
    Returns the attestation statement, attestation digest d_A,
    and derived keys for LF/HF/GEO/TR channels.

    Flow:
        prompt_commit = HMAC(k_prompt, normalize(prompt))
        seed_commit   = HMAC(k_seed, seed)
        statement     = build_attestation_statement(...)
        d_A           = SHA256(CanonicalJSON(statement))
        keys          = HKDF(K_master, d_A, context=*)
        lf_payload    = HMAC(k_LF, d_A)[:48]
        [trace_commit = HMAC(k_TR, trace_summary)]  # 若提供 latent_snapshots
        [payload_final = mix(lf_payload, trace_commit)]

    Args:
        k_master: Master key (hex str or bytes). Kept secret by caller.
        model_id: Generative model identifier (e.g., "sd3.5").
        prompt: Raw generation prompt (will be committed, not stored).
        seed: Integer generation seed (will be committed, not stored).
        plan_digest: SubspacePlanner plan digest for binding.
        k_prompt: Key for prompt HMAC commitment (hex str or bytes).
        k_seed: Key for seed HMAC commitment (hex str or bytes).
        event_nonce: Optional per-event nonce (uuid4() if None).
        time_bucket: Optional date bucket (today if None).
        latent_snapshots: Optional list of latent snapshots for trajectory commit.
        use_trajectory_mix: Whether to mix trace_commit into lf_payload (default True).

    Returns:
        Dict with:
        - "statement": AttestationStatement as dict.
        - "attestation_digest": d_A hex string.
        - "keys": dict with k_lf, k_hf, k_geo, k_tr as hex strings.
        - "lf_payload": bytes of LF attestation payload.
        - "lf_payload_hex": hex string of lf_payload.
        - "trace_commit": hex string if latent_snapshots provided, else None.
        - "geo_anchor_seed": int seed for geometry chain.
        - "attestation_status": "ok".

    Raises:
        TypeError: If required inputs have wrong type.
        ValueError: If required inputs are empty.
    """
    from main.watermarking.provenance.commitments import (
        compute_prompt_commit,
        compute_seed_commit,
    )
    from main.watermarking.provenance.attestation_statement import (
        build_attestation_statement,
        compute_attestation_digest,
        build_signed_attestation_bundle,
    )
    from main.watermarking.provenance.key_derivation import (
        derive_attestation_keys,
        compute_lf_attestation_payload,
        derive_geo_anchor_seed,
    )
    from main.watermarking.provenance.trajectory_commit import (
        compute_trajectory_commit,
        mix_payload_with_trace_commit,
    )

    # (1) 计算 prompt 和 seed 的 HMAC 承诺。
    prompt_commit = compute_prompt_commit(k_prompt, prompt)
    seed_commit = compute_seed_commit(k_seed, seed)

    # (2) 构造 generation attestation statement（不含版本字段）。
    statement = build_attestation_statement(
        model_id=model_id,
        prompt_commit=prompt_commit,
        seed_commit=seed_commit,
        plan_digest=plan_digest,
        event_nonce=event_nonce,
        time_bucket=time_bucket,
    )

    # (3) 计算 attestation digest d_A。
    d_a = compute_attestation_digest(statement)

    # (4) 先从 statement digest 派生 k_TR，用于计算 trajectory_commit。
    trace_keys = derive_attestation_keys(k_master, d_a)

    # (5) 计算轨迹承诺，再用 statement + trajectory_commit 联合派生事件密钥。
    trace_commit: "str | None" = None
    if latent_snapshots is not None and len(latent_snapshots) > 0:
        try:
            trace_commit = compute_trajectory_commit(trace_keys.k_tr, latent_snapshots)
        except Exception:
            # 轨迹承诺失败时降级为 statement-only 事件绑定，不中断主流程。
            trace_commit = None

    attest_keys = derive_attestation_keys(k_master, d_a, trajectory_commit=trace_commit)

    # (6) 生成以 event_binding_digest 为输入域的 LF 主通道 attestation payload。
    lf_payload = compute_lf_attestation_payload(
        k_lf=attest_keys.k_lf,
        attestation_digest=attest_keys.event_binding_digest,
        payload_length=48,
    )
    lf_payload_final = lf_payload
    if trace_commit is not None and use_trajectory_mix:
        lf_payload_final = mix_payload_with_trace_commit(
            lf_payload=lf_payload,
            trace_commit_hex=trace_commit,
            mix_bytes=8,
        )

    # (7) 派生几何链 anchor seed。
    geo_anchor_seed = derive_geo_anchor_seed(attest_keys.k_geo)

    return {
        "statement": statement.as_dict(),
        "attestation_digest": d_a,
        "event_binding_digest": attest_keys.event_binding_digest,
        "signed_bundle": build_signed_attestation_bundle(
            statement,
            d_a,
            k_master,
            lf_payload_hex=lf_payload_final.hex(),
            trace_commit=trace_commit,
            geo_anchor_seed=geo_anchor_seed,
        ),
        "keys": {
            "k_lf": attest_keys.k_lf,
            "k_hf": attest_keys.k_hf,
            "k_geo": attest_keys.k_geo,
            "k_tr": attest_keys.k_tr,
            "event_binding_digest": attest_keys.event_binding_digest,
        },
        "lf_payload": lf_payload_final,
        "lf_payload_hex": lf_payload_final.hex(),
        "trace_commit": trace_commit,
        "geo_anchor_seed": geo_anchor_seed,
        "attestation_status": "ok",
    }
