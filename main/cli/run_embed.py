"""
嵌入水印 CLI 入口

功能说明：
- 规范化输出目录路径，确保输出布局，加载合同与白名单，验证配置，解析实现，构造记录，绑定字段，写盘，并产出闭包。
- 包含详细的输入验证、错误处理与状态更新机制，确保健壮性与可维护性。
- 当前为可审计基线实现，新能力仅通过版本化追加，不改变既有冻结语义与字段口径。
"""

import sys
import argparse
import copy
import gc
from pathlib import Path
import uuid
from typing import Any, Callable, Dict, cast

from main.cli import assert_module_execution


assert_module_execution("run_embed")

from main.core import time_utils
from main.core import digests
from main.core.contracts import (
    load_frozen_contracts,
    bind_contract_to_record,
    get_contract_interpretation,
    FrozenContracts
)
from main.policy.runtime_whitelist import (
    load_runtime_whitelist,
    load_policy_path_semantics,
    assert_consistent_with_semantics,
    bind_whitelist_to_record,
    bind_semantics_to_record
)
from main.core import records_io
from main.core import config_loader
from main.core import schema
from main.core import status
from main.policy import path_policy
from main.core.errors import RunFailureReason
from main.registries import runtime_resolver, pipeline_registry
from main.diffusion.sd3 import pipeline_factory
from main.diffusion.sd3 import infer_runtime
from main.diffusion.sd3 import infer_trace
from main.diffusion.sd3 import pipeline_inspector
from main.watermarking.embed import orchestrator as embed_orchestrator
from main.watermarking.paper_faithfulness import injection_site_binder
from main.watermarking.paper_faithfulness import alignment_evaluator
from main.watermarking.embed.orchestrator import run_embed_orchestrator
from main.watermarking.geometry_chain.sync import SyncRuntimeContext, resolve_enable_latent_sync
from main.watermarking.content_chain.latent_modifier import (
    LatentModifier,
    LATENT_MODIFIER_ID,
    LATENT_MODIFIER_VERSION
)
from main.cli.run_common import (
    set_failure_status,
    format_fact_sources_mismatch,
    bind_impl_identity_fields as _bind_impl_identity_fields,
    build_seed_audit,
    build_determinism_controls,
    normalize_nondeterminism_notes,
    build_injection_context_from_plan,
    build_cli_config_migration_hint,
    resolve_attestation_env_inputs,
)


_build_content_inputs_for_embed = cast(
    Callable[[Dict[str, Any]], Dict[str, Any] | None],
    getattr(embed_orchestrator, "_build_content_inputs_for_embed"),
)
_build_planner_inputs_for_runtime = cast(
    Callable[..., Dict[str, Any]],
    getattr(embed_orchestrator, "_build_planner_inputs_for_runtime"),
)


EMBED_CONTENT_RUNTIME_PHASE_PRECOMPUTE = "embed_precompute"
PREVIEW_GENERATION_RUNTIME_PHASE_LABEL = "preview_generation"
STATEMENT_ONLY_RUNTIME_CAPTURE_PHASE_LABEL = "statement_only_runtime_capture"
EMBED_WATERMARKED_INFERENCE_PHASE_LABEL = "embed_watermarked_inference"
PREVIEW_GENERATION_RECORD_FILE_NAME = "preview_generation_record.json"
_PREVIEW_META_OMIT = object()


def _clone_preview_meta_value(value: Any) -> Any:
    """
    功能：递归复制 preview meta 可保留的轻量值。 

    Recursively clone lightweight preview-metadata values while dropping
    preview-only heavy objects.

    Args:
        value: Candidate metadata value.

    Returns:
        Cloned lightweight value, or an internal omit sentinel.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        cloned_mapping: Dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                continue
            cloned_value = _clone_preview_meta_value(nested_value)
            if cloned_value is _PREVIEW_META_OMIT:
                continue
            cloned_mapping[key] = cloned_value
        return cloned_mapping
    if isinstance(value, (list, tuple)):
        cloned_items = []
        for item in value:
            cloned_item = _clone_preview_meta_value(item)
            if cloned_item is _PREVIEW_META_OMIT:
                continue
            cloned_items.append(cloned_item)
        return cloned_items
    return _PREVIEW_META_OMIT


def _clone_preview_meta_mapping(mapping: Any) -> Dict[str, Any] | None:
    """
    功能：将 preview meta 子映射复制为轻量副本。 

    Clone a preview-metadata mapping into a lightweight detached copy.

    Args:
        mapping: Candidate metadata mapping.

    Returns:
        Detached lightweight mapping, or None when unavailable.
    """
    if not isinstance(mapping, dict):
        return None
    cloned_mapping = _clone_preview_meta_value(mapping)
    if not isinstance(cloned_mapping, dict):
        return None
    return cast(Dict[str, Any], cloned_mapping)


def _shrink_preview_generation_meta_payloads(preview_generation_meta: Dict[str, Any]) -> None:
    """
    功能：收紧 preview generation meta 中的运行期子载荷。 

    Shrink preview-generation runtime payloads to detached lightweight copies.

    Args:
        preview_generation_meta: Mutable preview-generation metadata mapping.

    Returns:
        None.
    """
    if not isinstance(preview_generation_meta, dict):
        return
    preview_generation_meta["pipeline_runtime_meta"] = _clone_preview_meta_mapping(
        preview_generation_meta.get("pipeline_runtime_meta")
    )
    preview_generation_meta["inference_runtime_meta"] = _clone_preview_meta_mapping(
        preview_generation_meta.get("inference_runtime_meta")
    )


def _resolve_embed_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析 embed 配置节点为字典映射。

    Resolve cfg["embed"] into a typed mapping.

    Args:
        cfg: Configuration mapping.

    Returns:
        Embed config mapping or empty dict.
    """
    embed_node = cfg.get("embed")
    return cast(Dict[str, Any], embed_node) if isinstance(embed_node, dict) else {}


def _resolve_preview_generation_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析 preview_generation 配置节点为字典映射。

    Resolve embed.preview_generation into a typed mapping.

    Args:
        cfg: Configuration mapping.

    Returns:
        Preview-generation config mapping or empty dict.
    """
    embed_cfg = _resolve_embed_cfg(cfg)
    preview_node = embed_cfg.get("preview_generation")
    return cast(Dict[str, Any], preview_node) if isinstance(preview_node, dict) else {}


def _normalize_preview_generation_artifact_rel_path(cfg: Dict[str, Any]) -> str | None:
    """
    功能：规范化 preview artifact 的相对路径。 

    Normalize the configured preview artifact relative path.

    Args:
        cfg: Configuration mapping.

    Returns:
        Artifact path relative to artifacts/, or None when absent.
    """
    preview_cfg = _resolve_preview_generation_cfg(cfg)
    artifact_rel_path = preview_cfg.get("artifact_rel_path")
    if not isinstance(artifact_rel_path, str):
        return None
    normalized_artifact_rel_path = artifact_rel_path.strip().replace("\\", "/")
    if not normalized_artifact_rel_path:
        return None
    return normalized_artifact_rel_path


def _resolve_preview_generation_record_rel_path(artifact_rel_path: str | None) -> str:
    """
    功能：从 preview artifact 相对路径派生 preview record 相对路径。 

    Derive the preview-generation record relative path from the configured
    preview artifact path.

    Args:
        artifact_rel_path: Artifact path relative to artifacts/, or None.

    Returns:
        Record path relative to artifacts/.
    """
    if artifact_rel_path is None:
        return PREVIEW_GENERATION_RECORD_FILE_NAME
    preview_rel_path = Path(artifact_rel_path)
    preview_parent = preview_rel_path.parent.as_posix()
    if preview_parent in {"", "."}:
        return PREVIEW_GENERATION_RECORD_FILE_NAME
    return f"{preview_parent}/{PREVIEW_GENERATION_RECORD_FILE_NAME}"


def _write_preview_generation_record(
    *,
    run_root: Path,
    artifacts_dir: Path,
    record_path: Path,
    payload: Dict[str, Any],
) -> None:
    """
    功能：写出 preview_generation 结构化记录工件。 

    Persist the structured preview-generation record artifact.

    Args:
        run_root: Run root directory.
        artifacts_dir: Artifacts directory.
        record_path: Target preview-generation record path.
        payload: Structured record payload.

    Returns:
        None.
    """
    record_path.parent.mkdir(parents=True, exist_ok=True)
    path_policy.validate_output_target(record_path, "artifact", run_root)
    records_io.write_artifact_json_unbound(
        run_root,
        artifacts_dir,
        str(record_path),
        payload,
    )


def _build_preview_generation_meta(
    *,
    cfg: Dict[str, Any],
    run_root: Path,
    artifacts_dir: Path,
    pipeline_result: Dict[str, Any],
    seed_value: int | None,
) -> Dict[str, Any]:
    """
    功能：执行并持久化 preview_generation 的正式工件与结构化元数据。 

    Execute preview generation, persist the authoritative preview artifact, and
    return structured observability metadata.

    Args:
        cfg: Mutable configuration mapping.
        run_root: Run root directory.
        artifacts_dir: Artifacts directory.
        pipeline_result: Pipeline build result mapping.
        seed_value: Effective runtime seed.

    Returns:
        Structured preview-generation metadata mapping.
    """
    preview_cfg = _resolve_preview_generation_cfg(cfg)
    preview_enabled = bool(preview_cfg.get("enabled", False))
    artifact_rel_path = _normalize_preview_generation_artifact_rel_path(cfg)
    record_rel_path = _resolve_preview_generation_record_rel_path(artifact_rel_path)
    preview_artifact_path = artifacts_dir / Path(artifact_rel_path) if artifact_rel_path is not None else None
    preview_record_path = artifacts_dir / Path(record_rel_path)
    preview_device = cfg.get("device", "cpu")

    preview_meta: Dict[str, Any] = {
        "enabled": preview_enabled,
        "status": "skipped",
        "reason": None,
        "exception_type": None,
        "exception_message": None,
        "pipeline_status": pipeline_result.get("pipeline_status"),
        "pipeline_error": pipeline_result.get("pipeline_error"),
        "pipeline_runtime_meta": _clone_preview_meta_mapping(pipeline_result.get("pipeline_runtime_meta")),
        "pipeline_provenance_canon_sha256": pipeline_result.get("pipeline_provenance_canon_sha256"),
        "model_provenance_canon_sha256": pipeline_result.get("model_provenance_canon_sha256"),
        "inference_status": None,
        "inference_error": None,
        "inference_runtime_meta": None,
        "output_image_present": False,
        "requested_artifact_rel_path": artifact_rel_path,
        "requested_artifact_path": str(preview_artifact_path) if preview_artifact_path is not None else None,
        "persisted_artifact_path": None,
        "persisted_artifact_rel_path": None,
        "persisted_artifact_sha256": None,
        "record_path": str(preview_record_path),
        "record_rel_path": record_rel_path,
        "seed": seed_value,
        "prompt": cfg.get("inference_prompt"),
        "device": preview_device,
        "creation_mode": "prompt_conditioned_preview",
    }

    if not preview_enabled:
        preview_meta["reason"] = "preview_generation_disabled"
        return preview_meta

    existing_input_image_path = cfg.get("__embed_input_image_path__")
    if isinstance(existing_input_image_path, str) and existing_input_image_path.strip():
        preview_meta["reason"] = "input_image_already_available"
        preview_meta["persisted_artifact_path"] = existing_input_image_path.strip()
        return preview_meta

    if preview_artifact_path is None:
        preview_meta["status"] = "failed"
        preview_meta["reason"] = "preview_generation_artifact_rel_path_missing"
        _write_preview_generation_record(
            run_root=run_root,
            artifacts_dir=artifacts_dir,
            record_path=preview_record_path,
            payload=preview_meta,
        )
        return preview_meta

    preview_pipeline_obj = pipeline_result.get("pipeline_obj")
    preview_infer_result: Dict[str, Any] | None = None
    preview_image: Any = None
    try:
        preview_infer_result = infer_runtime.run_sd3_inference(
            cfg,
            preview_pipeline_obj,
            preview_device,
            seed_value,
            runtime_phase_label=PREVIEW_GENERATION_RUNTIME_PHASE_LABEL,
            injection_context=None,
            injection_modifier=None,
            capture_final_latents=False,
        )
        preview_status = preview_infer_result.get("inference_status")
        if not isinstance(preview_status, str) or not preview_status:
            preview_status = preview_infer_result.get("status")
        if not isinstance(preview_status, str) or not preview_status:
            preview_status = infer_runtime.INFERENCE_STATUS_FAILED

        preview_meta["inference_status"] = preview_status
        preview_meta["inference_error"] = preview_infer_result.get("inference_error")
        preview_meta["inference_runtime_meta"] = _clone_preview_meta_mapping(
            preview_infer_result.get("inference_runtime_meta")
        )

        preview_image = preview_infer_result.get("output_image")
        preview_meta["output_image_present"] = preview_image is not None
        if preview_status == infer_runtime.INFERENCE_STATUS_OK and preview_image is not None:
            preview_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            path_policy.validate_output_target(preview_artifact_path, "artifact", run_root)
            preview_image.save(str(preview_artifact_path), format="PNG")
            preview_meta["status"] = "ok"
            preview_meta["persisted_artifact_path"] = str(preview_artifact_path)
            preview_meta["persisted_artifact_rel_path"] = artifact_rel_path
            preview_meta["persisted_artifact_sha256"] = digests.file_sha256(preview_artifact_path)
            cfg["__embed_input_image_path__"] = str(preview_artifact_path)
            print(f"[Preview Generation] 预览图已生成，路径：{preview_artifact_path}")
        elif preview_status == infer_runtime.INFERENCE_STATUS_OK:
            preview_meta["status"] = "failed"
            preview_meta["reason"] = "preview_inference_no_output_image"
            print("[Preview Generation] 推理成功但无 output_image，跳过。")
        else:
            preview_meta["status"] = "failed"
            inference_error = preview_infer_result.get("inference_error")
            if isinstance(inference_error, str) and inference_error:
                preview_meta["reason"] = inference_error
            else:
                preview_meta["reason"] = f"preview_inference_status={preview_status}"
            print(f"[Preview Generation] 推理状态非 ok（{preview_status}），跳过。")
    except Exception as exc:
        preview_meta["status"] = "failed"
        preview_meta["reason"] = "preview_generation_exception"
        preview_meta["exception_type"] = type(exc).__name__
        preview_meta["exception_message"] = str(exc)
        print(f"[Preview Generation] 推理异常：{exc}，跳过。")

    try:
        _shrink_preview_generation_meta_payloads(preview_meta)
        _write_preview_generation_record(
            run_root=run_root,
            artifacts_dir=artifacts_dir,
            record_path=preview_record_path,
            payload=preview_meta,
        )
        if preview_meta["status"] != "ok":
            cfg.pop("__embed_input_image_path__", None)
    finally:
        _release_preview_generation_transients(
            preview_infer_result=preview_infer_result,
            preview_image=preview_image,
        )
        preview_image = None
        preview_infer_result = None
        preview_pipeline_obj = None
    return preview_meta


def _release_preview_generation_transients(
    *,
    preview_infer_result: Dict[str, Any] | None,
    preview_image: Any,
) -> None:
    """
    功能：释放 preview generation 已完成使命的局部对象。 

    Release preview-only transient objects after the authoritative preview
    artifact and structured record have been persisted.

    Args:
        preview_infer_result: Preview inference result mapping.
        preview_image: Preview image object.

    Returns:
        None.
    """
    image_to_close = preview_image
    if image_to_close is None and isinstance(preview_infer_result, dict):
        image_to_close = preview_infer_result.get("output_image")

    if image_to_close is not None:
        close_method = getattr(image_to_close, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass

    if isinstance(preview_infer_result, dict):
        runtime_meta = preview_infer_result.get("inference_runtime_meta")
        if isinstance(runtime_meta, dict):
            runtime_meta.clear()
        trajectory_evidence = preview_infer_result.get("trajectory_evidence")
        if isinstance(trajectory_evidence, dict):
            trajectory_evidence.clear()
        injection_evidence = preview_infer_result.get("injection_evidence")
        if isinstance(injection_evidence, dict):
            injection_evidence.clear()
        trajectory_cache_capture_meta = preview_infer_result.get("trajectory_cache_capture_meta")
        if isinstance(trajectory_cache_capture_meta, dict):
            trajectory_cache_capture_meta.clear()
        preview_infer_result.pop("output_image", None)
        preview_infer_result.pop("final_latents", None)
        preview_infer_result.pop("runtime_self_attention_maps", None)
        preview_infer_result.pop("trajectory_evidence", None)
        preview_infer_result.pop("injection_evidence", None)
        preview_infer_result.pop("trajectory_cache_capture_meta", None)
        preview_infer_result.pop("inference_runtime_meta", None)
        preview_infer_result.clear()


def _release_preview_generation_runtime_pressure(
    cfg: Dict[str, Any],
    preview_generation_meta: Dict[str, Any] | None,
    *,
    statement_only_formal_path: bool,
) -> None:
    """
    功能：在 preview generation 与 statement_only second-pass 之间执行最小显存整理。 

    Apply minimal runtime-pressure cleanup between preview generation and the
    statement-only second pass.

    Args:
        cfg: Configuration mapping.
        preview_generation_meta: Structured preview-generation metadata.
        statement_only_formal_path: Whether the current run requires the
            statement-only second pass.

    Returns:
        None.
    """
    if not isinstance(cfg, dict):
        return
    if not statement_only_formal_path:
        return
    if not isinstance(preview_generation_meta, dict):
        return

    inference_status = preview_generation_meta.get("inference_status")
    if not isinstance(inference_status, str) or not inference_status:
        return

    runtime_device = cfg.get("device")
    if not isinstance(runtime_device, str) or not runtime_device.lower().startswith("cuda"):
        return

    _shrink_preview_generation_meta_payloads(preview_generation_meta)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _validate_real_pipeline_identity_fields(cfg: Dict[str, Any], config_path: str) -> None:
    """
    功能：在 embed 入口前置校验 real SD3 pipeline 所需身份字段。

    Validate the required identity fields for the real SD3 pipeline before
    pipeline-shell construction starts.

    Args:
        cfg: Runtime configuration mapping.
        config_path: Runtime config path string used for error reporting.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If the real pipeline requires explicit identity fields that are absent or invalid.
    """
    cfg_obj: Any = cfg
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg must be dict")
    cfg_mapping = cast(Dict[str, Any], cfg_obj)
    config_path_value: Any = config_path
    if not isinstance(config_path_value, str) or not config_path_value:
        raise TypeError("config_path must be non-empty str")

    pipeline_impl_id = cfg_mapping.get("pipeline_impl_id")
    if pipeline_impl_id is None:
        pipeline_node = cfg_mapping.get("pipeline")
        if isinstance(pipeline_node, dict):
            pipeline_impl_id = cast(Dict[str, Any], pipeline_node).get("pipeline_impl_id")
    if pipeline_impl_id != pipeline_registry.SD3_DIFFUSERS_REAL_ID:
        return

    paper_cfg_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_cfg_node) if isinstance(paper_cfg_node, dict) else {}
    build_required = bool(paper_cfg.get("enabled", False))
    if not build_required:
        pipeline_build_enabled = cfg.get("pipeline_build_enabled", True)
        if not isinstance(pipeline_build_enabled, bool):
            raise ValueError("pipeline_build_enabled must be bool")
        build_required = pipeline_build_enabled
    if not build_required:
        return

    model_source = cfg.get("model_source")
    allowed_model_sources = {"hf", "hf_hub", "local", "local_path"}
    if not isinstance(model_source, str) or model_source not in allowed_model_sources:
        raise ValueError(
            "real SD3 pipeline requires explicit config field model_source in "
            f"{config_path}; allowed values={sorted(allowed_model_sources)}"
        )

    hf_revision = cfg.get("hf_revision")
    if not isinstance(hf_revision, str) or not hf_revision.strip():
        raise ValueError(
            "real SD3 pipeline requires explicit non-empty config field hf_revision in "
            f"{config_path}"
        )


def _normalize_plan_digest(value: Any) -> str | None:
    """
    功能：将候选 plan digest 规范化为非空字符串。 

    Normalize a candidate plan digest into a non-empty string.

    Args:
        value: Candidate digest value.

    Returns:
        Normalized digest string, or None when the input is absent.
    """
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    if not normalized_value or normalized_value == "<absent>":
        return None
    return normalized_value


def _as_dict_payload(value: Any) -> Dict[str, Any] | None:
    """
    功能：将载荷对象规范化为 dict。 

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


def _requires_statement_only_formal_precompute(cfg: Dict[str, Any]) -> bool:
    """
    功能：判断当前是否属于 statement_only 两阶段 formal object 正式路径。 

    Determine whether the current run requires the two-stage statement-only
    formal object flow.

    Args:
        cfg: Configuration mapping.

    Returns:
        True when paper formal path requires statement-only scaffold plus
        runtime finalization.
    """
    paper_cfg_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_cfg_node) if isinstance(paper_cfg_node, dict) else {}
    if not bool(paper_cfg.get("enabled", False)):
        return False
    attestation_cfg_node = cfg.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_cfg_node) if isinstance(attestation_cfg_node, dict) else {}
    use_trajectory_mix = attestation_cfg.get("use_trajectory_mix")
    return isinstance(use_trajectory_mix, bool) and not use_trajectory_mix


def _resolve_subspace_precompute_failure_reason(subspace_result: Any) -> str:
    """
    功能：提取 precompute planner 缺失 formal plan 时的主失败原因。 

    Resolve the primary failure reason for a missing precomputed formal plan.

    Args:
        subspace_result: Precomputed planner result object or mapping.

    Returns:
        Structured failure-reason string.
    """
    payload = _as_dict_payload(subspace_result)
    if isinstance(payload, dict):
        reason = payload.get("plan_failure_reason")
        if isinstance(reason, str) and reason:
            return reason
        plan_stats = payload.get("plan_stats")
        if isinstance(plan_stats, dict):
            plan_stats_dict = cast(Dict[str, Any], plan_stats)
            for key_name in ["planner_absent_reason", "planner_failure_reason"]:
                stats_reason = plan_stats_dict.get(key_name)
                if isinstance(stats_reason, str) and stats_reason:
                    return stats_reason
        status_value = payload.get("status")
        if isinstance(status_value, str) and status_value:
            return f"precompute_status={status_value}"
    direct_reason = getattr(subspace_result, "plan_failure_reason", None)
    if isinstance(direct_reason, str) and direct_reason:
        return direct_reason
    status_value = getattr(subspace_result, "status", None)
    if isinstance(status_value, str) and status_value:
        return f"precompute_status={status_value}"
    return "precomputed_formal_plan_incomplete"


def _build_statement_only_formal_scaffold(
    cfg: Dict[str, Any],
    cfg_digest: str | None,
    seed_value: int | None,
    content_result_pre_payload: Dict[str, Any] | None,
    planner_inputs: Dict[str, Any] | None,
) -> tuple[Dict[str, Any] | None, str | None]:
    """
    功能：基于 pre-inference 静态输入域构造 statement_only formal scaffold。 

    Build the pre-inference scaffold for the statement-only formal path.

    Args:
        cfg: Configuration mapping.
        cfg_digest: Canonical config digest.
        seed_value: Resolved runtime seed.
        content_result_pre_payload: Pre-content payload.
        planner_inputs: Planner input mapping built before runtime finalization.

    Returns:
        Tuple of (formal_scaffold, failure_reason).
    """
    if not isinstance(content_result_pre_payload, dict):
        return None, "scaffold_content_payload_absent"
    if not isinstance(planner_inputs, dict):
        return None, "scaffold_planner_inputs_absent"

    mask_digest = _normalize_plan_digest(content_result_pre_payload.get("mask_digest"))
    if mask_digest is None:
        return None, "scaffold_mask_digest_absent"

    trace_signature = planner_inputs.get("trace_signature")
    if not isinstance(trace_signature, dict) or not trace_signature:
        return None, "scaffold_trace_signature_absent"

    mask_summary = planner_inputs.get("mask_summary")
    if not isinstance(mask_summary, dict) or not mask_summary:
        return None, "scaffold_mask_summary_absent"

    routing_digest = planner_inputs.get("routing_digest")
    if not isinstance(routing_digest, str) or not routing_digest:
        return None, "scaffold_routing_digest_absent"

    formal_scaffold: Dict[str, Any] = {
        "formal_object_stage": "pre_inference_scaffold",
        "formal_scaffold_version": "v1",
        "event_binding_mode": "statement_only",
        "policy_path": cfg.get("policy_path"),
        "cfg_digest": cfg_digest,
        "model_id": cfg.get("model_id"),
        "seed": seed_value,
        "prompt": cfg.get("inference_prompt"),
        "trace_signature": dict(cast(Dict[str, Any], trace_signature)),
        "mask_digest": mask_digest,
        "mask_summary": dict(cast(Dict[str, Any], mask_summary)),
        "routing_digest": routing_digest,
    }
    formal_scaffold["formal_scaffold_digest"] = digests.canonical_sha256(formal_scaffold)
    return formal_scaffold, None


def _resolve_runtime_executable_plan_failure_reason(
    subspace_result: Any,
    runtime_capture_status: str | None = None,
) -> str:
    """
    功能：解析 runtime executable formal plan finalization 的主失败原因。 

    Resolve the primary failure reason for runtime executable-plan finalization.

    Args:
        subspace_result: Finalization planner result object or mapping.
        runtime_capture_status: Optional runtime capture inference status.

    Returns:
        Structured failure-reason string.
    """
    reason = _resolve_subspace_precompute_failure_reason(subspace_result)
    if reason != "precomputed_formal_plan_incomplete":
        return reason
    if isinstance(runtime_capture_status, str) and runtime_capture_status and runtime_capture_status != infer_runtime.INFERENCE_STATUS_OK:
        return f"runtime_capture_status={runtime_capture_status}"
    return "runtime_executable_formal_plan_incomplete"


def _extract_subspace_plan_payload(subspace_result: Any) -> Dict[str, Any] | None:
    """
    功能：提取 subspace 结果中的内层 executable plan 负载。 

    Extract the inner executable plan payload from a subspace result.

    Args:
        subspace_result: Subspace result object or mapping.

    Returns:
        Plan payload mapping when available; otherwise None.
    """
    plan_candidate = getattr(subspace_result, "plan", None)
    if isinstance(plan_candidate, dict):
        return cast(Dict[str, Any], plan_candidate)
    if hasattr(subspace_result, "as_dict"):
        payload_candidate = subspace_result.as_dict()
        if isinstance(payload_candidate, dict):
            payload = cast(Dict[str, Any], payload_candidate)
            plan_node = payload.get("plan")
            if isinstance(plan_node, dict):
                return cast(Dict[str, Any], plan_node)
            return payload
    if isinstance(subspace_result, dict):
        payload = cast(Dict[str, Any], subspace_result)
        plan_node = payload.get("plan")
        if isinstance(plan_node, dict):
            return cast(Dict[str, Any], plan_node)
        return payload
    return None


def _extract_subspace_failure_diagnostics(subspace_result: Any) -> Dict[str, Any]:
    """
    功能：提取 planner 失败诊断字段。 

    Extract structured planner failure diagnostics from a subspace result.

    Args:
        subspace_result: Planner result object or mapping.

    Returns:
        Mapping with planner failure stage/detail/context fields.
    """
    payload = _as_dict_payload(subspace_result)

    def _resolve_value(field_name: str) -> Any:
        if isinstance(payload, dict) and field_name in payload:
            return payload.get(field_name)
        return getattr(subspace_result, field_name, None)

    planner_diagnostic_context = _resolve_value("planner_diagnostic_context")
    if not isinstance(planner_diagnostic_context, dict):
        planner_diagnostic_context = None

    return {
        "planner_failure_stage": _resolve_value("planner_failure_stage"),
        "planner_failure_detail_code": _resolve_value("planner_failure_detail_code"),
        "planner_failure_detail_message": _resolve_value("planner_failure_detail_message"),
        "planner_diagnostic_context": dict(cast(Dict[str, Any], planner_diagnostic_context)) if isinstance(planner_diagnostic_context, dict) else None,
    }


def _extract_runtime_capture_diagnostics(
    runtime_capture_result: Any,
    runtime_capture_cache: Any,
) -> Dict[str, Any]:
    """
    功能：提取 runtime trajectory cache 的结构化捕获诊断。 

    Extract structured trajectory-cache capture diagnostics from the runtime
    inference result.

    Args:
        runtime_capture_result: Runtime inference result mapping.
        runtime_capture_cache: In-memory trajectory cache object.

    Returns:
        Mapping with normalized capture diagnostics.
    """
    def _reconstruct_capture_payload(payload_source: Any) -> Dict[str, Any] | None:
        if not isinstance(payload_source, dict):
            return None
        payload_mapping = cast(Dict[str, Any], payload_source)
        nested_payload = payload_mapping.get("trajectory_cache_capture")
        if isinstance(nested_payload, dict):
            return dict(cast(Dict[str, Any], nested_payload))
        reconstructed_payload = {
            field_name: payload_mapping.get(field_name)
            for field_name in [
                "trajectory_cache_capture_status",
                "trajectory_cache_step_count",
                "trajectory_cache_capture_attempt_count",
                "trajectory_cache_capture_success_count",
                "trajectory_cache_capture_failure_count",
                "trajectory_cache_capture_failure_examples",
                "trajectory_cache_available_steps",
                "trajectory_cache_required_step_count",
                "trajectory_cache_missing_required_steps",
                "trajectory_cache_callback_invocation_count",
                "trajectory_cache_callback_latent_present_count",
                "trajectory_cache_tap_captured_step_count",
                "trajectory_cache_capture_detail_code",
                "trajectory_cache_capture_detail_message",
                "trajectory_cache_capture_error_message",
            ]
            if field_name in payload_mapping
        }
        if not reconstructed_payload:
            return None
        if all(value is None for value in reconstructed_payload.values()):
            return None
        return reconstructed_payload

    def _extract_tap_captured_step_count(payload_source: Any) -> int | None:
        if not isinstance(payload_source, dict):
            return None
        payload_mapping = cast(Dict[str, Any], payload_source)
        trajectory_evidence = payload_mapping.get("trajectory_evidence")
        if not isinstance(trajectory_evidence, dict):
            return None
        trajectory_metrics = trajectory_evidence.get("trajectory_metrics")
        if isinstance(trajectory_metrics, dict):
            steps_candidate = trajectory_metrics.get("steps")
            if isinstance(steps_candidate, list):
                return len(steps_candidate)
        trajectory_stats = trajectory_evidence.get("trajectory_stats")
        if isinstance(trajectory_stats, dict):
            steps_candidate = trajectory_stats.get("steps")
            if isinstance(steps_candidate, list):
                return len(steps_candidate)
        return None

    capture_payload: Dict[str, Any] | None = None
    inference_runtime_meta: Dict[str, Any] | None = None
    tap_captured_step_count_from_evidence: int | None = None
    runtime_capture_inference_status: str | None = None
    runtime_capture_inference_error: str | None = None
    if isinstance(runtime_capture_result, dict):
        runtime_capture_result_mapping = cast(Dict[str, Any], runtime_capture_result)
        runtime_capture_inference_status_candidate = runtime_capture_result_mapping.get("inference_status")
        if isinstance(runtime_capture_inference_status_candidate, str) and runtime_capture_inference_status_candidate:
            runtime_capture_inference_status = runtime_capture_inference_status_candidate
        runtime_capture_inference_error_candidate = runtime_capture_result_mapping.get("inference_error")
        if isinstance(runtime_capture_inference_error_candidate, str) and runtime_capture_inference_error_candidate:
            runtime_capture_inference_error = runtime_capture_inference_error_candidate
        inference_runtime_meta_candidate = runtime_capture_result_mapping.get("inference_runtime_meta")
        if isinstance(inference_runtime_meta_candidate, dict):
            inference_runtime_meta = dict(cast(Dict[str, Any], inference_runtime_meta_candidate))
        capture_node = runtime_capture_result_mapping.get("trajectory_cache_capture_meta")
        if isinstance(capture_node, dict):
            capture_payload = dict(cast(Dict[str, Any], capture_node))
        if capture_payload is None:
            capture_payload = _reconstruct_capture_payload(runtime_capture_result_mapping)
        tap_captured_step_count_from_evidence = _extract_tap_captured_step_count(runtime_capture_result_mapping)
    if capture_payload is None and isinstance(inference_runtime_meta, dict):
        capture_payload = _reconstruct_capture_payload(inference_runtime_meta)
        if runtime_capture_inference_status is None:
            inference_status_candidate = inference_runtime_meta.get("inference_status")
            if isinstance(inference_status_candidate, str) and inference_status_candidate:
                runtime_capture_inference_status = inference_status_candidate
        if runtime_capture_inference_error is None:
            inference_error_candidate = inference_runtime_meta.get("inference_error")
            if isinstance(inference_error_candidate, str) and inference_error_candidate:
                runtime_capture_inference_error = inference_error_candidate

    cache_diagnostics = None
    if runtime_capture_cache is not None and hasattr(runtime_capture_cache, "capture_diagnostics"):
        cache_diagnostics_candidate = runtime_capture_cache.capture_diagnostics()
        if isinstance(cache_diagnostics_candidate, dict):
            cache_diagnostics = dict(cast(Dict[str, Any], cache_diagnostics_candidate))

    available_steps: list[Any] = []
    if isinstance(capture_payload, dict):
        available_steps_candidate = capture_payload.get("trajectory_cache_available_steps")
        if isinstance(available_steps_candidate, list):
            available_steps = cast(list[Any], available_steps_candidate)
    elif isinstance(cache_diagnostics, dict):
        available_steps_candidate = cache_diagnostics.get("available_steps")
        if isinstance(available_steps_candidate, list):
            available_steps = cast(list[Any], available_steps_candidate)

    failure_examples: list[Any] = []
    if isinstance(capture_payload, dict):
        failure_examples_candidate = capture_payload.get("trajectory_cache_capture_failure_examples")
        if isinstance(failure_examples_candidate, list):
            failure_examples = cast(list[Any], failure_examples_candidate)
    elif isinstance(cache_diagnostics, dict):
        failure_examples_candidate = cache_diagnostics.get("failure_examples")
        if isinstance(failure_examples_candidate, list):
            failure_examples = cast(list[Any], failure_examples_candidate)

    missing_required_steps: list[Any] = []
    if isinstance(capture_payload, dict):
        missing_required_steps_candidate = capture_payload.get("trajectory_cache_missing_required_steps")
        if isinstance(missing_required_steps_candidate, list):
            missing_required_steps = cast(list[Any], missing_required_steps_candidate)

    normalized_available_steps = [int(value) for value in available_steps if isinstance(value, int)]
    normalized_failure_examples = [
        dict(cast(Dict[str, Any], item))
        for item in failure_examples
        if isinstance(item, dict)
    ]
    normalized_missing_required_steps = [int(value) for value in missing_required_steps if isinstance(value, int)]

    capture_error_message = None
    if normalized_failure_examples:
        first_failure_example = normalized_failure_examples[0]
        exception_message = first_failure_example.get("exception_message")
        if isinstance(exception_message, str) and exception_message:
            capture_error_message = exception_message

    capture_attempt_count = capture_payload.get("trajectory_cache_capture_attempt_count") if isinstance(capture_payload, dict) else None
    if not isinstance(capture_attempt_count, int) and isinstance(cache_diagnostics, dict):
        capture_attempt_count = cache_diagnostics.get("capture_attempt_count")

    capture_success_count = capture_payload.get("trajectory_cache_capture_success_count") if isinstance(capture_payload, dict) else None
    if not isinstance(capture_success_count, int) and isinstance(cache_diagnostics, dict):
        capture_success_count = cache_diagnostics.get("capture_success_count")

    capture_failure_count = capture_payload.get("trajectory_cache_capture_failure_count") if isinstance(capture_payload, dict) else None
    if not isinstance(capture_failure_count, int) and isinstance(cache_diagnostics, dict):
        capture_failure_count = cache_diagnostics.get("capture_failure_count")

    callback_invocation_count = capture_payload.get("trajectory_cache_callback_invocation_count") if isinstance(capture_payload, dict) else None
    callback_latent_present_count = capture_payload.get("trajectory_cache_callback_latent_present_count") if isinstance(capture_payload, dict) else None
    tap_captured_step_count = capture_payload.get("trajectory_cache_tap_captured_step_count") if isinstance(capture_payload, dict) else None
    if not isinstance(tap_captured_step_count, int):
        tap_captured_step_count = tap_captured_step_count_from_evidence
    if not isinstance(tap_captured_step_count, int):
        tap_captured_step_count = len(normalized_available_steps)

    if (
        capture_payload is None
        and isinstance(capture_failure_count, int)
        and capture_failure_count > 0
        and isinstance(capture_success_count, int)
        and capture_success_count <= 0
    ):
        capture_payload = {
            "trajectory_cache_capture_status": "all_failed",
            "trajectory_cache_step_count": len(normalized_available_steps),
            "trajectory_cache_capture_attempt_count": capture_attempt_count,
            "trajectory_cache_capture_success_count": capture_success_count,
            "trajectory_cache_capture_failure_count": capture_failure_count,
            "trajectory_cache_capture_failure_examples": normalized_failure_examples,
            "trajectory_cache_available_steps": normalized_available_steps,
            "trajectory_cache_required_step_count": None,
            "trajectory_cache_missing_required_steps": normalized_missing_required_steps,
            "trajectory_cache_callback_invocation_count": None,
            "trajectory_cache_callback_latent_present_count": None,
            "trajectory_cache_tap_captured_step_count": int(tap_captured_step_count),
            "trajectory_cache_capture_detail_code": "trajectory_cache_capture_all_failed",
            "trajectory_cache_capture_detail_message": capture_error_message or "trajectory_cache_capture_all_failed",
            "trajectory_cache_capture_error_message": capture_error_message,
        }

    if capture_payload is None and isinstance(runtime_capture_inference_status, str) and runtime_capture_inference_status != infer_runtime.INFERENCE_STATUS_OK:
        capture_payload = {
            "trajectory_cache_capture_status": "tap_steps_observed_but_cache_meta_missing" if tap_captured_step_count > 0 else None,
            "trajectory_cache_step_count": len(normalized_available_steps),
            "trajectory_cache_capture_attempt_count": capture_attempt_count,
            "trajectory_cache_capture_success_count": capture_success_count,
            "trajectory_cache_capture_failure_count": capture_failure_count,
            "trajectory_cache_capture_failure_examples": normalized_failure_examples,
            "trajectory_cache_available_steps": normalized_available_steps,
            "trajectory_cache_required_step_count": None,
            "trajectory_cache_missing_required_steps": normalized_missing_required_steps,
            "trajectory_cache_callback_invocation_count": None,
            "trajectory_cache_callback_latent_present_count": None,
            "trajectory_cache_tap_captured_step_count": int(tap_captured_step_count),
            "trajectory_cache_capture_detail_code": "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure",
            "trajectory_cache_capture_detail_message": "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure",
            "trajectory_cache_capture_error_message": runtime_capture_inference_error or capture_error_message,
        }
    elif capture_payload is None and tap_captured_step_count > 0:
        capture_payload = {
            "trajectory_cache_capture_status": "tap_steps_observed_but_cache_meta_missing",
            "trajectory_cache_step_count": len(normalized_available_steps),
            "trajectory_cache_capture_attempt_count": capture_attempt_count if isinstance(capture_attempt_count, int) else 0,
            "trajectory_cache_capture_success_count": capture_success_count if isinstance(capture_success_count, int) else len(normalized_available_steps),
            "trajectory_cache_capture_failure_count": capture_failure_count if isinstance(capture_failure_count, int) else 0,
            "trajectory_cache_capture_failure_examples": normalized_failure_examples,
            "trajectory_cache_available_steps": normalized_available_steps,
            "trajectory_cache_required_step_count": None,
            "trajectory_cache_missing_required_steps": normalized_missing_required_steps,
            "trajectory_cache_callback_invocation_count": None,
            "trajectory_cache_callback_latent_present_count": None,
            "trajectory_cache_tap_captured_step_count": int(tap_captured_step_count),
            "trajectory_cache_capture_detail_code": "trajectory_cache_capture_meta_missing_after_tap",
            "trajectory_cache_capture_detail_message": "trajectory_cache_capture_meta_missing_after_tap",
            "trajectory_cache_capture_error_message": runtime_capture_inference_error or capture_error_message,
        }
    elif isinstance(capture_payload, dict):
        capture_payload = dict(capture_payload)
        capture_status = capture_payload.get("trajectory_cache_capture_status")
        if (
            tap_captured_step_count > 0
            and isinstance(callback_invocation_count, int)
            and callback_invocation_count > 0
            and isinstance(callback_latent_present_count, int)
            and callback_latent_present_count > 0
            and isinstance(capture_attempt_count, int)
            and capture_attempt_count <= 0
            and isinstance(capture_success_count, int)
            and capture_success_count <= 0
            and isinstance(capture_failure_count, int)
            and capture_failure_count <= 0
            and not normalized_available_steps
        ):
            capture_payload["trajectory_cache_capture_status"] = "tap_steps_observed_but_cache_write_not_observed"
        elif not isinstance(capture_status, str) and tap_captured_step_count > 0 and not normalized_available_steps:
            capture_payload["trajectory_cache_capture_status"] = "tap_steps_observed_but_cache_meta_missing"

        capture_status = capture_payload.get("trajectory_cache_capture_status")
        capture_detail_code = capture_payload.get("trajectory_cache_capture_detail_code")
        if not isinstance(capture_detail_code, str) or not capture_detail_code:
            if capture_status == "tap_steps_observed_but_cache_write_not_observed":
                capture_payload["trajectory_cache_capture_detail_code"] = "trajectory_cache_write_not_observed_after_tap"
                capture_payload["trajectory_cache_capture_detail_message"] = "trajectory_cache_write_not_observed_after_tap"
            elif capture_status == "tap_steps_observed_but_cache_meta_missing":
                capture_payload["trajectory_cache_capture_detail_code"] = "trajectory_cache_capture_meta_missing_after_tap"
                capture_payload["trajectory_cache_capture_detail_message"] = "trajectory_cache_capture_meta_missing_after_tap"
            elif capture_status == "all_failed":
                capture_payload["trajectory_cache_capture_detail_code"] = "trajectory_cache_capture_all_failed"
                capture_payload["trajectory_cache_capture_detail_message"] = capture_error_message or "trajectory_cache_capture_all_failed"

        if runtime_capture_inference_error and not capture_payload.get("trajectory_cache_capture_error_message"):
            capture_payload["trajectory_cache_capture_error_message"] = runtime_capture_inference_error

    return {
        "runtime_capture_inference_status": runtime_capture_inference_status,
        "runtime_capture_inference_error": runtime_capture_inference_error,
        "trajectory_cache_capture_status": capture_payload.get("trajectory_cache_capture_status") if isinstance(capture_payload, dict) else None,
        "trajectory_cache_step_count": capture_payload.get("trajectory_cache_step_count") if isinstance(capture_payload, dict) else len(normalized_available_steps),
        "trajectory_cache_capture_attempt_count": capture_attempt_count,
        "trajectory_cache_capture_success_count": capture_success_count,
        "trajectory_cache_capture_failure_count": capture_failure_count,
        "trajectory_cache_capture_failure_examples": normalized_failure_examples,
        "trajectory_cache_available_steps": normalized_available_steps,
        "trajectory_cache_required_step_count": capture_payload.get("trajectory_cache_required_step_count") if isinstance(capture_payload, dict) else None,
        "trajectory_cache_missing_required_steps": normalized_missing_required_steps,
        "trajectory_cache_callback_invocation_count": capture_payload.get("trajectory_cache_callback_invocation_count") if isinstance(capture_payload, dict) else None,
        "trajectory_cache_callback_latent_present_count": capture_payload.get("trajectory_cache_callback_latent_present_count") if isinstance(capture_payload, dict) else None,
        "trajectory_cache_tap_captured_step_count": capture_payload.get("trajectory_cache_tap_captured_step_count") if isinstance(capture_payload, dict) else tap_captured_step_count,
        "trajectory_cache_capture_detail_code": capture_payload.get("trajectory_cache_capture_detail_code") if isinstance(capture_payload, dict) else None,
        "trajectory_cache_capture_detail_message": capture_payload.get("trajectory_cache_capture_detail_message") if isinstance(capture_payload, dict) else None,
        "trajectory_cache_capture_error_message": capture_payload.get("trajectory_cache_capture_error_message") if isinstance(capture_payload, dict) else runtime_capture_inference_error,
        "trajectory_cache_capture": capture_payload,
    }


def _is_runtime_capture_cache_usable(
    runtime_capture_diagnostics: Dict[str, Any],
    runtime_capture_cache: Any,
) -> bool:
    """
    功能：判定 statement_only finalization 是否具备 exact-only trajectory cache。 

    Determine whether the runtime capture cache is usable for the exact-only
    statement_only finalization path.

    Args:
        runtime_capture_diagnostics: Structured capture diagnostics.
        runtime_capture_cache: In-memory trajectory cache object.

    Returns:
        True when the cache is usable for planner handoff.
    """
    capture_status = runtime_capture_diagnostics.get("trajectory_cache_capture_status")
    if isinstance(capture_status, str) and capture_status:
        return capture_status == "complete"
    return bool(runtime_capture_cache is not None and hasattr(runtime_capture_cache, "is_empty") and not runtime_capture_cache.is_empty())


def _build_runtime_capture_precheck_failure(
    runtime_capture_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：构造 statement_only runtime cache 前置校验失败诊断。 

    Build planner-style diagnostics for statement_only runtime cache precheck
    failures before planner invocation.

    Args:
        runtime_capture_diagnostics: Structured capture diagnostics.

    Returns:
        Mapping with planner_failure_* fields.
    """
    capture_status = runtime_capture_diagnostics.get("trajectory_cache_capture_status")
    capture_detail_code = runtime_capture_diagnostics.get("trajectory_cache_capture_detail_code")
    available_steps = runtime_capture_diagnostics.get("trajectory_cache_available_steps")
    missing_required_steps = runtime_capture_diagnostics.get("trajectory_cache_missing_required_steps")
    if not isinstance(available_steps, list):
        available_steps = []
    if not isinstance(missing_required_steps, list):
        missing_required_steps = []

    planner_failure_detail_code = "trajectory_cache_absent_or_empty"
    planner_failure_detail_message = "trajectory_cache_absent_or_empty_cannot_build_basis"
    if capture_detail_code == "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure":
        planner_failure_detail_code = "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure"
        planner_failure_detail_message = "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure_cannot_build_basis"
    elif capture_status == "tap_steps_observed_but_cache_write_not_observed":
        planner_failure_detail_code = "trajectory_cache_write_not_observed_after_tap"
        planner_failure_detail_message = "trajectory_cache_write_not_observed_after_tap_cannot_build_basis"
    elif capture_status == "tap_steps_observed_but_cache_meta_missing":
        planner_failure_detail_code = "trajectory_cache_capture_meta_missing_after_tap"
        planner_failure_detail_message = "trajectory_cache_capture_meta_missing_after_tap_cannot_build_basis"
    elif missing_required_steps and available_steps:
        planner_failure_detail_code = "trajectory_cache_partial_missing_required_steps"
        planner_failure_detail_message = (
            "trajectory_cache_partial_missing_required_steps_cannot_build_basis:"
            f"{missing_required_steps[:8]}"
        )
    elif missing_required_steps:
        planner_failure_detail_code = "trajectory_cache_missing_required_steps"
        planner_failure_detail_message = (
            "trajectory_cache_missing_required_steps_cannot_build_basis:"
            f"{missing_required_steps[:8]}"
        )
    elif capture_status == "callback_not_observed":
        planner_failure_detail_code = "trajectory_callback_not_observed"
        planner_failure_detail_message = "trajectory_callback_not_observed_cannot_build_basis"
    elif capture_status == "callback_invoked_without_latents":
        planner_failure_detail_code = "trajectory_callback_invoked_without_latents"
        planner_failure_detail_message = "trajectory_callback_invoked_without_latents_cannot_build_basis"
    elif capture_status == "all_failed":
        planner_failure_detail_code = "trajectory_cache_capture_all_failed"
        planner_failure_detail_message = "trajectory_cache_capture_all_failed_cannot_build_basis"
    elif capture_status == "unsupported_pipeline":
        planner_failure_detail_code = "trajectory_callback_unsupported"
        planner_failure_detail_message = "trajectory_callback_unsupported_cannot_build_basis"
    elif available_steps:
        planner_failure_detail_code = "trajectory_cache_incomplete"
        planner_failure_detail_message = "trajectory_cache_incomplete_cannot_build_basis"

    planner_context_capture = runtime_capture_diagnostics.get("trajectory_cache_capture")
    if not isinstance(planner_context_capture, dict):
        planner_context_capture = None
    planner_context: Dict[str, Any] = {
        "diagnostic_source": "statement_only_runtime_capture_precheck",
        "trajectory_cache_capture": dict(cast(Dict[str, Any], planner_context_capture)) if isinstance(planner_context_capture, dict) else {
            key_name: value
            for key_name, value in runtime_capture_diagnostics.items()
            if key_name != "trajectory_cache_capture"
        },
    }
    return {
        "planner_failure_stage": "runtime_capture_cache_validation",
        "planner_failure_detail_code": planner_failure_detail_code,
        "planner_failure_detail_message": planner_failure_detail_message,
        "planner_diagnostic_context": planner_context,
    }


def _build_runtime_finalization_status_details(run_meta: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    功能：从 formal_two_stage 提取可进入 run_closure 的 runtime finalization 细节。 

    Build the runtime-finalization detail block persisted through run_closure.

    Args:
        run_meta: Mutable run metadata mapping.

    Returns:
        Structured detail mapping or None.
    """
    formal_two_stage = run_meta.get("formal_two_stage")
    if not isinstance(formal_two_stage, dict):
        return None
    formal_two_stage_mapping = cast(Dict[str, Any], formal_two_stage)
    runtime_finalization_status = formal_two_stage_mapping.get("runtime_finalization_status")
    runtime_executable_plan_status = formal_two_stage_mapping.get("runtime_executable_plan_status")
    if runtime_finalization_status is None and runtime_executable_plan_status is None:
        return None

    return {
        "runtime_finalization_status": runtime_finalization_status,
        "runtime_finalization_reason": formal_two_stage_mapping.get("runtime_finalization_reason"),
        "runtime_executable_plan_status": runtime_executable_plan_status,
        "runtime_executable_plan_reason": formal_two_stage_mapping.get("runtime_executable_plan_reason"),
        "runtime_capture_inference_status": formal_two_stage_mapping.get("runtime_capture_inference_status"),
        "runtime_capture_inference_error": formal_two_stage_mapping.get("runtime_capture_inference_error"),
        "runtime_capture_cuda_memory_profile": formal_two_stage_mapping.get("runtime_capture_cuda_memory_profile"),
        "trajectory_cache_capture_status": formal_two_stage_mapping.get("trajectory_cache_capture_status"),
        "trajectory_cache_step_count": formal_two_stage_mapping.get("trajectory_cache_step_count"),
        "trajectory_cache_capture_attempt_count": formal_two_stage_mapping.get("trajectory_cache_capture_attempt_count"),
        "trajectory_cache_capture_success_count": formal_two_stage_mapping.get("trajectory_cache_capture_success_count"),
        "trajectory_cache_capture_failure_count": formal_two_stage_mapping.get("trajectory_cache_capture_failure_count"),
        "trajectory_cache_capture_failure_examples": formal_two_stage_mapping.get("trajectory_cache_capture_failure_examples"),
        "trajectory_cache_available_steps": formal_two_stage_mapping.get("trajectory_cache_available_steps"),
        "trajectory_cache_required_step_count": formal_two_stage_mapping.get("trajectory_cache_required_step_count"),
        "trajectory_cache_missing_required_steps": formal_two_stage_mapping.get("trajectory_cache_missing_required_steps"),
        "trajectory_cache_callback_invocation_count": formal_two_stage_mapping.get("trajectory_cache_callback_invocation_count"),
        "trajectory_cache_callback_latent_present_count": formal_two_stage_mapping.get("trajectory_cache_callback_latent_present_count"),
        "trajectory_cache_tap_captured_step_count": formal_two_stage_mapping.get("trajectory_cache_tap_captured_step_count"),
        "trajectory_cache_capture_detail_code": formal_two_stage_mapping.get("trajectory_cache_capture_detail_code"),
        "trajectory_cache_capture_detail_message": formal_two_stage_mapping.get("trajectory_cache_capture_detail_message"),
        "trajectory_cache_capture_error_message": formal_two_stage_mapping.get("trajectory_cache_capture_error_message"),
        "trajectory_cache_capture": formal_two_stage_mapping.get("trajectory_cache_capture"),
        "planner_failure_stage": formal_two_stage_mapping.get("planner_failure_stage"),
        "planner_failure_detail_code": formal_two_stage_mapping.get("planner_failure_detail_code"),
        "planner_failure_detail_message": formal_two_stage_mapping.get("planner_failure_detail_message"),
        "planner_diagnostic_context": formal_two_stage_mapping.get("planner_diagnostic_context"),
    }


def _merge_runtime_finalization_status_details(run_meta: Dict[str, Any]) -> None:
    """
    功能：将 runtime finalization 诊断并入 run_meta.status_details。 

    Merge runtime-finalization diagnostics into run_meta.status_details so the
    failure path is persisted into run_closure.

    Args:
        run_meta: Mutable run metadata mapping.

    Returns:
        None.
    """
    runtime_finalization_details = _build_runtime_finalization_status_details(run_meta)
    if not isinstance(runtime_finalization_details, dict):
        return
    existing_status_details = run_meta.get("status_details")
    if not isinstance(existing_status_details, dict):
        run_meta["status_details"] = {"runtime_finalization": runtime_finalization_details}
        return
    merged_status_details = dict(cast(Dict[str, Any], existing_status_details))
    merged_status_details["runtime_finalization"] = runtime_finalization_details
    run_meta["status_details"] = merged_status_details


def _bind_embed_plan_digest_consistency(
    record: Dict[str, Any],
    content_evidence: Dict[str, Any],
    injection_plan_digest: Any,
    formal_plan_digest: Any,
) -> str:
    """
    功能：写入 embed 主链的 plan digest 一致性字段，并在 formal 分叉时显式标记。 

    Bind embed-side plan digest consistency fields and explicitly mark formal divergence.

    Args:
        record: Mutable embed record mapping.
        content_evidence: Mutable content evidence mapping.
        injection_plan_digest: Digest bound during injection-time precomputation.
        formal_plan_digest: Digest observed from the post-inference formal path.

    Returns:
        Consistency status string.
    """
    normalized_injection_plan_digest = _normalize_plan_digest(injection_plan_digest)
    normalized_formal_plan_digest = _normalize_plan_digest(formal_plan_digest)
    record["plan_digest_injection"] = normalized_injection_plan_digest or "<absent>"
    record["plan_digest_formal"] = normalized_formal_plan_digest or "<absent>"
    record["plan_digest_expected"] = normalized_injection_plan_digest or "<absent>"
    record["plan_digest_observed"] = normalized_formal_plan_digest or "<absent>"

    consistency_status = "ok"
    match_status = "match"
    mismatch_reason: str | None = None
    if normalized_injection_plan_digest is None:
        consistency_status = "absent"
        match_status = "absent"
        mismatch_reason = "plan_digest_injection_absent"
    elif normalized_formal_plan_digest is None:
        consistency_status = "absent"
        match_status = "absent"
        mismatch_reason = "plan_digest_formal_absent"
    elif normalized_injection_plan_digest != normalized_formal_plan_digest:
        consistency_status = "mismatch"
        match_status = "mismatch"
        mismatch_reason = "plan_digest_mismatch"

    record["plan_digest_status"] = consistency_status
    record["plan_digest_match_status"] = match_status

    if mismatch_reason is None:
        record.pop("plan_digest_mismatch_reason", None)
        content_evidence.pop("plan_digest_mismatch", None)
        if content_evidence.get("content_mismatch_reason") in {
            "plan_digest_mismatch",
            "plan_digest_formal_absent",
            "plan_digest_injection_absent",
        }:
            content_evidence.pop("content_mismatch_reason", None)
        return consistency_status

    record["plan_digest_mismatch_reason"] = mismatch_reason
    content_evidence["plan_digest_mismatch"] = {
        "plan_digest_injection": record["plan_digest_injection"],
        "plan_digest_formal": record["plan_digest_formal"],
        "mismatch_reason": mismatch_reason,
    }
    if content_evidence.get("status") == "ok":
        content_evidence["status"] = "mismatch"
    if not content_evidence.get("content_mismatch_reason"):
        content_evidence["content_mismatch_reason"] = mismatch_reason
    return consistency_status


def _extract_subspace_result_digests(subspace_result: Any) -> tuple[str | None, str | None]:
    """
    功能：提取 precomputed subspace 结果中的 plan/basis digest。 

    Extract plan and basis digests from a precomputed subspace result.

    Args:
        subspace_result: Precomputed planner result object or mapping.

    Returns:
        Tuple of normalized (plan_digest, basis_digest).
    """
    payload: Dict[str, Any] | None = None
    if isinstance(subspace_result, dict):
        payload = cast(Dict[str, Any], subspace_result)
    elif hasattr(subspace_result, "as_dict"):
        payload_candidate = subspace_result.as_dict()
        if isinstance(payload_candidate, dict):
            payload = cast(Dict[str, Any], payload_candidate)

    subspace_result_object = cast(Any, subspace_result)
    plan_digest = _normalize_plan_digest(getattr(subspace_result_object, "plan_digest", None))
    if plan_digest is None and isinstance(payload, dict):
        plan_digest = _normalize_plan_digest(payload.get("plan_digest"))

    basis_digest = _normalize_plan_digest(getattr(subspace_result_object, "basis_digest", None))
    if basis_digest is None and isinstance(payload, dict):
        basis_digest = _normalize_plan_digest(payload.get("basis_digest"))
        if basis_digest is None:
            plan_node = payload.get("plan")
            if isinstance(plan_node, dict):
                plan_node_payload = cast(Dict[str, Any], plan_node)
                basis_digest = _normalize_plan_digest(plan_node_payload.get("basis_digest"))

    return plan_digest, basis_digest


def _resolve_formal_subspace_override(subspace_result: Any) -> Any | None:
    """
    功能：仅在 executable formal plan 与 basis 都可用时复用 override。 

    Reuse the finalized subspace override only when both formal plan and
    basis digests are already available.

    Args:
        subspace_result: Precomputed planner result object or mapping.

    Returns:
        Original override payload when formal anchors are complete; otherwise None.
    """
    plan_digest, basis_digest = _extract_subspace_result_digests(subspace_result)
    if plan_digest is None or basis_digest is None:
        return None
    return subspace_result


def _write_embed_attestation_artifacts(
    record: Any,
    artifacts_dir: Path,
) -> None:
    """
    功能：将 embed 主链生成的 attestation 工件落盘到 artifacts/attestation。

    Persist embed-side attestation artifacts produced by the main path.

    Args:
        record: Embed record mapping.
        artifacts_dir: Current run artifacts directory.

    Returns:
        None.
    """
    if not isinstance(record, dict):
        return
    record_dict = cast(Dict[str, Any], record)
    attestation_node = record_dict.get("attestation")
    attestation_payload: Dict[str, Any] = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    if attestation_payload.get("status") != "ok":
        return

    attestation_dir = artifacts_dir / "attestation"
    attestation_dir.mkdir(parents=True, exist_ok=True)
    statement_payload: Dict[str, Any] = {
        "statement": attestation_payload.get("statement"),
        "attestation_digest": attestation_payload.get("attestation_digest"),
        "event_binding_digest": attestation_payload.get("event_binding_digest"),
        "lf_payload_hex": attestation_payload.get("lf_payload_hex"),
        "trace_commit": attestation_payload.get("trace_commit"),
        "geo_anchor_seed": attestation_payload.get("geo_anchor_seed"),
        "attestation_status": attestation_payload.get("status"),
    }
    records_io.write_artifact_json(str(attestation_dir / "attestation_statement.json"), statement_payload)
    signed_bundle = attestation_payload.get("signed_bundle")
    if isinstance(signed_bundle, dict):
        records_io.write_artifact_json(str(attestation_dir / "attestation_bundle.json"), cast(Dict[str, Any], signed_bundle))


def bind_impl_identity_fields(
    record: Dict[str, Any],
    identity: runtime_resolver.ImplIdentity,
    impl_set: runtime_resolver.BuiltImplSet,
    contracts: FrozenContracts
) -> None:
    """
    功能：绑定 impl_identity 相关字段。

    Bind impl identity, version, and digest fields into record.

    Args:
        record: Record dict to mutate.
        identity: Impl identity mapping.
        impl_set: Built implementation set.
        contracts: Loaded FrozenContracts.

    Returns:
        None.

    Raises:
        MissingRequiredFieldError: If required impl fields are missing.
        TypeError: If inputs are invalid.
    """
    return _bind_impl_identity_fields(record, identity, impl_set, contracts)


def _build_embed_runtime_reuse_signature(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构建 embed runtime 复用兼容性签名。

    Build the compatibility signature used for embed runtime session reuse.

    Args:
        cfg: Configuration mapping.

    Returns:
        Mapping containing the stable fields that must remain unchanged when a
        runtime session is reused.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    signature_cfg = copy.deepcopy(cfg)
    for field_name in [
        "inference_prompt",
        "seed",
        "paper_workflow_event",
        "input_image_path",
        "pw01_source_pool_preview",
    ]:
        signature_cfg.pop(field_name, None)

    embed_node = signature_cfg.get("embed")
    if isinstance(embed_node, dict):
        embed_cfg = dict(cast(Dict[str, Any], embed_node))
        embed_cfg.pop("input_image_path", None)
        signature_cfg["embed"] = embed_cfg

    model_source_binding = signature_cfg.get("model_source_binding")
    model_source_binding_digest = (
        digests.canonical_sha256(model_source_binding)
        if isinstance(model_source_binding, dict)
        else None
    )
    return {
        "policy_path": signature_cfg.get("policy_path"),
        "device": signature_cfg.get("device"),
        "model_id": signature_cfg.get("model_id"),
        "model_source": signature_cfg.get("model_source"),
        "hf_revision": signature_cfg.get("hf_revision"),
        "model_snapshot_path": signature_cfg.get("model_snapshot_path"),
        "model_source_binding_digest": model_source_binding_digest,
        "static_cfg_digest": digests.canonical_sha256(signature_cfg),
    }


def _validate_embed_runtime_session(runtime_session: Any) -> Dict[str, Any]:
    """
    功能：校验 embed runtime session 的最小结构。

    Validate the minimum structure required by an embed runtime session.

    Args:
        runtime_session: Candidate runtime session payload.

    Returns:
        Validated runtime session mapping.

    Raises:
        TypeError: If runtime_session is not a mapping.
        ValueError: If required fields are missing.
    """
    if not isinstance(runtime_session, dict):
        raise TypeError("runtime_session must be dict")

    required_keys = [
        "contracts",
        "whitelist",
        "semantics",
        "injection_scope_manifest",
        "interpretation",
        "snapshot",
        "pipeline_result",
        "impl_identity",
        "impl_set",
        "impl_set_capabilities_digest",
        "runtime_reuse_signature",
    ]
    missing_keys = [key_name for key_name in required_keys if key_name not in runtime_session]
    if missing_keys:
        raise ValueError(f"runtime_session missing required keys: {missing_keys}")
    return cast(Dict[str, Any], runtime_session)


def build_embed_runtime_session(
    config_path: Any,
    overrides: Any = None,
) -> Dict[str, Any]:
    """
    功能：构建可复用的 embed 静态 runtime session。

    Build reusable static embed runtime state for multiple embed executions.

    Args:
        config_path: YAML config path used to construct the static runtime.
        overrides: Optional CLI override args list.

    Returns:
        Mapping containing reusable fact-source objects, pipeline state, and
        implementation instances.

    Raises:
        ValueError: If config_path is invalid.
        TypeError: If overrides is invalid.
    """
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("config_path must be non-empty str")
    if overrides is not None:
        if not isinstance(overrides, list):
            raise TypeError("overrides must be list[str] or None")
        overrides_list = cast(list[Any], overrides)
        for override_item in overrides_list:
            if not isinstance(override_item, str):
                raise TypeError("overrides must be list[str] or None")

    validated_overrides = cast(list[str] | None, overrides)

    contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
    whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
    semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
    injection_scope_manifest = config_loader.load_injection_scope_manifest()
    assert_consistent_with_semantics(whitelist, semantics)
    interpretation = get_contract_interpretation(contracts)
    cfg, _, _ = config_loader.load_and_validate_config(
        config_path,
        whitelist,
        semantics,
        contracts,
        interpretation,
        overrides=validated_overrides,
    )

    _, _, seed_value, _ = build_seed_audit(cfg, "embed")
    cfg["seed"] = seed_value

    pipeline_result = pipeline_factory.build_pipeline_shell(cfg)
    impl_identity, impl_set, impl_set_capabilities_digest = runtime_resolver.build_runtime_impl_set_from_cfg(
        cfg,
        whitelist,
    )
    snapshot = records_io.build_fact_sources_snapshot(
        contracts,
        whitelist,
        semantics,
        injection_scope_manifest,
    )
    return {
        "contracts": contracts,
        "whitelist": whitelist,
        "semantics": semantics,
        "injection_scope_manifest": injection_scope_manifest,
        "interpretation": interpretation,
        "snapshot": snapshot,
        "pipeline_result": pipeline_result,
        "impl_identity": impl_identity,
        "impl_set": impl_set,
        "impl_set_capabilities_digest": impl_set_capabilities_digest,
        "runtime_reuse_signature": _build_embed_runtime_reuse_signature(cfg),
    }


def run_embed(
    output_dir: Any,
    config_path: Any,
    overrides: Any = None,
    input_image_path: Any = None,
    runtime_session: Any = None,
) -> None:
    """
    功能：执行嵌入流程，本阶段为基线实现。

    Execute embed workflow (baseline implementation).

    Args:
        output_dir: Run root directory for records/artifacts.
        config_path: YAML config path.
        overrides: Optional CLI override args list.
        input_image_path: Optional input image path for identity embed artifact flow.
        runtime_session: Optional reusable embed runtime session.
    """
    if not isinstance(output_dir, str) or not output_dir:
        # output_dir 输入不合法，必须 fail-fast。
        raise ValueError("output_dir must be non-empty str")
    if not isinstance(config_path, str) or not config_path:
        # config_path 输入不合法，必须 fail-fast。
        raise ValueError("config_path must be non-empty str")
    if overrides is not None and not isinstance(overrides, list):
        # overrides 输入不合法，必须 fail-fast。
        raise ValueError("overrides must be list or None")
    if input_image_path is not None and not isinstance(input_image_path, str):
        # input_image_path 输入不合法，必须 fail-fast。
        raise ValueError("input_image_path must be str or None")
    if runtime_session is not None and not isinstance(runtime_session, dict):
        # runtime_session 输入不合法，必须 fail-fast。
        raise TypeError("runtime_session must be dict or None")
    validated_overrides: list[str] | None = None
    if isinstance(overrides, list):
        override_candidates = cast(list[Any], overrides)
        if any(not isinstance(item, str) or not item for item in override_candidates):
            # overrides 列表项不合法，必须 fail-fast。
            raise ValueError("overrides items must be non-empty str")
        validated_overrides = cast(list[str], override_candidates)
    validated_input_image_path = input_image_path.strip() if isinstance(input_image_path, str) and input_image_path.strip() else None
    validated_runtime_session = (
        _validate_embed_runtime_session(runtime_session)
        if runtime_session is not None
        else None
    )

    # 创建 layout 和最小 run_meta。
    run_root = path_policy.derive_run_root(Path(output_dir))
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    layout_initialized = False
    allow_nonempty_run_root = False
    allow_nonempty_run_root_reason = None
    override_applied_for_layout = None
    
    # 初始化最小 run_meta。
    started_at = time_utils.now_utc_iso_z()
    run_meta: Dict[str, Any] = {
        "run_id": f"run-{uuid.uuid4().hex}",
        "command": "embed",
        "created_at_utc": started_at,
        "started_at": started_at,
        "output_dir_input": output_dir,
        "cfg_digest": "unknown",
        "policy_path": "unknown",
        "impl_id": "unknown",
        "impl_version": "unknown",
        "impl_identity": None,
        "impl_identity_digest": None,
        "pipeline_provenance_canon_sha256": "<absent>",
        "pipeline_status": "unbuilt",
        "pipeline_error": "<absent>",
        "pipeline_runtime_meta": None,
        "env_fingerprint_canon_sha256": "<absent>",
        "diffusers_version": "<absent>",
        "transformers_version": "<absent>",
        "safetensors_version": "<absent>",
        "model_provenance_canon_sha256": "<absent>",
        "status_ok": True,
        "status_reason": RunFailureReason.OK,
        "status_details": None,
        "manifest_rel_path": "unknown",
        "path_policy": {
            "allow_nonempty_run_root": False
        }
    }
    
    error = None
    pipeline_result: Dict[str, Any] | None = None
    record: Dict[str, Any] | None = None
    try:
        if validated_runtime_session is None:
            # 加载事实源。
            print("[Embed] Loading fact sources...")
            contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
            whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
            semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
            injection_scope_manifest = config_loader.load_injection_scope_manifest()
            interpretation = get_contract_interpretation(contracts)
            snapshot = records_io.build_fact_sources_snapshot(
                contracts,
                whitelist,
                semantics,
                injection_scope_manifest
            )
        else:
            print("[Embed] Reusing runtime session static state...")
            contracts = validated_runtime_session["contracts"]
            whitelist = validated_runtime_session["whitelist"]
            semantics = validated_runtime_session["semantics"]
            injection_scope_manifest = validated_runtime_session["injection_scope_manifest"]
            interpretation = validated_runtime_session["interpretation"]
            snapshot = dict(cast(Dict[str, Any], validated_runtime_session["snapshot"]))

        # 绑定冻结锚点到 run_meta。
        print("[Embed] Binding freeze anchors to run_meta...")
        status.bind_freeze_anchors_to_run_meta(
            run_meta,
            contracts,
            whitelist,
            semantics,
            injection_scope_manifest
        )

        run_meta["bound_fact_sources"] = snapshot

        # 验证 whitelist 与 semantics 一致性。
        print("[Embed] Validating whitelist-semantics consistency...")
        assert_consistent_with_semantics(whitelist, semantics)

        with records_io.bound_fact_sources(
            contracts,
            whitelist,
            semantics,
            run_root,
            records_dir,
            artifacts_dir,
            logs_dir,
            injection_scope_manifest=injection_scope_manifest
        ):
            bound_fact_sources = records_io.get_bound_fact_sources()
            run_meta["bound_fact_sources"] = bound_fact_sources
            if bound_fact_sources != snapshot:
                # 事实源快照不一致，必须 fail-fast。
                raise ValueError(format_fact_sources_mismatch(snapshot, bound_fact_sources))
            try:
                cfg, cfg_digest, cfg_audit_metadata = config_loader.load_and_validate_config(
                    config_path,
                    whitelist,
                    semantics,
                    contracts,
                    interpretation,
                    overrides=validated_overrides
                )
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.CONFIG_INVALID, exc)
                raise
            run_meta["cfg_digest"] = cfg_digest
            run_meta["policy_path"] = cfg["policy_path"]

            if validated_runtime_session is not None:
                runtime_reuse_signature = _build_embed_runtime_reuse_signature(cfg)
                if runtime_reuse_signature != validated_runtime_session["runtime_reuse_signature"]:
                    reuse_mismatch_exc = ValueError("runtime_session config signature mismatch")
                    set_failure_status(run_meta, RunFailureReason.CONFIG_INVALID, reuse_mismatch_exc)
                    raise reuse_mismatch_exc

            seed_parts, seed_digest, seed_value, seed_rule_id = build_seed_audit(cfg, "embed")
            cfg["seed"] = seed_value
            run_meta["seed_parts"] = seed_parts
            run_meta["seed_digest"] = seed_digest
            run_meta["seed_rule_id"] = seed_rule_id
            run_meta["seed_value"] = seed_value
            run_meta["cfg"] = cfg

            allow_nonempty_run_root = cfg.get("allow_nonempty_run_root", False)
            allow_nonempty_run_root_reason = cfg.get("allow_nonempty_run_root_reason")
            override_applied_for_layout = cfg.get("override_applied")
            layout = path_policy.ensure_output_layout(
                run_root,
                allow_nonempty_run_root=bool(allow_nonempty_run_root),
                allow_nonempty_run_root_reason=allow_nonempty_run_root_reason,
                override_applied=override_applied_for_layout,
            )
            layout_initialized = True
            path_policy.anchor_requirements(run_root)
            records_dir = layout["records_dir"]
            artifacts_dir = layout["artifacts_dir"]
            logs_dir = layout["logs_dir"]
            cfg["__run_root_dir__"] = str(run_root.resolve())
            cfg["__artifacts_dir__"] = str(artifacts_dir.resolve())

            determinism_controls = build_determinism_controls(cfg)
            if determinism_controls is not None:
                run_meta["determinism_controls"] = determinism_controls
            nondeterminism_notes = normalize_nondeterminism_notes(cfg.get("nondeterminism_notes"))
            if nondeterminism_notes is not None:
                run_meta["nondeterminism_notes"] = nondeterminism_notes

            _validate_real_pipeline_identity_fields(cfg, config_path)

            if validated_runtime_session is None:
                pipeline_result = pipeline_factory.build_pipeline_shell(cfg)
            else:
                pipeline_result = cast(Dict[str, Any], validated_runtime_session["pipeline_result"])
            run_meta["pipeline_provenance_canon_sha256"] = pipeline_result.get("pipeline_provenance_canon_sha256")
            run_meta["pipeline_status"] = pipeline_result.get("pipeline_status")
            run_meta["pipeline_error"] = pipeline_result.get("pipeline_error")
            run_meta["pipeline_runtime_meta"] = pipeline_result.get("pipeline_runtime_meta")
            run_meta["env_fingerprint_canon_sha256"] = pipeline_result.get("env_fingerprint_canon_sha256")
            run_meta["diffusers_version"] = pipeline_result.get("diffusers_version")
            run_meta["transformers_version"] = pipeline_result.get("transformers_version")
            run_meta["safetensors_version"] = pipeline_result.get("safetensors_version")
            run_meta["model_provenance_canon_sha256"] = pipeline_result.get("model_provenance_canon_sha256")

            if validated_runtime_session is None:
                try:
                    impl_identity, impl_set, impl_set_capabilities_digest = runtime_resolver.build_runtime_impl_set_from_cfg(
                        cfg,
                        whitelist,
                    )
                except Exception as exc:
                    set_failure_status(run_meta, RunFailureReason.IMPL_RESOLVE_FAILED, exc)
                    raise
            else:
                impl_identity = validated_runtime_session["impl_identity"]
                impl_set = validated_runtime_session["impl_set"]
                impl_set_capabilities_digest = validated_runtime_session["impl_set_capabilities_digest"]
            run_meta["impl_id"] = impl_identity.content_extractor_id
            run_meta["impl_version"] = impl_set.content_extractor.impl_version
            run_meta["impl_identity"] = impl_identity.as_dict()
            run_meta["impl_identity_digest"] = runtime_resolver.compute_impl_identity_digest(impl_identity)
            run_meta["impl_set_capabilities_digest"] = impl_set_capabilities_digest
            impl_set_capabilities_extended_digest = cfg.get("impl_set_capabilities_extended_digest")
            if isinstance(impl_set_capabilities_extended_digest, str) and impl_set_capabilities_extended_digest:
                run_meta["impl_set_capabilities_extended_digest"] = impl_set_capabilities_extended_digest

            if validated_input_image_path is not None:
                cfg["__embed_input_image_path__"] = validated_input_image_path
            else:
                embed_cfg = _resolve_embed_cfg(cfg)
                default_input = embed_cfg.get("input_image_path")
                if isinstance(default_input, str) and default_input.strip():
                    cfg["__embed_input_image_path__"] = default_input.strip()

            statement_only_formal_path = _requires_statement_only_formal_precompute(cfg)

            preview_generation_meta = _build_preview_generation_meta(
                cfg=cfg,
                run_root=run_root,
                artifacts_dir=artifacts_dir,
                pipeline_result=pipeline_result,
                seed_value=seed_value,
            )
            run_meta["preview_generation"] = preview_generation_meta
            _release_preview_generation_runtime_pressure(
                cfg,
                preview_generation_meta,
                statement_only_formal_path=statement_only_formal_path,
            )

            # 预先计算 content 与 subspace 计划，用于注入上下文。
            content_inputs_pre = _build_content_inputs_for_embed(cfg)
            if isinstance(content_inputs_pre, dict):
                content_inputs_pre = dict(content_inputs_pre)
            else:
                content_inputs_pre = {}
            content_inputs_pre["content_runtime_phase"] = EMBED_CONTENT_RUNTIME_PHASE_PRECOMPUTE
            content_result_pre = impl_set.content_extractor.extract(
                cfg,
                inputs=content_inputs_pre,
                cfg_digest=cfg_digest
            )
            content_result_pre_payload = _as_dict_payload(content_result_pre)
            mask_digest = None
            if isinstance(content_result_pre, dict):
                content_result_pre_dict = cast(Dict[str, Any], content_result_pre)
                mask_digest = content_result_pre_dict.get("mask_digest")
            elif hasattr(content_result_pre, "mask_digest"):
                mask_digest = content_result_pre.mask_digest

            plan_payload: Any = None
            plan_digest_precomputed = None
            basis_digest_precomputed = None
            subspace_result_pre: Any = None
            subspace_result_runtime_finalized: Any = None
            formal_scaffold: Dict[str, Any] | None = None
            runtime_finalization_meta: Dict[str, Any] = {
                "status": "not_required",
                "reason": None,
                "capture_inference_status": None,
                "trajectory_cache_bound": False,
                "final_plan_digest": None,
                "final_basis_digest": None,
            }

            planner_inputs = _build_planner_inputs_for_runtime(cfg, None, content_result_pre_payload)
            if statement_only_formal_path:
                formal_scaffold, scaffold_failure_reason = _build_statement_only_formal_scaffold(
                    cfg,
                    cfg_digest,
                    seed_value,
                    content_result_pre_payload,
                    planner_inputs,
                )
                if formal_scaffold is None:
                    raise ValueError(
                        "formal scaffold unavailable: statement_only pre-inference scaffold is required; "
                        f"reason={scaffold_failure_reason or 'scaffold_unavailable'}"
                    )
                cfg["__formal_scaffold__"] = formal_scaffold
                run_meta["formal_two_stage"] = {
                    "scaffold_status": "ok",
                    "scaffold_reason": None,
                    "formal_scaffold": formal_scaffold,
                    "runtime_executable_plan_status": "pending",
                    "runtime_executable_plan_reason": None,
                    "runtime_capture_inference_status": None,
                    "runtime_capture_inference_error": None,
                    "runtime_capture_cuda_memory_profile": None,
                    "final_plan_digest": None,
                    "final_basis_digest": None,
                }
            else:
                subspace_result_pre = impl_set.subspace_planner.plan(
                    cfg,
                    mask_digest=mask_digest,
                    cfg_digest=cfg_digest,
                    inputs=planner_inputs
                )
                plan_payload = _extract_subspace_plan_payload(subspace_result_pre)
                plan_digest_precomputed, basis_digest_precomputed = _extract_subspace_result_digests(subspace_result_pre)

            injection_context = None
            injection_modifier = None
            injection_site_spec = None
            injection_site_digest = None
            fallback_plan_digest = None
            injection_plan_digest = None
            injection_plan_source = None

            attestation_env_inputs = resolve_attestation_env_inputs(cfg, require_prompt_seed=True)
            if attestation_env_inputs.get("status") in {"ok", "absent"}:
                transient_secret_inputs: Dict[str, str] = {}
                for key_name in ["k_master", "k_prompt", "k_seed"]:
                    key_value = attestation_env_inputs.get(key_name)
                    if isinstance(key_value, str) and key_value:
                        transient_secret_inputs[key_name] = key_value
                if transient_secret_inputs:
                    cfg["__attestation_secret_inputs__"] = transient_secret_inputs

            pinned_event_nonce = cfg.get("__attestation_event_nonce__")
            if not isinstance(pinned_event_nonce, str) or not pinned_event_nonce:
                cfg["__attestation_event_nonce__"] = uuid.uuid4().hex
            pinned_time_bucket = cfg.get("__attestation_time_bucket__")
            if not isinstance(pinned_time_bucket, str) or not pinned_time_bucket:
                cfg["__attestation_time_bucket__"] = time_utils.now_utc_iso_z().split("T", 1)[0]

            pipeline_obj = pipeline_result.get("pipeline_obj")
            device = cfg.get("device", "cpu")

            if statement_only_formal_path:
                from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache

                runtime_capture_cache = LatentTrajectoryCache()
                runtime_capture_result = infer_runtime.run_sd3_inference(
                    cfg,
                    pipeline_obj,
                    device,
                    seed_value,
                    runtime_phase_label=STATEMENT_ONLY_RUNTIME_CAPTURE_PHASE_LABEL,
                    injection_context=None,
                    injection_modifier=None,
                    capture_final_latents=False,
                    trajectory_latent_cache=runtime_capture_cache,
                )
                runtime_capture_status_value = runtime_capture_result.get("inference_status")
                runtime_capture_status = (
                    runtime_capture_status_value
                    if isinstance(runtime_capture_status_value, str) and runtime_capture_status_value
                    else infer_runtime.INFERENCE_STATUS_FAILED
                )
                runtime_capture_error_value = runtime_capture_result.get("inference_error")
                runtime_capture_error = (
                    runtime_capture_error_value
                    if isinstance(runtime_capture_error_value, str) and runtime_capture_error_value
                    else None
                )
                runtime_capture_runtime_meta = runtime_capture_result.get("inference_runtime_meta")
                runtime_capture_cuda_memory_profile = None
                if isinstance(runtime_capture_runtime_meta, dict):
                    runtime_capture_runtime_meta_mapping = cast(Dict[str, Any], runtime_capture_runtime_meta)
                    nested_cuda_memory_profile = runtime_capture_runtime_meta_mapping.get("cuda_memory_profile")
                    if isinstance(nested_cuda_memory_profile, dict):
                        runtime_capture_cuda_memory_profile = dict(cast(Dict[str, Any], nested_cuda_memory_profile))
                runtime_capture_diagnostics = _extract_runtime_capture_diagnostics(
                    runtime_capture_result,
                    runtime_capture_cache,
                )
                runtime_capture_cache_usable = _is_runtime_capture_cache_usable(
                    runtime_capture_diagnostics,
                    runtime_capture_cache,
                )
                if pipeline_obj is not None:
                    cfg["__embed_pipeline_obj__"] = pipeline_obj
                if runtime_capture_cache_usable:
                    cfg["__embed_trajectory_latent_cache__"] = runtime_capture_cache
                else:
                    cfg.pop("__embed_trajectory_latent_cache__", None)
                    runtime_finalization_reason = "invalid_subspace_params"
                    runtime_capture_precheck_failure = _build_runtime_capture_precheck_failure(
                        runtime_capture_diagnostics,
                    )
                    runtime_finalization_meta = {
                        "status": "failed",
                        "reason": runtime_finalization_reason,
                        "runtime_finalization_status": "failed",
                        "runtime_finalization_reason": runtime_finalization_reason,
                        "capture_inference_status": runtime_capture_status,
                        "capture_inference_error": runtime_capture_error,
                        "runtime_capture_cuda_memory_profile": runtime_capture_cuda_memory_profile,
                        "trajectory_cache_bound": False,
                        "final_plan_digest": None,
                        "final_basis_digest": None,
                        **runtime_capture_diagnostics,
                        **runtime_capture_precheck_failure,
                    }
                    run_meta["formal_two_stage"] = {
                        "scaffold_status": "ok",
                        "scaffold_reason": None,
                        "formal_scaffold": formal_scaffold,
                        "runtime_executable_plan_status": "failed",
                        "runtime_executable_plan_reason": runtime_finalization_reason,
                        "runtime_finalization_status": "failed",
                        "runtime_finalization_reason": runtime_finalization_reason,
                        "runtime_capture_inference_status": runtime_capture_status,
                        "runtime_capture_inference_error": runtime_capture_error,
                        "runtime_capture_cuda_memory_profile": runtime_capture_cuda_memory_profile,
                        "final_plan_digest": None,
                        "final_basis_digest": None,
                        **runtime_capture_diagnostics,
                        **runtime_capture_precheck_failure,
                    }
                    raise ValueError(
                        "runtime executable formal plan unavailable: statement_only runtime finalization failed; "
                        f"reason={runtime_finalization_reason}"
                    )
                planner_inputs_runtime = _build_planner_inputs_for_runtime(
                    cfg,
                    runtime_capture_result.get("trajectory_evidence") if isinstance(runtime_capture_result.get("trajectory_evidence"), dict) else None,
                    content_result_pre_payload,
                )
                subspace_result_runtime_finalized = impl_set.subspace_planner.plan(
                    cfg,
                    mask_digest=mask_digest,
                    cfg_digest=cfg_digest,
                    inputs=planner_inputs_runtime,
                )
                plan_payload = _extract_subspace_plan_payload(subspace_result_runtime_finalized)
                plan_digest_precomputed, basis_digest_precomputed = _extract_subspace_result_digests(subspace_result_runtime_finalized)
                planner_failure_diagnostics = _extract_subspace_failure_diagnostics(subspace_result_runtime_finalized)
                if not isinstance(plan_payload, dict) or plan_digest_precomputed is None or basis_digest_precomputed is None:
                    runtime_finalization_reason = _resolve_runtime_executable_plan_failure_reason(
                        subspace_result_runtime_finalized,
                        runtime_capture_status=runtime_capture_status,
                    )
                    runtime_finalization_meta = {
                        "status": "failed",
                        "reason": runtime_finalization_reason,
                        "runtime_finalization_status": "failed",
                        "runtime_finalization_reason": runtime_finalization_reason,
                        "capture_inference_status": runtime_capture_status,
                        "capture_inference_error": runtime_capture_error,
                        "runtime_capture_cuda_memory_profile": runtime_capture_cuda_memory_profile,
                        "trajectory_cache_bound": runtime_capture_cache_usable,
                        "final_plan_digest": plan_digest_precomputed,
                        "final_basis_digest": basis_digest_precomputed,
                        **runtime_capture_diagnostics,
                        **planner_failure_diagnostics,
                    }
                    run_meta["formal_two_stage"] = {
                        "scaffold_status": "ok",
                        "scaffold_reason": None,
                        "formal_scaffold": formal_scaffold,
                        "runtime_executable_plan_status": "failed",
                        "runtime_executable_plan_reason": runtime_finalization_reason,
                        "runtime_finalization_status": "failed",
                        "runtime_finalization_reason": runtime_finalization_reason,
                        "runtime_capture_inference_status": runtime_capture_status,
                        "runtime_capture_inference_error": runtime_capture_error,
                        "runtime_capture_cuda_memory_profile": runtime_capture_cuda_memory_profile,
                        "final_plan_digest": plan_digest_precomputed,
                        "final_basis_digest": basis_digest_precomputed,
                        **runtime_capture_diagnostics,
                        **planner_failure_diagnostics,
                    }
                    raise ValueError(
                        "runtime executable formal plan unavailable: statement_only runtime finalization failed; "
                        f"reason={runtime_finalization_reason}"
                    )
                runtime_finalization_meta = {
                    "status": "ok",
                    "reason": None,
                    "runtime_finalization_status": "ok",
                    "runtime_finalization_reason": None,
                    "capture_inference_status": runtime_capture_status,
                    "capture_inference_error": runtime_capture_error,
                    "runtime_capture_cuda_memory_profile": runtime_capture_cuda_memory_profile,
                    "trajectory_cache_bound": runtime_capture_cache_usable,
                    "final_plan_digest": plan_digest_precomputed,
                    "final_basis_digest": basis_digest_precomputed,
                    **runtime_capture_diagnostics,
                    "planner_failure_stage": None,
                    "planner_failure_detail_code": None,
                    "planner_failure_detail_message": None,
                    "planner_diagnostic_context": None,
                }
                run_meta["formal_two_stage"] = {
                    "scaffold_status": "ok",
                    "scaffold_reason": None,
                    "formal_scaffold": formal_scaffold,
                    "runtime_executable_plan_status": "ok",
                    "runtime_executable_plan_reason": None,
                    "runtime_finalization_status": "ok",
                    "runtime_finalization_reason": None,
                    "runtime_capture_inference_status": runtime_capture_status,
                    "runtime_capture_inference_error": runtime_capture_error,
                    "runtime_capture_cuda_memory_profile": runtime_capture_cuda_memory_profile,
                    "final_plan_digest": plan_digest_precomputed,
                    "final_basis_digest": basis_digest_precomputed,
                    **runtime_capture_diagnostics,
                    "planner_failure_stage": None,
                    "planner_failure_detail_code": None,
                    "planner_failure_detail_message": None,
                    "planner_diagnostic_context": None,
                }

            if isinstance(plan_payload, dict) and isinstance(plan_digest_precomputed, str) and plan_digest_precomputed:
                early_attestation_payload = embed_orchestrator._prepare_embed_attestation_runtime_bindings(cfg, plan_digest_precomputed)  # pyright: ignore[reportPrivateUsage]
                if early_attestation_payload.get("attestation_status") == "ok":
                    embed_orchestrator._bind_attestation_runtime_to_cfg(cfg, early_attestation_payload)  # pyright: ignore[reportPrivateUsage]
                elif early_attestation_payload.get("attestation_status") not in {None, "absent"}:
                    print(
                        "[Paper-Faithful] [WARN] Early attestation runtime unavailable: "
                        f"{early_attestation_payload.get('attestation_failure_reason') or early_attestation_payload.get('attestation_absent_reason')}"
                    )
            
            # 延迟 injection_site_spec 创建到 POST-ORCHESTRATOR
            # 此处仅创建 injection_context 和 injection_modifier （驱动 inference）
            # injection_site_spec 将在 orchestrator 后根据真实 plan_digest 创建
            try:
                if isinstance(plan_payload, dict) and isinstance(plan_digest_precomputed, str) and plan_digest_precomputed:
                    # 计划存在：基于 PRE-COMPUTED plan 创建推理时所需的 context
                    # 注意：此 plan_digest 是临时的，真实 plan_digest 将由 orchestrator 计算
                    injection_context = build_injection_context_from_plan(cfg, cast(Dict[str, Any], plan_payload), plan_digest_precomputed)
                    injection_modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)
                    injection_plan_digest = plan_digest_precomputed
                    injection_plan_source = "precomputed"
                    print(f"[Paper-Faithful] Injection context created from PRE-COMPUTED plan (POST-ORCHESTRATOR决定最终spec)")
                else:
                    if statement_only_formal_path:
                        raise ValueError("runtime executable formal plan unavailable: finalized executable plan missing before injection")
                    # 计划缺失：基于 fallback plan 创建推理时所需的 context
                    fallback_plan_payload: Dict[str, Any] = {
                        "plan_status": "fallback_runtime_plan",
                        "planner_params": {
                            "rank": 8,
                            "source": "run_embed_fallback"
                        }
                    }
                    fallback_plan_digest = digests.canonical_sha256(fallback_plan_payload)
                    injection_context = build_injection_context_from_plan(cfg, fallback_plan_payload, fallback_plan_digest)
                    injection_modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)
                    injection_plan_digest = fallback_plan_digest
                    injection_plan_source = "fallback"
                    print(f"[Paper-Faithful] Injection context created from FALLBACK plan (POST-ORCHESTRATOR决定最终spec)")
            except Exception as inj_ctx_exc:
                print(f"[Paper-Faithful] [WARN] Injection context creation failed: {inj_ctx_exc}")
                injection_context = None
                injection_modifier = None
                injection_plan_digest = None
                injection_plan_source = None
            
            # (7.7) Real Dataflow Smoke: 在 pipeline_result 之后调用 inference
            seed = seed_value
            
            pipeline_fingerprint = None
            pipeline_fingerprint_digest = None
            if pipeline_obj is not None:
                try:
                    pipeline_fingerprint, pipeline_fingerprint_digest = pipeline_inspector.inspect_sd3_pipeline(
                        pipeline_obj, cfg
                    )
                    print(f"[Paper-Faithful] Pipeline fingerprint extracted: {pipeline_fingerprint_digest[:16]}...")
                except Exception as pipeline_insp_exc:
                    print(f"[Paper-Faithful] [WARN] Pipeline inspection failed: {pipeline_insp_exc}")
                    pipeline_fingerprint = {
                        "status": "failed",
                        "error": str(pipeline_insp_exc),
                        "transformer_num_blocks": "<absent>",
                        "scheduler_class_name": "<absent>",
                        "vae_latent_channels": "<absent>"
                    }
                    pipeline_fingerprint_digest = "<failed>"
            else:
                # Pipeline object 为 None：生成 absent 状态指纹，但仍包含必需的三个字段
                pipeline_fingerprint = {
                    "status": "absent",
                    "reason": "pipeline_obj_is_none",
                    "transformer_num_blocks": "<absent>",
                    "scheduler_class_name": "<absent>",
                    "vae_latent_channels": "<absent>"
                }
                pipeline_fingerprint_digest = digests.canonical_sha256(pipeline_fingerprint)
            
            enable_latent_sync = resolve_enable_latent_sync(cfg)
            # 为 planner 构造内存 latent 缓存（不写入 records）。
            from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache
            _traj_latent_cache = LatentTrajectoryCache()
            inference_result = infer_runtime.run_sd3_inference(
                cfg,
                pipeline_obj,
                device,
                seed,
                runtime_phase_label=EMBED_WATERMARKED_INFERENCE_PHASE_LABEL,
                injection_context=injection_context,
                injection_modifier=injection_modifier,
                capture_final_latents=enable_latent_sync,
                trajectory_latent_cache=_traj_latent_cache
            )
            inference_status_value = inference_result.get("inference_status")
            inference_status = (
                inference_status_value
                if isinstance(inference_status_value, str) and inference_status_value
                else infer_runtime.INFERENCE_STATUS_FAILED
            )
            inference_error = inference_result.get("inference_error")
            inference_runtime_meta = inference_result.get("inference_runtime_meta")
            trajectory_evidence = inference_result.get("trajectory_evidence")
            injection_evidence = inference_result.get("injection_evidence")
            final_latents = inference_result.get("final_latents") if enable_latent_sync else None
            # SD 推理输出图像（latent-per-step 模式下的水印图像）
            sd_output_image = inference_result.get("output_image")
            if pipeline_obj is not None:
                cfg["__embed_pipeline_obj__"] = pipeline_obj
            if final_latents is not None:
                cfg["__embed_final_latents__"] = final_latents
            # 将 latent 缓存传递给 planner（不写入 records，仅内存传递）。
            if not _traj_latent_cache.is_empty():
                cfg["__embed_trajectory_latent_cache__"] = _traj_latent_cache
            # 计算 embed 侧 latent 空间统计，作为 detect 侧几何同步的 cross-comparison 基线。
            if final_latents is not None:
                try:
                    import numpy as _np_lat
                    _lat = final_latents
                    if hasattr(_lat, "cpu"):
                        _lat = _lat.cpu()
                    if hasattr(_lat, "numpy"):
                        _lat = _lat.numpy()
                    _arr = _np_lat.asarray(_lat, dtype=_np_lat.float32)
                    if _arr.ndim == 4:
                        _se = _np_lat.mean(_np_lat.abs(_arr[0]), axis=0)
                        _std = float(_np_lat.std(_se))
                        _mean = float(cast(Any, _np_lat.mean(_se))) + 1e-12
                        _cr = _std / _mean
                        _flat = _se.reshape(-1)
                        _pk = float(_np_lat.max(_flat))
                        _median_value = cast(Any, _np_lat.median(_flat))
                        _med = float(_median_value) + 1e-12
                        _spec = _np_lat.fft.fftshift(_np_lat.abs(_np_lat.fft.fft2(_se)))
                        _cy, _cx = _spec.shape[0] // 2, _spec.shape[1] // 2
                        _spec[_cy, _cx] = 0.0
                        _sv = _np_lat.sort(_spec.reshape(-1))
                        _t1 = float(_sv[-1]) if _sv.size > 0 else 0.0
                        _t2 = float(_sv[-2]) if _sv.size > 1 else 0.0
                        _spr = float(_t1 / (_t2 + 1e-12)) if _t1 > 0.0 else 0.0
                        cfg["__embed_latent_spatial_stats__"] = {
                            "contrast_ratio": round(_cr, 6),
                            "peak_sharpness": round(float(_pk / _med), 6),
                            "spectral_peak_ratio": round(_spr, 6),
                            "stats_version": "v1",
                        }
                except Exception:
                    pass  # 统计计算失败不阻断主流程
            
            # 构造 infer_trace 并计算 digest
            infer_trace_obj = infer_trace.build_infer_trace(
                cfg,
                inference_status,
                inference_error,
                inference_runtime_meta,
                trajectory_evidence
            )
            infer_trace_canon_sha256 = infer_trace.compute_infer_trace_canon_sha256(infer_trace_obj)
            
            # 将 inference 相关字段写入 run_meta（append-only）
            run_meta["infer_trace"] = infer_trace_obj
            run_meta["infer_trace_canon_sha256"] = infer_trace_canon_sha256
            run_meta["inference_status"] = inference_status
            run_meta["inference_error"] = inference_error
            run_meta["inference_runtime_meta"] = inference_runtime_meta
            run_meta["trajectory_evidence"] = trajectory_evidence
            
            # 写入 cfg_audit 工件到 artifacts/cfg_audit/cfg_audit.json。
            cfg_audit_dir = artifacts_dir / "cfg_audit"
            cfg_audit_dir.mkdir(parents=True, exist_ok=True)
            cfg_audit_path = cfg_audit_dir / "cfg_audit.json"
            try:
                records_io.write_artifact_json(str(cfg_audit_path), cfg_audit_metadata)
            except records_io.FactSourcesNotInitializedError:
                records_io.write_artifact_json_unbound(run_root, artifacts_dir, str(cfg_audit_path), cfg_audit_metadata)
            
            # 将 cfg_audit digest 锚定字段写入 run_meta。
            run_meta["cfg_pruned_for_digest_canon_sha256"] = cfg_audit_metadata["cfg_pruned_for_digest_canon_sha256"]
            run_meta["cfg_audit_canon_sha256"] = cfg_audit_metadata["cfg_audit_canon_sha256"]

            # 记录路径策略配置与审计字段。
            run_meta["path_policy"] = {
                "allow_nonempty_run_root": bool(allow_nonempty_run_root),
                "allow_nonempty_run_root_reason": allow_nonempty_run_root_reason
            }
            run_meta["run_root_reuse_allowed"] = bool(allow_nonempty_run_root)
            run_meta["run_root_reuse_reason"] = allow_nonempty_run_root_reason

            # 构造 embed record，本阶段为基线实现。
            print("[Embed] Generating embed record (baseline)...")
            sync_runtime_context = SyncRuntimeContext(
                pipeline=pipeline_obj,
                latents=final_latents,
                rng=None,
                trajectory_evidence=trajectory_evidence
            )
            effective_subspace_result = subspace_result_runtime_finalized if statement_only_formal_path else subspace_result_pre
            subspace_result_override = _resolve_formal_subspace_override(effective_subspace_result)
            record_obj: Any = run_embed_orchestrator(
                cfg,
                impl_set,
                cfg_digest,
                trajectory_evidence=trajectory_evidence,
                injection_evidence=injection_evidence,
                sync_runtime_context=sync_runtime_context,
                content_result_override=cast(Any, content_result_pre),
                subspace_result_override=subspace_result_override,
            )
            cfg.pop("__attestation_secret_inputs__", None)
            cfg.pop("__attestation_event_nonce__", None)
            cfg.pop("__attestation_time_bucket__", None)
            cfg.pop("__embed_pipeline_obj__", None)
            cfg.pop("__embed_final_latents__", None)
            cfg.pop("__embed_trajectory_latent_cache__", None)
            cfg.pop("__formal_scaffold__", None)
            # 将 embed 侧 latent 空间统计写入 record，供 detect 侧几何同步 cross-comparison。
            _embed_latent_stats = cfg.pop("__embed_latent_spatial_stats__", None)
            if not isinstance(record_obj, dict):
                # record 类型不符合预期，必须 fail-fast。
                raise TypeError("orchestrator output must be dict")
            record = cast(Dict[str, Any], record_obj)
            record.pop("_lf_planner_risk_report_artifact", None)
            if isinstance(_embed_latent_stats, dict):
                record["latent_spatial_stats"] = _embed_latent_stats
            record["cfg_digest"] = cfg_digest
            record["policy_path"] = cfg["policy_path"]

            # 若 watermarked_path 仍为占位值（latent-per-step 模式下 orchestrator 不写磁盘图像），
            # 则将 sd_output_image 保存到 artifacts/watermarked/watermarked.png，并回填 record。
            if (
                record.get("watermarked_path") in ("<absent>", None)
                and sd_output_image is not None
                and hasattr(sd_output_image, "save")
            ):
                try:
                    _wm_output_rel = "watermarked/watermarked.png"
                    _artifacts_dir_str = cfg.get("__artifacts_dir__")
                    _run_root_str = cfg.get("__run_root_dir__")
                    if isinstance(_artifacts_dir_str, str) and isinstance(_run_root_str, str):
                        from pathlib import Path as _Path
                        _artifacts_dir = _Path(_artifacts_dir_str)
                        _run_root = _Path(_run_root_str)
                        _wm_out_path = (_artifacts_dir / _wm_output_rel).resolve()
                        _wm_out_path.parent.mkdir(parents=True, exist_ok=True)
                        from main.policy import path_policy as _pp
                        _pp.validate_output_target(_wm_out_path, "artifact", _run_root)
                        sd_output_image.save(str(_wm_out_path), format="PNG")
                        _wm_sha256 = digests.file_sha256(_wm_out_path)
                        try:
                            _wm_rel = str(_wm_out_path.relative_to(_run_root))
                        except ValueError:
                            _wm_rel = str(_wm_out_path)
                        record["watermarked_path"] = str(_wm_out_path)
                        record["artifact_sha256"] = _wm_sha256
                        record["artifact_rel_path"] = _wm_rel
                        record["watermarked_artifact_sha256"] = _wm_sha256
                        record["watermarked_artifact_rel_path"] = _wm_rel
                        record["watermarked_path_source"] = "sd_output_image"
                        print(f"[Embed] SD 输出图像已保存：{_wm_out_path} (sha256={_wm_sha256[:16]}...)")
                    else:
                        print("[Embed] [WARN] __artifacts_dir__ 或 __run_root_dir__ 未设置，跳过 SD 输出图像保存")
                except Exception as _wm_save_exc:
                    # 保存失败时写入审计字段，不中断流程。
                    record["watermarked_path_save_error"] = f"{type(_wm_save_exc).__name__}: {_wm_save_exc}"
                    print(f"[Embed] [WARN] SD 输出图像保存失败：{_wm_save_exc}")
            record["pipeline_impl_id"] = pipeline_result.get("pipeline_impl_id")
            record["pipeline_provenance"] = pipeline_result.get("pipeline_provenance")
            record["pipeline_provenance_canon_sha256"] = pipeline_result.get("pipeline_provenance_canon_sha256")
            record["pipeline_runtime_meta"] = pipeline_result.get("pipeline_runtime_meta")
            record["env_fingerprint_canon_sha256"] = pipeline_result.get("env_fingerprint_canon_sha256")
            record["diffusers_version"] = pipeline_result.get("diffusers_version")
            record["transformers_version"] = pipeline_result.get("transformers_version")
            record["safetensors_version"] = pipeline_result.get("safetensors_version")
            record["model_provenance_canon_sha256"] = pipeline_result.get("model_provenance_canon_sha256")
            
            # 将 inference 相关字段写入 record（append-only）
            record["infer_trace"] = run_meta.get("infer_trace")
            record["infer_trace_canon_sha256"] = run_meta.get("infer_trace_canon_sha256")
            record["inference_status"] = run_meta.get("inference_status")
            record["inference_error"] = run_meta.get("inference_error")
            record["inference_runtime_meta"] = run_meta.get("inference_runtime_meta")
            
            # 为 detect 侧保存关键输入信息：input_image_path 和输入配置（用于重建）
            # 注：final_latents 张量不序列化；detect 侧应通过自己的 inference 或从 embed 配置重建
            inputs_record = {}
            if validated_input_image_path is not None:
                inputs_record["input_image_path"] = validated_input_image_path
            else:
                cfg_input_image_path = cfg.get("__embed_input_image_path__")
                if isinstance(cfg_input_image_path, str) and cfg_input_image_path.strip():
                    inputs_record["input_image_path"] = cfg_input_image_path.strip()
            if inputs_record:
                record["inputs"] = inputs_record
            
            content_evidence_node = record.get("content_evidence")
            if isinstance(content_evidence_node, dict):
                content_evidence: Dict[str, Any] = cast(Dict[str, Any], content_evidence_node)
            else:
                content_evidence = {}
                record["content_evidence"] = content_evidence

            content_audit = content_evidence.get("audit")
            if not isinstance(content_audit, dict):
                content_audit = {}
                content_evidence["audit"] = content_audit
            if statement_only_formal_path:
                content_audit["formal_scaffold_status"] = "ok" if isinstance(formal_scaffold, dict) else "failed"
                if isinstance(formal_scaffold, dict):
                    content_audit["formal_scaffold_digest"] = formal_scaffold.get("formal_scaffold_digest")
                content_audit["runtime_executable_plan_status"] = runtime_finalization_meta.get("status")
                content_audit["runtime_executable_plan_reason"] = runtime_finalization_meta.get("reason")
                content_audit["runtime_finalization_status"] = runtime_finalization_meta.get("runtime_finalization_status")
                content_audit["runtime_finalization_reason"] = runtime_finalization_meta.get("runtime_finalization_reason")
                content_audit["runtime_capture_inference_status"] = runtime_finalization_meta.get("capture_inference_status")
                content_audit["runtime_capture_inference_error"] = runtime_finalization_meta.get("capture_inference_error")
                runtime_capture_cuda_memory_profile = runtime_finalization_meta.get("runtime_capture_cuda_memory_profile")
                if isinstance(runtime_capture_cuda_memory_profile, dict):
                    content_audit["runtime_capture_cuda_memory_profile"] = dict(
                        cast(Dict[str, Any], runtime_capture_cuda_memory_profile)
                    )
                content_audit["planner_failure_stage"] = runtime_finalization_meta.get("planner_failure_stage")
                content_audit["planner_failure_detail_code"] = runtime_finalization_meta.get("planner_failure_detail_code")
                content_audit["planner_failure_detail_message"] = runtime_finalization_meta.get("planner_failure_detail_message")
                content_audit["planner_diagnostic_context"] = runtime_finalization_meta.get("planner_diagnostic_context")
            
            # 写入 pipeline_fingerprint 和 pipeline_fingerprint_digest
            content_evidence["pipeline_fingerprint"] = pipeline_fingerprint
            content_evidence["pipeline_fingerprint_digest"] = pipeline_fingerprint_digest

            formal_plan_digest = _normalize_plan_digest(
                content_evidence.get("plan_digest") or record.get("plan_digest")
            )
            formal_basis_digest = _normalize_plan_digest(
                content_evidence.get("basis_digest") or record.get("basis_digest")
            )
            if formal_plan_digest is not None and not content_evidence.get("plan_digest"):
                content_evidence["plan_digest"] = formal_plan_digest
            if formal_basis_digest is not None and not content_evidence.get("basis_digest"):
                content_evidence["basis_digest"] = formal_basis_digest
            if injection_plan_source == "fallback" and isinstance(injection_plan_digest, str) and injection_plan_digest:
                content_evidence["fallback_plan_digest"] = injection_plan_digest
                if not isinstance(content_evidence.get("fallback_plan_digest_reason"), str) or not content_evidence.get("fallback_plan_digest_reason"):
                    content_evidence["fallback_plan_digest_reason"] = "fallback_injection_plan_used"
            _bind_embed_plan_digest_consistency(
                record,
                content_evidence,
                injection_plan_digest,
                formal_plan_digest,
            )
            
            # POST-ORCHESTRATOR 创建最终的 injection_site_spec
            # 注入位点必须绑定 injection-time 计划摘要，而不是 post-inference formal 摘要。
            try:
                if injection_plan_source == "precomputed" and injection_plan_digest is not None:
                    # injection-time plan_digest 存在 → subspace_projection 模式
                    injection_site_spec, injection_site_digest = injection_site_binder.build_injection_site_spec(
                        hook_type="callback_on_step_end",
                        target_module_name="StableDiffusion3Pipeline",
                        target_tensor_name="latents",
                        hook_timing="after_scheduler_step",
                        injection_rule_summary={
                            "plan_digest": injection_plan_digest,
                            "injection_mode": "subspace_projection"
                        },
                        cfg=cfg
                    )
                    print(f"[Paper-Faithful] Injection site spec built (POST-ORCHESTRATOR with injection plan_digest): {injection_site_digest[:16]}...")
                elif injection_plan_source == "fallback" and injection_plan_digest is not None:
                    injection_site_spec, injection_site_digest = injection_site_binder.build_injection_site_spec(
                        hook_type="callback_on_step_end",
                        target_module_name="StableDiffusion3Pipeline",
                        target_tensor_name="latents",
                        hook_timing="after_scheduler_step",
                        injection_rule_summary={
                            "plan_digest": injection_plan_digest,
                            "injection_mode": "latent_direct_fallback"
                        },
                        cfg=cfg
                    )
                    content_evidence["fallback_plan_digest"] = injection_plan_digest
                    if not isinstance(content_evidence.get("fallback_plan_digest_reason"), str) or not content_evidence.get("fallback_plan_digest_reason"):
                        content_evidence["fallback_plan_digest_reason"] = "fallback_injection_plan_used"
                    print(f"[Paper-Faithful] Injection site spec built (POST-ORCHESTRATOR fallback): {injection_site_digest[:16]}...")
                else:
                    # injection-time plan_digest 缺失 → fallback 模式
                    fallback_plan_payload: Dict[str, Any] = {
                        "plan_status": "fallback_runtime_plan_post_orchestrator",
                        "planner_params": {
                            "rank": 8,
                            "source": "post_orchestrator_fallback"
                        }
                    }
                    fallback_plan_digest = digests.canonical_sha256(fallback_plan_payload)
                    injection_site_spec, injection_site_digest = injection_site_binder.build_injection_site_spec(
                        hook_type="callback_on_step_end",
                        target_module_name="StableDiffusion3Pipeline",
                        target_tensor_name="latents",
                        hook_timing="after_scheduler_step",
                        injection_rule_summary={
                            "plan_digest": fallback_plan_digest,
                            "injection_mode": "latent_direct_fallback"
                        },
                        cfg=cfg
                    )
                    print(f"[Paper-Faithful] Injection site spec built (POST-ORCHESTRATOR fallback): {injection_site_digest[:16]}...")
                    # content_status!=ok 时，不写 fallback 摘要到语义 plan 字段。
                    content_status = content_evidence.get("status")
                    if content_status == "ok":
                        content_evidence["plan_digest"] = fallback_plan_digest
                    else:
                        content_evidence["fallback_plan_digest"] = fallback_plan_digest
                        content_evidence["fallback_plan_digest_reason"] = "content_status_not_ok"
            except Exception as final_inj_exc:
                print(f"[Paper-Faithful] [WARN] Final injection site building failed: {final_inj_exc}")
                injection_site_spec = {"status": "failed", "error": str(final_inj_exc)}
                injection_site_digest = "<failed>"
            
            # 写入 injection_site_spec 和 injection_site_digest
            content_evidence["injection_site_spec"] = injection_site_spec
            content_evidence["injection_site_digest"] = injection_site_digest
            
            # Paper Faithfulness: 调用 alignment_evaluator（必达）
            paper_spec: Dict[str, Any] = {"status": "absent", "reason": "paper_spec_uninitialized"}
            paper_spec_digest = "<absent>"
            try:
                # 加载 paper_faithfulness_spec.yaml（通过唯一入口）
                from pathlib import Path as PathLib
                spec_path = PathLib(__file__).parent.parent.parent / "configs" / "paper_faithfulness_spec.yaml"
                if spec_path.exists():
                    # 使用 config_loader 唯一入口加载 YAML
                    paper_spec_loaded, spec_provenance = config_loader.load_yaml_with_provenance(spec_path)
                    paper_spec = dict(paper_spec_loaded)
                    paper_spec_digest = spec_provenance.canon_sha256
                    print(f"[Paper-Faithful] Paper spec loaded: {spec_path.name}, digest: {paper_spec_digest[:16]}...")
                else:
                    print(f"[Paper-Faithful] [WARN] Paper spec not found: {spec_path}")
                    paper_spec = {"status": "absent", "reason": "spec_file_not_found"}
                    paper_spec_digest = "<absent>"
            except Exception as spec_load_exc:
                print(f"[Paper-Faithful] [WARN] Failed to load paper spec: {spec_load_exc}")
                paper_spec = {"status": "failed", "error": str(spec_load_exc)}
                paper_spec_digest = "<failed>"
            
            # 调用 alignment_evaluator
            alignment_report: Dict[str, Any] = {"status": "absent", "reason": "alignment_uninitialized"}
            alignment_digest = "<absent>"
            if paper_spec.get("status") not in ("absent", "failed"):
                try:
                    alignment_report, alignment_digest = alignment_evaluator.evaluate_alignment(
                        paper_spec=paper_spec,
                        pipeline_fingerprint=pipeline_fingerprint,
                        trajectory_evidence=trajectory_evidence,
                        injection_site_spec=injection_site_spec,
                        cfg=cfg
                    )
                    print(f"[Paper-Faithful] Alignment evaluated: {alignment_report['overall_status']}, digest: {alignment_digest[:16]}...")
                except Exception as align_exc:
                    print(f"[Paper-Faithful] [WARN] Alignment evaluation failed: {align_exc}")
                    alignment_report = {"status": "failed", "error": str(align_exc)}
                    alignment_digest = "<failed>"
            else:
                alignment_report = {"status": "absent", "reason": "paper_spec_not_available"}
                alignment_digest = "<absent>"
            
            # 写入 alignment_report 和 alignment_digest
            content_evidence["alignment_report"] = alignment_report
            content_evidence["alignment_digest"] = alignment_digest
            
            # 写入 paper_spec 绑定字段到顶层 record
            if not isinstance(record.get("paper_faithfulness"), dict):
                record["paper_faithfulness"] = {}
            record["paper_faithfulness"]["spec_version"] = paper_spec.get("paper_faithfulness_spec_version", "<absent>")
            record["paper_faithfulness"]["spec_digest"] = paper_spec_digest
            
            override_applied = cfg.get("override_applied")
            if override_applied is not None:
                record["override_applied"] = override_applied
            # 写入 impl_set_capabilities_digest。
            record["impl_set_capabilities_digest"] = impl_set_capabilities_digest
            if isinstance(impl_set_capabilities_extended_digest, str) and impl_set_capabilities_extended_digest:
                record["impl_set_capabilities_extended_digest"] = impl_set_capabilities_extended_digest

            schema.ensure_required_fields(record, cfg, interpretation)

            # 口径版本标识写入 run_meta。
            run_meta["thresholds_rule_id"] = record.get("thresholds_rule_id", "<absent>")
            run_meta["thresholds_rule_version"] = record.get("thresholds_rule_version", "<absent>")

            # 绑定 impl_identity 字段族。
            bind_impl_identity_fields(record, impl_identity, impl_set, contracts)

            # 绑定事实源字段。
            print("[Embed] Binding fact source fields...")
            bind_contract_to_record(record, contracts)
            bind_whitelist_to_record(record, whitelist)
            bind_semantics_to_record(record, semantics)

            try:
                schema.validate_record(record, interpretation=interpretation)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
                raise

            # 写盘，触发 freeze_gate.assert_prewrite。
            record_path = records_dir / "embed_record.json"
            path_policy.validate_output_target(record_path, "record", run_root)
            print(f"[Embed] Writing record to {record_path}...")
            try:
                records_io.write_json(str(record_path), record)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.GATE_FAILED, exc)
                raise

            _write_embed_attestation_artifacts(record, artifacts_dir)
    except Exception as exc:
        if run_meta.get("status_ok", True):
            set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
            _merge_runtime_finalization_status_details(run_meta)
        error = exc
    finally:
        # Fallback: 若布局未成功初始化，则强制创建默认布局。
        if not layout_initialized:
            try:
                layout = path_policy.ensure_output_layout(
                    run_root,
                    allow_nonempty_run_root=False,
                    allow_nonempty_run_root_reason=None,
                    override_applied=None
                )
                records_dir = layout["records_dir"]
                artifacts_dir = layout["artifacts_dir"]
                logs_dir = layout["logs_dir"]
            except Exception:
                pass
        # 绑定闭包最小合同字段，并产出 run_closure.json。
        if error is None and isinstance(run_meta.get("formal_two_stage"), dict):
            _merge_runtime_finalization_status_details(run_meta)
        run_meta["ended_at"] = time_utils.now_utc_iso_z()
        try:
            status.finalize_run(run_root, records_dir, artifacts_dir, run_meta)
        except Exception:
            # finalize_run 失败必须 fail-fast。
            raise
        if error is not None:
            raise error

    if record is None:
        # 成功路径缺失 record 不符合预期，必须 fail-fast。
        raise RuntimeError("embed record missing after successful execution")

    print(f"[Embed] [OK] Embed record written successfully")
    print(f"[Embed]   Record contains {len(record)} fields (15 fact source fields + {len(record) - 15} business fields)")



def main():
    """主流程。"""
    parser = argparse.ArgumentParser(
        description="Embed watermark (baseline implementation)"
    )
    parser.add_argument(
        "--out",
        default="tmp/cli_smoke/embed_run",
        help="Output run root directory"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config YAML path"
    )
    parser.add_argument(
        "--override",
        "--overrides",
        action="append",
        default=None,
        help="Override config key=value (JSON value). Can be repeated."
    )
    parser.add_argument(
        "--input-image",
        default=None,
        help="Input image path for identity embed artifact generation"
    )
    
    args = parser.parse_args()
    
    try:
        run_embed(args.out, args.config, args.override, args.input_image)
        sys.exit(0)
    except Exception as e:
        print(f"[Embed] [ERROR] Error: {e}", file=sys.stderr)
        hint = build_cli_config_migration_hint(e)
        if isinstance(hint, str) and hint:
            print(f"[Embed] [HINT] {hint}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
