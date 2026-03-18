"""
嵌入水印 CLI 入口

功能说明：
- 规范化输出目录路径，确保输出布局，加载合同与白名单，验证配置，解析实现，构造记录，绑定字段，写盘，并产出闭包。
- 包含详细的输入验证、错误处理与状态更新机制，确保健壮性与可维护性。
- 当前为可审计基线实现，新能力仅通过版本化追加，不改变既有冻结语义与字段口径。
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
import uuid

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
from main.core.errors import MissingRequiredFieldError, RunFailureReason
from main.registries import runtime_resolver
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
    set_value_by_field_path,
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


def _write_embed_attestation_artifacts(
    record: dict,
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
    attestation_node = record.get("attestation")
    attestation_payload = attestation_node if isinstance(attestation_node, dict) else {}
    if attestation_payload.get("status") != "ok":
        return

    attestation_dir = artifacts_dir / "attestation"
    attestation_dir.mkdir(parents=True, exist_ok=True)
    statement_payload = {
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
        records_io.write_artifact_json(str(attestation_dir / "attestation_bundle.json"), signed_bundle)


def bind_impl_identity_fields(
    record: dict,
    identity: runtime_resolver.ImplIdentity,
    impl_set: runtime_resolver.BuiltImplSet,
    contracts
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


def run_embed(
    output_dir: str,
    config_path: str,
    overrides: list[str] | None = None,
    input_image_path: str | None = None
) -> None:
    """
    功能：执行嵌入流程，本阶段为基线实现。

    Execute embed workflow (baseline implementation).

    Args:
        output_dir: Run root directory for records/artifacts.
        config_path: YAML config path.
        overrides: Optional CLI override args list.
        input_image_path: Optional input image path for identity embed artifact flow.
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

    # 创建 layout 和最小 run_meta。
    run_root = path_policy.derive_run_root(Path(output_dir))
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    layout_initialized = False
    allow_nonempty_run_root = False
    allow_nonempty_run_root_reason = None
    override_applied_for_layout = None
    
    # 初始化最小 run_meta，待完善。
    started_at = time_utils.now_utc_iso_z()
    run_meta = {
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
    pipeline_result = None
    try:
        # 加载事实源。
        print("[Embed] Loading fact sources...")
        contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
        whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
        semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
        injection_scope_manifest = config_loader.load_injection_scope_manifest()

        # 绑定冻结锚点到 run_meta。
        print("[Embed] Binding freeze anchors to run_meta...")
        status.bind_freeze_anchors_to_run_meta(
            run_meta,
            contracts,
            whitelist,
            semantics,
            injection_scope_manifest
        )

        # 生成事实源快照用于运行期一致性校验。
        snapshot = records_io.build_fact_sources_snapshot(
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
            interpretation = get_contract_interpretation(contracts)
            try:
                cfg, cfg_digest, cfg_audit_metadata = config_loader.load_and_validate_config(
                    config_path,
                    whitelist,
                    semantics,
                    contracts,
                    interpretation,
                    overrides=overrides
                )
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.CONFIG_INVALID, exc)
                raise
            run_meta["cfg_digest"] = cfg_digest
            run_meta["policy_path"] = cfg["policy_path"]

            seed_parts, seed_digest, seed_value, seed_rule_id = build_seed_audit(cfg, "embed")
            cfg["seed"] = seed_value
            run_meta["seed_parts"] = seed_parts
            run_meta["seed_digest"] = seed_digest
            run_meta["seed_rule_id"] = seed_rule_id
            run_meta["seed_value"] = seed_value
            run_meta["cfg"] = cfg

            determinism_controls = build_determinism_controls(cfg)
            if determinism_controls is not None:
                run_meta["determinism_controls"] = determinism_controls
            nondeterminism_notes = normalize_nondeterminism_notes(cfg.get("nondeterminism_notes"))
            if nondeterminism_notes is not None:
                run_meta["nondeterminism_notes"] = nondeterminism_notes

            pipeline_result = pipeline_factory.build_pipeline_shell(cfg)
            run_meta["pipeline_provenance_canon_sha256"] = pipeline_result.get("pipeline_provenance_canon_sha256")
            run_meta["pipeline_status"] = pipeline_result.get("pipeline_status")
            run_meta["pipeline_error"] = pipeline_result.get("pipeline_error")
            run_meta["pipeline_runtime_meta"] = pipeline_result.get("pipeline_runtime_meta")
            run_meta["env_fingerprint_canon_sha256"] = pipeline_result.get("env_fingerprint_canon_sha256")
            run_meta["diffusers_version"] = pipeline_result.get("diffusers_version")
            run_meta["transformers_version"] = pipeline_result.get("transformers_version")
            run_meta["safetensors_version"] = pipeline_result.get("safetensors_version")
            run_meta["model_provenance_canon_sha256"] = pipeline_result.get("model_provenance_canon_sha256")

            try:
                impl_identity, impl_set, impl_set_capabilities_digest = runtime_resolver.build_runtime_impl_set_from_cfg(
                    cfg,
                    whitelist,
                )
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.IMPL_RESOLVE_FAILED, exc)
                raise
            run_meta["impl_id"] = impl_identity.content_extractor_id
            run_meta["impl_version"] = impl_set.content_extractor.impl_version
            run_meta["impl_identity"] = impl_identity.as_dict()
            run_meta["impl_identity_digest"] = runtime_resolver.compute_impl_identity_digest(impl_identity)
            run_meta["impl_set_capabilities_digest"] = impl_set_capabilities_digest
            impl_set_capabilities_extended_digest = cfg.get("impl_set_capabilities_extended_digest")
            if isinstance(impl_set_capabilities_extended_digest, str) and impl_set_capabilities_extended_digest:
                run_meta["impl_set_capabilities_extended_digest"] = impl_set_capabilities_extended_digest

            if isinstance(input_image_path, str) and input_image_path.strip():
                cfg["__embed_input_image_path__"] = input_image_path.strip()
            else:
                embed_cfg = cfg.get("embed") if isinstance(cfg.get("embed"), dict) else {}
                default_input = embed_cfg.get("input_image_path") if isinstance(embed_cfg, dict) else None
                if isinstance(default_input, str) and default_input.strip():
                    cfg["__embed_input_image_path__"] = default_input.strip()

            # Preview Generation：若主链输入图仍为 None，且配置启用了 preview_generation，
            # 则先执行一次无注入 SD3 推理，将生成图作为语义掩码的输入，消除对外部图像的依赖。
            # preview 推理失败时不中断流程，失败语义传播至内容链（injection_mode 降级）。
            _embed_cfg_pg = (cfg.get("embed") or {}).get("preview_generation") or {}
            if _embed_cfg_pg.get("enabled", False) and cfg.get("__embed_input_image_path__") is None:
                preview_pipeline_obj = pipeline_result.get("pipeline_obj") if isinstance(pipeline_result, dict) else None
                preview_device = cfg.get("device", "cpu")
                preview_seed = seed_value
                _pg_status = "failed"
                _pg_reason = None
                _preview_tmp_path = None
                try:
                    import tempfile
                    _preview_infer_result = infer_runtime.run_sd3_inference(
                        cfg,
                        preview_pipeline_obj,
                        preview_device,
                        preview_seed,
                        injection_context=None,
                        injection_modifier=None,
                        capture_final_latents=False
                    )
                    _preview_status = None
                    if isinstance(_preview_infer_result, dict):
                        _preview_status = _preview_infer_result.get("inference_status")
                        if not isinstance(_preview_status, str) or not _preview_status:
                            _preview_status = _preview_infer_result.get("status")
                    if _preview_status == "ok":
                        _preview_image = _preview_infer_result.get("output_image")
                        if _preview_image is not None:
                            _tmp_fd, _preview_tmp_path = tempfile.mkstemp(suffix=".png", prefix="ceg_wm_preview_")
                            import os as _os
                            _os.close(_tmp_fd)
                            _preview_image.save(_preview_tmp_path)
                            cfg["__embed_input_image_path__"] = _preview_tmp_path
                            _pg_status = "ok"
                            print(f"[Preview Generation] 预览图已生成，路径：{_preview_tmp_path}")
                        else:
                            _pg_reason = "preview_inference_no_output_image"
                            print(f"[Preview Generation] 推理成功但无 output_image，跳过。")
                    else:
                        _pg_reason = f"preview_inference_status={_preview_status}"
                        print(f"[Preview Generation] 推理状态非 ok（{_preview_status}），跳过。")
                except Exception as _pg_exc:
                    _pg_reason = str(_pg_exc)
                    print(f"[Preview Generation] 推理异常：{_pg_exc}，跳过。")
                run_meta["preview_generation"] = {
                    "enabled": True,
                    "status": _pg_status,
                    "reason": _pg_reason,
                    "seed": preview_seed,
                    "tmp_path": _preview_tmp_path,
                }
                if _pg_status != "ok":
                    # preview 失败时保持 __embed_input_image_path__ 为 None，
                    # 内容链将以 status="failed" 返回，injection_mode 降级为 latent_direct_fallback。
                    cfg.pop("__embed_input_image_path__", None)
            elif not _embed_cfg_pg.get("enabled", False):
                run_meta["preview_generation"] = {"enabled": False, "status": "skipped", "reason": None, "seed": None, "tmp_path": None}

            # 预先计算 content 与 subspace 计划，用于注入上下文。
            content_inputs_pre = embed_orchestrator._build_content_inputs_for_embed(cfg)
            content_result_pre = impl_set.content_extractor.extract(
                cfg,
                inputs=content_inputs_pre,
                cfg_digest=cfg_digest
            )
            mask_digest = None
            if isinstance(content_result_pre, dict):
                mask_digest = content_result_pre.get("mask_digest")
            elif hasattr(content_result_pre, "mask_digest"):
                mask_digest = content_result_pre.mask_digest

            planner_inputs = embed_orchestrator._build_planner_inputs_for_runtime(cfg, None)
            subspace_result_pre = impl_set.subspace_planner.plan(
                cfg,
                mask_digest=mask_digest,
                cfg_digest=cfg_digest,
                inputs=planner_inputs
            )

            # 提取内层 plan dict（含 lf_basis/hf_basis 顶层键），供 injection context 使用。
            # 不能用 as_dict()（外层 evidence 包装），否则 _build_plan_for_injection 找不到
            # lf_basis 顶层键，会回退到运行时 basis 绑定，导致 embed/detect basis 不一致。
            if hasattr(subspace_result_pre, "plan") and isinstance(subspace_result_pre.plan, dict):
                plan_payload = subspace_result_pre.plan
            elif hasattr(subspace_result_pre, "as_dict"):
                _evidence_dict = subspace_result_pre.as_dict()
                plan_payload = _evidence_dict.get("plan") if isinstance(_evidence_dict.get("plan"), dict) else _evidence_dict
            else:
                plan_payload = subspace_result_pre
            plan_digest = getattr(subspace_result_pre, "plan_digest", None)
            if isinstance(plan_payload, dict) and not isinstance(plan_digest, str):
                plan_digest = plan_payload.get("plan_digest")

            injection_context = None
            injection_modifier = None
            injection_site_spec = None
            injection_site_digest = None
            
            # 延迟 injection_site_spec 创建到 POST-ORCHESTRATOR
            # 此处仅创建 injection_context 和 injection_modifier （驱动 inference）
            # injection_site_spec 将在 orchestrator 后根据真实 plan_digest 创建
            try:
                if isinstance(plan_payload, dict) and isinstance(plan_digest, str) and plan_digest:
                    # 计划存在：基于 PRE-COMPUTED plan 创建推理时所需的 context
                    # 注意：此 plan_digest 是临时的，真实 plan_digest 将由 orchestrator 计算
                    injection_context = build_injection_context_from_plan(cfg, plan_payload, plan_digest)
                    injection_modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)
                    print(f"[Paper-Faithful] Injection context created from PRE-COMPUTED plan (POST-ORCHESTRATOR决定最终spec)")
                else:
                    # 计划缺失：基于 fallback plan 创建推理时所需的 context
                    fallback_plan_payload = {
                        "plan_status": "fallback_runtime_plan",
                        "planner_params": {
                            "rank": 8,
                            "source": "run_embed_fallback"
                        }
                    }
                    fallback_plan_digest = digests.canonical_sha256(fallback_plan_payload)
                    injection_context = build_injection_context_from_plan(cfg, fallback_plan_payload, fallback_plan_digest)
                    injection_modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)
                    print(f"[Paper-Faithful] Injection context created from FALLBACK plan (POST-ORCHESTRATOR决定最终spec)")
            except Exception as inj_ctx_exc:
                print(f"[Paper-Faithful] [WARN] Injection context creation failed: {inj_ctx_exc}")
                injection_context = None
                injection_modifier = None
            
            # (7.7) Real Dataflow Smoke: 在 pipeline_result 之后调用 inference
            pipeline_obj = pipeline_result.get("pipeline_obj")
            device = cfg.get("device", "cpu")
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
                injection_context=injection_context,
                injection_modifier=injection_modifier,
                capture_final_latents=enable_latent_sync,
                trajectory_latent_cache=_traj_latent_cache
            )
            inference_status = inference_result.get("inference_status")
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
                        _mean = float(_np_lat.mean(_se) + 1e-12)
                        _cr = _std / _mean
                        _flat = _se.reshape(-1)
                        _pk = float(_np_lat.max(_flat))
                        _med = float(_np_lat.median(_flat) + 1e-12)
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
            
            # 提取 run_root 复用参数。
            allow_nonempty_run_root = cfg.get("allow_nonempty_run_root", False)
            allow_nonempty_run_root_reason = cfg.get("allow_nonempty_run_root_reason")
            override_applied_for_layout = cfg.get("override_applied")
            
            # 创建输出布局。
            layout = path_policy.ensure_output_layout(
                run_root,
                allow_nonempty_run_root=bool(allow_nonempty_run_root),
                allow_nonempty_run_root_reason=allow_nonempty_run_root_reason,
                override_applied=override_applied_for_layout
            )
            layout_initialized = True
            # 锚定依赖环境到 requirements.txt。
            path_policy.anchor_requirements(run_root)
            records_dir = layout["records_dir"]
            artifacts_dir = layout["artifacts_dir"]
            logs_dir = layout["logs_dir"]

            cfg["__run_root_dir__"] = str(run_root.resolve())
            cfg["__artifacts_dir__"] = str(artifacts_dir.resolve())
            if isinstance(input_image_path, str) and input_image_path.strip():
                cfg["__embed_input_image_path__"] = input_image_path.strip()
            else:
                embed_cfg = cfg.get("embed") if isinstance(cfg.get("embed"), dict) else {}
                default_input = embed_cfg.get("input_image_path") if isinstance(embed_cfg, dict) else None
                if isinstance(default_input, str) and default_input.strip():
                    cfg["__embed_input_image_path__"] = default_input.strip()
            
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

            attestation_env_inputs = resolve_attestation_env_inputs(cfg, require_prompt_seed=True)
            if isinstance(attestation_env_inputs, dict) and attestation_env_inputs.get("status") in {"ok", "absent"}:
                transient_secret_inputs = {
                    key_name: attestation_env_inputs.get(key_name)
                    for key_name in ["k_master", "k_prompt", "k_seed"]
                    if isinstance(attestation_env_inputs.get(key_name), str) and attestation_env_inputs.get(key_name)
                }
                if transient_secret_inputs:
                    cfg["__attestation_secret_inputs__"] = transient_secret_inputs
            
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
            record = run_embed_orchestrator(
                cfg,
                impl_set,
                cfg_digest,
                trajectory_evidence=trajectory_evidence,
                injection_evidence=injection_evidence,
                sync_runtime_context=sync_runtime_context
            )
            cfg.pop("__attestation_secret_inputs__", None)
            cfg.pop("__embed_pipeline_obj__", None)
            cfg.pop("__embed_final_latents__", None)
            cfg.pop("__embed_trajectory_latent_cache__", None)
            # 将 embed 侧 latent 空间统计写入 record，供 detect 侧几何同步 cross-comparison。
            _embed_latent_stats = cfg.pop("__embed_latent_spatial_stats__", None)
            if not isinstance(record, dict):
                # record 类型不符合预期，必须 fail-fast。
                raise TypeError("orchestrator output must be dict")
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
            if isinstance(pipeline_result, dict):
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
            if isinstance(input_image_path, str) and input_image_path.strip():
                inputs_record["input_image_path"] = input_image_path.strip()
            else:
                cfg_input_image_path = cfg.get("__embed_input_image_path__")
                if isinstance(cfg_input_image_path, str) and cfg_input_image_path.strip():
                    inputs_record["input_image_path"] = cfg_input_image_path.strip()
            if inputs_record:
                record["inputs"] = inputs_record
            
            content_evidence = record.get("content_evidence")
            if not isinstance(content_evidence, dict):
                content_evidence = {}
                record["content_evidence"] = content_evidence
            
            # 写入 pipeline_fingerprint 和 pipeline_fingerprint_digest
            if pipeline_fingerprint is not None:
                content_evidence["pipeline_fingerprint"] = pipeline_fingerprint
            if pipeline_fingerprint_digest is not None:
                content_evidence["pipeline_fingerprint_digest"] = pipeline_fingerprint_digest
            
            # POST-ORCHESTRATOR 创建最终的 injection_site_spec
            # 使用 orchestrator 计算的真实 plan_digest 来确定 injection_mode
            # 修复：orchestrator 通过 bind_plan_to_record 写的是 record["plan_digest"]（顶层），
            # 不写 content_evidence["plan_digest"]，需从顶层 record 读取真实值。
            orchestrator_plan_digest = content_evidence.get("plan_digest") or record.get("plan_digest")
            # 若顶层有真实 plan_digest，同步写入 content_evidence 供审计字段口径一致。
            if isinstance(orchestrator_plan_digest, str) and orchestrator_plan_digest:
                if not content_evidence.get("plan_digest"):
                    content_evidence["plan_digest"] = orchestrator_plan_digest
            try:
                if isinstance(orchestrator_plan_digest, str) and orchestrator_plan_digest:
                    # 真实 plan_digest 存在 → subspace_projection 模式
                    injection_site_spec, injection_site_digest = injection_site_binder.build_injection_site_spec(
                        hook_type="callback_on_step_end",
                        target_module_name="StableDiffusion3Pipeline",
                        target_tensor_name="latents",
                        hook_timing="after_scheduler_step",
                        injection_rule_summary={
                            "plan_digest": orchestrator_plan_digest,
                            "injection_mode": "subspace_projection"
                        },
                        cfg=cfg
                    )
                    print(f"[Paper-Faithful] Injection site spec built (POST-ORCHESTRATOR with real plan_digest): {injection_site_digest[:16]}...")
                else:
                    # plan_digest 缺失（orchestrator 也未能生成）→ fallback 模式
                    fallback_plan_payload = {
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
                    if not isinstance(orchestrator_plan_digest, str) or not orchestrator_plan_digest:
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
            # 同步校验plan_digest口径一致性
            if injection_site_spec is not None:
                content_evidence["injection_site_spec"] = injection_site_spec
                # 提取injection_site_spec中的plan_digest
                injection_rule_summary = injection_site_spec.get("injection_rule_summary")
                if isinstance(injection_rule_summary, dict):
                    spec_plan_digest = injection_rule_summary.get("plan_digest")
                    orchestrator_plan_digest = content_evidence.get("plan_digest")
                    # 校验口径一致性
                    if isinstance(spec_plan_digest, str) and spec_plan_digest:
                        if orchestrator_plan_digest is None or orchestrator_plan_digest == "":
                            # content_status!=ok 时，不同步 fallback digest 到语义 plan 字段。
                            content_status = content_evidence.get("status")
                            if content_status == "ok":
                                content_evidence["plan_digest"] = spec_plan_digest
                            else:
                                content_evidence["fallback_plan_digest"] = spec_plan_digest
                                content_evidence["fallback_plan_digest_reason"] = "content_status_not_ok"
                        elif orchestrator_plan_digest != spec_plan_digest:
                            # 口径不一致，写入mismatch原因但不静默覆盖
                            content_evidence["plan_digest_mismatch"] = {
                                "orchestrator_value": orchestrator_plan_digest,
                                "injection_site_value": spec_plan_digest,
                                "mismatch_reason": "pre_computation_vs_orchestrator_divergence"
                            }
            if injection_site_digest is not None:
                content_evidence["injection_site_digest"] = injection_site_digest
            
            # Paper Faithfulness: 调用 alignment_evaluator（必达）
            paper_spec = None
            paper_spec_digest = None
            try:
                # 加载 paper_faithfulness_spec.yaml（通过唯一入口）
                from pathlib import Path as PathLib
                spec_path = PathLib(__file__).parent.parent.parent / "configs" / "paper_faithfulness_spec.yaml"
                if spec_path.exists():
                    # 使用 config_loader 唯一入口加载 YAML
                    paper_spec, spec_provenance = config_loader.load_yaml_with_provenance(spec_path)
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
            alignment_report = None
            alignment_digest = None
            if isinstance(paper_spec, dict) and paper_spec.get("status") not in ("absent", "failed"):
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
            if alignment_report is not None:
                content_evidence["alignment_report"] = alignment_report
            if alignment_digest is not None:
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
        run_meta["ended_at"] = time_utils.now_utc_iso_z()
        try:
            status.finalize_run(run_root, records_dir, artifacts_dir, run_meta)
        except Exception:
            # finalize_run 失败必须 fail-fast。
            raise
        if error is not None:
            raise error

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
