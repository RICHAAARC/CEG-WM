"""
检测水印 CLI 入口

功能说明：
- 规范化输出目录路径，确保输出布局，加载合同与白名单，验证配置，解析实现，构造记录，绑定字段，写盘，并产出闭包。
- 包含详细的输入验证、错误处理与状态更新机制，确保健壮性与可维护性。
- 当前为可审计基线实现，新能力仅通过版本化追加，不改变既有冻结语义与字段口径。
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import uuid

from main.cli import assert_module_execution


assert_module_execution("run_detect")

from main.core import time_utils
from main.core.contracts import (
    load_frozen_contracts,
    bind_contract_to_record,
    get_contract_interpretation
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
from main.registries import runtime_resolver
from main.evaluation import protocol_loader
from main.watermarking.detect import orchestrator as detect_orchestrator
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.content_chain.latent_modifier import (
    LatentModifier,
    LATENT_MODIFIER_ID,
    LATENT_MODIFIER_VERSION
)
from main.watermarking.fusion import decision_writer
from main.core.errors import RunFailureReason
from main.diffusion.sd3 import pipeline_factory
from main.diffusion.sd3 import infer_runtime
from main.diffusion.sd3 import infer_trace
from main.diffusion.sd3 import trajectory_tap
from main.cli.run_common import (
    bind_impl_identity_fields,
    set_failure_status,
    format_fact_sources_mismatch,
    build_seed_audit,
    build_determinism_controls,
    normalize_nondeterminism_notes,
    build_injection_context_from_plan,
    build_cli_config_migration_hint
)


def resolve_content_override_from_input_record(input_record: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    功能：从输入记录中解析 detect 可复用的 content 结果覆盖项。

    Resolve reusable detect content override from input record.

    Args:
        input_record: Input record mapping loaded from --input.

    Returns:
        Detect-compatible content payload dict, or None if input only contains
        embed-mode content evidence.

    Raises:
        TypeError: If input_record is invalid.
    """
    if not isinstance(input_record, dict):
        # input_record 类型不合法，必须 fail-fast。
        raise TypeError("input_record must be dict")

    for content_key in ["content_evidence_payload", "content_result", "content_evidence"]:
        content_candidate = input_record.get(content_key)
        if not isinstance(content_candidate, dict):
            continue

        status_value = content_candidate.get("status")
        score_value = content_candidate.get("score")
        # content_result 与 content_evidence 均可能来自 embed 模式；
        # embed 模式下 status=ok 但 score=None，不能作为 detect 融合输入。
        # content_evidence_payload 是专属的 detect 输出格式，直接信任其 score 字段。
        if content_key in {"content_result", "content_evidence"}:
            if status_value != "ok" or not isinstance(score_value, (int, float)):
                # embed 侧提取结果无有效 score，不覆盖 detect 侧自行计算。
                continue

        return content_candidate

    return None


def _resolve_single_attack_condition_from_protocol(cfg: Dict[str, Any]) -> tuple[str, str] | None:
    """
    功能：从攻击协议中解析唯一的 attack 条件键（family::params_version）。

    Resolve the single declared attack condition from protocol spec.

    Args:
        cfg: Runtime configuration mapping.

    Returns:
        Tuple of (family, params_version) when protocol declares exactly one
        unique condition; otherwise None.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    protocol_spec = protocol_loader.load_attack_protocol_spec(cfg)
    if not isinstance(protocol_spec, dict):
        return None

    condition_keys: list[str] = []

    params_versions = protocol_spec.get("params_versions")
    if isinstance(params_versions, dict):
        for condition_key in params_versions.keys():
            if isinstance(condition_key, str) and "::" in condition_key:
                if condition_key not in condition_keys:
                    condition_keys.append(condition_key)

    families = protocol_spec.get("families")
    if isinstance(families, dict):
        for family_name, family_spec in families.items():
            if not isinstance(family_name, str) or not family_name:
                continue
            if not isinstance(family_spec, dict):
                continue
            family_versions = family_spec.get("params_versions")
            if not isinstance(family_versions, dict):
                continue
            for params_version in family_versions.keys():
                if not isinstance(params_version, str) or not params_version:
                    continue
                condition_key = f"{family_name}::{params_version}"
                if condition_key not in condition_keys:
                    condition_keys.append(condition_key)

    if len(condition_keys) != 1:
        return None

    family, params_version = condition_keys[0].split("::", 1)
    if not family or not params_version:
        return None
    return family, params_version


def _is_missing_attack_value(value: Any) -> bool:
    """Return True when attack metadata value is absent or unknown placeholder."""
    return value in (None, "", "<absent>", "unknown_attack", "unknown_params")


def _inject_attack_condition_fields(record: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    功能：在 detect record 中补充 attack family 与 params_version 字段。 

    Inject attack condition fields into detect record when protocol condition is unambiguous.

    Args:
        record: Detect record mapping to mutate in-place.
        cfg: Runtime configuration mapping.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    resolved_condition = _resolve_single_attack_condition_from_protocol(cfg)
    if resolved_condition is None:
        return

    family, params_version = resolved_condition

    if _is_missing_attack_value(record.get("attack_family")):
        record["attack_family"] = family
    if _is_missing_attack_value(record.get("attack_params_version")):
        record["attack_params_version"] = params_version

    attack_obj = record.get("attack")
    if not isinstance(attack_obj, dict):
        attack_obj = {}
    if _is_missing_attack_value(attack_obj.get("family")):
        attack_obj["family"] = family
    if _is_missing_attack_value(attack_obj.get("params_version")):
        attack_obj["params_version"] = params_version
    record["attack"] = attack_obj


def _build_hf_truncation_baseline_payload(record: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造 HF truncation baseline 对比载荷。 

    Build real HF truncation baseline payload from detect-time evidence for
    same-sample comparison in experiment matrix.

    Args:
        record: Detect record mapping.
        cfg: Runtime configuration mapping.

    Returns:
        HF truncation baseline payload mapping.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    result: Dict[str, Any] = {
        "status": "absent",
        "score": None,
        "score_source": "detect_record.content_evidence_payload.detect_hf_score",
        "baseline_impl_id": "high_freq_truncation_codec_v2",
        "baseline_version": "v1",
        "baseline_absent_reason": "hf_truncation_score_unavailable",
        "comparison_scope": "same_sample_real_pipeline",
        "trace": {
            "pipeline_impl_id": record.get("pipeline_impl_id", "<absent>"),
            "infer_trace_canon_sha256": record.get("infer_trace_canon_sha256", "<absent>"),
            "cfg_digest": record.get("cfg_digest", "<absent>"),
        },
    }

    content_payload = record.get("content_evidence_payload")
    if not isinstance(content_payload, dict):
        result["baseline_absent_reason"] = "content_evidence_payload_absent"
        return result

    paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False)) if isinstance(paper_cfg, dict) else False
    detect_runtime_mode = record.get("detect_runtime_mode")
    pipeline_runtime_meta = record.get("pipeline_runtime_meta") if isinstance(record.get("pipeline_runtime_meta"), dict) else {}

    if paper_enabled and bool(pipeline_runtime_meta.get("synthetic_pipeline", False)):
        result["baseline_absent_reason"] = "synthetic_pipeline_runtime"
        return result
    if paper_enabled and detect_runtime_mode != "real":
        result["baseline_absent_reason"] = f"detect_runtime_mode_not_real:{detect_runtime_mode}"
        return result

    watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
    hf_cfg = watermark_cfg.get("hf") if isinstance(watermark_cfg.get("hf"), dict) else {}
    if not bool(hf_cfg.get("enabled", False)):
        result["baseline_absent_reason"] = "hf_channel_disabled"
        return result

    score_candidate = content_payload.get("detect_hf_score")
    if not isinstance(score_candidate, (int, float)):
        score_candidate = content_payload.get("hf_score")
    if not isinstance(score_candidate, (int, float)):
        result["baseline_absent_reason"] = "hf_score_absent"
        return result

    score_value = float(score_candidate)
    if score_value != score_value or score_value in (float("inf"), float("-inf")):
        result["baseline_absent_reason"] = "hf_score_non_finite"
        return result

    result["status"] = "ok"
    result["score"] = score_value
    result["baseline_absent_reason"] = None
    return result


def run_detect(
    output_dir: str,
    config_path: str,
    input_record_path: str | None = None,
    overrides: list[str] | None = None,
    thresholds_path: str | None = None,
) -> None:
    """
    功能：执行检测流程，本阶段为基线实现。

    Execute detect workflow (baseline implementation).

    Args:
        output_dir: Run root directory for records/artifacts.
        config_path: YAML config path.
        input_record_path: Optional input record path (baseline if not provided).
        overrides: Optional CLI override args list.
        thresholds_path: Optional path to thresholds artifact for NP threshold injection.
    
    Returns:
        None.
    """
    if not output_dir:
        # output_dir 输入不合法，必须 fail-fast。
        raise ValueError("output_dir must be non-empty str")
    if not config_path:
        # config_path 输入不合法，必须 fail-fast。
        raise ValueError("config_path must be non-empty str")

    run_root = path_policy.derive_run_root(Path(output_dir))
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    layout_initialized = False
    allow_nonempty_run_root = False
    allow_nonempty_run_root_reason = None
    override_applied_for_layout = None

    started_at = time_utils.now_utc_iso_z()
    run_meta: Dict[str, Any] = {
        "run_id": f"run-{uuid.uuid4().hex}",
        "command": "detect",
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
        "pipeline_build_status": "<absent>",
        "pipeline_build_failure_reason": "<absent>",
        "pipeline_build_failure_summary": "<absent>",
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
        "path_policy": None,
        "run_root_reuse_allowed": False,
        "run_root_reuse_reason": None,
        # 统计口径锚点初始化。
        "target_fpr": "<absent>",
        "thresholds_digest": "<absent>",
        "threshold_metadata_digest": "<absent>"
    }

    error = None
    pipeline_result = None
    record = None
    try:
        # 加载事实源。
        print("[Detect] Loading fact sources...")
        contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
        whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
        semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
        injection_scope_manifest = config_loader.load_injection_scope_manifest()

        # 绑定冻结锚点到 run_meta。
        print("[Detect] Binding freeze anchors to run_meta...")
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
        print("[Detect] Validating whitelist-semantics consistency...")
        assert_consistent_with_semantics(whitelist, semantics)

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

        seed_parts, seed_digest, seed_value, seed_rule_id = build_seed_audit(cfg, "detect")
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
        run_meta["pipeline_build_status"] = pipeline_result.get("pipeline_build_status", "<absent>")
        run_meta["pipeline_build_failure_reason"] = pipeline_result.get("pipeline_build_failure_reason", "<absent>")
        run_meta["pipeline_build_failure_summary"] = pipeline_result.get("pipeline_build_failure_summary", "<absent>")
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

        # （1）thresholds_path 注入：若提供，加载 NP 阈值工件并注入 cfg["__thresholds_artifact__"]。
        # 加载失败时 fail-fast，不允许 silent fallback 至 test-only 阈值。
        if isinstance(thresholds_path, str) and thresholds_path:
            _tp_resolved = str(Path(thresholds_path).resolve())
            try:
                _thresholds_obj = detect_orchestrator.load_thresholds_artifact_controlled(_tp_resolved)
                cfg["__thresholds_artifact__"] = _thresholds_obj
                run_meta["thresholds_path_injected"] = _tp_resolved
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.CONFIG_INVALID, exc)
                raise

        # P1-A：calibration_artifact_path 自动发现：若 thresholds_path 参数未注入阈值工件，
        # 则尝试从 cfg.detect.calibration_artifact_path 加载（使同一受控加载函数）。
        # 加载失败时 fail-fast，不允许降级为 fallback；thresholds_path 参数优先于配置字段。
        if cfg.get("__thresholds_artifact__") is None:
            _detect_cfg_p1a = cfg.get("detect") or {}
            _cal_artifact_path = _detect_cfg_p1a.get("calibration_artifact_path")
            if isinstance(_cal_artifact_path, str) and _cal_artifact_path:
                _cal_resolved = str(Path(_cal_artifact_path).resolve())
                try:
                    _thresholds_obj = detect_orchestrator.load_thresholds_artifact_controlled(_cal_resolved)
                    cfg["__thresholds_artifact__"] = _thresholds_obj
                    run_meta["thresholds_path_injected"] = _cal_resolved
                    run_meta["thresholds_source"] = "detect_calibration_artifact_path"
                except Exception as exc:
                    # 加载失败时 fail-fast，不允许 silent fallback。
                    set_failure_status(run_meta, RunFailureReason.CONFIG_INVALID, exc)
                    raise

        # 预先计算 content 与 subspace 计划，用于注入上下文。
        # 这里必须使用 embed-mode 提取 mask_digest，避免 detect-mode 在无 detector_inputs 时返回 absent。
        cfg_for_preplan = dict(cfg)
        detect_cfg_for_preplan = cfg_for_preplan.get("detect")
        if isinstance(detect_cfg_for_preplan, dict):
            detect_cfg_for_preplan = dict(detect_cfg_for_preplan)
        else:
            detect_cfg_for_preplan = {}
        detect_content_cfg_for_preplan = detect_cfg_for_preplan.get("content")
        if isinstance(detect_content_cfg_for_preplan, dict):
            detect_content_cfg_for_preplan = dict(detect_content_cfg_for_preplan)
        else:
            detect_content_cfg_for_preplan = {}
        detect_content_cfg_for_preplan["enabled"] = False
        detect_cfg_for_preplan["content"] = detect_content_cfg_for_preplan
        cfg_for_preplan["detect"] = detect_cfg_for_preplan

        # P1b：在 bound_fact_sources 上下文之外，用标准库对 input_record_path 做最小解析，
        # 提取 watermarked_path 并作为 pre-plan 的图像输入，使 detect-side mask_digest
        # 与 embed-side 对齐（同一图像 → 同一 subspace injection_context）。
        # 优先级：(1) input_record_path 中的 watermarked_path；(2) cfg 中的 probe_image_path；(3) 无 inputs。
        _preplan_inputs: dict | None = None
        if input_record_path:
            try:
                import json as _json
                with open(input_record_path, "r", encoding="utf-8") as _f:
                    _pre_record = _json.load(_f)
                _wm_path = _pre_record.get("watermarked_path")
                if isinstance(_wm_path, str) and _wm_path and _wm_path != "<absent>":
                    _wm_path_resolved = str(Path(_wm_path).resolve())
                    if Path(_wm_path_resolved).is_file():
                        _preplan_inputs = {"image_path": _wm_path_resolved}
                    else:
                        print(f"[Detect][preplan] watermarked_path 不可达，跳过图像输入: {_wm_path_resolved}")
            except Exception as _pre_exc:
                # 最小解析失败时维持现有 no-input 行为，不阻断主流程。
                print(f"[Detect][preplan] input_record 最小解析失败，使用无输入模式: {_pre_exc}")
        if _preplan_inputs is None:
            _probe_img = cfg.get("detect", {}).get("content", {}).get("probe_image_path")
            if isinstance(_probe_img, str) and _probe_img:
                _preplan_inputs = {"image_path": _probe_img}

        content_result_pre = impl_set.content_extractor.extract(cfg_for_preplan, inputs=_preplan_inputs)
        mask_digest = None
        if isinstance(content_result_pre, dict):
            mask_digest = content_result_pre.get("mask_digest")
        elif hasattr(content_result_pre, "mask_digest"):
            mask_digest = content_result_pre.mask_digest

        planner_inputs = detect_orchestrator._build_planner_inputs_for_runtime(cfg, None)
        subspace_result_pre = impl_set.subspace_planner.plan(
            cfg,
            mask_digest=mask_digest,
            cfg_digest=cfg_digest,
            inputs=planner_inputs
        )

        plan_payload = subspace_result_pre.as_dict() if hasattr(subspace_result_pre, "as_dict") else subspace_result_pre
        plan_digest = getattr(subspace_result_pre, "plan_digest", None)
        if isinstance(plan_payload, dict) and not isinstance(plan_digest, str):
            plan_digest = plan_payload.get("plan_digest")

        injection_context = None
        injection_modifier = None
        if isinstance(plan_payload, dict) and isinstance(plan_digest, str) and plan_digest:
            injection_context = build_injection_context_from_plan(cfg, plan_payload, plan_digest)
            injection_modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)

        # (7.7) Real Dataflow Smoke：在 pipeline_result 之后调用 inference，并捕获最后的 latents 用于 detect 侧评分
        pipeline_obj = pipeline_result.get("pipeline_obj")
        device = cfg.get("device", "cpu")
        seed = seed_value
        runtime_self_attention_maps = None

        dependency_guard = assert_detect_runtime_dependencies(
            cfg,
            {},
            pipeline_obj
        )

        if pipeline_obj is None:
            inference_status = infer_runtime.INFERENCE_STATUS_FAILED
            inference_error = "detect_missing_pipeline_dependency"
            inference_runtime_meta = {
                "dependency_guard": {
                    "allow_missing_pipeline_for_detect": bool(dependency_guard.get("allow_flag", False)),
                    "allow_flag": bool(dependency_guard.get("allow_flag", False))
                }
            }
            trajectory_evidence = trajectory_tap.build_trajectory_evidence(
                cfg,
                inference_status,
                inference_runtime_meta,
                seed=seed,
                device=device,
            )
            injection_evidence = {
                "status": "absent",
                "injection_absent_reason": "inference_failed",
                "injection_failure_reason": None,
                "injection_trace_digest": None,
                "injection_params_digest": None,
                "injection_metrics": None,
                "subspace_binding_digest": None,
            }
            _detect_traj_cache = trajectory_tap.LatentTrajectoryCache()
        else:
            detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
            geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
            capture_attention = bool(
                geometry_cfg.get("enabled", False)
                and geometry_cfg.get("enable_attention_anchor", False)
            )
            # 为 detect 侧创建 per-step latent 缓存（与 embed 侧对称，不写入 records）。
            _detect_traj_cache = trajectory_tap.LatentTrajectoryCache()
            inference_result = infer_runtime.run_sd3_inference(
                cfg,
                pipeline_obj,
                device,
                seed,
                injection_context=injection_context,
                injection_modifier=injection_modifier,
                capture_final_latents=False,
                capture_attention=capture_attention,
                trajectory_latent_cache=_detect_traj_cache,
            )
            inference_status = inference_result.get("inference_status")
            inference_error = inference_result.get("inference_error")
            inference_runtime_meta = inference_result.get("inference_runtime_meta")
            trajectory_evidence = inference_result.get("trajectory_evidence")
            injection_evidence = inference_result.get("injection_evidence")
            runtime_self_attention_maps = inference_result.get("runtime_self_attention_maps")
            runtime_self_attention_source = None
            if isinstance(inference_runtime_meta, dict):
                runtime_self_attention_source = inference_runtime_meta.get("runtime_self_attention_source")
        
        if runtime_self_attention_maps is not None:
            cfg["__runtime_self_attention_maps__"] = runtime_self_attention_maps
        if isinstance(runtime_self_attention_source, str) and runtime_self_attention_source:
            cfg["__runtime_self_attention_source__"] = runtime_self_attention_source
        cfg["__detect_pipeline_obj__"] = pipeline_obj
        cfg["__pipeline_runtime_meta__"] = pipeline_result.get("pipeline_runtime_meta")
        # 将 detect 侧 per-step latent 缓存注入 cfg（内存传递，不写入 records）。
        if not _detect_traj_cache.is_empty():
            cfg["__detect_trajectory_latent_cache__"] = _detect_traj_cache
        
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

        allow_nonempty_run_root = cfg.get("allow_nonempty_run_root", False)
        allow_nonempty_run_root_reason = cfg.get("allow_nonempty_run_root_reason")
        override_applied_for_layout = cfg.get("override_applied")

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

        run_meta["path_policy"] = {
            "allow_nonempty_run_root": bool(allow_nonempty_run_root),
            "allow_nonempty_run_root_reason": allow_nonempty_run_root_reason
        }
        run_meta["run_root_reuse_allowed"] = bool(allow_nonempty_run_root)
        run_meta["run_root_reuse_reason"] = allow_nonempty_run_root_reason

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

            # 读取输入 record。
            if input_record_path:
                print(f"[Detect] Loading input record from {input_record_path}...")
                input_record = records_io.read_json(input_record_path)
                print(f"[Detect]   Loaded input record with {len(input_record)} fields")
                # 从 embed_record 读取 latent 空间统计，注入 cfg 供几何同步 cross-comparison。
                _input_latent_stats = input_record.get("latent_spatial_stats")
                if isinstance(_input_latent_stats, dict):
                    cfg["__embed_latent_spatial_stats__"] = _input_latent_stats
            else:
                print("[Detect] No input record provided, using baseline")
                input_record = {"baseline_input": True}

            # 构造 detect record，本阶段为基线实现。
            print("[Detect] Generating detect record (baseline)...")
            if input_record_path:
                content_override_for_orchestrator = resolve_content_override_from_input_record(input_record)

                plan_override_for_orchestrator = None
                input_subspace_plan = input_record.get("subspace_plan")
                input_plan_digest = input_record.get("plan_digest")
                input_basis_digest = input_record.get("basis_digest")
                input_plan_stats = input_record.get("plan_stats")
                if isinstance(input_subspace_plan, dict):
                    plan_override_for_orchestrator = {
                        "status": "ok",
                        "plan": input_subspace_plan,
                        "plan_digest": input_plan_digest if isinstance(input_plan_digest, str) and input_plan_digest else None,
                        "basis_digest": input_basis_digest if isinstance(input_basis_digest, str) and input_basis_digest else None,
                        "plan_stats": input_plan_stats if isinstance(input_plan_stats, dict) else None,
                        "plan_failure_reason": None,
                    }
            else:
                content_override_for_orchestrator = None
                plan_override_for_orchestrator = None
            record = run_detect_orchestrator(
                cfg,
                impl_set,
                input_record,
                cfg_digest=cfg_digest,
                trajectory_evidence=trajectory_evidence,
                injection_evidence=injection_evidence,
                content_result_override=content_override_for_orchestrator,
                detect_plan_result_override=plan_override_for_orchestrator
            )
            if record is None:
                exc = RuntimeError("record_construction_failed: record is None")
                set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
                raise exc

            _inject_attack_condition_fields(record, cfg)
            
            # ⭐ 增强项：从 input_record 继承 Embed 侧的摘要字段，用于完全对齐验证
            # 这使得 detect_record.content_evidence_payload 包含 Embed 的摘要，
            # 支持从 Notebook 生成的摘要对照表（checked_digests_alignment_report）
            if isinstance(input_record, dict) and "content_evidence" in input_record:
                embed_content_ev = input_record.get("content_evidence", {})
                detect_payload = record.get("content_evidence_payload", {})
                
                if isinstance(detect_payload, dict) and isinstance(embed_content_ev, dict):
                    # 从 Embed 继承关键摘要字段到 Detect（用于对齐验证）
                    digest_fields = [
                        "pipeline_fingerprint_digest",
                        "trajectory_digest",
                        "alignment_digest",
                        "injection_site_digest"
                    ]
                    for field in digest_fields:
                        if field not in detect_payload and field in embed_content_ev:
                            detect_payload[field] = embed_content_ev[field]
                    
                    record["content_evidence_payload"] = detect_payload
            
            record["cfg_digest"] = cfg_digest
            record["policy_path"] = cfg["policy_path"]
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
            override_applied = cfg.get("override_applied")
            if override_applied is not None:
                record["override_applied"] = override_applied
            # 写入 impl_set_capabilities_digest。
            record["impl_set_capabilities_digest"] = impl_set_capabilities_digest
            if isinstance(impl_set_capabilities_extended_digest, str) and impl_set_capabilities_extended_digest:
                record["impl_set_capabilities_extended_digest"] = impl_set_capabilities_extended_digest

            # 将 inference 相关字段写入 record（append-only）
            record["infer_trace"] = run_meta.get("infer_trace")
            record["infer_trace_canon_sha256"] = run_meta.get("infer_trace_canon_sha256")
            record["inference_status"] = run_meta.get("inference_status")
            record["inference_error"] = run_meta.get("inference_error")
            record["inference_runtime_meta"] = run_meta.get("inference_runtime_meta")
            record["hf_truncation_baseline"] = _build_hf_truncation_baseline_payload(record, cfg)

            schema.ensure_required_fields(record, cfg, interpretation)

            # 口径版本标识写入 run_meta。
            run_meta["thresholds_rule_id"] = record.get("thresholds_rule_id", "<absent>")
            run_meta["thresholds_rule_version"] = record.get("thresholds_rule_version", "<absent>")

            # 绑定 impl_identity 字段族。
            bind_impl_identity_fields(record, impl_identity, impl_set, contracts)

            # 绑定事实源字段。
            print("[Detect] Binding fact source fields...")
            bind_contract_to_record(record, contracts)
            bind_whitelist_to_record(record, whitelist)
            bind_semantics_to_record(record, semantics)

            # CLI 单点收口：融合判决唯一 decision 写入与序列化。
            # 位置：ensure_required_fields 之后、validate_record 之前。
            decision_writer.assert_decision_write_bypass_blocked(record, interpretation)
            fusion_result = record.get("fusion_result")
            if fusion_result is None:
                # fusion_result 缺失是致命错误，必须 fail-fast。
                raise ValueError("fusion_result is required in record but was None")
            # 验证 fusion_result 是 FusionDecision 对象。
            from main.watermarking.fusion.interfaces import FusionDecision
            if not isinstance(fusion_result, FusionDecision):
                raise ValueError(
                    f"fusion_result must be FusionDecision instance, got {type(fusion_result)}"
                )
            # 调用 decision_writer 写入 decision 字段。
            decision_writer.apply_fusion_decision_to_record(
                record,
                fusion_result,
                interpretation
            )
            # 序列化 fusion_result 为可 JSON 化的 dict，覆盖原值。
            record["fusion_result"] = fusion_result.to_dict()

            try:
                schema.validate_record(record, interpretation=interpretation)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
                raise
            
            # 写盘，触发 freeze_gate.assert_prewrite。
            record_path = records_dir / "detect_record.json"
            path_policy.validate_output_target(record_path, "record", run_root)
            print(f"[Detect] Writing record to {record_path}...")
            try:
                records_io.write_json(str(record_path), record)
            except Exception as exc:
                set_failure_status(run_meta, RunFailureReason.GATE_FAILED, exc)
                raise
    except Exception as exc:
        if run_meta.get("status_ok", True):
            set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
        error = exc
    finally:
        # 绑定闭包最小合同字段，并产出 run_closure.json。
        if not layout_initialized:
            try:
                layout = path_policy.ensure_output_layout(
                    run_root,
                    allow_nonempty_run_root=bool(allow_nonempty_run_root),
                    allow_nonempty_run_root_reason=allow_nonempty_run_root_reason,
                    override_applied=override_applied_for_layout
                )
                records_dir = layout["records_dir"]
                artifacts_dir = layout["artifacts_dir"]
                logs_dir = layout["logs_dir"]
            except Exception as exc:
                if run_meta.get("status_ok", True):
                    set_failure_status(run_meta, RunFailureReason.RUNTIME_ERROR, exc)
                run_meta["status_details"] = f"layout_init_failed: {exc}"
        run_meta["ended_at"] = time_utils.now_utc_iso_z()
        try:
            status.finalize_run(run_root, records_dir, artifacts_dir, run_meta)
        except Exception:
            # finalize_run 失败必须 fail-fast。
            raise
        if error is not None:
            raise error

    if record is not None:
        print("[Detect] [OK] Detect record written successfully")
        print(f"[Detect] Record contains {len(record)} fields (15 fact source fields + {len(record) - 15} business fields)")


def main():
    """主流程。"""
    parser = argparse.ArgumentParser(
        description="Detect watermark (baseline implementation)"
    )
    parser.add_argument(
        "--out",
        default="tmp/cli_smoke/detect_run",
        help="Output run root directory"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config YAML path"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional input record path"
    )
    parser.add_argument(
        "--override",
        "--overrides",
        action="append",
        default=None,
        help="Override config key=value (JSON value). Can be repeated."
    )
    parser.add_argument(
        "--thresholds-path",
        default=None,
        help="Path to thresholds artifact JSON for NP threshold injection."
    )
    
    args = parser.parse_args()
    
    try:
        run_detect(args.out, args.config, args.input, args.override, thresholds_path=args.thresholds_path)
        sys.exit(0)
    except Exception as e:
        print(f"[Detect] [ERROR] Error: {e}", file=sys.stderr)
        hint = build_cli_config_migration_hint(e)
        if isinstance(hint, str) and hint:
            print(f"[Detect] [HINT] {hint}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _resolve_allow_missing_pipeline_for_detect(cfg: Dict[str, Any]) -> bool:
    """
    功能：解析 detect 缺失 pipeline 显式放行开关。

    Resolve explicit allow flag for missing detect pipeline dependency.

    Args:
        cfg: Configuration mapping.

    Returns:
        True when explicit allow flag is enabled.
    """
    if not isinstance(cfg, dict):
        return False
    runtime_cfg = cfg.get("runtime")
    if isinstance(runtime_cfg, dict):
        allow_flag = runtime_cfg.get("allow_missing_pipeline_for_detect")
        if isinstance(allow_flag, bool):
            return allow_flag
    detect_cfg = cfg.get("detect")
    if isinstance(detect_cfg, dict):
        runtime_detect_cfg = detect_cfg.get("runtime")
        if isinstance(runtime_detect_cfg, dict):
            allow_flag = runtime_detect_cfg.get("allow_missing_pipeline_for_detect")
            if isinstance(allow_flag, bool):
                return allow_flag
    return False


def assert_detect_runtime_dependencies(
    cfg: Dict[str, Any],
    inputs: Dict[str, Any],
    pipeline_obj: Any
) -> Dict[str, Any]:
    """
    功能：detect 运行依赖前置校验。

    Enforce strict detect runtime dependency policy.
    Missing pipeline is allowed only via explicit allow flag.

    Args:
        cfg: Configuration mapping.
        inputs: Runtime inputs mapping (unused, kept for call-site compatibility).
        pipeline_obj: Built pipeline object.

    Returns:
        Guard decision mapping.

    Raises:
        RuntimeError: If pipeline dependency is missing without explicit allow flag.
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(inputs, dict):
        raise TypeError("inputs must be dict")

    allow_flag = _resolve_allow_missing_pipeline_for_detect(cfg)

    if pipeline_obj is None and not allow_flag:
        # 生产默认 fail-fast，缺失 pipeline 不再因 test_mode 放行。
        raise RuntimeError("detect_missing_pipeline_dependency")

    return {
        "allow_flag": bool(allow_flag),
        "allow_missing_pipeline_for_detect": bool(allow_flag),
    }


if __name__ == "__main__":
    main()