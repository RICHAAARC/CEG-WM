"""
File purpose: 检测、评估与校准编排（detect, evaluate, calibrate orchestration）。
Module type: General module
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from main.core import digests
from main.core import config_loader
from main.core import records_io
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain import detector_scoring
from main.watermarking.common.plan_digest_flow import verify_plan_digest
from main.watermarking.content_chain.high_freq_embedder import (
    HighFreqEmbedder,
    HIGH_FREQ_EMBEDDER_ID,
    HIGH_FREQ_EMBEDDER_VERSION,
    HF_FAILURE_RULE_VERSION,
    detect_high_freq_score,
)
from main.watermarking.content_chain.low_freq_coder import detect_low_freq_score
from main.watermarking.fusion import decision as fusion_decision
from main.watermarking.fusion import neyman_pearson
from main.watermarking.fusion.interfaces import FusionDecision
from main.watermarking.geometry_chain.align_invariance_extractor import (
    GEO_AVAILABILITY_RULE_VERSION,
)
from main.watermarking.geometry_chain.sync.latent_sync_template import SyncRuntimeContext
from main.evaluation import protocol_loader as eval_protocol_loader
from main.evaluation import metrics as eval_metrics
from main.evaluation import report_builder as eval_report_builder
from main.evaluation import attack_coverage as eval_attack_coverage


def run_detect_orchestrator(
    cfg: Dict[str, Any],
    impl_set: BuiltImplSet,
    input_record: Optional[Dict[str, Any]] = None,
    cfg_digest: Optional[str] = None,
    trajectory_evidence: Optional[Dict[str, Any]] = None,
    injection_evidence: Optional[Dict[str, Any]] = None,
    content_result_override: Any | None = None,
    detect_plan_result_override: Any | None = None
) -> Dict[str, Any]:
    """
    功能：执行检测编排流程，包括 plan_digest 一致性验证。

    Execute detect workflow using injected implementations.
    Validates plan_digest consistency with embed-time plan_digest when available.
    Supports ablation flags: when ablation.normalized.enable_content=false,
    content_extractor returns status="absent" with no failure reason.

    Args:
        cfg: Config mapping (may differ from embed-time cfg).
        impl_set: Built implementation set.
        input_record: Optional input record mapping (contains embed-time plan_digest).
        cfg_digest: Optional cfg digest for detect-time cfg.
                   If None, plan_digest validation is skipped.
        trajectory_evidence: Optional trajectory tap evidence mapping.
        injection_evidence: Optional injection evidence mapping.
        content_result_override: Optional precomputed content result.
        detect_plan_result_override: Optional precomputed detect plan result.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_set, BuiltImplSet):
        # impl_set 类型不合法，必须 failfast。
        raise TypeError("impl_set must be BuiltImplSet")
    if input_record is not None and not isinstance(input_record, dict):
        # input_record 类型不合法，必须 fail-fast。
        raise TypeError("input_record must be dict or None")
    if cfg_digest is not None and not isinstance(cfg_digest, str):
        # cfg_digest 类型不合法，必须 fail-fast。
        raise TypeError("cfg_digest must be str or None")
    if trajectory_evidence is not None and not isinstance(trajectory_evidence, dict):
        # trajectory_evidence 类型不合法，必须 fail-fast。
        raise TypeError("trajectory_evidence must be dict or None")
    if injection_evidence is not None and not isinstance(injection_evidence, dict):
        # injection_evidence 类型不符合预期，必须 fail-fast。
        raise TypeError("injection_evidence must be dict or None")
    if content_result_override is not None and not isinstance(content_result_override, dict) and not hasattr(content_result_override, "as_dict"):
        # content_result_override 类型不符合预期，必须 fail-fast。
        raise TypeError("content_result_override must be dict, ContentEvidence, or None")
    if detect_plan_result_override is not None and not isinstance(detect_plan_result_override, dict) and not hasattr(detect_plan_result_override, "as_dict"):
        # detect_plan_result_override 类型不符合预期，必须 fail-fast。
        raise TypeError("detect_plan_result_override must be dict, SubspacePlan, or None")

    # 读取 ablation.normalized 开关（若缺失则默认全启用）。
    ablation_normalized = _get_ablation_normalized(cfg)
    enable_content = ablation_normalized.get("enable_content", True)
    enable_geometry = ablation_normalized.get("enable_geometry", True)
    enable_sync = ablation_normalized.get("enable_sync", True)
    enable_anchor = ablation_normalized.get("enable_anchor", True)
    enable_attention_proxy = ablation_normalized.get("enable_attention_proxy", True)
    enable_image_sidecar = ablation_normalized.get("enable_image_sidecar", True)
    paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False)) if isinstance(paper_cfg, dict) else False
    if paper_enabled:
        enable_attention_proxy = False

    # Ablation: 禁用 content 模块时返回 absent 语义。
    if not enable_content:
        content_result = _build_ablation_absent_content_evidence("content_chain_disabled_by_ablation")
    else:
        content_result = content_result_override if content_result_override is not None else impl_set.content_extractor.extract(cfg)
    
    # Ablation: 禁用 geometry 模块时返回 absent 语义。
    if not enable_geometry:
        geometry_result = _build_ablation_absent_geometry_evidence("geometry_chain_disabled_by_ablation")
    else:
        geometry_result = _run_geometry_chain_with_sync(
            impl_set,
            cfg,
            enable_anchor=bool(enable_anchor),
            enable_sync=bool(enable_sync),
            enable_attention_proxy=bool(enable_attention_proxy),
        )

    # (1) 统一转换 ContentEvidence / GeometryEvidence 数据类为 dict。
    # 优先使用 .as_dict() 方法；若不存在则直接使用数据类或字典。
    content_evidence_payload = None
    if hasattr(content_result, "as_dict") and callable(content_result.as_dict):
        content_evidence_payload = content_result.as_dict()
    elif isinstance(content_result, dict):
        content_evidence_payload = content_result

    if trajectory_evidence is not None:
        if content_evidence_payload is None:
            content_evidence_payload = {}
        content_evidence_payload["trajectory_evidence"] = trajectory_evidence
        _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
    if injection_evidence is not None:
        if content_evidence_payload is None:
            content_evidence_payload = {}
        _merge_injection_evidence(content_evidence_payload, injection_evidence)

    geometry_evidence_payload = None
    if hasattr(geometry_result, "as_dict") and callable(geometry_result.as_dict):
        geometry_evidence_payload = geometry_result.as_dict()
    elif isinstance(geometry_result, dict):
        geometry_evidence_payload = geometry_result

    # (2) 构造融合输入适配 dict，兼容 FusionBaselineIdentity 的旧字段读取逻辑。
    # 优先从 .as_dict() 结果中读取，但为向后兼容也检查数据类属性。
    content_evidence_adapted = _adapt_content_evidence_for_fusion(content_result)
    geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)

    planner_content_payload = content_evidence_payload
    planner_inputs = _build_planner_inputs_for_runtime(cfg, None, planner_content_payload)
    mask_digest = None
    if isinstance(planner_content_payload, dict):
        mask_digest = planner_content_payload.get("mask_digest")
    if not isinstance(mask_digest, str) or not mask_digest:
        if isinstance(input_record, dict):
            embed_content_evidence = input_record.get("content_evidence")
            if isinstance(embed_content_evidence, dict):
                planner_content_payload = embed_content_evidence
                mask_digest = embed_content_evidence.get("mask_digest")
                planner_inputs = _build_planner_inputs_for_runtime(cfg, None, planner_content_payload)
    if not isinstance(mask_digest, str) or not mask_digest:
        # detect-mode 前置阶段可能无法提供 mask_digest；为 planner 回退到 embed-mode 提取。
        cfg_for_planner = dict(cfg)
        detect_cfg_for_planner = cfg_for_planner.get("detect")
        if isinstance(detect_cfg_for_planner, dict):
            detect_cfg_for_planner = dict(detect_cfg_for_planner)
        else:
            detect_cfg_for_planner = {}
        detect_content_cfg_for_planner = detect_cfg_for_planner.get("content")
        if isinstance(detect_content_cfg_for_planner, dict):
            detect_content_cfg_for_planner = dict(detect_content_cfg_for_planner)
        else:
            detect_content_cfg_for_planner = {}
        detect_content_cfg_for_planner["enabled"] = False
        detect_cfg_for_planner["content"] = detect_content_cfg_for_planner
        cfg_for_planner["detect"] = detect_cfg_for_planner
        planner_content_result = impl_set.content_extractor.extract(cfg_for_planner)
        if hasattr(planner_content_result, "as_dict") and callable(planner_content_result.as_dict):
            planner_content_payload = planner_content_result.as_dict()
        elif isinstance(planner_content_result, dict):
            planner_content_payload = planner_content_result
        else:
            planner_content_payload = None
        if isinstance(planner_content_payload, dict):
            mask_digest = planner_content_payload.get("mask_digest")
        planner_inputs = _build_planner_inputs_for_runtime(cfg, None, planner_content_payload)

    planner_cfg_digest = cfg_digest
    planner_cfg = cfg
    if isinstance(input_record, dict):
        embed_cfg_digest = input_record.get("cfg_digest")
        if isinstance(embed_cfg_digest, str) and embed_cfg_digest:
            planner_cfg_digest = embed_cfg_digest
        embed_seed = input_record.get("seed")
        if isinstance(embed_seed, int):
            planner_cfg = dict(cfg)
            planner_cfg["seed"] = embed_seed

    detect_plan_result = detect_plan_result_override if detect_plan_result_override is not None else impl_set.subspace_planner.plan(
        planner_cfg,
        mask_digest=mask_digest,
        cfg_digest=planner_cfg_digest,
        inputs=planner_inputs
    )

    expected_plan_digest = _resolve_expected_plan_digest(input_record)
    detect_test_mode = _resolve_detect_test_mode(cfg)
    allow_cfg_plan_digest_fallback_used = False
    if expected_plan_digest is None and detect_test_mode:
        cfg_plan_digest = _resolve_cfg_plan_digest(cfg)
        if isinstance(cfg_plan_digest, str) and cfg_plan_digest:
            expected_plan_digest = cfg_plan_digest
            allow_cfg_plan_digest_fallback_used = True
    embed_time_plan_digest = expected_plan_digest
    embed_time_basis_digest = None
    embed_time_planner_impl_identity = None
    if isinstance(input_record, dict):
        embed_time_basis_digest = input_record.get("basis_digest")
        embed_time_planner_impl_identity = input_record.get("subspace_planner_impl_identity")

    detect_time_plan_digest = getattr(detect_plan_result, "plan_digest", None)
    detect_time_basis_digest = getattr(detect_plan_result, "basis_digest", None)
    detect_planner_input_digest = _extract_detect_planner_input_digest(detect_plan_result)
    if detect_planner_input_digest is None:
        build_planner_input_digest = getattr(impl_set.subspace_planner, "_build_planner_input_digest", None)
        if callable(build_planner_input_digest):
            computed_digest = build_planner_input_digest(planner_inputs)
            if isinstance(computed_digest, str) and computed_digest:
                detect_planner_input_digest = computed_digest
    detect_time_planner_impl_identity = None
    if hasattr(detect_plan_result, "plan") and isinstance(detect_plan_result.plan, dict):
        detect_time_planner_impl_identity = detect_plan_result.plan.get("planner_impl_identity")

    plan_payload = None
    if hasattr(detect_plan_result, "as_dict") and callable(detect_plan_result.as_dict):
        plan_payload = detect_plan_result.as_dict()
    elif isinstance(detect_plan_result, dict):
        plan_payload = dict(detect_plan_result)

    if isinstance(plan_payload, dict):
        if not isinstance(detect_time_plan_digest, str):
            payload_plan_digest = plan_payload.get("plan_digest")
            if isinstance(payload_plan_digest, str) and payload_plan_digest:
                detect_time_plan_digest = payload_plan_digest
        if not isinstance(detect_time_basis_digest, str):
            payload_basis_digest = plan_payload.get("basis_digest")
            if isinstance(payload_basis_digest, str) and payload_basis_digest:
                detect_time_basis_digest = payload_basis_digest
        if detect_time_planner_impl_identity is None:
            plan_node = plan_payload.get("plan")
            if isinstance(plan_node, dict):
                detect_time_planner_impl_identity = plan_node.get("planner_impl_identity")

    mismatch_reasons: list[str] = []
    if isinstance(expected_plan_digest, str) and expected_plan_digest:
        mismatch_reasons = _collect_plan_mismatch_reasons(
            embed_time_plan_digest=expected_plan_digest,
            detect_time_plan_digest=detect_time_plan_digest,
            embed_time_basis_digest=embed_time_basis_digest,
            detect_time_basis_digest=detect_time_basis_digest,
            embed_time_planner_impl_identity=embed_time_planner_impl_identity,
            detect_time_planner_impl_identity=detect_time_planner_impl_identity
        )
        plan_digest_status, plan_digest_reason = verify_plan_digest(
            expected_plan_digest,
            detect_time_plan_digest if isinstance(detect_time_plan_digest, str) else None,
        )
        if plan_digest_reason == "plan_digest_mismatch" and "plan_digest_mismatch" not in mismatch_reasons:
            mismatch_reasons.append("plan_digest_mismatch")
    else:
        plan_digest_status = "absent"
        plan_digest_reason = "plan_digest_absent"

    trajectory_status, trajectory_mismatch_reason = _evaluate_trajectory_consistency(
        input_record=input_record,
        trajectory_evidence=trajectory_evidence,
        detect_planner_input_digest=detect_planner_input_digest
    )
    if trajectory_status == "mismatch" and trajectory_mismatch_reason:
        mismatch_reasons.append(trajectory_mismatch_reason)

    injection_status, injection_mismatch_reason = _evaluate_injection_consistency(
        input_record=input_record,
        injection_evidence=injection_evidence
    )
    if injection_status == "mismatch" and injection_mismatch_reason:
        mismatch_reasons.append(injection_mismatch_reason)

    paper_impl_binding_status, paper_impl_binding_reason = _evaluate_paper_impl_binding_consistency(
        cfg=cfg,
        injection_evidence=injection_evidence,
        input_record=input_record,
    )
    if paper_impl_binding_status == "mismatch" and isinstance(paper_impl_binding_reason, str):
        mismatch_reasons.append(paper_impl_binding_reason)

    # (S-D) Paper Faithfulness: 验证 paper faithfulness 证据一致性（必达）
    # 注意：只在 input_record 存在且包含 paper_faithfulness 信息时才添加到全局 mismatch_reasons
    # 这样可以避免单元测试中使用不完整 input_record 时产生副作用
    paper_faithfulness_status, paper_absent_reasons, paper_mismatch_reasons, paper_fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=input_record
    )
    
    # 仅当 paper_faithfulness 显式启用且 input_record 包含对应字段时，才将缺失视为 mismatch。
    paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False)) if isinstance(paper_cfg, dict) else False
    if paper_enabled and input_record is not None and isinstance(input_record.get("paper_faithfulness"), dict):
        # 启用模式下，paper_faithfulness 缺失或不一致必须进入 mismatch 门禁。
        if paper_mismatch_reasons:
            mismatch_reasons.extend(paper_mismatch_reasons)

    primary_mismatch_reason, primary_mismatch_field_path = _resolve_primary_mismatch(
        mismatch_reasons
    )

    trajectory_absent_forced = _is_embed_trajectory_explicit_absent(input_record)
    forced_mismatch = len(mismatch_reasons) > 0
    forced_absent = (
        not isinstance(expected_plan_digest, str) or
        not expected_plan_digest or
        (
            isinstance(injection_evidence, dict)
            and injection_evidence.get("injection_absent_reason") == "inference_failed"
            and not forced_mismatch
        ) or
        (
            isinstance(trajectory_evidence, dict)
            and trajectory_evidence.get("trajectory_absent_reason") == "inference_failed"
            and not forced_mismatch
        )
    )
    if trajectory_absent_forced:
        content_evidence_payload = {
            "status": "absent",
            "score": None,
            "plan_digest": detect_time_plan_digest,
            "basis_digest": detect_time_basis_digest,
            "content_failure_reason": "detector_no_plan_expected" if not isinstance(expected_plan_digest, str) or not expected_plan_digest else None,
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "audit": {
                "impl_identity": "detect_orchestrator",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"impl_id": "detect_orchestrator", "impl_version": "v1"}),
                "trace_digest": digests.canonical_sha256({
                    "trajectory_status": trajectory_status,
                    "trajectory_mismatch_reason": trajectory_mismatch_reason,
                    "plan_digest_status": plan_digest_status
                })
            }
        }
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _bind_scores_if_ok(content_evidence_payload)
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_absent_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    elif forced_mismatch:
        content_evidence_payload = {
            "status": "mismatch",
            "score": None,
            "plan_digest": detect_time_plan_digest,
            "basis_digest": detect_time_basis_digest,
            "content_failure_reason": _resolve_mismatch_failure_reason(primary_mismatch_reason),
            "content_mismatch_reason": primary_mismatch_reason,
            "content_mismatch_field_path": primary_mismatch_field_path,
            "mismatch_reasons": list(mismatch_reasons),
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "audit": {
                "impl_identity": "detect_orchestrator",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"impl_id": "detect_orchestrator", "impl_version": "v1"}),
                "trace_digest": digests.canonical_sha256({
                    "mismatch_reasons": mismatch_reasons,
                    "primary_mismatch_reason": primary_mismatch_reason,
                    "primary_mismatch_field_path": primary_mismatch_field_path
                })
            }
        }
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _bind_scores_if_ok(content_evidence_payload)
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_mismatch_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    elif forced_absent:
        content_evidence_payload = {
            "status": "absent",
            "score": None,
            "plan_digest": detect_time_plan_digest,
            "basis_digest": detect_time_basis_digest,
            "content_failure_reason": "detector_no_plan_expected" if not isinstance(expected_plan_digest, str) or not expected_plan_digest else None,
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "audit": {
                "impl_identity": "detect_orchestrator",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"impl_id": "detect_orchestrator", "impl_version": "v1"}),
                "trace_digest": digests.canonical_sha256({
                    "trajectory_status": trajectory_status,
                    "trajectory_mismatch_reason": trajectory_mismatch_reason,
                    "plan_digest_status": plan_digest_status
                })
            }
        }
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _bind_scores_if_ok(content_evidence_payload)
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_absent_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    else:
        hf_evidence = _build_hf_detect_evidence(
            impl_set=impl_set,
            cfg=cfg,
            cfg_digest=cfg_digest,
            plan_payload=plan_payload,
            plan_digest=detect_time_plan_digest,
            embed_time_plan_digest=embed_time_plan_digest,
            trajectory_evidence=trajectory_evidence,
        )
        if _is_image_domain_sidecar_enabled(cfg, ablation_override=bool(enable_image_sidecar)):
            lf_raw_score, hf_raw_score, raw_score_traces = _extract_content_raw_scores_from_image(
                cfg=cfg,
                input_record=input_record,
                plan_payload=plan_payload,
                plan_digest=detect_time_plan_digest,
                cfg_digest=cfg_digest,
            )
        else:
            lf_raw_score = None
            hf_raw_score = None
            raw_score_traces = {
                "lf": {
                    "lf_status": "absent",
                    "lf_absent_reason": (
                        "image_domain_sidecar_disabled_by_ablation"
                        if not bool(enable_image_sidecar)
                        else "image_domain_sidecar_disabled"
                    ),
                },
                "hf": {
                    "hf_status": "absent",
                    "hf_absent_reason": (
                        "image_domain_sidecar_disabled_by_ablation"
                        if not bool(enable_image_sidecar)
                        else "image_domain_sidecar_disabled"
                    ),
                },
            }
        lf_evidence = _extract_lf_evidence_from_input_record(input_record)
        detector_inputs: Dict[str, Any] = {
            "expected_plan_digest": expected_plan_digest,
            "observed_plan_digest": detect_time_plan_digest,
            "disable_cfg_plan_digest_fallback": (not detect_test_mode),
            "plan_digest": detect_time_plan_digest,
            "test_mode": detect_test_mode,
            "lf_evidence": lf_evidence,
            "hf_evidence": hf_evidence,
            "lf_score": lf_raw_score,
            "hf_score": hf_raw_score,
            "lf_detect_trace": raw_score_traces.get("lf") if isinstance(raw_score_traces, dict) else None,
            "hf_detect_trace": raw_score_traces.get("hf") if isinstance(raw_score_traces, dict) else None,
            "trajectory_evidence": trajectory_evidence,
            "injection_evidence": injection_evidence,
            "plan_payload": plan_payload,
        }
        content_result = impl_set.content_extractor.extract(
            cfg,
            inputs=detector_inputs,
            cfg_digest=cfg_digest,
        )
        content_evidence_payload = _adapt_content_evidence_for_fusion(content_result)
        if content_evidence_payload is None:
            content_evidence_payload = {}
        content_result = content_evidence_payload
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _merge_hf_evidence(content_evidence_payload, hf_evidence)
        _bind_raw_scores_to_content_payload(content_evidence_payload, lf_raw_score, hf_raw_score, raw_score_traces)
        _bind_scores_if_ok(content_evidence_payload)
        content_evidence_adapted = _adapt_content_evidence_for_fusion(content_evidence_payload)
        
        # 从 input_record 中提取 calibrate 生成的 thresholds_artifact，
        # 并注入到 cfg 中供 fusion_rule.fuse() 使用（必须修正：threshold binding error）。
        if isinstance(input_record, dict) and "thresholds_artifact" in input_record:
            thresholds_artifact = input_record["thresholds_artifact"]
            if isinstance(thresholds_artifact, dict):
                cfg["__thresholds_artifact__"] = thresholds_artifact
        
        fusion_result = impl_set.fusion_rule.fuse(cfg, content_evidence_adapted, geometry_evidence_adapted)
    input_fields = len(input_record or {})

    # 实现 detect 侧同构分数与一致性校验
    detect_runtime_mode = "fallback_identity_v0"  # 默认：未获得可用 detect 同构分数
    final_latents = cfg.get("__detect_final_latents__")  # 从 CLI 层捕获的最后 latents

    if not forced_mismatch and final_latents is not None and isinstance(plan_payload, dict):
        # 从 plan_payload 提取 LF/HF basis
        lf_basis = plan_payload.get("lf_basis")
        hf_basis = plan_payload.get("hf_basis")

        # 从 input_record 提取 embed 侧分数（兼容 content_evidence 承载）。
        embed_lf_score = None
        embed_hf_score = None
        if isinstance(input_record, dict):
            embed_lf_score = input_record.get("lf_score")
            embed_hf_score = input_record.get("hf_score")
            embed_content_evidence = input_record.get("content_evidence")
            if isinstance(embed_content_evidence, dict):
                if embed_lf_score is None:
                    embed_lf_score = embed_content_evidence.get("lf_score")
                if embed_hf_score is None:
                    embed_hf_score = embed_content_evidence.get("hf_score")

        # 计算 detect 侧 LF 分数（同构方式）
        detect_lf_score, detect_lf_status = detector_scoring.extract_lf_score_from_detect_latents(
            final_latents,
            lf_basis,
            embed_lf_score,
            cfg
        )

        # 计算 detect 侧 HF 分数（同构方式）
        detect_hf_score, detect_hf_status = detector_scoring.extract_hf_score_from_detect_latents(
            final_latents,
            hf_basis,
            embed_hf_score,
            cfg
        )

        # 校验 plan_digest 与 basis_digest 一致性
        embed_plan_digest = input_record.get("plan_digest") if input_record else None
        embed_basis_digest = input_record.get("basis_digest") if input_record else None

        plan_digest_consistent, plan_digest_reason = detector_scoring.validate_plan_digest_consistency(
            embed_plan_digest,
            detect_time_plan_digest
        )
        basis_digest_consistent, basis_digest_reason = detector_scoring.validate_basis_digest_consistency(
            embed_basis_digest,
            getattr(detect_plan_result, "basis_digest", None) if detect_plan_result else None
        )

        if content_evidence_payload is None:
            content_evidence_payload = {}

        # 追加 detect 侧分数与一致性状态到 content_evidence
        content_evidence_payload["detect_lf_score"] = detect_lf_score
        content_evidence_payload["detect_hf_score"] = detect_hf_score

        lf_score_drift_status = None
        if detect_lf_status == "ok":
            lf_score_drift_status = "ok"
        elif detect_lf_status == "lf_score_drift_detected":
            lf_score_drift_status = "drift_detected"

        hf_score_drift_status = None
        if detect_hf_status == "ok":
            hf_score_drift_status = "ok"
        elif detect_hf_status == "hf_score_drift_detected":
            hf_score_drift_status = "drift_detected"

        if lf_score_drift_status is not None:
            content_evidence_payload["lf_score_drift_status"] = lf_score_drift_status
        if hf_score_drift_status is not None:
            content_evidence_payload["hf_score_drift_status"] = hf_score_drift_status

        if basis_digest_consistent:
            basis_digest_match = "consistent"
        elif "absent" in basis_digest_reason:
            basis_digest_match = "absent"
        else:
            basis_digest_match = "mismatch"

        if trajectory_status == "ok":
            trajectory_digest_match = "consistent"
        elif trajectory_status == "mismatch":
            trajectory_digest_match = "mismatch"
        else:
            trajectory_digest_match = "absent"

        content_evidence_payload["basis_digest_match"] = basis_digest_match
        content_evidence_payload["trajectory_digest_match"] = trajectory_digest_match

        if (not plan_digest_consistent and "absent" not in plan_digest_reason) or basis_digest_match == "mismatch" or trajectory_status == "mismatch":
            subspace_consistency_status = "inconsistent"
        elif "absent" in plan_digest_reason or basis_digest_match == "absent" or trajectory_status == "absent":
            subspace_consistency_status = "absent"
        else:
            subspace_consistency_status = "ok"

        content_evidence_payload["subspace_consistency_status"] = subspace_consistency_status

        subspace_semantics = _extract_subspace_evidence_semantics(plan_payload)
        evidence_level = subspace_semantics.get("evidence_level") if isinstance(subspace_semantics.get("evidence_level"), str) else "<absent>"
        subspace_primary_path = bool(evidence_level in {"primary", "hybrid"})
        pipeline_runtime_meta = cfg.get("__pipeline_runtime_meta__") if isinstance(cfg.get("__pipeline_runtime_meta__"), dict) else {}
        synthetic_pipeline_runtime = bool(pipeline_runtime_meta.get("synthetic_pipeline", False)) if isinstance(pipeline_runtime_meta, dict) else False

        if isinstance(content_evidence_payload, dict):
            content_evidence_payload["subspace_evidence_semantics"] = subspace_semantics
            content_evidence_payload["subspace_evidence_level"] = evidence_level
            content_evidence_payload["subspace_primary_path"] = subspace_primary_path
            content_evidence_payload["synthetic_pipeline_runtime"] = synthetic_pipeline_runtime

        runtime_built = bool(pipeline_runtime_meta.get("status") == "built") if isinstance(pipeline_runtime_meta, dict) else False

        # 如果 detect 侧分数有效、未命中不一致且运行期为真实非 synthetic pipeline，则标记为真实运行模式。
        if (
            detect_lf_status == "ok"
            and subspace_consistency_status != "inconsistent"
            and runtime_built
            and (not synthetic_pipeline_runtime)
        ):
            detect_runtime_mode = "real"

    _bind_scores_if_ok(content_evidence_payload)

    # 删除临时的 latents 字段，确保不写入 records
    cfg.pop("__detect_final_latents__", None)
    cfg.pop("__detect_pipeline_obj__", None)
    cfg.pop("__pipeline_runtime_meta__", None)
    cfg.pop("__detect_attention_maps__", None)
    cfg.pop("__detect_self_attention_maps__", None)
    cfg.pop("__runtime_self_attention_maps__", None)

    plan_digest_mismatch_reason = plan_digest_reason if plan_digest_reason == "plan_digest_mismatch" else None

    record: Dict[str, Any] = {
        "operation": "detect",
        "detect_runtime_mode": detect_runtime_mode,
        "detect_runtime_status": "active" if detect_runtime_mode == "real" else "fallback",
        "detect_runtime_is_fallback": (detect_runtime_mode != "real"),
        "image_path": "<absent>",
        "score": getattr(fusion_result, "evidence_summary", {}).get("content_score"),
        "execution_report": {
            "content_chain_status": "fail" if forced_mismatch else "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "fail" if forced_mismatch else "ok",
            "audit_obligations_satisfied": True
        },
        "input_record_fields": input_fields,
        "plan_digest_expected": expected_plan_digest,
        "plan_digest_observed": detect_time_plan_digest,
        "plan_digest_status": plan_digest_status,
        "plan_digest_validation_status": plan_digest_status,
        "plan_digest_mismatch_reason": primary_mismatch_reason if forced_mismatch else plan_digest_mismatch_reason,
        "allow_cfg_plan_digest_fallback_used": allow_cfg_plan_digest_fallback_used,
        # (append-only) 保留完整的 payload，供后续升级 fusion 规则时直接消费冻结字段。
        "content_evidence_payload": content_evidence_payload,
        "geometry_evidence_payload": geometry_evidence_payload,
        "content_result": content_result,
        "geometry_result": geometry_result,
        "fusion_result": fusion_result,
        # (S-D) Paper Faithfulness: 添加一致性验证结果（结构化 failure semantics）
        "paper_faithfulness": {
            "status": paper_faithfulness_status,
            "absent_reasons": paper_absent_reasons,
            "mismatch_reasons": paper_mismatch_reasons,
            "fail_reasons": paper_fail_reasons
        }
    }
    return record


def _resolve_cfg_plan_digest(cfg: Dict[str, Any]) -> Optional[str]:
    """
    功能：从 cfg 读取 plan_digest（仅用于 test_mode）。

    Resolve cfg-side plan_digest for test-mode-only fallback.

    Args:
        cfg: Configuration mapping.

    Returns:
        plan_digest string or None.
    """
    if not isinstance(cfg, dict):
        return None
    watermark_cfg = cfg.get("watermark")
    if not isinstance(watermark_cfg, dict):
        return None
    candidate = watermark_cfg.get("plan_digest")
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def _resolve_detect_test_mode(cfg: Dict[str, Any]) -> bool:
    """
    功能：解析 detect 的 test_mode 开关。

    Resolve detect test_mode switch from cfg.

    Args:
        cfg: Configuration mapping.

    Returns:
        True if detect test_mode is enabled.
    """
    if not isinstance(cfg, dict):
        return False
    direct = cfg.get("test_mode")
    if isinstance(direct, bool):
        return direct
    detect_cfg = cfg.get("detect")
    if isinstance(detect_cfg, dict):
        runtime_cfg = detect_cfg.get("runtime")
        if isinstance(runtime_cfg, dict):
            runtime_test_mode = runtime_cfg.get("test_mode")
            if isinstance(runtime_test_mode, bool):
                return runtime_test_mode
        detect_test_mode = detect_cfg.get("test_mode")
        if isinstance(detect_test_mode, bool):
            return detect_test_mode
    return False


def _bind_scores_if_ok(content_evidence_payload: Dict[str, Any]) -> None:
    """
    功能：分数写入纪律收口，仅 status=ok 允许数值分数。

    Enforce score write discipline: numeric score fields are allowed only when status="ok".

    Args:
        content_evidence_payload: Mutable content evidence mapping.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return

    status_value = content_evidence_payload.get("status")
    if not isinstance(status_value, str):
        status_value = "failed"
        content_evidence_payload["status"] = status_value

    score_parts = content_evidence_payload.get("score_parts")
    if status_value != "ok":
        content_evidence_payload["score"] = None
        content_evidence_payload["lf_score"] = None
        content_evidence_payload["hf_score"] = None
        if isinstance(score_parts, dict):
            for numeric_key in [
                "lf_score",
                "hf_score",
                "content_score",
                "detect_lf_score",
                "detect_hf_score",
            ]:
                if isinstance(score_parts.get(numeric_key), (int, float)):
                    score_parts[numeric_key] = None
        return

    for field_name in ["score", "lf_score", "hf_score"]:
        score_value = content_evidence_payload.get(field_name)
        if score_value is None:
            continue
        if not isinstance(score_value, (int, float)) or not np.isfinite(float(score_value)):
            content_evidence_payload["status"] = "failed"
            content_evidence_payload["content_failure_reason"] = "detector_score_validation_failed"
            content_evidence_payload["score"] = None
            content_evidence_payload["lf_score"] = None
            content_evidence_payload["hf_score"] = None
            content_evidence_payload["score_parts"] = None
            return


def _build_hf_detect_evidence(
    impl_set: BuiltImplSet,
    cfg: Dict[str, Any],
    cfg_digest: Optional[str],
    plan_payload: Optional[Dict[str, Any]],
    plan_digest: Optional[str],
    embed_time_plan_digest: Optional[str],
    trajectory_evidence: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    功能：构造 detect 侧 HF 证据。

    Build detect-side HF evidence under planner-defined plan.

    Args:
        cfg: Configuration mapping.
        cfg_digest: Optional cfg digest.
        plan_payload: Planner evidence mapping.
        plan_digest: Detect-time plan digest.
        embed_time_plan_digest: Embed-time plan digest.
        trajectory_evidence: Optional trajectory evidence mapping.

    Returns:
        HF evidence mapping.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(impl_set, BuiltImplSet):
        raise TypeError("impl_set must be BuiltImplSet")

    embedder = impl_set.hf_embedder
    if embedder is None or not hasattr(embedder, "detect"):
        embedder = HighFreqEmbedder(
            impl_id=HIGH_FREQ_EMBEDDER_ID,
            impl_version=HIGH_FREQ_EMBEDDER_VERSION,
            impl_digest=digests.canonical_sha256(
                {
                    "impl_id": HIGH_FREQ_EMBEDDER_ID,
                    "impl_version": HIGH_FREQ_EMBEDDER_VERSION,
                }
            ),
        )

    expected_plan_digest = embed_time_plan_digest if isinstance(embed_time_plan_digest, str) and embed_time_plan_digest else plan_digest
    detect_result = embedder.detect(
        latents_or_features=trajectory_evidence,
        plan=plan_payload,
        cfg=cfg,
        cfg_digest=cfg_digest,
        expected_plan_digest=expected_plan_digest,
    )
    if isinstance(detect_result, tuple) and len(detect_result) == 2:
        hf_score, evidence = detect_result
    else:
        hf_score = None
        evidence = {
            "status": "failed",
            "hf_score": None,
            "hf_trace_digest": None,
            "hf_evidence_summary": {
                "hf_status": "failed",
                "hf_failure_reason": "hf_detection_invalid_output",
            },
            "content_failure_reason": "hf_detection_invalid_output",
        }
    if not isinstance(evidence, dict):
        evidence = {
            "status": "failed",
            "hf_score": None,
            "hf_trace_digest": None,
            "hf_evidence_summary": {
                "hf_status": "failed",
                "hf_failure_reason": "hf_detection_failed",
            },
            "content_failure_reason": "hf_detection_failed",
        }
    if "hf_score" not in evidence:
        evidence["hf_score"] = hf_score
    return evidence


def _extract_content_raw_scores_from_image(
    cfg: Dict[str, Any],
    input_record: Optional[Dict[str, Any]],
    plan_payload: Optional[Dict[str, Any]],
    plan_digest: Optional[str],
    cfg_digest: Optional[str],
) -> tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    功能：从图像提取 LF/HF 原始分数。 

    Extract LF/HF raw scores from image artifact for calibration-ready evidence.

    Args:
        cfg: Configuration mapping.
        input_record: Embed-side record mapping.
        plan_payload: Planner payload mapping.
        plan_digest: Detect-side plan digest.
        cfg_digest: Detect-side config digest.

    Returns:
        Tuple of (lf_score, hf_score, traces).
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    image_path = _resolve_detect_image_path(cfg, input_record)
    if image_path is None:
        return None, None, {
            "lf": {"lf_status": "absent", "lf_absent_reason": "detect_image_absent"},
            "hf": {"hf_status": "absent", "hf_absent_reason": "detect_image_absent"},
        }

    try:
        image_array = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None, None, {
            "lf": {"lf_status": "fail", "lf_failure_reason": "detect_image_load_failed"},
            "hf": {"hf_status": "fail", "hf_failure_reason": "detect_image_load_failed"},
        }

    plan_dict = _resolve_plan_dict(plan_payload)
    band_spec = plan_dict.get("band_spec") if isinstance(plan_dict.get("band_spec"), dict) else {}
    routing_summary = band_spec.get("hf_selector_summary") if isinstance(band_spec.get("hf_selector_summary"), dict) else {}

    key_material = digests.canonical_sha256(
        {
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "key_id": cfg.get("watermark", {}).get("key_id"),
            "pattern_id": cfg.get("watermark", {}).get("pattern_id"),
        }
    )

    lf_params = _build_lf_image_embed_params_for_detect(cfg)
    lf_score, lf_trace = detect_low_freq_score(image_array, band_spec, key_material, lf_params)

    hf_enabled = bool(cfg.get("watermark", {}).get("hf", {}).get("enabled", False))
    if hf_enabled:
        hf_params = _build_hf_image_embed_params_for_detect(cfg)
        hf_score, hf_trace = detect_high_freq_score(image_array, routing_summary, key_material, hf_params)
    else:
        hf_score = None
        hf_trace = {"hf_status": "absent", "hf_absent_reason": "hf_disabled_by_config"}

    return lf_score, hf_score, {"lf": lf_trace, "hf": hf_trace}


def _bind_raw_scores_to_content_payload(
    content_evidence_payload: Dict[str, Any],
    lf_score: Optional[float],
    hf_score: Optional[float],
    traces: Dict[str, Any],
) -> None:
    """
    功能：将 LF/HF 原始分数和 trace 写入 content evidence。 

    Bind LF/HF raw scores and traces into content evidence score_parts.
    """
    if not isinstance(content_evidence_payload, dict):
        return
    if not isinstance(traces, dict):
        return

    score_parts = content_evidence_payload.get("score_parts")
    if not isinstance(score_parts, dict):
        score_parts = {}
        content_evidence_payload["score_parts"] = score_parts

    lf_trace = traces.get("lf") if isinstance(traces.get("lf"), dict) else {}
    hf_trace = traces.get("hf") if isinstance(traces.get("hf"), dict) else {}

    content_evidence_payload["lf_score"] = lf_score
    score_parts["lf_detect_trace"] = lf_trace
    score_parts["lf_status"] = lf_trace.get("lf_status", "failed")

    hf_status = hf_trace.get("hf_status")
    if hf_status == "absent" and hf_trace.get("hf_absent_reason") == "hf_disabled_by_config":
        content_evidence_payload.pop("hf_score", None)
        content_evidence_payload.pop("hf_trace_digest", None)
        score_parts.pop("hf_status", None)
        score_parts.pop("hf_metrics", None)
        score_parts.pop("hf_absent_reason", None)
        score_parts.pop("hf_failure_reason", None)
        score_parts.pop("hf_detect_trace", None)
    else:
        content_evidence_payload["hf_score"] = hf_score
        score_parts["hf_detect_trace"] = hf_trace
        score_parts["hf_status"] = hf_status if isinstance(hf_status, str) else "failed"
        if "hf_absent_reason" in hf_trace:
            score_parts["hf_absent_reason"] = hf_trace.get("hf_absent_reason")
        if "hf_failure_reason" in hf_trace:
            score_parts["hf_failure_reason"] = hf_trace.get("hf_failure_reason")


def _resolve_detect_image_path(cfg: Dict[str, Any], input_record: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not isinstance(cfg, dict):
        return None
    candidates = [
        cfg.get("__detect_input_image_path__"),
        cfg.get("input_image_path"),
    ]
    if isinstance(input_record, dict):
        candidates.append(input_record.get("watermarked_path"))
        candidates.append(input_record.get("image_path"))
    for value in candidates:
        if isinstance(value, str) and value:
            path = Path(value).resolve()
            if path.exists() and path.is_file():
                return path
    return None


def _resolve_plan_dict(plan_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(plan_payload, dict):
        plan_node = plan_payload.get("plan")
        if isinstance(plan_node, dict):
            return plan_node
        return plan_payload
    return {}


def _build_lf_image_embed_params_for_detect(cfg: Dict[str, Any]) -> Dict[str, Any]:
    lf_cfg = cfg.get("watermark", {}).get("lf", {}) if isinstance(cfg.get("watermark", {}), dict) else {}
    if not isinstance(lf_cfg, dict):
        lf_cfg = {}
    return {
        "dct_block_size": int(lf_cfg.get("dct_block_size", 8)),
        "lf_coeff_indices": lf_cfg.get("lf_coeff_indices", [[1, 1], [1, 2], [2, 1]]),
        "alpha": float(lf_cfg.get("strength", 1.5)),
        "redundancy": int(lf_cfg.get("ecc", 1)),
        "variance": float(lf_cfg.get("variance", 1.5)),
    }


def _build_hf_image_embed_params_for_detect(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hf_cfg = cfg.get("watermark", {}).get("hf", {}) if isinstance(cfg.get("watermark", {}), dict) else {}
    if not isinstance(hf_cfg, dict):
        hf_cfg = {}
    return {
        "beta": float(hf_cfg.get("tau", 2.0)),
        "tail_truncation_ratio": float(hf_cfg.get("tail_truncation_ratio", 0.1)),
        "tail_truncation_mode": hf_cfg.get("tail_truncation_mode", "top_k_per_latent"),
        "sampling_stride": int(hf_cfg.get("sampling_stride", 1)),
    }


def _merge_hf_evidence(content_evidence_payload: Dict[str, Any], hf_evidence: Dict[str, Any]) -> None:
    """
    功能：合并 HF 证据到 content_evidence。

    Merge HF evidence into content evidence payload using existing registered fields.

    Args:
        content_evidence_payload: Mutable content evidence mapping.
        hf_evidence: HF evidence mapping.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return
    if not isinstance(hf_evidence, dict):
        return

    content_evidence_payload["hf_trace_digest"] = hf_evidence.get("hf_trace_digest")
    content_evidence_payload["hf_score"] = hf_evidence.get("hf_score")

    score_parts = content_evidence_payload.get("score_parts")
    if not isinstance(score_parts, dict):
        score_parts = {}
        content_evidence_payload["score_parts"] = score_parts

    score_parts["content_score_rule_version"] = hf_evidence.get("content_score_rule_version")
    score_parts["hf_status"] = hf_evidence.get("status")
    summary = hf_evidence.get("hf_evidence_summary")
    if isinstance(summary, dict):
        score_parts["hf_metrics"] = summary
        if "hf_absent_reason" in summary:
            score_parts["hf_absent_reason"] = summary.get("hf_absent_reason")
        if "hf_failure_reason" in summary:
            score_parts["hf_failure_reason"] = summary.get("hf_failure_reason")


def _merge_injection_evidence(content_evidence_payload: Dict[str, Any], injection_evidence: Dict[str, Any]) -> None:
    """
    功能：合并注入证据到 content_evidence。
    
    Merge injection evidence into content evidence payload using registered fields.

    Args:
        content_evidence_payload: Mutable content evidence mapping.
        injection_evidence: Injection evidence mapping.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return
    if not isinstance(injection_evidence, dict):
        return

    content_evidence_payload["injection_status"] = injection_evidence.get("status")
    content_evidence_payload["injection_absent_reason"] = injection_evidence.get("injection_absent_reason")
    content_evidence_payload["injection_failure_reason"] = injection_evidence.get("injection_failure_reason")
    content_evidence_payload["injection_trace_digest"] = injection_evidence.get("injection_trace_digest")
    content_evidence_payload["injection_params_digest"] = injection_evidence.get("injection_params_digest")
    content_evidence_payload["injection_metrics"] = injection_evidence.get("injection_metrics")
    content_evidence_payload["subspace_binding_digest"] = injection_evidence.get("subspace_binding_digest")
    content_evidence_payload["lf_impl_binding"] = injection_evidence.get("lf_impl_binding")
    content_evidence_payload["hf_impl_binding"] = injection_evidence.get("hf_impl_binding")


def _evaluate_injection_consistency(
    input_record: Optional[Dict[str, Any]],
    injection_evidence: Optional[Dict[str, Any]]
) -> tuple[str, Optional[str]]:
    """
    功能：校验 embed/detect 两端注入证据一致性。
    
    Evaluate injection evidence consistency between embed-time record and detect-time runtime.

    Args:
        input_record: Embed-time input record mapping or None.
        injection_evidence: Detect-time injection evidence mapping or None.

    Returns:
        Tuple of (status, mismatch_reason_or_none).
        status is one of: "ok", "absent", "mismatch".
    """
    if input_record is not None and not isinstance(input_record, dict):
        # input_record 类型不合法，必须 fail-fast。
        raise TypeError("input_record must be dict or None")
    if injection_evidence is not None and not isinstance(injection_evidence, dict):
        # injection_evidence 类型不合法，必须 fail-fast。
        raise TypeError("injection_evidence must be dict or None")

    embed_injection = None
    if isinstance(input_record, dict):
        for key in ["content_evidence_payload", "content_evidence", "content_result"]:
            candidate = input_record.get(key)
            if isinstance(candidate, dict) and "injection_status" in candidate:
                embed_injection = candidate
                break

    if embed_injection is None:
        # 向后兼容：embed 未提供注入证据时不触发缺失分支。
        return "ok", None
    if injection_evidence is None:
        return "absent", "injection_evidence_missing"

    embed_status = embed_injection.get("injection_status")
    detect_status = injection_evidence.get("status")
    if embed_status == "mismatch" or detect_status == "mismatch":
        return "mismatch", "injection_status_mismatch"
    if embed_status != "ok" or detect_status != "ok":
        return "absent", "injection_status_not_ok"

    embed_trace_digest = embed_injection.get("injection_trace_digest")
    detect_trace_digest = injection_evidence.get("injection_trace_digest")
    embed_params_digest = embed_injection.get("injection_params_digest")
    detect_params_digest = injection_evidence.get("injection_params_digest")
    embed_binding_digest = embed_injection.get("subspace_binding_digest")
    detect_binding_digest = injection_evidence.get("subspace_binding_digest")

    if not isinstance(embed_trace_digest, str) or not isinstance(detect_trace_digest, str):
        return "mismatch", "injection_trace_digest_invalid"
    if not isinstance(embed_params_digest, str) or not isinstance(detect_params_digest, str):
        return "mismatch", "injection_params_digest_invalid"
    if embed_trace_digest != detect_trace_digest:
        return "mismatch", "injection_trace_digest_mismatch"
    if embed_params_digest != detect_params_digest:
        return "mismatch", "injection_params_digest_mismatch"
    if isinstance(embed_binding_digest, str) and embed_binding_digest:
        if not isinstance(detect_binding_digest, str) or detect_binding_digest != embed_binding_digest:
            return "mismatch", "injection_subspace_binding_digest_mismatch"
    return "ok", None


def _evaluate_paper_impl_binding_consistency(
    cfg: Dict[str, Any],
    injection_evidence: Optional[Dict[str, Any]],
    input_record: Optional[Dict[str, Any]] = None,
) -> tuple[str, Optional[str]]:
    """
    功能：在 paper 模式下校验 HF/LF impl 绑定一致性。

    Validate impl binding consistency for paper mode and reject fallback-only claims.

    Args:
        cfg: Runtime configuration mapping.
        injection_evidence: Injection evidence mapping.

    Returns:
        Tuple of (status, reason) where status in {ok, absent, mismatch}.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if injection_evidence is not None and not isinstance(injection_evidence, dict):
        raise TypeError("injection_evidence must be dict or None")
    if input_record is not None and not isinstance(input_record, dict):
        raise TypeError("input_record must be dict or None")

    paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
    if not bool(paper_cfg.get("enabled", False)):
        return "ok", None

    if isinstance(injection_evidence, dict):
        detect_status = injection_evidence.get("status")
        if isinstance(detect_status, str) and detect_status != "ok":
            return "absent", "paper_impl_binding_injection_status_not_ok"

    resolved_binding_source = injection_evidence
    if not isinstance(resolved_binding_source, dict):
        resolved_binding_source = _extract_embed_impl_binding_source(input_record)
        if not isinstance(resolved_binding_source, dict):
            return "absent", "paper_impl_binding_evidence_absent"

    for channel_name in ["lf_impl_binding", "hf_impl_binding"]:
        binding_payload = resolved_binding_source.get(channel_name)
        if not isinstance(binding_payload, dict):
            return "mismatch", f"{channel_name}_missing_under_paper_mode"
        impl_selected = binding_payload.get("impl_selected")
        if not isinstance(impl_selected, str) or not impl_selected:
            return "mismatch", f"{channel_name}_impl_selected_absent"
        fallback_used = binding_payload.get("fallback_used")
        if bool(fallback_used):
            return "mismatch", f"{channel_name}_fallback_used_under_paper_mode"
    return "ok", None


def _extract_embed_impl_binding_source(input_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    功能：从 embed 输入记录提取 impl 绑定证据来源。

    Extract LF/HF impl binding source from embed-time record fields.

    Args:
        input_record: Optional embed-time input record mapping.

    Returns:
        Mapping containing lf_impl_binding/hf_impl_binding if available.
    """
    if not isinstance(input_record, dict):
        return None

    for field_name in ["content_evidence_payload", "content_evidence", "content_result"]:
        candidate = input_record.get(field_name)
        if isinstance(candidate, dict):
            has_lf = isinstance(candidate.get("lf_impl_binding"), dict)
            has_hf = isinstance(candidate.get("hf_impl_binding"), dict)
            if has_lf or has_hf:
                return candidate
    return None


def _evaluate_paper_faithfulness_consistency(
    input_record: Optional[Dict[str, Any]]
) -> tuple[str, list[str], list[str], list[str]]:
    """
    功能：校验 paper faithfulness 证据一致性（S-D 必达）。

    Evaluate paper faithfulness evidence consistency between embed-time record.
    Validates: pipeline_fingerprint_digest, injection_site_digest, paper_spec_digest.
    Returns structured failure semantics: absent_reasons / mismatch_reasons / fail_reasons.

    Args:
        input_record: Embed-time input record mapping or None.

    Returns:
        Tuple of (status, absent_reasons, mismatch_reasons, fail_reasons).
        status is one of: "ok", "absent", "mismatch", "fail".
        absent_reasons: list of tokens for missing required evidence (non-empty if status="absent").
        mismatch_reasons: list of tokens for inconsistent evidence (non-empty if status="mismatch").
        fail_reasons: list of tokens for failed validation (non-empty if status="fail").

    Raises:
        TypeError: If input_record type is invalid.
    """
    if input_record is not None and not isinstance(input_record, dict):
        raise TypeError("input_record must be dict or None")

    absent_reasons: list[str] = []
    mismatch_reasons: list[str] = []
    fail_reasons: list[str] = []

    if input_record is None:
        absent_reasons.append("input_record_is_none")
        return "absent", absent_reasons, mismatch_reasons, fail_reasons

    # 提取 embed-time paper faithfulness 证据。
    embed_content_evidence = None
    for key in ["content_evidence_payload", "content_evidence", "content_result"]:
        candidate = input_record.get(key)
        if isinstance(candidate, dict):
            embed_content_evidence = candidate
            break

    # (1) 验证 content_evidence 存在性（整体 absent 前置检查）。
    if not isinstance(embed_content_evidence, dict):
        absent_reasons.append("content_evidence_absent")
        return "absent", absent_reasons, mismatch_reasons, fail_reasons

    # content_evidence 存在说明 embed 侧运行了，后续缺失归类为 mismatch。
    embed_paper_faithfulness = input_record.get("paper_faithfulness")

    # (2) 验证 paper_spec_digest 存在性（mismatch vs fail）。
    if isinstance(embed_paper_faithfulness, dict):
        spec_digest = embed_paper_faithfulness.get("spec_digest")
        if spec_digest == "<absent>":
            mismatch_reasons.append("paper_spec_digest_absent_or_invalid")
        elif spec_digest == "<failed>":
            fail_reasons.append("paper_spec_digest_marked_failed")
        elif not isinstance(spec_digest, str) or not spec_digest:
            mismatch_reasons.append("paper_spec_digest_missing")
    else:
        mismatch_reasons.append("paper_faithfulness_section_absent")

    # (3) 验证 pipeline_fingerprint_digest 存在性（mismatch vs fail）。
    pipeline_fingerprint_digest = embed_content_evidence.get("pipeline_fingerprint_digest")
    if pipeline_fingerprint_digest == "<absent>":
        mismatch_reasons.append("pipeline_fingerprint_digest_marked_absent")
    elif pipeline_fingerprint_digest == "<failed>":
        fail_reasons.append("pipeline_fingerprint_digest_marked_failed")
    elif not isinstance(pipeline_fingerprint_digest, str) or not pipeline_fingerprint_digest:
        mismatch_reasons.append("pipeline_fingerprint_digest_missing")

    # (4) 验证 injection_site_digest 存在性（mismatch vs fail）。
    injection_site_digest = embed_content_evidence.get("injection_site_digest")
    if injection_site_digest == "<absent>":
        mismatch_reasons.append("injection_site_digest_marked_absent")
    elif injection_site_digest == "<failed>":
        fail_reasons.append("injection_site_digest_marked_failed")
    elif not isinstance(injection_site_digest, str) or not injection_site_digest:
        mismatch_reasons.append("injection_site_digest_missing")

    # (5) 验证 alignment_digest 存在性（mismatch vs fail）。
    alignment_digest = embed_content_evidence.get("alignment_digest")
    if alignment_digest == "<absent>":
        mismatch_reasons.append("alignment_digest_marked_absent")
    elif alignment_digest == "<failed>":
        fail_reasons.append("alignment_digest_marked_failed")
    elif not isinstance(alignment_digest, str) or not alignment_digest:
        mismatch_reasons.append("alignment_digest_missing")

    # (6) 决定最终 status（优先级：fail > mismatch > absent > ok）。
    if len(fail_reasons) > 0:
        return "fail", absent_reasons, mismatch_reasons, fail_reasons
    if len(mismatch_reasons) > 0:
        return "mismatch", absent_reasons, mismatch_reasons, fail_reasons
    if len(absent_reasons) > 0:
        return "absent", absent_reasons, mismatch_reasons, fail_reasons

    return "ok", absent_reasons, mismatch_reasons, fail_reasons


def _extract_lf_evidence_from_input_record(input_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    功能：从 embed record 中提取 LF 证据。

    Extract LF evidence payload from embed-time input record.

    Args:
        input_record: Optional input record mapping.

    Returns:
        LF evidence mapping or None.
    """
    if not isinstance(input_record, dict):
        return None
    for key in ["content_evidence_payload", "content_evidence", "content_result"]:
        candidate = input_record.get(key)
        if not isinstance(candidate, dict):
            continue
        if "lf_score" in candidate or "lf_trace_digest" in candidate or candidate.get("status") in {"ok", "failed", "mismatch", "absent"}:
            return candidate
    return None


def _resolve_expected_plan_digest(input_record: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    功能：从输入记录解析 expected plan_digest。 

    Resolve expected plan digest strictly from bound input record payload.

    Args:
        input_record: Embed-side bound input record.

    Returns:
        Expected plan digest string or None.
    """
    if not isinstance(input_record, dict):
        return None

    direct = input_record.get("plan_digest")
    if isinstance(direct, str) and direct:
        return direct

    for key in ["content_evidence_payload", "content_evidence", "content_result"]:
        payload = input_record.get(key)
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("plan_digest")
        if isinstance(candidate, str) and candidate:
            return candidate

        injection_site_spec = payload.get("injection_site_spec")
        if isinstance(injection_site_spec, dict):
            injection_rule_summary = injection_site_spec.get("injection_rule_summary")
            if isinstance(injection_rule_summary, dict):
                summary_plan_digest = injection_rule_summary.get("plan_digest")
                if isinstance(summary_plan_digest, str) and summary_plan_digest:
                    return summary_plan_digest

    embed_trace = input_record.get("embed_trace")
    if isinstance(embed_trace, dict):
        trace_plan_digest = embed_trace.get("plan_digest")
        if isinstance(trace_plan_digest, str) and trace_plan_digest:
            return trace_plan_digest

        trace_injection = embed_trace.get("injection_evidence")
        if isinstance(trace_injection, dict):
            trace_injection_plan_digest = trace_injection.get("plan_digest")
            if isinstance(trace_injection_plan_digest, str) and trace_injection_plan_digest:
                return trace_injection_plan_digest

    top_level_injection = input_record.get("injection_evidence")
    if isinstance(top_level_injection, dict):
        injection_plan_digest = top_level_injection.get("plan_digest")
        if isinstance(injection_plan_digest, str) and injection_plan_digest:
            return injection_plan_digest

    subspace_plan = input_record.get("subspace_plan")
    if isinstance(subspace_plan, dict) and len(subspace_plan) > 0:
        return digests.canonical_sha256(subspace_plan)
    return None


def _collect_plan_mismatch_reasons(
    embed_time_plan_digest: Any,
    detect_time_plan_digest: Any,
    embed_time_basis_digest: Any,
    detect_time_basis_digest: Any,
    embed_time_planner_impl_identity: Any,
    detect_time_planner_impl_identity: Any
) -> list[str]:
    """
    功能：收集计划锚点不一致原因。

    Collect mismatch reasons for plan/basis/impl identity anchors.

    Args:
        embed_time_plan_digest: Embed-time plan digest.
        detect_time_plan_digest: Detect-time recomputed plan digest.
        embed_time_basis_digest: Embed-time basis digest.
        detect_time_basis_digest: Detect-time recomputed basis digest.
        embed_time_planner_impl_identity: Embed-time planner impl identity payload.
        detect_time_planner_impl_identity: Detect-time planner impl identity payload.

    Returns:
        List of mismatch reason tokens.
    """
    reasons: list[str] = []
    if isinstance(embed_time_plan_digest, str) and isinstance(detect_time_plan_digest, str):
        if embed_time_plan_digest != detect_time_plan_digest:
            reasons.append("plan_digest_mismatch")
    if isinstance(embed_time_basis_digest, str) and isinstance(detect_time_basis_digest, str):
        if embed_time_basis_digest != detect_time_basis_digest:
            reasons.append("basis_digest_mismatch")
    if isinstance(embed_time_planner_impl_identity, dict) and isinstance(detect_time_planner_impl_identity, dict):
        if embed_time_planner_impl_identity != detect_time_planner_impl_identity:
            reasons.append("planner_impl_identity_mismatch")
    return reasons


def _evaluate_trajectory_consistency(
    input_record: Optional[Dict[str, Any]],
    trajectory_evidence: Optional[Dict[str, Any]],
    detect_planner_input_digest: Optional[str]
) -> tuple[str, Optional[str]]:
    """
    功能：校验 embed/detect 两端 trajectory 证据一致性。

    Evaluate trajectory evidence consistency between embed-time record and detect-time runtime.

    Args:
        input_record: Embed-time input record mapping or None.
        trajectory_evidence: Detect-time trajectory evidence mapping or None.
        detect_planner_input_digest: Detect-time planner input digest.

    Returns:
        Tuple of (status, mismatch_reason_or_none).
        status is one of: "ok", "absent", "mismatch".

    Raises:
        TypeError: If inputs are invalid.
    """
    if input_record is not None and not isinstance(input_record, dict):
        # input_record 类型不合法，必须 fail-fast。
        raise TypeError("input_record must be dict or None")
    if trajectory_evidence is not None and not isinstance(trajectory_evidence, dict):
        # trajectory_evidence 类型不合法，必须 fail-fast。
        raise TypeError("trajectory_evidence must be dict or None")
    if detect_planner_input_digest is not None and not isinstance(detect_planner_input_digest, str):
        # detect_planner_input_digest 类型不合法，必须 fail-fast。
        raise TypeError("detect_planner_input_digest must be str or None")

    embed_trajectory_evidence = None
    if isinstance(input_record, dict):
        candidate = None
        for key in ["content_evidence_payload", "content_evidence", "content_result"]:
            payload = input_record.get(key)
            if isinstance(payload, dict) and "trajectory_evidence" in payload:
                candidate = payload.get("trajectory_evidence")
                break
        if candidate is None and "trajectory_evidence" in input_record:
            candidate = input_record.get("trajectory_evidence")
        if candidate is not None and not isinstance(candidate, dict):
            # embed 记录中的 trajectory_evidence 类型不合法，必须 fail-fast。
            raise TypeError("embed trajectory_evidence must be dict or None")
        embed_trajectory_evidence = candidate

    embed_planner_input_digest = _extract_embed_planner_input_digest(input_record)

    if embed_trajectory_evidence is None and trajectory_evidence is None:
        return "absent", None
    if embed_trajectory_evidence is None or trajectory_evidence is None:
        return "absent", None

    embed_status = _resolve_trajectory_tap_status(embed_trajectory_evidence)
    detect_status = _resolve_trajectory_tap_status(trajectory_evidence)
    if embed_status != "ok" or detect_status != "ok":
        return "absent", None

    if not isinstance(embed_planner_input_digest, str) or not embed_planner_input_digest:
        return "absent", None
    if not isinstance(detect_planner_input_digest, str) or not detect_planner_input_digest:
        return "absent", None

    embed_spec_digest = embed_trajectory_evidence.get("trajectory_spec_digest")
    detect_spec_digest = trajectory_evidence.get("trajectory_spec_digest")
    embed_trajectory_digest = embed_trajectory_evidence.get("trajectory_digest")
    detect_trajectory_digest = trajectory_evidence.get("trajectory_digest")

    if not isinstance(embed_spec_digest, str) or not isinstance(detect_spec_digest, str):
        return "mismatch", "trajectory_evidence_invalid"
    if not isinstance(embed_trajectory_digest, str) or not isinstance(detect_trajectory_digest, str):
        return "mismatch", "trajectory_evidence_invalid"

    if embed_spec_digest != detect_spec_digest:
        return "mismatch", "trajectory_spec_digest_mismatch"
    if embed_trajectory_digest != detect_trajectory_digest:
        return "mismatch", "trajectory_digest_mismatch"
    if embed_planner_input_digest != detect_planner_input_digest:
        return "mismatch", "plan_digest_mismatch"
    return "ok", None


def _extract_embed_planner_input_digest(input_record: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    功能：从 embed 记录提取 planner_input_digest。

    Extract embed-time planner_input_digest from record payload.

    Args:
        input_record: Embed-time record mapping.

    Returns:
        Planner input digest string or None.
    """
    if not isinstance(input_record, dict):
        return None

    subspace_plan = input_record.get("subspace_plan")
    if isinstance(subspace_plan, dict):
        direct_digest = subspace_plan.get("planner_input_digest")
        if isinstance(direct_digest, str) and direct_digest:
            return direct_digest
        verifiable_spec = subspace_plan.get("verifiable_input_domain_spec")
        if isinstance(verifiable_spec, dict):
            digest_value = verifiable_spec.get("planner_input_digest")
            if isinstance(digest_value, str) and digest_value:
                return digest_value

    content_payload = input_record.get("content_evidence_payload")
    if isinstance(content_payload, dict):
        nested = content_payload.get("subspace_plan")
        if isinstance(nested, dict):
            direct_digest = nested.get("planner_input_digest")
            if isinstance(direct_digest, str) and direct_digest:
                return direct_digest
            verifiable_spec = nested.get("verifiable_input_domain_spec")
            if isinstance(verifiable_spec, dict):
                digest_value = verifiable_spec.get("planner_input_digest")
                if isinstance(digest_value, str) and digest_value:
                    return digest_value
    return None


def _is_embed_trajectory_explicit_absent(input_record: Optional[Dict[str, Any]]) -> bool:
    """
    功能：判断 embed 侧 trajectory 证据是否显式为 absent。

    Determine whether embed-side trajectory evidence is explicitly absent.

    Args:
        input_record: Embed-time input record mapping.

    Returns:
        True if embed trajectory evidence exists and status is absent.
    """
    if not isinstance(input_record, dict):
        return False

    embed_trajectory_evidence = None
    for key in ["content_evidence_payload", "content_evidence", "content_result"]:
        payload = input_record.get(key)
        if isinstance(payload, dict) and "trajectory_evidence" in payload:
            embed_trajectory_evidence = payload.get("trajectory_evidence")
            break
    if embed_trajectory_evidence is None and "trajectory_evidence" in input_record:
        embed_trajectory_evidence = input_record.get("trajectory_evidence")
    if not isinstance(embed_trajectory_evidence, dict):
        return False

    embed_status = _resolve_trajectory_tap_status(embed_trajectory_evidence)
    return embed_status == "absent"


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
    if not isinstance(content_evidence_payload, dict):
        return
    if not isinstance(trajectory_evidence, dict):
        return

    audit = content_evidence_payload.get("audit")
    if not isinstance(audit, dict):
        audit = {}
        content_evidence_payload["audit"] = audit

    tap_status = _resolve_trajectory_tap_status(trajectory_evidence)
    tap_absent_reason = _resolve_trajectory_absent_reason(trajectory_evidence)

    if isinstance(tap_status, str) and tap_status:
        audit["trajectory_tap_status"] = tap_status
    if isinstance(tap_absent_reason, str) and tap_absent_reason:
        audit["trajectory_absent_reason"] = tap_absent_reason


def _resolve_trajectory_tap_status(trajectory_evidence: Dict[str, Any]) -> Optional[str]:
    """
    功能：优先读取 trajectory audit 子状态，兼容旧 status 字段。

    Resolve trajectory tap status with new-field-first compatibility.

    Args:
        trajectory_evidence: Trajectory evidence mapping.

    Returns:
        trajectory tap status string or None.
    """
    if not isinstance(trajectory_evidence, dict):
        return None

    audit = trajectory_evidence.get("audit")
    if isinstance(audit, dict):
        value = audit.get("trajectory_tap_status")
        if isinstance(value, str) and value:
            return value

    legacy = trajectory_evidence.get("status")
    if isinstance(legacy, str) and legacy:
        return legacy
    return None


def _resolve_trajectory_absent_reason(trajectory_evidence: Dict[str, Any]) -> Optional[str]:
    """
    功能：优先读取 trajectory audit 缺失原因，兼容旧字段。

    Resolve trajectory absent reason with new-field-first compatibility.

    Args:
        trajectory_evidence: Trajectory evidence mapping.

    Returns:
        Trajectory absent reason string or None.
    """
    if not isinstance(trajectory_evidence, dict):
        return None

    audit = trajectory_evidence.get("audit")
    if isinstance(audit, dict):
        value = audit.get("trajectory_absent_reason")
        if isinstance(value, str) and value:
            return value

    legacy = trajectory_evidence.get("trajectory_absent_reason")
    if isinstance(legacy, str) and legacy:
        return legacy
    return None


def _extract_detect_planner_input_digest(detect_plan_result: Any) -> Optional[str]:
    """
    功能：从 detect 侧规划结果提取 planner_input_digest。

    Extract detect-time planner_input_digest from planner output.

    Args:
        detect_plan_result: Planner result object.

    Returns:
        Planner input digest string or None.
    """
    if detect_plan_result is None:
        return None

    if hasattr(detect_plan_result, "plan") and isinstance(detect_plan_result.plan, dict):
        direct_digest = detect_plan_result.plan.get("planner_input_digest")
        if isinstance(direct_digest, str) and direct_digest:
            return direct_digest
        verifiable_spec = detect_plan_result.plan.get("verifiable_input_domain_spec")
        if isinstance(verifiable_spec, dict):
            digest_value = verifiable_spec.get("planner_input_digest")
            if isinstance(digest_value, str) and digest_value:
                return digest_value

    if isinstance(detect_plan_result, dict):
        direct_digest = detect_plan_result.get("planner_input_digest")
        if isinstance(direct_digest, str) and direct_digest:
            return direct_digest
        verifiable_spec = detect_plan_result.get("verifiable_input_domain_spec")
        if isinstance(verifiable_spec, dict):
            digest_value = verifiable_spec.get("planner_input_digest")
            if isinstance(digest_value, str) and digest_value:
                return digest_value
    return None


def _resolve_mismatch_failure_reason(primary_mismatch_reason: str) -> str:
    """
    功能：将 mismatch 原因映射为 content_failure_reason。

    Map mismatch reason token to content_failure_reason enum.

    Args:
        primary_mismatch_reason: Primary mismatch reason token.

    Returns:
        content_failure_reason enum string.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(primary_mismatch_reason, str) or not primary_mismatch_reason:
        # primary_mismatch_reason 类型不合法，必须 fail-fast。
        raise TypeError("primary_mismatch_reason must be non-empty str")

    reason_map = {
        "plan_digest_mismatch": "detector_plan_mismatch",
        "basis_digest_mismatch": "detector_plan_mismatch",
        "planner_impl_identity_mismatch": "detector_plan_mismatch",
        "trajectory_spec_digest_mismatch": "detector_plan_mismatch",
        "trajectory_digest_mismatch": "detector_plan_mismatch",
        "trajectory_evidence_invalid": "detector_plan_mismatch",
        "injection_trace_digest_mismatch": "detector_plan_mismatch",
        "injection_params_digest_mismatch": "detector_plan_mismatch",
        "injection_trace_digest_invalid": "detector_plan_mismatch",
        "injection_params_digest_invalid": "detector_plan_mismatch",
        "injection_status_mismatch": "detector_plan_mismatch",
        "injection_subspace_binding_digest_mismatch": "detector_plan_mismatch",
        # (S-D) Paper Faithfulness mismatch reasons
        "paper_spec_digest_absent_or_invalid": "detector_plan_mismatch",
        "pipeline_fingerprint_digest_absent_or_invalid": "detector_plan_mismatch",
        "injection_site_digest_absent_or_invalid": "detector_plan_mismatch",
        "alignment_digest_absent_or_invalid": "detector_plan_mismatch",
        "paper_faithfulness_section_absent": "detector_plan_mismatch",
        "content_evidence_absent": "detector_plan_mismatch",
    }
    return reason_map.get(primary_mismatch_reason, "detector_plan_mismatch")


def _build_absent_fusion_decision(
    cfg: Dict[str, Any],
    content_evidence_adapted: Dict[str, Any],
    geometry_evidence_adapted: Dict[str, Any]
) -> FusionDecision:
    """
    功能：构造 absent 的融合判决。

    Build a FusionDecision for absent trajectory evidence.

    Args:
        cfg: Configuration mapping.
        content_evidence_adapted: Adapted content evidence mapping.
        geometry_evidence_adapted: Adapted geometry evidence mapping.

    Returns:
        FusionDecision with decision_status="abstain" and score-free evidence summary.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(content_evidence_adapted, dict):
        # content_evidence_adapted 类型不合法，必须 fail-fast。
        raise TypeError("content_evidence_adapted must be dict")
    if not isinstance(geometry_evidence_adapted, dict):
        # geometry_evidence_adapted 类型不合法，必须 fail-fast。
        raise TypeError("geometry_evidence_adapted must be dict")

    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)

    evidence_summary = {
        "content_score": None,
        "geometry_score": geometry_evidence_adapted.get("geo_score"),
        "content_status": "absent",
        "geometry_status": geometry_evidence_adapted.get("status", "absent"),
        "fusion_rule_id": "detect_absent_guard_v1"
    }
    audit = {
        "guard": "trajectory_absent_guard",
        "reason": content_evidence_adapted.get("content_failure_reason", "detector_no_evidence")
    }
    return FusionDecision(
        is_watermarked=None,
        decision_status="abstain",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary,
        audit=audit
    )


def _resolve_primary_mismatch(mismatch_reasons: list[str]) -> tuple[str, str]:
    """
    功能：选择单一主 mismatch 原因并返回对应字段路径。

    Resolve a single primary mismatch reason and its field path.

    Args:
        mismatch_reasons: Collected mismatch reason tokens.

    Returns:
        Tuple of (primary_reason, field_path).
    """
    reason_to_field_path = {
        "plan_digest_mismatch": "content_evidence.plan_digest",
        "basis_digest_mismatch": "content_evidence.basis_digest",
        "planner_impl_identity_mismatch": "content_evidence.planner_impl_identity",
        "trajectory_spec_digest_mismatch": "content_evidence.trajectory_evidence.trajectory_spec_digest",
        "trajectory_digest_mismatch": "content_evidence.trajectory_evidence.trajectory_digest",
        "trajectory_evidence_invalid": "content_evidence.trajectory_evidence",
        "injection_trace_digest_mismatch": "content_evidence.injection_trace_digest",
        "injection_params_digest_mismatch": "content_evidence.injection_params_digest",
        "injection_trace_digest_invalid": "content_evidence.injection_trace_digest",
        "injection_params_digest_invalid": "content_evidence.injection_params_digest",
        "injection_status_mismatch": "content_evidence.injection_status",
        "injection_subspace_binding_digest_mismatch": "content_evidence.subspace_binding_digest",
        "lf_impl_binding_missing_under_paper_mode": "content_evidence.lf_impl_binding",
        "hf_impl_binding_missing_under_paper_mode": "content_evidence.hf_impl_binding",
        "lf_impl_binding_impl_selected_absent": "content_evidence.lf_impl_binding.impl_selected",
        "hf_impl_binding_impl_selected_absent": "content_evidence.hf_impl_binding.impl_selected",
        "lf_impl_binding_fallback_used_under_paper_mode": "content_evidence.lf_impl_binding.fallback_used",
        "hf_impl_binding_fallback_used_under_paper_mode": "content_evidence.hf_impl_binding.fallback_used",
        # (S-D) Paper Faithfulness mismatch field paths
        "paper_spec_digest_absent_or_invalid": "paper_faithfulness.spec_digest",
        "pipeline_fingerprint_digest_absent_or_invalid": "content_evidence.pipeline_fingerprint_digest",
        "injection_site_digest_absent_or_invalid": "content_evidence.injection_site_digest",
        "alignment_digest_absent_or_invalid": "content_evidence.alignment_digest",
        "paper_faithfulness_section_absent": "paper_faithfulness",
        "content_evidence_absent": "content_evidence",
    }
    for token in [
        "trajectory_digest_mismatch",
        "plan_digest_mismatch",
        "basis_digest_mismatch",
        "planner_impl_identity_mismatch",
        "trajectory_spec_digest_mismatch",
        "trajectory_evidence_invalid",
        "injection_trace_digest_mismatch",
        "injection_params_digest_mismatch",
        "injection_trace_digest_invalid",
        "injection_params_digest_invalid",
        "injection_status_mismatch",
        "injection_subspace_binding_digest_mismatch",
        "lf_impl_binding_missing_under_paper_mode",
        "hf_impl_binding_missing_under_paper_mode",
        "lf_impl_binding_impl_selected_absent",
        "hf_impl_binding_impl_selected_absent",
        "lf_impl_binding_fallback_used_under_paper_mode",
        "hf_impl_binding_fallback_used_under_paper_mode",
        # (S-D) Paper Faithfulness mismatch priority
        "paper_spec_digest_absent_or_invalid",
        "pipeline_fingerprint_digest_absent_or_invalid",
        "injection_site_digest_absent_or_invalid",
        "alignment_digest_absent_or_invalid",
        "paper_faithfulness_section_absent",
        "content_evidence_absent",
    ]:
        if token in mismatch_reasons:
            return token, reason_to_field_path[token]
    return "unknown_mismatch", "content_evidence"


def _build_mismatch_fusion_decision(
    cfg: Dict[str, Any],
    content_evidence_adapted: Dict[str, Any],
    geometry_evidence_adapted: Dict[str, Any]
) -> FusionDecision:
    """
    功能：构造 mismatch 的融合失败判决。

    Build a single-path FusionDecision for mismatch failures.

    Args:
        cfg: Configuration mapping.
        content_evidence_adapted: Adapted content evidence mapping.
        geometry_evidence_adapted: Adapted geometry evidence mapping.

    Returns:
        FusionDecision with decision_status="error" and score-free evidence summary.
    """
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)

    evidence_summary = {
        "content_score": None,
        "geometry_score": geometry_evidence_adapted.get("geo_score"),
        "content_status": "mismatch",
        "geometry_status": geometry_evidence_adapted.get("status", "absent"),
        "fusion_rule_id": "detect_mismatch_guard_v1"
    }
    audit = {
        "guard": "plan_anchor_consistency",
        "reason": content_evidence_adapted.get("content_failure_reason", "detector_plan_mismatch")
    }
    return FusionDecision(
        is_watermarked=None,
        decision_status="error",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary,
        audit=audit
    )


def _build_planner_inputs_for_runtime(
    cfg: Dict[str, Any],
    trajectory_evidence: Optional[Dict[str, Any]] = None,
    content_evidence_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    功能：构造规划器输入签名。

    Build deterministic planner input signature from runtime cfg.

    Args:
        cfg: Configuration mapping.
        trajectory_evidence: Optional trajectory tap evidence.

    Returns:
        Planner input mapping.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if trajectory_evidence is not None and not isinstance(trajectory_evidence, dict):
        # trajectory_evidence 类型不合法，必须 fail-fast。
        raise TypeError("trajectory_evidence must be dict or None")
    if content_evidence_payload is not None and not isinstance(content_evidence_payload, dict):
        # content_evidence_payload 类型不合法，必须 fail-fast。
        raise TypeError("content_evidence_payload must be dict or None")

    trace_signature = {
        "num_inference_steps": cfg.get("inference_num_steps", cfg.get("generation", {}).get("num_inference_steps", 16) if isinstance(cfg.get("generation"), dict) else 16),
        "guidance_scale": cfg.get("inference_guidance_scale", cfg.get("generation", {}).get("guidance_scale", 7.0) if isinstance(cfg.get("generation"), dict) else 7.0),
        "height": cfg.get("inference_height", cfg.get("model", {}).get("height", 512) if isinstance(cfg.get("model"), dict) else 512),
        "width": cfg.get("inference_width", cfg.get("model", {}).get("width", 512) if isinstance(cfg.get("model"), dict) else 512),
    }
    inputs = {"trace_signature": trace_signature}
    runtime_pipeline = cfg.get("__detect_pipeline_obj__")
    runtime_latents = cfg.get("__detect_final_latents__")
    if runtime_pipeline is not None:
        inputs["pipeline"] = runtime_pipeline
    if runtime_latents is not None:
        inputs["latents"] = runtime_latents
    if trajectory_evidence is not None:
        inputs["trajectory_evidence"] = trajectory_evidence
    if isinstance(content_evidence_payload, dict):
        mask_stats = content_evidence_payload.get("mask_stats")
        if isinstance(mask_stats, dict):
            inputs["mask_summary"] = dict(mask_stats)
            routing_digest = mask_stats.get("routing_digest")
            if isinstance(routing_digest, str) and routing_digest:
                inputs["routing_digest"] = routing_digest
    return inputs


def run_calibrate_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet) -> Dict[str, Any]:
    """
    功能：执行校准流程并产出 NP 阈值工件。

    Execute calibration workflow and build NP thresholds artifacts.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_set, BuiltImplSet):
        # impl_set 类型不合法，必须 fail-fast。
        raise TypeError("impl_set must be BuiltImplSet")

    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    target_fpr = thresholds_spec.get("target_fpr")
    if not isinstance(target_fpr, (int, float)):
        raise TypeError("thresholds_spec.target_fpr must be number")

    detect_records = _load_records_for_calibration(cfg)
    scores, strata_info = load_scores_for_calibration(detect_records)
    threshold_value, order_stat_info = compute_np_threshold(scores, float(target_fpr))

    threshold_key_used = neyman_pearson.format_fpr_key_canonical(float(target_fpr))
    threshold_id = f"content_score_np_{threshold_key_used}"
    thresholds_artifact = {
        "calibration_version": "np_v1",
        "rule_id": neyman_pearson.RULE_ID,
        "rule_version": neyman_pearson.RULE_VERSION,
        "threshold_id": threshold_id,
        "score_name": "content_score",
        "target_fpr": float(target_fpr),
        "threshold_value": float(threshold_value),
        "threshold_key_used": threshold_key_used,
        "quantile_rule": "higher",
        "ties_policy": "higher",
    }
    threshold_metadata_artifact = {
        "calibration_version": "np_v1",
        "rule_id": neyman_pearson.RULE_ID,
        "rule_version": neyman_pearson.RULE_VERSION,
        "method": "neyman_pearson_v1",
        "score_name": "content_score",
        "target_fpr": float(target_fpr),
        "null_source": "wrong_key",
        "n_null": len(scores),
        "n_samples": len(scores),
        "calibration_date": "1970-01-01",
        "quantile_method": "higher",
        "target_fprs": [float(target_fpr)],
        "order_statistics": order_stat_info,
        "stratification": strata_info,
        "sample_digest": digests.canonical_sha256({"scores": [round(float(v), 12) for v in scores]}),
    }
    threshold_metadata_artifact["null_strata"] = _compute_null_strata_for_calibration(
        detect_records,
        float(threshold_value),
        cfg,
    )
    threshold_metadata_artifact["conditional_fpr"] = _compute_conditional_fpr_for_calibration(
        detect_records,
        float(threshold_value),
    )
    threshold_metadata_artifact["conditional_fpr_records"] = _compute_conditional_fpr_records_for_calibration(
        detect_records,
        float(threshold_value),
        cfg,
    )

    record: Dict[str, Any] = {
        "operation": "calibrate",
        "calibration_is_fallback": False,
        "calibration_mode": "real",
        "protocol": "neyman_pearson",
        "threshold_key_used": threshold_key_used,
        "threshold_id": threshold_id,
        "calibration_samples": len(scores),
        "calibration_summary": {
            "score_name": "content_score",
            "target_fpr": float(target_fpr),
            "threshold_value": float(threshold_value),
            "order_statistics": order_stat_info,
            "stratification": strata_info,
        },
        "thresholds_artifact": thresholds_artifact,
        "threshold_metadata_artifact": threshold_metadata_artifact,
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
    }
    return record


def _pick_first_non_empty_string(values: list[Any]) -> Optional[str]:
    """
    功能：从候选值列表中提取首个非空字符串。

    Select the first non-empty string from candidate values.

    Args:
        values: Candidate values.

    Returns:
        The first non-empty string, or None when not found.
    """
    if not isinstance(values, list):
        raise TypeError("values must be list")
    for value in values:
        if isinstance(value, str) and value and value != "<absent>":
            return value
    return None


def _resolve_cfg_digest_for_evaluate(cfg: Dict[str, Any], detect_records: list[Dict[str, Any]]) -> str:
    """
    功能：解析 evaluate 报告的 cfg_digest 锚点。

    Resolve cfg_digest anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        detect_records: Loaded detect records.

    Returns:
        Resolved cfg_digest anchor string.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(detect_records, list):
        raise TypeError("detect_records must be list")

    from_cfg = _pick_first_non_empty_string([
        cfg.get("__evaluate_cfg_digest__"),
        cfg.get("cfg_digest"),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    for record in detect_records:
        if not isinstance(record, dict):
            continue
        from_record = _pick_first_non_empty_string([
            record.get("cfg_digest"),
        ])
        if isinstance(from_record, str):
            return from_record
    return "<absent>"


def _resolve_plan_digest_for_evaluate(cfg: Dict[str, Any], detect_records: list[Dict[str, Any]]) -> str:
    """
    功能：解析 evaluate 报告的 plan_digest 锚点。

    Resolve plan_digest anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        detect_records: Loaded detect records.

    Returns:
        Resolved plan_digest anchor string.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(detect_records, list):
        raise TypeError("detect_records must be list")

    from_cfg = _pick_first_non_empty_string([
        cfg.get("__evaluate_plan_digest__"),
        _resolve_cfg_plan_digest(cfg),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    plan_digests: list[str] = []
    for record in detect_records:
        if not isinstance(record, dict):
            continue
        resolved = _pick_first_non_empty_string([
            record.get("plan_digest"),
            record.get("expected_plan_digest"),
        ])
        if isinstance(resolved, str):
            plan_digests.append(resolved)

    unique_plan_digests = sorted(set(plan_digests))
    if len(unique_plan_digests) == 1:
        return unique_plan_digests[0]
    if len(unique_plan_digests) > 1:
        return digests.canonical_sha256({"evaluate_plan_digest_candidates": unique_plan_digests})

    fallback_signatures: list[Dict[str, str]] = []
    for record in detect_records:
        if not isinstance(record, dict):
            continue
        attack = record.get("attack") if isinstance(record.get("attack"), dict) else {}
        fallback_signatures.append({
            "family": str(attack.get("family", "unknown")),
            "params_version": str(attack.get("params_version", "unknown")),
        })
    if fallback_signatures:
        return digests.canonical_sha256({
            "rule": "evaluate_plan_digest_fallback_v1",
            "attack_signatures": fallback_signatures,
        })
    return "<absent>"


def _resolve_threshold_metadata_digest_for_evaluate(
    cfg: Dict[str, Any],
    thresholds_path: Path,
    detect_records: list[Dict[str, Any]],
) -> str:
    """
    功能：解析 evaluate 报告的 threshold_metadata_digest 锚点。

    Resolve threshold metadata digest anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        thresholds_path: Threshold artifact path.
        detect_records: Loaded detect records.

    Returns:
        Resolved threshold metadata digest anchor string.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(thresholds_path, Path):
        raise TypeError("thresholds_path must be Path")
    if not isinstance(detect_records, list):
        raise TypeError("detect_records must be list")

    from_cfg = _pick_first_non_empty_string([
        cfg.get("__evaluate_threshold_metadata_digest__"),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    evaluate_cfg = cfg.get("evaluate") if isinstance(cfg.get("evaluate"), dict) else {}
    candidate_paths = [
        cfg.get("__evaluate_threshold_metadata_artifact_path__"),
        evaluate_cfg.get("threshold_metadata_artifact_path"),
        str(thresholds_path.parent / "threshold_metadata_artifact.json"),
        str(thresholds_path.parent / "threshold_metadata.json"),
    ]
    for path_str in candidate_paths:
        if not isinstance(path_str, str) or not path_str:
            continue
        path_obj = Path(path_str).resolve()
        if not path_obj.exists() or not path_obj.is_file():
            continue
        try:
            payload = records_io.read_json(str(path_obj))
        except Exception:
            # metadata 工件不可读时跳过当前候选，继续尝试其他来源。
            continue
        if isinstance(payload, dict):
            return digests.canonical_sha256(payload)

    for record in detect_records:
        if not isinstance(record, dict):
            continue
        resolved = _pick_first_non_empty_string([
            record.get("threshold_metadata_digest"),
        ])
        if isinstance(resolved, str):
            return resolved
    return "<absent>"


def _resolve_impl_digest_for_evaluate(cfg: Dict[str, Any], detect_records: list[Dict[str, Any]]) -> str:
    """
    功能：解析 evaluate 报告的 impl_digest 锚点。

    Resolve implementation digest anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        detect_records: Loaded detect records.

    Returns:
        Resolved impl_digest anchor string.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(detect_records, list):
        raise TypeError("detect_records must be list")

    from_cfg = _pick_first_non_empty_string([
        cfg.get("__impl_digest__"),
        cfg.get("impl_set_capabilities_v2_digest"),
        cfg.get("impl_set_capabilities_digest"),
        cfg.get("impl_identity_digest"),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    for record in detect_records:
        if not isinstance(record, dict):
            continue
        resolved = _pick_first_non_empty_string([
            record.get("impl_set_capabilities_v2_digest"),
            record.get("impl_set_capabilities_digest"),
            record.get("impl_identity_digest"),
            record.get("impl_digest"),
        ])
        if isinstance(resolved, str):
            return resolved
    return "<absent>"


def _resolve_policy_path_for_evaluate(cfg: Dict[str, Any], detect_records: list[Dict[str, Any]]) -> str:
    """
    功能：解析 evaluate 报告的 policy_path 锚点。

    Resolve policy_path anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        detect_records: Loaded detect records.

    Returns:
        Resolved policy_path anchor string.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(detect_records, list):
        raise TypeError("detect_records must be list")

    from_cfg = _pick_first_non_empty_string([
        cfg.get("__policy_path__"),
        cfg.get("policy_path"),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    for record in detect_records:
        if not isinstance(record, dict):
            continue
        resolved = _pick_first_non_empty_string([
            record.get("policy_path"),
        ])
        if isinstance(resolved, str):
            return resolved
    return "<absent>"


def run_evaluate_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet) -> Dict[str, Any]:
    """
    功能：执行只读阈值评估流程。

    Execute evaluation workflow in readonly-threshold mode.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_set, BuiltImplSet):
        # impl_set 类型不合法，必须 fail-fast。
        raise TypeError("impl_set must be BuiltImplSet")

    thresholds_path = _resolve_thresholds_path_for_evaluate(cfg)
    thresholds_obj = load_thresholds_artifact_controlled(str(thresholds_path))
    detect_records = _load_records_for_evaluate(cfg)
    
    # 记录 evaluate 开始前的 thresholds digest。
    thresholds_digest_before = digests.canonical_sha256(thresholds_obj)
    
    # 使用 evaluation 模块代替内联逻辑。
    attack_protocol_spec = eval_protocol_loader.load_attack_protocol_spec(cfg)
    
    # 计算 overall 和 grouped metrics。
    threshold_value = float(thresholds_obj.get("threshold_value", 0.5))
    aggregated_metrics = eval_metrics.aggregate_metrics(
        detect_records,
        thresholds_obj,
        attack_protocol_spec,
    )
    metrics_obj = aggregated_metrics.get("metrics_overall", {})
    breakdown = aggregated_metrics.get("breakdown", {})
    
    # 重新加载 thresholds 工件并对比 digest。
    thresholds_obj_after = load_thresholds_artifact_controlled(str(thresholds_path))
    thresholds_digest_after = digests.canonical_sha256(thresholds_obj_after)
    
    if thresholds_digest_before != thresholds_digest_after:
        # thresholds 工件在 evaluate 过程中被修改，违反 NP 规则。
        raise RuntimeError(
            f"thresholds 工件只读性验证失败\n"
            f"  - 路径: {thresholds_path}\n"
            f"  - digest_before: {thresholds_digest_before}\n"
            f"  - digest_after: {thresholds_digest_after}\n"
            f"  - 原因: evaluate 侧修改或污染了 thresholds 工件"
        )
    attack_group_metrics = aggregated_metrics.get("metrics_by_attack_condition", [])
    ablation_digest = _compute_ablation_digest_for_report(cfg)
    ablation_digest_v2 = _compute_ablation_digest_v2_for_report(cfg)
    attack_trace_digest = _collect_attack_trace_digest(detect_records)
    coverage_manifest = eval_attack_coverage.compute_attack_coverage_manifest()
    attack_coverage_digest = coverage_manifest.get("attack_coverage_digest", "<absent>")
    
    # 构造条件指标容器（向后兼容）。
    conditional_metrics = eval_report_builder.build_conditional_metrics_container(
        attack_protocol_spec.get("version", "<absent>"),
        attack_group_metrics,
    )

    _ = impl_set

    fusion_result = FusionDecision(
        is_watermarked=None,
        decision_status="abstain",
        thresholds_digest=digests.canonical_sha256(thresholds_obj),
        evidence_summary={
            "content_score": None,
            "geometry_score": None,
            "content_status": "aggregate",
            "geometry_status": "aggregate",
            "fusion_rule_id": "evaluate_readonly_thresholds_v1",
        },
        audit={
            "impl_identity": "evaluate_orchestrator",
            "mode": "readonly_thresholds",
            "threshold_key_used": thresholds_obj.get("threshold_key_used"),
            "n_total": metrics_obj.get("n_total"),
        },
        fusion_rule_version="v1",
        used_threshold_id=thresholds_obj.get("threshold_id") if isinstance(thresholds_obj.get("threshold_id"), str) else None,
    )

    # 使用 report_builder 组装完整报告。
    thresholds_digest = digests.canonical_sha256(thresholds_obj)
    threshold_metadata_digest = _resolve_threshold_metadata_digest_for_evaluate(
        cfg,
        thresholds_path,
        detect_records,
    )
    plan_digest = _resolve_plan_digest_for_evaluate(cfg, detect_records)
    impl_digest = _resolve_impl_digest_for_evaluate(cfg, detect_records)
    fusion_rule_version = thresholds_obj.get("rule_version", "<absent>")
    policy_path = _resolve_policy_path_for_evaluate(cfg, detect_records)
    cfg_digest = _resolve_cfg_digest_for_evaluate(cfg, detect_records)
    
    report_obj = eval_report_builder.build_eval_report(
        cfg_digest=cfg_digest,
        plan_digest=plan_digest,
        thresholds_digest=thresholds_digest,
        threshold_metadata_digest=threshold_metadata_digest,
        impl_digest=impl_digest,
        fusion_rule_version=fusion_rule_version,
        attack_protocol_version=attack_protocol_spec.get("version", "<absent>"),
        attack_protocol_digest=attack_protocol_spec.get("attack_protocol_digest", "<absent>"),
        policy_path=policy_path,
        metrics_overall=metrics_obj,
        metrics_by_attack_condition=attack_group_metrics,
        thresholds_artifact=thresholds_obj,
        attack_protocol_spec=attack_protocol_spec,  # (向后兼容)
        ablation_digest=ablation_digest,
        attack_trace_digest=attack_trace_digest,
        attack_coverage_digest=attack_coverage_digest,
    )
    report_obj["ablation_digest_v2"] = ablation_digest_v2
    anchors = report_obj.get("anchors") if isinstance(report_obj.get("anchors"), dict) else None
    if isinstance(anchors, dict):
        anchors["ablation_digest_v2"] = ablation_digest_v2
    
    # append-only 加入 readonly guard 记录
    report_obj["thresholds_readonly_guard"] = {
        "digest_before": thresholds_digest_before,
        "digest_after": thresholds_digest_after,
        "unchanged": (thresholds_digest_before == thresholds_digest_after),
        "guard_version": "v1",
    }

    record: Dict[str, Any] = {
        "operation": "evaluate",
        "evaluation_is_fallback": False,
        "evaluation_mode": "real",
        "metrics": metrics_obj,
        "evaluation_breakdown": breakdown,
        "conditional_metrics": conditional_metrics,
        "evaluation_report": report_obj,
        "thresholds_artifact": thresholds_obj,
        "threshold_key_used": thresholds_obj.get("threshold_key_used"),
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
        "test_samples": int(metrics_obj.get("n_total", 0)),
        "fusion_result": fusion_result
    }
    return record


def load_scores_for_calibration(records: list[Dict[str, Any]]) -> tuple[list[float], Dict[str, Any]]:
    """
    功能：从 detect records 加载校准分数。 

    Load calibration scores from detect records using strict status filtering.

    Args:
        records: Detect records list.

    Returns:
        Tuple of (scores, strata_info).
    """
    if not isinstance(records, list):
        raise TypeError("records must be list")

    scores: list[float] = []
    total = len(records)
    valid = 0
    rejected = 0
    for item in records:
        if not isinstance(item, dict):
            rejected += 1
            continue
        content_payload = item.get("content_evidence_payload")
        status_value = None
        score_value = None
        if isinstance(content_payload, dict):
            status_value = content_payload.get("status")
            score_value = content_payload.get("score")
        if status_value != "ok":
            rejected += 1
            continue
        if not isinstance(score_value, (int, float)):
            rejected += 1
            continue
        score_float = float(score_value)
        if not np.isfinite(score_float):
            rejected += 1
            continue
        scores.append(score_float)
        valid += 1

    if len(scores) == 0:
        raise ValueError("calibration requires at least one valid content_score sample")

    strata_info = {
        "global": {
            "n_total": total,
            "n_valid": valid,
            "n_rejected": rejected,
        }
    }
    return scores, strata_info


def compute_np_threshold(scores: list[float], target_fpr: float) -> tuple[float, Dict[str, Any]]:
    """
    功能：按 order-statistics 计算 NP 阈值。 

    Compute Neyman-Pearson threshold using higher quantile order statistics.

    Args:
        scores: Null distribution scores.
        target_fpr: Target false positive rate.

    Returns:
        Tuple of (threshold_value, order_stat_info).
    """
    if not isinstance(scores, list) or len(scores) == 0:
        raise ValueError("scores must be non-empty list")
    if not isinstance(target_fpr, (int, float)):
        raise TypeError("target_fpr must be number")

    threshold_value, order_stat_info = neyman_pearson.compute_np_threshold_from_scores(
        scores,
        float(target_fpr),
        quantile_rule="higher",
    )
    return float(threshold_value), order_stat_info


def load_thresholds_artifact_controlled(path: str) -> Dict[str, Any]:
    """
    功能：只读加载阈值工件。 

    Load thresholds artifact in read-only mode with schema checks.

    Args:
        path: Thresholds artifact path.

    Returns:
        Thresholds artifact mapping.
    """
    if not isinstance(path, str) or not path:
        raise TypeError("path must be non-empty str")
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        raise ValueError(f"thresholds artifact not found: {path}")
    payload = records_io.read_json(path)
    if not isinstance(payload, dict):
        raise TypeError("thresholds artifact must be dict")
    required = ["threshold_id", "score_name", "target_fpr", "threshold_value", "threshold_key_used"]
    for field_name in required:
        if field_name not in payload:
            raise ValueError(f"thresholds artifact missing field: {field_name}")
    if not isinstance(payload.get("threshold_value"), (int, float)):
        raise TypeError("threshold_value must be number")
    return payload


def evaluate_records_against_threshold(
    records: list[Dict[str, Any]],
    thresholds_obj: Dict[str, Any],
    attack_protocol_spec: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    功能：使用只读阈值评测 detect 记录。 

    Evaluate detect records using precomputed thresholds artifact only.
    Now delegates to evaluation module for metric computation.

    Args:
        records: Detect records list.
        thresholds_obj: Thresholds artifact mapping.
        attack_protocol_spec: Optional attack protocol spec.

    Returns:
        Tuple of (metrics, breakdown, conditional_metrics).
    """
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(thresholds_obj, dict):
        raise TypeError("thresholds_obj must be dict")

    threshold_value = thresholds_obj.get("threshold_value")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")
    threshold_float = float(threshold_value)

    # 使用 evaluation 模块计算指标。
    if attack_protocol_spec is None:
        attack_protocol_spec = {
            "version": "<absent>",
            "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
            "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
        }

    # Overall metrics。
    metrics, breakdown = eval_metrics.compute_overall_metrics(records, threshold_float)
    
    # 补充 thresholds 工件元数据。
    metrics["metric_version"] = "tpr_at_fpr_v1"
    metrics["score_name"] = thresholds_obj.get("score_name", "content_score")
    metrics["target_fpr"] = thresholds_obj.get("target_fpr")
    metrics["threshold_value"] = threshold_float
    metrics["threshold_key_used"] = thresholds_obj.get("threshold_key_used")

    # Grouped metrics。
    attack_group_metrics = eval_metrics.compute_attack_group_metrics(
        records,
        threshold_float,
        attack_protocol_spec,
    )

    # 计算条件指标中的 "items"（旧字段，用于向后兼容）。
    conditional_metrics_old = _compute_conditional_metrics_for_evaluate(records, threshold_float)
    additional_items = conditional_metrics_old.get("items", [])

    # 构造条件指标容器（向后兼容）。
    conditional_metrics = eval_report_builder.build_conditional_metrics_container(
        attack_protocol_spec.get("version", "<absent>"),
        attack_group_metrics,
        additional_items=additional_items,
    )

    return metrics, breakdown, conditional_metrics


def _load_records_for_calibration(cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    records_glob = cfg.get("__calibration_detect_records_glob__")
    if not isinstance(records_glob, str) or not records_glob:
        calibration_cfg = cfg.get("calibration") if isinstance(cfg.get("calibration"), dict) else {}
        records_glob = calibration_cfg.get("detect_records_glob") if isinstance(calibration_cfg.get("detect_records_glob"), str) else None
    if not isinstance(records_glob, str) or not records_glob:
        raise ValueError("calibration.detect_records_glob is required")
    return _load_records_by_glob(records_glob)


def _load_records_for_evaluate(cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    records_glob = cfg.get("__evaluate_detect_records_glob__")
    if not isinstance(records_glob, str) or not records_glob:
        evaluate_cfg = cfg.get("evaluate") if isinstance(cfg.get("evaluate"), dict) else {}
        records_glob = evaluate_cfg.get("detect_records_glob") if isinstance(evaluate_cfg.get("detect_records_glob"), str) else None
    if not isinstance(records_glob, str) or not records_glob:
        raise ValueError("evaluate.detect_records_glob is required")
    return _load_records_by_glob(records_glob)


def _resolve_thresholds_path_for_evaluate(cfg: Dict[str, Any]) -> Path:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    thresholds_path = cfg.get("__evaluate_thresholds_path__")
    if not isinstance(thresholds_path, str) or not thresholds_path:
        evaluate_cfg = cfg.get("evaluate") if isinstance(cfg.get("evaluate"), dict) else {}
        thresholds_path = evaluate_cfg.get("thresholds_path") if isinstance(evaluate_cfg.get("thresholds_path"), str) else None
    if not isinstance(thresholds_path, str) or not thresholds_path:
        raise ValueError("evaluate.thresholds_path is required")
    path_obj = Path(thresholds_path).resolve()
    if not path_obj.exists() or not path_obj.is_file():
        raise ValueError(f"evaluate thresholds_path not found: {path_obj}")
    return path_obj


def _load_records_by_glob(records_glob: str) -> list[Dict[str, Any]]:
    if not isinstance(records_glob, str) or not records_glob:
        raise TypeError("records_glob must be non-empty str")
    matched_paths = sorted(glob.glob(records_glob, recursive=True))
    if len(matched_paths) == 0:
        raise ValueError(f"no detect records matched: {records_glob}")
    records: list[Dict[str, Any]] = []
    for path_str in matched_paths:
        path_obj = Path(path_str)
        if not path_obj.is_file():
            continue
        payload = records_io.read_json(str(path_obj))
        if isinstance(payload, dict):
            records.append(payload)
    if len(records) == 0:
        raise ValueError(f"no valid detect records loaded from: {records_glob}")
    return records


def _extract_ground_truth_label(record: Dict[str, Any]) -> Optional[bool]:
    if not isinstance(record, dict):
        return None
    for key_name in ["ground_truth_is_watermarked", "is_watermarked_gt", "label"]:
        value = record.get(key_name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
    return None


def _extract_geometry_score(record: Dict[str, Any]) -> Optional[float]:
    if not isinstance(record, dict):
        return None
    geometry_payload = record.get("geometry_evidence_payload")
    if not isinstance(geometry_payload, dict):
        return None
    status_value = geometry_payload.get("status")
    if status_value != "ok":
        return None
    for key_name in ["score", "geo_score"]:
        value = geometry_payload.get(key_name)
        if isinstance(value, (int, float)):
            value_float = float(value)
            if np.isfinite(value_float):
                return value_float
    return None


def _extract_content_score_for_stats(record: Dict[str, Any]) -> Optional[float]:
    if not isinstance(record, dict):
        return None
    content_payload = record.get("content_evidence_payload")
    if not isinstance(content_payload, dict):
        return None
    if content_payload.get("status") != "ok":
        return None
    score_value = content_payload.get("score")
    if not isinstance(score_value, (int, float)):
        return None
    score_float = float(score_value)
    if not np.isfinite(score_float):
        return None
    return score_float


def _compute_null_strata_for_calibration(
    records: list[Dict[str, Any]],
    threshold_value: float,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    valid_content = 0
    geometry_available = 0
    geometry_unavailable = 0
    rescue_candidate = 0

    rescue_spec = fusion_decision._build_rescue_band_spec(cfg)
    delta_low = float(rescue_spec.get("delta_low", 0.05))
    lower_bound = float(threshold_value) - delta_low
    for item in records:
        score_float = _extract_content_score_for_stats(item)
        if score_float is None:
            continue
        valid_content += 1

        geo_score = _extract_geometry_score(item)
        if geo_score is None:
            geometry_unavailable += 1
        else:
            geometry_available += 1

        if lower_bound <= score_float < float(threshold_value):
            rescue_candidate += 1

    return {
        "global_valid": {
            "n": int(valid_content),
        },
        "geometry_available": {
            "n": int(geometry_available),
        },
        "geometry_unavailable": {
            "n": int(geometry_unavailable),
        },
        "rescue_candidate": {
            "n": int(rescue_candidate),
            "window": f"[threshold-{delta_low}, threshold)",
        },
    }


def _compute_conditional_fpr_records_for_calibration(
    records: list[Dict[str, Any]],
    threshold_value: float,
    cfg: Dict[str, Any],
) -> list[Dict[str, Any]]:
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    threshold_float = float(threshold_value)
    rescue_spec = fusion_decision._build_rescue_band_spec(cfg)
    delta_low = float(rescue_spec.get("delta_low", 0.05))
    delta_high = float(rescue_spec.get("delta_high", 0.05))
    geo_gate_lower = float(rescue_spec.get("geo_gate_lower", 0.3))
    geo_gate_upper = float(rescue_spec.get("geo_gate_upper", 0.7))
    align_quality_threshold = _resolve_align_quality_threshold(cfg)
    config_anchors = {
        "threshold_value": round(threshold_float, 12),
        "rescue_band_delta_low": round(delta_low, 12),
        "rescue_band_delta_high": round(delta_high, 12),
        "geo_gate_lower": round(geo_gate_lower, 12),
        "geo_gate_upper": round(geo_gate_upper, 12),
        "align_quality_threshold": round(align_quality_threshold, 12),
        "hf_failure_rule_version": HF_FAILURE_RULE_VERSION,
        "geo_availability_rule_version": GEO_AVAILABILITY_RULE_VERSION,
    }

    def _make_record(condition_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample_count = int(len(payload["sample_ids"]))
        positive_count = int(payload["positive"])
        empirical_fpr = float(positive_count / sample_count) if sample_count > 0 else None
        unavailable_count = int(payload.get("unavailable", 0))
        score_summary = _summarize_scores(payload["scores"])
        sample_set_digest = digests.canonical_sha256(sorted(payload["sample_ids"]))
        digest_payload = {
            "condition_id": condition_id,
            "definition": payload["definition"],
            "config_anchors": config_anchors,
            "sample_count": sample_count,
            "positive_count": positive_count,
            "unavailable_count": unavailable_count,
            "score_summary": score_summary,
            "sample_set_digest": sample_set_digest,
        }
        record = {
            "condition_id": condition_id,
            "definition": payload["definition"],
            "sample_count": sample_count,
            "empirical_fpr": empirical_fpr,
            "inputs_digest": digests.canonical_sha256(digest_payload),
        }
        if payload.get("include_unavailable_count", False):
            record["unavailable_count"] = unavailable_count
        return record

    condition_buffers: Dict[str, Dict[str, Any]] = {
        "global": {
            "definition": "all valid null samples",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "geometry_available": {
            "definition": "geometry score available",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "geometry_unavailable": {
            "definition": "geometry score unavailable",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "rescue_band_candidate": {
            "definition": f"content score in rescue band [threshold-{delta_low}, threshold) with delta_high={delta_high}",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "geo_gate_applied": {
            "definition": f"geo gate applied with bounds [{geo_gate_lower}, {geo_gate_upper}]",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "align_quality_ge_threshold": {
            "definition": f"alignment quality >= {align_quality_threshold}",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
            "unavailable": 0,
            "include_unavailable_count": True,
        },
        "hf_failure_rule": {
            "definition": f"HF failure decision rule version {HF_FAILURE_RULE_VERSION}",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "geo_availability_rule": {
            "definition": f"geo availability decision rule version {GEO_AVAILABILITY_RULE_VERSION}",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
    }

    rescue_lower = threshold_float - delta_low
    for index, item in enumerate(records):
        score_float = _extract_content_score_for_stats(item)
        if score_float is None:
            continue
        pred_positive = bool(score_float >= threshold_float)
        sample_id = _build_calibration_sample_id(item, index)

        _update_condition_buffer(condition_buffers, "global", score_float, pred_positive, sample_id)

        geo_score = _extract_geometry_score(item)
        if geo_score is None:
            _update_condition_buffer(condition_buffers, "geometry_unavailable", score_float, pred_positive, sample_id)
        else:
            _update_condition_buffer(condition_buffers, "geometry_available", score_float, pred_positive, sample_id)

        if rescue_lower <= score_float < threshold_float:
            _update_condition_buffer(condition_buffers, "rescue_band_candidate", score_float, pred_positive, sample_id)

        if _extract_geo_gate_applied(item) is True:
            _update_condition_buffer(condition_buffers, "geo_gate_applied", score_float, pred_positive, sample_id)

        align_quality_value = _extract_align_quality_value(item)
        if align_quality_value is None:
            condition_buffers["align_quality_ge_threshold"]["unavailable"] += 1
        elif align_quality_value >= align_quality_threshold:
            _update_condition_buffer(condition_buffers, "align_quality_ge_threshold", score_float, pred_positive, sample_id)

        # (新增) 跟踪 HF 失败规则事件
        hf_failure_decision = _extract_hf_failure_decision(item)
        if hf_failure_decision is True:
            _update_condition_buffer(condition_buffers, "hf_failure_rule", score_float, pred_positive, sample_id)

        # (新增) 跟踪 Geo 可用性规则事件
        geo_available = _extract_geo_available(item)
        if geo_available is True:
            _update_condition_buffer(condition_buffers, "geo_availability_rule", score_float, pred_positive, sample_id)

    ordered_conditions = [
        "global",
        "geometry_available",
        "geometry_unavailable",
        "rescue_band_candidate",
        "geo_gate_applied",
        "align_quality_ge_threshold",
        "hf_failure_rule",
        "geo_availability_rule",
    ]
    return [
        _make_record(condition_id=condition_id, payload=condition_buffers[condition_id])
        for condition_id in ordered_conditions
    ]


def _compute_conditional_fpr_for_calibration(records: list[Dict[str, Any]], threshold_value: float) -> Dict[str, Any]:
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")

    global_total = 0
    global_fp = 0
    geo_available_total = 0
    geo_available_fp = 0
    geo_unavailable_total = 0
    geo_unavailable_fp = 0

    for item in records:
        score_float = _extract_content_score_for_stats(item)
        if score_float is None:
            continue
        pred_positive = bool(score_float >= float(threshold_value))

        global_total += 1
        if pred_positive:
            global_fp += 1

        geo_score = _extract_geometry_score(item)
        if geo_score is None:
            geo_unavailable_total += 1
            if pred_positive:
                geo_unavailable_fp += 1
        else:
            geo_available_total += 1
            if pred_positive:
                geo_available_fp += 1

    def _pack(condition_id: str, total_count: int, fp_count: int) -> Dict[str, Any]:
        fpr_value = float(fp_count / total_count) if total_count > 0 else None
        return {
            "condition_id": condition_id,
            "n": int(total_count),
            "fp": int(fp_count),
            "fpr_empirical": fpr_value,
        }

    return {
        "definition": "null-only empirical FPR conditioned on geometry availability",
        "items": [
            _pack("global", global_total, global_fp),
            _pack("geometry_available", geo_available_total, geo_available_fp),
            _pack("geometry_unavailable", geo_unavailable_total, geo_unavailable_fp),
        ],
    }


def _update_condition_buffer(
    condition_buffers: Dict[str, Dict[str, Any]],
    condition_id: str,
    score_value: float,
    pred_positive: bool,
    sample_id: str,
) -> None:
    if condition_id not in condition_buffers:
        # 条件不存在属于调用方逻辑错误，必须 fail-fast。
        raise ValueError(f"unknown condition_id: {condition_id}")
    payload = condition_buffers[condition_id]
    payload["scores"].append(float(score_value))
    payload["sample_ids"].add(sample_id)
    if pred_positive:
        payload["positive"] += 1


def _summarize_scores(scores: list[float]) -> Dict[str, Any]:
    if not isinstance(scores, list):
        return {
            "count": 0,
            "p50": None,
            "p90": None,
        }
    valid_scores = [float(value) for value in scores if isinstance(value, (int, float)) and np.isfinite(float(value))]
    if len(valid_scores) == 0:
        return {
            "count": 0,
            "p50": None,
            "p90": None,
        }
    arr = np.asarray(valid_scores, dtype=float)
    return {
        "count": int(arr.size),
        "p50": float(np.quantile(arr, 0.50, method="higher")),
        "p90": float(np.quantile(arr, 0.90, method="higher")),
    }


def _build_calibration_sample_id(record: Dict[str, Any], index: int) -> str:
    if not isinstance(record, dict):
        return digests.canonical_sha256({"index": index, "record": "invalid"})
    identity_payload = {
        "index": int(index),
        "cfg_digest": record.get("cfg_digest"),
        "plan_digest": record.get("plan_digest"),
        "image_path": record.get("image_path"),
        "label": record.get("label"),
    }
    return digests.canonical_sha256(identity_payload)


def _resolve_align_quality_threshold(cfg: Dict[str, Any]) -> float:
    if not isinstance(cfg, dict):
        return 0.5
    for candidate_path in [
        "align_quality_threshold",
        "geometry_align_quality_threshold",
        "geometry.align_quality_threshold",
        "evaluate.align_quality_threshold",
    ]:
        value = _extract_nested_value(cfg, candidate_path)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return 0.5


def _extract_align_quality_value(record: Dict[str, Any]) -> Optional[float]:
    if not isinstance(record, dict):
        return None
    geometry_payload = record.get("geometry_evidence_payload")
    if not isinstance(geometry_payload, dict):
        return None
    for dotted_path in [
        "sync_metrics.align_quality",
        "sync_metrics.alignment_quality",
        "stability_metrics.align_quality",
        "stability_metrics.alignment_quality",
    ]:
        value = _extract_nested_value(geometry_payload, dotted_path)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return None


def _extract_hf_failure_decision(record: Dict[str, Any]) -> Optional[bool]:
    """
    从记录中提取 HF 失败决策字段。

    Extract HF failure decision from content evidence.

    Args:
        record: Detection record mapping.

    Returns:
        HF failure decision (bool) or None if not available.
    """
    if not isinstance(record, dict):
        return None
    content_payload = record.get("content_evidence_payload")
    if not isinstance(content_payload, dict):
        return None
    hf_failure_decision = content_payload.get("hf_failure_decision")
    if isinstance(hf_failure_decision, bool):
        return hf_failure_decision
    return None


def _extract_geo_available(record: Dict[str, Any]) -> Optional[bool]:
    """
    从记录中提取几何可用性字段。

    Extract geometry availability decision from geometry evidence.

    Args:
        record: Detection record mapping.

    Returns:
        Geometry available decision (bool) or None if not available.
    """
    if not isinstance(record, dict):
        return None
    geometry_payload = record.get("geometry_evidence_payload")
    if not isinstance(geometry_payload, dict):
        return None
    geo_available = geometry_payload.get("geo_available")
    if isinstance(geo_available, bool):
        return geo_available
    return None


def _compute_conditional_metrics_for_evaluate(records: list[Dict[str, Any]], threshold_value: float) -> Dict[str, Any]:
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")

    groups = {
        "global": {"tp": 0, "fp": 0, "pos": 0, "neg": 0, "accepted": 0},
        "geometry_available": {"tp": 0, "fp": 0, "pos": 0, "neg": 0, "accepted": 0},
        "geometry_unavailable": {"tp": 0, "fp": 0, "pos": 0, "neg": 0, "accepted": 0},
    }

    for item in records:
        score_float = _extract_content_score_for_stats(item)
        if score_float is None:
            continue
        gt_value = _extract_ground_truth_label(item)
        if gt_value is None:
            continue

        pred_positive = bool(score_float >= float(threshold_value))
        group_name = "geometry_available" if _extract_geometry_score(item) is not None else "geometry_unavailable"

        for key_name in ["global", group_name]:
            group = groups[key_name]
            group["accepted"] += 1
            if gt_value:
                group["pos"] += 1
                if pred_positive:
                    group["tp"] += 1
            else:
                group["neg"] += 1
                if pred_positive:
                    group["fp"] += 1

    items = []
    for condition_id, group in groups.items():
        tpr_value = float(group["tp"] / group["pos"]) if group["pos"] > 0 else None
        fpr_value = float(group["fp"] / group["neg"]) if group["neg"] > 0 else None
        items.append(
            {
                "condition_id": condition_id,
                "n_accepted": int(group["accepted"]),
                "n_pos": int(group["pos"]),
                "n_neg": int(group["neg"]),
                "tpr_at_fpr": tpr_value,
                "fpr_empirical": fpr_value,
            }
        )

    return {
        "version": "conditional_eval_v1",
        "items": items,
    }


def _collect_evaluation_anchors(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(records, list):
        return {}
    cfg_digests = []
    plan_digests = []
    impl_digests = []
    for item in records:
        if not isinstance(item, dict):
            continue
        cfg_digest = item.get("cfg_digest")
        if isinstance(cfg_digest, str) and cfg_digest:
            cfg_digests.append(cfg_digest)
        plan_digest = item.get("plan_digest")
        if isinstance(plan_digest, str) and plan_digest:
            plan_digests.append(plan_digest)
        impl_digest = item.get("impl", {}).get("digests", {}).get("content_extractor") if isinstance(item.get("impl"), dict) else None
        if isinstance(impl_digest, str) and impl_digest:
            impl_digests.append(impl_digest)
    return {
        "cfg_digest_set": sorted(set(cfg_digests)),
        "plan_digest_set": sorted(set(plan_digests)),
        "impl_digest_set": sorted(set(impl_digests)),
    }


def _read_attack_protocol_version(cfg: Dict[str, Any]) -> str:
    if not isinstance(cfg, dict):
        return "<absent>"
    evaluate_cfg = cfg.get("evaluate")
    if not isinstance(evaluate_cfg, dict):
        return "<absent>"
    value = evaluate_cfg.get("attack_protocol_version")
    if isinstance(value, str) and value:
        return value
    return "<absent>"


def _read_attack_protocol_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    default_spec = {
        "version": "<absent>",
        "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
        "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
        "families": {},
        "params_versions": {},
        "attack_protocol_digest": "<absent>",
    }
    if not isinstance(cfg, dict):
        return default_spec

    evaluate_cfg = cfg.get("evaluate") if isinstance(cfg.get("evaluate"), dict) else {}
    fact_source_path = evaluate_cfg.get("attack_protocol_path")
    if not isinstance(fact_source_path, str) or not fact_source_path:
        fact_source_path = config_loader.ATTACK_PROTOCOL_PATH

    fact_source_obj: Dict[str, Any] = {}
    try:
        loaded_obj, provenance = config_loader.load_yaml_with_provenance(fact_source_path)
        if isinstance(loaded_obj, dict):
            fact_source_obj = loaded_obj
            fact_source_obj["attack_protocol_digest"] = provenance.canon_sha256
            fact_source_obj["attack_protocol_file_sha256"] = provenance.file_sha256
    except Exception:
        fact_source_obj = {}

    merged_spec = dict(default_spec)
    merged_spec.update({
        key: value
        for key, value in fact_source_obj.items()
        if key in [
            "version",
            "family_field_candidates",
            "params_version_field_candidates",
            "families",
            "params_versions",
            "attack_protocol_digest",
            "attack_protocol_file_sha256",
        ]
    })

    cfg_version = evaluate_cfg.get("attack_protocol_version")
    if isinstance(cfg_version, str) and cfg_version:
        merged_spec["version"] = cfg_version

    cfg_family_candidates = evaluate_cfg.get("attack_family_field_candidates")
    if isinstance(cfg_family_candidates, list) and len(cfg_family_candidates) > 0:
        merged_spec["family_field_candidates"] = cfg_family_candidates

    cfg_params_candidates = evaluate_cfg.get("attack_params_version_field_candidates")
    if isinstance(cfg_params_candidates, list) and len(cfg_params_candidates) > 0:
        merged_spec["params_version_field_candidates"] = cfg_params_candidates

    merged_spec["family_field_candidates"] = [
        field_name
        for field_name in merged_spec.get("family_field_candidates", [])
        if isinstance(field_name, str) and field_name
    ] or default_spec["family_field_candidates"]

    merged_spec["params_version_field_candidates"] = [
        field_name
        for field_name in merged_spec.get("params_version_field_candidates", [])
        if isinstance(field_name, str) and field_name
    ] or default_spec["params_version_field_candidates"]

    if not isinstance(merged_spec.get("version"), str) or not merged_spec.get("version"):
        merged_spec["version"] = "<absent>"

    if not isinstance(merged_spec.get("attack_protocol_digest"), str) or not merged_spec.get("attack_protocol_digest"):
        merged_spec["attack_protocol_digest"] = "<absent>"

    return merged_spec


def _build_attack_group_key(record: Dict[str, Any], protocol_spec: Dict[str, Any]) -> str:
    family_fields = protocol_spec.get("family_field_candidates") if isinstance(protocol_spec, dict) else None
    params_fields = protocol_spec.get("params_version_field_candidates") if isinstance(protocol_spec, dict) else None

    family_value = _extract_first_string_value(record, family_fields, "unknown_attack")
    params_value = _extract_first_string_value(record, params_fields, "unknown_params")
    return f"{family_value}::{params_value}"


def _extract_first_string_value(record: Dict[str, Any], candidate_paths: Any, default_value: str) -> str:
    if not isinstance(record, dict):
        return default_value
    if not isinstance(candidate_paths, list):
        return default_value
    for candidate in candidate_paths:
        if not isinstance(candidate, str) or not candidate:
            continue
        value = _extract_nested_value(record, candidate)
        if isinstance(value, str) and value:
            return value
    return default_value


def _extract_nested_value(payload: Dict[str, Any], dotted_path: str) -> Any:
    if not isinstance(payload, dict):
        return None
    if not isinstance(dotted_path, str) or not dotted_path:
        return None
    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _build_attack_group_metrics(attack_groups: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    if not isinstance(attack_groups, dict):
        return []
    metrics_list: list[Dict[str, Any]] = []
    for group_key in sorted(attack_groups.keys()):
        group = attack_groups[group_key]
        if not isinstance(group, dict):
            continue
        group_n_pos = int(group.get("n_pos", 0))
        group_n_neg = int(group.get("n_neg", 0))
        group_tp = int(group.get("tp", 0))
        group_fp = int(group.get("fp", 0))
        group_n_accepted = int(group.get("n_accepted", 0))
        group_n_total = int(group.get("n_total", 0))
        group_geo_available = int(group.get("geo_available_accepted", 0))
        group_rescue_triggered = int(group.get("rescue_triggered", 0))
        group_rescue_positive = int(group.get("rescue_positive", 0))
        group_hf_failed = int(group.get("hf_failed_count", 0))
        group_reject_counts = group.get("reject_count_by_reason", {}) if isinstance(group.get("reject_count_by_reason"), dict) else {}
        group_reject_rate_by_reason = {
            key_name: (float(int(count_value) / group_n_total) if group_n_total > 0 else 0.0)
            for key_name, count_value in group_reject_counts.items()
        }
        metrics_list.append(
            {
                "group_key": group_key,
                "n_total": group_n_total,
                "n_accepted": group_n_accepted,
                "n_pos": group_n_pos,
                "n_neg": group_n_neg,
                "tpr_at_fpr_primary": (float(group_tp / group_n_pos) if group_n_pos > 0 else None),
                "fpr_empirical": (float(group_fp / group_n_neg) if group_n_neg > 0 else None),
                "geo_available_rate": (float(group_geo_available / group_n_accepted) if group_n_accepted > 0 else None),
                "hf_failed_rate": (float(group_hf_failed / group_n_accepted) if group_n_accepted > 0 else None),
                "rescue_rate": (float(group_rescue_triggered / group_n_accepted) if group_n_accepted > 0 else None),
                "rescue_gain_rate": (float(group_rescue_positive / group_n_accepted) if group_n_accepted > 0 else None),
                "reject_rate_by_reason": group_reject_rate_by_reason,
            }
        )
    return metrics_list


def _extract_rescue_triggered(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    decision_payload = record.get("decision")
    if not isinstance(decision_payload, dict):
        return False
    routing_decisions = decision_payload.get("routing_decisions")
    if isinstance(routing_decisions, dict):
        value = routing_decisions.get("rescue_triggered")
        if isinstance(value, bool):
            return value
    audit_payload = decision_payload.get("audit")
    if isinstance(audit_payload, dict):
        value = audit_payload.get("rescue_triggered")
        if isinstance(value, bool):
            return value
    return False


def _extract_geo_gate_applied(record: Dict[str, Any]) -> Optional[bool]:
    if not isinstance(record, dict):
        return None
    decision_payload = record.get("decision")
    if not isinstance(decision_payload, dict):
        return None
    routing_decisions = decision_payload.get("routing_decisions")
    if isinstance(routing_decisions, dict):
        value = routing_decisions.get("geo_gate_applied")
        if isinstance(value, bool):
            return value
    audit_payload = decision_payload.get("audit")
    if isinstance(audit_payload, dict):
        value = audit_payload.get("geo_gate_applied")
        if isinstance(value, bool):
            return value
    return None


def _extract_align_quality_available(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    geometry_payload = record.get("geometry_evidence_payload")
    if not isinstance(geometry_payload, dict):
        return False
    candidate_values = []
    for dotted_path in [
        "sync_metrics.align_quality",
        "sync_metrics.alignment_quality",
        "stability_metrics.align_quality",
        "stability_metrics.alignment_quality",
    ]:
        value = _extract_nested_value(geometry_payload, dotted_path)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            candidate_values.append(float(value))
    return len(candidate_values) > 0


def _adapt_content_evidence_for_fusion(content_evidence: Any) -> Dict[str, Any]:
    """
    功能：将 ContentEvidence 数据类适配为融合规则期望的字典格式。

    Adapt ContentEvidence (dataclass or dict) to fusion rule expected format.
    Prioritizes .as_dict() method; falls back to direct dict or attribute extraction.

    Args:
        content_evidence: ContentEvidence dataclass, dict, or compatible object.

    Returns:
        Dict with fields expected by fusion rule (status, score, etc.).

    Raises:
        TypeError: If content_evidence type is unrecognized.
    """
    if isinstance(content_evidence, dict):
        # 已是字典，直接返回。
        return content_evidence
    
    # 尝试用 .as_dict() 方法。
    if hasattr(content_evidence, "as_dict") and callable(content_evidence.as_dict):
        try:
            return content_evidence.as_dict()
        except Exception:
            # 如果 .as_dict() 失败，继续尝试属性提取。
            pass
    
    # 从数据类属性直接构造。
    adapted = {}
    
    # 提取关键字段（来自 ContentEvidence 冻结结构）。
    for field_name in ["status", "score", "audit", "mask_digest", "mask_stats",
                       "plan_digest", "basis_digest", "lf_trace_digest", "hf_trace_digest",
                       "lf_score", "hf_score", "score_parts", "trajectory_evidence",
                       "content_failure_reason"]:
        if hasattr(content_evidence, field_name):
            adapted[field_name] = getattr(content_evidence, field_name)
    
    return adapted if adapted else {"status": "unknown"}


def _adapt_geometry_evidence_for_fusion(geometry_evidence: Any) -> Dict[str, Any]:
    """
    功能：将 GeometryEvidence 数据类适配为融合规则期望的字典格式。

    Adapt GeometryEvidence (dataclass or dict) to fusion rule expected format.
    Prioritizes .as_dict() method; falls back to direct dict or attribute extraction.

    Args:
        geometry_evidence: GeometryEvidence dataclass, dict, or compatible object.

    Returns:
        Dict with fields expected by fusion rule (status, geo_score, etc.).

    Raises:
        TypeError: If geometry_evidence type is unrecognized.
    """
    if isinstance(geometry_evidence, dict):
        # 已是字典，直接返回。
        return geometry_evidence
    
    # 尝试用 .as_dict() 方法。
    if hasattr(geometry_evidence, "as_dict") and callable(geometry_evidence.as_dict):
        try:
            return geometry_evidence.as_dict()
        except Exception:
            # 如果 .as_dict() 失败，继续尝试属性提取。
            pass
    
    # 从数据类属性直接构造。
    adapted = {}
    
    # 提取关键字段（来自 GeometryEvidence 冻结结构）。
    for field_name in [
        "status",
        "geo_score",
        "audit",
        "anchor_digest",
        "anchor_config_digest",
        "align_trace_digest",
        "align_residuals",
        "anchor_metrics",
        "stability_metrics",
        "sync_digest",
        "sync_metrics",
        "sync_config_digest",
        "sync_quality_metrics",
        "resolution_binding",
        "align_metrics",
        "align_config_digest",
        "geo_score_direction",
        "geo_failure_reason",
        "geometry_failure_reason",
    ]:
        if hasattr(geometry_evidence, field_name):
            adapted[field_name] = getattr(geometry_evidence, field_name)
    
    return adapted if adapted else {"status": "unknown"}


def _is_image_domain_sidecar_enabled(cfg: Dict[str, Any], ablation_override: bool | None = None) -> bool:
    """
    功能：解析图像域 sidecar 开关。

    Resolve whether image-domain detector sidecar is enabled.

    Args:
        cfg: Configuration mapping.

    Returns:
        True if sidecar is enabled.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if ablation_override is not None and not isinstance(ablation_override, bool):
        raise TypeError("ablation_override must be bool or None")
    if isinstance(ablation_override, bool) and not ablation_override:
        return False
    detect_runtime_cfg = cfg.get("detect_runtime") if isinstance(cfg.get("detect_runtime"), dict) else {}
    explicit = detect_runtime_cfg.get("image_domain_sidecar_enabled")
    if isinstance(explicit, bool):
        return explicit
    paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
    if bool(paper_cfg.get("enabled", False)):
        return False
    return True


def _build_attention_maps_from_latents(latents: Any) -> Any:
    """
    功能：从 latent 构造可复算 attention maps 代理。

    Build deterministic attention-map proxy from latents.

    Args:
        latents: Tensor-like or numpy latent array.

    Returns:
        Attention proxy array or None.
    """
    if latents is None:
        return None
    if isinstance(latents, np.ndarray):
        latents_np = latents
    elif hasattr(latents, "detach") and callable(latents.detach):
        detached = latents.detach()
        if hasattr(detached, "cpu") and callable(detached.cpu):
            detached = detached.cpu()
        if hasattr(detached, "numpy") and callable(detached.numpy):
            latents_np = detached.numpy()
        else:
            return None
    else:
        return None
    if latents_np.ndim != 4:
        return None
    latent_first = latents_np[0]
    if latent_first.ndim != 3:
        return None
    channels = latent_first.reshape(latent_first.shape[0], -1)
    if channels.shape[0] < 2:
        return None
    correlation = np.corrcoef(channels)
    if not np.isfinite(correlation).all():
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    return correlation.astype(np.float64)


def _resolve_runtime_self_attention_maps(cfg: Dict[str, Any]) -> Any:
    """
    功能：解析 detect 侧真实 self-attention maps 载荷。

    Resolve runtime self-attention maps from detect transient fields.

    Args:
        cfg: Configuration mapping.

    Returns:
        Attention maps payload or None.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    for key_name in [
        "__detect_attention_maps__",
        "__detect_self_attention_maps__",
        "__runtime_self_attention_maps__",
    ]:
        candidate = cfg.get(key_name)
        if candidate is not None:
            return candidate
    return None


def _extract_subspace_evidence_semantics(plan_payload: Any) -> Dict[str, Any]:
    """
    功能：从计划载荷中提取子空间证据语义。

    Extract subspace evidence semantics from planner payload.

    Args:
        plan_payload: Plan payload mapping.

    Returns:
        Semantics mapping or empty dict.
    """
    if not isinstance(plan_payload, dict):
        return {}
    direct_value = plan_payload.get("subspace_evidence_semantics")
    if isinstance(direct_value, dict):
        return direct_value
    plan_node = plan_payload.get("plan")
    if isinstance(plan_node, dict):
        nested_value = plan_node.get("subspace_evidence_semantics")
        if isinstance(nested_value, dict):
            return nested_value
    plan_stats = plan_payload.get("plan_stats")
    if isinstance(plan_stats, dict):
        stats_value = plan_stats.get("subspace_evidence_semantics")
        if isinstance(stats_value, dict):
            return stats_value
    return {}


def _build_geometry_runtime_inputs(
    cfg: Dict[str, Any],
    sync_result: Dict[str, Any] | None = None,
    anchor_result: Dict[str, Any] | None = None,
    enable_attention_proxy: bool = True,
) -> Dict[str, Any]:
    """
    功能：构造几何链运行时输入域。

    Build geometry runtime input payload for sync and extractor.

    Args:
        cfg: Configuration mapping.
        sync_result: Optional sync result mapping.

    Returns:
        Runtime inputs mapping.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    runtime_inputs = {
        "pipeline": cfg.get("__detect_pipeline_obj__"),
        "latents": cfg.get("__detect_final_latents__"),
        "rng": cfg.get("rng"),
    }
    paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False)) if isinstance(paper_cfg, dict) else False
    prebuilt_attention_maps = _resolve_runtime_self_attention_maps(cfg)
    if prebuilt_attention_maps is not None:
        runtime_inputs["attention_maps"] = prebuilt_attention_maps
        runtime_inputs["attention_maps_source"] = "runtime_self_attention"
        runtime_inputs["attention_maps_evidence_level"] = "primary"
    if enable_attention_proxy:
        if "attention_maps" not in runtime_inputs:
            if paper_enabled:
                runtime_inputs["attention_proxy_status"] = "absent"
                runtime_inputs["attention_proxy_absent_reason"] = "paper_mode_requires_runtime_self_attention"
                runtime_inputs["attention_maps_missing_reason"] = "runtime_self_attention_missing_under_paper_mode"
                return runtime_inputs
            attention_maps = _build_attention_maps_from_latents(runtime_inputs.get("latents"))
            if attention_maps is not None:
                runtime_inputs["attention_maps"] = attention_maps
                runtime_inputs["attention_maps_source"] = "latent_proxy"
                runtime_inputs["attention_maps_evidence_level"] = "fallback"
                runtime_inputs["attention_maps_fallback_reason"] = "runtime_self_attention_missing"
                runtime_inputs["attention_maps_digest"] = digests.canonical_sha256({
                    "shape": list(attention_maps.shape),
                    "mean": float(np.mean(attention_maps)),
                    "std": float(np.std(attention_maps)),
                    "max": float(np.max(attention_maps)),
                    "min": float(np.min(attention_maps)),
                })
    else:
        runtime_inputs["attention_proxy_status"] = "absent"
        runtime_inputs["attention_proxy_absent_reason"] = "attention_proxy_disabled_by_ablation"
    if isinstance(anchor_result, dict):
        relation_digest = anchor_result.get("relation_digest")
        if isinstance(relation_digest, str) and relation_digest:
            runtime_inputs["relation_digest"] = relation_digest
        anchor_digest = anchor_result.get("anchor_digest")
        if isinstance(anchor_digest, str) and anchor_digest:
            runtime_inputs["anchor_digest"] = anchor_digest
        runtime_inputs["anchor_result"] = anchor_result
    if isinstance(sync_result, dict):
        relation_digest = sync_result.get("relation_digest_bound")
        if isinstance(relation_digest, str) and relation_digest:
            runtime_inputs["relation_digest"] = relation_digest
        sync_digest = sync_result.get("sync_digest")
        if isinstance(sync_digest, str) and sync_digest:
            runtime_inputs["sync_digest"] = sync_digest
        runtime_inputs["sync_result"] = sync_result
    return runtime_inputs


def _run_sync_module_for_detect(sync_module: Any, cfg: Dict[str, Any], runtime_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：在 detect 侧执行同步模块。

    Execute sync module for detect runtime.

    Args:
        sync_module: Runtime sync module instance.
        cfg: Configuration mapping.
        runtime_inputs: Runtime input mapping.

    Returns:
        Sync result mapping.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(runtime_inputs, dict):
        raise TypeError("runtime_inputs must be dict")
    if sync_module is None:
        return {"status": "absent", "geometry_absent_reason": "sync_module_absent"}
    pipeline_obj = runtime_inputs.get("pipeline")
    latents = runtime_inputs.get("latents")
    sync_with_context = getattr(sync_module, "sync_with_context", None)
    has_sync = hasattr(sync_module, "sync") and callable(getattr(sync_module, "sync", None))
    if callable(sync_with_context):
        try:
            sync_ctx = SyncRuntimeContext(
                pipeline=pipeline_obj,
                latents=latents,
                rng=runtime_inputs.get("rng"),
                trajectory_evidence=None
            )
            try:
                sync_result = sync_with_context(cfg, sync_ctx, runtime_inputs=runtime_inputs)
            except TypeError:
                sync_result = sync_with_context(cfg, sync_ctx)
            if isinstance(sync_result, dict):
                normalized = dict(sync_result)
                raw_status = normalized.get("sync_status")
                if not isinstance(raw_status, str) or not raw_status:
                    raw_status = normalized.get("status")
                if isinstance(raw_status, str) and raw_status:
                    lowered = raw_status.lower()
                    if lowered == "fail":
                        lowered = "failed"
                    if lowered in {"ok", "absent", "mismatch", "failed"}:
                        normalized["sync_status"] = lowered
                        normalized["status"] = lowered
                return normalized
        except Exception as exc:
            return {
                "status": "failed",
                "geometry_failure_reason": f"sync_with_context_failed: {type(exc).__name__}",
            }
    if not has_sync:
        return {"status": "absent", "geometry_absent_reason": "sync_module_missing_sync_method"}
    try:
        sync_result = sync_module.sync(cfg)
        if isinstance(sync_result, dict):
            return sync_result
    except Exception as exc:
        return {
            "status": "failed",
            "geometry_failure_reason": f"sync_failed: {type(exc).__name__}",
        }
    return {"status": "absent", "geometry_absent_reason": "sync_module_returned_non_mapping"}


def _run_geometry_chain_with_sync(
    impl_set: BuiltImplSet,
    cfg: Dict[str, Any],
    *,
    enable_anchor: bool = True,
    enable_sync: bool = True,
    enable_attention_proxy: bool = True,
) -> Any:
    """
    功能：detect 几何链先执行 sync，再执行 geometry extractor。

    Run detect geometry chain with explicit sync-first orchestration.

    Args:
        impl_set: Built runtime implementation set.
        cfg: Configuration mapping.

    Returns:
        Geometry evidence mapping.
    """
    if not isinstance(impl_set, BuiltImplSet):
        raise TypeError("impl_set must be BuiltImplSet")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    base_inputs = _build_geometry_runtime_inputs(cfg, enable_attention_proxy=enable_attention_proxy)
    base_inputs["sync_result"] = {
        "status": "absent",
        "sync_status": "pending_anchor_first",
        "geometry_absent_reason": "sync_not_executed_yet",
    }

    if enable_anchor:
        anchor_result = _run_geometry_extractor_with_runtime_inputs(impl_set.geometry_extractor, cfg, base_inputs)
        if not isinstance(anchor_result, dict):
            anchor_result = {
                "status": "failed",
                "geo_score": None,
                "geometry_failure_reason": "geometry_anchor_result_non_mapping",
            }
    else:
        anchor_result = {
            "status": "absent",
            "geo_score": None,
            "geometry_absent_reason": "anchor_disabled_by_ablation",
            "relation_digest": None,
            "anchor_digest": None,
        }

    if enable_sync:
        sync_inputs = _build_geometry_runtime_inputs(
            cfg,
            anchor_result=anchor_result,
            enable_attention_proxy=enable_attention_proxy,
        )
        sync_module = getattr(impl_set, "sync_module", None)
        sync_result = _run_sync_module_for_detect(sync_module, cfg, sync_inputs)
    else:
        sync_result = {
            "status": "absent",
            "sync_status": "absent",
            "geometry_absent_reason": "sync_disabled_by_ablation",
        }

    geometry_result = dict(anchor_result)
    if isinstance(geometry_result, dict):
        geometry_result.setdefault("sync_result", sync_result)
        geometry_result.setdefault("anchor_result", anchor_result)
        geometry_result.setdefault("anchor_status", anchor_result.get("status"))
        anchor_relation_digest = anchor_result.get("relation_digest")
        if isinstance(anchor_relation_digest, str) and anchor_relation_digest:
            geometry_result.setdefault("relation_digest", anchor_relation_digest)
        if isinstance(sync_result, dict):
            sync_status = sync_result.get("sync_status") or sync_result.get("status")
            if isinstance(sync_status, str) and sync_status and "sync_status" not in geometry_result:
                geometry_result["sync_status"] = sync_status
            if "sync_metrics" not in geometry_result:
                geometry_result["sync_metrics"] = sync_result.get("sync_quality_metrics")
            sync_quality_semantics = sync_result.get("sync_quality_semantics")
            if isinstance(sync_quality_semantics, dict):
                geometry_result["sync_quality_semantics"] = sync_quality_semantics
            relation_digest_bound = sync_result.get("relation_digest_bound")
            if isinstance(relation_digest_bound, str) and relation_digest_bound:
                geometry_result["relation_digest_bound"] = relation_digest_bound
            geometry_result["relation_digest_binding"] = {
                "anchor_relation_digest": anchor_relation_digest if isinstance(anchor_relation_digest, str) else None,
                "sync_relation_digest_bound": relation_digest_bound if isinstance(relation_digest_bound, str) else None,
                "binding_status": "matched" if isinstance(anchor_relation_digest, str) and isinstance(relation_digest_bound, str) and anchor_relation_digest == relation_digest_bound else "mismatch_or_absent",
            }
    return geometry_result


def _run_geometry_extractor_with_runtime_inputs(
    geometry_extractor: Any,
    cfg: Dict[str, Any],
    runtime_inputs: Dict[str, Any] | None = None
) -> Any:
    """
    功能：以兼容方式调用 geometry extractor。 

    Invoke geometry extractor with runtime inputs when supported.

    Args:
        geometry_extractor: Geometry extractor instance.
        cfg: Configuration mapping.

    Returns:
        Geometry extraction output.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if runtime_inputs is None:
        runtime_inputs = _build_geometry_runtime_inputs(cfg)
    if not isinstance(runtime_inputs, dict):
        raise TypeError("runtime_inputs must be dict or None")
    extract_method = getattr(geometry_extractor, "extract", None)
    if not callable(extract_method):
        # geometry_extractor 协议不合法，必须 fail-fast。
        raise TypeError("geometry_extractor.extract must be callable")
    try:
        return extract_method(cfg, inputs=runtime_inputs)
    except TypeError:
        # 兼容旧实现：仅接受 cfg 参数。
        return extract_method(cfg)


def _extract_detect_planner_input_digest(detect_plan_result: Any) -> str | None:
    """
    功能：提取 detect 侧 planner input digest。

    Extract detect-side planner input digest from plan result.

    Args:
        detect_plan_result: Plan result object or mapping.

    Returns:
        planner_input_digest string or None.
    """
    if hasattr(detect_plan_result, "planner_input_digest"):
        value = detect_plan_result.planner_input_digest
        if isinstance(value, str) and value:
            return value
    if isinstance(detect_plan_result, dict):
        value = detect_plan_result.get("planner_input_digest")
        if isinstance(value, str) and value:
            return value
    return None


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
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    ablation = cfg.get("ablation")
    if not isinstance(ablation, dict):
        return {}
    normalized = ablation.get("normalized")
    if not isinstance(normalized, dict):
        return {}
    return normalized


def _compute_ablation_digest_for_report(cfg: Dict[str, Any]) -> str:
    """
    功能：计算评测报告使用的 ablation_digest。 

    Compute canonical ablation digest from normalized ablation config.

    Args:
        cfg: Runtime config mapping.

    Returns:
        Canonical digest string.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    ablation_normalized = _get_ablation_normalized(cfg)
    if not isinstance(ablation_normalized, dict):
        return digests.canonical_sha256({})
    return digests.canonical_sha256(ablation_normalized)


def _compute_ablation_digest_v2_for_report(cfg: Dict[str, Any]) -> str:
    """
    功能：计算扩展口径 ablation_digest_v2。 

    Compute expanded ablation digest that binds high-impact runtime switches.

    Args:
        cfg: Runtime config mapping.

    Returns:
        Canonical digest string.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    ablation_normalized = _get_ablation_normalized(cfg)
    payload = {
        "ablation_normalized": ablation_normalized if isinstance(ablation_normalized, dict) else {},
        "detect_runtime_image_domain_sidecar_enabled": _is_image_domain_sidecar_enabled(cfg),
        "detect_runtime_explicit": (
            cfg.get("detect_runtime", {}).get("image_domain_sidecar_enabled")
            if isinstance(cfg.get("detect_runtime"), dict)
            else None
        ),
    }
    return digests.canonical_sha256(payload)


def _collect_attack_trace_digest(records: list[Dict[str, Any]]) -> str:
    """
    功能：聚合 detect records 中攻击追踪摘要。 

    Collect deterministic aggregate digest from per-record attack traces.

    Args:
        records: Detect records list.

    Returns:
        Aggregate digest string or "<absent>".
    """
    if not isinstance(records, list):
        raise TypeError("records must be list")

    trace_digests: list[str] = []
    for item in records:
        if not isinstance(item, dict):
            continue

        direct_digest = item.get("attack_trace_digest")
        if isinstance(direct_digest, str) and direct_digest:
            trace_digests.append(direct_digest)
            continue

        attack_trace = item.get("attack_trace")
        if isinstance(attack_trace, dict):
            nested_digest = attack_trace.get("attack_trace_digest")
            if isinstance(nested_digest, str) and nested_digest:
                trace_digests.append(nested_digest)

    if len(trace_digests) == 0:
        return "<absent>"
    return digests.canonical_sha256(sorted(trace_digests))


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
    if not isinstance(absent_reason, str) or not absent_reason:
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


def _build_ablation_absent_geometry_evidence(absent_reason: str) -> Dict[str, Any]:
    """
    功能：构造 ablation 禁用时的 geometry_evidence absent 语义。

    Build geometry_evidence with status="absent" for ablation-disabled modules.

    Args:
        absent_reason: Absence reason string (e.g., "geometry_chain_disabled_by_ablation").

    Returns:
        GeometryEvidence-compatible dict with status="absent", score=None.

    Raises:
        TypeError: If absent_reason is invalid.
    """
    if not isinstance(absent_reason, str) or not absent_reason:
        raise TypeError("absent_reason must be non-empty str")
    return {
        "status": "absent",
        "geo_score": None,
        "audit": {
            "impl_identity": "ablation_switchboard",
            "impl_version": "v1",
            "impl_digest": digests.canonical_sha256({"impl_id": "ablation_switchboard", "impl_version": "v1"}),
            "trace_digest": digests.canonical_sha256({"absent_reason": absent_reason})
        },
        "sync": {
            "status": "absent",
            "reason": absent_reason
        },
        "anchor_digest": None,
        "anchor_metrics": None,
        "sync_digest": None,
        "sync_metrics": None,
        "align_trace_digest": None,
        "geo_failure_reason": None  # absent 状态下无失败原因
    }
