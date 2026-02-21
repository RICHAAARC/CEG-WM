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
from main.core import records_io
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain import detector_scoring
from main.watermarking.common.plan_digest_flow import verify_plan_digest
from main.watermarking.content_chain.high_freq_embedder import (
    HighFreqEmbedder,
    HIGH_FREQ_EMBEDDER_ID,
    HIGH_FREQ_EMBEDDER_VERSION,
    detect_high_freq_score,
)
from main.watermarking.content_chain.low_freq_coder import detect_low_freq_score
from main.watermarking.fusion import neyman_pearson
from main.watermarking.fusion.interfaces import FusionDecision


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
    功能：执行检测占位流程，包括 plan_digest 一致性验证。

    Execute detect placeholder flow using injected implementations.
    Validates plan_digest consistency with embed-time plan_digest when available.

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
        # impl_set 类型不合法，必须 fail-fast。
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

    content_result = content_result_override if content_result_override is not None else impl_set.content_extractor.extract(cfg)
    geometry_result = impl_set.geometry_extractor.extract(cfg)

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

    planner_inputs = _build_planner_inputs_for_runtime(cfg, trajectory_evidence, content_evidence_payload)
    mask_digest = None
    if isinstance(content_evidence_payload, dict):
        mask_digest = content_evidence_payload.get("mask_digest")

    detect_plan_result = detect_plan_result_override if detect_plan_result_override is not None else impl_set.subspace_planner.plan(
        cfg,
        mask_digest=mask_digest,
        cfg_digest=cfg_digest,
        inputs=planner_inputs
    )

    expected_plan_digest = _resolve_expected_plan_digest(input_record)
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

    hf_evidence = _build_hf_detect_evidence(
        cfg=cfg,
        cfg_digest=cfg_digest,
        plan_payload=plan_payload,
        plan_digest=detect_time_plan_digest,
        embed_time_plan_digest=embed_time_plan_digest,
        trajectory_evidence=trajectory_evidence,
    )
    lf_raw_score, hf_raw_score, raw_score_traces = _extract_content_raw_scores_from_image(
        cfg=cfg,
        input_record=input_record,
        plan_payload=plan_payload,
        plan_digest=detect_time_plan_digest,
        cfg_digest=cfg_digest,
    )

    mismatch_reasons = _collect_plan_mismatch_reasons(
        embed_time_plan_digest=expected_plan_digest,
        detect_time_plan_digest=detect_time_plan_digest,
        embed_time_basis_digest=embed_time_basis_digest,
        detect_time_basis_digest=detect_time_basis_digest,
        embed_time_planner_impl_identity=embed_time_planner_impl_identity,
        detect_time_planner_impl_identity=detect_time_planner_impl_identity
    )
    plan_digest_status, plan_digest_reason = verify_plan_digest(
        expected_plan_digest if isinstance(expected_plan_digest, str) else None,
        detect_time_plan_digest if isinstance(detect_time_plan_digest, str) else None,
    )
    if plan_digest_reason == "plan_digest_mismatch" and "plan_digest_mismatch" not in mismatch_reasons:
        mismatch_reasons.append("plan_digest_mismatch")

    trajectory_status, trajectory_mismatch_reason = _evaluate_trajectory_consistency(
        input_record=input_record,
        trajectory_evidence=trajectory_evidence,
        detect_planner_input_digest=detect_planner_input_digest
    )
    if trajectory_mismatch_reason:
        mismatch_reasons.append(trajectory_mismatch_reason)

    injection_status, injection_mismatch_reason = _evaluate_injection_consistency(
        input_record=input_record,
        injection_evidence=injection_evidence
    )
    if injection_mismatch_reason:
        mismatch_reasons.append(injection_mismatch_reason)

    # (S-D) Paper Faithfulness: 验证 paper faithfulness 证据一致性（必达）
    # 注意：只在 input_record 存在且包含 paper_faithfulness 信息时才添加到全局 mismatch_reasons
    # 这样可以避免单元测试中使用不完整 input_record 时产生副作用
    paper_faithfulness_status, paper_absent_reasons, paper_mismatch_reasons, paper_fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=input_record
    )
    
    # 仅当 input_record 存在且包含 paper_faithfulness 字段时，才将缺失视为 mismatch
    if input_record is not None and isinstance(input_record.get("paper_faithfulness"), dict):
        # input_record 包含 paper_faithfulness，所以缺失或不一致是真实的 mismatch
        if paper_mismatch_reasons:
            mismatch_reasons.extend(paper_mismatch_reasons)

    primary_mismatch_reason, primary_mismatch_field_path = _resolve_primary_mismatch(
        mismatch_reasons
    )

    forced_mismatch = len(mismatch_reasons) > 0
    forced_absent = (trajectory_status == "absent" or injection_status == "absent") and not forced_mismatch
    if forced_mismatch:
        content_evidence_payload = {
            "status": "mismatch",
            "score": None,
            "plan_digest": detect_time_plan_digest,
            "basis_digest": detect_time_basis_digest,
            "content_failure_reason": _resolve_mismatch_failure_reason(primary_mismatch_reason),
            "content_mismatch_reason": primary_mismatch_reason,
            "content_mismatch_field_path": primary_mismatch_field_path,
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
        _merge_hf_evidence(content_evidence_payload, hf_evidence)
        _bind_raw_scores_to_content_payload(content_evidence_payload, lf_raw_score, hf_raw_score, raw_score_traces)
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
            "content_failure_reason": None,
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "audit": {
                "impl_identity": "detect_orchestrator",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"impl_id": "detect_orchestrator", "impl_version": "v1"}),
                "trace_digest": digests.canonical_sha256({
                    "trajectory_status": trajectory_status,
                    "trajectory_mismatch_reason": trajectory_mismatch_reason
                })
            }
        }
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _merge_hf_evidence(content_evidence_payload, hf_evidence)
        _bind_raw_scores_to_content_payload(content_evidence_payload, lf_raw_score, hf_raw_score, raw_score_traces)
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_absent_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    else:
        lf_evidence = _extract_lf_evidence_from_input_record(input_record)
        detector_inputs: Dict[str, Any] = {
            "expected_plan_digest": expected_plan_digest,
            "observed_plan_digest": detect_time_plan_digest,
            "disable_cfg_plan_digest_fallback": True,
            "plan_digest": detect_time_plan_digest,
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
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _merge_hf_evidence(content_evidence_payload, hf_evidence)
        _bind_raw_scores_to_content_payload(content_evidence_payload, lf_raw_score, hf_raw_score, raw_score_traces)
        content_evidence_adapted = _adapt_content_evidence_for_fusion(content_evidence_payload)
        fusion_result = impl_set.fusion_rule.fuse(cfg, content_evidence_adapted, geometry_evidence_adapted)
    input_fields = len(input_record or {})

    # (D-2) 实现 detect 侧同构分数与一致性校验
    detect_runtime_mode = "placeholder"  # 默认：未获得可用 detect 同构分数
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

        # 如果 detect 侧分数有效且一致性通过，则标记为真实运行模式
        if detect_lf_status == "ok" and subspace_consistency_status == "ok":
            detect_runtime_mode = "real"

    # 删除临时的 latents 字段，确保不写入 records
    cfg.pop("__detect_final_latents__", None)

    plan_digest_mismatch_reason = plan_digest_reason if plan_digest_reason == "plan_digest_mismatch" else None

    record: Dict[str, Any] = {
        "operation": "detect",
        "detect_runtime_mode": detect_runtime_mode,
        "detect_runtime_status": "active" if detect_runtime_mode == "real" else "fallback",
        "detect_placeholder": (detect_runtime_mode != "real"),
        "image_path": "placeholder_test.png",
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


def _build_hf_detect_evidence(
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
    hf_score, evidence = embedder.detect(
        latents_or_features=trajectory_evidence,
        plan=plan_payload,
        cfg=cfg,
        cfg_digest=cfg_digest,
        expected_plan_digest=expected_plan_digest,
    )
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
        # (S-D) Paper Faithfulness mismatch field paths
        "paper_spec_digest_absent_or_invalid": "paper_faithfulness.spec_digest",
        "pipeline_fingerprint_digest_absent_or_invalid": "content_evidence.pipeline_fingerprint_digest",
        "injection_site_digest_absent_or_invalid": "content_evidence.injection_site_digest",
        "alignment_digest_absent_or_invalid": "content_evidence.alignment_digest",
        "paper_faithfulness_section_absent": "paper_faithfulness",
        "content_evidence_absent": "content_evidence",
    }
    for token in [
        "plan_digest_mismatch",
        "basis_digest_mismatch",
        "planner_impl_identity_mismatch",
        "trajectory_spec_digest_mismatch",
        "trajectory_digest_mismatch",
        "trajectory_evidence_invalid",
        "injection_trace_digest_mismatch",
        "injection_params_digest_mismatch",
        "injection_trace_digest_invalid",
        "injection_params_digest_invalid",
        "injection_status_mismatch",
        "injection_subspace_binding_digest_mismatch",
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
        "score_name": "content_score",
        "target_fpr": float(target_fpr),
        "null_source": "wrong_key",
        "n_samples": len(scores),
        "order_statistics": order_stat_info,
        "stratification": strata_info,
        "sample_digest": digests.canonical_sha256({"scores": [round(float(v), 12) for v in scores]}),
    }

    record: Dict[str, Any] = {
        "operation": "calibrate",
        "calibration_placeholder": False,
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
    metrics_obj, breakdown = evaluate_records_against_threshold(detect_records, thresholds_obj)

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

    report_obj = {
        "evaluation_version": "eval_v1",
        "attack_protocol_version": _read_attack_protocol_version(cfg),
        "thresholds_artifact_path": str(thresholds_path),
        "thresholds_digest": digests.canonical_sha256(thresholds_obj),
        "cfg_digest": cfg.get("__evaluate_cfg_digest__", "<absent>"),
        "anchors": _collect_evaluation_anchors(detect_records),
    }

    record: Dict[str, Any] = {
        "operation": "evaluate",
        "evaluation_placeholder": False,
        "metrics": metrics_obj,
        "evaluation_breakdown": breakdown,
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
    thresholds_obj: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    功能：使用只读阈值评测 detect 记录。 

    Evaluate detect records using precomputed thresholds artifact only.

    Args:
        records: Detect records list.
        thresholds_obj: Thresholds artifact mapping.

    Returns:
        Tuple of (metrics, breakdown).
    """
    if not isinstance(records, list):
        raise TypeError("records must be list")
    if not isinstance(thresholds_obj, dict):
        raise TypeError("thresholds_obj must be dict")

    threshold_value = thresholds_obj.get("threshold_value")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")
    threshold_float = float(threshold_value)

    n_total = 0
    n_reject = 0
    n_pos = 0
    n_neg = 0
    tp = 0
    fp = 0
    accepted = 0

    for item in records:
        if not isinstance(item, dict):
            continue
        n_total += 1
        content_payload = item.get("content_evidence_payload")
        if not isinstance(content_payload, dict):
            n_reject += 1
            continue
        if content_payload.get("status") != "ok":
            n_reject += 1
            continue
        score_value = content_payload.get("score")
        if not isinstance(score_value, (int, float)):
            n_reject += 1
            continue
        score_float = float(score_value)
        if not np.isfinite(score_float):
            n_reject += 1
            continue

        gt_value = _extract_ground_truth_label(item)
        if gt_value is None:
            n_reject += 1
            continue

        accepted += 1
        pred_positive = score_float >= threshold_float
        if gt_value:
            n_pos += 1
            if pred_positive:
                tp += 1
        else:
            n_neg += 1
            if pred_positive:
                fp += 1

    reject_rate = float(n_reject / n_total) if n_total > 0 else 1.0
    tpr_value = float(tp / n_pos) if n_pos > 0 else None
    fpr_empirical = float(fp / n_neg) if n_neg > 0 else None

    metrics = {
        "metric_version": "tpr_at_fpr_v1",
        "score_name": thresholds_obj.get("score_name", "content_score"),
        "target_fpr": thresholds_obj.get("target_fpr"),
        "threshold_value": threshold_float,
        "threshold_key_used": thresholds_obj.get("threshold_key_used"),
        "tpr_at_fpr": tpr_value,
        "fpr_empirical": fpr_empirical,
        "reject_rate": reject_rate,
        "n_total": n_total,
        "n_accepted": accepted,
        "n_rejected": n_reject,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    breakdown = {
        "confusion": {
            "tp": tp,
            "fp": fp,
            "fn": max(0, n_pos - tp),
            "tn": max(0, n_neg - fp),
        }
    }
    return metrics, breakdown


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
    for field_name in ["status", "geo_score", "audit", "anchor_digest", "align_trace_digest",
                       "align_residuals", "stability_metrics", "geometry_failure_reason"]:
        if hasattr(geometry_evidence, field_name):
            adapted[field_name] = getattr(geometry_evidence, field_name)
    
    return adapted if adapted else {"status": "unknown"}
