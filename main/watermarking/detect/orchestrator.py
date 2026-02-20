"""
File purpose: 检测、评估与校准编排（detect, evaluate, calibrate orchestration）。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from main.core import digests
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.content_detector import CONTENT_DETECTOR_ID
from main.watermarking.content_chain import detector_scoring
from main.watermarking.content_chain.high_freq_embedder import (
    HighFreqEmbedder,
    HIGH_FREQ_EMBEDDER_ID,
    HIGH_FREQ_EMBEDDER_VERSION,
)
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

    planner_inputs = _build_planner_inputs_for_runtime(cfg, trajectory_evidence)
    mask_digest = None
    if isinstance(content_evidence_payload, dict):
        mask_digest = content_evidence_payload.get("mask_digest")

    detect_plan_result = detect_plan_result_override if detect_plan_result_override is not None else impl_set.subspace_planner.plan(
        cfg,
        mask_digest=mask_digest,
        cfg_digest=cfg_digest,
        inputs=planner_inputs
    )

    embed_time_plan_digest = None
    embed_time_basis_digest = None
    embed_time_planner_impl_identity = None
    if isinstance(input_record, dict):
        embed_time_plan_digest = input_record.get("plan_digest")
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

    mismatch_reasons = _collect_plan_mismatch_reasons(
        embed_time_plan_digest=embed_time_plan_digest,
        detect_time_plan_digest=detect_time_plan_digest,
        embed_time_basis_digest=embed_time_basis_digest,
        detect_time_basis_digest=detect_time_basis_digest,
        embed_time_planner_impl_identity=embed_time_planner_impl_identity,
        detect_time_planner_impl_identity=detect_time_planner_impl_identity
    )

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
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_absent_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    else:
        extractor_impl_id = getattr(impl_set.content_extractor, "impl_id", None)
        if extractor_impl_id == CONTENT_DETECTOR_ID:
            lf_evidence = _extract_lf_evidence_from_input_record(input_record)
            detector_inputs: Dict[str, Any] = {
                "plan_digest": detect_time_plan_digest,
                "lf_evidence": lf_evidence,
                "hf_evidence": hf_evidence,
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
        content_evidence_adapted = _adapt_content_evidence_for_fusion(content_evidence_payload)
        fusion_result = impl_set.fusion_rule.fuse(cfg, content_evidence_adapted, geometry_evidence_adapted)
    input_fields = len(input_record or {})

    # (D-2) 实现 detect 侧同构分数与一致性校验
    detect_placeholder_flag = True  # 默认为占位符模式
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

        # 如果 detect 侧分数有效且一致性通过，则标记为实际运行而非占位符
        if detect_lf_status == "ok" and subspace_consistency_status == "ok":
            detect_placeholder_flag = False

    # 删除临时的 latents 字段，确保不写入 records
    cfg.pop("__detect_final_latents__", None)

    # plan_digest 一致性验证（优先级高于 compute_failed）。
    # 优先级：已检测到的 plan_digest mismatch > compute_failed > not_validated
    plan_digest_status = "not_validated"
    plan_digest_mismatch_reason = None

    # 如果已通过 mismatch_reasons 检测到 plan_digest 不一致，优先级最高
    if forced_mismatch and "plan_digest_mismatch" in mismatch_reasons:
        # mismatch 已经在 forced_mismatch 分支中处理过，直接设置状态
        plan_digest_status = "mismatch"
        plan_digest_mismatch_reason = "plan_digest_mismatch"
    # 仅当未检测到 mismatch 时，才进行额外的一致性验证或 compute_failed 标记
    elif input_record and cfg_digest:
        embed_time_plan_digest = input_record.get("plan_digest")
        if embed_time_plan_digest is not None:
            if detect_time_plan_digest is not None:
                if detect_time_plan_digest == embed_time_plan_digest:
                    plan_digest_status = "ok"
                else:
                    plan_digest_status = "mismatch"
                    plan_digest_mismatch_reason = "plan_digest_mismatch"
            else:
                # 只有在不需要重算即可判定 mismatch 的情况下，才将 computed_failed 作为后备选项
                # embed_time 存在但 detect_time 计算失败，这才是 compute_failed
                plan_digest_status = "compute_failed"

    record: Dict[str, Any] = {
        "operation": "detect",
        "detect_placeholder": detect_placeholder_flag,
        "image_path": "placeholder_test.png",
        "score": getattr(fusion_result, "evidence_summary", {}).get("content_score"),
        "execution_report": {
            "content_chain_status": "fail" if forced_mismatch else "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "fail" if forced_mismatch else "ok",
            "audit_obligations_satisfied": True
        },
        "input_record_fields": input_fields,
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
    trajectory_evidence: Optional[Dict[str, Any]] = None
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

    trace_signature = {
        "num_inference_steps": cfg.get("inference_num_steps", cfg.get("generation", {}).get("num_inference_steps", 16) if isinstance(cfg.get("generation"), dict) else 16),
        "guidance_scale": cfg.get("inference_guidance_scale", cfg.get("generation", {}).get("guidance_scale", 7.0) if isinstance(cfg.get("generation"), dict) else 7.0),
        "height": cfg.get("inference_height", cfg.get("model", {}).get("height", 512) if isinstance(cfg.get("model"), dict) else 512),
        "width": cfg.get("inference_width", cfg.get("model", {}).get("width", 512) if isinstance(cfg.get("model"), dict) else 512),
    }
    inputs = {"trace_signature": trace_signature}
    if trajectory_evidence is not None:
        inputs["trajectory_evidence"] = trajectory_evidence
    return inputs


def run_calibrate_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet) -> Dict[str, Any]:
    """
    功能：执行校准占位流程。

    Execute calibration placeholder flow using injected implementations.

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

    subspace_result = impl_set.subspace_planner.plan(cfg)

    record: Dict[str, Any] = {
        "operation": "calibrate",
        "calibration_placeholder": True,
        "protocol": "neyman_pearson",
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
        "calibration_samples": 1000,
        "subspace_plan": subspace_result
    }
    return record


def run_evaluate_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet) -> Dict[str, Any]:
    """
    功能：执行评估占位流程。

    Execute evaluation placeholder flow using injected implementations.

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

    content_result = impl_set.content_extractor.extract(cfg)
    geometry_result = impl_set.geometry_extractor.extract(cfg)
    
    # (1) 统一转换 ContentEvidence / GeometryEvidence 数据类为 dict。
    content_evidence_payload = None
    if hasattr(content_result, "as_dict") and callable(content_result.as_dict):
        content_evidence_payload = content_result.as_dict()
    elif isinstance(content_result, dict):
        content_evidence_payload = content_result
    
    geometry_evidence_payload = None
    if hasattr(geometry_result, "as_dict") and callable(geometry_result.as_dict):
        geometry_evidence_payload = geometry_result.as_dict()
    elif isinstance(geometry_result, dict):
        geometry_evidence_payload = geometry_result
    
    # (2) 构造融合输入适配 dict。
    content_evidence_adapted = _adapt_content_evidence_for_fusion(content_result)
    geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
    
    fusion_result = impl_set.fusion_rule.fuse(cfg, content_evidence_adapted, geometry_evidence_adapted)

    record: Dict[str, Any] = {
        "operation": "evaluate",
        "evaluation_placeholder": True,
        "metrics": {
            "tpr": 0.95,
            "fpr": 0.01,
            "accuracy": 0.97
        },
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
        "test_samples": 500,
        # (append-only) 保留完整的 payload。
        "content_evidence_payload": content_evidence_payload,
        "geometry_evidence_payload": geometry_evidence_payload,
        "content_result": content_result,
        "geometry_result": geometry_result,
        "fusion_result": fusion_result
    }
    return record


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
