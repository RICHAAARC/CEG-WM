"""
统一内容链提取器（Embed + Detect 双模式）

功能说明：
- 根据 detect.content.enabled 配置自动切换 embed/detect 模式。
- Embed 模式：提取语义掩码并返回 mask_digest等结构证据。
- Detect 模式：执行完整的 LF/HF 检测并返回 content_score。
- 严格遵循冻结语义：absent/failed/mismatch/ok 的触发条件冻结。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

from main.core import digests

from .semantic_mask_provider import SemanticMaskProvider, SEMANTIC_MASK_PROVIDER_ID, SEMANTIC_MASK_PROVIDER_VERSION
from .interfaces import ContentEvidence


UNIFIED_CONTENT_EXTRACTOR_ID = "unified_content_extractor"
UNIFIED_CONTENT_EXTRACTOR_VERSION = "v2"
UNIFIED_CONTENT_TRACE_VERSION = "v3"
EMBED_CONTENT_RUNTIME_PHASE_PRECOMPUTE = "embed_precompute"


def _resolve_content_runtime_phase(inputs: Optional[Dict[str, Any]]) -> str | None:
    """
    功能：解析内容提取阶段语义控制字段。

    Resolve the optional runtime-phase control marker for content extraction.

    Args:
        inputs: Optional extractor input mapping.

    Returns:
        Normalized runtime-phase token when present; otherwise None.
    """
    if not isinstance(inputs, dict):
        return None
    runtime_phase = inputs.get("content_runtime_phase")
    if not isinstance(runtime_phase, str):
        return None
    normalized_phase = runtime_phase.strip()
    if not normalized_phase:
        return None
    return normalized_phase


def _strip_runtime_control_fields(inputs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    功能：剥离仅用于模式路由的控制字段，避免污染下游摘要。

    Remove runtime control markers before delegating to downstream providers.

    Args:
        inputs: Optional extractor input mapping.

    Returns:
        Sanitized input mapping, or None when the original input is absent.
    """
    if inputs is None:
        return None
    sanitized_inputs = dict(inputs)
    sanitized_inputs.pop("content_runtime_phase", None)
    return sanitized_inputs


class _UnifiedContentExtractorBase:
    """
    功能：统一内容链提取器基类（Embed + Detect 双模式，内部实现）。

    Internal base class for unified content extractor supporting both embed and detect modes.
    Embed mode delegates to SemanticMaskProvider. Detect mode performs formal
    LF/HF evidence aggregation directly in this class, without a separate
    legacy detector indirection.

    Embed mode (detect.content.enabled=False):
      - Extracts semantic mask via SemanticMaskProvider.
      - Returns ContentEvidence with mask_digest, mask_stats (structural evidence, score=None).

        Detect mode (detect.content.enabled=True):
            - Validates plan anchors and structured LF/HF evidence.
            - Returns ContentEvidence with content_score (valid score when status=ok).

    Args:
        impl_id: Implementation identifier string.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If constructor arguments are invalid.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        if not isinstance(impl_id, str) or not impl_id:
            raise ValueError("impl_id must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            raise ValueError("impl_version must be non-empty str")
        if not isinstance(impl_digest, str) or not impl_digest:
            raise ValueError("impl_digest must be non-empty str")

        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest

        # 初始化子模块实例（延迟构造以保持确定性）
        mask_provider_digest = digests.canonical_sha256({
            "impl_id": SEMANTIC_MASK_PROVIDER_ID,
            "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
        })
        self._mask_provider = SemanticMaskProvider(
            SEMANTIC_MASK_PROVIDER_ID,
            SEMANTIC_MASK_PROVIDER_VERSION,
            mask_provider_digest
        )

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        cfg_digest: Optional[str] = None
    ) -> ContentEvidence:
        """
        功能：提取内容链证据（embed 或 detect 模式自动选择）。

        Extract content evidence in embed or detect mode based on configuration.

                Mode selection:
                    - If detect.content.enabled=False -> Embed mode (SemanticMaskProvider).
                    - If detect.content.enabled=True -> Detect mode (formal LF/HF evidence aggregation).

        Args:
            cfg: Configuration mapping.
            inputs: Optional inputs mapping.
            cfg_digest: Optional canonical config digest.

        Returns:
            ContentEvidence instance.

        Raises:
            TypeError: If input types are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            raise TypeError("inputs must be dict or None")
        if cfg_digest is not None and not isinstance(cfg_digest, str):
            raise TypeError("cfg_digest must be str or None")

        runtime_phase = _resolve_content_runtime_phase(inputs)
        if runtime_phase == EMBED_CONTENT_RUNTIME_PHASE_PRECOMPUTE:
            return self._mask_provider.extract(
                cfg,
                inputs=_strip_runtime_control_fields(inputs),
                cfg_digest=cfg_digest,
            )

        # 判断模式
        detect_content_enabled = cfg.get("detect", {}).get("content", {}).get("enabled", False)
        if not isinstance(detect_content_enabled, bool):
            raise TypeError("detect.content.enabled must be bool")

        if detect_content_enabled:
            return self._extract_detect_mode(cfg, inputs=inputs, cfg_digest=cfg_digest)

        return self._mask_provider.extract(cfg, inputs=inputs, cfg_digest=cfg_digest)

    def _extract_detect_mode(
        self,
        cfg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]],
        cfg_digest: Optional[str],
    ) -> ContentEvidence:
        """
        功能：在统一提取器内部执行正式 detect 聚合。

        Aggregate formal LF/HF detect evidence directly inside UnifiedContentExtractor.

        Args:
            cfg: Configuration mapping.
            inputs: Structured detector inputs.
            cfg_digest: Optional canonical config digest.

        Returns:
            ContentEvidence with frozen status/score semantics.
        """
        normalized_inputs: Dict[str, Any] = inputs or {}
        expected_plan_digest = _resolve_expected_plan_digest(normalized_inputs)
        observed_plan_digest = _resolve_observed_plan_digest(normalized_inputs)
        plan_payload = normalized_inputs.get("plan_payload")
        basis_digest = _extract_plan_anchor(plan_payload, "basis_digest")
        trajectory_evidence = _extract_optional_mapping(normalized_inputs.get("trajectory_evidence"))

        if not isinstance(expected_plan_digest, str) or not expected_plan_digest:
            return self._build_detect_result(
                status="absent",
                score=None,
                plan_digest=None,
                basis_digest=basis_digest,
                content_failure_reason="detector_no_plan_expected",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
                trajectory_evidence=trajectory_evidence,
                lf_trace_digest=None,
                hf_trace_digest=None,
                lf_statistics_digest=_extract_statistics_digest(normalized_inputs, "lf"),
                hf_statistics_digest=_extract_statistics_digest(normalized_inputs, "hf"),
            )

        if isinstance(observed_plan_digest, str) and observed_plan_digest and observed_plan_digest != expected_plan_digest:
            return self._build_detect_result(
                status="mismatch",
                score=None,
                plan_digest=expected_plan_digest,
                basis_digest=basis_digest,
                content_failure_reason="detector_plan_mismatch",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
                trajectory_evidence=trajectory_evidence,
                lf_trace_digest=None,
                hf_trace_digest=None,
                lf_statistics_digest=_extract_statistics_digest(normalized_inputs, "lf"),
                hf_statistics_digest=_extract_statistics_digest(normalized_inputs, "hf"),
            )

        lf_status, lf_score, lf_failure_reason, lf_trace_digest, lf_summary = _extract_channel_evidence(
            evidence=normalized_inputs.get("lf_evidence"),
            score_key="lf_score",
            channel_name="lf",
        )

        hf_enabled = bool(cfg.get("watermark", {}).get("hf", {}).get("enabled", False))
        if hf_enabled:
            hf_status, hf_score, hf_failure_reason, hf_trace_digest, hf_summary = _extract_channel_evidence(
                evidence=normalized_inputs.get("hf_evidence"),
                score_key="hf_score",
                channel_name="hf",
            )
        else:
            hf_status = "absent"
            hf_score = None
            hf_failure_reason = "hf_disabled_by_config"
            hf_trace_digest = None
            hf_summary = {"hf_status": "absent", "hf_absent_reason": "hf_disabled_by_config"}

        _validate_non_negative_score(lf_score, "lf_score")
        _validate_non_negative_score(hf_score, "hf_score")

        if lf_status in {"absent", "failed", "mismatch"}:
            propagated_reason = lf_failure_reason
            if lf_status == "mismatch" and not isinstance(propagated_reason, str):
                propagated_reason = "detector_plan_mismatch"
            if lf_status == "failed" and not isinstance(propagated_reason, str):
                propagated_reason = "detector_extraction_failed"
            failure_score_parts: Dict[str, Any] = {
                "lf_status": lf_status,
            }
            if isinstance(lf_summary, dict):
                failure_score_parts["lf_metrics"] = lf_summary
            if isinstance(hf_summary, dict):
                failure_score_parts["hf_metrics"] = hf_summary
                hf_summary_status = hf_summary.get("hf_status")
                if isinstance(hf_summary_status, str) and hf_summary_status:
                    failure_score_parts["hf_status"] = hf_summary_status
                hf_absent_reason = hf_summary.get("hf_absent_reason")
                if isinstance(hf_absent_reason, str) and hf_absent_reason:
                    failure_score_parts["hf_absent_reason"] = hf_absent_reason
                hf_failure_reason_summary = hf_summary.get("hf_failure_reason")
                if isinstance(hf_failure_reason_summary, str) and hf_failure_reason_summary:
                    failure_score_parts["hf_failure_reason"] = hf_failure_reason_summary
            return self._build_detect_result(
                status=lf_status,
                score=None,
                plan_digest=expected_plan_digest,
                basis_digest=basis_digest,
                content_failure_reason=propagated_reason,
                score_parts=failure_score_parts,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
                trajectory_evidence=trajectory_evidence,
                lf_trace_digest=lf_trace_digest,
                hf_trace_digest=hf_trace_digest,
                lf_statistics_digest=_extract_statistics_digest(normalized_inputs, "lf"),
                hf_statistics_digest=_extract_statistics_digest(normalized_inputs, "hf"),
            )

        if hf_status == "mismatch":
            return self._build_detect_result(
                status="mismatch",
                score=None,
                plan_digest=expected_plan_digest,
                basis_digest=basis_digest,
                content_failure_reason=hf_failure_reason or "hf_plan_mismatch",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
                trajectory_evidence=trajectory_evidence,
                lf_trace_digest=lf_trace_digest,
                hf_trace_digest=hf_trace_digest,
                lf_statistics_digest=_extract_statistics_digest(normalized_inputs, "lf"),
                hf_statistics_digest=_extract_statistics_digest(normalized_inputs, "hf"),
            )

        if hf_status == "failed":
            return self._build_detect_result(
                status="failed",
                score=None,
                plan_digest=expected_plan_digest,
                basis_digest=basis_digest,
                content_failure_reason=hf_failure_reason or "hf_detection_failed",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
                trajectory_evidence=trajectory_evidence,
                lf_trace_digest=lf_trace_digest,
                hf_trace_digest=hf_trace_digest,
                lf_statistics_digest=_extract_statistics_digest(normalized_inputs, "lf"),
                hf_statistics_digest=_extract_statistics_digest(normalized_inputs, "hf"),
            )

        if lf_score is None and hf_score is None:
            return self._build_detect_result(
                status="failed",
                score=None,
                plan_digest=expected_plan_digest,
                basis_digest=basis_digest,
                content_failure_reason="detector_no_evidence",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
                trajectory_evidence=trajectory_evidence,
                lf_trace_digest=lf_trace_digest,
                hf_trace_digest=hf_trace_digest,
                lf_statistics_digest=_extract_statistics_digest(normalized_inputs, "lf"),
                hf_statistics_digest=_extract_statistics_digest(normalized_inputs, "hf"),
            )

        content_score, score_parts = self._compose_content_score(
            lf_score=lf_score,
            hf_score=hf_score,
            hf_status=hf_status,
            hf_failure_reason=hf_failure_reason,
        )
        score_parts["lf_status"] = lf_status
        if isinstance(lf_summary, dict):
            score_parts["lf_metrics"] = lf_summary
        if isinstance(hf_summary, dict):
            score_parts["hf_metrics"] = hf_summary
            hf_summary_status = hf_summary.get("hf_status")
            if isinstance(hf_summary_status, str) and hf_summary_status:
                score_parts["hf_status"] = hf_summary_status
            hf_absent_reason = hf_summary.get("hf_absent_reason")
            if isinstance(hf_absent_reason, str) and hf_absent_reason:
                score_parts["hf_absent_reason"] = hf_absent_reason
            hf_failure_reason_summary = hf_summary.get("hf_failure_reason")
            if isinstance(hf_failure_reason_summary, str) and hf_failure_reason_summary:
                score_parts["hf_failure_reason"] = hf_failure_reason_summary

        detection_result = {
            "plan_digest": expected_plan_digest,
            "basis_digest": basis_digest,
            "lf_score": lf_score,
            "hf_score": hf_score,
            "content_score": content_score,
            "content_score_rule_version": score_parts.get("content_score_rule_version"),
            "cfg_digest": cfg_digest,
        }
        return self._build_detect_result(
            status="ok",
            score=content_score,
            plan_digest=expected_plan_digest,
            basis_digest=basis_digest,
            content_failure_reason=None,
            score_parts=score_parts,
            lf_score=lf_score,
            hf_score=hf_score,
            cfg_digest=cfg_digest,
            detection_result=detection_result,
            trajectory_evidence=trajectory_evidence,
            lf_trace_digest=lf_trace_digest,
            hf_trace_digest=hf_trace_digest,
            lf_statistics_digest=_extract_statistics_digest(normalized_inputs, "lf"),
            hf_statistics_digest=_extract_statistics_digest(normalized_inputs, "hf"),
        )

    def _compose_content_score(
        self,
        lf_score: Optional[float],
        hf_score: Optional[float],
        hf_status: Optional[str],
        hf_failure_reason: Optional[str],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        功能：应用冻结的 LF/HF 组合规则。

        Apply the frozen LF-primary content score rule.

        Args:
            lf_score: LF score.
            hf_score: HF score.
            hf_status: HF status.
            hf_failure_reason: HF absent or failure reason.

        Returns:
            Tuple of content_score and score_parts.
        """
        if lf_score is None:
            raise ValueError("lf_score must be non-None when composing content score")

        if hf_status == "absent":
            hf_reason = hf_failure_reason if isinstance(hf_failure_reason, str) else "hf_disabled_by_config"
            return float(lf_score), {
                "content_score_rule_version": "content_score_rule_v1",
                "rule_id": "lf_only_when_hf_absent",
                "content_score_rule_id": "lf_only_when_hf_absent",
                "lf_score": lf_score,
                "hf_score": "<absent>",
                "hf_status": "absent",
                "hf_absent_reason": hf_reason,
            }

        if hf_score is None:
            return float(lf_score), {
                "content_score_rule_version": "content_score_rule_v1",
                "rule_id": "lf_only_default",
                "content_score_rule_id": "lf_only_default",
                "lf_score": lf_score,
                "hf_score": "<absent>",
                "hf_status": "absent",
                "hf_absent_reason": "hf_disabled_by_config",
            }

        weight_lf = 0.7
        weight_hf = 0.3
        content_score = float(round(lf_score * weight_lf + hf_score * weight_hf, 8))
        return content_score, {
            "content_score_rule_version": "content_score_rule_v1",
            "rule_id": "lf_hf_weighted_sum",
            "content_score_rule_id": "lf_hf_weighted_sum",
            "lf_score": lf_score,
            "hf_score": hf_score,
            "weights": {"lf": weight_lf, "hf": weight_hf},
            "hf_status": "ok",
        }

    def _build_detect_result(
        self,
        status: str,
        score: Optional[float],
        plan_digest: Optional[str],
        basis_digest: Optional[str],
        content_failure_reason: Optional[str],
        score_parts: Optional[Dict[str, Any]],
        lf_score: Optional[float],
        hf_score: Optional[float],
        cfg_digest: Optional[str],
        detection_result: Optional[Dict[str, Any]],
        trajectory_evidence: Optional[Dict[str, Any]],
        lf_trace_digest: Optional[str],
        hf_trace_digest: Optional[str],
        lf_statistics_digest: Optional[str],
        hf_statistics_digest: Optional[str],
    ) -> ContentEvidence:
        """
        功能：构造 detect 模式 ContentEvidence 结果。

        Build frozen ContentEvidence for detect mode.
        """
        trace_payload = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_version": UNIFIED_CONTENT_TRACE_VERSION,
            "plan_digest": plan_digest,
            "basis_digest": basis_digest,
            "cfg_digest": cfg_digest,
            "detection_result": detection_result,
        }
        trace_digest = digests.canonical_sha256(trace_payload)
        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": trace_digest,
        }
        return ContentEvidence(
            status=cast(Any, status),
            score=score if status == "ok" else None,
            audit=audit,
            plan_digest=plan_digest,
            basis_digest=basis_digest,
            lf_trace_digest=lf_trace_digest,
            hf_trace_digest=hf_trace_digest,
            lf_score=lf_score if status == "ok" else None,
            hf_score=hf_score if status == "ok" else None,
            score_parts=score_parts,
            trajectory_evidence=trajectory_evidence,
            lf_statistics_digest=lf_statistics_digest,
            hf_statistics_digest=hf_statistics_digest,
            content_failure_reason=content_failure_reason,
        )


def _extract_optional_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return cast(Dict[str, Any], value)
    return None


def _resolve_expected_plan_digest(inputs: Dict[str, Any]) -> Optional[str]:
    for key in ["expected_plan_digest", "plan_digest_expected", "plan_digest"]:
        value = inputs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _resolve_observed_plan_digest(inputs: Dict[str, Any]) -> Optional[str]:
    value = inputs.get("observed_plan_digest")
    if isinstance(value, str) and value:
        return value
    value = inputs.get("plan_digest")
    if isinstance(value, str) and value:
        return value
    return None


def _extract_plan_anchor(plan_payload: Any, field_name: str) -> Optional[str]:
    if not isinstance(field_name, str) or not field_name:
        raise TypeError("field_name must be non-empty str")
    if isinstance(plan_payload, dict):
        direct_value = plan_payload.get(field_name)
        if isinstance(direct_value, str) and direct_value:
            return direct_value
        nested_plan = plan_payload.get("plan")
        if isinstance(nested_plan, dict):
            nested_value = nested_plan.get(field_name)
            if isinstance(nested_value, str) and nested_value:
                return nested_value
    return None


def _extract_channel_evidence(
    evidence: Any,
    score_key: str,
    channel_name: str,
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    if channel_name not in {"lf", "hf"}:
        raise ValueError("channel_name must be lf or hf")

    if evidence is None:
        if channel_name == "hf":
            return "absent", None, "hf_disabled_by_config", None, {"hf_status": "absent", "hf_absent_reason": "hf_disabled_by_config"}
        return None, None, None, None, None

    if isinstance(evidence, dict):
        evidence_dict = cast(Dict[str, Any], evidence)
        status = evidence_dict.get("status")
        score = evidence_dict.get(score_key)
        if score is None:
            score = evidence_dict.get("score")
        failure_reason = evidence_dict.get("content_failure_reason")
        trace_digest = evidence_dict.get(f"{channel_name}_trace_digest")
        summary: Optional[Dict[str, Any]] = evidence_dict
        if channel_name == "hf":
            summary_node = evidence_dict.get("hf_evidence_summary")
            if isinstance(summary_node, dict):
                summary = cast(Dict[str, Any], summary_node)
                if status == "absent":
                    reason = summary.get("hf_absent_reason")
                    if isinstance(reason, str) and reason:
                        failure_reason = reason
                if status in {"failed", "mismatch"}:
                    reason = summary.get("hf_failure_reason")
                    if isinstance(reason, str) and reason:
                        failure_reason = reason
        if channel_name == "lf":
            summary_node = evidence_dict.get("lf_evidence_summary")
            if isinstance(summary_node, dict):
                summary = cast(Dict[str, Any], summary_node)
                if status == "absent":
                    reason = summary.get("lf_absent_reason")
                    if isinstance(reason, str) and reason:
                        failure_reason = reason
                if status in {"failed", "mismatch"}:
                    reason = summary.get("lf_failure_reason")
                    if isinstance(reason, str) and reason:
                        failure_reason = reason
        status_value = status if isinstance(status, str) else None
        trace_digest_value = trace_digest if isinstance(trace_digest, str) else None
        failure_reason_value = failure_reason if isinstance(failure_reason, str) else None
        return status_value, _safe_float(score), failure_reason_value, trace_digest_value, summary

    status = getattr(evidence, "status", None)
    score = getattr(evidence, score_key, None)
    if score is None:
        score = getattr(evidence, "score", None)
    failure_reason = getattr(evidence, "content_failure_reason", None)
    trace_digest = getattr(evidence, f"{channel_name}_trace_digest", None)
    return (
        status if isinstance(status, str) else None,
        _safe_float(score),
        failure_reason if isinstance(failure_reason, str) else None,
        trace_digest if isinstance(trace_digest, str) else None,
        None,
    )


def _extract_statistics_digest(inputs: Dict[str, Any], channel: str) -> Optional[str]:
    statistics = _extract_statistics_from_trajectory(inputs, channel)
    digest_value = statistics.get("statistics_digest") if isinstance(statistics, dict) else None
    return digest_value if isinstance(digest_value, str) and digest_value else None


def _extract_statistics_from_trajectory(inputs: Dict[str, Any], channel: str) -> Dict[str, Any]:
    if not isinstance(inputs, dict):
        return {"status": "absent", "reason": "invalid_inputs", "statistics_digest": None}
    if channel not in {"lf", "hf"}:
        raise ValueError("channel must be 'lf' or 'hf'")

    trajectory = inputs.get("trajectory_evidence")
    if not isinstance(trajectory, dict):
        return {"status": "absent", "reason": "trajectory_missing", "statistics_digest": None}
    if trajectory.get("status") != "ok":
        return {"status": "absent", "reason": "trajectory_not_ok", "statistics_digest": None}

    trajectory_metrics = trajectory.get("trajectory_metrics")
    if not isinstance(trajectory_metrics, dict):
        trajectory_metrics = trajectory.get("trajectory_stats")
    if not isinstance(trajectory_metrics, dict):
        return {"status": "absent", "reason": "trajectory_metrics_missing", "statistics_digest": None}

    steps = trajectory_metrics.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return {"status": "absent", "reason": "trajectory_steps_missing", "statistics_digest": None}

    mean_values = []
    std_values = []
    l2_values = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        stats = item.get("stats")
        if not isinstance(stats, dict):
            continue
        mean_val = _safe_float(stats.get("mean"))
        std_val = _safe_float(stats.get("std"))
        l2_val = _safe_float(stats.get("l2_norm"))
        if mean_val is not None:
            mean_values.append(abs(mean_val))
        if std_val is not None:
            std_values.append(std_val)
        if l2_val is not None:
            l2_values.append(l2_val)

    if len(std_values) == 0 or len(l2_values) == 0:
        return {"status": "absent", "reason": "trajectory_stats_invalid", "statistics_digest": None}

    mean_abs = float(sum(mean_values) / len(mean_values)) if mean_values else 0.0
    mean_std = float(sum(std_values) / len(std_values))
    mean_l2 = float(sum(l2_values) / len(l2_values))
    std_l2 = 0.0
    if len(l2_values) > 1:
        center = mean_l2
        std_l2 = float((sum((value - center) * (value - center) for value in l2_values) / len(l2_values)) ** 0.5)

    payload = {
        "channel": channel,
        "step_count": len(l2_values),
        "mean_abs": round(mean_abs, 8),
        "mean_std": round(mean_std, 8),
        "mean_l2": round(mean_l2, 8),
        "std_l2": round(std_l2, 8),
    }
    payload["statistics_digest"] = digests.canonical_sha256(payload)
    payload["status"] = "ok"
    payload["reason"] = None
    return payload


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not valid score")
    if not isinstance(value, (int, float)):
        raise TypeError("score must be float-like")
    return float(value)


def _validate_non_negative_score(score: Optional[float], score_name: str) -> None:
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")
    if score is None:
        return
    if score < 0:
        raise ValueError(f"{score_name} must be non-negative")


class UnifiedContentExtractor(_UnifiedContentExtractorBase):
    """
    功能：统一内容链提取器（稳定公共类，绑定 unified_content_extractor impl_id）。

    Stable public unified content extractor. All extraction logic is implemented
    in _UnifiedContentExtractorBase; this class provides the canonical public symbol.

    Args:
        impl_id: Implementation identifier string (must be unified_content_extractor).
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If constructor arguments are invalid.
    """

    pass

