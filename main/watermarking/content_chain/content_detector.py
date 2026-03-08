"""
内容链检测器实现

功能说明：
- 统一从 LF 和 HF 子空间读取证据并输出内容链得分。
- 保持 LF/HF 正交：HF disabled 不影响 LF 主链得分。
- 严格处理 mismatch/failed：任何阻断态下 score=None。
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple, cast

from main.core import digests

from .high_freq_embedder import CONTENT_SCORE_RULE_VERSION
from .interfaces import ContentEvidence


CONTENT_DETECTOR_ID = "content_detector_v1"
CONTENT_DETECTOR_VERSION = "v1"
CONTENT_DETECTOR_TRACE_VERSION = "v2"

ALLOWED_DETECTOR_FAILURE_REASONS = {
    "detector_no_plan",
    "detector_no_plan_expected",
    "detector_plan_mismatch",
    "detector_no_evidence",
    "detector_invalid_input",
    "detector_extraction_failed",
    "detector_score_validation_failed",
    "hf_plan_mismatch",
    "hf_detection_failed",
    "hf_subspace_missing",
}

HF_ABSENT_ALLOWLIST = {
    "hf_disabled_by_config",
}

DETECTOR_STATUS = Literal["ok", "absent", "failed", "mismatch"]


class ContentDetector:
    """
    功能：内容链检测器。

    Unified content detector that combines LF and HF evidence under frozen
    failure semantics and deterministic score rule.

    Args:
        impl_id: Implementation identifier string.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If any constructor argument is invalid.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        if not impl_id:
            raise ValueError("impl_id must be non-empty str")
        if not impl_version:
            raise ValueError("impl_version must be non-empty str")
        if not impl_digest:
            raise ValueError("impl_digest must be non-empty str")

        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        cfg_digest: Optional[str] = None
    ) -> ContentEvidence:
        """
        功能：检测内容链水印并输出 content_score。

        Extract content evidence and compute deterministic content score.

        Args:
            cfg: Configuration mapping.
            inputs: Optional detector inputs including lf_evidence/hf_evidence/plan_digest.
            cfg_digest: Optional canonical config digest.

        Returns:
            ContentEvidence instance.

        Raises:
            TypeError: If input types are invalid.
        """
        enabled = cfg.get("detect", {}).get("content", {}).get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("detect.content.enabled must be bool")

        if not enabled:
            return self._build_result(
                cfg=cfg,
                status="absent",
                score=None,
                plan_digest=None,
                content_failure_reason=None,
                score_parts={
                    "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
                    "content_score_rule_id": "lf_only_when_hf_absent_v1",
                    "hf_status": "absent",
                    "hf_absent_reason": "hf_disabled_by_config",
                    "hf_score": "<absent>",
                    "lf_score": "<absent>",
                },
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
            )

        normalized_inputs: Dict[str, Any] = inputs or {}
        test_mode = bool(normalized_inputs.get("test_mode", False))
        disable_cfg_plan_digest_fallback = bool(normalized_inputs.get("disable_cfg_plan_digest_fallback", False))
        expected_plan_digest = normalized_inputs.get("expected_plan_digest")
        observed_plan_digest = normalized_inputs.get("observed_plan_digest")
        if expected_plan_digest is None:
            legacy_expected = normalized_inputs.get("plan_digest_expected")
            if isinstance(legacy_expected, str) and legacy_expected:
                expected_plan_digest = legacy_expected

        if expected_plan_digest is None and test_mode and not disable_cfg_plan_digest_fallback:
            cfg_plan_digest = cfg.get("watermark", {}).get("plan_digest")
            if isinstance(cfg_plan_digest, str) and cfg_plan_digest:
                expected_plan_digest = cfg_plan_digest

        if expected_plan_digest is None:
            legacy_plan_digest = normalized_inputs.get("plan_digest")
            if isinstance(legacy_plan_digest, str) and legacy_plan_digest:
                expected_plan_digest = legacy_plan_digest
                if observed_plan_digest is None:
                    observed_plan_digest = legacy_plan_digest

        if not isinstance(expected_plan_digest, str) or not expected_plan_digest:
            return self._build_result(
                cfg=cfg,
                status="absent",
                score=None,
                plan_digest=None,
                content_failure_reason="detector_no_plan_expected",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
            )

        if observed_plan_digest is None:
            legacy_observed = normalized_inputs.get("plan_digest")
            if isinstance(legacy_observed, str) and legacy_observed:
                observed_plan_digest = legacy_observed

        if observed_plan_digest is not None and observed_plan_digest != expected_plan_digest:
            return self._build_result(
                cfg=cfg,
                status="mismatch",
                score=None,
                plan_digest=expected_plan_digest,
                content_failure_reason="detector_plan_mismatch",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
            )

        lf_status, lf_score, lf_failure_reason = _extract_channel(
            evidence=normalized_inputs.get("lf_evidence"),
            score_key="lf_score",
            default_score=normalized_inputs.get("lf_score"),
            channel_name="lf"
        )
        lf_statistics = extract_low_freq_statistics(normalized_inputs)
        lf_statistics_digest = lf_statistics.get("statistics_digest") if isinstance(lf_statistics, dict) else None
        if lf_score is None and isinstance(lf_statistics, dict) and lf_statistics.get("status") == "ok":
            lf_score = compute_lf_score(
                lf_statistics=lf_statistics,
                injection_evidence=normalized_inputs.get("injection_evidence"),
                lf_evidence=normalized_inputs.get("lf_evidence")
            )
            lf_status = "ok"
            lf_failure_reason = None

        hf_enabled = bool(cfg.get("watermark", {}).get("hf", {}).get("enabled", False))
        if hf_enabled:
            hf_status, hf_score, hf_failure_reason = _extract_channel(
                evidence=normalized_inputs.get("hf_evidence"),
                score_key="hf_score",
                default_score=normalized_inputs.get("hf_score"),
                channel_name="hf"
            )
            # 【P0-B】_extract_channel() 的 dict 分支忽略 default_score，
            # 当 hf_evidence dict 不含 hf_score 键时返回 None。
            # 若图像域旁路分数（detector_inputs["hf_score"] = hf_score_raw）可用，
            # 则以 max(0.0, hf_score_raw) 直接作为主评分，跳过 compute_hf_score()
            # 的 SD3 轨迹 L2 路径（轨迹 L2 均值 ≈262，对所有 SD3 图像无判别性）。
            if hf_score is None:
                hf_raw = normalized_inputs.get("hf_score")
                if isinstance(hf_raw, (int, float)) and not isinstance(hf_raw, bool):
                    hf_score = max(0.0, float(hf_raw))
                    hf_status = "ok"
                    hf_failure_reason = None
            hf_statistics = extract_high_freq_statistics(normalized_inputs)
            hf_statistics_digest = hf_statistics.get("statistics_digest") if isinstance(hf_statistics, dict) else None
            if hf_score is None and isinstance(hf_statistics, dict) and hf_statistics.get("status") == "ok":
                hf_score = compute_hf_score(
                    hf_statistics=hf_statistics,
                    injection_evidence=normalized_inputs.get("injection_evidence"),
                    hf_evidence=normalized_inputs.get("hf_evidence")
                )
                hf_status = "ok"
                hf_failure_reason = None
        else:
            hf_status, hf_score, hf_failure_reason = "absent", None, "hf_disabled_by_config"
            hf_statistics_digest = None

        try:
            _validate_non_negative_score(lf_score, "lf_score")
            _validate_non_negative_score(hf_score, "hf_score")
        except Exception:
            return self._build_result(
                cfg=cfg,
                status="failed",
                score=None,
                plan_digest=expected_plan_digest,
                content_failure_reason="detector_score_validation_failed",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
            )

        if lf_status in {"mismatch", "failed", "absent"}:
            propagated_reason = lf_failure_reason
            if lf_status == "mismatch" and not isinstance(propagated_reason, str):
                propagated_reason = "detector_plan_mismatch"
            if lf_status == "failed" and not isinstance(propagated_reason, str):
                propagated_reason = "detector_extraction_failed"
            return self._build_result(
                cfg=cfg,
                status=lf_status,
                score=None,
                plan_digest=expected_plan_digest,
                content_failure_reason=propagated_reason,
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
            )

        if hf_status == "mismatch":
            return self._build_result(
                cfg=cfg,
                status="mismatch",
                score=None,
                plan_digest=expected_plan_digest,
                content_failure_reason=hf_failure_reason or "hf_plan_mismatch",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
            )

        if hf_status == "failed":
            return self._build_result(
                cfg=cfg,
                status="failed",
                score=None,
                plan_digest=expected_plan_digest,
                content_failure_reason=hf_failure_reason or "hf_detection_failed",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
            )

        if lf_score is None and hf_score is None:
            return self._build_result(
                cfg=cfg,
                status="failed",
                score=None,
                plan_digest=expected_plan_digest,
                content_failure_reason="detector_no_evidence",
                score_parts=None,
                lf_score=None,
                hf_score=None,
                cfg_digest=cfg_digest,
                detection_result=None,
            )

        content_score, score_parts = self._compose_content_score(
            lf_score=lf_score,
            hf_score=hf_score,
            hf_status=hf_status,
            hf_failure_reason=hf_failure_reason,
        )

        detection_result_typed: Dict[str, Any] = {
            "plan_digest": expected_plan_digest,
            "lf_score": lf_score,
            "hf_score": hf_score,
            "content_score": content_score,
            "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
            "cfg_digest": cfg_digest,
        }

        return self._build_result(
            cfg=cfg,
            status="ok",
            score=content_score,
            plan_digest=expected_plan_digest,
            content_failure_reason=None,
            score_parts=score_parts,
            lf_score=lf_score,
            hf_score=hf_score,
            cfg_digest=cfg_digest,
            detection_result=detection_result_typed,
            lf_statistics_digest=lf_statistics_digest,
            hf_statistics_digest=hf_statistics_digest,
        )

    def _compose_content_score(
        self,
        lf_score: Optional[float],
        hf_score: Optional[float],
        hf_status: Optional[str],
        hf_failure_reason: Optional[str],
    ) -> Tuple[float, Dict[str, Any]]:
        if lf_score is None:
            raise ValueError("lf_score must be non-None when composing content score")

        if hf_status == "absent":
            hf_reason = hf_failure_reason if isinstance(hf_failure_reason, str) else "hf_disabled_by_config"
            score_parts: Dict[str, Any] = {
                "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
                "rule_id": "lf_only_when_hf_absent_v1",
                "content_score_rule_id": "lf_only_when_hf_absent_v1",
                "lf_score": lf_score,
                "hf_score": "<absent>",
                "hf_status": "absent",
                "hf_absent_reason": hf_reason,
            }
            return float(lf_score), score_parts

        if hf_score is None:
            score_parts: Dict[str, Any] = {
                "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
                "rule_id": "lf_only_default_v1",
                "content_score_rule_id": "lf_only_default_v1",
                "lf_score": lf_score,
                "hf_score": "<absent>",
                "hf_status": "absent",
                "hf_absent_reason": "hf_disabled_by_config",
            }
            return float(lf_score), score_parts

        weight_lf = 0.7
        weight_hf = 0.3
        content_score = float(round(lf_score * weight_lf + hf_score * weight_hf, 8))
        score_parts: Dict[str, Any] = {
            "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
            "rule_id": "lf_hf_weighted_sum_v1",
            "content_score_rule_id": "lf_hf_weighted_sum_v1",
            "lf_score": lf_score,
            "hf_score": hf_score,
            "weights": {
                "lf": weight_lf,
                "hf": weight_hf,
            },
            "hf_status": "ok",
        }
        return content_score, score_parts

    def _build_result(
        self,
        cfg: Dict[str, Any],
        status: DETECTOR_STATUS,
        score: Optional[float],
        plan_digest: Optional[str],
        content_failure_reason: Optional[str],
        score_parts: Optional[Dict[str, Any]],
        lf_score: Optional[float],
        hf_score: Optional[float],
        cfg_digest: Optional[str],
        detection_result: Optional[Dict[str, Any]],
        lf_statistics_digest: Optional[str] = None,
        hf_statistics_digest: Optional[str] = None,
    ) -> ContentEvidence:
        trace_payload = _build_detector_trace_payload(
            cfg=cfg,
            impl_id=self.impl_id,
            impl_version=self.impl_version,
            impl_digest=self.impl_digest,
            enabled=True,
            plan_digest=plan_digest,
            detection_result=detection_result,
        )
        trace_digest = digests.canonical_sha256(trace_payload)

        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": trace_digest,
        }

        if status != "ok":
            return ContentEvidence(
                status=status,
                score=None,
                audit=audit,
                plan_digest=plan_digest,
                score_parts=score_parts,
                lf_score=None,
                hf_score=None,
                content_failure_reason=content_failure_reason,
                lf_statistics_digest=lf_statistics_digest,
                hf_statistics_digest=hf_statistics_digest,
            )

        return ContentEvidence(
            status="ok",
            score=score,
            audit=audit,
            plan_digest=plan_digest,
            score_parts=score_parts,
            lf_score=lf_score,
            hf_score=hf_score,
            content_failure_reason=None,
            lf_statistics_digest=lf_statistics_digest,
            hf_statistics_digest=hf_statistics_digest,
        )


def _extract_channel(
    evidence: Any,
    score_key: str,
    default_score: Any,
    channel_name: str,
) -> Tuple[Optional[DETECTOR_STATUS], Optional[float], Optional[str]]:
    if not score_key:
        raise TypeError("score_key must be non-empty str")
    if channel_name not in {"lf", "hf"}:
        raise ValueError("channel_name must be lf or hf")

    if evidence is None:
        if default_score is None:
            if channel_name == "hf":
                return "absent", None, "hf_disabled_by_config"
            return None, None, None
        return "ok", float(default_score), None

    if isinstance(evidence, dict):
        evidence_dict = cast(Dict[str, Any], evidence)
        status = evidence_dict.get("status")
        score = evidence_dict.get(score_key)
        if score is None:
            score = evidence_dict.get("score")
        failure_reason = evidence_dict.get("content_failure_reason")
        if channel_name == "hf":
            summary = evidence_dict.get("hf_evidence_summary")
            if isinstance(summary, dict):
                summary_dict = cast(Dict[str, Any], summary)
                if status == "absent":
                    reason = summary_dict.get("hf_absent_reason")
                    if isinstance(reason, str) and reason:
                        failure_reason = reason
                if status in {"failed", "mismatch"}:
                    reason = summary_dict.get("hf_failure_reason")
                    if isinstance(reason, str) and reason:
                        failure_reason = reason
        status_literal: Optional[DETECTOR_STATUS] = None
        if isinstance(status, str) and status in {"ok", "absent", "failed", "mismatch"}:
            status_literal = cast(DETECTOR_STATUS, status)
        failure_reason_text: Optional[str] = failure_reason if isinstance(failure_reason, str) else None
        return status_literal, _safe_float(score), failure_reason_text

    status = getattr(evidence, "status", None)
    score = getattr(evidence, score_key, None)
    if score is None:
        score = getattr(evidence, "score", None)
    failure_reason = getattr(evidence, "content_failure_reason", None)
    status_literal: Optional[DETECTOR_STATUS] = None
    if isinstance(status, str) and status in {"ok", "absent", "failed", "mismatch"}:
        status_literal = cast(DETECTOR_STATUS, status)
    failure_reason_text = failure_reason if isinstance(failure_reason, str) else None
    return status_literal, _safe_float(score), failure_reason_text


def extract_low_freq_statistics(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：提取 LF 同构统计摘要。

    Extract deterministic LF statistics from trajectory evidence.

    Args:
        inputs: Detector inputs mapping.

    Returns:
        LF statistics mapping with status and digest.
    """
    return _extract_statistics_from_trajectory(inputs, channel="lf")


def extract_high_freq_statistics(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：提取 HF 同构统计摘要。

    Extract deterministic HF statistics from trajectory evidence.

    Args:
        inputs: Detector inputs mapping.

    Returns:
        HF statistics mapping with status and digest.
    """
    return _extract_statistics_from_trajectory(inputs, channel="hf")


def compute_lf_score(
    lf_statistics: Dict[str, Any],
    injection_evidence: Any,
    lf_evidence: Any
) -> float:
    """
    功能：根据 LF 统计计算分数（越大表示证据更强）。

    Compute LF score from deterministic statistics.

    Args:
        lf_statistics: LF statistics mapping.
        injection_evidence: Injection evidence mapping.
        lf_evidence: Embed LF evidence mapping.

    Returns:
        Non-negative LF score.
    """
    if not isinstance(lf_statistics, dict):
        raise TypeError("lf_statistics must be dict")

    mean_std = _safe_float(lf_statistics.get("mean_std")) or 0.0
    mean_abs = _safe_float(lf_statistics.get("mean_abs")) or 0.0
    expected_delta = _extract_expected_delta(injection_evidence, "lf_delta_norm_mean")
    embed_hint = _extract_embed_hint(lf_evidence, "lf_score")

    observed = mean_std + mean_abs
    alignment = 1.0 / (1.0 + abs(observed - expected_delta))
    boost = 1.0 + max(0.0, embed_hint)
    return float(round(max(0.0, observed * alignment * boost), 8))


def compute_hf_score(
    hf_statistics: Dict[str, Any],
    injection_evidence: Any,
    hf_evidence: Any
) -> float:
    """
    功能：根据 HF 统计计算分数（越大表示证据更强）。

    Compute HF score from deterministic statistics.

    Args:
        hf_statistics: HF statistics mapping.
        injection_evidence: Injection evidence mapping.
        hf_evidence: Embed HF evidence mapping.

    Returns:
        Non-negative HF score.
    """
    if not isinstance(hf_statistics, dict):
        raise TypeError("hf_statistics must be dict")

    mean_l2 = _safe_float(hf_statistics.get("mean_l2")) or 0.0
    std_l2 = _safe_float(hf_statistics.get("std_l2")) or 0.0
    expected_delta = _extract_expected_delta(injection_evidence, "hf_delta_norm_mean")
    embed_hint = _extract_embed_hint(hf_evidence, "hf_score")

    observed = mean_l2 + std_l2
    alignment = 1.0 / (1.0 + abs(observed - expected_delta))
    boost = 1.0 + max(0.0, embed_hint)
    return float(round(max(0.0, observed * alignment * boost), 8))


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


def _extract_expected_delta(injection_evidence: Any, key_name: str) -> float:
    if not isinstance(injection_evidence, dict):
        return 0.0
    metrics = injection_evidence.get("injection_metrics")
    if not isinstance(metrics, dict):
        return 0.0
    value = _safe_float(metrics.get(key_name))
    return value if value is not None else 0.0


def _extract_embed_hint(evidence: Any, key_name: str) -> float:
    if not isinstance(evidence, dict):
        return 0.0
    value = _safe_float(evidence.get(key_name))
    return value if value is not None else 0.0


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not valid score")
    if not isinstance(value, (int, float)):
        raise TypeError("score must be float-like")
    return float(value)


def _validate_non_negative_score(score: Optional[float], score_name: str) -> None:
    if not score_name:
        raise TypeError("score_name must be non-empty str")
    if score is None:
        return
    if score < 0:
        raise ValueError(f"{score_name} must be non-negative")


def _build_detector_trace_payload(
    cfg: Dict[str, Any],
    impl_id: str,
    impl_version: str,
    impl_digest: str,
    enabled: bool,
    plan_digest: Optional[str],
    detection_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    功能：构造可复算的内容检测追踪有效负载。

    Build deterministic trace payload for detector digest computation.

    Args:
        cfg: Configuration mapping.
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.
        enabled: Whether detection is enabled.
        plan_digest: Optional plan digest binding.
        detection_result: Optional detection result and scores.

    Returns:
        JSON-like dict for canonical SHA256 computation.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not impl_id:
        raise TypeError("impl_id must be non-empty str")

    payload: Dict[str, Any] = {
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "trace_version": CONTENT_DETECTOR_TRACE_VERSION,
        "enabled": enabled,
        "plan_digest": plan_digest,
        "content_score_rule_version": CONTENT_SCORE_RULE_VERSION,
    }

    if detection_result is not None:
        payload["detection_result"] = detection_result

    return payload
