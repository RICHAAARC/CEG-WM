"""
融合决策真实规则实现

功能说明：
- 实现 Neyman-Pearson 主链控制 + 几何辅助融合规则。
- 包含 NP 阈值选择（read-only from artifact）、rescue band 机制（几何门控增益）、唯一决策输出。
- 定义故障语义（abort/mismatch/fail）与审计摘要纪律（fusion_rule_digest、rescue_band_version）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from main.core import digests
from main.watermarking.fusion import neyman_pearson
from main.watermarking.fusion.interfaces import FusionDecision


# 融合规则实现的 impl_id 定义
FUSION_RULE_ID = "fusion_neyman_pearson_v1"
FUSION_RULE_VERSION = "v1"

# Rescue band 版本标记（用于审计可重现性）
RESCUE_BAND_VERSION = "v1"


def select_np_threshold_from_artifact(thresholds_artifact: Dict[str, Any]) -> float:
    """
    功能：从阈值工件中只读提取 NP 阈值。

    Select NP threshold value from thresholds artifact in readonly mode.

    Args:
        thresholds_artifact: Threshold artifact mapping.

    Returns:
        Selected threshold value.

    Raises:
        TypeError: If thresholds_artifact type is invalid.
        ValueError: If threshold_value is missing or invalid.
    """
    if not isinstance(thresholds_artifact, dict):
        # thresholds_artifact 类型不合法，必须 fail-fast。
        raise TypeError("thresholds_artifact must be dict")
    threshold_value = thresholds_artifact.get("threshold_value")
    if not isinstance(threshold_value, (int, float)):
        # 阈值工件缺少合法 threshold_value，必须 fail-fast。
        raise ValueError("thresholds_artifact.threshold_value must be number")
    return float(threshold_value)


def get_np_threshold(cfg: Dict[str, Any], thresholds_spec: Dict[str, Any]) -> Tuple[float, str, bool]:
    """
    功能：解析 NP 阈值并返回来源标签。

    Resolve NP threshold with strict artifact binding and test-only fallback gate.

    Args:
        cfg: Runtime configuration mapping.
        thresholds_spec: Deterministic thresholds spec mapping.

    Returns:
        Tuple of (threshold_value, threshold_source, fallback_enabled_for_tests).

    Raises:
        TypeError: If cfg or thresholds_spec type is invalid.
        ValueError: If threshold cannot be resolved under current policy.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(thresholds_spec, dict):
        # thresholds_spec 类型不合法，必须 fail-fast。
        raise TypeError("thresholds_spec must be dict")

    thresholds_artifact = cfg.get("__thresholds_artifact__")
    if isinstance(thresholds_artifact, dict):
        # 阈值来自 NP 校准工件时，threshold_source 标记为 np_canonical，
        # 与冻结决策门 threshold_source=np_canonical 约束对齐。
        return select_np_threshold_from_artifact(thresholds_artifact), "np_canonical", False

    fallback_enabled_for_tests = bool(cfg.get("allow_threshold_fallback_for_tests", False))
    if not fallback_enabled_for_tests:
        # 生产默认禁用 fallback，未提供工件时必须 fail-fast。
        raise ValueError(
            "np threshold artifact is required; set __thresholds_artifact__ or explicitly enable allow_threshold_fallback_for_tests"
        )

    target_fpr = thresholds_spec.get("target_fpr")
    if not isinstance(target_fpr, (int, float)):
        # fallback 被允许但 target_fpr 非法，仍然必须 fail-fast。
        raise ValueError("thresholds_spec.target_fpr must be number when fallback is enabled")
    return float(target_fpr), "fallback_target_fpr_test_only", True


class NeumanPearsonFusionRule:
    """
    功能：Neyman-Pearson 主链控制的融合规则实现。

    Fusion rule combining content evidence (NP-controlled) with geometry evidence (auxiliary/rescue).
    Implements deterministic decision logic with explicit threshold binding and rescue band audit.

    Args:
        impl_id: Implementation identifier string.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If any input is invalid.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 输入不合法，必须 fail-fast。
            raise ValueError("impl_id must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            # impl_version 输入不合法，必须 fail-fast。
            raise ValueError("impl_version must be non-empty str")
        if not isinstance(impl_digest, str) or not impl_digest:
            # impl_digest 输入不合法，必须 fail-fast。
            raise ValueError("impl_digest must be non-empty str")

        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest

    def fuse(
        self,
        cfg: Dict[str, Any],
        content_evidence: Dict[str, Any],
        geometry_evidence: Dict[str, Any]
    ) -> FusionDecision:
        """
        功能：融合内容与几何证据，输出唯一融合决策。

        Fuse content and geometry evidence with NP primary + geometry auxiliary logic.

        Args:
            cfg: Configuration mapping containing target_fpr and other threshold parameters.
            content_evidence: Content evidence mapping with content_score, status, and failure reasons.
            geometry_evidence: Geometry evidence mapping with geo_score, status.

        Returns:
            FusionDecision instance with decision_status ∈ {"decided", "abstain", "error"} and
            complete audit trail including fusion_rule_digest, rescue_band_version if applicable.

        Raises:
            TypeError: If inputs are not dict.
            ValueError: If evidence fields are invalid.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if not isinstance(content_evidence, dict):
            # content_evidence 类型不合法，必须 fail-fast。
            raise TypeError("content_evidence must be dict")
        if not isinstance(geometry_evidence, dict):
            # geometry_evidence 类型不合法，必须 fail-fast。
            raise TypeError("geometry_evidence must be dict")

        # (1) 构造 thresholds_spec 并计算摘要（NP 主控）
        thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
        thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)

        # (2) 验证融合输入：检查必需字段与单一主原因
        valid, primary_failure_reason = validate_fusion_inputs(content_evidence, geometry_evidence)
        if not valid:
            # 输入验证失败，返回 error 状态
            evidence_summary = {
                "content_score": None,
                "geometry_score": geometry_evidence.get("geo_score"),
                "content_status": content_evidence.get("status", "absent"),
                "geometry_status": geometry_evidence.get("status", "absent"),
                "fusion_rule_id": self.impl_id
            }
            audit = {
                "impl_id": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "decision_status": "error",
                "failure_reason": primary_failure_reason or "invalid_inputs"
            }
            return FusionDecision(
                is_watermarked=None,
                decision_status="error",
                thresholds_digest=thresholds_digest,
                evidence_summary=evidence_summary,
                audit=audit
            )

        # (3) 提取证据分数与状态
        content_status = content_evidence.get("status", "absent")
        geometry_status = geometry_evidence.get("status", "absent")
        content_score = content_evidence.get("content_score")
        geometry_score = geometry_evidence.get("geo_score")

        # (4) NP 主链：检查内容证据可决策性
        if content_status == "absent":
            # 无内容证据，禁止决策，返回 abstain
            evidence_summary = {
                "content_score": None,
                "geometry_score": geometry_score,
                "content_status": "absent",
                "geometry_status": geometry_status,
                "fusion_rule_id": self.impl_id
            }
            audit = {
                "impl_id": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "decision_status": "abstain",
                "reason": "content_evidence_absent"
            }
            return FusionDecision(
                is_watermarked=None,
                decision_status="abstain",
                thresholds_digest=thresholds_digest,
                evidence_summary=evidence_summary,
                audit=audit
            )

        # (5) NP 阈值选择：默认强制 artifact 绑定，fallback 仅测试显式开关可用
        thresholds_artifact = cfg.get("__thresholds_artifact__")
        np_threshold, threshold_source, fallback_enabled_for_tests = get_np_threshold(cfg, thresholds_spec)

        # (6) NP 主决策：内容分数 vs 阈值
        if not isinstance(content_score, (int, float)):
            # content_score 不合法，必须 fail-fast。
            raise ValueError(f"content_score must be number, got {type(content_score)}")

        content_decision = content_score >= np_threshold
        rescue_band_applied = False
        geo_gate_applied = False
        rescue_blocked_reason = None
        rescue_anchor_evidence_level = geometry_evidence.get("anchor_evidence_level", "proxy")
        rescue_sync_status = geometry_evidence.get("sync_status", geometry_status)
        rescue_sync_status_normalized = "ok" if rescue_sync_status in {"ok", "synced"} else rescue_sync_status

        # (7) 几何辅助（rescue band）：单侧救回策略
        # 口径修正：仅允许救回 False → True，禁止翻转 True → False
        rescue_band_spec = _build_rescue_band_spec(cfg)
        if (geometry_status == "ok" and 
            isinstance(geometry_score, (int, float)) and 
            not content_decision):  # NP 决策为 False，才考虑救回

            # 新增：几何可信度门控（proxy 几何默认不可信，除非 sync_status 已为 ok）。
            geo_trusted = (rescue_anchor_evidence_level != "proxy") or (rescue_sync_status_normalized == "ok")
            if not geo_trusted:
                rescue_blocked_reason = "proxy_geometry_not_trusted"
            else:
                # 检查是否在 rescue band 范围内（下界）
                if _is_rescue_candidate(content_score, np_threshold, rescue_band_spec):
                    # 检查几何门控条件
                    geo_gate_applied = _check_geo_gate(content_score, geometry_score, rescue_band_spec)
                    if geo_gate_applied:
                        # 几何门控通过：救回为 True（单侧）
                        rescue_band_applied = True
                        content_decision = True

        # (8) 计算融合规则摘要
        # 提取 target_fpr（用于摘要计算）
        target_fpr_value = None
        if thresholds_artifact is not None and isinstance(thresholds_artifact, dict):
            target_fpr_value = thresholds_artifact.get("target_fpr")
        if target_fpr_value is None:
            target_fpr_value = thresholds_spec.get("target_fpr")
        if target_fpr_value is None:
            target_fpr_value = 0.1  # 默认值，用于向后兼容
        
        fusion_rule_payload = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "rule_version": FUSION_RULE_VERSION,
            "np_threshold": float(np_threshold),
            "target_fpr": float(target_fpr_value),
            "rescue_band_version": RESCUE_BAND_VERSION if rescue_band_applied else None,
            "geo_gate_applied": geo_gate_applied,
            "allow_threshold_fallback_for_tests": fallback_enabled_for_tests,
            "rescue_blocked_reason": rescue_blocked_reason,
            "rescue_anchor_evidence_level": rescue_anchor_evidence_level,
            "rescue_sync_status": rescue_sync_status,
            "rescue_sync_status_normalized": rescue_sync_status_normalized,
        }
        fusion_rule_digest = compute_fusion_rule_digest(fusion_rule_payload)

        # (9) 封装最终决策与审计
        evidence_summary = {
            "content_score": content_score,
            "geometry_score": geometry_score,
            "content_status": content_status,
            "geometry_status": geometry_status,
            "fusion_rule_id": self.impl_id
        }
        audit = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "decision_status": "decided",
            "threshold_source": threshold_source,
            "np_threshold": float(np_threshold),  # 兼容旧字段名
            "used_threshold_value": float(np_threshold),
            "used_threshold_id": thresholds_artifact.get("threshold_id") if thresholds_artifact else None,
            "target_fpr": thresholds_artifact.get("target_fpr") if thresholds_artifact else target_fpr_value,
            "allow_threshold_fallback_for_tests": fallback_enabled_for_tests,
            "np_primary_decision": content_decision if not rescue_band_applied else (not content_decision),
            "content_decision": content_decision,
            "rescue_triggered": rescue_band_applied,
            "rescue_reason": "rescued_by_geo_gate" if rescue_band_applied else None,
            "rescue_band_version": RESCUE_BAND_VERSION if rescue_band_applied else None,
            "geo_gate_applied": geo_gate_applied,
            "rescue_blocked_reason": rescue_blocked_reason,
            "rescue_anchor_evidence_level": rescue_anchor_evidence_level,
            "rescue_sync_status": rescue_sync_status,
            "rescue_sync_status_normalized": rescue_sync_status_normalized,
            "fusion_rule_digest": fusion_rule_digest
        }

        return FusionDecision(
            is_watermarked=content_decision,
            decision_status="decided",
            thresholds_digest=thresholds_digest,
            evidence_summary=evidence_summary,
            audit=audit,
            fusion_rule_version=self.impl_version,
            used_threshold_id=thresholds_artifact.get("threshold_id") if thresholds_artifact else None,
            routing_decisions={
                "np_primary_decision": audit.get("np_primary_decision"),
                "rescue_triggered": rescue_band_applied,
                "rescue_reason": audit.get("rescue_reason"),
                "geo_gate_applied": geo_gate_applied
            },
            routing_digest=fusion_rule_digest
        )


def validate_fusion_inputs(
    content_evidence: Dict[str, Any],
    geometry_evidence: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    功能：验证融合输入，返回单一主失败原因。

    Validate fusion inputs with explicit priority: content_status > geometry_status > invalid.

    Args:
        content_evidence: Content evidence mapping.
        geometry_evidence: Geometry evidence mapping.

    Returns:
        Tuple of (valid: bool, primary_reason: Optional[str]).
        If valid, primary_reason is None.
        If invalid, primary_reason is single primary failure token.

    Raises:
        TypeError: If inputs are not dict.
    """
    if not isinstance(content_evidence, dict):
        # content_evidence 类型不合法，必须 fail-fast。
        raise TypeError("content_evidence must be dict")
    if not isinstance(geometry_evidence, dict):
        # geometry_evidence 类型不合法，必须 fail-fast。
        raise TypeError("geometry_evidence must be dict")

    # (1) 检查内容证据状态优先级
    content_status = content_evidence.get("status", "absent")
    if content_status == "mismatch":
        return False, "content_mismatch"
    if content_status == "fail":
        return False, "content_fail"

    # (2) 检查几何证据状态
    geometry_status = geometry_evidence.get("status", "absent")
    if geometry_status == "mismatch":
        return False, "geometry_mismatch"
    if geometry_status == "fail":
        return False, "geometry_fail"

    # (3) 有效状态组合：content 与 geometry 都在 {absent, ok}
    if content_status not in {"absent", "ok"}:
        return False, "invalid_content_status"
    if geometry_status not in {"absent", "ok"}:
        return False, "invalid_geometry_status"

    return True, None


def _build_rescue_band_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：从 cfg 构造 rescue band 规范。

    Build rescue band spec from config with conservative defaults.

    Args:
        cfg: Configuration mapping.

    Returns:
        Rescue band spec mapping with gate parameters.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    # 默认 rescue band：阈值 ±5% 范围内
    base_threshold = cfg.get("rescue_band_base_threshold", 0.5)
    delta_low = cfg.get("rescue_band_delta_low", 0.05)
    delta_high = cfg.get("rescue_band_delta_high", 0.05)

    spec = {
        "version": RESCUE_BAND_VERSION,
        "base_threshold": float(base_threshold),
        "delta_low": float(delta_low),
        "delta_high": float(delta_high),
        "geo_gate_lower": cfg.get("geo_gate_lower", 0.3),
        "geo_gate_upper": cfg.get("geo_gate_upper", 0.7)
    }
    return spec


def _is_rescue_candidate(
    content_score: float,
    np_threshold: float,
    rescue_band_spec: Dict[str, Any]
) -> bool:
    """
    功能：检查 content_score 是否在 rescue band 下界范围内（候选救回区间）。

    Determine if content_score is in rescue candidate zone (below threshold but within delta range).

    Args:
        content_score: Content score value.
        np_threshold: NP threshold value.
        rescue_band_spec: Rescue band spec mapping.

    Returns:
        bool indicating presence in rescue candidate zone.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(content_score, (int, float)):
        # content_score 类型不合法，必须 fail-fast。
        raise TypeError("content_score must be number")
    if not isinstance(np_threshold, (int, float)):
        # np_threshold 类型不合法，必须 fail-fast。
        raise TypeError("np_threshold must be number")
    if not isinstance(rescue_band_spec, dict):
        # rescue_band_spec 类型不合法，必须 fail-fast。
        raise TypeError("rescue_band_spec must be dict")

    delta_low = rescue_band_spec.get("delta_low", 0.05)
    
    # 仅检查下界区间：[threshold-delta, threshold)
    # 禁止上界翻转（content_score >= threshold 不允许救回）
    lower_bound = np_threshold - delta_low
    
    return lower_bound <= content_score < np_threshold


def _check_geo_gate(
    content_score: float,
    geometry_score: float,
    rescue_band_spec: Dict[str, Any]
) -> bool:
    """
    功能：检查几何门控条件。

    Check if geometry evidence passes gate condition for rescue band application.

    Args:
        content_score: Content score value.
        geometry_score: Geometry score value (normalized [0,1]).
        rescue_band_spec: Rescue band spec mapping.

    Returns:
        bool indicating whether geo_gate allows rescue band application.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(content_score, (int, float)):
        # content_score 类型不合法，必须 fail-fast。
        raise TypeError("content_score must be number")
    if not isinstance(geometry_score, (int, float)):
        # geometry_score 类型不合法，必须 fail-fast。
        raise TypeError("geometry_score must be number")
    if not isinstance(rescue_band_spec, dict):
        # rescue_band_spec 类型不合法，必须 fail-fast。
        raise TypeError("rescue_band_spec must be dict")

    # 几何门控：geo_score 应在 [geo_gate_lower, geo_gate_upper] 范围内
    geo_gate_lower = rescue_band_spec.get("geo_gate_lower", 0.3)
    geo_gate_upper = rescue_band_spec.get("geo_gate_upper", 0.7)

    if not (0 <= geometry_score <= 1):
        # geometry_score 归一化范围不合法，必须 fail-fast。
        raise ValueError(f"geometry_score must be in [0,1], got {geometry_score}")

    return geo_gate_lower <= geometry_score <= geo_gate_upper


def compute_fusion_rule_digest(payload: Dict[str, Any]) -> str:
    """
    功能：计算 fusion_rule_digest。

    Compute fusion rule digest using canonical semantic digest.

    Args:
        payload: Fusion rule parameters mapping.

    Returns:
        Canonical digest string (hex64).

    Raises:
        TypeError: If payload is not dict.
        ValueError: If digest output is invalid.
    """
    if not isinstance(payload, dict):
        # payload 类型不合法，必须 fail-fast。
        raise TypeError("payload must be dict")

    digests.normalize_for_digest(payload)
    digest_value = digests.semantic_digest(payload)

    if not isinstance(digest_value, str) or not digest_value:
        # 摘要输出不合法，必须 fail-fast。
        raise ValueError("fusion_rule_digest must be non-empty str")

    return digest_value
