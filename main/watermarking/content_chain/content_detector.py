"""
File purpose: Content detector implementation for S-06.
Module type: Core innovation module

功能说明：
- 统一从 LF 和 HF 子空间检测水印信号，输出融合得分。
- 严格区分 absent（无计划）、mismatch（参数不一致）、failed（异常）语义。
- 输出 content_score、score_parts（hf_score/lf_score）、content_evidence、failure_reason（枚举）。
- plan_digest 不一致时必须返回 mismatch 状态，score=None。
- 失败路径单一主因上报，不得写入"看似有效的分数"。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from main.core import digests

from .interfaces import ContentEvidence


CONTENT_DETECTOR_ID = "content_detector_v1"
CONTENT_DETECTOR_VERSION = "v1"
CONTENT_DETECTOR_TRACE_VERSION = "v1"

# 允许的失败原因枚举。
ALLOWED_DETECTOR_FAILURE_REASONS = {
    "detector_no_plan",                   # 缺失 plan_digest，无法校验一致性
    "detector_plan_mismatch",             # plan_digest 不一致，检测参数与计划脱离
    "detector_no_evidence",               # 内容证据缺失，无法提取分数
    "detector_invalid_input",             # 输入缺失或形状不符
    "detector_extraction_failed",         # 分数提取异常
    "detector_score_validation_failed",   # 分数有效性校验失败
}


class ContentDetector:
    """
    功能：内容链检测器，统一输出融合得分与失败语义。

    Content detector implementing unified watermark score extraction from
    content evidence with strict failure semantics and plan digest binding.

    Implements the ContentExtractor protocol frozen in interfaces.py.
    Emits ContentEvidence with content_score, score_parts (lf_score/hf_score),
    and strict failure semantics (absent, failed, mismatch) matching frozen enumeration.

    Detector validates plan_digest consistency and rejects mismatches with
    status="mismatch" and score=None to prevent false detection.

    Args:
        impl_id: Implementation identifier string (frozen to content_detector_v1).
        impl_version: Implementation version string (frozen to v1).
        impl_digest: Implementation digest computed from source code.

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

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        cfg_digest: Optional[str] = None
    ) -> ContentEvidence:
        """
        功能：检测内容链水印信号并输出融合得分。

        Detect watermark signal and extract unified content score.

        Detector verifies plan_digest consistency; mismatch returns status="mismatch".
        Extracts lf_score and hf_score from content evidence components and
        fuses into single content_score.

        Three semantic modes:
        1. enable_detector=false: Returns absent status (abstinence).
        2. enable_detector=true, plan available: Extracts score from evidence.
        3. enable_detector=true, plan missing or mismatch: Returns status="mismatch" or "failed".

        When status="ok": score is non-negative float in [0.0, infinity).
        When status!="ok": score must be None, content_failure_reason populated.

        Args:
            cfg: Configuration dict with optional keys:
                - "enable_detector" (bool, default False): Whether to enable detection.
                - "detector_threshold" (float, default 0.5): Unused; for documentation.
                - "watermark" (dict, optional): Sub-configuration containing plan_digest.
                  - "plan_digest" (str, optional): Expected plan digest binding.

            inputs: Optional input dict with keys:
                - "lf_score" (float, optional): Low-frequency channel score.
                - "hf_score" (float, optional): High-frequency channel score.
                - "raw_evidence" (dict, optional): Raw detection output.

            cfg_digest: Optional canonical SHA256 digest of cfg.

        Returns:
            ContentEvidence instance with frozen structure.
                - status: "ok" (valid score), "absent" (disabled), "failed" (error),
                  "mismatch" (plan inconsistency).
                - score: Non-None float when status="ok"; None otherwise.
                - score_parts: Dict with "lf_score", "hf_score" when ok.
                - lf_score: Low-frequency score component (when ok).
                - hf_score: High-frequency score component (when ok).
                - plan_digest: Echoed from inputs for consistency audit.
                - audit: Dict with impl_identity, impl_version, impl_digest, trace_digest.
                - content_failure_reason: Enumeration string when status!="ok".

        Raises:
            TypeError: If cfg or inputs types are invalid.
            ValueError: If critical fields are malformed.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            # inputs 类型不合法，必须 fail-fast。
            raise TypeError("inputs must be dict or None")

        # (1) 解析启用状态。对齐 injection_scope_manifest.yaml 中 cfg_digest_include_paths 的 detect.content.enabled。
        enabled = cfg.get("detect", {}).get("content", {}).get("enabled", False)
        if not isinstance(enabled, bool):
            # enabled 类型不合法，必须 fail-fast。
            raise TypeError("detect.content.enabled must be bool")

        # 若禁用，返回 absent 语义（非错误）。
        if not enabled:
            trace_payload = _build_detector_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=False,
                plan_digest=None,
                detection_result=None
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest
            }

            return ContentEvidence(
                status="absent",
                score=None,
                audit=audit,
                plan_digest=None,
                content_failure_reason=None,
                score_parts=None,
                lf_score=None,
                hf_score=None
            )

        # (2) 初始化输入。
        if inputs is None:
            inputs = {}

        # (3) 解析计划摘要与校验一致性。
        plan_digest = cfg.get("watermark", {}).get("plan_digest")
        if plan_digest is None:
            # 无计划，无法确定检测子空间。
            trace_payload = _build_detector_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=None,
                detection_result=None
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest
            }

            return ContentEvidence(
                status="mismatch",
                score=None,
                audit=audit,
                plan_digest=None,
                content_failure_reason="detector_no_plan",
                score_parts=None,
                lf_score=None,
                hf_score=None
            )

        # 验证输入的 plan_digest 是否与配置一致。
        input_plan_digest = inputs.get("plan_digest")
        if input_plan_digest is not None and input_plan_digest != plan_digest:
            # 计划摘要不一致，产生 mismatch。
            trace_payload = _build_detector_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=plan_digest,
                detection_result=None
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest
            }

            return ContentEvidence(
                status="mismatch",
                score=None,
                audit=audit,
                plan_digest=plan_digest,
                content_failure_reason="detector_plan_mismatch",
                score_parts=None,
                lf_score=None,
                hf_score=None
            )

        # (4) 提取得分分量。
        try:
            lf_score = inputs.get("lf_score")
            hf_score = inputs.get("hf_score")

            # 验证得分类型。
            if lf_score is not None and not isinstance(lf_score, (int, float)):
                # lf_score 类型不合法，必须 fail-fast。
                raise TypeError(f"lf_score must be float or None, got {type(lf_score).__name__}")
            if hf_score is not None and not isinstance(hf_score, (int, float)):
                # hf_score 类型不合法，必须 fail-fast。
                raise TypeError(f"hf_score must be float or None, got {type(hf_score).__name__}")

            # 验证得分范围（非负）。
            if lf_score is not None and lf_score < 0:
                # lf_score 值域不合法，必须 fail-fast。
                raise ValueError(f"lf_score must be non-negative, got {lf_score}")
            if hf_score is not None and hf_score < 0:
                # hf_score 值域不合法，必须 fail-fast。
                raise ValueError(f"hf_score must be non-negative, got {hf_score}")

            # 若无任何证据分数，返回失败。
            if lf_score is None and hf_score is None:
                trace_payload = _build_detector_trace_payload(
                    cfg=cfg,
                    impl_id=self.impl_id,
                    impl_version=self.impl_version,
                    impl_digest=self.impl_digest,
                    enabled=True,
                    plan_digest=plan_digest,
                    detection_result=None
                )
                trace_digest = digests.canonical_sha256(trace_payload)

                audit = {
                    "impl_identity": self.impl_id,
                    "impl_version": self.impl_version,
                    "impl_digest": self.impl_digest,
                    "trace_digest": trace_digest
                }

                return ContentEvidence(
                    status="failed",
                    score=None,
                    audit=audit,
                    plan_digest=plan_digest,
                    content_failure_reason="detector_no_evidence",
                    score_parts=None,
                    lf_score=None,
                    hf_score=None
                )

            # (5) 融合得分（简化：占位实现，加权平均）。
            # 在真实实现中，此处应执行实际分数融合算法。
            if lf_score is not None and hf_score is not None:
                # 两路都可用：加权融合（占位：等权）。
                content_score = (lf_score + hf_score) / 2.0
            elif lf_score is not None:
                # 仅 LF 可用。
                content_score = lf_score
            else:
                # 仅 HF 可用。
                content_score = hf_score

        except (TypeError, ValueError, KeyError) as e:
            # 检测异常，单一主因上报。
            failure_reason = "detector_extraction_failed"

            trace_payload = _build_detector_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=plan_digest,
                detection_result=None
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest
            }

            return ContentEvidence(
                status="failed",
                score=None,
                audit=audit,
                plan_digest=plan_digest,
                content_failure_reason=failure_reason,
                score_parts=None,
                lf_score=None,
                hf_score=None
            )

        # (6) 构造成功路径审计字段与得分分量。
        detection_result = {
            "lf_score": lf_score,
            "hf_score": hf_score,
            "content_score": content_score,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest
        }

        trace_payload = _build_detector_trace_payload(
            cfg=cfg,
            impl_id=self.impl_id,
            impl_version=self.impl_version,
            impl_digest=self.impl_digest,
            enabled=True,
            plan_digest=plan_digest,
            detection_result=detection_result
        )
        trace_digest = digests.canonical_sha256(trace_payload)

        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": trace_digest
        }

        score_parts = {
            "lf_score": lf_score,
            "hf_score": hf_score
        }

        # (7) 返回成功证据。
        return ContentEvidence(
            status="ok",
            score=content_score,
            audit=audit,
            plan_digest=plan_digest,
            content_failure_reason=None,
            score_parts=score_parts,
            lf_score=lf_score,
            hf_score=hf_score
        )


def _build_detector_trace_payload(
    cfg: Dict[str, Any],
    impl_id: str,
    impl_version: str,
    impl_digest: str,
    enabled: bool,
    plan_digest: Optional[str],
    detection_result: Optional[Dict[str, Any]]
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
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 类型不合法，必须 fail-fast。
        raise TypeError("impl_id must be non-empty str")
    if not isinstance(enabled, bool):
        # enabled 类型不合法，必须 fail-fast。
        raise TypeError("enabled must be bool")
    if plan_digest is not None and not isinstance(plan_digest, str):
        # plan_digest 类型不合法，必须 fail-fast。
        raise TypeError("plan_digest must be str or None")
    if detection_result is not None and not isinstance(detection_result, dict):
        # detection_result 类型不合法，必须 fail-fast。
        raise TypeError("detection_result must be dict or None")

    detector_cfg = cfg.get("watermark", {}).get("detector", {})

    payload = {
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "trace_version": CONTENT_DETECTOR_TRACE_VERSION,
        "enabled": enabled,
        "plan_digest": plan_digest,
        # detector_threshold 已在 docstring 中标记为 unused，直接使用默认值。
        "detector_threshold": 0.5 if enabled else None,
    }

    if detection_result is not None:
        payload["detection_result"] = detection_result

    return payload
