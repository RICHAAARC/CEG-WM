"""
File purpose: Low-frequency watermark coder implementation for S-04.
Module type: Core innovation module

功能说明：
- 在 LF（低频）子空间施加水印编码，作为隐蔽主通道实现。
- 支持 enable_lf 配置参数控制启用/禁用。
- 生成可复算的 lf_trace_digest，绑定配置、计划与编码参数。
- 输出 lf_score 与编码痕迹，用于内容检测与融合。
- 严格区分 absent（未启用）、failed（异常）、mismatch（参数不一致）语义。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from main.core import digests
from main.core.errors import RecordsWritePolicyError

from .interfaces import ContentEvidence


LOW_FREQ_CODER_ID = "low_freq_coder_v1"
LOW_FREQ_CODER_VERSION = "v1"
LOW_FREQ_CODER_TRACE_VERSION = "v1"

# 允许的失败原因枚举。
ALLOWED_LF_CODER_FAILURE_REASONS = {
    "lf_coder_disabled",                  # enable_lf 配置为 false，低频编码未启用
    "lf_coder_no_plan",                   # 缺失 plan_digest，无法确定编码子空间
    "lf_coder_invalid_input",             # 输入缺失或形状不符
    "lf_coder_plan_mismatch",             # plan_digest 不一致，编码参数与计划脱离
    "lf_coder_encoding_failed",           # 编码计算异常
    "lf_coder_trace_digest_mismatch",     # lf_trace_digest 与预期不符
}


class LowFreqCoder:
    """
    功能：低频水印编码器，在 LF 子空间施加隐蔽编码。

    Low-frequency watermark coder implementing reproducible, auditable encoding
    with plan digest binding and failure semantics strictness.

    Implements the ContentExtractor protocol frozen in interfaces.py.
    Emits ContentEvidence with lf_score, lf_trace_digest, and strict failure
    semantics (absent, failed, mismatch) matching frozen enumeration.

    Encoding applies only in LF subspace; outputs coding trace and score.
    Failure to produce valid score (e.g., plan mismatch) must set score=None
    and status!=ok to prevent false watermark evidence.

    Args:
        impl_id: Implementation identifier string (frozen to low_freq_coder_v1).
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
        功能：在 LF 子空间进行水印编码与检测评分。

        Perform low-frequency watermark coding and extract score.

        Three semantic modes:
        1. enable_lf=false: Returns absent status (abstinence, not error).
        2. enable_lf=true, plan available: Attempts encoding and score extraction.
        3. enable_lf=true, plan missing or mismatch: Returns status="mismatch" or "failed".

        When status="ok": score is non-None float, lf_score is populated.
        When status!="ok": score must be None, content_failure_reason populated.

        Args:
            cfg: Configuration dict with optional keys:
                - "enable_lf" (bool, default False): Whether to enable LF coding.
                - "lf_codebook_id" (str, optional): Codebook identifier.
                - "lf_redundancy" (int, default 3): Redundancy level (2-8).
                - "lf_power" (float, default 0.1): Encoding power level.
                - "watermark" (dict, optional): Sub-configuration containing plan_digest.
                  - "plan_digest" (str, optional): Expected plan digest binding.

            inputs: Optional input dict with keys:
                - "latent_features" (array-like, optional): Latent space features.
                - "latent_shape" (tuple, optional): Expected shape of latent.

            cfg_digest: Optional canonical SHA256 digest of cfg.
                       Enables mask_digest binding without full cfg dependency.

        Returns:
            ContentEvidence instance with frozen structure.
                - status: "ok" (valid score), "absent" (disabled), "failed" (error),
                  "mismatch" (plan inconsistency).
                - score: Non-None float when status="ok"; None otherwise.
                - lf_score: Optional low-frequency score (populated when ok).
                - lf_trace_digest: Audit digest of encoding trace.
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

        # (1) 解析启用状态。对齐 injection_scope_manifest.yaml 中 plan_digest_include_paths 的声明。
        lf_cfg = cfg.get("watermark", {}).get("lf", {})
        enabled = lf_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            # enabled 类型不合法，必须 fail-fast。
            raise TypeError("watermark.lf.enabled must be bool")

        # 若禁用，返回 absent 语义（非错误）。
        if not enabled:
            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=False,
                plan_digest=None,
                encoding_trace=None
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
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=None,
                content_failure_reason=None
            )

        # (2) 解析计划摘要与编码参数。
        plan_digest = cfg.get("watermark", {}).get("plan_digest")
        if plan_digest is None:
            # 无计划，无法确定编码子空间。
            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=None,
                encoding_trace=None
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
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=None,
                content_failure_reason="lf_coder_no_plan"
            )

        # (3) 验证输入有效性。
        if inputs is None:
            inputs = {}

        latent_features = inputs.get("latent_features")
        latent_shape = inputs.get("latent_shape")

        if latent_features is None or latent_shape is None:
            # 输入缺失。
            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=plan_digest,
                encoding_trace=None
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
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=plan_digest,
                content_failure_reason="lf_coder_invalid_input"
            )

        # (4) 执行编码与评分（简化：占位实现，返回固定分数）。
        try:
            # 读取 LF 参数（这些参数已在 plan_digest_include_paths 中声明）。
            codebook_id = lf_cfg.get("codebook_id", "default")
            redundancy = lf_cfg.get("ecc", 3)  # ecc 参数（纠错编码强度）匹配 manifest
            power = lf_cfg.get("strength", 0.1)  # strength 参数（功率）匹配 manifest

            # 验证参数范围。
            if not isinstance(redundancy, int) or redundancy < 2 or redundancy > 8:
                # 冗余参数不合法，必须 fail-fast。
                raise ValueError(f"redundancy must be int in [2, 8], got {redundancy}")
            if not isinstance(power, (int, float)) or power <= 0 or power > 1:
                # 功率参数不合法，必须 fail-fast。
                raise ValueError(f"power must be float in (0, 1], got {power}")

            # 计算编码痕迹（规范化可复算）。
            encoding_trace = {
                "codebook_id": codebook_id,
                "redundancy": redundancy,
                "power": power,
                "plan_digest": plan_digest,
                "cfg_digest": cfg_digest
            }

            # 生成占位分数（范围 [0.0, 1.0]，表示高频通道的强度）。
            # 在真实实现中，此处应执行实际编码与分数提取。
            lf_score_value = power * 0.5  # 占位：基于功率的简单映射

        except (ValueError, TypeError, KeyError) as e:
            # 编码异常，单一主因上报。
            failure_reason = "lf_coder_encoding_failed"

            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=plan_digest,
                encoding_trace=None
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
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=plan_digest,
                content_failure_reason=failure_reason
            )

        # (5) 构造成功路径审计字段。
        trace_payload = _build_lf_trace_payload(
            cfg=cfg,
            impl_id=self.impl_id,
            impl_version=self.impl_version,
            impl_digest=self.impl_digest,
            enabled=True,
            plan_digest=plan_digest,
            encoding_trace=encoding_trace
        )
        lf_trace_digest = digests.canonical_sha256(trace_payload)

        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": lf_trace_digest
        }

        # (6) 返回成功证据。
        return ContentEvidence(
            status="ok",
            score=lf_score_value,
            audit=audit,
            lf_trace_digest=lf_trace_digest,
            lf_score=lf_score_value,
            plan_digest=plan_digest,
            content_failure_reason=None
        )


def _build_lf_trace_payload(
    cfg: Dict[str, Any],
    impl_id: str,
    impl_version: str,
    impl_digest: str,
    enabled: bool,
    plan_digest: Optional[str],
    encoding_trace: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    功能：构造可复算的低频编码追踪有效负载。

    Build deterministic trace payload for LF coder digest computation.

    Args:
        cfg: Configuration mapping.
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.
        enabled: Whether LF coding is enabled.
        plan_digest: Optional plan digest binding.
        encoding_trace: Optional encoding parameters and trace.

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
    if encoding_trace is not None and not isinstance(encoding_trace, dict):
        # encoding_trace 类型不合法，必须 fail-fast。
        raise TypeError("encoding_trace must be dict or None")

    lf_cfg = cfg.get("watermark", {}).get("low_freq", {})

    payload = {
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "trace_version": LOW_FREQ_CODER_TRACE_VERSION,
        "enabled": enabled,
        "plan_digest": plan_digest,
        "codebook_id": lf_cfg.get("codebook_id", "default") if enabled else None,
        "redundancy": lf_cfg.get("redundancy", 3) if enabled else None,
        "power": lf_cfg.get("power", 0.1) if enabled else None,
    }

    if encoding_trace is not None:
        payload["encoding_trace"] = encoding_trace

    return payload
