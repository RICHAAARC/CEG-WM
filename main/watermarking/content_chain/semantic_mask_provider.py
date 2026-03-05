"""
语义掩码提供器实现

功能说明：
- 实现 SemanticMaskProvider，提供可复算、可审计的语义掩码提取。
- 掩码摘要通过规范 JSON + canonical_sha256 生成，确保可重复性。
- 支持 enable_mask 配置参数，当禁用时返回 absent 语义绑定。
- 实现语义位置绑定（resolution_binding）表达掩码与原始内容的空间关系。
- 所有失败路径单一主因上报，无沉默覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from main.core import digests

from .interfaces import ContentEvidence


SEMANTIC_MASK_PROVIDER_ID = "semantic_mask_provider_v1"
SEMANTIC_MASK_PROVIDER_VERSION = "v1"
SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_ID = "semantic_mask_provider_saliency_source_policy_v1"
SEMANTIC_MASK_PROVIDER_SALIENCY_POLICY_VERSION = "v1"
SEMANTIC_MASK_TRACE_VERSION = "v2"
SEMANTIC_SALIENCY_IMPL_ID = "semantic_saliency_v1"
SEMANTIC_SALIENCY_V2_IMPL_ID = "semantic_saliency_v2"
TEXTURE_FALLBACK_IMPL_ID = "texture_gradient_v1"

MASK_SOURCE_TYPE_SEMANTIC_MODEL_V2 = "semantic_model_v2"
MASK_SOURCE_TYPE_SEMANTIC_MODEL_V1 = "semantic_model_v1"
MASK_SOURCE_TYPE_TEXTURE_GRADIENT_PROXY = "texture_gradient_proxy"

SALIENCY_SOURCE_PROXY_V1 = "proxy_v1"
SALIENCY_SOURCE_MODEL_V2 = "model_v2"
SALIENCY_SOURCE_AUTO_FALLBACK = "auto_fallback"

SEMANTIC_MODEL_BACKEND_BASNET = "basnet"
SEMANTIC_MODEL_BACKEND_INSPYRENET = "inspyrenet"

MASK_ABSENT_REASON_DISABLED = "mask_disabled_by_config"
ROUTING_ABSENT_REASON_MASK_DISABLED = "routing_mask_disabled"

_SALIENCY_MODEL_CACHE: Dict[str, Any] = {}

# 允许的失败原因枚举。
ALLOWED_MASK_FAILURE_REASONS = {
    "mask_extraction_no_input",           # 输入缺失，无法执行掩码提取
    "mask_extraction_invalid_shape",      # 输入形状不符，掩码提取失败
    "mask_resolution_binding_mismatch",   # 分辨率绑定不一致，掩码损坏或配置错误
    "mask_digest_computation_failed",     # 掩码摘要计算异常
}


def _resolve_mask_source_type(mask_impl_id: str) -> str:
    """
    功能：将掩码实现 ID 映射为稳定的来源类型标签。 

    Map mask implementation identity to stable mask_source_type label.

    Args:
        mask_impl_id: Mask implementation identifier.

    Returns:
        One of semantic_model_v2 / semantic_model_v1 / texture_gradient_proxy.
    """
    if not isinstance(mask_impl_id, str):
        raise TypeError("mask_impl_id must be str")
    if mask_impl_id == SEMANTIC_SALIENCY_V2_IMPL_ID:
        return MASK_SOURCE_TYPE_SEMANTIC_MODEL_V2
    if mask_impl_id == TEXTURE_FALLBACK_IMPL_ID:
        return MASK_SOURCE_TYPE_TEXTURE_GRADIENT_PROXY
    return MASK_SOURCE_TYPE_SEMANTIC_MODEL_V1


@dataclass(frozen=True)
class SaliencySourceDecision:
    """
    功能：显著性来源策略决策。 

    Immutable decision for saliency source selection.

    Args:
        source_selected: Selected saliency source.
        source_attempted: Attempted saliency source list in order.
        fallback_used: Whether fallback is activated.
        fallback_reason: Explicit fallback reason.
        model_artifact_anchor: Model anchor digest payload.
        selected_impl_id: Selected mask implementation id.
    """

    source_selected: str
    source_attempted: List[str]
    fallback_used: bool
    fallback_reason: str
    model_artifact_anchor: Dict[str, Any]
    selected_impl_id: str

    def as_dict(self) -> Dict[str, Any]:
        """Serialize saliency source decision."""
        return {
            "saliency_source_selected": self.source_selected,
            "saliency_source_attempted": self.source_attempted,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "model_artifact_anchor": self.model_artifact_anchor,
            "selected_impl_id": self.selected_impl_id,
        }


class SemanticMaskProvider:
    """
    功能：语义掩码提供器，实现可复算、可审计的掩码提取与绑定。

    Semantic mask provider implementing reproducible, auditable mask extraction
    with resolution binding and reproducible digest computation.

    Implements the ContentExtractor protocol frozen in interfaces.py.
    Emits ContentEvidence with mask_digest, mask_stats, and resolution_binding
    bound to configuration and inputs via canonical SHA256.

    Args:
        impl_id: Implementation identifier string (frozen to semantic_mask_provider_v1).
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
        功能：提取语义掩码证据，支持可选启用与禁用模式。

        Extract semantic mask evidence with optional enable_mask configuration.
        Supports three semantic modes:
        1. enable_mask=false: Returns absent status with no failure reason (abstinence).
        2. enable_mask=true, input available: Computes mask and emits status="ok" with
           mask_digest/mask_stats (structural evidence, score=None).
        3. enable_mask=true, input missing/invalid: Returns status="failed" with
           single primary failure reason.

        When enable_mask=false, content_failure_reason is absent (not an error).
        When enable_mask=true but mask computation fails, single primary failure
        reason is reported.

        Args:
            cfg: Configuration dict with optional keys:
                - "enable_mask" (bool, default False): Whether to enable mask extraction.
                - "mask_inference_timeout" (int, default 30): Timeout in seconds.
                - "mask_resolution_width" (int, optional): Expected mask width.
                - "mask_resolution_height" (int, optional): Expected mask height.
                - Other config fields are included in trace digest for reproducibility.
            inputs: Optional input dict with keys:
                - "image" or "latent" (array-like): Input for mask computation.
                - "image_shape" (tuple, optional): Expected shape (H, W, C).
                - Other input fields included in trace digest.
            cfg_digest: Optional canonical SHA256 digest of cfg (computed from include_paths).
                       When provided, mask_digest will bind to this authoritative digest
                       instead of recomputing from full cfg. Prevents non-digest-scope fields
                       from affecting mask_digest and ensures reproducibility.

        Returns:
            ContentEvidence instance with:
            - status: "absent" if disabled, "ok" if mask computed successfully,
              "failed" if mask extraction error.
            - score: None (masking is structural evidence, not a detection score).
            - mask_digest: SHA256 of canonical JSON mask payload (only when status="ok").
            - mask_stats: Dict with mask_area_ratio, connected_components, boundary_complexity,
              resolution_binding (only when status="ok").
            - content_failure_reason: Absent if disabled/ok, required if status="failed".
            - audit: Dict with impl_identity, impl_version, impl_digest, trace_digest.

        Raises:
            TypeError: If cfg or inputs have invalid types.
            RecordsWritePolicyError: Must NOT be raised (all errors caught and wrapped in ContentEvidence).
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            # inputs 类型不合法，必须 fail-fast。
            raise TypeError("inputs must be dict or None")
        if cfg_digest is not None and not isinstance(cfg_digest, str):
            # cfg_digest 类型不合法，必须 fail-fast。
            raise TypeError("cfg_digest must be str or None")

        # 1. 解析配置参数。
        enable_mask = cfg.get("enable_mask", False)
        if not isinstance(enable_mask, bool):
            # enable_mask 类型不合法，必须 fail-fast。
            raise TypeError("enable_mask must be bool")

        # 2. 构造追踪有效负载，用于可复算的 trace_digest。
        trace_payload = _build_mask_trace_payload(
            cfg,
            inputs,
            self.impl_id,
            self.impl_version,
            self.impl_digest,
            enable_mask,
            cfg_digest=cfg_digest
        )
        trace_digest = digests.canonical_sha256(trace_payload)

        # 3. 审计字段集合。
        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": trace_digest
        }

        mask_params = _resolve_mask_params(cfg)

        # 4. 禁用路径：返回 absent 语义，无失败原因。
        if not enable_mask:
            return ContentEvidence(
                status="absent",
                score=None,
                audit={
                    **audit,
                    "mask_absent_reason": MASK_ABSENT_REASON_DISABLED,
                    "routing_absent_reason": ROUTING_ABSENT_REASON_MASK_DISABLED,
                },
                mask_digest=None,
                mask_stats=None,
                plan_digest=None,
                basis_digest=None,
                lf_trace_digest=None,
                hf_trace_digest=None,
                lf_score=None,
                hf_score=None,
                score_parts={
                    "routing_digest": "<absent>",
                    "routing_absent_reason": ROUTING_ABSENT_REASON_MASK_DISABLED,
                },
                content_failure_reason=None  # absent 状态下无失败原因
            )

        # 5. 启用路径：尝试计算掩码。
        # （1）输入校验。
        if inputs is None or not inputs:
            # 输入缺失，无法执行掩码提取。
            return ContentEvidence(
                status="failed",
                score=None,
                audit=audit,
                mask_digest=None,
                mask_stats=None,
                plan_digest=None,
                basis_digest=None,
                lf_trace_digest=None,
                hf_trace_digest=None,
                lf_score=None,
                hf_score=None,
                score_parts=None,
                content_failure_reason="mask_extraction_no_input"
            )

        # （2）提取输入语义（可选）。
        image_data = inputs.get("image")
        if image_data is None:
            image_data = inputs.get("latent")
        if image_data is None:
            image_data = inputs.get("image_path")
        image_shape = inputs.get("image_shape")

        if image_data is None:
            # 缺少核心输入：image 或 latent。
            return ContentEvidence(
                status="failed",
                score=None,
                audit=audit,
                mask_digest=None,
                mask_stats=None,
                plan_digest=None,
                basis_digest=None,
                lf_trace_digest=None,
                hf_trace_digest=None,
                lf_score=None,
                hf_score=None,
                score_parts=None,
                content_failure_reason="mask_extraction_no_input"
            )

        # （3）掩码计算（确定性轻量算法：梯度幅值 + 开闭操作）。
        _probe_available, _probe_failure_reason = _probe_model_v2_availability(mask_params)
        availability_probe = {
            "model_available": _probe_available
        }
        saliency_decision = select_saliency_source(cfg, availability_probe)

        if saliency_decision.source_selected == SALIENCY_SOURCE_MODEL_V2 and not availability_probe["model_available"] and not saliency_decision.fallback_used:
            return ContentEvidence(
                status="failed",
                score=None,
                audit={
                    **audit,
                    "saliency_source_selected": saliency_decision.source_selected,
                    "saliency_source_attempted": saliency_decision.source_attempted,
                    "fallback_used": saliency_decision.fallback_used,
                    "fallback_reason": saliency_decision.fallback_reason,
                    "model_artifact_anchor": saliency_decision.model_artifact_anchor,
                    "mask_source_type": MASK_SOURCE_TYPE_SEMANTIC_MODEL_V2,
                    "probe_failure_reason": _probe_failure_reason,
                },
                mask_digest=None,
                mask_stats=None,
                plan_digest=None,
                basis_digest=None,
                lf_trace_digest=None,
                hf_trace_digest=None,
                lf_score=None,
                hf_score=None,
                score_parts=None,
                content_failure_reason="saliency_source_model_unavailable"
            )

        try:
            mask_impl_id = saliency_decision.selected_impl_id
            fallback_reason = None
            if mask_impl_id in {SEMANTIC_SALIENCY_IMPL_ID, SEMANTIC_SALIENCY_V2_IMPL_ID}:
                try:
                    if mask_impl_id == SEMANTIC_SALIENCY_V2_IMPL_ID:
                        mask_array, saliency_map, mask_stats, mask_binding = build_semantic_saliency_mask_v2(
                            image=image_data,
                            image_shape=image_shape,
                            cfg=cfg,
                            params=mask_params,
                        )
                    else:
                        mask_array, saliency_map, mask_stats, mask_binding = build_semantic_saliency_mask_v1(
                            image=image_data,
                            image_shape=image_shape,
                            cfg=cfg,
                            params=mask_params,
                        )
                except Exception as saliency_exc:
                    fallback_reason = f"semantic_model_unavailable: {type(saliency_exc).__name__}"
                    if mask_impl_id == SEMANTIC_SALIENCY_V2_IMPL_ID:
                        if saliency_decision.source_selected == SALIENCY_SOURCE_MODEL_V2:
                            # 显式 model_v2 模式禁止自动降级，必须 fail-fast。
                            return ContentEvidence(
                                status="failed",
                                score=None,
                                audit={
                                    **audit,
                                    "saliency_source_selected": saliency_decision.source_selected,
                                    "saliency_source_attempted": saliency_decision.source_attempted,
                                    "fallback_used": False,
                                    "fallback_reason": fallback_reason,
                                    "model_artifact_anchor": saliency_decision.model_artifact_anchor,
                                    "mask_source_type": MASK_SOURCE_TYPE_SEMANTIC_MODEL_V2,
                                },
                                mask_digest=None,
                                mask_stats=None,
                                plan_digest=None,
                                basis_digest=None,
                                lf_trace_digest=None,
                                hf_trace_digest=None,
                                lf_score=None,
                                hf_score=None,
                                score_parts=None,
                                content_failure_reason="saliency_source_model_v2_runtime_failed"
                            )
                        try:
                            mask_array, saliency_map, mask_stats, mask_binding = build_semantic_saliency_mask_v1(
                                image=image_data,
                                image_shape=image_shape,
                                cfg=cfg,
                                params=mask_params,
                            )
                            mask_impl_id = SEMANTIC_SALIENCY_IMPL_ID
                        except Exception:
                            mask_impl_id = TEXTURE_FALLBACK_IMPL_ID
                            mask_array, mask_stats, mask_binding = build_texture_mask_v1(
                                image=image_data,
                                image_shape=image_shape,
                                cfg=cfg,
                                params=mask_params,
                            )
                            saliency_map = mask_array.astype(np.float32)
                    else:
                        mask_impl_id = TEXTURE_FALLBACK_IMPL_ID
                        mask_array, mask_stats, mask_binding = build_texture_mask_v1(
                            image=image_data,
                            image_shape=image_shape,
                            cfg=cfg,
                            params=mask_params,
                        )
                        saliency_map = mask_array.astype(np.float32)
            else:
                mask_array, mask_stats, mask_binding = build_texture_mask_v1(
                    image=image_data,
                    image_shape=image_shape,
                    cfg=cfg,
                    params=mask_params,
                )
                saliency_map = mask_array.astype(np.float32)
            mask_summary = summarize_mask_for_digest(
                mask_array=mask_array,
                mask_stats=mask_stats,
                binding=mask_binding,
                cfg_digest=cfg_digest,
                mask_params=mask_params,
            )
            mask_digest = compute_mask_digest(mask_summary)
            routing_summary = build_routing_summary(mask_stats, mask_params)
            routing_digest = compute_routing_digest(routing_summary)
            mask_params_digest = digests.canonical_sha256(mask_params)
        except (ValueError, TypeError, KeyError) as e:
            # 掩码计算异常，单一主因上报：根据异常类型推导主因。
            if "shape" in str(e).lower():
                failure_reason = "mask_extraction_invalid_shape"
            elif "digest" in str(e).lower():
                failure_reason = "mask_digest_computation_failed"
            else:
                failure_reason = "mask_extraction_invalid_shape"
            
            # 若主因不在允许枚举中，降级为通用异常信号。
            if failure_reason not in ALLOWED_MASK_FAILURE_REASONS:
                failure_reason = "mask_extraction_invalid_shape"

            return ContentEvidence(
                status="failed",
                score=None,
                audit=audit,
                mask_digest=None,
                mask_stats=None,
                plan_digest=None,
                basis_digest=None,
                lf_trace_digest=None,
                hf_trace_digest=None,
                lf_score=None,
                hf_score=None,
                score_parts=None,
                content_failure_reason=failure_reason
            )

        # （4）如启用掩码且计算成功，绑定至掩码统计。
        mask_stats_with_binding = dict(mask_stats or {})
        mask_stats_with_binding["mask_resolution_binding"] = mask_binding
        mask_stats_with_binding["resolution_binding"] = mask_binding
        mask_stats_with_binding["mask_source_impl_identity"] = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
        }
        mask_stats_with_binding["mask_impl_id"] = mask_impl_id
        mask_stats_with_binding["mask_source_type"] = _resolve_mask_source_type(mask_impl_id)
        mask_stats_with_binding["mask_fallback_reason"] = fallback_reason if isinstance(fallback_reason, str) else "<absent>"
        mask_stats_with_binding["saliency_mean"] = round(float(np.mean(saliency_map)), 8)
        mask_stats_with_binding["saliency_std"] = round(float(np.std(saliency_map)), 8)
        mask_stats_with_binding["mask_params_digest"] = mask_params_digest
        mask_stats_with_binding["routing_summary"] = routing_summary
        mask_stats_with_binding["routing_digest"] = routing_digest
        mask_stats_with_binding["saliency_source_selected"] = saliency_decision.source_selected
        mask_stats_with_binding["saliency_source_attempted"] = {
            "sequence": saliency_decision.source_attempted
        }
        mask_stats_with_binding["fallback_used"] = saliency_decision.fallback_used or isinstance(fallback_reason, str)
        mask_stats_with_binding["fallback_reason"] = fallback_reason if isinstance(fallback_reason, str) else saliency_decision.fallback_reason
        mask_stats_with_binding["model_artifact_anchor"] = saliency_decision.model_artifact_anchor
        mask_stats_with_binding["saliency_provenance"] = {
            "source_selected": saliency_decision.source_selected,
            "source_attempted": saliency_decision.source_attempted,
            "fallback_used": bool(saliency_decision.fallback_used or isinstance(fallback_reason, str)),
            "fallback_reason": fallback_reason if isinstance(fallback_reason, str) else saliency_decision.fallback_reason,
            "model_artifact_anchor": saliency_decision.model_artifact_anchor,
        }
        mask_stats_with_binding["mask_metadata"] = {
            "mask_impl_id": mask_impl_id,
            "model_id": mask_params.get("semantic_weights_id", "<absent>"),
            "model_version": mask_params.get("semantic_model_version", "<absent>"),
            "preprocess": mask_params.get("semantic_preprocess", "<absent>"),
            "thresholding": mask_params.get("semantic_thresholding", "<absent>"),
            "threshold": mask_stats.get("saliency_threshold", "<absent>"),
            "fallback_reason": fallback_reason if isinstance(fallback_reason, str) else "<absent>",
        }
        audit["mask_impl_id"] = mask_impl_id
        audit["mask_source_type"] = _resolve_mask_source_type(mask_impl_id)
        audit["mask_fallback_reason"] = fallback_reason if isinstance(fallback_reason, str) else "<absent>"
        audit["saliency_source_selected"] = saliency_decision.source_selected
        audit["saliency_source_attempted"] = saliency_decision.source_attempted
        audit["fallback_used"] = bool(saliency_decision.fallback_used or isinstance(fallback_reason, str))
        audit["fallback_reason"] = fallback_reason if isinstance(fallback_reason, str) else saliency_decision.fallback_reason
        audit["model_artifact_anchor"] = saliency_decision.model_artifact_anchor

        # 5. 成功路径：返回 ok 状态（掩码成功提取的结构证据）。
        # 注意：掩码是结构证据，不产生检测分数 (score=None)。
        # ContentEvidence 在接口层支持 status="ok" + score=None + mask_digest 非空的组合，
        # 表示"证据已成功提取"（不同于传统的检测分数）。
        return ContentEvidence(
            status="ok",
            score=None,  # 结构证据，不产生数值分数
            audit=audit,
            mask_digest=mask_digest,
            mask_stats=mask_stats_with_binding,
            plan_digest=None,
            basis_digest=None,
            lf_trace_digest=None,
            hf_trace_digest=None,
            lf_score=None,
            hf_score=None,
            score_parts={
                "routing_digest": routing_digest,
                "routing_summary": routing_summary,
            },
            content_failure_reason=None  # ok 状态下无失败原因
        )


def _build_mask_trace_payload(
    cfg: Dict[str, Any],
    inputs: Optional[Dict[str, Any]],
    impl_id: str,
    impl_version: str,
    impl_digest: str,
    enable_mask: bool,
    cfg_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：构造可复算的掩码追踪有效负载，用于 trace_digest 计算。

    Build deterministic trace payload for mask provider digest computation.
    Includes implementation identity, configuration, and input metadata
    to ensure reproducibility and auditability.

    Args:
        cfg: Configuration mapping.
        inputs: Optional input mapping.
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.
        enable_mask: Whether mask extraction is enabled.
        cfg_digest: Optional canonical SHA256 digest of cfg (computed from include_paths).

    Returns:
        JSON-like dict for canonical SHA256 computation.

    Raises:
        TypeError: If any input types are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if inputs is not None and not isinstance(inputs, dict):
        # inputs 类型不合法，必须 fail-fast。
        raise TypeError("inputs must be dict or None")
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 类型不合法，必须 fail-fast。
        raise TypeError("impl_id must be non-empty str")
    if not isinstance(impl_version, str) or not impl_version:
        # impl_version 类型不合法，必须 fail-fast。
        raise TypeError("impl_version must be non-empty str")
    if not isinstance(impl_digest, str) or not impl_digest:
        # impl_digest 类型不合法，必须 fail-fast。
        raise TypeError("impl_digest must be non-empty str")
    if cfg_digest is not None and not isinstance(cfg_digest, str):
        # cfg_digest 类型不合法，必须 fail-fast。
        raise TypeError("cfg_digest must be str or None")

    # 构造追踪有效负载：包含实现身份、配置摘要与输入元数据。
    # trace_version v2：从包含全量 cfg 改为仅包含 cfg_digest（append-only 版本化）。
    payload = {
        "trace_version": "v2",  # 版本化：表示不再包含全量 cfg
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "enable_mask": enable_mask,
        "inputs_keys": sorted(inputs.keys()) if inputs else [],
        "cfg_digest_provided": cfg_digest is not None,
        "cfg_digest_binding": cfg_digest  # 仅包含摘要，不包含全量 cfg
    }
    return payload


def _compute_semantic_mask(
    image_data: Any,
    image_shape: Optional[Any],
    cfg: Dict[str, Any],
    cfg_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：计算语义掩码有效负载（确定性 texture-based v1）。

    Compute semantic mask payload with deterministic texture-based v1 algorithm.
    The payload is reproducible and digest-friendly by construction:
    mask summary fields are canonicalized before hashing, and mask parameters
    are bound through mask_params_digest.

    Args:
        image_data: Image tensor or latent code.
        image_shape: Optional expected shape (H, W, C).
        cfg: Configuration dict.
        cfg_digest: Optional canonical SHA256 digest of cfg (computed from include_paths).
               When provided, this authoritative digest is used for cfg_digest_binding
               to avoid non-scope cfg fields affecting reproducibility.

    Returns:
        JSON-like mask payload dict.

    Raises:
        ValueError: If shape validation fails.
        TypeError: If image_data type is unsupported.
    """
    mask_params = _resolve_mask_params(cfg)
    mask_array, mask_stats, mask_binding = build_texture_mask_v1(
        image=image_data,
        image_shape=image_shape,
        cfg=cfg,
        params=mask_params,
    )
    return summarize_mask_for_digest(
        mask_array=mask_array,
        mask_stats=mask_stats,
        binding=mask_binding,
        cfg_digest=cfg_digest,
        mask_params=mask_params,
    )


def _extract_mask_statistics(
    mask_payload: Dict[str, Any],
    image_shape: Optional[Any]
) -> Dict[str, Any]:
    """
    功能：从掩码有效负载提取统计指标。

    Extract mask statistics from payload for diagnostic purposes.
    Statistics are optional fields in ContentEvidence and help
    with reproducibility verification and debugging.

    Args:
        mask_payload: Mask data dict from _compute_semantic_mask.
        image_shape: Optional shape tuple.

    Returns:
        Dict with mask_area_ratio, connected_components, boundary_complexity.

    Raises:
        ValueError: If payload structure is invalid.
    """
    if not isinstance(mask_payload, dict):
        raise ValueError(f"mask_payload must be dict, got {type(mask_payload).__name__}")

    mask_data = mask_payload.get("mask_data", {})
    if not isinstance(mask_data, dict):
        raise ValueError("mask_payload.mask_data must be dict")
    return {
        "area_ratio": mask_data.get("area_ratio", 0.0),
        "connected_components": mask_data.get("connected_components", 0),
        "mean_energy": mask_data.get("mean_energy", 0.0),
    }


def _bind_resolution(
    image_shape: Optional[Any],
    cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    功能：绑定掩码分辨率至原始输入与配置。

    Bind mask resolution to original input and configuration for consistency checking.
    Resolution binding enables detection of downstream scale mismatch or corruption.

    Args:
        image_shape: Optional original shape tuple (H, W, C).
        cfg: Configuration dict with optional mask_resolution_* parameters.

    Returns:
        Dict with width, height, aspect_ratio; None if binding cannot be computed.

    Raises:
        ValueError: If resolution validation fails.
    """
    if image_shape is None:
        # 无原始形状信息，binding 返回 None（absent 语义）。
        return None

    if not isinstance(image_shape, (list, tuple)) or len(image_shape) != 3:
        raise ValueError(
            f"image_shape must be 3-tuple, got {image_shape}"
        )

    height, width, channels = image_shape

    # 可选：校验配置中的预期分辨率。
    cfg_width = cfg.get("mask_resolution_width")
    cfg_height = cfg.get("mask_resolution_height")

    if cfg_width is not None and cfg_width != width:
        # 配置指定的宽度与实际不符，但允许 absent（不强制 binding）。
        pass

    if cfg_height is not None and cfg_height != height:
        # 配置指定的高度与实际不符，但允许 absent。
        pass

    # 返回分辨率绑定。
    aspect_ratio = width / height if height > 0 else 1.0

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4),
        "binding_version": "v1"
    }


def _resolve_mask_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析掩码算法参数并生成稳定参数集。 

    Resolve mask generation parameters from configuration.

    Args:
        cfg: Configuration mapping.

    Returns:
        Canonical mask parameter mapping.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    mask_cfg = cfg.get("mask") if isinstance(cfg.get("mask"), dict) else {}
    threshold_quantile = mask_cfg.get("threshold_quantile", 0.8)
    if not isinstance(threshold_quantile, (int, float)):
        threshold_quantile = 0.8
    threshold_quantile = max(0.5, min(float(threshold_quantile), 0.98))

    open_iters = mask_cfg.get("open_iters", 1)
    close_iters = mask_cfg.get("close_iters", 1)
    if not isinstance(open_iters, int):
        open_iters = 1
    if not isinstance(close_iters, int):
        close_iters = 1
    open_iters = max(0, min(open_iters, 3))
    close_iters = max(0, min(close_iters, 3))

    mask_impl_id = mask_cfg.get("impl_id", SEMANTIC_SALIENCY_V2_IMPL_ID)
    if not isinstance(mask_impl_id, str) or not mask_impl_id:
        mask_impl_id = SEMANTIC_SALIENCY_IMPL_ID

    semantic_model_source = mask_cfg.get("semantic_model_source", SEMANTIC_MODEL_BACKEND_BASNET)
    semantic_model_version = mask_cfg.get("semantic_model_version", "v1")
    semantic_weights_id = mask_cfg.get("semantic_weights_id", "builtin")
    semantic_model_path = mask_cfg.get("semantic_model_path")
    if semantic_model_path is not None and not isinstance(semantic_model_path, str):
        semantic_model_path = None
    semantic_preprocess = mask_cfg.get("semantic_preprocess", "rgb_normalized")
    semantic_thresholding = mask_cfg.get("semantic_thresholding", "quantile")
    saliency_threshold_quantile = mask_cfg.get("saliency_threshold_quantile", threshold_quantile)
    if not isinstance(saliency_threshold_quantile, (int, float)):
        saliency_threshold_quantile = threshold_quantile
    saliency_threshold_quantile = max(0.5, min(float(saliency_threshold_quantile), 0.98))
    saliency_source = mask_cfg.get("saliency_source", SALIENCY_SOURCE_AUTO_FALLBACK)
    if saliency_source not in {SALIENCY_SOURCE_PROXY_V1, SALIENCY_SOURCE_MODEL_V2, SALIENCY_SOURCE_AUTO_FALLBACK}:
        saliency_source = SALIENCY_SOURCE_AUTO_FALLBACK

    return {
        "mask_algo_version": mask_impl_id,
        "saliency_source": saliency_source,
        "threshold_quantile": threshold_quantile,
        "saliency_threshold_quantile": saliency_threshold_quantile,
        "semantic_model_source": semantic_model_source,
        "semantic_model_version": semantic_model_version,
        "semantic_weights_id": semantic_weights_id,
        "semantic_model_path": semantic_model_path,
        "semantic_preprocess": semantic_preprocess,
        "semantic_thresholding": semantic_thresholding,
        "open_iters": open_iters,
        "close_iters": close_iters,
    }


def _normalize_semantic_model_source(model_source: Any) -> str:
    """
    功能：归一化语义模型后端来源标记。

    Normalize semantic model backend source label.

    Args:
        model_source: Raw model source value from cfg.

    Returns:
        Normalized backend label with BASNet as default.
    """
    if not isinstance(model_source, str):
        return SEMANTIC_MODEL_BACKEND_BASNET

    normalized = model_source.strip().lower()
    if normalized in {"inspyrenet", "inspyre", "inspyre_net"}:
        return SEMANTIC_MODEL_BACKEND_INSPYRENET
    if normalized in {
        "basnet",
        "offline_heuristic",
        "model_v2",
        "offline",
        "local",
        "builtin",
        "default",
    }:
        return SEMANTIC_MODEL_BACKEND_BASNET
    return SEMANTIC_MODEL_BACKEND_BASNET


def _resolve_semantic_model_backend(mask_params: Dict[str, Any]) -> str:
    """
    功能：解析语义模型后端类型。

    Resolve semantic model backend type from mask parameters.

    Args:
        mask_params: Canonical mask parameter mapping.

    Returns:
        Backend label in {"basnet", "inspyrenet"}.
    """
    if not isinstance(mask_params, dict):
        raise TypeError("mask_params must be dict")
    return _normalize_semantic_model_source(mask_params.get("semantic_model_source"))


def _probe_model_v2_availability(mask_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    功能：探测 model_v2 显著性来源可用性，并返回失败原因以供审计。

    Probe deterministic availability for model-based saliency source.

    Args:
        mask_params: Canonical mask parameter mapping.

    Returns:
        Tuple of (available, failure_reason). failure_reason is None when available is True.
    """
    if not isinstance(mask_params, dict):
        return False, "probe_invalid_params_type"
    model_path = mask_params.get("semantic_model_path")
    if not isinstance(model_path, str) or not model_path:
        return False, "probe_model_path_missing"

    model_file = Path(model_path)
    if not (model_file.exists() and model_file.is_file()):
        return False, f"probe_model_file_not_found: {model_file}"

    backend = _resolve_semantic_model_backend(mask_params)
    try:
        model_obj = _load_saliency_model(mask_params)
    except Exception as exc:
        # 模型结构与权重最小加载校验失败，记录具体异常以供审计。
        return False, f"probe_load_failed: {type(exc).__name__}: {exc}"

    if backend == SEMANTIC_MODEL_BACKEND_INSPYRENET:
        if not callable(model_obj):
            return False, "probe_inspyrenet_not_callable"
        return True, None
    return True, None


def _extract_state_dict_payload(loaded_obj: Any) -> Optional[Dict[str, Any]]:
    """
    功能：从加载对象提取可用于 load_state_dict 的权重字典。

    Extract a state_dict-like payload from loaded checkpoint object.

    Args:
        loaded_obj: Object loaded by torch.load.

    Returns:
        State dictionary when available; otherwise None.
    """
    if not isinstance(loaded_obj, dict):
        return None

    for key_name in ["state_dict", "model_state_dict", "weights"]:
        payload = loaded_obj.get(key_name)
        if isinstance(payload, dict) and payload:
            return payload

    if loaded_obj and all(isinstance(key, str) for key in loaded_obj.keys()):
        return loaded_obj
    return None


def _instantiate_inspyrenet_model() -> Any:
    """
    功能：实例化 InSPyReNet 模型结构。

    Instantiate InSPyReNet model architecture from available runtime modules.
    Tries transparent_background and inspyrenet packages via static imports.

    Returns:
        Instantiated model object.

    Raises:
        RuntimeError: If no compatible InSPyReNet class can be constructed.
    """

    _CANDIDATE_KWARGS = [
        # 针对 InSPyReNet 基类：backbone + in_channels 为必选参数
        {"backbone": "res2net50_v1b_26w_4s", "in_channels": 3},
        {"backbone": "res2net50", "in_channels": 3},
        {"backbone": "res2net50_v1b_26w_4s", "in_channels": 3, "depth": 64, "base_size": 1024},
        {"backbone": "res2net50", "in_channels": 3, "depth": 64, "base_size": 1024},
        # 针对 InSPyReNet_SwinB：depth + pretrained + base_size 为必选参数
        {"depth": [2, 2, 6, 2], "pretrained": False, "base_size": 1024},
        {"depth": 4, "pretrained": False, "base_size": 1024},
        # 历史候选（容错保留）
        {},
        {"base_size": 1024, "backbone": "res2net50", "nclass": 1},
        {"backbone": "res2net50", "nclass": 1},
        {"backbone": "res2net50"},
    ]
    # 收集所有尝试路径的失败原因，用于最终 RuntimeError 诊断。
    _attempt_log: "list[str]" = []

    def _try_class(cls_name: str, cls: Any) -> "Any | None":
        """尝试用候选参数组逐一实例化，返回成功对象或 None。"""
        for _kw in _CANDIDATE_KWARGS:
            try:
                obj = cls(**_kw)
                _attempt_log.append(f"{cls_name}({_kw}): OK")
                return obj
            except Exception as _exc:
                _attempt_log.append(f"{cls_name}({_kw}): {type(_exc).__name__}: {_exc}")
        return None

    # (1) transparent_background 包：优先尝试派生类 InSPyReNet_Res2Net50（ckpt_base.pth 对应类）。
    try:
        import transparent_background.InSPyReNet as _tb_mod  # type: ignore[import]
        _attempt_log.append("import transparent_background.InSPyReNet module: OK")
        for _cls_name in ["InSPyReNet_Res2Net50", "InSPyReNet_SwinS", "InSPyReNet_SwinB", "InSPyReNet"]:
            _cls = getattr(_tb_mod, _cls_name, None)
            if _cls is not None:
                _obj = _try_class(f"tb.{_cls_name}", _cls)
                if _obj is not None:
                    return _obj
            else:
                _attempt_log.append(f"tb.{_cls_name}: not found in module")
    except ImportError as _e:
        _attempt_log.append(f"import transparent_background.InSPyReNet: ImportError: {_e}")
    except Exception as _e:
        _attempt_log.append(f"import transparent_background.InSPyReNet: {type(_e).__name__}: {_e}")

    # (2) transparent_background 顶层导出（有些版本路径不同）
    try:
        import transparent_background as _tb_top  # type: ignore[import]
        _attempt_log.append("import transparent_background top: OK")
        for _cls_name in ["InSPyReNet_Res2Net50", "InSPyReNet_SwinS", "InSPyReNet"]:
            _cls = getattr(_tb_top, _cls_name, None)
            if _cls is not None:
                _obj = _try_class(f"tbtop.{_cls_name}", _cls)
                if _obj is not None:
                    return _obj
    except ImportError as _e:
        _attempt_log.append(f"import transparent_background top: ImportError: {_e}")
    except Exception as _e:
        _attempt_log.append(f"import transparent_background top: {type(_e).__name__}: {_e}")

    # (3) inspyrenet 包（备用路径）
    try:
        from inspyrenet import InSPyReNet as _InspyInSPyReNet  # type: ignore[import]
        _attempt_log.append("import inspyrenet.InSPyReNet: OK")
        _obj = _try_class("InspyInSPyReNet", _InspyInSPyReNet)
        if _obj is not None:
            return _obj
    except ImportError as _e:
        _attempt_log.append(f"import inspyrenet InSPyReNet: ImportError: {_e}")
    except Exception as _e:
        _attempt_log.append(f"import inspyrenet InSPyReNet: {type(_e).__name__}: {_e}")

    # (4) inspyrenet.model 子模块
    try:
        from inspyrenet.model import InSPyReNet as _InspyModelInSPyReNet  # type: ignore[import]
        _attempt_log.append("import inspyrenet.model.InSPyReNet: OK")
        _obj = _try_class("InspyModelInSPyReNet", _InspyModelInSPyReNet)
        if _obj is not None:
            return _obj
    except ImportError as _e:
        _attempt_log.append(f"import inspyrenet.model InSPyReNet: ImportError: {_e}")
    except Exception as _e:
        _attempt_log.append(f"import inspyrenet.model InSPyReNet: {type(_e).__name__}: {_e}")

    _diag = " | ".join(_attempt_log) if _attempt_log else "no attempts logged"
    raise RuntimeError(f"inspyrenet model class unavailable. attempts: [{_diag}]")


def _compute_weights_sha256(model_path: Optional[str]) -> str:
    """
    功能：计算模型权重文件 SHA256。

    Compute SHA256 digest for model weights file.

    Args:
        model_path: Model file path.

    Returns:
        SHA256 hex digest or <absent>.
    """
    if not isinstance(model_path, str) or not model_path:
        return "<absent>"
    model_file = Path(model_path)
    if not (model_file.exists() and model_file.is_file()):
        return "<absent>"
    digest_hasher = hashlib.sha256()
    with model_file.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest_hasher.update(chunk)
    return digest_hasher.hexdigest()


def _load_saliency_model(mask_params: Dict[str, Any]) -> Any:
    """
    功能：加载并缓存显著性模型。

    Load and cache saliency model from local path.

    Args:
        mask_params: Mask parameter mapping.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If model file is missing.
        RuntimeError: If model cannot be loaded.
    """
    if not isinstance(mask_params, dict):
        raise TypeError("mask_params must be dict")
    backend = _resolve_semantic_model_backend(mask_params)
    model_path = mask_params.get("semantic_model_path")
    if not isinstance(model_path, str) or not model_path:
        raise FileNotFoundError("semantic_model_path is required for model_v2")

    model_file = Path(model_path)
    if not (model_file.exists() and model_file.is_file()):
        raise FileNotFoundError(f"semantic model path not found: {model_file}")

    cache_key = f"{backend}:{str(model_file.resolve())}"
    if cache_key in _SALIENCY_MODEL_CACHE:
        return _SALIENCY_MODEL_CACHE[cache_key]

    try:
        import torch
        # weights_only=False：ckpt_base.pth 为 state_dict，兼容 PyTorch >= 2.4 默认值变更。
        loaded_obj = torch.load(str(model_file), map_location="cpu", weights_only=False)

        if backend == SEMANTIC_MODEL_BACKEND_INSPYRENET:
            if callable(loaded_obj):
                model_obj = loaded_obj
            elif isinstance(loaded_obj, dict) and callable(loaded_obj.get("model")):
                model_obj = loaded_obj.get("model")
            else:
                state_dict = _extract_state_dict_payload(loaded_obj)
                if not isinstance(state_dict, dict) or not state_dict:
                    # inspyrenet 权重结构异常，必须显式失败。
                    raise RuntimeError("inspyrenet checkpoint missing state_dict payload")

                model_obj = _instantiate_inspyrenet_model()
                if not hasattr(model_obj, "load_state_dict") or not callable(model_obj.load_state_dict):
                    # 模型对象不支持 state_dict 加载，必须失败。
                    raise RuntimeError("inspyrenet model object missing load_state_dict")
                incompatible_keys = model_obj.load_state_dict(state_dict, strict=False)
                missing_keys = list(getattr(incompatible_keys, "missing_keys", []) or [])
                unexpected_keys = list(getattr(incompatible_keys, "unexpected_keys", []) or [])
                if missing_keys or unexpected_keys:
                    # key 不完全匹配时，评估是否可接受（允许部分不匹配，禁止全量缺失）。
                    # 若缺失 key 超过 state_dict 总 key 数的 90%，则判定为不可用。
                    total_keys = len(state_dict)
                    missing_ratio = len(missing_keys) / max(total_keys, 1)
                    if missing_ratio > 0.9:
                        # 绝大多数 key 缺失，架构与权重不兼容，禁止 silent fallback。
                        raise RuntimeError(
                            f"inspyrenet state_dict key mismatch too severe: "
                            f"missing={len(missing_keys)}/{total_keys} ({missing_ratio:.0%}), "
                            f"unexpected={len(unexpected_keys)}"
                        )
                    # 部分 key 不匹配，作为可接受的兼容加载，记录告警。
                    print(
                        f"[semantic_mask_provider] [WARN] InSPyReNet partial key mismatch "
                        f"(acceptable): missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
                    )
        else:
            model_obj = loaded_obj
            if isinstance(loaded_obj, dict) and "model" in loaded_obj:
                model_obj = loaded_obj.get("model")

        if hasattr(model_obj, "eval") and callable(model_obj.eval):
            model_obj.eval()
        _SALIENCY_MODEL_CACHE[cache_key] = model_obj
        return model_obj
    except Exception as exc:
        raise RuntimeError(f"saliency model load failed: {type(exc).__name__}: {exc}") from exc


def _run_saliency_inference(model: Any, image_tensor: Any, mask_params: Dict[str, Any]) -> np.ndarray:
    """
    功能：运行显著性模型推理并返回二维显著图。

    Run saliency model inference and return normalized saliency map.

    Args:
        model: Loaded model object.
        image_tensor: Input tensor in shape [1, 3, H, W].
        mask_params: Mask parameter mapping.

    Returns:
        Saliency map in [0, 1] with shape [H, W].

    Raises:
        RuntimeError: If inference fails.
    """
    if not isinstance(mask_params, dict):
        raise TypeError("mask_params must be dict")

    backend = _resolve_semantic_model_backend(mask_params)

    try:
        import torch
        if callable(model):
            with torch.no_grad():
                output = model(image_tensor)
        else:
            raise RuntimeError("saliency model is not callable")

        if backend == SEMANTIC_MODEL_BACKEND_INSPYRENET:
            if isinstance(output, dict):
                output = output.get("pred", output.get("saliency", output.get("out")))
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                output = output[0]
        else:
            if isinstance(output, (list, tuple)) and len(output) > 0:
                output = output[0]
            if isinstance(output, dict):
                output = output.get("saliency", output.get("pred", output.get("out")))

        if not hasattr(output, "detach"):
            raise RuntimeError("saliency model output must be tensor-like")
        saliency_tensor = output.detach().float().cpu()
        while saliency_tensor.ndim > 2:
            saliency_tensor = saliency_tensor[0]

        saliency_map = saliency_tensor.numpy().astype(np.float32)
        if saliency_map.ndim != 2:
            raise RuntimeError("saliency output must be 2D")

        min_value = float(np.min(saliency_map))
        max_value = float(np.max(saliency_map))
        if max_value > min_value:
            saliency_map = (saliency_map - min_value) / (max_value - min_value + 1e-8)
        else:
            saliency_map = np.zeros_like(saliency_map, dtype=np.float32)
        return np.clip(saliency_map, 0.0, 1.0)
    except Exception as exc:
        raise RuntimeError(f"saliency inference failed: {type(exc).__name__}") from exc


def _build_model_artifact_anchor(mask_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造模型工件锚点（路径安全）。 

    Build model artifact anchor digest without exposing raw absolute paths.

    Args:
        mask_params: Canonical mask parameter mapping.

    Returns:
        Model anchor payload with digest-safe fields.
    """
    if not isinstance(mask_params, dict):
        return {
            "status": "absent",
            "artifact_digest": "<absent>",
        }
    model_path = mask_params.get("semantic_model_path")
    path_name = "<absent>"
    if isinstance(model_path, str) and model_path:
        path_name = Path(model_path).name
    payload = {
        "weights_id": mask_params.get("semantic_weights_id", "<absent>"),
        "model_version": mask_params.get("semantic_model_version", "<absent>"),
        "model_path_name": path_name,
        "model_source": _resolve_semantic_model_backend(mask_params),
        "weights_sha256": _compute_weights_sha256(model_path if isinstance(model_path, str) else None),
    }
    return {
        "status": "ok",
        "artifact_digest": digests.canonical_sha256(payload),
        "payload": payload,
    }


def select_saliency_source(cfg: Dict[str, Any], availability_probe: Dict[str, Any]) -> SaliencySourceDecision:
    """
    功能：选择显著性来源策略。 

    Select saliency source policy with explicit fallback semantics.

    Args:
        cfg: Configuration mapping.
        availability_probe: Availability probe mapping.

    Returns:
        SaliencySourceDecision with append-only audit fields.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(availability_probe, dict):
        raise TypeError("availability_probe must be dict")

    mask_cfg = cfg.get("mask") if isinstance(cfg.get("mask"), dict) else {}
    saliency_source = mask_cfg.get("saliency_source", SALIENCY_SOURCE_AUTO_FALLBACK)
    if saliency_source not in {SALIENCY_SOURCE_PROXY_V1, SALIENCY_SOURCE_MODEL_V2, SALIENCY_SOURCE_AUTO_FALLBACK}:
        saliency_source = SALIENCY_SOURCE_AUTO_FALLBACK

    model_available = bool(availability_probe.get("model_available", False))
    anchor = _build_model_artifact_anchor(_resolve_mask_params(cfg))

    if saliency_source == SALIENCY_SOURCE_PROXY_V1:
        return SaliencySourceDecision(
            source_selected=SALIENCY_SOURCE_PROXY_V1,
            source_attempted=[SALIENCY_SOURCE_PROXY_V1],
            fallback_used=False,
            fallback_reason="<absent>",
            model_artifact_anchor=anchor,
            selected_impl_id=SEMANTIC_SALIENCY_IMPL_ID,
        )

    if saliency_source == SALIENCY_SOURCE_MODEL_V2:
        if model_available:
            return SaliencySourceDecision(
                source_selected=SALIENCY_SOURCE_MODEL_V2,
                source_attempted=[SALIENCY_SOURCE_MODEL_V2],
                fallback_used=False,
                fallback_reason="<absent>",
                model_artifact_anchor=anchor,
                selected_impl_id=SEMANTIC_SALIENCY_V2_IMPL_ID,
            )
        return SaliencySourceDecision(
            source_selected=SALIENCY_SOURCE_MODEL_V2,
            source_attempted=[SALIENCY_SOURCE_MODEL_V2],
            fallback_used=False,
            fallback_reason="model_artifact_unavailable",
            model_artifact_anchor=anchor,
            selected_impl_id=SEMANTIC_SALIENCY_V2_IMPL_ID,
        )

    if model_available:
        return SaliencySourceDecision(
            source_selected=SALIENCY_SOURCE_MODEL_V2,
            source_attempted=[SALIENCY_SOURCE_MODEL_V2],
            fallback_used=False,
            fallback_reason="<absent>",
            model_artifact_anchor=anchor,
            selected_impl_id=SEMANTIC_SALIENCY_V2_IMPL_ID,
        )

    return SaliencySourceDecision(
        source_selected=SALIENCY_SOURCE_PROXY_V1,
        source_attempted=[SALIENCY_SOURCE_MODEL_V2, SALIENCY_SOURCE_PROXY_V1],
        fallback_used=True,
        fallback_reason="model_artifact_unavailable_auto_fallback_proxy",
        model_artifact_anchor=anchor,
        selected_impl_id=SEMANTIC_SALIENCY_IMPL_ID,
    )


def build_semantic_saliency_mask_v1(
    image: Any,
    image_shape: Optional[Any],
    cfg: Dict[str, Any],
    params: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    功能：基于可复算显著性代理生成语义掩码。

    Build deterministic semantic saliency mask and map.

    Args:
        image: Input image payload.
        image_shape: Optional shape hint.
        cfg: Configuration mapping.
        params: Mask parameter mapping.

    Returns:
        Tuple of (mask, saliency_map, mask_stats, binding).
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    image_array = _materialize_image_array(image, image_shape)
    gray = _to_gray(image_array)
    gradient_energy = _compute_gradient_energy(gray)

    h, w = gray.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    center_prior = 1.0 - np.clip(np.sqrt(grid_x * grid_x + grid_y * grid_y), 0.0, 1.0)
    center_prior = center_prior.astype(np.float32)

    energy_norm = gradient_energy / (float(np.max(gradient_energy)) + 1e-8)
    gray_norm = gray / (float(np.max(gray)) + 1e-8)
    saliency_map = 0.45 * energy_norm + 0.35 * center_prior + 0.20 * gray_norm
    saliency_map = np.clip(saliency_map, 0.0, 1.0).astype(np.float32)

    threshold = float(np.quantile(saliency_map, float(params.get("saliency_threshold_quantile", 0.8))))
    mask = saliency_map >= threshold

    open_iters = int(params.get("open_iters", 1))
    close_iters = int(params.get("close_iters", 1))
    for _ in range(open_iters):
        mask = _binary_dilate(_binary_erode(mask))
    for _ in range(close_iters):
        mask = _binary_erode(_binary_dilate(mask))

    area_ratio = float(np.mean(mask)) if mask.size > 0 else 0.0
    component_count = _count_connected_components(mask)
    largest_component_ratio = _largest_component_ratio(mask)
    boundary_length = _mask_boundary_length(mask)
    perimeter_to_area = float(boundary_length / max(1.0, float(mask.sum())))
    downsample_grid = _build_downsample_binary_grid(mask, rows=8, cols=8)
    downsample_grid_digest = digests.canonical_sha256({"grid": downsample_grid.tolist()})
    true_indices = np.flatnonzero(downsample_grid.reshape(-1)).astype(int).tolist()

    mask_stats = {
        "area_ratio": round(area_ratio, 8),
        "connected_components": int(component_count),
        "largest_component_ratio": round(float(largest_component_ratio), 8),
        "boundary_length": int(boundary_length),
        "perimeter_to_area_ratio": round(float(perimeter_to_area), 8),
        "foreground_coverage_ratio": round(float(area_ratio), 8),
        "downsample_grid_shape": [8, 8],
        "downsample_grid_true_indices": true_indices,
        "downsample_grid_digest": downsample_grid_digest,
        "saliency_threshold": round(float(threshold), 8),
        "mean_energy": round(float(np.mean(gradient_energy[mask])) if np.any(mask) else 0.0, 8),
    }
    binding = {
        "space": "image_space",
        "height": int(image_array.shape[0]),
        "width": int(image_array.shape[1]),
        "aspect_ratio": round(float(image_array.shape[1]) / float(image_array.shape[0]), 4) if int(image_array.shape[0]) > 0 else 1.0,
        "channels": int(image_array.shape[2]),
        "resize_rule": "identity",
        "binding_version": "v1",
    }
    return mask.astype(bool), saliency_map, mask_stats, binding


def build_semantic_saliency_mask_v2(
    image: Any,
    image_shape: Optional[Any],
    cfg: Dict[str, Any],
    params: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    功能：基于语义模型锚点执行 v2 掩码提取（离线权重前提）。

    Build v2 semantic saliency mask with offline model anchors.

    Args:
        image: Input image payload.
        image_shape: Optional shape hint.
        cfg: Configuration mapping.
        params: Mask parameter mapping.

    Returns:
        Tuple of (mask, saliency_map, mask_stats, binding).

    Raises:
        FileNotFoundError: If semantic model path is provided but missing.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    model_source_raw = params.get("semantic_model_source")
    model_source = _resolve_semantic_model_backend(params)
    if isinstance(model_source_raw, str) and model_source_raw.strip().lower() in {"hf", "hf_hub", "online", "remote"}:
        # 禁止运行期联网下载语义模型权重。
        raise ValueError("semantic_model_source must be offline; provide semantic_model_path")

    model_path = params.get("semantic_model_path")
    if isinstance(model_path, str) and model_path:
        candidate = Path(model_path)
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"semantic model path not found: {candidate}")

    image_array = _materialize_image_array(image, image_shape)

    import torch

    model_obj = _load_saliency_model(params)
    image_tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    saliency_map = _run_saliency_inference(model_obj, image_tensor, params)

    threshold_mode = params.get("semantic_thresholding", "quantile")
    if threshold_mode == "otsu":
        hist, bin_edges = np.histogram(saliency_map.reshape(-1), bins=256, range=(0.0, 1.0))
        total = float(np.sum(hist))
        sum_total = float(np.sum(hist * np.arange(256)))
        sum_background = 0.0
        weight_background = 0.0
        var_max = -1.0
        threshold_index = 0
        for index in range(256):
            weight_background += hist[index]
            if weight_background <= 0.0:
                continue
            weight_foreground = total - weight_background
            if weight_foreground <= 0.0:
                break
            sum_background += index * hist[index]
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            if var_between > var_max:
                var_max = var_between
                threshold_index = index
        threshold = float(bin_edges[min(threshold_index, len(bin_edges) - 2)])
    else:
        threshold = float(np.quantile(saliency_map, float(params.get("saliency_threshold_quantile", 0.8))))

    mask = saliency_map >= threshold
    open_iters = int(params.get("open_iters", 1))
    close_iters = int(params.get("close_iters", 1))
    for _ in range(open_iters):
        mask = _binary_dilate(_binary_erode(mask))
    for _ in range(close_iters):
        mask = _binary_erode(_binary_dilate(mask))

    area_ratio = float(np.mean(mask)) if mask.size > 0 else 0.0
    model_path_name = Path(model_path).name if isinstance(model_path, str) and model_path else "<absent>"
    mask_stats = {
        "area_ratio": round(area_ratio, 8),
        "saliency_threshold": round(float(threshold), 8),
        "saliency_source_selected": SALIENCY_SOURCE_MODEL_V2,
        "model_artifact_anchor": {
            "model_source": model_source,
            "model_version": params.get("semantic_model_version", "v2"),
            "weights_id": params.get("semantic_weights_id", "<absent>"),
            "model_path_name": model_path_name,
            "weights_sha256": _compute_weights_sha256(model_path if isinstance(model_path, str) else None),
            "preprocess": params.get("semantic_preprocess", "rgb_normalized"),
            "thresholding": threshold_mode,
        },
        "saliency_provenance": {
            "source_selected": SALIENCY_SOURCE_MODEL_V2,
            "source_attempted": [SALIENCY_SOURCE_MODEL_V2],
            "fallback_used": False,
            "fallback_reason": "<absent>",
        },
    }
    binding = {
        "space": "image_space",
        "height": int(image_array.shape[0]),
        "width": int(image_array.shape[1]),
        "aspect_ratio": round(float(image_array.shape[1]) / float(image_array.shape[0]), 4) if int(image_array.shape[0]) > 0 else 1.0,
        "channels": int(image_array.shape[2]),
        "resize_rule": "identity",
        "binding_version": "v1",
    }
    return mask.astype(bool), saliency_map.astype(np.float32), mask_stats, binding


def build_texture_mask_v1(
    image: Any,
    image_shape: Optional[Any],
    cfg: Dict[str, Any],
    params: Dict[str, Any]
) -> tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    功能：基于梯度纹理生成确定性掩码。 

    Build deterministic texture mask via gradient magnitude thresholding.

    Args:
        image: Input image payload or path.
        image_shape: Optional shape hint.
        cfg: Configuration mapping.
        params: Mask parameter mapping.

    Returns:
        Tuple of (mask_array, mask_stats, mask_binding).
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    image_array = _materialize_image_array(image, image_shape)
    gray = _to_gray(image_array)
    gradient_energy = _compute_gradient_energy(gray)
    threshold = float(np.quantile(gradient_energy, float(params["threshold_quantile"])))
    mask = gradient_energy >= threshold

    for _ in range(int(params["open_iters"])):
        mask = _binary_dilate(_binary_erode(mask))
    for _ in range(int(params["close_iters"])):
        mask = _binary_erode(_binary_dilate(mask))

    mask = mask.astype(bool)
    area_ratio = float(np.mean(mask)) if mask.size > 0 else 0.0
    component_count = _count_connected_components(mask)
    mean_energy = float(np.mean(gradient_energy[mask])) if np.any(mask) else 0.0

    mask_stats = {
        "area_ratio": round(area_ratio, 8),
        "connected_components": int(component_count),
        "mean_energy": round(mean_energy, 8),
    }
    binding = {
        "space": "image_space",
        "height": int(image_array.shape[0]),
        "width": int(image_array.shape[1]),
        "aspect_ratio": round(float(image_array.shape[1]) / float(image_array.shape[0]), 4) if int(image_array.shape[0]) > 0 else 1.0,
        "channels": int(image_array.shape[2]),
        "resize_rule": "identity",
        "binding_version": "v1",
    }
    return mask, mask_stats, binding


def summarize_mask_for_digest(
    mask_array: np.ndarray,
    mask_stats: Dict[str, Any],
    binding: Dict[str, Any],
    cfg_digest: Optional[str],
    mask_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：构造掩码摘要化表示用于 digest。 

    Build canonical summary dictionary for mask digest computation.

    Args:
        mask_array: Boolean mask array.
        mask_stats: Mask statistics mapping.
        binding: Mask resolution binding mapping.
        cfg_digest: Optional cfg digest binding.
        mask_params: Mask algorithm parameter mapping.

    Returns:
        Canonical summary mapping.
    """
    if not isinstance(mask_array, np.ndarray):
        raise TypeError("mask_array must be ndarray")
    if not isinstance(mask_stats, dict):
        raise TypeError("mask_stats must be dict")
    if not isinstance(binding, dict):
        raise TypeError("binding must be dict")
    if not isinstance(mask_params, dict):
        raise TypeError("mask_params must be dict")

    row_sum = mask_array.sum(axis=1).astype(int).tolist() if mask_array.ndim == 2 else []
    col_sum = mask_array.sum(axis=0).astype(int).tolist() if mask_array.ndim == 2 else []
    row_digest = digests.canonical_sha256({"row_sum": row_sum})
    col_digest = digests.canonical_sha256({"col_sum": col_sum})
    return {
        "mask_summary_version": "v2",
        "mask_shape": [int(v) for v in mask_array.shape],
        "mask_population": int(mask_array.sum()),
        "row_projection_digest": row_digest,
        "col_projection_digest": col_digest,
        "mask_stats": mask_stats,
        "mask_resolution_binding": binding,
        "mask_params_digest": digests.canonical_sha256(mask_params),
        "cfg_digest_binding": cfg_digest if isinstance(cfg_digest, str) and cfg_digest else "<absent>",
    }


def compute_mask_digest(mask_summary_dict: Dict[str, Any]) -> str:
    """
    功能：计算掩码摘要 digest。 

    Compute mask digest via canonical sha256.

    Args:
        mask_summary_dict: Canonical mask summary mapping.

    Returns:
        SHA256 hex digest.
    """
    if not isinstance(mask_summary_dict, dict):
        raise TypeError("mask_summary_dict must be dict")
    return digests.canonical_sha256(mask_summary_dict)


def build_routing_summary(mask_stats: Dict[str, Any], planner_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造路由摘要。 

    Build minimal audit routing summary from mask statistics.

    Args:
        mask_stats: Mask statistics mapping.
        planner_params: Planner/mask params mapping.

    Returns:
        Routing summary mapping.
    """
    if not isinstance(mask_stats, dict):
        raise TypeError("mask_stats must be dict")
    if not isinstance(planner_params, dict):
        raise TypeError("planner_params must be dict")

    hf_ratio = float(mask_stats.get("area_ratio", 0.0))
    hf_ratio = max(0.0, min(hf_ratio, 1.0))
    lf_ratio = 1.0 - hf_ratio
    return {
        "routing_version": "v2",
        "hf_region_ratio": round(hf_ratio, 8),
        "lf_region_ratio": round(lf_ratio, 8),
        "connected_components": int(mask_stats.get("connected_components", 0)),
        "largest_component_ratio": float(mask_stats.get("largest_component_ratio", 0.0)),
        "perimeter_to_area_ratio": float(mask_stats.get("perimeter_to_area_ratio", 0.0)),
        "foreground_coverage_ratio": float(mask_stats.get("foreground_coverage_ratio", hf_ratio)),
        "downsample_grid_shape": mask_stats.get("downsample_grid_shape", [8, 8]),
        "downsample_grid_true_indices": mask_stats.get("downsample_grid_true_indices", []),
        "downsample_grid_digest": mask_stats.get("downsample_grid_digest", "<absent>"),
        "band_thresholds": {
            "gradient_quantile": planner_params.get("threshold_quantile")
        },
        "mask_to_band_mapping": {
            "mask_true": "hf",
            "mask_false": "lf"
        }
    }


def compute_routing_digest(routing_summary: Dict[str, Any]) -> str:
    """
    功能：计算 routing 摘要 digest。 

    Compute routing digest via canonical sha256.

    Args:
        routing_summary: Routing summary mapping.

    Returns:
        SHA256 hex digest.
    """
    if not isinstance(routing_summary, dict):
        raise TypeError("routing_summary must be dict")
    return digests.canonical_sha256(routing_summary)


def _materialize_image_array(image: Any, image_shape: Optional[Any]) -> np.ndarray:
    """
    功能：将输入转换为 HWC 图像数组。 

    Materialize image payload into deterministic HWC float array.

    Args:
        image: Image payload or path.
        image_shape: Optional shape hint.

    Returns:
        Numpy float array in HWC layout.
    """
    if isinstance(image, str) and image:
        path = Path(image)
        if path.exists() and path.is_file():
            try:
                from PIL import Image
            except Exception as exc:
                raise TypeError("PIL is required to read image path input") from exc
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                return np.asarray(rgb, dtype=np.float32)

    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[2] in {1, 3, 4}:
        if arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        if arr.shape[2] == 4:
            return arr[:, :, :3]
        return arr

    if isinstance(image_shape, (list, tuple)) and len(image_shape) == 3:
        h, w, c = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])
        if h <= 0 or w <= 0 or c <= 0:
            raise ValueError("image_shape must be positive")
        target_size = h * w * c
        flat = arr.flatten()
        if flat.size == 0:
            raise ValueError("image payload is empty")
        repeat_count = (target_size + flat.size - 1) // flat.size
        expanded = np.tile(flat, repeat_count)[:target_size]
        reshaped = expanded.reshape((h, w, c)).astype(np.float32)
        if c == 1:
            reshaped = np.repeat(reshaped, 3, axis=2)
        elif c > 3:
            reshaped = reshaped[:, :, :3]
        return reshaped

    raise ValueError("unsupported image payload shape for mask extraction")


def _to_gray(image_array: np.ndarray) -> np.ndarray:
    """
    功能：将 HWC 图像转为灰度。 

    Convert HWC image array to grayscale.

    Args:
        image_array: Image array in HWC format.

    Returns:
        Grayscale array in range [0, 1].
    """
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 3:
        raise ValueError("image_array must be HWC ndarray")
    rgb = image_array[:, :, :3]
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    max_value = float(np.max(gray)) if gray.size > 0 else 1.0
    if max_value <= 0:
        return np.zeros_like(gray, dtype=np.float32)
    return (gray / max_value).astype(np.float32)


def _compute_gradient_energy(gray: np.ndarray) -> np.ndarray:
    """
    功能：计算灰度图梯度能量。 

    Compute gradient magnitude energy map.

    Args:
        gray: Grayscale image array.

    Returns:
        Gradient energy array.
    """
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return np.sqrt(gx * gx + gy * gy)


def _binary_dilate(mask: np.ndarray) -> np.ndarray:
    """
    功能：执行 3x3 膨胀。 

    Perform 3x3 binary dilation.

    Args:
        mask: Boolean mask.

    Returns:
        Dilated mask.
    """
    padded = np.pad(mask.astype(np.uint8), 1, mode="edge")
    acc = np.zeros_like(mask, dtype=np.uint8)
    for di in range(3):
        for dj in range(3):
            acc = np.maximum(acc, padded[di:di + mask.shape[0], dj:dj + mask.shape[1]])
    return acc > 0


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    """
    功能：执行 3x3 腐蚀。 

    Perform 3x3 binary erosion.

    Args:
        mask: Boolean mask.

    Returns:
        Eroded mask.
    """
    padded = np.pad(mask.astype(np.uint8), 1, mode="edge")
    acc = np.ones_like(mask, dtype=np.uint8)
    for di in range(3):
        for dj in range(3):
            acc = np.minimum(acc, padded[di:di + mask.shape[0], dj:dj + mask.shape[1]])
    return acc > 0


def _count_connected_components(mask: np.ndarray) -> int:
    """
    功能：计算布尔掩码连通域数量。 

    Count connected components in boolean mask.

    Args:
        mask: Boolean mask array.

    Returns:
        Connected component count.
    """
    if mask.size == 0:
        return 0
    visited = np.zeros(mask.shape, dtype=bool)
    component_count = 0
    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue
            component_count += 1
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or nx >= h or ny < 0 or ny >= w:
                        continue
                    if visited[nx, ny] or not mask[nx, ny]:
                        continue
                    visited[nx, ny] = True
                    stack.append((nx, ny))
    return component_count


def _largest_component_ratio(mask: np.ndarray) -> float:
    """
    功能：计算最大连通域占前景比例。

    Compute largest connected component ratio among foreground pixels.

    Args:
        mask: Boolean mask array.

    Returns:
        Ratio in [0, 1].
    """
    if mask.size == 0:
        return 0.0
    total = int(mask.sum())
    if total <= 0:
        return 0.0
    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    max_count = 0
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue
            count = 0
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                x, y = stack.pop()
                count += 1
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or nx >= h or ny < 0 or ny >= w:
                        continue
                    if visited[nx, ny] or not mask[nx, ny]:
                        continue
                    visited[nx, ny] = True
                    stack.append((nx, ny))
            if count > max_count:
                max_count = count
    return float(max_count / total)


def _mask_boundary_length(mask: np.ndarray) -> int:
    """
    功能：估计掩码边界长度。

    Estimate boundary length by counting foreground-background edges.

    Args:
        mask: Boolean mask array.

    Returns:
        Boundary edge count.
    """
    if mask.size == 0:
        return 0
    boundary = 0
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                continue
            for nx, ny in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if nx < 0 or nx >= h or ny < 0 or ny >= w or not mask[nx, ny]:
                    boundary += 1
    return int(boundary)


def _build_downsample_binary_grid(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    功能：构建掩码低分辨率二值网格表示。

    Build downsampled binary grid for spatial structure anchoring.

    Args:
        mask: Boolean mask array.
        rows: Grid rows.
        cols: Grid cols.

    Returns:
        Boolean grid with shape (rows, cols).
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise TypeError("mask must be 2-D ndarray")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    h, w = mask.shape
    row_edges = np.linspace(0, h, rows + 1, dtype=np.int64)
    col_edges = np.linspace(0, w, cols + 1, dtype=np.int64)
    grid = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            block = mask[row_edges[i]:row_edges[i + 1], col_edges[j]:col_edges[j + 1]]
            if block.size == 0:
                grid[i, j] = False
            else:
                grid[i, j] = bool(np.mean(block.astype(np.float32)) >= 0.5)
    return grid
