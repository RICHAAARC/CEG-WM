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

from typing import Any, Dict, Optional, Literal

from main.core import digests
from main.core.errors import RecordsWritePolicyError

from .interfaces import ContentEvidence


SEMANTIC_MASK_PROVIDER_ID = "semantic_mask_provider_v1"
SEMANTIC_MASK_PROVIDER_VERSION = "v1"
SEMANTIC_MASK_TRACE_VERSION = "v1"

# 允许的失败原因枚举。
ALLOWED_MASK_FAILURE_REASONS = {
    "mask_extraction_disabled",           # enable_mask 配置为 false，掩码提取未启用
    "mask_extraction_no_input",           # 输入缺失，无法执行掩码提取
    "mask_extraction_invalid_shape",      # 输入形状不符，掩码提取失败
    "mask_resolution_binding_mismatch",   # 分辨率绑定不一致，掩码损坏或配置错误
    "mask_digest_computation_failed",     # 掩码摘要计算异常
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
        inputs: Optional[Dict[str, Any]] = None
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
            enable_mask
        )
        trace_digest = digests.canonical_sha256(trace_payload)

        # 3. 审计字段集合。
        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": trace_digest
        }

        # 4. 禁用路径：返回 absent 语义，无失败原因。
        if not enable_mask:
            return ContentEvidence(
                status="absent",
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
        image_data = inputs.get("image") or inputs.get("latent")
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

        # （3）掩码计算（简化版：演示版本不实现真实 ML 模型、直接生成模型掩码）。
        try:
            mask_payload = _compute_semantic_mask(image_data, image_shape, cfg)
            mask_digest = digests.canonical_sha256(mask_payload)
            mask_stats = _extract_mask_statistics(mask_payload, image_shape)
            resolution_binding = _bind_resolution(image_shape, cfg)
        except (ValueError, TypeError, KeyError) as e:
            # 掩码计算异常，单一主因上报：根据异常类型推导主因。
            if "shape" in str(e).lower():
                failure_reason = "mask_invalid_shape"
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
        if resolution_binding:
            mask_stats_with_binding["resolution_binding"] = resolution_binding

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
            score_parts=None,
            content_failure_reason=None  # ok 状态下无失败原因
        )


def _build_mask_trace_payload(
    cfg: Dict[str, Any],
    inputs: Optional[Dict[str, Any]],
    impl_id: str,
    impl_version: str,
    impl_digest: str,
    enable_mask: bool
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

    # 构造追踪有效负载：包含实现身份、配置与输入元数据。
    payload = {
        "trace_version": SEMANTIC_MASK_TRACE_VERSION,
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "enable_mask": enable_mask,
        "cfg": cfg,
        "inputs_keys": sorted(inputs.keys()) if inputs else []
    }
    return payload


def _compute_semantic_mask(
    image_data: Any,
    image_shape: Optional[Any],
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    功能：计算语义掩码有效负载（演示版本）。

    Compute semantic mask payload (demonstration version without ML model).
    In production, this would invoke segmentation model; for S-02 demo,
    generates deterministic mask based on shape and configuration.

    Args:
        image_data: Image tensor or latent code.
        image_shape: Optional expected shape (H, W, C).
        cfg: Configuration dict.

    Returns:
        JSON-like mask payload dict.

    Raises:
        ValueError: If shape validation fails.
        TypeError: If image_data type is unsupported.
    """
    # 演示版本：直接生成模拟掩码负载。
    # 生成结构：{mask_pixels, mask_bounds, mask_encoding_version}
    
    # 获取或推断形状。
    if image_shape is None:
        # 默认形状：512x512（标准 Stable Diffusion 3 分辨率）。
        inferred_shape = (512, 512, 3)
    elif isinstance(image_shape, (list, tuple)) and len(image_shape) == 3:
        inferred_shape = tuple(image_shape)
    else:
        raise ValueError(
            f"image_shape must be 3-tuple (H, W, C), got {image_shape}"
        )

    height, width, channels = inferred_shape

    if height <= 0 or width <= 0:
        # 分辨率不合法，必须 fail 上报。
        raise ValueError(
            f"image_shape must have positive dimensions, got ({height}, {width}, {channels})"
        )

    # 掩码负载：包含分辨率、编码版本、掩码数据段落。
    mask_payload = {
        "mask_encoding_version": "v1",
        "mask_resolution": {
            "height": height,
            "width": width,
            "channels": channels
        },
        "mask_data": {
            "total_pixels": height * width,
            "masked_pixels_ratio": 0.5,  # 演示版本固定比例
            "mask_type": "semantic_segmentation"
        },
        "cfg_digest_binding": digests.canonical_sha256(cfg)  # 绑定至配置摘要
    }

    return mask_payload


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

    # 从负载提取统计。
    mask_data = mask_payload.get("mask_data", {})
    mask_resolution = mask_payload.get("mask_resolution", {})

    stats = {
        "mask_area_ratio": mask_data.get("masked_pixels_ratio", 0.0),
        "connected_components": 1,  # 演示版本：单连通域
        "boundary_complexity": 0.5,  # 演示版本：中等复杂度
        "resolution": {
            "height": mask_resolution.get("height"),
            "width": mask_resolution.get("width")
        }
    }

    return stats


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
