"""
子空间规划器最小可用实现

功能说明：
- 实现 SubspacePlanner 的最小可用版本，支持可复算的 plan_digest 计算。
- 输出子空间定义摘要而非巨大数组，满足写盘约束。
- 支持规划器消融（关闭时输出 absent 语义）。
- 绑定 mask_digest 与 cfg_digest，确保完整的可复算性。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from main.core import digests
from main.watermarking.content_chain.subspace.planner_interface import (
    SubspacePlanEvidence
)


SUBSPACE_PLANNER_ID = "subspace_planner_v1"
SUBSPACE_PLANNER_VERSION = "v1"
SUBSPACE_PLANNER_TRACE_VERSION = "v2"

# 允许的失败原因枚举（status=failed 或 status=mismatch 时使用）。
ALLOWED_PLANNER_FAILURE_REASONS = {
    "planner_disabled_by_policy",       # 规划器被 enable_planner=false 禁用。
    "mask_absent",                      # 掩码摘要缺失，无法进行子空间规划。
    "invalid_subspace_params",          # 规划参数（k, topk）非法。
    "decomposition_failed",             # 特征分解过程失败。
    "rank_computation_failed",          # 秩计算失败。
    "unknown"                           # 未知错误。
}


class SubspacePlannerImpl:
    """
    功能：子空间规划器最小可用实现。

    Minimal working implementation of SubspacePlanner supporting:
    - Reproducible plan_digest computation.
    - Binding to mask_digest and cfg_digest.
    - Serializable plan output (indexes, not arrays).
    - Ablation support (when disabled, returns absent).

    Args:
        impl_id: Implementation identifier string.
        impl_version: Implementation version string.
        impl_digest: Implementation digest computed from source code.
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

    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: Optional[str] = None,
        cfg_digest: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None
    ) -> SubspacePlanEvidence:
        """
        功能：计算子空间规划证据。

        Compute subspace plan evidence from config and optional mask/cfg digests.

        Args:
            cfg: Configuration dict with optional keys:
                - "watermark.subspace.enabled" (bool, default False): Whether planning is enabled.
                - "watermark.subspace.k" (int, default 10): Subspace dimension.
                - "watermark.subspace.topk" (int, default 20): Top-k selection size.
                - Other subspace parameters.
            mask_digest: Optional SHA256 digest of mask.
            cfg_digest: Optional canonical SHA256 digest of cfg.
            inputs: Optional input dict (not used in minimal implementation).

        Returns:
            SubspacePlanEvidence instance.

        Raises:
            TypeError: If cfg or other inputs have invalid types.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if mask_digest is not None and not isinstance(mask_digest, str):
            # mask_digest 类型不合法，必须 fail-fast。
            raise TypeError("mask_digest must be str or None")
        if cfg_digest is not None and not isinstance(cfg_digest, str):
            # cfg_digest 类型不合法，必须 fail-fast。
            raise TypeError("cfg_digest must be str or None")
        if inputs is not None and not isinstance(inputs, dict):
            # inputs 类型不合法，必须 fail-fast。
            raise TypeError("inputs must be dict or None")

        # 1. 解析规划器启用状态。
        subspace_cfg = cfg.get("watermark", {}).get("subspace", {})
        enabled = subspace_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            # enabled 类型不合法，必须 fail-fast。
            raise TypeError("watermark.subspace.enabled must be bool")

        # 2. 构造追踪有效负载，用于可复算的 trace_digest。
        trace_payload = _build_planner_trace_payload(
            cfg,
            self.impl_id,
            self.impl_version,
            self.impl_digest,
            enabled,
            mask_digest=mask_digest,
            cfg_digest=cfg_digest
        )
        trace_digest = digests.canonical_sha256(trace_payload)

        # 3. 审计字段。
        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": trace_digest
        }

        # 4. 禁用路径：返回 absent 语义。
        # 注意：status=absent 时 plan_failure_reason 通常为 None，但为审计完整性可记录原因。
        if not enabled:
            return SubspacePlanEvidence(
                status="absent",
                plan=None,
                basis_digest=None,
                plan_digest=None,
                audit=audit,
                plan_stats=None,
                plan_failure_reason="planner_disabled_by_policy"  # 审计标记：禁用原因
            )

        # 5. 启用路径：尝试计算规划。
        try:
            # 提取规划参数。
            k = subspace_cfg.get("k", 10)
            topk = subspace_cfg.get("topk", 20)
            
            if not isinstance(k, int) or k <= 0:
                raise ValueError("k must be positive int")
            if not isinstance(topk, int) or topk <= 0:
                raise ValueError("topk must be positive int")
            if topk < k:
                raise ValueError("topk must be >= k")

            # 构造完整的 plan_digest 输入域（必须绑定所有关键因素）。
            # plan_digest = canonical_sha256(plan_digest_input)
            # 关键：plan_digest 必须绑定 cfg_digest、mask_digest、planner_params、impl_identity
            plan_digest_input = _compute_plan_digest_input(
                k=k,
                topk=topk,
                mask_digest=mask_digest,
                cfg_digest=cfg_digest,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest
            )
            plan_digest = digests.canonical_sha256(plan_digest_input)

            # 计算规划有效负载（可序列化，不含巨数组）。
            plan_payload = _compute_subspace_plan(
                k=k,
                topk=topk,
                mask_digest=mask_digest,
                cfg_digest=cfg_digest
            )
            
            plan_stats = _extract_plan_statistics(k, topk)
            basis_digest = digests.canonical_sha256(
                {"k": k, "basis_type": "eigen"}
            )

        except (ValueError, TypeError, KeyError) as e:
            # 规划异常，单一主因上报。
            failure_reason = "decomposition_failed"
            if "params" in str(e).lower():
                failure_reason = "invalid_subspace_params"
            
            if failure_reason not in ALLOWED_PLANNER_FAILURE_REASONS:
                failure_reason = "decomposition_failed"

            return SubspacePlanEvidence(
                status="failed",
                plan=None,
                basis_digest=None,
                plan_digest=None,
                audit=audit,
                plan_stats=None,
                plan_failure_reason=failure_reason
            )

        # 6. 成功路径：返回 ok 状态。
        return SubspacePlanEvidence(
            status="ok",
            plan=plan_payload,
            basis_digest=basis_digest,
            plan_digest=plan_digest,
            audit=audit,
            plan_stats=plan_stats,
            plan_failure_reason=None
        )


def _build_planner_trace_payload(
    cfg: Dict[str, Any],
    impl_id: str,
    impl_version: str,
    impl_digest: str,
    enabled: bool,
    mask_digest: Optional[str] = None,
    cfg_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：构造可复算的规划追踪有效负载。

    Build deterministic trace payload for planner digest computation.

    Args:
        cfg: Configuration mapping.
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.
        enabled: Whether planning is enabled.
        mask_digest: Optional mask digest.
        cfg_digest: Optional cfg digest.

    Returns:
        JSON-like dict for canonical SHA256 computation.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 类型不合法，必须 fail-fast。
        raise TypeError("impl_id must be non-empty str")
    if not isinstance(impl_version, str) or not impl_version:
        # impl_version 类型不合法，必须 fail-fast。
        raise TypeError("impl_version must be non-empty str")
    if not isinstance(impl_digest, str) or not impl_digest:
        # impl_digest 类型不合法，必须 fail-fast。
        raise TypeError("impl_digest must be non-empty str")

    payload = {
        "trace_version": "v2",  # 版本化：表示不再包含全量 cfg
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "enabled": enabled,
        "mask_digest_provided": mask_digest is not None,
        "mask_digest_binding": mask_digest,
        "cfg_digest_provided": cfg_digest is not None,
        "cfg_digest_binding": cfg_digest  # 仅包含摘要，不包含全量 cfg
    }
    return payload


def _compute_plan_digest_input(
    k: int,
    topk: int,
    mask_digest: Optional[str],
    cfg_digest: Optional[str],
    impl_id: str,
    impl_version: str,
    impl_digest: str
) -> Dict[str, Any]:
    """
    功能：构造 plan_digest 的完整输入域。
    
    Build complete input for plan_digest computation.
    This input must bind:
      - planner parameters (k, topk)
      - mask_digest (from SemanticMaskProvider)
      - cfg_digest (from config loader)
      - impl_identity (impl_id, impl_version, impl_digest)
    
    plan_digest = canonical_sha256(plan_digest_input)

    Args:
        k: Subspace dimension.
        topk: Top-k selection size.
        mask_digest: Optional mask digest binding.
        cfg_digest: Optional cfg digest binding.
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.

    Returns:
        JSON-like dict for canonical SHA256 computation.
    """
    return {
        "plan_digest_version": "v1",
        # 实现身份（impl_identity）：必须进入 digest
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        # 规划参数（planner_params）：必须进入 digest
        "k": k,
        "topk": topk,
        # 权威摘要绑定（digest_bindings）：必须进入 digest
        "mask_digest_binding": mask_digest,      # 来自 SemanticMaskProvider
        "cfg_digest_binding": cfg_digest,        # 来自 config_loader
        # 可空性标记：显式声明输入缺失时的消融语义
        "mask_digest_provided": mask_digest is not None,
        "cfg_digest_provided": cfg_digest is not None
    }



def _compute_subspace_plan(
    k: int,
    topk: int,
    mask_digest: Optional[str] = None,
    cfg_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：计算子空间规划有效负载（最小实现）。

    Compute subspace plan payload (minimal implementation).
    Returns serializable indexes and digests only (no large arrays).
    
    Important: This payload is for serialization only; plan_digest is computed
    from _compute_plan_digest_input, not from this payload.

    Args:
        k: Subspace dimension.
        topk: Top-k selection size.
        mask_digest: Optional mask digest binding.
        cfg_digest: Optional cfg digest binding.

    Returns:
        JSON-like plan dict (serializable, no large matrices).
    """
    plan = {
        "plan_version": "v1",
        "subspace_dimension": k,
        "topk_selected": topk,
        "selected_frequencies": list(range(min(topk, 32))),  # Demo：前 32 频率索引
        "basis_type": "eigen",
        "energy_ratio": 0.95,  # 演示值
        "hf_fraction": 0.3,
        "lf_fraction": 0.7,
        # 显式标注输入缺失情况（消融语义）
        "mask_digest_status": "provided" if mask_digest is not None else "absent",
        "cfg_digest_status": "provided" if cfg_digest is not None else "absent"
    }
    
    # 绑定摘要（仅当提供时）。
    if mask_digest is not None:
        plan["mask_digest_binding"] = mask_digest
    
    if cfg_digest is not None:
        plan["cfg_digest_binding"] = cfg_digest
    
    return plan


def _extract_plan_statistics(k: int, topk: int) -> Dict[str, Any]:
    """
    功能：从规划参数提取统计指标。

    Extract plan statistics for diagnostics.

    Args:
        k: Subspace dimension.
        topk: Top-k selection size.

    Returns:
        Dict with rank, energy_ratio, hf_fraction, lf_fraction.
    """
    return {
        "rank": k,
        "topk": topk,
        "energy_ratio": 0.95,
        "hf_fraction": 0.3,
        "lf_fraction": 0.7
    }
