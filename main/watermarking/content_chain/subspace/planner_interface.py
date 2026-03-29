"""
子空间规划器接口定义

功能说明：
- 定义子空间规划器的冻结协议接口。
- 规划器负责从掩码与配置输入计算子空间定义、矩阵分解参数、绑定摘要。
- 所有规划器必须遵循：输入域明确、输出可序列化、digest 可复算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol

from main.core.status import ALLOWED_STATUS_VALUES


@dataclass(frozen=True)
class SubspacePlanEvidence:
    """
    功能：子空间规划证据载体。
    
    Frozen structure for subspace plan evidence returned by planners.
    All values must be JSON-serializable and conform to frozen contracts.
    
    Attributes:
        status: Planner status enumeration from main.core.status.ALLOWED_STATUS_VALUES.
                One of: "ok", "absent", "failed", "mismatch". Semantics:
                  - "ok": Planning completed successfully, plan is valid.
                  - "absent": Planning not performed (e.g., disabled, optional).
                  - "failed": Planning failed due to error.
                  - "mismatch": Plan inconsistent with mask/input.
        plan: Optional serializable plan dict containing subspace definition (indexes/digest only, no large arrays).
        basis_digest: Optional SHA256 digest of subspace basis matrix summary.
        plan_digest: Optional SHA256 digest binding plan to mask_digest, planner params, and cfg_digest.
        audit: Audit metadata dict with required keys: impl_identity, impl_version, impl_digest, trace_digest.
        plan_stats: Optional dict with rank, energy_ratio, hf_fraction, lf_fraction for diagnostics.
        plan_failure_reason: Optional failure reason enumeration string.
               Required when status="failed" or status="mismatch".
    """
    status: Literal["ok", "absent", "failed", "mismatch"]
    plan: Optional[Dict[str, Any]] = None
    basis_digest: Optional[str] = None
    plan_digest: Optional[str] = None
    audit: Optional[Dict[str, Any]] = None
    plan_stats: Optional[Dict[str, Any]] = None
    plan_failure_reason: Optional[str] = None
    planner_failure_stage: Optional[str] = None
    planner_failure_detail_code: Optional[str] = None
    planner_failure_detail_message: Optional[str] = None
    planner_diagnostic_context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        # 校验 status 值。
        allowed_enum_set = set(ALLOWED_STATUS_VALUES)
        if self.status not in allowed_enum_set:
            raise ValueError(
                f"Invalid status: {self.status}. Must be one of {sorted(allowed_enum_set)}"
            )
        
        # 校验 plan, basis_digest, plan_digest 与 status 的一致性。
        if self.status == "ok":
            if self.plan is None:
                raise ValueError(
                    "plan must be non-None dict when status=ok"
                )
            if self.plan_digest is None:
                raise ValueError(
                    "plan_digest must be non-None str when status=ok"
                )
        else:
            if self.plan is not None:
                raise ValueError(
                    f"plan must be None when status={self.status}, got {self.plan}"
                )
        
        # 校验 audit 是 dict 且包含必需字段。
        if self.status == "ok":
            if self.audit is None or not isinstance(self.audit, dict):
                raise ValueError(
                    "audit must be dict when status=ok"
                )
            required_audit_fields = {"impl_identity", "impl_version", "impl_digest", "trace_digest"}
            missing_fields = required_audit_fields - set(self.audit.keys())
            if missing_fields:
                raise ValueError(
                    f"audit missing required fields: {missing_fields}"
                )
        
        # 校验 plan_failure_reason。
        if self.status in {"failed", "mismatch"}:
            if self.plan_failure_reason is None:
                raise ValueError(
                    f"plan_failure_reason is required when status={self.status}"
                )
    
    def as_dict(self) -> Dict[str, Any]:
        """
        功能：将 SubspacePlanEvidence 序列化为 dict。
        
        Serialize SubspacePlanEvidence to a JSON-serializable dict.
        
        Args:
            None.
        
        Returns:
            Dict containing frozen field values.
        """
        return {
            "status": self.status,
            "plan": self.plan,
            "basis_digest": self.basis_digest,
            "plan_digest": self.plan_digest,
            "audit": self.audit,
            "plan_stats": self.plan_stats,
            "plan_failure_reason": self.plan_failure_reason,
            "planner_failure_stage": self.planner_failure_stage,
            "planner_failure_detail_code": self.planner_failure_detail_code,
            "planner_failure_detail_message": self.planner_failure_detail_message,
            "planner_diagnostic_context": self.planner_diagnostic_context,
        }


class SubspacePlanner(Protocol):
    """
    功能：子空间规划器协议。
    
    Protocol for subspace planner implementations.
    Any class implementing this protocol must provide plan() method.
    
    Note: This is a frozen interface. Implementations must not change the signature.
    """
    
    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: Optional[str] = None,
        cfg_digest: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None
    ) -> SubspacePlanEvidence:
        """
        功能：从配置与掩码计算子空间规划证据。
        
        Compute subspace plan evidence from configuration and mask digest.
        
        Args:
            cfg: Configuration dict containing subspace parameters:
                - "watermark.subspace.enabled" (bool): Whether planning is enabled.
                - "watermark.subspace.k" (int): Subspace dimension.
                - "watermark.subspace.topk" (int): Top-k selection size.
                - Other subspace parameters as defined in injection_scope_manifest.
            mask_digest: Optional SHA256 digest of mask (when available).
            cfg_digest: Optional canonical SHA256 digest of cfg (computed from include_paths).
            inputs: Optional input dict with latent shape or other metadata.
        
        Returns:
            SubspacePlanEvidence instance with frozen structure.
        
        Raises:
            ValueError: If cfg or inputs are invalid.
            TypeError: If types are invalid.
        """
        ...
