"""
融合决策接口类型定义与冻结协议

功能说明：
- 定义融合决策接口类型，包括融合决策载体和融合规则协议。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol


@dataclass(frozen=True)
class FusionDecision:
    """
    功能：融合决策载体。
    
    Frozen structure for fusion decision returned by decision rules.
    
    Attributes:
        is_watermarked: Final binary decision. True means watermarking detected, False means not detected.
        decision_status: Outcome of fusion decision computation, one of: "decided", "abstain", "error".
                        Semantics:
                          - "decided": Fusion rule computed a valid binary decision (is_watermarked is definitive).
                          - "abstain": Fusion rule lacks sufficient evidence to decide (is_watermarked may be None or default).
                          - "error": Fusion rule encountered error during computation (is_watermarked is undefined).
        thresholds_digest: Deterministic hash of threshold parameters, model version, and decision rule version.
                          Used for reproducibility and audit tracing.
        evidence_summary: Structured summary dict containing:
                         - content_score: Score from content extractor or None.
                         - geometry_score: Score from geometry extractor or None.
                         - content_status: Status from content evidence ("ok", "absent", "fail", "mismatch").
                         - geometry_status: Status from geometry evidence ("synced", "absent", "failed").
                         - fusion_rule_id: Identifier of fusion rule applied.
                         - Additional fields as needed for audit and reporting.
        audit: Audit metadata dict containing decision context, rule parameters, and trace info.
    
    Raises:
        ValueError: If decision_status is not one of allowed values.
        ValueError: If evidence_summary is missing required fields.
    """
    is_watermarked: bool
    decision_status: Literal["decided", "abstain", "error"]
    thresholds_digest: str
    evidence_summary: Dict[str, Any]
    audit: Dict[str, Any]
    
    def __post_init__(self) -> None:
        # (1) 校验 decision_status 值。
        allowed_status = {"decided", "abstain", "error"}
        if self.decision_status not in allowed_status:
            raise ValueError(
                f"Invalid decision_status: {self.decision_status}. Must be one of {allowed_status}"
            )
        
        # (2) 校验 is_watermarked 是 bool。
        if not isinstance(self.is_watermarked, bool):
            raise ValueError(
                f"is_watermarked must be bool, got {type(self.is_watermarked)}"
            )
        
        # (3) 校验 thresholds_digest 是非空字符串。
        if not isinstance(self.thresholds_digest, str) or not self.thresholds_digest:
            raise ValueError(
                f"thresholds_digest must be non-empty str, got {self.thresholds_digest}"
            )
        
        # (4) 校验 evidence_summary 是 dict 且含必需字段。
        if not isinstance(self.evidence_summary, dict):
            raise ValueError(
                f"evidence_summary must be dict, got {type(self.evidence_summary)}"
            )
        
        required_evidence_fields = {
            "content_score", "geometry_score", "content_status", "geometry_status", "fusion_rule_id"
        }
        missing_fields = required_evidence_fields - set(self.evidence_summary.keys())
        if missing_fields:
            raise ValueError(
                f"evidence_summary missing required fields: {missing_fields}"
            )
        
        # (5) 校验 audit 是 dict。
        if not isinstance(self.audit, dict):
            raise ValueError(
                f"audit must be dict, got {type(self.audit)}"
            )


class FusionRule(Protocol):
    """
    功能：融合决策规则协议。
    
    Protocol for fusion rule implementations that combine content and geometry evidence.
    Any class implementing this protocol must provide fuse() method.
    
    Note: This is a frozen interface. Implementations must not change the signature.
    """
    
    def fuse(
        self,
        cfg: Dict[str, Any],
        content_evidence: Any,
        geometry_evidence: Any
    ) -> FusionDecision:
        """
        功能：融合内容与几何证据进行最终检测决策。
        
        Fuse content and geometry evidence to produce final watermarking detection decision.
        
        Args:
            cfg: Configuration dict containing decision thresholds, model parameters, etc.
            content_evidence: ContentEvidence from content extractor.
            geometry_evidence: GeometryEvidence from geometry extractor (may be None if absent).
        
        Returns:
            FusionDecision instance with frozen structure.
        
        Raises:
            ValueError: If cfg or evidence objects are invalid.
            RuntimeError: If fusion fails (decision_status should be "error" in FusionDecision, not raised here).
        """
        ...
