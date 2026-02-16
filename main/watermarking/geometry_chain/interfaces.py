"""
几何链证据接口类型定义与冻结协议

功能说明：
- 明确几何链证据的结构和提取器的接口。
- 冻结几何链证据的语义和提取器协议，确保实现的一致性和稳定性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol


@dataclass(frozen=True)
class GeometryEvidence:
    """
    功能：几何链证据载体。
    
    Frozen structure for geometry evidence returned by extractors.
    
    Attributes:
        sync_status: Synchronization detection status, one of: "synced", "absent", "failed".
                     Mutually exclusive values. Semantics:
                       - "synced": Geometry synchronization detected, score is valid.
                       - "absent": Geometry extraction not performed or result unavailable (score typically None).
                       - "failed": Extraction/sync comparison failed due to error (score None, audit explains).
        score: Synchronization score. Must be None only when sync_status != "synced".
               Higher score indicates stronger evidence of geometry synchronization.
               Must be non-negative float or None.
        audit: Audit metadata dict. Must contain keys: impl_identity, impl_version, impl_digest, 
               trace_digest, sync_status_detail.
               All values must be JSON-serializable strings or dicts.
    
    Raises:
        ValueError: If sync_status is not one of allowed values.
        ValueError: If score is not None when sync_status != "synced".
        ValueError: If audit is missing required fields.
    """
    sync_status: Literal["synced", "absent", "failed"]
    score: Optional[float]
    audit: Dict[str, Any]
    
    def __post_init__(self) -> None:
        # (1) 校验 sync_status 值。
        allowed_status = {"synced", "absent", "failed"}
        if self.sync_status not in allowed_status:
            raise ValueError(
                f"Invalid sync_status: {self.sync_status}. Must be one of {allowed_status}"
            )
        
        # (2) 校验 score 与 sync_status 一致性：只有 sync_status="synced" 时 score 才应该非 None。
        if self.score is not None and self.sync_status != "synced":
            raise ValueError(
                f"score must be None when sync_status != 'synced', got sync_status={self.sync_status}"
            )
        
        # (3) 校验 score 值范围。
        if self.score is not None and not isinstance(self.score, (int, float)):
            raise ValueError(
                f"score must be float or None, got {type(self.score)}"
            )
        if isinstance(self.score, float) and self.score < 0:
            raise ValueError(
                f"score must be non-negative, got {self.score}"
            )
        
        # (4) 校验 audit 是 dict。
        if not isinstance(self.audit, dict):
            raise ValueError(
                f"audit must be dict, got {type(self.audit)}"
            )
        
        # (5) 校验 audit 必有字段。
        required_audit_fields = {"impl_identity", "impl_version", "impl_digest", "trace_digest", "sync_status_detail"}
        missing_fields = required_audit_fields - set(self.audit.keys())
        if missing_fields:
            raise ValueError(
                f"audit missing required fields: {missing_fields}"
            )


class GeometryExtractor(Protocol):
    """
    功能：几何链提取器协议。
    
    Protocol for geometry evidence extraction and synchronization implementations.
    Any class implementing this protocol must provide extract() method.
    
    Note: This is a frozen interface. Implementations must not change the signature.
    """
    
    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any]) -> GeometryEvidence:
        """
        功能：从输入中提取几何链证据。
        
        Extract geometry evidence from inputs.
        
        Args:
            cfg: Configuration dict containing geometry parameters, sync thresholds, etc.
            inputs: Input dict containing latent features, trajectories, or spatial data.
        
        Returns:
            GeometryEvidence instance with frozen structure.
        
        Raises:
            ValueError: If cfg or inputs are invalid.
            RuntimeError: If extraction fails (sync_status should be "failed" in GeometryEvidence, not raised here).
        """
        ...
