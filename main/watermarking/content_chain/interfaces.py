"""
内容链证据接口类型定义与冻结协议

功能说明：
- 内容链证据接口类型定义。
- 冻结内容链证据结构和提取器接口，确保实现的一致性和可验证性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol


@dataclass(frozen=True)
class ContentEvidence:
    """
    功能：内容链证据载体。
    
    Frozen structure for content evidence returned by extractors.
    
    Attributes:
        status: Content detection status, one of: "ok", "absent", "fail", "mismatch".
                Mutually exclusive values. Semantics:
                  - "ok": Detection completed successfully, score is valid.
                  - "absent": Extraction not performed or result unavailable (score must be None).
                  - "fail": Detection failed due to error (score typically None, audit must explain).
                  - "mismatch": Detection completed but evidence inconsistent (score may or may not exist).
        score: Detection score. Must be None only when status="absent". 
               Higher score indicates stronger evidence of watermarking.
               Must be non-negative float or None.
        audit: Audit metadata dict. Must contain keys: impl_identity, impl_version, impl_digest, trace_digest.
               All values must be JSON-serializable strings or dicts.
    
    Raises:
        ValueError: If status is not one of allowed values.
        ValueError: If score is None but status is not "absent".
        ValueError: If audit is missing required fields.
    """
    status: Literal["ok", "absent", "fail", "mismatch"]
    score: Optional[float]
    audit: Dict[str, Any]
    
    def __post_init__(self) -> None:
        # 校验 status 值。
        allowed_status = {"ok", "absent", "fail", "mismatch"}
        if self.status not in allowed_status:
            raise ValueError(
                f"Invalid status: {self.status}. Must be one of {allowed_status}"
            )
        
        # 校验 score 与 status 一致性：只有 status="absent" 时 score 才能是 None。
        if self.score is None and self.status != "absent":
            raise ValueError(
                f"score must be None only when status='absent', got status={self.status}"
            )
        
        # 校验 score 值范围。
        if self.score is not None and not isinstance(self.score, (int, float)):
            raise ValueError(
                f"score must be float or None, got {type(self.score)}"
            )
        if isinstance(self.score, float) and self.score < 0:
            raise ValueError(
                f"score must be non-negative, got {self.score}"
            )
        
        # 校验 audit 是 dict。
        if not isinstance(self.audit, dict):
            raise ValueError(
                f"audit must be dict, got {type(self.audit)}"
            )
        
        # 校验 audit 必有字段。
        required_audit_fields = {"impl_identity", "impl_version", "impl_digest", "trace_digest"}
        missing_fields = required_audit_fields - set(self.audit.keys())
        if missing_fields:
            raise ValueError(
                f"audit missing required fields: {missing_fields}"
            )


class ContentExtractor(Protocol):
    """
    功能：内容链提取器协议。
    
    Protocol for content evidence extraction implementations.
    Any class implementing this protocol must provide extract() method.
    
    Note: This is a frozen interface. Implementations must not change the signature.
    """
    
    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any]) -> ContentEvidence:
        """
        功能：从输入中提取内容链证据。
        
        Extract content evidence from inputs.
        
        Args:
            cfg: Configuration dict containing model paths, thresholds, and parameters.
            inputs: Input dict containing latent features, embeddings, or raw data.
        
        Returns:
            ContentEvidence instance with frozen structure.
        
        Raises:
            ValueError: If cfg or inputs are invalid.
            RuntimeError: If extraction fails (status should be "fail" in ContentEvidence, not raised here).
        """
        ...
