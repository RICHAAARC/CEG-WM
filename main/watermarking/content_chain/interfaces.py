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
        mask_digest: Optional mask digest (sha256 hex string).
        mask_stats: Optional mask statistics dict.
        plan_digest: Optional content plan digest.
        basis_digest: Optional subspace basis digest.
        lf_trace_digest: Optional low-frequency trace digest.
        hf_trace_digest: Optional high-frequency trace digest.
        lf_score: Optional low-frequency score.
        hf_score: Optional high-frequency score.
        score_parts: Optional structured score components.
        content_failure_reason: Optional content failure reason string.
               All values must be JSON-serializable strings or dicts.
    
    Raises:
        ValueError: If status is not one of allowed values.
        ValueError: If score is None but status is not "absent".
        ValueError: If audit is missing required fields.
    """
    status: Literal["ok", "absent", "fail", "mismatch"]
    score: Optional[float]
    audit: Dict[str, Any]
    mask_digest: Optional[str] = None
    mask_stats: Optional[Dict[str, Any]] = None
    plan_digest: Optional[str] = None
    basis_digest: Optional[str] = None
    lf_trace_digest: Optional[str] = None
    hf_trace_digest: Optional[str] = None
    lf_score: Optional[float] = None
    hf_score: Optional[float] = None
    score_parts: Optional[Dict[str, Any]] = None
    content_failure_reason: Optional[str] = None
    
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

        _validate_optional_str(self.mask_digest, "mask_digest")
        _validate_optional_mapping(self.mask_stats, "mask_stats")
        _validate_optional_str(self.plan_digest, "plan_digest")
        _validate_optional_str(self.basis_digest, "basis_digest")
        _validate_optional_str(self.lf_trace_digest, "lf_trace_digest")
        _validate_optional_str(self.hf_trace_digest, "hf_trace_digest")
        _validate_optional_number(self.lf_score, "lf_score")
        _validate_optional_number(self.hf_score, "hf_score")
        _validate_optional_mapping(self.score_parts, "score_parts")
        _validate_optional_str(self.content_failure_reason, "content_failure_reason")

    def as_dict(self) -> Dict[str, Any]:
        """
        功能：将 ContentEvidence 序列化为可 JSON 化的字典。

        Serialize ContentEvidence to a JSON-serializable dict.

        Args:
            None.

        Returns:
            Dict containing frozen field values (all JSON-compatible types).

        Raises:
            ValueError: If any field is not JSON-serializable.
        """
        result = {
            "status": self.status,
            "score": self.score,
            "audit": self.audit,
            "mask_digest": self.mask_digest,
            "mask_stats": self.mask_stats,
            "plan_digest": self.plan_digest,
            "basis_digest": self.basis_digest,
            "lf_trace_digest": self.lf_trace_digest,
            "hf_trace_digest": self.hf_trace_digest,
            "lf_score": self.lf_score,
            "hf_score": self.hf_score,
            "score_parts": self.score_parts,
            "content_failure_reason": self.content_failure_reason
        }
        _validate_json_like(result, "content_evidence")
        return result


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


def _validate_optional_str(value: Any, field_name: str) -> None:
    """
    功能：校验可选字符串字段。

    Validate optional string field.

    Args:
        value: Field value.
        field_name: Field name for error context.

    Returns:
        None.

    Raises:
        TypeError: If field_name is invalid.
        ValueError: If value type is invalid.
    """
    if not isinstance(field_name, str) or not field_name:
        # field_name 类型不合法，必须 fail-fast。
        raise TypeError("field_name must be non-empty str")
    if value is None:
        return
    if not isinstance(value, str) or not value:
        # 字段类型不符合预期，必须 fail-fast。
        raise ValueError(f"Type mismatch at {field_name}: expected non-empty str")


def _validate_optional_number(value: Any, field_name: str) -> None:
    """
    功能：校验可选数值字段。

    Validate optional numeric field.

    Args:
        value: Field value.
        field_name: Field name for error context.

    Returns:
        None.

    Raises:
        TypeError: If field_name is invalid.
        ValueError: If value type is invalid.
    """
    if not isinstance(field_name, str) or not field_name:
        # field_name 类型不合法，必须 fail-fast。
        raise TypeError("field_name must be non-empty str")
    if value is None:
        return
    if not isinstance(value, (int, float)):
        # 字段类型不符合预期，必须 fail-fast。
        raise ValueError(f"Type mismatch at {field_name}: expected number")


def _validate_optional_mapping(value: Any, field_name: str) -> None:
    """
    功能：校验可选映射字段。

    Validate optional mapping field.

    Args:
        value: Field value.
        field_name: Field name for error context.

    Returns:
        None.

    Raises:
        TypeError: If field_name is invalid.
        ValueError: If value type is invalid.
    """
    if not isinstance(field_name, str) or not field_name:
        # field_name 类型不合法，必须 fail-fast。
        raise TypeError("field_name must be non-empty str")
    if value is None:
        return
    if not isinstance(value, dict):
        # 字段类型不符合预期，必须 fail-fast。
        raise ValueError(f"Type mismatch at {field_name}: expected dict")
    _validate_json_like(value, field_name)


def _validate_json_like(value: Any, field_path: str) -> None:
    """
    功能：校验 JSON-like 结构。

    Validate that a value is JSON-like (dict/list/scalars).

    Args:
        value: Value to validate.
        field_path: Field path for error messages.

    Returns:
        None.

    Raises:
        TypeError: If value contains non-JSON-like types.
        ValueError: If field_path is invalid.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    if _is_json_scalar(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_like(item, f"{field_path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                # JSON key 类型不合法，必须 fail-fast。
                raise TypeError(f"Type mismatch at {field_path}: expected str keys")
            _validate_json_like(item, f"{field_path}.{key}")
        return
    # 非 JSON-like 类型，必须 fail-fast。
    raise TypeError(f"Type mismatch at {field_path}: non-JSON-like {type(value).__name__}")


def _is_json_scalar(value: Any) -> bool:
    """
    功能：判断是否为 JSON 标量。

    Check whether a value is a JSON scalar.

    Args:
        value: Value to check.

    Returns:
        True if value is JSON scalar; otherwise False.
    """
    return isinstance(value, (str, int, float, bool)) or value is None
