"""
内容链证据接口类型定义与冻结协议

功能说明：
- 内容链证据接口类型定义。
- 冻结内容链证据结构和提取器接口，确保实现的一致性和可验证性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol

from main.core.status import ALLOWED_STATUS_VALUES


@dataclass(frozen=True)
class ContentEvidence:
    """
    功能：内容链证据载体。
    
    Frozen structure for content evidence returned by extractors.
    All values must be JSON-serializable and conform to frozen contracts.
    
    Attributes:
        status: Content detection status enumeration from main.core.status.ALLOWED_STATUS_VALUES.
                One of: "ok", "absent", "failed", "mismatch". Semantics:
                  - "ok": Detection completed successfully, score is valid (non-None float).
                  - "absent": Extraction not performed or unavailable (score must be None).
                  - "failed": Detection failed due to error (score must be None, content_failure_reason required).
                  - "mismatch": Evidence inconsistent with plan/config (score must be None, content_failure_reason required).
        score: Detection score. Valid only when status="ok"; must be None for all other statuses.
               When status="ok": non-negative float (higher indicates stronger watermark evidence).
               When status!="ok": must be None; failure cause expressed via content_failure_reason.
        audit: Audit metadata dict with required keys: impl_identity, impl_version, impl_digest, trace_digest.
        mask_digest: Optional mask digest (sha256 hex string).
        mask_stats: Optional mask statistics dict.
        plan_digest: Optional content plan digest.
        basis_digest: Optional subspace basis digest.
        lf_trace_digest: Optional low-frequency trace digest.
        hf_trace_digest: Optional high-frequency trace digest.
        lf_score: Optional low-frequency score.
        hf_score: Optional high-frequency score.
        score_parts: Optional structured score components.
        trajectory_evidence: Optional trajectory tap evidence mapping.
        content_failure_reason: Optional failure reason enumeration string.
               Required when status="failed" or status="mismatch".
               All values must be JSON-serializable.
    
    Raises:
        ValueError: If status violates frozen enumeration.
        ValueError: If score/content_failure_reason semantics mismatch status.
        ValueError: If audit lacks required fields.
        TypeError: If any field has invalid type.
    """
    status: Literal["ok", "absent", "failed", "mismatch"]
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
    trajectory_evidence: Optional[Dict[str, Any]] = None
    lf_statistics_digest: Optional[str] = None
    hf_statistics_digest: Optional[str] = None
    content_failure_reason: Optional[str] = None
    
    def __post_init__(self) -> None:
        # 校验 status 值：必须符合権威枚举源 ALLOWED_STATUS_VALUES。
        # frozen contracts 要求 status 枚举白名单由 main.core.status 统一定义。
        allowed_enum_set = set(ALLOWED_STATUS_VALUES)
        if self.status not in allowed_enum_set:
            raise ValueError(
                f"Invalid status: {self.status}. Must be one of {sorted(allowed_enum_set)}"
            )
        
        # 校验 score 与 status 一致性：
        # - status="ok" 时：
        #   - 通常 score 必须是非 None 的非负浮点数（检测内容）；
        #   - 特例：若 mask_digest 非 None，则为结构证据（掩码），score=None 允许。
        # - status!="ok" 时，score 必须为 None（失败/缺失/不一致态的分数无效）。
        if self.status == "ok":
            # 掩码等结构证据：score=None 且 mask_digest 非空。
            if self.score is None and self.mask_digest is not None:
                # 结构证据（掩码已提取），score=None 允许。
                pass
            # 常规检测内容：score 非 None。
            elif self.score is None and self.mask_digest is None:
                raise ValueError(
                    "score must be non-None float when status=ok (except for structural evidence with mask_digest)"
                )
            elif self.score is not None:
                # score 存在时必须是有效数值。
                if not isinstance(self.score, (int, float)):
                    raise ValueError(
                        f"score must be float or int, got {type(self.score).__name__}"
                    )
                if self.score < 0:
                    raise ValueError(
                        f"score must be non-negative, got {self.score}"
                    )
        else:
            if self.score is not None:
                raise ValueError(
                    f"score must be None when status={self.status}, got {self.score}"
                )
            if self.status in {"failed", "mismatch"}:
                if self.content_failure_reason is None:
                    raise ValueError(
                        f"content_failure_reason is required when status={self.status}"
                    )
        
        # 校验 audit 是 dict。
        if not isinstance(self.audit, dict):
            raise ValueError(
                f"audit must be dict, got {type(self.audit).__name__}"
            )
        
        # 校验 audit 包含必需的实现身份字段。
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
        _validate_optional_mapping(self.trajectory_evidence, "trajectory_evidence")
        _validate_optional_str(self.lf_statistics_digest, "lf_statistics_digest")
        _validate_optional_str(self.hf_statistics_digest, "hf_statistics_digest")
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
            "trajectory_evidence": self.trajectory_evidence,
            "lf_statistics_digest": self.lf_statistics_digest,
            "hf_statistics_digest": self.hf_statistics_digest,
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
    
    def extract(self, cfg: Dict[str, Any], inputs: Optional[Dict[str, Any]] = None, cfg_digest: Optional[str] = None) -> ContentEvidence:
        """
        功能：从输入中提取内容链证据。
        
        Extract content evidence from inputs.
        
        Args:
            cfg: Configuration dict containing model paths, thresholds, and parameters.
            inputs: Optional input dict containing latent features, embeddings, or raw data.
            cfg_digest: Optional canonical SHA256 digest of cfg (computed from include_paths).
                       When provided, mask_digest will bind to this authoritative digest
                       instead of recomputing from full cfg. Enables reproducibility and
                       prevents non-digest-scope fields from affecting mask_digest.
        
        Returns:
            ContentEvidence instance with frozen structure.
        
        Raises:
            ValueError: If cfg or inputs are invalid.
            RuntimeError: If extraction fails (status should be "failed" in ContentEvidence, not raised here).
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
