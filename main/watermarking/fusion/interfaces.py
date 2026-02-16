"""
融合决策接口与规则协议

功能说明：
- 定义了融合决策的载体类 `FusionDecision`，包含决策结果、状态、证据摘要和审计信息。
- 定义了融合规则的协议 `FusionRule`，要求实现 `fuse`方法来结合内容和几何证据生成融合决策。
- 实现了输入验证和错误处理，确保接口的健壮性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol


@dataclass(frozen=True)
class FusionDecision:
    """
    功能：融合决策载体。

    Frozen structure for fusion decision returned by decision rules.

    Args:
        is_watermarked: Final decision value. When decision_status is "decided", this must be bool. When
            decision_status is "abstain" or "error", this must be None.
        decision_status: Outcome status, one of: "decided", "abstain", "error".
        thresholds_digest: Deterministic digest of threshold parameters and rule version.
        evidence_summary: Structured summary dict with required fields: content_score, geometry_score,
            content_status, geometry_status, fusion_rule_id.
        audit: Audit metadata dict containing decision context and trace info.
        fusion_rule_version: Optional fusion rule version string.
        used_threshold_id: Optional threshold id used for decision.
        routing_decisions: Optional routing decision details.
        routing_digest: Optional routing decision digest.
        conditional_fpr_notes: Optional conditional FPR notes.

    Returns:
        None.

    Raises:
        ValueError: If decision_status is invalid.
        ValueError: If is_watermarked does not satisfy decision_status semantics.
        ValueError: If evidence_summary is missing required fields.
    """
    is_watermarked: Optional[bool]
    decision_status: Literal["decided", "abstain", "error"]
    thresholds_digest: str
    evidence_summary: Dict[str, Any]
    audit: Dict[str, Any]
    fusion_rule_version: Optional[str] = None
    used_threshold_id: Optional[str] = None
    routing_decisions: Optional[Dict[str, Any]] = None
    routing_digest: Optional[str] = None
    conditional_fpr_notes: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        # (1) 校验 decision_status 值。
        allowed_status = {"decided", "abstain", "error"}
        if self.decision_status not in allowed_status:
            raise ValueError(
                f"Invalid decision_status: {self.decision_status}. Must be one of {allowed_status}"
            )
        
        # (2) 校验 is_watermarked 与 decision_status 语义一致。
        if self.decision_status == "decided":
            if not isinstance(self.is_watermarked, bool):
                raise ValueError(
                    f"is_watermarked must be bool when decision_status='decided', got {type(self.is_watermarked)}"
                )
        else:
            if self.is_watermarked is not None:
                raise ValueError(
                    "is_watermarked must be None when decision_status is 'abstain' or 'error'"
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

        _validate_optional_str(self.fusion_rule_version, "fusion_rule_version")
        _validate_optional_str(self.used_threshold_id, "used_threshold_id")
        _validate_optional_mapping(self.routing_decisions, "routing_decisions")
        _validate_optional_str(self.routing_digest, "routing_digest")
        _validate_optional_mapping(self.conditional_fpr_notes, "conditional_fpr_notes")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        功能：将 FusionDecision 序列化为可 JSON 化的字典。

        Serialize FusionDecision to a JSON-serializable dict.

        Args:
            None.

        Returns:
            Dict containing frozen field values (all JSON-compatible types).

        Raises:
            ValueError: If any field is not JSON-serializable.
        """
        result = {
            "is_watermarked": self.is_watermarked,
            "decision_status": self.decision_status,
            "thresholds_digest": self.thresholds_digest,
            "evidence_summary": self.evidence_summary,
            "audit": self.audit,
            "fusion_rule_version": self.fusion_rule_version,
            "used_threshold_id": self.used_threshold_id,
            "routing_decisions": self.routing_decisions,
            "routing_digest": self.routing_digest,
            "conditional_fpr_notes": self.conditional_fpr_notes
        }
        _validate_json_like(result, "fusion_decision")
        return result


class FusionRule(Protocol):
    """
    功能：融合决策规则协议。

    Protocol for fusion rule implementations that combine content and geometry evidence.
    Implementations must not change the signature.

    Returns:
        None.
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
            cfg: Configuration mapping containing thresholds and rule parameters.
            content_evidence: Content evidence object or mapping.
            geometry_evidence: Geometry evidence object or mapping.

        Returns:
            FusionDecision instance.

        Raises:
            ValueError: If inputs are invalid.
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
