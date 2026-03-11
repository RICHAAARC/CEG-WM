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
        anchor_digest: Optional anchor digest.
        anchor_metrics: Optional anchor stability metrics.
        sync_digest: Optional sync template digest.
        sync_metrics: Optional sync quality metrics.
        align_trace_digest: Optional alignment trace digest. 【历史兼容字段，v2.0 正式路径不写出，仅用于读取历史 records】
        align_metrics: Optional geometry alignment statistics. 【历史兼容字段，v2.0 正式路径不写出，仅用于读取历史 records】
        align_config_digest: Optional geometry alignment config digest. 【历史兼容字段，v2.0 正式路径不写出，仅用于读取历史 records】
        geo_score: Optional geometry score.
        geo_failure_reason: Optional geometry failure reason.
               All values must be JSON-serializable strings or dicts.
    
    Raises:
        ValueError: If sync_status is not one of allowed values.
        ValueError: If score is not None when sync_status != "synced".
        ValueError: If audit is missing required fields.
    """
    sync_status: Literal["synced", "absent", "failed"]
    score: Optional[float]
    audit: Dict[str, Any]
    anchor_digest: Optional[str] = None
    anchor_config_digest: Optional[str] = None
    anchor_metrics: Optional[Dict[str, Any]] = None
    stability_metrics: Optional[Dict[str, Any]] = None
    resolution_binding: Optional[Dict[str, Any]] = None
    sync_digest: Optional[str] = None
    sync_metrics: Optional[Dict[str, Any]] = None
    sync_config_digest: Optional[str] = None
    sync_quality_metrics: Optional[Dict[str, Any]] = None
    align_trace_digest: Optional[str] = None
    align_metrics: Optional[Dict[str, Any]] = None
    align_config_digest: Optional[str] = None
    geo_score: Optional[float] = None
    geo_score_direction: Optional[str] = None
    geo_failure_reason: Optional[str] = None
    
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

        _validate_optional_str(self.anchor_digest, "anchor_digest")
        _validate_optional_str(self.anchor_config_digest, "anchor_config_digest")
        _validate_optional_mapping(self.anchor_metrics, "anchor_metrics")
        _validate_optional_mapping(self.stability_metrics, "stability_metrics")
        _validate_optional_mapping(self.resolution_binding, "resolution_binding")
        _validate_optional_str(self.sync_digest, "sync_digest")
        _validate_optional_mapping(self.sync_metrics, "sync_metrics")
        _validate_optional_str(self.sync_config_digest, "sync_config_digest")
        _validate_optional_mapping(self.sync_quality_metrics, "sync_quality_metrics")
        _validate_optional_str(self.align_trace_digest, "align_trace_digest")
        _validate_optional_mapping(self.align_metrics, "align_metrics")
        _validate_optional_str(self.align_config_digest, "align_config_digest")
        _validate_optional_number(self.geo_score, "geo_score")
        _validate_optional_str(self.geo_score_direction, "geo_score_direction")
        _validate_optional_str(self.geo_failure_reason, "geo_failure_reason")

    def as_dict(self) -> Dict[str, Any]:
        """
        功能：将 GeometryEvidence 序列化为可 JSON 化的字典。

        Serialize GeometryEvidence to a JSON-serializable dict.

        Args:
            None.

        Returns:
            Dict containing frozen field values (all JSON-compatible types).

        Raises:
            ValueError: If any field is not JSON-serializable.
        """
        result = {
            "sync_status": self.sync_status,
            "score": self.score,
            "audit": self.audit,
            "anchor_digest": self.anchor_digest,
            "anchor_config_digest": self.anchor_config_digest,
            "anchor_metrics": self.anchor_metrics,
            "stability_metrics": self.stability_metrics,
            "resolution_binding": self.resolution_binding,
            "sync_digest": self.sync_digest,
            "sync_metrics": self.sync_metrics,
            "sync_config_digest": self.sync_config_digest,
            "sync_quality_metrics": self.sync_quality_metrics,
            # 历史兼容字段（v2.0 正式路径不序列化）：align_trace_digest / align_metrics / align_config_digest
            # 已退出当前 formal path，仅保留 dataclass 属性供历史 records 读取层使用。
            "geo_score": self.geo_score,
            "geo_score_direction": self.geo_score_direction,
            "geo_failure_reason": self.geo_failure_reason
        }
        _validate_json_like(result, "geometry_evidence")
        return result


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
