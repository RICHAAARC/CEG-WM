"""
冻结契约与可审计性相关的异常类

功能说明：
- 定义与冻结契约加载、记录绑定、门禁执行、YAML 加载等相关的异常类。
- 每个异常类包含相关的上下文信息字段，便于调试和错误分析。
- 这些异常类在整个框架中被用于捕获和处理特定的错误情况，确保错误信息的清晰和一致性。
- 未来可以根据需要添加更多的异常类或扩展现有类的功能。
"""

from enum import Enum
from typing import List, Optional


class RunFailureReason(str, Enum):
    """
    功能：运行失败原因枚举。

    Run failure reason enumeration for run_closure.

    Args:
        None.

    Returns:
        None.
    """
    RECORDS_ABSENT = "records_absent"
    RECORDS_INCONSISTENT = "records_inconsistent"
    GATE_FAILED = "gate_failed"
    CONFIG_INVALID = "config_invalid"
    IMPL_RESOLVE_FAILED = "impl_resolve_failed"
    RUNTIME_ERROR = "runtime_error"
    OK = "ok"


class DigestCanonicalizationError(Exception):
    """
    功能：digest 规范化失败异常。
    
    Raised when canonical JSON serialization fails due to non-JSON-serializable types.
    
    Args:
        message: Error description.
        field_path: Optional field path that caused the error.
        offending_type: Optional type name of the non-serializable object.
    """
    def __init__(self, message: str, field_path: str | None = None, offending_type: str | None = None):
        self.field_path = field_path
        self.offending_type = offending_type
        super().__init__(message)


class ContractVersionMismatchError(Exception):
    """
    功能：契约版本不匹配异常。
    
    Raised when contract_version in record does not match the loaded contract.
    """
    pass


class DigestMismatchError(Exception):
    """
    功能：digest 不一致异常。
    
    Raised when recorded digest does not match recomputed digest.
    """
    pass


class WhitelistSemanticsMismatchError(Exception):
    """
    功能：whitelist 与 semantics 不一致异常。
    
    Raised when policy_path.allowed set is not consistent with policy_path_semantics keys.
    """
    pass


class MissingRequiredFieldError(Exception):
    """
    功能：必需字段缺失异常。
    
    Raised when required record fields are missing before write.
    """
    pass


class RecordBundleError(Exception):
    """
    功能：records bundle 闭包失败异常。
    
    Raised when records bundle closure fails due to missing or inconsistent anchors.

    Args:
        message: Error description.
        field_name: Optional anchor field name related to the failure.
        files: Optional list of file paths related to the failure.
    """
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        files: Optional[List[str]] = None
    ) -> None:
        self.field_name = field_name
        self.files = list(files) if files is not None else None
        super().__init__(message)


class GateRequirementNotImplementedError(Exception):
    """
    功能：门禁强制要求未实现异常。

    Raised when a must_enforce gate requirement lacks an implementation.

    Args:
        rule_name: Gate requirement name.
        rule_path: Optional contract path for the requirement.
    """
    def __init__(self, rule_name: str, rule_path: Optional[str] = None) -> None:
        self.rule_name = rule_name
        self.rule_path = rule_path
        message = f"Gate requirement not implemented: rule_name={rule_name}"
        if rule_path:
            message = f"{message}, rule_path={rule_path}"
        super().__init__(message)


class ImportEnvironmentError(Exception):
    """
    功能：入口执行环境不一致异常。

    Raised when a CLI entry point is executed outside module mode.

    Args:
        message: Error description.
    """
    pass


class RecordsWritePolicyError(Exception):
    """
    功能：records 写盘门禁策略异常。

    Raised when records write violates path-level gate policy.

    Args:
        message: Error description.
    """
    pass


class GateEnforcementError(Exception):
    """
    功能：门禁执行策略异常。

    Raised when a gate enforcement policy fails.

    Args:
        message: Error description.
        gate_name: Optional gate identifier.
        field_path: Optional field path causing the error.
        expected: Optional expected value description.
        actual: Optional actual value observed.
    """
    
    def __init__(
        self,
        message: str,
        gate_name: str | None = None,
        field_path: str | None = None,
        expected: str | None = None,
        actual: str | None = None
    ) -> None:
        self.gate_name = gate_name
        self.field_path = field_path
        self.expected = expected
        self.actual = actual
        
        # 自动拼接详细错误消息。
        if gate_name or field_path or expected or actual:
            details = []
            if gate_name:
                details.append(f"gate_name={gate_name}")
            if field_path:
                details.append(f"field_path={field_path}")
            if expected is not None:
                details.append(f"expected={expected}")
            if actual is not None:
                details.append(f"actual={actual}")
            full_message = f"{message} ({', '.join(details)})"
        else:
            full_message = message
        
        super().__init__(full_message)


class FrozenContractPathNotAuthoritativeError(Exception):
    """
    功能：冻结契约路径非权威异常。

    Raised when a non-authoritative frozen contracts path is used.

    Args:
        message: Error description.
        field_path: Field path for error context.
        actual_path: Actual path observed at runtime.
    """

    def __init__(self, message: str, field_path: str | None = None, actual_path: str | None = None) -> None:
        self.field_path = field_path
        self.actual_path = actual_path
        super().__init__(message)

class FactSourcesNotInitializedError(Exception):
    """
    功能：事实源未初始化异常。
    
    Raised when attempting to write before binding fact sources context.
    This enforces that CLI entry points must initialize fact sources before any record writes.
    """
    pass


class ContractInterpretationRequiredError(Exception):
    """
    功能：解释面缺失异常。

    Raised when a ContractInterpretation is required but not provided.

    Args:
        message: Error description.

    Returns:
        None.
    """
    pass

class YAMLLoadError(Exception):
    """
    功能：YAML 加载失败异常。
    
    Raised when YAML parsing or loading fails.
    """
    pass
