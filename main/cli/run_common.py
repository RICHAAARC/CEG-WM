"""
CLI 入口层公共辅助逻辑

功能说明：
- 提供 CLI 入口层共享的冻结面相关辅助函数，避免入口分叉导致的语义漂移。
"""

from typing import Dict, Any
from main.core.contracts import FrozenContracts, get_contract_interpretation
from main.core.errors import MissingRequiredFieldError, RunFailureReason
from main.registries import runtime_resolver


def set_value_by_field_path(record: Dict[str, Any], field_path: str, value: str) -> None:
    """
    功能：按点路径写入 record 字段值。

    Set a nested record value by dotted field path.

    Args:
        record: Record dict to mutate.
        field_path: Dotted field path.
        value: Value to set.

    Returns:
        None.

    Raises:
        TypeError: If record is not dict.
        ValueError: If field_path or value is invalid.
    """
    if not field_path:
        # value 输入不合法，必须 fail-fast。
        raise ValueError("value must be non-empty str")

    current = record
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if segment not in current or not isinstance(current[segment], dict):
            current[segment] = {}
        current = current[segment]
    last = segments[-1]
    if not last:
        # field_path 末段为空，必须 fail-fast。
        raise ValueError(f"Invalid field_path segment in {field_path}")
    current[last] = value


def bind_impl_identity_fields(
    record: Dict[str, Any],
    identity: runtime_resolver.ImplIdentity,
    impl_set: runtime_resolver.BuiltImplSet,
    contracts: FrozenContracts
) -> None:
    """
    功能：绑定 impl_identity 相关字段。

    Bind impl identity, version, and digest fields into record.

    Args:
        record: Record dict to mutate.
        identity: Impl identity mapping.
        impl_set: Built implementation set.
        contracts: Loaded FrozenContracts.

    Returns:
        None.

    Raises:
        MissingRequiredFieldError: If required impl fields are missing.
        TypeError: If inputs are invalid.
    """
    interpretation = get_contract_interpretation(contracts)
    impl_identity_spec = interpretation.impl_identity
    field_paths_by_domain = impl_identity_spec.field_paths_by_domain
    version_field_paths_by_domain = impl_identity_spec.version_field_paths_by_domain
    digest_field_paths_by_domain = impl_identity_spec.digest_field_paths_by_domain

    impl_id_by_domain = {
        "content_extractor": identity.content_extractor_id,
        "geometry_extractor": identity.geometry_extractor_id,
        "fusion_rule": identity.fusion_rule_id,
        "subspace_planner": identity.subspace_planner_id,
        "sync_module": identity.sync_module_id
    }
    impl_object_by_domain = {
        "content_extractor": impl_set.content_extractor,
        "geometry_extractor": impl_set.geometry_extractor,
        "fusion_rule": impl_set.fusion_rule,
        "subspace_planner": impl_set.subspace_planner,
        "sync_module": impl_set.sync_module
    }

    for domain, field_path in field_paths_by_domain.items():
        impl_id = impl_id_by_domain.get(domain)
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 缺失或类型不合法，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required impl_id for domain={domain}, field_path={field_path}"
            )
        set_value_by_field_path(record, field_path, impl_id)

    for domain, field_path in version_field_paths_by_domain.items():
        impl_object = impl_object_by_domain.get(domain)
        impl_version = getattr(impl_object, "impl_version", None)
        if not isinstance(impl_version, str) or not impl_version:
            # impl_version 缺失或类型不合法，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required impl_version for domain={domain}, field_path={field_path}"
            )
        set_value_by_field_path(record, field_path, impl_version)

    for domain, field_path in digest_field_paths_by_domain.items():
        impl_object = impl_object_by_domain.get(domain)
        impl_digest = getattr(impl_object, "impl_digest", None)
        if not isinstance(impl_digest, str) or not impl_digest:
            # impl_digest 缺失或类型不合法，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required impl_digest for domain={domain}, field_path={field_path}"
            )
        set_value_by_field_path(record, field_path, impl_digest)


def set_failure_status(run_meta: Dict[str, Any], reason: RunFailureReason, exc: Exception) -> None:
    """
    功能：设置失败状态与原因（结构化）。

    Set failure status with controlled reason and structured details.
    Ensures GateEnforcementError is serialized with structured fields.
    For all other exceptions, provides exc_type, message, and stack_fingerprint.

    Args:
        run_meta: Run metadata mapping to mutate.
        reason: Run failure reason enum.
        exc: Raised exception.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
    """
    from main.core.errors import GateEnforcementError
    import traceback
    import sys

    run_meta["status_ok"] = False
    run_meta["status_reason"] = reason

    # 结构化 GateEnforcementError。
    if isinstance(exc, GateEnforcementError):
        run_meta["status_details"] = {
            "gate_name": exc.gate_name if exc.gate_name is not None else "<absent>",
            "field_path": exc.field_path if exc.field_path is not None else "<absent>",
            "expected": exc.expected if exc.expected is not None else "<absent>",
            "actual": exc.actual if exc.actual is not None else "<absent>",
            "message": str(exc)
        }
    else:
        # 其他异常：提取 exc_type、message、stack_fingerprint。
        exc_type = type(exc).__name__
        message = str(exc)

        # 生成稳定的 stack_fingerprint：取首帧的 module:function:lineno。
        stack_fingerprint = "<absent>"
        try:
            tb = sys.exc_info()[2]
            if tb is not None:
                # 获取首帧（抛出点）。
                frame = tb.tb_frame
                filename = frame.f_code.co_filename
                function = frame.f_code.co_name
                lineno = tb.tb_lineno
                # 简化路径为模块名（去掉绝对路径前缀）。
                from pathlib import Path
                try:
                    module_path = Path(filename).relative_to(Path.cwd())
                    module_name = str(module_path).replace("\\", "/").replace(".py", "").replace("/", ".")
                except ValueError:
                    module_name = Path(filename).name.replace(".py", "")
                stack_fingerprint = f"{module_name}:{function}:{lineno}"
        except Exception:
            # 若提取失败，保持 <absent>。
            pass

        run_meta["status_details"] = {
            "exc_type": exc_type,
            "message": message,
            "stack_fingerprint": stack_fingerprint
        }



def format_fact_sources_mismatch(snapshot: Dict[str, Any], bound_fact_sources: Dict[str, Any]) -> str:
    """
    功能：格式化事实源快照不一致错误信息。

    Format snapshot mismatch details for fact sources.

    Args:
        snapshot: Snapshot mapping built before binding context.
        bound_fact_sources: Mapping from records_io.get_bound_fact_sources().

    Returns:
        Mismatch message with field path and both values.
    """
    keys = sorted(set(snapshot.keys()) | set(bound_fact_sources.keys()))
    for key in keys:
        left = snapshot.get(key)
        right = bound_fact_sources.get(key)
        if left != right:
            return (
                "bound_fact_sources snapshot mismatch: "
                f"field={key}, snapshot={left}, bound={right}"
            )

    return "bound_fact_sources snapshot mismatch: field=unknown"
