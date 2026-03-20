"""
injection_scope_manifest.yaml 并进行类型校验与规范化

功能说明：
- 加载 injection_scope_manifest.yaml 并进行类型校验与规范化。
- 计算 manifest 的 semantic digest、file_sha256、canon_sha256 与 bound digest。
- 提供统一加载入口，避免解释面分叉。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, cast

from main.core import config_loader
from main.core import digests
from main.core.errors import MissingRequiredFieldError, FrozenContractPathNotAuthoritativeError


@dataclass
class InjectionScopeManifest:
    """
    功能：注入范围事实源加载结果。

    Parsed injection scope manifest with computed digests.

    Attributes:
        data: Full YAML object.
        injection_scope_manifest_version: Version from YAML.
        injection_scope_manifest_digest: Semantic digest of manifest.
        injection_scope_manifest_file_sha256: SHA256 of raw file.
        injection_scope_manifest_canon_sha256: SHA256 of canonical JSON.
        injection_scope_manifest_bound_digest: Bound digest.
    """
    data: Dict[str, Any]
    injection_scope_manifest_version: str
    injection_scope_manifest_digest: str
    injection_scope_manifest_file_sha256: str
    injection_scope_manifest_canon_sha256: str
    injection_scope_manifest_bound_digest: str


def _is_test_environment() -> bool:
    """
    功能：判断当前是否在测试环境。

    Detect whether the current process is running under pytest.

    Args:
        None.

    Returns:
        True if executing in test environment; otherwise False.
    """
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


def _require_non_empty_str_list(values: Any, field_path: str) -> List[str]:
    """
    功能：校验字符串列表并返回标准化结果。

    Validate that values is a list of non-empty strings.

    Args:
        values: Value to validate.
        field_path: Field path for error context.

    Returns:
        Normalized list of strings.

    Raises:
        TypeError: If values or entries are invalid.
        ValueError: If field_path is invalid.
    """
    field_path_obj: Any = field_path
    if not isinstance(field_path_obj, str) or not field_path_obj:
        # field_path 类型不符合预期，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    normalized_field_path = field_path_obj
    if values is None:
        # values 缺失，必须 fail-fast。
        raise TypeError(f"{normalized_field_path} must be list")
    if not isinstance(values, list):
        # values 类型不符合预期，必须 fail-fast。
        raise TypeError(f"{normalized_field_path} must be list")

    normalized: List[str] = []
    value_items = cast(List[Any], values)
    for index, item in enumerate(value_items):
        if not isinstance(item, str) or not item:
            # list 成员不合法，必须 fail-fast。
            raise TypeError(f"{normalized_field_path}[{index}] must be non-empty str")
        normalized.append(item)
    return normalized


def _validate_manifest_schema(obj: Dict[str, Any]) -> None:
    """
    功能：校验 injection_scope_manifest 的结构与类型。

    Validate injection_scope_manifest schema structure and field types.

    Args:
        obj: Parsed YAML mapping.

    Returns:
        None.

    Raises:
        TypeError: If obj or fields are invalid types.
        MissingRequiredFieldError: If required fields are missing.
    """
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("injection_scope_manifest root must be dict")

    manifest = cast(Dict[str, Any], obj_value)

    manifest_version = manifest.get("injection_scope_manifest_version")
    if not isinstance(manifest_version, str) or not manifest_version:
        raise MissingRequiredFieldError("injection_scope_manifest_version missing or invalid")

    allowed_impl_ids = _require_non_empty_str_list(
        manifest.get("allowed_impl_ids"),
        "allowed_impl_ids"
    )
    _ = allowed_impl_ids

    digest_scope_obj: Any = manifest.get("digest_scope")
    digest_scope = cast(Dict[str, Any], digest_scope_obj) if isinstance(digest_scope_obj, dict) else None
    if not isinstance(digest_scope, dict):
        # digest_scope 类型不符合预期，必须 fail-fast。
        raise TypeError("digest_scope must be dict")

    _require_non_empty_str_list(
        digest_scope.get("cfg_digest_include_paths"),
        "digest_scope.cfg_digest_include_paths"
    )
    _require_non_empty_str_list(
        digest_scope.get("plan_digest_include_paths"),
        "digest_scope.plan_digest_include_paths"
    )

    versioning_policy_obj: Any = manifest.get("versioning_policy")
    versioning_policy = cast(Dict[str, Any], versioning_policy_obj) if isinstance(versioning_policy_obj, dict) else None
    if not isinstance(versioning_policy, dict):
        # versioning_policy 类型不符合预期，必须 fail-fast。
        raise TypeError("versioning_policy must be dict")

    bump_on_digest_scope_change: Any = versioning_policy.get(
        "bump_manifest_version_on_digest_scope_change"
    )
    if not isinstance(bump_on_digest_scope_change, bool):
        # bump_manifest_version_on_digest_scope_change 类型不符合预期，必须 fail-fast。
        raise TypeError(
            "versioning_policy.bump_manifest_version_on_digest_scope_change must be bool"
        )

    bump_on_impl_scope_change: Any = versioning_policy.get(
        "bump_manifest_version_on_impl_scope_change"
    )
    if not isinstance(bump_on_impl_scope_change, bool):
        # bump_manifest_version_on_impl_scope_change 类型不符合预期，必须 fail-fast。
        raise TypeError(
            "versioning_policy.bump_manifest_version_on_impl_scope_change must be bool"
        )

    _require_non_empty_str_list(
        manifest.get("frozen_surface_protected_files"),
        "frozen_surface_protected_files"
    )

    operational_notes = manifest.get("operational_notes")
    if operational_notes is None:
        return
    if isinstance(operational_notes, str):
        if not operational_notes:
            # operational_notes 值为空，必须 fail-fast。
            raise TypeError("operational_notes must be non-empty str when provided")
        return
    if isinstance(operational_notes, list):
        _require_non_empty_str_list(operational_notes, "operational_notes")
        return

    # operational_notes 类型不符合预期，必须 fail-fast。
    raise TypeError("operational_notes must be str or list[str] when provided")


def load_injection_scope_manifest(
    path: str = "configs/injection_scope_manifest.yaml",
    *,
    allow_non_authoritative: bool = False
) -> InjectionScopeManifest:
    """
    功能：加载注入范围事实源并计算 digest。

    Load injection_scope_manifest.yaml and compute all digests.

    Args:
        path: Path to injection_scope_manifest.yaml.
        allow_non_authoritative: Whether to allow non-authoritative paths for tests.

    Returns:
        InjectionScopeManifest instance.

    Raises:
        TypeError: If inputs are invalid.
        MissingRequiredFieldError: If required fields are missing.
        FrozenContractPathNotAuthoritativeError: If non-authoritative path is used.
    """
    path_obj: Any = path
    if not isinstance(path_obj, str) or not path_obj:
        # path 输入不合法，必须 fail-fast。
        raise TypeError("path must be non-empty str")
    allow_non_authoritative_obj: Any = allow_non_authoritative
    if not isinstance(allow_non_authoritative_obj, bool):
        # allow_non_authoritative 输入不合法，必须 fail-fast。
        raise TypeError("allow_non_authoritative must be bool")

    normalized_input_path = path_obj
    allow_non_authoritative_flag = allow_non_authoritative_obj

    normalized_path = Path(normalized_input_path).as_posix()
    if normalized_path != "configs/injection_scope_manifest.yaml":
        # 非权威路径例外仅测试环境可用。
        is_test = _is_test_environment()
        if not (allow_non_authoritative_flag and is_test):
            raise FrozenContractPathNotAuthoritativeError(
                "injection_scope_manifest path is not authoritative",
                field_path="injection_scope_manifest_source_path",
                actual_path=normalized_path
            )

    obj, provenance = config_loader.load_yaml_with_provenance(normalized_input_path)
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # YAML 根类型不符合预期，必须 fail-fast。
        raise TypeError("injection_scope_manifest root must be dict")

    manifest = cast(Dict[str, Any], obj_value)

    _validate_manifest_schema(manifest)

    manifest_version = manifest.get("injection_scope_manifest_version")
    if not isinstance(manifest_version, str) or not manifest_version:
        raise MissingRequiredFieldError("injection_scope_manifest_version missing in manifest")

    manifest_digest = digests.semantic_digest(manifest)
    manifest_file_sha256 = provenance.file_sha256
    manifest_canon_sha256 = provenance.canon_sha256
    manifest_bound_digest = digests.bound_digest(
        version=manifest_version,
        semantic_digest_value=manifest_digest,
        file_sha256_value=manifest_file_sha256,
        canon_sha256_value=manifest_canon_sha256
    )

    return InjectionScopeManifest(
        data=manifest,
        injection_scope_manifest_version=manifest_version,
        injection_scope_manifest_digest=manifest_digest,
        injection_scope_manifest_file_sha256=manifest_file_sha256,
        injection_scope_manifest_canon_sha256=manifest_canon_sha256,
        injection_scope_manifest_bound_digest=manifest_bound_digest
    )
