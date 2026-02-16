"""
File purpose: Injection scope manifest loading and digest anchoring.
Module type: General module

功能说明：
- 加载 injection_scope_manifest.yaml 并进行类型校验与规范化。
- 计算 manifest 的 semantic digest、file_sha256、canon_sha256 与 bound digest。
- 提供统一加载入口，避免解释面分叉。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
    if not isinstance(field_path, str) or not field_path:
        # field_path 类型不符合预期，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    if values is None:
        # values 缺失，必须 fail-fast。
        raise TypeError(f"{field_path} must be list")
    if not isinstance(values, list):
        # values 类型不符合预期，必须 fail-fast。
        raise TypeError(f"{field_path} must be list")

    normalized: List[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item:
            # list 成员不合法，必须 fail-fast。
            raise TypeError(f"{field_path}[{index}] must be non-empty str")
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
    if not isinstance(obj, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("injection_scope_manifest root must be dict")

    manifest_version = obj.get("injection_scope_manifest_version")
    if not isinstance(manifest_version, str) or not manifest_version:
        raise MissingRequiredFieldError("injection_scope_manifest_version missing or invalid")

    allowed_impl_id_set = _require_non_empty_str_list(
        obj.get("allowed_impl_id_set"),
        "allowed_impl_id_set"
    )
    _ = allowed_impl_id_set

    digest_inputs = obj.get("digest_inputs")
    if not isinstance(digest_inputs, dict):
        # digest_inputs 类型不符合预期，必须 fail-fast。
        raise TypeError("digest_inputs must be dict")

    _require_non_empty_str_list(
        digest_inputs.get("cfg_digest_include"),
        "digest_inputs.cfg_digest_include"
    )
    _require_non_empty_str_list(
        digest_inputs.get("plan_digest_include"),
        "digest_inputs.plan_digest_include"
    )

    version_bump_policy = obj.get("version_bump_policy")
    if not isinstance(version_bump_policy, dict):
        # version_bump_policy 类型不符合预期，必须 fail-fast。
        raise TypeError("version_bump_policy must be dict")

    bump_on_digest_inputs_change = version_bump_policy.get("bump_on_digest_inputs_change")
    if not isinstance(bump_on_digest_inputs_change, bool):
        # bump_on_digest_inputs_change 类型不符合预期，必须 fail-fast。
        raise TypeError("version_bump_policy.bump_on_digest_inputs_change must be bool")

    bump_on_impl_param_change = version_bump_policy.get("bump_on_impl_param_change")
    if not isinstance(bump_on_impl_param_change, bool):
        # bump_on_impl_param_change 类型不符合预期，必须 fail-fast。
        raise TypeError("version_bump_policy.bump_on_impl_param_change must be bool")

    _require_non_empty_str_list(
        obj.get("frozen_no_touch_files"),
        "frozen_no_touch_files"
    )

    notes = obj.get("notes")
    if notes is None:
        return
    if isinstance(notes, str):
        if not notes:
            # notes 值为空，必须 fail-fast。
            raise TypeError("notes must be non-empty str when provided")
        return
    if isinstance(notes, list):
        _require_non_empty_str_list(notes, "notes")
        return

    # notes 类型不符合预期，必须 fail-fast。
    raise TypeError("notes must be str or list[str] when provided")


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
    if not isinstance(path, str) or not path:
        # path 输入不合法，必须 fail-fast。
        raise TypeError("path must be non-empty str")
    if not isinstance(allow_non_authoritative, bool):
        # allow_non_authoritative 输入不合法，必须 fail-fast。
        raise TypeError("allow_non_authoritative must be bool")

    normalized_path = Path(path).as_posix()
    if normalized_path != "configs/injection_scope_manifest.yaml":
        # 非权威路径例外仅测试环境可用。
        is_test = _is_test_environment()
        if not (allow_non_authoritative and is_test):
            raise FrozenContractPathNotAuthoritativeError(
                "injection_scope_manifest path is not authoritative",
                field_path="injection_scope_manifest_source_path",
                actual_path=normalized_path
            )

    obj, provenance = config_loader.load_yaml_with_provenance(path)
    if not isinstance(obj, dict):
        # YAML 根类型不符合预期，必须 fail-fast。
        raise TypeError("injection_scope_manifest root must be dict")

    _validate_manifest_schema(obj)

    manifest_version = obj.get("injection_scope_manifest_version")
    if not isinstance(manifest_version, str) or not manifest_version:
        raise MissingRequiredFieldError("injection_scope_manifest_version missing in manifest")

    manifest_digest = digests.semantic_digest(obj)
    manifest_file_sha256 = provenance.file_sha256
    manifest_canon_sha256 = provenance.canon_sha256
    manifest_bound_digest = digests.bound_digest(
        version=manifest_version,
        semantic_digest_value=manifest_digest,
        file_sha256_value=manifest_file_sha256,
        canon_sha256_value=manifest_canon_sha256
    )

    return InjectionScopeManifest(
        data=obj,
        injection_scope_manifest_version=manifest_version,
        injection_scope_manifest_digest=manifest_digest,
        injection_scope_manifest_file_sha256=manifest_file_sha256,
        injection_scope_manifest_canon_sha256=manifest_canon_sha256,
        injection_scope_manifest_bound_digest=manifest_bound_digest
    )
