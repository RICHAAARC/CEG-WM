"""
records_schema_extensions.yaml 加载与解析

功能说明：
- 加载 records_schema_extensions.yaml 并进行类型校验与规范化。
- 计算 extension 的 semantic digest、file_sha256、canon_sha256 与 bound digest。
- 提供统一加载入口，避免解释面分叉。
- 支持向后兼容：若 YAML 不存在，允许返回空扩展（通过参数控制）。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from main.core import config_loader
from main.core import digests
from main.core.errors import MissingRequiredFieldError, FrozenContractPathNotAuthoritativeError


@dataclass
class RecordsSchemaExtension:
    """
    功能：单个字段扩展条目。

    Single field extension entry.

    Attributes:
        path: Field path (e.g., "content_evidence.mask_digest").
        layer: Layer classification (anchor, diagnostic, etc.).
        type: Type name (digest_hex64, metrics_dict, etc.).
        required: Whether field is required (append-only fields typically false).
        missing_semantics: Semantics when absent (e.g., "absent_ok").
        description: Field description.
    """
    path: str
    layer: str
    type: str
    required: bool
    missing_semantics: str
    description: str


@dataclass
class RecordsSchemaExtensions:
    """
    功能：记录 schema 扩展事实源加载结果。

    Parsed records schema extensions with computed digests.

    Attributes:
        data: Full YAML object.
        extensions_version: Version from YAML.
        extensions_digest: Semantic digest of extensions.
        extensions_file_sha256: SHA256 of raw file.
        extensions_canon_sha256: SHA256 of canonical JSON.
        extensions_bound_digest: Bound digest.
        entries: Parsed extension entries.
        entry_paths: List of field paths.
    """
    data: Dict[str, Any]
    extensions_version: str
    extensions_digest: str
    extensions_file_sha256: str
    extensions_canon_sha256: str
    extensions_bound_digest: str
    entries: List[RecordsSchemaExtension]
    entry_paths: List[str]


@dataclass
class EmptyRecordsSchemaExtensions:
    """
    功能：空扩展占位符（向后兼容）。

    Empty extensions placeholder for backward compatibility.

    Attributes:
        entries: Empty list.
        entry_paths: Empty list.
    """
    entries: List[RecordsSchemaExtension]
    entry_paths: List[str]


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


def _validate_extension_entry(entry: Any, index: int) -> Tuple[RecordsSchemaExtension, List[str]]:
    """
    功能：校验并解析单个字段扩展条目。

    Validate and parse a single extension entry.

    Args:
        entry: Entry object to validate.
        index: Entry index for error context.

    Returns:
        Tuple of (parsed entry, error list).

    Raises:
        TypeError: If entry type is invalid.
    """
    errors: List[str] = []

    if not isinstance(entry, dict):
        # entry 类型不符合预期，必须 fail-fast。
        raise TypeError(f"fields[{index}] must be dict")

    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value:
        errors.append(f"fields[{index}].path must be non-empty str")
        return None, errors  # type: ignore

    layer = entry.get("layer")
    if not isinstance(layer, str) or not layer:
        errors.append(f"fields[{index}].layer must be non-empty str")

    type_value = entry.get("type")
    if not isinstance(type_value, str) or not type_value:
        errors.append(f"fields[{index}].type must be non-empty str")

    required = entry.get("required")
    if not isinstance(required, bool):
        errors.append(f"fields[{index}].required must be bool")

    missing_semantics = entry.get("missing_semantics")
    if not isinstance(missing_semantics, str) or not missing_semantics:
        errors.append(f"fields[{index}].missing_semantics must be non-empty str")

    description = entry.get("description")
    if not isinstance(description, str):
        errors.append(f"fields[{index}].description must be str")

    if errors:
        return None, errors  # type: ignore

    return RecordsSchemaExtension(
        path=path_value,
        layer=layer,
        type=type_value,
        required=required,
        missing_semantics=missing_semantics,
        description=description
    ), []


def _validate_extensions_schema(obj: Dict[str, Any]) -> Tuple[List[str], List[RecordsSchemaExtension]]:
    """
    功能：校验 records_schema_extensions 的结构与类型。

    Validate records_schema_extensions schema structure and field types.

    Args:
        obj: Parsed YAML mapping.

    Returns:
        Tuple of (errors, entries).

    Raises:
        TypeError: If obj or fields are invalid types.
    """
    if not isinstance(obj, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("records_schema_extensions root must be dict")

    errors: List[str] = []
    entries: List[RecordsSchemaExtension] = []

    version = obj.get("version")
    if not isinstance(version, str) or not version:
        errors.append("version must be non-empty str")

    append_only = obj.get("append_only")
    if not isinstance(append_only, bool):
        errors.append("append_only must be bool")

    # 提取并校验 type_definitions（可选）
    type_definitions = obj.get("type_definitions")
    if type_definitions is not None:
        if not isinstance(type_definitions, dict):
            errors.append("type_definitions must be dict when present")

    # 提取并校验 fields（必须）
    fields = obj.get("fields")
    if not isinstance(fields, list):
        errors.append("fields must be list")
        return errors, []

    for index, entry in enumerate(fields):
        parsed_entry, entry_errors = _validate_extension_entry(entry, index)
        errors.extend(entry_errors)
        if parsed_entry is not None:
            entries.append(parsed_entry)

    return errors, entries


def load_records_schema_extensions(
    path: str = "configs/records_schema_extensions.yaml",
    *,
    allow_non_authoritative: bool = False,
    allow_missing: bool = False
) -> Any:
    """
    功能：加载记录 schema 扩展并计算 digest。

    Load records_schema_extensions.yaml and compute all digests.
    Supports backward compatibility: if file is missing and allow_missing=True,
    returns empty extensions structure instead of failing.

    Args:
        path: Path to records_schema_extensions.yaml.
        allow_non_authoritative: Whether to allow non-authoritative paths for tests.
        allow_missing: Whether to allow missing file (for backward compatibility).

    Returns:
        RecordsSchemaExtensions or EmptyRecordsSchemaExtensions instance.

    Raises:
        TypeError: If inputs are invalid.
        MissingRequiredFieldError: If required fields are missing (unless allow_missing=True).
        FrozenContractPathNotAuthoritativeError: If non-authoritative path is used.
    """
    if not isinstance(path, str) or not path:
        # path 输入不合法，必须 fail-fast。
        raise TypeError("path must be non-empty str")
    if not isinstance(allow_non_authoritative, bool):
        # allow_non_authoritative 输入不合法，必须 fail-fast。
        raise TypeError("allow_non_authoritative must be bool")
    if not isinstance(allow_missing, bool):
        # allow_missing 输入不合法，必须 fail-fast。
        raise TypeError("allow_missing must be bool")

    normalized_path = Path(path).as_posix()
    if normalized_path != "configs/records_schema_extensions.yaml":
        # 非权威路径例外仅测试环境可用。
        is_test = _is_test_environment()
        if not (allow_non_authoritative and is_test):
            raise FrozenContractPathNotAuthoritativeError(
                "records_schema_extensions path is not authoritative",
                field_path="records_schema_extensions_source_path",
                actual_path=normalized_path
            )

    # 检查文件是否存在
    path_obj = Path(path)
    if not path_obj.exists():
        # 文件缺失处理：若允许则返回空扩展，否则 fail-fast。
        if allow_missing:
            # 向后兼容：返回空扩展占位符。
            return EmptyRecordsSchemaExtensions(entries=[], entry_paths=[])
        else:
            raise MissingRequiredFieldError(
                "records_schema_extensions.yaml file not found"
            )

    obj, provenance = config_loader.load_yaml_with_provenance(path)
    if not isinstance(obj, dict):
        # YAML 根类型不符合预期，必须 fail-fast。
        raise TypeError("records_schema_extensions root must be dict")

    validation_errors, entries = _validate_extensions_schema(obj)
    if validation_errors:
        # 架构校验失败，必须 fail-fast。
        error_msg = "; ".join(validation_errors)
        raise TypeError(f"records_schema_extensions validation failed: {error_msg}")

    extensions_version = obj.get("version")
    if not isinstance(extensions_version, str) or not extensions_version:
        raise MissingRequiredFieldError("version missing in records_schema_extensions")

    extensions_digest = digests.semantic_digest(obj)
    extensions_file_sha256 = provenance.file_sha256
    extensions_canon_sha256 = provenance.canon_sha256
    extensions_bound_digest = digests.bound_digest(
        version=extensions_version,
        semantic_digest_value=extensions_digest,
        file_sha256_value=extensions_file_sha256,
        canon_sha256_value=extensions_canon_sha256
    )

    entry_paths = [entry.path for entry in entries]

    return RecordsSchemaExtensions(
        data=obj,
        extensions_version=extensions_version,
        extensions_digest=extensions_digest,
        extensions_file_sha256=extensions_file_sha256,
        extensions_canon_sha256=extensions_canon_sha256,
        extensions_bound_digest=extensions_bound_digest,
        entries=entries,
        entry_paths=entry_paths
    )
