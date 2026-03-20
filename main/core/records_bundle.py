"""
records bundle 闭包与 manifest 生成

功能说明：
- 实现闭包逻辑，确保跨文件锚点一致性，并生成包含文件列表和锚点信息的 manifest。
- 包含详细的输入验证和错误处理，确保健壮性和可维护性。
- 依赖 core.records_io 进行原子写入，依赖 core.digests 进行规范化 digest 计算。
- 扩展能力需通过版本化追加接入，且不得改变既有闭包语义与锚点约束。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

from . import digests
from . import records_io
from .errors import RecordBundleError


_REQUIRED_ANCHOR_FIELDS = [
    "contract_bound_digest",
    "whitelist_bound_digest",
    "policy_path_semantics_bound_digest",
]

_OPTIONAL_ANCHOR_FIELDS = [
    "contract_version",
    "whitelist_version",
    "policy_path_semantics_version",
    "policy_path",
]


def close_records_bundle(
    records_dir: Path,
    manifest_name: str = "records_manifest.json",
    manifest_dir: Path | None = None
) -> Path:
    """
    功能：关闭 records bundle 并生成 manifest。
    
    Close records bundle by validating cross-file invariants and writing a manifest.
    
    Args:
        records_dir: Directory containing records files.
        manifest_name: Manifest file name.
        manifest_dir: Directory to write manifest into.
    
    Returns:
        Path to the written manifest.
    
    Raises:
        RecordBundleError: If invariants fail or parsing fails.
    """
    _validate_records_dir(records_dir)
    _validate_manifest_name(manifest_name)
    manifest_dir = _normalize_manifest_dir(manifest_dir, records_dir)

    record_files = _scan_record_files(records_dir, manifest_name)
    if not record_files:
        # records_dir 下没有可闭包的 records 文件，必须 fail-fast。
        raise RecordBundleError(
            f"No records files found under {records_dir}"
        )

    records_by_file: Dict[Path, List[Dict[str, Any]]] = {}
    file_entries: List[Dict[str, Any]] = []

    for file_path in record_files:
        obj, records, file_type = _load_records_payload(file_path)
        records_by_file[file_path] = records
        file_entries.append(
            _build_file_entry(records_dir, file_path, file_type, obj, len(records))
        )

    anchor_values = _validate_anchor_fields(
        records_by_file,
        records_dir,
        _REQUIRED_ANCHOR_FIELDS,
        _OPTIONAL_ANCHOR_FIELDS
    )

    manifest = _build_manifest(
        records_dir=records_dir,
        manifest_name=manifest_name,
        file_entries=file_entries,
        anchor_values=anchor_values
    )
    _inject_artifact_audit_marker(manifest)
    manifest["bundle_canon_sha256"] = digests.canonical_sha256(
        _strip_bundle_digest(manifest)
    )

    manifest_path = manifest_dir / manifest_name
    try:
        records_io.write_artifact_json(str(manifest_path), manifest)
    except records_io.FactSourcesNotInitializedError:
        # 事实源未初始化时使用 unbound 写盘兜底，避免闭包中断。
        run_root = records_dir.parent
        records_io.write_artifact_json_unbound(
            run_root,
            manifest_dir,
            str(manifest_path),
            manifest
        )
    return manifest_path


def _normalize_manifest_dir(manifest_dir: Path | None, records_dir: Path) -> Path:
    """
    功能：规范化 manifest 输出目录。

    Normalize manifest output directory and ensure it is a directory.

    Args:
        manifest_dir: Optional manifest directory.
        records_dir: Records directory used as fallback.

    Returns:
        Normalized manifest directory path.

    Raises:
        RecordBundleError: If inputs are invalid.
    """
    if manifest_dir is None:
        manifest_dir = records_dir
    manifest_dir_obj: Any = manifest_dir
    if not isinstance(manifest_dir_obj, Path):
        # manifest_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("manifest_dir must be Path")
    normalized_manifest_dir = manifest_dir_obj
    if normalized_manifest_dir.exists() and not normalized_manifest_dir.is_dir():
        # manifest_dir 不是目录，必须 fail-fast。
        raise RecordBundleError(f"manifest_dir is not a directory: {normalized_manifest_dir}")
    normalized_manifest_dir.mkdir(parents=True, exist_ok=True)
    return normalized_manifest_dir.resolve()


def _inject_artifact_audit_marker(manifest: Dict[str, Any]) -> None:
    """
    功能：注入 records manifest 的审计标识。

    Inject deterministic audit marker into manifest to match artifact writes.

    Args:
        manifest: Manifest mapping to mutate.

    Returns:
        None.

    Raises:
        RecordBundleError: If inputs are invalid.
    """
    manifest_obj: Any = manifest
    if not isinstance(manifest_obj, dict):
        # manifest 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("manifest must be dict")
    manifest_mapping = cast(Dict[str, Any], manifest_obj)
    if "_artifact_audit" in manifest_mapping:
        return
    manifest_mapping["_artifact_audit"] = {
        "schema_version": "v1.0",
        "writer": "records_io"
    }


def _validate_records_dir(records_dir: Path) -> None:
    """
    功能：校验 records_dir 输入。
    
    Validate records_dir input.
    
    Args:
        records_dir: Directory path to validate.
    
    Raises:
        RecordBundleError: If input is invalid.
    """
    records_dir_obj: Any = records_dir
    if not isinstance(records_dir_obj, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    normalized_records_dir = records_dir_obj
    if not normalized_records_dir.exists():
        # records_dir 不存在，必须 fail-fast。
        raise RecordBundleError(f"records_dir does not exist: {normalized_records_dir}")
    if not normalized_records_dir.is_dir():
        # records_dir 不是目录，必须 fail-fast。
        raise RecordBundleError(f"records_dir is not a directory: {normalized_records_dir}")


def _validate_manifest_name(manifest_name: str) -> None:
    """
    功能：校验 manifest_name 输入。
    
    Validate manifest_name input.
    
    Args:
        manifest_name: Manifest file name.
    
    Raises:
        RecordBundleError: If input is invalid.
    """
    manifest_name_obj: Any = manifest_name
    if not isinstance(manifest_name_obj, str) or not manifest_name_obj:
        # manifest_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("manifest_name must be non-empty str")


def _scan_record_files(records_dir: Path, manifest_name: str) -> List[Path]:
    """
    功能：扫描 records_dir 下的 records 文件。
    
    Scan records_dir for *.json and *.jsonl, excluding manifest and temp files.
    
    Args:
        records_dir: Directory to scan.
        manifest_name: Manifest file name to exclude.
    
    Returns:
        Sorted list of record file paths.
    """
    records_dir_obj: Any = records_dir
    if not isinstance(records_dir_obj, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    manifest_name_obj: Any = manifest_name
    if not isinstance(manifest_name_obj, str) or not manifest_name_obj:
        # manifest_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("manifest_name must be non-empty str")

    normalized_records_dir = records_dir_obj
    normalized_manifest_name = manifest_name_obj

    candidates = list(normalized_records_dir.glob("*.json")) + list(normalized_records_dir.glob("*.jsonl"))
    filtered: List[Path] = []
    for path in candidates:
        if _is_excluded_file(path, normalized_manifest_name):
            continue
        filtered.append(path)

    return sorted(filtered, key=lambda p: p.as_posix())


def _is_excluded_file(path: Path, manifest_name: str) -> bool:
    """
    功能：判断是否排除文件。
    
    Determine whether a file should be excluded from scanning.
    
    Args:
        path: File path.
        manifest_name: Manifest file name.
    
    Returns:
        True if excluded.
    """
    path_obj: Any = path
    if not isinstance(path_obj, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("path must be Path")
    manifest_name_obj: Any = manifest_name
    if not isinstance(manifest_name_obj, str) or not manifest_name_obj:
        # manifest_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("manifest_name must be non-empty str")

    normalized_path = path_obj
    normalized_manifest_name = manifest_name_obj
    name = normalized_path.name
    if name == normalized_manifest_name:
        return True
    if name.startswith(".tmp-"):
        return True
    if name.endswith(".writing"):
        return True
    return False


def _load_records_payload(path: Path) -> Tuple[Any, List[Dict[str, Any]], str]:
    """
    功能：加载 records 文件并返回解析结果。
    
    Load a records file and return (obj, records_list, file_type).
    
    Args:
        path: File path to load.
    
    Returns:
        Tuple of parsed object, list of record dicts, and file type.
    
    Raises:
        RecordBundleError: If parsing fails.
    """
    path_obj: Any = path
    if not isinstance(path_obj, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("path must be Path")

    normalized_path = path_obj

    if normalized_path.suffix == ".json":
        obj = _read_json_file(normalized_path)
        obj_value: Any = obj
        if not isinstance(obj_value, dict):
            # json 文件内容类型不符合预期，必须 fail-fast。
            raise RecordBundleError(f"JSON file must contain dict: {normalized_path}")
        record_obj = cast(Dict[str, Any], obj_value)
        return record_obj, [record_obj], "json"

    if normalized_path.suffix == ".jsonl":
        records = _read_jsonl_file(normalized_path)
        return records, records, "jsonl"

    # 扫描结果中出现非支持后缀，必须 fail-fast。
    raise RecordBundleError(f"Unsupported records file type: {normalized_path}")


def _read_json_file(path: Path) -> Any:
    """
    功能：读取 JSON 文件。
    
    Read a JSON file.
    
    Args:
        path: File path.
    
    Returns:
        Parsed JSON object.
    
    Raises:
        RecordBundleError: If JSON parsing fails.
    """
    path_obj: Any = path
    if not isinstance(path_obj, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("path must be Path")

    normalized_path = path_obj

    try:
        with normalized_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # JSON 解析失败，必须 fail-fast。
        raise RecordBundleError(f"Failed to parse JSON file {normalized_path}: {e}") from e


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    """
    功能：读取 JSONL 文件。
    
    Read a JSONL file into a list of dict records.
    
    Args:
        path: File path.
    
    Returns:
        List of record dicts in file order.
    
    Raises:
        RecordBundleError: If JSONL parsing fails.
    """
    path_obj: Any = path
    if not isinstance(path_obj, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("path must be Path")

    normalized_path = path_obj

    records: List[Dict[str, Any]] = []
    try:
        with normalized_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                obj_value: Any = obj
                if not isinstance(obj_value, dict):
                    # JSONL 行内容不是 dict，必须 fail-fast。
                    raise RecordBundleError(
                        f"JSONL line must be dict at {normalized_path} line {line_idx}"
                    )
                records.append(cast(Dict[str, Any], obj_value))
    except RecordBundleError:
        raise
    except Exception as e:
        # JSONL 解析失败，必须 fail-fast。
        raise RecordBundleError(f"Failed to parse JSONL file {normalized_path}: {e}") from e

    if not records:
        # JSONL 为空，必须 fail-fast。
        raise RecordBundleError(f"JSONL file has no records: {normalized_path}")

    return records


def _build_file_entry(
    records_dir: Path,
    file_path: Path,
    file_type: str,
    obj: Any,
    record_count: int
) -> Dict[str, Any]:
    """
    功能：构造 manifest 的文件条目。
    
    Build a manifest file entry.
    
    Args:
        records_dir: Records directory.
        file_path: File path.
        file_type: File type string.
        obj: Parsed object for canonical hash.
        record_count: Number of records in file.
    
    Returns:
        Manifest file entry dict.
    
    Raises:
        RecordBundleError: If hashing fails.
    """
    records_dir_obj: Any = records_dir
    if not isinstance(records_dir_obj, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    file_path_obj: Any = file_path
    if not isinstance(file_path_obj, Path):
        # file_path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("file_path must be Path")
    file_type_obj: Any = file_type
    if not isinstance(file_type_obj, str) or not file_type_obj:
        # file_type 输入不合法，必须 fail-fast。
        raise RecordBundleError("file_type must be non-empty str")
    record_count_obj: Any = record_count
    if not isinstance(record_count_obj, int) or record_count_obj < 0:
        # record_count 输入不合法，必须 fail-fast。
        raise RecordBundleError("record_count must be non-negative int")

    normalized_records_dir = records_dir_obj
    normalized_file_path = file_path_obj
    normalized_file_type = file_type_obj
    normalized_record_count = record_count_obj

    try:
        file_sha256 = digests.file_sha256(normalized_file_path)
    except Exception as e:
        # file_sha256 计算失败，必须 fail-fast。
        raise RecordBundleError(f"Failed to compute file_sha256 for {normalized_file_path}: {e}") from e

    try:
        canon_sha256 = digests.canonical_sha256(obj)
    except Exception as e:
        # canon_sha256 计算失败，必须 fail-fast。
        raise RecordBundleError(f"Failed to compute canon_sha256 for {normalized_file_path}: {e}") from e

    return {
        "path": normalized_file_path.relative_to(normalized_records_dir).as_posix(),
        "file_type": normalized_file_type,
        "record_count": normalized_record_count,
        "file_sha256": file_sha256,
        "canon_sha256": canon_sha256
    }


def _validate_anchor_fields(
    records_by_file: Dict[Path, List[Dict[str, Any]]],
    records_dir: Path,
    required_fields: List[str],
    optional_fields: List[str]
) -> Dict[str, str]:
    """
    功能：校验跨文件一致性锚点。
    
    Validate cross-file anchor fields for consistency.
    
    Args:
        records_by_file: Mapping from file path to record list.
        records_dir: Records directory for path reporting.
        required_fields: Required anchor fields.
        optional_fields: Optional anchor fields.
    
    Returns:
        Dict of anchor field values.
    
    Raises:
        RecordBundleError: If anchors are missing or inconsistent.
    """
    records_by_file_obj: Any = records_by_file
    if not isinstance(records_by_file_obj, dict):
        # records_by_file 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_by_file must be dict")
    records_dir_obj: Any = records_dir
    if not isinstance(records_dir_obj, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")

    records_mapping = cast(Dict[Path, List[Dict[str, Any]]], records_by_file_obj)
    normalized_records_dir = records_dir_obj

    anchor_values: Dict[str, str] = {}

    for field_name in required_fields:
        value_by_file, missing_files, inconsistent_files = _collect_anchor_values(
            records_mapping, normalized_records_dir, field_name, require_presence=True
        )
        missing_files = sorted(set(missing_files))
        inconsistent_files = sorted(set(inconsistent_files))
        if missing_files:
            # 锚点字段缺失，必须 fail-fast。
            raise RecordBundleError(
                f"Anchor field missing: field_name={field_name}, files={missing_files}",
                field_name=field_name,
                files=missing_files
            )
        if inconsistent_files:
            # 锚点字段不一致，必须 fail-fast。
            raise RecordBundleError(
                f"Anchor field inconsistent: field_name={field_name}, files={inconsistent_files}",
                field_name=field_name,
                files=inconsistent_files
            )
        anchor_values[field_name] = _select_anchor_value(value_by_file, field_name)

    for field_name in optional_fields:
        value_by_file, missing_files, inconsistent_files = _collect_anchor_values(
            records_mapping, normalized_records_dir, field_name, require_presence=False
        )
        inconsistent_files = sorted(set(inconsistent_files))
        if inconsistent_files:
            # 可选锚点字段不一致，必须 fail-fast。
            raise RecordBundleError(
                f"Optional anchor field inconsistent: field_name={field_name}, files={inconsistent_files}",
                field_name=field_name,
                files=inconsistent_files
            )
        if value_by_file:
            anchor_values[field_name] = _select_anchor_value(value_by_file, field_name)
        _ = missing_files

    return anchor_values


def _collect_anchor_values(
    records_by_file: Dict[Path, List[Dict[str, Any]]],
    records_dir: Path,
    field_name: str,
    require_presence: bool
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    功能：收集单个字段的文件级值。
    
    Collect per-file values for a single anchor field.
    
    Args:
        records_by_file: Mapping from file path to record list.
        records_dir: Records directory for path reporting.
        field_name: Anchor field name.
        require_presence: Whether missing fields are errors.
    
    Returns:
        Tuple of (value_by_file, missing_files, inconsistent_files).
    """
    records_by_file_obj: Any = records_by_file
    if not isinstance(records_by_file_obj, dict):
        # records_by_file 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_by_file must be dict")
    records_dir_obj: Any = records_dir
    if not isinstance(records_dir_obj, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    field_name_obj: Any = field_name
    if not isinstance(field_name_obj, str) or not field_name_obj:
        # field_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("field_name must be non-empty str")
    require_presence_obj: Any = require_presence
    if not isinstance(require_presence_obj, bool):
        # require_presence 输入不合法，必须 fail-fast。
        raise RecordBundleError("require_presence must be bool")

    records_mapping = cast(Dict[Path, List[Dict[str, Any]]], records_by_file_obj)
    normalized_records_dir = records_dir_obj
    normalized_field_name = field_name_obj
    normalized_require_presence = require_presence_obj

    value_by_file: Dict[str, str] = {}
    missing_files: List[str] = []
    inconsistent_files: List[str] = []

    for file_path, records in records_mapping.items():
        file_key = file_path.relative_to(normalized_records_dir).as_posix()
        status, value = _extract_field_value(records, normalized_field_name, normalized_require_presence, file_key)
        if status == "missing":
            missing_files.append(file_key)
            continue
        if status == "inconsistent":
            inconsistent_files.append(file_key)
            continue
        value_by_file[file_key] = value

    if normalized_require_presence and missing_files:
        return {}, missing_files, []

    if value_by_file:
        unique_values = set(value_by_file.values())
        if len(unique_values) > 1:
            inconsistent_files.extend(sorted(value_by_file.keys()))

    return value_by_file, missing_files, inconsistent_files


def _extract_field_value(
    records: List[Dict[str, Any]],
    field_name: str,
    require_presence: bool,
    file_key: str
) -> Tuple[str, str]:
    """
    功能：抽取单文件字段值并检查一致性。
    
    Extract a field value from all records in a file and check consistency.
    
    Args:
        records: List of record dicts.
        field_name: Anchor field name.
        require_presence: Whether missing fields are errors.
        file_key: File key for error reporting.
    
    Returns:
        Tuple of (status, value).
    """
    records_obj: Any = records
    if not isinstance(records_obj, list):
        # records 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records must be list")
    field_name_obj: Any = field_name
    if not isinstance(field_name_obj, str) or not field_name_obj:
        # field_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("field_name must be non-empty str")
    require_presence_obj: Any = require_presence
    if not isinstance(require_presence_obj, bool):
        # require_presence 输入不合法，必须 fail-fast。
        raise RecordBundleError("require_presence must be bool")
    file_key_obj: Any = file_key
    if not isinstance(file_key_obj, str) or not file_key_obj:
        # file_key 输入不合法，必须 fail-fast。
        raise RecordBundleError("file_key must be non-empty str")

    record_list = cast(List[Dict[str, Any]], records_obj)
    normalized_field_name = field_name_obj
    normalized_require_presence = require_presence_obj
    normalized_file_key = file_key_obj

    first_value: str | None = None
    for record in record_list:
        if normalized_field_name not in record:
            if normalized_require_presence:
                return "missing", ""
            return "missing", ""
        value = record[normalized_field_name]
        if not isinstance(value, str):
            # 锚点字段类型不正确，必须 fail-fast。
            raise RecordBundleError(
                f"Anchor field must be str: field_name={normalized_field_name}, file={normalized_file_key}",
                field_name=normalized_field_name,
                files=[normalized_file_key]
            )
        if first_value is None:
            first_value = value
            continue
        if value != first_value:
            return "inconsistent", ""

    if first_value is None:
        return "missing", ""

    return "ok", first_value


def _select_anchor_value(value_by_file: Dict[str, str], field_name: str) -> str:
    """
    功能：选择 anchor 的统一值。
    
    Select a unified anchor value from per-file values.
    
    Args:
        value_by_file: Mapping from file key to value.
        field_name: Anchor field name.
    
    Returns:
        Selected anchor value.
    
    Raises:
        RecordBundleError: If values are inconsistent.
    """
    value_by_file_obj: Any = value_by_file
    if not isinstance(value_by_file_obj, dict):
        # value_by_file 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("value_by_file must be dict")
    field_name_obj: Any = field_name
    if not isinstance(field_name_obj, str) or not field_name_obj:
        # field_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("field_name must be non-empty str")

    value_mapping = cast(Dict[str, str], value_by_file_obj)
    values = list(value_mapping.values())
    if not values:
        # anchor 值为空，必须 fail-fast。
        raise RecordBundleError(f"No anchor values for {field_name_obj}")

    first_value = values[0]
    return first_value


def _build_manifest(
    records_dir: Path,
    manifest_name: str,
    file_entries: List[Dict[str, Any]],
    anchor_values: Dict[str, str]
) -> Dict[str, Any]:
    """
    功能：构造 manifest 对象。
    
    Build the manifest object.
    
    Args:
        records_dir: Records directory.
        manifest_name: Manifest file name.
        file_entries: File entries list.
        anchor_values: Anchor values dict.
    
    Returns:
        Manifest dict.
    """
    records_dir_obj: Any = records_dir
    if not isinstance(records_dir_obj, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    manifest_name_obj: Any = manifest_name
    if not isinstance(manifest_name_obj, str) or not manifest_name_obj:
        # manifest_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("manifest_name must be non-empty str")
    file_entries_obj: Any = file_entries
    if not isinstance(file_entries_obj, list):
        # file_entries 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("file_entries must be list")
    anchor_values_obj: Any = anchor_values
    if not isinstance(anchor_values_obj, dict):
        # anchor_values 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("anchor_values must be dict")

    normalized_records_dir = records_dir_obj
    normalized_manifest_name = manifest_name_obj
    normalized_file_entries = cast(List[Dict[str, Any]], file_entries_obj)
    normalized_anchor_values = cast(Dict[str, str], anchor_values_obj)

    return {
        "schema_version": "v1.0",
        "records_dir": normalized_records_dir.as_posix(),
        "manifest_name": normalized_manifest_name,
        "file_count": len(normalized_file_entries),
        "files": normalized_file_entries,
        "anchors": normalized_anchor_values
    }


def _strip_bundle_digest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：移除 bundle_canon_sha256 字段。
    
    Remove bundle_canon_sha256 for digest computation.
    
    Args:
        manifest: Manifest dict.
    
    Returns:
        Manifest dict without bundle_canon_sha256.
    """
    manifest_obj: Any = manifest
    if not isinstance(manifest_obj, dict):
        # manifest 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("manifest must be dict")

    stripped = dict(cast(Dict[str, Any], manifest_obj))
    stripped.pop("bundle_canon_sha256", None)
    return stripped
