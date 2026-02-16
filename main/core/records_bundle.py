"""
records bundle 闭包与 manifest 生成

功能说明：
- 实现闭包逻辑，确保跨文件锚点一致性，并生成包含文件列表和锚点信息的 manifest。
- 包含详细的输入验证和错误处理，确保健壮性和可维护性。
- 依赖 core.records_io 进行原子写入，依赖 core.digests 进行规范化 digest 计算。
- 未来可以扩展为支持更多类型的 records 文件、提供更丰富的 manifest 信息。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    if not isinstance(manifest_dir, Path):
        # manifest_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("manifest_dir must be Path")
    if manifest_dir.exists() and not manifest_dir.is_dir():
        # manifest_dir 不是目录，必须 fail-fast。
        raise RecordBundleError(f"manifest_dir is not a directory: {manifest_dir}")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir.resolve()


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
    if not isinstance(manifest, dict):
        # manifest 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("manifest must be dict")
    if "_artifact_audit" in manifest:
        return
    manifest["_artifact_audit"] = {
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
    if not isinstance(records_dir, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    if not records_dir.exists():
        # records_dir 不存在，必须 fail-fast。
        raise RecordBundleError(f"records_dir does not exist: {records_dir}")
    if not records_dir.is_dir():
        # records_dir 不是目录，必须 fail-fast。
        raise RecordBundleError(f"records_dir is not a directory: {records_dir}")


def _validate_manifest_name(manifest_name: str) -> None:
    """
    功能：校验 manifest_name 输入。
    
    Validate manifest_name input.
    
    Args:
        manifest_name: Manifest file name.
    
    Raises:
        RecordBundleError: If input is invalid.
    """
    if not isinstance(manifest_name, str) or not manifest_name:
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
    if not isinstance(records_dir, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    if not isinstance(manifest_name, str) or not manifest_name:
        # manifest_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("manifest_name must be non-empty str")

    candidates = list(records_dir.glob("*.json")) + list(records_dir.glob("*.jsonl"))
    filtered: List[Path] = []
    for path in candidates:
        if _is_excluded_file(path, manifest_name):
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
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("path must be Path")
    if not isinstance(manifest_name, str) or not manifest_name:
        # manifest_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("manifest_name must be non-empty str")

    name = path.name
    if name == manifest_name:
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
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("path must be Path")

    if path.suffix == ".json":
        obj = _read_json_file(path)
        if not isinstance(obj, dict):
            # json 文件内容类型不符合预期，必须 fail-fast。
            raise RecordBundleError(f"JSON file must contain dict: {path}")
        return obj, [obj], "json"

    if path.suffix == ".jsonl":
        records = _read_jsonl_file(path)
        return records, records, "jsonl"

    # 扫描结果中出现非支持后缀，必须 fail-fast。
    raise RecordBundleError(f"Unsupported records file type: {path}")


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
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("path must be Path")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # JSON 解析失败，必须 fail-fast。
        raise RecordBundleError(f"Failed to parse JSON file {path}: {e}") from e


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
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("path must be Path")

    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    # JSONL 行内容不是 dict，必须 fail-fast。
                    raise RecordBundleError(
                        f"JSONL line must be dict at {path} line {line_idx}"
                    )
                records.append(obj)
    except RecordBundleError:
        raise
    except Exception as e:
        # JSONL 解析失败，必须 fail-fast。
        raise RecordBundleError(f"Failed to parse JSONL file {path}: {e}") from e

    if not records:
        # JSONL 为空，必须 fail-fast。
        raise RecordBundleError(f"JSONL file has no records: {path}")

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
    if not isinstance(records_dir, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    if not isinstance(file_path, Path):
        # file_path 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("file_path must be Path")
    if not isinstance(file_type, str) or not file_type:
        # file_type 输入不合法，必须 fail-fast。
        raise RecordBundleError("file_type must be non-empty str")
    if not isinstance(record_count, int) or record_count < 0:
        # record_count 输入不合法，必须 fail-fast。
        raise RecordBundleError("record_count must be non-negative int")

    try:
        file_sha256 = digests.file_sha256(file_path)
    except Exception as e:
        # file_sha256 计算失败，必须 fail-fast。
        raise RecordBundleError(f"Failed to compute file_sha256 for {file_path}: {e}") from e

    try:
        canon_sha256 = digests.canonical_sha256(obj)
    except Exception as e:
        # canon_sha256 计算失败，必须 fail-fast。
        raise RecordBundleError(f"Failed to compute canon_sha256 for {file_path}: {e}") from e

    return {
        "path": file_path.relative_to(records_dir).as_posix(),
        "file_type": file_type,
        "record_count": record_count,
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
    if not isinstance(records_by_file, dict):
        # records_by_file 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_by_file must be dict")
    if not isinstance(records_dir, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")

    anchor_values: Dict[str, str] = {}

    for field_name in required_fields:
        value_by_file, missing_files, inconsistent_files = _collect_anchor_values(
            records_by_file, records_dir, field_name, require_presence=True
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
            records_by_file, records_dir, field_name, require_presence=False
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
    if not isinstance(records_by_file, dict):
        # records_by_file 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_by_file must be dict")
    if not isinstance(records_dir, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    if not isinstance(field_name, str) or not field_name:
        # field_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("field_name must be non-empty str")
    if not isinstance(require_presence, bool):
        # require_presence 输入不合法，必须 fail-fast。
        raise RecordBundleError("require_presence must be bool")

    value_by_file: Dict[str, str] = {}
    missing_files: List[str] = []
    inconsistent_files: List[str] = []

    for file_path, records in records_by_file.items():
        file_key = file_path.relative_to(records_dir).as_posix()
        status, value = _extract_field_value(records, field_name, require_presence, file_key)
        if status == "missing":
            missing_files.append(file_key)
            continue
        if status == "inconsistent":
            inconsistent_files.append(file_key)
            continue
        value_by_file[file_key] = value

    if require_presence and missing_files:
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
    if not isinstance(records, list):
        # records 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records must be list")
    if not isinstance(field_name, str) or not field_name:
        # field_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("field_name must be non-empty str")
    if not isinstance(require_presence, bool):
        # require_presence 输入不合法，必须 fail-fast。
        raise RecordBundleError("require_presence must be bool")
    if not isinstance(file_key, str) or not file_key:
        # file_key 输入不合法，必须 fail-fast。
        raise RecordBundleError("file_key must be non-empty str")

    first_value: str | None = None
    for record in records:
        if field_name not in record:
            if require_presence:
                return "missing", ""
            return "missing", ""
        value = record[field_name]
        if not isinstance(value, str):
            # 锚点字段类型不正确，必须 fail-fast。
            raise RecordBundleError(
                f"Anchor field must be str: field_name={field_name}, file={file_key}",
                field_name=field_name,
                files=[file_key]
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
    if not isinstance(value_by_file, dict):
        # value_by_file 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("value_by_file must be dict")
    if not isinstance(field_name, str) or not field_name:
        # field_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("field_name must be non-empty str")

    values = list(value_by_file.values())
    if not values:
        # anchor 值为空，必须 fail-fast。
        raise RecordBundleError(f"No anchor values for {field_name}")

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
    if not isinstance(records_dir, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("records_dir must be Path")
    if not isinstance(manifest_name, str) or not manifest_name:
        # manifest_name 输入不合法，必须 fail-fast。
        raise RecordBundleError("manifest_name must be non-empty str")
    if not isinstance(file_entries, list):
        # file_entries 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("file_entries must be list")
    if not isinstance(anchor_values, dict):
        # anchor_values 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("anchor_values must be dict")

    return {
        "schema_version": "v1.0",
        "records_dir": records_dir.as_posix(),
        "manifest_name": manifest_name,
        "file_count": len(file_entries),
        "files": file_entries,
        "anchors": anchor_values
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
    if not isinstance(manifest, dict):
        # manifest 类型不符合预期，必须 fail-fast。
        raise RecordBundleError("manifest must be dict")

    stripped = dict(manifest)
    stripped.pop("bundle_canon_sha256", None)
    return stripped
