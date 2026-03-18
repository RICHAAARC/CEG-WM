"""
记录字段追加规则与注册表一致性审计

功能说明：
- 校验 records_schema_extensions.yaml 中的新增字段是否注册到 frozen_contracts.yaml。
- 校验扩展字段条目结构的完整性与唯一性。
- 输出统一审计结果，用于门禁或审计链路。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


_AUDITS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _AUDITS_DIR.parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _candidate_path in (str(_REPO_ROOT), str(_AUDITS_DIR)):
    if _candidate_path not in sys.path:
        sys.path.insert(0, _candidate_path)

try:
    from scripts.audits.gate_label_mapping import resolve_audit_label
except Exception:
    from gate_label_mapping import resolve_audit_label
from main.core import config_loader


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """
    功能：加载 YAML 并要求为 dict。

    Load YAML content and require dict mapping.

    Args:
        path: YAML file path.

    Returns:
        Parsed mapping.

    Raises:
        TypeError: If inputs are invalid or parsed object is not dict.
        ValueError: If path is invalid.
    """
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise TypeError("path must be Path")
    if not path.as_posix():
        # path 值不合法，必须 fail-fast。
        raise ValueError("path must be non-empty")

    obj, _ = config_loader.load_yaml_with_provenance(path)
    if not isinstance(obj, dict):
        # YAML 根类型不符合预期，必须 fail-fast。
        raise TypeError("YAML root must be dict")
    return obj


def _extract_extension_paths(extensions: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    功能：提取扩展字段路径并返回错误列表。

    Extract extension field paths and collect validation errors.

    Args:
        extensions: Extensions mapping from records_schema_extensions.yaml.

    Returns:
        Tuple of (paths, errors).

    Raises:
        TypeError: If extensions is invalid.
    """
    if not isinstance(extensions, dict):
        # extensions 类型不符合预期，必须 fail-fast。
        raise TypeError("extensions must be dict")

    errors: List[str] = []
    paths: List[str] = []

    fields = extensions.get("fields")
    if not isinstance(fields, list):
        errors.append("fields must be list")
        return paths, errors

    for index, entry in enumerate(fields):
        if not isinstance(entry, dict):
            errors.append(f"fields[{index}] must be dict")
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"fields[{index}].path must be non-empty str")
            continue
        for key_name in ["layer", "type", "required", "missing_semantics", "description"]:
            if key_name not in entry:
                errors.append(f"fields[{index}] missing {key_name}")
        paths.append(path_value)

    return paths, errors


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行 records 字段追加规则审计。

    Execute records append-only fields audit.

    Args:
        repo_root: Repository root directory.

    Returns:
        Audit result dictionary following unified schema.

    Raises:
        TypeError: If repo_root is invalid.
    """
    if not isinstance(repo_root, Path):
        # repo_root 类型不符合预期，必须 fail-fast。
        raise TypeError("repo_root must be Path")

    extensions_path = repo_root / "configs" / "records_schema_extensions.yaml"
    contracts_path = repo_root / "configs" / "frozen_contracts.yaml"

    if not extensions_path.exists():
        label = resolve_audit_label(
            "S01.records_schema_extensions_presence",
            "gate.records_schema_extensions.presence"
        )
        return {
            "audit_id": label["audit_id"],
            "gate_name": label["gate_name"],
            "legacy_code": label["legacy_code"],
            "formal_description": label["formal_description"],
            "category": "S",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "records_schema_extensions.yaml 必须存在且可读取",
            "evidence": {"path": str(extensions_path)},
            "impact": "missing records_schema_extensions.yaml",
            "fix": "Create configs/records_schema_extensions.yaml with append-only fields",
        }

    extensions = _load_yaml_mapping(extensions_path)
    contracts = _load_yaml_mapping(contracts_path)

    extension_paths, extension_errors = _extract_extension_paths(extensions)

    records_schema = contracts.get("records_schema")
    registry = []
    if isinstance(records_schema, dict):
        registry = records_schema.get("field_paths_registry", [])

    if not isinstance(registry, list):
        registry = []
        extension_errors.append("records_schema.field_paths_registry must be list")

    registry_set = {p for p in registry if isinstance(p, str)}
    extension_set = {p for p in extension_paths}

    missing_in_registry = sorted(list(extension_set - registry_set))

    duplicate_paths = [p for p in extension_paths if extension_paths.count(p) > 1]
    duplicate_unique = sorted(list(set(duplicate_paths)))

    fail_reasons = []
    if extension_errors:
        fail_reasons.append("extension schema errors")
    if missing_in_registry:
        fail_reasons.append("extension fields not registered")
    if duplicate_unique:
        fail_reasons.append("duplicate extension field paths")

    result = "FAIL" if fail_reasons else "PASS"

    evidence = {
        "extension_errors": extension_errors,
        "extension_count": len(extension_paths),
        "missing_in_registry": missing_in_registry,
        "duplicate_paths": duplicate_unique,
    }

    impact = "; ".join(fail_reasons) if fail_reasons else "extensions are registered and well-formed"
    fix_suggestion = "Add missing fields to frozen_contracts.yaml field_paths_registry" if missing_in_registry else "N.A."

    label = resolve_audit_label(
        "S01.records_schema_extensions_append_only",
        "gate.records_schema_extensions.append_only"
    )
    return {
        "audit_id": label["audit_id"],
        "gate_name": label["gate_name"],
        "legacy_code": label["legacy_code"],
        "formal_description": label["formal_description"],
        "category": "S",
        "severity": "BLOCK",
        "result": result,
        "rule": "扩展字段必须登记在 frozen_contracts.yaml 的 field_paths_registry 中",
        "evidence": evidence,
        "impact": impact,
        "fix": fix_suggestion,
    }


def main() -> None:
    """
    功能：CLI 入口。

    CLI entry point.

    Args:
        None.

    Returns:
        None.
    """
    if len(sys.argv) < 2:
        repo_root = Path.cwd()
    else:
        repo_root = Path(sys.argv[1])

    result = run_audit(repo_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
