"""
File purpose: Audit injection scope manifest binding chain.
Module type: Core innovation module

功能说明：
- 检查 injection_scope_manifest.yaml 的存在性与加载入口。
- 检查 bound_fact_sources 与 schema.required_fields 是否覆盖 injection_scope_manifest 字段。
- 检查四个 CLI 入口是否加载并绑定 injection_scope_manifest。
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


_INJECTION_FIELDS = [
    "injection_scope_manifest_version",
    "injection_scope_manifest_digest",
    "injection_scope_manifest_file_sha256",
    "injection_scope_manifest_canon_sha256",
    "injection_scope_manifest_bound_digest"
]


def _read_text(path: Path) -> str:
    """
    功能：读取文本文件内容。

    Read text content from a file.

    Args:
        path: File path.

    Returns:
        File content as string.

    Raises:
        TypeError: If path is invalid.
    """
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise TypeError("path must be Path")
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _check_file_exists(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 injection_scope_manifest.yaml 是否存在。

    Check existence of configs/injection_scope_manifest.yaml.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result mapping.
    """
    manifest_path = repo_root / "configs" / "injection_scope_manifest.yaml"
    exists = manifest_path.exists() and manifest_path.is_file()
    return {
        "check": "manifest_exists",
        "pass": bool(exists),
        "path": str(manifest_path)
    }


def _check_loader_entry(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 config_loader 中的权威加载入口。

    Check config_loader for injection scope manifest loader entry.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result mapping.
    """
    loader_path = repo_root / "main" / "core" / "config_loader.py"
    source = _read_text(loader_path)
    has_const = "INJECTION_SCOPE_MANIFEST_PATH" in source
    has_loader = "load_injection_scope_manifest" in source
    return {
        "check": "config_loader_entry",
        "pass": bool(has_const and has_loader),
        "path": str(loader_path),
        "has_const": has_const,
        "has_loader": has_loader
    }


def _check_injection_scope_module(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 injection_scope 模块存在性。

    Check that main/core/injection_scope.py exists.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result mapping.
    """
    module_path = repo_root / "main" / "core" / "injection_scope.py"
    exists = module_path.exists() and module_path.is_file()
    return {
        "check": "injection_scope_module",
        "pass": bool(exists),
        "path": str(module_path)
    }


def _check_bound_fact_sources(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 bound_fact_sources 输出是否包含注入范围字段。

    Check records_io for bound_fact_sources injection fields.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result mapping.
    """
    records_io_path = repo_root / "main" / "core" / "records_io.py"
    source = _read_text(records_io_path)
    missing = [field for field in _INJECTION_FIELDS if field not in source]
    return {
        "check": "bound_fact_sources_fields",
        "pass": len(missing) == 0,
        "path": str(records_io_path),
        "missing_fields": missing
    }


def _check_schema_required_fields(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 schema.required_fields 是否覆盖注入范围字段。

    Check schema bound_fact_sources required fields.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result mapping.
    """
    schema_path = repo_root / "main" / "core" / "schema.py"
    source = _read_text(schema_path)
    missing = [field for field in _INJECTION_FIELDS if field not in source]
    return {
        "check": "schema_required_fields",
        "pass": len(missing) == 0,
        "path": str(schema_path),
        "missing_fields": missing
    }


def _check_cli_entries(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 CLI 入口是否加载并绑定 injection_scope_manifest。

    Check CLI entry points for injection scope loading and binding.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result mapping.
    """
    cli_files = [
        repo_root / "main" / "cli" / "run_embed.py",
        repo_root / "main" / "cli" / "run_detect.py",
        repo_root / "main" / "cli" / "run_calibrate.py",
        repo_root / "main" / "cli" / "run_evaluate.py"
    ]
    missing: List[str] = []
    details: Dict[str, Any] = {}
    for path in cli_files:
        source = _read_text(path)
        has_loader = "load_injection_scope_manifest" in source
        has_binding = "injection_scope_manifest" in source
        if not (has_loader and has_binding):
            missing.append(str(path))
        details[str(path)] = {
            "has_loader": has_loader,
            "has_binding": has_binding
        }
    return {
        "check": "cli_entries",
        "pass": len(missing) == 0,
        "missing_entries": missing,
        "details": details
    }


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行注入范围事实源绑定审计。

    Execute injection scope manifest binding audit.

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

    checks = [
        _check_file_exists(repo_root),
        _check_loader_entry(repo_root),
        _check_injection_scope_module(repo_root),
        _check_bound_fact_sources(repo_root),
        _check_schema_required_fields(repo_root),
        _check_cli_entries(repo_root)
    ]

    fail_reasons = [c["check"] for c in checks if not c.get("pass")]
    result = "FAIL" if fail_reasons else "PASS"

    impact = "missing binding checks: " + ", ".join(fail_reasons) if fail_reasons else "injection_scope_manifest binding is enforced"
    fix_suggestion = (
        "1. Add configs/injection_scope_manifest.yaml; "
        "2. Bind injection_scope_manifest in records_io and schema; "
        "3. Ensure CLI entry points load and bind injection_scope_manifest"
        if fail_reasons else "N.A."
    )

    return {
        "audit_id": "S00.injection_scope_manifest_binding",
        "gate_name": "gate.injection_scope_manifest_binding",
        "category": "S",
        "severity": "BLOCK",
        "result": result,
        "rule": "injection_scope_manifest 必须进入 bound_fact_sources 并受 schema 约束",
        "evidence": {
            "checks": checks
        },
        "impact": impact,
        "fix": fix_suggestion
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
