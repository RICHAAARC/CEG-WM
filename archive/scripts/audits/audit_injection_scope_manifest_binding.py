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

try:
    from scripts.audits.gate_label_mapping import resolve_audit_label
except Exception:
    from gate_label_mapping import resolve_audit_label

_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from main.core import config_loader


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


def _check_digest_inputs_non_empty(repo_root: Path) -> Dict[str, Any]:
    """
    功能：校验 injection_scope_manifest.digest_scope 不为空。

    Validate digest_scope.cfg_digest_include_paths and plan_digest_include_paths are non-empty.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result mapping.
    """
    manifest_path = repo_root / "configs" / "injection_scope_manifest.yaml"
    if not manifest_path.exists():
        return {
            "check": "digest_inputs_non_empty",
            "pass": False,
            "missing_fields": [
                "digest_scope.cfg_digest_include_paths",
                "digest_scope.plan_digest_include_paths"
            ],
            "reason": "manifest_missing",
            "path": str(manifest_path)
        }

    try:
        obj, _ = config_loader.load_yaml_with_provenance(manifest_path)
    except Exception as exc:
        return {
            "check": "digest_inputs_non_empty",
            "pass": False,
            "missing_fields": [
                "digest_scope.cfg_digest_include_paths",
                "digest_scope.plan_digest_include_paths"
            ],
            "reason": f"load_failed: {type(exc).__name__}: {exc}",
            "path": str(manifest_path)
        }

    digest_scope = obj.get("digest_scope") if isinstance(obj, dict) else None
    cfg_include = None
    plan_include = None
    if isinstance(digest_scope, dict):
        cfg_include = digest_scope.get("cfg_digest_include_paths")
        plan_include = digest_scope.get("plan_digest_include_paths")

    missing_fields: List[str] = []
    if not isinstance(cfg_include, list) or len(cfg_include) == 0:
        missing_fields.append("digest_scope.cfg_digest_include_paths")
    if not isinstance(plan_include, list) or len(plan_include) == 0:
        missing_fields.append("digest_scope.plan_digest_include_paths")

    return {
        "check": "digest_inputs_non_empty",
        "pass": len(missing_fields) == 0,
        "missing_fields": missing_fields,
        "path": str(manifest_path)
    }


def _check_injection_scope_impl_id_closure(repo_root: Path) -> Dict[str, Any]:
    """
    功能：检查 injection_scope_manifest.allowed_impl_ids 与 runtime_whitelist.impl_id.allowed_flat 的命名闭合性。

    Verify that injection_scope_manifest.allowed_impl_ids is a subset of 
    runtime_whitelist.impl_id.allowed_flat (命名闭合性）.

    Args:
        repo_root: Repository root directory.

    Returns:
        Check result mapping with:
        - "pass": True if allowed_impl_ids ⊆ allowed_flat.
        - "manifest_impl_ids": Set of impl_ids from injection_scope_manifest.
        - "whitelist_impl_ids": Set of impl_ids from runtime_whitelist.
        - "unmapped_impl_ids": Impl_ids in manifest but not in whitelist (BLOCK).
    """
    import yaml
    
    manifest_path = repo_root / "configs" / "injection_scope_manifest.yaml"
    whitelist_path = repo_root / "configs" / "runtime_whitelist.yaml"
    
    manifest_impl_ids = set()
    whitelist_impl_ids = set()
    unmapped = set()
    
    # 加载 injection_scope_manifest
    try:
        if manifest_path.exists():
            with manifest_path.open('r', encoding='utf-8') as f:
                manifest = yaml.safe_load(f) or {}
            manifest_impl_ids = set(manifest.get("allowed_impl_ids", []))
    except Exception as e:
        return {
            "check": "impl_id_closure",
            "pass": False,
            "reason": f"failed_to_load_manifest: {e}",
            "path": str(manifest_path)
        }
    
    # 加载 runtime_whitelist
    try:
        if whitelist_path.exists():
            with whitelist_path.open('r', encoding='utf-8') as f:
                whitelist = yaml.safe_load(f) or {}
            impl_id_cfg = whitelist.get("impl_id", {})
            whitelist_impl_ids = set(impl_id_cfg.get("allowed_flat", []))
    except Exception as e:
        return {
            "check": "impl_id_closure",
            "pass": False,
            "reason": f"failed_to_load_whitelist: {e}",
            "path": str(whitelist_path)
        }
    
    # 检查闭合性
    unmapped = manifest_impl_ids - whitelist_impl_ids
    
    return {
        "check": "impl_id_closure",
        "pass": len(unmapped) == 0,
        "manifest_impl_ids": sorted(manifest_impl_ids),
        "whitelist_impl_ids": sorted(whitelist_impl_ids),
        "unmapped_impl_ids": sorted(unmapped) if unmapped else None,
        "rule": "allowed_impl_ids ⊆ allowed_flat (命名闭合性)",
        "severity": "BLOCK_FREEZE" if unmapped else "PASS"
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
        _check_cli_entries(repo_root),
        _check_digest_inputs_non_empty(repo_root),
        _check_injection_scope_impl_id_closure(repo_root)  # 新增：impl_id 命名闭合性检查
    ]

    fail_reasons = [c["check"] for c in checks if not c.get("pass")]
    
    # 检查是否存在 BLOCK 级别的违规（impl_id 闭合性失败）
    has_block_failure = any(
        c.get("severity") == "BLOCK_FREEZE" for c in checks if not c.get("pass")
    )
    
    result = "FAIL" if fail_reasons else "PASS"
    severity = "BLOCK" if has_block_failure else "BLOCK"  # 本审计全部为 BLOCK 级别

    impact = "missing binding checks: " + ", ".join(fail_reasons) if fail_reasons else "injection_scope_manifest binding is enforced"
    fix_suggestion = (
        "1. Add configs/injection_scope_manifest.yaml; "
        "2. Bind injection_scope_manifest in records_io and schema; "
        "3. Ensure CLI entry points load and bind injection_scope_manifest; "
        "4. Fix impl_id naming: allowed_impl_ids must be subset of runtime_whitelist.allowed_flat"
        if fail_reasons else "N.A."
    )

    label = resolve_audit_label("S00.injection_scope_manifest_binding", "gate.injection_scope_manifest_binding")
    return {
        "audit_id": label["audit_id"],
        "gate_name": label["gate_name"],
        "legacy_code": label["legacy_code"],
        "formal_description": label["formal_description"],
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
    output = json.dumps(result, indent=2, ensure_ascii=False)
    # 处理 Windows GBK 编码限制：使用 ensure_ascii=True 或直接写为 UTF-8
    sys.stdout.buffer.write((output + '\n').encode('utf-8'))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
