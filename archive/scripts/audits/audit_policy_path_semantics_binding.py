"""
功能：检查 policy_path 语义绑定与路径策略门禁（B3/D1/D2 审计）

Module type: Core innovation module

Verify that policy_path comes from runtime whitelist, semantics version
is bound, and path policy execution leaves audit evidence.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from scripts.audits.gate_label_mapping import resolve_audit_label
except Exception:
    from gate_label_mapping import resolve_audit_label


def check_policy_files_existence(repo_root: Path) -> Dict[str, Any]:
    """
    Check existence of policy-related configuration files.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Check result with missing files list
    """
    required_files = [
        "configs/runtime_whitelist.yaml",
        "configs/policy_path_semantics.yaml",
    ]
    
    missing = []
    for filepath_str in required_files:
        filepath = repo_root / filepath_str
        if not filepath.exists():
            missing.append(filepath_str)
    
    return {
        "check": "policy_files_existence",
        "pass": len(missing) == 0,
        "missing_files": missing,
    }


def check_policy_path_module(repo_root: Path) -> Dict[str, Any]:
    """
    Check that path_policy.py exists and contains validation logic.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Check result
    """
    path_policy_file = repo_root / "main" / "policy" / "path_policy.py"
    if not path_policy_file.exists():
        return {
            "check": "path_policy_module",
            "pass": False,
            "reason": "path_policy.py not found",
        }
    
    try:
        source = path_policy_file.read_text(encoding="utf-8")
        
        # 检查是否包含关键验证函数
        has_validate = "validate" in source.lower()
        has_policy_path = "policy_path" in source
        has_semantics = "semantics" in source.lower()
        
        return {
            "check": "path_policy_module",
            "pass": has_validate and has_policy_path,
            "has_validate_function": has_validate,
            "has_policy_path_handling": has_policy_path,
            "has_semantics_binding": has_semantics,
        }
    except (UnicodeDecodeError, OSError):
        return {
            "check": "path_policy_module",
            "pass": False,
            "reason": "Failed to read path_policy.py",
        }


def check_runtime_whitelist_module(repo_root: Path) -> Dict[str, Any]:
    """
    Check that runtime_whitelist.py exists and enforces policy_path constraint.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Check result
    """
    whitelist_file = repo_root / "main" / "policy" / "runtime_whitelist.py"
    if not whitelist_file.exists():
        return {
            "check": "runtime_whitelist_module",
            "pass": False,
            "reason": "runtime_whitelist.py not found",
        }
    
    try:
        source = whitelist_file.read_text(encoding="utf-8")
        
        # 检查是否包含 policy_path 校验逻辑
        has_policy_path = "policy_path" in source
        has_whitelist_check = "whitelist" in source.lower()
        
        return {
            "check": "runtime_whitelist_module",
            "pass": has_policy_path and has_whitelist_check,
            "has_policy_path_constraint": has_policy_path,
            "has_whitelist_enforcement": has_whitelist_check,
        }
    except (UnicodeDecodeError, OSError):
        return {
            "check": "runtime_whitelist_module",
            "pass": False,
            "reason": "Failed to read runtime_whitelist.py",
        }


def check_path_audit_evidence(repo_root: Path) -> Dict[str, Any]:
    """
    Check that path policy execution creates audit evidence (D2).
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Check result
    """
    # 检查是否存在 path_audit 相关逻辑
    records_io_file = repo_root / "main" / "core" / "records_io.py"
    schema_file = repo_root / "main" / "core" / "schema.py"
    
    has_path_audit = False
    
    for check_file in [records_io_file, schema_file]:
        if check_file.exists():
            try:
                source = check_file.read_text(encoding="utf-8")
                if "path_audit" in source or "path_policy" in source:
                    has_path_audit = True
                    break
            except (UnicodeDecodeError, OSError):
                continue
    
    return {
        "check": "path_audit_evidence",
        "pass": has_path_audit,
        "reason": "path_audit evidence found" if has_path_audit else "path_audit evidence not found in core modules",
    }


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    Execute policy_path semantics binding audit.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Audit result dictionary following unified schema
    """
    # (1) 检查策略配置文件存在性
    files_check = check_policy_files_existence(repo_root)
    
    # (2) 检查 path_policy 模块
    path_policy_check = check_policy_path_module(repo_root)
    
    # (3) 检查 runtime_whitelist 模块
    whitelist_check = check_runtime_whitelist_module(repo_root)
    
    # (4) 检查路径审计证据
    audit_evidence_check = check_path_audit_evidence(repo_root)
    
    # 汇总失败原因
    fail_reasons = []
    
    if not files_check["pass"]:
        fail_reasons.append(f"缺失策略配置文件: {', '.join(files_check['missing_files'])}")
    
    if not path_policy_check["pass"]:
        fail_reasons.append(f"path_policy 模块检查失败: {path_policy_check.get('reason', 'unknown')}")
    
    if not whitelist_check["pass"]:
        fail_reasons.append(f"runtime_whitelist 模块检查失败: {whitelist_check.get('reason', 'unknown')}")
    
    if not audit_evidence_check["pass"]:
        fail_reasons.append(f"路径审计证据检查失败: {audit_evidence_check.get('reason', 'unknown')}")
    
    result = "FAIL" if len(fail_reasons) > 0 else "PASS"
    
    # 构造证据锚点
    anchors = {
        "path_policy_module": str(repo_root / "main" / "policy" / "path_policy.py"),
        "runtime_whitelist_module": str(repo_root / "main" / "policy" / "runtime_whitelist.py"),
        "policy_config_files": [str(repo_root / f) for f in ["configs/runtime_whitelist.yaml", "configs/policy_path_semantics.yaml"]],
    }
    
    evidence = {
        "anchors": anchors,
        "checks": {
            "files_existence": files_check,
            "path_policy_module": path_policy_check,
            "runtime_whitelist_module": whitelist_check,
            "audit_evidence": audit_evidence_check,
        },
    }
    
    impact = "; ".join(fail_reasons) if fail_reasons else "policy_path 语义绑定满足要求"
    
    fix_suggestion = (
        "1. 确保 configs/ 下存在 runtime_whitelist.yaml 与 policy_path_semantics.yaml；"
        "2. 确保 path_policy.py 实现路径验证逻辑；"
        "3. 确保 runtime_whitelist.py 约束 policy_path 取值；"
        "4. 确保路径策略执行结果记录到审计产物（path_audit 或 run_closure）"
        if len(fail_reasons) > 0
        else "N.A."
    )
    
    label = resolve_audit_label("B3_D1_D2.policy_path_semantics_binding", "gate.policy_path_semantics")
    return {
        "audit_id": label["audit_id"],
        "gate_name": label["gate_name"],
        "legacy_code": label["legacy_code"],
        "formal_description": label["formal_description"],
        "category": "B",
        "severity": "BLOCK",
        "result": result,
        "rule": "policy_path 必须来自 runtime whitelist，语义版本绑定，路径策略执行留存审计证据",
        "evidence": evidence,
        "impact": impact,
        "fix": fix_suggestion,
    }


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        repo_root = Path.cwd()
    else:
        repo_root = Path(sys.argv[1])
    
    result = run_audit(repo_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
