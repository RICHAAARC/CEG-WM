"""
File purpose: 审计发布面：禁止非初始化空文件存在于主源码树
Module type: Core innovation module

Audit that enforces: no zero-byte .py modules (except __init__.py) 
in main/ directory to prevent capability illusions and future silent failures.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def audit_no_empty_py_modules(repo_root: Path) -> Dict[str, Any]:
    """
    功能：扫描 main/ 目录并阻止任何非 __init__.py 的 0 字节 .py 文件。

    Audit that enforces absence of empty .py modules (non-__init__.py) 
    in main/ directory to prevent capability illusions and audit ambiguity.

    Args:
        repo_root: Repository root directory path.

    Returns:
        Audit result dictionary with PASS/FAIL status, evidence, and impact summary.

    Raises:
        TypeError: If repo_root is not a Path or string.
    """
    audit_id = "empty_py_modules"
    gate_name = "gate.empty_py_modules"
    category = "G"

    if isinstance(repo_root, str):
        repo_root = Path(repo_root)
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path or str")
    if not repo_root.is_dir():
        raise ValueError(f"repo_root {repo_root} is not a directory")

    main_dir = repo_root / "main"
    if not main_dir.is_dir():
        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": category,
            "severity": "NON_BLOCK",
            "result": "N.A.",
            "rule": "main/ directory not found; audit not applicable",
            "impact": "Empty Python modules check not applicable; main/ structure not present",
            "fix": "N.A.",
            "evidence": {
                "checked_path": str(main_dir),
                "directory_exists": False,
            },
        }

    # (1) 扫描所有 .py 文件，找出 0 字节的非 __init__.py 文件
    empty_modules: List[str] = []
    for py_file in main_dir.rglob("*.py"):
        # 相对路径（POSIX 形式）
        relative_path = py_file.relative_to(repo_root).as_posix()
        
        # (2) 检查条件：文件大小为 0 且不是 __init__.py
        if py_file.stat().st_size == 0 and py_file.name != "__init__.py":
            empty_modules.append(relative_path)

    # (3) 生成结果
    if empty_modules:
        # 违规：存在非初始化空文件
        empty_modules_sorted = sorted(empty_modules)
        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": category,
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"found {len(empty_modules_sorted)} zero-byte .py files (non-__init__.py) in main/",
            "impact": "release blocker; empty modules indicate incomplete implementation or capability illusions",
            "fix": "remove all non-__init__.py zero-byte .py files from main/ before signoff",
            "evidence": {
                "empty_modules": empty_modules_sorted,
                "count": len(empty_modules_sorted),
                "scanned_path": str(main_dir),
            },
        }
    else:
        # 通过：无违规文件
        return {
            "audit_id": audit_id,
            "gate_name": gate_name,
            "category": category,
            "severity": "NON_BLOCK",
            "result": "PASS",
            "rule": "no zero-byte .py files (non-__init__.py) found in main/",
            "impact": "release quality assurance: empty modules successfully eliminated",
            "fix": "N.A.",
            "evidence": {
                "scanned_path": str(main_dir),
                "empty_modules": [],
                "count": 0,
            },
        }


def main(argv: Optional[List[str]] = None) -> int:
    """
    功能：审计脚本主入口，命令行执行接口。

    CLI entry point for audit script.

    Args:
        argv: Command-line arguments (repo_root as first arg).

    Returns:
        Exit code: 0 for PASS, 1 for FAIL.
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: audit_no_empty_py_modules.py <repo_root>", file=sys.stderr)
        return 1

    repo_root_str = argv[0]
    try:
        result = audit_no_empty_py_modules(Path(repo_root_str))
    except Exception as exc:
        error_info = {
            "audit_id": "empty_py_modules",
            "gate_name": "gate.empty_py_modules",
            "category": "G",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "audit execution failed",
            "evidence": {
                "error": str(exc),
            },
            "impact": f"unexpected error: {str(exc)}",
            "fix": "fix audit runtime exception",
        }
        print(json.dumps(error_info, indent=2, ensure_ascii=False))
        return 1

    # (4) 输出审计结果（JSON 格式）
    audit_result = result.get("result", "UNKNOWN")
    
    # 统一输出为 dict（signoff 脚本要求 audit 输出 root 必须为 dict）
    output_dict = {}
    
    if audit_result == "FAIL":
        # 违规情况：输出错误信息 dict
        output_dict = {
            "audit_id": result.get("audit_id"),
            "gate_name": result.get("gate_name"),
            "category": result.get("category"),
            "severity": result.get("severity"),
            "result": result.get("result"),
            "rule": result.get("rule"),
            "impact": result.get("impact"),
            "fix": result.get("fix"),
            "evidence": result.get("evidence"),
        }
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))
        return 1
    elif audit_result == "PASS":
        # 通过：输出 PASS dict
        output_dict = {
            "audit_id": result.get("audit_id"),
            "gate_name": result.get("gate_name"),
            "category": result.get("category"),
            "severity": result.get("severity"),
            "result": result.get("result"),
            "rule": result.get("rule"),
            "impact": result.get("impact"),
            "fix": result.get("fix"),
            "evidence": result.get("evidence"),
        }
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))
        return 0
    elif audit_result == "N.A.":
        # 不适用：输出 N.A. dict
        output_dict = result
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))
        return 0
    else:
        # 未知状态：统一降级为 FAIL，避免输出非法枚举。
        output_dict = {
            "audit_id": result.get("audit_id", "empty_py_modules"),
            "gate_name": result.get("gate_name", "gate.empty_py_modules"),
            "category": result.get("category", "G"),
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "unexpected audit_result enum",
            "impact": f"invalid result value: {audit_result}",
            "fix": "ensure result is one of PASS/FAIL/N.A.",
            "evidence": {
                "raw_result": audit_result,
            },
        }
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
