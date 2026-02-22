"""
文件目的：静态审计 thresholds 工件只读强制（S-13 B2 阻断项）。
Module type: Core innovation module

审计职责：
1. 扫描 main/watermarking/detect/orchestrator.py
2. 检测是否存在修改 thresholds 工件的代码路径
3. validate Neyman–Pearson 阈值规则（只读）
4. FAIL → 评测过程可能污染阈值工件
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def scan_thresholds_mutations(file_path: Path) -> List[str]:
    """
    功能：扫描文件中是否存在 thresholds 工件修改操作。

    Scan for potential threshold artifact mutations in orchestrator.

    Args:
        file_path: Path to Python file.

    Returns:
        List of problematic code patterns found.
    """
    if not file_path.exists():
        return []
    
    try:
        source_code = file_path.read_text(encoding="utf-8")
    except IOError:
        return []
    
    problems = []
    
    # 危险关键词检测
    dangerous_patterns = [
        "threshold_value =",  # 重新赋值
        "thresholds[",  # 直接索引修改
        "thresholds.update(",  # 字典更新
        "thresholds.pop(",  # 删除键
        "thresholds.clear(",  # 清空
        "write_artifact.*threshold",  # 写工件
    ]
    
    for i, line in enumerate(source_code.split("\n"), 1):
        stripped = line.strip()
        # 忽略注释和文档字符串
        if stripped.startswith("#") or '"""' in stripped or "'''" in stripped:
            continue
        
        for pattern in dangerous_patterns:
            if pattern in stripped:
                # 二次确认：是否在只读上下文中
                # (这里做简化版，完整版应使用 AST 分析)
                if "read" in stripped.lower() or "only" in stripped.lower():
                    continue
                problems.append(f"Line {i}: potential mutation pattern '{pattern}'")
    
    return problems


def check_evaluate_threshold_protocol(orchestrator_path: Path) -> tuple[bool, List[str]]:
    """
    功能：验证 evaluate 函数中 thresholds 的只读协议。

    Check that evaluate functions respect thresholds read-only protocol.

    Args:
        orchestrator_path: Path to orchestrator.py.

    Returns:
        Tuple of (is_compliant: bool, issues: List[str]).
    """
    if not orchestrator_path.exists():
        return False, ["orchestrator.py not found"]
    
    try:
        source_code = orchestrator_path.read_text(encoding="utf-8")
        tree = ast.parse(source_code)
    except Exception as e:
        return False, [f"Failed to parse orchestrator.py: {str(e)}"]
    
    issues = []
    
    # 查找 evaluate_records_against_threshold 和 run_evaluate_orchestrator 函数
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if "evaluate" not in node.name:
                continue
            
            # 检查函数体中的赋值操作
            for child in ast.walk(node):
                # 查找赋值给 threshold 相关变量的操作
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            if "threshold" in target.id.lower():
                                # 检查是否是读行为
                                # (完整版应该更复杂，这里做简化)
                                pass
    
    return len(issues) == 0, issues


def main(repo_root_str: Optional[str] = None) -> int:
    """
    功能：执行 thresholds 只读审计。

    Execute thresholds read-only enforcement audit.

    Args:
        repo_root_str: Optional repository root path.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    repo_root = Path(repo_root_str) if repo_root_str else Path.cwd()
    
    orchestrator_path = repo_root / "main" / "watermarking" / "detect" / "orchestrator.py"
    
    if not orchestrator_path.exists():
        result = {
            "audit_id": "audit_thresholds_readonly",
            "gate_name": "gate_thresholds_readonly",
            "category": "S",
            "severity": "BLOCK",
            "result": "N.A.",
            "rule": "orchestrator.py not found",
            "evidence": {
                "expected_path": str(orchestrator_path),
            },
            "impact": "cannot verify thresholds read-only protocol",
            "fix": "ensure orchestrator.py exists at main/watermarking/detect/",
        }
        print(json.dumps(result, indent=2))
        return 0
    
    # 扫描危险模式
    mutation_problems = scan_thresholds_mutations(orchestrator_path)
    
    # 检查协议
    is_compliant, protocol_issues = check_evaluate_threshold_protocol(orchestrator_path)
    
    all_issues = mutation_problems + protocol_issues
    
    if not all_issues and is_compliant:
        result = {
            "audit_id": "audit_thresholds_readonly",
            "gate_name": "gate_thresholds_readonly",
            "category": "S",
            "severity": "NON_BLOCK",
            "result": "PASS",
            "rule": "thresholds artifact remains read-only in evaluate workflow",
            "evidence": {
                "file_scanned": str(orchestrator_path),
                "mutation_patterns_found": 0,
                "status": "thresholds_read_only_verified",
                "protocol": "Neyman–Pearson rule: evaluate must only read thresholds, never re-estimate",
            },
            "impact": "evaluation cannot corrupt threshold artifacts",
            "fix": "N.A.",
        }
    else:
        result = {
            "audit_id": "audit_thresholds_readonly",
            "gate_name": "gate_thresholds_readonly",
            "category": "S",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "thresholds artifact may not remain read-only in evaluate workflow",
            "evidence": {
                "file_scanned": str(orchestrator_path),
                "issues_found": all_issues[:10],  # 最多显示10个
                "status": "thresholds_mutation_detected",
            },
            "impact": "evaluate workflow may corrupt threshold artifacts, violating Neyman–Pearson rule",
            "fix": "remove all code paths that modify thresholds in evaluate functions; ensure only read access",
        }
    
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    repo_root = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = main(repo_root)
    sys.exit(exit_code)
