"""
静态审计 attack protocol
Module type: Core innovation module

审计职责：
1. 扫描 main/evaluation 模块
2. 检测是否存在硬编码的攻击参数定义（scale_min, scale_max, crop_ratio 等）
3. FAIL → 攻击参数脱离 configs 事实源
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# 禁止的硬编码参数关键词（必须来自 configs/attack_protocol.yaml）
FORBIDDEN_PARAMETER_PATTERNS = [
    r"scale_min\s*=\s*[\d\.]",  # 缩放最小值字面量
    r"scale_max\s*=\s*[\d\.]",  # 缩放最大值字面量
    r"crop_ratio\s*=\s*[\d\.]",  # 裁剪比例字面量
    r"rotate_angle\s*=\s*[\d\.]",  # 旋转角度字面量
    r"brightness_delta\s*=\s*[\d\.]",  # 亮度参数字面量
    r"contrast_factor\s*=\s*[\d\.]",  # 对比度参数字面量
    r"noise_std\s*=\s*[\d\.]",  # 噪声参数字面量
    r"attack_families\s*=\s*\[",  # 攻击族定义
    r"attack_params\s*=\s*\{",  # 攻击参数定义
    r"protocol_params\s*=\s*\[",  # 协议参数定义
]

# 允许的上下文（不算违规）
ALLOWED_CONTEXTS = [
    "config_loader",  # 从配置加载
    "protocol_spec",  # 从协议规范中读取
    "protocol_loader",  # 从 protocol_loader 加载
    "test_",  # 测试文件
    "_resolve_float",  # 这是参数提取函数，default 参数不算硬编码
    "# docstring",  # 文档
    '"""',  # 文档字符串
]


def scan_evaluation_module(eval_module_path: Path) -> List[Dict[str, Any]]:
    """
    功能：扫描 evaluation 模块中的硬编码参数。

    Scan evaluation module for hardcoded attack parameters.

    Args:
        eval_module_path: Path to main/evaluation directory.

    Returns:
        List of violation records.
    """
    violations = []
    
    if not eval_module_path.exists():
        return violations
    
    # 扫描所有 Python 文件
    for py_file in eval_module_path.glob("*.py"):
        # 跳过初始化文件
        if py_file.name == "__init__.py":
            continue
        
        try:
            content = py_file.read_text(encoding="utf-8")
        except IOError:
            continue
        
        lines = content.split("\n")
        
        for i, line in enumerate(lines, 1):
            # 跳过注释和字符串
            if line.strip().startswith("#"):
                continue
            
            # 检查每个禁止的模式
            for pattern in FORBIDDEN_PARAMETER_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # 二次确认：检查是否在允许的上下文中
                    in_allowed_context = any(ctx in line for ctx in ALLOWED_CONTEXTS)
                    
                    if not in_allowed_context:
                        violations.append({
                            "file": py_file.name,
                            "line": i,
                            "code": line.strip()[:100],
                            "pattern": pattern,
                        })
    
    return violations


def check_protocol_loader_single_source(repo_root: Path) -> tuple[bool, List[str]]:
    """
    功能：验证是否存在唯一的 protocol loader 事实源。

    Check that attack protocol loading happens through unified protocol_loader.

    Args:
        repo_root: Repository root.

    Returns:
        Tuple of (is_single_source: bool, issues: List[str]).
    """
    issues = []
    
    protocol_loader_path = repo_root / "main" / "evaluation" / "protocol_loader.py"
    
    if not protocol_loader_path.exists():
        issues.append("protocol_loader.py not found")
        return False, issues
    
    # 简单检查：是否导出了必要的函数
    try:
        content = protocol_loader_path.read_text(encoding="utf-8")
        if "load_attack_protocol_spec" not in content:
            issues.append("load_attack_protocol_spec function not found in protocol_loader.py")
        if "compute_attack_protocol_digest" not in content:
            issues.append("compute_attack_protocol_digest function not found")
    except IOError:
        issues.append("Cannot read protocol_loader.py")
    
    return len(issues) == 0, issues


def main(repo_root_str: Optional[str] = None) -> int:
    """
    功能：执行 attack protocol 事实源审计。

    Execute attack protocol fact source audit.

    Args:
        repo_root_str: Optional repository root path.

    Returns:
        Exit code (0 for success, 1 for audit N.A./FAIL).
    """
    repo_root = Path(repo_root_str) if repo_root_str else Path.cwd()
    
    eval_module_path = repo_root / "main" / "evaluation"
    
    if not eval_module_path.exists():
        result = {
            "audit_id": "audit_attack_protocol_hardcoding",
            "gate_name": "gate_attack_protocol_hardcoding",
            "category": "S",
            "severity": "NON_BLOCK",
            "result": "N.A.",
            "rule": "evaluation module not found",
            "evidence": {
                "search_path": str(eval_module_path),
            },
            "impact": "cannot audit attack protocol hardcoding",
            "fix": "ensure main/evaluation module exists",
        }
        print(json.dumps(result, indent=2))
        return 0
    
    # 扫描硬编码参数
    violations = scan_evaluation_module(eval_module_path)
    
    # 检查单一事实源
    is_single_source, source_issues = check_protocol_loader_single_source(repo_root)
    
    if not violations and is_single_source:
        result = {
            "audit_id": "audit_attack_protocol_hardcoding",
            "gate_name": "gate_attack_protocol_hardcoding",
            "category": "S",
            "severity": "NON_BLOCK",
            "result": "PASS",
            "rule": "attack protocol parameters come exclusively from configs fact source",
            "evidence": {
                "module_scanned": str(eval_module_path),
                "hardcoded_patterns_found": 0,
                "protocol_loader_exists": True,
                "status": "attack_protocol_fact_source_enforced",
            },
            "impact": "attack parameters are not hardcoded, ensuring consistency with configs/attack_protocol.yaml",
            "fix": "N.A.",
        }
    else:
        all_issues = [f"{v['file']}:{v['line']}: {v['code']}" for v in violations]
        all_issues.extend(source_issues)
        
        result = {
            "audit_id": "audit_attack_protocol_hardcoding",
            "gate_name": "gate_attack_protocol_hardcoding",
            "category": "S",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "attack protocol parameters must come exclusively from configs fact source",
            "evidence": {
                "module_scanned": str(eval_module_path),
                "violations_found": len(violations),
                "issues": all_issues[:10],  # 最多显示10个
                "status": "hardcoded_parameters_detected",
            },
            "impact": "attack protocol decoupled from configs/attack_protocol.yaml, risk of inconsistency",
            "fix": "remove hardcoded parameter definitions; load all attack parameters via protocol_loader.load_attack_protocol_spec()",
        }
    
    print(json.dumps(result, indent=2))
    return 0 if (not violations and is_single_source) else 1


if __name__ == "__main__":
    repo_root = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = main(repo_root)
    sys.exit(exit_code)
