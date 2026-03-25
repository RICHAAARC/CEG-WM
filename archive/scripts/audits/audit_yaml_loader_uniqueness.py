"""
功能：扫描仓库中不安全的 YAML 加载调用与非唯一解析入口（A6 对抗式审计）

Module type: Core innovation module

Scan for unsafe YAML loading (yaml.load, FullLoader) and non-unique loader entry points.
"""

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    from scripts.audits.gate_label_mapping import resolve_audit_label
except Exception:
    from gate_label_mapping import resolve_audit_label


class YAMLLoaderVisitor(ast.NodeVisitor):
    """
    AST visitor to detect YAML loading operations.
    
    Detects:
    - yaml.load (unsafe)
    - yaml.FullLoader, yaml.UnsafeLoader
    - yaml.safe_load (multiple entry points)
    """
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.matches: List[Dict[str, Any]] = []
        
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes."""
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno)
        
        # (1) yaml.load / yaml.unsafe_load
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            module_name = self._get_module_name(node.func.value)
            
            if module_name == "yaml":
                # 检测 unsafe load
                if attr_name in {"load", "unsafe_load"}:
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": f"yaml.{attr_name}",
                        "snippet": f"Unsafe yaml.{attr_name}() at line {lineno}",
                        "severity": "UNSAFE",
                    })
                # 检测 safe_load（用于统计入口数量）
                elif attr_name == "safe_load":
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": "yaml.safe_load",
                        "snippet": f"yaml.safe_load() at line {lineno}",
                        "severity": "SAFE_LOAD_ENTRY",
                    })
        
        # (2) 检测 FullLoader / UnsafeLoader 作为参数
        for kw in node.keywords:
            if kw.arg == "Loader":
                if isinstance(kw.value, ast.Attribute):
                    loader_name = kw.value.attr
                    if loader_name in {"FullLoader", "UnsafeLoader", "Loader"}:
                        self.matches.append({
                            "path": str(self.filepath),
                            "lineno_start": lineno,
                            "lineno_end": end_lineno,
                            "symbol": f"yaml.load with {loader_name}",
                            "snippet": f"yaml.load with {loader_name} at line {lineno}",
                            "severity": "UNSAFE",
                        })
        
        self.generic_visit(node)
    
    def _get_module_name(self, node: ast.AST) -> str:
        """Extract module name from node."""
        if isinstance(node, ast.Name):
            return node.id
        return ""


def scan_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Scan a single Python file for YAML loading operations.
    
    Args:
        filepath: Path to Python source file
        
    Returns:
        List of match dictionaries
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        visitor = YAMLLoaderVisitor(filepath)
        visitor.visit(tree)
        return visitor.matches
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    Execute YAML loader uniqueness and safety audit.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Audit result dictionary following unified schema
    """
    # 扫描整个仓库（包括 main、tests、scripts）
    all_matches = []
    for pyfile in repo_root.rglob("*.py"):
        # 跳过虚拟环境和隐藏目录
        if any(part.startswith(".") or part in {"venv", "env", "__pycache__"} 
               for part in pyfile.parts):
            continue
        matches = scan_file(pyfile)
        all_matches.extend(matches)
    
    # 分类命中点
    unsafe_matches = [m for m in all_matches if m["severity"] == "UNSAFE"]
    safe_load_entries = [m for m in all_matches if m["severity"] == "SAFE_LOAD_ENTRY"]
    
    # 统计 main/ 中的 safe_load 入口点
    main_safe_load_entries = [
        m for m in safe_load_entries
        if "main/" in m["path"] and "main/core/config_loader.py" not in m["path"]
    ]
    
    # 判定逻辑
    fail_reasons = []
    
    # (1) 存在不安全加载 → FAIL
    if len(unsafe_matches) > 0:
        fail_reasons.append(f"发现 {len(unsafe_matches)} 处不安全的 YAML 加载调用")
    
    # (2) main/ 中存在多处 safe_load 且不在权威入口 → FAIL
    if len(main_safe_load_entries) > 0:
        fail_reasons.append(
            f"main/ 中存在 {len(main_safe_load_entries)} 处非权威入口的 yaml.safe_load 调用，"
            "可能导致解释面分叉"
        )
    
    result = "FAIL" if len(fail_reasons) > 0 else "PASS"
    
    evidence = {
        "matches": all_matches,
        "unsafe_count": len(unsafe_matches),
        "safe_load_entry_count": len(safe_load_entries),
        "main_non_canonical_safe_load_count": len(main_safe_load_entries),
    }
    
    impact = "; ".join(fail_reasons) if fail_reasons else "所有 YAML 加载均使用安全入口"
    
    fix_suggestion = (
        "1. 将所有 yaml.load 替换为 yaml.safe_load；"
        "2. 统一通过 main/core/config_loader.py 作为唯一 YAML 解析入口；"
        "3. 禁止在业务逻辑中直接调用 yaml.safe_load"
        if len(fail_reasons) > 0
        else "N.A."
    )
    
    label = resolve_audit_label("A6.yaml_loader_uniqueness", "gate.yaml_loader_safety")
    return {
        "audit_id": label["audit_id"],
        "gate_name": label["gate_name"],
        "legacy_code": label["legacy_code"],
        "formal_description": label["formal_description"],
        "category": "A",
        "severity": "BLOCK",
        "result": result,
        "rule": "禁止不安全的 YAML 加载，且 YAML 解析入口必须唯一",
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
