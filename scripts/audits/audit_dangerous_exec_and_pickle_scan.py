"""
功能：扫描危险的动态执行与序列化调用（D9 对抗式审计）

Module type: Core innovation module

Scan for dangerous execution and serialization calls:
eval, exec, compile, pickle.load, dill, cloudpickle, dynamic imports.
"""

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


class DangerousExecVisitor(ast.NodeVisitor):
    """
    AST visitor to detect dangerous execution and serialization.
    
    Detects:
    - eval(), exec(), compile()
    - pickle.load, dill.load, cloudpickle.load
    - importlib.import_module, __import__ (when parameter can be influenced)
    """
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.matches: List[Dict[str, Any]] = []
        
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes."""
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno)
        
        # (1) eval / exec / compile
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in {"eval", "exec", "compile"}:
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": func_name,
                    "snippet": f"{func_name}() at line {lineno}",
                    "risk": "dynamic_execution",
                })
        
        # (2) pickle.load / dill.load / cloudpickle.load
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            module_name = self._get_module_name(node.func.value)
            
            if attr_name == "load" and module_name in {"pickle", "dill", "cloudpickle", "_pickle"}:
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": f"{module_name}.load",
                    "snippet": f"{module_name}.load() at line {lineno}",
                    "risk": "unsafe_deserialization",
                })
            
            # (3) importlib.import_module
            if attr_name == "import_module" and module_name == "importlib":
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": "importlib.import_module",
                    "snippet": f"importlib.import_module() at line {lineno}",
                    "risk": "dynamic_import",
                })
        
        # (4) __import__
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            self.matches.append({
                "path": str(self.filepath),
                "lineno_start": lineno,
                "lineno_end": end_lineno,
                "symbol": "__import__",
                "snippet": f"__import__() at line {lineno}",
                "risk": "dynamic_import",
            })
        
        self.generic_visit(node)
    
    def _get_module_name(self, node: ast.AST) -> str:
        """Extract module name from node."""
        if isinstance(node, ast.Name):
            return node.id
        return ""


def scan_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Scan a single Python file for dangerous execution calls.
    
    Args:
        filepath: Path to Python source file
        
    Returns:
        List of match dictionaries
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        visitor = DangerousExecVisitor(filepath)
        visitor.visit(tree)
        return visitor.matches
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []


def classify_match(match: Dict[str, Any], repo_root: Path) -> str:
    """
    Classify match as FAIL or WARNING.
    
    Args:
        match: Match dictionary
        repo_root: Repository root path
        
    Returns:
        Classification: FAIL / WARNING
    """
    filepath = Path(match["path"])
    rel_path = filepath.relative_to(repo_root).as_posix()
    
    # tests/ 或 scripts/ 中 → WARNING
    if rel_path.startswith("tests/") or rel_path.startswith("scripts/"):
        return "WARNING"
    
    # main/ 中 → FAIL
    if rel_path.startswith("main/"):
        return "FAIL"
    
    return "WARNING"


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    Execute dangerous execution and pickle scan audit.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Audit result dictionary following unified schema
    """
    # 扫描整个仓库
    all_matches = []
    for pyfile in repo_root.rglob("*.py"):
        # 跳过虚拟环境和隐藏目录
        if any(part.startswith(".") or part in {"venv", "env", "__pycache__"} 
               for part in pyfile.parts):
            continue
        matches = scan_file(pyfile)
        all_matches.extend(matches)
    
    # 分类命中点
    fail_matches = []
    warning_matches = []
    
    for match in all_matches:
        classification = classify_match(match, repo_root)
        match["classification"] = classification
        if classification == "FAIL":
            fail_matches.append(match)
        else:
            warning_matches.append(match)
    
    # 按风险类型统计
    dynamic_execution = [m for m in fail_matches if m["risk"] == "dynamic_execution"]
    unsafe_deserialization = [m for m in fail_matches if m["risk"] == "unsafe_deserialization"]
    dynamic_imports = [m for m in fail_matches if m["risk"] == "dynamic_import"]
    
    # 判定结果
    result = "FAIL" if len(fail_matches) > 0 else "PASS"
    
    evidence = {
        "matches": all_matches,
        "fail_count": len(fail_matches),
        "warning_count": len(warning_matches),
        "dynamic_execution_count": len(dynamic_execution),
        "unsafe_deserialization_count": len(unsafe_deserialization),
        "dynamic_import_count": len(dynamic_imports),
    }
    
    fail_reasons = []
    if len(dynamic_execution) > 0:
        fail_reasons.append(f"发现 {len(dynamic_execution)} 处动态执行调用（eval/exec/compile）")
    if len(unsafe_deserialization) > 0:
        fail_reasons.append(f"发现 {len(unsafe_deserialization)} 处不安全的反序列化调用（pickle.load）")
    if len(dynamic_imports) > 0:
        fail_reasons.append(f"发现 {len(dynamic_imports)} 处动态 import 调用")
    
    impact = "; ".join(fail_reasons) if fail_reasons else "未发现危险的动态执行或不安全序列化"
    
    fix_suggestion = (
        "1. 移除或严格隔离 eval/exec/compile 调用，禁止执行外部输入；"
        "2. 禁止使用 pickle.load 加载不可信数据；"
        "3. 若确需动态 import，必须在 whitelist 中显式声明并在审计中列举调用点；"
        "4. 对 tests/scripts 中的危险调用进行隔离，确保不进入发布包"
        if len(fail_matches) > 0
        else "N.A."
    )
    
    return {
        "audit_id": "D9.dangerous_exec_and_pickle_scan",
        "gate_name": "gate.dangerous_execution",
        "category": "D",
        "severity": "BLOCK",
        "result": result,
        "rule": "禁止 eval/exec/compile/pickle.load 等危险执行与不安全序列化",
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
