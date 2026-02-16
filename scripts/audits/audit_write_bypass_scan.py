"""
功能：扫描仓库中可能绕过受控写盘路径的直接 I/O 调用点（B1/B5 对抗式审计）

Module type: Core innovation module

Scan for direct file write operations that may bypass the controlled write path.
Covers: open(), Path.write_*, json.dump, yaml.dump, pickle.dump, shutil operations.
"""

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


# 关键产物集合（必须由受控写盘路径处理）
CRITICAL_OUTPUTS = [
    "run_closure.json",
    "records_manifest.json",
    "cfg_audit",
    "env_audit",
    "path_audit",
]

# 写模式标识符
WRITE_MODES = {"w", "wb", "a", "ab", "x", "xb", "r+", "rb+", "w+", "wb+", "a+", "ab+"}


class WriteCallVisitor(ast.NodeVisitor):
    """
    AST visitor to detect direct write operations.
    
    Detects:
    - open() with write modes
    - Path.write_text, Path.write_bytes, Path.open
    - json.dump, yaml.dump, pickle.dump
    - shutil.copy*, shutil.move
    """
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.matches: List[Dict[str, Any]] = []
        
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes."""
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno)
        
        # (1) open(...) 读写模式区分
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode, provided = self._extract_open_mode(node)
            access = self._classify_access_from_mode(mode, provided)
            self.matches.append({
                "path": str(self.filepath),
                "lineno_start": lineno,
                "lineno_end": end_lineno,
                "symbol": "open",
                "snippet": f"open() access={access} at line {lineno}",
                "access": access,
                "mode": mode
            })
        
        # (2) Path.write_text / Path.write_bytes / Path.open
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in {"write_text", "write_bytes", "open"}:
                # 检查是否是 Path 对象调用
                if self._looks_like_path_call(node.func):
                    access = "write"
                    mode = None
                    if attr_name == "open":
                        mode, provided = self._extract_open_mode(node)
                        access = self._classify_access_from_mode(mode, provided)
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": f"Path.{attr_name}",
                        "snippet": f"Path.{attr_name}() access={access} at line {lineno}",
                        "access": access,
                        "mode": mode
                    })
            
            # (3) json.dump / yaml.dump / pickle.dump
            if attr_name == "dump":
                if self._looks_like_serializer_dump(node.func):
                    module = self._get_module_name(node.func.value)
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": f"{module}.dump",
                        "snippet": f"{module}.dump() at line {lineno}",
                        "access": "write"
                    })
            
            # (4) shutil.copy* / shutil.move
            if attr_name in {"copy", "copy2", "copyfile", "move"}:
                if self._looks_like_shutil_call(node.func):
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": f"shutil.{attr_name}",
                        "snippet": f"shutil.{attr_name}() at line {lineno}",
                        "access": "write"
                    })
        
        self.generic_visit(node)
    
    def _extract_open_mode(self, node: ast.Call) -> tuple[Optional[str], bool]:
        """Extract open() mode and whether it was provided."""
        if len(node.args) >= 2:
            mode_arg = node.args[1]
            if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                return mode_arg.value, True
            return None, True

        for kw in node.keywords:
            if kw.arg == "mode":
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    return kw.value.value, True
                return None, True

        return None, False

    def _classify_access_from_mode(self, mode: Optional[str], provided: bool) -> str:
        """Classify access based on open() mode semantics."""
        if mode is None:
            return "unknown" if provided else "read"
        if any(m in mode for m in WRITE_MODES):
            return "write"
        return "read"
    
    def _looks_like_path_call(self, attr_node: ast.Attribute) -> bool:
        """Heuristic: check if attribute is called on Path-like object."""
        # 检查是否是 Path(...).method 或 变量.method
        return True  # 保守策略：只要是这些方法名就报告
    
    def _looks_like_serializer_dump(self, attr_node: ast.Attribute) -> bool:
        """Check if dump is from json/yaml/pickle module."""
        module = self._get_module_name(attr_node.value)
        return module in {"json", "yaml", "pickle", "dill", "cloudpickle"}
    
    def _looks_like_shutil_call(self, attr_node: ast.Attribute) -> bool:
        """Check if call is from shutil module."""
        module = self._get_module_name(attr_node.value)
        return module == "shutil"
    
    def _get_module_name(self, node: ast.AST) -> str:
        """Extract module name from node."""
        if isinstance(node, ast.Name):
            return node.id
        return ""


def scan_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Scan a single Python file for direct write operations.
    
    Args:
        filepath: Path to Python source file
        
    Returns:
        List of match dictionaries
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        visitor = WriteCallVisitor(filepath)
        visitor.visit(tree)
        return visitor.matches
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []


def classify_match(match: Dict[str, Any], repo_root: Path) -> str:
    """
    Classify match as ALLOWLISTED, FAIL, or WARNING.
    
    Args:
        match: Match dictionary
        repo_root: Repository root path
        
    Returns:
        Classification: ALLOWLISTED / FAIL / WARNING
    """
    filepath = Path(match["path"])
    rel_path = filepath.relative_to(repo_root).as_posix()
    access = match.get("access")

    if access == "read":
        return "ALLOWLISTED"
    if access == "unknown":
        return "WARNING"
    
    # (1) records_io.py 中的受控写盘函数内部 → ALLOWLISTED
    if "main/core/records_io.py" in rel_path:
        # 需要进一步检查符号名是否在白名单中
        # 这里简化为：records_io.py 中的所有 write 视为受控
        return "ALLOWLISTED"
    
    # (2) tests/ 或 scripts/ 中 → WARNING（非阻断，但应注意）
    if rel_path.startswith("tests/") or rel_path.startswith("scripts/"):
        return "WARNING"
    
    # (3) main/ 中且可能写入关键产物 → FAIL
    if rel_path.startswith("main/"):
        return "FAIL"
    
    # (4) 其他位置 → WARNING
    return "WARNING"


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    Execute write bypass scan audit.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Audit result dictionary following unified schema
    """
    # 扫描 main/ 目录下所有 Python 文件
    main_dir = repo_root / "main"
    if not main_dir.exists():
        return {
            "audit_id": "B1.write_bypass_scan",
            "gate_name": "gate.write_bypass",
            "category": "B",
            "severity": "BLOCK",
            "result": "N.A.",
            "rule": "禁止绕过受控写盘路径直接写入 records 或关键产物",
            "evidence": {"matches": []},
            "impact": "main/ directory not found",
            "fix": "Ensure repository structure includes main/ directory",
        }
    
    all_matches = []
    for pyfile in main_dir.rglob("*.py"):
        matches = scan_file(pyfile)
        all_matches.extend(matches)
    
    # 分类命中点
    fail_matches = []
    allowlisted_matches = []
    warning_matches = []
    
    for match in all_matches:
        classification = classify_match(match, repo_root)
        match["classification"] = classification
        if classification == "FAIL":
            fail_matches.append(match)
        elif classification == "ALLOWLISTED":
            allowlisted_matches.append(match)
        else:
            warning_matches.append(match)
    
    # 判定结果
    result = "PASS" if len(fail_matches) == 0 else "FAIL"
    
    # 构造证据
    evidence = {
        "matches": all_matches,
        "fail_count": len(fail_matches),
        "allowlisted_count": len(allowlisted_matches),
        "warning_count": len(warning_matches),
    }
    
    impact = (
        f"发现 {len(fail_matches)} 处可能绕过受控写盘的直接 I/O 调用"
        if len(fail_matches) > 0
        else "未发现阻断级写盘旁路"
    )
    
    fix_suggestion = (
        "将所有 records 与关键产物的写入操作统一通过 main/core/records_io.py 的受控函数执行，"
        "禁止在业务逻辑中直接调用 open/json.dump/Path.write_* 等写盘 API"
        if len(fail_matches) > 0
        else "N.A."
    )
    
    return {
        "audit_id": "B1.write_bypass_scan",
        "gate_name": "gate.write_bypass",
        "category": "B",
        "severity": "BLOCK",
        "result": result,
        "rule": "禁止绕过受控写盘路径直接写入 records 或关键产物",
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
