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

try:
    from scripts.audits.gate_label_mapping import resolve_audit_label
except Exception:
    from gate_label_mapping import resolve_audit_label


# 关键产物集合（必须由受控写盘路径处理）
CRITICAL_OUTPUTS = [
    "run_closure.json",
    "records_manifest.json",
    "cfg_audit",
    "env_audit",
    "path_audit",
]

# 受控写盘函数白名单（允许在关键脚本中调用）
CONTROLLED_WRITE_FUNCTIONS = {
    "write_artifact_json_unbound",
    "write_artifact_bytes_unbound",
    "write_artifact_json",
    "write_artifact_canon_json_unbound",
    "write_artifact_text_unbound",
}

# 写模式标识符
WRITE_MODES = {"w", "wb", "a", "ab", "x", "xb", "r+", "rb+", "w+", "wb+", "a+", "ab+"}

# 允许的工具类模块（可以直接写盘，无需通过 records_io）
# 这些是非关键产物的导出/工具函数，可在 main/ 中直接调用 Path.write_*
ALLOWLISTED_TOOL_MODULES = {
    "main/evaluation/table_export.py",
}


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
        self.source_lines: List[str] = []  # 用于上下文分析
        
    def set_source_lines(self, source: str) -> None:
        """Set source lines for context analysis."""
        self.source_lines = source.split("\n")
        
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
        
        # (1b) os.open(...) flags 模式推断（新增：去不确定化）
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "open" and self._looks_like_os_module(node.func):
                flags_value, flags_str = self._extract_os_open_flags(node)
                access = self._classify_access_from_os_flags(flags_value, flags_str)
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": "os.open",
                    "snippet": f"os.open() access={access} flags={flags_str} at line {lineno}",
                    "access": access,
                    "mode": None,
                    "os_flags": flags_str
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
    
    def _extract_os_open_flags(self, node: ast.Call) -> tuple[Optional[int], str]:
        """
        提取 os.open() 的 flags 参数并推断访问模式。

        Extract os.open() flags and infer access mode.
        
        Args:
            node: AST Call node.
        
        Returns:
            Tuple of (flags_value_or_none, flags_str_representation).
        """
        if len(node.args) >= 2:
            flags_arg = node.args[1]
            return self._parse_flags_expr(flags_arg)
        
        for kw in node.keywords:
            if kw.arg == "flags":
                return self._parse_flags_expr(kw.value)
        
        return None, "unknown"
    
    def _parse_flags_expr(self, expr: ast.AST) -> tuple[Optional[int], str]:
        """
        解析 flags 表达式（支持位或运算）。

        Parse flags expression (supports bitwise OR).
        
        Args:
            expr: AST expression node.
        
        Returns:
            Tuple of (flags_value_or_none, flags_str_representation).
        """
        # 处理常量
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
            return expr.value, str(expr.value)
        
        # 处理属性访问：os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            left_val, left_str = self._parse_flags_expr(expr.left)
            right_val, right_str = self._parse_flags_expr(expr.right)
            
            if left_val is not None and right_val is not None:
                combined_val = left_val | right_val
            else:
                combined_val = None
            
            combined_str = f"{left_str}|{right_str}"
            return combined_val, combined_str
        
        # 处理 os.O_* 常量
        if isinstance(expr, ast.Attribute):
            attr_name = expr.attr
            if attr_name.startswith("O_"):
                # 尝试从 os 模块获取实际值
                import os as os_module
                if hasattr(os_module, attr_name):
                    flag_value = getattr(os_module, attr_name)
                    return flag_value, f"os.{attr_name}"
            return None, f"os.{attr_name}"
        
        return None, "unknown"
    
    def _classify_access_from_os_flags(self, flags_value: Optional[int], flags_str: str) -> str:
        """
        根据 os.open() flags 推断访问模式。

        Classify access mode based on os.open() flags.
        
        Args:
            flags_value: Parsed flags value (or None if unparsable).
            flags_str: String representation of flags.
        
        Returns:
            Access classification: "read", "write", or "unknown".
        """
        import os as os_module
        
        # 如果能解析出数值，使用精确判断
        if flags_value is not None:
            # 检查是否包含写标志
            write_flags = [
                os_module.O_WRONLY,
                os_module.O_RDWR,
                os_module.O_CREAT,
                os_module.O_APPEND,
                os_module.O_TRUNC
            ]
            if any(flags_value & flag for flag in write_flags):
                return "write"
            # 如果只有 O_RDONLY（值为 0），则为读
            if flags_value == os_module.O_RDONLY:
                return "read"
            return "read"  # 默认为读
        
        # 如果无法解析数值，使用字符串模式匹配
        write_patterns = ["O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC"]
        if any(pattern in flags_str for pattern in write_patterns):
            return "write"
        
        # 如果包含 O_RDONLY，推断为读
        if "O_RDONLY" in flags_str:
            return "read"
        
        # 无法确定
        return "unknown"
    
    def _looks_like_os_module(self, attr_node: ast.Attribute) -> bool:
        """Check if attribute is from os module."""
        if isinstance(attr_node.value, ast.Name):
            return attr_node.value.id == "os"
        return False
    
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


def _is_controlled_write_call(filepath: Path, lineno: int) -> bool:
    """
    检查给定行是否调用了受控写盘函数。
    
    Check whether a line calls a controlled write function.
    
    Args:
        filepath: Source file path.
        lineno: Line number (1-indexed).
    
    Returns:
        True if line calls a controlled write function.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        lines = source.split("\n")
        if lineno <= 0 or lineno > len(lines):
            return False
        line = lines[lineno - 1]
        for func_name in CONTROLLED_WRITE_FUNCTIONS:
            if func_name in line:
                return True
        return False
    except Exception:
        return False


def _is_within_controlled_function(filepath: Path, lineno: int) -> bool:
    """
    检查给定行是否在受控写盘函数内部（用于 records_io.py 精确化）。

    Check whether a line is within a controlled write function.
    This is used to allowlist os.open() calls with access=unknown in records_io.py.
    
    Args:
        filepath: Source file path (should be records_io.py).
        lineno: Line number (1-indexed).
    
    Returns:
        True if line is within append_jsonl, write_json, write_text, etc.
    """
    # 受控写盘函数列表（records_io.py 中的规范入口函数及其核心私有辅助函数）
    controlled_functions = [
        "append_jsonl",
        "write_json",
        "write_text",
        "write_artifact_json",
        "write_artifact_bytes",
        "write_artifact_canon_json_unbound",
        "write_artifact_json_unbound",
        "write_artifact_bytes_unbound",
        "write_artifact_text_unbound",
        "copy_file_controlled",
        "copy_file_controlled_unbound",
        # 核心私有写盘辅助函数（由上层受控函数调用，包含 Windows WinError 5 回退路径）
        "_atomic_replace_write_bytes",
    ]
    
    try:
        source = filepath.read_text(encoding="utf-8")
        lines = source.split("\n")
        if lineno <= 0 or lineno > len(lines):
            return False
        
        # 向上搜索函数定义（简化启发式：找到 "def function_name(" 行）
        for i in range(lineno - 1, max(0, lineno - 100), -1):
            line = lines[i].strip()
            for func_name in controlled_functions:
                if line.startswith(f"def {func_name}("):
                    return True
        
        return False
    except Exception:
        return False


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
    lineno = match.get("lineno_start", -1)
    symbol = match.get("symbol", "")

    if access == "read":
        return "ALLOWLISTED"
    
    # (0) 允许的工具类模块 → ALLOWLISTED（非关键产物）
    for tool_module in ALLOWLISTED_TOOL_MODULES:
        if tool_module in rel_path:
            return "ALLOWLISTED"
    
    # (1) records_io.py 中的受控写盘函数内部 → ALLOWLISTED
    # 特别处理：os.open() access=unknown 时，通过函数上下文判断
    if "main/core/records_io.py" in rel_path:
        # records_io.py 中的所有 write/unknown 都视为受控写盘（仅限受控函数内部）
        # 受控函数包括：append_jsonl, write_json, write_text 等
        if access in ("write", "unknown") and _is_within_controlled_function(filepath, lineno):
            return "ALLOWLISTED"
        # 如果不在受控函数内部，则降级为 WARNING
        if access == "unknown":
            return "WARNING"
    
    if access == "unknown":
        return "WARNING"
    
    # (2) scripts/run_freeze_signoff.py 中调用受控写盘函数 → ALLOWLISTED
    if "scripts/run_freeze_signoff.py" in rel_path:
        if _is_controlled_write_call(filepath, lineno):
            return "ALLOWLISTED"
        # 否则对于 artifacts/ 写入 → FAIL
        if "artifacts" in match.get("snippet", ""):
            return "FAIL"
        # 其他脚本内的写入 → WARNING
        return "WARNING"
    
    # (3) tests/ 中 → WARNING（非阻断，但应注意）
    if rel_path.startswith("tests/"):
        return "WARNING"
    
    # (4) scripts/ 中：区分业务脚本 vs 审计工具脚本
    if rel_path.startswith("scripts/"):
        # 审计脚本本身可以输出临时结果 → WARNING
        if "audits/" in rel_path or rel_path.endswith("run_all_audits.py"):
            return "WARNING"
        # 业务脚本（run_*.py）不应直接调用写 API，必须使用受控路径 → FAIL
        if rel_path.endswith(".py"):
            return "FAIL"
        return "WARNING"
    
    # (5) main/ 中且可能写入关键产物 → FAIL
    if rel_path.startswith("main/"):
        return "FAIL"
    
    # (6) 其他位置 → WARNING
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
    all_matches = []
    
    if main_dir.exists():
        for pyfile in main_dir.rglob("*.py"):
            matches = scan_file(pyfile)
            all_matches.extend(matches)
    
    # 扫描 scripts/ 目录下所有 Python 文件（包括业务脚本和审计脚本）
    scripts_dir = repo_root / "scripts"
    if scripts_dir.exists():
        for pyfile in scripts_dir.rglob("*.py"):
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
        "将所有 records 与关键产物的写入操作统一通过 main/core/records_io.py 的受控函数执行。"
        "scripts/run_freeze_signoff.py 中的 artifacts/ 写入必须使用 write_artifact_json_unbound 或 write_artifact_bytes_unbound。"
        "禁止在业务逻辑中直接调用 open/json.dump/Path.write_* 等写盘 API"
        if len(fail_matches) > 0
        else "N.A."
    )
    
    label = resolve_audit_label("B1.write_bypass_scan", "gate.write_bypass")
    return {
        "audit_id": label["audit_id"],
        "gate_name": label["gate_name"],
        "legacy_code": label["legacy_code"],
        "formal_description": label["formal_description"],
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
