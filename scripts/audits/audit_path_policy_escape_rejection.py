"""
功能：路径策略逃逸拒绝审计（D1）

File purpose: D1 path policy escape rejection audit.
Module type: Core innovation module

Enumerate validate_output_target call sites in path_policy.py and records_io.py.
Scan records_io write call sites and verify validation happens before writes.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WRITE_FUNCTIONS = {
    "write_path_validation_audit",
    "write_json",
    "append_jsonl",
    "write_artifact_json",
    "write_artifact_json_unbound",
    "write_artifact_canon_json_unbound",
    "write_artifact_text_unbound",
    "write_artifact_bytes_unbound",
    "write_text",
}


def _read_source(file_path: Path) -> str:
    """
    功能：读取源码文本。

    Read source file content as UTF-8 text.

    Args:
        file_path: Source file path.

    Returns:
        Source text content.

    Raises:
        TypeError: If file_path is invalid.
        ValueError: If file does not exist.
    """
    if not isinstance(file_path, Path):
        # file_path 类型不符合预期，必须 fail-fast。
        raise TypeError("file_path must be Path")
    if not file_path.exists() or not file_path.is_file():
        # 文件不存在，必须 fail-fast。
        raise ValueError(f"file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def _parse_ast(file_path: Path, source: str) -> ast.AST:
    """
    功能：解析 AST。

    Parse Python source into AST.

    Args:
        file_path: Source file path.
        source: Source text content.

    Returns:
        Parsed AST tree.

    Raises:
        TypeError: If inputs are invalid.
        SyntaxError: If parsing fails.
    """
    if not isinstance(file_path, Path):
        # file_path 类型不符合预期，必须 fail-fast。
        raise TypeError("file_path must be Path")
    if not isinstance(source, str):
        # source 类型不符合预期，必须 fail-fast。
        raise TypeError("source must be str")
    return ast.parse(source, filename=str(file_path))


def _get_attr_root_name(node: ast.AST) -> Optional[str]:
    """
    功能：提取属性链根节点名称。

    Extract root name from an attribute chain.

    Args:
        node: AST node to inspect.

    Returns:
        Root name if available, otherwise None.
    """
    if not isinstance(node, ast.AST):
        # node 类型不符合预期，必须 fail-fast。
        raise TypeError("node must be AST")
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _get_attr_root_name(node.value)
    return None


def _collect_records_io_aliases(tree: ast.AST) -> Tuple[List[str], Dict[str, str]]:
    """
    功能：收集 records_io 导入别名。

    Collect records_io module aliases and imported function aliases.

    Args:
        tree: AST tree.

    Returns:
        Tuple of (module_aliases, function_aliases).
    """
    if not isinstance(tree, ast.AST):
        # tree 类型不符合预期，必须 fail-fast。
        raise TypeError("tree must be AST")

    module_aliases: List[str] = []
    function_aliases: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("records_io"):
                    module_aliases.append(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.endswith("records_io"):
                for alias in node.names:
                    function_aliases[alias.asname or alias.name] = alias.name

    return module_aliases, function_aliases


class _CallCollector(ast.NodeVisitor):
    """
    功能：收集 validate_output_target 与写盘调用点。

    Collect validate_output_target call sites and records_io write call sites.
    """

    def __init__(
        self,
        file_path: Path,
        module_aliases: List[str],
        function_aliases: Dict[str, str]
    ) -> None:
        if not isinstance(file_path, Path):
            # file_path 类型不符合预期，必须 fail-fast。
            raise TypeError("file_path must be Path")
        if not isinstance(module_aliases, list):
            # module_aliases 类型不符合预期，必须 fail-fast。
            raise TypeError("module_aliases must be list")
        if not isinstance(function_aliases, dict):
            # function_aliases 类型不符合预期，必须 fail-fast。
            raise TypeError("function_aliases must be dict")

        self.file_path = file_path
        self.module_aliases = set(module_aliases)
        self.function_aliases = dict(function_aliases)
        self.scope_stack: List[str] = []
        self.validate_calls: List[Dict[str, Any]] = []
        self.write_calls: List[Dict[str, Any]] = []

    def _current_scope(self) -> str:
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not isinstance(node, ast.ClassDef):
            # node 类型不符合预期，必须 fail-fast。
            raise TypeError("node must be ClassDef")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not isinstance(node, ast.FunctionDef):
            # node 类型不符合预期，必须 fail-fast。
            raise TypeError("node must be FunctionDef")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if not isinstance(node, ast.AsyncFunctionDef):
            # node 类型不符合预期，必须 fail-fast。
            raise TypeError("node must be AsyncFunctionDef")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node, ast.Call):
            # node 类型不符合预期，必须 fail-fast。
            raise TypeError("node must be Call")
        scope = self._current_scope()
        lineno = getattr(node, "lineno", 0)

        if self._is_validate_output_target_call(node):
            self.validate_calls.append({
                "path": str(self.file_path),
                "scope": scope,
                "lineno": lineno,
                "symbol": "validate_output_target",
            })

        write_info = self._resolve_write_call(node)
        if write_info is not None:
            write_info.update({
                "path": str(self.file_path),
                "scope": scope,
                "lineno": lineno,
            })
            self.write_calls.append(write_info)

        self.generic_visit(node)

    def _is_validate_output_target_call(self, node: ast.Call) -> bool:
        if not isinstance(node, ast.Call):
            # node 类型不符合预期，必须 fail-fast。
            raise TypeError("node must be Call")
        if isinstance(node.func, ast.Name):
            return node.func.id == "validate_output_target"
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "validate_output_target"
        return False

    def _resolve_write_call(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        if not isinstance(node, ast.Call):
            # node 类型不符合预期，必须 fail-fast。
            raise TypeError("node must be Call")
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if func_name in WRITE_FUNCTIONS:
                root = _get_attr_root_name(node.func.value)
                if root in self.module_aliases:
                    return {
                        "function": func_name,
                        "call_type": "records_io_module",
                        "root": root,
                    }
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in WRITE_FUNCTIONS:
                if func_name in self.function_aliases:
                    return {
                        "function": self.function_aliases[func_name],
                        "call_type": "records_io_import",
                        "root": func_name,
                    }
                if self.file_path.name == "records_io.py":
                    return {
                        "function": func_name,
                        "call_type": "records_io_local",
                        "root": func_name,
                    }
        return None


def _collect_validate_sites(file_path: Path) -> List[Dict[str, Any]]:
    """
    功能：收集 validate_output_target 调用点。

    Collect validate_output_target call sites in a file.

    Args:
        file_path: Target Python file path.

    Returns:
        List of call site dictionaries.
    """
    if not isinstance(file_path, Path):
        # file_path 类型不符合预期，必须 fail-fast。
        raise TypeError("file_path must be Path")

    source = _read_source(file_path)
    tree = _parse_ast(file_path, source)
    module_aliases, function_aliases = _collect_records_io_aliases(tree)
    collector = _CallCollector(file_path, module_aliases, function_aliases)
    collector.visit(tree)
    return collector.validate_calls


def _collect_write_sites(file_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    功能：收集写盘调用点与校验调用点。

    Collect records_io write call sites and validate_output_target call sites.

    Args:
        file_path: Target Python file path.

    Returns:
        Tuple of (write_calls, validate_calls).
    """
    if not isinstance(file_path, Path):
        # file_path 类型不符合预期，必须 fail-fast。
        raise TypeError("file_path must be Path")

    source = _read_source(file_path)
    tree = _parse_ast(file_path, source)
    module_aliases, function_aliases = _collect_records_io_aliases(tree)
    collector = _CallCollector(file_path, module_aliases, function_aliases)
    collector.visit(tree)
    return collector.write_calls, collector.validate_calls


def _collect_internal_guarded_functions(records_io_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    功能：识别 records_io 内部被 validate_output_target 保护的写盘函数。

    Identify records_io write functions guarded by validate_output_target.

    Args:
        records_io_path: Path to records_io.py.

    Returns:
        Tuple of (guarded_function_names, unguarded_definitions).
    """
    if not isinstance(records_io_path, Path):
        # records_io_path 类型不符合预期，必须 fail-fast。
        raise TypeError("records_io_path must be Path")

    source = _read_source(records_io_path)
    tree = _parse_ast(records_io_path, source)

    guarded: List[str] = []
    unguarded_defs: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in WRITE_FUNCTIONS:
            has_validate = False
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    if isinstance(inner.func, ast.Name) and inner.func.id == "validate_output_target":
                        has_validate = True
                        break
                    if isinstance(inner.func, ast.Attribute) and inner.func.attr == "validate_output_target":
                        has_validate = True
                        break
            if has_validate:
                guarded.append(node.name)
            else:
                unguarded_defs.append({
                    "path": str(records_io_path),
                    "function": node.name,
                    "lineno": getattr(node, "lineno", 0),
                })

    return guarded, unguarded_defs


def _scan_repo(repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行 D1 路径策略逃逸拒绝审计。

    Run the D1 path policy escape rejection audit.

    Args:
        repo_root: Repository root directory.

    Returns:
        Audit result dictionary.
    """
    if not isinstance(repo_root, Path):
        # repo_root 类型不符合预期，必须 fail-fast。
        raise TypeError("repo_root must be Path")

    path_policy_path = repo_root / "main" / "policy" / "path_policy.py"
    records_io_path = repo_root / "main" / "core" / "records_io.py"

    if not path_policy_path.exists() or not records_io_path.exists():
        return {
            "audit_id": "D1.path_policy_escape_rejection",
            "gate_name": "gate.path_policy_escape_rejection",
            "category": "D",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": "写盘必须通过路径策略门禁并拒绝路径逃逸",
            "evidence": {
                "missing_files": {
                    "path_policy": str(path_policy_path),
                    "records_io": str(records_io_path),
                }
            },
            "impact": "required modules missing",
            "fix": "ensure path_policy.py and records_io.py exist",
        }

    validate_sites = []
    validate_sites.extend(_collect_validate_sites(path_policy_path))
    validate_sites.extend(_collect_validate_sites(records_io_path))

    guarded_functions, unguarded_defs = _collect_internal_guarded_functions(records_io_path)

    write_sites: List[Dict[str, Any]] = []
    validate_by_scope: Dict[Tuple[str, str], List[int]] = {}

    scan_dirs = [repo_root / "main", repo_root / "scripts"]
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for pyfile in scan_dir.rglob("*.py"):
            try:
                write_calls, validate_calls = _collect_write_sites(pyfile)
            except (SyntaxError, UnicodeDecodeError, ValueError, TypeError, OSError):
                continue

            for call in validate_calls:
                key = (call["path"], call["scope"])
                validate_by_scope.setdefault(key, []).append(call["lineno"])

            for call in write_calls:
                key = (call["path"], call["scope"])
                local_validations = validate_by_scope.get(key, [])
                guarded_by_local = any(lineno < call["lineno"] for lineno in local_validations)
                guarded_by_internal = call["function"] in guarded_functions
                call["guarded_by_local_validate"] = guarded_by_local
                call["guarded_by_internal_validate"] = guarded_by_internal
                call["guarded"] = guarded_by_local or guarded_by_internal
                write_sites.append(call)

    missing_guards = [entry for entry in write_sites if not entry.get("guarded")]

    result = "PASS" if len(missing_guards) == 0 and len(unguarded_defs) == 0 else "FAIL"

    evidence = {
        "validate_call_sites": validate_sites,
        "write_call_sites": write_sites,
        "missing_guard_sites": missing_guards,
        "unguarded_records_io_functions": unguarded_defs,
        "write_call_count": len(write_sites),
        "missing_guard_count": len(missing_guards),
        "unguarded_function_count": len(unguarded_defs),
    }

    impact = (
        f"发现 {len(missing_guards)} 处写盘调用缺失路径策略门禁"
        if len(missing_guards) > 0 or len(unguarded_defs) > 0
        else "未发现路径策略门禁缺失"
    )

    fix = (
        "在写盘调用前增加 path_policy.validate_output_target 或确保 records_io 内部门禁完整"
        if result == "FAIL"
        else "N.A."
    )

    return {
        "audit_id": "D1.path_policy_escape_rejection",
        "gate_name": "gate.path_policy_escape_rejection",
        "category": "D",
        "severity": "BLOCK",
        "result": result,
        "rule": "写盘必须通过路径策略门禁并拒绝路径逃逸",
        "evidence": evidence,
        "impact": impact,
        "fix": fix,
    }


def _main() -> None:
    """
    功能：CLI 入口。

    CLI entry point.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    result = _scan_repo(repo_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
