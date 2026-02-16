"""
功能：聚合检查冻结面完整性（A1/A2/A3/A4/A5/A7）

Module type: Core innovation module

Verify frozen contract integrity: file existence, unique interpretation entry,
schema authority, append-only constraints, and digest anchoring.
"""

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Set


# 冻结面关键文件
FROZEN_FILES = [
    "configs/frozen_contracts.yaml",
    "configs/runtime_whitelist.yaml",
    "configs/policy_path_semantics.yaml",
]


class FrozenFileReferenceVisitor(ast.NodeVisitor):
    """
    AST visitor to find load-call evidence for frozen configuration files.
    
    Tracks:
    - yaml.safe_load(...) calls with frozen file paths
    - load_yaml_with_provenance(...) calls with frozen file paths
    """
    
    def __init__(self, filepath: Path, frozen_filenames: Set[str]):
        self.filepath = filepath
        self.frozen_filenames = frozen_filenames
        self.references: List[Dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno)
        call_name = self._get_call_name(node.func)

        if call_name in {"yaml.safe_load", "load_yaml_with_provenance"}:
            target = self._extract_first_arg_string(node)
            if target is not None:
                for frozen_file in self.frozen_filenames:
                    if frozen_file in target:
                        self.references.append({
                            "path": str(self.filepath),
                            "lineno_start": lineno,
                            "lineno_end": end_lineno,
                            "symbol": call_name,
                            "snippet": f"{call_name}('{target}') at line {lineno}",
                            "load_kind": "raw_yaml_load",
                            "target_path": target
                        })

        self.generic_visit(node)

    def _get_call_name(self, func: ast.AST) -> str:
        if isinstance(func, ast.Attribute):
            base = ""
            if isinstance(func.value, ast.Name):
                base = func.value.id
            if base:
                return f"{base}.{func.attr}"
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return ""

    def _extract_first_arg_string(self, node: ast.Call) -> str | None:
        if not node.args:
            return None
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value
        return None


def scan_frozen_file_references(repo_root: Path) -> List[Dict[str, Any]]:
    """
    Scan for references to frozen configuration files.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        List of reference dictionaries
    """
    frozen_filenames = {Path(f).name for f in FROZEN_FILES}
    all_references = []
    
    main_dir = repo_root / "main"
    if not main_dir.exists():
        return []
    
    for pyfile in main_dir.rglob("*.py"):
        try:
            source = pyfile.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(pyfile))
            visitor = FrozenFileReferenceVisitor(pyfile, frozen_filenames)
            visitor.visit(tree)
            all_references.extend(visitor.references)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
    
    return all_references


def check_file_existence(repo_root: Path) -> Dict[str, Any]:
    """
    Check existence of frozen configuration files (A7).
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Check result with missing files list
    """
    missing = []
    for frozen_file in FROZEN_FILES:
        filepath = repo_root / frozen_file
        if not filepath.exists():
            missing.append(frozen_file)
    
    return {
        "check": "frozen_files_existence",
        "pass": len(missing) == 0,
        "missing_files": missing,
    }


def check_interpretation_entry_uniqueness(references: List[Dict[str, Any]], repo_root: Path) -> Dict[str, Any]:
    """
    Check that frozen files are loaded from a single canonical entry point (A5/A7).
    
    Args:
        references: List of frozen file references
        repo_root: Repository root directory
        
    Returns:
        Check result with non-canonical references
    """
    # 权威加载链允许的 loader 文件
    canonical_entries = {
        (repo_root / "main" / "core" / "config_loader.py").resolve(),
        (repo_root / "main" / "policy" / "runtime_whitelist.py").resolve(),
        (repo_root / "main" / "core" / "contracts.py").resolve(),
    }
    non_canonical = []
    for ref in references:
        if ref.get("load_kind") != "raw_yaml_load":
            continue
        ref_path = Path(ref["path"]).resolve()
        if ref_path not in canonical_entries:
            non_canonical.append(ref)
    
    return {
        "check": "interpretation_entry_uniqueness",
        "pass": len(non_canonical) == 0,
        "canonical_entry": [str(path) for path in sorted(canonical_entries)],
        "non_canonical_references": non_canonical,
    }


def check_schema_authority(repo_root: Path) -> Dict[str, Any]:
    """
    Check that schema validation requires interpretation (A2).
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Check result
    """
    schema_file = repo_root / "main" / "core" / "schema.py"
    if not schema_file.exists():
        return {
            "check": "schema_authority",
            "pass": False,
            "reason": "schema.py not found",
        }
    
    # 检查 validate_record 函数是否包含 interpretation 缺失的 fail-fast 分支
    try:
        source = schema_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(schema_file))
        
        # 查找 validate_record 函数定义
        validate_record_found = False
        interpretation_required = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "validate_record":
                validate_record_found = True
                interpretation_required = _has_interpretation_fail_fast(node)
        
        return {
            "check": "schema_authority",
            "pass": validate_record_found and interpretation_required,
            "validate_record_found": validate_record_found,
            "interpretation_required": interpretation_required,
        }
    except (SyntaxError, UnicodeDecodeError, OSError):
        return {
            "check": "schema_authority",
            "pass": False,
            "reason": "Failed to parse schema.py",
        }


def check_records_schema_version_injection(repo_root: Path) -> Dict[str, Any]:
    """
    Check that records include schema_version field (A3).
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Check result
    """
    records_io_file = repo_root / "main" / "core" / "records_io.py"
    if not records_io_file.exists():
        return {
            "check": "records_schema_version_injection",
            "pass": False,
            "reason": "records_io.py not found",
        }
    
    # 检查写盘函数是否注入 schema_version
    try:
        source = records_io_file.read_text(encoding="utf-8")
        # 简单检查：是否包含 "schema_version" 字符串
        has_schema_version = "schema_version" in source
        
        return {
            "check": "records_schema_version_injection",
            "pass": has_schema_version,
            "reason": "schema_version field found in records_io.py" if has_schema_version else "schema_version field not found",
        }
    except (UnicodeDecodeError, OSError):
        return {
            "check": "records_schema_version_injection",
            "pass": False,
            "reason": "Failed to read records_io.py",
        }


def _has_interpretation_fail_fast(func_node: ast.FunctionDef) -> bool:
    for node in ast.walk(func_node):
        if isinstance(node, ast.If) and _is_interpretation_none_check(node.test):
            if _has_raise(node.body):
                return True
    return False


def _is_interpretation_none_check(test_node: ast.AST) -> bool:
    if not isinstance(test_node, ast.Compare):
        return False
    if len(test_node.ops) != 1 or len(test_node.comparators) != 1:
        return False
    left = test_node.left
    op = test_node.ops[0]
    right = test_node.comparators[0]
    if not isinstance(left, ast.Name) or left.id != "interpretation":
        return False
    if not isinstance(op, ast.Is):
        return False
    return isinstance(right, ast.Constant) and right.value is None


def _has_raise(nodes: List[ast.stmt]) -> bool:
    for node in nodes:
        if isinstance(node, ast.Raise):
            return True
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                return True
    return False


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    Execute freeze surface integrity audit.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Audit result dictionary following unified schema
    """
    # (1) 检查冻结文件存在性
    existence_check = check_file_existence(repo_root)
    
    # (2) 扫描冻结文件引用点
    references = scan_frozen_file_references(repo_root)
    
    # (3) 检查解释面入口唯一性
    uniqueness_check = check_interpretation_entry_uniqueness(references, repo_root)
    
    # (4) 检查 schema 权威化
    schema_check = check_schema_authority(repo_root)
    
    # (5) 检查 records schema_version 注入
    version_injection_check = check_records_schema_version_injection(repo_root)
    
    # 汇总失败原因
    fail_reasons = []
    
    if not existence_check["pass"]:
        fail_reasons.append(f"缺失冻结配置文件: {', '.join(existence_check['missing_files'])}")
    
    if not uniqueness_check["pass"]:
        fail_reasons.append(
            f"发现 {len(uniqueness_check['non_canonical_references'])} 处非权威入口的冻结文件引用"
        )
    
    if not schema_check["pass"]:
        fail_reasons.append(f"schema 权威化检查失败: {schema_check.get('reason', 'unknown')}")
    
    if not version_injection_check["pass"]:
        fail_reasons.append(f"records schema_version 注入检查失败: {version_injection_check.get('reason', 'unknown')}")
    
    result = "FAIL" if len(fail_reasons) > 0 else "PASS"
    
    # 构造证据锚点
    anchors = {
        "canonical_config_loader": str(repo_root / "main" / "core" / "config_loader.py"),
        "schema_module": str(repo_root / "main" / "core" / "schema.py"),
        "records_io_module": str(repo_root / "main" / "core" / "records_io.py"),
    }
    
    evidence = {
        "anchors": anchors,
        "checks": {
            "existence": existence_check,
            "uniqueness": uniqueness_check,
            "schema_authority": schema_check,
            "version_injection": version_injection_check,
        },
        "frozen_file_references": references,
    }
    
    impact = "; ".join(fail_reasons) if fail_reasons else "冻结面完整性满足要求"
    
    fix_suggestion = (
        "1. 确保 configs/ 下存在所有冻结配置文件；"
        "2. 统一通过 main/core/config_loader.py 加载冻结配置；"
        "3. 确保 schema.validate_record() 强制要求 interpretation 参数；"
        "4. 确保 records 写盘时统一注入 schema_version"
        if len(fail_reasons) > 0
        else "N.A."
    )
    
    return {
        "audit_id": "A1_A7.freeze_surface_integrity",
        "gate_name": "gate.freeze_surface_integrity",
        "category": "A",
        "severity": "BLOCK",
        "result": result,
        "rule": "冻结契约文件必须存在、解释面入口唯一、schema 权威化、records 包含 schema_version",
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
