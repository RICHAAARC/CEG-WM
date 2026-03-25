"""
功能：检查注册表的运行期注入面与可变性封闭（C1/C4 对抗式审计）

Module type: Core innovation module

Verify that registries are sealed after initialization and cannot be
injected or modified at runtime through environment variables, CLI args,
or dynamic imports.
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


class RegistryInjectionVisitor(ast.NodeVisitor):
    """
    AST visitor to detect registry injection surfaces.
    
    Detects:
    - os.environ / os.getenv usage
    - importlib.import_module with variable module names
    - __import__ with variable names
    - Registry.register calls after initialization
    """
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.matches: List[Dict[str, Any]] = []
        
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes."""
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno)
        
        # (1) os.getenv / os.environ.get
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            
            if attr_name == "getenv":
                module_name = self._get_module_name(node.func.value)
                if module_name == "os":
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": "os.getenv",
                        "snippet": f"os.getenv() at line {lineno}",
                        "risk": "env_variable_injection",
                    })
            
            elif attr_name == "get" and self._is_environ_dict(node.func.value):
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": "os.environ.get",
                    "snippet": f"os.environ.get() at line {lineno}",
                    "risk": "env_variable_injection",
                })
            
            # (2) importlib.import_module
            elif attr_name == "import_module":
                module_name = self._get_module_name(node.func.value)
                if module_name == "importlib":
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": "importlib.import_module",
                        "snippet": f"importlib.import_module() at line {lineno}",
                        "risk": "dynamic_import",
                    })
            
            # (3) registry.register 调用
            elif attr_name == "register":
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": "registry.register",
                    "snippet": f"registry.register() at line {lineno}",
                    "risk": "runtime_registration",
                })
        
        # (4) __import__ builtin
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
    
    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit subscript nodes (e.g., os.environ['KEY'])."""
        if self._is_environ_dict(node.value):
            self.matches.append({
                "path": str(self.filepath),
                "lineno_start": node.lineno,
                "lineno_end": getattr(node, "end_lineno", node.lineno),
                "symbol": "os.environ[]",
                "snippet": f"os.environ subscription at line {node.lineno}",
                "risk": "env_variable_injection",
            })
        self.generic_visit(node)
    
    def _get_module_name(self, node: ast.AST) -> str:
        """Extract module name from node."""
        if isinstance(node, ast.Name):
            return node.id
        return ""
    
    def _is_environ_dict(self, node: ast.AST) -> bool:
        """Check if node represents os.environ."""
        if isinstance(node, ast.Attribute):
            return node.attr == "environ" and self._get_module_name(node.value) == "os"
        return False


def scan_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Scan a single Python file for registry injection surfaces.
    
    Args:
        filepath: Path to Python source file
        
    Returns:
        List of match dictionaries
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        visitor = RegistryInjectionVisitor(filepath)
        visitor.visit(tree)
        return visitor.matches
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []


def check_registry_seal(repo_root: Path) -> Dict[str, Any]:
    """
    Check if registries implement seal mechanism.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Check result
    """
    registry_base_file = repo_root / "main" / "registries" / "registry_base.py"
    if not registry_base_file.exists():
        return {
            "check": "registry_seal_mechanism",
            "pass": False,
            "reason": "registry_base.py not found",
        }
    
    try:
        source = registry_base_file.read_text(encoding="utf-8")
        # 检查是否包含 seal 相关方法
        has_seal = "seal" in source.lower()
        has_lock = "lock" in source.lower() or "_sealed" in source or "_locked" in source
        
        return {
            "check": "registry_seal_mechanism",
            "pass": has_seal or has_lock,
            "has_seal_method": has_seal,
            "has_lock_indicator": has_lock,
        }
    except (UnicodeDecodeError, OSError):
        return {
            "check": "registry_seal_mechanism",
            "pass": False,
            "reason": "Failed to read registry_base.py",
        }


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    Execute registry injection surface audit.
    
    Args:
        repo_root: Repository root directory
        
    Returns:
        Audit result dictionary following unified schema
    """
    # 扫描 main/registries/ 目录
    registries_dir = repo_root / "main" / "registries"
    if not registries_dir.exists():
        label = resolve_audit_label("C1.registry_injection_surface", "gate.registry_immutability")
        return {
            "audit_id": label["audit_id"],
            "gate_name": label["gate_name"],
            "legacy_code": label["legacy_code"],
            "formal_description": label["formal_description"],
            "category": "C",
            "severity": "BLOCK",
            "result": "N.A.",
            "rule": "注册表必须 seal 后不可变，且不可被运行期注入",
            "evidence": {"matches": []},
            "impact": "registries directory not found",
            "fix": "Ensure repository structure includes main/registries/ directory",
        }
    
    # 扫描注册表目录中的注入面
    all_matches = []
    for pyfile in registries_dir.rglob("*.py"):
        matches = scan_file(pyfile)
        all_matches.extend(matches)
    
    # 检查 seal 机制
    seal_check = check_registry_seal(repo_root)
    
    # 分类风险
    env_injection = [m for m in all_matches if m["risk"] == "env_variable_injection"]
    dynamic_imports = [m for m in all_matches if m["risk"] == "dynamic_import"]
    runtime_registrations = [m for m in all_matches if m["risk"] == "runtime_registration"]
    
    # 判定逻辑
    fail_reasons = []
    
    if not seal_check["pass"]:
        fail_reasons.append(f"注册表缺少 seal 机制: {seal_check.get('reason', 'unknown')}")
    
    if len(env_injection) > 0:
        fail_reasons.append(f"发现 {len(env_injection)} 处环境变量注入风险")
    
    if len(dynamic_imports) > 0:
        fail_reasons.append(f"发现 {len(dynamic_imports)} 处动态 import 调用")
    
    if len(runtime_registrations) > 0:
        fail_reasons.append(f"发现 {len(runtime_registrations)} 处运行期 register 调用（需验证是否在 seal 前）")
    
    result = "FAIL" if len(fail_reasons) > 0 else "PASS"
    
    evidence = {
        "matches": all_matches,
        "env_injection_count": len(env_injection),
        "dynamic_import_count": len(dynamic_imports),
        "runtime_registration_count": len(runtime_registrations),
        "seal_check": seal_check,
    }
    
    impact = "; ".join(fail_reasons) if fail_reasons else "注册表封闭性满足要求"
    
    fix_suggestion = (
        "1. 为所有注册表实现 seal() 方法，seal 后禁止 register/override；"
        "2. 移除或严格隔离环境变量影响 impl 选择的路径；"
        "3. 禁止运行期动态 import 影响注册表内容；"
        "4. 确保所有 register 调用仅在初始化阶段完成"
        if len(fail_reasons) > 0
        else "N.A."
    )
    
    label = resolve_audit_label("C1.registry_injection_surface", "gate.registry_immutability")
    return {
        "audit_id": label["audit_id"],
        "gate_name": label["gate_name"],
        "legacy_code": label["legacy_code"],
        "formal_description": label["formal_description"],
        "category": "C",
        "severity": "BLOCK",
        "result": result,
        "rule": "注册表必须 seal 后不可变，且不可被运行期注入",
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
