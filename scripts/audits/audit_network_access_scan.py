"""
功能：扫描网络访问与隐式下载调用（D10 对抗式审计）

Module type: Core innovation module

Scan for network access and implicit download operations:
requests, urllib, httpx, hf_hub_download, subprocess with curl/wget/git.
"""

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


class NetworkAccessVisitor(ast.NodeVisitor):
    """
    AST visitor to detect network access and download operations.
    
    Detects:
    - requests.get/post/request
    - urllib.request.urlopen
    - httpx usage
    - hf_hub_download, snapshot_download
    - subprocess with curl/wget/git clone
    """
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.matches: List[Dict[str, Any]] = []
        
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes."""
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno)
        
        # (1) requests.* / httpx.*
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            module_name = self._get_module_name(node.func.value)
            
            if module_name in {"requests", "httpx"}:
                if attr_name in {"get", "post", "put", "delete", "request", "head", "options", "patch"}:
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": f"{module_name}.{attr_name}",
                        "snippet": f"{module_name}.{attr_name}() at line {lineno}",
                        "risk": "http_request",
                    })
            
            # (2) urllib.request.urlopen
            if attr_name == "urlopen":
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": "urllib.request.urlopen",
                    "snippet": f"urlopen() at line {lineno}",
                    "risk": "http_request",
                })
            
            # (3) hf_hub_download / snapshot_download
            if attr_name in {"hf_hub_download", "snapshot_download", "cached_download"}:
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": attr_name,
                    "snippet": f"{attr_name}() at line {lineno}",
                    "risk": "model_download",
                })
        
        # (4) 直接调用的 hf_hub_download 等（from huggingface_hub import ...）
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in {"hf_hub_download", "snapshot_download", "cached_download"}:
                self.matches.append({
                    "path": str(self.filepath),
                    "lineno_start": lineno,
                    "lineno_end": end_lineno,
                    "symbol": func_name,
                    "snippet": f"{func_name}() at line {lineno}",
                    "risk": "model_download",
                })
        
        # (5) subprocess 调用 curl/wget/git
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            module_name = self._get_module_name(node.func.value)
            
            if module_name == "subprocess" and attr_name in {"run", "call", "Popen", "check_output"}:
                # 检查参数中是否包含 curl/wget/git
                if self._has_download_command(node):
                    self.matches.append({
                        "path": str(self.filepath),
                        "lineno_start": lineno,
                        "lineno_end": end_lineno,
                        "symbol": f"subprocess.{attr_name}",
                        "snippet": f"subprocess.{attr_name}() with download command at line {lineno}",
                        "risk": "subprocess_download",
                    })
        
        self.generic_visit(node)
    
    def _get_module_name(self, node: ast.AST) -> str:
        """Extract module name from node."""
        if isinstance(node, ast.Name):
            return node.id
        return ""
    
    def _has_download_command(self, call_node: ast.Call) -> bool:
        """Check if subprocess call contains download commands."""
        # 检查第一个参数（命令列表或字符串）
        if len(call_node.args) > 0:
            first_arg = call_node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                cmd = first_arg.value.lower()
                return any(tool in cmd for tool in ["curl", "wget", "git clone", "git pull"])
            elif isinstance(first_arg, ast.List):
                # 检查列表第一个元素
                if len(first_arg.elts) > 0:
                    first_elem = first_arg.elts[0]
                    if isinstance(first_elem, ast.Constant) and isinstance(first_elem.value, str):
                        cmd = first_elem.value.lower()
                        return cmd in ["curl", "wget", "git"]
        return False


def scan_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Scan a single Python file for network access calls.
    
    Args:
        filepath: Path to Python source file
        
    Returns:
        List of match dictionaries
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        visitor = NetworkAccessVisitor(filepath)
        visitor.visit(tree)
        return visitor.matches
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []


def _has_controlled_download_gate(repo_root: Path) -> bool:
    """
    功能：检查下载是否受控制。

    Check if HF hub downloads are controlled (offline-only enforced).

    Args:
        repo_root: Repository root directory.

    Returns:
        True if downloads are fully controlled (offline-only); False otherwise.
    """
    weights_snapshot_path = repo_root / "main" / "diffusion" / "sd3" / "weights_snapshot.py"
    if not weights_snapshot_path.exists():
        return False

    try:
        source = weights_snapshot_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False

    # (P0-A) 检查离线强制模式：
    # Check for offline-only enforcement pattern:
    # "offline_only_enforcement" marker + hardcoded local_files_only=True
    offline_enforcement = "offline_only_enforcement" in source and '"local_files_only": True' in source
    if offline_enforcement:
        return True

    freeze_gate_path = repo_root / "main" / "policy" / "freeze_gate.py"
    if not freeze_gate_path.exists():
        return False

    try:
        freeze_source = freeze_gate_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False

    gate_ok = "assert_pipeline_hf_hub_download_allowed" in freeze_source and "local_files_only" in freeze_source
    # pipeline_realization policy removed; default to controlled if offline enforcement exists.
    return offline_enforcement or gate_ok


def classify_match(match: Dict[str, Any], repo_root: Path, controlled_download: bool) -> str:
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
    
    # main/ 中 → FAIL（需要审计）
    if rel_path.startswith("main/"):
        if match.get("risk") == "model_download" and controlled_download:
            return "CONTROLLED"
        return "FAIL"
    
    return "WARNING"


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    Execute network access scan audit.
    
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
    controlled_download = _has_controlled_download_gate(repo_root)
    fail_matches = []
    warning_matches = []
    controlled_matches = []
    
    for match in all_matches:
        classification = classify_match(match, repo_root, controlled_download)
        match["classification"] = classification
        if classification == "FAIL":
            fail_matches.append(match)
        elif classification == "CONTROLLED":
            controlled_matches.append(match)
        else:
            warning_matches.append(match)
    
    # 按风险类型统计
    http_requests = [m for m in fail_matches if m["risk"] == "http_request"]
    model_downloads = [m for m in fail_matches if m["risk"] == "model_download"]
    subprocess_downloads = [m for m in fail_matches if m["risk"] == "subprocess_download"]
    
    # 判定结果
    result = "FAIL" if len(fail_matches) > 0 else "PASS"
    
    evidence = {
        "matches": all_matches,
        "fail_count": len(fail_matches),
        "warning_count": len(warning_matches),
        "controlled_count": len(controlled_matches),
        "http_request_count": len(http_requests),
        "model_download_count": len(model_downloads),
        "subprocess_download_count": len(subprocess_downloads),
    }
    
    fail_reasons = []
    if len(http_requests) > 0:
        fail_reasons.append(f"发现 {len(http_requests)} 处 HTTP 请求调用")
    if len(model_downloads) > 0:
        fail_reasons.append(f"发现 {len(model_downloads)} 处模型下载调用（需记录来源与 hash）")
    if len(subprocess_downloads) > 0:
        fail_reasons.append(f"发现 {len(subprocess_downloads)} 处 subprocess 下载命令")
    
    if fail_reasons:
        impact = "; ".join(fail_reasons)
    elif controlled_matches:
        impact = f"发现 {len(controlled_matches)} 处受控下载调用（已由 freeze_gate + whitelist 覆盖）"
    else:
        impact = "未发现未审计的网络访问"
    
    fix_suggestion = (
        "1. 若允许模型下载，必须记录下载来源（repo + revision）与 hash，并锚定到 run_closure；"
        "2. 禁止运行时隐式网络访问，所有下载应在离线准备阶段完成；"
        "3. 对 HTTP 请求进行审计，确保不影响结果可复现性；"
        "4. 移除或隔离 subprocess 下载命令"
        if len(fail_matches) > 0
        else "N.A."
    )
    
    return {
        "audit_id": "D10.network_access_scan",
        "gate_name": "gate.network_access",
        "category": "D",
        "severity": "BLOCK",
        "result": result,
        "rule": "禁止运行时隐式网络访问或下载，所有网络操作必须审计并锚定来源",
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
