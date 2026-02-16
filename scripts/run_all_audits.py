"""
功能：审计聚合器，执行所有静态审计并输出统一报告

Module type: Core innovation module

Aggregate all audit results and compute freeze sign-off decision.
Output follows the unified audit result schema.
"""

import json
import subprocess
import sys
import locale
from pathlib import Path
from typing import List, Dict, Any, Optional


# 审计脚本清单（固定顺序）
AUDIT_SCRIPTS = [
    "audits/audit_write_bypass_scan.py",
    "audits/audit_yaml_loader_uniqueness.py",
    "audits/audit_freeze_surface_integrity.py",
    "audits/audit_registry_injection_surface.py",
    "audits/audit_policy_path_semantics_binding.py",
    "audits/audit_injection_scope_manifest_binding.py",
    "audits/audit_dangerous_exec_and_pickle_scan.py",
    "audits/audit_network_access_scan.py",
]


def _decode_bytes(data: Optional[bytes]) -> str:
    """
    功能：解码子进程字节输出。

    Decode subprocess bytes with stable fallback.

    Args:
        data: Bytes or None.

    Returns:
        Decoded text with replacement for invalid bytes.

    Raises:
        TypeError: If data is not bytes or None.
    """
    if data is None:
        return ""
    if not isinstance(data, (bytes, bytearray)):
        # data 类型不符合预期，必须 fail-fast。
        raise TypeError("data must be bytes or None")
    try:
        return bytes(data).decode("utf-8", errors="replace")
    except Exception:
        encoding = locale.getpreferredencoding(False)
        return bytes(data).decode(encoding, errors="replace")


def execute_audit_script(script_path: Path, repo_root: Path) -> Optional[Dict[str, Any]]:
    """
    Execute a single audit script.
    
    Args:
        script_path: Path to audit script
        repo_root: Repository root directory
        
    Returns:
        Audit result dictionary or None if execution failed
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(repo_root)],
            capture_output=True,
            text=False,
            timeout=60,
        )

        stdout_text = _decode_bytes(result.stdout)
        stderr_text = _decode_bytes(result.stderr)
        
        if result.returncode != 0:
            # 审计脚本异常退出
            return {
                "audit_id": f"ERROR.{script_path.stem}",
                "gate_name": f"gate.{script_path.stem}",
                "category": "G",
                "severity": "BLOCK",
                "result": "FAIL",
                "rule": f"审计脚本 {script_path.name} 执行失败",
                "evidence": {
                    "error": stderr_text[:500] if stderr_text else "Unknown error",
                    "exit_code": result.returncode,
                    "stdout_text": stdout_text[:1000],
                    "stderr_text": stderr_text[:1000],
                },
                "impact": f"审计脚本异常退出，无法完成对应检查项",
                "fix": f"修复审计脚本 {script_path.name} 的执行错误",
            }
        
        # 解析输出
        output = stdout_text
        if not output.strip():
            # 无输出
            return {
                "audit_id": f"ERROR.{script_path.stem}",
                "gate_name": f"gate.{script_path.stem}",
                "category": "G",
                "severity": "BLOCK",
                "result": "FAIL",
                "rule": f"审计脚本 {script_path.name} 无输出",
                "evidence": {
                    "error": "No output from audit script",
                    "stdout_text": stdout_text[:1000],
                    "stderr_text": stderr_text[:1000],
                    "exit_code": result.returncode,
                },
                "impact": "审计脚本未产生结果，无法完成对应检查项",
                "fix": f"修复审计脚本 {script_path.name}，确保输出符合规范",
            }
        
        audit_result = json.loads(output)
        return audit_result
        
    except subprocess.TimeoutExpired:
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "gate_name": f"gate.{script_path.stem}",
            "category": "G",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"审计脚本 {script_path.name} 执行超时",
            "evidence": {"error": "Execution timeout (60s)"},
            "impact": "审计脚本执行超时，无法完成对应检查项",
            "fix": f"优化审计脚本 {script_path.name} 的性能或增加超时时间",
        }
    except json.JSONDecodeError as e:
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "gate_name": f"gate.{script_path.stem}",
            "category": "G",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"审计脚本 {script_path.name} 输出格式错误",
            "evidence": {"error": str(e), "output_preview": output[:200]},
            "impact": "审计脚本输出不符合 JSON 规范",
            "fix": f"修复审计脚本 {script_path.name}，确保输出有效的 JSON",
        }
    except Exception as e:
        return {
            "audit_id": f"ERROR.{script_path.stem}",
            "gate_name": f"gate.{script_path.stem}",
            "category": "G",
            "severity": "BLOCK",
            "result": "FAIL",
            "rule": f"审计脚本 {script_path.name} 执行异常",
            "evidence": {"error": str(e)},
            "impact": "审计脚本执行过程中发生异常",
            "fix": f"修复审计脚本 {script_path.name} 的异常",
        }


def validate_audit_result(result: Dict[str, Any]) -> List[str]:
    """
    Validate audit result against unified schema.
    
    Args:
        result: Audit result dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # 必需字段
    required_fields = ["audit_id", "gate_name", "category", "severity", "result", "rule", "evidence", "impact", "fix"]
    for field in required_fields:
        if field not in result:
            errors.append(f"缺少必需字段: {field}")
    
    # 字段值校验
    if "result" in result and result["result"] not in {"PASS", "FAIL", "N.A."}:
        errors.append(f"result 字段值非法: {result['result']}")
    
    if "severity" in result and result["severity"] not in {"BLOCK", "NON_BLOCK"}:
        errors.append(f"severity 字段值非法: {result['severity']}")
    
    # 对抗式扫描必须输出命中列表
    if "scan" in result.get("audit_id", "").lower():
        if "evidence" not in result or "matches" not in result.get("evidence", {}):
            errors.append("对抗式扫描未输出命中列表（evidence.matches）")
    
    return errors


def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute audit summary and freeze sign-off decision.
    
    Args:
        results: List of audit result dictionaries
        
    Returns:
        Summary dictionary
    """
    pass_count = sum(1 for r in results if r["result"] == "PASS")
    fail_count = sum(1 for r in results if r["result"] == "FAIL")
    na_count = sum(1 for r in results if r["result"] == "N.A.")
    
    block_fails = [r for r in results if r["result"] == "FAIL" and r["severity"] == "BLOCK"]
    non_block_fails = [r for r in results if r["result"] == "FAIL" and r["severity"] == "NON_BLOCK"]
    
    # FreezeSignoffDecision
    if len(block_fails) > 0:
        decision = "BLOCK_FREEZE"
    else:
        decision = "ALLOW_FREEZE"
    
    # BlockingReasons
    blocking_reasons = []
    for fail in block_fails:
        blocking_reasons.append({
            "audit_id": fail["audit_id"],
            "rule": fail["rule"],
            "impact": fail["impact"],
            "evidence_summary": _summarize_evidence(fail["evidence"]),
            "fix": fail["fix"],
        })
    
    # RiskSummary
    if len(block_fails) > 0:
        risk_summary = "HIGH"
    elif fail_count > 0:
        risk_summary = "MED"
    else:
        risk_summary = "LOW"
    
    return {
        "FreezeSignoffDecision": decision,
        "BlockingReasons": blocking_reasons,
        "RiskSummary": risk_summary,
        "counts": {
            "PASS": pass_count,
            "FAIL": fail_count,
            "N.A.": na_count,
            "BLOCK_fails": len(block_fails),
            "NON_BLOCK_fails": len(non_block_fails),
        },
    }


def _summarize_evidence(evidence: Any) -> str:
    """
    Summarize evidence for blocking reason.
    
    Args:
        evidence: Evidence dictionary or value
        
    Returns:
        One-line evidence summary
    """
    if isinstance(evidence, dict):
        if "matches" in evidence:
            fail_count = evidence.get("fail_count", 0)
            return f"{fail_count} 处命中点" if fail_count > 0 else "命中列表为空"
        elif "checks" in evidence:
            failed_checks = [k for k, v in evidence["checks"].items() if not v.get("pass", False)]
            return f"失败的检查: {', '.join(failed_checks)}" if failed_checks else "检查通过"
        else:
            return str(evidence)[:100]
    return str(evidence)[:100]


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行所有审计脚本并生成聚合报告")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="仓库根目录（默认当前目录）")
    parser.add_argument("--output", type=Path, default=None, help="报告输出路径（默认 stdout）")
    parser.add_argument("--strict", action="store_true", help="严格模式：对抗式扫描未输出命中列表视为 FAIL")
    
    args = parser.parse_args()
    
    repo_root = args.repo_root.resolve()
    scripts_dir = Path(__file__).parent
    
    # 执行所有审计脚本
    all_results = []
    
    for script_name in AUDIT_SCRIPTS:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            # 审计脚本缺失
            all_results.append({
                "audit_id": f"ERROR.missing_{script_path.stem}",
                "gate_name": f"gate.{script_path.stem}",
                "category": "G",
                "severity": "BLOCK",
                "result": "FAIL",
                "rule": f"审计脚本 {script_name} 缺失",
                "evidence": {"missing_script": str(script_path)},
                "impact": "审计脚本缺失，无法完成对应检查项",
                "fix": f"创建审计脚本 {script_name}",
            })
            continue
        
        audit_result = execute_audit_script(script_path, repo_root)
        if audit_result is None:
            continue
        
        # 校验审计结果格式
        validation_errors = validate_audit_result(audit_result)
        if validation_errors:
            if args.strict:
                # 严格模式：格式错误视为 FAIL
                audit_result["result"] = "FAIL"
                audit_result["impact"] += f" (Validation errors: {'; '.join(validation_errors)})"
        
        all_results.append(audit_result)
    
    # 计算汇总
    summary = compute_summary(all_results)
    
    # 构造最终报告
    report = {
        "summary": summary,
        "results": all_results,
        "metadata": {
            "repo_root": str(repo_root),
            "audit_count": len(all_results),
            "audit_scripts": AUDIT_SCRIPTS,
        },
    }
    
    # 输出报告
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    
    if args.output:
        args.output.write_text(report_json, encoding="utf-8")
        print(f"审计报告已写入: {args.output}")
    else:
        # 输出到 stdout，使用 UTF-8 编码确保兼容性
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout.buffer.write(report_json.encode('utf-8'))
            sys.stdout.buffer.write(b'\n')
        else:
            print(report_json)
    
    # 返回退出码
    if summary["FreezeSignoffDecision"] == "BLOCK_FREEZE":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
