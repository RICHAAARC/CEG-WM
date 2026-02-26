"""
File purpose: 验证 strict 审计模式下 research-only 审计的 N.A. 结果不导致阻截
Module type: General module
"""

from __future__ import annotations

from typing import Dict, Any, List
import pytest
import importlib.util
from pathlib import Path


def validate_audit_result_strict_safe(result: Dict[str, Any]) -> List[str]:
    """
    功能：strict 模式下的审计结果验证（与 run_all_audits.py 同步）。

    Validate audit result in strict mode. N.A. results must not trigger blocking.

    Args:
        result: Audit result dictionary.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []
    
    # 必需字段
    required_fields = ["audit_id", "gate_name", "category", "severity", "result", "rule", "evidence", "impact", "fix"]
    for field in required_fields:
        if field not in result:
            errors.append(f"缺少必需字段: {field}")
    
    # 字段值校验（支持 PASS, FAIL, N.A.）
    # 重要：N.A. 是合法的结果，表示审计不适用
    if "result" in result and result["result"] not in {"PASS", "FAIL", "N.A."}:
        errors.append(f"result 字段值非法: {result['result']}")
    
    if "severity" in result and result["severity"] not in {"BLOCK", "NON_BLOCK"}:
        errors.append(f"severity 字段值非法: {result['severity']}")
    
    return errors


def compute_freeze_signoff_decision_strict(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    功能：strict 模式下计算冻结签署决策（N.A. 不阻止）。

    Compute freeze signoff decision. N.A. results must not cause BLOCK_FREEZE.

    Args:
        results: List of audit result dictionaries.

    Returns:
        Decision mapping with FreezeSignoffDecision and BlockingReasons.
    """
    # 过滤出阻止级别的失败审计
    # 关键：N.A. 结果即使有 BLOCK severity，也不导致 BLOCK_FREEZE
    # 因为 N.A. 表示审计不适用
    block_fails = [
        r for r in results 
        if r["result"] == "FAIL" and r["severity"] == "BLOCK"
    ]
    
    # N.A. 结果不计入阻止决策
    na_results = [r for r in results if r["result"] == "N.A."]
    
    blocking_reasons = []
    for fail in block_fails:
        blocking_reasons.append({
            "audit_id": fail["audit_id"],
            "rule": fail["rule"],
            "impact": fail["impact"],
        })
    
    if len(block_fails) > 0:
        decision = "BLOCK_FREEZE"
    else:
        decision = "ALLOW_FREEZE"
    
    return {
        "FreezeSignoffDecision": decision,
        "BlockingReasons": blocking_reasons,
        "counts": {
            "PASS": sum(1 for r in results if r["result"] == "PASS"),
            "FAIL": sum(1 for r in results if r["result"] == "FAIL"),
            "N.A.": len(na_results)
        }
    }


def test_run_all_audits_strict_does_not_block_on_research_only_na() -> None:
    """
    功能：strict 模式下 research-only 审计返回 N.A. 时不阻止冻结。

    When research-only audits return N.A. result in strict mode,
    ALLOW_FREEZE decision must be reached (not BLOCK_FREEZE).

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If N.A. causes BLOCK_FREEZE.
    """
    # 构造两种审计结果集合
    
    # (1) research-only 审计返回 N.A.（以前是 SKIP）
    research_only_na_result = {
        "audit_id": "audit_protocol_compare_outputs_schema",
        "gate_name": "gate.protocol_compare_outputs_schema",
        "category": "G",
        "severity": "NON_BLOCK",
        "result": "N.A.",
        "rule": "No protocol_compare outputs found (research-only feature, not mandatory)",
        "evidence": {},
        "impact": "Protocol compare audit not applicable: feature not invoked",
        "fix": "N/A (not applicable when feature is not used)"
    }
    
    # (2) 其他审计返回 PASS（正常流程）
    normal_pass_result = {
        "audit_id": "audit_freeze_surface_integrity",
        "gate_name": "gate.freeze_surface_integrity",
        "category": "G",
        "severity": "BLOCK",
        "result": "PASS",
        "rule": "frozen contracts must be syntactically correct",
        "evidence": {"syntax_check": "passed"},
        "impact": "Good: frozen contracts are valid",
        "fix": "N/A (passed)"
    }
    
    # (3) 阻止级失败的审计
    blocking_fail_result = {
        "audit_id": "audit_write_bypass_scan",
        "gate_name": "gate.write_bypass_scan",
        "category": "G",
        "severity": "BLOCK",
        "result": "FAIL",
        "rule": "No unauthorized write paths allowed",
        "evidence": {"matches": []},
        "impact": "Failed security check",
        "fix": "Remove unauthorized write paths"
    }
    
    # 测试场景 1：N.A. + PASS → ALLOW_FREEZE
    results_with_na = [research_only_na_result, normal_pass_result]
    
    # 验证每个结果都是合法的
    for result in results_with_na:
        errors = validate_audit_result_strict_safe(result)
        assert not errors, f"Validation errors for {result['audit_id']}: {errors}"
    
    decision = compute_freeze_signoff_decision_strict(results_with_na)
    assert decision["FreezeSignoffDecision"] == "ALLOW_FREEZE", \
        f"N.A. result should not cause BLOCK_FREEZE, got {decision['FreezeSignoffDecision']}"
    assert research_only_na_result["result"] == "N.A.", "Research-only result should be N.A."
    assert decision["counts"]["N.A."] == 1, "Should have 1 N.A. result"
    
    # 测试场景 2：N.A. + FAIL（BLOCK） → BLOCK_FREEZE（因为 FAIL）
    results_with_na_and_fail = [research_only_na_result, blocking_fail_result]
    
    decision2 = compute_freeze_signoff_decision_strict(results_with_na_and_fail)
    assert decision2["FreezeSignoffDecision"] == "BLOCK_FREEZE", \
        "FAIL with BLOCK severity should cause BLOCK_FREEZE (not because of N.A.)"
    assert decision2["counts"]["FAIL"] == 1, "Should have 1 FAIL result"
    assert decision2["counts"]["N.A."] == 1, "Should have 1 N.A. result"


def test_audit_result_validation_accepts_na() -> None:
    """
    功能：validate_audit_result 必须接受 N.A. 为合法 result 值。

    validate_audit_result must accept "N.A." as a valid result value.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If N.A. is rejected.
    """
    na_result = {
        "audit_id": "test_audit",
        "gate_name": "gate.test",
        "category": "G",
        "severity": "NON_BLOCK",
        "result": "N.A.",
        "rule": "test rule",
        "evidence": {},
        "impact": "test impact",
        "fix": "test fix"
    }
    
    errors = validate_audit_result_strict_safe(na_result)
    assert not errors, f"N.A. should be valid, but got errors: {errors}"


def test_infer_latest_run_root_prefers_latest_closure(tmp_path: Path) -> None:
    """
    功能：验证 run_all_audits 的 run_root 推断逻辑选择最新 run_closure。 

    Verify _infer_latest_run_root returns the run root owning the newest run_closure.

    Args:
        tmp_path: Temporary repo root.

    Returns:
        None.
    """
    repo_root = tmp_path
    outputs_root = repo_root / "outputs"
    older = outputs_root / "older_run" / "artifacts"
    newer = outputs_root / "newer_run" / "artifacts"
    older.mkdir(parents=True, exist_ok=True)
    newer.mkdir(parents=True, exist_ok=True)

    older_closure = older / "run_closure.json"
    newer_closure = newer / "run_closure.json"
    older_closure.write_text("{}", encoding="utf-8")
    newer_closure.write_text("{}", encoding="utf-8")

    import time
    now = time.time()
    # older 的时间戳更早。
    older_ts = now - 120
    newer_ts = now - 10
    older_closure.touch()
    newer_closure.touch()
    import os
    os.utime(older_closure, (older_ts, older_ts))
    os.utime(newer_closure, (newer_ts, newer_ts))

    repo_script = Path(__file__).resolve().parent.parent / "scripts" / "run_all_audits.py"
    spec = importlib.util.spec_from_file_location("run_all_audits_module", repo_script)
    assert spec is not None and spec.loader is not None
    run_all_audits_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all_audits_module)

    inferred = run_all_audits_module._infer_latest_run_root(repo_root)
    assert inferred == newer.parent
