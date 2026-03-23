"""
功能：experiment matrix 缺失语义（N.A. / FAIL）回归测试。

Module type: General module

验证 signoff 中 matrix 缺失策略开关的受控行为：
1. require=false 且缺失时返回 N.A.
2. require=true 且缺失时返回 FAIL，并触发 BLOCK_FREEZE。
"""

from pathlib import Path

from scripts.audits.audit_experiment_matrix_outputs_schema import run_audit_with_policy
from scripts.run_freeze_signoff import compute_signoff_decision


def test_matrix_audit_missing_is_na_when_not_required(tmp_path: Path) -> None:
    """
    功能：require=false 且 matrix 缺失时返回 N.A.。

    Verify missing matrix artifacts are treated as N.A. when matrix is not required.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.
    """
    repo_root = tmp_path / "repo"
    run_root = tmp_path / "run"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    result = run_audit_with_policy(
        repo_root=repo_root,
        run_root=run_root,
        require_experiment_matrix=False,
    )

    assert result.get("result") == "N.A.", "require=false 且缺失时必须返回 N.A."
    assert "N.A. because matrix not required" in str(result.get("impact", "")), \
        "N.A. 分支必须显式记录 matrix not required 原因"


def test_matrix_audit_missing_blocks_when_required(tmp_path: Path) -> None:
    """
    功能：require=true 且 matrix 缺失时返回 FAIL 并阻断冻结。

    Verify missing matrix artifacts fail audit and trigger BLOCK_FREEZE decision.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.
    """
    repo_root = tmp_path / "repo"
    run_root = tmp_path / "run"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    matrix_audit = run_audit_with_policy(
        repo_root=repo_root,
        run_root=run_root,
        require_experiment_matrix=True,
    )

    assert matrix_audit.get("result") == "FAIL", "require=true 且缺失时必须返回 FAIL"
    assert matrix_audit.get("severity") == "BLOCK", "matrix 缺失失败必须是 BLOCK 严重级别"

    decision = compute_signoff_decision(
        static_audits=[matrix_audit],
        evidence_report={"status": "ok", "errors": [], "warnings": []},
    )
    assert decision.get("decision") == "BLOCK_FREEZE", \
        "require=true 且 matrix 缺失时，signoff 决策必须为 BLOCK_FREEZE"
