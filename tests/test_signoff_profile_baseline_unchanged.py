"""
功能：baseline profile 审计清单与历史保持一致

Module type: General module

Regression test: baseline profile must preserve historical minimum audit set.
"""

import sys
from pathlib import Path

# 添加 repo 到 sys.path 以支持导入
_test_dir = Path(__file__).resolve().parent
_repo_root = _test_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.run_freeze_signoff import (
    MINIMUM_AUDIT_SCRIPTS,
    MATRIX_SCHEMA_AUDIT_SCRIPT,
    resolve_signoff_profile,
)


def test_signoff_profile_baseline_unchanged():
    """
    功能：baseline profile 审计清单与历史保持一致。

    Verify that baseline profile returns exactly the historical
    MINIMUM_AUDIT_SCRIPTS list, ensuring backward compatibility.

    GIVEN: MINIMUM_AUDIT_SCRIPTS constant (12 audits)
    WHEN: resolve_signoff_profile("baseline") is called
    THEN: Returned list equals MINIMUM_AUDIT_SCRIPTS.
    """
    baseline_audits = resolve_signoff_profile("baseline")

    # 必须与历史最小集合完全一致
    assert baseline_audits == list(MINIMUM_AUDIT_SCRIPTS), \
        f"baseline profile 必须返回 MINIMUM_AUDIT_SCRIPTS，实际为: {baseline_audits}"

    # 验证至少包含历史核心审计
    expected_core_audits = [
        "audits/audit_write_bypass_scan.py",
        "audits/audit_freeze_surface_integrity.py",
        "audits/audit_evaluation_report_schema.py",
        "audits/audit_thresholds_readonly_enforcement.py",
        "audits/audit_attack_protocol_hardcoding.py",
    ]
    for audit in expected_core_audits:
        assert audit in baseline_audits, \
            f"baseline profile 必须包含核心审计 '{audit}'，实际缺失"

    # 验证数量（历史为 12 个）
    assert len(baseline_audits) == 12, \
        f"baseline profile 必须包含 12 个审计，实际为 {len(baseline_audits)}"


def test_signoff_profile_baseline_audits_unchanged():
    """
    功能：baseline profile 不得引入 matrix schema 审计。

    Verify baseline profile remains unchanged and does not include matrix schema audit.

    Args:
        None.

    Returns:
        None.
    """
    baseline_audits = resolve_signoff_profile("baseline")

    assert baseline_audits == list(MINIMUM_AUDIT_SCRIPTS), \
        "baseline profile 必须与历史 MINIMUM_AUDIT_SCRIPTS 完全一致"
    assert MATRIX_SCHEMA_AUDIT_SCRIPT not in baseline_audits, \
        "baseline profile 不得新增 matrix schema 审计"
