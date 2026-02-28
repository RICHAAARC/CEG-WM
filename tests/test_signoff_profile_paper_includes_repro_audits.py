"""
功能：paper profile 包含 implementable 与 repro_bundle_integrity 审计

Module type: General module

Regression test: paper profile must include protocol implementability
and repro bundle integrity audits.
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
    PAPER_PROFILE_ADDITIONAL_AUDITS,
    resolve_signoff_profile,
)


def test_signoff_profile_paper_includes_implementable_and_repro_bundle():
    """
    功能：paper profile 包含 implementable 与 repro_bundle_integrity 审计。

    Verify that paper profile includes both baseline audits AND
    additional paper-grade audits (protocol implementability and repro bundle).

    GIVEN: paper profile specification
    WHEN: resolve_signoff_profile("paper") is called
    THEN: Returned list contains MINIMUM_AUDIT_SCRIPTS + PAPER_PROFILE_ADDITIONAL_AUDITS.
    """
    paper_audits = resolve_signoff_profile("paper")

    # 必须包含所有 baseline 审计（append-only）
    for audit in MINIMUM_AUDIT_SCRIPTS:
        assert audit in paper_audits, \
            f"paper profile 必须包含 baseline 审计 '{audit}'，实际缺失"

    # 必须包含论文级追加审计
    for audit in PAPER_PROFILE_ADDITIONAL_AUDITS:
        assert audit in paper_audits, \
            f"paper profile 必须包含论文级审计 '{audit}'，实际缺失"

    # 明确验证两个关键审计
    assert "audits/audit_attack_protocol_implementable.py" in paper_audits, \
        "paper profile 必须包含 audit_attack_protocol_implementable.py"
    assert "audits/audit_repro_bundle_integrity.py" in paper_audits, \
        "paper profile 必须包含 audit_repro_bundle_integrity.py"
    assert "audits/audit_protocol_compare_outputs_schema.py" in paper_audits, \
        "paper profile 必须包含 audit_protocol_compare_outputs_schema.py"

    # 验证数量（baseline 12 + paper 2 = 14）
    expected_count = len(MINIMUM_AUDIT_SCRIPTS) + len(PAPER_PROFILE_ADDITIONAL_AUDITS)
    assert len(paper_audits) == expected_count, \
        f"paper profile 必须包含 {expected_count} 个审计，实际为 {len(paper_audits)}"

    # publish profile 应该与 paper 相同
    publish_audits = resolve_signoff_profile("publish")
    assert publish_audits == paper_audits, \
        "publish profile 必须与 paper profile 相同"


def test_signoff_profile_paper_includes_matrix_schema_audit():
    """
    功能：paper profile 必须包含 matrix schema 审计。

    Verify paper profile includes experiment matrix outputs schema audit.

    Args:
        None.

    Returns:
        None.
    """
    paper_audits = resolve_signoff_profile("paper")
    assert MATRIX_SCHEMA_AUDIT_SCRIPT in paper_audits, \
        "paper profile 必须包含 audit_experiment_matrix_outputs_schema.py"


def test_signoff_profile_publish_includes_matrix_schema_audit():
    """
    功能：publish profile 必须包含 matrix schema 审计。

    Verify publish profile includes experiment matrix outputs schema audit.

    Args:
        None.

    Returns:
        None.
    """
    publish_audits = resolve_signoff_profile("publish")
    assert MATRIX_SCHEMA_AUDIT_SCRIPT in publish_audits, \
        "publish profile 必须包含 audit_experiment_matrix_outputs_schema.py"


def test_signoff_paper_protocol_compare_audit_enforces_run_root_and_all_ok() -> None:
    """
    功能：验证 paper/publish signoff 会为 protocol_compare 审计传入强绑定与全成功开关。

    Verify run_freeze_signoff passes --run-root, --require-compare-summary,
    and --require-all-ok for protocol compare audit in paper/publish profile.

    Args:
        None.

    Returns:
        None.
    """
    script_path = _repo_root / "scripts" / "run_freeze_signoff.py"
    content = script_path.read_text(encoding="utf-8")

    assert "elif relative_script == PROTOCOL_COMPARE_SCHEMA_AUDIT_SCRIPT" in content
    assert "--run-root" in content
    assert "--require-compare-summary" in content
    assert "--require-all-ok" in content
