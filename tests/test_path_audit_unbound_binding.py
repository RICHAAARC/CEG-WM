"""
功能：测试 path_audit 在未绑定事实源时不得落盘（B5/D2 覆盖）

Module type: Core innovation module

Test that path_audit is not written when fact sources are unbound,
and run_closure carries fixed status/error code.
"""

import json
from pathlib import Path

import pytest


def _build_minimal_run_meta() -> dict:
    """
    功能：构造最小 run_meta 用于 finalize_run。 

    Build minimal run_meta mapping for finalize_run.

    Args:
        None.

    Returns:
        Minimal run_meta dict.
    """
    from main.core import time_utils
    from main.core.errors import RunFailureReason

    now = time_utils.now_utc_iso_z()
    return {
        "run_id": "test_run_unbound_001",
        "command": "unit_test",
        "created_at_utc": now,
        "started_at": now,
        "ended_at": now,
        "cfg_digest": "cfg_digest_test",
        "policy_path": "test_policy_path",
        "impl_id": "test_impl",
        "impl_version": "1.0.0",
        "manifest_rel_path": "artifacts/records_manifest.json",
        "status_ok": False,
        "status_reason": RunFailureReason.RUNTIME_ERROR,
        "status_details": "fact_sources_unbound"
    }


def test_path_audit_not_written_when_fact_sources_unbound(tmp_run_root: Path):
    """
    Test that path_audit is not written when fact sources are unbound.

    当 fact sources 未初始化时，path_audit 不得落盘，run_closure 必须标记失败状态与固定错误码。
    """
    from main.core import status

    run_meta = _build_minimal_run_meta()
    records_dir = tmp_run_root / "records"
    artifacts_dir = tmp_run_root / "artifacts"

    run_closure_path = status.finalize_run(tmp_run_root, records_dir, artifacts_dir, run_meta)
    assert run_closure_path.exists()

    payload = json.loads(run_closure_path.read_text(encoding="utf-8"))
    assert payload.get("path_audit_status") == "failed"
    assert payload.get("path_audit_error_code") == "fact_sources_unbound"

    path_audits_dir = artifacts_dir / "path_audits"
    path_audit_files = sorted(path_audits_dir.glob("path_audit_*.json")) if path_audits_dir.exists() else []
    assert len(path_audit_files) == 0
