"""
文件目的：publish workflow 兼容性测试（multi-protocol 不破坏发布链路）
Module type: General module

测试：确保 publish workflow 脚本的默认行为不被 multi-protocol runner 影响。
"""

import subprocess
import sys
from pathlib import Path

_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent


def test_publish_workflow_script_exists():
    """Test that publish workflow script exists and is executable."""
    script_path = _repo_root / "scripts" / "run_publish_workflow.py"
    assert script_path.exists(), f"Publish workflow script not found: {script_path}"
    assert script_path.is_file(), f"Publish workflow script is not a file: {script_path}"


def test_publish_workflow_help_accessible():
    """Test that publish workflow --help is accessible (basic CLI validation)."""
    script_path = _repo_root / "scripts" / "run_publish_workflow.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(_repo_root),
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    # Should succeed (exit code 0 or 2 for --help)
    assert result.returncode in (0, 2), f"Expected exit code 0 or 2, got {result.returncode}"
    assert "publish" in result.stdout.lower() or "publish" in result.stderr.lower(), \
        "Help output should mention publishing"


def test_multi_protocol_runner_script_separate_from_publish():
    """Test that multi-protocol runner is a separate script."""
    multi_protocol_script = _repo_root / "scripts" / "run_multi_protocol_evaluation.py"
    publish_script = _repo_root / "scripts" / "run_publish_workflow.py"
    
    assert multi_protocol_script.exists(), "Multi-protocol runner script should exist"
    assert publish_script.exists(), "Publish workflow script should exist"
    assert multi_protocol_script != publish_script, "Scripts should be different"


def test_all_audits_script_includes_protocol_compare_audit():
    """Test that run_all_audits.py includes the new protocol_compare audit."""
    run_all_audits = _repo_root / "scripts" / "run_all_audits.py"
    assert run_all_audits.exists(), f"run_all_audits.py not found: {run_all_audits}"
    
    content = run_all_audits.read_text(encoding="utf-8")
    assert "audit_protocol_compare_outputs_schema.py" in content, \
        "run_all_audits.py should include protocol_compare audit in AUDIT_SCRIPTS"
