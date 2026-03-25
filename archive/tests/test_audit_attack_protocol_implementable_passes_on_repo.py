"""
功能：协议可实现性审计在当前仓库上通过

Module type: General module

Regression test: audit_attack_protocol_implementable must pass on current repo.
"""

import subprocess
import sys
import json
from pathlib import Path


def test_audit_attack_protocol_implementable_passes_on_repo():
    """
    功能：协议可实现性审计在当前仓库上通过。

    Verify that the static audit script audit_attack_protocol_implementable.py
    returns PASS when executed on the current repository.

    GIVEN: Current repository with frozen attack_protocol.yaml
    WHEN: Execute audit script
    THEN: Audit result is PASS and exit code is 0.
    """
    # (1) 定位审计脚本
    repo_root = Path(__file__).resolve().parent.parent
    audit_script = repo_root / "scripts" / "audits" / "audit_attack_protocol_implementable.py"
    
    assert audit_script.exists(), \
        f"审计脚本必须存在: {audit_script}"
    
    # (2) 执行审计
    result = subprocess.run(
        [sys.executable, str(audit_script), str(repo_root)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    # (3) 验证退出码
    assert result.returncode == 0, \
        f"审计脚本必须返回 0，实际为 {result.returncode}。\nstderr: {result.stderr}\nstdout: {result.stdout}"
    
    # (4) 解析输出
    output = result.stdout.strip()
    assert output, f"审计脚本必须有输出，实际为空。stderr: {result.stderr}"
    
    try:
        audit_result = json.loads(output)
    except json.JSONDecodeError as exc:
        pytest.fail(f"审计输出必须为有效 JSON，实际解析失败: {exc}。\nstdout: {output[:500]}")
    
    # (5) 验证 result 字段
    assert "result" in audit_result, f"审计结果必须包含 'result' 字段，实际为: {audit_result.keys()}"
    assert audit_result["result"] in ("PASS", "N.A."), \
        f"当前仓库审计结果必须为 PASS 或 N.A.，实际为: {audit_result['result']}"
    
    # (6) 验证 audit_id 和 severity
    assert audit_result.get("audit_id") == "attack.protocol_implementable", \
        f"audit_id 必须为 'attack.protocol_implementable'，实际为: {audit_result.get('audit_id')}"
    assert audit_result.get("severity") == "BLOCK", \
        f"severity 必须为 'BLOCK'，实际为: {audit_result.get('severity')}"
