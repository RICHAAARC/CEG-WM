"""
功能：experiment matrix 审计在最小 summary 上通过

Module type: General module

Regression test: audit_experiment_matrix_outputs_schema passes on minimal summary.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest


def test_audit_experiment_matrix_outputs_schema_passes_on_minimal_summary():
    """
    功能：experiment matrix 审计在最小 summary 上通过。

    Verify that audit_experiment_matrix_outputs_schema.py passes
    when grid_summary.json contains all required anchor fields.

    GIVEN: Minimal grid_summary.json with all required anchor fields
    WHEN: Execute audit script
    THEN: Audit result is PASS and exit code is 0.
    """
    repo_root = Path(__file__).resolve().parent.parent
    audit_script = repo_root / "scripts" / "audits" / "audit_experiment_matrix_outputs_schema.py"

    assert audit_script.exists(), \
        f"审计脚本必须存在: {audit_script}"

    # 创建临时目录结构并写入最小 grid_summary.json
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        outputs_dir = tmp_root / "outputs" / "experiment_matrix"
        artifacts_dir = outputs_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 构造包含所有必备锚点字段的最小 summary
        minimal_summary = {
            "strict": False,
            "executed": 1,
            "results": [],
            "cfg_digest": "abc123",
            "thresholds_digest": "def456",
            "threshold_metadata_digest": "ghi789",
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "jkl012",
            "attack_coverage_digest": "mno345",
            "impl_digest": "pqr678",
            "fusion_rule_version": "fusion_v1",
        }

        grid_summary_path = artifacts_dir / "grid_summary.json"
        with grid_summary_path.open("w", encoding="utf-8") as f:
            json.dump(minimal_summary, f, indent=2, ensure_ascii=False)

        # 执行审计
        result = subprocess.run(
            [sys.executable, str(audit_script), str(tmp_root)],
            capture_output=True,
            text=False,
            timeout=30,
        )

        stdout_text = result.stdout.decode("utf-8", errors="replace") if isinstance(result.stdout, (bytes, bytearray)) else str(result.stdout)
        stderr_text = result.stderr.decode("utf-8", errors="replace") if isinstance(result.stderr, (bytes, bytearray)) else str(result.stderr)

        # 验证退出码
        assert result.returncode == 0, \
            f"审计脚本必须返回 0，实际为 {result.returncode}。\nstderr: {stderr_text}\nstdout: {stdout_text}"

        # 解析输出
        output = stdout_text.strip()
        assert output, f"审计脚本必须有输出，实际为空。stderr: {stderr_text}"

        try:
            audit_result = json.loads(output)
        except json.JSONDecodeError as exc:
            pytest.fail(f"审计输出必须为有效 JSON，实际解析失败: {exc}。\nstdout: {output[:500]}")

        # 验证 result 字段
        assert "result" in audit_result, f"审计结果必须包含 'result' 字段，实际为: {audit_result.keys()}"
        assert audit_result["result"] == "PASS", \
            f"最小 summary 审计结果必须为 PASS，实际为: {audit_result['result']}。证据: {audit_result.get('evidence')}"

        # 验证 audit_id 和 severity
        assert audit_result.get("audit_id") == "experiment_matrix.outputs_schema", \
            f"audit_id 必须为 'experiment_matrix.outputs_schema'，实际为: {audit_result.get('audit_id')}"
        assert audit_result.get("severity") == "BLOCK", \
            f"severity 必须为 'BLOCK'，实际为: {audit_result.get('severity')}"
