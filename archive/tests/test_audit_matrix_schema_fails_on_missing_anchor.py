"""
功能：experiment matrix 审计在缺失锚点字段时失败

Module type: General module

Regression test: audit_experiment_matrix_outputs_schema fails on missing anchor field.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest


def test_audit_experiment_matrix_outputs_schema_fails_on_missing_anchor():
    """
    功能：experiment matrix 审计在缺失锚点字段时失败。

    Verify that audit_experiment_matrix_outputs_schema.py fails
    when grid_summary.json is missing a required anchor field.

    GIVEN: grid_summary.json missing cfg_digest field
    WHEN: Execute audit script
    THEN: Audit result is FAIL, exit code is 1, and evidence contains missing_fields.
    """
    repo_root = Path(__file__).resolve().parent.parent
    audit_script = repo_root / "scripts" / "audits" / "audit_experiment_matrix_outputs_schema.py"

    assert audit_script.exists(), \
        f"审计脚本必须存在: {audit_script}"

    # 创建临时目录结构并写入缺失 cfg_digest 的 summary
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        outputs_dir = tmp_root / "outputs" / "experiment_matrix"
        artifacts_dir = outputs_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 构造缺失 cfg_digest 的 summary
        incomplete_summary = {
            "strict": False,
            "executed": 1,
            "results": [],
            # cfg_digest 缺失
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
            json.dump(incomplete_summary, f, indent=2, ensure_ascii=False)

        # 执行审计
        result = subprocess.run(
            [sys.executable, str(audit_script), str(tmp_root)],
            capture_output=True,
            text=False,
            timeout=30,
        )

        # 验证退出码（必须为非零）
        stdout_text = result.stdout.decode("utf-8", errors="replace") if isinstance(result.stdout, (bytes, bytearray)) else str(result.stdout)
        stderr_text = result.stderr.decode("utf-8", errors="replace") if isinstance(result.stderr, (bytes, bytearray)) else str(result.stderr)

        assert result.returncode != 0, \
            f"审计脚本在缺失字段时必须返回非零退出码，实际为 {result.returncode}。\nstderr: {stderr_text}\nstdout: {stdout_text}"

        # 解析输出
        output = stdout_text.strip()
        assert output, f"审计脚本必须有输出，实际为空。stderr: {stderr_text}"

        try:
            audit_result = json.loads(output)
        except json.JSONDecodeError as exc:
            pytest.fail(f"审计输出必须为有效 JSON，实际解析失败: {exc}。\nstdout: {output[:500]}")

        # 验证 result 字段
        assert "result" in audit_result, f"审计结果必须包含 'result' 字段，实际为: {audit_result.keys()}"
        assert audit_result["result"] == "FAIL", \
            f"缺失字段时审计结果必须为 FAIL，实际为: {audit_result['result']}"

        # 验证证据字段包含 missing_fields 信息
        evidence = audit_result.get("evidence", {})
        assert "failures" in evidence or "missing_fields" in str(evidence).lower(), \
            f"证据必须包含缺失字段信息，实际为: {evidence}"

        # 验证 cfg_digest 被标识为缺失
        evidence_str = json.dumps(evidence)
        assert "cfg_digest" in evidence_str, \
            f"证据必须明确指出 cfg_digest 缺失，实际为: {evidence}"
