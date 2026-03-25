"""
功能：协议可实现性审计在非法协议上失败

Module type: General module

Regression test: audit_attack_protocol_implementable must fail when protocol contains unknown family.
"""

import subprocess
import sys
import json
import tempfile
import shutil
from pathlib import Path
import pytest


def test_audit_attack_protocol_implementable_fails_on_injected_bad_protocol():
    """
    功能：协议可实现性审计在注入非法协议后失败。

    Verify that the static audit script audit_attack_protocol_implementable.py
    returns FAIL when attack_protocol.yaml contains unknown family.

    GIVEN: Temporary config directory with attack_protocol.yaml containing unknown family
    WHEN: Execute audit script with injected bad protocol
    THEN: Audit result is FAIL and exit code is 1.
    """
    # (1) 定位审计脚本和真实仓库根
    real_repo_root = Path(__file__).resolve().parent.parent
    audit_script = real_repo_root / "scripts" / "audits" / "audit_attack_protocol_implementable.py"
    
    assert audit_script.exists(), \
        f"审计脚本必须存在: {audit_script}"
    
    # (2) 创建临时仓库结构并注入非法协议
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        tmp_configs = tmp_root / "configs"
        tmp_configs.mkdir(parents=True, exist_ok=True)
        
        # 构造非法协议（包含未知族）
        bad_protocol = {
            "protocol_version": "attack_protocol_v1",
            "protocol_digest": "dummy_digest_for_test",
            "protocol_meta": {
                "creation_date": "2025-01-06T00:00:00Z",
                "frozen_by": "test_harness",
            },
            "families": [
                "UNKNOWN_ATTACK_FAMILY_NEVER_EXISTS",
            ],
            "params_versions": {
                "UNKNOWN_ATTACK_FAMILY_NEVER_EXISTS": {
                    "v1": {
                        "family": "UNKNOWN_ATTACK_FAMILY_NEVER_EXISTS",
                        "default_params": {},
                    },
                },
            },
        }
        
        # 写入临时配置
        import yaml
        bad_protocol_path = tmp_configs / "attack_protocol.yaml"
        with bad_protocol_path.open("w", encoding="utf-8") as f:
            yaml.dump(bad_protocol, f, allow_unicode=True)
        
        # 复制真实仓库的 main 包到临时目录（审计脚本需要导入）
        # 注意：此测试依赖审计脚本能够加载临时目录下的协议，
        # 如果审计脚本硬编码了路径，此测试将失败（这是预期行为，暴露硬编码问题）
        # 为简化测试，这里采用环境隔离策略：将 tmp_root 作为 repo_root 传给审计脚本
        
        # 审计脚本期望在 {repo_root}/configs/attack_protocol.yaml 找到协议
        # 已创建 tmp_configs/attack_protocol.yaml
        
        # 审计脚本还需要导入 main 包，所以需要复制或模拟 main 包
        # 为避免完整复制，直接修改 PYTHONPATH 让审计脚本能找到真实 main 包
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(real_repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        
        # (3) 执行审计（传入临时仓库根目录）
        result = subprocess.run(
            [sys.executable, str(audit_script), str(tmp_root)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        
        # (4) 验证退出码（必须为非零，表示审计失败）
        assert result.returncode != 0, \
            f"审计脚本在非法协议下必须返回非零退出码，实际为 {result.returncode}。\nstderr: {result.stderr}\nstdout: {result.stdout}"
        
        # (5) 解析输出
        output = result.stdout.strip()
        assert output, f"审计脚本必须有输出，实际为空。stderr: {result.stderr}"
        
        try:
            audit_result = json.loads(output)
        except json.JSONDecodeError as exc:
            pytest.fail(f"审计输出必须为有效 JSON，实际解析失败: {exc}。\nstdout: {output[:500]}")
        
        # (6) 验证 result 字段
        assert "result" in audit_result, f"审计结果必须包含 'result' 字段，实际为: {audit_result.keys()}"
        assert audit_result["result"] == "FAIL", \
            f"非法协议审计结果必须为 FAIL，实际为: {audit_result['result']}"
        
        # (7) 验证证据字段包含不支持的族名
        evidence = audit_result.get("evidence", {})
        assert "unsupported_families" in str(evidence).lower() or "unknown" in str(evidence).lower(), \
            f"证据必须指出不支持的族，实际为: {evidence}"
