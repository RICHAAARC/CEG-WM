"""
文件目的：attack runner 协议驱动与无硬编码回归测试。
Module type: Core innovation module
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from main.evaluation import attack_runner
from main.evaluation import protocol_loader


def test_attack_runner_uses_protocol_only_no_hardcoded_params(tmp_path: Path) -> None:
    """
    功能：验证 attack runner 参数来自协议且硬编码参数静态审计可阻断。

    Verify attack runner resolves params from protocol only and static audit blocks hardcoded params.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    condition_spec = attack_runner.resolve_condition_spec_from_protocol(protocol_spec, "rotate::v1")

    trace_records = []
    result = attack_runner.run_attacks_for_condition(
        images_or_latents={"sample": [1, 2, 3]},
        condition_spec=condition_spec,
        seed=123,
        record_hook=lambda item: trace_records.append(item),
    )

    assert result.attack_status == "ok"
    assert result.attack_condition_key == "rotate::v1"
    assert result.params_canon_sha256 == condition_spec["params_canon_sha256"]
    assert len(trace_records) == 1

    # 构造临时仓库，注入硬编码参数样例，验证静态审计 FAIL。
    fake_repo = tmp_path / "fake_repo"
    eval_dir = fake_repo / "main" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    (eval_dir / "protocol_loader.py").write_text(
        "def load_attack_protocol_spec(cfg=None):\n    return {}\n"
        "def compute_attack_protocol_digest(protocol_spec):\n    return 'x'\n",
        encoding="utf-8",
    )
    (eval_dir / "attack_runner.py").write_text(
        "scale_min = 0.8\n"
        "def run_attacks_for_condition():\n    return None\n",
        encoding="utf-8",
    )

    audit_script = Path(__file__).resolve().parent.parent / "scripts" / "audits" / "audit_attack_protocol_hardcoding.py"
    proc = subprocess.run(
        [sys.executable, str(audit_script), str(fake_repo)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    parsed = json.loads(proc.stdout)
    assert parsed.get("result") == "FAIL"
