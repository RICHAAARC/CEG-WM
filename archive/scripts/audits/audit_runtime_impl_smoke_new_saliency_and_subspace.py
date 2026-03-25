"""
新 impl 运行时冒烟审计。
Module type: Core innovation module

审计职责：
1. 固定执行新 impl_id 的 runtime smoke pytest。
2. 验证 saliency_source policy 与 mask-conditioned subspace planner 在审计入口可执行。
3. 失败时输出可定位的 stdout/stderr 摘要。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


SMOKE_TEST_PATH = "tests/test_runtime_impl_smoke_new_saliency_and_subspace.py"


def _decode_output(data: Optional[bytes]) -> str:
    """
    功能：解码子进程输出字节。 

    Decode subprocess output bytes with codec fallback.

    Args:
        data: Raw subprocess output bytes.

    Returns:
        Decoded text.
    """
    if data is None:
        return ""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes or None")

    for codec in ("utf-8", "gbk", "cp936", "latin-1"):
        try:
            return bytes(data).decode(codec, errors="strict")
        except Exception:
            continue
    return bytes(data).decode("utf-8", errors="replace")


def run_smoke_pytest(repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行新 impl smoke pytest。 

    Run fixed smoke pytest for new runtime impl IDs.

    Args:
        repo_root: Repository root directory.

    Returns:
        Structured execution result.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    test_file = repo_root / SMOKE_TEST_PATH
    if not test_file.exists():
        return {
            "status": "na",
            "exit_code": 0,
            "command": f"pytest -q {SMOKE_TEST_PATH}",
            "stdout": "",
            "stderr": "",
            "reason": "smoke_test_missing",
        }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(test_file)],
            cwd=str(repo_root),
            capture_output=True,
            text=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "exit_code": 1,
            "command": f"pytest -q {SMOKE_TEST_PATH}",
            "stdout": "",
            "stderr": "pytest timeout (180s)",
            "reason": "timeout",
        }

    stdout_text = _decode_output(result.stdout)
    stderr_text = _decode_output(result.stderr)
    status = "pass" if result.returncode == 0 else "fail"

    return {
        "status": status,
        "exit_code": result.returncode,
        "command": f"pytest -q {SMOKE_TEST_PATH}",
        "stdout": stdout_text,
        "stderr": stderr_text,
        "reason": "<absent>" if status == "pass" else "pytest_failed",
    }


def main(repo_root_str: Optional[str] = None) -> int:
    """
    功能：运行新 impl runtime smoke 审计入口。 

    Execute runtime smoke audit for new saliency/subspace impl IDs.

    Args:
        repo_root_str: Optional repository root path string.

    Returns:
        Exit code (0 pass, 1 fail).
    """
    repo_root = Path(repo_root_str) if isinstance(repo_root_str, str) and repo_root_str else Path.cwd()
    execution = run_smoke_pytest(repo_root)

    if execution["status"] == "na":
        result = {
            "audit_id": "audit_runtime_impl_smoke_new_saliency_and_subspace",
            "gate_name": "gate.runtime_impl_smoke_new_saliency_and_subspace",
            "category": "G",
            "severity": "NON_BLOCK",
            "result": "N.A.",
            "rule": "new impl runtime smoke test file is absent",
            "evidence": {
                "test_path": SMOKE_TEST_PATH,
                "status": execution["status"],
                "reason": execution["reason"],
            },
            "impact": "cannot verify runtime smoke coverage for new impl ids",
            "fix": f"add {SMOKE_TEST_PATH} and ensure pytest can execute it",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if execution["status"] == "pass":
        result = {
            "audit_id": "audit_runtime_impl_smoke_new_saliency_and_subspace",
            "gate_name": "gate.runtime_impl_smoke_new_saliency_and_subspace",
            "category": "G",
            "severity": "NON_BLOCK",
            "result": "PASS",
            "rule": "new impl runtime smoke tests are executable in audit entry",
            "evidence": {
                "test_path": SMOKE_TEST_PATH,
                "command": execution["command"],
                "status": execution["status"],
                "exit_code": execution["exit_code"],
                "stdout_tail": execution["stdout"][-500:],
                "stderr_tail": execution["stderr"][-500:],
            },
            "impact": "runtime resolver + whitelist + evidence anchors for new impl ids remain verifiable",
            "fix": "N.A.",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    result = {
        "audit_id": "audit_runtime_impl_smoke_new_saliency_and_subspace",
        "gate_name": "gate.runtime_impl_smoke_new_saliency_and_subspace",
        "category": "G",
        "severity": "NON_BLOCK",
        "result": "FAIL",
        "rule": "new impl runtime smoke tests failed in audit entry",
        "evidence": {
            "test_path": SMOKE_TEST_PATH,
            "command": execution["command"],
            "status": execution["status"],
            "exit_code": execution["exit_code"],
            "stdout_tail": execution["stdout"][-1000:],
            "stderr_tail": execution["stderr"][-1000:],
            "reason": execution["reason"],
        },
        "impact": "new impl ids may not be safely executable through audit entry",
        "fix": "fix failing smoke tests and keep runtime whitelist/registry/evidence anchors consistent",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    repo_root_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(repo_root_arg))
