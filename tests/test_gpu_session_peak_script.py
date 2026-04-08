"""
文件目的：验证 GPU 会话峰值包装脚本的基础 CLI 与透传合同。
Module type: General module

职责边界：
1. 覆盖 gpu_session_peak.py 的 help、stdout/stderr 透传、JSON 写出与返回码保持语义。
2. 验证 nvidia-smi 缺失时仅降级为 absent，不阻断被包装命令执行。
3. 不依赖真实 GPU 或真实 nvidia-smi，仅通过 monkeypatch 控制观测层输入。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from scripts import gpu_session_peak
from scripts.notebook_runtime_common import build_repo_import_subprocess_env


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "gpu_session_peak.py"


def test_gpu_session_peak_script_supports_help() -> None:
    """
    功能：验证 GPU peak wrapper 暴露 help 入口。

    Verify that the GPU session peak wrapper exposes a working help entrypoint.

    Args:
        None.

    Returns:
        None.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        env=build_repo_import_subprocess_env(repo_root=REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_gpu_session_peak_wrapper_passthrough_writes_json_and_preserves_return_code(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """
    功能：验证 wrapper 透传业务输出、写出 JSON 并保持退出码。

    Verify that the wrapper forwards the wrapped command output, writes the
    JSON summary, and returns the wrapped command exit code.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture.

    Returns:
        None.
    """
    output_json_path = tmp_path / "gpu_session_peak.json"
    wrapped_command = [sys.executable, "fake_command.py", "--flag"]
    wrapped_stdout = '{"business":"stdout"}\n'
    wrapped_stderr = "business-stderr\n"
    nvidia_smi_path = "C:/Windows/System32/nvidia-smi.exe"
    sample_outputs: List[str] = [
        "0, GPU-uuid-0, Fake GPU, 1024, 24576\n",
        "0, GPU-uuid-0, Fake GPU, 4096, 24576\n",
        "0, GPU-uuid-0, Fake GPU, 2048, 24576\n",
    ]
    sample_index = {"value": 0}

    def fake_run(command, check=False, capture_output=False, text=False, encoding=None, errors=None):
        command_list = [str(item) for item in command]
        if command_list[0] == nvidia_smi_path:
            response_index = min(sample_index["value"], len(sample_outputs) - 1)
            sample_index["value"] += 1
            return subprocess.CompletedProcess(command_list, 0, stdout=sample_outputs[response_index], stderr="")

        assert command_list == wrapped_command
        time.sleep(0.03)
        sys.stdout.write(wrapped_stdout)
        sys.stderr.write(wrapped_stderr)
        return subprocess.CompletedProcess(command_list, 7, stdout=None, stderr=None)

    monkeypatch.setattr(gpu_session_peak.shutil, "which", lambda _: nvidia_smi_path)
    monkeypatch.setattr(gpu_session_peak.subprocess, "run", fake_run)

    exit_code = gpu_session_peak.main(
        [
            "--output-json",
            str(output_json_path),
            "--label",
            "PW01_Source_Event_Shards",
            "--sample-interval-ms",
            "10",
            "--",
            *wrapped_command,
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(output_json_path.read_text(encoding="utf-8"))

    assert exit_code == 7
    assert captured.out == wrapped_stdout
    assert captured.err == wrapped_stderr
    assert payload["status"] == "ok"
    assert payload["label"] == "PW01_Source_Event_Shards"
    assert payload["monitor_source"] == "nvidia-smi"
    assert payload["nvidia_smi_available"] is True
    assert payload["nvidia_smi_path"] == nvidia_smi_path
    assert payload["sample_interval_ms"] == 10
    assert payload["sample_count"] >= 2
    assert payload["wrapped_command"] == wrapped_command
    assert payload["wrapped_return_code"] == 7
    assert payload["visible_gpu_count"] == 1
    assert payload["visible_gpus"] == [
        {
            "index": 0,
            "uuid": "GPU-uuid-0",
            "name": "Fake GPU",
            "memory_total_mib": 24576,
            "peak_memory_used_mib": 4096,
        }
    ]
    assert payload["session_board_memory_used_mib_at_start"] == 1024
    assert payload["session_board_memory_used_mib_at_end"] == 2048
    assert payload["session_board_peak_memory_used_mib"] == 4096
    assert payload["session_board_peak_memory_used_bytes"] == 4096 * 1024 * 1024
    assert payload["session_board_peak_increment_mib"] == 3072
    assert payload["peak_gpu_index"] == 0
    assert payload["peak_gpu_uuid"] == "GPU-uuid-0"
    assert payload["peak_gpu_name"] == "Fake GPU"
    assert payload["sampling_error_count"] == 0


def test_gpu_session_peak_wrapper_marks_absent_when_nvidia_smi_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """
    功能：验证 nvidia-smi 缺失时 wrapper 不阻断业务命令。

    Verify that the wrapper keeps executing the wrapped command and emits an
    absent monitor summary when nvidia-smi is unavailable.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture.

    Returns:
        None.
    """
    output_json_path = tmp_path / "gpu_session_peak_absent.json"
    wrapped_command = [sys.executable, "fake_command.py"]
    wrapped_stdout = '{"status":"wrapped-ok"}\n'
    wrapped_stderr = "wrapped-stderr\n"

    def fake_run(command, check=False, capture_output=False, text=False, encoding=None, errors=None):
        command_list = [str(item) for item in command]
        assert command_list == wrapped_command
        sys.stdout.write(wrapped_stdout)
        sys.stderr.write(wrapped_stderr)
        return subprocess.CompletedProcess(command_list, 0, stdout=None, stderr=None)

    monkeypatch.setattr(gpu_session_peak.shutil, "which", lambda _: None)
    monkeypatch.setattr(gpu_session_peak.subprocess, "run", fake_run)

    exit_code = gpu_session_peak.main(
        [
            "--output-json",
            str(output_json_path),
            "--label",
            "PW01_Source_Event_Shards",
            "--sample-interval-ms",
            "20",
            "--",
            *wrapped_command,
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(output_json_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured.out == wrapped_stdout
    assert captured.err == wrapped_stderr
    assert payload["status"] == "absent"
    assert payload["nvidia_smi_available"] is False
    assert payload["nvidia_smi_path"] is None
    assert payload["sample_count"] == 0
    assert payload["visible_gpu_count"] == 0
    assert payload["visible_gpus"] == []
    assert payload["wrapped_command"] == wrapped_command
    assert payload["wrapped_return_code"] == 0
    assert payload["session_board_peak_memory_used_mib"] is None
    assert payload["sampling_error_count"] == 0
