"""
File purpose: Calibrate CLI 配置迁移提示端到端测试。
Module type: General module
"""

from __future__ import annotations

import sys

import pytest

from main.cli import run_calibrate as run_calibrate_module


def test_run_calibrate_main_prints_hint_for_paper_lf_ecc_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    功能：验证 run_calibrate CLI main 在门禁错误时输出迁移提示。

    Verify run_calibrate CLI main prints migration hint when paper LF ECC gate fails.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture for stdio.

    Returns:
        None.
    """

    def _raise_gate_error(*args, **kwargs):
        _ = args
        _ = kwargs
        raise ValueError(
            "paper_faithfulness requires watermark.lf.ecc='sparse_ldpc'; "
            "legacy int ecc is not allowed (got 3)"
        )

    monkeypatch.setattr(run_calibrate_module, "run_calibrate", _raise_gate_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_calibrate",
            "--out",
            "tmp/cli_test_calibrate_hint",
            "--config",
            "configs/default.yaml",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        run_calibrate_module.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "[Calibrate] [ERROR]" in captured.err
    assert "[Calibrate] [HINT]" in captured.err
    assert "watermark.lf.ecc" in captured.err
    assert "sparse_ldpc" in captured.err
