"""
File purpose: ExperimentMatrix CLI 配置迁移提示端到端测试。
Module type: General module
"""

from __future__ import annotations

import sys

import pytest

from main.cli import run_experiment_matrix as run_experiment_matrix_module


def test_run_experiment_matrix_main_prints_hint_for_paper_lf_ecc_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    功能：验证 run_experiment_matrix CLI main 在门禁错误时输出迁移提示。

    Verify run_experiment_matrix CLI main prints migration hint
    when paper LF ECC gate fails.

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

    monkeypatch.setattr(
        run_experiment_matrix_module,
        "run_experiment_matrix",
        _raise_gate_error,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_matrix",
            "--config",
            "configs/default.yaml",
        ],
    )

    with pytest.raises(ValueError):
        run_experiment_matrix_module.main()

    captured = capsys.readouterr()
    assert "[ExperimentMatrix] [ERROR]" in captured.err
    assert "[ExperimentMatrix] [HINT]" in captured.err
    assert "watermark.lf.ecc" in captured.err
    assert "sparse_ldpc" in captured.err
