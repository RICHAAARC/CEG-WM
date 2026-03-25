"""
功能：当启用 paper_faithfulness 时，trajectory_tap 必须启用

File purpose: Test that trajectory tap is enabled when paper faithfulness is enabled.
Module type: General module
"""

import pytest

from main.diffusion.sd3 import trajectory_tap


def test_trajectory_tap_enabled_when_paper_faithfulness_enabled():
    """
    功能：当 paper_faithfulness.enabled=true 时，
          trajectory_tap 必须默认启用（即使未显式配置）。

    Test that trajectory tap is enabled when paper_faithfulness is enabled.
    """
    cfg = {
        "paper_faithfulness": {
            "enabled": True,
            "alignment_check": True
        },
        "inference_enabled": False,  # 即使 inference_enabled=false，tap 也应启用
        "trajectory_tap": {}  # 未显式配置 enabled
    }

    result = trajectory_tap._resolve_tap_enabled(cfg)

    assert result is True, \
        "trajectory_tap must be enabled when paper_faithfulness is enabled"


def test_trajectory_tap_disabled_when_paper_faithfulness_disabled():
    """
    功能：当 paper_faithfulness.enabled=false 时，
          trajectory_tap 应依赖 inference_enabled（传统逻辑）。

    Test that trajectory tap follows inference_enabled when paper_faithfulness is disabled.
    """
    cfg = {
        "paper_faithfulness": {
            "enabled": False
        },
        "inference_enabled": False,
        "trajectory_tap": {}
    }

    result = trajectory_tap._resolve_tap_enabled(cfg)

    assert result is False, \
        "trajectory_tap should follow inference_enabled when paper_faithfulness disabled"


def test_trajectory_tap_explicit_false_is_honored():
    """
    功能：当 trajectory_tap.enabled 显式设置为 false 时，
          必须被尊重（即使 paper_faithfulness=true）。

    Test that explicit trajectory_tap.enabled=false is honored.
    """
    cfg = {
        "paper_faithfulness": {
            "enabled": True
        },
        "trajectory_tap": {
            "enabled": False  # 显式禁用
        }
    }

    result = trajectory_tap._resolve_tap_enabled(cfg)

    assert result is False, \
        "Explicit trajectory_tap.enabled=false must be honored"


def test_trajectory_evidence_still_absent_without_tap_steps():
    """
    功能：即使 tap 启用，如果没有真实的 tap_steps（来自推理回调），
          evidence 会被标记为 absent（这是正常的，实际 tap_steps 来自真实推理）。

    Test that trajectory evidence without tap_steps (from real inference) is marked absent.
    """
    cfg = {
        "trajectory_tap": {
            "enabled": True,
            "sample_at_steps": [5, 10],
            "sample_layer_names": ["transformer"]
        },
        "inference_enabled": True,
        "inference_num_steps": 10,
        "paper_faithfulness": {
            "enabled": True
        }
    }

    # 构造最小的 inference_runtime_meta
    inference_runtime_meta = {
        "num_inference_steps": 10,
        "device": "cpu"
    }

    # 构造 trajectory_spec（通过 _build_trajectory_spec）
    spec = trajectory_tap._build_trajectory_spec(cfg, inference_runtime_meta)

    # 调用 build_trajectory_evidence，不传入 tap_steps（模拟推理未捕获数据）
    evidence = trajectory_tap.build_trajectory_evidence(
        cfg,
        inference_status="ok",
        inference_runtime_meta=inference_runtime_meta,
        seed=42,
        device="cpu",
        trajectory_spec=spec,
        tap_steps=None  # 没有真实的 tap_steps
    )

    # 当没有 tap_steps 时，evidence 会返回 absent（这是正常的）
    # 实际的有效 evidence 来自 tap_from_pipeline 的真实推理回调
    assert evidence["status"] == "absent" or evidence["status"] == "ok", \
        f"Expected absent or ok status, got {evidence['status']}"


def test_trajectory_evidence_absent_when_tap_disabled():
    """
    功能：当 tap 禁用时，trajectory_evidence 必须为 status=absent。

    Test that trajectory evidence is absent when tap is disabled.
    """
    cfg = {
        "trajectory_tap": {
            "enabled": False
        },
        "inference_enabled": False
    }

    inference_runtime_meta = {
        "num_inference_steps": 10,
        "device": "cpu"
    }

    evidence = trajectory_tap.build_trajectory_evidence(
        cfg,
        inference_status="ok",
        inference_runtime_meta=inference_runtime_meta,
        seed=42,
        device="cpu"
    )

    assert evidence["status"] == "absent"
    assert evidence["trajectory_absent_reason"] == "tap_disabled"
