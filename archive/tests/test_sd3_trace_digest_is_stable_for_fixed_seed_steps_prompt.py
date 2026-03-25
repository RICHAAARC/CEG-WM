"""
File purpose: 测试 SD3 trajectory digest 在固定种子、步数、prompt 下的稳定性。
Module type: Core innovation module

Test that SD3 trajectory digest is stable and reproducible for the same
seed/steps/prompt configuration.
"""

import pytest
from main.diffusion.sd3 import trajectory_tap
from main.core import digests


def test_sd3_trace_digest_is_stable_for_fixed_seed_steps_prompt():
    """
    功能：测试固定 seed/steps/prompt 下 trajectory digest 的稳定性。

    Test that trajectory_digest is deterministic when all sampling parameters
    (seed, num_inference_steps, scheduler config, etc.) are fixed.

    Args:
        None.

    Returns:
        None.
    """
    # 构造最小 cfg（固定 seed、steps、prompt）
    cfg = {
        "inference_enabled": True,
        "inference_num_steps": 10,
        "inference_prompt": "a photo of a dog",
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
        "trajectory_tap_enabled": True,
        "trajectory_sample_stride": 2  # 每 2 步采样一次
    }
    
    inference_runtime_meta = {
        "num_inference_steps": 10,
        "scheduler_class_name": "FlowMatchEulerDiscreteScheduler",
        "scheduler_num_train_timesteps": 1000
    }
    
    seed = 42
    device = "cpu"
    
    # 构造 mock tap_steps（量化统计，不含原始 tensor）
    tap_steps = [
        {
            "step_index": 0,
            "scheduler_step": 0,
            "stats": {
                "mean": 0.001234,
                "std": 0.123456,
                "l2_norm": 1.234567,
                "min": -0.5,
                "max": 0.5
            }
        },
        {
            "step_index": 2,
            "scheduler_step": 2,
            "stats": {
                "mean": 0.001235,
                "std": 0.123457,
                "l2_norm": 1.234568,
                "min": -0.5,
                "max": 0.5
            }
        },
        {
            "step_index": 4,
            "scheduler_step": 4,
            "stats": {
                "mean": 0.001236,
                "std": 0.123458,
                "l2_norm": 1.234569,
                "min": -0.5,
                "max": 0.5
            }
        },
        {
            "step_index": 6,
            "scheduler_step": 6,
            "stats": {
                "mean": 0.001237,
                "std": 0.123459,
                "l2_norm": 1.234570,
                "min": -0.5,
                "max": 0.5
            }
        },
        {
            "step_index": 8,
            "scheduler_step": 8,
            "stats": {
                "mean": 0.001238,
                "std": 0.123460,
                "l2_norm": 1.234571,
                "min": -0.5,
                "max": 0.5
            }
        }
    ]
    
    # 第一次调用 build_trajectory_evidence
    evidence_1 = trajectory_tap.build_trajectory_evidence(
        cfg=cfg,
        inference_status="ok",
        inference_runtime_meta=inference_runtime_meta,
        seed=seed,
        device=device,
        tap_steps=tap_steps
    )
    
    # 第二次调用 build_trajectory_evidence（相同参数）
    evidence_2 = trajectory_tap.build_trajectory_evidence(
        cfg=cfg,
        inference_status="ok",
        inference_runtime_meta=inference_runtime_meta,
        seed=seed,
        device=device,
        tap_steps=tap_steps
    )
    
    # 断言：trajectory_digest 必须一致（可复算性）
    digest_1 = evidence_1.get("trajectory_digest")
    digest_2 = evidence_2.get("trajectory_digest")
    
    assert digest_1 == digest_2, f"Expected same digest, got {digest_1} vs {digest_2}"
    
    # 断言：trajectory_spec_digest 必须一致
    spec_digest_1 = evidence_1.get("trajectory_spec_digest")
    spec_digest_2 = evidence_2.get("trajectory_spec_digest")
    
    assert spec_digest_1 == spec_digest_2, \
        f"Expected same spec_digest, got {spec_digest_1} vs {spec_digest_2}"
    
    # 断言：digest 是有效的 sha256（64 位小写十六进制）
    assert isinstance(digest_1, str), "trajectory_digest must be str"
    assert len(digest_1) == 64, f"Expected 64-char digest, got {len(digest_1)}"
    assert digest_1.islower(), "digest must be lowercase"
    assert all(c in "0123456789abcdef" for c in digest_1), "digest must be hex"


def test_sd3_trace_digest_changes_when_steps_change():
    """
    功能：测试改变 tap_steps 会导致 digest 变化。

    Test that changing the tap_steps (e.g., different latent norms)
    results in a different trajectory_digest.

    Args:
        None.

    Returns:
        None.
    """
    cfg = {
        "inference_enabled": True,
        "inference_num_steps": 10,
        "trajectory_tap_enabled": True,
        "trajectory_sample_stride": 2
    }
    
    inference_runtime_meta = {
        "num_inference_steps": 10,
        "scheduler_class_name": "FlowMatchEulerDiscreteScheduler",
        "scheduler_num_train_timesteps": 1000
    }
    
    seed = 42
    device = "cpu"
    
    # 第一组 tap_steps
    tap_steps_1 = [
        {
            "step_index": 0,
            "scheduler_step": 0,
            "stats": {
                "mean": 0.001234,
                "std": 0.123456,
                "l2_norm": 1.234567,
                "min": -0.5,
                "max": 0.5
            }
        },
        {
            "step_index": 2,
            "scheduler_step": 2,
            "stats": {
                "mean": 0.001235,
                "std": 0.123457,
                "l2_norm": 1.234568,
                "min": -0.5,
                "max": 0.5
            }
        }
    ]
    
    # 第二组 tap_steps（l2_norm 不同）
    tap_steps_2 = [
        {
            "step_index": 0,
            "scheduler_step": 0,
            "stats": {
                "mean": 0.001234,
                "std": 0.123456,
                "l2_norm": 9.999999,  # 不同的 l2_norm
                "min": -0.5,
                "max": 0.5
            }
        },
        {
            "step_index": 2,
            "scheduler_step": 2,
            "stats": {
                "mean": 0.001235,
                "std": 0.123457,
                "l2_norm": 9.999998,
                "min": -0.5,
                "max": 0.5
            }
        }
    ]
    
    # 调用 build_trajectory_evidence
    evidence_1 = trajectory_tap.build_trajectory_evidence(
        cfg=cfg,
        inference_status="ok",
        inference_runtime_meta=inference_runtime_meta,
        seed=seed,
        device=device,
        tap_steps=tap_steps_1
    )
    
    evidence_2 = trajectory_tap.build_trajectory_evidence(
        cfg=cfg,
        inference_status="ok",
        inference_runtime_meta=inference_runtime_meta,
        seed=seed,
        device=device,
        tap_steps=tap_steps_2
    )
    
    # 断言：trajectory_digest 必须不同（步长变化导致 digest 变化）
    digest_1 = evidence_1.get("trajectory_digest")
    digest_2 = evidence_2.get("trajectory_digest")
    
    assert digest_1 != digest_2, \
        f"Expected different digests, got {digest_1} vs {digest_2}"


def test_sd3_trace_digest_absent_when_tap_disabled():
    """
    功能：测试 trajectory_tap_enabled=false 时返回 absent 语义。

    Test that when trajectory_tap_enabled is false, the evidence
    has status='absent' and absent_reason='tap_disabled'.

    Args:
        None.

    Returns:
        None.
    """
    cfg = {
        "inference_enabled": True,
        "trajectory_tap": {"enabled": False}  # 显式禁用
    }
    
    inference_runtime_meta = {}
    seed = 42
    device = "cpu"
    
    evidence = trajectory_tap.build_trajectory_evidence(
        cfg=cfg,
        inference_status="ok",
        inference_runtime_meta=inference_runtime_meta,
        seed=seed,
        device=device
    )
    
    # 断言：status 必须为 absent
    assert evidence.get("status") == "absent", \
        f"Expected status='absent', got {evidence.get('status')}"
    
    # 断言：absent_reason 必须为 tap_disabled
    absent_reason = evidence.get("trajectory_absent_reason")
    assert absent_reason == "tap_disabled", \
        f"Expected absent_reason='tap_disabled', got {absent_reason}"
    
    # 断言：trajectory_digest 必须为 None 或 "<absent>"
    trajectory_digest = evidence.get("trajectory_digest")
    assert trajectory_digest in (None, "<absent>"), \
        f"Expected trajectory_digest to be None or '<absent>', got {trajectory_digest}"


def test_sd3_trace_digest_absent_when_inference_disabled():
    """
    功能：测试 inference_enabled=false 时返回 absent 语义。

    Test that when inference_enabled is false, the trajectory evidence
    has status='absent' and absent_reason='inference_disabled'.

    Args:
        None.

    Returns:
        None.
    """
    cfg = {
        "inference_enabled": True,  # 启用推理和 tap
        "trajectory_tap": {"enabled": True}
    }
    
    inference_runtime_meta = {}
    seed = 42
    device = "cpu"
    
    evidence = trajectory_tap.build_trajectory_evidence(
        cfg=cfg,
        inference_status="disabled",  # inference_status 为 disabled
        inference_runtime_meta=inference_runtime_meta,
        seed=seed,
        device=device
    )
    
    # 断言：status 必须为 absent
    assert evidence.get("status") == "absent", \
        f"Expected status='absent', got {evidence.get('status')}"
    
    # 断言：absent_reason 必须为 inference_disabled
    absent_reason = evidence.get("trajectory_absent_reason")
    assert absent_reason == "inference_disabled", \
        f"Expected absent_reason='inference_disabled', got {absent_reason}"

