"""
File purpose: 验证 detect 侧 trajectory-only 评分路径（Phase 5：彻底移除 legacy path）。
Module type: General module

Tests that after full legacy path removal:
1. extract_lf_score_from_detect_latents and extract_hf_score_from_detect_latents
   no longer exist in detector_scoring (deleted, not just blocked).
2. resolve_detect_trajectory_latent_for_timestep is exact-only (no strict_mode param,
   no nearest-step fallback).
3. Empty cache and None cache return "absent_empty_cache".
4. Exact timestep miss returns "absent_exact_timestep_mismatch_*" (hard fail).
5. Exact timestep hit returns "ok_exact".
"""

from __future__ import annotations

import numpy as np
import inspect

from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache
from main.watermarking.content_chain import detector_scoring


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_cache_with_step(step_index: int) -> LatentTrajectoryCache:
    """Create a LatentTrajectoryCache with a single captured step."""
    cache = LatentTrajectoryCache()
    cache.capture(step_index, np.ones(8, dtype=np.float32))
    return cache


# ---------------------------------------------------------------------------
# 场景 1：验证 legacy 函数已被彻底删除（不存在于模块中）
# ---------------------------------------------------------------------------

def test_legacy_lf_detect_latents_function_removed() -> None:
    """
    extract_lf_score_from_detect_latents must be completely removed from detector_scoring.
    It must not exist as an attribute; Phase 5 deletes it (not just blocks it).
    """
    assert not hasattr(detector_scoring, "extract_lf_score_from_detect_latents"), (
        "extract_lf_score_from_detect_latents still exists in detector_scoring — should be deleted"
    )


def test_legacy_hf_detect_latents_function_removed() -> None:
    """
    extract_hf_score_from_detect_latents must be completely removed from detector_scoring.
    Phase 5 deletes it entirely.
    """
    assert not hasattr(detector_scoring, "extract_hf_score_from_detect_latents"), (
        "extract_hf_score_from_detect_latents still exists in detector_scoring — should be deleted"
    )


def test_legacy_latents_path_label_constant_removed() -> None:
    """_LATENTS_PATH_LEGACY_LABEL constant must no longer exist in detector_scoring."""
    assert not hasattr(detector_scoring, "_LATENTS_PATH_LEGACY_LABEL"), (
        "_LATENTS_PATH_LEGACY_LABEL constant still exists — should be deleted"
    )


# ---------------------------------------------------------------------------
# 场景 2：resolve_detect_trajectory_latent_for_timestep 无 strict_mode 参数
# ---------------------------------------------------------------------------

def test_resolve_has_no_strict_mode_param() -> None:
    """
    resolve_detect_trajectory_latent_for_timestep must not accept a strict_mode param.
    It is exact-only; strict_mode concept is removed.
    """
    sig = inspect.signature(detector_scoring.resolve_detect_trajectory_latent_for_timestep)
    assert "strict_mode" not in sig.parameters, (
        f"strict_mode still present in signature: {list(sig.parameters)}"
    )


def test_extract_lf_trajectory_has_no_strict_mode_param() -> None:
    """extract_lf_score_from_detect_trajectory must not accept strict_mode."""
    sig = inspect.signature(detector_scoring.extract_lf_score_from_detect_trajectory)
    assert "strict_mode" not in sig.parameters, (
        f"strict_mode still present in LF trajectory signature: {list(sig.parameters)}"
    )


def test_extract_hf_trajectory_has_no_strict_mode_param() -> None:
    """extract_hf_score_from_detect_trajectory must not accept strict_mode."""
    sig = inspect.signature(detector_scoring.extract_hf_score_from_detect_trajectory)
    assert "strict_mode" not in sig.parameters, (
        f"strict_mode still present in HF trajectory signature: {list(sig.parameters)}"
    )


# ---------------------------------------------------------------------------
# 场景 3：trajectory cache 缺失 → 显式失败（empty / None cache）
# ---------------------------------------------------------------------------

def test_empty_cache_returns_absent_empty_cache() -> None:
    """
    Resolving trajectory latent from an empty cache must return 'absent_empty_cache'.
    No fallback path exists.
    """
    empty_cache = LatentTrajectoryCache()
    z_t, status = detector_scoring.resolve_detect_trajectory_latent_for_timestep(
        trajectory_cache=empty_cache,
        edit_timestep=5,
    )
    assert z_t is None
    assert status == "absent_empty_cache", f"Expected absent_empty_cache, got: {status!r}"


def test_none_cache_returns_absent_empty_cache() -> None:
    """Resolving trajectory latent from None cache returns absent_empty_cache."""
    z_t, status = detector_scoring.resolve_detect_trajectory_latent_for_timestep(
        trajectory_cache=None,
        edit_timestep=3,
    )
    assert z_t is None
    assert status == "absent_empty_cache", f"Expected absent_empty_cache, got: {status!r}"


# ---------------------------------------------------------------------------
# 场景 4：exact timestep 缺失 → 硬失败，无 nearest-step 回退
# ---------------------------------------------------------------------------

def test_exact_miss_fails_with_absent_mismatch_status() -> None:
    """
    When the exact edit_timestep is absent from the cache, resolution must fail with
    'absent_exact_timestep_mismatch_*' status. No nearest-step fallback exists.
    """
    # Cache contains step 3; edit_timestep=2 is absent.
    cache = _make_cache_with_step(step_index=3)
    z_t, status = detector_scoring.resolve_detect_trajectory_latent_for_timestep(
        trajectory_cache=cache,
        edit_timestep=2,
    )
    assert z_t is None, f"z_t must be None when exact step misses"
    assert status.startswith("absent_exact_timestep_mismatch"), (
        f"Expected absent_exact_timestep_mismatch prefix, got: {status!r}"
    )
    # 确认 available steps 信息包含在 status 中（可审计）。
    assert "available" in status, f"Status must contain available steps info: {status!r}"


def test_nearest_step_no_longer_returned() -> None:
    """
    nearest_step must never appear in any resolution status.
    This validates that the fallback mechanism is fully removed.
    """
    for edit_step in (0, 1, 5, 10):
        cache = _make_cache_with_step(step_index=3)
        _z_t, status = detector_scoring.resolve_detect_trajectory_latent_for_timestep(
            trajectory_cache=cache,
            edit_timestep=edit_step,
        )
        if edit_step != 3:
            assert "nearest_step" not in status, (
                f"nearest_step must not appear for edit_step={edit_step}, got: {status!r}"
            )


# ---------------------------------------------------------------------------
# 场景 5：exact timestep 命中 → ok_exact
# ---------------------------------------------------------------------------

def test_exact_hit_returns_ok_exact() -> None:
    """
    When the exact edit_timestep is present in the cache, resolution must succeed
    with 'ok_exact' status.
    """
    cache = _make_cache_with_step(step_index=5)
    z_t, status = detector_scoring.resolve_detect_trajectory_latent_for_timestep(
        trajectory_cache=cache,
        edit_timestep=5,
    )
    assert z_t is not None, "Exact hit must return the latent"
    assert status == "ok_exact", f"Expected ok_exact, got: {status!r}"
