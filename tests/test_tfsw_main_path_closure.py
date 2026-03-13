"""
测试用例：TFSW 主路径收口（trajectory-consistent LF 检测 + planner exact-only）。

功能说明：
- 验证 sparse_ldpc 唯一路径：trajectory cache z_{t_e} → φ → PRC 解码，无 legacy latent 分支。
- 验证 planner exact-only 约束：缓存缺失或时步不命中均 raise ValueError（无最近时步兜底）。
- 验证 planner 无 latents 时 raise ValueError（无 fallback 降级）。

Module type: General module
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any, Dict, cast

from main.watermarking.detect import orchestrator as detect_orchestrator
from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SubspacePlannerImpl,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
    build_runtime_jvp_operator_from_cache,
)
from main.core import digests


# ---------------------------------------------------------------------------
# 共享测试 fixture 工具
# ---------------------------------------------------------------------------

def _make_trajectory_cache(step_latents: Dict[int, Any]) -> LatentTrajectoryCache:
    """构造包含指定时步数据的 LatentTrajectoryCache。"""
    cache = LatentTrajectoryCache()
    for step_idx, latent in step_latents.items():
        cache.capture(step_idx, latent)
    return cache


def _make_lf_basis(feature_dim: int = 4, edit_timestep: int = 0, projection_seed: int = 0) -> Dict[str, Any]:
    """构造满足 paper 模式 TFSW 路径的最小 lf_basis。"""
    return {
        "trajectory_feature_spec": {
            "feature_operator": "masked_normalized_random_projection",
            "feature_dim": feature_dim,
            "projection_seed": projection_seed,
            "edit_timestep": edit_timestep,
        }
    }


def _minimal_planner_cfg(
    feature_dim: int = 4,
    sample_count: int = 4,
    timestep_end: int = 3,
) -> Dict[str, Any]:
    """构造最小化 planner 配置（始终启用 paper_faithfulness）。"""
    cfg: Dict[str, Any] = {
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 2,
                "sample_count": sample_count,
                "feature_dim": feature_dim,
                "seed": 0,
                "timestep_start": 0,
                "timestep_end": timestep_end,
                "trajectory_step_stride": 1,
                "edit_timestep": 0,
                "num_inference_steps": timestep_end + 1,
            }
        },
        "paper_faithfulness": {"enabled": True},
    }
    return cfg


class _FakePipeline:
    """仅存在作为 has_pipeline 路由触发器，不执行任何真实推理。"""
    transformer = None


# ---------------------------------------------------------------------------
# 测试一：orchestrator LF 路径选择（Problem A：主路径收口）
# ---------------------------------------------------------------------------

def test_lf_uses_trajectory_path_label() -> None:
    """
    功能： paper 模式下 LF 检测路径标签必须为 low_freq_template_trajectory。

    Verify that paper_faithfulness.enabled=True routes LF scoring through
    trajectory cache path ("low_freq_template_trajectory"), using exact trajectory cache hit.

    Args:
        None.

    Returns:
        None.
    """
    # z_t shape (1, 1, 2, 3) = 6 elements >= feature_dim=4
    z_t = np.ones((1, 1, 2, 3), dtype=np.float32)
    cache = _make_trajectory_cache({0: z_t})
    lf_basis = _make_lf_basis(feature_dim=4, edit_timestep=0, projection_seed=42)

    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "key_id": "k1",
            "pattern_id": "p1",
            "lf": {"enabled": True, "ecc": "sparse_ldpc"},
            "hf": {"enabled": False},
        },
        "__detect_trajectory_latent_cache__": cache,
    }

    _, _, traces = detect_orchestrator._extract_content_raw_scores_from_image(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record=None,
        plan_payload={"plan": {"lf_basis": lf_basis, "band_spec": {}}},
        plan_digest="test_plan_digest",
        cfg_digest="test_cfg_digest",
    )

    lf_trace = cast(Dict[str, Any], traces.get("lf", {}))
    assert lf_trace.get("lf_detect_path") == "low_freq_template_trajectory"
    assert lf_trace.get("lf_status") in {"ok", "failed", "absent"}


def test_lf_absent_when_trajectory_cache_absent() -> None:
    """
    功能：trajectory cache 为 None 时 LF 状态应为 absent。

    Verify that absent trajectory cache results in
    lf_status="absent" with reason "trajectory_cache_absent".

    Args:
        None.

    Returns:
        None.
    """
    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "key_id": "k1",
            "pattern_id": "p1",
            "lf": {"enabled": True, "ecc": "sparse_ldpc"},
            "hf": {"enabled": False},
        },
    }
    lf_basis = _make_lf_basis()

    _, _, traces = detect_orchestrator._extract_content_raw_scores_from_image(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record=None,
        plan_payload={"plan": {"lf_basis": lf_basis, "band_spec": {}}},
        plan_digest="test_plan_digest",
        cfg_digest="test_cfg_digest",
    )

    lf_trace = cast(Dict[str, Any], traces.get("lf", {}))
    assert lf_trace.get("lf_detect_path") == "low_freq_template_trajectory"
    assert lf_trace.get("lf_status") == "absent"
    assert lf_trace.get("lf_absent_reason") == "trajectory_cache_absent"


def test_lf_absent_when_trajectory_cache_empty() -> None:
    """
    功能：trajectory cache 为空时 LF 状态应为 absent。

    Verify that an empty (captured nothing) trajectory cache results in
    lf_status="absent" with reason "trajectory_cache_absent".

    Args:
        None.

    Returns:
        None.
    """
    cache = LatentTrajectoryCache()  # 空缓存
    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "key_id": "k1",
            "pattern_id": "p1",
            "lf": {"enabled": True, "ecc": "sparse_ldpc"},
            "hf": {"enabled": False},
        },
        "__detect_trajectory_latent_cache__": cache,
    }
    lf_basis = _make_lf_basis()

    _, _, traces = detect_orchestrator._extract_content_raw_scores_from_image(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record=None,
        plan_payload={"plan": {"lf_basis": lf_basis, "band_spec": {}}},
        plan_digest="test_plan_digest",
        cfg_digest="test_cfg_digest",
    )

    lf_trace = cast(Dict[str, Any], traces.get("lf", {}))
    assert lf_trace.get("lf_detect_path") == "low_freq_template_trajectory"
    assert lf_trace.get("lf_status") == "absent"
    assert lf_trace.get("lf_absent_reason") == "trajectory_cache_absent"


def test_lf_absent_when_lf_basis_missing() -> None:
    """
    功能：plan 中缺少 lf_basis 时 LF 状态应为 absent。

    Verify that missing lf_basis in plan_payload results in
    lf_status="absent" with reason "lf_basis_missing_for_trajectory_path".

    Args:
        None.

    Returns:
        None.
    """
    z_t = np.ones((4,), dtype=np.float32)
    cache = _make_trajectory_cache({0: z_t})
    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "key_id": "k1",
            "pattern_id": "p1",
            "lf": {"enabled": True, "ecc": "sparse_ldpc"},
            "hf": {"enabled": False},
        },
        "__detect_trajectory_latent_cache__": cache,
    }

    _, _, traces = detect_orchestrator._extract_content_raw_scores_from_image(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record=None,
        plan_payload={"plan": {"band_spec": {}}},  # lf_basis 缺失
        plan_digest="test_plan_digest",
        cfg_digest="test_cfg_digest",
    )

    lf_trace = cast(Dict[str, Any], traces.get("lf", {}))
    assert lf_trace.get("lf_detect_path") == "low_freq_template_trajectory"
    assert lf_trace.get("lf_status") == "absent"
    assert lf_trace.get("lf_absent_reason") == "lf_basis_missing_for_trajectory_path"


def test_lf_absent_on_exact_timestep_mismatch() -> None:
    """
    功能：paper 模式下 edit_timestep 精确不命中时 LF 状态应为 absent。

    Verify that a trajectory cache without the required edit_timestep
    results in lf_status="absent" with trajectory_latent_absent reason.

    Args:
        None.

    Returns:
        None.
    """
    # cache 有 step=5，但 edit_timestep=0 需要 step=0
    z_t = np.ones((4,), dtype=np.float32)
    cache = _make_trajectory_cache({5: z_t})
    lf_basis = _make_lf_basis(edit_timestep=0)  # 需要 step=0

    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "key_id": "k1",
            "pattern_id": "p1",
            "lf": {"enabled": True, "ecc": "sparse_ldpc"},
            "hf": {"enabled": False},
        },
        "__detect_trajectory_latent_cache__": cache,
    }

    _, _, traces = detect_orchestrator._extract_content_raw_scores_from_image(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record=None,
        plan_payload={"plan": {"lf_basis": lf_basis, "band_spec": {}}},
        plan_digest="test_plan_digest",
        cfg_digest="test_cfg_digest",
    )

    lf_trace = cast(Dict[str, Any], traces.get("lf", {}))
    assert lf_trace.get("lf_detect_path") == "low_freq_template_trajectory"
    assert lf_trace.get("lf_status") == "absent"
    absent_reason = lf_trace.get("lf_absent_reason", "")
    assert "trajectory_latent_absent" in absent_reason


# ---------------------------------------------------------------------------
# 测试二：planner exact-only（无最近时步兜底，无 fallback 降级）
# ---------------------------------------------------------------------------

def _make_planner() -> SubspacePlannerImpl:
    return SubspacePlannerImpl(
        impl_id=SUBSPACE_PLANNER_ID,
        impl_version=SUBSPACE_PLANNER_VERSION,
        impl_digest=digests.canonical_sha256({
            "impl_id": SUBSPACE_PLANNER_ID,
            "impl_version": SUBSPACE_PLANNER_VERSION,
        }),
    )


def test_planner_raises_when_cache_absent() -> None:
    """
    功能：trajectory cache 为 None 时 planner 应 raise ValueError。

    Verify that SubspacePlannerImpl.plan() raises ValueError (propagated as
    plan_failure_reason="decomposition_failed") when no trajectory_latent_cache
    is provided.

    Args:
        None.

    Returns:
        None.
    """
    planner = _make_planner()
    cfg = _minimal_planner_cfg(feature_dim=4, sample_count=4, timestep_end=3)

    fake_pipeline = _FakePipeline()

    inputs: Dict[str, Any] = {
        "pipeline": fake_pipeline,
        "trace_signature": {
            "num_inference_steps": 4,
            "guidance_scale": 7.0,
            "height": 64,
            "width": 64,
        },
        # trajectory_latent_cache 故意不提供
    }

    result = planner.plan(cfg, mask_digest="m1", cfg_digest="c1", inputs=inputs)
    # 计划应以 failed 状态结束（ValueError 被捕获）
    assert result.status == "failed"
    plan_dict = result.plan if isinstance(result.plan, dict) else {}
    plan_failure = plan_dict.get("plan_failure_reason", "")
    assert "decomposition_failed" in plan_failure or result.plan_digest is None


def test_planner_raises_on_exact_timestep_miss() -> None:
    """
    功能：trajectory cache 时步精确不命中时 planner 应 raise ValueError（无最近时步兜底）。

    Verify that SubspacePlannerImpl.plan() raises ValueError when the trajectory
    cache has a different timestep than required (no nearest-step fallback allowed).

    Args:
        None.

    Returns:
        None.
    """
    planner = _make_planner()
    cfg = _minimal_planner_cfg(feature_dim=4, sample_count=4, timestep_end=3)

    # 构造 cache：仅有 step=100，不在规划器的 [0,1,2,3] 范围内
    z_t = np.ones((1, 1, 2, 3), dtype=np.float32)
    cache = _make_trajectory_cache({100: z_t})

    fake_pipeline = _FakePipeline()

    inputs: Dict[str, Any] = {
        "pipeline": fake_pipeline,
        "trajectory_latent_cache": cache,
        "jvp_operator": build_runtime_jvp_operator_from_cache(cfg, cache),
        "trace_signature": {
            "num_inference_steps": 4,
            "guidance_scale": 7.0,
            "height": 64,
            "width": 64,
        },
    }

    result = planner.plan(cfg, mask_digest="m1", cfg_digest="c1", inputs=inputs)
    # 精确不命中 → ValueError → plan failed
    assert result.status == "failed"
    plan_dict = result.plan if isinstance(result.plan, dict) else {}
    plan_failure = plan_dict.get("plan_failure_reason", "")
    assert "decomposition_failed" in plan_failure or result.plan_digest is None


def test_planner_succeeds_on_exact_cache_hit() -> None:
    """
    功能：所有时步均精确命中 trajectory cache 时 planner 应返回 ok。

    Verify that SubspacePlannerImpl.plan() returns status="ok" when the
    trajectory cache contains all required timesteps (exact hit).

    Args:
        None.

    Returns:
        None.
    """
    planner = _make_planner()
    # timestep_end=3, sample_count=4 → 时步序列 = [0, 1, 2, 3]
    cfg = _minimal_planner_cfg(feature_dim=4, sample_count=4, timestep_end=3)

    # 使用多样化 z_t，避免归一化后全零 → SVD 秩不足。
    rng = np.random.default_rng(42)
    cache = _make_trajectory_cache(
        {i: rng.standard_normal((1, 1, 2, 3)).astype(np.float32) for i in range(4)}
    )

    fake_pipeline = _FakePipeline()

    inputs: Dict[str, Any] = {
        "pipeline": fake_pipeline,
        "trajectory_latent_cache": cache,
        "jvp_operator": build_runtime_jvp_operator_from_cache(cfg, cache),
        "trace_signature": {
            "num_inference_steps": 4,
            "guidance_scale": 7.0,
            "height": 64,
            "width": 64,
        },
    }

    result = planner.plan(cfg, mask_digest="m1", cfg_digest="c1", inputs=inputs)
    assert result.status == "ok"
    assert result.plan_digest is not None
