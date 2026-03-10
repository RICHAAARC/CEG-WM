"""
功能：验证辅锚点仅在主同步成功后启用的硬门控

Module type: Core innovation module

当 sync_primary_anchor_secondary 模式启用时（paper 路径），
attention 辅锚点必须在主同步失败时被门控拒绝，防止产出
"稳定但不可信"的伪几何证据。
"""

from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np
import pytest

from main.watermarking.detect import orchestrator as detect_orchestrator
from main.watermarking.detect.orchestrator import BuiltImplSet


class _GeometryStub:
    """辅锚点 stub：始终返回 ok"""

    def extract(self, cfg: Dict[str, Any], inputs: Any = None) -> Dict[str, Any]:
        _ = cfg
        _ = inputs
        return {
            "status": "ok",
            "geo_score": None,
            "relation_digest": "r" * 64,
            "anchor_digest": "a" * 64,
        }


class _SyncOkStub:
    """主同步 stub：返回 ok"""

    def sync_with_context(
        self,
        cfg: Dict[str, Any],
        context: Any,
        runtime_inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        _ = cfg
        _ = context
        _ = runtime_inputs
        return {
            "status": "ok",
            "sync_status": "ok",
            "sync_digest": "s" * 64,
            "sync_quality_metrics": {"quality_score": 0.9},
        }


class _SyncFailStub:
    """主同步 stub：返回 failed"""

    def sync_with_context(
        self,
        cfg: Dict[str, Any],
        context: Any,
        runtime_inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        _ = cfg
        _ = context
        _ = runtime_inputs
        return {
            "status": "failed",
            "sync_status": "failed",
            "geometry_failure_reason": "sync_quality_below_threshold",
        }


def _build_paper_cfg(latents: np.ndarray) -> Dict[str, Any]:
    """构造启用 sync_primary_anchor_secondary 的 paper 模式配置"""
    _ = latents  # latents 不再使用
    return {
        "__detect_pipeline_obj__": object(),
        "paper_faithfulness": {"enabled": True},
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "sync_primary_anchor_secondary": True,
            }
        },
    }


def test_anchor_gated_when_sync_fails_under_paper_mode() -> None:
    """
    功能：paper 模式下 sync 失败时辅锚点必须被门控拒绝。

    When sync_primary_anchor_secondary is enabled and sync fails,
    anchor must be gated out with absent status and explicit gate detail.

    Args:
        None.

    Returns:
        None.
    """
    latents = np.random.default_rng(20260228).normal(size=(1, 4, 16, 16)).astype(np.float32)
    cfg = _build_paper_cfg(latents)
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=_SyncFailStub(),
    )

    run_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    result = run_chain(impl_set, cfg)
    result = cast(Dict[str, Any], result)

    # 辅锚点应被门控拒绝
    assert result.get("anchor_gated_by_sync") is True
    assert result.get("anchor_status") == "absent"
    assert result.get("anchor_result", {}).get("geometry_absent_reason") == "anchor_gated_by_sync_failure"

    # 整体几何证据状态应由 sync 主导（failed）
    assert result.get("status") == "failed"
    assert result.get("geo_score") is None


def test_anchor_allowed_when_sync_succeeds_under_paper_mode() -> None:
    """
    功能：paper 模式下 sync 成功时辅锚点应正常执行。

    When sync succeeds, anchor extraction should proceed normally.

    Args:
        None.

    Returns:
        None.
    """
    latents = np.random.default_rng(20260228).normal(size=(1, 4, 16, 16)).astype(np.float32)
    cfg = _build_paper_cfg(latents)
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=_SyncOkStub(),
    )

    run_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    result = run_chain(impl_set, cfg)
    result = cast(Dict[str, Any], result)

    # 辅锚点应正常执行
    assert result.get("anchor_gated_by_sync") is False
    # anchor_status 来自 stub，应为 ok
    assert result.get("anchor_status") == "ok"
