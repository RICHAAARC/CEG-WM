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

from main.core import digests
from main.watermarking.detect import orchestrator as detect_orchestrator
from main.watermarking.detect.orchestrator import BuiltImplSet
from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache
from main.watermarking.geometry_chain.attention_anchor_extractor import AttentionAnchorExtractor


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


class _SyncCaptureStub:
    """主同步 stub：捕获 runtime_inputs 并返回 ok"""

    def __init__(self) -> None:
        self.runtime_inputs: Dict[str, Any] | None = None

    def sync_with_context(
        self,
        cfg: Dict[str, Any],
        context: Any,
        runtime_inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        _ = cfg
        _ = context
        self.runtime_inputs = runtime_inputs
        relation_digest = runtime_inputs.get("relation_digest") if isinstance(runtime_inputs, dict) else None
        return {
            "status": "ok",
            "sync_status": "ok",
            "sync_digest": "s" * 64,
            "sync_quality_metrics": {"quality_score": 0.95},
            "relation_digest_bound": relation_digest,
            "geo_score": 0.95,
        }


class _SyncLegacyReasonStub:
    """主同步 stub：返回旧版缺失 reason，验证 detect path 会做语义归一化"""

    def __init__(self) -> None:
        self.runtime_inputs: Dict[str, Any] | None = None

    def sync_with_context(
        self,
        cfg: Dict[str, Any],
        context: Any,
        runtime_inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        _ = cfg
        _ = context
        self.runtime_inputs = runtime_inputs
        return {
            "status": "absent",
            "sync_status": "absent",
            "geometry_absent_reason": "relation_digest_absent_embed_mode",
        }


class _SyncLatentsMissingStub:
    """主同步 stub：返回 latents 缺失语义，并捕获 runtime_inputs"""

    def __init__(self) -> None:
        self.runtime_inputs: Dict[str, Any] | None = None

    def sync_with_context(
        self,
        cfg: Dict[str, Any],
        context: Any,
        runtime_inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        _ = cfg
        _ = context
        self.runtime_inputs = runtime_inputs
        return {
            "status": "absent",
            "sync_status": "absent",
            "geometry_absent_reason": "latents_missing",
        }


class _GeometryNoRelationStub:
    """辅锚点 stub：不提供 relation 绑定，用于验证 detect path reason 归一化"""

    def extract(self, cfg: Dict[str, Any], inputs: Any = None) -> Dict[str, Any]:
        _ = cfg
        _ = inputs
        return {
            "status": "failed",
            "geo_score": None,
            "relation_digest": None,
            "anchor_digest": None,
            "geometry_failure_reason": "runtime_self_attention_unavailable_proxy_forbidden_in_v2",
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


def _build_latent_cache(latents: np.ndarray) -> LatentTrajectoryCache:
    cache = LatentTrajectoryCache()
    cache.capture(27, latents)
    return cache


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


def test_sync_primary_receives_relation_binding_and_latents() -> None:
    """
    功能：paper 模式下首轮 sync 必须拿到正式 relation binding 与 detect latent。 

    Verify sync-primary runtime inputs include relation_digest and detect latents.
    """
    latents = np.random.default_rng(20260314).normal(size=(1, 4, 16, 16)).astype(np.float32)
    cfg = _build_paper_cfg(latents)
    cfg["__runtime_self_attention_maps__"] = [np.zeros((1, 1), dtype=np.float32)]
    cfg["__detect_trajectory_latent_cache__"] = _build_latent_cache(latents)
    sync_stub = _SyncCaptureStub()
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=sync_stub,
    )

    run_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    result = cast(Dict[str, Any], run_chain(impl_set, cfg))

    assert isinstance(sync_stub.runtime_inputs, dict)
    assert sync_stub.runtime_inputs.get("relation_digest") == "r" * 64
    assert sync_stub.runtime_inputs.get("latents") is not None
    assert result.get("relation_digest_bound") == "r" * 64
    assert result.get("status") == "ok"


def test_detect_path_normalizes_legacy_relation_digest_reason() -> None:
    """
    功能：detect path 不应继续输出带 embed_mode 含义的旧缺失 reason。 

    Verify detect sync path rewrites legacy embed-mode relation reason.
    """
    latents = np.random.default_rng(20260315).normal(size=(1, 4, 16, 16)).astype(np.float32)
    cfg = _build_paper_cfg(latents)
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryNoRelationStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=_SyncLegacyReasonStub(),
    )

    run_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    result = cast(Dict[str, Any], run_chain(impl_set, cfg))

    assert result.get("geometry_absent_reason") == "detect_sync_relation_binding_missing"
    assert result.get("status") == "absent"


def test_detect_path_preserves_pre_sync_binding_failure_semantics() -> None:
    """
    功能：真实 pre-sync extractor 失败语义不能在 binding 阶段被吞掉。

    Verify detect path keeps original pre-sync binding failure semantics.
    """
    latents = np.random.default_rng(20260316).normal(size=(1, 4, 16, 16)).astype(np.float32)
    cfg = _build_paper_cfg(latents)
    sync_stub = _SyncLegacyReasonStub()
    geometry_extractor = AttentionAnchorExtractor(
        impl_id="attention_anchor_extractor",
        impl_version="v1",
        impl_digest=digests.canonical_sha256({"impl_id": "attention_anchor_extractor", "impl_version": "v1"}),
    )
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=geometry_extractor,
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=sync_stub,
    )

    run_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    result = cast(Dict[str, Any], run_chain(impl_set, cfg))

    assert isinstance(sync_stub.runtime_inputs, dict)
    assert sync_stub.runtime_inputs.get("relation_digest") is None
    diagnostics = result.get("relation_binding_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("binding_status") in {"failed", "mismatch", "absent", "invalid"}
    assert diagnostics.get("binding_source") == "authentic_runtime_self_attention"
    assert diagnostics.get("geometry_failure_reason") or diagnostics.get("geometry_absent_reason")
    assert result.get("geometry_absent_reason") == "detect_sync_relation_binding_missing"
    assert result.get("geometry_absent_reason_raw") == "relation_digest_absent_embed_mode"


def test_detect_path_does_not_fabricate_missing_latents() -> None:
    """
    功能：detect sync 缺失 latent 时不得伪造输入。

    Verify detect path does not fabricate sync latents when cache is absent.
    """
    latents = np.random.default_rng(20260317).normal(size=(1, 4, 16, 16)).astype(np.float32)
    cfg = _build_paper_cfg(latents)
    cfg["__runtime_self_attention_maps__"] = [np.zeros((1, 1), dtype=np.float32)]
    sync_stub = _SyncLatentsMissingStub()
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=sync_stub,
    )

    run_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    result = cast(Dict[str, Any], run_chain(impl_set, cfg))

    assert isinstance(sync_stub.runtime_inputs, dict)
    assert sync_stub.runtime_inputs.get("latents") is None
    assert result.get("geometry_absent_reason") == "detect_sync_latents_missing"
    assert result.get("geometry_absent_reason_raw") == "latents_missing"
