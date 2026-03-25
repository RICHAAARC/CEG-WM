"""
File purpose: 论文主路径收口回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np

from main.core import config_loader
from main.registries import runtime_resolver
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.high_freq_embedder import HIGH_FREQ_TRUNCATION_CODEC_ID
from main.watermarking.content_chain.low_freq_coder import LOW_FREQ_TEMPLATE_CODEC_ID
from main.watermarking.detect import orchestrator as detect_orchestrator
from main.watermarking.embed import orchestrator as embed_orchestrator


def _load_paper_full_cuda_cfg() -> Dict[str, Any]:
    cfg, _ = config_loader.load_yaml_with_provenance("configs/paper_full_cuda.yaml")
    return cfg


def test_paper_full_cuda_fusion_is_np() -> None:
    cfg = _load_paper_full_cuda_cfg()
    raw_impl_cfg = cfg.get("impl")
    impl_cfg = cast(Dict[str, Any], raw_impl_cfg if isinstance(raw_impl_cfg, dict) else {})
    assert impl_cfg.get("fusion_rule_id") == "fusion_neyman_pearson"

    identity = runtime_resolver.parse_impl_identity_from_cfg(cfg)
    assert identity.fusion_rule_id == "fusion_neyman_pearson"


def test_runtime_resolver_builds_hf_lf_from_impl_identity() -> None:
    cfg = _load_paper_full_cuda_cfg()
    impl_cfg = cast(Dict[str, Any], cfg.get("impl") if isinstance(cfg.get("impl"), dict) else {})
    identity, impl_set, _ = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)
    assert impl_cfg.get("hf_embedder_id") == HIGH_FREQ_TRUNCATION_CODEC_ID
    assert impl_cfg.get("lf_coder_id") == LOW_FREQ_TEMPLATE_CODEC_ID
    assert identity.hf_embedder_id == impl_cfg.get("hf_embedder_id")
    assert identity.lf_coder_id == impl_cfg.get("lf_coder_id")
    assert impl_set.hf_embedder is not None
    assert impl_set.lf_coder is not None
    assert getattr(impl_set.hf_embedder, "impl_id", None) == impl_cfg.get("hf_embedder_id")
    assert getattr(impl_set.lf_coder, "impl_id", None) == impl_cfg.get("lf_coder_id")


class _SyncStub:
    def sync_with_context(self, cfg: Dict[str, Any], context: Any) -> Dict[str, Any]:
        _ = cfg
        _ = context
        return {
            "status": "ok",
            "sync_status": "ok",
            "sync_digest": "s" * 64,
            "sync_quality_metrics": {"quality_score": 0.9, "uncertainty": 0.1},
            "relation_digest_bound": "r" * 64,
        }


class _GeometryStub:
    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        _ = cfg
        assert isinstance(inputs, dict)
        # attention_maps 仅在真实 self-attention 可用时存在；latent proxy 路径已移除
        assert "sync_result" in inputs
        return {
            "status": "ok",
            "geo_score": 0.8,
            "relation_digest": "r" * 64,
            "anchor_digest": "a" * 64,
        }


def test_detect_geometry_chain_runs_sync_then_extract() -> None:
    cfg: Dict[str, Any] = {
        "__detect_pipeline_obj__": object(),
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
            }
        }
    }
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=_SyncStub(),
    )
    run_geometry_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    geometry_result = run_geometry_chain(impl_set, cfg)
    assert isinstance(geometry_result, dict)
    geometry_payload = cast(Dict[str, Any], geometry_result)
    assert geometry_payload.get("status") == "ok"
    assert geometry_payload.get("sync_status") == "ok"


class _SyncRelationConsumerStub:
    def sync_with_context(self, cfg: Dict[str, Any], context: Any, runtime_inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        _ = cfg
        _ = context
        if not isinstance(runtime_inputs, dict):
            return {
                "status": "mismatch",
                "sync_status": "mismatch",
                "geometry_failure_reason": "runtime_inputs_missing",
            }
        relation_digest = runtime_inputs.get("relation_digest")
        if not isinstance(relation_digest, str) or not relation_digest:
            return {
                "status": "mismatch",
                "sync_status": "mismatch",
                "geometry_failure_reason": "relation_digest_missing_for_v2",
            }
        return {
            "status": "ok",
            "sync_status": "ok",
            "sync_digest": "s" * 64,
            "sync_quality_metrics": {"quality_score": 0.92, "uncertainty": 0.08},
            "relation_digest_bound": relation_digest,
        }


class _SyncMismatchStub:
    def sync_with_context(self, cfg: Dict[str, Any], context: Any, runtime_inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        _ = cfg
        _ = context
        _ = runtime_inputs
        return {
            "status": "mismatch",
            "sync_status": "mismatch",
            "geometry_failure_reason": "sync_quality_below_threshold",
        }


def test_detect_geometry_chain_anchor_then_sync_relation_digest_bound() -> None:
    cfg: Dict[str, Any] = {
        "__detect_pipeline_obj__": object(),
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
            }
        }
    }
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=_SyncRelationConsumerStub(),
    )
    run_geometry_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    geometry_result = run_geometry_chain(impl_set, cfg)
    assert isinstance(geometry_result, dict)
    geometry_payload = cast(Dict[str, Any], geometry_result)
    assert geometry_payload.get("anchor_status") == "ok"
    assert geometry_payload.get("sync_status") == "ok"
    assert geometry_payload.get("relation_digest") == "r" * 64
    assert geometry_payload.get("relation_digest_bound") == "r" * 64
    relation_binding = geometry_payload.get("relation_digest_binding")
    assert isinstance(relation_binding, dict)
    assert relation_binding.get("binding_status") == "matched"


def test_detect_geometry_chain_paper_path_sync_primary_enabled() -> None:
    """
    功能：paper 路径默认启用 sync 主证据，sync 失败时应主导几何状态。 

    Verify paper path enables sync-primary semantics by default.

    Args:
        None.

    Returns:
        None.
    """
    latents = np.random.default_rng(202602281).normal(size=(1, 4, 16, 16)).astype(np.float32)
    _ = latents  # latents 不再直接注入 cfg
    cfg: Dict[str, Any] = {
        "__detect_pipeline_obj__": object(),
        "__runtime_self_attention_maps__": [np.zeros((1, 1), dtype=np.float32)],
        "paper_faithfulness": {"enabled": True},
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
            }
        }
    }
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=_SyncMismatchStub(),
    )
    run_geometry_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    geometry_result = run_geometry_chain(impl_set, cfg)
    geometry_payload = cast(Dict[str, Any], geometry_result)
    assert geometry_payload.get("status") == "mismatch"
    assert geometry_payload.get("geo_score") is None
    hierarchy = geometry_payload.get("geometry_evidence_hierarchy")
    assert isinstance(hierarchy, dict)
    assert hierarchy.get("switch_enabled") is True
    assert hierarchy.get("primary_source") == "sync"


def test_detect_geometry_chain_paper_path_sync_primary_can_rollback() -> None:
    """
    功能：paper 路径显式关闭开关时，保持 anchor 主证据旧语义。 

    Verify explicit rollback switch preserves anchor-primary compatibility.

    Args:
        None.

    Returns:
        None.
    """
    latents = np.random.default_rng(202602282).normal(size=(1, 4, 16, 16)).astype(np.float32)
    _ = latents  # latents 不再直接注入 cfg
    cfg: Dict[str, Any] = {
        "__detect_pipeline_obj__": object(),
        "__runtime_self_attention_maps__": [np.zeros((1, 1), dtype=np.float32)],
        "paper_faithfulness": {"enabled": True},
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "sync_primary_anchor_secondary": False,
            }
        }
    }
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=_GeometryStub(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=_SyncMismatchStub(),
    )
    run_geometry_chain = getattr(detect_orchestrator, "_run_geometry_chain_with_sync")
    geometry_result = run_geometry_chain(impl_set, cfg)
    geometry_payload = cast(Dict[str, Any], geometry_result)
    assert geometry_payload.get("status") == "ok"
    assert geometry_payload.get("geo_score") == 0.8
    hierarchy = geometry_payload.get("geometry_evidence_hierarchy")
    assert isinstance(hierarchy, dict)
    assert hierarchy.get("switch_enabled") is False
    assert hierarchy.get("primary_source") == "anchor"


def test_image_domain_sidecar_disabled_in_paper_mode() -> None:
    """
    功能：验证 paper_full_cuda 配置下 sidecar 禁用状态。
    v2.0 收口后，论文正式路径（paper_faithfulness.enabled=True）禁止 image-domain sidecar。
    """
    cfg = _load_paper_full_cuda_cfg()
    sidecar_enabled = getattr(detect_orchestrator, "_is_image_domain_sidecar_enabled")
    # v2.0 收口：paper 正式路径 image_domain_sidecar_enabled=false，sidecar 必须禁用
    assert sidecar_enabled(cfg) is False


def test_embed_trace_mode_no_stub_marker() -> None:
    cfg = {"paper_faithfulness": {"enabled": True}}
    evidence = {
        "status": "ok",
        "injection_trace_digest": "a" * 64,
        "injection_params_digest": "b" * 64,
    }
    build_trace = getattr(embed_orchestrator, "_build_latent_step_embed_trace")
    trace = build_trace(cfg, evidence)
    assert trace.get("embed_mode") == "latent_step_injection_v1"
    assert "stub" not in trace.get("embed_mode", "")
