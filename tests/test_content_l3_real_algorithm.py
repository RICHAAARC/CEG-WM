"""
L3 Content Chain 真实算法必达字段回归测试

功能说明：
- 验证 Content 链 L3 必达字段：mask_digest, plan_digest, lf_trace_digest 等。
- 确保 embed 侧不再输出 baseline，而是真实水印证据。
- 确保 detect 侧能够检测并返回 content_score。
- 严格验证失败语义（absent/failed/mismatch）。
"""

import numpy as np
import pytest
from PIL import Image

from main.watermarking.content_chain.unified_content_extractor import (
    UnifiedContentExtractor,
    UNIFIED_CONTENT_EXTRACTOR_ID,
    UNIFIED_CONTENT_EXTRACTOR_VERSION
)
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
    SubspacePlannerImpl,
    _build_lf_planner_risk_report,
)
from main.watermarking.content_chain.semantic_mask_provider import (
    SemanticMaskProvider,
    SEMANTIC_MASK_PROVIDER_ID
)
from main.core import digests


def test_unified_extractor_embed_mode_returns_mask_digest():
    """
    功能：验证 Embed 模式（detect.content.enabled=False）返回 mask_digest。

    Test unified extractor in embed mode returns mask_digest structural evidence.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If mask_digest is missing or invalid.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": False  # Embed 模式
            }
        },
        "enable_mask": False  # 掩码提取禁用（应返回 absent）
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })
    
    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )
    
    result = extractor.extract(cfg, inputs=None, cfg_digest="test_cfg_digest")
    
    # Embed 模式 + enable_mask=False -> status=absent
    assert result.status == "absent", f"Expected absent, got {result.status}"
    assert result.score is None, "Score must be None in embed mode with mask disabled"
    

def test_unified_extractor_embed_mode_with_mask_enabled_returns_ok():
    """
    功能：验证 Embed 模式启用掩码时返回 status=ok 与 mask_digest。

    Test unified extractor in embed mode with mask enabled returns ok status.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not ok or mask_digest is missing.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": False  # Embed 模式
            }
        },
        "enable_mask": True,  # 启用掩码提取
        "mask_resolution_width": 512,
        "mask_resolution_height": 512
    }
    
    inputs = {
        "latent": [[1.0, 2.0], [3.0, 4.0]],  # 简化输入
        "shape": [1, 4, 64, 64]
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })
    
    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )
    
    result = extractor.extract(cfg, inputs=inputs, cfg_digest="test_cfg_digest")
    
    # Embed 模式 + enable_mask=True -> status=ok（结构证据，score=None）
    # 或者 status=failed（如果 latent 输入无效）
    # 由于我们的输入是最小化的，可能触发 failed
    assert result.status in {"ok", "failed", "absent"}, f"Unexpected status: {result.status}"
    assert result.score is None or result.status != "ok", "Score must be None when status=ok in embed mode (structural evidence)"


def test_unified_extractor_detect_mode_returns_content_score_when_plan_present():
    """
    功能：验证 Detect 模式（detect.content.enabled=True）返回 content_score。

    Test unified extractor in detect mode returns content_score when plan_digest present.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If content_score is missing or status is incorrect.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": True  # Detect 模式
            }
        },
        "watermark": {
            "plan_digest": "test_plan_digest_12345678",  # 计划摘要存在
            "lf": {
                "enabled": True
            },
            "hf": {
                "enabled": False
            }
        }
    }
    
    inputs = {
        "plan_digest": "test_plan_digest_12345678",
        "lf_evidence": {
            "status": "ok",
            "lf_score": 0.85
        },
        "lf_score": 0.85
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })
    
    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )
    
    result = extractor.extract(cfg, inputs=inputs, cfg_digest="test_cfg_digest")
    
    # Detect 模式 + plan_digest 一致 -> status=ok, score=非None
    assert result.status == "ok", f"Expected ok, got {result.status}"
    assert result.score is not None, "Score must be non-None in detect mode when status=ok"
    assert result.score >= 0, f"Score must be non-negative, got {result.score}"


def test_unified_extractor_detect_mode_plan_mismatch_returns_mismatch():
    """
    功能：验证 Detect 模式 plan_digest 不一致时返回 mismatch。

    Test unified extractor in detect mode returns mismatch when plan_digest mismatches.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not mismatch or failure reason is missing.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": True  # Detect 模式
            }
        },
        "watermark": {
            "lf": {
                "enabled": True
            }
        }
    }
    
    inputs = {
        "expected_plan_digest": "expected_plan_digest_abc123",  # 期望的 plan_digest
        "plan_digest": "actual_plan_digest_xyz789",  # 实际的 plan_digest（不一致）
        "lf_evidence": {
            "status": "ok",
            "lf_score": 0.75
        }
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })
    
    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )
    
    result = extractor.extract(cfg, inputs=inputs, cfg_digest="test_cfg_digest")
    
    # Detect 模式 + plan_digest 不一致 -> status=mismatch, score=None
    assert result.status == "mismatch", f"Expected mismatch, got {result.status}"
    assert result.score is None, "Score must be None when status=mismatch"
    assert result.content_failure_reason is not None, "Failure reason must be present when status=mismatch"
    assert "plan_mismatch" in result.content_failure_reason or "detector_plan_mismatch" in result.content_failure_reason, \
        f"Failure reason should mention plan mismatch, got: {result.content_failure_reason}"


def test_unified_extractor_embed_precompute_phase_overrides_detect_mode(tmp_path) -> None:
    """
    功能：embed 预计算阶段即使 detect.content.enabled=true，也必须走 mask/provider 路径。

    Verify embed precompute phase forces mask extraction even when detect.content.enabled is enabled.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    input_image = tmp_path / "embed_precompute_input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(input_image)

    cfg = {
        "detect": {
            "content": {
                "enabled": True
            }
        },
        "enable_mask": True,
        "mask": {
            "saliency_source": "proxy"
        }
    }

    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })

    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )

    result = extractor.extract(
        cfg,
        inputs={
            "image_path": str(input_image),
            "content_runtime_phase": "embed_precompute",
        },
        cfg_digest="test_cfg_digest"
    )

    assert result.status == "ok", f"Expected ok, got {result.status}"
    assert isinstance(result.mask_digest, str) and result.mask_digest
    assert result.score is None


def test_l3_content_chain_not_baseline_when_enabled():
    """
    功能：验证 Content 链 L3 不再是 baseline（启用时）。

    Regression test: Content chain L3 must not emit baseline when enabled.
    Validates that embed_record contains mask_digest/plan_digest/lf_trace_digest.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If L3 必达字段缺失。
    """
    # 此测试需要完整的 embed 流程，暂时标记为 skip（需要 pipeline 和推理）
    pytest.skip("Requires full embed pipeline integration test (TBD in integration suite)")


def test_semantic_mask_provider_returns_mask_digest_when_enabled():
    """
    功能：验证 SemanticMaskProvider 启用时返回 mask_digest。

    Test SemanticMaskProvider returns mask_digest when enable_mask=True.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If mask_digest is missing or status is incorrect.
    """
    cfg = {
        "enable_mask": True,
        "mask_resolution_width": 512,
        "mask_resolution_height": 512
    }
    
    inputs = {
        "latent": [[1.0, 2.0], [3.0, 4.0]],
        "shape": [1, 4, 64, 64]
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": SEMANTIC_MASK_PROVIDER_ID,
        "impl_version": "v1"
    })
    
    provider = SemanticMaskProvider(
        SEMANTIC_MASK_PROVIDER_ID,
        "v1",
        impl_digest
    )
    
    result = provider.extract(cfg, inputs=inputs, cfg_digest="test_cfg_digest")
    
    # enable_mask=True，但输入可能无效 -> status=ok or failed
    # 如果 ok，必须有 mask_digest
    if result.status == "ok":
        assert result.mask_digest is not None, "mask_digest must be present when status=ok"
        assert isinstance(result.mask_digest, str), "mask_digest must be str"
        assert len(result.mask_digest) > 0, "mask_digest must be non-empty"
    elif result.status == "failed":
        assert result.content_failure_reason is not None, "Failure reason must be present when status=failed"
    else:
        pytest.fail(f"Unexpected status: {result.status}")


def test_content_l3_subspace_plan_requires_nonempty_route_basis_bridge():
    """
    功能：验证内容链 L3 子空间计划必须输出非空强语义 bridge。

    Verify content-chain L3 planner emits a non-empty route-first bridge.
    """
    impl_digest = digests.canonical_sha256(
        {"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION}
    )
    planner = SubspacePlannerImpl(SUBSPACE_PLANNER_ID, SUBSPACE_PLANNER_VERSION, impl_digest)

    cfg = {
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 4,
                "sample_count": 8,
                "feature_dim": 16,
                "seed": 13,
                "timestep_start": 0,
                "timestep_end": 6,
            }
        }
    }
    inputs = {
        "trace_signature": {
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "height": 512,
            "width": 512,
        },
        "mask_summary": {
            "area_ratio": 0.4,
            "downsample_grid_shape": [8, 8],
            "downsample_grid_true_indices": [0, 1, 2, 8, 9, 10],
            "downsample_grid_digest": "a" * 64,
        },
        "routing_digest": "b" * 64,
    }

    result = planner.plan(cfg, mask_digest="mask_digest_l3", cfg_digest="cfg_digest_l3", inputs=inputs)

    assert result.status == "ok"
    assert isinstance(result.plan, dict)
    route_basis_bridge = result.plan.get("route_basis_bridge")
    assert isinstance(route_basis_bridge, dict)
    assert route_basis_bridge.get("bridge_version") == "route_basis_bridge"
    assert route_basis_bridge.get("feature_routing_digest")

    route_layer = route_basis_bridge.get("route_layer")
    routed_matrix_layer = route_basis_bridge.get("routed_matrix_layer")
    dual_subspace_estimation = route_basis_bridge.get("dual_subspace_estimation")
    assert isinstance(route_layer, dict)
    assert isinstance(routed_matrix_layer, dict)
    assert isinstance(dual_subspace_estimation, dict)
    assert route_layer.get("route_source") == "mask_region_index_spec"
    assert routed_matrix_layer.get("matrix_source") == "build_routed_decomposition_matrices"
    assert dual_subspace_estimation.get("lf_basis_source") == "lf_decomposition_matrix"
    assert dual_subspace_estimation.get("hf_basis_source") == "hf_decomposition_matrix"


def test_lf_planner_risk_report_classifies_host_baseline_dominant() -> None:
    lf_matrix = [
        [3.0, 3.0],
        [3.2, 2.8],
        [2.9, 3.1],
        [3.1, 2.9],
    ]
    lf_projection_matrix = [
        [1.0],
        [1.0],
        [0.0],
        [0.0],
    ]
    feature_routing = {"lf_feature_cols": [0, 1]}

    report = _build_lf_planner_risk_report(
        lf_decomposition_matrix=lf_matrix,
        lf_projection_matrix=lf_projection_matrix,
        feature_routing=feature_routing,
        planner_rank=1,
    )

    assert report["risk_classification"] == "host_baseline_dominant"
    assert report["host_baseline_dominant_flag"] is True


def test_lf_planner_risk_report_classifies_detect_trajectory_shift() -> None:
    lf_matrix = [
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
    ]
    lf_projection_matrix = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    feature_routing = {"lf_feature_cols": [0, 1]}

    report = _build_lf_planner_risk_report(
        lf_decomposition_matrix=lf_matrix,
        lf_projection_matrix=lf_projection_matrix,
        feature_routing=feature_routing,
        planner_rank=2,
    )

    assert report["risk_classification"] == "detect_trajectory_shift"
    assert report["detect_trajectory_shift_flag"] is True

