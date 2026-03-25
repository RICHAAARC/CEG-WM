"""
File purpose: 禁止随机 basis fallback 的回归测试。
Module type: General module
"""

from __future__ import annotations

import inspect
import numpy as np

from main.core import digests
from main.diffusion.sd3.infer_runtime import _build_plan_for_injection
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
    SubspacePlannerImpl,
)


def test_missing_torch_latents_blocks_runtime_basis_construction() -> None:
    """
    功能：缺失 torch 张量输入时必须 absent，且不得生成随机 basis。
    """
    plan_payload = {"rank": 4}
    plan = _build_plan_for_injection(
        plan_ref=plan_payload,
        plan_digest="a" * 64,
        latents=np.zeros((1, 4, 8, 8), dtype=np.float32),
        injection_cfg={"lf_enabled": True, "hf_enabled": True},
    )

    runtime_binding = plan.get("runtime_subspace_binding")
    assert isinstance(runtime_binding, dict)
    assert runtime_binding.get("status") == "absent"
    assert "lf_basis" not in plan
    assert "hf_basis" not in plan


def test_route_basis_bridge_uses_routed_feature_matrices() -> None:
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
                "seed": 5,
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
            "area_ratio": 0.5,
            "downsample_grid_shape": [8, 8],
            "downsample_grid_true_indices": [0, 1, 8, 9],
            "downsample_grid_digest": "a" * 64,
        },
        "routing_digest": "b" * 64,
    }

    result = planner.plan(cfg, mask_digest="mask_digest_route", cfg_digest="cfg_digest_route", inputs=inputs)

    assert result.status == "ok"
    assert isinstance(result.plan, dict)
    route_basis_bridge = result.plan.get("route_basis_bridge")
    assert isinstance(route_basis_bridge, dict)
    assert route_basis_bridge.get("bridge_version") == "route_basis_bridge"

    routed_matrix_layer = route_basis_bridge.get("routed_matrix_layer")
    dual_subspace_estimation = route_basis_bridge.get("dual_subspace_estimation")
    route_layer = route_basis_bridge.get("route_layer")
    assert isinstance(routed_matrix_layer, dict)
    assert isinstance(dual_subspace_estimation, dict)
    assert isinstance(route_layer, dict)
    assert routed_matrix_layer.get("matrix_source") == "build_routed_decomposition_matrices"
    assert routed_matrix_layer.get("lf_decomposition_matrix_source") == "stack(lf_trajectory_matrix, lf_jvp_matrix)"
    assert routed_matrix_layer.get("hf_decomposition_matrix_source") == "stack(hf_trajectory_matrix, hf_jvp_matrix)"
    assert dual_subspace_estimation.get("estimation_input") == "routed_decomposition_matrices"
    assert dual_subspace_estimation.get("lf_basis_source") == "lf_decomposition_matrix"
    assert dual_subspace_estimation.get("hf_basis_source") == "hf_decomposition_matrix"
    assert route_layer.get("routing_stage_order") == [
        "build_feature_routing_from_mask",
        "build_routed_decomposition_matrices",
        "estimate_routed_dual_subspaces",
    ]
    assert dual_subspace_estimation.get("lf_basis_matrix_shape") == [16, 4]
    assert dual_subspace_estimation.get("hf_basis_matrix_shape") == [16, 4]
    assert isinstance(routed_matrix_layer.get("lf_decomposition_matrix_digest"), str)
    assert isinstance(routed_matrix_layer.get("hf_decomposition_matrix_digest"), str)


def test_route_precedes_basis_estimation() -> None:
    source = inspect.getsource(SubspacePlannerImpl._estimate_low_dim_subspace)

    route_index = source.index("_build_feature_routing_from_mask")
    matrix_index = source.index("_build_routed_decomposition_matrices")
    basis_index = source.index("_estimate_routed_dual_subspaces")

    assert route_index < matrix_index < basis_index


def test_lf_hf_basis_are_estimated_from_routed_decomposition_matrices() -> None:
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
                "seed": 5,
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
            "area_ratio": 0.5,
            "downsample_grid_shape": [8, 8],
            "downsample_grid_true_indices": [0, 1, 8, 9],
            "downsample_grid_digest": "a" * 64,
        },
        "routing_digest": "b" * 64,
    }

    result = planner.plan(cfg, mask_digest="mask_digest_route", cfg_digest="cfg_digest_route", inputs=inputs)

    assert result.status == "ok"
    assert isinstance(result.plan, dict)
    route_basis_bridge = result.plan.get("route_basis_bridge")
    assert isinstance(route_basis_bridge, dict)

    routed_matrix_layer = route_basis_bridge.get("routed_matrix_layer")
    dual_subspace_estimation = route_basis_bridge.get("dual_subspace_estimation")
    assert isinstance(routed_matrix_layer, dict)
    assert isinstance(dual_subspace_estimation, dict)
    assert dual_subspace_estimation.get("lf_basis_source") == "lf_decomposition_matrix"
    assert dual_subspace_estimation.get("hf_basis_source") == "hf_decomposition_matrix"
    lf_decomposition_matrix_shape = routed_matrix_layer.get("lf_decomposition_matrix_shape")
    hf_decomposition_matrix_shape = routed_matrix_layer.get("hf_decomposition_matrix_shape")
    lf_trajectory_matrix_shape = routed_matrix_layer.get("lf_trajectory_matrix_shape")
    hf_trajectory_matrix_shape = routed_matrix_layer.get("hf_trajectory_matrix_shape")
    lf_jvp_matrix_shape = routed_matrix_layer.get("lf_jvp_matrix_shape")
    hf_jvp_matrix_shape = routed_matrix_layer.get("hf_jvp_matrix_shape")
    assert lf_decomposition_matrix_shape == [
        lf_trajectory_matrix_shape[0] + lf_jvp_matrix_shape[0],
        len(route_basis_bridge.get("lf_feature_cols", [])),
    ]
    assert hf_decomposition_matrix_shape == [
        hf_trajectory_matrix_shape[0] + hf_jvp_matrix_shape[0],
        len(route_basis_bridge.get("hf_feature_cols", [])),
    ]

