"""
文件目的：plan digest append-only 与可复算回归测试。
Module type: General module
"""

from main.core import digests
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SubspacePlannerImpl,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
)


def _build_cfg() -> dict:
    return {
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 6,
                "sample_count": 12,
                "feature_dim": 32,
                "seed": 7,
                "timestep_start": 0,
                "timestep_end": 12,
            }
        }
    }


def _build_inputs() -> dict:
    return {
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


def test_plan_digest_append_only_and_recomputable() -> None:
    impl_digest = digests.canonical_sha256(
        {"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION}
    )
    planner = SubspacePlannerImpl(SUBSPACE_PLANNER_ID, SUBSPACE_PLANNER_VERSION, impl_digest)

    cfg = _build_cfg()
    inputs = _build_inputs()

    result_1 = planner.plan(cfg, mask_digest="mask_digest_anchor", cfg_digest="cfg_digest_anchor", inputs=inputs)
    result_2 = planner.plan(cfg, mask_digest="mask_digest_anchor", cfg_digest="cfg_digest_anchor", inputs=inputs)

    assert result_1.status == "ok"
    assert result_2.status == "ok"
    assert result_1.plan_digest == result_2.plan_digest

    plan = result_1.plan
    assert isinstance(plan, dict)
    assert isinstance(plan.get("lf_basis"), dict)
    assert isinstance(plan.get("hf_basis"), dict)
    assert isinstance(plan.get("lf_region_index_spec"), dict)
    assert isinstance(plan.get("hf_region_index_spec"), dict)
    assert isinstance(plan.get("region_index_digest"), str)
