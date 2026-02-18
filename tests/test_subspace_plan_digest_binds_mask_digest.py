"""
File purpose: Mask digest binding test for subspace planner.
Module type: General module
"""

from main.core import digests
from main.watermarking.content_chain.subspace.placeholder_planner import (
    SubspacePlannerImpl,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION
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
                "timestep_end": 12
            }
        },
        "model": {
            "model_id": "sd3-test",
            "model_revision": "rev-1"
        }
    }


def _build_inputs() -> dict:
    return {
        "trace_signature": {
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "height": 512,
            "width": 512
        }
    }


def test_subspace_plan_digest_binds_mask_digest() -> None:
    impl_digest = digests.canonical_sha256({"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION})
    planner = SubspacePlannerImpl(SUBSPACE_PLANNER_ID, SUBSPACE_PLANNER_VERSION, impl_digest)

    cfg = _build_cfg()
    inputs = _build_inputs()

    result_1 = planner.plan(cfg, mask_digest="mask_digest_A", cfg_digest="cfg_digest_same", inputs=inputs)
    result_2 = planner.plan(cfg, mask_digest="mask_digest_B", cfg_digest="cfg_digest_same", inputs=inputs)

    assert result_1.status == "ok"
    assert result_2.status == "ok"
    assert result_1.plan_digest != result_2.plan_digest
