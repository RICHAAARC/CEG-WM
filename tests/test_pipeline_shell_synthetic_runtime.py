"""
File purpose: Pipeline shell synthetic runtime regression tests.
Module type: General module
"""

from __future__ import annotations

from main.diffusion.sd3.pipeline_factory import build_pipeline_shell
from main.registries import pipeline_registry


def test_pipeline_shell_builds_executable_synthetic_pipeline() -> None:
    cfg = {
        "pipeline_impl_id": pipeline_registry.SD3_DIFFUSERS_SHELL_ID,
        "inference_num_steps": 8,
        "inference_height": 64,
        "inference_width": 64,
    }

    result = build_pipeline_shell(cfg)

    assert result["pipeline_status"] == "built"
    pipeline_obj = result.get("pipeline_obj")
    assert pipeline_obj is not None
    assert callable(pipeline_obj)
    assert bool(getattr(pipeline_obj, "is_synthetic_pipeline", False)) is True

    output = pipeline_obj(
        prompt="synthetic smoke",
        num_inference_steps=4,
        height=32,
        width=32,
    )
    assert hasattr(output, "images")
    assert isinstance(output.images, list)
    assert len(output.images) == 1
