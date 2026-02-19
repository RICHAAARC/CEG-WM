"""
File purpose: trajectory tap absent 语义稳定性测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

from main.diffusion.sd3.infer_runtime import run_sd3_inference


class _NoCallbackPipelineStub:
    """
    功能：不支持 callback 的推理桩。

    Pipeline stub without callback support.

    Args:
        None.

    Returns:
        None.
    """

    def __call__(self, **kwargs: Any) -> Any:
        class _Output:
            images = [object()]

        return _Output()


def test_trajectory_tap_absent_is_stable() -> None:
    """
    功能：不支持 callback 时，absent_reason 必须稳定为 unsupported_pipeline。

    Unsupported callback path must produce stable absent reason.

    Args:
        None.

    Returns:
        None.
    """
    cfg: Dict[str, Any] = {
        "inference_enabled": True,
        "inference_prompt": "prompt",
        "inference_num_steps": 3,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
        "trajectory_tap": {"enabled": True},
        "watermark": {
            "subspace": {
                "sample_count": 3,
                "feature_dim": 16,
                "timestep_start": 0,
                "timestep_end": 2
            }
        }
    }

    result = run_sd3_inference(cfg, _NoCallbackPipelineStub(), "cpu", None)
    evidence = result["trajectory_evidence"]

    assert result["inference_status"] == "ok"
    assert evidence["status"] == "absent"
    assert evidence["trajectory_absent_reason"] == "unsupported_pipeline"
    assert evidence["audit"]["trajectory_tap_status"] == "absent"
    assert evidence["audit"]["trajectory_absent_reason"] == "unsupported_pipeline"
    assert evidence["trajectory_digest"] is None
    assert evidence["trajectory_spec_digest"] is None
    assert evidence["trajectory_metrics"] is None
    assert evidence["trajectory_stats"] is None
