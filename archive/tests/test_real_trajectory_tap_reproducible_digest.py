"""
File purpose: 真实 trajectory tap 可复算摘要回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from main.diffusion.sd3 import trajectory_tap


class _PipelineOutput:
    def __init__(self) -> None:
        self.images = [object()]


class _CallbackPipelineStub:
    """
    功能：支持 callback_on_step_end 的推理桩。

    Callback-capable pipeline stub for deterministic latent generation.

    Args:
        base_seed: Base seed for deterministic latent simulation.

    Returns:
        None.
    """

    def __init__(self, base_seed: int) -> None:
        self._base_seed = base_seed

    def __call__(
        self,
        *,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        callback_on_step_end: Any = None,
        callback_on_step_end_tensor_inputs: Any = None
    ) -> _PipelineOutput:
        if callback_on_step_end is not None:
            for step_index in range(num_inference_steps):
                rng = np.random.default_rng(self._base_seed + step_index)
                latents = rng.normal(0.0, 1.0, size=(1, 4, 8, 8)).astype(np.float32)
                callback_on_step_end(
                    self,
                    step_index,
                    step_index,
                    {"latents": latents}
                )
        return _PipelineOutput()


def _build_cfg() -> Dict[str, Any]:
    return {
        "inference_enabled": True,
        "trajectory_tap": {
            "enabled": True,
            "stats_precision_digits": 6,
            "tensor_types": ["latent"],
            "module_paths": ["transformer"]
        },
        "watermark": {
            "subspace": {
                "sample_count": 4,
                "feature_dim": 32,
                "timestep_start": 0,
                "timestep_end": 3,
                "trajectory_step_stride": 1
            }
        }
    }


def test_real_trajectory_tap_reproducible_digest() -> None:
    """
    功能：同一 cfg/seed/model 下 trajectory digest 必须可复算一致。

    Trajectory digest must be reproducible under identical cfg/seed/model conditions.

    Args:
        None.

    Returns:
        None.
    """
    cfg = _build_cfg()
    infer_kwargs: Dict[str, Any] = {
        "prompt": "test prompt",
        "num_inference_steps": 4,
        "guidance_scale": 7.0,
        "height": 512,
        "width": 512,
    }
    runtime_meta: Dict[str, Any] = {
        "num_inference_steps": 4,
        "guidance_scale": 7.0,
        "height": 512,
        "width": 512,
    }

    pipeline = _CallbackPipelineStub(base_seed=123)
    first = trajectory_tap.tap_from_pipeline(
        cfg,
        pipeline,
        infer_kwargs,
        runtime_meta,
        seed=2026,
        device="cpu"
    )
    second = trajectory_tap.tap_from_pipeline(
        cfg,
        pipeline,
        infer_kwargs,
        runtime_meta,
        seed=2026,
        device="cpu"
    )

    evidence_a = first["trajectory_evidence"]
    evidence_b = second["trajectory_evidence"]

    assert evidence_a["status"] == "ok"
    assert evidence_b["status"] == "ok"
    assert evidence_a["trajectory_spec_digest"] == evidence_b["trajectory_spec_digest"]
    assert evidence_a["trajectory_digest"] == evidence_b["trajectory_digest"]
    assert evidence_a["trajectory_metrics"]["steps"] == evidence_b["trajectory_metrics"]["steps"]
    assert evidence_a.get("trajectory_metrics") is not None
    assert evidence_a["trajectory_stats"] == evidence_a["trajectory_metrics"]
    assert evidence_b["trajectory_stats"] == evidence_b["trajectory_metrics"]
    assert evidence_a["audit"]["trajectory_tap_status"] == "ok"
    assert evidence_a["audit"]["trajectory_absent_reason"] is None
    assert "latents" not in str(evidence_a)
