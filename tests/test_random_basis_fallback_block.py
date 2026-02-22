"""
File purpose: 禁止随机 basis fallback 的回归测试。
Module type: General module
"""

from __future__ import annotations

import numpy as np

from main.diffusion.sd3.infer_runtime import _build_plan_for_injection


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

