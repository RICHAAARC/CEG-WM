"""
文件目的：LatentModifier 对 basis/region_spec strict 校验回归测试。
Module type: General module
"""

import numpy as np
import pytest

from main.watermarking.content_chain.latent_modifier import (
    LATENT_MODIFIER_ID,
    LATENT_MODIFIER_VERSION,
    LatentModifier,
)


def test_latent_modifier_requires_basis_and_region_spec() -> None:
    modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)
    latents = np.zeros((1, 4, 8, 8), dtype=np.float32)

    cfg = {
        "require_basis_region_spec": True,
        "lf_enabled": True,
        "hf_enabled": True,
    }
    bad_plan = {
        "lf_basis": {"projection_matrix": [[1.0], [0.0]]},
    }

    with pytest.raises(ValueError):
        modifier.apply_latent_update(latents=latents, plan=bad_plan, cfg=cfg, step_index=0, key=7)
