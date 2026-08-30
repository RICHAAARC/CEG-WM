from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cegwm.runtime.geometry_v4_sd35 import FinalLatentAnchorCallback, run_sd35_final_latent_pair


class _FixturePipeline:
    def __call__(self, *, callback_on_step_end=None, **kwargs):
        state = {"latents": torch.zeros((1, 4, 8, 8), dtype=torch.float32)}
        if callback_on_step_end is not None:
            for step in range(20): state = callback_on_step_end(self, step, None, state)
        return SimpleNamespace(images=[Image.new("RGB", (256, 256), "gray")])


@pytest.mark.integration
def test_final_callback_is_sole_step_19_and_pair_materializes_rgb() -> None:
    callback = FinalLatentAnchorCallback("0123456789abcdef")
    early = {"latents": torch.zeros((1, 4, 8, 8), dtype=torch.float32)}
    assert callback(None, 18, None, early) is early
    updated = callback(None, 19, None, early)
    assert callback.called and not torch.equal(updated["latents"], early["latents"])
    with pytest.raises(RuntimeError, match="more than once"):
        callback(None, 19, None, early)
    pair = run_sd35_final_latent_pair(_FixturePipeline(), "a test prompt", "0123456789abcdef", height=256, width=256, generator=torch.Generator().manual_seed(7))
    assert pair.clean.mode == pair.marked.mode == "RGB"
