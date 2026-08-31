from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from cegwm.runtime.geometry_v6_sd35 import run_sd35_geometry_v6_r0_arm


class _Distribution:
    def __init__(self, value): self._value = value
    def mode(self): return self._value


class _VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
    def decode(self, value, return_dict=True): return SimpleNamespace(sample=value * self.anchor)
    def encode(self, value): return SimpleNamespace(latent_dist=_Distribution(value * self.anchor))


class _Pipeline:
    def __init__(self):
        self.vae = _VAE()
        self.events = []
    def __call__(self, **kwargs):
        callback = kwargs["callback_on_step_end"]
        latents = torch.zeros(1, 4, 16, 16)
        for step in range(20):
            latents = latents + 1.0  # fake scheduler update happens before callback
            self.events.append(("scheduler", step, float(latents.mean())))
            latents = callback(self, step, None, {"latents": latents})["latents"]
            self.events.append(("callback", step, float(latents.mean())))
        self.events.append(("decode", float(latents.mean())))
        return SimpleNamespace(images=[Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), "RGB")])


def test_fake_pipeline_fixes_step19_after_scheduler_and_before_final_decode():
    pipeline = _Pipeline()
    output = run_sd35_geometry_v6_r0_arm(
        pipeline, "frozen prompt", "geometry_only", content_key=None,
        geometry_key="geometry-key-0001", amplitude=0.0025, content_assets=None,
        height=256, width=256,
    )
    assert output.arm == "geometry_only"
    scheduler_19 = next(event for event in pipeline.events if event[:2] == ("scheduler", 19))
    callback_19 = next(event for event in pipeline.events if event[:2] == ("callback", 19))
    decode = pipeline.events[-1]
    assert scheduler_19[2] == 20.0
    assert callback_19[2] != scheduler_19[2]
    assert decode[0] == "decode" and decode[1] == callback_19[2]
