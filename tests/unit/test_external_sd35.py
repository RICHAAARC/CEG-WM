import numpy as np
import torch

from cegwm.baselines.external_sd35 import GaussianShadingCarrier, ShallowDiffuseCarrier, TreeRingCarrier, _circle, chacha20, score_rgb


def test_tree_ring_16_channel_fixed_carrier_and_score_direction() -> None:
    carrier = TreeRingCarrier.fixed(); base = torch.randn((1, 16, 64, 64))
    assert carrier.mask.shape == base.shape and carrier.mask[:, 0].any() and not carrier.mask[:, 1].any()
    assert carrier.score(carrier.inject(base)) > carrier.score(base)
    assert len(carrier.digest) == 64


def test_gaussian_chacha_vector_repeat_abs_magnitude_and_vote_roundtrip() -> None:
    assert chacha20(bytes(64), key=bytes(32), nonce=bytes(12)).hex().startswith("76b8e0ada0f13d90405d6ae55386bd28")
    carrier = GaussianShadingCarrier.fixed(); base = torch.randn((1, 16, 64, 64)); marked = carrier.embed(base)
    assert torch.equal(marked.abs(), base.abs()) and carrier.score(marked) == 1.0
    assert carrier.watermark.shape == (1, 16, 8, 8) and len(carrier.digest) == 64


def test_shallow_schedule_fusion_carrier_and_score_direction() -> None:
    carrier = ShallowDiffuseCarrier.fixed(); base = torch.randn((1, 16, 64, 64)); marked = carrier.inject(base)
    assert carrier.mask[:, 0].any() and not carrier.mask[:, 1].any()
    assert carrier.score(marked) > carrier.score(base)


def test_circle_matches_archived_reversed_y_axis() -> None:
    mask = _circle(64, 0)
    assert mask.sum().item() == 1 and bool(mask[31, 32]) and not bool(mask[32, 32])


def test_rgb_scorers_fix_detection_conditioning() -> None:
    class Pipe:
        _execution_device = "cpu"
        def __init__(self): self.calls = []
        def get_image_latents(self, image, *, sample=False): return torch.zeros((1,16,64,64))
        def invert_flow_matching_latent(self, latent, **kwargs): self.calls.append(kwargs); return latent
        def invert_to_edit_timestep(self, latent, **kwargs): self.calls.append(kwargs); return latent
    rgb = np.zeros((2,2,3), dtype=np.uint8)
    for carrier in (TreeRingCarrier.fixed(), GaussianShadingCarrier.fixed()):
        pipe=Pipe(); score_rgb(rgb,pipe,carrier); assert pipe.calls == [{"prompt":"", "num_inference_steps":20, "guidance_scale":1.0}]
    pipe=Pipe(); score_rgb(rgb,pipe,ShallowDiffuseCarrier.fixed()); assert pipe.calls == [{"num_inference_steps":20, "edit_fraction":.2}]
