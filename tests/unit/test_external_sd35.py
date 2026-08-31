import numpy as np
import torch

from cegwm.baselines.external_sd35 import GaussianShadingCarrier, ShallowDiffuseCarrier, TreeRingCarrier, chacha20


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
