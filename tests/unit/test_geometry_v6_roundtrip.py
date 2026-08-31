import math
from types import SimpleNamespace

import torch

from cegwm.method.geometry_v6_roundtrip import (
    R0_AMPLITUDE_CANDIDATES,
    apply_roundtrip_adjoint_update,
    derive_geometry_keys,
    keyed_template,
    midfrequency_support,
)


class _Distribution:
    def __init__(self, value): self._value = value
    def mode(self): return self._value


class _VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
    def decode(self, value, return_dict=True):
        return SimpleNamespace(sample=value * self.anchor)
    def encode(self, value):
        return SimpleNamespace(latent_dist=_Distribution(value * self.anchor))


def test_geometry_keys_are_domain_separated_and_template_is_midfrequency_only():
    left = derive_geometry_keys("geometry-key-0001")
    right = derive_geometry_keys("geometry-key-0002")
    assert len({left.search, left.fit, left.validate}) == 3
    assert left.validate != right.validate
    latents = torch.randn(1, 4, 16, 16)
    template = keyed_template(latents, "geometry-key-0001")
    spectrum = torch.fft.fft2(template, dim=(-2, -1))
    outside = spectrum * (~midfrequency_support(latents))[None, None]
    assert float(outside.abs().max()) < 1e-5
    assert math.isclose(float(torch.linalg.vector_norm(template)), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_single_adjoint_update_is_global_amplitude_only_and_finite():
    vae = _VAE()
    latents = torch.randn(1, 4, 16, 16)
    updated = apply_roundtrip_adjoint_update(latents, "geometry-key-0001", R0_AMPLITUDE_CANDIDATES[0], vae)
    assert updated.shape == latents.shape
    assert updated.dtype == latents.dtype
    assert bool(torch.isfinite(updated).all())
    assert not torch.equal(updated, latents)
