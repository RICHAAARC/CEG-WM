from types import SimpleNamespace
import torch
from cegwm.method.geometry_v6_roundtrip import R0_AMPLITUDE_CANDIDATES, apply_roundtrip_adjoint_update, midfrequency_support, public_pilot_partition, public_pilot_template

class _Dist:
    def __init__(self,x): self.x=x
    def mode(self): return self.x
class _VAE(torch.nn.Module):
    def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.tensor(1.0)); self.config=SimpleNamespace(scaling_factor=1.0,shift_factor=0.0)
    def decode(self,x,return_dict=True): return SimpleNamespace(sample=x*self.p)
    def encode(self,x): return SimpleNamespace(latent_dist=_Dist(x*self.p))

def test_public_template_is_deterministic_and_partition_is_exact():
    z=torch.randn(1,4,16,16); left,right=public_pilot_template(z),public_pilot_template(z.clone())
    assert torch.equal(left,right)
    support=midfrequency_support(z); part=public_pilot_partition(z)
    assert torch.equal(part.search|part.fit|part.validate,support)
    assert not bool((part.search&part.fit).any() or (part.search&part.validate).any() or (part.fit&part.validate).any())
    assert all(int(mask.sum())>0 for mask in (part.search,part.fit,part.validate))
    for y in range(16):
        for x in range(16):
            mate=((-y)%16,(-x)%16)
            assert sum(bool(mask[y,x]) for mask in (part.search,part.fit,part.validate)) == sum(bool(mask[mate]) for mask in (part.search,part.fit,part.validate))

def test_public_adjoint_works_under_no_grad_and_inference_mode_without_vae_grads():
    vae=_VAE(); z=torch.randn(1,4,16,16)
    with torch.no_grad(): a=apply_roundtrip_adjoint_update(z,R0_AMPLITUDE_CANDIDATES[0],vae)
    with torch.inference_mode(): b=apply_roundtrip_adjoint_update(z,R0_AMPLITUDE_CANDIDATES[0],vae)
    assert bool(torch.isfinite(a).all() and torch.isfinite(b).all()) and all(p.grad is None for p in vae.parameters())
