"""Actual-dtype fixed-component checks; synthetic engineering, not model evidence."""
import pytest
import torch
from cegwm.method.fixed_lf_hf import FixedLFReference, solve_fixed_lf_hf, prepare_fixed_lf_reference, embed_fixed_lf_hf
from cegwm.method.survival_allocator import allocation_from_logits

pytestmark=pytest.mark.unit


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32,torch.bfloat16])
def test_fixed_actual_lf_and_cross_term_budget(dtype):
    base=torch.ones(1,1,8,8,dtype=dtype)
    lf_image=(base.double()+.008).to(dtype)
    reference=FixedLFReference(base.clone(),lf_image.clone(),1.,1.,.5,.5)
    frozen=reference.lf_image.clone()
    for slope in (.2,1.,3.):
        hf=torch.linspace(-slope,slope+1,64).reshape_as(base).double()*.01
        image,measurement=solve_fixed_lf_hf(reference,hf)
        assert torch.equal(reference.lf_image,frozen)
        delta_l=reference.lf_image.double()-base.double()
        delta_h=image.double()-reference.lf_image.double()
        assert torch.equal(image.double()-base.double(),delta_l+delta_h)
        total=float(torch.sum((image.double()-base.double())**2)/torch.sum(base.double()**2))
        recomposed=measurement['lf_actual_relative_l2']**2+measurement['hf_actual_relative_l2']**2+measurement['lf_hf_cross_relative_squared']
        assert total==pytest.approx(recomposed,abs=1e-15)
        assert 0<measurement['combined_relative_l2']<=.012
        assert measurement['hf_actual_relative_l2']>0
        assert measurement['spectral_lf_invariance_claim'] is False
    assert measurement['lf_hf_cross_relative_squared']!=0


def test_overbudget_lf_and_zero_hf_are_explicit_failures():
    base=torch.ones(1,1,8,8,dtype=torch.float64)
    bad=FixedLFReference(base,base+.02,1.,1.,.5,.5)
    with pytest.raises(RuntimeError,match='LF_ALONE_EXCEEDS'):
        solve_fixed_lf_hf(bad,torch.ones_like(base))
    good=FixedLFReference(base,base+.005,1.,1.,.5,.5)
    with pytest.raises(RuntimeError,match='NO_NONZERO_HF_FOUND'):
        solve_fixed_lf_hf(good,torch.zeros_like(base))


def test_reference_scale_once_and_same_base_required(monkeypatch):
    from cegwm.method import fixed_lf_hf as method
    base=torch.ones(1,1,8,8,dtype=torch.float32)
    calls=[]
    def deltas(*args,**kwargs):
        calls.append(kwargs)
        return torch.ones_like(base,dtype=torch.float64)*.005,torch.linspace(-.02,.02,64).reshape_as(base).double()
    monkeypatch.setattr(method,'_content_unweighted_branch_deltas',deltas)
    allocation=allocation_from_logits([0]*4)
    reference=prepare_fixed_lf_reference(base,b'key',None,None,allocation,1.2)
    saved=reference.lf_image.clone()
    embed_fixed_lf_hf(base,b'key',None,None,allocation,reference)
    embed_fixed_lf_hf(base,b'key',None,None,allocation_from_logits([.5,0,0,0]),reference)
    assert torch.equal(saved,reference.lf_image)
    assert calls==[{}, {'hf_weight_interpolation':'bilinear'}, {'hf_weight_interpolation':'bilinear'}]
    with pytest.raises(RuntimeError,match='BASE_REPLAY_MISMATCH'):
        embed_fixed_lf_hf(base+1,b'key',None,None,allocation,reference)

def test_original_cancellation_does_not_hide_overbudget_lf(monkeypatch):
    from cegwm.method import fixed_lf_hf as method
    base=torch.ones(1,1,8,8,dtype=torch.float64)
    monkeypatch.setattr(method,'_content_unweighted_branch_deltas',lambda *a,**k:(base*.02,base*-.018))
    with pytest.raises(RuntimeError,match='LF_ALONE_EXCEEDS_TOTAL_BUDGET'):
        prepare_fixed_lf_reference(base,b'key',None,None,allocation_from_logits([0]*4),1.)
