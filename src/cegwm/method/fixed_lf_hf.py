"""Development-only fixed actual LF component and HF-only budget solving.

LF component means y_L - base in actual dtype arithmetic promoted to float64.
It is not the LF spectral projection of the final jointly perturbed latent.
"""
from dataclasses import dataclass
import math
import torch
from cegwm.method.content_adaptive import COMBINED_RELATIVE_L2
from cegwm.method.content_unweighted import _content_unweighted_branch_deltas


@dataclass(frozen=True)
class FixedLFReference:
    base: torch.Tensor
    lf_image: torch.Tensor
    reference_scale: float
    beta: float
    lf_share: float
    hf_share: float


def _ratio(base, candidate):
    base64 = base.double()
    return float((torch.linalg.vector_norm(candidate.double()-base64) /
                  torch.linalg.vector_norm(base64)).item())


def prepare_fixed_lf_reference(base, key, hf_assets, lf_assets, allocation, beta):
    """Determine a_ref once using original v2 LF+HF, then freeze cast(base+a_ref L)."""
    if not math.isfinite(float(beta)) or not 1 <= float(beta) <= 2:
        raise ValueError('FIXED_LF_INVALID_ISS_BETA')
    if not bool(torch.isfinite(base).all()) or float(torch.linalg.vector_norm(base.double())) == 0:
        raise ValueError('FIXED_LF_INVALID_BASE')
    lf, hf = _content_unweighted_branch_deltas(base,key,hf_assets,lf_assets,allocation)
    lf = lf * float(beta)
    base64 = base.double()
    low, high = 0., 2.
    for _ in range(96):
        middle = (low+high)/2
        candidate = (base64+middle*(lf+hf)).to(base.dtype)
        if _ratio(base,candidate) <= COMBINED_RELATIVE_L2:
            low = middle
        else:
            high = middle
    lf_image = (base64+low*lf).to(base.dtype)
    ratio = _ratio(base,lf_image)
    if ratio > COMBINED_RELATIVE_L2:
        raise RuntimeError('FIXED_LF_ALONE_EXCEEDS_TOTAL_BUDGET')
    if ratio == 0:
        raise RuntimeError('FIXED_LF_ACTUAL_COMPONENT_ZERO')
    return FixedLFReference(base.detach().clone(),lf_image.detach().clone(),low,float(beta),
                            allocation.lf_branch_share,allocation.hf_branch_share)


def solve_fixed_lf_hf(reference, hf_direction):
    """Solve only positive HF amplitude; every accepted quantized candidate is checked."""
    base, lf_image = reference.base, reference.lf_image
    base64, lf64, direction = base.double(), lf_image.double(), hf_direction.double()
    base_norm = torch.linalg.vector_norm(base64)
    lf_actual = lf64-base64
    if _ratio(base,lf_image) > COMBINED_RELATIVE_L2:
        raise RuntimeError('FIXED_LF_ALONE_EXCEEDS_TOTAL_BUDGET')
    if direction.shape != base.shape or not bool(torch.isfinite(direction).all()):
        raise ValueError('FIXED_LF_INVALID_HF_DIRECTION')
    aa = float(torch.sum(direction*direction))
    if aa <= 0:
        raise RuntimeError('FIXED_LF_NO_NONZERO_HF_FOUND_IN_BOUNDED_SEARCH')
    bb = 2*float(torch.sum(lf_actual*direction))
    cc = float(torch.sum(lf_actual*lf_actual)) - float(base_norm*COMBINED_RELATIVE_L2)**2
    discriminant_root = math.sqrt(max(0.,bb*bb-4*aa*cc))
    root = ((-2*cc)/(bb+discriminant_root) if bb >= 0 and cc < 0
            else (-bb+discriminant_root)/(2*aa))
    if root <= 0 or not math.isfinite(root):
        raise RuntimeError('FIXED_LF_NO_NONZERO_HF_FOUND_IN_BOUNDED_SEARCH')
    # Continuous root only locates a bounded search; actual-dtype quantization may
    # be nonmonotone. This is conservative feasibility search, not global optimization.
    low, high = 0., root*2
    best, scale = lf_image, 0.
    for _ in range(96):
        middle = (low+high)/2
        candidate = (lf64+middle*direction).to(base.dtype)
        if _ratio(base,candidate) <= COMBINED_RELATIVE_L2:
            low, best, scale = middle, candidate, middle
        else:
            high = middle
    hf_actual = best.double()-lf64
    hf_norm = torch.linalg.vector_norm(hf_actual)
    if float(hf_norm) == 0:
        raise RuntimeError('FIXED_LF_NO_NONZERO_HF_FOUND_IN_BOUNDED_SEARCH')
    ratio = _ratio(base,best)
    if ratio > COMBINED_RELATIVE_L2:
        raise RuntimeError('FIXED_LF_TOTAL_BUDGET_EXCEEDED')
    # Exact decomposition in float64 of representable actual-dtype images.
    cross = 2*float(torch.sum(lf_actual*hf_actual))/(float(base_norm)**2)
    return best, {'combined_relative_l2':ratio,
        'lf_actual_relative_l2':float(torch.linalg.vector_norm(lf_actual)/base_norm),
        'hf_actual_relative_l2':float(hf_norm/base_norm),
        'lf_hf_cross_relative_squared':cross, 'reference_scale':reference.reference_scale,
        'hf_scale':scale, 'lf_branch_share':reference.lf_share,'hf_branch_share':reference.hf_share,
        'component_definition':'delta_L=cast(base+a_ref*L)-base; delta_H=final-cast(base+a_ref*L)',
        'spectral_lf_invariance_claim':False}


def embed_fixed_lf_hf(base,key,hf_assets,lf_assets,allocation,reference):
    if base.dtype != reference.base.dtype or not torch.equal(base,reference.base):
        raise RuntimeError('FIXED_LF_BASE_REPLAY_MISMATCH')
    if (allocation.lf_branch_share,allocation.hf_branch_share) != (reference.lf_share,reference.hf_share):
        raise RuntimeError('FIXED_LF_REFERENCE_SHARE_MISMATCH')
    _, hf = _content_unweighted_branch_deltas(base,key,hf_assets,lf_assets,allocation,
                                             hf_weight_interpolation='bilinear')
    return solve_fixed_lf_hf(reference,hf)
