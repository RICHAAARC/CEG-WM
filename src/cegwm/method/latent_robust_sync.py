"""Isolated four-arm CPU rotation reader; the historical reader is unchanged.

Transfer is continuous zero-extension bilinear sampling and fixed mild spatial
smoothing on both sides. It is a robustness approximation, not an exact integral
over displacement probabilities: in ideal continuous convolution, smoothing both
sides gives an effective kernel width sqrt(2)*0.5. The discrete finite kernels
and reflect boundaries only approximate that identity. Geometry correlation is
descriptive and can be negative.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import affine_transform,gaussian_filter
from scipy.optimize import minimize_scalar

from cegwm.method.latent_sync import public_template,similarity,warp_field


ARMS=('original','whitening','transfer','combined')
BOUNDS=(-15.,15.)
XATOL=.01


def _field(value):
    field=np.asarray(value,dtype=float)
    if field.ndim!=3 or field.shape[0]<1 or min(field.shape[1:])<2 or not np.isfinite(field).all():
        raise ValueError('finite CHW field with positive channels and spatial dimensions >=2 required')
    return field


def preprocess_field(field,transfer=False):
    """Fixed sigma .5 smoothing when requested, then the original sigma3 highpass."""
    value=_field(field)
    if transfer:
        value=gaussian_filter(value,(0,.5,.5),mode='reflect',truncate=4.)
    return value-gaussian_filter(value,(0,3,3),mode='reflect',truncate=4.)


def warp_continuous(field,H):
    """Reference->observed H, bilinear continuous zero extension (grid-constant)."""
    value=_field(field)
    H=np.asarray(H,dtype=float)
    if H.shape!=(3,3) or not np.isfinite(H).all() or not np.allclose(H[2],[0,0,1]):
        raise ValueError('finite affine 3x3 reference-to-observed H required')
    inverse=np.linalg.inv(H)
    matrix=inverse[:2,:2][::-1,::-1]; offset=inverse[:2,2][::-1]
    return np.stack([affine_transform(channel,matrix,offset,order=1,mode='grid-constant',
                     cval=0.,prefilter=False) for channel in value])


@dataclass
class RotationObjective:
    score: Callable[[float],float]
    processed_observation: np.ndarray
    processed_template: Callable[[float],np.ndarray]
    prewhitening_observation: np.ndarray
    W: np.ndarray | None
    diagnostics: dict


def make_objective(observation,arm='original'):
    """Construct one observation-dependent W, fixed throughout all angle queries."""
    if arm not in ARMS: raise ValueError('unknown reader arm')
    obs=_field(observation)
    c,h,w=obs.shape
    transfer=arm in ('transfer','combined')
    whitening=arm in ('whitening','combined')
    filtered=preprocess_field(obs,transfer)
    template=public_template(c,h,w)
    W=np.eye(c); whitening_diagnostics={'enabled':False}
    if whitening:
        from cegwm.method.latent_channel_whitening import estimate_whitener,apply_whitener
        W,whitening_diagnostics=estimate_whitener(filtered)
        processed=np.zeros_like(filtered) if W is None else apply_whitener(W,filtered)
    else:
        processed=filtered  # no identity matrix multiply: preserve baseline arithmetic
    norm=float(np.linalg.norm(processed))
    degenerate=(W is None or not np.isfinite(norm) or norm<1e-12)
    diagnostics=dict(arm=arm,shape=list(obs.shape),bounds=list(BOUNDS),xatol=XATOL,
        center_xy=[(w-1)/2,(h-1)/2],nonzero_processed_observation=bool(norm>=1e-12),
        observation_norm=norm,whitening=whitening_diagnostics,
        whitener_estimations=int(whitening),numerically_degenerate=bool(degenerate),
        smoothing_sigma=.5 if transfer else 0.,highpass_sigma=3.,
        warp_mode='grid-constant' if transfer else 'constant',
        application_recenters_channels=False,content_scores_used=False,truth_used=False)
    def processed_template(angle):
        if not np.isfinite(angle): raise ValueError('finite angle required')
        H=similarity((h,w),float(angle))
        rendered=warp_continuous(template,H) if transfer else warp_field(template,H)
        rendered=preprocess_field(rendered,transfer)
        if whitening:
            return np.zeros_like(rendered) if W is None else apply_whitener(W,rendered)
        return rendered
    def score(angle):
        if degenerate: return 0.
        rendered=processed_template(angle)
        # Same sign/arithmetic as the original correlation: negatives are valid.
        return float(np.sum(processed*rendered)/(norm*np.linalg.norm(rendered)+1e-12))
    return RotationObjective(score,processed,processed_template,filtered,W,diagnostics)


def estimate_robust_rotation(observation,arm='original',*,objective=None):
    """Fixed bounded scalar optimization; optional built objective avoids re-estimating W."""
    objective=make_objective(observation,arm) if objective is None else objective
    if objective.diagnostics['arm']!=arm: raise ValueError('objective arm differs')
    if objective.diagnostics['numerically_degenerate']:
        return dict(angle=0.,correlation=0.,H_reference_to_observed_latent=np.eye(3).tolist(),
                    nfev=0,nit=0,success=False,status='DEGENERATE_OBSERVATION',diagnostics=objective.diagnostics)
    result=minimize_scalar(lambda angle:-objective.score(angle),bounds=BOUNDS,
                           method='bounded',options={'xatol':XATOL})
    h,w=objective.processed_observation.shape[-2:]
    return dict(angle=float(result.x),correlation=-float(result.fun),
                H_reference_to_observed_latent=similarity((h,w),result.x).tolist(),
                nfev=int(result.nfev),nit=int(result.nit),success=bool(result.success),
                status=str(result.message),diagnostics=objective.diagnostics)
