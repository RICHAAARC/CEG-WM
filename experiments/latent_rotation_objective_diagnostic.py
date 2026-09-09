"""CPU-only inspection of the existing public-template rotation objective.

The sampled curve and alternate centre are diagnostics, never production H
selection. These functions accept no truth; a caller may annotate errors only
after optimizations and curves have completed. No RGB, content score, embedding
latent, prompt, per-image template, compensation, or new carrier is consumed.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

from cegwm.method.latent_sync import AnchorSpec, public_template, warp_field


BOUNDS=(-15.,15.)
CURVE_ANGLES=np.linspace(-15.,15.,121)
LOCAL_OFFSETS=(-.01,-.001,0.,.001,.01)


def rotation_matrix(center_xy,angle):
    center=np.asarray(center_xy,dtype=float)
    theta=np.deg2rad(angle)
    linear=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    H=np.eye(3); H[:2,:2]=linear; H[:2,2]=center-linear@center
    return H


def mapped_diagnostic_center(latent_shape,rgb_size=(512,512),rgb_center=(255.,255.)):
    """Map a predefined diagnostic RGB pixel centre to latent pixel centres.

    latent_shape is (height,width); rgb_size is (width,height). This centre is
    an explicit counterfactual, not a modification of the production reader.
    """
    h,w=latent_shape
    scale=np.asarray(rgb_size,dtype=float)/np.array([w,h],dtype=float)
    if np.any(scale<=0): raise ValueError('positive dimensions required')
    return ((np.asarray(rgb_center,dtype=float)-(scale-1)/2)/scale).tolist()


def _analyze_centers(observation,centers):
    """Return baseline/tight optimization, fixed curve, and centre sensitivity.

    observation: a single final-RGB VAE re-encoding, finite CHW numpy-compatible.
    centers are predefined diagnostic centres, never derived by optimization.
    """
    obs=np.asarray(observation,dtype=float)
    if obs.ndim!=3 or min(obs.shape)<2 or not np.isfinite(obs).all():
        raise ValueError('finite CHW observation required')
    c,h,w=obs.shape
    template=public_template(c,h,w,AnchorSpec())
    filtered=obs-gaussian_filter(obs,(0,3,3))
    norm=np.linalg.norm(filtered)
    for center in centers.values():
        center_array=np.asarray(center,dtype=float)
        if center_array.shape!=(2,) or not np.isfinite(center_array).all():
            raise ValueError('finite predefined diagnostic centre (x,y) required')
    report=dict(shape=list(obs.shape),bounds=list(BOUNDS),curve_step_degrees=.25,
                observation_filtered_norm=float(norm),nonzero_filtered_observation=bool(norm>=1e-12),
                center_results={},production_method_changed=False,
                curve_used_to_select_optimizer=False,truth_used_in_objective=False)
    for name,center in centers.items():
        # Algebra and interpolation match estimate_rotation exactly at default centre.
        def objective(angle):
            rendered=warp_field(template,rotation_matrix(center,angle))
            rendered-=gaussian_filter(rendered,(0,3,3))
            return -float(np.sum(filtered*rendered)/(norm*np.linalg.norm(rendered)+1e-12))
        entry=dict(center_xy=center,optimizers={})
        for label,tolerance in (('original',.01),('tight',1e-6)):
            if norm<1e-12:
                result=dict(angle=0.,correlation=0.,nfev=0,nit=0,success=True,status='zero_signal_identity')
            else:
                fit=minimize_scalar(objective,bounds=BOUNDS,method='bounded',options={'xatol':tolerance})
                result=dict(angle=float(fit.x),correlation=-float(fit.fun),nfev=int(fit.nfev),
                            nit=int(fit.nit),success=bool(fit.success),status=str(fit.message))
            result['xatol']=tolerance
            result['local_samples']=[dict(offset_degrees=offset,
                sampled_angle=float(np.clip(result['angle']+offset,*BOUNDS)),
                correlation=-objective(float(np.clip(result['angle']+offset,*BOUNDS)))) for offset in LOCAL_OFFSETS]
            for sample in result['local_samples']:
                sample['correlation_minus_optimized']=sample['correlation']-result['correlation']
            result['H_reference_to_observed_latent']=rotation_matrix(center,result['angle']).tolist()
            entry['optimizers'][label]=result
        curve=[dict(angle=float(angle),correlation=-objective(float(angle))) for angle in CURVE_ANGLES]
        # Report sampled maximum only; it does not seed or replace either result.
        maximum=max(curve,key=lambda row:row['correlation'])
        entry.update(curve=curve,curve_evaluations=len(curve),sampled_curve_maximum=maximum,
            tight_minus_original_degrees=entry['optimizers']['tight']['angle']-entry['optimizers']['original']['angle'],
            tight_minus_original_correlation=entry['optimizers']['tight']['correlation']-entry['optimizers']['original']['correlation'])
        report['center_results'][name]=entry
    return report


def analyze_observation(observation):
    """Blind original-centre report: no truth/centre override argument is accepted."""
    shape=np.shape(observation)
    if len(shape)!=3: raise ValueError('CHW observation required')
    _,h,w=shape
    report=_analyze_centers(observation,{'production':[(w-1)/2,(h-1)/2]})
    entry=report.pop('center_results')['production']
    return dict(**report,**entry)


def center_sensitivity(observation,center):
    """Separate diagnostic counterfactual; never substitutes the production result."""
    report=_analyze_centers(observation,{'diagnostic':list(center)})
    entry=report.pop('center_results')['diagnostic']
    return dict(**report,**entry,diagnostic_center_override=True)
