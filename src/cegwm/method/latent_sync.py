"""Public intermediate-latent anchors; geometry supplies no content evidence.

All H matrices map reference pixel centers to observed pixel centers. Template
matching is a development mechanism, not evidence of real VAE equivariance.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import minimize
from PIL import Image


@dataclass(frozen=True)
class AnchorSpec:
    rms: float = 0.04
    seed: int = 741091
    step_index: int = 18

    def __post_init__(self):
        if not np.isfinite(self.rms) or self.rms < 0 or self.step_index != 18:
            raise ValueError('anchor RMS must be nonnegative; injection is step 18')


def public_template(channels, height, width, spec=AnchorSpec()):
    """Five asymmetric locations, three widths, fixed public channel codes."""
    if min(channels, height, width) < 2:
        raise ValueError('template dimensions must exceed one')
    y, x = np.mgrid[:height, :width]
    x = (x + .5) / width; y = (y + .5) / height
    rng = np.random.default_rng(spec.seed)
    template = np.zeros((channels, height, width), dtype=np.float64)
    for cx, cy, sigma in ((.19,.22,.035),(.73,.17,.055),(.38,.54,.075),
                          (.81,.69,.035),(.22,.84,.055)):
        r2 = (x-cx)**2 + (y-cy)**2
        blob = np.exp(-r2/(2*sigma**2)) - .25*np.exp(-r2/(8*sigma**2))
        code = rng.choice((-1.,1.), channels)
        template += code[:,None,None]*blob
    template -= template.mean(axis=(1,2),keepdims=True)
    return template / np.sqrt(np.mean(template**2))


def similarity(shape, angle=0., scale=1., tx=0., ty=0.):
    h,w = shape; center = np.array([(w-1)/2, (h-1)/2])
    theta = np.deg2rad(angle)
    linear = scale*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    H = np.eye(3); H[:2,:2] = linear
    H[:2,2] = center + np.array([tx,ty])-linear@center
    return H


def warp_field(field, forward):
    """Render CHW field with inverse-H bilinear sampling and zero fill."""
    inverse = np.linalg.inv(forward)
    matrix = inverse[:2,:2][::-1,::-1]; offset = inverse[:2,2][::-1]
    return np.stack([affine_transform(c,matrix,offset,order=1,mode='constant',cval=0.,prefilter=False)
                     for c in field])


def latent_to_rgb_h(H, latent_shape, rgb_shape):
    lh,lw = latent_shape; rh,rw = rgb_shape
    C = np.array([[rw/lw,0,(rw/lw-1)/2],[0,rh/lh,(rh/lh-1)/2],[0,0,1.]])
    return C@H@np.linalg.inv(C)


def rectify_once(image, reference_to_observed):
    """One RGB resampling, full fixed canvas, no scoring-domain masking."""
    H = np.asarray(reference_to_observed, dtype=float)
    if H.shape != (3,3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-10:
        raise ValueError('invalid homography')
    # Pillow uses corner coordinates; convert from pixel-center coordinates.
    plus = np.array([[1,0,.5],[0,1,.5],[0,0,1.]])
    sampling = plus@H@np.linalg.inv(plus)
    sampling /= sampling[2,2]
    return image.transform(image.size,Image.Transform.PERSPECTIVE,tuple(sampling.ravel()[:8]),
                           resample=Image.Resampling.BILINEAR,fillcolor=(0,0,0))


def estimate_similarity(observation, spec=AnchorSpec()):
    """Coarse hypotheses followed by continuous template-only optimization.

    Search does not query content scores. Confidence is descriptive only; even
    an unmarked image may produce H. Entire pre/post path needs calibration.
    """
    obs = np.asarray(observation,dtype=float)
    if obs.ndim != 3 or not np.isfinite(obs).all():
        raise ValueError('finite CHW observation required')
    c,h,w = obs.shape
    template = public_template(c,h,w,spec)
    obs = obs-gaussian_filter(obs,(0,3,3))
    norm = np.linalg.norm(obs)
    if norm < 1e-12:
        return dict(H=np.eye(3), parameters=[0.,1.,0.,0.], correlation=0.)
    def objective(p):
        rendered = warp_field(template,similarity((h,w),*p))
        rendered -= gaussian_filter(rendered,(0,3,3))
        return -float(np.sum(obs*rendered)/(norm*np.linalg.norm(rendered)+1e-12))
    bounds = [(-25,25),(.6,1.45),(-.15*w,.15*w),(-.15*h,.15*h)]
    seeds = [(a,s,tx*w,ty*h) for a in (-20,-10,0,10,20) for s in (.75,1.,1.25)
             for tx,ty in ((0,0),(-.08,0),(.08,0),(0,-.08),(0,.08))]
    ranked = sorted((objective(p),p) for p in seeds)
    candidates = [minimize(objective,p,method='Powell',bounds=bounds,
                 options={'maxiter':35,'xtol':1e-3,'ftol':1e-5}) for _,p in ranked[:3]]
    result = min(candidates,key=lambda r:r.fun)
    return dict(H=similarity((h,w),*result.x),parameters=result.x.tolist(),
                correlation=-float(result.fun),optimizer_success=bool(result.success))
