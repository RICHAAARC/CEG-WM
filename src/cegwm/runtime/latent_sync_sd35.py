"""V2 content unchanged; step-18 public anchor replaces RGB SyncSeal embedding."""
import numpy as np
import torch
from cegwm.method.latent_sync import AnchorSpec, public_template, estimate_similarity, latent_to_rgb_h, rectify_once
from cegwm.method.content_iss import score_content_iss_image, iss_beta
from cegwm.runtime.content_iss_sd35 import ContentISSInjectionCallback
from cegwm.runtime.diffusers_sd35 import run_sd35_plain
from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image
from cegwm.runtime.paper_detection_v2 import score_current_rgb


class LatentAnchorCallback:
    def __init__(self, content_callback, spec=AnchorSpec()):
        self.content_callback = content_callback
        self.spec = spec
        self.injected = False
        self.delta_rms = None
        self.remaining_steps = 0

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        if step_index == self.spec.step_index and self.injected:
            raise RuntimeError('duplicate anchor injection')
        updated = self.content_callback(pipeline,step_index,timestep,callback_kwargs)
        if self.injected and step_index > self.spec.step_index:
            self.remaining_steps += 1
        if step_index != self.spec.step_index:
            return updated
        latent = updated['latents']
        if latent.ndim != 4 or latent.shape[0] != 1:
            raise ValueError('single NCHW latent required')
        template = public_template(*latent.shape[1:],self.spec)
        delta = torch.as_tensor(template,device=latent.device,dtype=latent.dtype)[None]*self.spec.rms
        updated = dict(updated); updated['latents'] = latent+delta
        self.delta_rms = float(delta.float().square().mean().sqrt())
        self.injected = True
        return updated


def run_pair(pipeline,prompt,key,iss_assets,seed,spec=AnchorSpec()):
    device = pipeline._execution_device
    generator = lambda: torch.Generator(device=device).manual_seed(seed)
    plain = require_ordinary_rgb_image(run_sd35_plain(pipeline,prompt,height=512,width=512,generator=generator()))
    beta = iss_beta(score_content_iss_image(plain,key,iss_assets.lf_public_assets),iss_assets.iss_asset)
    callback = LatentAnchorCallback(ContentISSInjectionCallback(key,iss_assets,beta),spec)
    result = pipeline(prompt=prompt,height=512,width=512,num_inference_steps=20,generator=generator(),
                      output_type='pil',callback_on_step_end=callback,
                      callback_on_step_end_tensor_inputs=['latents'])
    if not callback.injected or callback.remaining_steps < 1:
        raise RuntimeError('pipeline missed step 18 or remaining sampling')
    # Deliberately no embed_final_rgb: old and new sync are mutually exclusive.
    return plain,require_ordinary_rgb_image(result.images[0]),dict(anchor_delta_rms=callback.delta_rms)


def score_image(image,key,assets,pipeline,spec=AnchorSpec(),tau=None):
    """Blind production-shaped inputs; forced pre/post for development analysis.

    tau must be calibrated for this complete new path, never inherited from V2.
    Frozen pipeline is used solely as a public image processor/VAE here.
    """
    pre = score_current_rgb(image,key,assets).value
    observation = encode_final_rgb_image(image,pipeline.image_processor,pipeline.vae)
    estimate = estimate_similarity(observation[0].float().cpu().numpy(),spec)
    H = latent_to_rgb_h(estimate['H'],observation.shape[-2:],(image.height,image.width))
    recovered = rectify_once(image,H)
    post = score_current_rgb(recovered,key,assets).value
    payload = dict(pre=float(pre),post=float(post),score=float(max(pre,post)),
                   H_reference_to_observed_pixels=H.tolist(),correlation=estimate['correlation'],
                   parameters_latent=estimate['parameters'],rgb_rectifications=1)
    if tau is not None:
        if not np.isfinite(tau): raise ValueError('finite calibrated threshold required')
        payload.update(tau=float(tau),positive=max(pre,post)>tau)
    return payload


def tolerance_surface(image,key,assets,truth_reference_to_observed,offsets):
    """DEVELOPMENT ONLY: truth/oracle input must never enter score_image.

    Offsets are (angle degrees, scale, tx pixels, ty pixels). Each row resamples
    the attacked RGB once with a perturbed truth H and preserves content scorer.
    """
    from cegwm.method.latent_sync import similarity
    rows=[]
    for offset in offsets:
        H = np.asarray(truth_reference_to_observed)@similarity((image.height,image.width),*offset)
        rows.append(dict(offset=list(offset),score=float(score_current_rgb(rectify_once(image,H),key,assets).value)))
    return rows
