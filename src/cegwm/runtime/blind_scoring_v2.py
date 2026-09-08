"""Optional candidate-local observation reuse. V1 scoring remains unchanged.

Only encode is shared; LF/HF transformations and per-key arithmetic retain
their original device, dtype and reduction order. No cache survives a call.
"""
import math
import torch

from cegwm.method import content_whitening as lf_method
from cegwm.method import frequency as frequency_method
from cegwm.method.hf import _spec
from cegwm.method.content_weighted_joint import weighted_joint_score
from cegwm.method.blind_detection import statistic_from_weighted_scores
from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image
from cegwm.runtime.content_weighted_joint_sd35 import blind_weighted_scores, derive_stability_wrong_keys
from cegwm.shared.keys import normalize_detection_key


def _lf(observation,key,assets):
    # content_whitening._detection_observation + score_content_whitened_lf_image.
    observation=observation.detach().to(device="cpu",dtype=torch.float32).contiguous()
    if tuple(observation.shape)!=lf_method.OBSERVATION_SHAPE or tuple(observation.stride())!=lf_method.OBSERVATION_STRIDE:
        raise ValueError("content-whitening observation shape/order differs")
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("content-whitening observation must be finite")
    carrier=lf_method.reconstruct_lf_carrier(key,lf_method.OBSERVATION_SHAPE,assets.carrier_assets,
        dtype=torch.float32,device="cpu").contiguous()
    if tuple(carrier.shape)!=lf_method.OBSERVATION_SHAPE or not bool(torch.isfinite(carrier).all()):
        raise ValueError("content-whitening reconstructed LF carrier is invalid")
    observation_dct=lf_method._detrended_dct(observation.to(torch.float64))
    carrier_dct=lf_method._detrended_dct(carrier.to(torch.float64))
    weights=lf_method.decode_whitening_weights(assets.whitening_asset).to(torch.float64)
    masks=lf_method._ring_masks()
    observation_parts=[]
    carrier_parts=[]
    for channel in range(lf_method.WHITENING_SHAPE[0]):
        for band,mask in enumerate(masks):
            weight=weights[channel,band]
            observation_parts.append(observation_dct[0,channel][mask]*weight)
            carrier_parts.append(carrier_dct[0,channel][mask]*weight)
    observed=torch.cat(observation_parts)
    expected=torch.cat(carrier_parts)
    denominator=torch.linalg.vector_norm(observed)*torch.linalg.vector_norm(expected)
    if not bool(torch.isfinite(denominator)) or float(denominator.item())<=0:
        raise ValueError("content-whitening LF matched-cosine denominator must be positive")
    score=float(torch.dot(observed,expected).div(denominator).item())
    if not math.isfinite(score) or not -1<=score<=1:
        raise ValueError("content-whitening LF matched cosine must be finite in [-1,1]")
    return score


def _hf(observation,key,assets):
    # frequency.score_frequency_image after its image-only encode.
    spec=_spec(assets)
    carrier=frequency_method.reconstruct_frequency_carrier(key,tuple(observation.shape),spec,
        dtype=torch.float32,device=observation.device)
    observed_spectrum=torch.fft.rfft2(observation.to(torch.float32),norm="ortho")
    carrier_spectrum=torch.fft.rfft2(carrier,norm="ortho")
    _,_,height,width=observation.shape
    mask=torch.from_numpy(frequency_method.radial_frequency_mask(height,width,spec)).to(observation.device)
    observed=observed_spectrum.real[...,mask].reshape(-1).to(torch.float64)
    expected=carrier_spectrum.real[...,mask].reshape(-1).to(torch.float64)
    observed=observed-observed.mean()
    expected=expected-expected.mean()
    denominator=torch.linalg.vector_norm(observed)*torch.linalg.vector_norm(expected)
    if not bool(torch.isfinite(denominator)) or float(denominator.item())==0:
        raise ValueError("blind frequency score requires non-constant finite image evidence")
    score=float(torch.dot(observed,expected).item()/denominator.item())
    if not math.isfinite(score): raise RuntimeError("blind frequency score is not finite")
    return score


def score_branches_v2(image,key,assets,*,reuse_observation=False):
    """Return complete 17-key branches and execution mode; accepts RGB, never latent."""
    current=require_ordinary_rgb_image(image)
    key=normalize_detection_key(key)
    wrong=derive_stability_wrong_keys(key)
    lf_assets=assets.content_assets.iss_assets.lf_public_assets
    hf_assets=assets.content_assets.iss_assets.hf_public_assets
    carrier=lf_assets.carrier_assets
    shared_context=carrier.vae is hf_assets.vae and carrier.image_processor is hf_assets.image_processor
    if not reuse_observation or not shared_context:
        branches=blind_weighted_scores(current,key,wrong,assets.content_assets,assets.weighted_joint_asset)
        return branches,"original" if not reuse_observation else "fallback_distinct_contexts"
    observation=encode_final_rgb_image(current,hf_assets.image_processor,hf_assets.vae)
    labels=("registered",*(f"wrong_{i:02d}" for i in range(16)))
    pairs=[(_lf(observation,k,lf_assets),_hf(observation,k,hf_assets)) for k in (key,*wrong)]
    if any(not math.isfinite(x) or not -1<=x<=1 for pair in pairs for x in pair):
        raise ValueError("blind branch scores must be finite in [-1,1]")
    branches={"lf":{label:pair[0] for label,pair in zip(labels,pairs,strict=True)},
              "hf":{label:pair[1] for label,pair in zip(labels,pairs,strict=True)},
              "weighted_joint":{label:weighted_joint_score(*pair,assets.weighted_joint_asset)
                  for label,pair in zip(labels,pairs,strict=True)}}
    return branches,"shared_candidate_observation"


def score_statistic_v2(image,key,assets,*,reuse_observation=False):
    branches,_=score_branches_v2(image,key,assets,reuse_observation=reuse_observation)
    return statistic_from_weighted_scores(branches["weighted_joint"])
