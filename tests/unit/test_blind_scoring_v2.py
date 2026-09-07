"""Real LF/HF arithmetic on an injected deterministic VAE, not model evidence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import FrozenLFPublicAssets, LF_BALANCED_BLOCKS_CARRIER_METHOD_ID, LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
from cegwm.method.content_whitening import FrozenContentWhiteningLFPublicAssets
from cegwm.runtime.blind_scoring_v2 import score_branches_v2
from experiments.run_blind_detection_v1 import load_whitening_asset_semantic, load_weighted_asset_semantic


class Processor:
    def preprocess(self,image):
        x=np.asarray(image.resize((64,64)),dtype=np.float32)/255
        return torch.from_numpy(x.copy()).permute(2,0,1)[None].contiguous()


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(()))
        self.config=SimpleNamespace(scaling_factor=.5,shift_factor=.1)
        self.calls=0
    def encode(self,x):
        self.calls+=1
        mode=torch.cat([x]*5+[x.mean(dim=1,keepdim=True)],dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:mode))


def fixture():
    vae,processor=VAE(),Processor()
    identifier="stabilityai/stable-diffusion-3.5-medium:image_processor"
    carrier=FrozenLFPublicAssets(vae,processor,identifier,LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID)
    root=Path(__file__).resolve().parents[2]
    lf=FrozenContentWhiteningLFPublicAssets(carrier,load_whitening_asset_semantic(root))
    hf=FrozenHFPublicAssets(vae,processor,identifier)
    assets=SimpleNamespace(content_assets=SimpleNamespace(iss_assets=SimpleNamespace(lf_public_assets=lf,hf_public_assets=hf)),
        weighted_joint_asset=load_weighted_asset_semantic(root))
    pixels=np.random.default_rng(37).integers(0,256,(512,512,3),dtype=np.uint8)
    return Image.fromarray(pixels),assets,vae


@pytest.mark.quick
def test_all_17_branch_values_match_with_34_to_1_encodes():
    image,assets,vae=fixture()
    key=b"v2-injected-vae-key"
    old,mode=score_branches_v2(image,key,assets)
    assert vae.calls==34 and mode=="original"
    new,mode=score_branches_v2(image,key,assets,reuse_observation=True)
    assert vae.calls==35 and mode=="shared_candidate_observation"
    assert old==new
    assert all(len(new[branch])==17 for branch in new)


@pytest.mark.quick
def test_candidate_observation_does_not_leak_across_image_key_or_context():
    image,assets,vae=fixture()
    key=b"v2-injected-vae-key"
    a,_=score_branches_v2(image,key,assets,reuse_observation=True)
    other=Image.fromarray(255-np.asarray(image))
    b,_=score_branches_v2(other,key,assets,reuse_observation=True)
    a_again,_=score_branches_v2(image,key,assets,reuse_observation=True)
    changed_key,_=score_branches_v2(image,b"v2-different-vae-key",assets,reuse_observation=True)
    assert a==a_again and a!=b and a!=changed_key and vae.calls==4
    hf=assets.content_assets.iss_assets.hf_public_assets
    assets.content_assets.iss_assets.hf_public_assets=replace(hf,vae=VAE())
    fallback,mode=score_branches_v2(image,key,assets,reuse_observation=True)
    assert mode=="fallback_distinct_contexts" and fallback==a
    assert vae.calls==21 and assets.content_assets.iss_assets.hf_public_assets.vae.calls==17


@pytest.mark.quick
def test_nonfinite_encode_is_not_cached_or_silently_replaced():
    image,assets,vae=fixture()
    original=vae.encode
    vae.encode=lambda x:SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:torch.full((1,16,64,64),float('nan'))))
    with pytest.raises(ValueError,match="finite"):
        score_branches_v2(image,b"v2-injected-vae-key",assets,reuse_observation=True)
    vae.encode=original
    assert score_branches_v2(image,b"v2-injected-vae-key",assets,reuse_observation=True)[1]=="shared_candidate_observation"
