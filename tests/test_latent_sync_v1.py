import numpy as np
import pytest
from PIL import Image
from cegwm.method.latent_sync import (public_template,similarity,warp_field,
    estimate_similarity,latent_to_rgb_h,rectify_once)


def test_template_public_and_multichannel():
    template=public_template(16,32,32)
    assert np.array_equal(template,public_template(16,32,32))
    assert np.sqrt(np.mean(template**2)) == pytest.approx(1.)
    assert not np.allclose(template,template[:,:,::-1])


def test_similarity_recovers_offcenter_scale_rotation_translation():
    template=public_template(8,32,32)
    truth=similarity((32,32),10,.85,1.2,-.8)
    observed=warp_field(template,truth)
    result=estimate_similarity(observed)
    points=np.array([[4,9,1],[25,7,1],[18,25,1]]).T
    assert np.max(np.abs(result['H']@points-truth@points)) < .35


def test_coordinate_conversion_and_rgb_rectification_direction():
    latent=similarity((8,8),0,1,1,-1)
    H=latent_to_rgb_h(latent,(8,8),(32,32))
    assert np.allclose(H[:2,2],[4,-4])
    pixels=np.zeros((32,32,3),dtype=np.uint8)
    pixels[9,17]=255  # observed point: canonical (13,13) + (4,-4)
    recovered=np.asarray(rectify_once(Image.fromarray(pixels),H))
    assert np.array_equal(recovered[13,13],[255,255,255])
    assert np.count_nonzero(recovered)==3


def test_zero_observation_is_descriptive_not_positive():
    result=estimate_similarity(np.zeros((4,16,16)))
    assert result['correlation']==0
    assert 'positive' not in result


def test_callback_injects_after_content_once_at_18():
    import torch
    from cegwm.runtime.latent_sync_sd35 import LatentAnchorCallback
    from cegwm.method.latent_sync import AnchorSpec
    calls=[]
    def content(pipe,step,timestep,state):
        calls.append(step)
        return dict(latents=state['latents']+2) if step==18 else state
    callback=LatentAnchorCallback(content,AnchorSpec(rms=.03))
    state=dict(latents=torch.zeros(1,4,16,16))
    assert callback(None,17,None,state) is state
    result=callback(None,18,None,state)
    assert float((result['latents']-2).square().mean().sqrt())==pytest.approx(.03,abs=1e-6)
    with pytest.raises(RuntimeError): callback(None,18,None,state)
    assert calls==[17,18]


def test_blind_score_only_content_can_be_positive(monkeypatch):
    import torch
    from types import SimpleNamespace
    import cegwm.runtime.latent_sync_sd35 as runtime
    calls=[]
    def scorer(image,key,assets):
        calls.append((key,assets,image.size))
        return SimpleNamespace(value=-2.)
    monkeypatch.setattr(runtime,'score_current_rgb',scorer)
    monkeypatch.setattr(runtime,'encode_final_rgb_image',lambda *a:torch.zeros(1,4,16,16))
    pipe=SimpleNamespace(image_processor=None,vae=None)
    assets=object(); image=Image.new('RGB',(32,32))
    result=runtime.score_image(image,b'key',assets,pipe,tau=0.)
    assert not result['positive']
    assert result['rgb_rectifications']==1
    assert calls==[(b'key',assets,(32,32))]*2


def test_fit_selector_can_select_different_objectives_and_reject_validation():
    from experiments.select_latent_sync_candidates import select_candidates
    def report(seed,geometry,loss):
        return dict(split='fit',errors=[],missing_paths=0,failed_paths=0,
          anchor_spec=dict(rms=.04,seed=seed),pair_id='fit1',prompt='a',
          local_distortion_mse_2x2=[1.]*4,
          rows=[dict(role='positive',condition=str(i),corner_rmse_pixels=geometry,
                oracle_minus_post_loss=loss) for i in range(9)])
    first=report(1,1.,3.); second=report(2,2.,1.)
    result=select_candidates([first,second],2.)
    assert result['selected']['geometry']['seed']==1
    assert result['selected']['content_tolerance']['seed']==2
    second['split']='validation'
    with pytest.raises(ValueError,match='fit only'): select_candidates([first,second],2.)
