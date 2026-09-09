"""CPU-only observation and failure-flow checks; no model downloads."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from experiments.run_latent_rotation_cache_diagnostic import encode_reader_observation,cache_observations,diagnose_cache,SOURCE_NAMES
from cegwm.runtime.observation import encode_final_rgb_image

pytestmark=pytest.mark.unit


class Processor:
    config={'do_normalize':True}
    def preprocess(self,image):return torch.tensor(np.asarray(image).copy()).permute(2,0,1)[None].float()/127.5-1


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1,dtype=torch.float16))
        self.config=SimpleNamespace(shift_factor=.0609,scaling_factor=1.5305)
    def encode(self,pixels):
        assert pixels.dtype==torch.float16
        value=pixels[:,:,::8,::8]*.371
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value))


def test_cache_preserves_original_reader_rounding_order():
    image=Image.fromarray(np.random.default_rng(9).integers(0,256,(32,32,3),dtype=np.uint8))
    runtime=SimpleNamespace(vae=VAE(),image_processor=Processor(),model_id='fake-CPU')
    expected=encode_final_rgb_image(image,runtime.image_processor,runtime.vae)[0].float().numpy()
    actual,metadata=encode_reader_observation(image,runtime)
    assert np.array_equal(actual,expected)
    assert metadata['normalized_observation_dtype']=='torch.float16'
    assert metadata['stored_dtype']=='float32' and metadata['distribution'].startswith('latent_dist.mode')


def test_eight_encode_units_and_missing_image_failures_remain(tmp_path):
    source=tmp_path/'input';out=tmp_path/'output';source.mkdir();out.mkdir()
    for name in SOURCE_NAMES[:3]:Image.new('RGB',(512,512),(10,20,30)).save(source/(name+'.png'))
    calls=[];retained=[]
    def encoder(image,runtime):
        calls.append(image)
        return np.ones((4,64,64),dtype=np.float32),{'test':'fake'}
    rows,count=cache_observations(source,out,runtime=None,encoder=encoder,on_row=retained.append)
    assert len(rows)==len(retained)==8 and count==len(calls)==6
    assert sum(r['error'] is not None for r in rows)==2
    assert len(list(out.glob('*.npz')))==6
    assert np.asarray(calls[0])[0,0].tolist()==[10,20,30]
    assert np.asarray(calls[1])[0,0].tolist()==[0,0,0]


def test_cached_analysis_does_not_pass_truth_to_blind_optimizer(tmp_path,monkeypatch):
    import experiments.latent_rotation_objective_diagnostic as analysis
    values=np.ones((4,64,64),dtype=np.float32);np.savez_compressed(tmp_path/'one.npz',observation=values)
    calls=[]
    def analyze(observation):
        calls.append(observation)
        return {'optimizers':{'original':{'angle':-11.},'tight':{'angle':-11.001}}}
    monkeypatch.setattr(analysis,'analyze_observation',analyze)
    monkeypatch.setattr(analysis,'center_sensitivity',lambda observation,center:{'diagnostic':True})
    rows=[dict(source='plain',condition='rotation',error=None,cache_file='one.npz')]
    result=diagnose_cache(tmp_path,rows)
    assert len(calls)==1 and np.array_equal(calls[0],values)
    assert result[0]['angle_errors_degrees']['original']==-1.
    assert result[0]['error'] is None
