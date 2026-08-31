from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cegwm.runtime.geometry_v6_sd35 as runtime
from cegwm.runtime.geometry_v6_sd35 import run_sd35_geometry_v6_r0_arm
from experiments import geometry_v6_r0_engine as engine

class _D:
    def __init__(self,x): self.x=x
    def mode(self): return self.x
class _V(torch.nn.Module):
    def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.tensor(1.0)); self.config=SimpleNamespace(scaling_factor=1.0,shift_factor=0.0)
    def decode(self,x,return_dict=True): return SimpleNamespace(sample=x*self.p)
    def encode(self,x): return SimpleNamespace(latent_dist=_D(x*self.p))
class _P:
    def __init__(self): self.vae=_V(); self.events=[]
    def __call__(self,**kwargs):
        cb=kwargs['callback_on_step_end']; z=torch.zeros(1,4,16,16)
        for i in range(20):
            z=z+1; self.events.append(('scheduler',i,float(torch.linalg.vector_norm(z)))); z=cb(self,i,None,{'latents':z})['latents']; self.events.append(('callback',i,float(torch.linalg.vector_norm(z))))
        self.events.append(('decode',float(torch.linalg.vector_norm(z)))); return SimpleNamespace(images=[Image.fromarray(np.zeros((16,16,3),dtype=np.uint8),'RGB')])

def test_geometry_only_runs_step19_after_scheduler_before_decode(monkeypatch):
    monkeypatch.setattr(runtime,'apply_roundtrip_adjoint_update',lambda z,amplitude,vae:z+3)
    p=_P(); run_sd35_geometry_v6_r0_arm(p,'prompt','geometry_only',content_key=None,amplitude=.0025,content_assets=None,height=256,width=256)
    s=next(x for x in p.events if x[:2]==('scheduler',19)); c=next(x for x in p.events if x[:2]==('callback',19))
    assert c[2]!=s[2] and p.events[-1]==('decode',c[2])

def test_combined_path_keeps_content18_then_public_pilot19_then_decode(monkeypatch):
    writes=[]
    class C:
        def __init__(self,*args): self.measurement=None
        def __call__(self,p,i,t,k):
            if i==18: self.measurement='step18'; k=dict(k); k['latents']=k['latents']+10
            return k
    monkeypatch.setattr(runtime,'ContentAdaptiveInjectionCallback',C); monkeypatch.setattr(runtime,'apply_roundtrip_adjoint_update',lambda z,amplitude,vae:writes.append((amplitude,float(z.mean()))) or z+3); p=_P()
    out=run_sd35_geometry_v6_r0_arm(p,'prompt','content_geometry',content_key='content-key-0001',amplitude=.0025,content_assets=object(),height=256,width=256)
    a=next(x for x in p.events if x[:2]==('callback',18)); s=next(x for x in p.events if x[:2]==('scheduler',19)); c=next(x for x in p.events if x[:2]==('callback',19))
    assert out.content_measurement=='step18' and s[2]>a[2] and c[2]!=s[2] and p.events[-1]==('decode',c[2]) and writes==[(.0025,30.0)]

def test_notebook_has_no_geometry_secret_and_fixed_handoff():
    nb=json.loads(Path('notebooks/geometry_v6_r0_colab.ipynb').read_text()); text=json.dumps(nb)
    assert nb['cells'][0]['source']==['from google.colab import drive\n',"drive.mount('/content/drive')\n"]
    assert all(c['execution_count'] is None and c['outputs']==[] for c in nb['cells'])
    for constant in ("APPROVED_EXACT='76b7fdbbeb4ba0d97f7cdfc3411d1082701ddccb'", "R0_UNIT_ID='geometry-v6-r0-dev-0001'", "PROMPT='A watchmaker sorting steel springs beneath a magnifying lamp'", 'SEED=2026082400', 'HEIGHT=512', 'WIDTH=512'):
        assert constant in text
    assert 'CEG_WM_GEOMETRY_V6_APPROVED_EXACT' not in text and 'CEG_WM_GEOMETRY_V6_R0_PROMPT' not in text and 'CEG_WM_GEOMETRY_V6_R0_SEED' not in text
    assert 'GEOMETRY_KEY' not in text and 'wrong_geometry' not in text and '--amplitude' not in text
    for flag in ('--expected-exact','--prompt','--seed','--height','--width'):
        assert flag in text

def test_public_pilot_pairs_present_arms_only_with_their_absent_baselines():
    present={'status':'success','public_pilot_observation':{'aggregate_score':2.,'search_score':2.,'fit_score':2.,'validate_score':2.}}
    absent={'status':'success','public_pilot_observation':{'aggregate_score':1.,'search_score':1.,'fit_score':1.,'validate_score':1.}}
    delta=engine._pilot_present_vs_absent(present,absent)
    assert delta['status']=='RAW_OBSERVABILITY_ONLY' and delta['aggregate_delta']==1.
    source=Path('experiments/geometry_v6_r0_engine.py').read_text()
    assert '_pilot_present_vs_absent(combined_record, content_only_record)' in source
    assert '_pilot_present_vs_absent(geometry_only_record, unwatermarked_record)' in source
    assert '_pilot_present_vs_absent(combined_record, geometry_only_record)' not in source
