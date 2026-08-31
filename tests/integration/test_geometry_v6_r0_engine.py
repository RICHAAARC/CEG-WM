from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cegwm.runtime.geometry_v6_sd35 as runtime
from cegwm.runtime.geometry_v6_sd35 import run_sd35_geometry_v6_r0_arm

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
    class C:
        def __init__(self,*args): self.measurement=None
        def __call__(self,p,i,t,k):
            if i==18: self.measurement='step18'; k=dict(k); k['latents']=k['latents']+10
            return k
    monkeypatch.setattr(runtime,'ContentAdaptiveInjectionCallback',C); monkeypatch.setattr(runtime,'apply_roundtrip_adjoint_update',lambda z,amplitude,vae:z+3); p=_P()
    out=run_sd35_geometry_v6_r0_arm(p,'prompt','content_geometry',content_key='content-key-0001',amplitude=.0025,content_assets=object(),height=256,width=256)
    a=next(x for x in p.events if x[:2]==('callback',18)); s=next(x for x in p.events if x[:2]==('scheduler',19)); c=next(x for x in p.events if x[:2]==('callback',19))
    assert out.content_measurement=='step18' and s[2]>a[2] and c[2]!=s[2] and p.events[-1]==('decode',c[2])

def test_notebook_has_no_geometry_secret_and_fixed_handoff():
    nb=json.loads(Path('notebooks/geometry_v6_r0_colab.ipynb').read_text()); text=json.dumps(nb)
    assert nb['cells'][0]['source']==['from google.colab import drive\n',"drive.mount('/content/drive')\n"]
    assert all(c['execution_count'] is None and c['outputs']==[] for c in nb['cells'])
    assert 'GEOMETRY_KEY' not in text and 'wrong_geometry' not in text and '--amplitude' not in text
