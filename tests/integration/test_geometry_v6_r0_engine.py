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


def test_failure_diagnostic_is_bounded_and_redacts_prompt_and_common_secret_forms():
    content_key = 'content-key-0001'
    token = 'hf_token-0002'
    prompt = 'A private prompt that must never enter an operational artifact'
    api_key = 'api-key-0003'
    access_token = 'access-token-0004'
    auth_token = 'auth-token-0005'
    json_token = 'json-token-0006'
    single_quoted_token = 'single-quoted-token-0007'
    try:
        raise RuntimeError(
            f'{prompt} {content_key} {token} Bearer bearer-token '
            f'api_key={api_key} access_token={access_token} auth-token={auth_token} '
            f'{{"token": "{json_token}"}} \'token\': \'{single_quoted_token}\'\nunrelated_context=retained ' + ('message ' * 100)
        )
    except RuntimeError as error:
        diagnostic = engine._failure_diagnostic(error, content_key, token, prompt)
    serialized = json.dumps(diagnostic)
    assert diagnostic['failure_class'] == 'RuntimeError'
    assert diagnostic['failure_stage'] == engine.FAILURE_STAGE
    assert len(diagnostic['sanitized_message']) <= engine.FAILURE_MESSAGE_LIMIT
    assert len(diagnostic['sanitized_traceback_tail']) <= engine.FAILURE_TRACEBACK_TAIL_LIMIT
    for secret in (prompt, content_key, token, 'bearer-token', api_key, access_token, auth_token, json_token, single_quoted_token):
        assert secret not in diagnostic['sanitized_message']
        assert secret not in diagnostic['sanitized_traceback_tail']
        assert secret not in serialized
    assert 'unrelated_context=retained' in diagnostic['sanitized_message']
    assert 'unrelated_context=retained' in diagnostic['sanitized_traceback_tail']
    assert '[REDACTED]' in serialized


def test_runtime_environment_is_public_allowlisted_and_optional_versions_can_be_unavailable(monkeypatch):
    monkeypatch.setattr(engine, '_package_version', lambda package: engine.UNAVAILABLE)
    monkeypatch.setattr(engine.torch.cuda, 'is_available', lambda: False)
    environment = engine._runtime_environment(_P())
    assert set(environment) == engine._RUNTIME_ENVIRONMENT_FIELDS
    assert environment['diffusers_version'] == engine.UNAVAILABLE
    assert environment['transformers_version'] == engine.UNAVAILABLE
    assert environment['cuda_device_name'] == engine.UNAVAILABLE
    assert environment['model_id'] == engine.MODEL_ID
    assert environment['vae_parameter_dtype'] == 'torch.float32'


def test_failed_fixed_arms_are_retained_independently_without_retry(monkeypatch, tmp_path):
    content_key = 'content-key-0001'
    token = 'hf_token-0002'
    calls = []
    monkeypatch.setenv(engine.CONTENT_KEY_ENV, content_key)
    monkeypatch.setenv('HF_TOKEN', token)
    monkeypatch.setattr(engine, '_exact', lambda repo_root, expected: expected)
    monkeypatch.setattr(engine, '_load_assets', lambda received_token: (_P(), SimpleNamespace(lf_public_assets=object(), hf_public_assets=object())))
    monkeypatch.setattr(engine, '_runtime_environment', lambda pipeline: {'runtime': 'public-test-only'})
    monkeypatch.setattr(engine, 'load_calibration_asset', lambda *paths: object())
    monkeypatch.setattr(engine, 'FrozenContentWhiteningLFPublicAssets', lambda *values: object())
    monkeypatch.setattr(engine, 'load_frozen_content_whitening_asset', lambda repo_root: object())
    monkeypatch.setattr(engine.torch, 'Generator', lambda device: SimpleNamespace(manual_seed=lambda seed: (device, seed)))
    def fail_arm(pipeline, prompt, arm, **kwargs):
        calls.append((arm, kwargs['amplitude']))
        raise RuntimeError(f'{content_key} {token} arm={arm}')
    monkeypatch.setattr(engine, 'run_sd35_geometry_v6_r0_arm', fail_arm)
    payload = engine._run(SimpleNamespace(repo_root=tmp_path, expected_exact='a' * 40, prompt='public prompt', seed=7, height=512, width=512))
    expected_calls = [('content_only', None), ('unwatermarked', None)] + [item for amplitude in engine.R0_AMPLITUDE_CANDIDATES for item in (('content_geometry', amplitude), ('geometry_only', amplitude))]
    assert calls == expected_calls
    failures = [payload['baselines']['content_only']['failure_reason'], payload['baselines']['unwatermarked']['failure_reason']]
    failures += [record[arm]['failure_reason'] for record in payload['amplitudes'] for arm in ('content_geometry', 'geometry_only')]
    assert len(failures) == len(expected_calls)
    assert all(item['failure_class'] == 'RuntimeError' and item['failure_stage'] == engine.FAILURE_STAGE for item in failures)
    assert all(content_key not in json.dumps(item) and token not in json.dumps(item) for item in failures)
    assert payload['evidence_ceiling'] == 'user_run_nonformal_colab_diagnostic; science_denominator=0'
    assert all(record['content_compatibility'] == 'NOT_EVALUABLE_OPERATIONAL_FAILURE' for record in payload['amplitudes'])
