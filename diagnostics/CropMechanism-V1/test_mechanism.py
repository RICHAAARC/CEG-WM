"""Synthetic CPU engineering checks. No GPU or model downloads."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import nbformat
import numpy as np
from PIL import Image
import pytest
import torch

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'CropResidual-V1'
REPO=next(p for p in (HERE.parents[1],HERE.parents[1]/'alive/CEG-WM') if (p/'src/cegwm').is_dir())
sys.path[:0]=[str(HERE),str(PREVIOUS),str(REPO),str(REPO/'src')]
import run_mechanism as d
spec=importlib.util.spec_from_file_location('residual_test_support',PREVIOUS/'test_residual.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
torch.set_num_threads(1)


def test_phase_sign_norm_inverse_and_nyquist_rejection():
    x=torch.arange(64)[None,:]
    z=torch.cos(2*torch.pi*7*x/64).repeat(64,1)[None,None]
    actual=d.phase_shift(d.project(z),.25,-.5)
    expected=torch.cos(2*torch.pi*7*(x-.25)/64).repeat(64,1)[None,None]
    torch.testing.assert_close(actual,expected,atol=4e-6,rtol=1e-5)
    assert float(actual.norm()/z.norm())==pytest.approx(1,abs=1e-6)
    assert d.relative(d.phase_shift(actual,-.25,.5),d.project(z))<1e-6
    nyquist=(-1.)**x.repeat(64,1)[None,None]
    with pytest.raises(ValueError,match='real common subspace'):
        d.phase_shift(nyquist,.25,0)
    assert float(d.project(nyquist).abs().max())==0


def test_stride_alignment_and_interpolation_controls():
    z=d.project(torch.randn(1,2,64,64,generator=torch.Generator().manual_seed(12)))
    for dx,dy in ((-1,0),(1,0),(0,1)):
        torch.testing.assert_close(d.phase_shift(z,dx,dy),torch.roll(z,(dy,dx),(-2,-1)),atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(d.padded_phase(z,dx,dy),d.integer_roll(z,dx,dy,False),atol=3e-6,rtol=1e-5)
    assert d.bilinear_shift(z,.25,0,True).norm()<.9*z.norm()
    assert d.phase_shift(z,.25,0).norm()==pytest.approx(float(z.norm()),rel=1e-6)
    torch.testing.assert_close(d.padded_phase(z,0,0),z,atol=2e-6,rtol=1e-5)
    for pad in (32,64):
        assert torch.isfinite(d.padded_phase(z,.25,0,pad)).all()


def test_rgb_exact_integer_copy_boundary_and_source_unchanged():
    a=np.zeros((512,512,3),dtype=np.uint8)
    a[27:485,27:485]=np.arange(458,dtype=np.uint16)[None,:,None]%256
    image=Image.fromarray(a)
    for dx,dy in d.SHIFTS:
        actual=np.asarray(d.rgb_shift(image,dx,dy))
        np.testing.assert_array_equal(actual,np.roll(a,(dy,dx),(0,1)))
        np.testing.assert_array_equal(np.asarray(image),a)
    edge=np.full((512,512,3),127,dtype=np.uint8)
    shifted=np.asarray(d.rgb_shift(Image.fromarray(edge),-2,0))
    assert np.all(shifted[:,-2:]==0) and np.all(shifted[:,:-2]==127)


def test_cached_full_hf_matches_original_seventeen_keys():
    from cegwm.method.hf import FrozenHFPublicAssets
    from cegwm.shared.keys import normalize_detection_key
    from cegwm.runtime.content_weighted_joint_sd35 import derive_stability_wrong_keys
    from cegwm.runtime.blind_scoring_v2 import _hf
    assets=FrozenHFPublicAssets(SimpleNamespace(encode=lambda:None),SimpleNamespace(preprocess=lambda:None),'test')
    key=normalize_detection_key('00'*32)
    reader=SimpleNamespace(hf=assets,keys=(key,*derive_stability_wrong_keys(key)))
    z=torch.randn(1,16,64,64,generator=torch.Generator().manual_seed(20))
    probe=d.HFProbe(reader,tuple(z.shape),z.device)
    actual=list(probe.score(z,'full')['scores'].values())
    expected=[_hf(z,k,assets) for k in reader.keys]
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
    # Removing Nyquist does not change the common-mask score (up to FFT roundoff).
    assert probe.score(z)['registered']==pytest.approx(probe.score(d.project(z))['registered'],abs=1e-8)


def test_operator_controls_and_signed_identity():
    z=torch.randn(1,2,64,64,generator=torch.Generator().manual_seed(3))
    black=.01*torch.randn_like(z)
    fields,metrics=d.operator_fields(z,black,z,z,0,0)
    assert metrics['phase_roundtrip_relative_l2']<1e-6
    for value in fields.values():
        torch.testing.assert_close(d.project(value),d.project(z),atol=3e-6,rtol=1e-5)
    scores={name:{'registered':i*.1,'gap':i*.07} for i,name in enumerate(fields)}
    for result in d.decompose(scores).values():
        assert result['total_loss']==pytest.approx(result['phase_loss']+result['boundary_model_delta']+result['extra_vs_padded_phase'])
        assert result['extra_vs_padded_phase']==pytest.approx(result['bilinear_model_delta']+result['extra_vs_bilinear'])


class FakeProbe:
    def __init__(self,*args): pass
    def score(self,z,mode='common'):
        own=2+float(z.mean())
        return {'scores':{'registered':own,**{f'wrong_{i:02d}':0. for i in range(16)}},'registered':own,'wrong_max':0.,'gap':own}


def setup(monkeypatch):
    monkeypatch.setenv('CEG_WM_ROOT_KEY','00'*32)
    monkeypatch.setattr(d,'Reader',prior.FakeReader)
    monkeypatch.setattr(d,'HFProbe',FakeProbe)
    monkeypatch.setattr(d,'version',lambda name:'synthetic')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:'synthetic')
    monkeypatch.setattr(d.subprocess,'check_output',lambda cmd,**kw:d.EXACT if 'rev-parse' in cmd else '')
    prior.FakeReader.fail_at=None


def test_interrupt_resume_fixed_denominators_and_source_immutable(monkeypatch,tmp_path):
    source,output=tmp_path/'source',tmp_path/'output'
    prior.make_source(source)
    before={str(p.relative_to(source)):d.sha(p) for p in source.rglob('*') if p.is_file()}
    setup(monkeypatch)
    prior.FakeReader.fail_at=17
    with pytest.raises(RuntimeError,match='synthetic interruption'):
        d.run(source,output,REPO)
    assert d.read(output/'replay_gate.json')['status']=='PASSED'
    control=output/'records'/d.IDS[0]/'positive'/f'{d.shift_id(0,0)}.json'
    stamp=control.stat().st_mtime_ns
    prior.FakeReader.fail_at=None
    d.run(source,output,REPO)
    result=d.read(output/'summary.json')
    assert result['status']=='COMPLETE_DIAGNOSTIC'
    assert result['planned_records']==112==len(result['records'])
    assert all(c['scored']==c['planned']==8 for c in result['counts'].values())
    assert control.stat().st_mtime_ns==stamp
    assert before=={str(p.relative_to(source)):d.sha(p) for p in source.rglob('*') if p.is_file()}
    assert any(d.read(p)['status']=='OPERATIONAL_ERROR' for p in (output/'attempts').rglob('*.json'))
    calls=prior.FakeReader.calls
    d.run(source,output,REPO)
    assert prior.FakeReader.calls==calls


@pytest.mark.parametrize('failure',['missing','replay'])
def test_input_and_replay_gates_stop_before_probe(monkeypatch,tmp_path,failure):
    source,output=tmp_path/'source',tmp_path/'output'
    prior.make_source(source)
    if failure=='missing': next(source.rglob('oracle_h.png')).unlink()
    else:
        summary=d.read(source/'summary.json')
        next(r for r in summary['records'] if r['route']=='oracle_h')['branches']['hf']['registered']+=1
        d.write(source/'summary.json',summary)
    setup(monkeypatch)
    with pytest.raises((FileNotFoundError,RuntimeError)):
        d.run(source,output,REPO)
    assert not list((output/'records').rglob('*.json'))
    if failure=='missing': assert prior.FakeReader.calls==0
    else: assert d.read(output/'replay_gate.json')['status']=='FAILED'


def test_notebook_embeds_exact_scripts_and_logs_child_failures(tmp_path):
    nb=nbformat.read(HERE/'CEG_WM_crop_mechanism_v1.ipynb',as_version=4)
    nbformat.validate(nb)
    texts=[c.source for c in nb.cells if c.cell_type=='code']
    for text in texts: compile(text,'cell','exec')
    embedded=next(t for t in texts if t.startswith('# Embedded'))
    exec(embedded,{'script_dir':tmp_path,'script':tmp_path/'run_mechanism.py'})
    assert (tmp_path/'run_mechanism.py').read_bytes()==(HERE/'run_mechanism.py').read_bytes()
    assert d.sha(tmp_path/'run_residual.py')==d.SUPPORT_SHA
    for forbidden in ('_main_pair(', '_build_runtime(', 'run_sd35_plain(', 'StableDiffusion3Pipeline.from_pretrained'):
        assert forbidden not in embedded
    code=next(t for t in texts if 'def run_logged' in t)
    function=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='run_logged')
    ns={'output':tmp_path,'child_env':{'HF_TOKEN':'fake_secret','CEG_WM_ROOT_KEY':'fake_key'},'subprocess':subprocess,'time':time}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'helper','exec'),ns)
    with pytest.raises(RuntimeError,match='exit code 2'):
        ns['run_logged']([sys.executable,'-c',"print('detail fake_secret');raise SystemExit(2)"],'failure')
    log=next(tmp_path.rglob('*.log')).read_text()
    assert 'detail [REDACTED]' in log and 'fake_secret' not in log


def test_notebook_plot_runs_with_synthetic_rows(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    nb=nbformat.read(HERE/'CEG_WM_crop_mechanism_v1.ipynb',as_version=4)
    plot=next(c.source for c in nb.cells if c.cell_type=='code' and 'fig, axes' in c.source)
    rows=[]
    for role in d.ROLES:
        for dx,dy in d.SHIFTS:
            for i in range(8):
                fields={name:{'registered':i*.01+(dx+dy)*j*.001,'gap':i*.01+(dx+dy)*j*.0005}
                        for j,name in enumerate(d.operator_fields(torch.randn(1,1,64,64),torch.zeros(1,1,64,64),
                            torch.randn(1,1,64,64),torch.randn(1,1,64,64),0,0)[0])}
                rows.append({'role':role,'dx_rgb':dx,'dy_rgb':dy,'decomposition':d.decompose(fields),
                             'metrics':{'nyquist_removed_energy_fraction_base_residual':.01}})
    exec(plot,{'summary':{'status':'COMPLETE_DIAGNOSTIC','records':rows},'output':tmp_path})
    assert (tmp_path/'mechanism_contrasts.png').stat().st_size>10000
