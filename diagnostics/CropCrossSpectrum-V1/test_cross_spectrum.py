"""Synthetic CPU checks only: no real image experiment or model downloads."""
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
import pytest
import torch

HERE=Path(__file__).resolve().parent
REPO=next(p for p in (HERE.parents[1],HERE.parents[1]/'alive/CEG-WM') if (p/'src/cegwm').is_dir())
sys.path[:0]=[str(HERE),str(HERE.parent/'CropResidual-V1'),str(HERE.parent/'CropMechanism-V1'),str(REPO),str(REPO/'src')]
import run_cross_spectrum as d
spec=importlib.util.spec_from_file_location('prior_residual_tests',HERE.parent/'CropResidual-V1/test_residual.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
torch.set_num_threads(1)


@pytest.fixture(scope='module')
def real_probe():
    from cegwm.method.hf import FrozenHFPublicAssets,_spec
    from cegwm.shared.keys import normalize_detection_key
    from cegwm.runtime.content_weighted_joint_sd35 import derive_stability_wrong_keys
    from cegwm.method.frequency import reconstruct_frequency_carrier
    key=normalize_detection_key('00'*32)
    hf=FrozenHFPublicAssets(SimpleNamespace(encode=lambda:None),SimpleNamespace(preprocess=lambda:None),'test')
    reader=SimpleNamespace(hf=hf,keys=(key,*derive_stability_wrong_keys(key)))
    carrier=reconstruct_frequency_carrier(key,(1,16,64,64),_spec(hf),dtype=torch.float32,device='cpu')
    return d.FixedProbe(reader),carrier


def test_actual_candidate_keys_phase_gain_and_noncommon_channel_response(real_probe):
    probe,carrier=real_probe
    baseline=probe.score(carrier)
    assert baseline['C']['registered']==pytest.approx(1.,abs=1e-12)
    assert baseline['C']['top1'] and baseline['C']['gap']>.5
    for shift in d.SHIFTS[1:]:
        scores=probe.score(d.robust_phase_shift(carrier,*shift))
        np.testing.assert_allclose(list(scores['C']['scores'].values()),list(baseline['C']['scores'].values()),atol=1e-12,rtol=0)
    scores=probe.score(carrier*3)
    assert scores['C']['registered']==pytest.approx(1.,abs=1e-12)
    changed=carrier.clone()
    changed[:,::2]=torch.roll(changed[:,::2],1,dims=-1)
    assert probe.score(changed)['C']['registered']<.9
    with pytest.raises(RuntimeError,match='Undefined HF'):
        probe.score(torch.zeros_like(carrier))
    single=carrier.clone()
    single[:,1:]=0
    with pytest.raises(ValueError,match='Undefined cross-spectrum'):
        probe.score(single)


def test_cross_features_match_direct_complex_formula_and_common_gain(real_probe):
    probe,_=real_probe
    z=torch.randn(1,16,64,64,generator=torch.Generator().manual_seed(45),dtype=torch.float64)
    actual=d.cross_features(z,probe.mask)
    f=torch.fft.rfft2(z,norm='ortho')[0,:,probe.mask]
    expected=torch.cat([f[c]*f[e].conj()/f.abs().square().sum(0) for c in range(16) for e in range(c+1,16)])
    torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)
    # A real even frequency response common to all channels preserves real images.
    spectrum=torch.fft.rfft2(z,norm='ortho')
    gain=1+torch.fft.rfftfreq(64,dtype=torch.float64)[None,:]
    filtered=torch.fft.irfft2(spectrum*gain,s=(64,64),norm='ortho')
    torch.testing.assert_close(d.cross_features(filtered,probe.mask),actual,rtol=1e-10,atol=1e-12)


def synthetic_record(own_b,own_c,role='positive'):
    wrong=np.linspace(-.08,.08,16).tolist()
    return {'status':'SCORED','readers':{'B':d.keyed([own_b,*wrong]),'C':d.keyed([own_c,*wrong])}}


def zero_records():
    return {(u,r):synthetic_record(.3,.3) if r=='positive' else synthetic_record(0,0)
            for u in d.IDS for r in d.ROLES}


def test_zero_gates_scales_ties_and_retention():
    zero=zero_records()
    gate=d.assess_zero(zero)
    assert gate['status']=='PASSED'
    assert gate['scales']['B']==pytest.approx(np.std(np.linspace(-.08,.08,16)))
    # C must keep at least 80% of B standardized median margin.
    for u in d.IDS: zero[u,'positive']=synthetic_record(.3,.2)
    assert not d.assess_zero(zero)['checks']['baseline_margin_retention_80pct']
    zero=zero_records()
    for u in d.IDS[:2]: zero[u,'positive']=synthetic_record(.3,.08)
    assert d.assess_zero(zero)['positive_top1']==6
    assert d.assess_zero(zero)['status']=='FAILED'
    zero=zero_records()
    for u in d.IDS[:3]: zero[u,'negative']=synthetic_record(0,.09)
    assert not d.assess_zero(zero)['checks']['negative_top1_below_3_of_8']
    zero=zero_records()
    for u in d.IDS: zero[u,'positive']=synthetic_record(.07,.3)
    assert d.assess_zero(zero)['status']=='REFERENCE_UNINFORMATIVE'


def test_shift_gates_prevent_flat_or_only_integer_success():
    zero=zero_records()
    gate=d.assess_zero(zero)
    records={(u,r,0,0):v for (u,r),v in zero.items()}
    for s in d.SHIFTS[1:]:
        for u in d.IDS:
            records[u,'positive',*s]=synthetic_record(.12,.26)
            records[u,'negative',*s]=synthetic_record(0,0)
    passed=d.assess_shifts(records,gate)
    assert passed['status']=='PASSED' and passed['improved_sources']==8
    for s in d.SMALL:
        for u in d.IDS: records[u,'positive',*s]=synthetic_record(.12,.08)
    failed=d.assess_shifts(records,gate)
    assert failed['status']=='FAILED'
    assert not failed['checks']['dx-2_dy+0/top1_6_of_8']


class CountReader(prior.FakeReader):
    observations=0
    fail_at_observation=None
    def observe(self,image):
        CountReader.observations+=1
        if CountReader.observations==CountReader.fail_at_observation: raise RuntimeError('synthetic interrupted encoding')
        return super().observe(image)


class FakeLinear:
    def score(self,z,mode='full'):
        return {'scores':{'registered':2+float(z.mean()),**{k:0. for k in d.LABELS[1:]}}}


class FakeProbe:
    def __init__(self,*args): self.linear=FakeLinear()
    def score(self,z): return synthetic_record(2+float(z.mean()),2+float(z.mean()))


def setup_run(monkeypatch,source):
    prior.FakeReader.fail_at=None
    prior.make_source(source)
    CountReader.observations=0
    CountReader.fail_at_observation=None
    monkeypatch.setenv('CEG_WM_ROOT_KEY','00'*32)
    monkeypatch.setattr(d,'Reader',CountReader)
    # Return only the reader dictionary, matching FixedProbe.score's public shape.
    class Probe(FakeProbe):
        def score(self,z): return super().score(z)['readers']
    monkeypatch.setattr(d,'FixedProbe',Probe)
    monkeypatch.setattr(d,'version',lambda name:'synthetic')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:'synthetic')
    monkeypatch.setattr(d.subprocess,'check_output',lambda cmd,**kw:d.EXACT if 'rev-parse' in cmd else '')


def test_zero_failure_persists_and_never_calls_any_shift(monkeypatch,tmp_path):
    source,output=tmp_path/'source',tmp_path/'out'
    setup_run(monkeypatch,source)
    before={str(p):d.sha(p) for p in source.rglob('*') if p.is_file()}
    monkeypatch.setattr(d.geometry,'rgb_shift',lambda *a,**kw:pytest.fail('Forbidden RGB shift after failed zero gate'))
    monkeypatch.setattr(d,'robust_phase_shift',lambda *a,**kw:pytest.fail('Forbidden phase probe after failed zero gate'))
    d.run(source,output,REPO)
    summary=d.read(output/'summary.json')
    assert summary['status']=='STOPPED_ZERO_GATE'
    assert CountReader.observations==16
    assert sum(r['status']=='SCORED' for r in summary['records'])==16
    assert sum(r['status']=='NOT_RUN_ZERO_GATE' for r in summary['records'])==96
    assert summary['phase_not_run_zero_gate']==96
    assert not list((output/'phase_records').rglob('*.json'))
    d.run(source,output,REPO)
    assert CountReader.observations==16
    assert before=={str(p):d.sha(p) for p in source.rglob('*') if p.is_file()}


def test_pass_gate_allows_bounded_probes_and_resume(monkeypatch,tmp_path):
    source,output=tmp_path/'source',tmp_path/'out'
    setup_run(monkeypatch,source)
    assess=d.assess_zero
    # Only orchestration is mocked here; actual gate arithmetic is tested separately.
    monkeypatch.setattr(d,'assess_zero',lambda z:{**assess(z),'status':'PASSED'})
    CountReader.fail_at_observation=18 # all zero/phase records and first RGB shift succeed
    with pytest.raises(RuntimeError,match='synthetic interrupted encoding'):
        d.run(source,output,REPO)
    done=d.record_path(output,d.IDS[0],'positive',(-2,0))
    stamp=done.stat().st_mtime_ns
    assert len(list((output/'phase_records').rglob('*.json')))==96
    CountReader.fail_at_observation=None
    d.run(source,output,REPO)
    summary=d.read(output/'summary.json')
    assert summary['status']=='COMPLETE_CANDIDATE_REJECTED'
    assert all(r['status']=='SCORED' for r in summary['records'])
    assert len(summary['records'])==112 and summary['phase_scored']==96
    assert done.stat().st_mtime_ns==stamp
    assert any(d.read(p)['status']=='OPERATIONAL_ERROR' for p in (output/'attempts').rglob('*.json'))
    calls=CountReader.observations
    d.run(source,output,REPO)
    assert CountReader.observations==calls


def test_replay_failure_prevents_candidate_gate_and_shifts(monkeypatch,tmp_path):
    source,output=tmp_path/'source',tmp_path/'out'
    setup_run(monkeypatch,source)
    summary=d.read(source/'summary.json')
    next(r for r in summary['records'] if r['route']=='oracle_h')['branches']['hf']['wrong_01']=99
    d.write(source/'summary.json',summary)
    with pytest.raises(RuntimeError,match='replay failed'): d.run(source,output,REPO)
    assert not (output/'zero_gate.json').exists()
    assert not list((output/'phase_records').rglob('*.json'))


def test_notebook_embeds_fixed_sources_and_redacts_failure_logs(tmp_path):
    nb=nbformat.read(HERE/'CEG_WM_crop_cross_spectrum_v1.ipynb',as_version=4)
    nbformat.validate(nb)
    texts=[c.source for c in nb.cells if c.cell_type=='code']
    for text in texts: compile(text,'notebook cell','exec')
    exec(next(t for t in texts if t.startswith('# Embedded')),{'script_dir':tmp_path})
    assert (tmp_path/'run_cross_spectrum.py').read_bytes()==(HERE/'run_cross_spectrum.py').read_bytes()
    assert d.sha(tmp_path/'run_residual.py')==d.SUPPORT_SHA
    assert d.sha(tmp_path/'run_mechanism.py')==d.GEOMETRY_SHA
    for forbidden in ('StableDiffusion3Pipeline.from_pretrained','_build_runtime(', 'run_sd35_plain('):
        assert forbidden not in (tmp_path/'run_cross_spectrum.py').read_text()
    setup=next(t for t in texts if 'def run_logged' in t)
    fn=next(n for n in ast.parse(setup).body if isinstance(n,ast.FunctionDef) and n.name=='run_logged')
    ns={'output':tmp_path,'child_env':{'HF_TOKEN':'fake_secret','CEG_WM_ROOT_KEY':'fake_key'},'subprocess':subprocess,'time':time}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'helper','exec'),ns)
    with pytest.raises(RuntimeError,match='exit code 2'):
        ns['run_logged']([sys.executable,'-c',"print('pip detail fake_secret');raise SystemExit(2)"],'failure')
    log=next(tmp_path.rglob('*.log')).read_text()
    assert 'pip detail [REDACTED]' in log and 'fake_secret' not in log


@pytest.mark.parametrize('complete',[False,True])
def test_notebook_plots_both_terminal_paths_with_synthetic_rows(tmp_path,complete):
    import matplotlib
    matplotlib.use('Agg')
    nb=nbformat.read(HERE/'CEG_WM_crop_cross_spectrum_v1.ipynb',as_version=4)
    code=next(c.source for c in nb.cells if c.cell_type=='code' and 'fig,axes' in c.source)
    zero=zero_records()
    gate=d.assess_zero(zero)
    rows=[{**zero[u,r],'unit_id':u,'role':r,'dx_rgb':dx,'dy_rgb':dy}
          for u in d.IDS for r in d.ROLES for dx,dy in (d.SHIFTS if complete else ((0,0),))]
    summary={'records':rows,'zero_gate':gate,'status':'COMPLETE_CANDIDATE_PASSED' if complete else 'STOPPED_ZERO_GATE'}
    exec(code,{'summary':summary,'output':tmp_path})
    assert (tmp_path/'zero_key_gaps.png').stat().st_size>10000
    assert (tmp_path/'shift_key_gaps.png').exists()==complete
