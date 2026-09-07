from dataclasses import dataclass
from types import SimpleNamespace
import json
import pytest
from PIL import Image
from cegwm.geometry_v7.contracts import GeometryEstimate
from cegwm.geometry_v7.syncseal import SyncSealTorchScript
from diagnostics.continuous_corners_v1 import runner
from test_continuous_corners import FractionalFixture

pytestmark=pytest.mark.unit

@dataclass
class Statistic:
    value: float

def test_shared_raw_and_forced_post_do_not_change_direct_route():
    fixture=FractionalFixture()
    values=iter([2.,-3.,-4.])
    row=runner.observe(Image.new('RGB',(512,512)),SyncSealTorchScript(fixture),lambda _:Statistic(next(values)))
    assert fixture.calls==1
    assert [p['statistic']['value'] for p in row['posts'].values()]==[-3.,-4.]
    assert all(r['route']=='DIRECT_POSITIVE' for r in row['derived_routes'].values())

def test_one_geometry_failure_and_pre_error_preserve_other_post(monkeypatch):
    pair=runner.estimate_pair(SyncSealTorchScript(FractionalFixture()),Image.new('RGB',(512,512)))
    pair['rounded']=GeometryEstimate.error_record(ValueError('fixture failure'))
    monkeypatch.setattr(runner,'estimate_pair',lambda *args:pair)
    calls=[]
    def score(image):
        calls.append(image)
        if len(calls)==1: raise RuntimeError('pre failure')
        return Statistic(3.)
    row=runner.observe(Image.new('RGB',(512,512)),None,score)
    assert len(calls)==2
    assert row['pre']['error'] and row['posts']['rounded']['error']
    assert row['posts']['continuous']['statistic']['value']==3.
    assert all(r['route']=='ERROR_FAIL_CLOSED' for r in row['derived_routes'].values())

def test_failed_generation_retains_fixed_denominator_and_manual_boundary(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'validate_runtime',lambda *args:'fixture')
    monkeypatch.setattr(runner,'public_key_digest',lambda key:'fixture')
    def failed(*args,**kwargs): raise RuntimeError('generation fixture')
    monkeypatch.setattr(runner,'run_content_iss_evaluation_pair',failed)
    session=runner.Session(None,None,'fixture',tmp_path/'output')
    pilot=session.pilot()
    assert len(session.rows)==18 and pilot['actual_score_calls']==0
    assert pilot['complete_statistics']==0 and pilot['route_errors']==54
    with pytest.raises(ValueError): session.pilot()
    summary=session.remaining()
    assert summary['recorded_observations']==72 and summary['route_errors']==216
    assert summary['actual_score_calls']==0
    rows=[json.loads(x) for x in (session.output/'rows.jsonl').read_text().splitlines()]
    assert len({(r['unit_id'],r['arm'],r['condition']) for r in rows})==72
    with pytest.raises(ValueError): session.remaining()

def test_failed_cg_embedding_keeps_g_and_u(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'validate_runtime',lambda *args:'fixture')
    monkeypatch.setattr(runner,'public_key_digest',lambda key:'fixture')
    content=Image.new('RGB',(512,512),'red')
    plain=Image.new('RGB',(512,512),'blue')
    def pair(*args,**kwargs):
        assert kwargs==dict(height=512,width=512,seed=2026112101)
        return SimpleNamespace(image=content,primary_null=plain)
    monkeypatch.setattr(runner,'run_content_iss_evaluation_pair',pair)
    backend=SyncSealTorchScript(FractionalFixture())
    def embed(image,strength):
        assert strength==.75
        if image is content: raise RuntimeError('CG embedding failure')
        assert image is plain
        return image.copy()
    monkeypatch.setattr(backend,'embed_final_rgb',embed)
    assets=SimpleNamespace(content_assets=SimpleNamespace(iss_assets=None),geometry_backend=backend)
    session=runner.Session(None,assets,'fixture',tmp_path/'output')
    session.score=lambda image:Statistic(0.)
    receipt=session.pilot()
    assert receipt['actual_score_calls']==36
    assert len(session.rows)==18
    assert all(r['pre']['statistic'] is None for r in session.rows if r['arm']=='CG')
    assert all(r['pre']['statistic'] is not None for r in session.rows if r['arm'] in ('G','U'))
