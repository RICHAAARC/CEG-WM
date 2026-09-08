import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from cegwm import formal_experiment as v1
from cegwm import formal_experiment_v2 as v2
from cegwm.runtime import blind_detection as detector
from cegwm.runtime import blind_scoring_v2
from cegwm.method.blind_detection import registered_minus_wrong_key_max
from cegwm.geometry_v7.r1a import _pixel_output_to_source
from cegwm.geometry_v7.syncseal import SyncSealTorchScript
from test_continuous_corners import FractionalFixture

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_complete_new_rosters_and_unchanged_denominators():
    config = v2.load_formal_config(ROOT/'configs/paper_experiment/formal_experiment_v2.json')
    old = v1.load_formal_config(ROOT/'configs/paper_experiment/formal_experiment_v1.json')
    roster, previous = v2.expand_rosters(ROOT, config), v1.expand_rosters(ROOT, old)
    assert [len(x) for x in roster.values()] == [2000, 3000, 1000]
    old_seeds = {u.seed for units in previous.values() for u in units}
    new_seeds = {u.seed for units in roster.values() for u in units}
    assert not old_seeds & new_seeds and len(new_seeds)==6000
    assert len(v2.FORMAL_CONDITIONS)==6
    assert sum(len(x['conditions'])*2*100 for x in config['ablations']['variants'].values())==1600


def test_black_rotation_matches_validated_sampler_and_other_conditions_unchanged():
    image = Image.fromarray(np.random.default_rng(1).integers(0,256,(512,512,3),dtype=np.uint8))
    angle=math.radians(10.)
    c,s=math.cos(angle),math.sin(angle)
    expected=image.transform((512,512),Image.Transform.PERSPECTIVE,
        _pixel_output_to_source(((c,-s,0),(s,c,0),(0,0,1))),
        resample=Image.Resampling.BILINEAR,fillcolor=(0,0,0))
    assert np.array_equal(np.asarray(v2.apply_attack(image,v2.ROTATION)),np.asarray(expected))
    for condition in v1.FORMAL_CONDITIONS:
        assert np.array_equal(np.asarray(v2.apply_attack(image,condition)),np.asarray(v1.apply_attack(image,condition)))


def test_single_continuous_h_and_pre_early_return(monkeypatch):
    fixture=FractionalFixture()
    backend=SyncSealTorchScript(fixture)
    assets=SimpleNamespace(geometry_backend=backend)
    calls=[]
    values=iter([2.,0.,2.])
    def score(*args,**kwargs):
        calls.append(kwargs)
        return registered_minus_wrong_key_max(next(values),(0.,)*16)
    monkeypatch.setattr(blind_scoring_v2,'score_statistic_v2',score)
    image=Image.new('RGB',(512,512))
    first=detector._detect_core(image,'fixture-key-0001',assets,1.,continuous_corners=True,reuse_observation=True)
    assert first.route=='DIRECT_POSITIVE' and fixture.calls==0
    second=detector._detect_core(image,'fixture-key-0001',assets,1.,continuous_corners=True,reuse_observation=True)
    assert second.route=='GEOMETRY_RECOVERED' and fixture.calls==1
    assert len(calls)==3 and all(c['reuse_observation'] for c in calls)


def test_preflight_runs_one_pair_all_conditions_and_no_formal_worker(monkeypatch,tmp_path):
    from experiments import run_paper_main_worker_v2 as worker
    from experiments.run_paper_v2 import real_preflight
    generated=[];forced=[]
    monkeypatch.setattr(worker,'_build_runtime',lambda path:dict(key='key',assets=None))
    def pair(*args):
        generated.append(1)
        return Image.new('RGB',(512,512),'white'),Image.new('RGB',(512,512),'gray')
    monkeypatch.setattr(worker,'_main_pair',pair)
    monkeypatch.setattr(worker,'_quality',lambda *a:dict(psnr=40.,ssim=.99,lpips=.01))
    def forced_score(*args):
        forced.append(1)
        return -2.,'GEOMETRY_RECOVERED'
    monkeypatch.setattr(worker,'_calibration_score',forced_score)
    monkeypatch.setattr(blind_scoring_v2,'score_branches_v2',lambda *a,**k:({'weighted_joint':{'registered':0.}},'fixture'))
    assert real_preflight('main',tmp_path,tmp_path/'runtime')==0
    report=json.loads((tmp_path/'preflight/main/preflight.json').read_text())
    assert len(generated)==1 and len(forced)==12 and report['science_denominator']==0
    assert len({(r['condition'],r['role']) for r in report['rows']})==12


def test_revision_change_does_not_block_resume(tmp_path):
    identity=dict(schema_version='v2',job_id='job',run_id='run',method_id='method',stage='test',expected_exact='old')
    store=v2.FormalRunStore(tmp_path,identity,['one'])
    store.initialize()
    resumed=v2.FormalRunStore(tmp_path,dict(identity,expected_exact='new'),['one'])
    resumed.initialize()
    assert resumed.identity['expected_exact']=='old'


def test_finalizer_exports_all_results_with_failures_retained(tmp_path):
    import csv
    from experiments import run_paper_results_finalize_v2 as finalizer
    from experiments import run_paper_main_worker_v2 as main_worker
    from cegwm.paper_tables_v2 import export_tables_figures
    config=v2.load_formal_config(main_worker.CONFIG_PATH)
    paths={main_worker.METHOD_ID:tmp_path/'main'/finalizer.MAIN_JOB_ID/'method_final.json'}
    paths.update({m:tmp_path/'baselines'/j/'method_final.json' for m,j in finalizer.BASELINE_JOBS.items()})
    for method,path in paths.items():
        result=finalizer._missing_method_result(method,'fixture')
        if method==main_worker.METHOD_ID: result['ablations']=main_worker._empty_ablations(config)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result))
    path=tmp_path/'reconstruction'/finalizer.RECONSTRUCTION_JOB_ID/'reconstruction_final.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(finalizer._missing_reconstruction('fixture')))
    assert finalizer.run_finalize(drive_root=tmp_path,expected_exact='fixture-new',baseline_exact='fixture-new')==0
    output=tmp_path/'finalized/paper-formal-v2'
    package=output/'unified_result_package.json'
    assert export_tables_figures(package)==dict(binary_rows=83,quality_rows=15)
    for name,count in [('unified_main_table_long.csv',60),('all_binary_results.csv',83),('quality_results.csv',15)]:
        with (output/name).open() as stream: assert len(list(csv.DictReader(stream)))==count
    for name in ('main_conditions.png','main_conditions.pdf','quality.png','quality.pdf'):
        assert (output/name).is_file()
    payload=json.loads(package.read_text())
    assert payload['status']=='INCOMPLETE_OPERATIONAL'
    assert payload['methods'][main_worker.METHOD_ID]['clean_negative_test']['n_missing']==3000


def test_failed_preflight_can_retry_without_mixing_rows(monkeypatch,tmp_path):
    from experiments import run_paper_main_worker_v2 as worker
    from experiments.run_paper_v2 import real_preflight
    def failed(path): raise RuntimeError('fixture dependency failure')
    monkeypatch.setattr(worker,'_build_runtime',failed)
    assert real_preflight('main',tmp_path,tmp_path/'runtime')==3
    assert real_preflight('main',tmp_path,tmp_path/'runtime')==3
    root=tmp_path/'preflight/main'
    assert (root/'failed-attempt-1/preflight.json').is_file()
    assert json.loads((root/'preflight.json').read_text())['missing_observations']==12
