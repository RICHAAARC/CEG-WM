import json
import numpy as np
import pytest
pytestmark = pytest.mark.unit
import torch
from cegwm.method.survival_allocator import allocation_from_logits, macro_features, fit_allocator
from experiments.run_survival_allocator_dev import validate_roster, main


def test_allocator_full_support_mean_budget_and_fixed_lf():
    for logits in ([0]*4, [1e6,-1e6,0,0], [4,3,2,1]):
        a = allocation_from_logits(logits)
        w = np.asarray(a.hf_tile_weights)
        assert np.all(w > .5) and np.all(w < 1.5)
        assert abs(w.mean()-1) < 1e-12
        assert a.lf_tile_weights == (1.,)*16
        assert a.lf_branch_share == a.hf_branch_share == .5
    assert allocation_from_logits([0]*4).lf_tile_weights == (1.,)*16


def test_macro_coordinates_and_feature_prediction():
    grid = np.repeat(np.repeat(np.arange(4).reshape(2,2),2,0),2,1)
    x = macro_features(grid.ravel(), grid.ravel(), torch.ones(1,16,8,8))
    np.testing.assert_array_equal(x[:,0], np.arange(4))
    np.testing.assert_array_equal(x[:,2], np.ones(4))
    model = fit_allocator(x, np.arange(4), ['fit', 'seed:1'])
    weights = np.asarray(model.predict(x).hf_tile_weights).reshape(4,4)
    assert weights[3,3] > weights[0,0]
    with pytest.raises(ValueError, match='overlaps'):
        validate_roster([{'id':'val','seed':1,'prompt':'p'}], 'validation', model)
    validate_roster([{'id':'val','seed':2,'prompt':'p'}], 'validation', model)


def test_plan_does_not_initialize_models(tmp_path, capsys):
    roster = tmp_path/'r.json'
    roster.write_text(json.dumps([{'id':'a','seed':1,'prompt':'a landscape'}]))
    assert main(['--roster',str(roster),'--output',str(tmp_path/'unused')]) == 0
    assert not (tmp_path/'unused').exists()
    assert json.loads(capsys.readouterr().out)['real_execution'] is False


def test_callback_requires_end_to_end_remainder(monkeypatch):
    from types import SimpleNamespace
    from cegwm.runtime import survival_allocator_sd35 as runtime
    # Engineering double tests allocation replacement wiring, not model robustness.
    monkeypatch.setattr(runtime, '_adaptive_allocation', lambda *a: allocation_from_logits([0]*4))
    monkeypatch.setattr(runtime, '_decode_callback_latents', lambda *a: object())
    monkeypatch.setattr(runtime, 'dino_last_layer_cls_patch_tiles', lambda *a: [1.]*16)
    monkeypatch.setattr(runtime, 'rgb_texture_tiles', lambda *a: [1.]*16)
    seen = []
    def embed(latent, key, hf, lf, allocation, beta, **kwargs):
        seen.append((key, beta, allocation))
        return latent + .01, None
    monkeypatch.setattr(runtime, 'embed_content_iss', embed)
    monkeypatch.setattr(runtime, 'prepare_fixed_lf_reference', lambda *a:SimpleNamespace(beta=.7))
    monkeypatch.setattr(runtime, 'embed_fixed_lf_hf', lambda latent,key,hf,lf,allocation,reference:embed(latent,key,hf,lf,allocation,reference.beta))
    assets = SimpleNamespace(embed_assets=SimpleNamespace(dino_processor=None,dino_model=None,hf_public_assets=None,lf_public_assets=None))
    cb = runtime.AllocatorCallback(b'key', assets, .7, 'uniform')
    latent = torch.ones(1,16,8,8)
    state = cb(object(),18,None,{'latents':latent})
    assert not torch.equal(state['latents'],latent)
    cb(object(),19,None,state)
    assert cb.remaining_steps == [19] and seen[0][:2] == (b'key',.7)
    with pytest.raises(RuntimeError, match='duplicate'):
        cb(object(),18,None,state)


def test_generate_variant_replay_seed_sync_and_missing_remainder(monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from cegwm.runtime import survival_allocator_sd35 as module
    calls = []
    class Pipeline:
        include_last = True
        def __call__(self, **kwargs):
            calls.append(('pipeline', kwargs['generator'].initial_seed()))
            assert kwargs['num_inference_steps'] == 20
            assert kwargs['callback_on_step_end_tensor_inputs'] == ['latents']
            state = {'latents':torch.ones(1,16,8,8)}
            callback = kwargs['callback_on_step_end']
            for step in range(20 if self.include_last else 19):
                state = callback(self, step, None, state)
            # Actual pipeline is replaced only for engineering wiring assertions.
            assert torch.allclose(state['latents'],torch.full((1,16,8,8),1.01))
            return SimpleNamespace(images=[Image.new('RGB',(512,512))])
    def sync(image, multiplier):
        calls.append(('sync',multiplier))
        return image
    assets = SimpleNamespace(embed_assets=SimpleNamespace(dino_processor=None,dino_model=None,
        hf_public_assets=None,lf_public_assets=None), lf_public_assets=None, iss_asset=None)
    runtime = {'pipeline':Pipeline(), 'device':'cpu', 'key':b'key', 'assets':SimpleNamespace(
        content_assets=SimpleNamespace(iss_assets=assets),
        geometry_backend=SimpleNamespace(embed_final_rgb=sync))}
    monkeypatch.setattr(module,'_adaptive_allocation',lambda *a:allocation_from_logits([0]*4))
    monkeypatch.setattr(module,'score_content_iss_image',lambda *args:.2)
    monkeypatch.setattr(module,'iss_beta',lambda *args:.7)
    monkeypatch.setattr(module,'_decode_callback_latents',lambda *args:object())
    monkeypatch.setattr(module,'dino_last_layer_cls_patch_tiles',lambda *args:[1.]*16)
    monkeypatch.setattr(module,'rgb_texture_tiles',lambda *args:[1.]*16)
    def embed(latent,key,hf,lf,allocation,beta,**kwargs):
        calls.append(('embed', beta))
        return latent+.01,None
    monkeypatch.setattr(module,'embed_content_iss',embed)
    monkeypatch.setattr(module,'prepare_fixed_lf_reference',lambda *a:SimpleNamespace(beta=.7))
    monkeypatch.setattr(module,'embed_fixed_lf_hf',lambda latent,key,hf,lf,allocation,reference:embed(latent,key,hf,lf,allocation,reference.beta))
    image, features, budget = module.generate_variant(runtime,'test prompt',917,Image.new('RGB',(512,512)),'uniform')
    assert image.mode == 'RGB' and features.shape == (4,3)
    assert [call[0] for call in calls] == ['pipeline','embed','sync']
    assert calls[0][1] == 917
    calls.clear()
    runtime['pipeline'].include_last = False
    with pytest.raises(RuntimeError,match='actual final denoising step 19'):
        module.generate_variant(runtime,'test prompt',917,Image.new('RGB',(512,512)),'uniform')
    assert [call[0] for call in calls] == ['pipeline','embed']


def test_fit_setup_failure_retains_all_planned_rows(monkeypatch,tmp_path):
    from experiments import run_paper_main_worker_v2 as v2
    def fail_runtime(*args):
        raise RuntimeError('engineering injected initialization failure')
    monkeypatch.setattr(v2,'_build_runtime',fail_runtime)
    roster = tmp_path/'fit.json'
    roster.write_text(json.dumps([{'id':'fit-a','prompt':'test prompt','seed':717}]))
    out = tmp_path/'output'
    assert main(['--mode','fit','--roster',str(roster),'--output',str(out),
        '--runtime-root',str(tmp_path/'runtime')]) == 2
    report = json.loads((out/'report.json').read_text())
    assert report['planned_score_paths'] == 21
    assert len(report['rows']) == 21
    assert len({(row['id'],row['variant'],row['condition']) for row in report['rows']}) == 21
    assert all('initialization failure' in row['error'] for row in report['rows'])
    assert not (out/'allocator.json').exists()


def test_hf_replacement_preserves_original_lf_and_shares():
    from cegwm.method.content_adaptive import ContentAllocation
    reference = ContentAllocation((1.,)*16, tuple([.8]*8+[1.2]*8), .43, .57, (0.,)*6)
    original = allocation_from_logits([0]*4, reference)
    assert original.hf_tile_weights == (1.,)*16
    uniform = allocation_from_logits([0]*4, reference, uniform=True)
    assert uniform.hf_tile_weights == (1.,)*16
    probe = allocation_from_logits([.5,0,0,0],reference)
    for allocation in (original,uniform,probe):
        assert allocation.lf_tile_weights == reference.lf_tile_weights
        assert allocation.lf_branch_share == .43 and allocation.hf_branch_share == .57
    assert probe.hf_tile_weights != reference.hf_tile_weights


def test_instrumented_v2_path_matches_existing_margin_without_repeat_observation(monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from experiments import run_paper_main_worker_v2 as v2
    from experiments.run_survival_allocator_dev import score_details
    from cegwm.runtime import blind_scoring_v2 as scoring
    pre, post = Image.new('RGB',(4,4)), Image.new('RGB',(4,4))
    seen = []
    def branches(image,*args,**kwargs):
        seen.append(image)
        values = {'registered': 5. if image is pre else 7., **{f'wrong_{i:02d}':3. for i in range(16)}}
        return {'weighted_joint':values},'fake_engineering'
    runtime = {'key':b'key','assets':SimpleNamespace(geometry_backend=SimpleNamespace(detect_geometry_continuous=lambda image:object()))}
    monkeypatch.setattr(scoring,'score_branches_v2',branches)
    monkeypatch.setattr(v2,'_geometry_disposition',lambda x:('VALID',None))
    monkeypatch.setattr(v2,'_raw_h',lambda x:np.eye(3))
    monkeypatch.setattr(v2,'rectify_attacked_rgb',lambda *args:post)
    monkeypatch.setattr(v2,'_score_current_rgb',lambda image,*args:SimpleNamespace(value=2. if image is pre else 4.))
    result = score_details(runtime,pre)
    assert (result['score'],result['route']) == v2._calibration_score(runtime,pre)
    assert result['selected']=='post' and result['registered']==7. and result['max16wrong']==3.
    assert seen == [pre,post]
    monkeypatch.setattr(v2,'_geometry_disposition',lambda x:('INVALID_H',None))
    assert score_details(runtime,pre)['score'] == v2._calibration_score(runtime,pre)[0]


def test_negative_upper_tail_and_actual_paired_separation():
    from experiments.run_survival_allocator_dev import summarize_separation
    quality = {'psnr':40.,'ssim':.99,'lpips':.002}
    rows=[]
    for unit,negative,positive in [('a',-1.,2.),('b',1.,3.)]:
        for variant,value in [('clean',negative),('original',positive)]:
            rows.append({'id':unit,'variant':variant,'condition':'clean','error':None,'score':value,
                'registered':value+2.,'max16wrong':2.,'quality':quality})
    report=summarize_separation(rows,3)[0]
    assert report['negative_upper_tail']['planned']==3
    assert report['negative_upper_tail']['observed']==2
    assert report['negative_upper_tail']['max']==1.
    assert report['paired_score_separations']==[3.,2.]
    assert report['positive_min_minus_negative_max']==1.
    assert report['quality_matching_claim'] is False


def test_actual_latent_hf_map_is_smooth_and_default_unchanged():
    from cegwm.method.content_unweighted import _hf_weighted_amplitude
    carrier=torch.ones(1,1,64,64,dtype=torch.float64)
    weights=tuple([.75,.75,1.25,1.25]*4)
    nearest=_hf_weighted_amplitude(carrier,weights,torch.tensor(1.))
    explicit=_hf_weighted_amplitude(carrier,weights,torch.tensor(1.),interpolation='nearest')
    smooth=_hf_weighted_amplitude(carrier,weights,torch.tensor(1.),interpolation='bilinear')
    assert torch.equal(nearest,explicit)
    # Spatial jumps at old tile boundaries are reduced at actual latent resolution.
    assert torch.diff(smooth[0,0],dim=1).abs().max() < torch.diff(nearest[0,0],dim=1).abs().max()/8
    assert torch.all(smooth>0)
    assert torch.isclose(torch.linalg.vector_norm(smooth),torch.tensor(1.,dtype=torch.float64))
    uniform=(1.,)*16
    assert torch.equal(_hf_weighted_amplitude(carrier,uniform,torch.tensor(1.)),
        _hf_weighted_amplitude(carrier,uniform,torch.tensor(1.),interpolation='bilinear'))

@pytest.mark.parametrize('fail_probe', [False, True])
def test_fit_fake_wiring_produces_four_labels_and_separation(monkeypatch,tmp_path,fail_probe):
    from PIL import Image
    from experiments import run_paper_main_worker_v2 as v2
    from experiments import run_survival_allocator_dev as runner
    from cegwm.runtime import survival_allocator_sd35 as runtime
    monkeypatch.setattr(v2,'_build_runtime',lambda *args:{})
    monkeypatch.setattr(v2,'_plain',lambda *args:Image.new('RGB',(32,32)))
    monkeypatch.setattr(v2,'_quality',lambda *args:{'psnr':40.,'ssim':.99,'lpips':.002})
    def generate(*args):
        variant=args[4]
        if fail_probe and variant == 'probe2':
            raise RuntimeError('FIXED_LF_NO_NONZERO_HF_FOUND_IN_BOUNDED_SEARCH')
        color=20+int(variant[-1]) if variant.startswith('probe') else 10
        return Image.new('RGB',(32,32),(color,)*3),np.arange(12).reshape(4,3),{'iss_beta':1.2,'embedding':{}}
    monkeypatch.setattr(runtime,'generate_variant',generate)
    def score(runtime,image):
        value=float(np.asarray(image).mean())/255.
        return {'score':value,'registered':value+1.,'max16wrong':1.,'route':'fake'}
    monkeypatch.setattr(runner,'score_details',score)
    roster=tmp_path/'roster.json'
    roster.write_text(json.dumps([{'id':'fit-one','prompt':'fake only','seed':4}]))
    out=tmp_path/'output'
    assert main(['--mode','fit','--roster',str(roster),'--output',str(out),
        '--runtime-root',str(tmp_path/'runtime')]) == (2 if fail_probe else 0)
    report=json.loads((out/'report.json').read_text())
    if fail_probe:
        assert len(report['rows']) == 21 and report['row_errors'] == 3
        assert report['failed_fit_units'] == ['fit-one']
        assert all('FIXED_LF_NO_NONZERO_HF_FOUND' in row['error']
                   for row in report['rows'] if row['variant'] == 'probe2')
        assert not (out/'allocator.json').exists()
        return
    labels=[json.loads(row) for row in (out/'labels.jsonl').read_text().splitlines()]
    assert len(report['rows'])==21 and report['row_errors']==0
    assert len(labels)==4 and all(label['reference_variant']=='uniform' for label in labels)
    assert all(len(label['attacked_negative'])==3 for label in labels)
    assert (out/'allocator.json').exists() and report['separation']


def test_hf_smoothing_does_not_change_lf_preprojection(monkeypatch):
    from cegwm.method import content_unweighted as method
    carrier=torch.linspace(.1,1.,64*64,dtype=torch.float64).reshape(1,1,64,64)
    monkeypatch.setattr(method,'reconstruct_hf_carrier',lambda *args,**kwargs:carrier)
    monkeypatch.setattr(method,'reconstruct_lf_carrier',lambda *args,**kwargs:carrier)
    allocation=allocation_from_logits([.5,0,0,0])
    nearest=method._content_unweighted_branch_deltas(carrier,b'key',None,None,allocation)
    smooth=method._content_unweighted_branch_deltas(carrier,b'key',None,None,allocation,hf_weight_interpolation='bilinear')
    assert torch.equal(nearest[0],smooth[0])
    assert not torch.equal(nearest[1],smooth[1])
    assert torch.isclose(torch.linalg.vector_norm(nearest[1]),torch.linalg.vector_norm(smooth[1]))
