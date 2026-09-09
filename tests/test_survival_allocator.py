import json
import numpy as np
import pytest
pytestmark = pytest.mark.unit
import torch
from cegwm.method.survival_allocator import allocation_from_logits, macro_features, fit_allocator
from experiments.run_survival_allocator_dev import validate_roster, main


def test_allocator_full_support_mean_budget_and_shared_branches():
    for logits in ([0]*4, [1e6,-1e6,0,0], [4,3,2,1]):
        a = allocation_from_logits(logits)
        w = np.asarray(a.lf_tile_weights)
        assert np.all(w > .5) and np.all(w < 1.5)
        assert abs(w.mean()-1) < 1e-12
        assert a.lf_tile_weights == a.hf_tile_weights
        assert a.lf_branch_share == a.hf_branch_share == .5
    assert allocation_from_logits([0]*4).lf_tile_weights == (1.,)*16


def test_macro_coordinates_and_feature_prediction():
    grid = np.repeat(np.repeat(np.arange(4).reshape(2,2),2,0),2,1)
    x = macro_features(grid.ravel(), grid.ravel(), torch.ones(1,16,8,8))
    np.testing.assert_array_equal(x[:,0], np.arange(4))
    np.testing.assert_array_equal(x[:,2], np.ones(4))
    model = fit_allocator(x, np.arange(4), ['fit', 'seed:1'])
    weights = np.asarray(model.predict(x).lf_tile_weights).reshape(4,4)
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
    monkeypatch.setattr(runtime, '_decode_callback_latents', lambda *a: object())
    monkeypatch.setattr(runtime, 'dino_last_layer_cls_patch_tiles', lambda *a: [1.]*16)
    monkeypatch.setattr(runtime, 'rgb_texture_tiles', lambda *a: [1.]*16)
    seen = []
    def embed(latent, key, hf, lf, allocation, beta):
        seen.append((key, beta, allocation))
        return latent + .01, None
    monkeypatch.setattr(runtime, 'embed_content_iss', embed)
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
    monkeypatch.setattr(module,'score_content_iss_image',lambda *args:.2)
    monkeypatch.setattr(module,'iss_beta',lambda *args:.7)
    monkeypatch.setattr(module,'_decode_callback_latents',lambda *args:object())
    monkeypatch.setattr(module,'dino_last_layer_cls_patch_tiles',lambda *args:[1.]*16)
    monkeypatch.setattr(module,'rgb_texture_tiles',lambda *args:[1.]*16)
    def embed(latent,key,hf,lf,allocation,beta):
        calls.append(('embed', beta))
        return latent+.01,None
    monkeypatch.setattr(module,'embed_content_iss',embed)
    image, features = module.generate_variant(runtime,'test prompt',917,Image.new('RGB',(512,512)),'uniform')
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
    assert report['planned_score_paths'] == 63
    assert len(report['rows']) == 63
    assert len({(row['id'],row['variant'],row['condition']) for row in report['rows']}) == 63
    assert all('initialization failure' in row['error'] for row in report['rows'])
    assert not (out/'allocator.json').exists()
