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

