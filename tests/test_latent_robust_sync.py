import numpy as np
import pytest
from cegwm.method.latent_sync import public_template,similarity,warp_field,estimate_rotation
from cegwm.method.latent_robust_sync import (
    ARMS,make_objective,estimate_robust_rotation,warp_continuous,preprocess_field)

pytestmark=pytest.mark.unit


def test_original_arm_reproduces_original_reader_and_signed_score():
    observation=warp_field(public_template(4,32,32),similarity((32,32),-10)).astype(np.float32)
    old=estimate_rotation(observation); new=estimate_robust_rotation(observation)
    assert new['angle']==pytest.approx(old['parameters'][0],abs=1e-12)
    assert new['correlation']==pytest.approx(old['correlation'],abs=1e-12)
    opposite=make_objective(-public_template(4,32,32),'original')
    assert opposite.score(0)<0


def test_grid_constant_crossing_is_continuous_instead_of_full_value_jump():
    field=np.ones((1,16,16)); epsilon=1e-6
    def translation(dx):
        H=np.eye(3); H[0,2]=dx; return H
    before=warp_continuous(field,translation(-epsilon))[0,8,0]
    after=warp_continuous(field,translation(epsilon))[0,8,0]
    assert abs(before-after)<=2*epsilon
    old_before=warp_field(field,translation(-epsilon))[0,8,0]
    old_after=warp_field(field,translation(epsilon))[0,8,0]
    assert abs(old_before-old_after)==1.


@pytest.mark.parametrize('arm',ARMS)
def test_pure_template_direction_and_degenerate_observation(arm):
    template=public_template(4,32,32)
    observation=warp_continuous(template,similarity((32,32),-10)).astype(np.float32)
    result=estimate_robust_rotation(observation,arm)
    assert -11<result['angle']<-9
    assert result['success']
    empty=estimate_robust_rotation(np.zeros((4,32,32)),arm)
    assert empty['status']=='DEGENERATE_OBSERVATION'
    assert empty['nfev']==0 and empty['correlation']==0


def test_whitener_is_estimated_once_and_shared_across_angle_queries(monkeypatch):
    import cegwm.method.latent_channel_whitening as module
    original=module.estimate_whitener; calls=[]
    def estimate(value):
        calls.append(value.copy()); return original(value)
    monkeypatch.setattr(module,'estimate_whitener',estimate)
    observation=public_template(4,32,32)
    objective=make_objective(observation,'combined')
    assert len(calls)==1
    assert np.array_equal(calls[0],preprocess_field(observation,True))
    objective.score(-10); objective.score(0); objective.score(10)
    estimate_robust_rotation(observation,'combined',objective=objective)
    assert len(calls)==1
    assert np.array_equal(objective.processed_observation,module.apply_whitener(objective.W,calls[0]))


def test_covariance_inverse_root_and_rejected_floor_reporting():
    from cegwm.method.latent_channel_whitening import estimate_whitener,apply_whitener
    rng=np.random.default_rng(613)
    x=rng.normal(size=(4,9,9)); W,diagnostics=estimate_whitener(x)
    centered=x.reshape(4,-1)-x.reshape(4,-1).mean(axis=1,keepdims=True)
    S=centered@centered.T/81
    K=.75*S+.25*np.trace(S)/4*np.eye(4)
    np.testing.assert_allclose(W@K@W.T,np.eye(4),atol=3e-14)
    y=rng.normal(size=x.shape)
    np.testing.assert_allclose(apply_whitener(W,x+y),apply_whitener(W,x)+apply_whitener(W,y),atol=3e-14)
    assert diagnostics['reliable'] and not diagnostics['floor_applied']
    W,diagnostics=estimate_whitener(np.zeros_like(x))
    assert W is None and not diagnostics['reliable']
    assert diagnostics['floor_required'] and not diagnostics['floor_applied']
