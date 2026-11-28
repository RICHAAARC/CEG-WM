"""Engineering checks only. No checkpoint, GPU generation or training."""
import itertools
import numpy as np
import pytest
from core import biased_content, normalize_logits, select_set
from geometry import (N, K, MASK, V0, IDENTITY, R, addresses, groups, compose,
                      transform, support, render, score, token_field, build_coefficients)


@pytest.fixture(scope='module')
def geometry_assets():
    return build_coefficients()


def test_balanced_assets():
    assert np.array_equal(np.bincount(groups()), np.full(16, 256))
    assert np.array_equal(np.bincount(addresses()), np.full(16, 64))
    assert np.array_equal(groups(), groups())


def test_coordinates_and_composition():
    field = np.eye(16)[addresses()]
    expected = field.reshape(32,32,16).repeat(2,0).repeat(2,1).reshape(4096,16)
    np.testing.assert_array_equal(render(field, IDENTITY), expected)
    # A +8 mathematical-y shift moves the symbol field two reader rows UP.
    shifted = render(field, (0.,1.,0.,8.)).reshape(64,64,16)
    np.testing.assert_array_equal(shifted[:-2], expected.reshape(64,64,16)[2:])
    np.testing.assert_array_equal(shifted[-2:], np.broadcast_to(V0, (2,64,16)))
    r, d = (9.,1.1,3.,-2.), (-6.,.9,12.,-8.)
    pts = np.array([[0,0],[256,256],[32,180],[128,128]])
    np.testing.assert_allclose(transform(pts, compose(r,d)), transform(transform(pts,d),r), atol=1e-12)
    assert support((0.,1.25,0.,0.)) == 1.0  # output support != retained source area


@pytest.mark.parametrize('partial', [False, True])
def test_coefficient_identity_against_direct_renderer(geometry_assets, partial):
    coeff, pairs, _ = geometry_assets
    assert len(pairs) > 1 and np.max(np.abs(coeff)) > 0
    rng = np.random.default_rng(28112801)
    tokens = rng.integers(K, size=N)
    if partial:
        tokens[rng.random(N) < .4] = MASK
    field = token_field(tokens)
    algebra = np.einsum('hig,ig->h', coeff, field)
    direct = [score(render(field,r),r) - score(render(field,r),h) for r,h in pairs]
    np.testing.assert_allclose(algebra, direct, atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(coeff.sum(2), 0., atol=1e-17)
    for h in R:
        assert abs(score(np.tile(V0,(4096,1)),h)) < 1e-15


def test_full_candidate_tilt_and_content_budget():
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(32), size=20)
    p[0,0] = 0
    p /= p.sum(1,keepdims=True)
    mass = np.linspace(.01,.99,20)
    kernel = np.column_stack((p*mass[:,None], 1-mass))
    g, a = np.arange(32)%16, np.arange(20)%16
    q,beta,reported = biased_content(kernel,g,a)
    np.testing.assert_allclose(q.sum(1),1.,atol=1e-14)
    assert q[0,0] == 0 and np.all((0 <= beta) & (beta <= 1))
    terms = np.zeros_like(q)
    valid = q > 0
    terms[valid] = q[valid]*np.log(q[valid]/p[valid])
    measured = terms.sum(1)
    np.testing.assert_allclose(measured,reported,atol=1e-14)
    assert measured.max() <= .10+1e-14
    # Probe the region in which beta=1 exceeds the budget.
    q,beta,kl = biased_content(np.array([[.25,.25,.5]]), np.array([0,1]),np.array([0]))
    assert 0 < beta[0] < 1 and .09999 < kl[0] <= .1
    q,beta,_ = biased_content(np.array([[0.,.5,.5],[.5,0.,.5],[0.,0.,1.]]),np.array([0,1]),np.zeros(3,int))
    np.testing.assert_array_equal(beta,0)
    np.testing.assert_array_equal(q,[[0,1],[1,0],[0,0]])


def test_tiny_exact_pushforward_kernel_and_kl():
    # Enumerate two masked positions, two content classes. F swaps one event.
    m = np.array([.4,.7])
    p = np.array([[.2,.8],[.6,.4]])
    q = np.array([[.4,.6],[.7,.3]])
    events = list(itertools.product([False,True],repeat=2))
    p0 = {a:float(np.prod(np.where(a,1-m,m))) for a in events}
    pi = dict.fromkeys(events,0.)
    for a in events:
        dest = (False,True) if a == (True,False) else a
        pi[dest] += p0[a]
    output_mass, joint_kl = 0.,0.
    for y in itertools.product(range(3),repeat=2):
        s = tuple(c != 2 for c in y)
        Q = pi[s]*np.prod([q[i,c] if c != 2 else 1. for i,c in enumerate(y)])
        P = np.prod([m[i] if c == 2 else (1-m[i])*p[i,c] for i,c in enumerate(y)])
        output_mass += Q
        if Q > 0:
            joint_kl += Q*np.log(Q/P)
    decomposition = sum(v*np.log(v/p0[a]) for a,v in pi.items() if v > 0)
    decomposition += sum(v*sum(np.sum(q[i]*np.log(q[i]/p[i])) for i in range(2) if a[i]) for a,v in pi.items())
    assert abs(output_mass-1.) < 1e-14
    assert abs(joint_kl-decomposition) < 1e-14
    # Multiplying MASK probabilities a second time must fail normalization.
    wrong_mass = sum(v*np.prod([m[i] for i in range(2) if not a[i]]) for a,v in pi.items())
    assert wrong_mass < .9


def test_same_history_ledger_changes_choice():
    kwargs = dict(initial=np.array([True,False,False]),current_mask=np.ones(3,bool),
        mask_probability=np.full(3,.5),timestep=30,
        contribution=np.array([[0.,1.,0.],[0.,1.,2.]]),ledger=np.array([2.,0.]),
        target_probability=np.full(3,.2),address=np.zeros(3,int),visible_counts=np.zeros(16))
    p,rp = select_set(**kwargs,objective='P')
    pe,re = select_set(**kwargs,objective='P-E')
    np.testing.assert_array_equal(p,[False,False,True])
    np.testing.assert_array_equal(pe,[False,True,False])
    for chosen,record in ((p,rp),(pe,re)):
        assert chosen.sum() == 1 and len(record['swaps']) <= 4
        assert all(x['after'] > x['before'] for x in record['swaps'])
    kwargs['mask_probability'] = np.array([.5,.001,.999])
    chosen,record = select_set(**kwargs)
    np.testing.assert_array_equal(chosen,kwargs['initial'])
    assert not record['swaps']


def test_selection_limits_determinism_and_ties():
    n=80
    kwargs=dict(initial=np.arange(n)<40,current_mask=np.ones(n,bool),mask_probability=np.full(n,.5),
        timestep=40,contribution=np.array([np.arange(n)/100]),ledger=np.zeros(1),
        target_probability=np.arange(n)/100,address=np.arange(n)%16,visible_counts=np.zeros(16))
    for arm in ('P','P-E','B1','B2'):
        chosen,record=select_set(**kwargs,objective=arm)
        again,repeat=select_set(**kwargs,objective=arm)
        assert record == repeat and np.array_equal(chosen,again)
        assert chosen.sum()==40 and len(record['swaps'])<=4
        assert record['out_pool_count']==record['in_pool_count']==32
        ids=[x[k] for x in record['swaps'] for k in ('out','into')]
        assert len(ids)==len(set(ids))
    kwargs['target_probability']=np.ones(n)
    _,record=select_set(**kwargs,objective='B1')
    assert not record['swaps']


def test_logits_normalization_rejects_nonfinite():
    np.testing.assert_allclose(normalize_logits([[10000.,10001.]]),[[1/(1+np.e),np.e/(1+np.e)]])
    with pytest.raises(ValueError):
        normalize_logits([[np.nan,0]])


@pytest.mark.parametrize('return_dict',[True,False])
def test_native_scheduler_capture_rng_and_restore(return_dict):
    import torch
    from diffusers import VQDiffusionScheduler
    from observe import native_trace
    # Real scheduler implementation, small synthetic model outputs. NOT checkpoint evidence.
    scheduler=VQDiffusionScheduler(num_vec_classes=17,num_train_timesteps=100)
    scheduler.set_timesteps(100)
    original_step,original_q=scheduler.step,scheduler.q_posterior
    model_rng=torch.Generator().manual_seed(12)
    logits=torch.randn((1,16,32),generator=model_rng).log_softmax(1)
    initial=torch.full((1,32),16,dtype=torch.long)
    traces=[]
    for capture in (False,True):
        record=[]
        def sink(t,raw,history,tokens,state,replay):
            record.append((t,tokens.copy(),state.copy()))
            if capture:
                assert replay
                assert raw.shape==(1,16+(t!=0),32)
        rng=torch.Generator().manual_seed(25)
        x=initial.clone()
        with native_trace(scheduler,sink,capture_kernel=capture):
            for t in (60,25,1,0):
                output=scheduler.step(logits,t,x,generator=rng,return_dict=return_dict)
                x=output.prev_sample if return_dict else output[0]
        traces.append(record)
    for first,second in zip(*traces):
        assert first[0]==second[0]
        np.testing.assert_array_equal(first[1],second[1])
        np.testing.assert_array_equal(first[2],second[2])
    assert scheduler.step == original_step and scheduler.q_posterior == original_q
    with pytest.raises(RuntimeError,match='intentional'):
        with native_trace(scheduler,lambda *args:None,capture_kernel=True):
            raise RuntimeError('intentional')
    assert scheduler.step == original_step and scheduler.q_posterior == original_q


def test_real_shape_kernel_diagnostics_on_synthetic_law():
    import torch
    from g0_gpu import kernel_record
    probabilities=np.full((1,K+1,N),.02/K,dtype=np.float64)
    probabilities[:,MASK,:]=.98
    raw=torch.from_numpy(np.log(probabilities))
    history=np.full((1,N),MASK,dtype=np.int64)
    tokens=history.copy()
    tokens[0,:20]=0
    row=kernel_record(60,raw,history,tokens)
    assert row['eligible_count']==N and row['exchange_opportunity']
    assert row['out_pool_count']==20 and row['in_pool_count']==32
    assert row['conditional_kl_max']<=.1 and row['kernel_normalization_error']<1e-12
    with pytest.raises(AssertionError,match='shape'):
        kernel_record(0,raw,history,tokens)
