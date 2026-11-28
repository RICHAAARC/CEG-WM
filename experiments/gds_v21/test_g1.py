"""CPU engineering tests, no model download/generation or training."""
import json
from pathlib import Path
import numpy as np
import pytest
from core import select_set,pool_order
from geometry import N,K,MASK
from g1_sampling import ARMS,DOMAINS,stream_seed,draw_factorized,sampling_context
from g1_runner import plan,validate_g0


def test_fixed_roster_and_independent_rng_domains():
    config=json.loads((Path(__file__).parent/'g1_config.json').read_text())
    rows=plan(config)
    assert len(rows)==48 and len({r['unit'] for r in rows})==48
    assert len({r['source_id'] for r in rows})==8
    assert all(sum(r['arm']==a for r in rows)==8 for a in ARMS)
    seeds=[stream_seed(281128101,a,d) for a in ARMS for d in DOMAINS]
    assert len(set(seeds))==18
    config['max_generation_attempts']=49
    with pytest.raises(ValueError): plan(config)


@pytest.mark.parametrize('arm',['P','P-E','B1','B2'])
def test_optimized_selection_matches_naive_reference(arm):
    rng=np.random.default_rng(98)
    for _ in range(8):
        n=14
        initial=rng.random(n)<.4
        contribution=rng.normal(size=(5,n))*.03
        ledger=rng.normal(size=5)*.02
        target=rng.random(n); address=np.arange(n)%16; counts=rng.random(16)
        m=rng.uniform(.05,.95,n); current=np.ones(n,bool)
        def value(s):
            if arm in ('P','P-E'):
                return float(np.min((ledger if arm=='P' else np.zeros(5))+contribution[:,s].sum(1)))
            if arm=='B1': return float(target[s].sum())
            return float(np.log1p(counts+np.bincount(address[s],weights=target[s],minlength=16)).sum())
        expected=initial.copy()
        outgoing=pool_order(np.flatnonzero(initial),30)[:32]
        incoming=pool_order(np.flatnonzero(~initial),30)[:32]
        for _ in range(4):
            before=value(expected); best=None
            for i in sorted(outgoing):
                for j in sorted(incoming):
                    candidate=expected.copy(); candidate[i]=False;candidate[j]=True
                    after=value(candidate)
                    if after-before>1e-10*max(1.,abs(before),abs(after)) and (best is None or after>best[0]):
                        best=(after,i,j)
            if best is None: break
            _,i,j=best; expected[i]=False;expected[j]=True; outgoing.remove(i);incoming.remove(j)
        chosen,record=select_set(initial,current,m,30,contribution,ledger,target,address,counts,objective=arm)
        np.testing.assert_array_equal(chosen,expected)
        assert abs(record['final_value']-value(expected))<1e-12


def test_factorized_dirac_and_visible_natural_remask():
    import torch
    kernel=np.zeros((3,K+1));q=np.zeros((3,K))
    kernel[0,5]=.2;kernel[0,MASK]=.8;q[0,5]=1
    kernel[1,8]=.3;kernel[1,MASK]=.7;q[1,8]=1
    kernel[2,MASK]=1
    history=np.array([MASK,MASK,2]);selected=np.array([True,False,False])
    streams={d:torch.Generator().manual_seed(10+i) for i,d in enumerate(DOMAINS)}
    before=streams['event'].get_state().clone()
    for _ in range(10):
        result=draw_factorized(kernel,q,history,selected,streams,'cpu')
        np.testing.assert_array_equal(result,[5,MASK,MASK])
    assert torch.equal(before,streams['event'].get_state())


def test_g0_gate_fails_closed():
    expected={'files':{'known.py':'known'}}
    report=dict(stage='G0_REAL_MODEL',status='passed',files=expected['files'],
        checks=dict(all_100_token_and_rng_states_equal=True,decoded_rgb_equal=True,final_rng_equal=True,
                    all_native_gumbel_replays_equal=True,intervention_window_records=50),
        observed_native_opportunity_timesteps=list(range(60,10,-1)),attempts=[dict(status='complete')]*2)
    validate_g0(report,expected)
    report['checks']['final_rng_equal']=False
    with pytest.raises(ValueError): validate_g0(report,expected)


@pytest.mark.parametrize('arm',ARMS)
def test_actual_scheduler_interface_and_unmodified_bypass(arm):
    import torch
    from diffusers import VQDiffusionScheduler
    old_threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        scheduler=VQDiffusionScheduler(num_vec_classes=K+1,num_train_timesteps=100)
        scheduler.set_timesteps(100)
        logits=torch.full((1,K,N),-np.log(K),dtype=torch.float32)
        initial=torch.full((1,N),MASK,dtype=torch.long)
        coeff=np.zeros((2,N,16),dtype=np.float64)
        original_step,original_q=scheduler.step,scheduler.q_posterior
        rng=torch.Generator().manual_seed(7)
        expected=original_step(logits,60,initial,generator=rng).prev_sample
        expected_state=rng.get_state().clone()
        rng=torch.Generator().manual_seed(7);records=[];calls=[]
        def counted(*args,**kwargs):
            calls.append(1);return original_q(*args,**kwargs)
        scheduler.q_posterior=counted
        with sampling_context(scheduler,arm,281128101,coeff,lambda row,h,z:records.append((row,h,z)),device='cpu'):
            output=scheduler.step(logits,60,initial,generator=rng).prev_sample
            assert len(calls)==1  # no extra posterior or generator evaluation for shadows
            if arm=='N':
                assert torch.equal(expected,output) and torch.equal(expected_state,rng.get_state())
            else:
                row=records[0][0]
                selected=np.array(row['executed_selected'],dtype=int)
                assert len(selected)==row['initial_count']
                assert int((output!=MASK).sum())==len(selected)
                assert not np.any(output.cpu().numpy().ravel()[selected]==MASK)
            # Outside the intervention window, same input/RNG must match native exactly.
            rng_a=torch.Generator().manual_seed(77);rng_b=torch.Generator().manual_seed(77)
            want=original_step(logits,10,output,generator=rng_a).prev_sample
            got=scheduler.step(logits,10,output,generator=rng_b).prev_sample
            assert torch.equal(want,got) and torch.equal(rng_a.get_state(),rng_b.get_state())
        assert scheduler.step==original_step and scheduler.q_posterior==counted
    finally:
        torch.set_num_threads(old_threads)
