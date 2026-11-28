"""GDS v2.1 actual transition intervention, with same-history shadow decisions."""
import contextlib
import hashlib
import time
import numpy as np
from core import normalize_logits,biased_content,select_set
from geometry import MASK,N,K,groups,addresses,token_field

ARMS=('N','B0','B1','B2','P','P-E')
DOMAINS=('event','content','visible')


def stream_seed(base,arm,domain):
    if arm not in ARMS or domain not in DOMAINS:
        raise ValueError('Unregistered RNG stream')
    message=f'GDS-v2.1-G1:{base}:{arm}:{domain}'.encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8],'big') % (2**63-1)


def aggregate(probabilities,group):
    return np.column_stack([probabilities[:,np.flatnonzero(group==g)].sum(1) for g in range(16)])


def prepare_history(kernel,history,initial,coeff,group,address,t):
    """Deterministic shared H/S0/pools; all four shadows use the same inputs."""
    current_mask=history==MASK
    q,beta,kl=biased_content(kernel,group,address)
    qgroup=aggregate(q,group)
    kgroup=aggregate(kernel[:,:K],group)
    contribution=np.einsum('hig,ig->hi',coeff,qgroup,optimize=False)
    expected_visible=np.einsum('hig,ig->hi',coeff,kgroup,optimize=False)
    ledger=expected_visible[:,~current_mask].sum(1)
    target=qgroup[np.arange(N),address]
    visible_target=kgroup[np.arange(N),address]*(~current_mask)
    counts=np.bincount(address,weights=visible_target,minlength=16)
    selections={'B0':initial.copy()}
    records={}
    for arm in ('B1','B2','P','P-E'):
        selected,record=select_set(initial,current_mask,kernel[:,MASK],t,contribution,ledger,
                                  target,address,counts,objective=arm)
        if selected.sum()!=initial.sum() or np.any(selected&~current_mask):
            raise AssertionError('Invalid selected set')
        selections[arm]=selected
        record['selected']=np.flatnonzero(selected).tolist()
        scores=ledger+contribution[:,selected].sum(1)
        record['full_ledger_J']=float(scores.min())
        record['bottleneck']=int(np.argmin(scores))
        records[arm]=record
    full_before=ledger+contribution[:,initial].sum(1)
    records['B0']=dict(selected=np.flatnonzero(initial).tolist(),swaps=[],
        full_ledger_J=float(full_before.min()),bottleneck=int(np.argmin(full_before)))
    return q,beta,kl,selections,records,ledger,contribution


def draw_categorical(probabilities,generator,device):
    """Independent full-candidate categorical draws; zero mass remains impossible."""
    import torch
    if not len(probabilities):
        return np.empty(0,dtype=np.int64)
    if not np.isfinite(probabilities).all() or np.any(probabilities<0) or np.any(probabilities.sum(1)<=0):
        raise ValueError('Invalid categorical law')
    tensor=torch.as_tensor(probabilities,dtype=torch.float64,device=device)
    return torch.multinomial(tensor,1,replacement=True,generator=generator).flatten().cpu().numpy()


def draw_factorized(kernel,q,history,selected,streams,device):
    """The selected set has already absorbed all MASK-event probabilities.

    Unselected current-MASK sites are Dirac MASK; no extra Bernoulli factor.
    Visible sites retain the full original K, including any natural remasking.
    """
    visible=history!=MASK
    if np.any(selected&visible):
        raise ValueError('Cannot newly select visible sites')
    output=np.full(len(history),MASK,dtype=np.int64)
    output[selected]=draw_categorical(q[selected],streams['content'],device)
    output[visible]=draw_categorical(kernel[visible],streams['visible'],device)
    return output


@contextlib.contextmanager
def sampling_context(scheduler,arm,base_seed,coeff,sink,device='cuda'):
    import torch
    from diffusers.schedulers.scheduling_vq_diffusion import VQDiffusionSchedulerOutput
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    group,address=groups(),addresses()
    streams={d:torch.Generator(device=device).manual_seed(stream_seed(base_seed,arm,d)) for d in DOMAINS}
    original_step,original_q=scheduler.step,scheduler.q_posterior
    pending={}
    seen=set()
    def posterior(log_p_x_0,x_t,t):
        result=original_q(log_p_x_0,x_t,t)
        if 11<=int(t)<=60:
            pending[int(t)]=result.detach()
        return result
    def step(model_output,timestep,sample,generator=None,return_dict=True):
        t=int(timestep)
        if t in seen or generator is None:
            raise RuntimeError('Duplicate timestep or missing native RNG')
        begin=time.monotonic()
        history=sample.detach().cpu().numpy().ravel().copy()
        row=dict(timestep=t,intervened=bool(arm!='N' and 11<=t<=60))
        if not 11<=t<=60:
            result=original_step(model_output,timestep,sample,generator=generator,return_dict=return_dict)
            tokens=(result.prev_sample if return_dict else result[0]).detach().cpu().numpy().ravel()
        else:
            if arm=='N':
                result=original_step(model_output,timestep,sample,generator=generator,return_dict=return_dict)
                native=(result.prev_sample if return_dict else result[0]).detach().cpu().numpy().ravel()
                raw=pending.pop(t)
            else:
                # Exactly the native step's q_posterior after the pipeline's existing
                # guidance/truncation/clamp; no surrogate predictor or extra forward.
                raw=original_q(model_output,sample,timestep)
            if tuple(raw.shape)!=(1,K+1,N):
                raise ValueError('Unexpected transition interface')
            kernel=normalize_logits(raw.detach().cpu().numpy()[0].T)
            current_mask=history==MASK
            events=torch.rand(N,device=device,dtype=torch.float64,generator=streams['event']).cpu().numpy()
            initial=current_mask&(events<1-kernel[:,MASK])
            q,beta,kl,selections,records,ledger,contribution=prepare_history(kernel,history,initial,coeff,group,address,t)
            if np.max(kl)>.1+1e-12 or np.max(beta)>1 or not np.isfinite(q).all():
                raise ArithmeticError('Content budget invalid')
            selected=selections['B0' if arm=='N' else arm]
            # N uses the untouched native draw; its S0 and four shadows are diagnostics.
            tokens=native if arm=='N' else draw_factorized(kernel,q,history,selected,streams,device)
            if arm!='N':
                tensor=torch.as_tensor(tokens.reshape(1,N),device=sample.device,dtype=sample.dtype)
                result=VQDiffusionSchedulerOutput(prev_sample=tensor) if return_dict else (tensor,)
            eligible=current_mask&(kernel[:,MASK]>=.005)&(kernel[:,MASK]<=.995)
            delta=np.count_nonzero(selections['P']!=selections['P-E'])
            expected=ledger+contribution[:,selected].sum(1)
            row.update(initial_selected=np.flatnonzero(initial).tolist(),
                executed_selected=[] if arm=='N' else np.flatnonzero(selected).tolist(),
                shadows=records,ledger=ledger.tolist(),initial_count=int(initial.sum()),
                eligible_count=int(eligible.sum()),
                opportunity=bool(np.any(initial&eligible) and np.any(~initial&eligible)),
                P_vs_PE_different=bool(delta),P_vs_PE_symmetric_difference=int(delta),
                P_vs_PE_difference_ratio=float(delta/max(1,initial.sum())),
                expected_J=None if arm=='N' else float(expected.min()),
                expected_bottleneck=None if arm=='N' else int(expected.argmin()),
                selected_conditional_kl_sum=None if arm=='N' else float(kl[selected].sum()),
                conditional_kl_max=float(kl.max()),beta_max=float(beta.max()),
                mask_probability_range=([float(kernel[current_mask,MASK].min()),float(kernel[current_mask,MASK].max())]
                                        if current_mask.any() else None))
        actual=np.einsum('hig,ig->h',coeff,token_field(tokens),optimize=False)
        is_content=tokens!=MASK
        hit=np.zeros(N,bool)
        hit[is_content]=group[tokens[is_content]]==address[is_content]
        row.update(incoming_mask_count=int((history==MASK).sum()),outgoing_mask_count=int((tokens==MASK).sum()),
            newly_unmasked=int(((history==MASK)&is_content).sum()),actual_address_hits=int(hit.sum()),
            actual_ideal_min_margin=float(actual.min()),actual_bottleneck=int(actual.argmin()),
            scheduler_and_diagnostic_seconds=time.monotonic()-begin,
            rng_states={d:streams[d].get_state().cpu().numpy().tolist() for d in DOMAINS},
            native_rng_state=generator.get_state().cpu().numpy().tolist())
        sink(row,history,tokens.copy())
        seen.add(t)
        return result
    scheduler.q_posterior,scheduler.step=posterior,step
    try:
        yield seen
    finally:
        scheduler.step,scheduler.q_posterior=original_step,original_q
