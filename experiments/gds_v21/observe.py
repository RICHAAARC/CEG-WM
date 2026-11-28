"""Read-only native sampler observer. Never performs a watermark intervention."""
import contextlib


@contextlib.contextmanager
def native_trace(scheduler, sink, capture_kernel=False):
    """Single-threaded. Capture the ACTUAL pre-Gumbel return, then replay on a clone RNG.

    Baseline mode only records the output of the untouched native step. Kernel mode
    additionally taps q_posterior (model_output at t=0), without replacing its result.
    Replay checks the capture point; it cannot change the generator used by step.
    """
    import torch
    from diffusers.schedulers.scheduling_vq_diffusion import gumbel_noised
    original_step, original_posterior = scheduler.step, scheduler.q_posterior
    pending, seen = {}, set()

    def posterior(log_p_x_0, x_t, t):
        result = original_posterior(log_p_x_0, x_t, t)
        if int(t) in pending:
            raise RuntimeError('Duplicate posterior call')
        pending[int(t)] = result.detach().clone()
        return result

    def step(model_output, timestep, sample, generator=None, return_dict=True):
        t = int(timestep)
        if t in seen or generator is None:
            raise RuntimeError('Repeated timestep or missing explicit generator')
        history = sample.detach().cpu().numpy().copy()
        before = generator.get_state().clone()
        if capture_kernel and t == 0:
            pending[t] = model_output.detach().clone()
        result = original_step(model_output,timestep,sample,generator=generator,return_dict=return_dict)
        after = generator.get_state().clone()
        tokens = result.prev_sample if return_dict else result[0]
        raw, replay_ok = None, None
        if capture_kernel:
            raw = pending.pop(t)
            replica = torch.Generator(device=generator.device)
            replica.set_state(before)
            replay = gumbel_noised(raw,replica).argmax(dim=1)
            replay_ok = bool(torch.equal(replay,tokens) and torch.equal(replica.get_state(),after))
            if not replay_ok or not torch.equal(generator.get_state(),after):
                raise AssertionError('Captured raw logits do not reproduce the native step/RNG')
        sink(t,raw,history,tokens.detach().cpu().numpy().copy(),after.cpu().numpy().copy(),replay_ok)
        if not torch.equal(generator.get_state(),after):
            raise AssertionError('Observer changed native RNG state')
        seen.add(t)
        return result

    scheduler.step = step
    if capture_kernel:
        scheduler.q_posterior = posterior
    try:
        yield seen
    finally:
        scheduler.step, scheduler.q_posterior = original_step, original_posterior
