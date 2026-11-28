"""Probability and bounded set selection shared by G0 and later authorized G1."""
import hashlib
import numpy as np


def normalize_logits(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('expected finite [positions, classes] logits')
    p = np.exp(x - x.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def biased_content(kernel, group, address):
    """Full candidates; zeros preserved; KL concerns content only."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if not np.allclose(kernel.sum(1), 1., atol=1e-12, rtol=1e-12) or np.any(kernel < 0):
        raise ValueError('kernel not normalized')
    mass = kernel[:, :-1].sum(1)
    p = np.divide(kernel[:, :-1], mass[:, None], out=np.zeros_like(kernel[:, :-1]),
                  where=mass[:,None] > 0)
    target = np.asarray(group)[None,:] == np.asarray(address)[:,None]
    u = (p * target).sum(1)
    allowed = (u >= 1e-4) & (u <= 1 - 1e-4)
    def kl_at(beta):
        z = 1 + u * np.expm1(beta)
        tilted = u * np.exp(beta) / z
        return tilted * beta - np.log(z)
    lo, hi = np.zeros(len(u)), allowed.astype(float)
    full = allowed & (kl_at(hi) <= .10)
    lo[full] = 1.
    for _ in range(24):
        mid = (lo + hi) / 2
        good = kl_at(mid) <= .10
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)
    beta = np.where(allowed, lo, 0.)
    q = p * np.exp(beta[:,None] * target)
    denom = q.sum(1)
    q = np.divide(q, denom[:,None], out=np.zeros_like(q), where=denom[:,None] > 0)
    return q, beta, np.maximum(0., kl_at(beta))


def pool_order(ids, timestep):
    return sorted(ids, key=lambda i: (
        hashlib.sha256(b'GDS-v2-pool:' + int(timestep).to_bytes(4,'big')
                       + int(i).to_bytes(4,'big')).digest(), int(i)))


def select_set(initial, current_mask, mask_probability, timestep,
               contribution, ledger, target_probability, address, visible_counts,
               objective='P', limit=4, pool_size=32):
    """No draws. Same history/initial/pool gives comparable shadow decisions."""
    initial = np.asarray(initial, dtype=bool)
    current_mask = np.asarray(current_mask, dtype=bool)
    if np.any(initial & ~current_mask):
        raise ValueError('initial set contains visible positions')
    eligible = current_mask & (mask_probability >= .005) & (mask_probability <= .995)
    outgoing = pool_order(np.flatnonzero(initial & eligible), timestep)[:pool_size]
    incoming = pool_order(np.flatnonzero(~initial & eligible), timestep)[:pool_size]
    selected = initial.copy()
    changes = []
    def value(s):
        if objective in ('P', 'P-E'):
            offsets = ledger if objective == 'P' else np.zeros_like(ledger)
            return float(np.min(offsets + contribution[:,s].sum(1)))
        if objective == 'B1':
            return float(target_probability[s].sum())
        if objective == 'B2':
            counts = visible_counts + np.bincount(address[s], weights=target_probability[s], minlength=16)
            return float(np.log1p(counts).sum())
        raise ValueError('unknown objective')
    for _ in range(limit):
        before = value(selected)
        best = None
        # Numeric index tie-breaking, independent of hash pool ordering.
        for i in sorted(outgoing):
            for j in sorted(incoming):
                candidate = selected.copy()
                candidate[i], candidate[j] = False, True
                after = value(candidate)
                tolerance = 1e-10 * max(1., abs(before), abs(after))
                if after - before > tolerance and (best is None or after > best[0]):
                    best = (after, i, j)
        if best is None:
            break
        after, i, j = best
        selected[i], selected[j] = False, True
        outgoing.remove(i)
        incoming.remove(j)
        # Original event mass ratio only, never labelled a KL estimate.
        mi, mj = mask_probability[i], mask_probability[j]
        ratio = np.log(mi) + np.log1p(-mj) - np.log1p(-mi) - np.log(mj)
        changes.append(dict(out=int(i), into=int(j), before=before, after=after,
                            original_event_log_mass_change=float(ratio)))
    return selected, dict(swaps=changes, initial_count=int(initial.sum()),
                         eligible_count=int(eligible.sum()),
                         out_pool_count=min(pool_size, int((initial & eligible).sum())),
                         in_pool_count=min(pool_size, int((~initial & eligible).sum())),
                         objective=objective, final_value=value(selected))
