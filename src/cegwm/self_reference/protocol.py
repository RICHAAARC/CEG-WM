"""Public key mapping, paired schedules, fixed spatial aggregation and gates."""
import hashlib
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import DCT_ORDER, SHIFTS


def digest(domain, *values):
    return hashlib.sha256(json.dumps(["cegwm-self-reference-v1", domain, *values],
                                     separators=(",", ":"), ensure_ascii=True).encode()).digest()


def key_bits(key):
    raw = digest("message-bits", key)[:4]
    return torch.tensor([1.0 if (byte >> bit) & 1 else -1.0
                         for byte in raw for bit in range(7, -1, -1)])


def candidates(key):
    """K17 in fixed order; skip duplicate strings AND 32-bit collisions."""
    keys, seen_bits = [key], {tuple(key_bits(key).tolist())}
    counter = 0
    while len(keys) < 17:
        query = "wrong-v1:" + digest("wrong-key-query", key, counter).hex()
        counter += 1
        bits = tuple(key_bits(query).tolist())
        if query not in keys and bits not in seen_bits:
            keys.append(query)
            seen_bits.add(bits)
    return keys


def reference(nonce, size):
    """Nonce alone seeds p. No source, latent, message or key argument exists."""
    generator = torch.Generator().manual_seed(int.from_bytes(digest("reference", nonce)[:8], "big"))
    coefficients = torch.randn(len(DCT_ORDER), generator=generator, dtype=torch.float64)
    x = torch.arange(size, dtype=torch.float64) + 0.5
    basis = torch.stack([torch.cos(math.pi * u * x[:, None] / size)
                         * torch.cos(math.pi * v * x[None, :] / size) for u, v in DCT_ORDER])
    p = (coefficients[:, None, None] * basis).sum(0)
    return (p / p.square().mean().sqrt()).float()[None]


def schedule(seed, step, rows, batch_size):
    rng = random.Random(int.from_bytes(digest("schedule", seed, step), "big"))
    # Nonces depend only on seed/update/slot, independently of selected source/key.
    return [(rng.randrange(rows), rng.randrange(len(SHIFTS)),
             digest("train-reference-nonce", seed, step, slot).hex()) for slot in range(batch_size)]


def shift_rgb(pixels, dx, dy):
    """Exact integer translation, zero RGB padding; supports uint8 or float NCHW."""
    if type(dx) is not int or type(dy) is not int:
        raise TypeError("RGB translation must be integer pixels")
    h, w = pixels.shape[-2:]
    result = torch.zeros_like(pixels)
    if abs(dx) < w and abs(dy) < h:
        result[..., max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
            pixels[..., max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return result


def reference_target(p, dx, dy, downsample):
    """Bilinear LATENT supervision proxy, NOT a model of exact RGB/VAE equivariance."""
    n, _, h, w = p.shape
    theta = p.new_tensor([[1, 0, -2 * dx / (downsample * w)],
                          [0, 1, -2 * dy / (downsample * h)]])[None].expand(n, -1, -1)
    grid = F.affine_grid(theta, p.shape, align_corners=False)
    return F.grid_sample(p, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def interior(x, border):
    if border <= 0 or min(x.shape[-2:]) <= 2 * border:
        raise ValueError("empty or unsupported fixed scoring region")
    return x[..., border:-border, border:-border]


def relation(p, q, border, epsilon=1e-8):
    p, q = interior(p, border).float(), interior(q, border).float()
    return 2 * (p * q).sum((-2, -1)) / (p.square().sum((-2, -1)) + q.square().sum((-2, -1)) + epsilon)


def ordinary_logits(q, border):
    return interior(q, border).float().mean((-2, -1))


def score(evidence, query_bits):
    # The only query-dependent operation in the reader path.
    return (evidence[:, None, :] * query_bits[None, :, :]).mean(-1)


def summarize_prediction(evidence, key):
    values = evidence.detach().float().cpu()
    bits = key_bits(key)
    scores = score(values[None], torch.stack([key_bits(k) for k in candidates(key)]))[0]
    correct = (values.sign() == bits)
    finite = bool(torch.isfinite(values).all())
    return {"top1": bool(finite and (scores[0] > scores[1:].max())),
            "bit_accuracy": float(correct.float().mean()) if finite else 0.0,
            "exact": bool(finite and correct.all()), "scores": scores.tolist() if finite else None}


def wilson(successes, total):
    if total == 0:
        return [0.0, 1.0]
    z, p = 1.95996398454, successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [max(0, center-half), min(1, center+half)]


def stage_a(rates, complete, config):
    arm_pass = {a: rates[a] >= config.a_top1_min for a in ("B0", "B1", "C")}
    if not complete:
        return {"pass": False, "status": "engineering_error", "B": "skipped", "arm_pass": arm_pass}
    passed = arm_pass["C"] and all(
        rates["C"] >= rates[a] - config.a_max_baseline_gap for a in ("B0", "B1"))
    status = "continue_pilot" if passed else (
        "stop_C_v1" if any(arm_pass[a] for a in ("B0", "B1")) else "implementation_budget_optimization_unresolved")
    return {"pass": passed, "status": status, "B": "eligible" if passed else "skipped", "arm_pass": arm_pass}


def bootstrap_sources(values, seed, count):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(count)]
    return np.quantile(means, [0.025, 0.975]).tolist()
