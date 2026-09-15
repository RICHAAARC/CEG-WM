"""Independent V1 landmark carrier. CPU functions receive observations, never truth.

Image reader is in runtime/latent_landmark_sd35.py; synthetic observations here
are engineering test inputs and do not constitute a final-image observation.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import numpy as np
import torch
import torch.nn.functional as F

CENTERS = np.array([(0.22, .23), (.49, .19), (.77, .27), (.81, .53),
                    (.71, .78), (.44, .81), (.20, .69), (.18, .44)]) * 63
DOMAIN = b"CEG-WM/Latent-Landmark-Sync-V1/channel-codes\x00"


def codes(key):
    key = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    if not key:
        raise ValueError("empty detection key")
    prefix = DOMAIN + len(key).to_bytes(8, "big") + key
    # One independent SHA256 word per row-major entry; unsigned big-endian.
    a = np.array([(int.from_bytes(hashlib.sha256(prefix + i.to_bytes(4, "big")).digest()[:8], "big")
                   + .5) / 2**64 * 2 - 1 for i in range(128)]).reshape(16, 8)
    q, r = np.linalg.qr(a, mode="reduced")
    if np.min(np.abs(np.diag(r))) < 1e-12:
        raise ValueError("degenerate channel code matrix")
    return q * np.where(np.diag(r) >= 0, 1., -1.)


def kernel(dx, dy):
    """Raised cosine disk r<3.5, weighted DC removal, then unit L2.

    The nearest integer-centered 7x7 footprint is sampled at fractional offsets.
    Multiplying the DC correction by the same window retains edge taper.
    """
    r = np.sqrt(dx * dx + dy * dy)
    w = np.where(r < 3.5, .5 * (1 + np.cos(np.pi * np.minimum(r, 3.5) / 3.5)), 0.)
    g = np.exp(-r * r / 2)
    k = w * (g - np.sum(w * g) / np.sum(w))
    return k / np.linalg.norm(k)


def carrier(key, rms):
    if rms not in (.005, .01, .02):
        raise ValueError("V1 RMS must be one of the three frozen candidates")
    result = np.zeros((16, 64, 64), dtype=np.float64)
    code = codes(key)
    for i, (x, y) in enumerate(CENTERS):
        ix, iy = np.floor(np.array([x, y]) + .5).astype(int)
        yy, xx = np.mgrid[iy-3:iy+4, ix-3:ix+4]
        result[:, iy-3:iy+4, ix-3:ix+4] += code[:, i, None, None] * kernel(xx-x, yy-y)
    result *= rms / np.sqrt(np.mean(result * result))
    return result[None]


def responses(observation, key):
    z = torch.as_tensor(observation, dtype=torch.float64, device="cpu")
    if tuple(z.shape) != (1, 16, 64, 64) or not bool(torch.isfinite(z).all()):
        raise ValueError("observation must be finite 1x16x64x64")
    # Same linear high-pass on observation and template, zero padding throughout.
    hp = z - F.avg_pool2d(z, 9, stride=1, padding=4)
    projected = torch.einsum("bcyx,ci->biyx", hp, torch.from_numpy(codes(key)))
    yy, xx = np.mgrid[-3:4, -3:4]
    raw = torch.zeros((1, 1, 15, 15), dtype=torch.float64)
    raw[0, 0, 4:11, 4:11] = torch.from_numpy(kernel(xx, yy))
    template = raw - F.avg_pool2d(raw, 9, stride=1, padding=4)
    template /= torch.linalg.vector_norm(template)
    numerator = F.conv2d(projected.reshape(8, 1, 64, 64), template, padding=7)
    energy = F.conv2d(projected.reshape(8, 1, 64, 64).square(), torch.ones_like(template), padding=7)
    response = torch.where(energy > 1e-24, numerator / energy.sqrt().clamp_min(1e-12), 0.)
    response = response[:, 0].numpy()
    # Full 15x15 high-passed support; boundary peaks are not eligible.
    response[:, :7] = -np.inf; response[:, -7:] = -np.inf
    response[:, :, :7] = -np.inf; response[:, :, -7:] = -np.inf
    return response


def find_peaks(response):
    peaks = []
    for identity, original in enumerate(response):
        work = original.copy()
        local_max = F.max_pool2d(torch.from_numpy(original)[None,None], 3, 1, 1)[0,0].numpy()
        work[original < local_max] = -np.inf
        for rank in range(3):
            flat = int(np.argmax(work))  # ties: y then x
            y, x = np.unravel_index(flat, work.shape)
            score = float(work[y, x])
            if not math.isfinite(score) or score <= 0:
                break
            p = np.array([float(x), float(y)])
            status = "integer_fallback"
            if 7 < x < 56 and 7 < y < 56:
                hxx = original[y, x+1] - 2*score + original[y, x-1]
                hyy = original[y+1, x] - 2*score + original[y-1, x]
                hxy = (original[y+1,x+1]-original[y+1,x-1]-original[y-1,x+1]+original[y-1,x-1])/4
                h = np.array([[hxx, hxy], [hxy, hyy]])
                gradient = np.array([original[y,x+1]-original[y,x-1], original[y+1,x]-original[y-1,x]])/2
                if np.max(np.linalg.eigvalsh(h)) < -1e-10 and np.linalg.cond(h) < 1e6:
                    delta = -np.linalg.solve(h, gradient)
                    if np.max(np.abs(delta)) <= .75:
                        p += delta; status = "quadratic"
            peaks.append(dict(identity=identity, rank=rank, xy=p.tolist(), score=score, refinement=status))
            yy, xx = np.ogrid[:64, :64]
            work[(xx-x)**2+(yy-y)**2 <= 9] = -np.inf
    return peaks


def transform(h, points):
    points = np.asarray(points)
    return points @ h[:2, :2].T + h[:2, 2]


def fit_rigid(source, target):
    source, target = np.asarray(source), np.asarray(target)
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    rotation = vt.T @ np.diag([1., np.linalg.det(vt.T @ u.T)]) @ u.T
    h = np.eye(3); h[:2,:2] = rotation; h[:2,2] = b - rotation @ a
    return h


def legal(h):
    return abs(math.degrees(math.atan2(h[1,0], h[0,0]))) <= 20 + 1e-9


def assign(peaks, h):
    """Exact deterministic maximum-cardinality assignment (<=24 candidate peaks).

    Each identity once; observations <=0.75 pixels apart conflict even across
    identities. Objective: count, summed response, negative squared residual.
    """
    groups = [[] for _ in range(8)]
    for p in peaks:
        error = float(np.linalg.norm(transform(h, [p["xy"]])[0] - CENTERS[p["identity"]]))
        if error <= .5:
            groups[p["identity"]].append((p, error))
    best, best_key = [], (-1, -1., -math.inf)
    def visit(i, selected):
        nonlocal best, best_key
        if len(selected) + 8-i < best_key[0]:
            return
        if i == 8:
            objective = (len(selected), sum(p["score"] for p,e in selected), -sum(e*e for p,e in selected))
            if objective > best_key:
                best, best_key = selected[:], objective
            return
        for p,e in groups[i]:
            if all(np.linalg.norm(np.array(p["xy"])-q["xy"]) > .75 for q,_ in selected):
                visit(i+1, selected+[(p,e)])
        visit(i+1, selected)
    visit(0, [])
    return best, best_key


def solve(peaks, peak_threshold=0., ambiguity_threshold=0.):
    eligible = [p for p in peaks if p["score"] > peak_threshold]
    hypotheses = []
    for p, q in itertools.combinations(eligible, 2):
        if p["identity"] == q["identity"]:
            continue
        target = CENTERS[[p["identity"], q["identity"]]]
        source = np.array([p["xy"], q["xy"]])
        if min(np.linalg.norm(source[0]-source[1]), np.linalg.norm(target[0]-target[1])) < 8:
            continue
        h = fit_rigid(source, target)
        if not legal(h):
            continue
        chosen, objective = assign(eligible, h)
        if len(chosen) < 4:
            continue
        for _ in range(2):
            h = fit_rigid([p["xy"] for p,e in chosen], CENTERS[[p["identity"] for p,e in chosen]])
            chosen, objective = assign(eligible, h)
            if len(chosen) < 4:
                break
        if len(chosen) < 4 or not legal(h):
            continue
        points = np.array([p["xy"] for p,e in chosen])
        if np.max(np.linalg.norm(points[:,None]-points[None,:], axis=-1)) < .35*64:
            continue
        hypotheses.append((objective, h, chosen))
    hypotheses.sort(key=lambda item: item[0], reverse=True)
    unique = []
    for entry in hypotheses:
        # Duplicate if max displacement of the 8 public centers <=0.25 latent px.
        if not any(np.max(np.linalg.norm(transform(entry[1], CENTERS)-transform(old[1], CENTERS), axis=1)) <= .25 for old in unique):
            unique.append(entry)
    result = dict(accepted=False, reason="insufficient_consistent_identities", H_latent=None,
                  matches=[], hypothesis_count=len(unique), runner_up=None, ambiguity_margin=None)
    if not unique:
        return result
    objective, h, chosen = unique[0]
    # Integer identity advantage dominates bounded mean correlation advantage.
    strength = lambda row: row[0][0] + row[0][1] / row[0][0]
    margin = strength(unique[0]) - strength(unique[1]) if len(unique)>1 else 2.
    accepted = margin > ambiguity_threshold
    result.update(accepted=accepted, reason="accepted" if accepted else "ambiguous", H_latent=h.tolist(),
                  matches=[dict(p, residual=e) for p,e in chosen], ambiguity_margin=margin,
                  runner_up=None if len(unique)<2 else dict(H_latent=unique[1][1].tolist(), count=unique[1][0][0]))
    return result


def latent_to_rgb_h(h):
    """Chosen cell-center convention, not a claim about learned VAE centers."""
    a = np.array([[8.,0.,3.5],[0.,8.,3.5],[0.,0.,1.]])
    return a @ np.asarray(h) @ np.linalg.inv(a)


def public_h(h):
    n = np.array([[2/511,0,-1],[0,2/511,-1],[0,0,1.]])
    return n @ latent_to_rgb_h(h) @ np.linalg.inv(n)


def pil_sampling_public_h(h):
    """Adapt integer-center geometry to PIL's half-integer sampling centers.

    Keep public_h/latent_to_rgb_h unchanged for physical error reporting. The
    main rectifier passes its denormalized inverse H directly to PIL, so this
    candidate-specific boundary conversion is needed for rotation (not shifts).
    """
    n = np.array([[2/511,0,-1],[0,2/511,-1],[0,0,1.]])
    t = np.array([[1.,0,.5],[0,1.,.5],[0,0,1.]])
    return n @ t @ latent_to_rgb_h(h) @ np.linalg.inv(t) @ np.linalg.inv(n)


def read_observation(observation, key, peak_threshold=0., ambiguity_threshold=0.):
    peaks = find_peaks(responses(observation, key))
    return dict(peaks=peaks, geometry=solve(peaks, peak_threshold, ambiguity_threshold))
