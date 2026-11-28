"""Frozen GDS-v2.1 ideal-symbol geometry. Not an RGB synchronizer."""
from functools import lru_cache
import hashlib
import numpy as np

N, K, MASK = 1024, 4096, 4096
V0 = np.full(16, 1 / 16, dtype=np.float64)
IDENTITY = (0., 1., 0., 0.)
R = (IDENTITY, (9.,1.,0.,0.), (-9.,1.,0.,0.), (0.,1.,18.,0.),
     (0.,1.,-18.,0.), (0.,1.,0.,18.), (0.,1.,0.,-18.),
     (0.,1.25,0.,0.), (0.,1.25,12.,-12.))
DELTAS = ((0.,1.,12.,0.), (0.,1.,-12.,0.), (0.,1.,0.,12.),
          (0.,1.,0.,-12.), (6.,1.,0.,0.), (-6.,1.,0.,0.),
          (0.,.9,0.,0.), (0.,1.1,0.,0.))


def addresses():
    row, col = np.indices((32, 32))
    return (4 * (row // 8) + col // 8).ravel()


def groups():
    order = sorted(range(K), key=lambda c: (
        hashlib.sha256(b'GDS-v2-group:' + c.to_bytes(4, 'big')).digest(), c))
    g = np.empty(K, dtype=np.int64)
    g[order] = np.arange(K) % 16
    return g


def rotation(degrees):
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def transform(points, h):
    theta, scale, tx, ty = h
    return (np.asarray(points) - 128.) @ rotation(theta).T * scale + 128. + [tx, ty]


def compose(r, delta):
    translation = np.array(r[2:]) + r[1] * rotation(r[0]) @ np.array(delta[2:])
    return (r[0] + delta[0], r[1] * delta[1], *translation.tolist())


def distance(r, h):
    corners = np.array([[0,0],[0,256],[256,0],[256,256],[128,128]])
    return float(np.linalg.norm(transform(corners, r) - transform(corners, h), axis=1).max())


def in_range(h):
    return -12 <= h[0] <= 12 and .9 <= h[1] <= 1.25 and max(abs(h[2]), abs(h[3])) <= 24


@lru_cache(maxsize=256)
def source_indices(h):
    """[4096 reader cells, 16 subpixels]; -1 means outside source domain."""
    rr, cc = np.indices((64, 64))
    sy, sx = np.indices((4, 4))
    x = cc.ravel()[:, None] * 4 + sx.ravel()[None, :] + .5
    y = 256 - (rr.ravel()[:, None] * 4 + sy.ravel()[None, :] + .5)
    points = np.stack((x, y), axis=-1)
    theta, scale, tx, ty = h
    original = (points - 128. - [tx, ty]) @ rotation(theta) / scale + 128.
    ox, oy = original[..., 0], original[..., 1]
    valid = (ox >= 0) & (ox < 256) & (oy > 0) & (oy <= 256)
    idx = 32 * np.floor((256 - oy) / 8).astype(int) + np.floor(ox / 8).astype(int)
    return np.where(valid, idx, -1)


def render(field, h):
    """Linear one-hot/soft field render; outside points have uniform groups."""
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (N, 16):
        raise ValueError('field must be [1024,16]')
    ids = source_indices(tuple(h))
    return np.where((ids >= 0)[..., None], field[np.maximum(ids, 0)], V0).mean(axis=1)


def template(h):
    return render(np.eye(16)[addresses()], h) - V0


def support(h):
    return float((source_indices(tuple(h)) >= 0).mean())


def score(observation, h):
    return float(np.sum(observation * template(h)) / 4096)


def hypothesis_pairs():
    pairs, excluded = [], []
    for r in R:
        for h in [compose(r, d) for d in DELTAS] + [IDENTITY]:
            h = tuple(float(x) for x in h)
            reason = ('range' if not in_range(h) else 'near_peak' if distance(r,h) <= 8
                      else 'support' if support(h) < .60 else None)
            if reason:
                excluded.append(dict(r=r, h=h, reason=reason))
            elif (r,h) not in pairs:
                pairs.append((r,h))
    if not pairs:
        raise ValueError('empty H: stop, do not silently change geometry')
    return pairs, excluded


def build_coefficients():
    pairs, excluded = hypothesis_pairs()
    coeff = np.zeros((len(pairs), N, 16), dtype=np.float64)
    for k, (r,h) in enumerate(pairs):
        ids = source_indices(r)
        diff = template(r) - template(h)
        for sub in range(16):
            keep = ids[:,sub] >= 0
            np.add.at(coeff[k], ids[keep,sub], diff[keep] / (16 * 4096))
    return coeff, pairs, excluded


def token_field(tokens):
    tokens = np.asarray(tokens).ravel()
    if tokens.shape != (N,) or np.any((tokens < 0) | (tokens > MASK)):
        raise ValueError('invalid token field')
    field = np.tile(V0, (N,1))
    valid = tokens != MASK
    field[valid] = np.eye(16)[groups()[tokens[valid]]]
    return field
