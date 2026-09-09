"""Shared 2x2 survival allocator; embedding only, no detector-side masks."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn.functional as F
from cegwm.method.content_adaptive import ContentAllocation


def allocation_from_logits(logits):
    values = np.asarray(logits, dtype=float).reshape(2, 2)
    if not np.isfinite(values).all():
        raise ValueError('allocator logits must be finite')
    bounded = np.tanh(values)
    bounded -= bounded.mean()
    weights = 1 + .24 * F.interpolate(torch.tensor(bounded)[None, None],
        size=(4, 4), mode='bilinear', align_corners=False).numpy().reshape(-1)
    weights /= weights.mean()
    result = tuple(float(x) for x in weights)
    return ContentAllocation(result, result, .5, .5, (0.,) * 6)


def macro_features(semantic, texture, latents):
    def pool(values):
        return np.asarray(values, dtype=float).reshape(2, 2, 2, 2).mean(axis=(1, 3)).reshape(4)
    # Latent energy is a candidate feature, not a generation Jacobian.
    energy = F.adaptive_avg_pool2d(latents.detach().double().square().mean(1, keepdim=True), (2, 2))
    return np.column_stack((pool(semantic), pool(texture), energy.cpu().numpy().reshape(4)))


@dataclass(frozen=True)
class SurvivalAllocator:
    mean: tuple
    scale: tuple
    coefficients: tuple
    fit_ids: tuple
    cost_penalty: float = 1.0

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        logits = (x - np.asarray(self.mean)) / np.asarray(self.scale) @ np.asarray(self.coefficients)
        return allocation_from_logits(logits)

    def to_dict(self):
        return asdict(self)


def fit_allocator(features, utilities, fit_ids, *, cost_penalty=1., ridge=1.):
    x = np.asarray(features, dtype=float).reshape(-1, 3)
    y = np.asarray(utilities, dtype=float).reshape(-1)
    if len(y) != len(x) or len(y) < 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite complete fit observations required')
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    z = (x - mean) / scale
    coef = np.linalg.solve(z.T @ z + ridge * np.eye(3), z.T @ (y - y.mean()))
    return SurvivalAllocator(tuple(mean), tuple(scale), tuple(coef), tuple(fit_ids), cost_penalty)
