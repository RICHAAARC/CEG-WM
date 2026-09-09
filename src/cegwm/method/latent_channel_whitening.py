"""Fixed channel covariance shrinkage for one processed observation, NumPy only.

The observation can contain anchors and is not a pure background sample.
A returned matrix is numerically usable, not evidence of synchronization.
Estimate once per observation and hold W fixed across all candidate angles.
"""
import numpy as np

SHRINKAGE = 0.25


def _spectrum_diagnostics(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    positive = np.maximum(eigenvalues, 0.0)
    total = float(positive.sum())
    if total > 0:
        probabilities = positive[positive > 0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        effective_rank = 0.0
    tolerance = np.finfo(np.float64).eps * len(eigenvalues) * max(float(positive.max()), 0.0)
    numerical_rank = int(np.count_nonzero(positive > tolerance))
    condition = (float(eigenvalues[-1] / eigenvalues[0])
                 if eigenvalues[0] > tolerance else None)
    return {'eigenvalues': eigenvalues.tolist(), 'condition_number': condition,
            'effective_rank': effective_rank, 'numerical_rank': numerical_rank}


def estimate_whitener(x):
    """Return (K^-1/2, diagnostics), or (None, unreliable diagnostics).

    S = centered_X @ centered_X.T / N; s = trace(S) / C;
    K = .75*S + .25*s*I. Spatial channel means are removed only to
    estimate covariance. Rank-deficient S is allowed when K is usable.
    Floor is determined by float64 precision and dimension, never fitted.
    """
    observation = np.asarray(x, dtype=np.float64)
    if observation.ndim != 3 or observation.shape[0] < 1 or np.prod(observation.shape[1:]) < 2:
        raise ValueError('channel whitening requires CHW with at least two spatial samples')
    channels = observation.shape[0]
    diagnostics = {'reliable': False, 'reason': None, 'shrinkage': SHRINKAGE,
        'channels': channels, 'spatial_samples': int(np.prod(observation.shape[1:])),
        'source': 'current processed observation, possibly containing anchor; not pure background',
        'S': None, 'K': None, 'isotropic_variance': None,
        'floor': None, 'floor_required': False, 'floor_applied': False}
    if not np.isfinite(observation).all():
        diagnostics['reason'] = 'NONFINITE_OBSERVATION'
        return None, diagnostics
    flat = observation.reshape(channels, -1)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        centered = flat - flat.mean(axis=1, keepdims=True)
        covariance = centered @ centered.T / flat.shape[1]
        covariance = (covariance + covariance.T) * 0.5
        isotropic = float(np.trace(covariance) / channels)
    if not np.isfinite(covariance).all() or not np.isfinite(isotropic):
        diagnostics['reason'] = 'NONFINITE_COVARIANCE'
        return None, diagnostics
    diagnostics['isotropic_variance'] = isotropic
    kernel = (1.0-SHRINKAGE)*covariance + SHRINKAGE*isotropic*np.eye(channels)
    try:
        s_eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    except np.linalg.LinAlgError:
        diagnostics['reason'] = 'EIGENDECOMPOSITION_FAILED'
        return None, diagnostics
    diagnostics['S'] = _spectrum_diagnostics(s_eigenvalues)
    diagnostics['K'] = _spectrum_diagnostics(eigenvalues)
    floor = max(np.finfo(np.float64).tiny,
                np.finfo(np.float64).eps * channels * max(float(eigenvalues[-1]), 0.0))
    diagnostics['floor'] = floor
    diagnostics['floor_required'] = bool(np.any(eigenvalues < floor))
    if isotropic <= 0:
        diagnostics['reason'] = 'ZERO_SPATIAL_VARIANCE'
        return None, diagnostics
    if not np.isfinite(eigenvalues).all() or float(eigenvalues[0]) <= floor:
        diagnostics['reason'] = 'NUMERICALLY_DEGENERATE_SHRUNK_COVARIANCE'
        return None, diagnostics
    # Relative floor can only guard numerical roundoff; it is not a method parameter.
    inverse_roots = 1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    diagnostics['floor_applied'] = diagnostics['floor_required']
    whitening = (eigenvectors * inverse_roots[None, :]) @ eigenvectors.T
    whitening = (whitening + whitening.T) * 0.5
    if not np.isfinite(whitening).all():
        diagnostics['reason'] = 'NONFINITE_WHITENER'
        return None, diagnostics
    diagnostics.update(reliable=True, reason='NUMERICALLY_USABLE')
    return whitening, diagnostics


def apply_whitener(W, x):
    """Left-multiply channels by the supplied fixed W; no centering or re-estimation."""
    whitening = np.asarray(W, dtype=np.float64)
    observation = np.asarray(x, dtype=np.float64)
    if observation.ndim != 3 or whitening.shape != (observation.shape[0], observation.shape[0]):
        raise ValueError('whitener shape must match CHW channels')
    if not np.isfinite(whitening).all() or not np.isfinite(observation).all():
        raise ValueError('whitener and observation must be finite')
    return (whitening @ observation.reshape(observation.shape[0], -1)).reshape(observation.shape)
