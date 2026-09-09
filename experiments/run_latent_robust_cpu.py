"""Fixed four-arm CPU diagnosis of eight existing observations; no model loading.

Angle truth is read only after all blind fits and diagnostic curves are complete.
Paired differences are privileged diagnostics, never inputs to the blind reader.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cegwm.method.latent_sync import estimate_rotation, warp_field
from cegwm.method.latent_robust_sync import (
    make_objective, estimate_robust_rotation, preprocess_field, warp_continuous,
)
from cegwm.method.latent_channel_whitening import apply_whitener

ARMS = ('original', 'whitening', 'transfer', 'combined')
ROLES = ('plain', 'content_only', 'latent_anchor', 'v2_rgb_sync')
CONDITIONS = ('clean', 'rotation')


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def norm(x):
    return float(np.linalg.norm(x))


def metrics(a, b):
    aa, bb, ab = float(np.sum(a*a)), float(np.sum(b*b)), float(np.sum(a*b))
    gain = ab / bb if bb else None
    return dict(actual_norm=np.sqrt(aa), predicted_norm=np.sqrt(bb),
                cosine=ab/(np.sqrt(aa*bb)+1e-12), least_squares_gain=gain,
                gain_adjusted_relative_residual=norm(a-gain*b)/np.sqrt(aa)
                if aa and bb else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations, objectives, rows = {}, {}, []
    grid = np.linspace(-15., 15., 121)
    boundary_angles = [-11.1222293, -11.1221293]
    # Exact eight-file roster: generated diagnostic mask NPZs are not inputs.
    for role in ROLES:
        for condition in CONDITIONS:
            key = role+'__'+condition
            with np.load(args.cache_dir/(key+'.npz'), allow_pickle=False) as data:
                obs = data['observation']
                if obs.shape != (16, 64, 64) or obs.dtype != np.float32 or not np.isfinite(obs).all():
                    raise ValueError('Invalid original observation: '+key)
                observations[key] = obs.astype(np.float64)
            for arm in ARMS:
                row = dict(observation=key, role=role, condition=condition, arm=arm)
                try:
                    obj = make_objective(observations[key], arm)
                    objectives[key, arm] = obj
                    fit = estimate_robust_rotation(observations[key], arm, objective=obj)
                    values = np.array([obj.score(float(a)) for a in grid])
                    best = int(np.argmax(values))
                    local = [i for i in range(121)
                             if (i == 0 or values[i] >= values[i-1])
                             and (i == 120 or values[i] >= values[i+1])]
                    ranked = sorted(local, key=lambda i: values[i], reverse=True)
                    boundary_values = [obj.score(a) for a in boundary_angles]
                    row.update(fit=fit, identity_correlation=obj.score(0.),
                               diagnostic_curve=dict(angles=grid, correlations=values),
                               sampled_max_angle=float(grid[best]), sampled_max_correlation=float(values[best]),
                               sampled_local_maxima=[dict(angle=float(grid[i]), correlation=float(values[i])) for i in ranked],
                               sampled_peak_gap=float(values[ranked[0]]-values[ranked[1]]) if len(ranked)>1 else None,
                               boundary_interval=dict(angles=boundary_angles, correlations=boundary_values,
                                                      delta=boundary_values[1]-boundary_values[0]),
                               processed_observation_norm=norm(obj.processed_observation),
                               processed_template_identity_norm=norm(obj.processed_template(0.)),
                               diagnostics=obj.diagnostics, error=None)
                    if arm == 'original':
                        old = estimate_rotation(observations[key])
                        row['baseline_reproduction'] = dict(
                            angle_abs_error=abs(fit['angle']-old['parameters'][0]),
                            correlation_abs_error=abs(fit['correlation']-old['correlation']))
                except Exception as exc:
                    row['error'] = type(exc).__name__+': '+str(exc)
                rows.append(row)
    # No truth annotations or privileged paired data above this point.
    source_report = json.loads((args.cache_dir/'report.json').read_text())
    audit = source_report['coordinate_audit']
    true_h = np.asarray(audit['H_latent64_reference_to_observed'], float)
    lookup = {(r['observation'], r['arm']): r for r in rows}
    for row in rows:
        truth = 0. if row['condition'] == 'clean' else float(audit['forward_screen_degrees'])
        row['angle_annotation'] = dict(angle=truth, reader_center=[31.5, 31.5],
            absolute_error=abs(row['fit']['angle']-truth) if not row['error'] else None,
            denotes_anchor_presence=row['role']=='latent_anchor')
    decomposition = []
    for condition in CONDITIONS:
        full = observations['latent_anchor__'+condition]
        background = observations['content_only__'+condition]
        delta = full-background
        for arm in ARMS:
            row = lookup['latent_anchor__'+condition, arm]
            if row['error']:
                decomposition.append(dict(condition=condition, arm=arm, error=row['error']))
                continue
            obj = objectives['latent_anchor__'+condition, arm]
            transfer = arm in ('transfer', 'combined')
            use_w = arm in ('whitening', 'combined')
            def process(x):
                value = preprocess_field(x, transfer=transfer)
                return apply_whitener(obj.W, value) if use_w else value
            if use_w and obj.W is None:
                decomposition.append(dict(condition=condition, arm=arm, error='degenerate whitening'))
                continue
            arrays = {'full':process(full), 'background':process(background), 'delta':process(delta)}
            angular = []
            for label, angle in [('truth_angle_only', row['angle_annotation']['angle']),
                                 ('returned_angle', row['fit']['angle'])]:
                template = obj.processed_template(angle)
                denominator = norm(arrays['full'])*norm(template)+1e-12
                contributions = {k:float(np.sum(v*template))/denominator for k,v in arrays.items()}
                angular.append(dict(label=label, angle=angle, common_denominator=denominator,
                    contributions=contributions,
                    additivity_error=contributions['full']-contributions['background']-contributions['delta']))
            raw_processed = {k:preprocess_field(v, transfer=transfer)
                             for k,v in [('full',full),('background',background),('delta',delta)]}
            decomposition.append(dict(condition=condition, arm=arm, error=None,
                W_source='anchor_full_current_observation_only' if use_w else 'disabled_identity', angular=angular,
                processed_norms={k:norm(v) for k,v in arrays.items()},
                before_whitening_norms={k:norm(v) for k,v in raw_processed.items()},
                delta_background_norm_ratio=norm(arrays['delta'])/(norm(arrays['background'])+1e-12),
                before_whitening_delta_background_norm_ratio=norm(raw_processed['delta'])/(norm(raw_processed['background'])+1e-12),
                linearity_max_abs_error=float(np.max(np.abs(arrays['full']-arrays['background']-arrays['delta']))),
                returned_minus_truth={k:angular[1]['contributions'][k]-angular[0]['contributions'][k] for k in arrays}))
    transfer_results = {}
    delta_clean = observations['latent_anchor__clean']-observations['content_only__clean']
    delta_rotation = observations['latent_anchor__rotation']-observations['content_only__rotation']
    for arm in ARMS:
        obj = objectives.get(('latent_anchor__rotation',arm))
        if obj is None or (arm in ('whitening','combined') and obj.W is None):
            transfer_results[arm] = dict(error='unavailable objective')
            continue
        transfer = arm in ('transfer','combined')
        warped = (warp_continuous if transfer else warp_field)(delta_clean, true_h)
        actual, predicted = [preprocess_field(v, transfer=transfer) for v in (delta_rotation,warped)]
        if arm in ('whitening','combined'):
            actual, predicted = [apply_whitener(obj.W,v) for v in (actual,predicted)]
        transfer_results[arm] = metrics(actual,predicted)
    output = dict(design=dict(observations=8, arms=list(ARMS), planned_fits=32,
        physical_image_pairs=1, shrinkage=.25, transfer_sigma=.5, highpass_sigma=3.,
        bounds=[-15.,15.], xatol=.01, sampled_curve_used_for_selection=False,
        truth_used_by_reader=False, content_scores_used=False, model_calls=0,
        note='Both-side sigma.5 smoothing is a robust comparison approximation, not exact displacement marginalization.'),
        completed_rows=len(rows), failed_rows=sum(r['error'] is not None for r in rows),
        numerically_degenerate_rows=sum(r.get('diagnostics',{}).get('numerically_degenerate',False) for r in rows),
        unsuccessful_optimizer_rows=sum(not r['fit']['success'] for r in rows if not r['error']),
        rows=rows, privileged_decomposition=decomposition,
        privileged_difference_transfer=transfer_results, coordinate_audit=audit,
        limitations=['Eight observations originate from one already-seen development image pair.',
            'No content detection, FPR, new-image generalization or VAE-only causal conclusion.',
            'Absolute correlations from different objectives are not comparable evidence of improvement.',
            'Bounded optimizer return is not a global optimum certificate; grid is report-only.',
            'Whitening estimates mixed host and anchor covariance, and can suppress both.',
            'Known H and reference differences are diagnostic-only, never blind reader inputs.'])
    (args.output_dir/'robust_cpu_result.json').write_text(json.dumps(output, indent=2, default=serial, allow_nan=False))
    print(json.dumps(dict(rows=[{k:r.get(k) for k in ('observation','arm','error')} |
        dict(angle=r.get('fit',{}).get('angle'), sampled_max_angle=r.get('sampled_max_angle')) for r in rows],
        failed_rows=output['failed_rows']), indent=2))


if __name__ == '__main__':
    main()
