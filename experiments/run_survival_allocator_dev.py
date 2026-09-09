"""Allocator development: explicitly selected real fit/validation, never formal FPR."""
import argparse
import json
from pathlib import Path
import numpy as np
import platform
import subprocess
from importlib.metadata import version, PackageNotFoundError
from cegwm.method.survival_allocator import SurvivalAllocator, fit_allocator


def validate_roster(units, stage, asset=None):
    ids, seeds = set(), set()
    for unit in units:
        if not isinstance(unit['id'], str) or not unit['id'] or not isinstance(unit['prompt'], str) or not unit['prompt'].strip():
            raise ValueError('each unit needs id, prompt and seed')
        if type(unit['seed']) is not int or unit['seed'] < 0:
            raise ValueError('seed must be nonnegative integer')
        if unit['id'] in ids or unit['seed'] in seeds:
            raise ValueError('duplicate unit id or seed')
        ids.add(unit['id']); seeds.add(unit['seed'])
    if not units:
        raise ValueError('empty development roster')
    if stage == 'validation':
        if asset is None:
            raise ValueError('validation requires fit asset')
        identities = ids | {'seed:' + str(s) for s in seeds}
        if identities.intersection(asset.fit_ids):
            raise ValueError('validation overlaps fit ids or seeds')


def local_cost(clean, marked):
    diff = (np.asarray(marked, dtype=float) - np.asarray(clean, dtype=float)) / 255.
    h, w = diff.shape[:2]
    return [float(np.mean(diff[y*h//2:(y+1)*h//2, x*w//2:(x+1)*w//2] ** 2))
            for y in range(2) for x in range(2)]



def score_details(runtime, image):
    """Instrument the existing v2 forced pre/post path; 17 keys per observation once."""
    from experiments import run_paper_main_worker_v2 as v2
    from cegwm.runtime.blind_scoring_v2 import score_branches_v2
    def observe(current):
        branches, mode = score_branches_v2(current, runtime['key'], runtime['assets'], reuse_observation=True)
        scores = branches['weighted_joint']
        registered = scores['registered']
        wrong = max(scores[f'wrong_{i:02d}'] for i in range(16))
        if not np.isfinite([registered, wrong]).all():
            raise ValueError('nonfinite content evidence')
        return {'score':registered-wrong, 'registered':registered, 'max16wrong':wrong,
                'branches':branches, 'observation_mode':mode}
    pre = observe(image)
    result = {'pre':pre, 'post':None, 'selected':'pre', **pre}
    geometry = runtime['assets'].geometry_backend.detect_geometry_continuous(image)
    disposition, detail = v2._geometry_disposition(geometry)
    if disposition == 'OPERATIONAL':
        raise RuntimeError('geometry operational failure: '+str(detail))
    if disposition == 'INVALID_H':
        return {**result, 'route':'GEOMETRY_FAIL_CLOSED'}
    try:
        matrix = v2._raw_h(geometry)
    except LookupError:
        return {**result, 'route':'GEOMETRY_NO_H'}
    except (TypeError, ValueError):
        return {**result, 'route':'GEOMETRY_FAIL_CLOSED'}
    try:
        recovered = v2.rectify_attacked_rgb(image, matrix)
    except Exception:
        return {**result, 'route':'RECTIFICATION_FAIL_CLOSED'}
    post = observe(recovered)
    selected = 'post' if post['score'] > pre['score'] else 'pre'
    return {'pre':pre, 'post':post, 'selected':selected,
            **(post if selected == 'post' else pre), 'route':'GEOMETRY_RECOVERED'}


def summarize_separation(rows, planned_units):
    """Descriptive tails and paired separation at observed quality; no threshold fitting."""
    summaries = []
    for condition in sorted({r['condition'] for r in rows}):
        selected = [r for r in rows if r['condition'] == condition]
        neg = {r['id']:r for r in selected if r['variant']=='clean' and not r['error']}
        negatives = [r['score'] for r in neg.values()]
        tail = {'planned':planned_units, 'observed':len(negatives),
            'max':max(negatives) if negatives else None,
            'empirical_q95':float(np.quantile(negatives,.95)) if negatives else None}
        for component in ('registered', 'max16wrong'):
            component_values=[r[component] for r in neg.values()]
            tail[component]={'max':max(component_values) if component_values else None,
                'empirical_q95':float(np.quantile(component_values,.95)) if component_values else None}
        originals = {r['id']:r for r in selected if r['variant']=='original' and not r['error']}
        for variant in sorted({r['variant'] for r in selected if r['variant']!='clean'}):
            positives = [r for r in selected if r['variant']==variant and not r['error']]
            pairs = [r for r in positives if r['id'] in neg]
            summaries.append({'condition':condition, 'variant':variant, 'negative_upper_tail':tail,
                'planned_positive':planned_units, 'observed_positive':len(positives),
                'paired_count':len(pairs),
                'paired_score_separations':[r['score']-neg[r['id']]['score'] for r in pairs],
                'positive_min_minus_negative_max':min(r['score'] for r in positives)-max(negatives) if positives and negatives else None,
                'paired_quality_and_evidence':[{'id':r['id'], 'quality':r['quality'],
                    'positive_registered':r['registered'], 'positive_max16wrong':r['max16wrong'],
                    'negative_registered':neg[r['id']]['registered'], 'negative_max16wrong':neg[r['id']]['max16wrong']} for r in pairs],
                'quality_matching_claim':False,
                'comparisons_to_original':[{'id':r['id'],
                    'score_delta':r['score']-originals[r['id']]['score'],
                    'quality_deltas':{k:r['quality'][k]-originals[r['id']]['quality'][k] for k in ('psnr','ssim','lpips')},
                    'quality_no_worse':all((r['quality'][k]>=originals[r['id']]['quality'][k]) if k!='lpips' else (r['quality'][k]<=originals[r['id']]['quality'][k]) for k in ('psnr','ssim','lpips'))}
                    for r in pairs if r['id'] in originals]})
    return summaries


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=('plan', 'fit', 'validation'), default='plan')
    p.add_argument('--roster', type=Path, required=True, help='JSON list of {id,prompt,seed}')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--allocator', type=Path)
    p.add_argument('--cost-penalty', type=float, default=1., help='fixed coefficient on incremental final LPIPS')
    args = p.parse_args(argv)
    units = json.loads(args.roster.read_text())
    allocator = SurvivalAllocator(**json.loads(args.allocator.read_text())) if args.allocator else None
    validate_roster(units, args.mode, allocator)
    if not np.isfinite(args.cost_penalty) or args.cost_penalty < 0:
        p.error('cost penalty must be finite and nonnegative')
    if args.mode == 'plan':
        print(json.dumps({'units': len(units), 'fit_images': len(units)*7,
            'validation_images': len(units)*4, 'conditions': ['clean','awgn_002','jpeg50'], 'fit_macro_continuations':len(units)*4, 'fit_score_paths':len(units)*21, 'validation_score_paths':len(units)*12, 'real_execution': False}))
        return 0
    if args.runtime_root is None:
        p.error('--runtime-root required for real execution')
    args.output.mkdir(parents=True, exist_ok=False)
    from experiments import run_paper_main_worker_v2 as v2
    from experiments.parallel_method_protocol_v1 import ACTIVE_B_ATTACKS as CORE_ATTACKS, render_attack
    from cegwm.runtime.survival_allocator_sd35 import generate_variant
    variants = ('clean', 'original', 'uniform', 'probe0', 'probe1', 'probe2', 'probe3') if args.mode == 'fit' else ('clean', 'original', 'uniform', 'survival')
    rows, fit_x, fit_y, failed_units = [], [], [], []
    report = {'stage': args.mode, 'unit_count': len(units), 'planned_images': len(units)*len(variants),
        'planned_score_paths': len(units)*len(variants)*len(CORE_ATTACKS),
        'formal_science_denominator': 0, 'claim': 'development only; no fixed-FPR or robustness conclusion',
        'allocator_scope':'HF spatial weights only; per-image LF/share/ISS fixed',
        'conditions':[a.name for a in CORE_ATTACKS],
        'planned_macro_continuations':len(units)*4 if args.mode=='fit' else 0,
        'label': 'relative to uniform HF: mean attacked max(pre,post) registered-minus-max16wrong increment minus penalty times final LPIPS increment',
        'cost_penalty': args.cost_penalty, 'rows': rows}
    report['environment'] = {'python': platform.python_version(), 'packages': {}}
    for package in ('torch', 'diffusers', 'transformers', 'numpy', 'Pillow'):
        try: report['environment']['packages'][package] = version(package)
        except PackageNotFoundError: report['environment']['packages'][package] = None
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    report['environment']['git_revision'] = revision.stdout.strip() if revision.returncode == 0 else None
    try:
        runtime = v2._build_runtime(args.runtime_root)
    except Exception as error:
        report['setup_error'] = repr(error)
        for unit in units:
            for variant in variants:
                for attack in CORE_ATTACKS:
                    rows.append({'id':unit['id'],'variant':variant,'condition':attack.name,'error':'setup: '+repr(error)})
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        return 2
    for unit in units:
        values, qualities, features = {}, {}, None
        clean = None
        reference_cache = {}
        details = {}
        for variant in variants:
            try:
                if variant == 'clean':
                    clean = v2._plain(runtime, unit['prompt'], unit['seed'])
                    image = clean
                    budget = None
                else:
                    if clean is None:
                        raise RuntimeError('plain generation failed')
                    image, x, budget = generate_variant(runtime, unit['prompt'], unit['seed'], clean, variant, allocator, reference_cache)
                    if variant == 'uniform':
                        features = x
                # Save generated outputs with ordinal names; arbitrary roster ids never form paths.
                index = len({r['id'] for r in rows if r['id'] != unit['id']})
                image.save(args.output / f'unit-{index:04d}-{variant}.png')
                quality = v2._quality(clean, image) if variant != 'clean' else {'psnr':None,'ssim':1.,'lpips':0.}
                quality['macro_mse'] = local_cost(clean, image)
                qualities[variant] = quality
            except Exception as error:
                for attack in CORE_ATTACKS:
                    row = {'id':unit['id'],'variant':variant,'condition':attack.name,'error':repr(error)}
                    rows.append(row)
                    with (args.output/'rows.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                continue
            values[variant] = []
            for attack_index, attack in enumerate(CORE_ATTACKS):
                row = {'id':unit['id'], 'variant':variant, 'condition':attack.name, 'quality':quality, 'budget':budget, 'error':None}
                try:
                    attacked, _ = render_attack(image, attack, noise_seed=unit['seed'] + 100000 + attack_index)
                    observed = score_details(runtime, attacked)
                    row.update(observed)
                    values[variant].append(observed['score'])
                    details[(variant, attack_index)] = observed
                except Exception as error:
                    row['error'] = repr(error)
                rows.append(row)
                with (args.output/'rows.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if args.mode == 'fit':
            needed = ('clean', 'uniform', 'probe0', 'probe1', 'probe2', 'probe3')
            if features is not None and all(len(values.get(v, [])) == len(CORE_ATTACKS) for v in needed):
                base = np.asarray(values['uniform'])
                for block in range(4):
                    variant = f'probe{block}'
                    increment = (np.asarray(values[variant]) - base).tolist()
                    cost = qualities[variant]['lpips'] - qualities['uniform']['lpips']
                    utility = float(np.mean(increment) - args.cost_penalty * cost)
                    fit_x.append(features[block].tolist()); fit_y.append(utility)
                    with (args.output/'labels.jsonl').open('a') as f:
                        f.write(json.dumps({'id':unit['id'], 'block':block, 'features':features[block].tolist(),
                            'score_increments':increment, 'lpips_increment':cost, 'utility':utility,
                            'reference_variant':'uniform',
                            'attacked_probe':[details[(variant,i)] for i in range(len(CORE_ATTACKS))],
                            'attacked_reference':[details[('uniform',i)] for i in range(len(CORE_ATTACKS))],
                            'attacked_negative':[details[('clean',i)] for i in range(len(CORE_ATTACKS))]})+'\n')
            else:
                failed_units.append(unit['id'])
    # Retain all failures. A complete roster is required only to fit a coherent model;
    # report output is unconditional, and adverse scores do not stop fitting.
    if args.mode == 'fit' and not failed_units:
        identities = [u['id'] for u in units] + ['seed:'+str(u['seed']) for u in units]
        fitted = fit_allocator(fit_x, fit_y, identities, cost_penalty=args.cost_penalty)
        (args.output/'allocator.json').write_text(json.dumps(fitted.to_dict(), indent=2))
    report['separation'] = summarize_separation(rows,len(units))
    report['failed_fit_units'] = failed_units
    report['row_errors'] = sum(r['error'] is not None for r in rows)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    return 2 if report['row_errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

