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
            'validation_images': len(units)*4, 'conditions': 9, 'real_execution': False}))
        return 0
    if args.runtime_root is None:
        p.error('--runtime-root required for real execution')
    args.output.mkdir(parents=True, exist_ok=False)
    from experiments import run_paper_main_worker_v2 as v2
    from experiments.parallel_method_protocol_v1 import CORE_ATTACKS, render_attack
    from cegwm.runtime.survival_allocator_sd35 import generate_variant
    variants = ('clean', 'uniform', 'original', 'probe0', 'probe1', 'probe2', 'probe3') if args.mode == 'fit' else ('clean', 'uniform', 'original', 'survival')
    rows, fit_x, fit_y, failed_units = [], [], [], []
    report = {'stage': args.mode, 'unit_count': len(units), 'planned_images': len(units)*len(variants),
        'planned_score_paths': len(units)*len(variants)*len(CORE_ATTACKS),
        'formal_science_denominator': 0, 'claim': 'development only; no fixed-FPR or robustness conclusion',
        'label': 'mean attacked max(pre,post) registered-minus-max16wrong increment minus penalty times final LPIPS increment',
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
        for variant in variants:
            try:
                if variant == 'clean':
                    clean = v2._plain(runtime, unit['prompt'], unit['seed'])
                    image = clean
                else:
                    if clean is None:
                        raise RuntimeError('plain generation failed')
                    image, x = generate_variant(runtime, unit['prompt'], unit['seed'], clean, variant, allocator)
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
                row = {'id':unit['id'], 'variant':variant, 'condition':attack.name, 'quality':quality, 'error':None}
                try:
                    attacked, _ = render_attack(image, attack, noise_seed=unit['seed'] + 100000 + attack_index)
                    score, route = v2._calibration_score(runtime, attacked)
                    if not np.isfinite(score): raise ValueError('nonfinite content score')
                    row.update(score=score, route=route)
                    values[variant].append(score)
                except Exception as error:
                    row['error'] = repr(error)
                rows.append(row)
                with (args.output/'rows.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if args.mode == 'fit':
            needed = ('uniform', 'probe0', 'probe1', 'probe2', 'probe3')
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
                            'score_increments':increment, 'lpips_increment':cost, 'utility':utility})+'\n')
            else:
                failed_units.append(unit['id'])
    # Retain all failures. A complete roster is required only to fit a coherent model;
    # report output is unconditional, and adverse scores do not stop fitting.
    if args.mode == 'fit' and not failed_units:
        identities = [u['id'] for u in units] + ['seed:'+str(u['seed']) for u in units]
        fitted = fit_allocator(fit_x, fit_y, identities, cost_penalty=args.cost_penalty)
        (args.output/'allocator.json').write_text(json.dumps(fitted.to_dict(), indent=2))
    report['failed_fit_units'] = failed_units
    report['row_errors'] = sum(r['error'] is not None for r in rows)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    return 2 if report['row_errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

