"""One selected worker per process; small real preflight is the default."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from cegwm.formal_experiment_v2 import EXPERIMENT_ID, FORMAL_CONDITIONS, apply_attack

BASELINES = ('t2smark', 'tree_ring', 'gaussian_shading', 'shallow_diffuse')
JOBS = dict(main='paper-main-v2', reconstruction='paper-main-reconstruction-v2',
    t2smark='paper-baseline-t2smark-v2', tree_ring='paper-baseline-treering-v2',
    gaussian_shading='paper-baseline-gaussian-shading-v2', shallow_diffuse='paper-baseline-shallow-diffuse-v2')


def real_preflight(worker, root, runtime_root):
    """One newly generated pair, all six conditions; no scientific pass threshold."""
    import torch
    from experiments import run_paper_main_worker_v2 as main
    from experiments import run_paper_baseline_worker_v2 as baseline
    output = root / 'preflight' / worker
    output.mkdir(parents=True, exist_ok=True)
    final = output / 'preflight.json'
    if final.exists():
        previous = json.loads(final.read_text())
        if previous['status'] == 'PREFLIGHT_EXECUTED':
            print(json.dumps(previous, indent=2))
            return 0
        attempt = 1
        while (output/f'failed-attempt-{attempt}').exists(): attempt += 1
        archived = output/f'failed-attempt-{attempt}'
        archived.mkdir()
        for name in ('preflight.json', 'rows.jsonl', 'clean.png', 'watermarked.png'):
            path = output/name
            if path.exists(): path.rename(archived/name)
    started = time.perf_counter()
    report = dict(experiment_id=EXPERIMENT_ID, worker=worker, science_denominator=0,
        planned_observations=12, rows=[], torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    try:
        tick = time.perf_counter()
        runtime = (main._build_runtime(runtime_root) if worker == 'main' else
            baseline._build_runtime(worker, runtime_root, os.environ.get('HF_TOKEN', '')))
        report['initialization_seconds'] = time.perf_counter()-tick
        tick = time.perf_counter()
        clean, marked = (main._main_pair(runtime, main.PREFLIGHT_PROMPT, main.PREFLIGHT_SEED)
            if worker == 'main' else runtime.pair(main.PREFLIGHT_PROMPT, main.PREFLIGHT_SEED))
        report['pair_seconds'] = time.perf_counter()-tick
        clean.save(output/'clean.png'); marked.save(output/'watermarked.png')
        try: report['quality'] = main._quality(clean, marked)
        except Exception as error: report['quality_error'] = type(error).__name__+': '+str(error)
        for condition in FORMAL_CONDITIONS:
            for role, image in (('negative', clean), ('positive', marked)):
                tick = time.perf_counter()
                row = dict(condition=condition, role=role, error=None)
                try:
                    attacked = apply_attack(image, condition)
                    if worker == 'main':
                        # Calibration-style forced pre + single continuous H + post:
                        # this checks the new route even when production would early-return.
                        value, route = main._calibration_score(runtime, attacked)
                        row.update(max_pre_post=value, forced_route=route)
                        if condition == FORMAL_CONDITIONS[-1] and role == 'positive':
                            from cegwm.runtime.blind_scoring_v2 import score_branches_v2
                            original, _ = score_branches_v2(attacked, runtime['key'], runtime['assets'])
                            reused, mode = score_branches_v2(attacked, runtime['key'], runtime['assets'], reuse_observation=True)
                            row['reuse_mode'] = mode
                            row['max_abs_branch_difference'] = max(abs(reused[b][k]-v)
                                for b in original for k, v in original[b].items())
                    else:
                        row.update(baseline._score_payload(runtime, attacked))
                except Exception as error:
                    row['error'] = type(error).__name__+': '+str(error)
                row['seconds'] = time.perf_counter()-tick
                report['rows'].append(row)
                with (output/'rows.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
    except Exception as error:
        report['setup_error'] = type(error).__name__+': '+str(error)
    report['seconds'] = time.perf_counter()-started
    report['row_errors'] = sum(r['error'] is not None for r in report['rows'])
    report['missing_observations'] = 12-len(report['rows'])
    report['status'] = ('PREFLIGHT_EXECUTED' if not report['row_errors'] and not report['missing_observations']
                        else 'PREFLIGHT_INCOMPLETE')
    final.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PREFLIGHT_EXECUTED' else 3


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('main', *BASELINES, 'reconstruction', 'finalize'), default='main')
    parser.add_argument('--mode', choices=('preflight', 'formal'), default='preflight')
    parser.add_argument('--drive-root', required=True, help='Parent CEG-WM directory; the umbrella ID is appended')
    parser.add_argument('--runtime-root', required=True)
    args = parser.parse_args(argv)
    root = Path(args.drive_root)/EXPERIMENT_ID
    runtime_root = Path(args.runtime_root)/args.worker
    runtime_root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    exact = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if args.worker == 'finalize':
        from experiments.run_paper_results_finalize_v2 import run_finalize
        if args.mode == 'preflight':
            print(json.dumps(dict(experiment_id=EXPERIMENT_ID, jobs=JOBS, model_execution=False)))
            return 0
        code = run_finalize(drive_root=root, expected_exact=exact, baseline_exact=exact)
        package = root/'finalized'/'paper-formal-v2'/'unified_result_package.json'
        if package.exists():
            from cegwm.paper_tables_v2 import export_tables_figures
            export_tables_figures(package)
        return code
    if args.mode == 'preflight' and args.worker != 'reconstruction':
        return real_preflight(args.worker, root, runtime_root)
    if args.worker == 'main':
        from experiments.run_paper_main_worker_v2 import run_worker
        return run_worker(job_id=JOBS['main'], expected_exact=exact, drive_root=root/'main', runtime_root=runtime_root)
    if args.worker == 'reconstruction':
        from experiments.run_paper_reconstruction_worker_v2 import run_worker, run_engineering_canary
        if args.mode == 'preflight':
            return run_engineering_canary(job_id='reconstruction', expected_exact=exact,
                drive_root=root/'preflight', runtime_root=runtime_root)
        return run_worker(job_id=JOBS['reconstruction'], main_job_id=JOBS['main'],
            expected_exact=exact, drive_root=root, runtime_root=runtime_root)
    from experiments.run_paper_baseline_worker_v2 import run_worker
    return run_worker(method=args.worker, job_id=JOBS[args.worker], expected_exact=exact,
                      drive_root=root/'baselines', runtime_root=runtime_root)


if __name__ == '__main__':
    raise SystemExit(main())
