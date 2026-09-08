"""Export the existing result package as paper tables and descriptive figures."""
import csv
import json
from pathlib import Path


def export_tables_figures(package_path):
    package_path = Path(package_path)
    package = json.loads(package_path.read_text())
    output = package_path.parent
    binary, quality = [], []
    def add(method, stage, condition, role, summary, variant=''):
        binary.append(dict(method=method, stage=stage, variant=variant,
            condition=condition, role=role, **summary))
    for method, result in package['methods'].items():
        add(method, 'clean_test', 'clean', 'negative', result['clean_negative_test'])
        for key, summary in result['evaluation'].items():
            condition, role = key.rsplit(':', 1)
            add(method, 'evaluation', condition, role, summary)
        for key, summary in result.get('ablations', {}).items():
            variant, condition, role = key.split(':')
            add(method, 'ablation', condition, role, summary, variant)
        for metric, summary in result['quality']['metrics'].items():
            quality.append(dict(method=method, metric=metric, **summary))
    reconstruction = package['reconstruction_supplement']
    for role, summary in reconstruction['summaries'].items():
        add(reconstruction['method_id'], 'reconstruction', 'sdxl_img2img', role, summary)
    for name, rows in (('all_binary_results.csv', binary), ('quality_results.csv', quality)):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    # Missing or failed statistics stay visible as gaps; never plotted as zero.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = package['method_order']
    labels = {m: ('CEG-WM' if m.startswith('cegwm_') else
        {'t2smark':'T2SMark','tree_ring':'Tree-Ring','gaussian_shading':'Gaussian Shading',
         'shallow_diffuse':'Shallow Diffuse'}.get(m,m)) for m in methods}
    conditions = list(dict.fromkeys(r['condition'] for r in binary if r['stage']=='evaluation'))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, role in zip(axes, ('positive', 'negative')):
        metric = 'scored_only_tpr' if role=='positive' else 'scored_only_fpr'
        for method in methods:
            rows = [r for r in binary if r['stage']=='evaluation' and r['method']==method and r['role']==role]
            by_condition = {r['condition']:r for r in rows}
            axis.plot(range(len(conditions)), [by_condition[c].get(metric) for c in conditions], marker='o', label=labels[method])
        axis.set_xticks(range(len(conditions)), ['clean', 'JPEG50', 'resize .5', 'crop .8', 'blur 1', 'rotation +10'])
        axis.set_ylabel('TPR (scored rows)' if role=='positive' else 'Attacked FPR (scored rows)')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('V2 descriptive results; failures and fixed denominators retained in tables')
    for suffix in ('png', 'pdf'): fig.savefig(output/f'main_conditions.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric in zip(axes, ('psnr', 'ssim', 'lpips')):
        rows = {r['method']:r for r in quality if r['metric']==metric}
        axis.bar(range(len(methods)), [rows[m]['mean'] if rows[m]['mean'] is not None else float('nan') for m in methods])
        axis.set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=35, ha='right', fontsize=9)
        axis.set_title(metric.upper())
    for suffix in ('png', 'pdf'): fig.savefig(output/f'quality.{suffix}', dpi=180)
    plt.close(fig)
    (output/'tables_and_figures.md').write_text(
        '# V2 result exports\n\nThe unified main CSV contains 60 evaluation rows. '
        'all_binary_results.csv additionally includes clean confirmation, all four main-method ablations '
        'and the 100-pair reconstruction supplement. quality_results.csv contains the five-method '
        'PSNR/SSIM/LPIPS summaries. Figures use scored-only rates; consult planned denominators, '
        'failure/missing counts, intervals and planned bounds in the tables. FPR is report-only.\n')
    return dict(binary_rows=len(binary), quality_rows=len(quality))
