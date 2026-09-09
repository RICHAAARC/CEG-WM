"""Fit-only two-objective public anchor selection from completed real reports."""
import argparse
import json
from pathlib import Path
import math


def select_candidates(reports,max_mse):
    if not math.isfinite(max_mse) or max_mse <= 0: raise ValueError('positive quality budget required')
    groups={}
    for report in reports:
        if report['split']!='fit': raise ValueError('selection accepts fit only')
        if report.get('errors') or report['missing_paths'] or report['failed_paths']:
            raise ValueError('retain and resolve failed fit records before candidate comparison')
        spec=report['anchor_spec']; identity=(spec['rms'],spec['seed'])
        groups.setdefault(identity,[]).append(report)
    if len(groups)<2: raise ValueError('at least two candidate anchors required')
    summaries=[]; baseline=None
    for (rms,seed),items in groups.items():
        roster={(r['pair_id'],r['prompt']) for r in items}
        if len(roster)!=len(items): raise ValueError('duplicate candidate pair')
        if baseline is None: baseline=roster
        if roster!=baseline: raise ValueError('candidate pair rosters differ')
        quality=[sum(r['local_distortion_mse_2x2'])/4 for r in items]
        rows=[row for r in items for row in r['rows'] if row['role']=='positive']
        if len(rows)!=9*len(items): raise ValueError('incomplete conditions')
        for r in items:
            if len({row['condition'] for row in r['rows'] if row['role']=='positive'})!=9:
                raise ValueError('duplicate conditions')
        summary=dict(rms=rms,seed=seed,mean_mse=sum(quality)/len(quality),
            worst_pair_mse=max(quality),geometry=sum(r['corner_rmse_pixels'] for r in rows)/len(rows),
            content_tolerance=sum(r['oracle_minus_post_loss'] for r in rows)/len(rows))
        if not all(math.isfinite(v) for v in summary.values()): raise ValueError('nonfinite candidate metric')
        summaries.append(summary)
    eligible=[r for r in summaries if r['worst_pair_mse']<=max_mse]
    if not eligible: raise ValueError('no candidate meets the common image MSE budget')
    selected={objective:{k:best[k] for k in ('rms','seed')} for objective in ('geometry','content_tolerance')
              for best in [min(eligible,key=lambda r:(r[objective],r['mean_mse'],r['rms'],r['seed']))]}
    return dict(schema='latent_sync_fit_selection_v1',quality_budget_mse=max_mse,
                fit_pairs=sorted([list(x) for x in baseline]),candidates=summaries,selected=selected,
                content_objective='mean positive-part oracle-score minus predicted-post-score',
                claim='development selection; no fixed-FPR or real robustness claim')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,nargs='+',required=True)
    p.add_argument('--max-mse',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    result=select_candidates([json.loads(path.read_text()) for path in a.reports],a.max_mse)
    with a.output.open('x') as stream: json.dump(result,stream,indent=2)
    return 0


if __name__=='__main__': raise SystemExit(main())
