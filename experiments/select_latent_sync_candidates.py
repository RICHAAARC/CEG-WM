"""Compatibility filename: descriptive single-candidate comparison, no selection.

Own oracle-minus-post gaps can reward unreadable anchors and are not an objective.
No fitted winner or public asset is emitted. Failures remain in report counts.
"""
import argparse
import json
from pathlib import Path


def compare_report(report):
    rows=report['rows']; output=[]
    for condition in sorted({r['condition'] for r in rows}):
        values={}
        for method in ('content_only','latent_anchor','v2_rgb_sync'):
            subset=[r for r in rows if r['condition']==condition and r['method']==method]
            if len(subset)!=2 or any(r.get('error') for r in subset):
                values[method]=dict(status='incomplete',failed_or_missing=2-sum(not r.get('error') for r in subset))
                continue
            keyed={r['role']:r for r in subset}
            if set(keyed)!={'positive','negative'}: raise ValueError('duplicate roles')
            pos=float(keyed['positive']['post']); neg=float(keyed['negative']['post'])
            whole_pos=float(keyed['positive']['score']); whole_neg=float(keyed['negative']['score'])
            values[method]=dict(status='complete',positive_post=pos,negative_post=neg,post_separation=pos-neg,
                                positive_whole_path=whole_pos,negative_whole_path=whole_neg,
                                whole_path_separation=whole_pos-whole_neg)
        baseline=values['content_only']
        for method in ('latent_anchor','v2_rgb_sync'):
            if baseline['status']=='complete' and values[method]['status']=='complete':
                values[method]['separation_minus_content_only']=values[method]['post_separation']-baseline['post_separation']
                values[method]['positive_post_minus_content_only']=values[method]['positive_post']-baseline['positive_post']
                values[method]['whole_path_separation_minus_content_only']=values[method]['whole_path_separation']-baseline['whole_path_separation']
        output.append(dict(condition=condition,methods=values))
    return dict(rows=output,selection_performed=False,
                claim='paired descriptive post scores; one pair is not detection rate or FPR')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    with args.output.open('x') as stream:
        json.dump(compare_report(json.loads(args.report.read_text())),stream,indent=2)
    return 0


if __name__=='__main__': raise SystemExit(main())
