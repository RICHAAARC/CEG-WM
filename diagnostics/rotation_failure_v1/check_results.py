"""Read-only fixed-roster diagnostic coverage audit, including unfinished runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
from diagnostic import CONDITIONS, STATUS, TAU


def inspect(root):
    root=Path(root)
    entries=json.loads((Path(__file__).parent/'input_reference.json').read_text())['entries']
    groups={}
    mismatches=[]
    for entry in entries:
        for condition in CONDITIONS:
            for arm in ('clean','watermarked'):
                name=entry['sample_id']+'__'+condition+'__'+arm+'.json'
                key=condition+'/'+arm+'/'+entry['selection_stratum']
                group=groups.setdefault(key,Counter(planned=0,present=0,missing=0,invalid=0,replay_mismatch=0))
                group['planned']+=1
                path=root/name
                if not path.exists():
                    group['missing']+=1
                    continue
                group['present']+=1
                try:
                    row=json.loads(path.read_text())
                    expected={'sample_id':entry['sample_id'],'selection_stratum':entry['selection_stratum'],
                              'condition':condition,'arm':arm,'truth_role':'negative' if arm=='clean' else 'positive',
                              'status':STATUS,'science_denominator':0,'threshold':TAU,'execution_kind':'FROZEN_REAL_RUNTIME'}
                    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('row identity differs')
                    if row.get('unit_status') not in ('COMPLETE','PARTIAL_DIAGNOSTIC','FAILED'):raise ValueError('unknown unit status')
                    group[row['unit_status']]+=1
                    group['replay_mismatch']+=row.get('replay_matches_original') is False
                    for metric in ('pre_score','syncseal_post_score','oracle_post_score'):
                        if row.get(metric) is not None:group[metric+'_available']+=1
                except Exception as error:
                    group['invalid']+=1
                    mismatches.append({'file':name,'error':f'{type(error).__name__}: {error}'})
    return {'science_denominator':0,'planned_rows':400,'groups':groups,'invalid_rows':mismatches,
            'note':'Coverage only; failure rows retained; not a cause adjudication or method PASS.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(inspect(parser.parse_args().output),indent=2))
