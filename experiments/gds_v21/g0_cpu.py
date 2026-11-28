"""Explicit local G0 check and artifact generation; not a real-model pass."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from geometry import build_coefficients, support, distance, groups, addresses, R


def manifest():
    root=Path(__file__).resolve().parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
            if p.suffix in ('.py','.json','.txt')}


def run(output):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    root=Path(__file__).resolve().parent
    result=subprocess.run([sys.executable,'-m','pytest','-o','addopts=',str(root/'test_g0.py'),'-q'],
                          cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout,flush=True)
    (out/'pytest.log').write_text(result.stdout)
    report=dict(stage='G0_CPU',status='failed',real_model_checked=False,files=manifest(),
                scope='engineering checks only; full G0 still requires same-runtime CUDA checkpoint check')
    if result.returncode == 0:
        coeff,pairs,excluded=build_coefficients()
        report.update(status='passed',hypothesis_count=len(pairs),excluded_count=len(excluded),
            coefficient_max_abs=float(abs(coeff).max()),coefficient_bytes=int(coeff.nbytes),
            hypotheses=[dict(r=r,h=h,v_r=support(r),v_h=support(h),distance=distance(r,h),
                coefficient_max_abs=float(abs(coeff[i]).max())) for i,(r,h) in enumerate(pairs)],
            excluded=excluded,reference_supports=[dict(r=r,support=support(r)) for r in R],
            group_counts=np.bincount(groups()).tolist(),address_counts=np.bincount(addresses()).tolist())
    report['wall_seconds']=time.monotonic()-start
    (out/'report.json').write_text(json.dumps(report,indent=2))
    if result.returncode:
        raise RuntimeError(f'G0 CPU failed; inspect {out}/pytest.log')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.output)
    print(json.dumps({k:result[k] for k in ('stage','status','hypothesis_count','coefficient_max_abs','real_model_checked')},indent=2))
