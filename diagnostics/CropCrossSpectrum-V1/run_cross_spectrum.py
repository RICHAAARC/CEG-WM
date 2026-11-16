"""One fixed cross-channel reader; a failed zero-shift gate forbids shift probes."""
from __future__ import annotations
import argparse
import csv
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import uuid

import run_residual as support
import run_mechanism as geometry
from run_residual import (Reader, EXACT, IDS, ROLES, ATOL, RTOL, read, write, sha,
                         bind, error_data, inspect_source, load_saved_image, replay_comparison)

SUPPORT_SHA='27945052b071b4a207d674c60cfd79f8d326448a09f5bdb3f54524cc31c1c701'
GEOMETRY_SHA='9bc94ed7a467f2030bedb33bb915eba3969aceaa43db3e5b03458fabceb64f52'
SHIFTS=((0,0),(-2,0),(2,0),(0,-2),(0,2),(8,0),(0,8))
SMALL=SHIFTS[1:5]
LABELS=('registered',*(f'wrong_{j:02d}' for j in range(16)))
RULES={'zero_top1_min':7,'zero_pair_min':7,'zero_margin_retention':.8,
       'drift_ratio_max':.7,'improved_sources_min':6,'shift_top1_min':6,
       'shift_pair_min':6,'shift_margin_retention':.5,'negative_top1_stop':3,
       'phase_score_atol':1e-6}
TERMINAL={'STOPPED_ZERO_GATE','STOPPED_UNINFORMATIVE_REFERENCE',
          'COMPLETE_CANDIDATE_PASSED','COMPLETE_CANDIDATE_REJECTED'}


def cross_features(z, mask):
    import torch
    z=z.detach().to(device='cpu',dtype=torch.float64)
    if tuple(z.shape)!=(1,16,64,64) or not torch.isfinite(z).all():
        raise ValueError('Expected finite 1x16x64x64 observation')
    spectrum=torch.fft.rfft2(z,norm='ortho')[0,:,mask]
    energy=spectrum.abs().square().sum(dim=0)
    c,d=torch.triu_indices(16,16,offset=1)
    # Zero-energy frequencies contribute zero; no data-dependent mask or tuning.
    denom=torch.where(energy>0,energy,torch.ones_like(energy))
    return (spectrum[c]*spectrum[d].conj()/denom).reshape(-1)


def keyed(values):
    values=[float(v) for v in values]
    if len(values)!=17 or not all(math.isfinite(v) for v in values):
        raise ValueError('Seventeen finite scores required')
    wrong=max(values[1:])
    return {'scores':dict(zip(LABELS,values)), 'registered':values[0],
            'wrong_max':wrong,'gap':values[0]-wrong,'top1':values[0]>wrong}


class FixedProbe:
    def __init__(self,reader):
        import torch
        from cegwm.method.hf import _spec
        from cegwm.method.frequency import reconstruct_frequency_carrier
        self.linear=geometry.HFProbe(reader,(1,16,64,64),'cpu')
        self.mask=self.linear.masks['common']
        templates=[]
        for key in reader.keys:
            carrier=reconstruct_frequency_carrier(key,(1,16,64,64),_spec(reader.hf),
                                                  dtype=torch.float32,device='cpu')
            vector=cross_features(carrier,self.mask)
            norm=vector.norm()
            if not torch.isfinite(norm) or float(norm)==0: raise ValueError('Undefined key template')
            templates.append(vector/norm)
        self.templates=torch.stack(templates)
        # Coincident relation templates would invalidate this fixed 17-key test.
        for i in range(17):
            for j in range(i):
                if float((self.templates[i]-self.templates[j]).norm())<1e-10:
                    raise ValueError('Relation-template collision in fixed keys')

    def score(self,z):
        import torch
        z=z.detach().cpu()
        linear=self.linear.score(z,'common')
        feature=cross_features(z,self.mask)
        norm=feature.norm()
        if not torch.isfinite(norm) or float(norm)==0:
            raise ValueError('Undefined cross-spectrum score; cannot count a flat reader as stable')
        values=(self.templates.conj()@(feature/norm)).real.tolist()
        return {'B':keyed(linear['scores'].values()),'C':keyed(values)}


def robust_phase_shift(z,dx,dy):
    """Double-precision pure phase control of Qz, no encoded-black subtraction."""
    import torch
    f=torch.fft.fft2(z.detach().cpu().double(),norm='ortho')
    f[...,32,:]=0
    f[...,:,32]=0
    freq=torch.fft.fftfreq(64,dtype=torch.float64)
    phase=torch.exp(-2j*math.pi*(freq[None,:]*dx/8+freq[:,None]*dy/8))
    moved=torch.fft.ifft2(f*phase,norm='ortho')
    if float(moved.imag.norm())>1e-10*max(float(moved.real.norm()),1e-12):
        raise RuntimeError('Non-real pure-phase control')
    return moved.real


def scale_from_negatives(zero,method):
    # Population within-image dispersion of the same 16 wrong keys, pooled over 8 negatives.
    variances=[]
    for uid in IDS:
        scores=zero[uid,'negative']['readers'][method]['scores']
        wrong=[scores[label] for label in LABELS[1:]]
        mean=statistics.mean(wrong)
        variances.extend((v-mean)**2 for v in wrong)
    value=math.sqrt(statistics.mean(variances))
    if not math.isfinite(value) or value<=0: raise ValueError('Zero/nonfinite baseline wrong-key scale')
    return value


def assess_zero(zero):
    scales={m:scale_from_negatives(zero,m) for m in ('B','C')}
    medians={m:statistics.median(zero[u,'positive']['readers'][m]['gap']/scales[m] for u in IDS)
             for m in ('B','C')}
    top1=sum(zero[u,'positive']['readers']['C']['top1'] for u in IDS)
    negative_top1=sum(zero[u,'negative']['readers']['C']['top1'] for u in IDS)
    pair=sum(zero[u,'positive']['readers']['C']['registered']>
             zero[u,'negative']['readers']['C']['registered'] for u in IDS)
    checks={'positive_top1_7_of_8':top1>=RULES['zero_top1_min'],
            'paired_positive_gt_negative_7_of_8':pair>=RULES['zero_pair_min'],
            'positive_median_gap':medians['C']>0,
            'baseline_margin_retention_80pct':medians['C']>=RULES['zero_margin_retention']*medians['B'],
            'negative_top1_below_3_of_8':negative_top1<RULES['negative_top1_stop']}
    reference_ok=medians['B']>0
    return {'status':'PASSED' if reference_ok and all(checks.values()) else
            ('FAILED' if reference_ok else 'REFERENCE_UNINFORMATIVE'),
            'checks':checks,'scales':scales,'median_standardized_gap':medians,
            'positive_top1':top1,'paired_positive_gt_negative':pair,'negative_top1':negative_top1,
            'denominator':8,'scale_definition':'sqrt(mean over 8 negative images x 16 wrong keys of within-image squared score deviations)'}


def assess_shifts(records,zero_gate):
    scales=zero_gate['scales']
    def score(uid,role,shift,method): return records[uid,role,*shift]['readers'][method]
    drift={m:[max(abs(score(u,'positive',s,m)['gap']-score(u,'positive',(0,0),m)['gap'])/scales[m]
                    for s in SMALL) for u in IDS] for m in ('B','C')}
    median={m:statistics.median(drift[m]) for m in ('B','C')}
    improved=sum(c<b for b,c in zip(drift['B'],drift['C']))
    checks={'reference_drift_positive':median['B']>0,
            'median_drift_reduced_30pct':median['C']<=RULES['drift_ratio_max']*median['B'],
            'improved_6_of_8':improved>=RULES['improved_sources_min']}
    conditions={}
    for shift in SHIFTS:
        pos=[score(u,'positive',shift,'C') for u in IDS]
        neg=[score(u,'negative',shift,'C') for u in IDS]
        top1=sum(p['top1'] for p in pos)
        negative_top1=sum(n['top1'] for n in neg)
        pair=sum(p['registered']>n['registered'] for p,n in zip(pos,neg))
        gap=statistics.median(p['gap']/scales['C'] for p in pos)
        sid=geometry.shift_id(*shift)
        conditions[sid]={'positive_top1':top1,'negative_top1':negative_top1,
                         'paired_positive_gt_negative':pair,'median_standardized_gap':gap,'denominator':8}
        checks[sid+'/negative_top1_below_3']=negative_top1<RULES['negative_top1_stop']
        if shift in SMALL:
            checks[sid+'/top1_6_of_8']=top1>=RULES['shift_top1_min']
            checks[sid+'/pair_6_of_8']=pair>=RULES['shift_pair_min']
            checks[sid+'/margin_retention_50pct']=gap>=RULES['shift_margin_retention']*zero_gate['median_standardized_gap']['C']
    return {'status':'PASSED' if all(checks.values()) else 'FAILED','checks':checks,
            'worst_standardized_gap_change_by_source':dict(zip(IDS,zip(drift['B'],drift['C']))),
            'median_worst_change':median,'improved_sources':improved,'conditions':conditions}


def record_path(output,uid,role,shift,phase=False):
    return output/('phase_records' if phase else 'records')/uid/role/(geometry.shift_id(*shift)+'.json')


def collect(output,phase=False):
    result={}
    for uid in IDS:
        for role in ROLES:
            for shift in SHIFTS[1:] if phase else SHIFTS:
                path=record_path(output,uid,role,shift,phase)
                if path.exists(): result[uid,role,*shift]=read(path)
    return result


def summarize(output,status='INCOMPLETE'):
    actual,phase=collect(output),collect(output,True)
    zero_gate=read(output/'zero_gate.json') if (output/'zero_gate.json').exists() else None
    skipped=status in {'STOPPED_ZERO_GATE','STOPPED_UNINFORMATIVE_REFERENCE'}
    rows=[]
    for uid in IDS:
        for role in ROLES:
            for shift in SHIFTS:
                r=actual.get((uid,role,*shift),{'status':'NOT_RUN_ZERO_GATE' if skipped and shift!=(0,0) else 'PENDING'})
                rows.append({**r,'unit_id':uid,'role':role,'dx_rgb':shift[0],'dy_rgb':shift[1]})
    counts={}
    for role in ROLES:
        for shift in SHIFTS:
            selected=[r for r in rows if r['role']==role and (r['dx_rgb'],r['dy_rgb'])==shift]
            counts[role+'/'+geometry.shift_id(*shift)]={'planned':8,**{s:sum(r['status']==s for r in selected)
                for s in ('SCORED','PENDING','NOT_RUN_ZERO_GATE','OPERATIONAL_ERROR')}}
    summary={'status':status,'planned_actual_records':112,'planned_phase_records':96,
             'phase_scored':sum(r['status']=='SCORED' for r in phase.values()),
             'phase_not_run_zero_gate':96 if skipped else 0,'counts':counts,'zero_gate':zero_gate,
             'shift_gate':read(output/'shift_gate.json') if (output/'shift_gate.json').exists() else None,
             'records':rows,'claim_ceiling':'development_only_fixed_reader_no_FPR_no_blind_geometry_claim'}
    write(output/'summary.json',summary)
    flat=[]
    for r in rows:
        for method in ('B','C'):
            row={k:r[k] for k in ('unit_id','role','dx_rgb','dy_rgb','status')}
            row['reader']=method
            if r['status']=='SCORED':
                s=r['readers'][method]
                row.update({k:s[k] for k in ('registered','wrong_max','gap','top1')})
                row.update(s['scores'])
                if zero_gate: row['standardized_gap']=s['gap']/zero_gate['scales'][method]
            flat.append(row)
    cols=list(dict.fromkeys(k for r in flat for k in r))
    tmp=output/'readout.csv.tmp'
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(flat)
    os.replace(tmp,output/'readout.csv')
    return summary


def save_attempt(output,uid,role,shift,compute,phase=False):
    path=record_path(output,uid,role,shift,phase)
    if path.exists() and read(path)['status']=='SCORED': return read(path)
    attempt=output/'attempts'/('phase' if phase else 'actual')/uid/role/geometry.shift_id(*shift)/(uuid.uuid4().hex+'.json')
    write(attempt,{'status':'STARTED'})
    try:
        result={'status':'SCORED',**compute()}
        write(attempt,result);write(path,result)
        return result
    except Exception as error:
        result={'status':'OPERATIONAL_ERROR',**error_data(error)}
        write(attempt,result);write(path,result);summarize(output)
        raise


def run(source,output,repo):
    import numpy as np
    import torch
    from cegwm.shared.keys import normalize_detection_key,public_key_digest
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError('SOURCE and OUTPUT must not overlap')
    if sha(support.__file__)!=SUPPORT_SHA or sha(geometry.__file__)!=GEOMETRY_SHA:
        raise ValueError('Validated support identity differs')
    protocol,old=inspect_source(source)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=EXACT:
        raise ValueError('Historical source revision differs')
    if subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--'],text=True):
        raise ValueError('Historical tracked source modified')
    if public_key_digest(normalize_detection_key(os.environ['CEG_WM_ROOT_KEY']))!=protocol['key_digest']:
        raise ValueError('Key differs from source run')
    identity={'schema':'crop_cross_spectrum_v1','script_sha':sha(__file__),'support_sha':SUPPORT_SHA,
              'geometry_sha':GEOMETRY_SHA,'source':str(source.resolve()),'source_exact':EXACT,
              'source_hashes':{n:sha(source/n) for n in ('summary.json','protocol.json','environment.json','model_revisions.json')},
              'shifts_rgb':[list(s) for s in SHIFTS],'rules':RULES,
              'candidate':'all_120_complex_offdiagonal_channel_pairs_per_frequency_energy_normalized_cosine',
              'mask':'original_HF_common_no_Nyquist','baseline':'original_common_mask_real_centered_cosine',
              'scale':'zero_negative_within_image_16_wrong_key_population_RMS',
              'no_black_subtraction':True,'no_search':True}
    bind(output/'protocol.json',identity)
    # Scientific stop/completion is persistent: Run all cannot quietly reopen it.
    if (output/'summary.json').exists() and read(output/'summary.json')['status'] in TERMINAL:
        print('Existing terminal result: '+read(output/'summary.json')['status'],flush=True)
        return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA VAE reader required; no generation fallback')
    bind(output/'environment.json',{'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),
        'cuda':torch.version.cuda,'packages':{p:version(p) for p in ('torch','torchvision','diffusers',
            'transformers','numpy','Pillow','huggingface_hub','safetensors')}})
    summarize(output)
    reader=Reader(repo,source)
    probe=FixedProbe(reader)
    references={}
    for uid in IDS:
        for role in ROLES:
            image=load_saved_image(source,old[uid,role,'oracle_h']['image'])
            z=reader.observe(image)
            if tuple(z.shape)!=(1,16,64,64): raise ValueError('Unexpected VAE shape')
            production=reader.score(z)
            check=replay_comparison(production,old[uid,role,'oracle_h'])
            full=probe.linear.score(z.detach().cpu(),'full')
            hf_ok=np.allclose(list(full['scores'].values()),list(production['branches']['hf'].values()),atol=ATOL,rtol=RTOL)
            if not check['passed'] or not hf_ok:
                write(output/'replay_gate.json',{'status':'FAILED','unit_id':uid,'role':role,'check':check,'full_HF_matches':bool(hf_ok)})
                raise RuntimeError('Historical score replay failed; no shifts allowed')
            scores=probe.score(z)
            path=record_path(output,uid,role,(0,0))
            if path.exists() and read(path)['status']=='SCORED':
                for method in ('B','C'):
                    if not np.allclose(list(scores[method]['scores'].values()),list(read(path)['readers'][method]['scores'].values()),atol=ATOL,rtol=RTOL):
                        raise RuntimeError('Saved zero-shift candidate score changed on resume')
            else:
                write(path,{'status':'SCORED','readers':scores,'historical_replay':check,'production_reference':production})
            references[uid,role]=(image,z.detach().cpu())
            print(f'zero {uid[-4:]} {role}: B gap={scores["B"]["gap"]:.6f}, C gap={scores["C"]["gap"]:.6f}',flush=True)
    write(output/'replay_gate.json',{'status':'PASSED','planned':16,'atol':ATOL,'rtol':RTOL})
    zero={(u,r):read(record_path(output,u,r,(0,0))) for u in IDS for r in ROLES}
    gate=assess_zero(zero)
    bind(output/'zero_gate.json',gate)
    print('ZERO GATE: '+json.dumps(gate,ensure_ascii=False),flush=True)
    if gate['status']!='PASSED':
        status='STOPPED_ZERO_GATE' if gate['status']=='FAILED' else 'STOPPED_UNINFORMATIVE_REFERENCE'
        summarize(output,status)
        print(status+': no phase or RGB shift probes started.',flush=True)
        return
    # Only after the scientific zero-shift gate: pure phase controls for all 17 keys.
    for uid in IDS:
        for role in ROLES:
            _,z=references[uid,role]
            for shift in SHIFTS[1:]:
                def phase_compute():
                    scores=probe.score(robust_phase_shift(z,*shift))
                    diff=max(abs(scores['C']['scores'][k]-zero[uid,role]['readers']['C']['scores'][k]) for k in LABELS)
                    if diff>RULES['phase_score_atol']:
                        raise RuntimeError(f'Pure phase invariance failed: {diff}')
                    return {'readers':scores,'max_abs_C_score_change':diff}
                save_attempt(output,uid,role,shift,phase_compute,True)
    phase=collect(output,True)
    write(output/'phase_gate.json',{'status':'PASSED','planned':96,
          'max_abs_C_score_change':max(r['max_abs_C_score_change'] for r in phase.values())})
    for shift in SHIFTS[1:]:
        for uid in IDS:
            for role in ROLES:
                image,_=references[uid,role]
                def actual_compute():
                    zero_rgb=geometry.rgb_shift(image,*shift,False)
                    circular=geometry.rgb_shift(image,*shift,True)
                    if not np.array_equal(np.asarray(zero_rgb),np.asarray(circular)):
                        raise RuntimeError('RGB boundary controls differ; stop, do not select favorable boundary')
                    return {'readers':probe.score(reader.observe(zero_rgb)),
                            'rgb_boundary_identical':True,'rgb_interpolation':'none_integer_copy'}
                save_attempt(output,uid,role,shift,actual_compute)
        summarize(output)
        print('Completed actual shift '+geometry.shift_id(*shift),flush=True)
    final_gate=assess_shifts(collect(output),gate)
    write(output/'shift_gate.json',final_gate)
    status='COMPLETE_CANDIDATE_PASSED' if final_gate['status']=='PASSED' else 'COMPLETE_CANDIDATE_REJECTED'
    summarize(output,status)
    print(status+'\n'+json.dumps(final_gate,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','output','repo'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()) or args.source.resolve().is_relative_to(args.output.resolve()):
        raise ValueError('SOURCE and OUTPUT must not overlap')
    sys.path[:0]=[str(args.repo),str(args.repo/'src')]
    try: run(args.source,args.output,args.repo)
    except Exception as error:
        write(args.output/'last_failure.json',error_data(error))
        summarize(args.output,'OPERATIONAL_ERROR')
        raise


if __name__=='__main__': main()
