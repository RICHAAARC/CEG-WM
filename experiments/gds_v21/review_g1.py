"""CPU-only read-only review of a completed Drive G1. Never imports model code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import K,MASK,N,groups,addresses,build_coefficients,rotation,support

ARMS=['N','B0','B1','B2','P','P-E']


def fraction(a,b):
    return float(a/b) if b else float('nan')


def retention(states,steps,group,address):
    """Recompute repeated-event counts AND a unique-position latest-success cohort."""
    def hits(z):
        valid=z!=MASK; result=np.zeros(N,bool)
        result[valid]=group[z[valid]]==address[valid]
        return result
    h10,h0=hits(states[10]),hits(states[0])
    latest={};events=[]
    for row in steps:
        t=row['timestep']; chosen=np.array(row.get('executed_selected',[]),dtype=int)
        if len(chosen):
            if len(set(chosen.tolist()))!=len(chosen) or chosen.min()<0 or chosen.max()>=N or not 11<=t<=60:
                raise ValueError('Invalid saved selected positions')
            successful=chosen[hits(states[t])[chosen]]
            for i in successful: latest[int(i)]=int(states[t][i])
            events.append(dict(timestep=t,selected=len(chosen),hits_when_sampled=len(successful),
                address_hits_at_t10=int(h10[successful].sum()),address_hits_at_t0=int(h0[successful].sum()),
                exact_token_retained_t10=int((states[t][successful]==states[10][successful]).sum()),
                exact_token_retained_t0=int((states[t][successful]==states[0][successful]).sum())))
    ids=np.array(sorted(latest),dtype=int)
    old=np.array([latest[i] for i in ids],dtype=int)
    total=sum(e['hits_when_sampled'] for e in events)
    summary=dict(selected_events=sum(e['selected'] for e in events),successful_events=total,
        event_hit_rate=fraction(total,sum(e['selected'] for e in events)),unique_success_positions=len(ids),
        event_group_retention_t10=fraction(sum(e['address_hits_at_t10'] for e in events),total),
        event_group_retention_t0=fraction(sum(e['address_hits_at_t0'] for e in events),total),
        unique_group_retention_t10=fraction(h10[ids].sum(),len(ids)),
        unique_group_retention_t0=fraction(h0[ids].sum(),len(ids)),
        unique_exact_token_retention_t10=fraction((states[10][ids]==old).sum(),len(ids)),
        unique_exact_token_retention_t0=fraction((states[0][ids]==old).sum(),len(ids)),
        total_address_hits_t11=int(hits(states[11]).sum()),total_address_hits_t10=int(h10.sum()),
        total_address_hits_t0=int(h0.sum()),retention_applicable=bool(len(ids)))
    return events,summary


def relative_geometry(r,h):
    translation=rotation(r[0]).T@(np.array(h[2:])-np.array(r[2:]))/r[1]
    theta=h[0]-r[0];scale=h[1]/r[1]
    changed=[name for name,yes in [('rotation',abs(theta)>1e-9),('scale',abs(scale-1)>1e-9),
                ('translation',np.linalg.norm(translation)>1e-9)] if yes]
    return dict(relative_type='+'.join(changed) or 'identity',relative_angle=theta,relative_scale=scale,
                relative_tx=float(translation[0]),relative_ty=float(translation[1]))


def classify_failure(at11,at10,at0):
    if at0>0: return 'positive_final'
    if at11>0: return 'positive_at_11_nonpositive_final'
    if at10>0: return 'nonpositive_at_11_recovered_at_10_lost_by_final'
    return 'already_nonpositive_at_11_and_10'


def save_csv(data,path):
    frame=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(path,index=False)
    return frame


def run(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('Analysis output must be separate from the original G1 directory')
    output.mkdir(parents=True,exist_ok=False)
    before={}
    def read_path(relative):
        path=(source/relative).resolve()
        if source not in path.parents or not path.is_file():
            raise FileNotFoundError(f'Missing or invalid G1 input: {relative}')
        key=str(path.relative_to(source))
        if key not in before: before[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def load(relative): return json.loads(read_path(relative).read_text())
    report=load('report.json')
    if report.get('stage')!='G1' or report.get('status')!='complete':
        raise ValueError('Expected a completed G1; do not silently drop failed units')
    attempts=report['attempts']
    if len(attempts)!=48 or report['generation_attempts_started']!=48 or any(r['status']!='complete' for r in attempts):
        raise ValueError('48/48 completed ledger required')
    units={(r['source_id'],r['arm']) for r in attempts}
    if len(units)!=48 or {a for _,a in units}!=set(ARMS) or len({s for s,_ in units})!=8:
        raise ValueError('Unexpected roster')
    for r in attempts:
        if r['unit']!=r['source_id']+'_'+r['arm'] or '/' in r['unit'] or '\\' in r['unit'] or r['generator_forwards']!=100:
            raise ValueError('Invalid unit or NFE count')
    saved_summary=pd.read_csv(read_path('summary.csv'))
    if len(saved_summary)!=48 or set(zip(saved_summary.source_id,saved_summary.arm))!=units:
        raise ValueError('Saved summary roster mismatch')
    if not (saved_summary.status=='complete').all() or not (saved_summary.generator_forwards==100).all():
        raise ValueError('Saved summary status/NFE mismatch')
    geometry_hash=hashlib.sha256(Path(__file__).with_name('geometry.py').read_bytes()).hexdigest()
    if geometry_hash!=report['files']['geometry.py']:
        raise ValueError('Geometry source differs from the generating run')
    coeff,pairs,_=build_coefficients()
    assets=load('geometry_assets.json')
    if assets['pairs']!=[[list(r),list(h)] for r,h in pairs] or assets['coefficient_sha256']!=hashlib.sha256(coeff.tobytes()).hexdigest():
        raise ValueError('Frozen geometry asset mismatch')
    lookup=np.concatenate([coeff,np.zeros((len(pairs),N,1))],axis=2)
    group=groups();address=addresses();class_group=np.r_[group,16]
    pair_info=[]
    for i,(r,h) in enumerate(pairs):
        pair_info.append(dict(pair_id=i,r_angle=r[0],r_scale=r[1],r_tx=r[2],r_ty=r[3],
            h_angle=h[0],h_scale=h[1],h_tx=h[2],h_ty=h[3],r_output_support=support(r),h_output_support=support(h),
            **relative_geometry(r,h)))
    save_csv(pair_info,output/'geometry_pair_definitions.csv')
    summaries=[];event_rows=[];pair_rows=[];trajectories=[];worst_rows=[];fidelity=[]
    for number,attempt in enumerate(attempts):
        unit=attempt['unit'];prefix=dict(source_id=attempt['source_id'],arm=attempt['arm'],unit=unit)
        metric=load(unit+'/metrics.json')
        steps=[json.loads(s) for s in read_path(unit+'/steps.jsonl').read_text().splitlines() if s.strip()]
        if [r['timestep'] for r in steps]!=list(range(99,-1,-1)):
            raise ValueError(f'{unit}: step ledger incomplete')
        with np.load(read_path(unit+'/private_tokens.npz'),allow_pickle=False) as f:
            times=f['timesteps'];tokens=f['tokens']
        if not np.array_equal(times,np.arange(99,-1,-1)) or tokens.shape!=(100,N) or tokens.dtype.kind not in 'iu' or np.any((tokens<0)|(tokens>MASK)):
            raise ValueError(f'{unit}: invalid token archive')
        states={int(t):z for t,z in zip(times,tokens)}
        scores={int(t):lookup[:,np.arange(N),class_group[z]].sum(1) for t,z in states.items()}
        events,ret=retention(states,steps,group,address)
        if events!=metric['survival']: raise ValueError(f'{unit}: saved retention does not match tokens/logs')
        final=scores[0];worst=int(np.argmin(final))
        if not np.isclose(final.min(),metric['final_ideal_min_margin'],atol=2e-12,rtol=0):
            raise ValueError(f'{unit}: final margin mismatch')
        if ret['total_address_hits_t0']!=metric['final_address_hits'] or ret['total_address_hits_t0']/N!=metric['final_address_hit_fraction']:
            raise ValueError(f'{unit}: final hit mismatch')
        tab=saved_summary[(saved_summary.source_id==attempt['source_id'])&(saved_summary.arm==attempt['arm'])].iloc[0]
        for key in ('final_address_hit_fraction','final_ideal_min_margin'):
            if not np.isclose(tab[key],metric[key],atol=2e-12,rtol=0):
                raise ValueError(f'{unit}: CSV/metrics mismatch: {key}')
        for row in steps:
            t=row['timestep']
            if not np.isclose(scores[t].min(),row['actual_ideal_min_margin'],atol=2e-12,rtol=0):
                raise ValueError(f'{unit} t={t}: actual margin mismatch')
            valid=states[t]!=MASK
            hit_count=int((group[states[t][valid]]==address[valid]).sum())
            if hit_count!=row['actual_address_hits'] or int((~valid).sum())!=row['outgoing_mask_count']:
                raise ValueError(f'{unit} t={t}: step counts disagree with tokens')
            if 11<=t<=60:
                p_set=set(row['shadows']['P']['selected']);pe_set=set(row['shadows']['P-E']['selected'])
                if len(p_set^pe_set)!=row['P_vs_PE_symmetric_difference'] or bool(p_set^pe_set)!=row['P_vs_PE_different']:
                    raise ValueError(f'{unit} t={t}: ledger-trigger flags disagree with selected sets')
                chosen=np.array(row.get('executed_selected',[]),dtype=int)
                if attempt['arm']=='N':
                    if len(chosen): raise ValueError('Native arm unexpectedly recorded interventions')
                elif len(chosen)!=row['initial_count'] or np.any(states[t+1][chosen]!=MASK) or np.any(states[t][chosen]==MASK):
                    raise ValueError(f'{unit} t={t}: recorded intervention violates MASK selection rule')
            trajectories.append(dict(**prefix,timestep=t,min_margin=float(scores[t].min()),
                current_worst_pair=int(scores[t].argmin()),final_worst_pair_margin=float(scores[t][worst]),
                address_hits=row['actual_address_hits'],remaining_mask=row['outgoing_mask_count'],
                expected_local_min_of_means=row.get('expected_J')))
        for e in events: event_rows.append(dict(**prefix,**e))
        for pair in pair_info:
            i=pair['pair_id'];a,b,c=(float(scores[t][i]) for t in (11,10,0))
            pair_rows.append(dict(**prefix,**pair,margin_t11=a,margin_t10=b,margin_t0=c,
                final_nonpositive=c<=0,near_numerical_tie=abs(c)<=1e-12,change_11_to_0=c-a,
                failure_stage=classify_failure(a,b,c)))
        worst_rows.append(dict(**prefix,**pair_info[worst],margin_t11=float(scores[11][worst]),
            margin_t10=float(scores[10][worst]),margin_t0=float(final[worst]),
            failure_stage=classify_failure(scores[11][worst],scores[10][worst],final[worst])))
        summaries.append(dict(**prefix,**ret,final_address_hit_fraction=metric['final_address_hit_fraction'],
            min_margin_t11=float(scores[11].min()),min_margin_t10=float(scores[10].min()),min_margin_t0=float(final.min()),
            nonpositive_pairs_t0=int((final<=0).sum()),same_history_different_steps=metric['same_history_P_PE_different_steps'],
            swaps=metric['executed_swap_count'],wall_seconds=attempt['wall_seconds'],
            content_only_kl=metric['conditional_content_kl_sum']))
        with Image.open(read_path(unit+'/image.png')) as im:
            im.load()
            if im.size!=(256,256) or im.mode!='RGB': raise ValueError(f'{unit}: unexpected image size/mode')
        fidelity.append(dict(unit=unit,token_log_metrics_consistent=True,image_decodes=True))
        print(f'Checked {number+1}/48: {unit}',flush=True)
    summary=save_csv(summaries,output/'source_summary.csv')
    save_csv(event_rows,output/'retention_events.csv');save_csv(trajectories,output/'trajectories.csv')
    pair_frame=save_csv(pair_rows,output/'all_pair_margins.csv')
    worst_frame=save_csv(worst_rows,output/'worst_pair_per_image.csv')
    save_csv(worst_frame[worst_frame.arm=='P'],output/'P_worst_pairs.csv')
    numeric=[c for c in summary.select_dtypes(include=np.number).columns]
    arm_summary=summary.groupby('arm')[numeric].mean().reindex(ARMS)
    arm_summary['sources']=summary.groupby('arm').size()
    arm_summary['positive_min_sources']=summary.groupby('arm')['min_margin_t0'].apply(lambda x:int((x>0).sum()))
    arm_summary.reset_index().to_csv(output/'arm_summary.csv',index=False)
    freq=pair_frame.groupby(['arm','pair_id','relative_type'],as_index=False).agg(
        nonpositive_sources=('final_nonpositive','sum'),mean_margin=('margin_t0','mean'))
    freq.sort_values(['arm','nonpositive_sources'],ascending=[True,False]).to_csv(output/'failure_pair_frequency.csv',index=False)
    save_csv(fidelity,output/'integrity_checks.csv')
    # Blind sheets use the existing fixed anonymization. No model scores or quality verdicts.
    key=pd.read_csv(read_path('quality_key_PRIVATE.csv'))
    if len(key)!=48 or set(key.unit)!=set(r['unit'] for r in attempts) or key.blind_id.nunique()!=48:
        raise ValueError('Invalid quality anonymization key')
    existing=pd.read_csv(read_path('blind_quality/unscored_review.csv'))
    if len(existing)!=96 or existing[['blind_id','reviewer']].drop_duplicates().shape[0]!=96 or set(existing.blind_id)!=set(key.blind_id):
        raise ValueError('Quality form missing images/reviewer rows')
    existing.to_csv(output/'quality_review_copy.csv',index=False)
    prompt={r['unit']:r['prompt'] for r in attempts}
    for page in range(8):
        fig,axes=plt.subplots(2,3,figsize=(13,10))
        for ax,(_,row) in zip(axes.ravel(),key.iloc[page*6:(page+1)*6].iterrows()):
            original=read_path(row.unit+'/image.png')
            blind=read_path('blind_quality/'+row.blind_id+'.png')
            if original.read_bytes()!=blind.read_bytes(): raise ValueError('Anonymous image differs from source')
            with Image.open(blind) as im: ax.imshow(np.asarray(im))
            ax.set_title(row.blind_id+'\n'+textwrap.fill(prompt[row.unit],42),fontsize=9);ax.axis('off')
        fig.suptitle('G1 blind quality review — no automatic quality score',fontsize=14)
        fig.tight_layout();fig.savefig(output/f'quality_blind_{page+1:02}.png',dpi=120);plt.close(fig)
    # Descriptive paired-source margin plot; n=8 source units, not 70 pairs or 400 steps.
    fig,ax=plt.subplots(figsize=(9,5))
    for src,data in summary.groupby('source_id'):
        vals=data.set_index('arm').loc[ARMS,'min_margin_t0']
        ax.plot(ARMS,vals*1000,marker='o',alpha=.55,label=src)
    ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Final ideal minimum margin × 1000')
    ax.set_title('Eight source-level outcomes (not RGB detection rates)')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(output/'margin_by_source.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for arm in ARMS[1:]:
        data=summary[summary.arm==arm]
        ax.plot([10,0],[data.unique_group_retention_t10.mean(),data.unique_group_retention_t0.mean()],marker='o',label=arm)
    ax.invert_xaxis();ax.set_xticks([10,0]);ax.set_ylim(0,1.05)
    ax.set_xlabel('Observed endpoint after step t (successful-position cohort fixed after t11)')
    ax.set_ylabel('Mean per-source unique successful-position group retention')
    ax.legend();fig.tight_layout();fig.savefig(output/'retention.png',dpi=150);plt.close(fig)
    p_heat=pair_frame[pair_frame.arm=='P'].pivot(index='source_id',columns='pair_id',values='margin_t0')*1000
    fig,ax=plt.subplots(figsize=(14,4))
    scale=max(float(np.abs(p_heat.values).max()),1e-12)
    heat=ax.imshow(p_heat.values,aspect='auto',cmap='RdYlGn',vmin=-scale,vmax=scale)
    ax.set_yticks(range(len(p_heat)),p_heat.index);ax.set_xlabel('Frozen geometry pair ID (0-based)')
    ax.set_title('P: all 70 final ideal margins; red = incorrect/tied competition, not RGB failure rate')
    fig.colorbar(heat,ax=ax,label='Margin × 1000');fig.tight_layout()
    fig.savefig(output/'P_pair_heatmap.png',dpi=150);plt.close(fig)
    changed=[name for name,digest in before.items() if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest]
    if changed: raise RuntimeError('Inputs changed during analysis: '+str(changed))
    (output/'input_sha256.json').write_text(json.dumps(before,indent=2))
    review=dict(source=str(source),completed_units=48,verified_inputs=len(before),original_files_unchanged=True,
        review_scope='Existing artifacts only; no model loading, generation, training or original-result modification',
        human_quality_review='pending: contact sheets and score form provided, not an automated visual judgment',
        G2_decision='pending_joint_review_of_quality_retention_and_failed_pairs',
        P_worst_pairs=worst_frame[worst_frame.arm=='P'].to_dict(orient='records'),
        warnings=['Repeated successful-write events are correlated; unique-position retention is also reported.',
                  'A retained group may disappear and reappear; endpoint retention is not a causal survival guarantee.',
                  'Pair failures are correlated within a source. Do not count 70 pairs as independent samples.',
                  'Minimum of conditional mean margins differs from expectation of a sampled minimum.',
                  'Content-only KL excludes set-selection distribution change.',
                  'No automatic G2 pass threshold has been added.'])
    (output/'review.json').write_text(json.dumps(review,indent=2))
    (output/'README.txt').write_text('Source read-only. Derived review artifacts only.\n'
        'Read arm_summary.csv, source_summary.csv, P_worst_pairs.csv and failure_pair_frequency.csv.\n'
        'Quality sheets contain anonymous existing images; score quality_review_copy.csv independently.\n'
        'Do not infer RGB readability, robustness, quality equivalence or automatic G2 approval.\n')
    archive=shutil.make_archive(str(output)+'_review','zip',output)
    print('Review directory:',output);print('Review archive:',archive)
    return review


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run(args.source,args.output)
