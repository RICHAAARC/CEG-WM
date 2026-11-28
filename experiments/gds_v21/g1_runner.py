"""Fixed 8-source / 48-attempt G1. No training and no hidden retries."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid
import numpy as np
from compat import compatible_checkpoint,ensure_vq_component_compatibility,source_timeout
from geometry import build_coefficients,token_field,render,score,groups,addresses,MASK
from g1_sampling import ARMS,DOMAINS,stream_seed,sampling_context


def write_json(path,data):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False))
    os.replace(temporary,path)


def plan(config):
    if config['source_count']!=8 or len(config['sources'])!=8 or config['max_generation_attempts']!=48 or tuple(config['arms'])!=ARMS:
        raise ValueError('Frozen 8-source/48-attempt protocol changed')
    if len({s['seed'] for s in config['sources']})!=8 or len({s['prompt'] for s in config['sources']})!=8:
        raise ValueError('Duplicate source')
    # Counterbalanced order avoids always placing P at the same loading/thermal position.
    rows=[]
    for index,source in enumerate(config['sources']):
        ordered=ARMS[index%6:]+ARMS[:index%6]
        for arm in ordered:
            rows.append(dict(**source,arm=arm,unit=source['source_id']+'_'+arm,status='not_run'))
    assert len(rows)==48
    return rows


def validate_g0(report,expected):
    if report.get('stage')!='G0_REAL_MODEL' or report.get('status')!='passed':
        raise ValueError('G0 real-model pass required')
    checks=report.get('checks',{})
    keys=('all_100_token_and_rng_states_equal','decoded_rgb_equal','final_rng_equal','all_native_gumbel_replays_equal')
    if not all(checks.get(k) is True for k in keys) or checks.get('intervention_window_records')!=50:
        raise ValueError('Incomplete G0 checks')
    if report.get('files')!=expected['files']:
        raise ValueError('G0 report is not from the reviewed G0 payload')
    if report.get('observed_native_opportunity_timesteps')!=list(range(60,10,-1)):
        raise ValueError('G0 opportunity record differs from the supplied completed run')
    if len(report.get('attempts',[]))!=2 or any(r.get('status')!='complete' for r in report['attempts']):
        raise ValueError('Incomplete G0 attempt ledger')


def source_metrics(rows,states,coeff,pairs):
    final=states[0]
    field=token_field(final)
    algebra=np.einsum('hig,ig->h',coeff,field,optimize=False)
    direct=np.array([score(render(field,r),r)-score(render(field,r),h) for r,h in pairs])
    err=float(abs(algebra-direct).max())
    if err>2e-12:
        raise AssertionError('Final ideal geometry identity failed')
    group,address=groups(),addresses()
    def hits(tokens):
        hit=np.zeros(len(tokens),bool)
        valid=tokens!=MASK
        hit[valid]=group[tokens[valid]]==address[valid]
        return hit
    hit0,hit10=hits(final),hits(states[10])
    survival=[]
    for row in rows:
        chosen=np.array(row.get('executed_selected',[]),dtype=int)
        if len(chosen):
            t=row['timestep']
            successful=chosen[hits(states[t])[chosen]]
            survival.append(dict(timestep=t,selected=len(chosen),hits_when_sampled=len(successful),
                address_hits_at_t10=int(hit10[successful].sum()),address_hits_at_t0=int(hit0[successful].sum()),
                exact_token_retained_t10=int((states[t][successful]==states[10][successful]).sum()),
                exact_token_retained_t0=int((states[t][successful]==final[successful]).sum())))
    window=[r for r in rows if 11<=r['timestep']<=60]
    return dict(final_address_hits=int(hit0.sum()),final_address_hit_fraction=float(hit0.mean()),
        final_ideal_min_margin=float(algebra.min()),geometry_identity_max_error=err,
        same_history_opportunity_steps=sum(r.get('opportunity',False) for r in window),
        same_history_P_PE_different_steps=sum(r.get('P_vs_PE_different',False) for r in window),
        executed_swap_count=sum(len(r.get('shadows',{}).get(r.get('executed_arm',''),{}).get('swaps',[])) for r in window),
        scheduler_and_diagnostic_seconds=sum(r['scheduler_and_diagnostic_seconds'] for r in rows),
        conditional_content_kl_sum=sum(r.get('selected_conditional_kl_sum') or 0 for r in rows),
        kl_scope='sum of realized conditional content KL only; NOT total transition or path KL',
        survival=survival)


def summarize(status,out):
    columns=['source_id','arm','status','final_address_hit_fraction','final_ideal_min_margin',
             'same_history_P_PE_different_steps','executed_swap_count','wall_seconds','generator_forwards']
    with (out/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns)
        writer.writeheader()
        writer.writerows({k:row.get(k) for k in columns} for row in status['attempts'])
    complete=[r for r in status['attempts'] if r['status']=='complete']
    p=[r for r in complete if r['arm']=='P']
    status['summary']=dict(expected_sources=8,fixed_denominator=48,completed=len(complete),
        failed=sum(r['status']=='failed' for r in status['attempts']),
        unresolved_running=sum(r['status']=='running' for r in status['attempts']),
        not_run=sum(r['status']=='not_run' for r in status['attempts']),
        P_same_history_different_steps=sum(r['same_history_P_PE_different_steps'] for r in p),
        all_histories_different_steps=sum(r['same_history_P_PE_different_steps'] for r in complete),
        decision=('incomplete_stop_review_records' if len(complete)!=48 else
                  'stop_no_ledger_trigger_on_any_history' if not any(r['same_history_P_PE_different_steps'] for r in complete) else
                  'mechanism_triggered_review_reference_quality_and_cost_before_any_next_stage'),
        scope='Token/ideal-symbol diagnostics only; no RGB detector, FPR, robustness or quality-equivalence conclusion')
    write_json(out/'report.json',status)


def symbol_svg(labels,path):
    palette=['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896',
             '#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#bcbd22']
    cells=''.join(f'<rect x="{i%32*8}" y="{i//32*8}" width="8" height="8" fill="{palette[int(g)]}"/>'
                  for i,g in enumerate(labels))
    Path(path).write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'+cells+'</svg>')


def quality_sheet(status,out):
    """Unscored, anonymized human-review material; never substitutes for review."""
    folder=out/'blind_quality';folder.mkdir(exist_ok=True)
    order=np.random.default_rng(281128109).permutation(48)
    keys=[];scores=[]
    for slot,index in enumerate(order):
        row=status['attempts'][int(index)]
        blind_id=f'image-{slot:03}'
        keys.append(dict(blind_id=blind_id,unit=row['unit'],status=row['status']))
        if row['status']=='complete':
            shutil.copyfile(out/row['unit']/'image.png',folder/f'{blind_id}.png')
        for reviewer in ('reviewer-1','reviewer-2'):
            scores.append(dict(blind_id=blind_id,prompt=row['prompt'],available=row['status']=='complete',
                reviewer=reviewer,visual_quality_1_to_5='',prompt_match_1_to_5='',severe_defect_0_or_1=''))
    for path,data in ((out/'quality_key_PRIVATE.csv',keys),(folder/'unscored_review.csv',scores)):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)


def run(args):
    root=Path(__file__).resolve().parent
    config=json.loads((root/'g1_config.json').read_text())
    g0=json.loads(Path(args.g0_report).read_text())
    expected=json.loads((root/'g1_g0_evidence.json').read_text())
    validate_g0(g0,expected)
    roster=plan(config)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)  # rerunning cannot overwrite or silently restart this 48-run roster
    started=time.monotonic()
    status=dict(stage='G1',status='running',config=config,attempts=roster,
        runtime_id=uuid.uuid4().hex,generation_attempts_started=0,reader_training_started=False,
        G2_started=False,G0_origin=str(args.g0_report),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.suffix in ('.py','.json','.txt')})
    write_json(out/'report.json',status)
    shutil.copyfile(args.g0_report,out/'accepted_g0_report.json')
    pipe=None
    try:
        import torch
        from diffusers import VQDiffusionPipeline
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU model-generation fallback')
        versions={p:importlib.metadata.version(p) for p in ('torch','diffusers','transformers','numpy','Pillow')}
        if versions!=g0['versions'] or torch.cuda.get_device_name(0)!=g0['gpu']:
            raise RuntimeError('Runtime versions/GPU type differ from accepted G0. Stop for review; do not rerun G0 automatically.')
        status.update(versions=versions,gpu=torch.cuda.get_device_name(0))
        coeff,pairs,excluded=build_coefficients()
        write_json(out/'geometry_assets.json',dict(pairs=pairs,excluded=excluded,
            coefficient_sha256=hashlib.sha256(coeff.tobytes()).hexdigest()))
        symbol_svg(addresses(),out/'target_template.svg')
        ensure_vq_component_compatibility()
        print('Loading the G0-verified pinned checkpoint.',flush=True)
        snapshot=VQDiffusionPipeline.download(config['model_id'],revision=config['model_revision'])
        with compatible_checkpoint(snapshot) as view:
            pipe=VQDiffusionPipeline.from_pretrained(view,local_files_only=True,torch_dtype=torch.float16).to('cuda')
        pipe.vqvae.to(dtype=torch.float32)
        for name in ('vqvae','transformer','text_encoder','learned_classifier_free_sampling_embeddings'):
            module=getattr(pipe,name,None)
            if isinstance(module,torch.nn.Module): module.eval().requires_grad_(False)
        if (pipe.scheduler.config.num_train_timesteps!=100 or pipe.transformer.config.sample_size!=32
                or pipe.transformer.config.num_vector_embeds!=4097 or pipe.vqvae.config.num_vq_embeddings!=4096):
            raise RuntimeError('Model interface mismatch')
        torch.backends.cudnn.benchmark=False
        pipe.set_progress_bar_config(disable=True)
        for index,attempt in enumerate(roster):
            if time.monotonic()-started>config['max_run_seconds']:
                status['stop_reason']='wall_time_budget'; break
            if status['generation_attempts_started']>=48:
                raise RuntimeError('Hard generation cap reached')
            directory=out/attempt['unit']; directory.mkdir(exist_ok=False)
            attempt.update(status='running',stream_seeds={d:stream_seed(attempt['seed'],attempt['arm'],d) for d in DOMAINS})
            status['generation_attempts_started']+=1
            write_json(out/'report.json',status)
            start=time.monotonic(); rows=[]; states={}; count=[0]
            print(f'G1 {index+1}/48 | {attempt["unit"]}',flush=True)
            def forward_hook(module,inputs,output): count[0]+=1
            hook=pipe.transformer.register_forward_hook(forward_hook)
            torch.cuda.reset_peak_memory_stats()
            try:
                with (directory/'steps.jsonl').open('w') as log:
                    def sink(row,history,tokens):
                        row['executed_arm']=attempt['arm']
                        rows.append(row); states[row['timestep']]=tokens.copy()
                        log.write(json.dumps(row)+'\n'); log.flush()
                        if row['timestep']%20==0:
                            print(f'  t={row["timestep"]}; MASK={row["outgoing_mask_count"]}; hits={row["actual_address_hits"]}',flush=True)
                    with source_timeout(config['max_source_seconds']),torch.inference_mode(),sampling_context(
                        pipe.scheduler,attempt['arm'],attempt['seed'],coeff,sink) as seen:
                        output=pipe(attempt['prompt'],num_inference_steps=100,guidance_scale=5.,truncation_rate=1.,
                            generator=torch.Generator(device='cuda').manual_seed(attempt['seed']))
                if seen!=set(range(100)) or count[0]!=100 or np.any(states[0]==MASK):
                    raise AssertionError('Step/NFE/final-token invariant failed')
                output.images[0].save(directory/'image.png')
                symbol_svg(groups()[states[0]],directory/'final_symbol_map.svg')
                np.savez_compressed(directory/'private_tokens.npz',timesteps=np.arange(99,-1,-1),
                                    tokens=np.stack([states[t] for t in range(99,-1,-1)]))
                metric=source_metrics(rows,states,coeff,pairs)
                write_json(directory/'metrics.json',metric)
                attempt.update({k:v for k,v in metric.items() if k!='survival'},status='complete')
            except Exception as error:
                attempt.update(status='failed',error=f'{type(error).__name__}: {error}')
                (directory/'failure.txt').write_text(traceback.format_exc())
                status['stop_reason']='unit_failed_no_retry'
            finally:
                hook.remove()
                attempt.update(wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),generator_forwards=count[0])
                status['wall_seconds']=time.monotonic()-started
                summarize(status,out)
                print(json.dumps({k:attempt.get(k) for k in ('unit','status','final_address_hit_fraction',
                    'same_history_P_PE_different_steps','executed_swap_count','wall_seconds','error')}),flush=True)
            if attempt['status']!='complete': break
        status['status']='complete' if all(r['status']=='complete' for r in roster) else 'stopped'
        quality_sheet(status,out)
        status['quality_review']='not performed; anonymized images and blank two-reviewer sheet saved'
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        status['wall_seconds']=time.monotonic()-started
        summarize(status,out)
    print(json.dumps(status['summary'],indent=2),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--g0-report',required=True)
    args=parser.parse_args()
    result=run(args)
    if result['status']!='complete': raise SystemExit(2)
