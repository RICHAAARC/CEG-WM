"""Explicit G0 checkpoint verification: two native generations, no embedding/training.

The two repeats are engineering diagnostics, separate from the future 48-run G1.
No retries, resume, threshold fitting, or automatic transition to G1.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
from compat import compatible_checkpoint,ensure_vq_component_compatibility,source_timeout
from core import normalize_logits,biased_content
from geometry import groups,addresses,MASK,N,K
from g0_cpu import manifest
from observe import native_trace


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def kernel_record(t,raw,history,tokens):
    expected=(1,K+(t!=0),N)
    if tuple(raw.shape)!=expected:
        raise AssertionError(f'Unexpected actual-kernel shape {tuple(raw.shape)} != {expected}')
    # All timesteps inspected for finite actual logits; full content diagnostics only
    # in the frozen intervention window, so this observer cannot create new draws.
    values=raw.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ArithmeticError('Non-finite actual kernel logits')
    incoming=np.asarray(history).ravel()==MASK
    outgoing=np.asarray(tokens).ravel()==MASK
    record=dict(timestep=t,shape=list(expected),incoming_mask=int(incoming.sum()),
                outgoing_mask=int(outgoing.sum()),newly_unmasked=int((incoming&~outgoing).sum()))
    if 11<=t<=60:
        p=normalize_logits(values[0].T)
        q,beta,kl=biased_content(p,groups(),addresses())
        m=p[:,MASK]
        eligible=incoming&(m>=.005)&(m<=.995)
        out_count=int((eligible&~outgoing).sum())
        in_count=int((eligible&outgoing).sum())
        target=(q*(groups()[None,:]==addresses()[:,None])).sum(1)
        qmass=q.sum(1)
        content_supported=p[:,:MASK].sum(1)>0
        if not np.allclose(qmass[content_supported],1.,atol=1e-12,rtol=1e-12):
            raise ArithmeticError('Conditional content law does not normalize')
        if np.any((beta<0)|(beta>1)) or kl.max()>.10+1e-12:
            raise ArithmeticError('Frozen beta/content-KL budget violated')
        if np.any((p[:,:MASK]==0)&(q!=0)):
            raise ArithmeticError('Content tilt created unsupported classes')
        record.update(eligible_count=int(eligible.sum()),out_pool_count=min(out_count,32),
            in_pool_count=min(in_count,32),exchange_opportunity=bool(out_count and in_count),
            mask_probability_range_on_incoming=([float(m[incoming].min()),float(m[incoming].max())]
                                                if incoming.any() else None),
            conditional_kl_max=float(kl.max()),beta_max=float(beta.max()),
            target_probability_mean_on_incoming=float(target[incoming].mean()) if incoming.any() else None,
            kernel_normalization_error=float(abs(p.sum(1)-1).max()))
    return record


def run(args):
    root=Path(__file__).resolve().parent
    config=json.loads((root/'g0_config.json').read_text())
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status=dict(stage='G0_REAL_MODEL',status='running',config=config,files=manifest(),
        scope='same-runtime native-path engineering evidence only; no watermark effectiveness claim',
        attempts=[dict(name='native_baseline',status='not_run'),dict(name='native_kernel_observer',status='not_run')],
        G1_started=False,reader_training_started=False)
    def save():
        status['wall_seconds']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        cpu=json.loads(Path(args.cpu_report).read_text())
        if cpu.get('stage')!='G0_CPU' or cpu.get('status')!='passed' or cpu.get('files')!=status['files']:
            raise RuntimeError('Matching G0 CPU report is required; run CPU checks on this exact payload first')
        import torch
        from diffusers import VQDiffusionPipeline
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required. CPU checks cannot pass the real-model G0 gate.')
        versions={name:importlib.metadata.version(name) for name in ('torch','diffusers','transformers','numpy','Pillow')}
        for name in ('diffusers','transformers'):
            if versions[name]!=config[name+'_version']:
                raise RuntimeError(f'Install {name}=={config[name+"_version"]}')
        status.update(versions=versions,python=platform.python_version(),gpu=torch.cuda.get_device_name(0),
            gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,cuda=torch.version.cuda,
            native_global_rng_initial=digest(torch.cuda.get_rng_state().cpu().numpy()))
        torch.backends.cudnn.benchmark=False
        save()
        ensure_vq_component_compatibility()
        print('Loading pinned checkpoint; original cache/weights remain unchanged.',flush=True)
        snapshot=VQDiffusionPipeline.download(config['model_id'],revision=config['model_revision'])
        status['checkpoint_snapshot']=str(snapshot)
        with compatible_checkpoint(snapshot) as view:
            pipe=VQDiffusionPipeline.from_pretrained(view,local_files_only=True,torch_dtype=torch.float16).to('cuda')
        pipe.vqvae.to(dtype=torch.float32)
        for name in ('vqvae','transformer','text_encoder','learned_classifier_free_sampling_embeddings'):
            module=getattr(pipe,name,None)
            if isinstance(module,torch.nn.Module):
                module.eval().requires_grad_(False)
        if (pipe.scheduler.config.num_train_timesteps!=100 or pipe.transformer.config.sample_size!=32
                or pipe.transformer.config.num_vector_embeds!=4097 or pipe.vqvae.config.num_vq_embeddings!=4096):
            raise RuntimeError('Pinned model has an unexpected scheduler/grid/codebook interface')
        import inspect
        status['scheduler_step_sha256']=hashlib.sha256(inspect.getsource(type(pipe.scheduler).step).encode()).hexdigest()
        status['posterior_sha256']=hashlib.sha256(inspect.getsource(type(pipe.scheduler).q_posterior).encode()).hexdigest()
        status['scheduler_config']=dict(pipe.scheduler.config)
        baseline_trace,baseline_image,baseline_rng=None,None,None
        kernel_rows=[]
        for index,attempt in enumerate(status['attempts']):
            attempt.update(status='running',seed=config['seed'])
            save()  # fixed attempt ledger BEFORE generation; no automatic retry
            print(f'G0 attempt {index+1}/2: {attempt["name"]}',flush=True)
            trace=[]
            start=time.monotonic()
            rng=torch.Generator(device='cuda').manual_seed(config['seed'])
            torch.cuda.reset_peak_memory_stats()
            def sink(t,raw,history,tokens,rng_state,replay_ok):
                trace.append((t,tokens.copy(),rng_state.copy()))
                if index==1:
                    row=kernel_record(t,raw,history,tokens)
                    row['native_gumbel_replay_equal']=replay_ok
                    kernel_rows.append(row)
                    with (out/'actual_kernels.jsonl').open('a') as f:
                        f.write(json.dumps(row)+'\n')
                if t%10==0:
                    print(f'  t={t}; remaining MASK={(tokens==MASK).sum()}',flush=True)
            try:
                with source_timeout(config['max_source_seconds']),torch.inference_mode(),native_trace(pipe.scheduler,sink,capture_kernel=index==1) as seen:
                    generated=pipe(config['prompt'],num_inference_steps=config['num_inference_steps'],
                        guidance_scale=config['guidance_scale'],truncation_rate=config['truncation_rate'],generator=rng)
                if seen!=set(range(100)) or len(trace)!=100:
                    raise AssertionError('Missing or repeated native sampler timesteps')
                generated.images[0].save(out/f'{attempt["name"]}.png')
                image=np.asarray(generated.images[0])
                np.savez_compressed(out/f'{attempt["name"]}_private_trace.npz',
                    timesteps=np.array([x[0] for x in trace]),tokens=np.stack([x[1] for x in trace]),
                    rng_states=np.stack([x[2] for x in trace]))
                if index==0:
                    baseline_trace,baseline_image,baseline_rng=trace,image.copy(),rng.get_state().clone()
                else:
                    equal=all(a[0]==b[0] and np.array_equal(a[1],b[1]) and np.array_equal(a[2],b[2])
                              for a,b in zip(baseline_trace,trace))
                    status['checks']=dict(all_100_token_and_rng_states_equal=equal,
                        decoded_rgb_equal=bool(np.array_equal(baseline_image,image)),
                        final_rng_equal=bool(torch.equal(baseline_rng,rng.get_state())),
                        all_native_gumbel_replays_equal=all(r['native_gumbel_replay_equal'] for r in kernel_rows),
                        intervention_window_records=sum(11<=r['timestep']<=60 for r in kernel_rows))
                    if not all(status['checks'][key] for key in ('all_100_token_and_rng_states_equal',
                        'decoded_rgb_equal','final_rng_equal','all_native_gumbel_replays_equal')):
                        raise AssertionError('Same-runtime zero-intervention consistency failed')
                attempt['status']='complete'
            except Exception as error:
                attempt.update(status='failed',error=f'{type(error).__name__}: {error}')
                raise
            finally:
                attempt.update(wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                save()
        opportunities=[r['timestep'] for r in kernel_rows if r.get('exchange_opportunity')]
        status['observed_native_opportunity_timesteps']=opportunities
        if status['checks']['intervention_window_records']!=50:
            raise AssertionError('Frozen intervention window was not fully inspected')
        status['status']='passed' if opportunities else 'blocked_no_native_exchange_opportunity'
        status['decision']=('G0 checks passed on this runtime; review records before preparing the bounded G1 notebook'
                            if opportunities else 'Stop: no native exchange opportunity in the frozen window; do not widen it silently')
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({k:status[k] for k in ('stage','status','decision','checks','observed_native_opportunity_timesteps')},indent=2),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--cpu-report',required=True)
    args=parser.parse_args()
    result=run(args)
    if result['status']!='passed':
        raise SystemExit(2)
