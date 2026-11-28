"""One frozen VQ encode per saved RGB. No generation, training or attack."""
import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import time
import traceback
import zipfile
import numpy as np
import pandas as pd
import torch
from geometry import groups, IDENTITY, hypothesis_pairs, score
from g2a_reader import (load_saved_g1, make_priors, evaluate, token_probs,
                       digest, write_json, write_csv, rgb_tensor)


def validate_loading(info):
    for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
        if key not in info or info[key]:
            raise RuntimeError(f'VQ loading integrity failed: {key}={info.get(key)}')


def hard_field(indices):
    labels=groups()[np.asarray(indices).reshape(32,32)]
    return np.eye(16,dtype=np.float32)[labels].repeat(2,axis=0).repeat(2,axis=1)


@torch.inference_mode()
def read_rgb(model, image, device):
    """Blind interface: no source tokens, address template or metadata."""
    h=model.encode(rgb_tensor(image).unsqueeze(0).to(device)).latents
    if h.shape!=(1,128,32,32) or not torch.isfinite(h).all():
        raise ValueError('Unexpected native encoded feature interface')
    _,_,info=model.quantize(h)
    indices=info[2].reshape(-1)
    if indices.shape!=(1024,) or torch.any((indices<0)|(indices>=4096)):
        raise ValueError('Invalid native quantization indices')
    # Same native cdist metric/order, on the SAME features. Not a second encode.
    flat=h.permute(0,2,3,1).contiguous().reshape(1024,128)
    distances=torch.cdist(flat,model.quantize.embedding.weight)
    if not torch.isfinite(distances).all() or not torch.equal(distances.argmin(1),indices):
        raise ValueError('Native index/distance interface disagreement')
    return indices.cpu().numpy(), distances.cpu().numpy()


def token_diagnostics(pred, truth, distances, train_counts):
    pred=np.asarray(pred,dtype=np.int64).reshape(-1)
    truth=np.asarray(truth,dtype=np.int64).reshape(-1)
    if distances.shape!=(1024,4096) or len(pred)!=1024 or len(truth)!=1024:
        raise ValueError('Unexpected diagnostic shapes')
    # Evaluation labels enter here only, AFTER the blind reader output is fixed.
    true_distance=distances[np.arange(1024),truth]
    rank=1+(distances<true_distance[:,None]).sum(1)+((distances==true_distance[:,None]) &
             (np.arange(4096)[None,:]<truth[:,None])).sum(1)
    exact=pred==truth
    same_group=groups()[pred]==groups()[truth]
    if np.any(exact & ~same_group) or not np.array_equal(rank==1,exact):
        raise ValueError('Token/group/rank consistency failure')
    errors=~exact
    row=dict(token_accuracy=float(exact.mean()),group_accuracy=float(same_group.mean()),
        token_errors=int(errors.sum()),same_group_token_errors=int((errors&same_group).sum()),
        cross_group_token_errors=int((errors&~same_group).sum()),
        cross_group_given_token_error=float((errors&~same_group).sum()/errors.sum()) if errors.any() else None,
        true_token_top1=float((rank<=1).mean()),true_token_top4=float((rank<=4).mean()),
        true_token_top16=float((rank<=16).mean()),true_token_rank_median=float(np.median(rank)))
    nearest=distances[np.arange(1024),pred]
    for label,mask in [('seen',train_counts[truth]>0),('unseen',train_counts[truth]==0)]:
        row[label+'_positions']=int(mask.sum())
        row[label+'_group_accuracy']=float(same_group[mask].mean()) if mask.any() else None
    return row,dict(true_token_rank=rank,true_minus_nearest_distance=true_distance-nearest)


def clean_geometry(data, outputs, old_outputs):
    wrong=[h for r,h in hypothesis_pairs()[0] if r==IDENTITY]
    if len(wrong)!=8:raise ValueError('Frozen clean hypothesis roster changed')
    rows=[]
    for d in data:
        ideal=np.eye(16)[d['target']].repeat(2,axis=0).repeat(2,axis=1)
        for method,field in [('ideal_tokens',ideal),('native_vq',outputs[d['unit']]),
                             ('saved_unet',old_outputs[d['unit']])]:
            obs=field.reshape(4096,16)
            s0=score(obs,IDENTITY)
            for i,h in enumerate(wrong):
                sh=score(obs,h)
                rows.append({k:d[k] for k in ('unit','source_id','arm','split')} | dict(
                    method=method,pair_id=i,wrong_h=json.dumps(h),identity_score=s0,
                    wrong_score=sh,margin=s0-sh))
    return rows


def plot_results(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=pd.read_csv(out/'token_summary.csv')
    c=pd.read_csv(out/'reader_comparison.csv')
    h=c[(c.split=='heldout') & c.arm.isin(['N','B0','P'])]
    fig,ax=plt.subplots(figsize=(10,4))
    x=np.arange(len(h));ax.bar(x-.18,h.native_balanced_accuracy,.36,label='Frozen native VQ')
    ax.bar(x+.18,h.unet_balanced_accuracy,.36,label='Saved U-Net')
    ax.axhline(1/16,color='grey',linestyle='--',label='1/16 reference')
    ax.set_xticks(x,h.unit,rotation=25,ha='right');ax.set_ylim(0,1)
    ax.set_ylabel('Group balanced accuracy');ax.set_title('Heldout clean readback: two existing sources')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
    fig.tight_layout();fig.savefig(out/'heldout_readback.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    for arm,frame in t.groupby('arm'):
        ax.scatter(frame.token_accuracy,frame.group_accuracy,label=arm)
    ax.plot([0,1],[0,1],'--',color='grey');ax.set(xlim=(0,1),ylim=(0,1),
        xlabel='Exact Token accuracy',ylabel='Group accuracy',title='48 images; points are not independent sources')
    ax.legend();fig.tight_layout();fig.savefig(out/'token_vs_group.png',dpi=140);plt.close(fig)
    g=pd.read_csv(out/'clean_geometry.csv')
    g=g[(g.arm=='P') & (g.split=='heldout')].groupby(['unit','method']).margin.min().unstack()
    ax=g[['ideal_tokens','native_vq','saved_unet']].plot.bar(figsize=(8,4),rot=0)
    ax.axhline(0,color='grey');ax.set_ylabel('Minimum identity margin')
    ax.set_title('Clean identity vs 8 wrong hypotheses (no RGB attack)')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
    ax.figure.tight_layout();ax.figure.savefig(out/'clean_identity.png',dpi=140);plt.close(ax.figure)


def run(source, reference, output):
    root=Path(__file__).resolve().parent
    config=json.loads((root/'vq_readback_config.json').read_text())
    split_config=json.loads((root/'g2a_config.json').read_text())
    source,reference,out=map(lambda p:Path(p).resolve(),(source,reference,output))
    for original in (source,reference):
        if out==original or original in out.parents or out in original.parents:
            raise ValueError('Output must be separate from previous results')
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report=dict(stage=config['stage'],status='running',config=config,expected_units=48,
        encoder_forwards=0,decoder_forwards=0,optimizer_steps=0,new_generations=0,attacks=0,
        units=[dict(unit=f'g1-{i:03}_{a}',status='not_run') for i in range(8) for a in ('N','B0','B1','B2','P','P-E')],
        scope='Frozen clean RGB reencoding only; no robustness, full blind synchronization or FPR claim')
    write_json(out/'report.json',report)
    hashes={};reference_hashes={};hooks=[];native=None;old_handler=None
    try:
        def timeout(signum,frame):raise TimeoutError('30-minute job budget exhausted')
        old_handler=signal.signal(signal.SIGALRM,timeout);signal.alarm(config['max_run_seconds'])
        if not torch.cuda.is_available():raise RuntimeError('Select GPU runtime; no implicit CPU fallback')
        if importlib.metadata.version('diffusers')!=config['diffusers_version']:
            raise RuntimeError('Unexpected diffusers version; run the notebook setup cell')
        for name in ('report.json','input_hashes.json','metrics.csv','group_probabilities_64.npz'):
            path=reference/name
            if not path.is_file():raise FileNotFoundError(f'Missing G2-A record: {path}')
            reference_hashes[name]=digest(path)
        ref=json.loads((reference/'report.json').read_text())
        if ref.get('status')!='complete' or ref.get('step')!=2000 or ref.get('expected_units')!=48:
            raise ValueError('Expected the completed original G2-A run')
        for name in ('geometry.py','g2a_reader.py','g2a_config.json'):
            if digest(root/name)!=ref['code_hashes'][name]:
                raise ValueError(f'Reused helper differs from G2-A: {name}')
        data,hashes,split=load_saved_g1(source,split_config)
        if hashes!=json.loads((reference/'input_hashes.json').read_text()) or split!=ref['source_split']:
            raise ValueError('G1 inputs/source split differ from G2-A')
        old_table=pd.read_csv(reference/'metrics.csv')
        if len(old_table)!=336 or old_table.unit.nunique()!=48:
            raise ValueError('Incomplete saved U-Net comparison table')
        with np.load(reference/'group_probabilities_64.npz',allow_pickle=False) as z:
            if set(z.files)!={d['unit'] for d in data}:raise ValueError('Incomplete saved U-Net probabilities')
            old_outputs={u:z[u] for u in z.files}
        for a in old_outputs.values():
            if a.shape!=(64,64,16) or not np.isfinite(a).all() or np.any(a<0) or not np.allclose(a.sum(-1),1,atol=2e-6):
                raise ValueError('Invalid saved U-Net probabilities')
        truths={}
        for d in data:
            with np.load(source/d['unit']/'private_tokens.npz',allow_pickle=False) as z:
                truths[d['unit']]=z['tokens'][-1].copy()
        train=np.concatenate([truths[d['unit']] for d in data if d['split']=='train' and d['arm'] in ('N','B0')])
        counts=np.bincount(train,minlength=4096)
        write_json(out/'input_hashes.json',dict(G1=hashes,G2A=reference_hashes))
        write_json(out/'source_split.json',split)
        (out/'protocol.txt').write_text((root/'VQ_READBACK_README.txt').read_text())
        report.update(source=str(source),reference=str(reference),source_split=split,
            versions={p:importlib.metadata.version(p) for p in ('torch','diffusers','transformers','numpy','Pillow','pandas')},
            gpu=torch.cuda.get_device_name(0),code_hashes={p.name:digest(p) for p in root.iterdir() if p.suffix in ('.py','.json','.txt')})
        write_json(out/'report.json',report)
        print('Loading frozen original VQ component only.',flush=True)
        from huggingface_hub import snapshot_download
        from diffusers import VQModel
        snapshot=Path(snapshot_download(config['model_id'],revision=config['revision'],
            allow_patterns=['vqvae/config.json','vqvae/diffusion_pytorch_model.bin']))
        component=snapshot/'vqvae'
        report['checkpoint_hashes']={n:digest(component/n) for n in ('config.json','diffusion_pytorch_model.bin')}
        native,info=VQModel.from_pretrained(component,local_files_only=True,use_safetensors=False,
            torch_dtype=torch.float32,low_cpu_mem_usage=False,output_loading_info=True)
        write_json(out/'loading_info.json',info);validate_loading(info)
        if native.config.num_vq_embeddings!=4096 or native.config.vq_embed_dim!=128 or native.quantize.remap is not None:
            raise ValueError('Wrong VQ codebook interface')
        if not list(native.encoder.parameters()) or not list(native.quant_conv.parameters()):
            raise ValueError('Missing encoder/quant_conv parameters')
        report['runtime_source_hashes']={name:digest(inspect.getsourcefile(obj)) for name,obj in
                                         [('VQModel',VQModel),('VectorQuantizer',type(native.quantize))]}
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        native.eval().requires_grad_(False).to('cuda')
        def count_encoder(module,args):
            if report['encoder_forwards']>=48:raise RuntimeError('48 encoder-forward cap reached')
            report['encoder_forwards']+=1
        def prohibit_decoder(module,args):
            report['decoder_forwards']+=1
            raise RuntimeError('Decoder calls are outside the authorized readback scope')
        hooks=[native.encoder.register_forward_pre_hook(count_encoder),native.decoder.register_forward_pre_hook(prohibit_decoder)]
        torch.cuda.reset_peak_memory_stats()
        outputs={};summary=[]
        ledger={r['unit']:r for r in report['units']}
        for d in data:
            unit=d['unit'];row=ledger[unit]
            row.update(status='running',source_id=d['source_id'],arm=d['arm'],split=d['split'])
            write_json(out/'report.json',report)
            tick=time.monotonic()
            try:
                pred,distances=read_rgb(native,d['rgb'],'cuda')
                metric,diag=token_diagnostics(pred,truths[unit],distances,counts)
                outputs[unit]=hard_field(pred)
                folder=out/unit;folder.mkdir(exist_ok=False)
                np.savez_compressed(folder/'readback.npz',predicted_token=pred,predicted_group=groups()[pred],**diag)
                record={k:d[k] for k in ('unit','source_id','arm','split')}|metric
                summary.append(record);write_json(folder/'metrics.json',record)
                write_csv(out/'token_summary.csv',summary)
                row.update(status='complete',wall_seconds=time.monotonic()-tick,**metric)
            except BaseException as error:
                row.update(status='failed',error=f'{type(error).__name__}: {error}')
                raise
            finally:
                report.update(wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                write_json(out/'report.json',report)
            print(json.dumps(dict(unit=unit,status=row['status'],token_accuracy=metric['token_accuracy'],
                group_accuracy=metric['group_accuracy'],encoder_forwards=report['encoder_forwards'])),flush=True)
        if report['encoder_forwards']!=48 or report['decoder_forwards']!=0:
            raise RuntimeError('Forward-count invariant failed')
        rows,matrices=evaluate(data,outputs,make_priors(data),split_config['seed'])
        for row in rows:
            row.pop('ce')  # no clipped hard-vs-soft cross entropy comparison
            if row['method']=='reader':row['method']='native_vq'
        write_csv(out/'group_metrics.csv',rows)
        np.savez_compressed(out/'group_confusions.npz',**matrices)
        native_table=pd.DataFrame(rows);native_table=native_table[native_table.method=='native_vq']
        old=old_table[old_table.method=='reader']
        comparison=native_table[['unit','source_id','arm','split','accuracy','balanced_accuracy']].merge(
            old[['unit','accuracy','balanced_accuracy']],on='unit',suffixes=('_native','_unet'),validate='one_to_one')
        comparison=comparison.rename(columns={'accuracy_native':'native_accuracy','balanced_accuracy_native':'native_balanced_accuracy',
            'accuracy_unet':'unet_accuracy','balanced_accuracy_unet':'unet_balanced_accuracy'})
        comparison.to_csv(out/'reader_comparison.csv',index=False)
        write_csv(out/'clean_geometry.csv',clean_geometry(data,outputs,old_outputs))
        plot_results(out)
        report.update(status='complete',completed_units=48,decision='review_native_readability_and_clean_reference_before_any_geometry_attacks')
    except BaseException as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}',decision='stop_review_no_retry_no_expansion')
        (out/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        signal.alarm(0)
        if old_handler is not None:signal.signal(signal.SIGALRM,old_handler)
        for h in hooks:h.remove()
        changed=[]
        for parent,records in ((source,hashes),(reference,reference_hashes)):
            for name,h in records.items():
                if not (parent/name).is_file() or digest(parent/name)!=h:changed.append(str(parent/name))
        report.update(original_files_unchanged=not changed,changed_inputs=changed,wall_seconds=time.monotonic()-start,
            completed_units=sum(r['status']=='complete' for r in report['units']))
        if changed:report.update(status='stopped',decision='input_integrity_failure')
        write_json(out/'report.json',report)
        with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(out.rglob('*')):
                if file.is_file() and not file.name.endswith('.tmp'):z.write(file,str(file.relative_to(out)))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True)
    parser.add_argument('--reference',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.source,args.reference,args.output)
    if result['status']!='complete':raise SystemExit(2)
