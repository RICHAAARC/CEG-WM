"""Fit-only grouped LOO diagnostic. No validation inputs; no fitted asset export.

Run: python -m experiments.run_survival_region_loo --input FIT_DIR --output NEW_DIR
Frozen design: three original features, ridge=1, 8 leave-image-out folds,
old global centering versus image-centered X and y, 4096 within-image permutations.
"""
from pathlib import Path
import json
import argparse
import numpy as np

RIDGE=1.0
PERMUTATIONS=4096
SEED=20260910


def ranks(x):
    # Average zero-based ranks, including exact ties.
    return (x[..., :, None]>x[..., None, :]).sum(-1)+.5*((x[..., :, None]==x[..., None, :]).sum(-1)-1)


def metrics(predictions, targets):
    pr,yr=ranks(predictions),ranks(targets)
    pr,yr=pr-pr.mean(-1,keepdims=True),yr-yr.mean(-1,keepdims=True)
    den=np.sqrt((pr*pr).sum(-1)*(yr*yr).sum(-1))
    rho=np.divide((pr*yr).sum(-1),den,out=np.zeros_like(den),where=den>0)
    selected=predictions.argmax(-1)  # deterministic first block on a prediction tie
    picked=np.take_along_axis(targets,selected[...,None],axis=-1)[...,0]
    top=(picked==targets.max(-1)).astype(float)
    gain=picked-targets.mean(-1)
    return {'spearman':rho,'topblock_hit':top,'selected_centered_utility':gain,
            'selected_block':selected,'constant_rank_pair':den==0}


def operator(x, test_index, centered):
    train_indices=[i for i in range(8) if i!=test_index]
    design=x-x.mean(1,keepdims=True) if centered else x.copy()
    train=design[train_indices].reshape(28,3)
    mean=train.mean(0)
    scale=np.maximum(train.std(0),1e-8)
    z=(train-mean)/scale
    test=(design[test_index]-mean)/scale
    projection=test@np.linalg.solve(z.T@z+RIDGE*np.eye(3),z.T)
    return train_indices,projection,mean,scale


def predictions(x,all_y,centered):
    result=np.empty_like(all_y)
    targets=all_y-all_y.mean(-1,keepdims=True) if centered else all_y
    for held in range(8):
        train,projection,_,_=operator(x,held,centered)
        y=targets[:,train,:].reshape(len(all_y),28)
        y=y-y.mean(1,keepdims=True)
        result[:,held,:]=y@projection.T
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Fit directory with labels.jsonl and allocator.json; read only')
    parser.add_argument('--output',type=Path,required=True,help='New diagnostic output directory; no allocator is exported')
    args=parser.parse_args(argv)
    input_root=args.input
    output_root=args.output
    if output_root.exists():
        raise FileExistsError('Diagnostic output directory must be new')
    rows=[json.loads(line) for line in (input_root/'labels.jsonl').read_text().splitlines()]
    ids=sorted({r['id'] for r in rows})
    if len(rows)!=32 or len(ids)!=8 or len({(r['id'],r['block']) for r in rows})!=32:
        raise ValueError('Expected exactly eight images, four unique blocks per image')
    ordered=[]
    for identity in ids:
        group=sorted([r for r in rows if r['id']==identity],key=lambda r:r['block'])
        if [r['block'] for r in group]!=list(range(4)):raise ValueError('Invalid macroblock roster')
        ordered.append(group)
    x=np.asarray([[r['features'] for r in group] for group in ordered],dtype=float)
    y=np.asarray([[r['utility'] for r in group] for group in ordered],dtype=float)
    if x.shape!=(8,4,3) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid feature or utility observation')
    asset=json.loads((input_root/'allocator.json').read_text())
    penalty=asset['cost_penalty']
    rebuilt=np.asarray([[np.mean(r['score_increments'])-penalty*r['lpips_increment'] for r in group] for group in ordered])
    if not np.allclose(y,rebuilt,atol=1e-14,rtol=0):raise ValueError('Utility reconstruction differs')
    # Reproduce the OLD fitted asset only. No full-data centered candidate is fitted/exported.
    flat=x.reshape(32,3);mean=flat.mean(0);scale=np.maximum(flat.std(0),1e-8)
    z=(flat-mean)/scale
    coef=np.linalg.solve(z.T@z+RIDGE*np.eye(3),z.T@(y.ravel()-y.mean()))
    old_asset_reproduction={name:float(np.max(np.abs(actual-np.asarray(asset[name]))))
        for name,actual in [('mean',mean),('scale',scale),('coefficients',coef)]}
    if max(old_asset_reproduction.values())>1e-12:raise ValueError('Old training recipe does not reproduce saved asset')
    rng=np.random.default_rng(SEED)
    all_y=np.empty((PERMUTATIONS+1,8,4));all_y[0]=y
    for b in range(1,PERMUTATIONS+1):
        for i in range(8):all_y[b,i]=y[i,rng.permutation(4)]
    outputs={}
    null_arrays={}
    actual_predictions={}
    for name,centered in [('old_global',False),('image_centered',True)]:
        pred=predictions(x,all_y,centered)
        actual_predictions[name]=pred[0].tolist()
        m=metrics(pred,all_y)
        averages={k:v.mean(1) for k,v in m.items() if k in ['spearman','topblock_hit','selected_centered_utility']}
        null_arrays[name]=averages
        outputs[name]={'mean':{k:float(v[0]) for k,v in averages.items()},
            'per_image':[{'id':identity,'spearman':float(m['spearman'][0,i]),
                'topblock_hit':bool(m['topblock_hit'][0,i]),'selected_block':int(m['selected_block'][0,i]),
                'true_top_blocks':np.flatnonzero(y[i]==y[i].max()).tolist(),
                'selected_centered_utility':float(m['selected_centered_utility'][0,i]),
                'constant_rank_pair':bool(m['constant_rank_pair'][0,i]),
                'true_utilities':y[i].tolist(),'predictions':pred[0,i].tolist()} for i,identity in enumerate(ids)],
            'permutation':{k:{'null_mean':float(v[1:].mean()),
                'null_q025':float(np.quantile(v[1:],.025)),'null_q975':float(np.quantile(v[1:],.975)),
                'one_sided_p':float((1+np.count_nonzero(v[1:]>=v[0]-1e-15))/(PERMUTATIONS+1))} for k,v in averages.items()}}
    difference={}
    for key in null_arrays['old_global']:
        delta=null_arrays['image_centered'][key]-null_arrays['old_global'][key]
        difference[key]={'observed':float(delta[0]),'null_mean':float(delta[1:].mean()),
            'null_q025':float(np.quantile(delta[1:],.025)),'null_q975':float(np.quantile(delta[1:],.975)),
            'one_sided_p':float((1+np.count_nonzero(delta[1:]>=delta[0]-1e-15))/(PERMUTATIONS+1))}
    result={'design':{'features':['semantic','texture','latent_energy'],'ridge':RIDGE,
        'folds':8,'train_images_per_fold':7,'test_images_per_fold':1,'blocks_per_image':4,
        'target':'utility minus same-image four-block utility mean for image_centered; fixed existing utility definition',
        'utility_definition':'mean three-condition score increment minus original penalty times LPIPS increment',
        'feature_centering':'image_centered subtracts each image four-block feature means before training-only standardization',
        'permutations':PERMUTATIONS,'seed':SEED,'permutation_unit':'four labels independently permuted within each image; whole LOO refitted',
        'primary_metric':'selected_centered_utility','undefined_spearman_rule':'constant rank pair contributes zero as uninformative; flagged per image',
        'tie_selection':'lowest block index among prediction maxima; true maximum tie counts as hit',
        'validation24_read':False,'new_full_data_asset_fitted':False},
        'old_asset_max_abs_reproduction_error':old_asset_reproduction,'results':outputs,
        'centered_minus_old':difference,
        'limitations':['Only eight fit images; folds overlap in training images.',
          'Permutation exchangeability within image is a development null, not a universal data-generating guarantee.',
          'Three metrics are descriptive; auxiliary p values are not multiplicity-adjusted confirmations.',
          'Ranking four single-block probes does not establish benefit of a combined soft allocator or GPU candidate.',
          'No validation24 content read and no new full-data allocator fitted/exported.']}
    result['development_assessment']='Descriptive CPU diagnostic only; no automatic candidate decision or p-value gate.'
    output_root.mkdir(parents=True,exist_ok=False)
    (output_root/'loo_region_centering_result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(output_root/'loo_region_centering_null.npz',
        **{name+'__'+metric:arr[1:] for name,stats in null_arrays.items() for metric,arr in stats.items()})
    print(json.dumps({'results':{name:{'mean':value['mean'],'permutation':value['permutation']} for name,value in outputs.items()},
        'centered_minus_old':difference,'assessment':result['development_assessment']},indent=2))

if __name__=='__main__':main()
