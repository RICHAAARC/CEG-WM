"""Explicit real-model development entry; default prints plan without loading models."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def build_runtime():
    # Build frozen content assets without downloading/loading the old sync model.
    from experiments import run_blind_detection_v1 as old
    from experiments import content_unweighted_engine
    from cegwm.shared.keys import normalize_detection_key
    root = Path(__file__).resolve().parents[1]
    config = old.load_runtime_config(root)
    key = normalize_detection_key(os.environ['CEG_WM_ROOT_KEY'])
    pipeline,embed = content_unweighted_engine._load_pipeline_and_assets(config['content_model_id'],os.environ['HF_TOKEN'])
    lf = old.FrozenContentWhiteningLFPublicAssets(embed.lf_public_assets,old.load_whitening_asset_semantic(root))
    iss = old.ContentISSEvaluationAssets(embed,lf,old.load_iss_asset_semantic(root))
    assets = SimpleNamespace(content_assets=old.ContentCalibrationAssets(iss),
                             weighted_joint_asset=old.load_weighted_asset_semantic(root))
    return pipeline,key,assets


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('outputs/latent-sync-dev'))
    parser.add_argument('--prompt',default='A ceramic teapot on a wooden table beside flowers, natural window light.')
    parser.add_argument('--seed',type=int,default=2031010000)
    parser.add_argument('--anchor-rms',type=float,default=.04)
    parser.add_argument('--anchor-seed',type=int,default=741091)
    parser.add_argument('--anchor-config',type=Path)
    parser.add_argument('--objective',choices=('geometry','content_tolerance'),default='geometry')
    parser.add_argument('--split',choices=('fit','validation'),default='validation')
    parser.add_argument('--tolerance',action='store_true')
    args=parser.parse_args(argv)
    if args.anchor_config:
        selection=json.loads(args.anchor_config.read_text())
        if args.split=='validation' and any(str(args.seed)==str(pair[0]) for pair in selection['fit_pairs']):
            raise ValueError('validation seed overlaps fit selection')
        frozen=selection['selected'][args.objective]
        args.anchor_rms=frozen['rms']; args.anchor_seed=frozen['seed']
    plan=dict(generations=2,images=2,attacks_per_image=9,blind_paths=18,
              oracle_paths=9,additional_tolerance_paths=108 if args.tolerance else 0,
              formal_claims=False,threshold=None,anchor_rms=args.anchor_rms,anchor_seed=args.anchor_seed)
    if not args.execute:
        print(json.dumps(plan,indent=2)); return 0
    import numpy as np
    from experiments.parallel_method_protocol_v1 import CORE_ATTACKS,render_attack
    from cegwm.method.latent_sync import AnchorSpec
    from cegwm.runtime.latent_sync_sd35 import run_pair,score_image,tolerance_surface
    from experiments.run_paper_main_worker_v2 import _quality
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(plan=plan,rows=[],errors=[],science_denominator=0,split=args.split,
                pair_id=str(args.seed),prompt=args.prompt,
                anchor_spec=dict(rms=args.anchor_rms,seed=args.anchor_seed))
    import subprocess
    import importlib.metadata
    report['code_revision']=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    report['versions']={}
    for package in ('torch','diffusers','numpy','scipy'):
        try: report['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: report['versions'][package]=None
    try:
        pipeline,key,assets=build_runtime()
        spec=AnchorSpec(rms=args.anchor_rms,seed=args.anchor_seed)
        clean,marked,measurement=run_pair(pipeline,args.prompt,key,assets.content_assets.iss_assets,args.seed,spec)
        clean.save(args.output/'plain.png'); marked.save(args.output/'marked.png')
        report['measurement']=measurement
        try:
            report['quality']=_quality(clean,marked)
        except Exception as error:
            report['quality_error']=str(error)
        delta=(np.asarray(marked,dtype=float)-np.asarray(clean,dtype=float))/255.
        report['local_distortion_mse_2x2']=[float(np.mean(tile**2)) for band in np.array_split(delta,2,axis=0)
                                         for tile in np.array_split(band,2,axis=1)]
        for index,attack in enumerate(CORE_ATTACKS):
            for role,image in [('negative',clean),('positive',marked)]:
                row=dict(condition=attack.name,role=role,error=None)
                try:
                    attacked,H=render_attack(image,attack,noise_seed=args.seed+index)
                    row.update(score_image(attacked,key,assets,pipeline,spec))
                    if role=='positive':
                        predicted=np.array(row['H_reference_to_observed_pixels'])
                        corners=np.array([[0,0,1],[511,0,1],[511,511,1],[0,511,1]]).T
                        row['corner_rmse_pixels']=float(np.sqrt(np.mean(np.sum(((predicted-H)@corners)[:2]**2,axis=0))))
                        offsets=[(0,1,0,0)]
                        if args.tolerance:
                            offsets += [(a,1,0,0) for a in (-2,-1,1,2)]
                            offsets += [(0,s,0,0) for s in (.98,1.02)]
                            offsets += [(0,1,x,y) for x,y in ((-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,1))]
                        row['development_oracle_tolerance']=tolerance_surface(attacked,key,assets,H,offsets)
                        row['oracle_minus_post_loss']=max(0.,row['development_oracle_tolerance'][0]['score']-row['post'])
                except Exception as error:
                    row['error']=type(error).__name__+': '+str(error)
                report['rows'].append(row)
                with (args.output/'rows.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
        existing={(row['condition'],row['role']) for row in report['rows']}
        for attack in CORE_ATTACKS:
            for role in ('negative','positive'):
                if (attack.name,role) not in existing:
                    row=dict(condition=attack.name,role=role,error=report['errors'][-1])
                    report['rows'].append(row)
                    with (args.output/'rows.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
    report['missing_paths']=18-len(report['rows'])
    report['failed_paths']=sum(row['error'] is not None for row in report['rows'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    return 3 if report['errors'] or report['missing_paths'] or report['failed_paths'] else 0


if __name__=='__main__':
    raise SystemExit(main())
