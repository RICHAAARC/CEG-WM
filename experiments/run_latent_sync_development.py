"""One fixed latent anchor readability check; default prints the small plan."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def build_runtime():
    from experiments import run_blind_detection_v1 as old
    from experiments import content_unweighted_engine
    from cegwm.shared.keys import normalize_detection_key
    root=Path(__file__).resolve().parents[1]
    config=old.load_runtime_config(root)
    key=normalize_detection_key(os.environ['CEG_WM_ROOT_KEY'])
    pipeline,embed=content_unweighted_engine._load_pipeline_and_assets(config['content_model_id'],os.environ['HF_TOKEN'])
    lf=old.FrozenContentWhiteningLFPublicAssets(embed.lf_public_assets,old.load_whitening_asset_semantic(root))
    iss=old.ContentISSEvaluationAssets(embed,lf,old.load_iss_asset_semantic(root))
    assets=SimpleNamespace(content_assets=old.ContentCalibrationAssets(iss),
                           weighted_joint_asset=old.load_weighted_asset_semantic(root))
    return pipeline,key,assets


METHODS=('content_only','latent_anchor','v2_rgb_sync')


def development_plan():
    return dict(generations=3,images=4,methods=list(METHODS),conditions=['clean','v2_rotation_plus10'],
                blind_paths=12,oracle_paths=0,tolerance_paths=0,anchor_candidates=1,
                additional_content_only_reader_calls=2,
                threshold=None,formal_claims=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('outputs/latent-sync-readability'))
    parser.add_argument('--prompt',default='A ceramic teapot on a wooden table beside flowers, natural window light.')
    parser.add_argument('--seed',type=int,default=2031010000)
    args=parser.parse_args(argv)
    plan=development_plan()
    if not args.execute:
        print(json.dumps(plan,indent=2)); return 0
    import numpy as np
    from experiments.parallel_method_protocol_v1 import ACTIVE_A_ATTACKS,render_attack
    from cegwm.method.latent_sync import AnchorSpec
    from cegwm.runtime.latent_sync_sd35 import run_pair,score_image,read_anchor
    from cegwm.runtime.paper_detection_v2 import score_current_rgb
    from experiments.run_paper_main_worker_v2 import _quality
    from experiments import run_blind_detection_v1 as old
    from experiments.select_latent_sync_candidates import compare_report
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(plan=plan,rows=[],errors=[],quality={},science_denominator=0,
                pair_id=str(args.seed),prompt=args.prompt,anchor_spec=dict(rms=.04,seed=741091),
                claim='fixed candidate readability only; no demonstrated V4/V6 readout repair')
    import subprocess
    import importlib.metadata
    report['code_revision']=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    report['versions']={}
    for package in ('torch','diffusers','numpy','scipy'):
        try: report['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: report['versions'][package]=None
    try:
        pipeline,key,assets=build_runtime()
        clean,anchor,measurement=run_pair(pipeline,args.prompt,key,assets.content_assets.iss_assets,args.seed)
        _,content,_=run_pair(pipeline,args.prompt,key,assets.content_assets.iss_assets,args.seed,
                            AnchorSpec(rms=0),primary_null=clean)
        images=dict(content_only=content,latent_anchor=anchor)
        clean.save(args.output/'plain.png'); report['measurement']=measurement
        # Old sync loads only for the reference, never in the anchor scorer.
        reference_error=None
        try:
            root=Path(__file__).resolve().parents[1]; config=old.load_runtime_config(root)
            checkpoint=args.output/config['syncseal_filename']
            old.download_official_syncseal_torchscript(checkpoint)
            backend=old.SyncSealTorchScript.from_file(checkpoint,device=config['device'])
            images['v2_rgb_sync']=backend.embed_final_rgb(content,.75)
            reference_assets=SimpleNamespace(content_assets=assets.content_assets,
                    weighted_joint_asset=assets.weighted_joint_asset,geometry_backend=backend)
        except Exception as error: reference_error=type(error).__name__+': '+str(error)
        for method,image in images.items():
            image.save(args.output/(method+'.png')); entry={}
            try: entry.update(_quality(clean,image))
            except Exception as error: entry['error']=str(error)
            delta=(np.asarray(image,dtype=float)-np.asarray(clean,dtype=float))/255.
            entry['local_mse_2x2']=[float(np.mean(tile**2)) for band in np.array_split(delta,2,axis=0)
                                   for tile in np.array_split(band,2,axis=1)]
            report['quality'][method]=entry
        for attack in ACTIVE_A_ATTACKS:
            for method in METHODS:
                for role in ('negative','positive'):
                    row=dict(condition=attack.name,method=method,role=role,error=None)
                    try:
                        if method=='v2_rgb_sync' and reference_error: raise RuntimeError(reference_error)
                        image=clean if role=='negative' else images[method]
                        attacked,_=render_attack(image,attack)
                        if method=='latent_anchor': row.update(score_image(attacked,key,assets,pipeline))
                        elif method=='content_only':
                            value=float(score_current_rgb(attacked,key,assets).value)
                            row.update(pre=value,post=value,score=value,rgb_rectifications=0)
                            if role=='positive': row['anchor_absent_reader_diagnostic']=read_anchor(attacked,pipeline)
                        else:
                            from cegwm.geometry_v7.r1b import rectify_attacked_rgb
                            from cegwm.runtime.blind_detection import _geometry_disposition,_raw_h
                            pre=float(score_current_rgb(attacked,key,reference_assets).value)
                            geometry=backend.detect_geometry_continuous(attacked)
                            disposition,detail=_geometry_disposition(geometry)
                            if disposition=='OPERATIONAL': raise RuntimeError(detail)
                            post=pre; rectifications=0
                            if disposition!='INVALID_H':
                                try: H=_raw_h(geometry)
                                except (LookupError,TypeError,ValueError): H=None
                                if H is not None:
                                    try: recovered=rectify_attacked_rgb(attacked,H)
                                    except Exception: recovered=None
                                    if recovered is not None:
                                        post=float(score_current_rgb(recovered,key,reference_assets).value)
                                        rectifications=1
                            row.update(pre=pre,post=post,score=max(pre,post),rgb_rectifications=rectifications)
                    except Exception as error: row['error']=type(error).__name__+': '+str(error)
                    report['rows'].append(row)
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    existing={(r['condition'],r['method'],r['role']) for r in report['rows']}
    for attack in ACTIVE_A_ATTACKS:
        for method in METHODS:
            for role in ('negative','positive'):
                if (attack.name,method,role) not in existing:
                    report['rows'].append(dict(condition=attack.name,method=method,role=role,
                         error='setup/generation failed: '+'; '.join(report['errors'])))
    report['missing_paths']=12-len(report['rows'])
    report['failed_paths']=sum(r['error'] is not None for r in report['rows'])
    report['comparison']=compare_report(report)
    (args.output/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in report['rows']))
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    return 3 if report['errors'] or report['failed_paths'] else 0


if __name__=='__main__': raise SystemExit(main())
