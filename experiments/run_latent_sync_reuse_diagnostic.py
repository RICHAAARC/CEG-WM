"""Reuse four final RGB files; 0 generations, 8 reader and 32 content-score calls.

Oracle and fixed +/-1 degree rows are diagnostic inputs, never H selection or
detector candidates. Historical methods and the public anchor remain unchanged.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image

from experiments.parallel_method_protocol_v1 import ACTIVE_A_ATTACKS, render_attack, latent_to_rgb_matrix
from cegwm.method.latent_sync import rectify_once


SOURCE_NAMES = ('plain', 'content_only', 'latent_anchor', 'v2_rgb_sync')
VARIANTS = ('clean_raw', 'clean_identity', 'clean_predicted', 'rotation_raw',
            'rotation_truth', 'rotation_predicted', 'rotation_truth_minus1', 'rotation_truth_plus1')
DEFAULT_INPUT = '/content/drive/MyDrive/CEG-WM/development/latent-sync-v1/diagnostic-20260909-152107-462244'


def diagnostic_plan():
    return dict(source_images=4,source_names=list(SOURCE_NAMES),generations=0,
                logical_content_inputs=32,max_actual_content_scorer_calls=32,
                logical_reader_calls=8,max_actual_reader_calls=8,cache='none',
                expected_vae_encodes_if_complete=40,
                loaded_models=['SD3.5 public VAE only'],
                scoring='unchanged V2 registered-minus-max16wrong weighted LF/HF',
                variants=list(VARIANTS),threshold=None,oracle_for_diagnosis_only=True,
                tolerance='fixed -1/+1 degrees around true rotation; coarse probes, not a precise tolerance bound')


def rotation_about(center, degrees):
    angle=math.radians(degrees)
    linear=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    H=np.eye(3); H[:2,:2]=linear; H[:2,2]=np.asarray(center)-linear@center
    return H


def coordinate_audit():
    """Read the actual historical renderer H, including its half-pixel convention."""
    _,H=render_attack(Image.new('RGB',(512,512)),ACTIVE_A_ATTACKS[1])
    angle=float(np.degrees(np.arctan2(H[1,0],H[0,0])))
    center=np.linalg.solve(np.eye(2)-H[:2,:2],H[:2,2])
    S=latent_to_rgb_matrix((512,512),(64,64))
    latent_H=np.linalg.inv(S)@H@S
    latent_center=(np.linalg.inv(S)@np.r_[center,1])[:2]
    reader_center=(S@np.array([31.5,31.5,1.]))[:2]
    reader_exact_angle_H=rotation_about(reader_center,angle)
    return dict(forward_direction='reference RGB integer pixel centres to observed RGB integer pixel centres',
                named_attack=ACTIVE_A_ATTACKS[1].name,forward_screen_degrees=angle,
                actual_rgb_center=center.tolist(),actual_latent64_center=latent_center.tolist(),
                reader_rgb_center=reader_center.tolist(),reader_latent_center=[31.5,31.5],
                exact_angle_reader_center_translation_error_pixels=float(np.linalg.norm(reader_exact_angle_H[:2,2]-H[:2,2])),
                H_reference_to_observed=H.tolist(),H_observed_to_reference=np.linalg.inv(H).tolist(),
                H_latent64_reference_to_observed=latent_H.tolist(),latent_to_rgb=S.tolist(),
                rectify_sampling='pass forward H to one RGB output-to-input sampling; do not invert it',
                interpolation='historical attack: bilinear black fill; correction: bilinear black fill',
                identity='explicit rectify_once(I), no application shortcut; output equality recorded')


def load_scoring_runtime():
    """Load only the same VAE and detector public assets; never generation models."""
    import torch
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from cegwm.method.hf import FrozenHFPublicAssets
    from cegwm.method.lf import (FrozenLFPublicAssets, LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID)
    from cegwm.method.content_whitening import FrozenContentWhiteningLFPublicAssets
    from cegwm.shared.keys import normalize_detection_key
    from experiments.run_blind_detection_v1 import (load_runtime_config,
        load_whitening_asset_semantic,load_weighted_asset_semantic)
    root=Path(__file__).resolve().parents[1]
    config=load_runtime_config(root); model=config['content_model_id']
    key=normalize_detection_key(os.environ['CEG_WM_ROOT_KEY'])
    vae=AutoencoderKL.from_pretrained(model,subfolder='vae',torch_dtype=torch.float16,
                                    token=os.environ.get('HF_TOKEN'))
    vae.to(config['device']); vae.eval(); vae.requires_grad_(False)
    # Exact StableDiffusion3Pipeline constructor formula/default processor.
    processor=VaeImageProcessor(vae_scale_factor=2**(len(vae.config.block_out_channels)-1))
    hf=FrozenHFPublicAssets(vae=vae,image_processor=processor,image_processor_id=f'{model}:image_processor')
    lf_carrier=FrozenLFPublicAssets(vae=vae,image_processor=processor,image_processor_id=f'{model}:image_processor',
        candidate_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        evaluated_candidate_id=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID)
    lf=FrozenContentWhiteningLFPublicAssets(lf_carrier,load_whitening_asset_semantic(root))
    # Read-only scoring needs these assets, not ISS embedding/DINO objects.
    assets=SimpleNamespace(content_assets=SimpleNamespace(iss_assets=SimpleNamespace(
        lf_public_assets=lf,hf_public_assets=hf)),weighted_joint_asset=load_weighted_asset_semantic(root))
    return SimpleNamespace(vae=vae,image_processor=processor),key,assets


def detailed_content_score(image,key,assets):
    """Exactly the existing V2 branch/scalar path, with its intermediate values retained."""
    from cegwm.runtime.blind_scoring_v2 import score_branches_v2
    from cegwm.method.blind_detection import statistic_from_weighted_scores
    branches,mode=score_branches_v2(image,key,assets,reuse_observation=True)
    statistic=statistic_from_weighted_scores(branches['weighted_joint'])
    return dict(raw_content_score=statistic.value,
                registered_weighted_joint=statistic.registered_weighted_joint,
                wrong_key_max=statistic.wrong_key_max,
                wrong_key_weighted_joint=list(statistic.wrong_key_weighted_joint),
                branches=branches,observation_execution_mode=mode)


def evaluate_images(images, *, scorer, reader, on_row=None, on_reader_row=None):
    """Injected callbacks permit CPU path tests; real entry uses unchanged methods."""
    audit=coordinate_audit(); truth=np.asarray(audit['H_reference_to_observed'])
    center=audit['actual_rgb_center']; angle=audit['forward_screen_degrees']
    report=dict(plan=diagnostic_plan(),coordinate_audit=audit,rows=[],reader_rows=[],
                actual_content_scorer_calls=0,actual_reader_calls=0,science_denominator=0,
                lifecycle=dict(step18='historical embedding; not replayed or observable from final PNG',
                    step19='historical remaining sampling; not replayed or isolated by this diagnostic',
                    vae='fresh deterministic mode re-encoding of each scored/read final RGB',
                    rgb='each correction starts from original clean or attacked RGB; no serial warps'))
    def retain(row,reader_row=False):
        report['reader_rows' if reader_row else 'rows'].append(row)
        callback=on_reader_row if reader_row else on_row
        if callback is not None: callback(row)
    for name in SOURCE_NAMES:
        image=images.get(name)
        if image is None:
            for variant in VARIANTS: retain(dict(source=name,variant=variant,error='source image unavailable'))
            for condition in ('clean','rotation'): retain(dict(source=name,condition=condition,error='source image unavailable'),True)
            continue
        if image.mode!='RGB' or image.size!=(512,512): raise ValueError('source must be RGB 512x512')
        attacked,_=render_attack(image,ACTIVE_A_ATTACKS[1])
        predicted={}
        for condition,current in (('clean',image),('rotation',attacked)):
            row=dict(source=name,condition=condition,error=None)
            try:
                report['actual_reader_calls']+=1
                diagnostic=reader(current)
                matrix=np.asarray(diagnostic['H_reference_to_observed_pixels'],dtype=float)
                if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-10:
                    raise ValueError('reader H must be finite invertible 3x3')
                predicted[condition]=matrix
                row.update(diagnostic)
                expected=np.eye(3) if condition=='clean' else truth
                corners=np.array([[0,0,1],[511,0,1],[511,511,1],[0,511,1]]).T
                row['corner_rmse_pixels']=float(np.sqrt(np.mean(np.sum(((predicted[condition]-expected)@corners)[:2]**2,axis=0))))
            except Exception as error: row['error']=type(error).__name__+': '+str(error)
            retain(row,True)
        for variant in VARIANTS:
            row=dict(source=name,variant=variant,error=None,
                diagnostic_only=variant in ('clean_identity','rotation_truth','rotation_truth_minus1','rotation_truth_plus1'))
            try:
                base=image if variant.startswith('clean') else attacked
                H=None
                if variant=='clean_identity': H=np.eye(3)
                elif variant=='clean_predicted': H=predicted['clean']
                elif variant=='rotation_truth': H=truth
                elif variant=='rotation_predicted': H=predicted['rotation']
                elif variant=='rotation_truth_minus1': H=rotation_about(center,angle-1.)
                elif variant=='rotation_truth_plus1': H=rotation_about(center,angle+1.)
                candidate=base if H is None else rectify_once(base,H)
                row.update(correction_resamplings=int(H is not None),serial_correction=False,
                           H_sampling_reference_to_observed=None if H is None else H.tolist())
                if variant=='clean_identity':
                    row.update(identity_application_shortcut=False,
                               identity_pixels_equal=bool(np.array_equal(np.asarray(candidate),np.asarray(image))))
                report['actual_content_scorer_calls']+=1
                scored=scorer(candidate)
                if isinstance(scored,dict):
                    value=float(scored['raw_content_score']); row.update(scored)
                else: value=float(scored)  # lightweight callback tests only
                if not math.isfinite(value): raise ValueError('nonfinite content score')
                row['raw_content_score']=value
            except Exception as error: row['error']=type(error).__name__+': '+str(error)
            retain(row)
    report['failed_content_rows']=sum(r['error'] is not None for r in report['rows'])
    report['failed_reader_rows']=sum(r['error'] is not None for r in report['reader_rows'])
    report['path_comparisons']=[]
    for name in SOURCE_NAMES:
        keyed={r['variant']:r for r in report['rows'] if r['source']==name}
        for condition,pre,post in (('clean','clean_raw','clean_predicted'),('rotation','rotation_raw','rotation_predicted')):
            row=dict(source=name,condition=condition,error=None)
            if keyed[pre]['error'] or keyed[post]['error']:
                row['error']='pre or estimated-post score unavailable'
            else:
                a=keyed[pre]['raw_content_score']; b=keyed[post]['raw_content_score']
                row.update(pre=a,estimated_post=b,whole_pre_estimated_post=max(a,b),
                           oracle_used_in_whole_path=False,threshold=None)
            report['path_comparisons'].append(row)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path(DEFAULT_INPUT))
    parser.add_argument('--output',type=Path,default=Path('outputs/latent-sync-reuse-diagnostic'))
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(plan=diagnostic_plan(),input_dir=str(args.input_dir),coordinate_audit=coordinate_audit()),indent=2))
        return 0
    args.output.mkdir(parents=True,exist_ok=False)
    images={}; source_errors={}
    for name in SOURCE_NAMES:
        try:
            with Image.open(args.input_dir/(name+'.png')) as image:
                if image.mode!='RGB' or image.size!=(512,512): raise ValueError('expected RGB 512x512')
                images[name]=image.copy()
        except Exception as error: source_errors[name]=type(error).__name__+': '+str(error)
    setup_error=None
    with (args.output/'rows.jsonl').open('w') as row_stream, (args.output/'reader_rows.jsonl').open('w') as reader_stream:
        def persist(stream,row):
            stream.write(json.dumps(row)+'\n'); stream.flush()
        callbacks=dict(on_row=lambda row:persist(row_stream,row),on_reader_row=lambda row:persist(reader_stream,row))
        try: pipeline,key,assets=load_scoring_runtime()
        except Exception as error: setup_error=type(error).__name__+': '+str(error)
        if setup_error:
            report=evaluate_images({},scorer=None,reader=None,**callbacks)
        else:
            from cegwm.runtime.latent_sync_sd35 import read_anchor
            report=evaluate_images(images,scorer=lambda image:detailed_content_score(image,key,assets),
                                   reader=lambda image:read_anchor(image,pipeline),**callbacks)
    report.update(source_dir=str(args.input_dir),source_errors=source_errors,setup_error=setup_error)
    report['code_revision']=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    report['versions']={}
    for package in ('torch','diffusers','numpy','scipy','Pillow'):
        try: report['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: report['versions'][package]=None
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    return 3 if setup_error or source_errors or report['failed_content_rows'] or report['failed_reader_rows'] else 0


if __name__=='__main__': raise SystemExit(main())
