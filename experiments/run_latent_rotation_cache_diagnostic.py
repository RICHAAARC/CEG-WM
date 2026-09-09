"""Cache eight final-RGB reader observations, then diagnose the unchanged objective.

No generation, content scoring, detection key, or replacement formal estimator.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
from PIL import Image

from experiments.run_latent_sync_reuse_diagnostic import SOURCE_NAMES, DEFAULT_INPUT, coordinate_audit
from experiments.parallel_method_protocol_v1 import ACTIVE_A_ATTACKS, render_attack

CONDITIONS=('clean','rotation')


def plan():
    return dict(source_images=4,observations=8,generations=0,content_scorer_calls=0,
        maximum_vae_encodes=8,loaded_models=['original public SD3.5 VAE only'],
        followup='automatic CPU objective analysis in the same run',
        formal_reader_changed=False,curve_role='diagnostic only; no new detector selection',
        cache_definition='final RGB-derived CHW float32 array passed to the original estimate_rotation')


def load_vae_runtime():
    import torch
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from experiments.run_blind_detection_v1 import load_runtime_config
    config=load_runtime_config(Path(__file__).resolve().parents[1]); model=config['content_model_id']
    vae=AutoencoderKL.from_pretrained(model,subfolder='vae',torch_dtype=torch.float16,token=os.environ.get('HF_TOKEN'))
    vae.to(config['device']); vae.eval(); vae.requires_grad_(False)
    processor=VaeImageProcessor(vae_scale_factor=2**(len(vae.config.block_out_channels)-1))
    return SimpleNamespace(vae=vae,image_processor=processor,model_id=model)


def encode_reader_observation(image,runtime):
    """Invoke the original normalization path once, without changing rounding order."""
    from cegwm.runtime.observation import encode_final_rgb_image
    details={}
    class RecordingProcessor:
        def preprocess(self,rgb):
            pixels=runtime.image_processor.preprocess(rgb)
            details.update(preprocessed_shape=list(pixels.shape),preprocessed_dtype=str(pixels.dtype),
                preprocessed_min=float(pixels.min()),preprocessed_max=float(pixels.max()))
            return pixels
    observation=encode_final_rgb_image(image,RecordingProcessor(),runtime.vae)
    array=observation[0].float().cpu().numpy()
    parameter=next(runtime.vae.parameters())
    details.update(model_id=runtime.model_id,subfolder='vae',vae_parameter_dtype=str(parameter.dtype),
        vae_device=str(parameter.device),processor_class=type(runtime.image_processor).__name__,
        processor_config=dict(getattr(runtime.image_processor,'config',{})),
        distribution='latent_dist.mode(), never sample()',
        shift_factor=float(runtime.vae.config.shift_factor),scaling_factor=float(runtime.vae.config.scaling_factor),
        normalized_observation_dtype=str(observation.dtype),
        normalization_order='(mode - shift_factor) * scaling_factor in original tensor dtype, then [0].float().cpu().numpy()',
        stored_shape=list(array.shape),stored_dtype=str(array.dtype),stored_layout='CHW',
        source_is='observed final RGB; not original embedding latent')
    return array,details


def cache_observations(input_dir,output_dir,*,runtime,encoder=encode_reader_observation,on_row=None):
    """All eight fixed units remain, including missing-source and encoding failures."""
    rows=[];calls=0
    for source in SOURCE_NAMES:
        image=None;error=None
        try:
            with Image.open(input_dir/(source+'.png')) as opened:
                if opened.mode!='RGB' or opened.size!=(512,512): raise ValueError('source must be RGB512x512')
                image=opened.copy()
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        for condition in CONDITIONS:
            row=dict(source=source,condition=condition,error=error,cache_file=None)
            try:
                if error:raise ValueError(error)
                current=image if condition=='clean' else render_attack(image,ACTIVE_A_ATTACKS[1])[0]
                calls+=1
                observation,metadata=encoder(current,runtime)
                if observation.ndim!=3 or observation.dtype!=np.float32 or not np.isfinite(observation).all():
                    raise ValueError('cache must be the finite CHW float32 original-reader input')
                name=source+'__'+condition+'.npz'
                np.savez_compressed(output_dir/name,observation=observation)
                row.update(cache_file=name,encoding=metadata,error=None)
            except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
            rows.append(row)
            if on_row:on_row(row)
    return rows,calls


def diagnose_cache(cache_dir,cache_rows,*,on_row=None):
    from experiments.latent_rotation_objective_diagnostic import (analyze_observation,
        center_sensitivity,mapped_diagnostic_center)
    rows=[]
    for entry in cache_rows:
        row=dict(source=entry['source'],condition=entry['condition'],error=entry['error'])
        if not row['error']:
            try:
                with np.load(cache_dir/entry['cache_file'],allow_pickle=False) as cached:
                    observation=cached['observation']
                # The analyzer receives no truth, original image, content score, or key.
                row['objective_diagnostic']=analyze_observation(observation)
                row['center_sensitivity_diagnostic_only']=center_sensitivity(
                    observation,mapped_diagnostic_center(observation.shape[-2:]))
                truth=0. if entry['condition']=='clean' else -10.
                row['reference_for_error_reporting_only']=dict(
                    true_screen_angle=truth,
                    source='known historical diagnostic attack; never passed to estimator')
                row['angle_errors_degrees']={name:result['angle']-truth for name,result in
                    row['objective_diagnostic']['optimizers'].items()}
            except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
        rows.append(row)
        if on_row:on_row(row)
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode',choices=('plan','encode-and-diagnose','diagnose-cache'),default='plan')
    parser.add_argument('--input-dir',type=Path,default=Path(DEFAULT_INPUT))
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/latent-rotation-objective'))
    args=parser.parse_args(argv)
    if args.mode=='plan':print(json.dumps(plan(),indent=2));return 0
    if args.mode=='diagnose-cache' and args.cache_dir is None:parser.error('--cache-dir required')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(plan=plan(),mode=args.mode,source_dir=str(args.input_dir),vae_encoding_attempts=0,
        setup_error=None,coordinate_audit=coordinate_audit(),cache_rows=[],diagnostic_rows=[],
        interpretation='No content scores or causal correction claim. Discrete curve peak is not a continuous global optimum.')
    report['code_revision']=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,
                                         cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    report['versions']={}
    for name in ('torch','diffusers','numpy','scipy','Pillow'):
        try:report['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:report['versions'][name]=None
    def save_report(): (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (args.output/'cache_rows.jsonl').open('w') as cache_stream,(args.output/'rows.jsonl').open('w') as row_stream:
        def persist(stream,row):stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        if args.mode=='encode-and-diagnose':
            try:runtime=load_vae_runtime()
            except Exception as exc:report['setup_error']=type(exc).__name__+': '+str(exc)
            if report['setup_error']:
                report['cache_rows']=[dict(source=s,condition=c,error=report['setup_error'],cache_file=None) for s in SOURCE_NAMES for c in CONDITIONS]
                for row in report['cache_rows']:persist(cache_stream,row)
            else:
                report['cache_rows'],report['vae_encoding_attempts']=cache_observations(args.input_dir,args.output,
                    runtime=runtime,on_row=lambda row:persist(cache_stream,row))
                # GPU work is finished before CPU curves; cached observations are sufficient.
                del runtime
            cache_dir=args.output
        else:
            cache_dir=args.cache_dir
            try:
                report['cache_rows']=[json.loads(line) for line in (cache_dir/'cache_rows.jsonl').read_text().splitlines()]
                expected={(s,c) for s in SOURCE_NAMES for c in CONDITIONS}
                actual={(r['source'],r['condition']) for r in report['cache_rows']}
                if len(report['cache_rows'])!=8 or actual!=expected:raise ValueError('expected all eight unique cached units')
            except Exception as exc:
                report['setup_error']=type(exc).__name__+': '+str(exc)
                report['cache_rows']=[dict(source=s,condition=c,error=report['setup_error'],cache_file=None) for s in SOURCE_NAMES for c in CONDITIONS]
            for row in report['cache_rows']:persist(cache_stream,row)
        save_report()
        report['diagnostic_rows']=diagnose_cache(cache_dir,report['cache_rows'],on_row=lambda row:persist(row_stream,row))
    report['failed_units']=sum(r['error'] is not None for r in report['diagnostic_rows'])
    report['cached_observations']=sum(r['error'] is None for r in report['cache_rows'])
    save_report()
    print(json.dumps(dict(output=str(args.output),vae_encoding_attempts=report['vae_encoding_attempts'],
        content_scorer_calls=0,generations=0,failed_units=report['failed_units'])))
    return 3 if report['setup_error'] or report['failed_units'] else 0


if __name__=='__main__':raise SystemExit(main())
