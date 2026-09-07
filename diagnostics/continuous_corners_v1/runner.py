"""Fixed independent development; one raw geometry and three score routes."""
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate, CANONICAL_CORNERS_NORMALIZED
from cegwm.geometry_v7.r1a import (
    _pixel_output_to_source, condition_by_id, render_r1a_attack, corner_rmse,
    apply_homography,
)
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.runtime.blind_detection import BlindProductionAssets, _score_current_rgb, _geometry_disposition, _raw_h
from cegwm.runtime.content_iss_sd35 import run_content_iss_evaluation_pair
from cegwm.protocol.content_chain import CONTENT_CHAIN_PUBLIC_KEY_DIGEST
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from diagnostics.continuous_corners_v1.paired_geometry import estimate_pair

ROOT=Path(__file__).resolve().parents[2]
TAU=1.2657276026437319


def validate_runtime(pipeline,assets,key):
    from experiments.run_blind_detection_v1 import load_weighted_asset_semantic,load_whitening_asset_semantic,load_iss_asset_semantic
    if type(assets) is not BlindProductionAssets: raise TypeError('requires production assets')
    normalized=normalize_detection_key(key)
    if public_key_digest(normalized)!=CONTENT_CHAIN_PUBLIC_KEY_DIGEST: raise ValueError('original detection key differs')
    iss=assets.content_assets.iss_assets
    if getattr(pipeline,'vae',None) is not iss.hf_public_assets.vae or getattr(pipeline,'image_processor',None) is not iss.hf_public_assets.image_processor:
        raise ValueError('pipeline and scoring observation assets differ')
    if any(a.payload!=b.payload for a,b in (
        (assets.weighted_joint_asset,load_weighted_asset_semantic(ROOT)),
        (iss.lf_public_assets.whitening_asset,load_whitening_asset_semantic(ROOT)),
        (iss.iss_asset,load_iss_asset_semantic(ROOT)))):
        raise ValueError('public computation assets differ')
    return normalized


def render_condition(image,condition):
    if image.mode!='RGB' or image.size!=(512,512): raise ValueError('requires RGB512')
    if condition['id']=='identity': return image.copy(),np.eye(3)
    if condition['id']=='core_fixed_canvas_zoom_0_8':
        spec=condition_by_id(condition['id'])
        return render_r1a_attack(image,spec),np.asarray(spec.truth_observed_to_canonical)
    angle=math.radians(condition['angle_degrees'])
    c,s=math.cos(angle),math.sin(angle)
    truth=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    coefficients=list(_pixel_output_to_source(truth))
    source=image
    if condition['fill']=='reflect':
        source=Image.fromarray(np.pad(np.asarray(image),((128,128),(128,128),(0,0)),mode='reflect'))
        coefficients[2]+=128
        coefficients[5]+=128
    return source.transform((512,512),Image.Transform.PERSPECTIVE,coefficients,
        resample=Image.Resampling.BILINEAR,fillcolor=(0,0,0)),truth


def score_record(image,score):
    begin=time.perf_counter()
    row=dict(statistic=None,error=None,scorer_called=True)
    try: row['statistic']=asdict(score(image))
    except Exception as error: row['error']=type(error).__name__+': '+str(error)
    row['seconds']=time.perf_counter()-begin
    return row


def derived_route(pre,post):
    """Offline description of early-return policy, never a calibrated decision."""
    if pre['statistic'] is None: return dict(route='ERROR_FAIL_CLOSED',positive=False,method_complete=False)
    if pre['statistic']['value']>TAU: return dict(route='DIRECT_POSITIVE',positive=True,method_complete=True)
    if post['error']:
        route={'geometry_invalid':'GEOMETRY_FAIL_CLOSED','geometry_no_h':'GEOMETRY_NO_H',
            'rectification':'RECTIFICATION_FAIL_CLOSED'}.get(post.get('failure_stage'),'ERROR_FAIL_CLOSED')
        return dict(route=route,positive=False,method_complete=route!='ERROR_FAIL_CLOSED')
    return dict(route='GEOMETRY_RECOVERED',positive=post['statistic']['value']>TAU,method_complete=True)


def observe(image,backend,score):
    pre=score_record(image,score)
    start=time.perf_counter()
    try: pair=estimate_pair(backend,image)
    except Exception as error: pair={m:GeometryEstimate.error_record(error) for m in ('rounded','continuous')}
    geometry_seconds=time.perf_counter()-start
    posts={}
    for mode,geometry in pair.items():
        row=dict(statistic=None,error=None,scorer_called=False,failure_stage=None,seconds=0.,rectification_seconds=0.)
        try:
            disposition,error=_geometry_disposition(geometry)
            if disposition=='OPERATIONAL':
                row.update(error=error or 'geometry operational error',failure_stage='geometry')
            elif disposition=='INVALID_H':
                row.update(error=error or 'invalid H',failure_stage='geometry_invalid')
            else:
                try: h=_raw_h(geometry)
                except LookupError:
                    row.update(error='H absent',failure_stage='geometry_no_h')
                except (TypeError,ValueError) as exc:
                    row.update(error=str(exc),failure_stage='geometry_invalid')
                if row['error'] is None:
                    start=time.perf_counter()
                    try: recovered=rectify_attacked_rgb(image,h)
                    except Exception as exc: row.update(error=type(exc).__name__+': '+str(exc),failure_stage='rectification')
                    row['rectification_seconds']=time.perf_counter()-start
                    if row['error'] is None:
                        row.update(score_record(recovered,score))
                        if row['error']: row['failure_stage']='content_post'
        except Exception as exc:
            row.update(error=type(exc).__name__+': '+str(exc),failure_stage='geometry')
        posts[mode]=row
    return dict(pre=pre,posts=posts,geometry={m:asdict(g) for m,g in pair.items()},
        geometry_seconds=geometry_seconds,geometry_inference_calls=1,
        derived_routes={m:derived_route(pre,p) for m,p in posts.items()})


def unavailable(reason):
    pre=dict(statistic=None,error=reason,scorer_called=False,seconds=0.)
    posts={m:dict(pre,failure_stage='input',rectification_seconds=0.) for m in ('rounded','continuous')}
    return dict(pre=pre,posts=posts,geometry=None,geometry_seconds=0.,geometry_inference_calls=0,
        derived_routes={m:derived_route(pre,p) for m,p in posts.items()})


class Session:
    def __init__(self,pipeline,assets,key,output):
        self.key=validate_runtime(pipeline,assets,key)
        self.pipeline,self.assets=pipeline,assets
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.plan=json.loads((Path(__file__).with_name('development_roster.json')).read_text())
        (self.output/'plan.json').write_text(json.dumps(dict(self.plan,status='SESSION_CREATED',key_identity=public_key_digest(self.key)),indent=2))
        (self.output/'rows.jsonl').open('x').close()
        self.next_unit=0
        self.rows=[]
        self.score=lambda image:_score_current_rgb(image,self.key,self.assets)

    def run_unit(self,index):
        if index!=self.next_unit or index not in range(4): raise ValueError('fixed units once in order; no retry')
        self.next_unit+=1
        unit=self.plan['units'][index]
        directory=self.output/unit['unit_id']
        directory.mkdir()
        start=time.perf_counter()
        arms={}
        generation_error=None
        tick=time.perf_counter()
        try:
            result=run_content_iss_evaluation_pair(self.pipeline,unit['prompt'],self.key,
                self.assets.content_assets.iss_assets,height=512,width=512,seed=unit['seed'])
            arms['U']=result.primary_null
            content=result.image
        except Exception as error: generation_error=type(error).__name__+': '+str(error)
        generation_seconds=time.perf_counter()-tick
        embedding={}
        if generation_error is None:
            for name,source in (('CG',content),('G',arms['U'])):
                tick=time.perf_counter()
                try:
                    arms[name]=self.assets.geometry_backend.embed_final_rgb(source,.75)
                    embedding[name]=dict(error=None,seconds=time.perf_counter()-tick)
                except Exception as error:
                    embedding[name]=dict(error=type(error).__name__+': '+str(error),seconds=time.perf_counter()-tick)
        image_save_errors={}
        if generation_error is None:
            try: content.save(directory/'C.png')
            except Exception as error: image_save_errors['C']=type(error).__name__+': '+str(error)
        for name,image in arms.items():
            try: image.save(directory/(name+'.png'))
            except Exception as error: image_save_errors[name]=type(error).__name__+': '+str(error)
        unit_rows=[]
        with (self.output/'rows.jsonl').open('a') as file:
            for arm in self.plan['arms']:
                for condition in self.plan['conditions']:
                    if arm not in arms:
                        observation=unavailable(generation_error or embedding.get(arm,{}).get('error') or 'source arm unavailable')
                        observation['input_failure_stage']='generation' if generation_error else 'sync_embedding'
                    else:
                        tick=time.perf_counter()
                        try:
                            attacked,truth=render_condition(arms[arm],condition)
                            render_seconds=time.perf_counter()-tick
                        except Exception as error:
                            observation=unavailable(type(error).__name__+': '+str(error))
                            observation['input_failure_stage']='render'
                        else:
                            observation=observe(attacked,self.assets.geometry_backend,self.score)
                            observation['render_seconds']=render_seconds
                            # Truth is only evaluated AFTER all score routes.
                            observation['geometry_error_px']={}
                            try:
                                actual=apply_homography(truth,CANONICAL_CORNERS_NORMALIZED)
                                for mode,g in observation['geometry'].items():
                                    points=g['observed_corners_in_canonical_normalized']
                                    observation['geometry_error_px'][mode]=corner_rmse(points,actual)*255.5*math.sqrt(2) if points else None
                            except Exception as error:
                                observation['truth_analysis_error']=type(error).__name__+': '+str(error)
                    observation.update(unit_id=unit['unit_id'],arm=arm,condition=condition['id'],science_denominator=0)
                    file.write(json.dumps(observation)+'\n')
                    file.flush()
                    self.rows.append(observation)
                    unit_rows.append(observation)
        receipt=dict(unit_id=unit['unit_id'],generation_seconds=generation_seconds,generation_error=generation_error,
            sync_embedding=embedding,image_save_errors=image_save_errors,geometry_seconds=sum(r['geometry_seconds'] for r in unit_rows),
            content_score_seconds=sum(r['pre']['seconds']+sum(x['seconds'] for x in r['posts'].values()) for r in unit_rows),
            elapsed_seconds=time.perf_counter()-start,observations=len(unit_rows),planned_score_routes=54,
            actual_score_calls=sum(r['pre']['scorer_called']+sum(x['scorer_called'] for x in r['posts'].values()) for r in unit_rows))
        (directory/'timing.json').write_text(json.dumps(receipt,indent=2))
        return receipt

    def pilot(self):
        result=self.run_unit(0)
        return dict(result,remaining_units=3,remaining_max_score_routes=162,rough_remaining_seconds=result['elapsed_seconds']*3)

    def remaining(self):
        if self.next_unit!=1: raise ValueError('inspect pilot before remaining')
        receipts=[self.run_unit(i) for i in range(1,4)]
        flat=[s for r in self.rows for s in (r['pre'],*r['posts'].values())]
        summary=dict(planned_observations=72,recorded_observations=len(self.rows),planned_score_routes=216,
            actual_score_calls=sum(x['scorer_called'] for x in flat),complete_statistics=sum(x['statistic'] is not None for x in flat),
            route_errors=sum(x['error'] is not None for x in flat),geometry_inference_calls=sum(x['geometry_inference_calls'] for x in self.rows),
            remaining_unit_timings=receipts,science_denominator=0)
        (self.output/'summary.json').write_text(json.dumps(summary,indent=2))
        with (self.output/'paired.csv').open('x',newline='') as file:
            fields=['unit','arm','condition','pre_m','rounded_post_m','continuous_post_m','rounded_derived','continuous_derived','errors']
            writer=csv.DictWriter(file,fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                value=lambda x:x['statistic']['value'] if x['statistic'] else None
                writer.writerow(dict(unit=row['unit_id'],arm=row['arm'],condition=row['condition'],pre_m=value(row['pre']),
                    rounded_post_m=value(row['posts']['rounded']),continuous_post_m=value(row['posts']['continuous']),
                    rounded_derived=json.dumps(row['derived_routes']['rounded']),continuous_derived=json.dumps(row['derived_routes']['continuous']),
                    errors=json.dumps([x['error'] for x in (row['pre'],*row['posts'].values())])))
        return summary
