"""Eight prepared continuous-corner scores; no geometry inference or search."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

from cegwm.geometry_v7.contracts import homography_observed_to_canonical
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.runtime.blind_detection import _score_current_rgb
from cegwm.shared.keys import public_key_digest
from diagnostics.rotation_renderer.content_plan import (
    UNITS, ARMS, FORMAL_TAU_REFERENCE, prepare_sources, validate_scoring_assets,
)
from diagnostics.rotation_renderer.run import CONDITIONS, render

ROSTER=tuple((u,a,c) for u in UNITS for c in CONDITIONS for a in ARMS)


def continuous_h(raw):
    points=np.asarray(raw,dtype=np.float64)
    if points.shape!=(4,2) or not np.isfinite(points).all():
        raise ValueError('missing or invalid four raw corners')
    points=2*(points*128+128)/255-1  # Only remove the original round operation.
    h=homography_observed_to_canonical(points)
    return points.tolist(),h


def _one(image, raw_geometry, score):
    row=dict(continuous_corners=None,continuous_h=None,statistic=None,
        above_reference_tau=None,scorer_called=False,error=None)
    begin=time.perf_counter()
    try:
        points,h=continuous_h(raw_geometry['raw_syncseal_corners'])
        row.update(continuous_corners=points,continuous_h=h)
        current=rectify_attacked_rgb(image,h)
        row['scorer_called']=True
        statistic=score(current)
        row.update(statistic=asdict(statistic),above_reference_tau=statistic.value>FORMAL_TAU_REFERENCE)
    except Exception as error:
        row['error']=type(error).__name__+': '+str(error)
    row['seconds']=time.perf_counter()-begin
    return row


class SubpixelSession:
    """Prepare from saved records, then run once only after execution approval."""
    def __init__(self,inputs,cg_reference,previous_content,output,key,assets):
        r0,self.sources=prepare_sources(inputs)
        self.key=validate_scoring_assets(assets,key,r0)
        previous=Path(previous_content)
        old_plan=json.loads((previous/'plan.json').read_text(encoding='utf-8-sig'))
        if old_plan['key_identity']!=public_key_digest(self.key):
            raise ValueError('previous scoring key differs')
        old_rows=[json.loads(x) for name in ('pilot.jsonl','remaining.jsonl')
            for x in (previous/name).read_text().splitlines()]
        self.references={}
        self.raw={}
        for row in old_rows:
            if 'case' not in row: continue
            u,a,c,f=row['case']
            if f!='black': continue
            k=(u,a,c)
            if k in self.references: raise ValueError('duplicate previous case')
            if row['source']['original_path']!=self.sources[u,a]['original_path']:
                raise ValueError('previous case image differs')
            self.references[k]=row['scores']
            if a=='G': self.raw[k]=row['geometry']
        cg=json.loads(Path(cg_reference).read_text())['rows']
        for row in cg:
            if row['fill']!='black' or row['interpolation']!='bilinear': continue
            k=(row['unit_id'],'CG',row['condition_id'])
            if k in self.raw: raise ValueError('duplicate saved CG')
            if row['original_image_file']!=self.sources[k[:2]]['original_path']:
                raise ValueError('CG geometry image differs')
            self.raw[k]=row['geometry']
        if set(self.raw)!=set(ROSTER) or set(self.references)!=set(ROSTER):
            raise ValueError('fixed eight cases missing from saved records')
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.score=lambda image:_score_current_rgb(image,self.key,assets)
        metadata=dict(status='PREPARED_SESSION',roster=ROSTER,planned_routes=8,
            science_denominator=0,formula='2*(128*raw+128)/255-1; no round',
            geometry_inference_calls=0,threshold_reference=FORMAL_TAU_REFERENCE,
            key_identity=public_key_digest(self.key),previous_content=str(previous),
            cg_reference=str(cg_reference),sources=list(self.sources.values()),
            candidate_selection='fixed single conversion, no truth or score selection')
        (self.output/'plan.json').write_text(json.dumps(metadata,indent=2))

    def run(self):
        results=[]
        with (self.output/'rows.jsonl').open('x') as file:
            for case in ROSTER:
                u,a,c=case
                try:
                    with Image.open(self.sources[u,a]['path']) as source:
                        image=render(source,c,'black','bilinear')
                    row=_one(image,self.raw[case],self.score)
                except Exception as error:
                    row=dict(statistic=None,continuous_corners=None,continuous_h=None,
                        scorer_called=False,above_reference_tau=None,error='render:'+type(error).__name__+': '+str(error))
                row.update(case=case,raw_geometry=self.raw[case],reference_scores=self.references[case])
                file.write(json.dumps(row)+'\n')
                file.flush()
                results.append(row)
        fields=['unit','arm','condition','old_post_m','continuous_m','oracle_m','above_reference_tau','scorer_called','error']
        with (self.output/'paired.csv').open('x',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields)
            writer.writeheader()
            for row in results:
                reference={x['route']:x['statistic'] for x in row['reference_scores']}
                u,a,c=row['case']
                writer.writerow(dict(unit=u,arm=a,condition=c,
                    old_post_m=reference['predicted_h_post']['value'] if reference['predicted_h_post'] else None,
                    oracle_m=reference['oracle']['value'] if reference['oracle'] else None,
                    continuous_m=row['statistic']['value'] if row['statistic'] else None,
                    above_reference_tau=row['above_reference_tau'],scorer_called=row['scorer_called'],error=row['error']))
        summary=dict(planned_routes=8,recorded_routes=len(results),
            actual_score_calls=sum(x['scorer_called'] for x in results),
            complete_statistics=sum(x['statistic'] is not None for x in results),
            errors=sum(x['error'] is not None for x in results),geometry_inference_calls=0,science_denominator=0)
        (self.output/'summary.json').write_text(json.dumps(summary,indent=2))
        return summary
