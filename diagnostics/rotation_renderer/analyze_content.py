"""Offline audit of saved content statistics and geometry; no model calls."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

TAU = 1.2657276026437319
Q = np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rmse(points, truth):
    return float(np.sqrt(np.mean(np.sum(((points-truth)*255.5)**2,axis=1))))


def components(points, truth):
    design = np.column_stack((Q,np.ones(4)))
    fit = np.linalg.lstsq(design,points,rcond=None)[0]
    linear = fit[:2].T
    u,s,vh = np.linalg.svd(linear)
    rotation = u@vh
    rigid_points = Q@rotation.T + fit[2]
    return dict(corner_rmse_px=rmse(points,truth),
        angle_deg=float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0]))),
        corner_centroid_shift_px=(fit[2]*255.5).tolist(),
        affine_singular_values=s.tolist(),
        affine_fit_residual_px=rmse(points,design@fit),
        rigid_projection_rmse_px=rmse(rigid_points,truth))


def analyze(content, inputs):
    content,inputs=Path(content),Path(inputs)
    originals=read(inputs/'r1a-result.json')
    specs={s['condition_id']:np.asarray(s['truth_observed_to_canonical']) for s in originals['condition_specs']}
    records=[json.loads(s) for name in ('pilot.jsonl','remaining.jsonl') for s in (content/name).read_text().splitlines()]
    rows=[]
    geometry=[]
    for record in records:
        if 'case' in record:
            unit,arm,condition,fill=record['case']
            scores=record['scores']
            g=record['geometry']
            raw=np.asarray(g['raw_syncseal_corners'])
            public=np.asarray(g['observed_corners_in_canonical_normalized'])
            h=specs[condition]
            homogeneous=np.column_stack((Q,np.ones(4)))@h.T
            truth=homogeneous[:,:2]/homogeneous[:,2:]
            unrounded=2*(128*raw+128)/255-1
            official_integer=2*np.trunc(np.round(128*raw+128)*511/255)/511-1
            variants=dict(public_current=public,unrounded_same_256_conversion=unrounded,
                raw_as_normalized_diagnostic=raw,native_second_integer_step=official_integer)
            geometry.append(dict(unit=unit,arm=arm,condition=condition,fill=fill,
                variants={k:components(v,truth) for k,v in variants.items()},
                rounding_displacement_px=((public-unrounded)*255.5).tolist(),
                rounding_displacement_rmse_px=rmse(public,unrounded),
                public_error_vectors_px=((public-truth)*255.5).tolist(),
                h_center_shift_px=(np.asarray(g['homography_observed_to_canonical'])[:2,2]*255.5).tolist(),
                h_perspective_row=g['homography_observed_to_canonical'][2]))
        else:
            unit,arm,condition,fill=record['unit'],record['arm'],'identity','none'
            scores=[record]
        for score in scores:
            stat=score['statistic']
            wrong=stat['wrong_key_weighted_joint']
            if len(wrong)!=16: raise ValueError('wrong-key count differs')
            m=stat['registered_weighted_joint']-max(wrong)
            if not np.isfinite(m): raise ValueError('nonfinite statistic')
            rows.append(dict(unit=unit,arm=arm,condition=condition,fill=fill,route=score['route'],
                m=stat['value'],registered=stat['registered_weighted_joint'],wrong_max=max(wrong),
                wrong_max_index=int(np.argmax(wrong)),margin=m-TAU,
                m_recompute_error=stat['value']-m,
                wrong_max_error=stat['wrong_key_max']-max(wrong),
                boundary_matches=score['above_reference_tau']==(m>TAU),
                scorer_called=score['scorer_called'],error=score['error']))
    keys=[(r['unit'],r['arm'],r['condition'],r['fill'],r['route']) for r in rows]
    expected={(u,a,c,f,t) for u in ('content-v6-iss-eval-0001','content-v6-iss-eval-0002')
        for a in ('CG','G') for c in ('core_rotation_neg15','core_rotation_pos15')
        for f in ('black','reflect') for t in ('pre','predicted_h_post','oracle')}
    expected|={(u,a,'identity','none','unattacked') for u in ('content-v6-iss-eval-0001','content-v6-iss-eval-0002') for a in ('CG','G')}
    if len(keys)!=52 or set(keys)!=expected: raise ValueError('fixed 52 matrix differs')
    csv_rows=list(csv.DictReader((content/'paired.csv').open()))
    csv_lookup={(r['unit'],r['arm'],r['condition'],r['fill'],r['route']):r for r in csv_rows}
    if len(csv_rows)!=52 or set(csv_lookup)!=expected: raise ValueError('CSV coverage differs')
    csv_error=max(abs(float(csv_lookup[k]['m'])-r['m']) for k,r in zip(keys,rows))
    groups=[]
    for arm in ('CG','G'):
        for fill in ('black','reflect'):
            for route in ('pre','predicted_h_post','oracle'):
                selected=[r for r in rows if (r['arm'],r['fill'],r['route'])==(arm,fill,route)]
                groups.append(dict(arm=arm,fill=fill,route=route,denominator=4,
                    above_reference=sum(r['margin']>0 for r in selected),
                    m_values=[r['m'] for r in selected]))
    summary=dict(rows=len(rows),unique=len(set(keys)),scorer_calls=sum(r['scorer_called'] for r in rows),
        errors=sum(r['error'] is not None for r in rows),
        m_recompute_max_error=max(abs(r['m_recompute_error']) for r in rows),
        wrong_max_max_error=max(abs(r['wrong_max_error']) for r in rows),
        all_boundaries_match=all(r['boundary_matches'] for r in rows),csv_max_error=csv_error,
        groups=groups, science_denominator=0,
        source_note='Local JSON is controller reserialization of cloud original objects; not byte originals. Cloud originals preserved.')
    return dict(summary=summary,statistics=rows,geometry=geometry)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--content',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.content,args.inputs)
    with args.output.open('x') as file: json.dump(result,file,indent=2)
    print(json.dumps(result['summary'],indent=2))
