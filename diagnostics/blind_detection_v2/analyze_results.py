"""Read-only recomputation of the fixed stage-2 canary; never repairs raw rows."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import numpy as np


def key(row):
    return tuple(row.get(k) for k in ("unit_id","kind","arm","strength","angle","angle_error","dx","dy"))


def analyze(root):
    root=Path(root)
    rows=[json.loads(line) for line in (root/"rows.jsonl").read_text().splitlines()]
    plan=json.loads((root/"plan.json").read_text())
    expected=[]
    for unit in plan["roster"]:
        for strength in plan["strengths"]:
            for angle in plan["angles"]:
                expected.append(key(dict(unit_id=unit["unit_id"],kind="strength",arm="positive",strength=strength,angle=angle)))
        for angle in plan["angles"]:
            expected.append(key(dict(unit_id=unit["unit_id"],kind="baseline_negative",arm="negative",angle=angle)))
        for arm in ("positive","negative"):
            for a,x,y in plan["perturbations"]:
                expected.append(key(dict(unit_id=unit["unit_id"],kind="tolerance",arm=arm,angle_error=a,dx=x,dy=y)))
    observed=[key(row) for row in rows]
    tau=plan["reference_threshold"]
    report={"science_denominator":0,"base_pairs":len(plan["roster"]),"rows":len(rows),
            "unique_rows":len(set(observed)),"missing":len(set(expected)-set(observed)),
            "unexpected":len(set(observed)-set(expected)),"kinds":dict(Counter(r["kind"] for r in rows)),
            "errors":[],"strength":[],"tolerance":[]}
    for row in rows:
        if row["error"]:
            report["errors"].append({"key":key(row),"original_error":row["error"],
                "display_status":"PUBLIC_H_UNAVAILABLE" if row.get("geometry",{}).get("homography_observed_to_canonical") is None else "ERROR",
                "v1_route":row.get("v1",{}).get("route"),"v1_complete":row.get("v1",{}).get("method_complete"),
                "native_warp_score":row.get("native_warp_score")})
    for strength in plan["strengths"]:
        for angle in plan["angles"]:
            group=[r for r in rows if r["kind"]=="strength" and r["strength"]==strength and r["angle"]==angle]
            report["strength"].append({"multiplier":strength,"angle":angle,"n":len(group),
                "v1_positive":sum(r["v1"]["positive"] for r in group),
                "public_positive":sum(r.get("adapter_warp_score",float('-inf'))>tau for r in group),
                "native_positive":sum(r.get("native_warp_score",float('-inf'))>tau for r in group),
                "oracle_positive":sum(r.get("oracle_warp_score",float('-inf'))>tau for r in group),
                "corner_rmse_median":median(r["corner_rmse_pixels"] for r in group),
                "psnr_vs_content_median":median(r["quality_vs_content"]["psnr_db"] for r in group)})
    for a,x,y in plan["perturbations"]:
        group=[r for r in rows if r["kind"]=="tolerance" and (r["angle_error"],r["dx"],r["dy"])==(a,x,y)]
        report["tolerance"].append({"angle_error":a,"dx":x,"dy":y,
            **{arm:sum(r["score"]>tau for r in group if r["arm"]==arm) for arm in ("positive","negative")}})
    fits=[]
    square=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    for row in rows:
        if row["kind"]=="strength" and row["strength"]==.75 and row["angle"]!=0:
            points=np.asarray(row["geometry"]["observed_corners_in_canonical_normalized"])
            u,_,vt=np.linalg.svd(square.T@(points-points.mean(axis=0)))
            correction=np.diag([1.,np.linalg.det(u@vt)])
            rot=(u@correction@vt).T
            angle=float(np.degrees(np.arctan2(rot[1,0],rot[0,0])))
            fits.append({"unit_id":row["unit_id"],"attack":row["angle"],"fit_angle":angle,"angle_error":abs(angle-row["angle"])})
    report["rigid_fits_diagnostic_only"]=fits
    report["single_score_seconds_median"]=median(r["seconds"] for r in rows if r["kind"]=="tolerance")
    report["row_seconds_sum_excludes_generation"]=sum(r["seconds"] for r in rows)
    negative=[r for r in rows if r["kind"]=="baseline_negative"]
    report["negative"]={"rows":len(negative),"v1_positive":sum(r["v1"]["positive"] for r in negative),
        "public_available":sum("adapter_warp_score" in r for r in negative),"native_available":sum("native_warp_score" in r for r in negative)}
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("strength","tolerance","rigid_fits_diagnostic_only")}))
